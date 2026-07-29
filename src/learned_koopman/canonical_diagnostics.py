from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from learned_koopman.canonical_model import CanonicalKoopmanNetwork


@dataclass(frozen=True)
class OrbitDiagnostics:
    """Independent geometry, phase-law, and conjugacy checks for one orbit."""

    radial_coefficient_of_variation: float
    phase_step_coefficient_of_variation: float
    phase_law_rmse_radians: float
    normalized_conjugacy_rmse: float
    mean_phase_step_radians: float
    verdict: str


@dataclass(frozen=True)
class ResidualHarmonic:
    """One Fourier component of ``delta I = -partial_phi G``."""

    order: int
    sine_coefficient: float
    cosine_coefficient: float
    action_kick_amplitude: float
    generating_function_amplitude: float
    phase_radians: float


@dataclass(frozen=True)
class ResidualSpectrum:
    """Least-squares residual spectrum with an explicit fit remainder."""

    intercept: float
    rmse: float
    normalized_rmse: float
    r2: float
    sample_count: int
    harmonics: tuple[ResidualHarmonic, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["harmonics"] = [asdict(row) for row in self.harmonics]
        return result


def wrap_angle(values: np.ndarray) -> np.ndarray:
    """Map angles to ``[-pi, pi)`` without changing shape."""

    array = np.asarray(values, dtype=np.float64)
    return (array + np.pi) % (2.0 * np.pi) - np.pi


def fit_residual_spectrum(
    angle: np.ndarray,
    delta_action: np.ndarray,
    *,
    max_order: int,
) -> ResidualSpectrum:
    """Fit a truncated type-2 residual generating-function spectrum.

    With

    ``G(phi) = K_m cos(m phi + phase_m)``,

    the associated action kick is

    ``delta I = -partial_phi G = m K_m sin(m phi + phase_m)``.

    The returned ``generating_function_amplitude`` therefore divides the fitted
    action-kick amplitude by the harmonic order. This convention is stated here
    because silently mixing the two amplitudes changes island-width formulas.
    """

    if max_order < 1:
        raise ValueError("max_order must be positive")
    phi = np.asarray(angle, dtype=np.float64).reshape(-1)
    kick = np.asarray(delta_action, dtype=np.float64).reshape(-1)
    if phi.shape != kick.shape:
        raise ValueError("angle and delta_action must have matching shapes")
    if len(phi) < 2 * max_order + 2:
        raise ValueError("not enough samples for the requested harmonic order")
    if not np.isfinite(phi).all() or not np.isfinite(kick).all():
        raise ValueError("residual spectrum inputs must be finite")

    columns = [np.ones_like(phi)]
    for order in range(1, max_order + 1):
        columns.extend((np.sin(order * phi), np.cos(order * phi)))
    design = np.column_stack(columns)
    coefficients = np.linalg.lstsq(design, kick, rcond=None)[0]
    prediction = design @ coefficients
    residual = kick - prediction
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    scale = max(float(np.sqrt(np.mean(np.square(kick - kick.mean())))), 1e-12)
    total = float(np.square(kick - kick.mean()).sum())
    unexplained = float(np.square(residual).sum())

    harmonics = []
    for order in range(1, max_order + 1):
        sine = float(coefficients[2 * order - 1])
        cosine = float(coefficients[2 * order])
        kick_amplitude = float(np.hypot(sine, cosine))
        harmonics.append(
            ResidualHarmonic(
                order=order,
                sine_coefficient=sine,
                cosine_coefficient=cosine,
                action_kick_amplitude=kick_amplitude,
                generating_function_amplitude=kick_amplitude / order,
                phase_radians=float(np.arctan2(cosine, sine)),
            )
        )
    return ResidualSpectrum(
        intercept=float(coefficients[0]),
        rmse=rmse,
        normalized_rmse=rmse / scale,
        r2=1.0 - unexplained / max(total, 1e-24),
        sample_count=len(phi),
        harmonics=tuple(harmonics),
    )


def diagnose_canonical_orbits(
    network: CanonicalKoopmanNetwork,
    states: np.ndarray | torch.Tensor,
    *,
    radial_tolerance: float = 0.08,
    phase_tolerance: float = 0.08,
    conjugacy_tolerance: float = 0.08,
) -> list[OrbitDiagnostics]:
    """Diagnose whether complete observed orbits support the learned chart.

    The checks are deliberately orthogonal:

    - radial variation tests whether the chart circularizes the orbit;
    - phase-step variation tests whether the observed angle advances uniformly;
    - phase-law error tests the learned ``omega(I)`` law;
    - conjugacy error tests the complete latent one-step prediction.

    A single state cannot establish these properties. Callers should use this
    on complete trajectories before treating an action-range check as support.
    """

    values = torch.as_tensor(states, dtype=torch.float32)
    if values.ndim == 2:
        values = values.unsqueeze(0)
    if values.ndim != 3 or values.shape[-1] != 2 or values.shape[1] < 3:
        raise ValueError("expected one or more trajectories with shape (time, 2)")
    if min(radial_tolerance, phase_tolerance, conjugacy_tolerance) <= 0.0:
        raise ValueError("diagnostic tolerances must be positive")

    with torch.no_grad():
        latent = network.encode(values)
        q, p = latent.unbind(dim=-1)
        radius = torch.sqrt(torch.clamp(q * q + p * p, min=1e-12))
        radial_cv = radius.std(dim=1, correction=0) / radius.mean(dim=1).clamp_min(1e-12)

        phase = torch.atan2(-p, q)
        observed_step = torch.atan2(
            torch.sin(phase[:, 1:] - phase[:, :-1]),
            torch.cos(phase[:, 1:] - phase[:, :-1]),
        )
        mean_step = observed_step.mean(dim=1)
        phase_step_cv = observed_step.std(dim=1, correction=0) / mean_step.abs().clamp_min(
            1e-12
        )

        action = network.action_from_latent(latent[:, :-1])
        expected_step = network.dt * network.hamiltonian.frequency(action)
        phase_error = torch.atan2(
            torch.sin(observed_step - expected_step),
            torch.cos(observed_step - expected_step),
        )
        phase_rmse = torch.sqrt(torch.mean(torch.square(phase_error), dim=1))

        predicted_latent = network.latent_step(latent[:, :-1])
        conjugacy_rmse = torch.sqrt(
            torch.mean(torch.square(predicted_latent - latent[:, 1:]), dim=(1, 2))
        )
        latent_scale = torch.sqrt(torch.mean(torch.square(latent[:, 1:]), dim=(1, 2)))
        normalized_conjugacy = conjugacy_rmse / latent_scale.clamp_min(1e-12)

    result = []
    for index in range(values.shape[0]):
        supported = (
            float(radial_cv[index]) <= radial_tolerance
            and float(phase_step_cv[index]) <= phase_tolerance
            and float(normalized_conjugacy[index]) <= conjugacy_tolerance
        )
        result.append(
            OrbitDiagnostics(
                radial_coefficient_of_variation=float(radial_cv[index]),
                phase_step_coefficient_of_variation=float(phase_step_cv[index]),
                phase_law_rmse_radians=float(phase_rmse[index]),
                normalized_conjugacy_rmse=float(normalized_conjugacy[index]),
                mean_phase_step_radians=float(mean_step[index]),
                verdict=(
                    "supported_by_orbit_residuals"
                    if supported
                    else "chart_residual_exceeds_threshold"
                ),
            )
        )
    return result


def summarize_orbit_diagnostics(rows: list[OrbitDiagnostics]) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one orbit diagnostic is required")
    metrics = (
        "radial_coefficient_of_variation",
        "phase_step_coefficient_of_variation",
        "phase_law_rmse_radians",
        "normalized_conjugacy_rmse",
    )
    summary: dict[str, Any] = {
        "trajectory_count": len(rows),
        "supported_trajectory_count": sum(
            row.verdict == "supported_by_orbit_residuals" for row in rows
        ),
        "per_trajectory": [asdict(row) for row in rows],
    }
    for name in metrics:
        values = np.asarray([getattr(row, name) for row in rows], dtype=np.float64)
        summary[f"mean_{name}"] = float(values.mean())
        summary[f"maximum_{name}"] = float(values.max())
    return summary
