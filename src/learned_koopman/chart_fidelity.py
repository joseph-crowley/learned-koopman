from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from learned_koopman.canonical_diagnostics import fit_residual_spectrum


@dataclass(frozen=True)
class ChartFidelityConfig:
    """Controlled first-order identifiability experiment."""

    harmonic_order: int = 4
    kick_amplitude: float = 0.01
    base_frequency: float = 1.0
    twist: float = 0.3
    gauge_amplitude: float = 0.45
    off_resonance_action_offset: float = 1.0
    angle_samples: int = 8192
    chart_error_levels: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5)


def _validate_config(config: ChartFidelityConfig) -> None:
    if config.harmonic_order < 1:
        raise ValueError("harmonic_order must be positive")
    if config.kick_amplitude <= 0.0 or config.twist == 0.0:
        raise ValueError("kick_amplitude must be positive and twist must be nonzero")
    if config.angle_samples < 64:
        raise ValueError("angle_samples must be at least 64")
    if not config.chart_error_levels or min(config.chart_error_levels) < 0.0:
        raise ValueError("chart_error_levels must be nonempty and nonnegative")


def _twist_kick_step(
    action: np.ndarray,
    angle: np.ndarray,
    config: ChartFidelityConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """An exact symplectic kick followed by an action-dependent drift."""

    updated_action = action + config.kick_amplitude * np.sin(
        config.harmonic_order * angle
    )
    updated_angle = (
        angle + config.base_frequency + config.twist * updated_action
    ) % (2.0 * np.pi)
    return updated_action, updated_angle


def _observe(
    action: np.ndarray,
    angle: np.ndarray,
    amplitude: float,
    harmonic_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the exact canonical shear J = I + a cos(m phi), psi = phi."""

    return action + amplitude * np.cos(harmonic_order * angle), angle


def _unobserve(
    observed_action: np.ndarray,
    observed_angle: np.ndarray,
    amplitude: float,
    harmonic_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        observed_action - amplitude * np.cos(harmonic_order * observed_angle),
        observed_angle,
    )


def _probe_action(
    action_value: float,
    *,
    label: str,
    config: ChartFidelityConfig,
) -> dict[str, Any]:
    angle = np.linspace(0.0, 2.0 * np.pi, config.angle_samples, endpoint=False)
    action = np.full_like(angle, action_value)
    next_action, next_angle = _twist_kick_step(action, angle, config)
    observed_action, observed_angle = _observe(
        action,
        angle,
        config.gauge_amplitude,
        config.harmonic_order,
    )
    observed_next_action, observed_next_angle = _observe(
        next_action,
        next_angle,
        config.gauge_amplitude,
        config.harmonic_order,
    )

    true_generator_amplitude = config.kick_amplitude / config.harmonic_order
    rows = []
    for chart_error in config.chart_error_levels:
        estimated_amplitude = config.gauge_amplitude * (1.0 + chart_error)
        recovered_action, recovered_angle = _unobserve(
            observed_action,
            observed_angle,
            estimated_amplitude,
            config.harmonic_order,
        )
        recovered_next_action, _ = _unobserve(
            observed_next_action,
            observed_next_angle,
            estimated_amplitude,
            config.harmonic_order,
        )
        spectrum = fit_residual_spectrum(
            recovered_angle,
            recovered_next_action - recovered_action,
            max_order=config.harmonic_order,
        )
        harmonic = spectrum.harmonics[config.harmonic_order - 1]
        relative_error = abs(
            harmonic.generating_function_amplitude - true_generator_amplitude
        ) / true_generator_amplitude
        rows.append(
            {
                "chart_relative_error": chart_error,
                "recovered_generating_function_amplitude": (
                    harmonic.generating_function_amplitude
                ),
                "true_generating_function_amplitude": true_generator_amplitude,
                "relative_amplitude_error": relative_error,
                "spectrum_normalized_rmse": spectrum.normalized_rmse,
            }
        )
    return {
        "label": label,
        "action": action_value,
        "frequency": config.base_frequency + config.twist * action_value,
        "harmonic_rotation_number": (
            config.harmonic_order
            * (config.base_frequency + config.twist * action_value)
            / (2.0 * np.pi)
        ),
        "measurements": rows,
    }


def run_chart_fidelity_experiment(
    config: ChartFidelityConfig | None = None,
) -> dict[str, Any]:
    """Test whether a resonant residual survives a misspecified canonical chart.

    This is an oracle experiment: the physical map, observation chart, and chart
    error are known. It verifies the expected cohomological cancellation in a
    controlled setting. It does *not* establish that a learned chart has the same
    protection; that is the next falsifier.
    """

    resolved = config or ChartFidelityConfig()
    _validate_config(resolved)
    resonance_frequency = 2.0 * np.pi / resolved.harmonic_order
    resonant_action = (
        resonance_frequency - resolved.base_frequency
    ) / resolved.twist
    off_resonant_action = resonant_action + resolved.off_resonance_action_offset
    resonant = _probe_action(resonant_action, label="resonant", config=resolved)
    off_resonant = _probe_action(
        off_resonant_action,
        label="off_resonant",
        config=resolved,
    )

    nonzero = [
        index
        for index, value in enumerate(resolved.chart_error_levels)
        if value > 0.0
    ]
    ratios = []
    for index in nonzero:
        resonant_error = resonant["measurements"][index]["relative_amplitude_error"]
        off_error = off_resonant["measurements"][index]["relative_amplitude_error"]
        ratios.append(off_error / max(resonant_error, 1e-12))
    minimum_protection = min(ratios)
    return {
        "schema_version": 1,
        "experiment": "controlled_chart_fidelity_separation",
        "config": asdict(resolved),
        "resonant_probe": resonant,
        "off_resonant_probe": off_resonant,
        "comparison": {
            "off_to_resonant_error_ratio_by_nonzero_chart_error": ratios,
            "minimum_off_to_resonant_error_ratio": minimum_protection,
            "passes_controlled_threefold_protection_gate": minimum_protection >= 3.0,
        },
        "claim_boundary": {
            "supported": (
                "For this known canonical shear and synthetic symplectic map, "
                "the target residual harmonic is substantially more stable at resonance."
            ),
            "not_supported": (
                "The experiment does not show that optimizer error in a learned chart "
                "obeys the same bound or that measured-system residuals are identifiable."
            ),
            "next_falsifier": (
                "Repeat with learned chart ensembles and adversarial chart errors at a "
                "fixed held-out map error."
            ),
        },
    }
