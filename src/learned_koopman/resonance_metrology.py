from __future__ import annotations

import hashlib
import html
import json
import platform
import random
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from learned_koopman import __version__
from learned_koopman.canonical_experiment import (
    CanonicalExperimentConfig,
    _fit_network,
    _rollout_metrics,
    _set_seed,
    _split_indices,
)
from learned_koopman.canonical_model import (
    CanonicalKoopmanModel,
    CanonicalKoopmanNetwork,
    save_canonical_model,
)
from learned_koopman.map_fixtures import (
    ExactGauge,
    KickHarmonic,
    MapTrajectoryBundle,
    ObservationChart,
    TwistKickMap,
    simulate_map_trajectories,
    wrap_angle,
    write_map_trajectory_csv,
)
from learned_koopman.trajectory import TrajectoryDataset, load_trajectory_csv


@dataclass(frozen=True)
class ArchitectureSpec:
    hidden_dim: int
    shear_layers: int

    @property
    def label(self) -> str:
        return f"h{self.hidden_dim}-s{self.shear_layers}"


@dataclass(frozen=True)
class MetrologyConfig:
    """Frozen execution profile for learned-chart resonance metrology."""

    profile: str
    output: Path
    seeds: tuple[int, ...]
    architectures: tuple[ArchitectureSpec, ...]
    epochs: int
    trajectories: int
    steps: int
    target_order: int = 3
    base_frequency: float = 1.6
    twist: float = 0.3
    kick_amplitude: float = 0.0075
    kick_phase: float = 0.9
    action_band: tuple[float, float] = (0.7, 2.6)
    bins: int = 14
    max_order: int = 8
    split_seed: int = 20260728
    minimum_accepted_charts: int = 6

    @classmethod
    def full(cls, output: Path) -> MetrologyConfig:
        return cls(
            profile="full",
            output=output,
            seeds=(7, 17, 29, 41),
            architectures=(
                ArchitectureSpec(24, 6),
                ArchitectureSpec(40, 8),
            ),
            epochs=600,
            trajectories=48,
            steps=400,
        )

    @classmethod
    def ci(cls, output: Path) -> MetrologyConfig:
        return cls(
            profile="ci",
            output=output,
            seeds=(7, 17, 29, 41, 53, 67),
            architectures=(ArchitectureSpec(24, 6),),
            epochs=80,
            trajectories=36,
            steps=240,
        )


@dataclass(frozen=True)
class SpectrumEstimate:
    coefficient: complex
    standard_error: float
    normalized_remainder: float
    condition_number: float
    sample_count: int
    angular_resultant: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_source_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_worktree_clean": None}
    return {"git_commit": commit, "git_worktree_clean": not dirty}


def _fit_complex_spectrum(
    angle: np.ndarray,
    delta_action: np.ndarray,
    *,
    order: int,
    max_order: int,
) -> SpectrumEstimate:
    phi = np.asarray(angle, dtype=np.float64).reshape(-1)
    residual = np.asarray(delta_action, dtype=np.float64).reshape(-1)
    if phi.shape != residual.shape:
        raise ValueError("angle and residual must have matching shapes")
    if order < 1 or order > max_order:
        raise ValueError("order must be between one and max_order")
    if len(phi) < 2 * max_order + 2:
        raise ValueError("not enough samples for the spectrum")
    columns = [np.ones_like(phi)]
    for harmonic in range(1, max_order + 1):
        columns.extend(
            (np.sin(harmonic * phi), np.cos(harmonic * phi))
        )
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, residual, rcond=None)
    prediction = design @ coefficients
    remainder = residual - prediction
    degrees = max(len(phi) - design.shape[1], 1)
    variance = float(remainder @ remainder) / degrees
    covariance = variance * np.linalg.pinv(design.T @ design)
    sine_index = 2 * order - 1
    cosine_index = 2 * order
    coefficient = complex(
        float(coefficients[sine_index]),
        float(coefficients[cosine_index]),
    )
    standard_error = float(
        np.sqrt(
            max(
                covariance[sine_index, sine_index]
                + covariance[cosine_index, cosine_index],
                0.0,
            )
        )
    )
    centered_scale = max(float(np.std(residual)), 1e-12)
    return SpectrumEstimate(
        coefficient=coefficient,
        standard_error=standard_error,
        normalized_remainder=float(np.sqrt(np.mean(remainder**2)))
        / centered_scale,
        condition_number=float(np.linalg.cond(design)),
        sample_count=len(phi),
        angular_resultant=float(abs(np.mean(np.exp(1j * order * phi)))),
    )


def weighted_birkhoff_mean(values: np.ndarray) -> float:
    """Das–Yorke-style bump-weighted average of a scalar sequence."""

    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(samples) < 8:
        raise ValueError("weighted Birkhoff average needs at least eight samples")
    positions = (np.arange(len(samples), dtype=np.float64) + 0.5) / len(
        samples
    )
    weights = np.exp(-1.0 / (positions * (1.0 - positions)))
    weights /= weights.sum()
    return float(weights @ samples)


def _frequency_profile(
    action: np.ndarray,
    angle: np.ndarray,
) -> dict[str, Any]:
    mean_actions = []
    weighted_frequencies = []
    ordinary_frequencies = []
    for orbit_action, orbit_angle in zip(action, angle, strict=True):
        increments = wrap_angle(np.diff(orbit_angle))
        if abs(float(increments.sum())) < 4.0 * np.pi:
            continue
        mean_actions.append(float(np.mean(orbit_action[:-1])))
        weighted_frequencies.append(weighted_birkhoff_mean(increments))
        ordinary_frequencies.append(float(np.mean(increments)))
    if len(mean_actions) < 4:
        return {
            "status": "insufficient_circulating_orbits",
            "circulating_orbits": len(mean_actions),
        }
    coefficients = np.polyfit(mean_actions, weighted_frequencies, 2)
    prediction = np.polyval(coefficients, mean_actions)
    return {
        "status": "available",
        "circulating_orbits": len(mean_actions),
        "polynomial_coefficients_descending": coefficients.tolist(),
        "weighted_frequencies": weighted_frequencies,
        "ordinary_frequencies": ordinary_frequencies,
        "mean_actions": mean_actions,
        "weighted_vs_ordinary_max_difference": float(
            np.max(
                np.abs(
                    np.asarray(weighted_frequencies)
                    - np.asarray(ordinary_frequencies)
                )
            )
        ),
        "fit_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        prediction
                        - np.asarray(weighted_frequencies, dtype=np.float64)
                    )
                    ** 2
                )
            )
        ),
    }


def _resonance_crossing(
    frequency_coefficients: Sequence[float],
    *,
    order: int,
    band: tuple[float, float],
) -> tuple[bool, int, float]:
    coefficients = np.asarray(frequency_coefficients, dtype=np.float64)
    middle = 0.5 * (band[0] + band[1])
    resonance_index = max(
        1,
        int(round(order * np.polyval(coefficients, middle) / (2.0 * np.pi))),
    )
    target = 2.0 * np.pi * resonance_index
    values = order * np.polyval(coefficients, np.asarray(band)) - target
    crossing = bool(values[0] == 0.0 or values[1] == 0.0 or values[0] * values[1] < 0)
    roots = np.roots(
        order * coefficients
        - np.asarray((0.0, 0.0, target), dtype=np.float64)
    )
    real_roots = [
        float(root.real)
        for root in roots
        if abs(root.imag) < 1e-8 and band[0] <= root.real <= band[1]
    ]
    location = real_roots[0] if real_roots else float("nan")
    return crossing, resonance_index, location


def _rotate_coefficient(coefficient: complex, offset: float, order: int) -> complex:
    """Express a coefficient fitted at ``phi+offset`` in the ``phi`` gauge."""

    sine = coefficient.real
    cosine = coefficient.imag
    phase = order * offset
    return complex(
        sine * np.cos(phase) - cosine * np.sin(phase),
        sine * np.sin(phase) + cosine * np.cos(phase),
    )


def _per_bin_spectra(
    action: np.ndarray,
    angle: np.ndarray,
    *,
    order: int,
    band: tuple[float, float],
    bins: int,
    max_order: int,
    reference_angle: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    current_action = action[:, :-1].reshape(-1)
    current_angle = angle[:, :-1].reshape(-1)
    delta_action = (action[:, 1:] - action[:, :-1]).reshape(-1)
    reference = (
        np.asarray(reference_angle, dtype=np.float64)[:, :-1].reshape(-1)
        if reference_angle is not None
        else None
    )
    edges = np.linspace(band[0], band[1], bins + 1)
    rows = []
    for index in range(bins):
        mask = (current_action >= edges[index]) & (
            current_action < edges[index + 1]
        )
        if int(mask.sum()) < 2 * max_order + 2:
            continue
        estimate = _fit_complex_spectrum(
            current_angle[mask],
            delta_action[mask],
            order=order,
            max_order=max_order,
        )
        alignment = 0.0
        coefficient = estimate.coefficient
        if reference is not None:
            alignment = float(
                np.angle(
                    np.mean(
                        np.exp(
                            1j * (current_angle[mask] - reference[mask])
                        )
                    )
                )
            )
            coefficient = _rotate_coefficient(coefficient, alignment, order)
        rows.append(
            {
                "index": index,
                "center": 0.5 * (edges[index] + edges[index + 1]),
                "lower": edges[index],
                "upper": edges[index + 1],
                "coefficient": coefficient,
                "standard_error": estimate.standard_error,
                "normalized_remainder": estimate.normalized_remainder,
                "condition_number": estimate.condition_number,
                "sample_count": estimate.sample_count,
                "angular_resultant": estimate.angular_resultant,
                "alignment_offset": alignment,
            }
        )
    return rows


def _band_regression(
    rows: list[dict[str, Any]],
    frequency_coefficients: Sequence[float],
    *,
    order: int,
    band: tuple[float, float],
    chart_degree: int = 1,
    weighted: bool = True,
) -> dict[str, Any]:
    crossing, resonance_index, resonance_action = _resonance_crossing(
        frequency_coefficients,
        order=order,
        band=band,
    )
    if not crossing:
        return {
            "verdict": "no_resonance_crossing",
            "resonance_index": resonance_index,
            "resonance_action": resonance_action,
        }
    minimum_count = 8 * (2 * 8 + 1)
    usable = [
        row
        for row in rows
        if row["sample_count"] >= minimum_count
        and row["condition_number"] <= 10.0
        and row["angular_resultant"] <= 0.85
        and row["normalized_remainder"] <= 0.8
    ]
    unknowns = 2 + 2 * (chart_degree + 1)
    if len(usable) * 2 < unknowns + 2:
        return {
            "verdict": "insufficient_coverage",
            "resonance_index": resonance_index,
            "resonance_action": resonance_action,
            "usable_bins": len(usable),
        }
    center = float(np.mean([row["center"] for row in usable]))
    design_rows = []
    targets = []
    weights = []
    for row in usable:
        multiplier = np.exp(
            1j
            * order
            * np.polyval(frequency_coefficients, row["center"])
        ) - 1.0
        powers = [
            (row["center"] - center) ** degree
            for degree in range(chart_degree + 1)
        ]
        real_row = [1.0, 0.0]
        imag_row = [0.0, 1.0]
        for power in powers:
            real_row.extend(
                (power * multiplier.real, -power * multiplier.imag)
            )
            imag_row.extend(
                (power * multiplier.imag, power * multiplier.real)
            )
        design_rows.extend((real_row, imag_row))
        targets.extend((row["coefficient"].real, row["coefficient"].imag))
        weight = (
            1.0 / max(float(row["standard_error"]), 1e-10)
            if weighted
            else 1.0
        )
        weights.extend((weight, weight))
    design = np.asarray(design_rows, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    weighted_design = design * weight_array[:, None]
    weighted_target = target * weight_array
    solution, *_ = np.linalg.lstsq(
        weighted_design,
        weighted_target,
        rcond=None,
    )
    condition = float(np.linalg.cond(weighted_design))
    coefficient = complex(float(solution[0]), float(solution[1]))
    prediction = design @ solution
    remainder = target - prediction
    return {
        "verdict": "value" if condition <= 25.0 else "ill_conditioned",
        "coefficient": coefficient,
        "condition_number": condition,
        "resonance_index": resonance_index,
        "resonance_action": resonance_action,
        "usable_bins": len(usable),
        "total_bins": len(rows),
        "chart_degree": chart_degree,
        "weighted": weighted,
        "weighted_residual_rms": float(
            np.sqrt(
                np.mean(
                    np.square(
                        (target - prediction)
                        * weight_array
                    )
                )
            )
        ),
        "unweighted_residual_rms": float(np.sqrt(np.mean(remainder**2))),
    }


def estimate_resonant_block(
    models: Sequence[CanonicalKoopmanModel | CanonicalKoopmanNetwork],
    states: np.ndarray,
    *,
    order: int,
    band: tuple[float, float],
    bins: int = 14,
    max_order: int = 8,
    reference_actions: np.ndarray | None = None,
    reference_angles: np.ndarray | None = None,
) -> dict[str, Any]:
    """Estimate one resonant block from shared trajectory transitions.

    With oracle coordinates this returns truth-aligned synthetic evidence.
    Without them, charts are aligned to the first chart and the floor remains
    a pairwise lower bound; the function does not claim calibrated accuracy.
    """

    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError("states must have shape (trajectory, time, 2)")
    if len(models) < 1:
        return {
            "order": order,
            "band": list(band),
            "oracle_alignment": reference_angles is not None,
            "floor_multiplicative_basis": (
                "oracle"
                if reference_angles is not None
                else "pairwise_lower_bound"
            ),
            "charts": [],
            "successful_chart_count": 0,
            "consensus_coefficient": None,
            "componentwise_iqr": None,
            "reference_action_available": reference_actions is not None,
        }
    resolved_networks = [
        model.network if isinstance(model, CanonicalKoopmanModel) else model
        for model in models
    ]
    chart_arrays = []
    for network in resolved_networks:
        with torch.no_grad():
            tensor = torch.tensor(values, dtype=torch.float32)
            action = network.action(tensor).numpy().astype(np.float64)
            angle = network.angle(tensor).numpy().astype(np.float64)
        chart_arrays.append((action, angle))
    alignment_reference = (
        np.asarray(reference_angles, dtype=np.float64)
        if reference_angles is not None
        else chart_arrays[0][1]
    )
    results = []
    for index, (network, (action, angle)) in enumerate(
        zip(resolved_networks, chart_arrays, strict=True)
    ):
        profile = _frequency_profile(action, angle)
        if profile["status"] != "available":
            results.append(
                {
                    "chart": index,
                    "verdict": "insufficient_circulating_orbits",
                }
            )
            continue
        rows = _per_bin_spectra(
            action,
            angle,
            order=order,
            band=band,
            bins=bins,
            max_order=max_order,
            reference_angle=alignment_reference,
        )
        learned_coefficients = (
            network.hamiltonian.frequency_coefficients()
            .detach()
            .numpy()
            .astype(np.float64)[::-1]
        )
        estimate = _band_regression(
            rows,
            profile["polynomial_coefficients_descending"],
            order=order,
            band=band,
        )
        learned_h_estimate = _band_regression(
            rows,
            learned_coefficients,
            order=order,
            band=band,
        )
        learned_frequency = []
        if rows:
            with torch.no_grad():
                learned_frequency = (
                    network.hamiltonian.frequency(
                        torch.tensor(
                            [row["center"] for row in rows],
                            dtype=torch.float32,
                        )
                    )
                    .numpy()
                    .astype(np.float64)
                    .tolist()
                )
        results.append(
            {
                "chart": index,
                "verdict": estimate["verdict"],
                "frequency_profile": profile,
                "learned_frequency_at_bin_centers": learned_frequency,
                "bins": rows,
                "estimate": estimate,
                "learned_h_estimate": learned_h_estimate,
            }
        )
    successful = [
        row
        for row in results
        if row["verdict"] == "value"
    ]
    coefficients = np.asarray(
        [
            [
                row["estimate"]["coefficient"].real,
                row["estimate"]["coefficient"].imag,
            ]
            for row in successful
        ],
        dtype=np.float64,
    )
    consensus = (
        complex(*np.median(coefficients, axis=0))
        if len(coefficients)
        else None
    )
    spread = (
        float(
            np.linalg.norm(
                np.percentile(coefficients, 75, axis=0)
                - np.percentile(coefficients, 25, axis=0)
            )
        )
        if len(coefficients)
        else None
    )
    return {
        "order": order,
        "band": list(band),
        "oracle_alignment": reference_angles is not None,
        "floor_multiplicative_basis": (
            "oracle" if reference_angles is not None else "pairwise_lower_bound"
        ),
        "charts": results,
        "successful_chart_count": len(successful),
        "consensus_coefficient": consensus,
        "componentwise_iqr": spread,
        "reference_action_available": reference_actions is not None,
    }


def _fixture(
    config: MetrologyConfig,
    *,
    kick_scale: float,
    initial_actions: np.ndarray,
    initial_angles: np.ndarray,
) -> tuple[TwistKickMap, ObservationChart, MapTrajectoryBundle]:
    kick = KickHarmonic(
        config.target_order,
        config.kick_amplitude * kick_scale,
        config.kick_phase,
    )
    system = TwistKickMap(
        base_frequency=config.base_frequency,
        twist=config.twist,
        kicks=(kick,) if kick_scale != 0.0 else (),
    )
    chart = ObservationChart()
    bundle = simulate_map_trajectories(
        system,
        chart,
        initial_actions=initial_actions,
        initial_angles=initial_angles,
        steps=config.steps,
    )
    return system, chart, bundle


def _model_label(seed: int, architecture: ArchitectureSpec) -> str:
    return f"seed-{seed}-{architecture.label}"


def _observed_frequency_initialization(
    states: np.ndarray,
    trajectory_indices: np.ndarray,
    *,
    degree: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Seed the rotation law from trajectory-level raw polar increments.

    This uses only observed states from the training split. Averaging the unit
    phase increments around each orbit suppresses periodic chart distortion
    while retaining the winding rate. The initializer is deliberately not an
    estimator of the final canonical action or residual coefficient.
    """

    selected = np.asarray(states, dtype=np.float64)[trajectory_indices]
    if selected.ndim != 3 or selected.shape[-1] != 2:
        raise ValueError("states must have shape (trajectory, time, 2)")
    if degree < 1:
        raise ValueError("frequency degree must be positive")
    q, p = selected[..., 0], selected[..., 1]
    raw_action = 0.5 * (q * q + p * p)
    raw_angle = np.arctan2(-p, q)
    increments = wrap_angle(np.diff(raw_angle, axis=1))
    phasors = np.mean(np.exp(1j * increments), axis=1)
    concentration = np.abs(phasors)
    frequencies = np.angle(phasors)
    mean_action = raw_action[:, :-1].mean(axis=1)
    order = np.argsort(mean_action)
    unwrapped_frequency = np.unwrap(frequencies[order])
    fit_degree = min(degree - 1, len(mean_action) - 1)
    descending = np.polyfit(
        mean_action[order],
        unwrapped_frequency,
        fit_degree,
    )
    prediction = np.polyval(descending, mean_action[order])
    coefficients = np.zeros(degree, dtype=np.float64)
    coefficients[: fit_degree + 1] = descending[::-1]
    sampled_prediction = np.polynomial.polynomial.polyval(
        mean_action,
        coefficients,
    )
    if coefficients[0] <= 1e-4 or np.any(sampled_prediction <= 0.0):
        raise ValueError(
            "observed winding is nonpositive or crosses the angular branch; "
            "this profile requires positive sub-Nyquist phase advance"
        )
    return coefficients, {
        "method": "training_orbit_circular_mean_raw_polar_increment",
        "uses_oracle_coordinates": False,
        "trajectory_count": int(len(mean_action)),
        "polynomial_degree": int(fit_degree),
        "frequency_coefficients_ascending": coefficients.tolist(),
        "orbit_fit_rmse_radians_per_step": float(
            np.sqrt(np.mean(np.square(prediction - unwrapped_frequency)))
        ),
        "minimum_circular_concentration": float(np.min(concentration)),
        "maximum_circular_concentration": float(np.max(concentration)),
        "claim_boundary": (
            "Optimization initializer only; not a canonical action, frequency "
            "certificate, or residual estimate."
        ),
    }


def _train_chart(
    dataset: TrajectoryDataset,
    *,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    architecture: ArchitectureSpec,
    seed: int,
    epochs: int,
    model_path: Path,
) -> tuple[CanonicalKoopmanModel, dict[str, Any]]:
    _set_seed(seed)
    training = dataset.states[train_indices].reshape(-1, 2)
    config = replace(
        CanonicalExperimentConfig.quick(seed),
        epochs=epochs,
        batch_size=512,
        rollout_horizon=4,
        hidden_dim=architecture.hidden_dim,
        shear_layers=architecture.shear_layers,
    )
    network = CanonicalKoopmanNetwork(
        dt=1.0,
        hidden_dim=architecture.hidden_dim,
        shear_layers=architecture.shear_layers,
        hamiltonian_degree=3,
        initial_center=tuple(float(value) for value in training.mean(axis=0)),
    )
    initial_frequency, initialization = _observed_frequency_initialization(
        dataset.states,
        train_indices,
        degree=network.hamiltonian.degree,
    )
    base_frequency = float(initial_frequency[0])
    inverse_softplus = float(np.log(np.expm1(base_frequency - 1e-4)))
    with torch.no_grad():
        network.hamiltonian.raw_base_frequency.fill_(inverse_softplus)
        network.hamiltonian.higher_frequency_coefficients.copy_(
            torch.tensor(initial_frequency[1:], dtype=torch.float32)
        )
    started = time.perf_counter()
    history = _fit_network(network, dataset, train_indices, config)
    network.eval()
    scale = training.std(axis=0)
    training_metrics, _ = _rollout_metrics(
        network,
        dataset.states,
        train_indices,
        scale,
    )
    held_out_metrics, _ = _rollout_metrics(
        network,
        dataset.states,
        test_indices,
        scale,
    )
    with torch.no_grad():
        training_action = (
            network.action(
                torch.tensor(dataset.states[train_indices], dtype=torch.float32)
            )
            .mean(dim=1)
            .numpy()
        )
    padding = max(float(np.std(training_action)) * 0.15, 1e-6)
    fit_supported = (
        held_out_metrics["normalized_one_step_rmse"] < 0.25
        and held_out_metrics["normalized_one_step_rmse"]
        < held_out_metrics["persistence_normalized_rollout_rmse"]
    )
    model = CanonicalKoopmanModel(
        network=network,
        state_columns=("position", "momentum"),
        action_min=max(0.0, float(np.min(training_action)) - padding),
        action_max=float(np.max(training_action)) + padding,
        certificate_status=(
            "supported_on_held_out_trajectories"
            if fit_supported
            else "fit_not_supported"
        ),
    )
    save_canonical_model(model_path, model)
    return model, {
        "seed": seed,
        "architecture": asdict(architecture),
        "training": training_metrics,
        "held_out": held_out_metrics,
        "training_seconds": time.perf_counter() - started,
        "frequency_initialization": initialization,
        "model_fit_status": model.certificate_status,
        "history_final": history[-1],
        "model": str(model_path.name),
        "model_sha256": _sha256(model_path),
    }


def _fit_gauged_rotation(
    network: CanonicalKoopmanNetwork,
    states: np.ndarray,
    gauge: ExactGauge,
) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.tensor(states, dtype=torch.float32)
        action = network.action(tensor).numpy().astype(np.float64)
        angle = network.angle(tensor).numpy().astype(np.float64)
    transformed_action, transformed_angle = gauge.forward(action, angle)
    increments = wrap_angle(
        transformed_angle[:, 1:] - transformed_angle[:, :-1]
    )
    return np.polyfit(
        transformed_action[:, :-1].reshape(-1),
        increments.reshape(-1),
        2,
    )


def _gauged_prediction_error(
    network: CanonicalKoopmanNetwork,
    train_states: np.ndarray,
    held_out_states: np.ndarray,
    gauge: ExactGauge,
    state_scale: np.ndarray,
) -> float:
    frequency = _fit_gauged_rotation(network, train_states, gauge)
    current = held_out_states[:, :-1]
    truth = held_out_states[:, 1:]
    with torch.no_grad():
        tensor = torch.tensor(current, dtype=torch.float32)
        latent = network.encode(tensor).numpy().astype(np.float64)
    q, p = latent[..., 0], latent[..., 1]
    action = 0.5 * (q * q + p * p)
    angle = np.arctan2(-p, q)
    transformed_action, transformed_angle = gauge.forward(action, angle)
    predicted_angle = wrap_angle(
        transformed_angle + np.polyval(frequency, transformed_action)
    )
    rebuilt_action, rebuilt_angle = gauge.inverse(
        transformed_action,
        predicted_angle,
    )
    radius = np.sqrt(2.0 * np.maximum(rebuilt_action, 1e-12))
    predicted_latent = np.stack(
        (
            radius * np.cos(rebuilt_angle),
            -radius * np.sin(rebuilt_angle),
        ),
        axis=-1,
    )
    with torch.no_grad():
        prediction = (
            network.decode(
                torch.tensor(predicted_latent, dtype=torch.float32)
            )
            .numpy()
            .astype(np.float64)
        )
    return float(
        np.sqrt(np.mean(np.square((prediction - truth) / state_scale)))
    )


def _gauged_block(
    network: CanonicalKoopmanNetwork,
    states: np.ndarray,
    gauge: ExactGauge,
    *,
    order: int,
    band: tuple[float, float],
    bins: int,
    max_order: int,
    reference_angle: np.ndarray,
) -> dict[str, Any]:
    with torch.no_grad():
        tensor = torch.tensor(states, dtype=torch.float32)
        action = network.action(tensor).numpy().astype(np.float64)
        angle = network.angle(tensor).numpy().astype(np.float64)
    transformed_action, transformed_angle = gauge.forward(action, angle)
    profile = _frequency_profile(transformed_action, transformed_angle)
    if profile["status"] != "available":
        return {"verdict": profile["status"]}
    rows = _per_bin_spectra(
        transformed_action,
        transformed_angle,
        order=order,
        band=band,
        bins=bins,
        max_order=max_order,
        reference_angle=reference_angle,
    )
    return _band_regression(
        rows,
        profile["polynomial_coefficients_descending"],
        order=order,
        band=band,
    )


def _static_error_harmonics(
    learned_action: np.ndarray,
    learned_angle: np.ndarray,
    true_action: np.ndarray,
    true_angle: np.ndarray,
    *,
    target_order: int,
) -> dict[str, float]:
    relative_action_error = (
        learned_action - true_action
    ) / np.maximum(true_action, 1e-8)
    phase_error = wrap_angle(learned_angle - true_angle)
    action_spectrum = _fit_complex_spectrum(
        true_angle.reshape(-1),
        relative_action_error.reshape(-1),
        order=target_order,
        max_order=2 * target_order,
    )
    phase_spectrum = _fit_complex_spectrum(
        true_angle.reshape(-1),
        phase_error.reshape(-1),
        order=2 * target_order,
        max_order=2 * target_order,
    )
    return {
        "relative_action_harmonic_m": abs(action_spectrum.coefficient),
        "angle_error_harmonic_2m": abs(phase_spectrum.coefficient),
    }


def _circle_probe(
    network: CanonicalKoopmanNetwork,
    system: TwistKickMap,
    observation: ObservationChart,
    *,
    action: float,
    order: int,
    max_order: int,
    samples: int = 4096,
) -> dict[str, Any]:
    """Measure one chart on a uniformly sampled oracle circle.

    This is synthetic evaluation evidence only. The chart receives observed
    states; oracle angle is used afterward solely to align the complex phase.
    """

    reference_angle = np.linspace(-np.pi, np.pi, samples, endpoint=False)
    reference_action = np.full(samples, action, dtype=np.float64)
    next_action, next_angle = system.step(reference_action, reference_angle)
    current_states = observation.observe(reference_action, reference_angle)
    next_states = observation.observe(next_action, next_angle)
    with torch.no_grad():
        current = torch.tensor(current_states, dtype=torch.float32)
        following = torch.tensor(next_states, dtype=torch.float32)
        learned_action = network.action(current).numpy().astype(np.float64)
        learned_next_action = (
            network.action(following).numpy().astype(np.float64)
        )
        learned_angle = network.angle(current).numpy().astype(np.float64)
    estimate = _fit_complex_spectrum(
        learned_angle,
        learned_next_action - learned_action,
        order=order,
        max_order=max_order,
    )
    alignment = float(
        np.angle(np.mean(np.exp(1j * (learned_angle - reference_angle))))
    )
    coefficient = _rotate_coefficient(
        estimate.coefficient,
        alignment,
        order,
    )
    return {
        "coefficient": coefficient,
        "alignment_offset": alignment,
        "standard_error": estimate.standard_error,
        "normalized_remainder": estimate.normalized_remainder,
        "condition_number": estimate.condition_number,
        "sample_count": estimate.sample_count,
        "uses_oracle_for_phase_alignment": True,
        "claim_boundary": (
            "Uniform-circle synthetic diagnostic; not used to fit or accept "
            "the chart and unavailable on unreferenced measured data."
        ),
    }


def _shuffled_angle_control(
    action: np.ndarray,
    angle: np.ndarray,
    *,
    order: int,
    band: tuple[float, float],
    bins: int,
    max_order: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Permute current angles within action bins while retaining WBA geometry."""

    profile = _frequency_profile(action, angle)
    if profile["status"] != "available":
        return {"verdict": profile["status"], "frequency_profile": profile}
    shuffled = np.asarray(angle, dtype=np.float64).copy()
    current_action = np.asarray(action, dtype=np.float64)[:, :-1].reshape(-1)
    current_angle = shuffled[:, :-1].reshape(-1).copy()
    edges = np.linspace(band[0], band[1], bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        indices = np.flatnonzero(
            (current_action >= lower) & (current_action < upper)
        )
        if len(indices):
            current_angle[indices] = current_angle[rng.permutation(indices)]
    shuffled[:, :-1] = current_angle.reshape(shuffled[:, :-1].shape)
    rows = _per_bin_spectra(
        action,
        shuffled,
        order=order,
        band=band,
        bins=bins,
        max_order=max_order,
    )
    estimate = _band_regression(
        rows,
        profile["polynomial_coefficients_descending"],
        order=order,
        band=band,
    )
    magnitudes = [abs(row["coefficient"]) for row in rows]
    return {
        "verdict": estimate["verdict"],
        "frequency_profile": profile,
        "bins": rows,
        "estimate": estimate,
        "permutation": "current angles within fixed action bins",
        "median_bin_coefficient_magnitude": (
            float(np.median(magnitudes)) if magnitudes else None
        ),
        "maximum_bin_coefficient_magnitude": (
            float(np.max(magnitudes)) if magnitudes else None
        ),
    }


def _complex_payload(value: complex | None) -> list[float] | None:
    return [float(value.real), float(value.imag)] if value is not None else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, complex):
        return _complex_payload(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _plot_report(path: Path, manifest: dict[str, Any]) -> None:
    charts = manifest["ensemble"]["accepted"]
    truth = manifest["oracle"]["kick_amplitude"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    names = [row["label"] for row in charts]
    short_names = [
        name.replace("seed-", "").replace("-h", " / h").replace("-s", " s")
        for name in names
    ]
    one_step = [row["held_out_one_step_rmse"] for row in charts]
    errors = [row["complex_error"] for row in charts]
    axes[0, 0].scatter(one_step, errors, color="#4057c9", s=55)
    for name, x_value, y_value in zip(
        short_names,
        one_step,
        errors,
        strict=True,
    ):
        axes[0, 0].annotate(name, (x_value, y_value), fontsize=8)
    axes[0, 0].axhline(0.2, color="#b23a33", linestyle="--")
    axes[0, 0].set(
        title="Prediction quality vs resonant-block error",
        xlabel="held-out normalized one-step RMSE",
        ylabel="aligned complex relative error",
    )

    if charts:
        coefficients = np.asarray(
            [row["coefficient"] for row in charts],
            dtype=np.float64,
        )
        axes[0, 1].scatter(
            coefficients[:, 0],
            coefficients[:, 1],
            color="#2a8b68",
            s=55,
            label="learned charts",
        )
    oracle = np.asarray(manifest["oracle"]["coefficient"], dtype=np.float64)
    axes[0, 1].scatter(
        [oracle[0]],
        [oracle[1]],
        marker="*",
        s=180,
        color="#d85140",
        label="oracle",
    )
    axes[0, 1].set(
        title="Aligned complex kick coefficient",
        xlabel="sine coefficient",
        ylabel="cosine coefficient",
    )
    axes[0, 1].legend()

    floor = [
        (
            row["floor_total"] / truth
            if row["floor_total"] is not None
            else np.nan
        )
        for row in charts
    ]
    realized = [row["absolute_error"] / truth for row in charts]
    positions = np.arange(len(charts))
    axes[1, 0].bar(
        positions - 0.18,
        realized,
        0.36,
        color="#d85140",
        label="realized error",
    )
    axes[1, 0].bar(
        positions + 0.18,
        np.asarray(floor) * 2.0,
        0.36,
        color="#4057c9",
        label="2 x modeled floor",
    )
    axes[1, 0].set(
        title="Error coverage by the empirical floor model",
        xlabel="estimable chart",
        ylabel="fraction of planted kick",
    )
    for position, row in zip(positions, charts, strict=True):
        if row["floor_total"] is None:
            axes[1, 0].annotate(
                "floor unavailable",
                (position + 0.18, 0.01),
                rotation=90,
                ha="center",
                va="bottom",
                color="#4057c9",
                fontsize=8,
            )
    axes[1, 0].set_xticks(positions, short_names, rotation=25, ha="right")
    axes[1, 0].legend()

    stress = manifest["controls"]["exact_2m_gauge_stress"]
    ladder = stress["ladder"]
    max_shift = [
        max(
            (
                row["complex_shift"]
                for row in stress["per_scale"][str(scale)]
                if row.get("comparable_block", False)
            ),
            default=0.0,
        )
        for scale in ladder
    ]
    in_envelope = [
        any(row["inside_prediction_envelope"] for row in stress["per_scale"][str(scale)])
        for scale in ladder
    ]
    axes[1, 1].plot(ladder, max_shift, marker="o", color="#7c3fb7")
    for scale, shift, inside in zip(ladder, max_shift, in_envelope, strict=True):
        axes[1, 1].annotate(
            "in envelope" if inside else "outside",
            (scale, shift),
        )
    axes[1, 1].axhline(0.2, color="#b23a33", linestyle="--")
    axes[1, 1].set(
        title="Exact 2m gauge identifiability stress",
        xlabel="peak action fraction s",
        ylabel="maximum complex-block shift",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Learned-chart resonance metrology",
        fontsize=16,
        y=0.99,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    accepted = manifest["ensemble"]["accepted"]
    gates = manifest["empirical_gates"]
    initializers = [
        row["frequency_initialization"]
        for row in manifest["ensemble"]["training"]["s1"]
    ]
    initialization_rmse = float(
        np.median(
            [
                row["orbit_fit_rmse_radians_per_step"]
                for row in initializers
            ]
        )
    )
    status_explanations = {
        "resolved_supported": (
            "The planted block survived every frozen recovery, control, "
            "stability, and exact-gauge gate on this synthetic fixture."
        ),
        "resolved_refuted": (
            "A frozen falsifier failed by the predeclared decisive margin. "
            "The claimed residual precision is refuted on this fixture."
        ),
        "not_resolved_abstained": (
            "The instrument withheld the coefficient because a support, "
            "control, truncation, or gauge condition was unresolved."
        ),
        "invalid_ensemble": (
            "Too few learned charts passed the prediction precondition, so "
            "no residual coefficient was interpreted."
        ),
    }
    consensus = manifest["ensemble_consensus"]
    oracle = manifest["oracle"]
    stress = manifest["controls"]["exact_2m_gauge_stress"]
    visible_scale = stress["smallest_visible_scale"]
    visible_text = (
        f"{visible_scale:.3g}" if visible_scale is not None else "not observed"
    )
    variant_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{row['chart_count']}</td>"
        f"<td>{html.escape(str(row['consensus']))}</td>"
        f"<td>{html.escape(str(row['relative_deviation_from_primary']))}</td>"
        "</tr>"
        for name, row in manifest["controls"]["variant_stability"][
            "variants"
        ].items()
    )
    gate_rows = "\n".join(
        f"<li class=\"{'pass' if row['passed'] else 'fail'}\">"
        f"{'PASS' if row['passed'] else 'FAIL'} — "
        f"{html.escape(name.replace('_', ' '))}: "
        f"{html.escape(str(row['value']))}</li>"
        for name, row in gates.items()
    )
    chart_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['label'])}</td>"
        f"<td>{row['held_out_one_step_rmse']:.5f}</td>"
        f"<td>{row['complex_error']:.2%}</td>"
        f"<td>{row['magnitude_error']:.2%}</td>"
        f"<td>{row['floor_total']:.3e}</td>"
        f"<td>{'yes' if row['covered'] else 'no'}</td>"
        "</tr>"
        for row in accepted
        if row["floor_available"]
    )
    unavailable_floor_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['label'])}</td>"
        f"<td>{row['held_out_one_step_rmse']:.5f}</td>"
        f"<td>{row['complex_error']:.2%}</td>"
        f"<td>{row['magnitude_error']:.2%}</td>"
        "<td>unavailable</td><td>not evaluated</td>"
        "</tr>"
        for row in accepted
        if not row["floor_available"]
    )
    block_abstention_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['label'])}</td>"
        f"<td>{html.escape(row['verdict'])}</td>"
        f"<td>{html.escape(str(row['condition_number']))}</td>"
        f"<td>{html.escape(str(row['usable_bins']))}</td>"
        "</tr>"
        for row in manifest["ensemble"]["block_ledger"]
        if row["verdict"] != "value"
    )
    shuffled = manifest["controls"]["shuffled_angle_runs"]
    shuffled_abstentions = sum(row["verdict"] != "value" for row in shuffled)
    certification = manifest["ledgers"]["certification"]
    unevaluable_variants = manifest["controls"]["variant_stability"][
        "unevaluable_trigger_variants"
    ]
    path.write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Resonance metrology</title><style>
body {{margin:0;background:#f3f0e8;color:#20242b;font:17px/1.55 system-ui}}
main {{max-width:1050px;margin:auto;padding:52px 24px 80px}}
h1 {{font:700 clamp(40px,7vw,72px)/1.02 Georgia,serif;margin:.2em 0}}
.eyebrow {{color:#4057c9;font-weight:800;letter-spacing:.08em;text-transform:uppercase}}
.status {{display:inline-block;padding:8px 14px;border-radius:99px;background:#e5eaf7;
font-weight:800}} .card {{background:white;border-radius:18px;padding:25px;margin:24px 0;
box-shadow:0 12px 35px #1e263112}} img {{width:100%;border-radius:12px}}
table {{width:100%;border-collapse:collapse}} th,td {{padding:9px;
border-bottom:1px solid #ddd;text-align:right}}
th:first-child,td:first-child {{text-align:left}}
.pass {{color:#176e50}} .fail {{color:#ae352d}} code {{overflow-wrap:anywhere}}
</style></head><body><main>
<div class="eyebrow">learned-koopman · residual normal form</div>
<h1>Can the resonance survive the chart?</h1>
<p class="status">{html.escape(manifest['status'])}</p>
<p>{html.escape(manifest['claim_boundary'])}</p>
<p><strong>{html.escape(status_explanations[manifest['status']])}</strong></p>
<p>The rotation law was initialized from circular-mean raw polar increments
on the training trajectories only (median orbit-fit RMSE
{initialization_rmse:.3g} rad/step). That seed uses no oracle coordinates or
planted map parameters and is recorded separately from the learned result.</p>
<div class="card"><img src="overview.png"
alt="Resonance-metrology diagnostics"></div>
<div class="card"><h2>Five-layer output</h2>
<table><tbody>
<tr><th>prediction-accepted charts</th>
<td>{manifest['ensemble']['accepted_count']} /
{len(manifest['ensemble']['training']['s1'])}</td></tr>
<tr><th>charts with an estimable block</th>
<td>{manifest['ensemble']['estimable_count']}</td></tr>
<tr><th>action-kick coefficient (sine, cosine)</th>
<td>{html.escape(str(consensus['coefficient']))}</td></tr>
<tr><th>generating amplitude</th>
<td>{consensus['generating_function_amplitude']:.6g}</td></tr>
<tr><th>leading island halfwidth</th>
<td>{consensus['island_half_width']:.6g}</td></tr>
<tr><th>synthetic oracle generating amplitude</th>
<td>{oracle['generating_function_amplitude']:.6g}</td></tr>
</tbody></table></div>
<div class="card"><h2>Predeclared gates</h2><ul>{gate_rows}</ul></div>
<div class="card"><h2>Estimable learned charts</h2><table><thead><tr>
<th>chart</th><th>one-step RMSE</th><th>complex error</th>
<th>magnitude error</th><th>floor</th><th>covered</th></tr></thead>
<tbody>{chart_rows}{unavailable_floor_rows}</tbody></table></div>
<div class="card"><h2>Block abstention map</h2>
<p>Prediction acceptance and coefficient estimability are separate gates.
These accepted predictors withheld a primary block:</p>
<table><thead><tr><th>chart</th><th>verdict</th>
<th>condition number</th><th>usable bins</th></tr></thead>
<tbody>{block_abstention_rows}</tbody></table></div>
<div class="card"><h2>Identifiability margin</h2>
<p>Smallest exact-gauge rung leaving the prediction envelope:
<strong>{visible_text}</strong>. Maximum in-envelope complex shift:
<strong>{stress['maximum_in_envelope_complex_shift']}</strong>; magnitude
shift: <strong>{stress['maximum_in_envelope_magnitude_shift']}</strong>.</p>
<p>In-envelope gauge fits without a comparable block:
<strong>{stress['noncomparable_in_envelope_count']}</strong>. A non-comparable
fit forbids support; only a measured comparable shift beyond twice the frozen
tolerance can produce a gauge refutation.</p>
</div>
<div class="card"><h2>Estimator variants</h2><table><thead><tr>
<th>variant</th><th>charts</th><th>consensus</th>
<th>relative deviation</th></tr></thead><tbody>{variant_rows}</tbody></table>
<p>Every trigger variant must be evaluable before G9 can pass. Unevaluable
triggers: {html.escape(str(unevaluable_variants))}.</p>
</div>
<div class="card"><h2>Control interpretation</h2>
<p>All {shuffled_abstentions} / {len(shuffled)} shuffled trajectory-band fits
abstained. The stricter retained-bin statistic was
{manifest['controls']['shuffled_angle_over_truth']:.2%} of the planted
coefficient, against a 20% limit; this is recorded separately so abstention
cannot pass the shuffle control vacuously.</p>
<p>Certification ledger: formal guarantees =
<strong>{html.escape(certification['formal_guarantees'])}</strong>.
This is an empirical falsification instrument, not a formal certificate.</p>
</div>
<h2>What the number means</h2>
<p>The primary estimator fits the complex residual harmonic across an action
band that crosses the target resonance. It separates a smooth physical block
from the chart's coboundary-shaped detuning signature. Results are aligned on
shared states, checked against an oracle only in this synthetic reference run,
and stress-tested with exact symplectic chart gauges.</p>
<h2>What it does not mean</h2>
<p>No formal KAM, torus, interval, probability, noise, hardware, or
coordinate-global guarantee is claimed. Static action ripple is a chart
diagnostic, not an island-width floor. The empirical floor model is promoted
only if its predeclared coverage and detection checks pass.</p>
<p>Artifacts: <code>manifest.json</code>, <code>overview.png</code>,
<code>s1-trajectories.csv</code>, <code>s2-null-trajectories.csv</code>, and
the saved chart ensemble under <code>models/</code>.</p>
</main></body></html>
""",
        encoding="utf-8",
    )


def _analyze_coordinate_arrays(
    action: np.ndarray,
    angle: np.ndarray,
    *,
    order: int,
    band: tuple[float, float],
    bins: int,
    max_order: int,
    reference_angle: np.ndarray | None = None,
) -> dict[str, Any]:
    profile = _frequency_profile(action, angle)
    if profile["status"] != "available":
        return {"verdict": profile["status"], "frequency_profile": profile}
    rows = _per_bin_spectra(
        action,
        angle,
        order=order,
        band=band,
        bins=bins,
        max_order=max_order,
        reference_angle=reference_angle,
    )
    estimate = _band_regression(
        rows,
        profile["polynomial_coefficients_descending"],
        order=order,
        band=band,
    )
    return {
        "verdict": estimate["verdict"],
        "frequency_profile": profile,
        "bins": rows,
        "estimate": estimate,
    }


def _consensus(values: Sequence[complex]) -> complex:
    components = np.asarray(
        [[value.real, value.imag] for value in values],
        dtype=np.float64,
    )
    median = np.median(components, axis=0)
    return complex(float(median[0]), float(median[1]))


def _variant_panel(
    accepted_rows: list[dict[str, Any]],
    *,
    order: int,
    band: tuple[float, float],
) -> dict[str, Any]:
    variants: dict[str, list[complex]] = {
        "primary_linear": [],
        "constant_chart": [],
        "quadratic_chart": [],
        "inner_band": [],
        "learned_h_frequency": [],
        "unweighted": [],
    }
    for row in accepted_rows:
        chart = row["analysis"]
        bins = chart["bins"]
        wba_coefficients = chart["frequency_profile"][
            "polynomial_coefficients_descending"
        ]
        primary = chart["estimate"]
        if primary["verdict"] != "value":
            continue
        variants["primary_linear"].append(primary["coefficient"])
        constant = _band_regression(
            bins,
            wba_coefficients,
            order=order,
            band=band,
            chart_degree=0,
        )
        quadratic = _band_regression(
            bins,
            wba_coefficients,
            order=order,
            band=band,
            chart_degree=2,
        )
        inner_rows = [
            item for item in bins if 0.89 <= item["center"] <= 2.41
        ]
        inner = _band_regression(
            inner_rows,
            wba_coefficients,
            order=order,
            band=(0.89, 2.41),
        )
        learned_h = chart["learned_h_estimate"]
        unweighted = _band_regression(
            bins,
            wba_coefficients,
            order=order,
            band=band,
            weighted=False,
        )
        for name, estimate in (
            ("constant_chart", constant),
            ("quadratic_chart", quadratic),
            ("inner_band", inner),
            ("learned_h_frequency", learned_h),
            ("unweighted", unweighted),
        ):
            if estimate["verdict"] == "value":
                variants[name].append(estimate["coefficient"])
    payload = {}
    primary_consensus = _consensus(variants["primary_linear"])
    for name, values in variants.items():
        consensus = _consensus(values) if values else None
        deviation = (
            abs(consensus - primary_consensus)
            / max(abs(primary_consensus), 1e-12)
            if consensus is not None
            else None
        )
        payload[name] = {
            "chart_count": len(values),
            "consensus": consensus,
            "relative_deviation_from_primary": deviation,
        }
    trigger_names = (
        "primary_linear",
        "quadratic_chart",
        "inner_band",
        "learned_h_frequency",
    )
    unevaluable = [
        name
        for name in trigger_names
        if payload[name]["relative_deviation_from_primary"] is None
    ]
    evaluated_deviations = [
        float(payload[name]["relative_deviation_from_primary"])
        for name in trigger_names
        if payload[name]["relative_deviation_from_primary"] is not None
    ]
    maximum = max(evaluated_deviations, default=1e9)
    return {
        "variants": payload,
        "maximum_trigger_deviation": maximum,
        "all_trigger_variants_evaluable": not unevaluable,
        "unevaluable_trigger_variants": unevaluable,
        "blocks_supported_status": not unevaluable and maximum <= 0.2,
        "warning": bool(unevaluable) or maximum > 0.1,
    }


def run_resonance_metrology(
    config: MetrologyConfig,
) -> dict[str, Any]:
    """Train chart ensembles and run the frozen resonance-metrology protocol."""

    if config.profile not in {"ci", "full"}:
        raise ValueError("profile must be 'ci' or 'full'")
    if config.target_order < 1 or config.max_order < config.target_order:
        raise ValueError("invalid harmonic orders")
    if config.bins < 6 or config.epochs < 1:
        raise ValueError("bins and epochs must be positive")
    if len(config.seeds) * len(config.architectures) < 6:
        raise ValueError("resonance metrology requires at least six charts")
    started = time.perf_counter()
    config.output.mkdir(parents=True, exist_ok=True)
    model_dir = config.output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.split_seed)
    rng = np.random.default_rng(config.split_seed)
    initial_actions = np.linspace(
        config.action_band[0] + 0.05,
        config.action_band[1] - 0.05,
        config.trajectories,
    )
    initial_angles = rng.uniform(-np.pi, np.pi, config.trajectories)
    system, observation, s1_bundle = _fixture(
        config,
        kick_scale=1.0,
        initial_actions=initial_actions,
        initial_angles=initial_angles,
    )
    _, _, s2_bundle = _fixture(
        config,
        kick_scale=0.0,
        initial_actions=initial_actions,
        initial_angles=initial_angles,
    )
    s1_csv = write_map_trajectory_csv(
        config.output / "s1-trajectories.csv",
        s1_bundle,
    )
    s2_csv = write_map_trajectory_csv(
        config.output / "s2-null-trajectories.csv",
        s2_bundle,
    )
    s1_dataset = load_trajectory_csv(
        s1_csv,
        state_columns=("position", "momentum"),
    )
    s2_dataset = load_trajectory_csv(
        s2_csv,
        state_columns=("position", "momentum"),
    )
    train_indices, test_indices = _split_indices(
        config.trajectories,
        0.75,
        config.split_seed,
    )
    trained: dict[str, dict[str, CanonicalKoopmanModel]] = {"s1": {}, "s2": {}}
    training_rows: dict[str, list[dict[str, Any]]] = {"s1": [], "s2": []}
    for system_name, dataset in (("s1", s1_dataset), ("s2", s2_dataset)):
        for architecture in config.architectures:
            for seed in config.seeds:
                label = _model_label(seed, architecture)
                model, metrics = _train_chart(
                    dataset,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    architecture=architecture,
                    seed=seed,
                    epochs=config.epochs,
                    model_path=model_dir / f"{system_name}-{label}.pt",
                )
                trained[system_name][label] = model
                metrics["label"] = label
                training_rows[system_name].append(metrics)
    best_error = min(
        row["held_out"]["normalized_one_step_rmse"]
        for row in training_rows["s1"]
    )
    acceptance_limit = 1.5 * best_error
    absolute_acceptance_limit = 0.25
    accepted_labels = []
    for row in training_rows["s1"]:
        held_out = row["held_out"]
        relative_quality = (
            held_out["normalized_one_step_rmse"] <= acceptance_limit
        )
        absolute_quality = (
            held_out["normalized_one_step_rmse"]
            < absolute_acceptance_limit
            and held_out["normalized_one_step_rmse"]
            < held_out["persistence_normalized_rollout_rmse"]
        )
        if relative_quality and (
            absolute_quality or config.profile != "full"
        ):
            accepted_labels.append(row["label"])
    dropped = [
        {
            "label": row["label"],
            "held_out_one_step_rmse": row["held_out"][
                "normalized_one_step_rmse"
            ],
            "relative_limit": acceptance_limit,
            "absolute_limit": absolute_acceptance_limit,
            "beats_persistence": (
                row["held_out"]["normalized_one_step_rmse"]
                < row["held_out"]["persistence_normalized_rollout_rmse"]
            ),
        }
        for row in training_rows["s1"]
        if row["label"] not in accepted_labels
    ]
    ensemble_healthy = len(accepted_labels) >= config.minimum_accepted_charts
    accepted_models = [trained["s1"][label] for label in accepted_labels]
    held_out_states = s1_bundle.states[test_indices]
    held_out_actions = s1_bundle.actions[test_indices]
    held_out_angles = s1_bundle.angles[test_indices]
    primary = estimate_resonant_block(
        accepted_models,
        held_out_states,
        order=config.target_order,
        band=config.action_band,
        bins=config.bins,
        max_order=config.max_order,
        reference_actions=held_out_actions,
        reference_angles=held_out_angles,
    )
    truth = config.kick_amplitude * np.exp(1j * config.kick_phase)
    oracle = _analyze_coordinate_arrays(
        held_out_actions,
        held_out_angles,
        order=config.target_order,
        band=config.action_band,
        bins=config.bins,
        max_order=config.max_order,
        reference_angle=held_out_angles,
    )
    raw_action = 0.5 * np.square(held_out_states).sum(axis=-1)
    raw_angle = np.arctan2(
        -held_out_states[..., 1],
        held_out_states[..., 0],
    )
    raw = _analyze_coordinate_arrays(
        raw_action,
        raw_angle,
        order=config.target_order,
        band=config.action_band,
        bins=config.bins,
        max_order=config.max_order,
        reference_angle=held_out_angles,
    )
    primary_by_label = {
        label: row
        for label, row in zip(
            accepted_labels,
            primary["charts"],
            strict=True,
        )
    }
    block_ledger = []
    for label in accepted_labels:
        chart_result = primary_by_label[label]
        estimate = chart_result.get(
            "estimate",
            {"verdict": chart_result["verdict"]},
        )
        block_ledger.append(
            {
                "label": label,
                "verdict": estimate["verdict"],
                "condition_number": estimate.get("condition_number"),
                "usable_bins": estimate.get("usable_bins"),
                "total_bins": estimate.get("total_bins"),
            }
        )
    null_results: dict[str, dict[str, Any]] = {}
    accepted_rows = []
    training_by_label = {
        row["label"]: row for row in training_rows["s1"]
    }
    for label in accepted_labels:
        model = trained["s1"][label]
        chart_result = primary_by_label[label]
        null = estimate_resonant_block(
            [trained["s2"][label]],
            s2_bundle.states[test_indices],
            order=config.target_order,
            band=config.action_band,
            bins=config.bins,
            max_order=config.max_order,
            reference_actions=s2_bundle.actions[test_indices],
            reference_angles=s2_bundle.angles[test_indices],
        )
        null_results[label] = null
        estimate = chart_result.get(
            "estimate",
            {"verdict": chart_result["verdict"]},
        )
        if estimate["verdict"] != "value":
            continue
        coefficient = estimate["coefficient"]
        null_coefficient = (
            null["charts"][0]["estimate"]["coefficient"]
            if null["charts"][0]["verdict"] == "value"
            else None
        )
        with torch.no_grad():
            tensor = torch.tensor(held_out_states, dtype=torch.float32)
            learned_action = model.network.action(tensor).numpy()
            learned_angle = model.network.angle(tensor).numpy()
        harmonics = _static_error_harmonics(
            learned_action,
            learned_angle,
            held_out_actions,
            held_out_angles,
            target_order=config.target_order,
        )
        floor_additive = (
            abs(null_coefficient) if null_coefficient is not None else None
        )
        floor_second_order = (
            2.34
            * harmonics["relative_action_harmonic_m"] ** 2
            * abs(coefficient)
        )
        floor_multiplicative = (
            2.3
            * harmonics["angle_error_harmonic_2m"]
            * abs(coefficient)
        )
        floor_total = (
            floor_additive + floor_second_order + floor_multiplicative
            if floor_additive is not None
            else None
        )
        absolute_error = abs(coefficient - truth)
        frequency_coefficients = (
            model.network.hamiltonian.frequency_coefficients()
            .detach()
            .numpy()
            .astype(np.float64)[::-1]
        )
        circle_probe = _circle_probe(
            model.network,
            system,
            observation,
            action=system.resonance_action(config.target_order),
            order=config.target_order,
            max_order=config.max_order,
        )
        circle_coefficient = circle_probe["coefficient"]
        circle_error = abs(circle_coefficient - truth) / abs(truth)
        accepted_rows.append(
            {
                "label": label,
                "coefficient": _complex_payload(coefficient),
                "held_out_one_step_rmse": training_by_label[label][
                    "held_out"
                ]["normalized_one_step_rmse"],
                "complex_error": absolute_error / abs(truth),
                "magnitude_error": abs(abs(coefficient) - abs(truth))
                / abs(truth),
                "location_error_radians": abs(np.angle(coefficient / truth))
                / config.target_order,
                "circle_probe": {
                    **circle_probe,
                    "coefficient": _complex_payload(circle_coefficient),
                    "complex_error": circle_error,
                },
                "absolute_error": absolute_error,
                "floor_additive": floor_additive,
                "floor_second_order": floor_second_order,
                "floor_multiplicative": floor_multiplicative,
                "floor_total": floor_total,
                "floor_available": floor_total is not None,
                "covered": (
                    absolute_error <= 2.0 * floor_total
                    if floor_total is not None
                    else None
                ),
                "chart_error_harmonics": harmonics,
                "analysis": chart_result,
                "frequency_coefficients_descending": (
                    frequency_coefficients.tolist()
                ),
            }
        )
    successful_coefficients = [
        complex(*row["coefficient"]) for row in accepted_rows
    ]
    consensus = (
        _consensus(successful_coefficients)
        if successful_coefficients
        else 0.0j
    )
    consensus_complex_error = abs(consensus - truth) / abs(truth)
    median_per_chart_complex_error = (
        float(np.median([row["complex_error"] for row in accepted_rows]))
        if accepted_rows
        else 1e9
    )
    charts_above_complex_threshold = sum(
        row["complex_error"] > 0.20 for row in accepted_rows
    )
    consensus_magnitude_error = abs(abs(consensus) - abs(truth)) / abs(truth)
    consensus_location_error = (
        abs(np.angle(consensus / truth)) / config.target_order
    )
    floor_rows = [row for row in accepted_rows if row["floor_available"]]
    coverage_fraction = (
        float(np.mean([row["covered"] for row in floor_rows]))
        if floor_rows
        else 0.0
    )
    modeled_floor = (
        float(np.median([row["floor_total"] for row in floor_rows]))
        if floor_rows
        else config.kick_amplitude
    )
    circle_errors = [
        row["circle_probe"]["complex_error"] for row in accepted_rows
    ]
    circle_median_error = (
        float(np.median(circle_errors)) if circle_errors else 1e9
    )
    trajectory_to_circle_ratio = (
        consensus_complex_error / max(circle_median_error, 1e-12)
    )
    trajectory_vs_circle_passed = (
        bool(circle_errors)
        and (
            consensus_complex_error <= 0.20
            or trajectory_to_circle_ratio <= 2.5
        )
    )
    prototype_circle_transfer_passed = circle_median_error <= 0.02

    wrong_harmonics = {}
    for wrong_order in (2, 4, 6, 8):
        wrong_harmonics[str(wrong_order)] = estimate_resonant_block(
            accepted_models,
            held_out_states,
            order=wrong_order,
            band=config.action_band,
            bins=config.bins,
            max_order=config.max_order,
            reference_actions=held_out_actions,
            reference_angles=held_out_angles,
        )
    off_band = (config.action_band[0], 1.35)
    off_band_control = estimate_resonant_block(
        accepted_models,
        held_out_states,
        order=config.target_order,
        band=off_band,
        bins=config.bins,
        max_order=config.max_order,
        reference_actions=held_out_actions,
        reference_angles=held_out_angles,
    )

    shuffled_results = []
    shuffle_rng = np.random.default_rng(20260728)
    for model in accepted_models:
        with torch.no_grad():
            tensor = torch.tensor(held_out_states, dtype=torch.float32)
            action = model.network.action(tensor).numpy().astype(np.float64)
            angle = model.network.angle(tensor).numpy().astype(np.float64)
        shuffled_results.append(
            _shuffled_angle_control(
                action,
                angle,
                order=config.target_order,
                band=config.action_band,
                bins=config.bins,
                max_order=config.max_order,
                rng=shuffle_rng,
            )
        )

    state_scale = s1_bundle.states[train_indices].reshape(-1, 2).std(axis=0)
    gauge_ladder = (0.01, 0.02, 0.04, 0.10)
    stress_per_scale: dict[str, list[dict[str, Any]]] = {}
    prediction_envelope = min(
        acceptance_limit,
        absolute_acceptance_limit,
    )
    for scale in gauge_ladder:
        rows = []
        for phase in (0.0, 0.5 * np.pi):
            gauge = ExactGauge(
                amplitude=scale / (2 * config.target_order),
                order=2 * config.target_order,
                phase=phase,
            )
            for label in accepted_labels:
                model = trained["s1"][label]
                prediction_error = _gauged_prediction_error(
                    model.network,
                    s1_bundle.states[train_indices],
                    held_out_states,
                    gauge,
                    state_scale,
                )
                block = _gauged_block(
                    model.network,
                    held_out_states,
                    gauge,
                    order=config.target_order,
                    band=config.action_band,
                    bins=config.bins,
                    max_order=config.max_order,
                    reference_angle=held_out_angles,
                )
                base = primary_by_label[label]["estimate"]
                comparable = (
                    block["verdict"] == "value"
                    and base["verdict"] == "value"
                )
                shift = (
                    abs(block["coefficient"] - base["coefficient"])
                    / max(abs(base["coefficient"]), 1e-12)
                    if comparable
                    else 1e9
                )
                magnitude_shift = (
                    abs(
                        abs(block["coefficient"])
                        - abs(base["coefficient"])
                    )
                    / max(abs(base["coefficient"]), 1e-12)
                    if comparable
                    else 1e9
                )
                rows.append(
                    {
                        "label": label,
                        "phase": phase,
                        "prediction_error": prediction_error,
                        "inside_prediction_envelope": (
                            prediction_error <= prediction_envelope
                        ),
                        "block_verdict": block["verdict"],
                        "comparable_block": comparable,
                        "complex_shift": shift,
                        "magnitude_shift": magnitude_shift,
                    }
                )
        stress_per_scale[str(scale)] = rows
    visible_scales = [
        scale
        for scale in gauge_ladder
        if any(
            not row["inside_prediction_envelope"]
            for row in stress_per_scale[str(scale)]
        )
    ]
    in_envelope_rows = [
        row
        for scale in gauge_ladder
        for row in stress_per_scale[str(scale)]
        if row["inside_prediction_envelope"]
    ]
    comparable_in_envelope_rows = [
        row for row in in_envelope_rows if row["comparable_block"]
    ]
    noncomparable_in_envelope_count = (
        len(in_envelope_rows) - len(comparable_in_envelope_rows)
    )
    maximum_in_envelope_complex_shift = max(
        (row["complex_shift"] for row in comparable_in_envelope_rows),
        default=1e9,
    )
    maximum_in_envelope_magnitude_shift = max(
        (row["magnitude_shift"] for row in comparable_in_envelope_rows),
        default=1e9,
    )
    gauge_stress_passed = (
        noncomparable_in_envelope_count == 0
        and maximum_in_envelope_complex_shift <= 0.20
        and maximum_in_envelope_magnitude_shift <= 0.15
    )
    gauge_stress_refuted = (
        any(
            row["comparable_block"]
            and (
                row["complex_shift"] > 0.40
                or row["magnitude_shift"] > 0.30
            )
            for row in comparable_in_envelope_rows
        )
    )
    variant_panel = (
        _variant_panel(
            accepted_rows,
            order=config.target_order,
            band=config.action_band,
        )
        if accepted_rows
        else {
            "variants": {},
            "maximum_trigger_deviation": 1e9,
            "all_trigger_variants_evaluable": False,
            "unevaluable_trigger_variants": [
                "primary_linear",
                "quadratic_chart",
                "inner_band",
                "learned_h_frequency",
            ],
            "blocks_supported_status": False,
            "warning": True,
        }
    )

    shuffled_magnitudes = [
        row["median_bin_coefficient_magnitude"]
        for row in shuffled_results
        if row["median_bin_coefficient_magnitude"] is not None
    ]
    shuffled_level = (
        float(np.median(shuffled_magnitudes)) / abs(truth)
        if shuffled_magnitudes
        else 0.0
    )
    wrong_traps_pass = True
    wrong_harmonic_checks: dict[str, Any] = {}
    for wrong_order, result in wrong_harmonics.items():
        order_value = int(wrong_order)
        if order_value in (2, 4):
            rows = [
                {
                    "label": label,
                    "verdict": row["verdict"],
                    "coefficient_magnitude": None,
                    "passed": row["verdict"] == "no_resonance_crossing",
                }
                for label, row in zip(
                    accepted_labels,
                    result["charts"],
                    strict=True,
                )
            ]
            criterion = "each chart must report no_resonance_crossing"
            threshold = None
        else:
            threshold = 2.0 * modeled_floor
            rows = [
                {
                    "label": label,
                    "verdict": row["verdict"],
                    "coefficient_magnitude": (
                        abs(row["estimate"]["coefficient"])
                        if row["verdict"] == "value"
                        else None
                    ),
                    "passed": (
                        abs(row["estimate"]["coefficient"]) <= threshold
                        if row["verdict"] == "value"
                        else True
                    ),
                }
                for label, row in zip(
                    accepted_labels,
                    result["charts"],
                    strict=True,
                )
            ]
            criterion = (
                "each value must be at or below two modeled floors; "
                "abstention passes"
            )
        check_passed = all(row["passed"] for row in rows)
        wrong_harmonic_checks[wrong_order] = {
            "criterion": criterion,
            "threshold": threshold,
            "passed": check_passed,
            "charts": rows,
        }
        wrong_traps_pass &= check_passed
    off_band_passed = all(
        row["verdict"] == "no_resonance_crossing"
        for row in off_band_control["charts"]
    )
    null_coefficients = [
        result["charts"][0]["estimate"]["coefficient"]
        for result in null_results.values()
        if result["charts"]
        and result["charts"][0]["verdict"] == "value"
    ]
    null_level = (
        abs(_consensus(null_coefficients)) if null_coefficients else 0.0
    )
    null_fit_healthy = sum(
        row["held_out"]["normalized_one_step_rmse"] < 0.25
        and row["held_out"]["normalized_one_step_rmse"]
        < row["held_out"]["persistence_normalized_rollout_rmse"]
        for row in training_rows["s2"]
        if row["label"] in accepted_labels
    )
    null_instrument_available = (
        len(null_coefficients) >= config.minimum_accepted_charts
        and null_fit_healthy >= config.minimum_accepted_charts
    )
    null_passed = null_level <= 2.0 * modeled_floor
    false_positive_passed = (
        wrong_traps_pass
        and off_band_passed
        and null_instrument_available
        and null_passed
        and shuffled_level <= 0.2
    )
    sweep_levels: dict[str, Any] = {}
    detected_scale = None
    for scale in (0.0, 0.25, 0.5, 1.0, 2.0):
        _, _, sweep_bundle = _fixture(
            config,
            kick_scale=scale,
            initial_actions=initial_actions,
            initial_angles=initial_angles,
        )
        sweep = estimate_resonant_block(
            accepted_models,
            sweep_bundle.states[test_indices],
            order=config.target_order,
            band=config.action_band,
            bins=config.bins,
            max_order=config.max_order,
            reference_actions=sweep_bundle.actions[test_indices],
            reference_angles=sweep_bundle.angles[test_indices],
        )
        values = [
            row["estimate"]["coefficient"]
            for row in sweep["charts"]
            if row["verdict"] == "value"
        ]
        magnitude = abs(_consensus(values)) if values else None
        detected = magnitude is not None and magnitude > 2.0 * modeled_floor
        if detected and detected_scale is None and scale > 0.0:
            detected_scale = scale
        sweep_levels[str(scale)] = {
            "consensus_magnitude": magnitude,
            "detected_above_two_floors": detected,
        }
    predicted_scale = (
        2.0 * modeled_floor / config.kick_amplitude
    )
    roc_ratio = (
        detected_scale / max(predicted_scale, 1e-12)
        if detected_scale is not None
        else 1e9
    )
    roc_passed = roc_ratio <= 3.0

    empirical_gates = {
        "G1_complex_recovery": {
            "value": consensus_complex_error,
            "aggregation": "error of componentwise-median coefficient",
            "median_per_chart_error": median_per_chart_complex_error,
            "charts_above_threshold": charts_above_complex_threshold,
            "threshold": 0.20,
            "passed": consensus_complex_error <= 0.20,
        },
        "G2_magnitude_recovery": {
            "value": consensus_magnitude_error,
            "threshold": 0.15,
            "passed": consensus_magnitude_error <= 0.15,
        },
        "G3_location": {
            "value": consensus_location_error,
            "threshold": 2.0 * np.pi / (8.0 * config.target_order),
            "passed": (
                consensus_location_error
                <= 2.0 * np.pi / (8.0 * config.target_order)
            ),
        },
        "G4_floor_coverage": {
            "value": coverage_fraction,
            "evaluated_chart_count": len(floor_rows),
            "unavailable_chart_labels": [
                row["label"]
                for row in accepted_rows
                if not row["floor_available"]
            ],
            "threshold": 0.80,
            "passed": coverage_fraction >= 0.80,
        },
        "G5_false_positives": {
            "value": {
                "null_consensus_magnitude": null_level,
                "null_successful_chart_count": len(null_coefficients),
                "null_prediction_healthy_count": null_fit_healthy,
                "null_instrument_available": null_instrument_available,
                "null_below_two_floors": null_passed,
                "wrong_harmonics_passed": wrong_traps_pass,
                "wrong_harmonic_checks": wrong_harmonic_checks,
                "off_band_abstained": off_band_passed,
                "shuffled_over_truth": shuffled_level,
                "shuffled_band_fit_abstention_count": sum(
                    row["verdict"] != "value" for row in shuffled_results
                ),
            },
            "threshold": (
                "at least 6 prediction-healthy/estimable null charts; "
                "null/crossed traps <= 2 floors; no-crossing and off-band "
                "controls abstain; shuffled <= 0.2 of truth"
            ),
            "passed": false_positive_passed,
        },
        "G6_detection_roc": {
            "value": roc_ratio,
            "threshold": 3.0,
            "passed": roc_passed,
        },
        "G7_trajectory_vs_circle": {
            "value": {
                "trajectory_complex_error": consensus_complex_error,
                "circle_median_complex_error": circle_median_error,
                "trajectory_to_circle_ratio": trajectory_to_circle_ratio,
                "prototype_circle_transfer_passed": (
                    prototype_circle_transfer_passed
                ),
            },
            "threshold": (
                "trajectory error <= 0.20 or <= 2.5x circle error; "
                "prototype transfer requires circle median <= 0.02"
            ),
            "passed": trajectory_vs_circle_passed,
        },
        "G8_exact_gauge_stress": {
            "value": {
                "noncomparable_in_envelope_count": (
                    noncomparable_in_envelope_count
                ),
                "maximum_in_envelope_complex_shift": (
                    maximum_in_envelope_complex_shift
                ),
                "maximum_in_envelope_magnitude_shift": (
                    maximum_in_envelope_magnitude_shift
                ),
            },
            "threshold": {"complex": 0.20, "magnitude": 0.15},
            "passed": gauge_stress_passed,
        },
        "G9_variant_stability": {
            "value": variant_panel["maximum_trigger_deviation"],
            "all_trigger_variants_evaluable": variant_panel[
                "all_trigger_variants_evaluable"
            ],
            "unevaluable_trigger_variants": variant_panel[
                "unevaluable_trigger_variants"
            ],
            "threshold": 0.20,
            "passed": variant_panel["blocks_supported_status"],
        },
    }
    decision_gates = [
        empirical_gates[f"G{index}_{name}"]["passed"]
        for index, name in (
            (1, "complex_recovery"),
            (2, "magnitude_recovery"),
            (3, "location"),
            (4, "floor_coverage"),
            (5, "false_positives"),
            (6, "detection_roc"),
            (8, "exact_gauge_stress"),
            (9, "variant_stability"),
        )
    ]
    passed_empirical_gates = ensemble_healthy and all(decision_gates)
    status_reason = "gate_failure_1x_2x"
    if not ensemble_healthy:
        status = "invalid_ensemble"
        status_reason = "invalid_ensemble"
    elif config.profile != "full":
        status = "not_resolved_abstained"
        status_reason = "non_decisive_profile"
    elif gauge_stress_refuted:
        status = "resolved_refuted"
        status_reason = "gauge_freedom"
    elif not gauge_stress_passed:
        status = "not_resolved_abstained"
        status_reason = "gauge_freedom"
    elif not variant_panel["blocks_supported_status"]:
        status = "not_resolved_abstained"
        status_reason = "truncation_instability"
    elif not false_positive_passed:
        status = "not_resolved_abstained"
        status_reason = "false_positive_control"
    elif passed_empirical_gates:
        status = "resolved_supported"
        status_reason = "all_predeclared_gates_passed"
    else:
        severe = (
            consensus_complex_error > 0.40
            or consensus_magnitude_error > 0.30
        )
        status = "resolved_refuted" if severe else "not_resolved_abstained"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "resonance-metrology",
        "package_version": __version__,
        "profile": config.profile,
        "status": status,
        "status_reason": status_reason,
        "passed_empirical_gates": passed_empirical_gates,
        "matches_reference_precision": False,
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "source_revision": _git_source_state(),
        "fixture": {
            "map": {
                "base_frequency": config.base_frequency,
                "twist": config.twist,
                "kick_order": config.target_order,
                "kick_amplitude": config.kick_amplitude,
                "kick_phase": config.kick_phase,
                "resonant_action": system.resonance_action(
                    config.target_order
                ),
                "island_half_width": system.island_half_width(
                    config.target_order
                ),
            },
            "observation_chart": asdict(observation),
            "paired_initial_conditions": True,
            "training_trajectory_ids": [
                s1_bundle.trajectory_ids[index] for index in train_indices
            ],
            "held_out_trajectory_ids": [
                s1_bundle.trajectory_ids[index] for index in test_indices
            ],
        },
        "ensemble": {
            "acceptance_limit": acceptance_limit,
            "absolute_acceptance_limit": absolute_acceptance_limit,
            "minimum_accepted_charts": config.minimum_accepted_charts,
            "accepted_count": len(accepted_labels),
            "accepted_labels": accepted_labels,
            "estimable_count": len(accepted_rows),
            "block_ledger": block_ledger,
            "accepted": accepted_rows,
            "dropped": dropped,
            "training": training_rows,
        },
        "oracle": {
            "coefficient": _complex_payload(truth),
            "kick_amplitude": config.kick_amplitude,
            "generating_function_amplitude": (
                config.kick_amplitude / config.target_order
            ),
            "island_half_width": system.island_half_width(config.target_order),
            "analysis": oracle,
        },
        "raw_coordinate_baseline": raw,
        "ensemble_consensus": {
            "coefficient": _complex_payload(consensus),
            "complex_error": consensus_complex_error,
            "median_per_chart_complex_error": (
                median_per_chart_complex_error
            ),
            "charts_above_complex_threshold": (
                charts_above_complex_threshold
            ),
            "magnitude_error": consensus_magnitude_error,
            "location_error_radians": consensus_location_error,
            "generating_function_amplitude": abs(consensus)
            / config.target_order,
            "island_half_width": 2.0
            * np.sqrt(
                (abs(consensus) / config.target_order) / abs(config.twist)
            ),
        },
        "empirical_gates": empirical_gates,
        "controls": {
            "null_runs": null_results,
            "wrong_harmonics": wrong_harmonics,
            "wrong_harmonic_checks": wrong_harmonic_checks,
            "off_band": off_band_control,
            "shuffled_angle_over_truth": shuffled_level,
            "shuffled_angle_runs": shuffled_results,
            "kick_sweep": {
                "levels": sweep_levels,
                "modeled_floor": modeled_floor,
                "predicted_detection_scale": predicted_scale,
                "observed_detection_scale": detected_scale,
                "observed_to_predicted_ratio": roc_ratio,
            },
            "exact_2m_gauge_stress": {
                "ladder": list(gauge_ladder),
                "phases": [0.0, 0.5 * np.pi],
                "prediction_envelope": prediction_envelope,
                "per_scale": stress_per_scale,
                "smallest_visible_scale": (
                    min(visible_scales) if visible_scales else None
                ),
                "maximum_in_envelope_complex_shift": (
                    maximum_in_envelope_complex_shift
                ),
                "maximum_in_envelope_magnitude_shift": (
                    maximum_in_envelope_magnitude_shift
                ),
                "noncomparable_in_envelope_count": (
                    noncomparable_in_envelope_count
                ),
            },
            "variant_stability": variant_panel,
        },
        "ledgers": {
            "structural": {
                "fixture_map": "exact symplectic kick-drift",
                "observation_chart": "exact canonical shears plus SL(2)",
                "learned_charts": "exact symplectic by construction",
            },
            "empirical": {
                "passed_empirical_gates": passed_empirical_gates,
                "scope": "one noiseless synthetic return-map fixture",
            },
            "stability": {
                "accepted_charts": len(accepted_labels),
                "estimable_charts": len(accepted_rows),
                "shared_bias_stress": "exact 2m gauge ladder",
            },
            "certification": {
                "formal_guarantees": "none",
                "explicitly_not_claimed": [
                    "KAM or a-posteriori torus proof",
                    "interval bounds",
                    "calibrated probabilities",
                    "noise robustness",
                    "measured-system validation",
                ],
            },
        },
        "claim_boundary": (
            "On one noiseless synthetic exact-symplectic kicked twist map "
            "observed through a fixed nontrivial canonical chart, the "
            f"trajectory-band ensemble returned {status!r} under the frozen "
            "learned-chart, control, and exact-gauge stress gates."
        ),
        "not_supported": [
            "formal identifiability",
            "transfer beyond the tested synthetic map",
            "measured vibration data",
            "noise, irregular sampling, or partial observation",
            "physical coefficients at harmonics without an in-band resonance crossing",
        ],
        "next_falsifier": (
            "Repeat only after preserving this result: independent systems, "
            "noise/sampling sweeps, and a measured return map."
        ),
        "runtime_seconds": time.perf_counter() - started,
        "artifacts": {
            "manifest": "manifest.json",
            "report": "report.html",
            "overview": "overview.png",
            "s1_data": s1_csv.name,
            "s2_data": s2_csv.name,
        },
    }
    safe_manifest = _json_safe(manifest)
    _plot_report(config.output / "overview.png", safe_manifest)
    _write_report(config.output / "report.html", safe_manifest)
    safe_manifest["artifacts"].update(
        {
            "report_sha256": _sha256(config.output / "report.html"),
            "overview_sha256": _sha256(config.output / "overview.png"),
            "s1_data_sha256": _sha256(s1_csv),
            "s2_data_sha256": _sha256(s2_csv),
        }
    )
    _write_json(config.output / "manifest.json", safe_manifest)
    return safe_manifest


def validate_resonance_manifest(
    manifest: dict[str, Any],
    *,
    require_data_artifacts: bool = False,
) -> list[str]:
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported resonance-metrology schema")
    if manifest.get("experiment") != "resonance-metrology":
        raise ValueError("not a resonance-metrology manifest")
    serialized = json.dumps(manifest, allow_nan=False)
    if not serialized:
        raise ValueError("empty manifest")
    valid_statuses = {
        "resolved_supported",
        "resolved_refuted",
        "not_resolved_abstained",
        "invalid_ensemble",
    }
    if manifest["status"] not in valid_statuses:
        raise ValueError("unknown metrology status")
    if manifest["profile"] != "full" and manifest["status"] in {
        "resolved_supported",
        "resolved_refuted",
    }:
        raise ValueError("non-full profile emitted a decisive scientific status")
    if "passed_empirical_gates" not in manifest:
        raise ValueError("empirical-gate status is missing")
    artifacts = manifest["artifacts"]
    root = Path(manifest.get("_artifact_root", "."))
    artifact_checks = []
    for name in ("report", "overview", "s1_data", "s2_data"):
        digest = artifacts.get(f"{name}_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{name} artifact digest is missing")
        target = root / artifacts[name]
        required = name in {"report", "overview"} or require_data_artifacts
        if required and not target.is_file():
            raise ValueError(f"{name} artifact is missing")
        if target.is_file() and _sha256(target) != digest:
            raise ValueError(f"{name} artifact digest is stale")
        artifact_checks.append(
            f"{name} digest verified"
            if target.is_file()
            else f"{name} digest recorded; source artifact is not shipped"
        )
    accepted = manifest["ensemble"]["accepted"]
    if len(accepted) != manifest["ensemble"]["estimable_count"]:
        raise ValueError("estimable-chart count is stale")
    accepted_labels = manifest["ensemble"]["accepted_labels"]
    if len(accepted_labels) != manifest["ensemble"]["accepted_count"]:
        raise ValueError("prediction-accepted chart count is stale")
    if any(row["label"] not in accepted_labels for row in accepted):
        raise ValueError("an estimable chart was not prediction-accepted")
    if any(row["complex_error"] < 0.0 for row in accepted):
        raise ValueError("chart error cannot be negative")
    for row in accepted:
        if row["floor_available"] != (row["floor_total"] is not None):
            raise ValueError("floor availability is inconsistent")
        if not row["floor_available"] and row["covered"] is not None:
            raise ValueError("unavailable floor silently emitted coverage")
    floor_gate = manifest["empirical_gates"]["G4_floor_coverage"]
    if floor_gate["evaluated_chart_count"] != sum(
        row["floor_available"] for row in accepted
    ):
        raise ValueError("floor-coverage denominator is stale")
    variant_gate = manifest["empirical_gates"]["G9_variant_stability"]
    if (
        variant_gate["passed"]
        and not variant_gate["all_trigger_variants_evaluable"]
    ):
        raise ValueError("G9 passed with an unevaluable trigger variant")
    trap_checks = manifest["controls"]["wrong_harmonic_checks"]
    if any(
        check["passed"]
        != all(row["passed"] for row in check["charts"])
        for check in trap_checks.values()
    ):
        raise ValueError("wrong-harmonic per-chart ledger is inconsistent")
    false_positive_gate = manifest["empirical_gates"]["G5_false_positives"]
    if (
        false_positive_gate["passed"]
        and not all(check["passed"] for check in trap_checks.values())
    ):
        raise ValueError("G5 passed with a failed wrong-harmonic chart")
    for system_name in ("s1", "s2"):
        for row in manifest["ensemble"]["training"][system_name]:
            initialization = row.get("frequency_initialization", {})
            if initialization.get("uses_oracle_coordinates") is not False:
                raise ValueError("frequency initialization must be observation-only")
            coefficients = initialization.get(
                "frequency_coefficients_ascending"
            )
            if not isinstance(coefficients, list) or not coefficients:
                raise ValueError("frequency initialization is missing")
            if not np.isfinite(coefficients).all():
                raise ValueError("frequency initialization is non-finite")
    if (
        manifest["profile"] == "full"
        and manifest["source_revision"].get("git_worktree_clean") is not True
    ):
        raise ValueError("full-profile evidence must come from a clean source revision")
    if (
        manifest["status"] == "resolved_refuted"
        and manifest["status_reason"] == "gauge_freedom"
    ):
        stress = manifest["controls"]["exact_2m_gauge_stress"]
        if not (
            stress["maximum_in_envelope_complex_shift"] > 0.40
            or stress["maximum_in_envelope_magnitude_shift"] > 0.30
        ):
            raise ValueError("gauge refutation lacks a comparable 2x shift")
    for width in (
        manifest["oracle"]["island_half_width"],
        manifest["ensemble_consensus"]["island_half_width"],
    ):
        if width < 0.0:
            raise ValueError("island halfwidth cannot be negative")
    return [
        "profile semantics and status are consistent",
        "accepted-chart ledger is internally consistent",
        "floor and per-chart control abstentions are explicit",
        "all chart initializers are observation-only and finite",
        *artifact_checks,
        "island widths use nonnegative generating amplitudes",
    ]
