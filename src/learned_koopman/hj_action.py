from __future__ import annotations

import hashlib
import html
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np

from learned_koopman import __version__
from learned_koopman.trajectory import TrajectoryDataset


class CoordinateModel(Protocol):
    state_columns: tuple[str, ...]

    def coordinate(self, states: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class CycleMeasurement:
    """One closed-orbit action measurement between consecutive positive maxima."""

    start_time: float
    end_time: float
    period: float
    action: float
    frequency: float
    closure_error: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_finite(value: Any, path: str = "manifest") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} is not finite")


def _positive_maximum_events(
    times: np.ndarray,
    position: np.ndarray,
    momentum: np.ndarray,
) -> list[tuple[float, float]]:
    """Locate p=0 crossings from positive to negative momentum."""

    momentum_scale = max(float(np.max(np.abs(momentum))), 1.0)
    tolerance = max(1e-12, momentum_scale * 1e-8)
    midpoint = float(np.median(position))
    events: list[tuple[float, float]] = []
    if (
        abs(float(momentum[0])) <= tolerance
        and float(momentum[1]) < -tolerance
        and float(position[0]) > midpoint
    ):
        events.append((float(times[0]), float(position[0])))
    for index in range(len(momentum) - 1):
        left = float(momentum[index])
        right = float(momentum[index + 1])
        if left <= tolerance or right > tolerance:
            continue
        alpha = left / (left - right)
        event_time = float(times[index] + alpha * (times[index + 1] - times[index]))
        event_position = float(
            position[index] + alpha * (position[index + 1] - position[index])
        )
        if event_position > midpoint:
            events.append((event_time, event_position))
    return events


def _measure_cycles(
    times: np.ndarray,
    position: np.ndarray,
    momentum: np.ndarray,
) -> list[CycleMeasurement]:
    events = _positive_maximum_events(times, position, momentum)
    position_scale = max(float(np.ptp(position)), 1e-12)
    measurements: list[CycleMeasurement] = []
    for (start_time, start_position), (end_time, end_position) in zip(
        events,
        events[1:],
        strict=False,
    ):
        interior = (times > start_time) & (times < end_time)
        q = np.concatenate(([start_position], position[interior], [end_position]))
        p = np.concatenate(([0.0], momentum[interior], [0.0]))
        line_integral = float(np.sum(0.5 * (p[:-1] + p[1:]) * np.diff(q)))
        period = end_time - start_time
        if period <= 0.0:
            continue
        measurements.append(
            CycleMeasurement(
                start_time=start_time,
                end_time=end_time,
                period=period,
                action=abs(line_integral) / (2.0 * math.pi),
                frequency=2.0 * math.pi / period,
                closure_error=abs(end_position - start_position) / position_scale,
            )
        )
    return measurements


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_ranks = np.argsort(np.argsort(left, kind="stable"), kind="stable")
    right_ranks = np.argsort(np.argsort(right, kind="stable"), kind="stable")
    if np.std(left_ranks) <= 0.0 or np.std(right_ranks) <= 0.0:
        return 0.0
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _coordinate_alignment(
    coordinate: np.ndarray,
    action: np.ndarray,
    trajectory_ids: list[str],
) -> dict[str, Any]:
    design = np.column_stack((coordinate, np.ones(len(coordinate), dtype=np.float64)))
    slope, intercept = np.linalg.lstsq(design, action, rcond=None)[0]
    prediction = slope * coordinate + intercept
    residual = float(np.square(prediction - action).sum())
    total = float(np.square(action - action.mean()).sum())
    rank_correlation = _rank_correlation(coordinate, action)
    orientation = 1.0 if rank_correlation >= 0.0 else -1.0
    oriented = orientation * coordinate
    order = np.argsort(oriented)
    calibration_indices = order[::2]
    if order[-1] not in calibration_indices:
        calibration_indices = np.append(calibration_indices, order[-1])
    validation_indices = np.setdiff1d(order, calibration_indices)
    polynomial_degree = 3
    coefficients = np.polyfit(
        oriented[calibration_indices],
        action[calibration_indices],
        polynomial_degree,
    )
    held_out_prediction = np.polyval(coefficients, oriented[validation_indices])
    held_out_truth = action[validation_indices]
    held_out_residual = float(np.square(held_out_prediction - held_out_truth).sum())
    held_out_total = float(
        np.square(held_out_truth - held_out_truth.mean()).sum()
    )
    grid = np.linspace(float(oriented.min()), float(oriented.max()), 512)
    derivative = np.polyval(np.polyder(coefficients), grid)
    return {
        "affine_r2": 1.0 - residual / max(total, 1e-12),
        "absolute_rank_correlation": abs(rank_correlation),
        "orientation": orientation,
        "affine_slope": float(slope),
        "affine_intercept": float(intercept),
        "calibration": {
            "kind": "trajectory-interleaved monotone polynomial",
            "degree": polynomial_degree,
            "coefficients": coefficients.tolist(),
            "calibration_trajectory_ids": [
                trajectory_ids[index] for index in calibration_indices
            ],
            "held_out_trajectory_ids": [
                trajectory_ids[index] for index in validation_indices
            ],
            "held_out_r2": 1.0 - held_out_residual / max(held_out_total, 1e-12),
            "held_out_max_absolute_error": float(
                np.max(np.abs(held_out_prediction - held_out_truth))
            ),
            "monotone_over_observed_range": bool(np.all(derivative > 0.0)),
            "minimum_derivative": float(np.min(derivative)),
        },
    }


def _hj_identity(
    action: np.ndarray,
    energy: np.ndarray,
    frequency: np.ndarray,
) -> dict[str, Any]:
    order = np.argsort(action)
    sorted_action = action[order]
    sorted_energy = energy[order]
    sorted_frequency = frequency[order]
    if len(action) < 6 or np.any(np.diff(sorted_action) <= 1e-10):
        return {
            "available": False,
            "reason": "need at least six trajectories with distinct empirical actions",
        }
    derivative = np.gradient(sorted_energy, sorted_action, edge_order=2)
    interior = slice(1, -1)
    error = derivative[interior] - sorted_frequency[interior]
    frequency_scale = max(float(np.sqrt(np.mean(sorted_frequency[interior] ** 2))), 1e-12)
    relative = np.abs(error) / np.maximum(np.abs(sorted_frequency[interior]), 1e-12)
    return {
        "available": True,
        "equation": "dH/dJ = omega",
        "normalized_rmse": float(np.sqrt(np.mean(error**2)) / frequency_scale),
        "median_relative_error": float(np.median(relative)),
        "max_relative_error": float(np.max(relative)),
        "action": sorted_action.tolist(),
        "energy": sorted_energy.tolist(),
        "measured_frequency": sorted_frequency.tolist(),
        "dH_dJ": derivative.tolist(),
    }


def validate_hj_action_manifest(manifest: dict[str, Any]) -> list[str]:
    """Recompute the audit verdict and reject non-finite or stale evidence."""

    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported HJ-action audit schema")
    _assert_finite(manifest)
    measurements = manifest["measurements"]
    if len(measurements) < 6:
        raise ValueError("HJ-action audit needs at least six measured trajectories")
    comparisons = {
        "enough_closed_orbits": (
            len(measurements)
            >= max(6, math.ceil(0.8 * manifest["dataset"]["trajectory_count"]))
        ),
        "orbit_closure_is_accurate": (
            manifest["aggregate"]["max_closure_error"] < 0.02
        ),
        "action_is_ordered": (
            manifest["aggregate"]["action_rank_vs_amplitude"] > 0.95
        ),
    }
    hj_identity = manifest["hj_identity"]
    if hj_identity["available"]:
        comparisons["hamilton_jacobi_identity_holds"] = (
            hj_identity["normalized_rmse"] < 0.08
            and hj_identity["median_relative_error"] < 0.05
        )
    learned_alignment = manifest["learned_coordinate_alignment"]
    if learned_alignment["available"]:
        calibration = learned_alignment["calibration"]
        comparisons["learned_invariant_calibrates_to_action"] = (
            calibration["held_out_r2"] > 0.98
            and calibration["monotone_over_observed_range"]
            and learned_alignment["absolute_rank_correlation"] > 0.95
        )
    if comparisons != manifest["certificate"]["decisive_comparisons"]:
        raise ValueError("HJ-action certificate is stale")
    expected = (
        "supported_on_supplied_periodic_orbits"
        if all(comparisons.values())
        else "not_supported_by_current_dataset"
    )
    if manifest["certificate"]["status"] != expected:
        raise ValueError("HJ-action status disagrees with measured evidence")
    artifacts = manifest["artifacts"]
    for name in ("overview", "report"):
        digest = artifacts.get(f"{name}_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{name} artifact SHA-256 is missing")
    return [
        "closed-orbit actions were measured from supplied canonical coordinates",
        "the certificate was recomputed from measured closure and ordering",
        (
            "the Hamilton-Jacobi identity was checked"
            if hj_identity["available"]
            else "the Hamilton-Jacobi identity was not claimed without energy"
        ),
        (
            "the learned invariant was compared with canonical action"
            if learned_alignment["available"]
            else "no learned-coordinate calibration was claimed"
        ),
    ]


def _plot_overview(path: Path, manifest: dict[str, Any]) -> None:
    measurements = manifest["measurements"]
    action = np.asarray([row["action"] for row in measurements])
    frequency = np.asarray([row["frequency"] for row in measurements])
    amplitude = np.asarray([row["position_amplitude"] for row in measurements])
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(amplitude, action, "o-", color="#4057c9")
    axes[0].set(xlabel="position amplitude", ylabel=r"action $J$", title="Orbit area")
    axes[0].grid(alpha=0.25)
    axes[1].plot(action, frequency, "o-", color="#d85140", label="measured")
    hj_identity = manifest["hj_identity"]
    if hj_identity["available"]:
        axes[1].plot(
            hj_identity["action"],
            hj_identity["dH_dJ"],
            "--",
            color="#2a8b68",
            label=r"$dH/dJ$",
        )
        axes[1].legend()
    axes[1].set(xlabel=r"action $J$", ylabel=r"frequency $\omega$", title="HJ identity")
    axes[1].grid(alpha=0.25)
    alignment = manifest["learned_coordinate_alignment"]
    if alignment["available"]:
        coordinate = np.asarray(alignment["coordinate"])
        axes[2].scatter(coordinate, action, color="#7c3fb7")
        grid = np.linspace(float(coordinate.min()), float(coordinate.max()), 100)
        calibration = alignment["calibration"]
        axes[2].plot(
            grid,
            np.polyval(
                calibration["coefficients"],
                alignment["orientation"] * grid,
            ),
            color="#252525",
        )
        axes[2].set(
            xlabel="learned invariant",
            ylabel=r"action $J$",
            title="Gauge calibration",
        )
    else:
        axes[2].text(
            0.5,
            0.5,
            "Pass --model to test whether\nthe learned invariant calibrates\nto canonical action.",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
        axes[2].set(title="Gauge calibration")
        axes[2].set_xticks([])
        axes[2].set_yticks([])
    axes[2].grid(alpha=0.25)
    figure.suptitle("Koopman + Hamilton–Jacobi canonical-action audit", fontsize=14)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    certificate = manifest["certificate"]
    hj_identity = manifest["hj_identity"]
    learned = manifest["learned_coordinate_alignment"]
    checks = certificate["decisive_comparisons"]
    check_rows = "\n".join(
        f"<li class=\"{'pass' if passed else 'fail'}\">"
        f"{'PASS' if passed else 'FAIL'} — {html.escape(name.replace('_', ' '))}</li>"
        for name, passed in checks.items()
    )
    hj_text = (
        f"normalized RMSE {hj_identity['normalized_rmse']:.3%}; "
        f"median relative error {hj_identity['median_relative_error']:.3%}"
        if hj_identity["available"]
        else html.escape(hj_identity["reason"])
    )
    learned_text = (
        f"held-out monotone-calibration R² "
        f"{learned['calibration']['held_out_r2']:.4f}; "
        f"|rank correlation| {learned['absolute_rank_correlation']:.4f}"
        if learned["available"]
        else "No model supplied; no learned-coordinate claim was made."
    )
    path.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Koopman + HJ action audit</title>
<style>
body {{ margin: 0; background: #f4f1ea; color: #20242b; font: 17px/1.55 system-ui; }}
main {{ max-width: 980px; margin: auto; padding: 54px 24px 80px; }}
.eyebrow {{ color: #4057c9; font-weight: 750; letter-spacing: .08em; text-transform: uppercase; }}
h1 {{ font: 700 clamp(36px,7vw,68px)/1.02 Georgia,serif; margin: 10px 0 24px; }}
h2 {{ margin-top: 40px; }} .card {{ background: white; padding: 26px; border-radius: 18px;
box-shadow: 0 10px 35px #1e263112; margin: 22px 0; }}
.status {{ display: inline-block; border-radius: 99px; padding: 7px 13px;
background: #e8edf9; font-weight: 700; }} img {{ width: 100%; border-radius: 12px; }}
.pass {{ color: #176e50; }} .fail {{ color: #ae352d; }} code {{ font-size: .9em; }}
</style></head><body><main>
<div class="eyebrow">learned-koopman · canonical mechanics</div>
<h1>Does the learned foliation line up with Hamilton–Jacobi action?</h1>
<p class="status">{html.escape(certificate['status'])}</p>
<p>This audit measures <strong>J = (1/2π)∮p dq</strong> directly from each supplied
closed orbit. It then tests <strong>dH/dJ = ω</strong> when a reference Hamiltonian is
available and calibrates the learned invariant against J when a model is supplied.</p>
<div class="card"><img src="overview.png" alt="Canonical-action audit plots"></div>
<div class="card"><h2>Measured evidence</h2>
<ul>{check_rows}</ul>
<p><strong>Hamilton–Jacobi identity:</strong> {hj_text}.</p>
<p><strong>Learned coordinate:</strong> {learned_text}.</p></div>
<h2>Scientific boundary</h2>
<p>This is a one-degree-of-freedom, autonomous, conservative, periodic-orbit audit.
The two supplied state columns must be canonical position and momentum. Velocity is
momentum only after the correct mass scaling. This result does not claim a global
generating function, a phase chart through a turning point, multi-degree integrability,
or a Hamilton–Jacobi–Bellman solution.</p>
<h2>What it unlocks</h2>
<p>The existing neural invariant is free up to a monotone gauge. Empirical action
anchors that gauge to a canonical physical quantity. The next construction can learn
a conjugate phase, enforce {{φ,J}} = 1, and test the HJ PDE and Koopman eigenfunction
residuals on held-out tori.</p>
<p>Machine-readable evidence: <code>manifest.json</code>.</p>
</main></body></html>
""",
        encoding="utf-8",
    )


def run_hj_action_audit(
    dataset: TrajectoryDataset,
    output_dir: Path,
    *,
    model: CoordinateModel | None = None,
) -> dict[str, Any]:
    """Measure canonical action and test its HJ and learned-coordinate relations."""

    if dataset.state_dim != 2:
        raise ValueError("HJ-action audit requires exactly two canonical state columns (q, p)")
    if model is not None and model.state_columns != dataset.state_columns:
        raise ValueError(
            "model state columns do not match the supplied canonical coordinate order"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    measured: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for index, trajectory_id in enumerate(dataset.trajectory_ids):
        position = dataset.states[index, :, 0]
        momentum = dataset.states[index, :, 1]
        cycles = _measure_cycles(dataset.times[index], position, momentum)
        if not cycles:
            rejected.append(
                {"trajectory_id": trajectory_id, "reason": "no complete closed cycle"}
            )
            continue
        actions = np.asarray([cycle.action for cycle in cycles])
        periods = np.asarray([cycle.period for cycle in cycles])
        frequencies = np.asarray([cycle.frequency for cycle in cycles])
        measured.append(
            {
                "trajectory_id": trajectory_id,
                "cycle_count": len(cycles),
                "position_amplitude": float(np.max(np.abs(position - np.mean(position)))),
                "action": float(np.mean(actions)),
                "period": float(np.mean(periods)),
                "frequency": float(np.mean(frequencies)),
                "max_closure_error": float(
                    max(cycle.closure_error for cycle in cycles)
                ),
                "action_coefficient_of_variation": float(
                    np.std(actions) / max(float(np.mean(actions)), 1e-12)
                ),
                "period_coefficient_of_variation": float(
                    np.std(periods) / max(float(np.mean(periods)), 1e-12)
                ),
                "cycles": [asdict(cycle) for cycle in cycles],
            }
        )
    if len(measured) < 6:
        raise ValueError(
            "need at least six trajectories containing a complete positive-maximum cycle"
        )
    indices = np.asarray(
        [dataset.trajectory_ids.index(row["trajectory_id"]) for row in measured],
        dtype=np.int64,
    )
    actions = np.asarray([row["action"] for row in measured], dtype=np.float64)
    frequencies = np.asarray([row["frequency"] for row in measured], dtype=np.float64)
    amplitudes = np.asarray(
        [row["position_amplitude"] for row in measured],
        dtype=np.float64,
    )
    hj_identity: dict[str, Any]
    if dataset.reference_values is None:
        hj_identity = {
            "available": False,
            "reason": "no reference Hamiltonian column was supplied",
        }
    else:
        hj_identity = _hj_identity(
            actions,
            dataset.reference_values[indices],
            frequencies,
        )
    learned_alignment: dict[str, Any]
    if model is None:
        learned_alignment = {
            "available": False,
            "reason": "no mechanics-workbench model was supplied",
        }
    else:
        coordinate = model.coordinate(dataset.states[indices]).mean(axis=1)
        learned_alignment = {
            "available": True,
            "coordinate": coordinate.tolist(),
            **_coordinate_alignment(
                coordinate,
                actions,
                [row["trajectory_id"] for row in measured],
            ),
        }
    aggregate = {
        "measured_trajectory_count": len(measured),
        "rejected_trajectory_count": len(rejected),
        "max_closure_error": float(
            max(row["max_closure_error"] for row in measured)
        ),
        "median_closure_error": float(
            np.median([row["max_closure_error"] for row in measured])
        ),
        "action_rank_vs_amplitude": _rank_correlation(actions, amplitudes),
        "action_min": float(actions.min()),
        "action_max": float(actions.max()),
        "frequency_min": float(frequencies.min()),
        "frequency_max": float(frequencies.max()),
    }
    comparisons = {
        "enough_closed_orbits": (
            len(measured) >= max(6, math.ceil(0.8 * dataset.trajectory_count))
        ),
        "orbit_closure_is_accurate": aggregate["max_closure_error"] < 0.02,
        "action_is_ordered": aggregate["action_rank_vs_amplitude"] > 0.95,
    }
    if hj_identity["available"]:
        comparisons["hamilton_jacobi_identity_holds"] = (
            hj_identity["normalized_rmse"] < 0.08
            and hj_identity["median_relative_error"] < 0.05
        )
    if learned_alignment["available"]:
        calibration = learned_alignment["calibration"]
        comparisons["learned_invariant_calibrates_to_action"] = (
            calibration["held_out_r2"] > 0.98
            and calibration["monotone_over_observed_range"]
            and learned_alignment["absolute_rank_correlation"] > 0.95
        )
    status = (
        "supported_on_supplied_periodic_orbits"
        if all(comparisons.values())
        else "not_supported_by_current_dataset"
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "package_version": __version__,
        "scientific_contract": {
            "system_class": "one-degree-of-freedom autonomous conservative mechanics",
            "coordinate_contract": (
                "state column 1 is canonical q; state column 2 is canonical p"
            ),
            "measured_equations": [
                "J = (1 / (2 pi)) integral_closed_orbit p dq",
                "omega = 2 pi / T",
                "dH/dJ = omega",
            ],
            "not_claimed": [
                "global Hamilton-Jacobi generating function",
                "global angle chart through turning points or separatrices",
                "multi-degree-of-freedom Liouville integrability",
                "Hamilton-Jacobi-Bellman control solution",
            ],
        },
        "dataset": {
            "source": dataset.source,
            "source_sha256": dataset.source_sha256,
            "trajectory_count": dataset.trajectory_count,
            "step_count": dataset.step_count,
            "dt": dataset.dt,
            "canonical_columns": list(dataset.state_columns),
            "reference_column": dataset.reference_column,
        },
        "measurements": measured,
        "rejected_trajectories": rejected,
        "aggregate": aggregate,
        "hj_identity": hj_identity,
        "learned_coordinate_alignment": learned_alignment,
        "certificate": {
            "status": status,
            "decisive_comparisons": comparisons,
            "scope": (
                "the supplied periodic trajectories under the declared canonical "
                "coordinate and conservative-system assumptions"
            ),
        },
        "artifacts": {
            "overview": "overview.png",
            "report": "report.html",
            "manifest": "manifest.json",
        },
    }
    _plot_overview(output_dir / "overview.png", manifest)
    _write_report(output_dir / "report.html", manifest)
    manifest["artifacts"].update(
        {
            "overview_sha256": _sha256(output_dir / "overview.png"),
            "report_sha256": _sha256(output_dir / "report.html"),
        }
    )
    manifest["validation_checks"] = validate_hj_action_manifest(manifest)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest
