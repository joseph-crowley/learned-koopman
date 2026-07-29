from __future__ import annotations

import hashlib
import html
import json
import math
import platform
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from learned_koopman import __version__
from learned_koopman.invariant_experiment import train_invariant_model
from learned_koopman.models.invariant import LearnedInvariant
from learned_koopman.operator_family import (
    FiberedKoopmanModel,
    fit_fibered_operator,
    observable_feature_names,
    spectral_summary,
)
from learned_koopman.trajectory import TrajectoryDataset


@dataclass(frozen=True)
class WorkbenchConfig:
    profile: str
    seed: int
    train_fraction: float
    hidden_dim: int
    epochs: int
    learning_rate: float
    graph_neighbors: int
    constancy_weight: float
    graph_weight: float
    variance_weight: float
    centering_weight: float
    family_degree: int
    observable_degree: int
    ridge: float

    @classmethod
    def quick(cls, seed: int = 7) -> WorkbenchConfig:
        return cls(
            profile="quick",
            seed=seed,
            train_fraction=0.75,
            hidden_dim=32,
            epochs=160,
            learning_rate=3e-3,
            graph_neighbors=2,
            constancy_weight=8.0,
            graph_weight=0.15,
            variance_weight=1.0,
            centering_weight=0.02,
            family_degree=2,
            observable_degree=2,
            ridge=1e-5,
        )

    @classmethod
    def full(cls, seed: int = 7) -> WorkbenchConfig:
        return cls(
            profile="full",
            seed=seed,
            train_fraction=0.75,
            hidden_dim=48,
            epochs=500,
            learning_rate=2e-3,
            graph_neighbors=2,
            constancy_weight=10.0,
            graph_weight=0.12,
            variance_weight=1.0,
            centering_weight=0.02,
            family_degree=2,
            observable_degree=2,
            ridge=1e-6,
        )


@dataclass
class MechanicsModel:
    """Loadable engineer-facing model exported by the mechanics workbench."""

    invariant: LearnedInvariant
    operator: FiberedKoopmanModel
    state_mean: np.ndarray
    state_scale: np.ndarray
    state_columns: tuple[str, ...]
    invariant_min: float
    invariant_max: float
    state_support_samples: np.ndarray
    state_support_radius: float
    certificate_status: str
    decisive_comparisons: dict[str, bool]

    def coordinate(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=np.float64)
        if values.ndim == 0 or values.shape[-1] != len(self.state_columns):
            raise ValueError(
                f"expected state vectors with {len(self.state_columns)} values"
            )
        normalized = (values - self.state_mean) / self.state_scale
        with torch.no_grad():
            result = self.invariant(
                torch.tensor(normalized, dtype=torch.float32)
            ).cpu()
        return result.numpy().astype(np.float64)

    def rollout(
        self,
        initial_states: np.ndarray,
        *,
        steps: int,
        allow_extrapolation: bool = False,
    ) -> np.ndarray:
        if steps < 1:
            raise ValueError("steps must be positive")
        if (
            self.certificate_status != "supported_on_held_out_trajectories"
            and not allow_extrapolation
        ):
            raise ValueError(
                f"model fit is not certified ({self.certificate_status}); "
                "pass allow_extrapolation=True to override"
            )
        values = np.asarray(initial_states, dtype=np.float64)
        one_state = values.ndim == 1
        if one_state:
            values = values[None, :]
        support = self.support_status(values)
        if np.any(support != "supported") and not allow_extrapolation:
            unsupported = ", ".join(sorted(set(support.tolist())))
            raise ValueError(
                f"initial state is outside fitted support ({unsupported}); "
                "pass allow_extrapolation=True to override"
            )
        normalized = (values - self.state_mean) / self.state_scale
        invariants = self.coordinate(values)
        prediction = self.operator.rollout(
            normalized,
            invariants,
            steps=steps,
        )
        physical = prediction * self.state_scale + self.state_mean
        return physical[0] if one_state else physical

    def support_status(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=np.float64)
        coordinate = np.atleast_1d(self.coordinate(values))
        normalized = (values - self.state_mean) / self.state_scale
        state_distance = np.atleast_1d(
            _nearest_state_distance(normalized, self.state_support_samples)
        )
        invariant_supported = (coordinate >= self.invariant_min) & (
            coordinate <= self.invariant_max
        )
        state_supported = state_distance <= self.state_support_radius
        result = np.full(coordinate.shape, "supported", dtype=object)
        result[~invariant_supported & state_supported] = "invariant_extrapolation"
        result[invariant_supported & ~state_supported] = "state_extrapolation"
        result[~invariant_supported & ~state_supported] = (
            "invariant_and_state_extrapolation"
        )
        if self.certificate_status != "supported_on_held_out_trajectories":
            result[:] = "fit_not_certified"
        return result


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_source_state() -> dict[str, Any]:
    """Record the repository revision when running from a Git checkout."""

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


def _split_indices(count: int, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.5, 1.0)")
    generator = np.random.default_rng(seed)
    order = generator.permutation(count)
    test_count = max(2, int(round(count * (1.0 - train_fraction))))
    test_count = min(test_count, count - 4)
    if test_count < 2:
        raise ValueError("need enough trajectories for four training and two held-out runs")
    return np.sort(order[test_count:]), np.sort(order[:test_count])


def _orbit_coordinate(
    model: LearnedInvariant,
    trajectories: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        values = model(torch.tensor(trajectories, dtype=torch.float32)).numpy()
    return values, values.mean(axis=1), values.std(axis=1)


def _normalized_rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def _per_state_rmse(
    prediction: np.ndarray,
    truth: np.ndarray,
    scale: np.ndarray,
    state_columns: tuple[str, ...],
) -> dict[str, float]:
    values = np.sqrt(
        np.mean(
            np.square((prediction - truth) * scale),
            axis=(0, 1),
        )
    )
    return {
        column: float(value)
        for column, value in zip(state_columns, values, strict=True)
    }


def _nearest_state_distance(
    states: np.ndarray,
    support_samples: np.ndarray,
) -> np.ndarray:
    """Distance to sampled training-state support in normalized coordinates."""

    values = np.asarray(states, dtype=np.float64)
    support = np.asarray(support_samples, dtype=np.float64)
    if values.ndim == 0 or support.ndim != 2:
        raise ValueError("state support expects state vectors and a sample matrix")
    if values.shape[-1] != support.shape[-1]:
        raise ValueError("state and support dimensions disagree")
    flat = values.reshape(-1, values.shape[-1])
    distances = []
    for start in range(0, len(flat), 512):
        chunk = flat[start : start + 512]
        squared = np.square(chunk[:, None, :] - support[None, :, :]).sum(axis=-1)
        distances.append(np.sqrt(squared.min(axis=1)))
    return np.concatenate(distances).reshape(values.shape[:-1])


def _one_step_rmse(
    model: FiberedKoopmanModel,
    trajectories: np.ndarray,
    invariants: np.ndarray,
) -> float:
    repeated = np.repeat(invariants[:, None], trajectories.shape[1] - 1, axis=1)
    prediction = model.predict_one_step(trajectories[:, :-1], repeated)
    return _normalized_rmse(prediction, trajectories[:, 1:])


def _mean_valid_time(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    dt: float,
    threshold: float = 0.5,
) -> float:
    errors = np.sqrt(np.mean(np.square(prediction - truth), axis=-1))
    valid_times = []
    for row in errors:
        failures = np.flatnonzero(row > threshold)
        valid_times.append((int(failures[0]) if len(failures) else len(row) - 1) * dt)
    return float(np.mean(valid_times))


def _reference_alignment(
    coordinate: np.ndarray,
    reference: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict[str, float]:
    design = np.column_stack(
        (coordinate[train_indices], np.ones(len(train_indices), dtype=np.float64))
    )
    slope, intercept = np.linalg.lstsq(
        design,
        reference[train_indices],
        rcond=None,
    )[0]
    prediction = slope * coordinate[test_indices] + intercept
    truth = reference[test_indices]
    residual = float(np.square(prediction - truth).sum())
    total = float(np.square(truth - truth.mean()).sum())
    ranks_coordinate = np.argsort(np.argsort(coordinate[test_indices], kind="stable"))
    ranks_reference = np.argsort(np.argsort(truth, kind="stable"))
    rank_correlation = (
        float(np.corrcoef(ranks_coordinate, ranks_reference)[0, 1])
        if np.std(ranks_coordinate) > 0.0 and np.std(ranks_reference) > 0.0
        else 0.0
    )
    return {
        "fit_uses_training_trajectories_only": True,
        "held_out_affine_r2": 1.0 - residual / max(total, 1e-12),
        "held_out_absolute_rank_correlation": abs(rank_correlation),
        "orientation": 1.0 if slope >= 0.0 else -1.0,
        "slope": float(slope),
        "intercept": float(intercept),
    }


def _assert_finite(value: Any, path: str = "manifest") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} is not finite")


def validate_workbench_manifest(manifest: dict[str, Any]) -> list[str]:
    """Reject leakage, collapsed coordinates, and stale promotion verdicts."""

    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported mechanics-workbench schema")
    _assert_finite(manifest)
    split = manifest["split"]
    if set(split["training_trajectory_ids"]).intersection(
        split["held_out_trajectory_ids"]
    ):
        raise ValueError("trajectory leakage between training and held-out data")
    contract = manifest["scientific_contract"]
    if "reference column" not in contract["training_excludes"]:
        raise ValueError("reference-label exclusion is missing")
    invariant = manifest["invariant"]
    if invariant["training_coordinate_std"] <= 0.05:
        raise ValueError("candidate invariant collapsed")
    if invariant["held_out_mean_normalized_drift"] >= 0.5:
        raise ValueError("candidate invariant drifts excessively")
    operator = manifest["operator_family"]
    if operator.get("held_out_conditioning") != "initial_state_only":
        raise ValueError("held-out rollout may be conditioned on future samples")
    errors = operator["held_out_errors"]
    state_support = manifest["state_support"]
    comparisons = {
        "invariant_is_stable": (
            invariant["held_out_max_normalized_drift"] < 0.2
        ),
        "held_out_coordinates_are_in_range": (
            invariant["held_out_interpolation_coverage"] == 1.0
        ),
        "held_out_initial_states_are_near_training_data": (
            state_support["held_out_initial_coverage"] == 1.0
        ),
        "beats_global_edmd_rollout": (
            errors["fibered"]["normalized_rollout_rmse"]
            < errors["global_edmd"]["normalized_rollout_rmse"]
        ),
        "beats_persistence_rollout": (
            errors["fibered"]["normalized_rollout_rmse"]
            < errors["persistence"]["normalized_rollout_rmse"]
        ),
    }
    if comparisons != manifest["certificate"]["decisive_comparisons"]:
        raise ValueError("mechanics-workbench certificate is stale")
    expected = (
        "supported_on_held_out_trajectories"
        if all(comparisons.values())
        else "not_supported_by_current_dataset"
    )
    if manifest["certificate"]["status"] != expected:
        raise ValueError("mechanics-workbench status disagrees with measured errors")
    artifacts = manifest["artifacts"]
    for name in ("model", "overview", "report"):
        digest = artifacts.get(f"{name}_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{name} artifact SHA-256 is missing")
    return [
        "training and held-out trajectories are disjoint",
        "all held-out initial states pass invariant and sampled-state support gates",
        "candidate invariant is noncollapsed and trajectory-stable on every held-out run",
        f"fibered operator verdict is {expected}",
    ]


def _plot_overview(
    path: Path,
    *,
    dataset: TrajectoryDataset,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    orbit_means: np.ndarray,
    orbit_stds: np.ndarray,
    errors: dict[str, dict[str, float]],
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    spectrum: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    order = np.argsort(orbit_means)
    training = set(int(index) for index in train_indices)
    colors = ["#2563eb" if int(index) in training else "#f97316" for index in order]
    axes[0, 0].errorbar(
        np.arange(len(order)),
        orbit_means[order],
        yerr=orbit_stds[order],
        fmt="none",
        ecolor="#94a3b8",
        alpha=0.7,
    )
    axes[0, 0].scatter(np.arange(len(order)), orbit_means[order], c=colors, s=28)
    axes[0, 0].set_title("Discovered trajectory coordinate")
    axes[0, 0].set_xlabel("trajectories sorted by learned coordinate")
    axes[0, 0].set_ylabel("mean ± within-trajectory drift")

    labels = ["fibered", "global_edmd", "persistence"]
    axes[0, 1].bar(
        ["fibered\nK(I)", "global\nEDMD", "persistence"],
        [errors[label]["normalized_rollout_rmse"] for label in labels],
        color=["#7c3aed", "#64748b", "#cbd5e1"],
    )
    axes[0, 1].set_title("Held-out recursive rollout")
    axes[0, 1].set_ylabel("normalized state RMSE; lower is better")

    time = dataset.times[test_indices[0]] - dataset.times[test_indices[0], 0]
    axes[1, 0].plot(time, truth[0, :, 0], color="#111827", label="measured")
    axes[1, 0].plot(
        time,
        predictions["fibered"][0, :, 0],
        color="#7c3aed",
        label="fibered K(I)",
    )
    axes[1, 0].plot(
        time,
        predictions["global_edmd"][0, :, 0],
        color="#64748b",
        linestyle="--",
        label="global EDMD",
    )
    axes[1, 0].set_title(
        f"Held-out run {dataset.trajectory_ids[int(test_indices[0])]!r}: "
        f"{dataset.state_columns[0]}"
    )
    axes[1, 0].set_xlabel("time")
    axes[1, 0].legend()

    spectral_coordinates = [row["invariant"] for row in spectrum]
    frequencies = [
        row["lowest_nonzero_principal_frequency_hz"]
        if row["lowest_nonzero_principal_frequency_hz"] is not None
        else 0.0
        for row in spectrum
    ]
    axes[1, 1].plot(spectral_coordinates, frequencies, "o-", color="#0891b2")
    axes[1, 1].set_title("Fitted finite-operator spectrum across fibers")
    axes[1, 1].set_xlabel("learned invariant coordinate")
    axes[1, 1].set_ylabel("lowest principal-branch frequency [Hz]")

    figure.suptitle("Koopman Mechanics Workbench", fontsize=18, fontweight="bold")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    dataset = manifest["dataset"]
    invariant = manifest["invariant"]
    errors = manifest["operator_family"]["held_out_errors"]
    certificate = manifest["certificate"]
    reference = manifest["reference_evaluation"]
    comparison_rows = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{values['normalized_rollout_rmse']:.5f}</td>"
        "<td>"
        + ", ".join(
            f"<code>{html.escape(column)}</code>: {error:.5f}"
            for column, error in values["per_state_rmse"].items()
        )
        + "</td>"
        f"<td>{values['mean_valid_time']:.4f}</td>"
        "</tr>"
        for name, values in errors.items()
    )
    reference_html = (
        "<p>No reference invariant was supplied. The candidate is judged only "
        "by label-free constancy, noncollapse, held-out behavior, and operator utility.</p>"
        if reference is None
        else (
            f"<p>Optional post-training comparison to "
            f"<code>{html.escape(reference['column'])}</code>: held-out affine "
            f"R² <strong>{reference['held_out_affine_r2']:.4f}</strong>, absolute "
            "rank correlation <strong>"
            f"{reference['held_out_absolute_rank_correlation']:.4f}</strong>."
            " The reference was excluded from optimization.</p>"
        )
    )
    invariant_drift = invariant["held_out_mean_normalized_drift"]
    fibered_rollout = errors["fibered"]["normalized_rollout_rmse"]
    global_rollout = errors["global_edmd"]["normalized_rollout_rmse"]
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Koopman Mechanics Workbench</title>
<style>
body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; color: #172033;
       background: #f5f7fb; line-height: 1.55; }}
main {{ max-width: 1050px; margin: 0 auto; padding: 42px 24px 80px; }}
h1 {{ font-size: 2.35rem; margin-bottom: .25rem; }}
h2 {{ margin-top: 2.2rem; }}
.lede {{ font-size: 1.15rem; color: #475569; max-width: 850px; }}
.status {{ display: inline-block; padding: 7px 12px; border-radius: 999px;
           background: #ede9fe; color: #5b21b6; font-weight: 700; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(190px,1fr));
          gap: 14px; margin: 24px 0; }}
.card {{ background: white; border: 1px solid #dde3ee; border-radius: 12px; padding: 17px; }}
.metric {{ font-size: 1.65rem; font-weight: 750; }}
img {{ width: 100%; background: white; border: 1px solid #dde3ee; border-radius: 12px; }}
table {{ width: 100%; border-collapse: collapse; background: white; }}
th, td {{ padding: 10px 12px; border-bottom: 1px solid #e2e8f0; text-align: left; }}
code {{ background: #e9eef7; padding: 2px 5px; border-radius: 4px; }}
</style>
</head>
<body><main>
<p class="status">{html.escape(certificate['status'])}</p>
<h1>Koopman Mechanics Workbench</h1>
<p class="lede">A label-free candidate invariant organizes complete mechanical
trajectories, then indexes a polynomial family of local Koopman operators.
Every promoted statement below comes from held-out complete runs.</p>
<div class="cards">
  <div class="card"><div class="metric">{dataset['trajectory_count']}</div>trajectories</div>
  <div class="card"><div class="metric">{invariant_drift:.4f}</div>
  held-out invariant drift</div>
  <div class="card"><div class="metric">{fibered_rollout:.4f}</div>
  fibered rollout RMSE</div>
  <div class="card"><div class="metric">{global_rollout:.4f}</div>
  global EDMD RMSE</div>
</div>
<img src="overview.png" alt="Invariant, rollout, trajectory, and spectrum diagnostics">
<h2>What the model learned</h2>
<p>The neural scalar <em>I</em><sub>θ</sub>(x) was optimized to remain constant
inside each trajectory while remaining noncollapsed and smooth between nearby
trajectory sets. A transparent observable dictionary ψ(x) then evolves under
<strong>K(ĉ)=K₀+ĉK₁+ĉ²K₂</strong>, where ĉ is the normalized learned
coordinate. This is a finite, testable operator family—not a claim of one exact
global linearization.</p>
{reference_html}
<h2>Held-out model comparison</h2>
<table><thead><tr><th>model</th><th>normalized rollout RMSE</th>
<th>per-state RMSE in each column's native unit</th><th>mean valid time</th></tr></thead>
<tbody>{comparison_rows}</tbody></table>
<h2>Trust boundary</h2>
<ul>
<li>Training and evaluation are split by complete trajectory IDs.</li>
<li>The optional reference column never enters invariant or operator training.</li>
<li>The certificate is specific to this sampling interval, state definition,
observed invariant range, and sampled training-state neighborhood.</li>
<li>The exported model carries this fit verdict and refuses unsupported initial
states by default.</li>
<li>A negative certificate is useful evidence; it means this dictionary and
operator family should not be used as a surrogate for the supplied data.</li>
</ul>
<p>Machine-readable evidence: <code>manifest.json</code>. Loadable model:
<code>model.pt</code>.</p>
</main></body></html>
"""
    path.write_text(document, encoding="utf-8")


def _save_model(
    path: Path,
    *,
    invariant: LearnedInvariant,
    operator: FiberedKoopmanModel,
    state_mean: np.ndarray,
    state_scale: np.ndarray,
    state_columns: tuple[str, ...],
    hidden_dim: int,
    invariant_min: float,
    invariant_max: float,
    state_support_samples: np.ndarray,
    state_support_radius: float,
    certificate_status: str,
    decisive_comparisons: dict[str, bool],
) -> None:
    torch.save(
        {
            "schema_version": 1,
            "learned_koopman_version": __version__,
            "input_dim": len(state_columns),
            "hidden_dim": hidden_dim,
            "state_columns": list(state_columns),
            "state_mean": torch.tensor(state_mean, dtype=torch.float64),
            "state_scale": torch.tensor(state_scale, dtype=torch.float64),
            "invariant_min": invariant_min,
            "invariant_max": invariant_max,
            "state_support_samples": torch.tensor(
                state_support_samples,
                dtype=torch.float64,
            ),
            "state_support_radius": state_support_radius,
            "certificate_status": certificate_status,
            "decisive_comparisons": decisive_comparisons,
            "invariant_state_dict": invariant.state_dict(),
            "operator": operator.to_dict(),
        },
        path,
    )


def load_mechanics_model(path: Path) -> MechanicsModel:
    """Load a workbench export for invariant evaluation and recursive rollout."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported mechanics-model schema")
    invariant = LearnedInvariant(
        int(payload["hidden_dim"]),
        input_dim=int(payload["input_dim"]),
    )
    invariant.load_state_dict(payload["invariant_state_dict"])
    invariant.eval()
    operator_data = payload["operator"]
    operator = FiberedKoopmanModel(
        matrices=np.asarray(operator_data["matrices"], dtype=np.float64),
        invariant_center=float(operator_data["invariant_center"]),
        invariant_scale=float(operator_data["invariant_scale"]),
        state_dim=int(operator_data["state_dim"]),
        observable_degree=int(operator_data["observable_degree"]),
        dt=float(operator_data["dt"]),
        ridge=float(operator_data["ridge"]),
    )
    return MechanicsModel(
        invariant=invariant,
        operator=operator,
        state_mean=payload["state_mean"].numpy(),
        state_scale=payload["state_scale"].numpy(),
        state_columns=tuple(payload["state_columns"]),
        invariant_min=float(payload["invariant_min"]),
        invariant_max=float(payload["invariant_max"]),
        state_support_samples=payload["state_support_samples"].numpy(),
        state_support_radius=float(payload["state_support_radius"]),
        certificate_status=str(payload["certificate_status"]),
        decisive_comparisons={
            str(key): bool(value)
            for key, value in payload["decisive_comparisons"].items()
        },
    )


def run_mechanics_workbench(
    dataset: TrajectoryDataset,
    output_dir: Path,
    *,
    config: WorkbenchConfig,
) -> dict[str, Any]:
    """Discover an invariant, fit a fibered operator, and issue a trust certificate."""

    if config.epochs < 1 or config.hidden_dim < 1:
        raise ValueError("epochs and hidden_dim must be positive")
    _set_seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_indices, test_indices = _split_indices(
        dataset.trajectory_count,
        config.train_fraction,
        config.seed,
    )
    training_states = dataset.states[train_indices]
    state_mean = training_states.reshape(-1, dataset.state_dim).mean(axis=0)
    state_scale = training_states.reshape(-1, dataset.state_dim).std(axis=0)
    if np.any(state_scale <= 1e-10):
        collapsed = [
            column
            for column, scale in zip(dataset.state_columns, state_scale, strict=True)
            if scale <= 1e-10
        ]
        raise ValueError(f"state columns have no training variation: {collapsed}")
    normalized = (dataset.states - state_mean) / state_scale
    state_support_samples = normalized[train_indices, ::4].reshape(
        -1,
        dataset.state_dim,
    )
    support_calibration = normalized[train_indices, 2::4].reshape(
        -1,
        dataset.state_dim,
    )
    calibration_distances = _nearest_state_distance(
        support_calibration,
        state_support_samples,
    )
    state_support_radius = float(
        max(np.quantile(calibration_distances, 0.995) * 1.2, 1e-8)
    )
    held_out_initial_state_distances = _nearest_state_distance(
        normalized[test_indices, 0],
        state_support_samples,
    )
    held_out_initial_state_coverage = float(
        np.mean(held_out_initial_state_distances <= state_support_radius)
    )

    invariant_model, history = train_invariant_model(
        torch.tensor(normalized[train_indices], dtype=torch.float32),
        hidden_dim=config.hidden_dim,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        seed=config.seed,
        graph_neighbors=config.graph_neighbors,
        constancy_weight=config.constancy_weight,
        graph_weight=config.graph_weight,
        variance_weight=config.variance_weight,
        centering_weight=config.centering_weight,
    )
    coordinate_values, orbit_means, orbit_stds = _orbit_coordinate(
        invariant_model,
        normalized,
    )
    training_coordinate_std = float(orbit_means[train_indices].std())
    held_out_normalized_drift = orbit_stds[test_indices] / max(
        training_coordinate_std,
        1e-12,
    )
    training_min = float(coordinate_values[train_indices].min())
    training_max = float(coordinate_values[train_indices].max())
    test_invariants = coordinate_values[test_indices, 0]
    interpolation_coverage = float(
        np.mean(
            (test_invariants >= training_min)
            & (test_invariants <= training_max)
        )
    )

    fibered = fit_fibered_operator(
        normalized[train_indices],
        orbit_means[train_indices],
        dt=dataset.dt,
        family_degree=config.family_degree,
        observable_degree=config.observable_degree,
        ridge=config.ridge,
    )
    global_edmd = fit_fibered_operator(
        normalized[train_indices],
        orbit_means[train_indices],
        dt=dataset.dt,
        family_degree=0,
        observable_degree=config.observable_degree,
        ridge=config.ridge,
    )

    truth = normalized[test_indices]
    predictions = {
        "fibered": fibered.rollout(
            truth[:, 0],
            test_invariants,
            steps=dataset.step_count,
        ),
        "global_edmd": global_edmd.rollout(
            truth[:, 0],
            test_invariants,
            steps=dataset.step_count,
        ),
        "persistence": np.repeat(
            truth[:, :1],
            dataset.step_count,
            axis=1,
        ),
    }
    held_out_errors: dict[str, dict[str, float]] = {}
    for name, prediction in predictions.items():
        held_out_errors[name] = {
            "normalized_rollout_rmse": _normalized_rmse(prediction, truth),
            "per_state_rmse": _per_state_rmse(
                prediction,
                truth,
                state_scale,
                dataset.state_columns,
            ),
            "mean_valid_time": _mean_valid_time(
                prediction,
                truth,
                dt=dataset.dt,
            ),
        }
    held_out_errors["fibered"]["normalized_one_step_rmse"] = _one_step_rmse(
        fibered,
        truth,
        test_invariants,
    )
    held_out_errors["global_edmd"]["normalized_one_step_rmse"] = _one_step_rmse(
        global_edmd,
        truth,
        test_invariants,
    )

    comparisons = {
        "invariant_is_stable": (
            float(held_out_normalized_drift.max()) < 0.2
        ),
        "held_out_coordinates_are_in_range": interpolation_coverage == 1.0,
        "held_out_initial_states_are_near_training_data": (
            held_out_initial_state_coverage == 1.0
        ),
        "beats_global_edmd_rollout": (
            held_out_errors["fibered"]["normalized_rollout_rmse"]
            < held_out_errors["global_edmd"]["normalized_rollout_rmse"]
        ),
        "beats_persistence_rollout": (
            held_out_errors["fibered"]["normalized_rollout_rmse"]
            < held_out_errors["persistence"]["normalized_rollout_rmse"]
        ),
    }
    status = (
        "supported_on_held_out_trajectories"
        if all(comparisons.values())
        else "not_supported_by_current_dataset"
    )
    quantiles = np.quantile(
        orbit_means[train_indices],
        [0.05, 0.25, 0.5, 0.75, 0.95],
    )
    spectrum = spectral_summary(fibered, quantiles)

    reference_evaluation = None
    if dataset.reference_values is not None:
        reference_evaluation = {
            "column": dataset.reference_column,
            "reference_max_relative_drift": dataset.reference_max_relative_drift,
            **_reference_alignment(
                orbit_means,
                dataset.reference_values,
                train_indices,
                test_indices,
            ),
        }

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "tool": "koopman_mechanics_workbench",
        "learned_koopman_version": __version__,
        "profile": config.profile,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "source_revision": _git_source_state(),
        "dataset": {
            "source": dataset.source,
            "source_sha256": dataset.source_sha256,
            "trajectory_count": dataset.trajectory_count,
            "samples_per_trajectory": dataset.step_count,
            "state_columns": list(dataset.state_columns),
            "trajectory_column": dataset.trajectory_column,
            "time_column": dataset.time_column,
            "state_dimension": dataset.state_dim,
            "sampling_interval": dataset.dt,
            "original_lengths": list(dataset.original_lengths),
        },
        "split": {
            "seed": config.seed,
            "training_trajectory_ids": [
                dataset.trajectory_ids[int(index)] for index in train_indices
            ],
            "held_out_trajectory_ids": [
                dataset.trajectory_ids[int(index)] for index in test_indices
            ],
        },
        "scientific_contract": {
            "training_inputs": [
                "state trajectories",
                "time samples",
                "trajectory membership",
            ],
            "training_excludes": [
                "reference column",
                "known energy or Hamiltonian",
                "trajectory ordering labels",
                "future held-out trajectories",
            ],
            "candidate_invariant": (
                "I_theta is approximately constant on observed trajectories "
                "and is generally identifiable only up to a smooth monotone "
                "reparameterization"
            ),
            "operator_family": (
                "psi(x_next) is regressed against a polynomial family K(I); "
                "finite residuals do not imply an exact Koopman-invariant subspace"
            ),
            "held_out_conditioning": (
                "recursive forecasts use I_theta(x[0]) from the initial held-out "
                "state only; future held-out samples never condition a forecast"
            ),
        },
        "config": asdict(config),
        "normalization": {
            "state_mean": state_mean.tolist(),
            "state_scale": state_scale.tolist(),
        },
        "invariant": {
            "training_coordinate_std": training_coordinate_std,
            "held_out_mean_normalized_drift": float(
                held_out_normalized_drift.mean()
            ),
            "held_out_max_normalized_drift": float(
                held_out_normalized_drift.max()
            ),
            "held_out_interpolation_coverage": interpolation_coverage,
            "training": {
                "initial_loss": history[0]["loss"],
                "final_loss": history[-1]["loss"],
                "final_constancy": history[-1]["constancy"],
                "final_orbit_std": history[-1]["orbit_std"],
            },
            "per_trajectory": [
                {
                    "trajectory_id": trajectory_id,
                    "split": (
                        "train" if index in set(train_indices.tolist()) else "held_out"
                    ),
                    "mean": float(orbit_means[index]),
                    "standard_deviation": float(orbit_stds[index]),
                }
                for index, trajectory_id in enumerate(dataset.trajectory_ids)
            ],
        },
        "state_support": {
            "method": (
                "nearest sampled training state in normalized coordinates; "
                "radius is 1.2 times the 99.5th percentile of interleaved "
                "training calibration distances"
            ),
            "sample_count": int(len(state_support_samples)),
            "radius": state_support_radius,
            "held_out_initial_coverage": held_out_initial_state_coverage,
            "held_out_initial_max_distance": float(
                held_out_initial_state_distances.max()
            ),
            "held_out_initial_distances": {
                dataset.trajectory_ids[int(index)]: float(distance)
                for index, distance in zip(
                    test_indices,
                    held_out_initial_state_distances,
                    strict=True,
                )
            },
        },
        "operator_family": {
            "equation": "psi(x[k+1]) = psi(x[k]) @ sum_r c_hat^r K_r",
            "held_out_conditioning": "initial_state_only",
            "model": fibered.to_dict(),
            "observable_features": list(
                observable_feature_names(
                    dataset.state_columns,
                    degree=config.observable_degree,
                )
            ),
            "held_out_errors": held_out_errors,
            "local_spectrum": spectrum,
        },
        "reference_evaluation": reference_evaluation,
        "certificate": {
            "status": status,
            "decisive_comparisons": comparisons,
            "scope": (
                "complete held-out trajectories at the supplied state definition, "
                "sampling interval, observed invariant range, and sampled-state "
                "support radius"
            ),
            "warnings": [
                warning
                for warning, active in (
                    (
                        "some held-out invariant values are outside the training range",
                        interpolation_coverage < 1.0,
                    ),
                    (
                        "candidate invariant has appreciable held-out drift",
                        float(held_out_normalized_drift.max()) >= 0.2,
                    ),
                    (
                        "some held-out initial states are far from sampled "
                        "training states",
                        held_out_initial_state_coverage < 1.0,
                    ),
                    (
                        "fibered operator does not improve on global EDMD",
                        not comparisons["beats_global_edmd_rollout"],
                    ),
                )
                if active
            ],
        },
        "artifacts": {
            "report": "report.html",
            "overview": "overview.png",
            "model": "model.pt",
            "manifest": "manifest.json",
        },
    }
    _save_model(
        output_dir / "model.pt",
        invariant=invariant_model,
        operator=fibered,
        state_mean=state_mean,
        state_scale=state_scale,
        state_columns=dataset.state_columns,
        hidden_dim=config.hidden_dim,
        invariant_min=training_min,
        invariant_max=training_max,
        state_support_samples=state_support_samples,
        state_support_radius=state_support_radius,
        certificate_status=status,
        decisive_comparisons=comparisons,
    )
    _plot_overview(
        output_dir / "overview.png",
        dataset=dataset,
        train_indices=train_indices,
        test_indices=test_indices,
        orbit_means=orbit_means,
        orbit_stds=orbit_stds,
        errors=held_out_errors,
        truth=truth * state_scale + state_mean,
        predictions={
            name: prediction * state_scale + state_mean
            for name, prediction in predictions.items()
        },
        spectrum=spectrum,
    )
    _write_report(output_dir / "report.html", manifest)
    manifest["artifacts"].update(
        {
            "model_sha256": _sha256(output_dir / "model.pt"),
            "overview_sha256": _sha256(output_dir / "overview.png"),
            "report_sha256": _sha256(output_dir / "report.html"),
        }
    )
    checks = validate_workbench_manifest(manifest)
    manifest["validation_checks"] = checks
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest
