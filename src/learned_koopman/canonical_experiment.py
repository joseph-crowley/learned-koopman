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
from torch.nn import functional as F

from learned_koopman import __version__
from learned_koopman.canonical_diagnostics import (
    diagnose_canonical_orbits,
    summarize_orbit_diagnostics,
)
from learned_koopman.canonical_model import (
    CanonicalKoopmanModel,
    CanonicalKoopmanNetwork,
    save_canonical_model,
)
from learned_koopman.hj_action import run_hj_action_audit
from learned_koopman.trajectory import TrajectoryDataset


@dataclass(frozen=True)
class CanonicalExperimentConfig:
    profile: str
    seed: int
    train_fraction: float
    hidden_dim: int
    shear_layers: int
    hamiltonian_degree: int
    epochs: int
    batch_size: int
    rollout_horizon: int
    learning_rate: float
    action_weight: float
    latent_weight: float

    @classmethod
    def quick(cls, seed: int = 7) -> CanonicalExperimentConfig:
        return cls(
            profile="quick",
            seed=seed,
            train_fraction=0.75,
            hidden_dim=24,
            shear_layers=6,
            hamiltonian_degree=3,
            epochs=350,
            batch_size=384,
            rollout_horizon=6,
            learning_rate=2e-3,
            action_weight=0.35,
            latent_weight=0.3,
        )

    @classmethod
    def full(cls, seed: int = 7) -> CanonicalExperimentConfig:
        return cls(
            profile="full",
            seed=seed,
            train_fraction=0.75,
            hidden_dim=40,
            shear_layers=8,
            hamiltonian_degree=4,
            epochs=1200,
            batch_size=768,
            rollout_horizon=12,
            learning_rate=1.2e-3,
            action_weight=0.45,
            latent_weight=0.35,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


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


def _split_indices(
    count: int,
    train_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    order = generator.permutation(count)
    test_count = max(2, int(round(count * (1.0 - train_fraction))))
    test_count = min(test_count, count - 4)
    if test_count < 2:
        raise ValueError("need four training and two held-out trajectories")
    return np.sort(order[test_count:]), np.sort(order[:test_count])


def _training_batch(
    states: torch.Tensor,
    train_indices: np.ndarray,
    *,
    batch_size: int,
    horizon: int,
    generator: torch.Generator,
) -> torch.Tensor:
    trajectory_choices = torch.randint(
        0,
        len(train_indices),
        (batch_size,),
        generator=generator,
    )
    trajectory_indices = torch.tensor(train_indices, dtype=torch.long)[
        trajectory_choices
    ]
    starts = torch.randint(
        0,
        states.shape[1] - horizon,
        (batch_size,),
        generator=generator,
    )
    offsets = torch.arange(horizon + 1)
    return states[trajectory_indices[:, None], starts[:, None] + offsets[None, :]]


def _fit_network(
    network: CanonicalKoopmanNetwork,
    dataset: TrajectoryDataset,
    train_indices: np.ndarray,
    config: CanonicalExperimentConfig,
) -> list[dict[str, float]]:
    states = torch.tensor(dataset.states, dtype=torch.float32)
    state_scale = states[train_indices].reshape(-1, 2).std(dim=0).clamp_min(1e-4)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed + 103)
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        batch = _training_batch(
            states,
            train_indices,
            batch_size=config.batch_size,
            horizon=config.rollout_horizon,
            generator=generator,
        )
        observed_latent = network.encode(batch)
        observed_action = network.action_from_latent(observed_latent)
        action_scale = observed_action[:, 0].std().detach().clamp_min(0.05)
        latent_scale = observed_latent.detach().std(dim=(0, 1)).clamp_min(0.05)
        action_loss = torch.mean(
            torch.square(
                (observed_action - observed_action[:, :1]) / action_scale
            )
        )
        state = batch[:, 0]
        rollout_loss = torch.zeros((), dtype=torch.float32)
        latent_loss = torch.zeros((), dtype=torch.float32)
        for step in range(1, config.rollout_horizon + 1):
            predicted_latent = network.latent_step(network.encode(state))
            state = network.decode(predicted_latent)
            rollout_loss = rollout_loss + torch.mean(
                torch.square((state - batch[:, step]) / state_scale)
            )
            latent_loss = latent_loss + torch.mean(
                torch.square(
                    (predicted_latent - observed_latent[:, step]) / latent_scale
                )
            )
        rollout_loss = rollout_loss / config.rollout_horizon
        latent_loss = latent_loss / config.rollout_horizon
        sampled_frequency = network.hamiltonian.frequency(observed_action.detach())
        positive_frequency_penalty = torch.mean(
            torch.square(F.relu(0.05 - sampled_frequency))
        )
        coefficient_penalty = torch.mean(
            torch.square(network.hamiltonian.higher_frequency_coefficients)
        )
        loss = (
            rollout_loss
            + config.latent_weight * latent_loss
            + config.action_weight * action_loss
            + 0.1 * positive_frequency_penalty
            + 1e-5 * coefficient_penalty
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
        optimizer.step()
        if epoch % max(1, config.epochs // 40) == 0 or epoch == config.epochs - 1:
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": float(loss.detach()),
                    "rollout_loss": float(rollout_loss.detach()),
                    "latent_loss": float(latent_loss.detach()),
                    "action_loss": float(action_loss.detach()),
                }
            )
    return history


def _rollout_metrics(
    network: CanonicalKoopmanNetwork,
    states: np.ndarray,
    indices: np.ndarray,
    state_scale: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    truth = torch.tensor(states[indices], dtype=torch.float32)
    with torch.no_grad():
        prediction = network.rollout(truth[:, 0], steps=truth.shape[1])
        one_step = network(truth[:, :-1])
    scale = torch.tensor(state_scale, dtype=torch.float32)
    rollout_rmse = torch.sqrt(
        torch.mean(torch.square((prediction - truth) / scale))
    )
    one_step_rmse = torch.sqrt(
        torch.mean(torch.square((one_step - truth[:, 1:]) / scale))
    )
    persistence = truth[:, :1].expand_as(truth)
    persistence_rmse = torch.sqrt(
        torch.mean(torch.square((persistence - truth) / scale))
    )
    return (
        {
            "normalized_one_step_rmse": float(one_step_rmse),
            "normalized_rollout_rmse": float(rollout_rmse),
            "persistence_normalized_rollout_rmse": float(persistence_rmse),
        },
        prediction.numpy().astype(np.float64),
    )


def _structure_metrics(
    network: CanonicalKoopmanNetwork,
    states: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    truth = torch.tensor(states[indices], dtype=torch.float32)
    with torch.no_grad():
        latent = network.encode(truth)
        action = network.action_from_latent(latent)
        action_means = action.mean(dim=1)
        action_scale = action_means.std().clamp_min(1e-8)
        normalized_action_drift = action.std(dim=1) / action_scale
        q, p = latent.unbind(dim=-1)
        radius = torch.sqrt(torch.clamp(q * q + p * p, min=1e-12))
        eigenfunction = torch.complex(q / radius, -p / radius)
        frequency = network.hamiltonian.frequency(action[:, :-1])
        multiplier = torch.exp(torch.complex(torch.zeros_like(frequency), frequency * network.dt))
        koopman_residual = torch.abs(
            eigenfunction[:, 1:] - multiplier * eigenfunction[:, :-1]
        )
        reconstructed = network.decode(latent)
        inverse_error = torch.max(torch.abs(reconstructed - truth))
        predicted = network.rollout(truth[:, 0], steps=truth.shape[1])
        predicted_action = network.action(predicted)
        exact_action_drift = torch.max(
            torch.abs(predicted_action - predicted_action[:, :1])
        )
    samples = truth[:, :: max(1, truth.shape[1] // 4)].reshape(-1, 2)[:16]
    symplectic_form = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
    defects = []
    for sample in samples:
        point = sample.detach().requires_grad_(True)
        jacobian = torch.autograd.functional.jacobian(
            lambda value: network(value.unsqueeze(0))[0],
            point,
        )
        defect = jacobian.T @ symplectic_form @ jacobian - symplectic_form
        defects.append(float(torch.max(torch.abs(defect))))
    orbit_diagnostics = summarize_orbit_diagnostics(
        diagnose_canonical_orbits(network, states[indices])
    )
    return {
        "held_out_mean_normalized_action_drift": float(
            normalized_action_drift.mean()
        ),
        "held_out_max_normalized_action_drift": float(
            normalized_action_drift.max()
        ),
        "held_out_mean_koopman_eigenfunction_residual": float(
            koopman_residual.mean()
        ),
        "maximum_inverse_error": float(inverse_error),
        "maximum_model_rollout_action_drift": float(exact_action_drift),
        "maximum_symplectic_defect": max(defects),
        "held_out_mean_radial_coefficient_of_variation": orbit_diagnostics[
            "mean_radial_coefficient_of_variation"
        ],
        "held_out_mean_phase_step_coefficient_of_variation": orbit_diagnostics[
            "mean_phase_step_coefficient_of_variation"
        ],
        "held_out_mean_phase_law_rmse_radians": orbit_diagnostics[
            "mean_phase_law_rmse_radians"
        ],
        "held_out_mean_normalized_conjugacy_rmse": orbit_diagnostics[
            "mean_normalized_conjugacy_rmse"
        ],
        "held_out_orbit_diagnostic_verdicts": orbit_diagnostics["per_trajectory"],
    }


def _learned_hamiltonian_metrics(
    network: CanonicalKoopmanNetwork,
    dataset: TrajectoryDataset,
    action_audit: dict[str, Any],
    trajectory_ids: set[str],
) -> dict[str, Any]:
    all_measurements = action_audit["measurements"]
    all_coordinate = np.asarray(
        action_audit["learned_coordinate_alignment"]["coordinate"],
        dtype=np.float64,
    )
    selected = np.asarray(
        [
            index
            for index, row in enumerate(all_measurements)
            if row["trajectory_id"] in trajectory_ids
        ],
        dtype=np.int64,
    )
    measurements = [all_measurements[index] for index in selected]
    coordinate = all_coordinate[selected]
    measured_frequency = np.asarray(
        [row["frequency"] for row in measurements],
        dtype=np.float64,
    )
    with torch.no_grad():
        values = torch.tensor(coordinate, dtype=torch.float32)
        learned_frequency = (
            network.hamiltonian.frequency(values).numpy().astype(np.float64)
        )
        learned_energy = network.hamiltonian(values).numpy().astype(np.float64)
    frequency_error = learned_frequency - measured_frequency
    result: dict[str, Any] = {
        "training_uses_reference_hamiltonian": False,
        "evaluation_trajectory_ids": [
            row["trajectory_id"] for row in measurements
        ],
        "equations": ["H(q,p) = h(I(q,p))", "omega(I) = dh/dI"],
        "frequency_normalized_rmse": float(
            np.sqrt(np.mean(frequency_error**2))
            / max(float(np.sqrt(np.mean(measured_frequency**2))), 1e-12)
        ),
        "frequency_median_relative_error": float(
            np.median(
                np.abs(frequency_error) / np.maximum(np.abs(measured_frequency), 1e-12)
            )
        ),
        "learned_action": coordinate.tolist(),
        "measured_frequency": measured_frequency.tolist(),
        "learned_frequency": learned_frequency.tolist(),
        "learned_hamiltonian": learned_energy.tolist(),
    }
    if dataset.reference_values is None:
        result.update(
            {
                "reference_energy_available": False,
                "reason": "no reference Hamiltonian column was supplied",
            }
        )
        return result
    reference_by_id = {
        trajectory_id: float(dataset.reference_values[index])
        for index, trajectory_id in enumerate(dataset.trajectory_ids)
    }
    reference_energy = np.asarray(
        [reference_by_id[row["trajectory_id"]] for row in measurements],
        dtype=np.float64,
    )
    additive_offset = float(np.mean(reference_energy - learned_energy))
    adjusted_energy = learned_energy + additive_offset
    energy_error = adjusted_energy - reference_energy
    result.update(
        {
            "reference_energy_available": True,
            "reference_energy": reference_energy.tolist(),
            "energy_additive_offset": additive_offset,
            "energy_normalized_rmse": float(
                np.sqrt(np.mean(energy_error**2))
                / max(float(np.std(reference_energy)), 1e-12)
            ),
            "energy_max_absolute_error": float(np.max(np.abs(energy_error))),
        }
    )
    return result


def _held_out_action_alignment(
    action_audit: dict[str, Any],
    trajectory_ids: set[str],
) -> dict[str, Any]:
    measurements = action_audit["measurements"]
    all_coordinate = np.asarray(
        action_audit["learned_coordinate_alignment"]["coordinate"],
        dtype=np.float64,
    )
    selected = np.asarray(
        [
            index
            for index, row in enumerate(measurements)
            if row["trajectory_id"] in trajectory_ids
        ],
        dtype=np.int64,
    )
    coordinate = all_coordinate[selected]
    action = np.asarray(
        [measurements[index]["action"] for index in selected],
        dtype=np.float64,
    )
    design = np.column_stack((coordinate, np.ones(len(coordinate))))
    slope, intercept = np.linalg.lstsq(design, action, rcond=None)[0]
    prediction = slope * coordinate + intercept
    residual = float(np.square(prediction - action).sum())
    total = float(np.square(action - action.mean()).sum())
    coordinate_rank = np.argsort(np.argsort(coordinate, kind="stable"), kind="stable")
    action_rank = np.argsort(np.argsort(action, kind="stable"), kind="stable")
    return {
        "scope": "complete held-out trajectories only",
        "trajectory_ids": [
            measurements[index]["trajectory_id"] for index in selected
        ],
        "affine_r2": 1.0 - residual / max(total, 1e-12),
        "absolute_rank_correlation": abs(
            float(np.corrcoef(coordinate_rank, action_rank)[0, 1])
        ),
        "affine_slope": float(slope),
        "affine_intercept": float(intercept),
        "learned_action": coordinate.tolist(),
        "empirical_action": action.tolist(),
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


def _chart_diagnostic_comparisons(structure: dict[str, Any]) -> dict[str, bool]:
    required = {
        "held_out_mean_radial_coefficient_of_variation",
        "held_out_mean_phase_step_coefficient_of_variation",
        "held_out_mean_normalized_conjugacy_rmse",
    }
    if not required.issubset(structure):
        return {}
    return {
        "chart_circularizes_held_out_orbits": (
            structure["held_out_mean_radial_coefficient_of_variation"] < 0.08
        ),
        "observed_phase_step_is_nearly_uniform": (
            structure["held_out_mean_phase_step_coefficient_of_variation"] < 0.08
        ),
        "complete_latent_conjugacy_fits_observed_steps": (
            structure["held_out_mean_normalized_conjugacy_rmse"] < 0.08
        ),
    }


def validate_canonical_manifest(manifest: dict[str, Any]) -> list[str]:
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported canonical Koopman experiment schema")
    _assert_finite(manifest)
    if set(manifest["split"]["training_trajectory_ids"]).intersection(
        manifest["split"]["held_out_trajectory_ids"]
    ):
        raise ValueError("trajectory leakage between training and held-out data")
    held_out = manifest["held_out_evaluation"]
    structure = manifest["structure_evaluation"]
    action_alignment = manifest["canonical_action_evaluation"]
    learned_hamiltonian = manifest["learned_hamiltonian_evaluation"]
    comparisons = {
        "held_out_rollout_is_accurate": (
            held_out["normalized_rollout_rmse"] < 0.25
        ),
        "beats_persistence_rollout": (
            held_out["normalized_rollout_rmse"]
            < held_out["persistence_normalized_rollout_rmse"]
        ),
        "observed_orbits_have_stable_action": (
            structure["held_out_mean_normalized_action_drift"] < 0.08
        ),
        "koopman_phase_law_fits_observed_steps": (
            structure["held_out_mean_koopman_eigenfunction_residual"] < 0.08
        ),
        "map_is_numerically_invertible": (
            structure["maximum_inverse_error"] < 1e-5
        ),
        "map_is_numerically_symplectic": (
            structure["maximum_symplectic_defect"] < 2e-5
        ),
        "latent_action_is_exactly_conserved_by_model": (
            structure["maximum_model_rollout_action_drift"] < 2e-4
        ),
        "symplectic_gauge_matches_empirical_action": (
            action_alignment["affine_r2"] > 0.97
            and abs(abs(action_alignment["affine_slope"]) - 1.0) < 0.2
        ),
        "learned_hamiltonian_matches_frequency": (
            learned_hamiltonian["frequency_normalized_rmse"] < 0.05
        ),
    }
    comparisons.update(_chart_diagnostic_comparisons(structure))
    if learned_hamiltonian["reference_energy_available"]:
        comparisons["learned_hamiltonian_matches_reference_shape"] = (
            learned_hamiltonian["energy_normalized_rmse"] < 0.05
        )
    if comparisons != manifest["certificate"]["decisive_comparisons"]:
        raise ValueError("canonical Koopman certificate is stale")
    expected = (
        "supported_on_held_out_trajectories"
        if all(comparisons.values())
        else "not_supported_by_current_dataset"
    )
    if manifest["certificate"]["status"] != expected:
        raise ValueError("canonical Koopman status disagrees with measured evidence")
    artifacts = manifest["artifacts"]
    for name in ("model", "overview", "report", "action_audit_manifest"):
        digest = artifacts.get(f"{name}_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{name} artifact SHA-256 is missing")
    return [
        "training and held-out trajectories are disjoint",
        "rollout quality and observed Koopman residual were measured on held-out runs",
        "invertibility, symplecticity, and model action conservation were differentiated",
        "latent action was checked against the physical closed-orbit area",
        f"canonical model verdict is {expected}",
    ]


def _plot_overview(
    path: Path,
    *,
    network: CanonicalKoopmanNetwork,
    dataset: TrajectoryDataset,
    test_indices: np.ndarray,
    prediction: np.ndarray,
    history: list[dict[str, float]],
    action_audit: dict[str, Any],
) -> None:
    truth = dataset.states[test_indices]
    with torch.no_grad():
        latent = network.encode(torch.tensor(truth, dtype=torch.float32)).numpy()
    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    showcase = -1
    axes[0, 0].plot(
        truth[showcase, :, 0],
        truth[showcase, :, 1],
        color="#1e293b",
        label="held-out truth",
    )
    axes[0, 0].plot(
        prediction[showcase, :, 0],
        prediction[showcase, :, 1],
        "--",
        color="#d85140",
        label="canonical model",
    )
    axes[0, 0].set(title="Recursive held-out rollout", xlabel="q", ylabel="p")
    axes[0, 0].legend()
    for row in latent:
        axes[0, 1].plot(row[:, 0], row[:, 1], alpha=0.8)
    axes[0, 1].set(
        title="Learned canonical chart",
        xlabel="Q",
        ylabel="P",
        aspect="equal",
    )
    measurements = action_audit["measurements"]
    physical_action = np.asarray([row["action"] for row in measurements])
    learned_action = np.asarray(
        action_audit["learned_coordinate_alignment"]["coordinate"]
    )
    axes[1, 0].scatter(physical_action, learned_action, color="#4057c9")
    limits = (
        min(float(physical_action.min()), float(learned_action.min())),
        max(float(physical_action.max()), float(learned_action.max())),
    )
    axes[1, 0].plot(limits, limits, color="#252525", alpha=0.7)
    axes[1, 0].set(
        title="Symplectic gauge fixing",
        xlabel=r"empirical $J=(2\pi)^{-1}\oint p\,dq$",
        ylabel=r"latent $I=(Q^2+P^2)/2$",
    )
    axes[1, 1].semilogy(
        [row["epoch"] for row in history],
        [row["loss"] for row in history],
        color="#2a8b68",
        label="total",
    )
    axes[1, 1].semilogy(
        [row["epoch"] for row in history],
        [row["action_loss"] for row in history],
        color="#7c3fb7",
        label="action",
    )
    axes[1, 1].set(title="Training", xlabel="epoch", ylabel="loss")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle("Exact-symplectic canonical Koopman world model", fontsize=15)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    certificate = manifest["certificate"]
    held_out = manifest["held_out_evaluation"]
    structure = manifest["structure_evaluation"]
    action = manifest["canonical_action_evaluation"]
    learned_hamiltonian = manifest["learned_hamiltonian_evaluation"]
    radial_residual = structure.get(
        "held_out_mean_radial_coefficient_of_variation",
        float("nan"),
    )
    phase_step_residual = structure.get(
        "held_out_mean_phase_step_coefficient_of_variation",
        float("nan"),
    )
    conjugacy_residual = structure.get(
        "held_out_mean_normalized_conjugacy_rmse",
        float("nan"),
    )
    rows = "\n".join(
        f"<li class=\"{'pass' if passed else 'fail'}\">"
        f"{'PASS' if passed else 'FAIL'} — {html.escape(name.replace('_', ' '))}</li>"
        for name, passed in certificate["decisive_comparisons"].items()
    )
    path.write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Canonical Koopman world model</title><style>
body {{ margin:0; background:#f2efe8; color:#20242b; font:17px/1.55 system-ui; }}
main {{ max-width:1000px; margin:auto; padding:54px 24px 80px; }}
h1 {{ font:700 clamp(38px,7vw,70px)/1.02 Georgia,serif; margin:8px 0 22px; }}
.eyebrow {{ color:#4057c9; font-weight:800; text-transform:uppercase; letter-spacing:.08em; }}
.card {{ background:white; padding:26px; border-radius:18px; margin:24px 0;
box-shadow:0 12px 38px #1e263112; }} img {{ width:100%; border-radius:12px; }}
.status {{ display:inline-block; padding:7px 13px; border-radius:99px;
background:#e7ecf8; font-weight:750; }} .pass {{color:#176e50}} .fail {{color:#ae352d}}
</style></head><body><main><div class="eyebrow">learned-koopman · Koopman + HJ</div>
<h1>A canonical world model, not another unconstrained rollout net.</h1>
<p class="status">{html.escape(certificate['status'])}</p>
<p>The model learns an exactly invertible symplectic map from physical
<strong>(q,p)</strong> into latent <strong>(Q,P)</strong>. There, motion is an
action-conditioned rotation generated by a learned scalar Hamiltonian
<strong>h(I)</strong>. Its inverse returns the prediction to physical coordinates.</p>
<div class="card"><img src="overview.png" alt="Canonical model evaluation"></div>
<div class="card"><h2>Hard checks</h2><ul>{rows}</ul></div>
<h2>What the numbers mean</h2>
<p>Held-out recursive rollout RMSE: <strong>{held_out['normalized_rollout_rmse']:.4f}</strong>.
Observed Koopman phase residual:
<strong>{structure['held_out_mean_koopman_eigenfunction_residual']:.4f}</strong>.
Radial chart residual:
<strong>{radial_residual:.4f}</strong>.
Phase-step residual:
<strong>{phase_step_residual:.4f}</strong>.
Complete conjugacy residual:
<strong>{conjugacy_residual:.4f}</strong>.
Numerical symplectic defect: <strong>{structure['maximum_symplectic_defect']:.2e}</strong>.
Empirical-action affine R²: <strong>{action['affine_r2']:.4f}</strong>;
slope: <strong>{action['affine_slope']:.4f}</strong>.
Learned-h frequency error:
<strong>{learned_hamiltonian['frequency_normalized_rmse']:.3%}</strong>.</p>
<h2>Why this is Hamilton–Jacobi structure</h2>
<p>The canonical map is a learned local generating transformation. In the latent
chart, H becomes h(I), the angle advances at ω(I)=dh/dI, and
exp(ikφ) is a fiberwise Koopman eigenfunction. Symplecticity fixes the
phase-space area scale; radial and phase-law residuals test whether the learned
orbit is actually a circle with uniform angle advance at that scale.</p>
<h2>Boundary</h2>
<p>This certificate covers complete held-out trajectories from autonomous,
conservative, periodic one-degree-of-freedom data inside the observed action
range. A single state's action-range check is weaker than these orbit residuals.
It does not yet cover separatrix charts,
multi-frequency tori, forcing, dissipation, impacts, or HJB control.</p>
<p>Artifacts: <code>model.pt</code>, <code>manifest.json</code>, and
<code>action-audit/manifest.json</code>.</p></main></body></html>
""",
        encoding="utf-8",
    )


def run_canonical_experiment(
    dataset: TrajectoryDataset,
    output_dir: Path,
    *,
    config: CanonicalExperimentConfig,
) -> dict[str, Any]:
    if dataset.state_dim != 2:
        raise ValueError("canonical experiment requires exactly two state columns (q, p)")
    if config.epochs < 1 or config.rollout_horizon < 1:
        raise ValueError("epochs and rollout horizon must be positive")
    _set_seed(config.seed)
    source_revision = _git_source_state()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_indices, test_indices = _split_indices(
        dataset.trajectory_count,
        config.train_fraction,
        config.seed,
    )
    training_flat = dataset.states[train_indices].reshape(-1, 2)
    initial_center = tuple(float(value) for value in training_flat.mean(axis=0))
    network = CanonicalKoopmanNetwork(
        dt=dataset.dt,
        hidden_dim=config.hidden_dim,
        shear_layers=config.shear_layers,
        hamiltonian_degree=config.hamiltonian_degree,
        initial_center=(initial_center[0], initial_center[1]),
    )
    history = _fit_network(network, dataset, train_indices, config)
    network.eval()
    state_scale = training_flat.std(axis=0)
    train_evaluation, _ = _rollout_metrics(
        network,
        dataset.states,
        train_indices,
        state_scale,
    )
    held_out_evaluation, held_out_prediction = _rollout_metrics(
        network,
        dataset.states,
        test_indices,
        state_scale,
    )
    structure = _structure_metrics(network, dataset.states, test_indices)
    with torch.no_grad():
        training_action = network.action(
            torch.tensor(dataset.states[train_indices], dtype=torch.float32)
        ).mean(dim=1)
    action_padding = max(float(torch.std(training_action)) * 0.15, 1e-6)
    provisional = CanonicalKoopmanModel(
        network=network,
        state_columns=(dataset.state_columns[0], dataset.state_columns[1]),
        action_min=float(torch.min(training_action)) - action_padding,
        action_max=float(torch.max(training_action)) + action_padding,
        certificate_status="not_supported_by_current_dataset",
    )
    action_audit = run_hj_action_audit(
        dataset,
        output_dir / "action-audit",
        model=provisional,
    )
    all_action_alignment = action_audit["learned_coordinate_alignment"]
    held_out_ids = {
        dataset.trajectory_ids[index] for index in test_indices
    }
    action_alignment = _held_out_action_alignment(action_audit, held_out_ids)
    learned_hamiltonian = _learned_hamiltonian_metrics(
        network,
        dataset,
        action_audit,
        held_out_ids,
    )
    comparisons = {
        "held_out_rollout_is_accurate": (
            held_out_evaluation["normalized_rollout_rmse"] < 0.25
        ),
        "beats_persistence_rollout": (
            held_out_evaluation["normalized_rollout_rmse"]
            < held_out_evaluation["persistence_normalized_rollout_rmse"]
        ),
        "observed_orbits_have_stable_action": (
            structure["held_out_mean_normalized_action_drift"] < 0.08
        ),
        "koopman_phase_law_fits_observed_steps": (
            structure["held_out_mean_koopman_eigenfunction_residual"] < 0.08
        ),
        "map_is_numerically_invertible": (
            structure["maximum_inverse_error"] < 1e-5
        ),
        "map_is_numerically_symplectic": (
            structure["maximum_symplectic_defect"] < 2e-5
        ),
        "latent_action_is_exactly_conserved_by_model": (
            structure["maximum_model_rollout_action_drift"] < 2e-4
        ),
        "symplectic_gauge_matches_empirical_action": (
            action_alignment["affine_r2"] > 0.97
            and abs(abs(action_alignment["affine_slope"]) - 1.0) < 0.2
        ),
        "learned_hamiltonian_matches_frequency": (
            learned_hamiltonian["frequency_normalized_rmse"] < 0.05
        ),
    }
    comparisons.update(_chart_diagnostic_comparisons(structure))
    if learned_hamiltonian["reference_energy_available"]:
        comparisons["learned_hamiltonian_matches_reference_shape"] = (
            learned_hamiltonian["energy_normalized_rmse"] < 0.05
        )
    status = (
        "supported_on_held_out_trajectories"
        if all(comparisons.values())
        else "not_supported_by_current_dataset"
    )
    model = CanonicalKoopmanModel(
        network=network,
        state_columns=provisional.state_columns,
        action_min=provisional.action_min,
        action_max=provisional.action_max,
        certificate_status=status,
    )
    save_canonical_model(output_dir / "model.pt", model)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "package_version": __version__,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "source_revision": source_revision,
        "config": asdict(config),
        "scientific_contract": {
            "system_class": "autonomous conservative one-degree-of-freedom mechanics",
            "coordinate_contract": "ordered canonical state columns (q, p)",
            "architecture": (
                "exact symplectic conjugacy to the Hamiltonian rotation generated "
                "by h(I), with I=(Q^2+P^2)/2 and omega=dh/dI"
            ),
            "training_excludes": [
                "reference Hamiltonian column",
                "empirical closed-orbit action",
                "held-out trajectories",
            ],
        },
        "dataset": {
            "source": dataset.source,
            "source_sha256": dataset.source_sha256,
            "trajectory_count": dataset.trajectory_count,
            "step_count": dataset.step_count,
            "dt": dataset.dt,
            "canonical_columns": list(dataset.state_columns),
            "reference_column_used_for_training": False,
        },
        "split": {
            "training_trajectory_ids": [
                dataset.trajectory_ids[index] for index in train_indices
            ],
            "held_out_trajectory_ids": [
                dataset.trajectory_ids[index] for index in test_indices
            ],
        },
        "model": {
            "canonical_map": (
                f"{config.shear_layers} alternating neural symplectic shears"
            ),
            "hamiltonian_degree": config.hamiltonian_degree,
            "frequency_coefficients": (
                network.hamiltonian.frequency_coefficients().detach().tolist()
            ),
            "action_support": [model.action_min, model.action_max],
        },
        "training_history": history,
        "training_evaluation": train_evaluation,
        "held_out_evaluation": held_out_evaluation,
        "structure_evaluation": structure,
        "canonical_action_evaluation": action_alignment,
        "all_trajectory_action_diagnostic": all_action_alignment,
        "hamilton_jacobi_evaluation": action_audit["hj_identity"],
        "learned_hamiltonian_evaluation": learned_hamiltonian,
        "certificate": {
            "status": status,
            "decisive_comparisons": comparisons,
            "scope": (
                "complete held-out trajectories inside the observed one-degree-of-"
                "freedom periodic action range"
            ),
        },
        "artifacts": {
            "model": "model.pt",
            "overview": "overview.png",
            "report": "report.html",
            "manifest": "manifest.json",
            "action_audit_manifest": "action-audit/manifest.json",
        },
    }
    _plot_overview(
        output_dir / "overview.png",
        network=network,
        dataset=dataset,
        test_indices=test_indices,
        prediction=held_out_prediction,
        history=history,
        action_audit=action_audit,
    )
    _write_report(output_dir / "report.html", manifest)
    manifest["artifacts"].update(
        {
            "model_sha256": _sha256(output_dir / "model.pt"),
            "overview_sha256": _sha256(output_dir / "overview.png"),
            "report_sha256": _sha256(output_dir / "report.html"),
            "action_audit_manifest_sha256": _sha256(
                output_dir / "action-audit" / "manifest.json"
            ),
        }
    )
    manifest["validation_checks"] = validate_canonical_manifest(manifest)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest
