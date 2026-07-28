from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch

from learned_koopman.models.invariant import LearnedInvariant
from learned_koopman.physics import angle_from_state, simulate


@dataclass(frozen=True)
class InvariantExperimentConfig:
    """Controls for the label-free invariant-discovery experiment."""

    profile: str
    dt: float
    train_trajectories: int
    train_steps: int
    evaluation_trajectories: int
    evaluation_steps: int
    hidden_dim: int
    epochs: int
    learning_rate: float
    graph_neighbors: int
    constancy_weight: float
    graph_weight: float
    variance_weight: float
    centering_weight: float

    @classmethod
    def quick(cls) -> InvariantExperimentConfig:
        return cls(
            profile="quick",
            dt=0.025,
            train_trajectories=18,
            train_steps=180,
            evaluation_trajectories=15,
            evaluation_steps=280,
            hidden_dim=32,
            epochs=180,
            learning_rate=3e-3,
            graph_neighbors=2,
            constancy_weight=8.0,
            graph_weight=0.15,
            variance_weight=1.0,
            centering_weight=0.02,
        )

    @classmethod
    def full(cls) -> InvariantExperimentConfig:
        return cls(
            profile="full",
            dt=0.02,
            train_trajectories=32,
            train_steps=480,
            evaluation_trajectories=27,
            evaluation_steps=900,
            hidden_dim=48,
            epochs=700,
            learning_rate=2e-3,
            graph_neighbors=2,
            constancy_weight=10.0,
            graph_weight=0.12,
            variance_weight=1.0,
            centering_weight=0.02,
        )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _amplitude_centers(count: int, *, offset: float) -> np.ndarray:
    """Return interleaved orbit shells without calculating their energy."""

    edges = np.linspace(0.18, 3.05, count + 1, dtype=np.float64)
    width = edges[1] - edges[0]
    return 0.5 * (edges[:-1] + edges[1:]) + offset * width


def _trajectory_tensor(
    count: int,
    steps: int,
    dt: float,
    *,
    offset: float = 0.0,
) -> torch.Tensor:
    amplitudes = _amplitude_centers(count, offset=offset)
    # Stagger the start of every observed segment. This prevents the model from
    # seeing only the p=0 turning-point section while keeping phase/frequency
    # targets entirely outside the experiment.
    warmup_steps = max(steps * 3, 640)
    warmup, _, _ = simulate(
        amplitudes,
        np.zeros_like(amplitudes),
        steps=warmup_steps,
        dt=dt,
    )
    start_indices = (31 + 67 * np.arange(count)) % (warmup_steps - 1)
    initial = warmup[np.arange(count), start_indices]
    states, _, _ = simulate(
        angle_from_state(initial),
        initial[:, 2],
        steps=steps,
        dt=dt,
    )
    return torch.tensor(states, dtype=torch.float32)


def _orbit_neighbor_indices(
    trajectories: torch.Tensor,
    graph_neighbors: int,
) -> torch.Tensor:
    """Infer orbit adjacency from symmetric trajectory-set distance."""

    sample_count = min(32, trajectories.shape[1])
    sample_indices = torch.linspace(
        0,
        trajectories.shape[1] - 1,
        sample_count,
    ).round().long()
    representatives = trajectories[:, sample_indices]
    count = len(trajectories)
    distances = torch.full((count, count), torch.inf)
    for left in range(count):
        for right in range(left + 1, count):
            point_distances = torch.cdist(
                representatives[left],
                representatives[right],
            )
            symmetric_chamfer = 0.5 * (
                point_distances.min(dim=0).values.mean()
                + point_distances.min(dim=1).values.mean()
            )
            distances[left, right] = symmetric_chamfer
            distances[right, left] = symmetric_chamfer
    return distances.topk(
        k=min(graph_neighbors, count - 1),
        largest=False,
    ).indices


def invariant_discovery_loss(
    model: LearnedInvariant,
    trajectories: torch.Tensor,
    *,
    graph_neighbors: int,
    constancy_weight: float,
    graph_weight: float,
    variance_weight: float,
    centering_weight: float,
    neighbor_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build the objective from states and orbit membership only.

    Exact energy, amplitude values, class labels, and their ordering are absent.
    A k-nearest-neighbor graph is inferred from the observed section states.
    The non-collapse term fixes the otherwise arbitrary scale of the quotient
    coordinate.
    """

    values = model(trajectories)
    orbit_means = values.mean(dim=1)
    constancy = (values - orbit_means[:, None]).square().mean()

    with torch.no_grad():
        neighbors = (
            neighbor_indices
            if neighbor_indices is not None
            else _orbit_neighbor_indices(trajectories, graph_neighbors)
        )
    neighbor_values = orbit_means[neighbors]
    graph_smoothness = (orbit_means[:, None] - neighbor_values).square().mean()

    orbit_center = orbit_means.mean()
    orbit_std = orbit_means.std(unbiased=False)
    variance = (orbit_std - 1.0).square()
    centering = orbit_center.square()
    loss = (
        constancy_weight * constancy
        + graph_weight * graph_smoothness
        + variance_weight * variance
        + centering_weight * centering
    )
    return loss, {
        "constancy": constancy,
        "graph_smoothness": graph_smoothness,
        "orbit_std": orbit_std,
        "centering": centering,
    }


def train_invariant_model(
    trajectories: torch.Tensor,
    *,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    graph_neighbors: int = 2,
    constancy_weight: float = 8.0,
    graph_weight: float = 0.15,
    variance_weight: float = 1.0,
    centering_weight: float = 0.02,
) -> tuple[LearnedInvariant, list[dict[str, float]]]:
    """Train from a tensor of grouped state trajectories, with no labels."""

    _set_seed(seed)
    model = LearnedInvariant(hidden_dim)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )
    with torch.no_grad():
        neighbor_indices = _orbit_neighbor_indices(trajectories, graph_neighbors)
    history: list[dict[str, float]] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss, terms = invariant_discovery_loss(
            model,
            trajectories,
            graph_neighbors=graph_neighbors,
            constancy_weight=constancy_weight,
            graph_weight=graph_weight,
            variance_weight=variance_weight,
            centering_weight=centering_weight,
            neighbor_indices=neighbor_indices,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        history.append(
            {
                "loss": float(loss.detach()),
                **{key: float(value.detach()) for key, value in terms.items()},
            }
        )
    model.eval()
    return model, history


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _evaluate(
    model: LearnedInvariant,
    trajectories: torch.Tensor,
) -> dict[str, float]:
    from learned_koopman.physics import pendulum_energy_from_state

    with torch.no_grad():
        predictions = model(trajectories).cpu().numpy().astype(np.float64)
    states = trajectories.cpu().numpy().astype(np.float64)

    # Physics enters for the first time here, after optimization is complete.
    energies = pendulum_energy_from_state(states[:, 0])
    orbit_means = predictions.mean(axis=1)
    orbit_std = predictions.std(axis=1)

    design = np.column_stack((orbit_means, np.ones_like(orbit_means)))
    slope, intercept = np.linalg.lstsq(design, energies, rcond=None)[0]
    aligned = slope * orbit_means + intercept
    residual = np.square(aligned - energies).sum()
    total = np.square(energies - energies.mean()).sum()
    aligned_r2 = 1.0 - residual / max(total, 1e-12)

    spearman = float(np.corrcoef(_rank(orbit_means), _rank(energies))[0, 1])
    signal_scale = max(float(orbit_means.std()), 1e-8)
    normalized_drift = orbit_std / signal_scale

    energy_order = np.argsort(energies)
    oriented = orbit_means * (1.0 if slope >= 0.0 else -1.0)
    ordered_gaps = np.diff(oriented[energy_order])
    pooled_noise = max(float(np.sqrt(np.mean(np.square(orbit_std)))), 1e-8)
    time = np.linspace(-0.5, 0.5, predictions.shape[1], dtype=np.float64)
    normalized_time_trends = []
    for values in predictions:
        slope = float(np.dot(values - values.mean(), time) / np.dot(time, time))
        normalized_time_trends.append(abs(slope) / signal_scale)
    return {
        "affine_aligned_energy_r2": float(aligned_r2),
        "absolute_spearman_rank": abs(spearman),
        "mean_normalized_trajectory_drift": float(normalized_drift.mean()),
        "max_normalized_trajectory_drift": float(normalized_drift.max()),
        "median_cross_shell_signal_to_drift": float(
            np.median(np.maximum(ordered_gaps, 0.0)) / pooled_noise
        ),
        "monotonic_shell_fraction": float(np.mean(ordered_gaps > 0.0)),
        "max_normalized_time_trend": max(normalized_time_trends),
        "quotient_coordinate_std": signal_scale,
        "evaluation_shells": float(len(energies)),
    }


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = runs[0]["metrics"].keys()
    return {
        key: {
            "mean": mean(float(run["metrics"][key]) for run in runs),
            "std": pstdev(float(run["metrics"][key]) for run in runs),
            "min": min(float(run["metrics"][key]) for run in runs),
            "max": max(float(run["metrics"][key]) for run in runs),
        }
        for key in keys
    }


def run_invariant_experiment(
    *,
    profile: str = "quick",
    seeds: tuple[int, ...] | None = None,
    config: InvariantExperimentConfig | None = None,
) -> dict[str, Any]:
    """Run a reproducible label-free discovery benchmark.

    ``quick`` defaults to two independent seeds; ``full`` defaults to five.
    Passing a config is useful for focused tests and ablations.
    """

    selected = config or (
        InvariantExperimentConfig.quick()
        if profile == "quick"
        else InvariantExperimentConfig.full()
    )
    if selected.profile not in {"quick", "full"}:
        raise ValueError(f"unknown profile: {selected.profile}")
    selected_seeds = seeds or ((7, 17) if selected.profile == "quick" else (7, 17, 29, 41, 53))

    training_trajectories = _trajectory_tensor(
        selected.train_trajectories,
        selected.train_steps,
        selected.dt,
    )
    # Offset centers put evaluation between training shells.
    evaluation_trajectories = _trajectory_tensor(
        selected.evaluation_trajectories,
        selected.evaluation_steps,
        selected.dt,
        offset=0.19,
    )

    runs: list[dict[str, Any]] = []
    for seed in selected_seeds:
        model, history = train_invariant_model(
            training_trajectories,
            hidden_dim=selected.hidden_dim,
            epochs=selected.epochs,
            learning_rate=selected.learning_rate,
            seed=seed,
            graph_neighbors=selected.graph_neighbors,
            constancy_weight=selected.constancy_weight,
            graph_weight=selected.graph_weight,
            variance_weight=selected.variance_weight,
            centering_weight=selected.centering_weight,
        )
        runs.append(
            {
                "seed": seed,
                "metrics": _evaluate(model, evaluation_trajectories),
                "training": {
                    "initial_loss": history[0]["loss"],
                    "final_loss": history[-1]["loss"],
                    "final_constancy": history[-1]["constancy"],
                    "final_orbit_std": history[-1]["orbit_std"],
                },
            }
        )

    return {
        "experiment": "label_free_invariant_discovery",
        "profile": selected.profile,
        "scientific_contract": {
            "training_inputs": ["circular state trajectories", "trajectory membership"],
            "training_excludes": [
                "physical energy",
                "amplitude labels",
                "shell ordering",
                "frequency targets",
            ],
            "evaluation_oracle": "physical energy is used only after training",
            "phase_coverage": "trajectory segments begin at staggered, nonzero-velocity states",
            "held_out_structure": "complete evaluation shells are absent from training",
            "identifiability": "the scalar coordinate is defined up to orientation and scale",
        },
        "config": asdict(selected),
        "seeds": list(selected_seeds),
        "runs": runs,
        "aggregate": _aggregate(runs),
    }
