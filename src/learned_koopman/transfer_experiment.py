from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from learned_koopman.models.transfer import SimplexTransferOperator
from learned_koopman.physics import state_from_angle

Array = NDArray[np.float64]


@dataclass(frozen=True)
class TransferExperimentConfig:
    """Configuration for the stochastic-pendulum transfer experiment."""

    seed: int = 7
    dt: float = 0.04
    lag_steps: int = 5
    damping: float = 0.15
    noise_strength: float = 0.55
    trajectories: int = 36
    trajectory_steps: int = 260
    training_fraction: float = 0.75
    n_states: int = 6
    hidden_dim: int = 32
    batch_size: int = 512
    epochs: int = 80
    learning_rate: float = 3e-3
    count_pseudocount: float = 0.25
    branching_replicas: int = 192
    branching_lags: int = 5

    @classmethod
    def quick(cls, seed: int = 7) -> TransferExperimentConfig:
        return cls(
            seed=seed,
            trajectories=20,
            trajectory_steps=180,
            hidden_dim=24,
            batch_size=512,
            epochs=36,
            branching_replicas=128,
        )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _simulate_stochastic_pendulum(config: TransferExperimentConfig) -> Array:
    """Euler-Maruyama trajectories for a damped, thermally forced pendulum."""

    rng = np.random.default_rng(config.seed)
    theta = rng.uniform(-np.pi, np.pi, size=config.trajectories)
    omega = rng.normal(0.0, 1.15, size=config.trajectories)
    states = np.empty((config.trajectories, config.trajectory_steps, 3), dtype=np.float64)
    states[:, 0] = state_from_angle(theta, omega)
    diffusion = config.noise_strength * math.sqrt(config.dt)
    for step in range(1, config.trajectory_steps):
        omega += (
            -np.sin(theta) - config.damping * omega
        ) * config.dt + diffusion * rng.normal(size=config.trajectories)
        theta += omega * config.dt
        states[:, step] = state_from_angle(theta, omega)
    return states


def _branch_from_states(
    states: Array,
    *,
    steps: int,
    replicas: int,
    config: TransferExperimentConfig,
) -> Array:
    """Launch independent physical-noise paths from each exact anchor state."""

    rng = np.random.default_rng(config.seed + 10_007)
    theta = np.repeat(np.arctan2(states[:, 0], states[:, 1]), replicas)
    omega = np.repeat(states[:, 2], replicas)
    diffusion = config.noise_strength * math.sqrt(config.dt)
    for _ in range(steps):
        omega += (
            -np.sin(theta) - config.damping * omega
        ) * config.dt + diffusion * rng.normal(size=len(theta))
        theta += omega * config.dt
    return state_from_angle(theta, omega).reshape(len(states), replicas, 3)


def _lagged_triplets(states: Array, lag: int) -> tuple[Array, Array, Array]:
    if states.shape[1] <= 2 * lag:
        raise ValueError("trajectory is too short for the requested lag")
    return (
        states[:, : -2 * lag].reshape(-1, 3),
        states[:, lag:-lag].reshape(-1, 3),
        states[:, 2 * lag :].reshape(-1, 3),
    )


def _squared_distances(samples: Array, centroids: Array) -> Array:
    return ((samples[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)


def _fit_kmeans(samples: Array, n_states: int, seed: int) -> Array:
    """Small deterministic k-means implementation with k-means++ seeding."""

    rng = np.random.default_rng(seed)
    centroids = [samples[int(rng.integers(len(samples)))]]
    closest = ((samples - centroids[0]) ** 2).sum(axis=-1)
    for _ in range(1, n_states):
        probabilities = closest / closest.sum()
        centroids.append(samples[int(rng.choice(len(samples), p=probabilities))])
        candidate_distance = ((samples - centroids[-1]) ** 2).sum(axis=-1)
        closest = np.minimum(closest, candidate_distance)
    centers = np.stack(centroids)

    for _ in range(40):
        labels = _squared_distances(samples, centers).argmin(axis=1)
        updated = centers.copy()
        distance = _squared_distances(samples, centers).min(axis=1)
        for index in range(n_states):
            selected = samples[labels == index]
            if len(selected):
                updated[index] = selected.mean(axis=0)
            else:
                updated[index] = samples[int(distance.argmax())]
        if np.allclose(updated, centers, atol=1e-7):
            break
        centers = updated
    circle_norm = np.linalg.norm(centers[:, :2], axis=1, keepdims=True).clip(min=1e-8)
    centers[:, :2] /= circle_norm
    return centers


def _assign(samples: Array, centroids: Array) -> NDArray[np.int64]:
    return _squared_distances(samples, centroids).argmin(axis=1).astype(np.int64)


def _count_transition(
    current: NDArray[np.int64],
    future: NDArray[np.int64],
    n_states: int,
    pseudocount: float,
) -> Array:
    counts = np.full((n_states, n_states), pseudocount, dtype=np.float64)
    np.add.at(counts, (current, future), 1.0)
    return counts / counts.sum(axis=1, keepdims=True)


def _categorical_nll(probability: Array, labels: NDArray[np.int64]) -> float:
    selected = probability[np.arange(len(labels)), labels]
    return float(-np.log(selected.clip(min=1e-12)).mean())


def _circular_mse(prediction: Array, target: Array) -> float:
    circle = ((prediction[:, :2] - target[:, :2]) ** 2).sum(axis=-1)
    velocity = 0.25 * (prediction[:, 2] - target[:, 2]) ** 2
    return float((circle + velocity).mean())


def _normalize_states(states: Array) -> Array:
    normalized = states.copy()
    norm = np.linalg.norm(normalized[:, :2], axis=1, keepdims=True).clip(min=1e-8)
    normalized[:, :2] /= norm
    return normalized


def _stationary_distribution(transition: Array) -> Array:
    stationary = np.full(transition.shape[0], 1.0 / transition.shape[0])
    for _ in range(10_000):
        updated = stationary @ transition
        if np.max(np.abs(updated - stationary)) < 1e-13:
            break
        stationary = updated
    return updated / updated.sum()


def _weighted_matrix_rmse(first: Array, second: Array, occupancy: Array) -> float:
    weights = occupancy / occupancy.sum()
    return float(np.sqrt((weights[:, None] * (first - second) ** 2).sum()))


def _entropy(probability: Array, axis: int = -1) -> Array:
    return -(probability * np.log(probability.clip(min=1e-12))).sum(axis=axis)


def _branching_evidence(
    config: TransferExperimentConfig,
    model: SimplexTransferOperator,
    centroids: Array,
    count_transition: Array,
    occupancy: Array,
) -> dict[str, Any]:
    endpoints = _branch_from_states(
        centroids,
        steps=config.lag_steps * config.branching_lags,
        replicas=config.branching_replicas,
        config=config,
    )
    endpoint_labels = _assign(endpoints.reshape(-1, 3), centroids).reshape(
        config.n_states,
        config.branching_replicas,
    )
    counts = np.zeros((config.n_states, config.n_states), dtype=np.float64)
    for anchor in range(config.n_states):
        counts[anchor] = np.bincount(
            endpoint_labels[anchor],
            minlength=config.n_states,
        )
    empirical = (counts + config.count_pseudocount) / (
        counts.sum(axis=1, keepdims=True)
        + config.count_pseudocount * config.n_states
    )
    with torch.no_grad():
        anchor_tensor = torch.tensor(centroids, dtype=torch.float32)
        model_probability = model(anchor_tensor, steps=config.branching_lags).numpy()
        no_operator_probability = model.membership(anchor_tensor).numpy()
    occupancy_probability = np.broadcast_to(occupancy, empirical.shape)
    empirical_ulam_probability = np.linalg.matrix_power(
        count_transition,
        config.branching_lags,
    )

    velocity_variance = endpoints[:, :, 2].var(axis=1)
    circular_resultant = np.linalg.norm(endpoints[:, :, :2].mean(axis=1), axis=1)
    observed_destinations = (counts > 0).sum(axis=1)
    return {
        "source": "independent process-noise rollouts from identical anchor states",
        "replicas_per_anchor": config.branching_replicas,
        "anchor_count": config.n_states,
        "horizon_lags": config.branching_lags,
        "horizon_time": config.dt * config.lag_steps * config.branching_lags,
        "empirical_transition_matrix": empirical.tolist(),
        "mean_observed_destinations": float(observed_destinations.mean()),
        "minimum_observed_destinations": int(observed_destinations.min()),
        "mean_transition_entropy": float(_entropy(empirical).mean()),
        "mean_endpoint_velocity_variance": float(velocity_variance.mean()),
        "mean_endpoint_circular_dispersion": float((1.0 - circular_resultant).mean()),
        "model_cross_entropy": float(
            -(empirical * np.log(model_probability.clip(min=1e-12))).sum(axis=1).mean()
        ),
        "no_operator_cross_entropy": float(
            -(empirical * np.log(no_operator_probability.clip(min=1e-12)))
            .sum(axis=1)
            .mean()
        ),
        "empirical_ulam_cross_entropy": float(
            -(empirical * np.log(empirical_ulam_probability.clip(min=1e-12)))
            .sum(axis=1)
            .mean()
        ),
        "occupancy_baseline_cross_entropy": float(
            -(empirical * np.log(occupancy_probability.clip(min=1e-12))).sum(axis=1).mean()
        ),
    }


def _train_model(
    config: TransferExperimentConfig,
    centroids: Array,
    initial_transition: Array,
    train_triplets: tuple[Array, Array, Array],
    train_labels: tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]],
) -> tuple[SimplexTransferOperator, list[float]]:
    model = SimplexTransferOperator(
        state_dim=3,
        n_states=config.n_states,
        hidden_dim=config.hidden_dim,
    )
    model.initialize_prototypes(torch.tensor(centroids, dtype=torch.float32))
    model.initialize_transition(torch.tensor(initial_transition, dtype=torch.float32))

    tensors = [
        torch.tensor(value, dtype=torch.float32)
        for value in train_triplets
    ] + [
        torch.tensor(value, dtype=torch.long)
        for value in train_labels
    ]
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(*tensors),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[float] = []
    model.train()
    for _ in range(config.epochs):
        total = 0.0
        for current, future, future_two, label0, label1, label2 in loader:
            optimizer.zero_grad()
            logits0 = model.membership_logits(current)
            logits1 = model.membership_logits(future)
            logits2 = model.membership_logits(future_two)
            membership0 = torch.softmax(logits0, dim=-1)
            predicted1 = model.propagate(membership0)
            predicted2 = model.propagate(membership0, steps=2)
            classification = (
                F.cross_entropy(logits0, label0)
                + F.cross_entropy(logits1, label1)
                + F.cross_entropy(logits2, label2)
            ) / 3.0
            transition_nll = F.nll_loss(predicted1.clamp_min(1e-8).log(), label1)
            ck_nll = F.nll_loss(predicted2.clamp_min(1e-8).log(), label2)
            loss = classification + transition_nll + 0.5 * ck_nll
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total += float(loss.detach())
        history.append(total / len(loader))
    model.eval()
    return model, history


def _evaluate(
    config: TransferExperimentConfig,
    model: SimplexTransferOperator,
    centroids: Array,
    train_labels: tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]],
    validation_triplets: tuple[Array, Array, Array],
    validation_labels: tuple[
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.int64],
    ],
    count_transition: Array,
) -> dict[str, Any]:
    current, future, _ = validation_triplets
    label0, label1, label2 = validation_labels
    with torch.no_grad():
        current_tensor = torch.tensor(current, dtype=torch.float32)
        membership = model.membership(current_tensor).numpy()
        predicted = model(current_tensor).numpy()
        predicted_two = model(current_tensor, steps=2).numpy()
        transition = model.transition_matrix().numpy()
        decoded = model.decode_membership(torch.tensor(predicted)).numpy()

    occupancy = np.bincount(train_labels[1], minlength=config.n_states).astype(np.float64)
    occupancy /= occupancy.sum()
    occupancy_prediction = np.broadcast_to(occupancy, predicted.shape)
    hard_prediction = count_transition[label0]
    hard_prediction_two = (count_transition @ count_transition)[label0]
    identity = np.eye(config.n_states) * 0.995 + (1.0 - np.eye(config.n_states)) * (
        0.005 / (config.n_states - 1)
    )

    direct_two_lag = _count_transition(
        train_labels[0],
        train_labels[2],
        config.n_states,
        config.count_pseudocount,
    )
    learned_ck = transition @ transition
    count_ck = count_transition @ count_transition
    direct_two_lag_prediction = direct_two_lag[label0]

    stationary = _stationary_distribution(transition)
    eigenvalues = np.linalg.eigvals(transition)
    eigenvalues = sorted(eigenvalues, key=lambda value: abs(value), reverse=True)
    second_eigenvalue = float(abs(eigenvalues[1]))
    lag_time = config.dt * config.lag_steps
    implied_timescale = (
        float(-lag_time / math.log(second_eigenvalue))
        if 1e-12 < second_eigenvalue < 1.0 - 1e-12
        else 0.0
    )

    design = np.concatenate(
        (validation_triplets[0], np.ones((len(current), 1))),
        axis=1,
    )
    train_design = np.concatenate(
        (validation_triplets[0], np.ones((len(current), 1))),
        axis=1,
    )
    # A held-out-only linear fit would leak.  This baseline is intentionally
    # omitted here and the finite-state prediction is compared to persistence.
    del design, train_design

    decoded = _normalize_states(decoded)
    simplex_row_error = np.max(np.abs(membership.sum(axis=1) - 1.0))
    transition_row_error = np.max(np.abs(transition.sum(axis=1) - 1.0))
    empirical_occupancy = np.bincount(label1, minlength=config.n_states).astype(np.float64)
    empirical_occupancy /= empirical_occupancy.sum()
    stationary_residual = np.abs(stationary @ transition - stationary).sum()
    membership_mean = membership.mean(axis=0)
    membership_entropy = _entropy(membership)
    membership_mean_entropy = float(_entropy(membership_mean))
    branching = _branching_evidence(
        config,
        model,
        centroids,
        count_transition,
        occupancy,
    )
    learned_one_nll = _categorical_nll(predicted, label1)
    no_operator_one_nll = _categorical_nll(membership, label1)
    learned_two_nll = _categorical_nll(predicted_two, label2)
    no_operator_two_nll = _categorical_nll(membership, label2)
    learned_ck_rmse = _weighted_matrix_rmse(
        learned_ck,
        direct_two_lag,
        occupancy,
    )
    ulam_ck_rmse = _weighted_matrix_rmse(
        count_ck,
        direct_two_lag,
        occupancy,
    )
    decisive_comparisons = {
        "one_lag_beats_no_operator": learned_one_nll < no_operator_one_nll,
        "one_lag_beats_empirical_ulam": (
            learned_one_nll < _categorical_nll(hard_prediction, label1)
        ),
        "two_lag_beats_no_operator": learned_two_nll < no_operator_two_nll,
        "two_lag_beats_direct_ulam": (
            learned_two_nll
            < _categorical_nll(direct_two_lag_prediction, label2)
        ),
        "branching_beats_no_operator": (
            branching["model_cross_entropy"]
            < branching["no_operator_cross_entropy"]
        ),
        "branching_beats_empirical_ulam": (
            branching["model_cross_entropy"]
            < branching["empirical_ulam_cross_entropy"]
        ),
        "branching_beats_occupancy": (
            branching["model_cross_entropy"]
            < branching["occupancy_baseline_cross_entropy"]
        ),
        "ck_beats_empirical_ulam": learned_ck_rmse < ulam_ck_rmse,
    }
    constraints_pass = bool(
        simplex_row_error < 1e-5
        and transition_row_error < 1e-5
        and membership.min() >= 0.0
        and transition.min() > 0.0
    )
    failed_comparisons = [
        name for name, passed in decisive_comparisons.items() if not passed
    ]
    operator_status = (
        "not_falsified_on_this_profile"
        if constraints_pass and not failed_comparisons
        else "falsified_by_current_profile"
    )

    return {
        "constraints": {
            "membership_max_sum_error": float(simplex_row_error),
            "membership_min_probability": float(membership.min()),
            "transition_max_row_sum_error": float(transition_row_error),
            "transition_min_probability": float(transition.min()),
        },
        "held_out": {
            "one_step_nll": learned_one_nll,
            "no_operator_one_step_nll": no_operator_one_nll,
            "empirical_ulam_nll": _categorical_nll(hard_prediction, label1),
            "occupancy_baseline_nll": _categorical_nll(occupancy_prediction, label1),
            "persistence_membership_nll": _categorical_nll(identity[label0], label1),
            "membership_classification_accuracy": float(
                (membership.argmax(axis=1) == label0).mean()
            ),
            "finite_state_circular_mse": _circular_mse(decoded, future),
            "physical_persistence_circular_mse": _circular_mse(current, future),
            "state_change_rate": float((label0 != label1).mean()),
            "samples": int(len(label1)),
        },
        "two_lag_held_out": {
            "learned_k_squared_nll": learned_two_nll,
            "no_operator_membership_nll": no_operator_two_nll,
            "empirical_ulam_squared_nll": _categorical_nll(
                hard_prediction_two,
                label2,
            ),
            "direct_two_lag_ulam_nll": _categorical_nll(
                direct_two_lag_prediction,
                label2,
            ),
            "occupancy_baseline_nll": _categorical_nll(
                occupancy_prediction,
                label2,
            ),
            "samples": int(len(label2)),
        },
        "representation": {
            "mean_membership": membership_mean.tolist(),
            "mean_membership_entropy": membership_mean_entropy,
            "effective_state_count": float(math.exp(membership_mean_entropy)),
            "active_states_above_one_percent": int((membership_mean > 0.01).sum()),
            "mean_sample_entropy": float(membership_entropy.mean()),
            "maximum_mean_membership": float(membership_mean.max()),
        },
        "chapman_kolmogorov": {
            "learned_two_lag_weighted_rmse": learned_ck_rmse,
            "empirical_ulam_two_lag_weighted_rmse": ulam_ck_rmse,
            "direct_two_lag_matrix": direct_two_lag.tolist(),
        },
        "stationary": {
            "distribution": stationary.tolist(),
            "sum_error": float(abs(stationary.sum() - 1.0)),
            "minimum_probability": float(stationary.min()),
            "fixed_point_l1_residual": float(stationary_residual),
            "held_out_occupancy_l1_error": float(
                np.abs(stationary - empirical_occupancy).sum()
            ),
            "second_eigenvalue_magnitude": second_eigenvalue,
            "implied_timescale": implied_timescale,
            "spectral_gap": float(1.0 - second_eigenvalue),
            "mean_row_entropy": float(_entropy(transition).mean()),
        },
        "process_noise_evidence": branching,
        "operator_verdict": {
            "status": operator_status,
            "constraints_pass": constraints_pass,
            "decisive_comparisons": decisive_comparisons,
            "failed_comparisons": failed_comparisons,
            "interpretation": (
                "The stochastic constraints are mathematically valid, but the "
                "learned propagation is not useful on this profile when any "
                "no-operator, branching, or Chapman-Kolmogorov counterfactual fails."
                if operator_status == "falsified_by_current_profile"
                else (
                    "The learned propagation passed this profile's counterfactuals; "
                    "this is not evidence of robustness across seeds, lags, or systems."
                )
            ),
        },
        "baselines": {
            "empirical_ulam": (
                "train-only hard state counts with a pseudocount; no neural encoder"
            ),
            "occupancy": "the train-set state frequencies, independent of current state",
            "no_operator": (
                "the learned soft membership at the current state, propagated zero times"
            ),
            "physical_persistence": "the current physical state reused at the next lag",
        },
        "transition_matrix": transition.tolist(),
        "prototypes": centroids.tolist(),
    }


def run_transfer_experiment(
    *,
    config: TransferExperimentConfig | None = None,
    quick: bool = False,
    seed: int = 7,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Train and evaluate a categorical transfer operator on held-out paths."""

    resolved = config or (
        TransferExperimentConfig.quick(seed=seed)
        if quick
        else TransferExperimentConfig(seed=seed)
    )
    _set_seed(resolved.seed)
    trajectories = _simulate_stochastic_pendulum(resolved)
    train_count = int(resolved.trajectories * resolved.training_fraction)
    if train_count < 2 or train_count >= resolved.trajectories:
        raise ValueError("training_fraction must leave train and validation trajectories")
    train_triplets = _lagged_triplets(trajectories[:train_count], resolved.lag_steps)
    validation_triplets = _lagged_triplets(trajectories[train_count:], resolved.lag_steps)

    train_states = trajectories[:train_count].reshape(-1, 3)
    centroids = _fit_kmeans(train_states, resolved.n_states, resolved.seed)
    train_labels = tuple(_assign(values, centroids) for values in train_triplets)
    validation_labels = tuple(_assign(values, centroids) for values in validation_triplets)
    count_transition = _count_transition(
        train_labels[0],
        train_labels[1],
        resolved.n_states,
        resolved.count_pseudocount,
    )
    model, history = _train_model(
        resolved,
        centroids,
        count_transition,
        train_triplets,
        train_labels,
    )
    evaluation = _evaluate(
        resolved,
        model,
        centroids,
        train_labels,
        validation_triplets,
        validation_labels,
        count_transition,
    )
    result: dict[str, Any] = {
        "experiment": "simplex_transfer_operator",
        "config": asdict(resolved),
        "data": {
            "system": "damped stochastic pendulum",
            "integrator": "Euler-Maruyama",
            "noise_source": "Gaussian process noise injected into the physical velocity equation",
            "decoder": "deterministic prototype expectation",
            "split": "held out by complete trajectory",
            "train_trajectories": train_count,
            "validation_trajectories": resolved.trajectories - train_count,
            "lag_time": resolved.dt * resolved.lag_steps,
            "independent_branching_rollouts": (
                resolved.branching_replicas * resolved.n_states
            ),
        },
        "objective": {
            "latent_family": "categorical simplex",
            "operator": "positive row-stochastic matrix",
            "terms": [
                "coarse-state categorical classification",
                "one-lag categorical negative log likelihood",
                "two-lag categorical negative log likelihood",
            ],
            "claim": "finite-state transfer experiment, not a variational autoencoder",
        },
        "training": {
            "epochs": resolved.epochs,
            "initial_loss": history[0],
            "final_loss": history[-1],
        },
        **evaluation,
        "limitations": [
            "The coarse states are initialized and supervised by train-only k-means labels.",
            "The stochastic simulator is a controlled benchmark, not calibrated experimental data.",
            (
                "The finite-state decoder is diagnostic; it is not expected "
                "to beat local physical predictors."
            ),
            (
                "A valid simplex and stochastic matrix are structural checks only; "
                "the operator verdict is controlled by no-operator and Ulam baselines."
            ),
            "One seed and one lag do not establish a metastability or convergence result.",
        ],
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "transfer_metrics.json").write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        torch.save(model.state_dict(), output_dir / "transfer_model.pt")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_transfer_experiment(
        quick=arguments.quick,
        seed=arguments.seed,
        output_dir=arguments.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
