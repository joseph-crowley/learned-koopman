from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch

from learned_koopman.config import ExperimentConfig
from learned_koopman.data import EvaluationTrajectory, evaluation_trajectories, training_dataset
from learned_koopman.models import EnergyConditionedRotation, FixedKoopmanAE
from learned_koopman.models.baselines import small_angle_step
from learned_koopman.physics import pendulum_energy_from_state
from learned_koopman.training import TrainedModels


def _rollout_step(
    initial: torch.Tensor,
    steps: int,
    step_function: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    states = [initial]
    state = initial
    with torch.no_grad():
        for _ in range(steps):
            state = step_function(state)
            states.append(state)
    return torch.stack(states)


def _rollout_fixed(model: FixedKoopmanAE, initial: torch.Tensor, steps: int) -> torch.Tensor:
    states = [initial]
    with torch.no_grad():
        latent = model.encode(initial)
        for _ in range(steps):
            latent = model.step_latent(latent)
            states.append(model.decode(latent))
    return torch.stack(states)


def _rollout_conditioned(
    model: EnergyConditionedRotation,
    initial: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    states = [initial]
    with torch.no_grad():
        condition = model.normalized_energy(initial)
        phase = model.encode_phase(initial)
        for _ in range(steps):
            phase = model.rotate(phase, condition)
            states.append(model.decode(phase, condition))
    return torch.stack(states)


def model_rollouts(
    models: TrainedModels,
    trajectory: EvaluationTrajectory,
    config: ExperimentConfig,
    dmd_operator: np.ndarray,
) -> dict[str, np.ndarray]:
    initial = torch.tensor(trajectory.states[0], dtype=torch.float32)
    steps = len(trajectory.states) - 1
    dmd_states = [trajectory.states[0]]
    for _ in range(steps):
        next_state = dmd_states[-1] @ dmd_operator
        next_state[:2] /= max(float(np.linalg.norm(next_state[:2])), 1e-12)
        dmd_states.append(next_state)
    return {
        "persistence": initial.unsqueeze(0).repeat(steps + 1, 1).numpy(),
        "dmd": np.stack(dmd_states),
        "small_angle": _rollout_step(
            initial,
            steps,
            lambda state: small_angle_step(state, config.dt),
        ).numpy(),
        "mlp": _rollout_step(initial, steps, models.mlp.step).numpy(),
        "fixed_koopman": _rollout_fixed(models.fixed, initial, steps).numpy(),
        "energy_conditioned": _rollout_conditioned(
            models.conditioned,
            initial,
            steps,
        ).numpy(),
    }


def _angular_frequency(states: np.ndarray, dt: float) -> float | None:
    theta = np.arctan2(states[:, 0], states[:, 1])
    omega = states[:, 2]
    crossings = np.flatnonzero((theta[:-1] > 0.0) & (theta[1:] <= 0.0) & (omega[1:] < 0.0))
    if len(crossings) < 2:
        return None
    period = float(np.diff(crossings).mean() * dt)
    return 2.0 * math.pi / period


def rollout_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    dt: float,
) -> dict[str, float | int | None]:
    prediction_theta = np.arctan2(prediction[:, 0], prediction[:, 1])
    reference_theta = np.arctan2(reference[:, 0], reference[:, 1])
    angle_difference = np.arctan2(
        np.sin(prediction_theta - reference_theta),
        np.cos(prediction_theta - reference_theta),
    )
    omega_difference = prediction[:, 2] - reference[:, 2]
    combined = np.sqrt(angle_difference**2 + 0.25 * omega_difference**2)
    failures = np.flatnonzero(combined > 0.15)
    valid_steps = int(failures[0]) if len(failures) else len(reference) - 1
    energy = pendulum_energy_from_state(prediction)
    return {
        "angle_rmse": float(np.sqrt(np.mean(angle_difference**2))),
        "omega_rmse": float(np.sqrt(np.mean(omega_difference**2))),
        "max_energy_drift": float(np.max(np.abs(energy - energy[0]))),
        "valid_steps": valid_steps,
        "valid_time": float(valid_steps * dt),
        "angular_frequency": _angular_frequency(prediction, dt),
    }


def evaluate(
    models: TrainedModels,
    config: ExperimentConfig,
) -> tuple[dict[str, dict[str, dict[str, float | int | None]]], dict[float, dict[str, np.ndarray]]]:
    metrics: dict[str, dict[str, dict[str, float | int | None]]] = {}
    all_rollouts: dict[float, dict[str, np.ndarray]] = {}
    sequences = training_dataset(config).tensors[0].numpy()
    dmd_input = sequences[:, :-1].reshape(-1, 3)
    dmd_target = sequences[:, 1:].reshape(-1, 3)
    dmd_operator, *_ = np.linalg.lstsq(dmd_input, dmd_target, rcond=None)
    for trajectory in evaluation_trajectories(config):
        key = f"{trajectory.amplitude:.2f}"
        rollouts = model_rollouts(models, trajectory, config, dmd_operator)
        rollouts["reference"] = trajectory.states
        all_rollouts[trajectory.amplitude] = rollouts
        metrics[key] = {
            name: rollout_metrics(prediction, trajectory.states, dt=config.dt)
            for name, prediction in rollouts.items()
            if name != "reference"
        }
        metrics[key]["reference"] = rollout_metrics(
            trajectory.states,
            trajectory.states,
            dt=config.dt,
        )
    return metrics, all_rollouts
