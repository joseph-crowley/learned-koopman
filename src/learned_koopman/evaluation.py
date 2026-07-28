from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch

from learned_koopman.config import ExperimentConfig
from learned_koopman.data import EvaluationTrajectory, evaluation_trajectories, training_dataset
from learned_koopman.models import (
    EnergyConditionedRotation,
    FixedKoopmanAE,
    SeparatrixAtlas,
)
from learned_koopman.models.baselines import small_angle_step
from learned_koopman.physics import circular_state_error, pendulum_energy_from_state
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


def _rollout_saddle_only(
    model: SeparatrixAtlas,
    initial: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    states = [initial]
    state = initial
    with torch.no_grad():
        for _ in range(steps):
            state = model.saddle_step(state)
            states.append(state)
    return torch.stack(states)


def _rollout_projected_conditioned(
    atlas: SeparatrixAtlas,
    initial: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    states = [initial]
    with torch.no_grad():
        condition = atlas.regular.normalized_energy(initial)
        phase = atlas.regular.encode_phase(initial)
        for _ in range(steps):
            phase = atlas.regular.rotate(phase, condition)
            prediction = atlas.regular.decode(phase, condition)
            states.append(atlas.project_to_energy_shell(prediction, condition))
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
    rollouts = {
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
    if models.atlas is not None:
        with torch.no_grad():
            atlas_states, _ = models.atlas.rollout(initial, steps)
        rollouts["energy_projected_conditioned"] = _rollout_projected_conditioned(
            models.atlas,
            initial,
            steps,
        ).numpy()
        rollouts["saddle_chart_only"] = _rollout_saddle_only(
            models.atlas,
            initial,
            steps,
        ).numpy()
        rollouts["separatrix_atlas"] = atlas_states.numpy()
    return rollouts


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


def _atlas_local_residuals(
    model: SeparatrixAtlas,
    trajectory: EvaluationTrajectory,
) -> tuple[float, float]:
    """Measure selected-chart transition error on a held-out reference trajectory."""

    state = torch.tensor(trajectory.states[:-1], dtype=torch.float32)
    target = torch.tensor(trajectory.states[1:], dtype=torch.float32)
    initial = state[0]
    condition = model.regular.normalized_energy(initial).expand(len(state), -1)
    with torch.no_grad():
        route = model.route_index(state, condition)
        regular = model.project_to_energy_shell(
            model.regular_step(state, condition),
            condition,
        )
        saddle = model.project_to_energy_shell(model.saddle_step(state), condition)
        prediction = torch.where(route.unsqueeze(-1).bool(), saddle, regular)
        residual = torch.sqrt(circular_state_error(prediction, target))
    return float(residual.mean()), float(residual.max())


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
        if models.atlas is not None:
            initial = torch.tensor(trajectory.states[0], dtype=torch.float32)
            with torch.no_grad():
                _, diagnostics = models.atlas.rollout(
                    initial,
                    len(trajectory.states) - 1,
                )
            route_index = diagnostics["route_index"].cpu().numpy()
            switch_disagreement = diagnostics["switch_disagreement"].cpu().numpy()
            route_switches = (
                int(np.count_nonzero(route_index[1:] != route_index[:-1]))
                if len(route_index) > 1
                else 0
            )
            mean_local_residual, max_local_residual = _atlas_local_residuals(
                models.atlas,
                trajectory,
            )
            metrics[key]["separatrix_atlas"].update(
                {
                    "saddle_fraction": float(np.mean(route_index == 1)),
                    "route_switches": route_switches,
                    "max_switch_disagreement": (
                        float(np.max(switch_disagreement)) if len(switch_disagreement) else 0.0
                    ),
                    "mean_local_chart_residual": mean_local_residual,
                    "max_local_chart_residual": max_local_residual,
                }
            )
    return metrics, all_rollouts
