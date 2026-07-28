from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from learned_koopman.control import (
    ActionConditionedPendulum,
    ControlledTrajectories,
    ExactUnitGainOracle,
    GainOnlyControlledPendulum,
    baseline_rollout,
    generate_controlled_pulses,
    mechanical_work,
    simulate_controlled,
    small_angle_controlled_step,
    upward_crossing_steps,
)
from learned_koopman.physics import circular_state_error, torch_energy


@dataclass(frozen=True)
class ControlExperimentProfile:
    seed: int = 7
    dt: float = 0.04
    steps: int = 180
    train_trajectories: int = 72
    evaluation_trajectories: int = 12
    hidden_dim: int = 32
    batch_size: int = 512
    epochs: int = 90
    learning_rate: float = 4e-3
    rollout_training_horizon: int = 6
    crossing_window_radius: int = 18

    @classmethod
    def quick(cls, seed: int = 7) -> ControlExperimentProfile:
        return cls(
            seed=seed,
            steps=150,
            train_trajectories=48,
            evaluation_trajectories=8,
            hidden_dim=24,
            epochs=60,
            rollout_training_horizon=6,
            crossing_window_radius=12,
        )

    @classmethod
    def full(cls, seed: int = 7) -> ControlExperimentProfile:
        return cls(seed=seed)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _training_windows(
    data: ControlledTrajectories,
    horizon: int,
) -> TensorDataset:
    state_windows: list[np.ndarray] = []
    control_windows: list[np.ndarray] = []
    stride = max(horizon, 2)
    for start in range(0, data.step_count - horizon, stride):
        state_windows.append(data.states[:, start : start + horizon + 1])
        control_windows.append(data.controls[:, start : start + horizon])
    states = torch.tensor(np.concatenate(state_windows), dtype=torch.float32)
    controls = torch.tensor(np.concatenate(control_windows), dtype=torch.float32)
    return TensorDataset(states, controls)


def train_action_conditioned_model(
    profile: ControlExperimentProfile,
    training_data: ControlledTrajectories,
) -> tuple[ActionConditionedPendulum, list[float]]:
    """Fit on short windows; all reported long rollouts are free-running."""

    _set_seed(profile.seed)
    model = ActionConditionedPendulum(profile.dt, profile.hidden_dim)
    dataset = _training_windows(training_data, profile.rollout_training_horizon)
    generator = torch.Generator().manual_seed(profile.seed)
    loader = DataLoader(
        dataset,
        batch_size=profile.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=profile.learning_rate)
    history: list[float] = []
    model.train()
    for _ in range(profile.epochs):
        total = 0.0
        for states, controls in loader:
            optimizer.zero_grad()
            prediction = model.rollout(states[:, 0], controls)
            state_loss = circular_state_error(prediction[:, 1:], states[:, 1:]).mean()
            target_average_acceleration = (
                states[:, 1, 2] - states[:, 0, 2]
            ) / profile.dt
            acceleration_loss = (
                model.acceleration(states[:, 0], controls[:, 0])
                - target_average_acceleration
            ).square().mean()
            loss = state_loss + 0.1 * acceleration_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total += float(loss.detach())
        history.append(total / len(loader))
    model.eval()
    return model, history


def train_gain_only_model(
    profile: ControlExperimentProfile,
    training_data: ControlledTrajectories,
) -> tuple[GainOnlyControlledPendulum, list[float]]:
    """Identify actuator effectiveness from the same windows as the residual model."""

    _set_seed(profile.seed)
    model = GainOnlyControlledPendulum(profile.dt, initial_gain=0.35)
    dataset = _training_windows(training_data, profile.rollout_training_horizon)
    generator = torch.Generator().manual_seed(profile.seed)
    loader = DataLoader(
        dataset,
        batch_size=profile.batch_size,
        shuffle=True,
        generator=generator,
    )
    # A single scalar has a much smoother objective than the residual network;
    # the larger step keeps the quick profile a genuine identification run.
    optimizer = torch.optim.Adam(model.parameters(), lr=5.0 * profile.learning_rate)
    history: list[float] = []
    model.train()
    for _ in range(profile.epochs):
        total = 0.0
        for states, controls in loader:
            optimizer.zero_grad()
            prediction = model.rollout(states[:, 0], controls)
            state_loss = circular_state_error(prediction[:, 1:], states[:, 1:]).mean()

            # The kick-drift-kick discretization gives a directly auditable
            # actuator impulse target without differentiating noisy angles.
            observed_control_force = (
                (states[:, 1:, 2] - states[:, :-1, 2]) / profile.dt
                + 0.5 * (states[:, :-1, 0] + states[:, 1:, 0])
            )
            identified_control_force = model.control_gain * controls
            identification_loss = (
                identified_control_force - observed_control_force
            ).square().mean()
            loss = state_loss + identification_loss
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        history.append(total / len(loader))
    model.eval()
    return model, history


def _first_crossing_metrics(
    true_energies: np.ndarray,
    predicted_energies: np.ndarray,
) -> dict[str, float | int | None]:
    truth = upward_crossing_steps(true_energies)
    prediction = upward_crossing_steps(predicted_energies)
    truth_positive = truth >= 0
    prediction_positive = prediction >= 0
    true_positive = truth_positive & prediction_positive
    false_positive = ~truth_positive & prediction_positive
    false_negative = truth_positive & ~prediction_positive
    true_negative = ~truth_positive & ~prediction_positive

    accuracy = float(np.mean(truth_positive == prediction_positive))
    precision_denominator = int(np.sum(true_positive | false_positive))
    recall_denominator = int(np.sum(true_positive | false_negative))
    precision = (
        float(np.sum(true_positive) / precision_denominator)
        if precision_denominator
        else None
    )
    recall = (
        float(np.sum(true_positive) / recall_denominator)
        if recall_denominator
        else None
    )
    timing = (
        float(np.mean(np.abs(prediction[true_positive] - truth[true_positive])))
        if np.any(true_positive)
        else None
    )
    return {
        "event_accuracy": accuracy,
        "event_precision": precision,
        "event_recall": recall,
        "timing_mae_steps_on_detected_crossings": timing,
        "true_positive": int(np.sum(true_positive)),
        "false_positive": int(np.sum(false_positive)),
        "false_negative": int(np.sum(false_negative)),
        "true_negative": int(np.sum(true_negative)),
    }


def _crossing_window_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    crossing_steps: np.ndarray,
    radius: int,
) -> float | None:
    errors: list[torch.Tensor] = []
    for row, crossing in enumerate(crossing_steps):
        if crossing < 0:
            continue
        start = max(0, int(crossing) - radius)
        stop = min(target.shape[1], int(crossing) + radius + 1)
        errors.append(circular_state_error(prediction[row, start:stop], target[row, start:stop]))
    if not errors:
        return None
    return float(torch.cat(errors).mean())


def _one_step_prediction(
    model: GainOnlyControlledPendulum,
    data: ControlledTrajectories,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = torch.tensor(data.states[:, :-1], dtype=torch.float32).flatten(0, 1)
    control = torch.tensor(data.controls, dtype=torch.float32).flatten()
    with torch.no_grad():
        prediction = model.step(state, control)
    target = torch.tensor(data.states[:, 1:], dtype=torch.float32).flatten(0, 1)
    return prediction, target


def evaluate_control_model(
    gain_only_model: GainOnlyControlledPendulum,
    residual_model: ActionConditionedPendulum,
    data: ControlledTrajectories,
    profile: ControlExperimentProfile,
) -> dict[str, object]:
    initial = torch.tensor(data.states[:, 0], dtype=torch.float32)
    controls = torch.tensor(data.controls, dtype=torch.float32)
    target = torch.tensor(data.states, dtype=torch.float32)
    oracle = ExactUnitGainOracle(profile.dt)
    flat_state = torch.tensor(data.states[:, :-1], dtype=torch.float32).flatten(0, 1)
    flat_control = controls.flatten()
    one_target = torch.tensor(data.states[:, 1:], dtype=torch.float32).flatten(0, 1)
    with torch.no_grad():
        oracle_prediction = oracle.rollout(initial, controls)
        gain_only_prediction = gain_only_model.rollout(initial, controls)
        gain_only_blind_prediction = gain_only_model.rollout(
            initial,
            torch.zeros_like(controls),
        )
        residual_prediction = residual_model.rollout(initial, controls)
        baseline = baseline_rollout(initial, controls, dt=profile.dt)
        small_angle_blind = baseline_rollout(
            initial,
            controls,
            dt=profile.dt,
            ignore_control=True,
        )
        oracle_one, _ = _one_step_prediction(oracle, data)
        gain_only_one, _ = _one_step_prediction(gain_only_model, data)
        gain_only_blind_one = gain_only_model.step(
            flat_state,
            torch.zeros_like(flat_control),
        )
        residual_one, _ = _one_step_prediction(residual_model, data)
        baseline_one = small_angle_controlled_step(
            flat_state,
            flat_control,
            profile.dt,
        )
        small_angle_blind_one = small_angle_controlled_step(
            flat_state,
            torch.zeros_like(flat_control),
            profile.dt,
        )

    crossing_steps = upward_crossing_steps(data.energies)
    actual_crossings = crossing_steps >= 0

    autonomous_controls = np.zeros_like(data.controls)
    autonomous_states, _, _, autonomous_energies = simulate_controlled(
        data.angles[:, 0],
        data.angular_velocities[:, 0],
        autonomous_controls,
        dt=profile.dt,
    )
    del autonomous_states
    work = mechanical_work(data.controls, data.angular_velocities, profile.dt)
    energy_change = data.energies[:, -1] - data.energies[:, 0]

    methods = {
        "exact_unit_gain_oracle": (oracle_prediction, oracle_one),
        "learned_gain_only": (gain_only_prediction, gain_only_one),
        # Compatibility name for the promoted action-conditioned system.
        "action_conditioned": (gain_only_prediction, gain_only_one),
        "residual_ablation": (residual_prediction, residual_one),
        "small_angle_controlled": (baseline, baseline_one),
        # Same learned architecture with its action channel removed.
        "learned_gain_only_control_blind": (
            gain_only_blind_prediction,
            gain_only_blind_one,
        ),
        # Backward-compatible short name for the direct learned-model ablation.
        "control_blind": (gain_only_blind_prediction, gain_only_blind_one),
        "small_angle_control_blind": (small_angle_blind, small_angle_blind_one),
    }
    method_metrics: dict[str, object] = {}
    for name, (rollout, one_step) in methods.items():
        energies = torch_energy(rollout).numpy()
        method_metrics[name] = {
            "one_step_error": float(circular_state_error(one_step, one_target).mean()),
            "recursive_rollout_error": float(circular_state_error(rollout, target).mean()),
            "crossing_window_rollout_error": _crossing_window_error(
                rollout,
                target,
                crossing_steps,
                profile.crossing_window_radius,
            ),
            "crossing": _first_crossing_metrics(data.energies, energies),
        }

    return {
        "actual_crossing_count": int(np.sum(actual_crossings)),
        "actual_crossing_rate": float(np.mean(actual_crossings)),
        "autonomous_replay_crossing_count": int(
            np.sum(upward_crossing_steps(autonomous_energies) >= 0)
        ),
        "maximum_absolute_torque": float(np.max(np.abs(data.controls))),
        "external_work_energy_change_mae": float(np.mean(np.abs(work - energy_change))),
        "methods": method_metrics,
    }


def run_control_experiment(
    profile: ControlExperimentProfile | None = None,
) -> dict[str, object]:
    """Train, recursively evaluate, and return a JSON-serializable result."""

    if profile is None:
        profile = ControlExperimentProfile.quick()
    training_data = generate_controlled_pulses(
        seed=profile.seed,
        trajectories=profile.train_trajectories,
        steps=profile.steps,
        dt=profile.dt,
    )
    evaluation_data = generate_controlled_pulses(
        seed=profile.seed + 10_003,
        trajectories=profile.evaluation_trajectories,
        steps=profile.steps,
        dt=profile.dt,
        evaluation=True,
    )
    gain_only_model, gain_history = train_gain_only_model(profile, training_data)
    residual_model, residual_history = train_action_conditioned_model(
        profile,
        training_data,
    )
    evaluation = evaluate_control_model(
        gain_only_model,
        residual_model,
        evaluation_data,
        profile,
    )
    # Predeclared for identifiability: the plant has one unknown actuator gain.
    # The held-out evaluation set is not used for model selection.
    promoted_method = "learned_gain_only"
    return {
        "experiment": "torque_controlled_separatrix_crossing",
        "profile": asdict(profile),
        "training": {
            "crossing_count": int(
                np.sum(upward_crossing_steps(training_data.energies) >= 0)
            ),
            "crossing_rate": float(
                np.mean(upward_crossing_steps(training_data.energies) >= 0)
            ),
            "gain_only": {
                "initial_loss": gain_history[0],
                "final_loss": gain_history[-1],
                "initial_control_gain": 0.35,
                "learned_control_gain": float(gain_only_model.control_gain.detach()),
            },
            "residual_ablation": {
                "initial_loss": residual_history[0],
                "final_loss": residual_history[-1],
                "learned_control_gain": float(residual_model.control_gain.detach()),
            },
        },
        "promoted_learned_system": {
            "method": promoted_method,
            "selection_basis": "predeclared identifiable scalar actuator gain",
        },
        "evaluation": evaluation,
        "interpretation": {
            "exact_unit_gain_oracle": (
                "The full unit-gain plant equation is supplied, not learned, and "
                "defines the numerical error floor."
            ),
            "learned_gain_only": (
                "This is a scalar actuator system-identification exercise, not a "
                "claim of novel control theory."
            ),
            "learned_gain_only_control_blind": (
                "This is the identified gain-only model rolled recursively with "
                "the same architecture but a zeroed action channel."
            ),
        },
        "claim_boundary": (
            "Controls are known future inputs at every transition. Training uses "
            "short state windows; reported long-horizon metrics are recursive and "
            "receive no true future states. The experiment predicts forced crossing; "
            "it does not synthesize a control policy."
        ),
    }
