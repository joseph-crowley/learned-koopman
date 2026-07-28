import json

import numpy as np
import torch

from learned_koopman.control import (
    ExactUnitGainOracle,
    generate_controlled_pulses,
    mechanical_work,
    simulate_controlled,
    upward_crossing_steps,
)
from learned_koopman.control_experiment import (
    ControlExperimentProfile,
    _first_crossing_metrics,
    run_control_experiment,
)


def test_controlled_dataset_crosses_only_when_torque_injects_energy() -> None:
    data = generate_controlled_pulses(
        seed=19,
        trajectories=12,
        steps=180,
        dt=0.04,
        evaluation=True,
    )
    crossing = upward_crossing_steps(data.energies)
    assert np.any(crossing >= 0)
    assert np.any(crossing < 0)
    assert np.all(data.energies[:, 0] < 1.0)
    assert np.max(np.abs(data.controls)) <= 1.17 + 1e-12

    _, _, autonomous_omega, autonomous_energy = simulate_controlled(
        data.angles[:, 0],
        data.angular_velocities[:, 0],
        np.zeros_like(data.controls),
        dt=0.04,
    )
    assert np.all(upward_crossing_steps(autonomous_energy) < 0)
    assert np.max(np.abs(autonomous_energy - autonomous_energy[:, :1])) < 2e-4

    work = mechanical_work(data.controls, data.angular_velocities, 0.04)
    energy_change = data.energies[:, -1] - data.energies[:, 0]
    assert np.mean(np.abs(work - energy_change)) < 2e-3
    assert autonomous_omega.shape == data.angular_velocities.shape


def test_quick_control_experiment_is_recursive_and_machine_readable() -> None:
    profile = ControlExperimentProfile.quick(seed=11)
    result = run_control_experiment(profile)
    json.dumps(result, allow_nan=False)

    evaluation = result["evaluation"]
    assert evaluation["actual_crossing_count"] > 0
    assert evaluation["autonomous_replay_crossing_count"] == 0
    gain_training = result["training"]["gain_only"]
    assert gain_training["final_loss"] < gain_training["initial_loss"]
    assert abs(gain_training["learned_control_gain"] - 1.0) < 0.05

    methods = evaluation["methods"]
    learned = methods["learned_gain_only"]
    residual = methods["residual_ablation"]
    blind = methods["control_blind"]
    assert methods["action_conditioned"] == learned
    assert methods["learned_gain_only_control_blind"] == blind
    assert learned["one_step_error"] < blind["one_step_error"]
    assert learned["crossing_window_rollout_error"] < blind["crossing_window_rollout_error"]
    assert learned["crossing"]["event_recall"] >= blind["crossing"]["event_recall"]
    assert learned["crossing_window_rollout_error"] < residual["crossing_window_rollout_error"]
    assert result["promoted_learned_system"]["method"] == "learned_gain_only"
    assert (
        result["promoted_learned_system"]["selection_basis"]
        == "predeclared identifiable scalar actuator gain"
    )
    assert "system-identification" in result["interpretation"]["learned_gain_only"]
    assert "supplied" in result["interpretation"]["exact_unit_gain_oracle"]
    assert "recursive" in result["claim_boundary"]


def test_exact_unit_gain_oracle_matches_the_controlled_simulator() -> None:
    data = generate_controlled_pulses(
        seed=23,
        trajectories=6,
        steps=90,
        dt=0.04,
        evaluation=True,
    )
    oracle = ExactUnitGainOracle(0.04).double()
    initial = torch.tensor(data.states[:, 0], dtype=torch.float64)
    controls = torch.tensor(data.controls, dtype=torch.float64)
    with torch.no_grad():
        prediction = oracle.rollout(initial, controls).numpy()
    np.testing.assert_allclose(prediction, data.states, atol=1e-12)


def test_undefined_crossing_precision_and_recall_are_none() -> None:
    no_crossing = np.array([[-0.8, -0.7, -0.6]])
    metrics = _first_crossing_metrics(no_crossing, no_crossing)
    assert metrics["event_precision"] is None
    assert metrics["event_recall"] is None

    true_crossing = np.array([[-0.8, 0.9, 1.1]])
    missed = _first_crossing_metrics(true_crossing, no_crossing)
    assert missed["event_precision"] is None
    assert missed["event_recall"] == 0.0
