import inspect

import torch

from learned_koopman.invariant_experiment import (
    InvariantExperimentConfig,
    _trajectory_tensor,
    invariant_discovery_loss,
    run_invariant_experiment,
    train_invariant_model,
)
from learned_koopman.models.invariant import LearnedInvariant


def _toy_orbits() -> torch.Tensor:
    phase = torch.linspace(0.0, 2.0 * torch.pi, 32)
    radii = torch.tensor([0.3, 0.6, 0.9, 1.2])
    trajectories = []
    for radius in radii:
        trajectories.append(
            torch.stack(
                (
                    torch.sin(radius * torch.cos(phase)),
                    torch.cos(radius * torch.cos(phase)),
                    radius * torch.sin(phase),
                ),
                dim=-1,
            )
        )
    return torch.stack(trajectories)


def test_training_api_has_no_energy_or_amplitude_label() -> None:
    parameters = inspect.signature(train_invariant_model).parameters
    assert "energy" not in parameters
    assert "energies" not in parameters
    assert "amplitude" not in parameters
    assert "labels" not in parameters


def test_training_segments_start_away_from_a_single_turning_point_section() -> None:
    trajectories = _trajectory_tensor(8, 20, 0.03)
    assert torch.count_nonzero(trajectories[:, 0, 2].abs() > 1e-3) >= 6


def test_anti_collapse_term_penalizes_constant_coordinate() -> None:
    model = LearnedInvariant(hidden_dim=8)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    loss, terms = invariant_discovery_loss(
        model,
        _toy_orbits(),
        graph_neighbors=1,
        constancy_weight=8.0,
        graph_weight=0.15,
        variance_weight=1.0,
        centering_weight=0.02,
    )
    assert float(loss.detach()) >= 0.99
    assert float(terms["orbit_std"].detach()) == 0.0


def test_small_training_run_produces_noncollapsed_coordinate() -> None:
    _, history = train_invariant_model(
        _toy_orbits(),
        hidden_dim=12,
        epochs=60,
        learning_rate=5e-3,
        seed=3,
    )
    assert history[-1]["loss"] < history[0]["loss"]
    assert history[-1]["orbit_std"] > 0.2


def test_same_seed_reproduces_training_exactly() -> None:
    kwargs = {
        "hidden_dim": 8,
        "epochs": 8,
        "learning_rate": 5e-3,
        "seed": 11,
    }
    first, first_history = train_invariant_model(_toy_orbits(), **kwargs)
    second, second_history = train_invariant_model(_toy_orbits(), **kwargs)
    assert first_history == second_history
    for first_parameter, second_parameter in zip(
        first.parameters(),
        second.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(first_parameter, second_parameter)


def test_quick_experiment_schema_and_label_boundary() -> None:
    config = InvariantExperimentConfig(
        profile="quick",
        dt=0.03,
        train_trajectories=8,
        train_steps=50,
        evaluation_trajectories=7,
        evaluation_steps=60,
        hidden_dim=12,
        epochs=35,
        learning_rate=5e-3,
        graph_neighbors=1,
        constancy_weight=8.0,
        graph_weight=0.15,
        variance_weight=1.0,
        centering_weight=0.02,
    )
    result = run_invariant_experiment(config=config, seeds=(3, 5))
    assert result["experiment"] == "label_free_invariant_discovery"
    assert result["seeds"] == [3, 5]
    assert result["scientific_contract"]["evaluation_oracle"].endswith("after training")
    assert len(result["runs"]) == 2
    assert "affine_aligned_energy_r2" in result["aggregate"]
    assert result["aggregate"]["quotient_coordinate_std"]["min"] > 0.05
