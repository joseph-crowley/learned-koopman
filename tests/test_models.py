import math

import torch

from learned_koopman.models import (
    EnergyConditionedRotation,
    FixedKoopmanAE,
    ResidualMLP,
    SeparatrixAtlas,
)


def _states(batch: int = 8) -> torch.Tensor:
    theta = torch.linspace(-2.0, 2.0, batch)
    omega = torch.linspace(-0.2, 0.2, batch)
    return torch.stack((torch.sin(theta), torch.cos(theta), omega), dim=-1)


def test_models_preserve_shapes_and_circle_representation() -> None:
    states = _states()
    models = [
        ResidualMLP(16),
        FixedKoopmanAE(16, 4),
        EnergyConditionedRotation(16, 0.02),
    ]
    for model in models:
        prediction = model.step(states)
        assert prediction.shape == states.shape
        torch.testing.assert_close(
            prediction[:, :2].norm(dim=-1),
            torch.ones(len(states)),
            atol=1e-6,
            rtol=1e-6,
        )


def test_energy_conditioned_frequency_is_bounded() -> None:
    model = EnergyConditionedRotation(16, 0.02)
    frequencies = model.angular_frequency(torch.tensor([[0.0], [0.5], [1.0]]))
    assert torch.all(frequencies > 0.0)
    assert torch.all(frequencies < 1.05)


def test_fixed_operator_is_orthogonal_by_construction() -> None:
    model = FixedKoopmanAE(16, 4)
    with torch.no_grad():
        model.generator.normal_()
    operator = model.operator_matrix()
    torch.testing.assert_close(
        operator.T @ operator,
        torch.eye(4),
        atol=1e-5,
        rtol=1e-5,
    )


def test_saddle_chart_is_symplectic_by_construction() -> None:
    regular = EnergyConditionedRotation(16, 0.02)
    atlas = SeparatrixAtlas(regular, 0.02)
    operator = atlas.saddle_operator_matrix()
    torch.testing.assert_close(
        torch.linalg.det(operator),
        torch.ones(()),
        atol=1e-6,
        rtol=1e-6,
    )


def test_atlas_rollout_records_categorical_routes() -> None:
    regular = EnergyConditionedRotation(16, 0.02)
    atlas = SeparatrixAtlas(regular, 0.02)
    initial = _states(batch=1)[0]
    states, diagnostics = atlas.rollout(initial, steps=5)
    assert states.shape == (6, 3)
    assert diagnostics["route_index"].shape == (5,)
    assert diagnostics["route_switch_step"].ndim == 1
    assert int(diagnostics["total_route_switches"]) == len(
        diagnostics["route_switch_step"]
    )
    assert int(diagnostics["rapid_route_reversals"]) == 0
    torch.testing.assert_close(
        states[:, :2].norm(dim=-1),
        torch.ones(6),
        atol=1e-6,
        rtol=1e-6,
    )


def test_stateful_atlas_router_uses_hysteresis_and_minimum_dwell() -> None:
    regular = EnergyConditionedRotation(16, 0.02)
    atlas = SeparatrixAtlas(regular, 0.02)
    condition = torch.tensor([0.95])

    def state_at_saddle_distance(distance: float) -> torch.Tensor:
        return torch.tensor(
            [-math.sin(distance), -math.cos(distance), 0.0],
            dtype=torch.float32,
        )

    overlap = state_at_saddle_distance(1.45)
    assert int(atlas.route_index(overlap, condition)) == 0
    assert (
        int(
            atlas.route_index(
                overlap,
                condition,
                previous_route=1,
                steps_since_switch=atlas.minimum_route_dwell_steps,
            )
        )
        == 1
    )

    outside_exit = state_at_saddle_distance(1.55)
    assert (
        int(
            atlas.route_index(
                outside_exit,
                condition,
                previous_route=1,
                steps_since_switch=atlas.minimum_route_dwell_steps - 1,
            )
        )
        == 1
    )
    assert (
        int(
            atlas.route_index(
                outside_exit,
                condition,
                previous_route=1,
                steps_since_switch=atlas.minimum_route_dwell_steps,
            )
        )
        == 0
    )


def test_route_trace_summary_detects_rapid_reversals_and_alternations() -> None:
    route_trace = torch.tensor([0, 0, 1, 0, 1, 1, 0], dtype=torch.long)
    diagnostics = SeparatrixAtlas.summarize_route_trace(route_trace)

    assert diagnostics["route_switch_step"].tolist() == [2, 3, 4, 6]
    assert int(diagnostics["total_route_switches"]) == 4
    assert int(diagnostics["route_alternations"]) == 2
    assert int(diagnostics["rapid_route_reversals"]) == 3
    assert int(diagnostics["max_route_switches_in_window"]) == 4
