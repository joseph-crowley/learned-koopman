from __future__ import annotations

import numpy as np

from learned_koopman.map_fixtures import (
    ExactGauge,
    KickHarmonic,
    ObservationChart,
    TwistKickMap,
    simulate_map_trajectories,
)


def _determinant(function, point: np.ndarray, step: float = 1e-6) -> float:
    columns = []
    for index in range(2):
        offset = np.zeros(2)
        offset[index] = step
        columns.append(
            (np.asarray(function(point + offset)) - np.asarray(function(point - offset)))
            / (2.0 * step)
        )
    return float(np.linalg.det(np.column_stack(columns)))


def test_twist_kick_map_is_area_preserving_and_uses_generating_amplitude() -> None:
    system = TwistKickMap(
        base_frequency=1.6,
        twist=0.3,
        kicks=(KickHarmonic(3, 0.0075, 0.9),),
    )

    determinant = _determinant(
        lambda value: np.asarray(system.step(value[:1], value[1:])).reshape(2),
        np.array([1.4, 0.7]),
    )

    assert abs(determinant - 1.0) < 2e-9
    assert np.isclose(system.island_half_width(3), 2.0 * np.sqrt(0.0025 / 0.3))


def test_exact_gauge_round_trips_and_preserves_area() -> None:
    gauge = ExactGauge(amplitude=0.08 / 6.0, order=6, phase=0.4)
    action = np.linspace(0.8, 2.4, 50)
    angle = np.linspace(-np.pi, np.pi, 50, endpoint=False)

    transformed = gauge.forward(action, angle)
    rebuilt = gauge.inverse(*transformed)

    np.testing.assert_allclose(rebuilt[0], action, atol=2e-12)
    np.testing.assert_allclose(rebuilt[1], angle, atol=2e-12)
    determinant = _determinant(
        lambda value: np.asarray(
            gauge.forward(value[:1], value[1:])
        ).reshape(2),
        np.array([1.2, -0.5]),
    )
    assert abs(determinant - 1.0) < 2e-9


def test_observation_chart_round_trips_and_fixture_randomizes_phase() -> None:
    chart = ObservationChart()
    actions = np.linspace(0.7, 2.6, 48)
    angles = np.random.default_rng(20260728).uniform(-np.pi, np.pi, 48)

    states = chart.observe(actions, angles)
    rebuilt = chart.unobserve(states)

    np.testing.assert_allclose(rebuilt[0], actions, atol=2e-12)
    np.testing.assert_allclose(rebuilt[1], angles, atol=2e-12)
    assert abs(np.linalg.det(chart.matrix) - 1.0) < 1e-12

    system = TwistKickMap(
        base_frequency=1.6,
        twist=0.3,
        kicks=(KickHarmonic(3, 0.0075, 0.9),),
    )
    bundle = simulate_map_trajectories(
        system,
        chart,
        initial_actions=actions,
        initial_angles=angles,
        steps=40,
    )
    assert bundle.states.shape == (48, 40, 2)
    assert np.std(bundle.angles[:, 0]) > 1.0
    recovered = chart.unobserve(bundle.states)
    np.testing.assert_allclose(recovered[0], bundle.actions, atol=2e-12)
