from __future__ import annotations

import numpy as np

from learned_koopman.operator_family import (
    fit_fibered_operator,
    observable_feature_names,
    polynomial_observables,
)


def test_polynomial_observables_keep_physical_state_decodable() -> None:
    states = np.array([[2.0, -3.0], [0.5, 4.0]])
    lifted = polynomial_observables(states, degree=2)

    assert observable_feature_names(("q", "v"), degree=2) == (
        "1",
        "q",
        "v",
        "q*q",
        "q*v",
        "v*v",
    )
    np.testing.assert_allclose(lifted[:, 1:3], states)
    np.testing.assert_allclose(lifted[:, 3:], [[4.0, -6.0, 9.0], [0.25, 2.0, 16.0]])


def test_fibered_operator_recovers_a_parameterized_linear_family() -> None:
    coordinates = np.linspace(-1.0, 1.0, 12)
    trajectories = []
    for coordinate in coordinates:
        matrix = np.array(
            [
                [0.98, -0.04 - 0.015 * coordinate],
                [0.04 + 0.015 * coordinate, 0.98],
            ]
        )
        state = np.array([1.0, 0.2])
        run = [state]
        for _ in range(59):
            state = state @ matrix
            run.append(state)
        trajectories.append(run)
    data = np.asarray(trajectories)

    fibered = fit_fibered_operator(
        data[:9],
        coordinates[:9],
        dt=0.05,
        family_degree=1,
        observable_degree=1,
        ridge=1e-10,
    )
    global_model = fit_fibered_operator(
        data[:9],
        coordinates[:9],
        dt=0.05,
        family_degree=0,
        observable_degree=1,
        ridge=1e-10,
    )
    fibered_prediction = fibered.rollout(
        data[9:, 0],
        coordinates[9:],
        steps=data.shape[1],
    )
    global_prediction = global_model.rollout(
        data[9:, 0],
        coordinates[9:],
        steps=data.shape[1],
    )
    fibered_error = np.sqrt(np.mean(np.square(fibered_prediction - data[9:])))
    global_error = np.sqrt(np.mean(np.square(global_prediction - data[9:])))

    assert fibered_error < 1e-7
    assert fibered_error < global_error * 1e-4
