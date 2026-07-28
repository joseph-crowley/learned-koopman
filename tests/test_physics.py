import numpy as np

from learned_koopman.physics import pendulum_energy_from_state, pendulum_frequency, simulate


def test_velocity_verlet_has_small_bounded_energy_error() -> None:
    states, _, _ = simulate(
        np.array([2.0]),
        np.array([0.0]),
        steps=5_000,
        dt=0.01,
    )
    energy = pendulum_energy_from_state(states[0])
    assert np.max(np.abs(energy - energy[0])) < 1e-4


def test_circular_state_stays_on_unit_circle() -> None:
    states, _, _ = simulate(
        np.array([0.5, 2.5]),
        np.zeros(2),
        steps=100,
        dt=0.02,
    )
    radius = np.linalg.norm(states[..., :2], axis=-1)
    np.testing.assert_allclose(radius, 1.0, atol=1e-12)


def test_exact_frequency_slows_with_amplitude() -> None:
    frequencies = pendulum_frequency(np.array([0.1, 1.0, 2.8]))
    assert np.all(np.diff(frequencies) < 0.0)
    assert frequencies[0] < 1.0
    assert frequencies[0] > 0.99
