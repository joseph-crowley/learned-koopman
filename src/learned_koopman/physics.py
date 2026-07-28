from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

Array = NDArray[np.float64]


def state_from_angle(theta: Array, omega: Array) -> Array:
    """Represent the pendulum on its natural circular angle topology."""

    return np.stack((np.sin(theta), np.cos(theta), omega), axis=-1)


def angle_from_state(state: Array) -> Array:
    return np.arctan2(state[..., 0], state[..., 1])


def pendulum_energy(theta: Array, omega: Array) -> Array:
    """Dimensionless Hamiltonian with separatrix energy H=1."""

    return 0.5 * omega**2 - np.cos(theta)


def pendulum_energy_from_state(state: Array) -> Array:
    return 0.5 * state[..., 2] ** 2 - state[..., 1]


def pendulum_frequency(amplitude: Array) -> Array:
    """Exact libration frequency from the complete elliptic integral via AGM."""

    parameter = np.sin(amplitude / 2.0) ** 2
    arithmetic = np.ones_like(parameter)
    geometric = np.sqrt(1.0 - parameter)
    for _ in range(24):
        arithmetic, geometric = (
            0.5 * (arithmetic + geometric),
            np.sqrt(arithmetic * geometric),
        )
    return arithmetic


def torch_energy(state: torch.Tensor) -> torch.Tensor:
    return 0.5 * state[..., 2] ** 2 - state[..., 1]


def velocity_verlet_step(theta: Array, omega: Array, dt: float) -> tuple[Array, Array]:
    """One reversible symplectic step for theta'' = -sin(theta)."""

    omega_half = omega - 0.5 * dt * np.sin(theta)
    theta_next = theta + dt * omega_half
    omega_next = omega_half - 0.5 * dt * np.sin(theta_next)
    return theta_next, omega_next


def simulate(
    theta0: Array,
    omega0: Array,
    *,
    steps: int,
    dt: float,
) -> tuple[Array, Array, Array]:
    """Simulate many initial conditions and return states, angles, and energies."""

    theta = np.empty((len(theta0), steps), dtype=np.float64)
    omega = np.empty_like(theta)
    theta[:, 0] = theta0
    omega[:, 0] = omega0
    for index in range(1, steps):
        theta[:, index], omega[:, index] = velocity_verlet_step(
            theta[:, index - 1],
            omega[:, index - 1],
            dt,
        )
    states = state_from_angle(theta, omega)
    energies = pendulum_energy(theta, omega)
    return states, theta, energies


def normalize_circular_state(state: torch.Tensor) -> torch.Tensor:
    circle = torch.nn.functional.normalize(state[..., :2], dim=-1, eps=1e-8)
    return torch.cat((circle, state[..., 2:3]), dim=-1)


def circular_state_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    circle = (prediction[..., :2] - target[..., :2]).square().sum(dim=-1)
    velocity = (prediction[..., 2] - target[..., 2]).square()
    return circle + 0.25 * velocity
