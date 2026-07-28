from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from learned_koopman.physics import pendulum_energy, state_from_angle

Array = NDArray[np.float64]


@dataclass(frozen=True)
class ControlledTrajectories:
    """A batch of torque-driven trajectories and their reproducibility metadata."""

    states: Array
    angles: Array
    angular_velocities: Array
    controls: Array
    energies: Array
    pulse_amplitudes: Array
    pulse_starts: NDArray[np.int64]
    pulse_durations: NDArray[np.int64]

    @property
    def trajectory_count(self) -> int:
        return int(self.states.shape[0])

    @property
    def step_count(self) -> int:
        return int(self.states.shape[1])


def controlled_velocity_verlet_step(
    theta: Array,
    omega: Array,
    torque: Array,
    dt: float,
) -> tuple[Array, Array]:
    """One second-order step for theta'' = -sin(theta) + torque.

    Torque is held constant over the step. This is the same kick-drift-kick
    structure as the autonomous integrator, with the external generalized
    force included in both kicks.
    """

    omega_half = omega + 0.5 * dt * (-np.sin(theta) + torque)
    theta_next = theta + dt * omega_half
    omega_next = omega_half + 0.5 * dt * (-np.sin(theta_next) + torque)
    return theta_next, omega_next


def simulate_controlled(
    theta0: Array,
    omega0: Array,
    controls: Array,
    *,
    dt: float,
) -> tuple[Array, Array, Array, Array]:
    """Simulate batched piecewise-constant torque sequences.

    ``controls`` has shape ``(trajectory, steps - 1)`` because each value acts
    on the transition from state ``k`` to state ``k + 1``.
    """

    if controls.ndim != 2:
        raise ValueError("controls must have shape (trajectory, steps - 1)")
    if len(theta0) != len(omega0) or len(theta0) != controls.shape[0]:
        raise ValueError("initial conditions and controls must have matching batches")

    steps = controls.shape[1] + 1
    theta = np.empty((len(theta0), steps), dtype=np.float64)
    omega = np.empty_like(theta)
    theta[:, 0] = theta0
    omega[:, 0] = omega0
    for index in range(steps - 1):
        theta[:, index + 1], omega[:, index + 1] = controlled_velocity_verlet_step(
            theta[:, index],
            omega[:, index],
            controls[:, index],
            dt,
        )
    states = state_from_angle(theta, omega)
    energies = pendulum_energy(theta, omega)
    return states, theta, omega, energies


def upward_crossing_steps(energies: Array, threshold: float = 1.0) -> NDArray[np.int64]:
    """Return the first below-to-above separatrix crossing, or -1 per trajectory."""

    if energies.ndim == 1:
        energies = energies[None, :]
    crossed = (energies[:, :-1] < threshold) & (energies[:, 1:] >= threshold)
    result = np.full(energies.shape[0], -1, dtype=np.int64)
    has_crossing = crossed.any(axis=1)
    result[has_crossing] = np.argmax(crossed[has_crossing], axis=1) + 1
    return result


def mechanical_work(controls: Array, angular_velocities: Array, dt: float) -> Array:
    """Trapezoidal estimate of external work, integral(torque * omega dt)."""

    average_velocity = 0.5 * (
        angular_velocities[:, :-1] + angular_velocities[:, 1:]
    )
    return np.sum(controls * average_velocity * dt, axis=1)


def make_pulse_controls(
    amplitudes: Array,
    starts: NDArray[np.int64],
    durations: NDArray[np.int64],
    *,
    steps: int,
) -> Array:
    """Construct one bounded, rectangular torque pulse per trajectory."""

    if not (len(amplitudes) == len(starts) == len(durations)):
        raise ValueError("pulse arrays must have equal length")
    controls = np.zeros((len(amplitudes), steps - 1), dtype=np.float64)
    for row, (amplitude, start, duration) in enumerate(
        zip(amplitudes, starts, durations, strict=True)
    ):
        stop = min(int(start + duration), steps - 1)
        controls[row, int(start) : stop] = amplitude
    return controls


def generate_controlled_pulses(
    *,
    seed: int,
    trajectories: int,
    steps: int,
    dt: float,
    evaluation: bool = False,
) -> ControlledTrajectories:
    """Generate bounded pulses spanning subcritical and true crossing outcomes.

    Training uses randomized pulse parameters. Evaluation uses a deterministic
    held-out grid with both signs, unseen amplitudes, and subcritical cases.
    """

    if trajectories < 4:
        raise ValueError("at least four trajectories are needed")
    rng = np.random.default_rng(seed)

    if evaluation:
        magnitudes = np.resize(
            np.array([0.78, 0.86, 0.93, 1.01, 1.09, 1.17], dtype=np.float64),
            trajectories,
        )
        directions = np.where(np.arange(trajectories) % 2 == 0, 1.0, -1.0)
        amplitudes = magnitudes * directions
        starts = 8 + np.arange(trajectories, dtype=np.int64) % 7
        durations = np.full(trajectories, min(112, steps - 20), dtype=np.int64)
        theta0 = directions * np.linspace(0.12, 0.34, trajectories)
        omega0 = directions * np.linspace(0.0, 0.08, trajectories)
    else:
        directions = rng.choice(np.array([-1.0, 1.0]), size=trajectories)
        # Include non-crossing pulses so crossing is an outcome, not a label leak.
        magnitudes = rng.uniform(0.72, 1.22, size=trajectories)
        amplitudes = magnitudes * directions
        starts = rng.integers(6, 18, size=trajectories, dtype=np.int64)
        duration_low = min(82, steps - 30)
        duration_high = min(122, steps - 18)
        durations = rng.integers(
            duration_low,
            duration_high + 1,
            size=trajectories,
            dtype=np.int64,
        )
        theta0 = directions * rng.uniform(0.08, 0.38, size=trajectories)
        omega0 = directions * rng.uniform(-0.04, 0.10, size=trajectories)

    controls = make_pulse_controls(amplitudes, starts, durations, steps=steps)
    states, angles, angular_velocities, energies = simulate_controlled(
        theta0,
        omega0,
        controls,
        dt=dt,
    )
    return ControlledTrajectories(
        states=states,
        angles=angles,
        angular_velocities=angular_velocities,
        controls=controls,
        energies=energies,
        pulse_amplitudes=amplitudes,
        pulse_starts=starts,
        pulse_durations=durations,
    )


class GainOnlyControlledPendulum(nn.Module):
    """Identify the scalar actuator gain in an otherwise supplied plant model."""

    def __init__(self, dt: float, initial_gain: float = 0.35) -> None:
        super().__init__()
        self.dt = dt
        self.control_gain = nn.Parameter(torch.tensor(initial_gain))

    def acceleration(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        return -state[..., 0] + self.control_gain * control

    def step(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        theta = torch.atan2(state[..., 0], state[..., 1])
        omega = state[..., 2]
        acceleration = self.acceleration(state, control)
        omega_half = omega + 0.5 * self.dt * acceleration
        theta_next = theta + self.dt * omega_half
        provisional = torch.stack(
            (torch.sin(theta_next), torch.cos(theta_next), omega_half),
            dim=-1,
        )
        acceleration_next = self.acceleration(provisional, control)
        omega_next = omega_half + 0.5 * self.dt * acceleration_next
        return torch.stack(
            (torch.sin(theta_next), torch.cos(theta_next), omega_next),
            dim=-1,
        )

    def rollout(self, initial: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        state = initial
        result = [state]
        for index in range(controls.shape[1]):
            state = self.step(state, controls[:, index])
            result.append(state)
        return torch.stack(result, dim=1)


class ExactUnitGainOracle(GainOnlyControlledPendulum):
    """Supplied simulator equation, exposed as an error-floor oracle."""

    def __init__(self, dt: float) -> None:
        super().__init__(dt=dt, initial_gain=1.0)
        self.control_gain.requires_grad_(False)


class ActionConditionedPendulum(GainOnlyControlledPendulum):
    """Higher-capacity residual ablation for the torque-conditioned predictor.

    The conservative force is supplied as a physics prior. PyTorch learns the
    control effectiveness and a bounded residual acceleration. Controls remain
    explicit inputs at every predicted step. This model is deliberately
    reported as an ablation against the more identifiable gain-only system.
    """

    def __init__(self, dt: float, hidden_dim: int = 32) -> None:
        super().__init__(dt=dt, initial_gain=0.35)
        self.residual = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def acceleration(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        control_column = control.reshape(state.shape[:-1] + (1,))
        features = torch.cat((state, control_column), dim=-1)
        correction = 0.25 * torch.tanh(self.residual(features).squeeze(-1))
        return super().acceleration(state, control) + correction


def small_angle_controlled_step(
    state: torch.Tensor,
    control: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Transparent global-linear baseline: theta'' = -theta + torque."""

    theta = torch.atan2(state[..., 0], state[..., 1])
    omega = state[..., 2]
    omega_half = omega + 0.5 * dt * (-theta + control)
    theta_next = theta + dt * omega_half
    omega_next = omega_half + 0.5 * dt * (-theta_next + control)
    return torch.stack(
        (torch.sin(theta_next), torch.cos(theta_next), omega_next),
        dim=-1,
    )


def baseline_rollout(
    initial: torch.Tensor,
    controls: torch.Tensor,
    *,
    dt: float,
    ignore_control: bool = False,
) -> torch.Tensor:
    """Recursively roll out the small-angle baseline with known future controls."""

    state = initial
    result = [state]
    for index in range(controls.shape[1]):
        control = torch.zeros_like(controls[:, index]) if ignore_control else controls[:, index]
        state = small_angle_controlled_step(state, control, dt)
        result.append(state)
    return torch.stack(result, dim=1)
