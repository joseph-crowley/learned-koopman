from __future__ import annotations

import torch
from torch import nn

from learned_koopman.physics import normalize_circular_state


class ResidualMLP(nn.Module):
    """A strong nonlinear one-step baseline with no Koopman bottleneck."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),
        )

    def step(self, state: torch.Tensor) -> torch.Tensor:
        return normalize_circular_state(state + self.network(state))


def persistence_rollout(initial: torch.Tensor, steps: int) -> torch.Tensor:
    return initial.unsqueeze(0).repeat(steps + 1, 1)


def small_angle_step(state: torch.Tensor, dt: float) -> torch.Tensor:
    theta = torch.atan2(state[..., 0], state[..., 1])
    omega = state[..., 2]
    omega_half = omega - 0.5 * dt * theta
    theta_next = theta + dt * omega_half
    omega_next = omega_half - 0.5 * dt * theta_next
    return torch.stack((torch.sin(theta_next), torch.cos(theta_next), omega_next), dim=-1)
