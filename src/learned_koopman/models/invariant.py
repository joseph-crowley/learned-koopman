from __future__ import annotations

import torch
from torch import nn


class LearnedInvariant(nn.Module):
    """A scalar state function learned without access to a conserved quantity.

    The network deliberately has no physics-specific input or energy shortcut:
    it receives only the circular pendulum state ``(sin(theta), cos(theta),
    omega)``.  The experiment loss, defined separately, makes the scalar
    constant on observed trajectories and smooth between neighboring orbit
    representatives.
    """

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state).squeeze(-1)
