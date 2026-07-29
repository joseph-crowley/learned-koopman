from __future__ import annotations

import torch
from torch import nn


class LearnedInvariant(nn.Module):
    """A scalar state function learned without access to a conserved quantity.

    The experiment loss, defined separately, makes the scalar constant on
    observed trajectories and smooth between neighboring orbit
    representatives. ``input_dim=3`` preserves the circular-pendulum research
    cell, while the mechanics workbench can use arbitrary measured state
    vectors.
    """

    def __init__(self, hidden_dim: int = 32, *, input_dim: int = 3) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state).squeeze(-1)
