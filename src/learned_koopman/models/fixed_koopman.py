from __future__ import annotations

import torch
from torch import nn

from learned_koopman.physics import normalize_circular_state


class FixedKoopmanAE(nn.Module):
    """A coherent fixed-operator Koopman autoencoder baseline."""

    def __init__(self, hidden_dim: int, latent_dim: int, dt: float = 0.02) -> None:
        super().__init__()
        self.dt = dt
        self.encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.generator = nn.Parameter(torch.zeros(latent_dim, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),
        )

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return normalize_circular_state(self.decoder(latent))

    def reconstruct(self, state: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(state))

    def step_latent(self, latent: torch.Tensor) -> torch.Tensor:
        operator = torch.matrix_exp(self.dt * (self.generator - self.generator.T))
        return latent @ operator.T

    def operator_matrix(self) -> torch.Tensor:
        return torch.matrix_exp(self.dt * (self.generator - self.generator.T))

    def step(self, state: torch.Tensor) -> torch.Tensor:
        return self.decode(self.step_latent(self.encode(state)))
