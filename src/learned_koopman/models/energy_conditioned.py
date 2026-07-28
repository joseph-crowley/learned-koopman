from __future__ import annotations

import torch
from torch import nn

from learned_koopman.physics import normalize_circular_state, torch_energy


class EnergyConditionedRotation(nn.Module):
    """A fibered rotation model: one linear phase update per energy shell."""

    def __init__(self, hidden_dim: int, dt: float) -> None:
        super().__init__()
        self.dt = dt
        self.phase_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )
        self.frequency = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),
        )

    @staticmethod
    def normalized_energy(state: torch.Tensor) -> torch.Tensor:
        return ((torch_energy(state) + 1.0) / 2.0).clamp(0.0, 1.0).unsqueeze(-1)

    def encode_phase(self, state: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.phase_encoder(state), dim=-1, eps=1e-8)

    def angular_frequency(self, energy: torch.Tensor) -> torch.Tensor:
        return 1.05 * self.frequency(energy)

    def rotate(self, phase: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        angle = self.angular_frequency(energy) * self.dt
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        first = cosine * phase[..., 0:1] - sine * phase[..., 1:2]
        second = sine * phase[..., 0:1] + cosine * phase[..., 1:2]
        return torch.cat((first, second), dim=-1)

    def decode(self, phase: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        return normalize_circular_state(self.decoder(torch.cat((phase, energy), dim=-1)))

    def reconstruct(self, state: torch.Tensor) -> torch.Tensor:
        energy = self.normalized_energy(state)
        return self.decode(self.encode_phase(state), energy)

    def step(self, state: torch.Tensor, energy: torch.Tensor | None = None) -> torch.Tensor:
        condition = self.normalized_energy(state) if energy is None else energy
        phase = self.rotate(self.encode_phase(state), condition)
        return self.decode(phase, condition)
