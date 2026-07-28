from __future__ import annotations

import math

import torch
from torch import nn

from learned_koopman.models.energy_conditioned import EnergyConditionedRotation


class SeparatrixAtlas(nn.Module):
    """Two structured local charts with an explicit geometric router.

    The regular chart is the trained energy-conditioned rotation. Near the
    unstable upright equilibrium, a hyperbolic canonical chart advances local
    saddle coordinates. Chart transitions pass through the physical state, so
    the rollout remains autonomous while avoiding a fictitious global latent
    coordinate across the separatrix neighborhood.
    """

    minimum_saddle_energy = 0.80
    maximum_saddle_distance = 1.40

    def __init__(self, regular: EnergyConditionedRotation, dt: float) -> None:
        super().__init__()
        self.dt = dt
        self.regular = regular
        self.regular.requires_grad_(False)
        inverse_softplus_one = math.log(math.expm1(1.0))
        self.raw_saddle_rate = nn.Parameter(torch.tensor(inverse_softplus_one))

    @property
    def saddle_rate(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_saddle_rate) + 1e-4

    @staticmethod
    def saddle_coordinates(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return local q=0 saddle displacement and canonical momentum."""

        displacement = torch.atan2(-state[..., 0], -state[..., 1])
        momentum = state[..., 2]
        return displacement, momentum

    def saddle_operator_matrix(self) -> torch.Tensor:
        """Exact flow of qdot=p, pdot=lambda^2 q for one sample interval."""

        rate = self.saddle_rate
        scaled_time = rate * self.dt
        cosine = torch.cosh(scaled_time)
        sine = torch.sinh(scaled_time)
        return torch.stack(
            (
                torch.stack((cosine, sine / rate)),
                torch.stack((rate * sine, cosine)),
            )
        )

    def saddle_step(self, state: torch.Tensor) -> torch.Tensor:
        displacement, momentum = self.saddle_coordinates(state)
        operator = self.saddle_operator_matrix()
        next_displacement = operator[0, 0] * displacement + operator[0, 1] * momentum
        next_momentum = operator[1, 0] * displacement + operator[1, 1] * momentum
        return torch.stack(
            (-torch.sin(next_displacement), -torch.cos(next_displacement), next_momentum),
            dim=-1,
        )

    def route_index(
        self,
        state: torch.Tensor,
        normalized_energy: torch.Tensor,
    ) -> torch.Tensor:
        displacement, _ = self.saddle_coordinates(state)
        physical_energy = normalized_energy.squeeze(-1) * 2.0 - 1.0
        local_validity = (physical_energy > self.minimum_saddle_energy) & (
            displacement.abs() < self.maximum_saddle_distance
        )
        return local_validity.long()

    def project_to_energy_shell(
        self,
        state: torch.Tensor,
        normalized_energy: torch.Tensor,
    ) -> torch.Tensor:
        """Enforce the known invariant only in the atlas's high-energy regime."""

        circle = torch.nn.functional.normalize(state[..., :2], dim=-1, eps=1e-8)
        physical_energy = normalized_energy.squeeze(-1) * 2.0 - 1.0
        available_kinetic = torch.clamp(
            2.0 * (physical_energy + circle[..., 1]),
            min=0.0,
        )
        momentum_sign = torch.where(
            state[..., 2] < 0.0,
            -torch.ones_like(state[..., 2]),
            torch.ones_like(state[..., 2]),
        )
        projected = torch.cat(
            (
                circle,
                (momentum_sign * torch.sqrt(available_kinetic)).unsqueeze(-1),
            ),
            dim=-1,
        )
        use_projection = physical_energy > self.minimum_saddle_energy
        return torch.where(use_projection.unsqueeze(-1), projected, state)

    def regular_step(
        self,
        state: torch.Tensor,
        normalized_energy: torch.Tensor,
    ) -> torch.Tensor:
        phase = self.regular.encode_phase(state)
        return self.regular.decode(
            self.regular.rotate(phase, normalized_energy),
            normalized_energy,
        )

    def rollout(
        self,
        initial: torch.Tensor,
        steps: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Autonomously evolve one state while carrying chart-local coordinates."""

        if initial.ndim != 1:
            raise ValueError("SeparatrixAtlas.rollout currently expects one state vector.")
        states = [initial]
        route_indices: list[torch.Tensor] = []
        switch_disagreements: list[torch.Tensor] = []
        state = initial
        condition = self.regular.normalized_energy(initial)
        phase = self.regular.encode_phase(initial)
        previous_route: int | None = None

        for _ in range(steps):
            route = int(self.route_index(state, condition).item())
            route_indices.append(torch.tensor(route, device=state.device))

            fresh_regular = self.project_to_energy_shell(
                self.regular_step(state, condition),
                condition,
            )
            saddle_prediction = self.project_to_energy_shell(
                self.saddle_step(state),
                condition,
            )
            if previous_route is not None and route != previous_route:
                difference = fresh_regular - saddle_prediction
                switch_disagreements.append(torch.sqrt(torch.mean(difference.square())))

            if route == 1:
                state = saddle_prediction
            else:
                if previous_route == 1:
                    phase = self.regular.encode_phase(state)
                phase = self.regular.rotate(phase, condition)
                state = self.regular.decode(phase, condition)

            state = self.project_to_energy_shell(state, condition)
            states.append(state)
            previous_route = route

        empty = torch.empty(0, dtype=initial.dtype, device=initial.device)
        diagnostics = {
            "route_index": torch.stack(route_indices) if route_indices else empty.long(),
            "switch_disagreement": (
                torch.stack(switch_disagreements) if switch_disagreements else empty
            ),
        }
        return torch.stack(states), diagnostics
