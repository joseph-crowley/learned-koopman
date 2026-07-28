from __future__ import annotations

import torch
from torch import nn

from learned_koopman.physics import normalize_circular_state


class SimplexTransferOperator(nn.Module):
    """A finite-state transfer model with learned soft state memberships.

    The encoder returns a categorical probability vector for each physical
    state.  The transition operator is parameterized row-wise with a softmax,
    so it is positive and row stochastic by construction.  This is a
    categorical latent-state model, not a variational autoencoder: its
    likelihood and latent geometry are both discrete.
    """

    def __init__(
        self,
        state_dim: int = 3,
        n_states: int = 6,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if n_states < 2:
            raise ValueError("n_states must be at least two")
        self.state_dim = state_dim
        self.n_states = n_states
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_states),
        )
        self.raw_transition = nn.Parameter(torch.zeros(n_states, n_states))
        self.register_buffer("prototypes", torch.zeros(n_states, state_dim))

    def membership_logits(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)

    def membership(self, state: torch.Tensor) -> torch.Tensor:
        """Return non-negative memberships that sum to one."""

        return torch.softmax(self.membership_logits(state), dim=-1)

    def transition_matrix(self) -> torch.Tensor:
        """Return the positive row-stochastic one-lag transfer operator."""

        return torch.softmax(self.raw_transition, dim=-1)

    def propagate(
        self,
        membership: torch.Tensor,
        *,
        steps: int = 1,
    ) -> torch.Tensor:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        propagated = membership
        transition = self.transition_matrix()
        for _ in range(steps):
            propagated = propagated @ transition
        return propagated

    def forward(self, state: torch.Tensor, *, steps: int = 1) -> torch.Tensor:
        return self.propagate(self.membership(state), steps=steps)

    def decode_membership(self, membership: torch.Tensor) -> torch.Tensor:
        """Decode a probability distribution to its prototype expectation."""

        state = membership @ self.prototypes
        if self.state_dim == 3:
            state = normalize_circular_state(state)
        return state

    @torch.no_grad()
    def initialize_transition(self, transition: torch.Tensor) -> None:
        if transition.shape != (self.n_states, self.n_states):
            raise ValueError("transition has the wrong shape")
        if torch.any(transition < 0):
            raise ValueError("transition probabilities must be non-negative")
        row_sums = transition.sum(dim=-1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
            raise ValueError("transition rows must sum to one")
        self.raw_transition.copy_(transition.clamp_min(1e-8).log())

    @torch.no_grad()
    def initialize_prototypes(self, prototypes: torch.Tensor) -> None:
        if prototypes.shape != (self.n_states, self.state_dim):
            raise ValueError("prototypes have the wrong shape")
        self.prototypes.copy_(prototypes)
