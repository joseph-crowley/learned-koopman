from __future__ import annotations

import numpy as np
import torch

from learned_koopman.canonical_model import CanonicalKoopmanNetwork


def test_canonical_network_is_invertible_action_preserving_and_symplectic() -> None:
    torch.manual_seed(7)
    model = CanonicalKoopmanNetwork(
        dt=0.03,
        hidden_dim=12,
        shear_layers=4,
        hamiltonian_degree=3,
    )
    states = torch.tensor(
        [[0.4, -0.8], [1.1, 0.2], [-0.7, 0.5]],
        dtype=torch.float32,
    )

    reconstructed = model.decode(model.encode(states))
    assert torch.max(torch.abs(reconstructed - states)).item() < 1e-6

    rollout = model.rollout(states, steps=40)
    actions = model.action(rollout)
    assert torch.max(torch.abs(actions - actions[:, :1])).item() < 2e-6

    point = states[0].detach().requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(
        lambda value: model(value.unsqueeze(0))[0],
        point,
    )
    symplectic_form = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
    defect = jacobian.T @ symplectic_form @ jacobian - symplectic_form
    assert torch.max(torch.abs(defect)).item() < 2e-5

    latent = model.encode(states).detach().numpy()
    action = model.action(states).detach().numpy()
    np.testing.assert_allclose(action, 0.5 * np.square(latent).sum(axis=-1))
