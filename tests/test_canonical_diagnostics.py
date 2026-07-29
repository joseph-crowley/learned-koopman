from __future__ import annotations

import numpy as np
import torch

from learned_koopman.canonical_diagnostics import (
    diagnose_canonical_orbits,
    fit_residual_spectrum,
    summarize_orbit_diagnostics,
)
from learned_koopman.canonical_model import CanonicalKoopmanNetwork


def test_residual_spectrum_recovers_generating_function_convention() -> None:
    angle = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
    delta_action = 0.3 * np.sin(angle - 0.2) + 0.12 * np.sin(3.0 * angle + 0.4)

    spectrum = fit_residual_spectrum(angle, delta_action, max_order=4)

    assert spectrum.r2 > 0.999999999
    np.testing.assert_allclose(
        spectrum.harmonics[0].generating_function_amplitude,
        0.3,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        spectrum.harmonics[2].generating_function_amplitude,
        0.04,
        rtol=1e-10,
    )
    np.testing.assert_allclose(spectrum.harmonics[0].phase_radians, -0.2, atol=1e-10)
    np.testing.assert_allclose(spectrum.harmonics[2].phase_radians, 0.4, atol=1e-10)


def test_orbit_diagnostics_separate_geometry_phase_and_conjugacy() -> None:
    torch.manual_seed(7)
    network = CanonicalKoopmanNetwork(
        dt=0.03,
        hidden_dim=12,
        shear_layers=4,
        hamiltonian_degree=3,
    )
    initial = torch.tensor([[0.8, 0.1], [1.2, -0.2]], dtype=torch.float32)
    trajectories = network.rollout(initial, steps=180)

    rows = diagnose_canonical_orbits(network, trajectories)
    summary = summarize_orbit_diagnostics(rows)

    assert summary["supported_trajectory_count"] == 2
    assert summary["maximum_radial_coefficient_of_variation"] < 1e-5
    assert summary["maximum_phase_step_coefficient_of_variation"] < 1e-4
    assert summary["maximum_normalized_conjugacy_rmse"] < 1e-5

    corrupted = trajectories.detach().clone()
    scale = torch.linspace(0.75, 1.25, corrupted.shape[1])
    corrupted[0] = corrupted[0] * scale[:, None]
    bad = diagnose_canonical_orbits(network, corrupted)
    assert bad[0].radial_coefficient_of_variation > 0.08
    assert bad[0].verdict == "chart_residual_exceeds_threshold"
