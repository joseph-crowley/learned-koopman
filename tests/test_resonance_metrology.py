from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from learned_koopman.canonical_model import CanonicalKoopmanNetwork
from learned_koopman.cli import _independent_model_digests
from learned_koopman.map_fixtures import (
    KickHarmonic,
    ObservationChart,
    TwistKickMap,
)
from learned_koopman.resonance_metrology import (
    MetrologyConfig,
    _analyze_coordinate_arrays,
    _circle_probe,
    _observed_frequency_initialization,
    _shuffled_angle_control,
    run_resonance_metrology,
    validate_resonance_manifest,
    weighted_birkhoff_mean,
)


def _twist_kick_trajectories() -> tuple[np.ndarray, np.ndarray]:
    order = 3
    base_frequency = 1.6
    twist = 0.3
    amplitude = 0.0075
    phase = 0.9
    generator = np.random.default_rng(11)
    actions = np.empty((48, 400))
    angles = np.empty_like(actions)
    actions[:, 0] = np.linspace(0.75, 2.55, 48)
    angles[:, 0] = generator.uniform(-np.pi, np.pi, 48)
    for step in range(1, actions.shape[1]):
        actions[:, step] = actions[:, step - 1] + amplitude * np.sin(
            order * angles[:, step - 1] + phase
        )
        angles[:, step] = (
            angles[:, step - 1]
            + base_frequency
            + twist * actions[:, step]
            + np.pi
        ) % (2.0 * np.pi) - np.pi
    return actions, angles


def test_weighted_birkhoff_and_band_estimator_recover_known_map() -> None:
    actions, angles = _twist_kick_trajectories()

    result = _analyze_coordinate_arrays(
        actions,
        angles,
        order=3,
        band=(0.7, 2.6),
        bins=14,
        max_order=8,
        reference_angle=angles,
    )
    truth = 0.0075 * np.exp(0.9j)

    assert weighted_birkhoff_mean(np.full(400, 0.2345)) == pytest.approx(
        0.2345,
        abs=1e-12,
    )
    assert result["verdict"] == "value"
    assert result["estimate"]["condition_number"] < 10.0
    assert result["estimate"]["coefficient"] == pytest.approx(
        truth,
        rel=1e-9,
        abs=1e-11,
    )


def test_observed_frequency_initializer_uses_no_oracle_and_escapes_unit_basin() -> None:
    actions, angles = _twist_kick_trajectories()
    radius = np.sqrt(2.0 * actions)
    states = np.stack(
        (radius * np.cos(angles), -radius * np.sin(angles)),
        axis=-1,
    )

    coefficients, diagnostics = _observed_frequency_initialization(
        states,
        np.arange(len(states)),
        degree=3,
    )

    assert diagnostics["uses_oracle_coordinates"] is False
    assert diagnostics["minimum_circular_concentration"] > 0.99
    assert diagnostics["orbit_fit_rmse_radians_per_step"] < 0.002
    assert coefficients[0] == pytest.approx(1.6, abs=0.01)
    assert coefficients[1] == pytest.approx(0.3, abs=0.01)
    assert abs(coefficients[0] - 1.0) > 0.5


def test_user_instrument_rejects_copied_chart_files(tmp_path) -> None:
    first = tmp_path / "first.pt"
    copied = tmp_path / "copied.pt"
    distinct = tmp_path / "distinct.pt"
    first.write_bytes(b"one fitted chart")
    copied.write_bytes(first.read_bytes())
    distinct.write_bytes(b"a different fitted chart")

    with pytest.raises(ValueError, match="distinct fitted charts"):
        _independent_model_digests([first, copied])
    assert len(set(_independent_model_digests([first, distinct]))) == 2


def test_circle_probe_and_within_bin_shuffle_are_independent_controls() -> None:
    actions, angles = _twist_kick_trajectories()

    network = CanonicalKoopmanNetwork(dt=1.0)
    network.hamiltonian.raw_base_frequency.data.fill_(
        float(np.log(np.expm1(1.6 - 1e-4)))
    )
    network.hamiltonian.higher_frequency_coefficients.data.copy_(
        torch.tensor((0.3, 0.0))
    )
    for parameter in network.canonical_map.parameters():
        parameter.data.zero_()
    system = TwistKickMap(
        1.6,
        0.3,
        (KickHarmonic(3, 0.0075, 0.9),),
    )
    circle = _circle_probe(
        network,
        system,
        ObservationChart(
            action_shears=(),
            angle_offset=0.0,
            angle_twist=0.0,
            linear_q_q=1.0,
            linear_q_p=0.0,
            linear_p_q=0.0,
        ),
        action=system.resonance_action(3),
        order=3,
        max_order=8,
    )
    shuffled = _shuffled_angle_control(
        actions,
        angles,
        order=3,
        band=(0.7, 2.6),
        bins=14,
        max_order=8,
        rng=np.random.default_rng(20260728),
    )

    assert circle["coefficient"] == pytest.approx(
        0.0075 * np.exp(0.9j),
        rel=2e-5,
    )
    assert circle["uses_oracle_for_phase_alignment"] is True
    assert shuffled["permutation"] == "current angles within fixed action bins"
    assert shuffled["median_bin_coefficient_magnitude"] < 0.2 * 0.0075


def test_ci_profile_is_real_but_cannot_emit_a_decisive_status(tmp_path) -> None:
    config = replace(
        MetrologyConfig.ci(tmp_path / "metrology"),
        epochs=2,
        steps=80,
    )

    result = run_resonance_metrology(config)
    result["_artifact_root"] = str(config.output)

    assert result["status"] in {
        "not_resolved_abstained",
        "invalid_ensemble",
    }
    assert result["status"] not in {"resolved_supported", "resolved_refuted"}
    assert validate_resonance_manifest(result)
    assert (config.output / "manifest.json").is_file()
    assert (config.output / "report.html").is_file()
    assert (config.output / "overview.png").is_file()

    stale = copy.deepcopy(result)
    stale["profile"] = "ci"
    stale["status"] = "resolved_supported"
    with pytest.raises(ValueError, match="non-full"):
        validate_resonance_manifest(stale)

    unsupported_refutation = copy.deepcopy(result)
    unsupported_refutation["profile"] = "full"
    unsupported_refutation["source_revision"]["git_worktree_clean"] = True
    unsupported_refutation["status"] = "resolved_refuted"
    unsupported_refutation["status_reason"] = "gauge_freedom"
    stress = unsupported_refutation["controls"]["exact_2m_gauge_stress"]
    stress["maximum_in_envelope_complex_shift"] = 0.2
    stress["maximum_in_envelope_magnitude_shift"] = 0.15
    with pytest.raises(ValueError, match="lacks a comparable 2x shift"):
        validate_resonance_manifest(unsupported_refutation)

    silent_variant = copy.deepcopy(result)
    silent_variant["empirical_gates"]["G9_variant_stability"].update(
        {
            "passed": True,
            "all_trigger_variants_evaluable": False,
        }
    )
    with pytest.raises(ValueError, match="unevaluable trigger"):
        validate_resonance_manifest(silent_variant)

    stale_trap = copy.deepcopy(result)
    first_check = next(
        iter(stale_trap["controls"]["wrong_harmonic_checks"].values())
    )
    first_check["passed"] = not all(
        row["passed"] for row in first_check["charts"]
    )
    with pytest.raises(ValueError, match="per-chart ledger"):
        validate_resonance_manifest(stale_trap)

    report = config.output / "report.html"
    report_payload = report.read_bytes()
    report.unlink()
    with pytest.raises(ValueError, match="report artifact is missing"):
        validate_resonance_manifest(result)
    report.write_bytes(report_payload)

    s1_data = config.output / "s1-trajectories.csv"
    s1_data.unlink()
    optional_checks = validate_resonance_manifest(result)
    assert any("source artifact is not shipped" in row for row in optional_checks)
    with pytest.raises(ValueError, match="s1_data artifact is missing"):
        validate_resonance_manifest(result, require_data_artifacts=True)
