from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from learned_koopman.resonance_metrology import (
    MetrologyConfig,
    _analyze_coordinate_arrays,
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
