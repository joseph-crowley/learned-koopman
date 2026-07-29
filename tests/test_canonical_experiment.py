from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from learned_koopman.canonical_experiment import (
    CanonicalExperimentConfig,
    run_canonical_experiment,
    validate_canonical_manifest,
)
from learned_koopman.canonical_model import load_canonical_model
from learned_koopman.trajectory import load_trajectory_csv, write_duffing_example


def test_canonical_experiment_learns_a_certified_symplectic_model(tmp_path) -> None:
    source = write_duffing_example(
        tmp_path / "duffing.csv",
        trajectories=18,
        steps=220,
        dt=0.03,
    )
    dataset = load_trajectory_csv(
        source,
        state_columns=("position", "velocity"),
        reference_column="energy",
    )
    config = replace(
        CanonicalExperimentConfig.quick(seed=7),
        epochs=80,
        batch_size=256,
        rollout_horizon=5,
    )
    output = tmp_path / "canonical"

    result = run_canonical_experiment(dataset, output, config=config)

    json.dumps(result, allow_nan=False)
    assert result["certificate"]["status"] == "supported_on_held_out_trajectories"
    assert result["held_out_evaluation"]["normalized_rollout_rmse"] < (
        result["held_out_evaluation"]["persistence_normalized_rollout_rmse"]
    )
    assert result["structure_evaluation"]["maximum_symplectic_defect"] < 2e-5
    assert result["canonical_action_evaluation"]["affine_r2"] > 0.99
    assert result["learned_hamiltonian_evaluation"]["frequency_normalized_rmse"] < 0.05
    assert validate_canonical_manifest(result)
    for relative in (
        "manifest.json",
        "model.pt",
        "overview.png",
        "report.html",
        "action-audit/manifest.json",
    ):
        assert (output / relative).is_file()

    model = load_canonical_model(output / "model.pt")
    prediction = model.rollout(np.array([1.1, 0.0]), steps=80)
    assert prediction.shape == (80, 2)
    assert np.isfinite(prediction).all()
    assert model.support_status(np.array([1.1, 0.0]))[0] == "supported"

    stale = copy.deepcopy(result)
    stale["certificate"]["decisive_comparisons"][
        "map_is_numerically_symplectic"
    ] = False
    with pytest.raises(ValueError, match="stale"):
        validate_canonical_manifest(stale)
