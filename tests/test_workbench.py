from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from learned_koopman.trajectory import load_trajectory_csv, write_duffing_example
from learned_koopman.workbench import (
    WorkbenchConfig,
    load_mechanics_model,
    run_mechanics_workbench,
    validate_workbench_manifest,
)


def test_mechanics_workbench_runs_on_nonpendulum_trajectory_data(tmp_path) -> None:
    source = write_duffing_example(
        tmp_path / "duffing.csv",
        trajectories=18,
        steps=160,
        dt=0.03,
    )
    dataset = load_trajectory_csv(
        source,
        state_columns=("position", "velocity"),
        reference_column="energy",
    )
    config = WorkbenchConfig.quick(seed=7)
    config = WorkbenchConfig(**{**config.__dict__, "epochs": 90})
    output = tmp_path / "result"

    result = run_mechanics_workbench(dataset, output, config=config)
    json.dumps(result, allow_nan=False)
    assert validate_workbench_manifest(result)
    assert result["certificate"]["status"] == "supported_on_held_out_trajectories"
    errors = result["operator_family"]["held_out_errors"]
    assert (
        errors["fibered"]["normalized_rollout_rmse"]
        < errors["global_edmd"]["normalized_rollout_rmse"]
    )
    assert result["invariant"]["held_out_mean_normalized_drift"] < 0.1
    assert result["reference_evaluation"]["fit_uses_training_trajectories_only"]
    assert (output / "manifest.json").is_file()
    assert (output / "overview.png").is_file()
    assert (output / "report.html").is_file()
    assert (output / "model.pt").is_file()

    model = load_mechanics_model(output / "model.pt")
    prediction = model.rollout(np.array([1.1, 0.0]), steps=40)
    assert prediction.shape == (40, 2)
    assert np.isfinite(prediction).all()
    assert model.support_status(np.array([1.1, 0.0]))[0] == "supported"
    with pytest.raises(ValueError, match="steps must be positive"):
        model.rollout(np.array([1.1, 0.0]), steps=0)
    with pytest.raises(ValueError, match="2 values"):
        model.coordinate(np.array([1.1]))
    assert model.support_status(np.array([-10.0, 0.0]))[0] != "supported"
    with pytest.raises(ValueError, match="outside fitted support"):
        model.rollout(np.array([-10.0, 0.0]), steps=5)

    stale_manifest = copy.deepcopy(result)
    stale_manifest["certificate"]["decisive_comparisons"][
        "beats_global_edmd_rollout"
    ] = False
    with pytest.raises(ValueError, match="stale"):
        validate_workbench_manifest(stale_manifest)

    rejected_output = tmp_path / "rejected"
    rejected = run_mechanics_workbench(
        dataset,
        rejected_output,
        config=replace(config, family_degree=0),
    )
    assert rejected["certificate"]["status"] == "not_supported_by_current_dataset"
    rejected_model = load_mechanics_model(rejected_output / "model.pt")
    assert rejected_model.support_status(np.array([1.1, 0.0]))[0] == (
        "fit_not_certified"
    )
    with pytest.raises(ValueError, match="model fit is not certified"):
        rejected_model.rollout(np.array([1.1, 0.0]), steps=5)
