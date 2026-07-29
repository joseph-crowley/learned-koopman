from __future__ import annotations

import copy
import json

import pytest

from learned_koopman.hj_action import (
    run_hj_action_audit,
    validate_hj_action_manifest,
)
from learned_koopman.trajectory import load_trajectory_csv, write_duffing_example


def test_hj_action_audit_measures_action_and_hj_identity(tmp_path) -> None:
    source = write_duffing_example(
        tmp_path / "duffing.csv",
        trajectories=18,
        steps=280,
        dt=0.03,
    )
    dataset = load_trajectory_csv(
        source,
        state_columns=("position", "velocity"),
        reference_column="energy",
    )
    output = tmp_path / "hj-action"

    result = run_hj_action_audit(dataset, output)

    json.dumps(result, allow_nan=False)
    assert result["certificate"]["status"] == "supported_on_supplied_periodic_orbits"
    assert result["hj_identity"]["available"]
    assert result["hj_identity"]["normalized_rmse"] < 0.08
    assert result["aggregate"]["max_closure_error"] < 0.02
    assert result["learned_coordinate_alignment"]["available"] is False
    assert validate_hj_action_manifest(result)
    assert (output / "manifest.json").is_file()
    assert (output / "overview.png").is_file()
    assert (output / "report.html").is_file()

    stale = copy.deepcopy(result)
    stale["certificate"]["decisive_comparisons"]["action_is_ordered"] = False
    with pytest.raises(ValueError, match="stale"):
        validate_hj_action_manifest(stale)


def test_hj_action_audit_abstains_without_complete_cycles(tmp_path) -> None:
    source = write_duffing_example(
        tmp_path / "short.csv",
        trajectories=8,
        steps=64,
        dt=0.03,
    )
    dataset = load_trajectory_csv(
        source,
        state_columns=("position", "velocity"),
        reference_column="energy",
    )

    with pytest.raises(ValueError, match="complete positive-maximum cycle"):
        run_hj_action_audit(dataset, tmp_path / "rejected")
