from __future__ import annotations

import csv

import numpy as np
import pytest

from learned_koopman.trajectory import load_trajectory_csv, write_duffing_example


def test_duffing_csv_round_trip_preserves_complete_trials(tmp_path) -> None:
    source = write_duffing_example(
        tmp_path / "duffing.csv",
        trajectories=8,
        steps=48,
        dt=0.03,
    )
    dataset = load_trajectory_csv(
        source,
        state_columns=("position", "velocity"),
        reference_column="energy",
    )

    assert dataset.states.shape == (8, 48, 2)
    assert dataset.times.shape == (8, 48)
    assert dataset.trajectory_ids[0] == "run-000"
    assert dataset.dt == pytest.approx(0.03)
    assert dataset.reference_values is not None
    assert dataset.reference_max_relative_drift is not None
    assert dataset.reference_max_relative_drift < 0.01
    assert np.all(np.diff(dataset.times, axis=1) > 0.0)


def test_csv_loader_rejects_irregular_sampling(tmp_path) -> None:
    source = tmp_path / "irregular.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("trajectory_id", "time", "q"))
        for trajectory in range(6):
            for step in range(32):
                time = 0.1 * step
                if trajectory == 2 and step == 14:
                    time += 0.04
                writer.writerow((trajectory, time, np.sin(time)))

    with pytest.raises(ValueError, match="uniformly sampled"):
        load_trajectory_csv(source, state_columns=("q",))
