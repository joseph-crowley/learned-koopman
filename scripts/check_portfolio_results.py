"""Validate the committed portfolio benchmark and robustness evidence."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/portfolio/metrics.json"
ROBUSTNESS = ROOT / "results/portfolio/robustness.json"


def main() -> None:
    result = json.loads(RESULTS.read_text())
    robustness = json.loads(ROBUSTNESS.read_text())
    metrics = result["metrics"]["2.00"]

    conditioned = metrics["energy_conditioned"]
    assert conditioned["valid_time"] > metrics["mlp"]["valid_time"]
    assert conditioned["valid_time"] > metrics["fixed_koopman"]["valid_time"]
    assert conditioned["angle_rmse"] < metrics["mlp"]["angle_rmse"]

    near_separatrix = result["metrics"]["3.05"]["energy_conditioned"]
    assert near_separatrix["valid_time"] < 1.0

    assert robustness["seeds"] == [7, 17, 29]
    aggregate = robustness["aggregate"]["2.00"]

    comparisons = robustness["comparisons"]
    assert comparisons["conditioned_valid_time_wins_over_mlp"] == 2
    assert comparisons["conditioned_angle_rmse_wins_over_mlp"] == 2
    assert (
        aggregate["energy_conditioned"]["valid_time"]["mean"]
        > aggregate["mlp"]["valid_time"]["mean"]
    )
    assert (
        aggregate["energy_conditioned"]["angle_rmse"]["mean"]
        < aggregate["mlp"]["angle_rmse"]["mean"]
    )
    for run in robustness["runs"].values():
        assert run["training_loss_final"]["energy_conditioned"] < 0.05
        assert run["metrics"]["2.00"]["energy_conditioned"]["valid_time"] > 2.0
        assert run["metrics"]["3.05"]["energy_conditioned"]["valid_time"] < 1.0

    print("Portfolio single-seed and robustness evidence is internally coherent.")


if __name__ == "__main__":
    main()
