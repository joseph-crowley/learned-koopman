"""Fail when the README outruns the committed benchmark evidence."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/portfolio/metrics.json"
ROBUSTNESS = ROOT / "results/portfolio/robustness.json"
README = ROOT / "README.md"


def main() -> None:
    result = json.loads(RESULTS.read_text())
    robustness = json.loads(ROBUSTNESS.read_text())
    metrics = result["metrics"]["2.00"]
    counts = result["parameter_counts"]
    readme = README.read_text()

    rows = {
        "Persistence": ("persistence", counts["persistence"]),
        "Global DMD": ("dmd", counts["dmd"]),
        "Small-angle physics": ("small_angle", counts["small_angle"]),
        "Residual MLP": ("mlp", counts["mlp"]),
        "Fixed Koopman AE": ("fixed_koopman", counts["fixed_koopman"]),
        "**Energy-conditioned rotation**": (
            "energy_conditioned",
            counts["energy_conditioned"],
        ),
    }
    for label, (model, parameters) in rows.items():
        values = metrics[model]
        expected = (
            f"| {label} | "
            f"{'**' if label.startswith('**') else ''}{parameters:,}"
            f"{'**' if label.startswith('**') else ''} |"
        )
        assert expected in readme, f"README parameter count is stale for {model}"
        for value in (
            f"{float(values['valid_time']):.2f}",
            f"{float(values['angle_rmse']):.3f}",
            f"{float(values['max_energy_drift']):.3f}",
        ):
            assert value in readme, f"README metric {value} is missing for {model}"

    conditioned = metrics["energy_conditioned"]
    assert conditioned["valid_time"] > metrics["mlp"]["valid_time"]
    assert conditioned["valid_time"] > metrics["fixed_koopman"]["valid_time"]
    assert conditioned["angle_rmse"] < metrics["mlp"]["angle_rmse"]

    near_separatrix = result["metrics"]["3.05"]["energy_conditioned"]
    assert near_separatrix["valid_time"] < 1.0

    assert robustness["seeds"] == [7, 17, 29]
    aggregate = robustness["aggregate"]["2.00"]
    robust_rows = {
        "Residual MLP": "mlp",
        "Fixed Koopman AE": "fixed_koopman",
        "**Energy-conditioned rotation**": "energy_conditioned",
    }
    for label, model in robust_rows.items():
        line = next(
            line for line in readme.splitlines() if line.startswith(f"| {label} |") and "±" in line
        )
        values = aggregate[model]
        for metric, precision in (
            ("valid_time", 2),
            ("angle_rmse", 3),
            ("max_energy_drift", 3),
        ):
            mean = values[metric]["mean"]
            std = values[metric]["std"]
            expected = f"{mean:.{precision}f} ± {std:.{precision}f}"
            assert expected in line, f"README robustness value is stale for {model}/{metric}"

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

    print("Portfolio claims match the committed single-seed and robustness evidence.")


if __name__ == "__main__":
    main()
