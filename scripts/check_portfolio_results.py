"""Fail when the README outruns the committed benchmark evidence."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/portfolio/metrics.json"
README = ROOT / "README.md"


def main() -> None:
    result = json.loads(RESULTS.read_text())
    metrics = result["metrics"]["2.00"]
    counts = result["parameter_counts"]
    readme = README.read_text()

    rows = {
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
    print("Portfolio claims match the committed benchmark.")


if __name__ == "__main__":
    main()
