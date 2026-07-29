"""Fail when the separatrix-atlas claims outrun committed evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from learned_koopman.route_validation import validate_route_truth

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/atlas/metrics.json"
ROBUSTNESS = ROOT / "results/atlas/robustness.json"
README = ROOT / "README.md"


def main() -> None:
    result = json.loads(RESULTS.read_text())
    robustness = json.loads(ROBUSTNESS.read_text())
    readme = README.read_text()

    assert robustness["seeds"] == [7, 17, 29, 41, 53]
    assert result["config"]["train_max_amplitude"] == 3.12
    assert result["config"]["summary_band_min_amplitude"] == 2.95
    assert result["config"]["evaluation_amplitudes"] == [
        0.25,
        1.0,
        2.0,
        2.8,
        2.95,
        3.05,
        3.1,
    ]

    config = result["config"]
    edges = np.linspace(
        config["train_min_amplitude"],
        config["train_max_amplitude"],
        config["train_amplitudes"] + 1,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    for amplitude in config["evaluation_amplitudes"]:
        assert not np.any(np.isclose(centers, amplitude)), (
            f"evaluation amplitude {amplitude} leaked into the training grid"
        )

    showcase = result["metrics"]["3.05"]
    atlas = showcase["separatrix_atlas"]
    assert atlas["valid_time"] > showcase["mlp"]["valid_time"]
    assert atlas["valid_time"] > showcase["energy_conditioned"]["valid_time"]
    assert atlas["valid_time"] > showcase["energy_projected_conditioned"]["valid_time"]
    assert atlas["valid_time"] > showcase["saddle_chart_only"]["valid_time"]
    assert atlas["max_energy_drift"] < 1e-5
    assert 0.0 < atlas["saddle_fraction"] < 1.0
    assert atlas["route_switches"] > 0
    assert atlas["mean_local_chart_residual"] < 0.02

    diagnostics = result["model_diagnostics"]["separatrix_atlas"]
    assert abs(diagnostics["saddle_operator_determinant"] - 1.0) < 1e-5
    assert diagnostics["router"] == "explicit geometric validity rule"

    checked_route_traces = 0
    maximum_observed_switches = 0
    for amplitude, model_metrics in result["metrics"].items():
        maximum_observed_switches = max(
            maximum_observed_switches,
            validate_route_truth(
                model_metrics["separatrix_atlas"],
                expected_steps=int(config["rollout_steps"]),
                label=f"representative amplitude {amplitude}",
            ),
        )
        checked_route_traces += 1

    for seed, run in robustness["runs"].items():
        run_steps = int(run["config"]["rollout_steps"])
        for amplitude, model_metrics in run["metrics"].items():
            maximum_observed_switches = max(
                maximum_observed_switches,
                validate_route_truth(
                    model_metrics["separatrix_atlas"],
                    expected_steps=run_steps,
                    label=f"seed {seed} amplitude {amplitude}",
                ),
            )
            checked_route_traces += 1

    band = robustness["high_energy_band"]
    assert band["amplitudes"] == [2.95, 3.05, 3.1]
    comparisons = band["comparisons"]
    assert comparisons["atlas_valid_time_wins_over_mlp"] >= 4
    assert comparisons["atlas_valid_time_wins_over_conditioned"] == 5
    aggregate = band["aggregate"]
    atlas_band = aggregate["separatrix_atlas"]
    mlp_band = aggregate["mlp"]
    conditioned_band = aggregate["energy_conditioned"]
    assert atlas_band["valid_time"]["mean"] > mlp_band["valid_time"]["mean"]
    assert atlas_band["valid_time"]["mean"] > conditioned_band["valid_time"]["mean"]
    assert atlas_band["valid_time"]["std"] < mlp_band["valid_time"]["std"]
    assert atlas_band["max_energy_drift"]["max"] < 1e-5

    for amplitude in ("0.25", "1.00", "2.00"):
        ordinary = robustness["aggregate"][amplitude]
        assert (
            ordinary["separatrix_atlas"]["valid_time"]
            == ordinary["energy_conditioned"]["valid_time"]
        )

    expected_claims = (
        f"${atlas_band['valid_time']['mean']:.2f}"
        f"\\pm{atlas_band['valid_time']['std']:.2f}$",
        f"${conditioned_band['valid_time']['mean']:.2f}"
        f"\\pm{conditioned_band['valid_time']['std']:.2f}$",
    )
    for claim in expected_claims:
        assert claim in readme, f"README atlas claim is stale or missing: {claim}"

    print(
        "Atlas claims and route truth match the committed evidence: "
        f"{checked_route_traces} traces checked, "
        f"maximum {maximum_observed_switches} switches."
    )


if __name__ == "__main__":
    main()
