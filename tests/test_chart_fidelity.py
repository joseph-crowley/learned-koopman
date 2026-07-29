from __future__ import annotations

import numpy as np

from learned_koopman.chart_fidelity import (
    ChartFidelityConfig,
    run_chart_fidelity_experiment,
)


def test_controlled_chart_fidelity_experiment_separates_resonance() -> None:
    result = run_chart_fidelity_experiment(
        ChartFidelityConfig(
            angle_samples=2048,
            chart_error_levels=(0.0, 0.02, 0.05, 0.1),
        )
    )

    assert result["comparison"]["passes_controlled_threefold_protection_gate"]
    resonant = result["resonant_probe"]["measurements"]
    off_resonant = result["off_resonant_probe"]["measurements"]
    assert max(row["relative_amplitude_error"] for row in resonant) < 1e-6
    assert off_resonant[-1]["relative_amplitude_error"] > 1.0
    assert np.isfinite(result["comparison"]["minimum_off_to_resonant_error_ratio"])
