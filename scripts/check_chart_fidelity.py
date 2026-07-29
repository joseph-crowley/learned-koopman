from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from learned_koopman.chart_fidelity import (
    ChartFidelityConfig,
    run_chart_fidelity_experiment,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "chart-fidelity.json"


def main() -> None:
    committed = json.loads(RESULT.read_text(encoding="utf-8"))
    config = dict(committed["config"])
    config["chart_error_levels"] = tuple(config["chart_error_levels"])
    rebuilt = run_chart_fidelity_experiment(ChartFidelityConfig(**config))

    assert committed["schema_version"] == 1
    assert committed["experiment"] == "oracle_chart_pipeline_regression"
    assert committed["comparison"]["oracle_pipeline_regression_gate"]
    assert committed["claim_boundary"]["next_falsifier"].startswith(
        "Repeat with learned chart ensembles"
    )

    for probe in ("resonant_probe", "off_resonant_probe"):
        for actual, expected in zip(
            committed[probe]["measurements"],
            rebuilt[probe]["measurements"],
            strict=True,
        ):
            np.testing.assert_allclose(
                actual["recovered_generating_function_amplitude"],
                expected["recovered_generating_function_amplitude"],
                rtol=1e-12,
                atol=1e-14,
            )
            np.testing.assert_allclose(
                actual["relative_amplitude_error"],
                expected["relative_amplitude_error"],
                rtol=1e-10,
                atol=1e-12,
            )
    print(
        "oracle chart-pipeline regression is reproducible; learned-chart "
        "identifiability is evaluated separately by resonance-metrology"
    )


if __name__ == "__main__":
    main()
