from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from learned_koopman.canonical_diagnostics import (
    diagnose_canonical_orbits,
    summarize_orbit_diagnostics,
)
from learned_koopman.canonical_model import load_canonical_model
from learned_koopman.trajectory import load_trajectory_csv

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "koopman-hj" / "orbit-diagnostics.json"


def main() -> None:
    committed = json.loads(RESULT.read_text(encoding="utf-8"))
    model = load_canonical_model(ROOT / committed["model"])
    dataset = load_trajectory_csv(
        ROOT / committed["input"],
        state_columns=tuple(committed["state_columns"]),
    )
    rebuilt = summarize_orbit_diagnostics(
        diagnose_canonical_orbits(model.network, dataset.states)
    )

    assert committed["schema_version"] == 1
    assert committed["claim_boundary"].endswith(
        "They are not a KAM or physical-system certificate."
    )
    assert committed["diagnostics"]["trajectory_count"] == dataset.trajectory_count
    for name in (
        "mean_radial_coefficient_of_variation",
        "mean_phase_step_coefficient_of_variation",
        "mean_phase_law_rmse_radians",
        "mean_normalized_conjugacy_rmse",
    ):
        np.testing.assert_allclose(
            committed["diagnostics"][name],
            rebuilt[name],
            rtol=1e-7,
            atol=1e-10,
        )
    print(
        "canonical orbit diagnostics reproduce from the committed model and CSV; "
        f"{rebuilt['supported_trajectory_count']}/{rebuilt['trajectory_count']} "
        "complete trajectories pass the empirical residual gate"
    )


if __name__ == "__main__":
    main()
