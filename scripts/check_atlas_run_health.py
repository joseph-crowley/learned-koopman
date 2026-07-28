"""Reject a trained atlas run when its structured mechanism has collapsed."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    args = parser.parse_args()
    result = json.loads(args.metrics.read_text())

    conditioned_loss = float(result["training_loss_final"]["energy_conditioned"])
    saddle_loss = float(result["training_loss_final"]["atlas_saddle"])
    assert conditioned_loss < 0.05, (
        f"conditioned coordinate fit collapsed: final loss {conditioned_loss:.4f}"
    )
    assert saddle_loss < 1e-5, f"local saddle fit collapsed: final loss {saddle_loss:.4g}"

    diagnostics = result["model_diagnostics"]["separatrix_atlas"]
    determinant = float(diagnostics["saddle_operator_determinant"])
    assert abs(determinant - 1.0) < 1e-5, (
        f"saddle operator lost symplectic determinant: {determinant}"
    )

    showcase = result["metrics"]["3.05"]
    atlas = showcase["separatrix_atlas"]
    assert float(atlas["valid_time"]) > float(showcase["energy_conditioned"]["valid_time"])
    assert float(atlas["valid_time"]) > float(showcase["saddle_chart_only"]["valid_time"])
    assert float(atlas["max_energy_drift"]) < 1e-5
    assert int(atlas["route_switches"]) > 0

    for amplitude, models in result["metrics"].items():
        for model, metrics in models.items():
            for name in ("angle_rmse", "omega_rmse", "max_energy_drift", "valid_time"):
                value = float(metrics[name])
                assert math.isfinite(value), f"{amplitude}/{model}/{name} is not finite"

    print("Trained atlas run passed the mechanism and stability health checks.")


if __name__ == "__main__":
    main()
