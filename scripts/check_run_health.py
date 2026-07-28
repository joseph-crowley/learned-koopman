"""Reject a trained run when the structured coordinate fit has collapsed."""

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
    assert conditioned_loss < 0.05, (
        f"conditioned training collapsed: final loss {conditioned_loss:.4f}"
    )

    showcase = result["metrics"]["2.00"]["energy_conditioned"]
    assert float(showcase["valid_time"]) > 2.0, (
        f"conditioned rollout is unhealthy: valid time {showcase['valid_time']}"
    )

    for amplitude, models in result["metrics"].items():
        for model, metrics in models.items():
            for name in ("angle_rmse", "omega_rmse", "max_energy_drift", "valid_time"):
                value = float(metrics[name])
                assert math.isfinite(value), f"{amplitude}/{model}/{name} is not finite"

    print("Trained run passed the stability health check.")


if __name__ == "__main__":
    main()
