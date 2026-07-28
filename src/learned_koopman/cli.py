from __future__ import annotations

import argparse
import json
from pathlib import Path

from learned_koopman.config import ExperimentConfig
from learned_koopman.experiment import run_experiment


def _summary(result: dict[str, object]) -> str:
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    showcase = metrics["2.00"]
    rows = []
    for name in [
        "persistence",
        "dmd",
        "small_angle",
        "mlp",
        "fixed_koopman",
        "energy_conditioned",
    ]:
        values = showcase[name]
        rows.append(
            {
                "model": name,
                "valid_time": values["valid_time"],
                "angle_rmse": values["angle_rmse"],
                "energy_drift": values["max_energy_drift"],
            }
        )
    return json.dumps(rows, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="learned-koopman",
        description="Reproducible structured-latent pendulum experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="Run the end-to-end demonstration.")
    demo.add_argument("--quick", action="store_true", help="Use the CI-sized experiment.")
    demo.add_argument("--output", type=Path, help="Override the result directory.")
    benchmark = subparsers.add_parser("benchmark", help="Run the portfolio benchmark.")
    benchmark.add_argument("--output", type=Path, default=Path("results/portfolio"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "demo" and args.quick:
        config = ExperimentConfig.quick(args.output or Path("results/quick"))
    else:
        output = args.output or Path("results/portfolio")
        config = ExperimentConfig(output_dir=output)
    print("Training and evaluating learned dynamics…")
    result = run_experiment(config)
    print(_summary(result))
    print(f"Figure: {config.output_dir / 'comparison.png'}")
    print(f"Metrics: {config.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
