from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from learned_koopman import __version__
from learned_koopman.config import ExperimentConfig
from learned_koopman.experiment import run_experiment, run_robustness_sweep


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
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="Run the end-to-end demonstration.")
    demo.add_argument("--quick", action="store_true", help="Use the CI-sized experiment.")
    demo.add_argument("--output", type=Path, help="Override the result directory.")
    demo.add_argument("--seed", type=int, default=7, help="Training seed (default: 7).")
    benchmark = subparsers.add_parser("benchmark", help="Run the portfolio benchmark.")
    benchmark.add_argument("--output", type=Path, default=Path("results/portfolio"))
    benchmark.add_argument("--seed", type=int, default=7, help="Training seed (default: 7).")
    robustness = subparsers.add_parser(
        "robustness",
        help="Run a multi-seed robustness sweep.",
    )
    robustness.add_argument("--output", type=Path, default=Path("results/robustness"))
    robustness.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    robustness.add_argument("--quick", action="store_true", help="Use shorter evaluation rollouts.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "robustness":
        config = (
            ExperimentConfig.quick(output_dir=args.output)
            if args.quick
            else ExperimentConfig(output_dir=args.output)
        )
        print(f"Training and evaluating {len(args.seeds)} independent seeds…")
        result = run_robustness_sweep(config, args.seeds)
        aggregate = result["aggregate"]["2.00"]  # type: ignore[index]
        summary = {
            name: {
                metric: values[metric]
                for metric in ("valid_time", "angle_rmse", "max_energy_drift")
            }
            for name, values in aggregate.items()
            if name in {"mlp", "fixed_koopman", "energy_conditioned"}
        }
        print(json.dumps(summary, indent=2))
        print(f"Robustness metrics: {config.output_dir / 'robustness.json'}")
        return
    if args.command == "demo" and args.quick:
        base = ExperimentConfig.quick(args.output or Path("results/quick"))
        config = replace(base, seed=args.seed)
    else:
        output = args.output or Path("results/portfolio")
        config = ExperimentConfig(seed=args.seed, output_dir=output)
    print("Training and evaluating learned dynamics…")
    result = run_experiment(config)
    print(_summary(result))
    print(f"Figure: {config.output_dir / 'comparison.png'}")
    print(f"Metrics: {config.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
