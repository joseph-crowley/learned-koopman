from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from learned_koopman import __version__
from learned_koopman.config import ExperimentConfig
from learned_koopman.control_experiment import (
    ControlExperimentProfile,
    run_control_experiment,
)
from learned_koopman.experiment import run_experiment, run_robustness_sweep
from learned_koopman.invariant_experiment import run_invariant_experiment
from learned_koopman.research_lab import run_research_lab
from learned_koopman.transfer_experiment import run_transfer_experiment


def _write_json(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _summary(result: dict[str, object]) -> str:
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    config = result["config"]
    assert isinstance(config, dict)
    showcase = metrics[f"{float(config['showcase_amplitude']):.2f}"]
    rows = []
    for name in showcase:
        if name == "reference":
            continue
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
        description="PyTorch experiments in learned nonlinear-dynamics structure.",
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
    atlas = subparsers.add_parser(
        "atlas",
        help="Run the two-chart near-separatrix experiment.",
    )
    atlas.add_argument("--output", type=Path)
    atlas.add_argument("--seed", type=int, default=7, help="Training seed (default: 7).")
    atlas.add_argument("--quick", action="store_true", help="Use a shorter evaluation rollout.")
    atlas_robustness = subparsers.add_parser(
        "atlas-robustness",
        help="Run the atlas experiment across five independent seeds.",
    )
    atlas_robustness.add_argument("--output", type=Path)
    atlas_robustness.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[7, 17, 29, 41, 53],
    )
    atlas_robustness.add_argument(
        "--quick",
        action="store_true",
        help="Use shorter evaluation rollouts.",
    )
    lab = subparsers.add_parser(
        "lab",
        aliases=["research-lab"],
        help="Run the connected atlas, invariant, transfer, and control experiments.",
    )
    lab.add_argument("--output", type=Path)
    lab.add_argument("--seed", type=int, default=7, help="Primary seed (default: 7).")
    lab.add_argument("--quick", action="store_true", help="Use CPU-friendly profiles.")
    invariant = subparsers.add_parser(
        "invariant",
        help="Discover a scalar invariant without physical-energy labels.",
    )
    invariant.add_argument("--output", type=Path, default=Path("results/invariant"))
    invariant.add_argument("--seeds", type=int, nargs="+")
    invariant.add_argument("--quick", action="store_true", help="Use the two-seed profile.")
    transfer = subparsers.add_parser(
        "transfer",
        help="Learn a mass-preserving stochastic simplex transfer operator.",
    )
    transfer.add_argument("--output", type=Path, default=Path("results/transfer"))
    transfer.add_argument("--seed", type=int, default=7)
    transfer.add_argument("--quick", action="store_true")
    control = subparsers.add_parser(
        "control",
        help="Learn torque-conditioned prediction across separatrix crossings.",
    )
    control.add_argument("--output", type=Path, default=Path("results/control"))
    control.add_argument("--seed", type=int, default=7)
    control.add_argument("--quick", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command in {"lab", "research-lab"}:
        output = args.output or (
            Path("results/research-lab-quick")
            if args.quick
            else Path("results/research-lab")
        )
        print("Running the four-experiment nonlinear-dynamics lab…")
        result = run_research_lab(
            output,
            quick=args.quick,
            seed=args.seed,
        )
        print(json.dumps(result["summary"], indent=2))
        print(f"Overview: {output / 'overview.png'}")
        print(f"Manifest: {output / 'manifest.json'}")
        return
    if args.command == "invariant":
        profile = "quick" if args.quick else "full"
        print("Learning an invariant from trajectory membership without energy labels…")
        result = run_invariant_experiment(
            profile=profile,
            seeds=tuple(args.seeds) if args.seeds else None,
        )
        target = args.output / "metrics.json"
        _write_json(target, result)
        print(json.dumps(result["aggregate"], indent=2))
        print(f"Metrics: {target}")
        return
    if args.command == "transfer":
        print("Learning a positive, row-stochastic transfer operator…")
        result = run_transfer_experiment(
            quick=args.quick,
            seed=args.seed,
            output_dir=args.output,
        )
        print(json.dumps(result["held_out"], indent=2))
        print(f"Metrics: {args.output / 'transfer_metrics.json'}")
        return
    if args.command == "control":
        print("Identifying actuator gain through real controlled crossings…")
        profile = (
            ControlExperimentProfile.quick(args.seed)
            if args.quick
            else ControlExperimentProfile.full(args.seed)
        )
        result = run_control_experiment(profile)
        target = args.output / "metrics.json"
        _write_json(target, result)
        print(json.dumps(result["evaluation"], indent=2))
        print(f"Metrics: {target}")
        return
    if args.command in {"robustness", "atlas-robustness"}:
        include_atlas = args.command == "atlas-robustness"
        output = args.output or (
            Path("results/atlas-robustness-quick")
            if include_atlas and args.quick
            else Path("results/atlas")
            if include_atlas
            else Path("results/robustness")
        )
        config = (
            (
                ExperimentConfig.quick_atlas(output_dir=output)
                if include_atlas
                else ExperimentConfig.quick(output_dir=output)
            )
            if args.quick
            else (
                ExperimentConfig.atlas(output_dir=output)
                if include_atlas
                else ExperimentConfig(output_dir=output)
            )
        )
        print(f"Training and evaluating {len(args.seeds)} independent seeds…")
        result = run_robustness_sweep(
            config,
            args.seeds,
            include_atlas=include_atlas,
        )
        aggregate = result["aggregate"][f"{config.showcase_amplitude:.2f}"]  # type: ignore[index]
        promoted_models = {
            "mlp",
            "fixed_koopman",
            "energy_conditioned",
            "separatrix_atlas",
        }
        summary = {
            name: {
                metric: values[metric]
                for metric in ("valid_time", "angle_rmse", "max_energy_drift")
            }
            for name, values in aggregate.items()
            if name in promoted_models
        }
        print(json.dumps(summary, indent=2))
        print(f"Robustness metrics: {config.output_dir / 'robustness.json'}")
        return
    if args.command == "atlas":
        output = args.output or (
            Path("results/atlas-quick") if args.quick else Path("results/atlas")
        )
        config = ExperimentConfig.atlas(output_dir=output)
        if args.quick:
            config = replace(config, rollout_steps=500)
        config = replace(config, seed=args.seed)
        print("Training and evaluating the separatrix atlas…")
        result = run_experiment(config, include_atlas=True)
        print(_summary(result))
        print(f"Figure: {config.output_dir / 'comparison.png'}")
        print(f"Metrics: {config.output_dir / 'metrics.json'}")
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
