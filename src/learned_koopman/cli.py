from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from learned_koopman import __version__
from learned_koopman.canonical_diagnostics import (
    diagnose_canonical_orbits,
    summarize_orbit_diagnostics,
)
from learned_koopman.canonical_experiment import (
    CanonicalExperimentConfig,
    run_canonical_experiment,
)
from learned_koopman.canonical_model import load_canonical_model
from learned_koopman.chart_fidelity import (
    ChartFidelityConfig,
    run_chart_fidelity_experiment,
)
from learned_koopman.config import ExperimentConfig
from learned_koopman.control_experiment import (
    ControlExperimentProfile,
    run_control_experiment,
)
from learned_koopman.experiment import run_experiment, run_robustness_sweep
from learned_koopman.hj_action import run_hj_action_audit
from learned_koopman.invariant_experiment import run_invariant_experiment
from learned_koopman.research_lab import run_research_lab
from learned_koopman.resonance_metrology import (
    MetrologyConfig,
    run_resonance_metrology,
)
from learned_koopman.trajectory import load_trajectory_csv, write_duffing_example
from learned_koopman.transfer_experiment import run_transfer_experiment
from learned_koopman.workbench import (
    WorkbenchConfig,
    load_mechanics_model,
    run_mechanics_workbench,
)


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


def _load_coordinate_model(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "hamiltonian_degree" in payload and "state_dict" in payload:
        return load_canonical_model(path)
    if "invariant_state_dict" in payload and "operator" in payload:
        return load_mechanics_model(path)
    raise ValueError("unsupported coordinate-model bundle")


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
        help="Identify actuator gain across torque-driven separatrix crossings.",
    )
    control.add_argument("--output", type=Path, default=Path("results/control"))
    control.add_argument("--seed", type=int, default=7)
    control.add_argument("--quick", action="store_true")
    example = subparsers.add_parser(
        "generate-example",
        help="Write conservative Duffing-oscillator trajectories as CSV.",
    )
    example.add_argument(
        "--output",
        type=Path,
        default=Path("examples/duffing-trajectories.csv"),
    )
    example.add_argument("--trajectories", type=int, default=30)
    example.add_argument("--steps", type=int, default=360)
    example.add_argument("--dt", type=float, default=0.025)
    analyze = subparsers.add_parser(
        "analyze",
        help="Discover an invariant and fit a fibered Koopman model to trajectory CSV.",
    )
    analyze.add_argument("input", type=Path)
    analyze.add_argument("--state-columns", nargs="+", required=True)
    analyze.add_argument("--trajectory-column", default="trajectory_id")
    analyze.add_argument("--time-column", default="time")
    analyze.add_argument("--reference-column")
    analyze.add_argument(
        "--output",
        type=Path,
        default=Path("results/mechanics-workbench"),
    )
    analyze.add_argument("--seed", type=int, default=7)
    analyze.add_argument("--quick", action="store_true")
    analyze.add_argument("--epochs", type=int)
    analyze.add_argument("--family-degree", type=int, choices=(0, 1, 2, 3))
    analyze.add_argument("--observable-degree", type=int, choices=(1, 2))
    hj_audit = subparsers.add_parser(
        "hj-audit",
        aliases=["canonical-audit"],
        help="Measure canonical action and test the Koopman/Hamilton-Jacobi bridge.",
    )
    hj_audit.add_argument("input", type=Path)
    hj_audit.add_argument("--position-column", required=True)
    hj_audit.add_argument("--momentum-column", required=True)
    hj_audit.add_argument("--trajectory-column", default="trajectory_id")
    hj_audit.add_argument("--time-column", default="time")
    hj_audit.add_argument("--reference-column")
    hj_audit.add_argument(
        "--model",
        type=Path,
        help="Optional workbench or canonical model for coordinate calibration.",
    )
    hj_audit.add_argument("--output", type=Path, default=Path("results/hj-action"))
    canonical = subparsers.add_parser(
        "canonical-train",
        aliases=["koopman-hj"],
        help="Train an exact-symplectic canonical Koopman world model.",
    )
    canonical.add_argument("input", type=Path)
    canonical.add_argument("--position-column", required=True)
    canonical.add_argument("--momentum-column", required=True)
    canonical.add_argument("--trajectory-column", default="trajectory_id")
    canonical.add_argument("--time-column", default="time")
    canonical.add_argument("--reference-column")
    canonical.add_argument("--output", type=Path, default=Path("results/koopman-hj"))
    canonical.add_argument("--seed", type=int, default=7)
    canonical.add_argument("--quick", action="store_true")
    canonical.add_argument("--epochs", type=int)
    canonical_predict = subparsers.add_parser(
        "canonical-predict",
        help="Roll out a saved exact-symplectic canonical Koopman model.",
    )
    canonical_predict.add_argument("model", type=Path)
    canonical_predict.add_argument("--initial", type=float, nargs=2, required=True)
    canonical_predict.add_argument("--steps", type=int, default=200)
    canonical_predict.add_argument("--allow-unsupported", action="store_true")
    canonical_predict.add_argument(
        "--output",
        type=Path,
        default=Path("results/canonical-prediction.csv"),
    )
    canonical_diagnose = subparsers.add_parser(
        "canonical-diagnose",
        help="Test complete trajectories against a saved canonical chart.",
    )
    canonical_diagnose.add_argument("model", type=Path)
    canonical_diagnose.add_argument("input", type=Path)
    canonical_diagnose.add_argument("--position-column", required=True)
    canonical_diagnose.add_argument("--momentum-column", required=True)
    canonical_diagnose.add_argument("--trajectory-column", default="trajectory_id")
    canonical_diagnose.add_argument("--time-column", default="time")
    canonical_diagnose.add_argument(
        "--output",
        type=Path,
        default=Path("results/canonical-diagnostics.json"),
    )
    chart_fidelity = subparsers.add_parser(
        "chart-fidelity",
        help="Run the closed-form oracle chart-pipeline regression.",
    )
    chart_fidelity.add_argument(
        "--output",
        type=Path,
        default=Path("results/chart-fidelity.json"),
    )
    chart_fidelity.add_argument("--angle-samples", type=int, default=8192)
    chart_fidelity.add_argument("--harmonic-order", type=int, default=4)
    resonance = subparsers.add_parser(
        "resonance-metrology",
        help="Measure a resonant normal-form block across learned canonical charts.",
    )
    resonance.add_argument(
        "--output",
        type=Path,
        default=Path("results/resonance-metrology"),
    )
    resonance.add_argument(
        "--profile",
        choices=("ci", "full"),
        default="ci",
        help="Use the non-decisive CI smoke or the frozen full experiment.",
    )
    resonance.add_argument(
        "--epochs",
        type=int,
        help="Override training epochs for implementation smokes only.",
    )
    predict = subparsers.add_parser(
        "predict",
        help="Roll out a saved mechanics-workbench model.",
    )
    predict.add_argument("model", type=Path)
    predict.add_argument("--initial", type=float, nargs="+", required=True)
    predict.add_argument("--steps", type=int, default=200)
    predict.add_argument(
        "--allow-unsupported",
        "--allow-extrapolation",
        dest="allow_unsupported",
        action="store_true",
        help="Override a negative fit certificate or out-of-support initial state.",
    )
    predict.add_argument(
        "--output",
        type=Path,
        default=Path("results/mechanics-prediction.csv"),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "generate-example":
        target = write_duffing_example(
            args.output,
            trajectories=args.trajectories,
            steps=args.steps,
            dt=args.dt,
        )
        print(f"Duffing trajectories: {target}")
        print(
            "Next: learned-koopman analyze "
            f"{target} --state-columns position velocity --reference-column energy --quick"
        )
        return
    if args.command == "analyze":
        if args.epochs is not None and args.epochs < 1:
            parser.error("--epochs must be positive")
        dataset = load_trajectory_csv(
            args.input,
            state_columns=tuple(args.state_columns),
            trajectory_column=args.trajectory_column,
            time_column=args.time_column,
            reference_column=args.reference_column,
        )
        config = (
            WorkbenchConfig.quick(args.seed)
            if args.quick
            else WorkbenchConfig.full(args.seed)
        )
        updates = {
            key: value
            for key, value in (
                ("epochs", args.epochs),
                ("family_degree", args.family_degree),
                ("observable_degree", args.observable_degree),
            )
            if value is not None
        }
        if updates:
            config = replace(config, **updates)
        print(
            f"Analyzing {dataset.trajectory_count} complete trajectories "
            f"with {dataset.state_dim} state variables…"
        )
        result = run_mechanics_workbench(dataset, args.output, config=config)
        errors = result["operator_family"]["held_out_errors"]
        print(f"Model status: {result['certificate']['status']}")
        print(
            "Held-out rollout RMSE: "
            f"fibered {errors['fibered']['normalized_rollout_rmse']:.5f}, "
            f"global EDMD {errors['global_edmd']['normalized_rollout_rmse']:.5f}"
        )
        print(f"Report: {args.output / 'report.html'}")
        print(f"Model: {args.output / 'model.pt'}")
        return
    if args.command in {"hj-audit", "canonical-audit"}:
        dataset = load_trajectory_csv(
            args.input,
            state_columns=(args.position_column, args.momentum_column),
            trajectory_column=args.trajectory_column,
            time_column=args.time_column,
            reference_column=args.reference_column,
        )
        model = _load_coordinate_model(args.model) if args.model else None
        print(
            f"Measuring canonical action on {dataset.trajectory_count} trajectories…"
        )
        result = run_hj_action_audit(dataset, args.output, model=model)
        hj_identity = result["hj_identity"]
        print(f"Audit status: {result['certificate']['status']}")
        if hj_identity["available"]:
            print(
                "HJ identity dH/dJ = omega: "
                f"{hj_identity['normalized_rmse']:.3%} normalized RMSE"
            )
        alignment = result["learned_coordinate_alignment"]
        if alignment["available"]:
            print(
                "Model coordinate -> action (held-out monotone calibration): "
                f"R² {alignment['calibration']['held_out_r2']:.4f}, "
                f"|rank correlation| {alignment['absolute_rank_correlation']:.4f}"
            )
        print(f"Report: {args.output / 'report.html'}")
        print(f"Manifest: {args.output / 'manifest.json'}")
        return
    if args.command in {"canonical-train", "koopman-hj"}:
        if args.epochs is not None and args.epochs < 1:
            parser.error("--epochs must be positive")
        dataset = load_trajectory_csv(
            args.input,
            state_columns=(args.position_column, args.momentum_column),
            trajectory_column=args.trajectory_column,
            time_column=args.time_column,
            reference_column=args.reference_column,
        )
        config = (
            CanonicalExperimentConfig.quick(args.seed)
            if args.quick
            else CanonicalExperimentConfig.full(args.seed)
        )
        if args.epochs is not None:
            config = replace(config, epochs=args.epochs)
        print(
            "Training an exact-symplectic conjugacy to Hamiltonian latent "
            "rotation…"
        )
        result = run_canonical_experiment(dataset, args.output, config=config)
        evaluation = result["held_out_evaluation"]
        structure = result["structure_evaluation"]
        print(f"Model status: {result['certificate']['status']}")
        print(
            "Held-out recursive rollout RMSE: "
            f"{evaluation['normalized_rollout_rmse']:.5f}"
        )
        print(
            "Symplectic defect: "
            f"{structure['maximum_symplectic_defect']:.3e}; "
            "observed Koopman residual: "
            f"{structure['held_out_mean_koopman_eigenfunction_residual']:.5f}"
        )
        print(f"Report: {args.output / 'report.html'}")
        print(f"Model: {args.output / 'model.pt'}")
        return
    if args.command == "canonical-predict":
        if args.steps < 1:
            parser.error("--steps must be positive")
        model = load_canonical_model(args.model)
        initial = np.asarray(args.initial, dtype=np.float64)
        support = str(model.support_status(initial)[0])
        if support != "supported" and not args.allow_unsupported:
            parser.error(
                f"prediction is unsupported ({support}); "
                "pass --allow-unsupported to override"
            )
        prediction = model.rollout(
            initial,
            steps=args.steps,
            allow_extrapolation=args.allow_unsupported,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("time", *model.state_columns))
            for step, state in enumerate(prediction):
                writer.writerow((step * model.network.dt, *state))
        print(f"Prediction support: {support}")
        print(f"Prediction: {args.output}")
        return
    if args.command == "canonical-diagnose":
        model = load_canonical_model(args.model)
        dataset = load_trajectory_csv(
            args.input,
            state_columns=(args.position_column, args.momentum_column),
            trajectory_column=args.trajectory_column,
            time_column=args.time_column,
        )
        rows = diagnose_canonical_orbits(model.network, dataset.states)
        result = {
            "schema_version": 1,
            "model": str(args.model),
            "input": str(args.input),
            "state_columns": [args.position_column, args.momentum_column],
            "thresholds": {
                "radial_coefficient_of_variation": 0.08,
                "phase_step_coefficient_of_variation": 0.08,
                "normalized_conjugacy_rmse": 0.08,
            },
            "diagnostics": summarize_orbit_diagnostics(rows),
            "claim_boundary": (
                "These are empirical complete-orbit support checks. "
                "They are not a KAM or physical-system certificate."
            ),
        }
        _write_json(args.output, result)
        print(f"Canonical diagnostics: {args.output}")
        return
    if args.command == "chart-fidelity":
        result = run_chart_fidelity_experiment(
            ChartFidelityConfig(
                angle_samples=args.angle_samples,
                harmonic_order=args.harmonic_order,
            )
        )
        _write_json(args.output, result)
        print(f"Chart-fidelity experiment: {args.output}")
        return
    if args.command == "resonance-metrology":
        config = (
            MetrologyConfig.full(args.output)
            if args.profile == "full"
            else MetrologyConfig.ci(args.output)
        )
        if args.epochs is not None:
            if args.epochs < 1:
                parser.error("--epochs must be positive")
            config = replace(config, epochs=args.epochs)
        print(
            "Training exact-symplectic chart ensembles and measuring the "
            "trajectory-sampled resonant block…"
        )
        result = run_resonance_metrology(config)
        consensus = result["ensemble_consensus"]
        print(f"Metrology status: {result['status']} ({result['status_reason']})")
        print(
            "Recovered generating amplitude: "
            f"{consensus['generating_function_amplitude']:.6g}; "
            "complex error: "
            f"{consensus['complex_error']:.2%}"
        )
        print(f"Report: {args.output / 'report.html'}")
        print(f"Manifest: {args.output / 'manifest.json'}")
        return
    if args.command == "predict":
        if args.steps < 1:
            parser.error("--steps must be positive")
        model = load_mechanics_model(args.model)
        if len(args.initial) != len(model.state_columns):
            parser.error(
                f"expected {len(model.state_columns)} initial values for "
                f"{', '.join(model.state_columns)}"
            )
        initial = np.asarray(args.initial, dtype=np.float64)
        support = str(model.support_status(initial)[0])
        if support != "supported" and not args.allow_unsupported:
            parser.error(
                f"prediction is unsupported ({support}); "
                "pass --allow-unsupported to override"
            )
        prediction = model.rollout(
            initial,
            steps=args.steps,
            allow_extrapolation=args.allow_unsupported,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("time", *model.state_columns))
            for step, state in enumerate(prediction):
                writer.writerow((step * model.operator.dt, *state))
        print(f"Prediction support: {support}")
        print(f"Prediction: {args.output}")
        return
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
