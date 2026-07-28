from __future__ import annotations

import json
import platform
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from learned_koopman import __version__
from learned_koopman.config import ExperimentConfig
from learned_koopman.evaluation import evaluate
from learned_koopman.training import TrainedModels, train_models


def _parameter_counts(models: TrainedModels) -> dict[str, int]:
    return {
        "persistence": 0,
        "dmd": 9,
        "small_angle": 0,
        "mlp": sum(parameter.numel() for parameter in models.mlp.parameters()),
        "fixed_koopman": sum(parameter.numel() for parameter in models.fixed.parameters()),
        "energy_conditioned": sum(
            parameter.numel() for parameter in models.conditioned.parameters()
        ),
    }


def _plot_rollouts(
    rollouts: dict[float, dict[str, np.ndarray]],
    metrics: dict[str, dict[str, dict[str, float | int | None]]],
    config: ExperimentConfig,
    output: Path,
) -> None:
    colors = {
        "reference": "#111827",
        "persistence": "#9ca3af",
        "dmd": "#0ea5e9",
        "small_angle": "#f59e0b",
        "mlp": "#10b981",
        "fixed_koopman": "#6366f1",
        "energy_conditioned": "#e11d48",
    }
    showcase = 2.0
    time = np.arange(config.rollout_steps + 1) * config.dt
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    for name, states in rollouts[showcase].items():
        theta = np.arctan2(states[:, 0], states[:, 1])
        axes[0].plot(time, theta, label=name.replace("_", " "), color=colors[name], lw=1.7)
    axes[0].set_title(rf"Autonomous rollout, $\theta_0={showcase}$")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel(r"$\theta$")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    model_names = [
        "persistence",
        "dmd",
        "small_angle",
        "mlp",
        "fixed_koopman",
        "energy_conditioned",
    ]
    amplitudes = [float(value) for value in metrics]
    for name in model_names:
        valid = [float(metrics[f"{amplitude:.2f}"][name]["valid_time"]) for amplitude in amplitudes]
        axes[1].plot(
            amplitudes,
            valid,
            marker="o",
            label=name.replace("_", " "),
            color=colors[name],
        )
    axes[1].set_title("Valid autonomous prediction time")
    axes[1].set_xlabel("initial amplitude")
    axes[1].set_ylabel("time before error > 0.15")

    reference_frequency = [
        metrics[f"{amplitude:.2f}"]["reference"]["angular_frequency"] for amplitude in amplitudes
    ]
    axes[2].plot(
        amplitudes,
        reference_frequency,
        color=colors["reference"],
        marker="o",
        label="reference",
    )
    for name in ["fixed_koopman", "energy_conditioned"]:
        values = [
            metrics[f"{amplitude:.2f}"][name]["angular_frequency"] for amplitude in amplitudes
        ]
        axes[2].plot(
            amplitudes,
            [np.nan if value is None else value for value in values],
            marker="o",
            label=name.replace("_", " "),
            color=colors[name],
        )
    axes[2].set_title("Rollout frequency vs reference")
    axes[2].set_xlabel("initial amplitude")
    axes[2].set_ylabel("angular frequency")
    axes[2].legend(frameon=False, fontsize=8)

    figure.suptitle("Structured latent dynamics for the nonlinear pendulum", fontweight="bold")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_models(models: TrainedModels, output_dir: Path) -> None:
    torch.save(models.mlp.state_dict(), output_dir / "mlp.pt")
    torch.save(models.fixed.state_dict(), output_dir / "fixed_koopman.pt")
    torch.save(models.conditioned.state_dict(), output_dir / "energy_conditioned.pt")


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    models = train_models(config)
    metrics, rollouts = evaluate(models, config)
    _plot_rollouts(rollouts, metrics, config, output_dir / "comparison.png")
    _save_models(models, output_dir)
    result: dict[str, object] = {
        "config": config.to_dict(),
        "environment": {
            "learned_koopman": __version__,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "parameter_counts": _parameter_counts(models),
        "training_loss_final": {name: values[-1] for name, values in models.histories.items()},
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _aggregate_seed_metrics(
    runs: dict[str, dict[str, object]],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    metric_names = ("valid_time", "angle_rmse", "omega_rmse", "max_energy_drift")
    first_run = next(iter(runs.values()))
    first_metrics = first_run["metrics"]
    assert isinstance(first_metrics, dict)
    aggregate: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for amplitude, model_metrics in first_metrics.items():
        assert isinstance(model_metrics, dict)
        aggregate[amplitude] = {}
        for model_name in model_metrics:
            aggregate[amplitude][model_name] = {}
            for metric_name in metric_names:
                values = np.array(
                    [
                        run["metrics"][amplitude][model_name][metric_name]  # type: ignore[index]
                        for run in runs.values()
                    ],
                    dtype=np.float64,
                )
                aggregate[amplitude][model_name][metric_name] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
    return aggregate


def run_robustness_sweep(
    config: ExperimentConfig,
    seeds: Sequence[int],
) -> dict[str, object]:
    """Train independent seeded runs and summarize variation in every core metric."""

    unique_seeds = list(dict.fromkeys(seeds))
    if len(unique_seeds) < 2:
        raise ValueError("A robustness sweep requires at least two distinct seeds.")
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict[str, object]] = {}
    for seed in unique_seeds:
        seed_config = replace(config, seed=seed, output_dir=output_dir / f"seed-{seed}")
        run = run_experiment(seed_config)
        runs[str(seed)] = {
            "config": run["config"],
            "environment": run["environment"],
            "parameter_counts": run["parameter_counts"],
            "training_loss_final": run["training_loss_final"],
            "metrics": run["metrics"],
        }

    aggregate = _aggregate_seed_metrics(runs)
    comparisons = {
        "showcase_amplitude": 2.0,
        "conditioned_valid_time_wins_over_mlp": sum(
            run["metrics"]["2.00"]["energy_conditioned"]["valid_time"]  # type: ignore[index]
            > run["metrics"]["2.00"]["mlp"]["valid_time"]  # type: ignore[index]
            for run in runs.values()
        ),
        "conditioned_angle_rmse_wins_over_mlp": sum(
            run["metrics"]["2.00"]["energy_conditioned"]["angle_rmse"]  # type: ignore[index]
            < run["metrics"]["2.00"]["mlp"]["angle_rmse"]  # type: ignore[index]
            for run in runs.values()
        ),
        "seed_count": len(unique_seeds),
    }
    result: dict[str, object] = {
        "seeds": unique_seeds,
        "base_config": config.to_dict(),
        "runs": runs,
        "aggregate": aggregate,
        "comparisons": comparisons,
    }
    (output_dir / "robustness.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
