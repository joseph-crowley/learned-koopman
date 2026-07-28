from __future__ import annotations

import json
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from learned_koopman.config import ExperimentConfig
from learned_koopman.evaluation import evaluate
from learned_koopman.training import TrainedModels, train_models


def _parameter_counts(models: TrainedModels) -> dict[str, int]:
    return {
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
    axes[2].set_title("Recovered amplitude–frequency law")
    axes[2].set_xlabel("initial amplitude")
    axes[2].set_ylabel("angular frequency")
    axes[2].legend(frameon=False, fontsize=8)

    figure.suptitle("Learned Koopman — honest long-horizon comparison", fontweight="bold")
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
