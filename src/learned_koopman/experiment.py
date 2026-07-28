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
    counts = {
        "persistence": 0,
        "dmd": 9,
        "small_angle": 0,
        "mlp": sum(parameter.numel() for parameter in models.mlp.parameters()),
        "fixed_koopman": sum(parameter.numel() for parameter in models.fixed.parameters()),
        "energy_conditioned": sum(
            parameter.numel() for parameter in models.conditioned.parameters()
        ),
    }
    if models.atlas is not None:
        counts["energy_projected_conditioned"] = counts["energy_conditioned"]
        counts["saddle_chart_only"] = 1
        counts["separatrix_atlas"] = sum(
            parameter.numel() for parameter in models.atlas.parameters()
        )
    return counts


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
        "energy_projected_conditioned": "#fb7185",
        "saddle_chart_only": "#a16207",
        "separatrix_atlas": "#7c3aed",
    }
    showcase = config.showcase_amplitude
    time = np.arange(config.rollout_steps + 1) * config.dt
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    atlas_mode = "separatrix_atlas" in next(iter(metrics.values()))

    rollout_names = (
        [
            "reference",
            "mlp",
            "energy_conditioned",
            "saddle_chart_only",
            "separatrix_atlas",
        ]
        if atlas_mode
        else list(rollouts[showcase])
    )
    for name in rollout_names:
        states = rollouts[showcase][name]
        theta = np.arctan2(states[:, 0], states[:, 1])
        axes[0].plot(
            time,
            theta,
            label=name.replace("_", " "),
            color=colors[name],
            lw=2.3 if name in {"reference", "separatrix_atlas"} else 1.6,
            ls="--" if name == "saddle_chart_only" else "-",
        )
    axes[0].set_title(rf"Autonomous rollout, $\theta_0={showcase}$")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel(r"$\theta$")
    axes[0].legend(frameon=True, framealpha=0.9, fontsize=8, loc="lower left")

    first_amplitude = next(iter(metrics.values()))
    model_names = (
        [
            "mlp",
            "energy_conditioned",
            "saddle_chart_only",
            "separatrix_atlas",
        ]
        if atlas_mode
        else [name for name in first_amplitude if name != "reference"]
    )
    amplitudes = [float(value) for value in metrics]
    for name in model_names:
        valid = [float(metrics[f"{amplitude:.2f}"][name]["valid_time"]) for amplitude in amplitudes]
        axes[1].plot(
            amplitudes,
            valid,
            marker="o",
            label=name.replace("_", " "),
            color=colors[name],
            ls="--" if name == "saddle_chart_only" else "-",
        )
    axes[1].set_title("Valid autonomous prediction time")
    axes[1].set_xlabel("initial amplitude")
    axes[1].set_ylabel("time before error > 0.15")

    if atlas_mode:
        saddle_fraction = [
            float(metrics[f"{amplitude:.2f}"]["separatrix_atlas"]["saddle_fraction"])
            for amplitude in amplitudes
        ]
        axes[2].plot(
            amplitudes,
            saddle_fraction,
            marker="o",
            label="saddle-chart fraction",
            color=colors["separatrix_atlas"],
        )
        axes[2].set_title("Geometric chart use")
        axes[2].set_xlabel("initial amplitude")
        axes[2].set_ylabel("fraction of steps in saddle chart")
        axes[2].set_ylim(-0.03, 1.03)
    else:
        reference_frequency = [
            metrics[f"{amplitude:.2f}"]["reference"]["angular_frequency"]
            for amplitude in amplitudes
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

    title = (
        "A two-chart model near the pendulum separatrix"
        if atlas_mode
        else "Structured latent dynamics for the nonlinear pendulum"
    )
    figure.suptitle(title, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_models(models: TrainedModels, output_dir: Path) -> None:
    torch.save(models.mlp.state_dict(), output_dir / "mlp.pt")
    torch.save(models.fixed.state_dict(), output_dir / "fixed_koopman.pt")
    torch.save(models.conditioned.state_dict(), output_dir / "energy_conditioned.pt")
    if models.atlas is not None:
        torch.save(models.atlas.state_dict(), output_dir / "separatrix_atlas.pt")


def run_experiment(
    config: ExperimentConfig,
    *,
    include_atlas: bool = False,
) -> dict[str, object]:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    models = train_models(config, include_atlas=include_atlas)
    metrics, rollouts = evaluate(models, config)
    _plot_rollouts(rollouts, metrics, config, output_dir / "comparison.png")
    _save_models(models, output_dir)
    model_diagnostics: dict[str, object] = {}
    if models.atlas is not None:
        saddle_operator = models.atlas.saddle_operator_matrix().detach()
        model_diagnostics["separatrix_atlas"] = {
            "saddle_rate": float(models.atlas.saddle_rate.detach()),
            "saddle_operator_determinant": float(torch.linalg.det(saddle_operator)),
            "minimum_saddle_energy": models.atlas.minimum_saddle_energy,
            "maximum_saddle_distance": models.atlas.maximum_saddle_distance,
            "router": "explicit geometric validity rule",
            "high_energy_projection": "initial invariant energy shell",
        }
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
        "model_diagnostics": model_diagnostics,
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


def _aggregate_band_metrics(
    runs: dict[str, dict[str, object]],
    amplitudes: Sequence[float],
) -> dict[str, object]:
    """Summarize each seed over a predeclared amplitude band before averaging seeds."""

    keys = [f"{amplitude:.2f}" for amplitude in amplitudes]
    metric_names = ("valid_time", "angle_rmse", "omega_rmse", "max_energy_drift")
    first_run = next(iter(runs.values()))
    first_metrics = first_run["metrics"]
    assert isinstance(first_metrics, dict)
    first_amplitude = first_metrics[keys[0]]
    assert isinstance(first_amplitude, dict)
    per_seed: dict[str, dict[str, dict[str, float]]] = {}
    for seed, run in runs.items():
        per_seed[seed] = {}
        for model_name in first_amplitude:
            per_seed[seed][model_name] = {}
            for metric_name in metric_names:
                values = [
                    float(run["metrics"][key][model_name][metric_name])  # type: ignore[index]
                    for key in keys
                ]
                per_seed[seed][model_name][metric_name] = float(np.mean(values))

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for model_name in first_amplitude:
        aggregate[model_name] = {}
        for metric_name in metric_names:
            values = np.array(
                [seed_metrics[model_name][metric_name] for seed_metrics in per_seed.values()],
                dtype=np.float64,
            )
            aggregate[model_name][metric_name] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return {
        "amplitudes": list(amplitudes),
        "per_seed": per_seed,
        "aggregate": aggregate,
    }


def run_robustness_sweep(
    config: ExperimentConfig,
    seeds: Sequence[int],
    *,
    include_atlas: bool = False,
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
        run = run_experiment(seed_config, include_atlas=include_atlas)
        runs[str(seed)] = {
            "config": run["config"],
            "environment": run["environment"],
            "parameter_counts": run["parameter_counts"],
            "training_loss_final": run["training_loss_final"],
            "metrics": run["metrics"],
        }

    aggregate = _aggregate_seed_metrics(runs)
    comparisons = {
        "showcase_amplitude": config.showcase_amplitude,
        "conditioned_valid_time_wins_over_mlp": sum(
            run["metrics"][f"{config.showcase_amplitude:.2f}"]["energy_conditioned"][  # type: ignore[index]
                "valid_time"
            ]
            > run["metrics"][f"{config.showcase_amplitude:.2f}"]["mlp"]["valid_time"]  # type: ignore[index]
            for run in runs.values()
        ),
        "conditioned_angle_rmse_wins_over_mlp": sum(
            run["metrics"][f"{config.showcase_amplitude:.2f}"]["energy_conditioned"][  # type: ignore[index]
                "angle_rmse"
            ]
            < run["metrics"][f"{config.showcase_amplitude:.2f}"]["mlp"]["angle_rmse"]  # type: ignore[index]
            for run in runs.values()
        ),
        "seed_count": len(unique_seeds),
    }
    if include_atlas:
        showcase_key = f"{config.showcase_amplitude:.2f}"
        comparisons.update(
            {
                "atlas_valid_time_wins_over_conditioned": sum(
                    run["metrics"][showcase_key]["separatrix_atlas"]["valid_time"]  # type: ignore[index]
                    > run["metrics"][showcase_key]["energy_conditioned"]["valid_time"]  # type: ignore[index]
                    for run in runs.values()
                ),
                "atlas_valid_time_wins_over_mlp": sum(
                    run["metrics"][showcase_key]["separatrix_atlas"]["valid_time"]  # type: ignore[index]
                    > run["metrics"][showcase_key]["mlp"]["valid_time"]  # type: ignore[index]
                    for run in runs.values()
                ),
            }
        )
    result: dict[str, object] = {
        "seeds": unique_seeds,
        "base_config": config.to_dict(),
        "runs": runs,
        "aggregate": aggregate,
        "comparisons": comparisons,
    }
    if include_atlas:
        band_amplitudes = [
            amplitude
            for amplitude in config.evaluation_amplitudes
            if amplitude >= config.summary_band_min_amplitude
        ]
        high_energy_band = _aggregate_band_metrics(runs, band_amplitudes)
        per_seed = high_energy_band["per_seed"]
        assert isinstance(per_seed, dict)
        high_energy_band["comparisons"] = {
            "atlas_valid_time_wins_over_mlp": sum(
                seed_metrics["separatrix_atlas"]["valid_time"]  # type: ignore[index]
                > seed_metrics["mlp"]["valid_time"]  # type: ignore[index]
                for seed_metrics in per_seed.values()
            ),
            "atlas_valid_time_wins_over_conditioned": sum(
                seed_metrics["separatrix_atlas"]["valid_time"]  # type: ignore[index]
                > seed_metrics["energy_conditioned"]["valid_time"]  # type: ignore[index]
                for seed_metrics in per_seed.values()
            ),
            "seed_count": len(unique_seeds),
        }
        result["high_energy_band"] = high_energy_band
    (output_dir / "robustness.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
