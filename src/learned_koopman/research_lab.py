from __future__ import annotations

import json
import math
import platform
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from learned_koopman import __version__
from learned_koopman.config import ExperimentConfig
from learned_koopman.control_experiment import (
    ControlExperimentProfile,
    run_control_experiment,
)
from learned_koopman.experiment import run_experiment
from learned_koopman.invariant_experiment import run_invariant_experiment
from learned_koopman.route_validation import validate_route_truth
from learned_koopman.transfer_experiment import run_transfer_experiment


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _plot_overview(
    atlas: dict[str, Any],
    invariant: dict[str, Any],
    transfer: dict[str, Any],
    control: dict[str, Any],
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))

    showcase = f"{float(atlas['config']['showcase_amplitude']):.2f}"
    atlas_metrics = atlas["metrics"][showcase]
    atlas_names = ["mlp", "energy_conditioned", "separatrix_atlas"]
    atlas_values = [atlas_metrics[name]["valid_time"] for name in atlas_names]
    axes[0, 0].bar(
        ["residual\nMLP", "single\nchart", "atlas"],
        atlas_values,
        color=["#10b981", "#e11d48", "#7c3aed"],
    )
    axes[0, 0].set_title(f"Autonomous valid time at amplitude {showcase}")
    axes[0, 0].set_ylabel("time before error threshold")

    invariant_values = [
        invariant["aggregate"]["affine_aligned_energy_r2"]["mean"],
        invariant["aggregate"]["absolute_spearman_rank"]["mean"],
        max(
            0.0,
            1.0
            - invariant["aggregate"]["mean_normalized_trajectory_drift"]["mean"],
        ),
    ]
    axes[0, 1].bar(
        ["energy $R^2$", "rank", "1 - drift"],
        invariant_values,
        color=["#2563eb", "#0ea5e9", "#06b6d4"],
    )
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].set_title("Invariant discovery, held-out shells")

    transfer_metrics = transfer["held_out"]
    axes[1, 0].bar(
        ["learned\nK", "no\noperator", "empirical\nUlam", "occupancy"],
        [
            transfer_metrics["one_step_nll"],
            transfer_metrics["no_operator_one_step_nll"],
            transfer_metrics["empirical_ulam_nll"],
            transfer_metrics["occupancy_baseline_nll"],
        ],
        color=["#8b5cf6", "#f43f5e", "#64748b", "#cbd5e1"],
    )
    ck = transfer["chapman_kolmogorov"]
    axes[1, 0].set_title(
        "Transfer falsified: "
        f"CK {ck['learned_two_lag_weighted_rmse']:.3f} "
        f"vs Ulam {ck['empirical_ulam_two_lag_weighted_rmse']:.3f}"
    )
    axes[1, 0].set_ylabel("lower is better")

    control_methods = control["evaluation"]["methods"]
    control_values = [
        max(float(control_methods[name]["crossing_window_rollout_error"]), 1e-14)
        for name in (
            "exact_unit_gain_oracle",
            "learned_gain_only",
            "residual_ablation",
            "small_angle_controlled",
            "learned_gain_only_control_blind",
        )
    ]
    axes[1, 1].bar(
        ["exact\noracle", "learned\ngain", "neural\nresidual", "small-\nangle", "action-\nblind"],
        control_values,
        color=["#1e293b", "#f97316", "#fb7185", "#f59e0b", "#94a3b8"],
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Actuator identification around real crossings")
    axes[1, 1].set_ylabel("circular state error, log scale")

    figure.suptitle(
        "Learned geometry, local laws, probability flow, and control",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _summary(
    atlas: dict[str, Any],
    invariant: dict[str, Any],
    transfer: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, Any]:
    showcase = f"{float(atlas['config']['showcase_amplitude']):.2f}"
    atlas_metrics = atlas["metrics"][showcase]["separatrix_atlas"]
    transfer_metrics = transfer["held_out"]
    control_methods = control["evaluation"]["methods"]
    return {
        "atlas": {
            "showcase_amplitude": float(showcase),
            "valid_time": atlas_metrics["valid_time"],
            "route_switches": atlas_metrics["route_switches"],
            "rapid_route_reversals": atlas_metrics["rapid_route_reversals"],
            "maximum_energy_drift": atlas_metrics["max_energy_drift"],
        },
        "invariant": {
            "affine_aligned_energy_r2": invariant["aggregate"][
                "affine_aligned_energy_r2"
            ]["mean"],
            "absolute_spearman_rank": invariant["aggregate"][
                "absolute_spearman_rank"
            ]["mean"],
            "mean_normalized_trajectory_drift": invariant["aggregate"][
                "mean_normalized_trajectory_drift"
            ]["mean"],
        },
        "transfer": {
            "one_step_nll": transfer_metrics["one_step_nll"],
            "no_operator_one_step_nll": transfer_metrics[
                "no_operator_one_step_nll"
            ],
            "empirical_ulam_nll": transfer_metrics["empirical_ulam_nll"],
            "occupancy_baseline_nll": transfer_metrics["occupancy_baseline_nll"],
            "learned_ck_rmse": transfer["chapman_kolmogorov"][
                "learned_two_lag_weighted_rmse"
            ],
            "empirical_ulam_ck_rmse": transfer["chapman_kolmogorov"][
                "empirical_ulam_two_lag_weighted_rmse"
            ],
            "operator_verdict": transfer["operator_verdict"]["status"],
        },
        "control": {
            "actual_crossings": control["evaluation"]["actual_crossing_count"],
            "autonomous_replay_crossings": control["evaluation"][
                "autonomous_replay_crossing_count"
            ],
            "promoted_learned_system": control["promoted_learned_system"]["method"],
            "learned_control_gain": control["training"]["gain_only"][
                "learned_control_gain"
            ],
            "crossing_event_recall": control_methods["learned_gain_only"]["crossing"][
                "event_recall"
            ],
            "exact_oracle_crossing_window_error": control_methods[
                "exact_unit_gain_oracle"
            ]["crossing_window_rollout_error"],
            "learned_gain_crossing_window_error": control_methods[
                "learned_gain_only"
            ]["crossing_window_rollout_error"],
            "residual_crossing_window_error": control_methods["residual_ablation"][
                "crossing_window_rollout_error"
            ],
            "small_angle_crossing_window_error": control_methods[
                "small_angle_controlled"
            ]["crossing_window_rollout_error"],
            "control_blind_crossing_window_error": control_methods["control_blind"][
                "crossing_window_rollout_error"
            ],
        },
    }


def run_research_lab(
    output_dir: Path = Path("results/research-lab"),
    *,
    quick: bool = False,
    seed: int = 7,
) -> dict[str, Any]:
    """Run the four connected experiments and write one validated manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    profile = "quick" if quick else "full"

    atlas_config = (
        ExperimentConfig.quick_atlas(output_dir / "atlas")
        if quick
        else ExperimentConfig.atlas(output_dir / "atlas")
    )
    atlas = run_experiment(
        replace(atlas_config, seed=seed),
        include_atlas=True,
    )

    invariant_seeds = (
        (seed, seed + 10)
        if quick
        else (seed, seed + 10, seed + 22, seed + 34, seed + 46)
    )
    invariant = run_invariant_experiment(
        profile=profile,
        seeds=invariant_seeds,
    )
    _write_json(output_dir / "invariant" / "metrics.json", invariant)

    transfer = run_transfer_experiment(
        quick=quick,
        seed=seed,
        output_dir=output_dir / "transfer",
    )

    control_profile = (
        ControlExperimentProfile.quick(seed)
        if quick
        else ControlExperimentProfile.full(seed)
    )
    control = run_control_experiment(control_profile)
    _write_json(output_dir / "control" / "metrics.json", control)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "learned_koopman_research_lab",
        "profile": profile,
        "seed": seed,
        "thesis": (
            "Learn the geometry, local laws, and transitions that organize "
            "nonlinear dynamics without assuming one global linearization."
        ),
        "environment": {
            "learned_koopman": __version__,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "scientific_contract": {
            "atlas": (
                "Autonomous routing uses predicted state only and exposes the "
                "complete route trace plus chatter diagnostics."
            ),
            "invariant": (
                "Training receives state trajectories and membership only; "
                "physical energy appears only in post-training evaluation."
            ),
            "transfer": (
                "Physical process noise drives a categorical simplex model "
                "with a positive row-stochastic operator."
            ),
            "control": (
                "A scalar actuator gain is identified from forced trajectories; "
                "known controls drive recursive prediction and no future true "
                "state is supplied."
            ),
        },
        "artifacts": {
            "overview": {"path": "overview.png"},
            "atlas": {
                "json_pointer": "#/experiments/atlas",
                "five_seed_evidence": "../atlas/robustness.json",
            },
            "invariant": {"json_pointer": "#/experiments/invariant"},
            "transfer": {"json_pointer": "#/experiments/transfer"},
            "control": {"json_pointer": "#/experiments/control"},
        },
        "summary": _summary(atlas, invariant, transfer, control),
        "experiments": {
            "atlas": atlas,
            "invariant": invariant,
            "transfer": transfer,
            "control": control,
        },
        "claim_boundary": [
            "The lab is a reproducible PyTorch research example, not a new theorem.",
            "The atlas still receives analytic energy and supplied local geometry.",
            "Invariant recovery is demonstrated on noiseless libration shells only.",
            transfer["operator_verdict"]["interpretation"],
            (
                "The controlled result identifies one scalar actuator gain in a "
                "supplied plant equation; it is not a closed-loop controller."
            ),
        ],
    }
    validate_research_lab(manifest)
    _plot_overview(
        atlas,
        invariant,
        transfer,
        control,
        output_dir / "overview.png",
    )
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _assert_finite(value: Any, path: str = "manifest") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} is not finite")


def validate_research_lab(manifest: dict[str, Any]) -> list[str]:
    """Reject missing, numerically invalid, or scientifically incoherent runs."""

    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported research-lab schema")
    experiments = manifest.get("experiments")
    if not isinstance(experiments, dict):
        raise ValueError("manifest is missing experiments")
    required = {"atlas", "invariant", "transfer", "control"}
    if set(experiments) != required:
        raise ValueError("manifest does not contain the four required experiments")
    _assert_finite(manifest)

    checks: list[str] = []
    atlas_result = experiments["atlas"]
    expected_steps = int(atlas_result["config"]["rollout_steps"])
    trace_count = 0
    for amplitude, model_metrics in atlas_result["metrics"].items():
        try:
            validate_route_truth(
                model_metrics["separatrix_atlas"],
                expected_steps=expected_steps,
                label=f"research-lab amplitude {amplitude}",
            )
        except AssertionError as error:
            raise ValueError(f"atlas route truth failed: {error}") from error
        trace_count += 1
    checks.append(
        f"atlas route truth is independently reconstructed for {trace_count} amplitudes"
    )

    invariant = experiments["invariant"]
    exclusions = invariant["scientific_contract"]["training_excludes"]
    if "physical energy" not in exclusions or "frequency targets" not in exclusions:
        raise ValueError("invariant training boundary is incomplete")
    if invariant["aggregate"]["affine_aligned_energy_r2"]["mean"] < 0.8:
        raise ValueError("invariant does not organize held-out energy shells")
    if invariant["aggregate"]["mean_normalized_trajectory_drift"]["mean"] > 0.2:
        raise ValueError("invariant drifts excessively along held-out trajectories")
    if invariant["aggregate"]["quotient_coordinate_std"]["min"] <= 0.05:
        raise ValueError("invariant coordinate collapsed")
    checks.append("label-free invariant is noncollapsed and orders held-out shells")

    transfer = experiments["transfer"]
    constraints = transfer["constraints"]
    if constraints["membership_max_sum_error"] >= 1e-5:
        raise ValueError("transfer memberships do not remain on the simplex")
    if constraints["transition_max_row_sum_error"] >= 1e-5:
        raise ValueError("transfer matrix does not preserve mass")
    if constraints["membership_min_probability"] < 0.0:
        raise ValueError("transfer memberships contain negative probability")
    if constraints["transition_min_probability"] <= 0.0:
        raise ValueError("transfer matrix contains non-positive probability")
    held_out = transfer["held_out"]
    if held_out["one_step_nll"] >= held_out["empirical_ulam_nll"]:
        raise ValueError("transfer operator does not beat empirical Ulam at one lag")
    if held_out["one_step_nll"] >= held_out["occupancy_baseline_nll"]:
        raise ValueError("transfer operator does not beat the occupancy baseline")
    if transfer["representation"]["active_states_above_one_percent"] < 4:
        raise ValueError("transfer representation collapsed")
    noise = transfer["process_noise_evidence"]
    if (
        noise["mean_observed_destinations"] <= 1.0
        or noise["mean_endpoint_velocity_variance"] <= 0.0
    ):
        raise ValueError("transfer experiment lacks physical stochastic branching")
    comparisons = transfer["operator_verdict"]["decisive_comparisons"]
    expected_comparisons = {
        "one_lag_beats_no_operator": (
            held_out["one_step_nll"] < held_out["no_operator_one_step_nll"]
        ),
        "one_lag_beats_empirical_ulam": (
            held_out["one_step_nll"] < held_out["empirical_ulam_nll"]
        ),
        "two_lag_beats_no_operator": (
            transfer["two_lag_held_out"]["learned_k_squared_nll"]
            < transfer["two_lag_held_out"]["no_operator_membership_nll"]
        ),
        "two_lag_beats_direct_ulam": (
            transfer["two_lag_held_out"]["learned_k_squared_nll"]
            < transfer["two_lag_held_out"]["direct_two_lag_ulam_nll"]
        ),
        "branching_beats_no_operator": (
            noise["model_cross_entropy"] < noise["no_operator_cross_entropy"]
        ),
        "branching_beats_empirical_ulam": (
            noise["model_cross_entropy"] < noise["empirical_ulam_cross_entropy"]
        ),
        "branching_beats_occupancy": (
            noise["model_cross_entropy"]
            < noise["occupancy_baseline_cross_entropy"]
        ),
        "ck_beats_empirical_ulam": (
            transfer["chapman_kolmogorov"]["learned_two_lag_weighted_rmse"]
            < transfer["chapman_kolmogorov"][
                "empirical_ulam_two_lag_weighted_rmse"
            ]
        ),
    }
    if comparisons != expected_comparisons:
        raise ValueError("transfer operator verdict disagrees with measured comparisons")
    expected_status = (
        "not_falsified_on_this_profile"
        if all(expected_comparisons.values())
        else "falsified_by_current_profile"
    )
    if transfer["operator_verdict"]["status"] != expected_status:
        raise ValueError("transfer operator verdict status is stale")
    checks.append(
        "transfer constraints pass and counterfactual verdict is "
        f"{expected_status}"
    )

    control = experiments["control"]
    evaluation = control["evaluation"]
    if evaluation["actual_crossing_count"] <= 0:
        raise ValueError("controlled dataset contains no separatrix crossings")
    if evaluation["autonomous_replay_crossing_count"] != 0:
        raise ValueError("crossing attribution is not specific to applied control")
    methods = evaluation["methods"]
    promotion = control["promoted_learned_system"]
    if promotion["method"] != "learned_gain_only":
        raise ValueError("control experiment did not promote the identifiable model")
    if (
        promotion.get("selection_basis")
        != "predeclared identifiable scalar actuator gain"
    ):
        raise ValueError("control promotion basis is stale or outcome-selected")
    gain = float(control["training"]["gain_only"]["learned_control_gain"])
    if abs(gain - 1.0) >= 0.05:
        raise ValueError("control experiment failed to identify actuator gain")
    learned = methods["learned_gain_only"]
    oracle = methods["exact_unit_gain_oracle"]
    if learned["crossing"]["event_recall"] != 1.0:
        raise ValueError("gain-only model misses controlled crossing events")
    if learned["crossing_window_rollout_error"] >= min(
        methods["residual_ablation"]["crossing_window_rollout_error"],
        methods["small_angle_controlled"]["crossing_window_rollout_error"],
        methods["learned_gain_only_control_blind"]["crossing_window_rollout_error"],
    ):
        raise ValueError("gain-only model does not win its learned and physical ablations")
    absolute_oracle_tolerance = (
        1e-6 if manifest.get("profile") == "quick" else 1e-10
    )
    if learned["crossing_window_rollout_error"] > max(
        absolute_oracle_tolerance,
        20.0 * oracle["crossing_window_rollout_error"],
    ):
        raise ValueError("identified model is not close to the supplied oracle")
    if evaluation["external_work_energy_change_mae"] >= 0.01:
        raise ValueError("controlled simulator fails its external-work audit")
    checks.append("actuator gain is identified and matches the supplied oracle")
    return checks
