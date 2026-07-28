from __future__ import annotations

import pytest

from learned_koopman.research_lab import validate_research_lab


def _valid_manifest() -> dict[str, object]:
    atlas_metrics = {
        "route_trace": [0, 0, 1, 1],
        "route_switch_steps": [2],
        "route_switches": 1,
        "rapid_route_reversals": 0,
        "route_alternations": 0,
        "max_route_switches_in_window": 1,
        "valid_steps": 4,
        "switches_within_valid_horizon": 1,
        "alternations_within_valid_horizon": 0,
        "rapid_reversals_within_valid_horizon": 0,
        "max_route_switches_in_window_within_valid_horizon": 1,
    }
    return {
        "schema_version": 1,
        "experiments": {
            "atlas": {
                "config": {
                    "showcase_amplitude": 3.05,
                    "rollout_steps": 4,
                },
                "metrics": {
                    "3.05": {
                        "separatrix_atlas": atlas_metrics,
                    }
                },
            },
            "invariant": {
                "scientific_contract": {
                    "training_excludes": ["physical energy", "frequency targets"],
                },
                "aggregate": {
                    "affine_aligned_energy_r2": {"mean": 0.95},
                    "mean_normalized_trajectory_drift": {"mean": 0.04},
                    "quotient_coordinate_std": {"min": 0.8},
                },
            },
            "transfer": {
                "constraints": {
                    "membership_max_sum_error": 1e-7,
                    "transition_max_row_sum_error": 1e-7,
                    "membership_min_probability": 0.01,
                    "transition_min_probability": 0.01,
                },
                "held_out": {
                    "one_step_nll": 0.4,
                    "no_operator_one_step_nll": 0.35,
                    "empirical_ulam_nll": 0.6,
                    "occupancy_baseline_nll": 1.2,
                },
                "two_lag_held_out": {
                    "learned_k_squared_nll": 0.5,
                    "no_operator_membership_nll": 0.6,
                    "direct_two_lag_ulam_nll": 0.45,
                },
                "representation": {
                    "active_states_above_one_percent": 5,
                },
                "process_noise_evidence": {
                    "mean_observed_destinations": 2.0,
                    "mean_endpoint_velocity_variance": 0.1,
                    "model_cross_entropy": 0.7,
                    "no_operator_cross_entropy": 1.0,
                    "empirical_ulam_cross_entropy": 0.6,
                    "occupancy_baseline_cross_entropy": 0.8,
                },
                "chapman_kolmogorov": {
                    "learned_two_lag_weighted_rmse": 0.3,
                    "empirical_ulam_two_lag_weighted_rmse": 0.1,
                },
                "operator_verdict": {
                    "status": "falsified_by_current_profile",
                    "decisive_comparisons": {
                        "one_lag_beats_no_operator": False,
                        "one_lag_beats_empirical_ulam": True,
                        "two_lag_beats_no_operator": True,
                        "two_lag_beats_direct_ulam": False,
                        "branching_beats_no_operator": True,
                        "branching_beats_empirical_ulam": False,
                        "branching_beats_occupancy": True,
                        "ck_beats_empirical_ulam": False,
                    },
                },
            },
            "control": {
                "training": {
                    "gain_only": {
                        "learned_control_gain": 1.0,
                    },
                },
                "promoted_learned_system": {
                    "method": "learned_gain_only",
                    "selection_basis": "predeclared identifiable scalar actuator gain",
                },
                "evaluation": {
                    "actual_crossing_count": 5,
                    "autonomous_replay_crossing_count": 0,
                    "external_work_energy_change_mae": 1e-4,
                    "methods": {
                        "exact_unit_gain_oracle": {
                            "crossing_window_rollout_error": 0.001,
                        },
                        "learned_gain_only": {
                            "crossing_window_rollout_error": 0.002,
                            "crossing": {"event_recall": 1.0},
                        },
                        "residual_ablation": {
                            "crossing_window_rollout_error": 0.03,
                        },
                        "small_angle_controlled": {
                            "crossing_window_rollout_error": 0.8,
                        },
                        "learned_gain_only_control_blind": {
                            "crossing_window_rollout_error": 2.0,
                        },
                    },
                },
            },
        },
    }


def test_validator_accepts_four_scientifically_coherent_experiments() -> None:
    checks = validate_research_lab(_valid_manifest())
    assert len(checks) == 4


def test_validator_uses_a_bounded_quick_profile_oracle_tolerance() -> None:
    manifest = _valid_manifest()
    manifest["profile"] = "quick"
    methods = manifest["experiments"]["control"]["evaluation"]["methods"]
    methods["exact_unit_gain_oracle"]["crossing_window_rollout_error"] = 1e-13
    methods["learned_gain_only"]["crossing_window_rollout_error"] = 5e-7
    validate_research_lab(manifest)

    methods["learned_gain_only"]["crossing_window_rollout_error"] = 2e-6
    with pytest.raises(ValueError, match="not close"):
        validate_research_lab(manifest)


def test_validator_rejects_hidden_route_chatter() -> None:
    manifest = _valid_manifest()
    atlas = manifest["experiments"]["atlas"]["metrics"]["3.05"]["separatrix_atlas"]
    atlas["rapid_route_reversals"] = 1
    with pytest.raises(ValueError, match="route truth"):
        validate_research_lab(manifest)


def test_validator_reconstructs_route_truth_instead_of_trusting_summaries() -> None:
    manifest = _valid_manifest()
    atlas = manifest["experiments"]["atlas"]["metrics"]["3.05"]["separatrix_atlas"]
    atlas.update(
        {
            "route_trace": [0, 1, 0, 0],
            "route_switch_steps": [1, 2],
            "route_switches": 2,
            "route_alternations": 1,
            "rapid_route_reversals": 1,
            "max_route_switches_in_window": 2,
            "valid_steps": 4,
            "switches_within_valid_horizon": 2,
            "alternations_within_valid_horizon": 1,
            "rapid_reversals_within_valid_horizon": 1,
            "max_route_switches_in_window_within_valid_horizon": 2,
        }
    )
    with pytest.raises(ValueError, match="alternation|reversal|chatter"):
        validate_research_lab(manifest)


def test_validator_rejects_a_collapsed_invariant() -> None:
    manifest = _valid_manifest()
    invariant = manifest["experiments"]["invariant"]["aggregate"]
    invariant["quotient_coordinate_std"]["min"] = 0.0
    with pytest.raises(ValueError, match="collapsed"):
        validate_research_lab(manifest)
