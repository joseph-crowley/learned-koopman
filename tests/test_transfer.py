from __future__ import annotations

import json

import torch

from learned_koopman.models.transfer import SimplexTransferOperator
from learned_koopman.transfer_experiment import run_transfer_experiment


def test_simplex_and_transition_constraints_hold_by_construction() -> None:
    torch.manual_seed(3)
    model = SimplexTransferOperator(n_states=5, hidden_dim=12)
    state = torch.randn(17, 3)

    membership = model.membership(state)
    transition = model.transition_matrix()
    propagated = model.propagate(membership, steps=3)

    assert torch.all(membership >= 0)
    assert torch.allclose(membership.sum(dim=-1), torch.ones(17), atol=1e-6)
    assert torch.all(transition > 0)
    assert torch.allclose(transition.sum(dim=-1), torch.ones(5), atol=1e-6)
    assert torch.allclose(propagated.sum(dim=-1), torch.ones(17), atol=1e-6)


def test_quick_transfer_experiment_is_reproducible_and_writes_metrics(tmp_path) -> None:
    result = run_transfer_experiment(quick=True, seed=11, output_dir=tmp_path)

    assert result["experiment"] == "simplex_transfer_operator"
    assert result["objective"]["latent_family"] == "categorical simplex"
    assert result["constraints"]["membership_max_sum_error"] < 1e-5
    assert result["constraints"]["transition_max_row_sum_error"] < 1e-5
    assert result["constraints"]["membership_min_probability"] >= 0.0
    assert result["constraints"]["transition_min_probability"] > 0.0
    assert result["held_out"]["membership_classification_accuracy"] > 0.75
    assert result["held_out"]["one_step_nll"] < result["held_out"]["occupancy_baseline_nll"]
    assert result["held_out"]["no_operator_one_step_nll"] >= 0.0
    assert result["held_out"]["state_change_rate"] > 0.0
    assert result["two_lag_held_out"]["learned_k_squared_nll"] >= 0.0
    assert result["two_lag_held_out"]["no_operator_membership_nll"] >= 0.0
    assert result["two_lag_held_out"]["empirical_ulam_squared_nll"] >= 0.0
    assert result["two_lag_held_out"]["direct_two_lag_ulam_nll"] >= 0.0
    assert result["two_lag_held_out"]["occupancy_baseline_nll"] >= 0.0
    assert result["stationary"]["sum_error"] < 1e-10
    assert result["stationary"]["fixed_point_l1_residual"] < 1e-6
    assert result["representation"]["active_states_above_one_percent"] >= 4
    assert result["representation"]["effective_state_count"] > 3.0
    assert result["process_noise_evidence"]["mean_endpoint_velocity_variance"] > 0.0
    assert result["process_noise_evidence"]["mean_observed_destinations"] > 1.0
    assert result["process_noise_evidence"]["no_operator_cross_entropy"] >= 0.0
    assert result["baselines"]["empirical_ulam"]
    assert result["baselines"]["no_operator"]
    assert result["chapman_kolmogorov"]["learned_two_lag_weighted_rmse"] >= 0.0
    comparisons = result["operator_verdict"]["decisive_comparisons"]
    assert comparisons["one_lag_beats_no_operator"] == (
        result["held_out"]["one_step_nll"]
        < result["held_out"]["no_operator_one_step_nll"]
    )
    assert comparisons["one_lag_beats_empirical_ulam"] == (
        result["held_out"]["one_step_nll"]
        < result["held_out"]["empirical_ulam_nll"]
    )
    assert comparisons["two_lag_beats_no_operator"] == (
        result["two_lag_held_out"]["learned_k_squared_nll"]
        < result["two_lag_held_out"]["no_operator_membership_nll"]
    )
    assert comparisons["branching_beats_no_operator"] == (
        result["process_noise_evidence"]["model_cross_entropy"]
        < result["process_noise_evidence"]["no_operator_cross_entropy"]
    )
    assert comparisons["branching_beats_empirical_ulam"] == (
        result["process_noise_evidence"]["model_cross_entropy"]
        < result["process_noise_evidence"]["empirical_ulam_cross_entropy"]
    )
    assert comparisons["ck_beats_empirical_ulam"] == (
        result["chapman_kolmogorov"]["learned_two_lag_weighted_rmse"]
        < result["chapman_kolmogorov"]["empirical_ulam_two_lag_weighted_rmse"]
    )
    expected_status = (
        "not_falsified_on_this_profile"
        if all(comparisons.values())
        else "falsified_by_current_profile"
    )
    assert result["operator_verdict"]["status"] == expected_status
    assert result["operator_verdict"]["status"] == "falsified_by_current_profile"
    assert not comparisons["one_lag_beats_no_operator"]
    assert not comparisons["ck_beats_empirical_ulam"]
    assert result["operator_verdict"]["constraints_pass"]
    assert result["training"]["final_loss"] < result["training"]["initial_loss"]

    written = json.loads((tmp_path / "transfer_metrics.json").read_text())
    assert written["held_out"]["samples"] == result["held_out"]["samples"]
    assert (tmp_path / "transfer_model.pt").is_file()
