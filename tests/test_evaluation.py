import pytest
import torch

from learned_koopman.evaluation import _route_trace_metrics
from learned_koopman.models import SeparatrixAtlas
from learned_koopman.route_validation import validate_route_truth


def test_route_trace_metrics_separate_valid_horizon_from_full_rollout() -> None:
    route_trace = torch.tensor([0, 0, 1, 0, 1, 1, 0], dtype=torch.long)
    diagnostics = {
        "route_index": route_trace,
        "switch_disagreement": torch.empty(0),
        **SeparatrixAtlas.summarize_route_trace(route_trace),
    }

    metrics = _route_trace_metrics(diagnostics, valid_steps=4)

    assert metrics["route_trace"] == [0, 0, 1, 0, 1, 1, 0]
    assert metrics["route_switch_steps"] == [2, 3, 4, 6]
    assert metrics["route_switches"] == 4
    assert metrics["rapid_route_reversals"] == 3
    assert metrics["route_alternations"] == 2
    assert metrics["switches_within_valid_horizon"] == 2
    assert metrics["rapid_reversals_within_valid_horizon"] == 1
    assert metrics["alternations_within_valid_horizon"] == 1


def _stored_route_metrics(route_trace: torch.Tensor, valid_steps: int) -> dict[str, object]:
    diagnostics = {
        "route_index": route_trace,
        "switch_disagreement": torch.empty(0),
        **SeparatrixAtlas.summarize_route_trace(route_trace),
    }
    return {
        **_route_trace_metrics(diagnostics, valid_steps=valid_steps),
        "valid_steps": valid_steps,
    }


def test_route_truth_validator_accepts_clean_trace_and_rejects_chatter() -> None:
    clean = torch.tensor([0] * 5 + [1] * 15 + [0] * 10, dtype=torch.long)
    clean_metrics = _stored_route_metrics(clean, valid_steps=25)
    assert validate_route_truth(
        clean_metrics,
        expected_steps=len(clean),
        label="clean",
    ) == 2

    chatter = torch.tensor([0, 0, 1, 0, 1, 1, 0], dtype=torch.long)
    chatter_metrics = _stored_route_metrics(chatter, valid_steps=len(chatter))
    with pytest.raises(AssertionError, match="alternation|reversal|chatter"):
        validate_route_truth(
            chatter_metrics,
            expected_steps=len(chatter),
            label="chatter",
        )


def test_route_truth_validator_rejects_noninteger_trace_and_invalid_horizon() -> None:
    clean = torch.tensor([0] * 5 + [1] * 15 + [0] * 10, dtype=torch.long)
    metrics = _stored_route_metrics(clean, valid_steps=25)

    float_trace = {**metrics, "route_trace": [float(value) for value in clean]}
    with pytest.raises(AssertionError, match="integer categories"):
        validate_route_truth(
            float_trace,
            expected_steps=len(clean),
            label="float trace",
        )

    for invalid_steps in (-1, len(clean) + 1):
        invalid_horizon = {**metrics, "valid_steps": invalid_steps}
        with pytest.raises(AssertionError, match="outside"):
            validate_route_truth(
                invalid_horizon,
                expected_steps=len(clean),
                label="invalid horizon",
            )
