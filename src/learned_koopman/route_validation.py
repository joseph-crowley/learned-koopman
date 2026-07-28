from __future__ import annotations

import math
from typing import Any

import numpy as np

RAPID_REVERSAL_WINDOW_STEPS = 10
MAXIMUM_SWITCH_FRACTION = 0.02


def validate_route_truth(
    metrics: dict[str, Any],
    *,
    expected_steps: int,
    label: str,
) -> int:
    """Reconstruct every route metric from a stored categorical trace.

    The checker intentionally does not trust summary counters emitted by the
    model. It is shared by trained-result validators and unit tests so the same
    independent contract applies to committed, CI, and ad-hoc runs.
    """

    trace = metrics.get("route_trace")
    assert isinstance(trace, list), f"{label} is stale: missing explicit route_trace"
    assert len(trace) == expected_steps, (
        f"{label} route trace has {len(trace)} steps, expected {expected_steps}"
    )
    raw_route = np.asarray(trace)
    assert np.issubdtype(raw_route.dtype, np.integer) and not np.issubdtype(
        raw_route.dtype, np.bool_
    ), f"{label} route trace must contain integer categories"
    route = raw_route.astype(np.int64, copy=False)
    assert np.all((route == 0) | (route == 1)), f"{label} contains a non-categorical route"

    switch_steps = np.flatnonzero(route[1:] != route[:-1]) + 1
    expected_switch_steps = [int(value) for value in switch_steps]
    assert metrics.get("route_switch_steps") == expected_switch_steps, (
        f"{label} route_switch_steps do not match the stored trace"
    )
    assert int(metrics["route_switches"]) == len(switch_steps), (
        f"{label} route_switches does not match the stored trace"
    )

    alternations = int(
        np.count_nonzero(
            (route[:-2] == route[2:]) & (route[1:-1] != route[:-2])
        )
    )
    rapid_reversals = int(
        np.count_nonzero(np.diff(switch_steps) <= RAPID_REVERSAL_WINDOW_STEPS)
    )
    maximum_switches_in_window = max(
        (
            int(
                np.count_nonzero(
                    (switch_steps >= start)
                    & (switch_steps <= start + RAPID_REVERSAL_WINDOW_STEPS)
                )
            )
            for start in switch_steps
        ),
        default=0,
    )
    assert int(metrics["route_alternations"]) == alternations
    assert int(metrics["rapid_route_reversals"]) == rapid_reversals
    assert int(metrics["max_route_switches_in_window"]) == maximum_switches_in_window

    valid_steps = int(metrics["valid_steps"])
    assert 0 <= valid_steps <= expected_steps, (
        f"{label} valid_steps={valid_steps} is outside [0, {expected_steps}]"
    )
    valid_route = route[:valid_steps]
    valid_switch_steps = np.flatnonzero(valid_route[1:] != valid_route[:-1]) + 1
    valid_alternations = int(
        np.count_nonzero(
            (valid_route[:-2] == valid_route[2:])
            & (valid_route[1:-1] != valid_route[:-2])
        )
    )
    valid_rapid_reversals = int(
        np.count_nonzero(
            np.diff(valid_switch_steps) <= RAPID_REVERSAL_WINDOW_STEPS
        )
    )
    valid_maximum_switches_in_window = max(
        (
            int(
                np.count_nonzero(
                    (valid_switch_steps >= start)
                    & (valid_switch_steps <= start + RAPID_REVERSAL_WINDOW_STEPS)
                )
            )
            for start in valid_switch_steps
        ),
        default=0,
    )
    assert int(metrics["switches_within_valid_horizon"]) == len(valid_switch_steps)
    assert int(metrics["alternations_within_valid_horizon"]) == valid_alternations
    assert (
        int(metrics["rapid_reversals_within_valid_horizon"])
        == valid_rapid_reversals
    )
    assert (
        int(metrics["max_route_switches_in_window_within_valid_horizon"])
        == valid_maximum_switches_in_window
    )

    maximum_switches = max(
        4,
        int(math.ceil(expected_steps * MAXIMUM_SWITCH_FRACTION)),
    )
    assert alternations == 0, f"{label} contains one-step route alternation"
    assert rapid_reversals == 0, f"{label} contains rapid route reversal"
    assert maximum_switches_in_window <= 1, f"{label} contains route-window chatter"
    assert len(switch_steps) <= maximum_switches, (
        f"{label} has pathological switch density: "
        f"{len(switch_steps)} switches in {expected_steps} steps"
    )
    return len(switch_steps)
