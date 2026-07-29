from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from learned_koopman.island_area import (
    IslandAreaConfig,
    _claim_state,
    bounded_libration_mask,
    quadrilateral_cell_areas,
    run_island_area_audit,
    validate_island_area_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results" / "resonance-metrology" / "manifest.json"


def test_quadrilateral_cell_areas_measure_structured_mesh() -> None:
    x, y = np.meshgrid(
        np.asarray((0.0, 1.0, 2.0)),
        np.asarray((0.0, 1.0, 2.0, 3.0)),
        indexing="ij",
    )
    vertices = np.stack((x, y), axis=-1)
    areas = quadrilateral_cell_areas(vertices)
    assert areas.shape == (2, 3)
    np.testing.assert_allclose(areas, 1.0)


def test_bounded_libration_mask_separates_winding_topology() -> None:
    time = np.arange(200, dtype=np.float64)
    bounded = 0.2 * np.sin(0.08 * time)
    circulating = 0.04 * time
    mask, diagnostics = bounded_libration_mask(
        np.stack((bounded, circulating), axis=1),
        order=3,
    )
    assert mask.tolist() == [True, False]
    assert (
        diagnostics["maximum_bounded_span"]
        < diagnostics["minimum_unbounded_span"]
    )


def test_quick_island_area_audit_has_real_controls(tmp_path: Path) -> None:
    result = run_island_area_audit(
        IslandAreaConfig(
            output=tmp_path,
            resonance_manifest=REFERENCE,
            radial_cells=13,
            angular_cells=36,
            steps=80,
        )
    )
    result["_artifact_root"] = str(tmp_path)
    checks = validate_island_area_manifest(
        result,
        require_clean_source=False,
    )
    assert result["experiment"] == "island-area-audit"
    assert result["ensemble"]["accepted_count"] == 8
    assert result["config"]["profile"] == "exploratory"
    assert result["status"] == "not_resolved_abstained"
    assert result["status_reason"] == "non_reference_profile"
    assert result["controls"]["noncanonical_scale"]["relative_area_shift"] > 0.19
    assert result["controls"]["exact_gauge_stress"]["rows"]
    assert checks


def test_validator_rejects_duplicate_chart_artifacts(tmp_path: Path) -> None:
    result = run_island_area_audit(
        IslandAreaConfig(
            output=tmp_path,
            resonance_manifest=REFERENCE,
            radial_cells=9,
            angular_cells=24,
            steps=40,
        )
    )
    result["_artifact_root"] = str(tmp_path)
    result["ensemble"]["charts"][1]["model_sha256"] = result["ensemble"]["charts"][0][
        "model_sha256"
    ]
    with pytest.raises(ValueError, match="independent charts"):
        validate_island_area_manifest(
            result,
            require_clean_source=False,
        )


def test_validator_recomputes_consensus_and_gate_arithmetic(tmp_path: Path) -> None:
    result = run_island_area_audit(
        IslandAreaConfig(
            output=tmp_path,
            resonance_manifest=REFERENCE,
            radial_cells=9,
            angular_cells=24,
            steps=40,
        )
    )
    result["_artifact_root"] = str(tmp_path)
    stale = copy.deepcopy(result)
    stale["ensemble"]["consensus_area"] += 0.25
    with pytest.raises(ValueError, match="consensus area"):
        validate_island_area_manifest(
            stale,
            require_clean_source=False,
        )


def test_claim_state_refuses_exploratory_support_and_can_refute() -> None:
    passing = {
        "direct_map_matches_leading_area": {"passed": True},
        "learned_consensus_matches_direct_area": {"passed": True},
        "every_chart_matches_direct_area": {"passed": True},
        "membership_matches_direct_topology": {"passed": True},
        "exact_gauges_preserve_area": {"passed": True},
        "null_area_stays_within_resolution_fraction_ceiling": {"passed": True},
        "noncanonical_area_scaling_plumbing": {"passed": True},
        "learned_charts_preserve_domain_area": {"passed": True},
        "probe_mesh_within_model_support": {"passed": True},
        "learned_chart_beats_raw_polar_baseline": {"passed": True},
    }
    assert _claim_state(passing, is_reference_profile=False) == (
        "not_resolved_abstained",
        "non_reference_profile",
    )
    failed_invariance = copy.deepcopy(passing)
    failed_invariance["exact_gauges_preserve_area"]["passed"] = False
    assert _claim_state(failed_invariance, is_reference_profile=True) == (
        "resolved_refuted",
        "gauge_invariance_failed",
    )


def test_relative_reference_manifest_is_cwd_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "audit"
    result = run_island_area_audit(
        IslandAreaConfig(
            output=output,
            resonance_manifest=Path("results/resonance-metrology/manifest.json"),
            radial_cells=9,
            angular_cells=24,
            steps=40,
        )
    )
    assert result["ensemble"]["accepted_count"] == 8
