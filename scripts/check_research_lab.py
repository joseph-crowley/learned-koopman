from __future__ import annotations

import argparse
import json
from pathlib import Path

from learned_koopman.research_lab import validate_research_lab

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results/research-lab/manifest.json"
README = ROOT / "README.md"


def _validate_readme_claims(payload: dict[str, object]) -> None:
    """Tie the compact public research-lab claims to the committed manifest."""

    summary = payload["summary"]
    experiments = payload["experiments"]
    invariant = summary["invariant"]
    transfer = summary["transfer"]
    control = summary["control"]
    control_training = experiments["control"]["training"]["gain_only"]
    control_evaluation = experiments["control"]["evaluation"]
    crossing_count = int(control["actual_crossings"])
    crossing_total = round(
        crossing_count / float(control_evaluation["actual_crossing_rate"])
    )
    readme = README.read_text(encoding="utf-8")
    expected_claims = (
        f"held-out energy $R^2={invariant['affine_aligned_energy_r2']:.3f}$",
        f"rank $={invariant['absolute_spearman_rank']:.3f}$",
        f"drift $={invariant['mean_normalized_trajectory_drift']:.4f}$",
        f"actuator gain ${control_training['initial_control_gain']:.2f}"
        f"\\rightarrow{control['learned_control_gain']:.3f}$",
        f"with {crossing_count}/{crossing_total}",
        "real crossings recovered",
    )
    for claim in expected_claims:
        assert claim in readme, f"README research-lab claim is stale or missing: {claim}"
    assert transfer["operator_verdict"] == "falsified_by_current_profile"
    assert "stronger baselines falsify the learned propagation" in readme


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a learned-koopman research-lab manifest.",
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    arguments = parser.parse_args()
    payload = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    checks = validate_research_lab(payload)
    if arguments.manifest.resolve() == DEFAULT_MANIFEST.resolve():
        _validate_readme_claims(payload)
        checks.append("README research-lab values match the committed manifest")
    print("Research lab is internally coherent:")
    for check in checks:
        print(f"- {check}")


if __name__ == "__main__":
    main()
