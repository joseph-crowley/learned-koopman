from __future__ import annotations

import argparse
import json
from pathlib import Path

from learned_koopman.research_lab import validate_research_lab

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results/research-lab/manifest.json"
README = ROOT / "README.md"


def _scientific_latex(value: float) -> str:
    mantissa, exponent = f"{value:.1e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def _validate_readme_claims(payload: dict[str, object]) -> None:
    """Tie the public v3 headline claims to the committed full manifest."""

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
        f"held-out energy \\(R^2={invariant['affine_aligned_energy_r2']:.3f}\\)",
        "normalized drift "
        f"\\(={invariant['mean_normalized_trajectory_drift']:.4f}\\)",
        f"one-lag NLL **{transfer['one_step_nll']:.3f}** versus "
        f"**{transfer['no_operator_one_step_nll']:.3f}**",
        f"CK **{transfer['learned_ck_rmse']:.3f}** versus Ulam "
        f"**{transfer['empirical_ulam_ck_rmse']:.3f}**",
        f"`{transfer['operator_verdict']}`",
        f"gain **{control_training['initial_control_gain']:.2f} → "
        f"{control['learned_control_gain']:.3f}**",
        f"**{crossing_count} / {crossing_total}** real crossings",
        "crossing-window error "
        f"\\({_scientific_latex(control['learned_gain_crossing_window_error'])}\\)",
    )
    for claim in expected_claims:
        assert claim in readme, f"README research-lab claim is stale or missing: {claim}"


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
        checks.append("README v3 headline values match the committed manifest")
    print("Research lab is internally coherent:")
    for check in checks:
        print(f"- {check}")


if __name__ == "__main__":
    main()
