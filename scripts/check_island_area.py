from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from learned_koopman.island_area import (
    IslandAreaConfig,
    run_island_area_audit,
    validate_island_area_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results" / "island-area-audit" / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--reproduce",
        action="store_true",
        help="Rerun the saved mesh and compare the promoted numerical result.",
    )
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    payload["_artifact_root"] = str(args.manifest.parent)
    checks = validate_island_area_manifest(payload)
    if args.reproduce:
        with tempfile.TemporaryDirectory(
            prefix="learned-koopman-island-area-reproduction-"
        ) as temporary:
            saved = payload["config"]
            reference_manifest = Path(saved["resonance_manifest"])
            if not reference_manifest.is_absolute():
                reference_manifest = ROOT / reference_manifest
            rebuilt = run_island_area_audit(
                IslandAreaConfig(
                    output=Path(temporary),
                    resonance_manifest=reference_manifest,
                    radial_cells=int(saved["radial_cells"]),
                    angular_cells=int(saved["angular_cells"]),
                    steps=int(saved["steps"]),
                    action_margin=float(saved["action_margin"]),
                    batch_size=int(saved["batch_size"]),
                    gauge_scales=tuple(saved["gauge_scales"]),
                    gauge_phases=tuple(saved["gauge_phases"]),
                    libration_span_limit=float(saved["libration_span_limit"]),
                )
            )
            np.testing.assert_allclose(
                rebuilt["ensemble"]["consensus_area"],
                payload["ensemble"]["consensus_area"],
                rtol=5e-4,
                atol=5e-5,
            )
            np.testing.assert_allclose(
                rebuilt["controls"]["exact_gauge_stress"][
                    "maximum_relative_area_shift"
                ],
                payload["controls"]["exact_gauge_stress"][
                    "maximum_relative_area_shift"
                ],
                rtol=5e-3,
                atol=5e-6,
            )
            if rebuilt["status"] != payload["status"]:
                raise ValueError("fresh island-area run changed the claim state")
            checks.append("fresh mesh rerun reproduced the consensus and status")
    print("; ".join(checks))


if __name__ == "__main__":
    main()
