from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from learned_koopman.resonance_metrology import (
    MetrologyConfig,
    run_resonance_metrology,
    validate_resonance_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results" / "resonance-metrology" / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--reproduce",
        action="store_true",
        help="Rerun the saved profile and compare the main numerical result.",
    )
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    payload["_artifact_root"] = str(args.manifest.parent)
    checks = validate_resonance_manifest(payload)
    if args.reproduce:
        with tempfile.TemporaryDirectory(
            prefix="learned-koopman-metrology-reproduction-"
        ) as temporary:
            output = Path(temporary)
            saved = payload["config"]
            config = (
                MetrologyConfig.full(output)
                if saved["profile"] == "full"
                else MetrologyConfig.ci(output)
            )
            rebuilt = run_resonance_metrology(config)
            np.testing.assert_allclose(
                rebuilt["ensemble_consensus"]["coefficient"],
                payload["ensemble_consensus"]["coefficient"],
                rtol=5e-3,
                atol=2e-5,
            )
            assert rebuilt["status"] == payload["status"]
            checks.append("fresh-data rerun reproduced the consensus and status")
    print("; ".join(checks))


if __name__ == "__main__":
    main()
