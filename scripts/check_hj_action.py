from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from learned_koopman.hj_action import validate_hj_action_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a canonical-action/Hamilton-Jacobi audit."
    )
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=Path("results/hj-action/manifest.json"),
    )
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    checks = validate_hj_action_manifest(payload)
    assert checks == payload["validation_checks"], "validation check list is stale"
    source = Path(payload["dataset"]["source"])
    assert source.is_file(), f"source dataset is missing: {source}"
    assert _sha256(source) == payload["dataset"]["source_sha256"], (
        "source dataset fingerprint is stale"
    )
    artifacts = payload["artifacts"]
    for name in ("overview", "report"):
        artifact = args.manifest.parent / artifacts[name]
        assert artifact.is_file(), f"{name} artifact is missing: {artifact}"
        assert _sha256(artifact) == artifacts[f"{name}_sha256"], (
            f"{name} artifact fingerprint is stale"
        )
    print(
        "HJ-action audit verified: "
        f"{payload['certificate']['status']}; "
        f"dH/dJ error {payload['hj_identity'].get('normalized_rmse', 'not measured')}"
    )


if __name__ == "__main__":
    main()
