from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from learned_koopman.canonical_experiment import validate_canonical_manifest
from learned_koopman.canonical_model import load_canonical_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a canonical Koopman manifest and its live model."
    )
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=Path("results/koopman-hj/manifest.json"),
    )
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    checks = validate_canonical_manifest(payload)
    assert checks == payload["validation_checks"], "validation check list is stale"
    source = Path(payload["dataset"]["source"])
    assert source.is_file(), f"source dataset is missing: {source}"
    assert _sha256(source) == payload["dataset"]["source_sha256"], (
        "source dataset fingerprint is stale"
    )
    artifacts = payload["artifacts"]
    for name in ("model", "overview", "report", "action_audit_manifest"):
        artifact = args.manifest.parent / artifacts[name]
        assert artifact.is_file(), f"{name} artifact is missing: {artifact}"
        assert _sha256(artifact) == artifacts[f"{name}_sha256"], (
            f"{name} artifact fingerprint is stale"
        )
    model = load_canonical_model(args.manifest.parent / artifacts["model"])
    support_midpoint = 0.5 * (
        payload["model"]["action_support"][0]
        + payload["model"]["action_support"][1]
    )
    initial = np.array([np.sqrt(max(2.0 * support_midpoint, 0.1)), 0.0])
    prediction = model.rollout(initial, steps=200, allow_extrapolation=True)
    assert prediction.shape == (200, 2)
    assert np.isfinite(prediction).all()
    action = model.coordinate(prediction)
    relative_action_drift = float(
        np.max(np.abs(action - action[0])) / max(abs(float(action[0])), 1e-8)
    )
    assert relative_action_drift < 2e-4, "loaded model does not conserve latent action"
    revision = payload["source_revision"]
    if revision["git_commit"] and revision["git_worktree_clean"] is True:
        root = Path(__file__).resolve().parents[1]
        subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                str(revision["git_commit"]),
                "HEAD",
                "--",
                "pyproject.toml",
                "src",
            ],
            cwd=root,
            check=True,
        )
    print(
        "canonical Koopman artifact verified: "
        f"{payload['certificate']['status']}; "
        f"relative 200-step action drift {relative_action_drift:.3e}"
    )


if __name__ == "__main__":
    main()
