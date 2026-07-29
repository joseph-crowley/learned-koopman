from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from learned_koopman.operator_family import fit_fibered_operator
from learned_koopman.trajectory import load_trajectory_csv
from learned_koopman.workbench import (
    load_mechanics_model,
    validate_workbench_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results/mechanics-workbench/manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_valid_time(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    dt: float,
) -> float:
    errors = np.sqrt(np.mean(np.square(prediction - truth), axis=-1))
    times = []
    for row in errors:
        failures = np.flatnonzero(row > 0.5)
        times.append((int(failures[0]) if len(failures) else len(row) - 1) * dt)
    return float(np.mean(times))


def _errors(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    scale: np.ndarray,
    columns: tuple[str, ...],
    dt: float,
) -> dict[str, object]:
    per_state = np.sqrt(
        np.mean(np.square((prediction - truth) * scale), axis=(0, 1))
    )
    return {
        "normalized_rollout_rmse": float(
            np.sqrt(np.mean(np.square(prediction - truth)))
        ),
        "per_state_rmse": {
            column: float(value)
            for column, value in zip(columns, per_state, strict=True)
        },
        "mean_valid_time": _mean_valid_time(prediction, truth, dt=dt),
    }


def _assert_close_tree(
    measured: dict[str, object],
    stored: dict[str, object],
    *,
    model_name: str,
) -> None:
    for key, value in measured.items():
        expected = stored[key]
        if isinstance(value, dict):
            assert isinstance(expected, dict)
            _assert_close_tree(
                value,
                expected,
                model_name=f"{model_name}.{key}",
            )
        else:
            assert np.isclose(float(value), float(expected), rtol=1e-6, atol=1e-9), (
                f"{model_name}.{key} disagrees with source data and model bundle: "
                f"reconstructed={value}, stored={expected}"
            )


def _reconstruct_errors(
    payload: dict[str, object],
    *,
    source: Path,
    model_path: Path,
) -> None:
    dataset_info = payload["dataset"]
    assert isinstance(dataset_info, dict)
    reference = payload["reference_evaluation"]
    reference_column = reference["column"] if isinstance(reference, dict) else None
    columns = tuple(str(value) for value in dataset_info["state_columns"])
    dataset = load_trajectory_csv(
        source,
        state_columns=columns,
        trajectory_column=str(dataset_info["trajectory_column"]),
        time_column=str(dataset_info["time_column"]),
        reference_column=str(reference_column) if reference_column else None,
    )
    split = payload["split"]
    assert isinstance(split, dict)
    id_to_index = {
        trajectory_id: index
        for index, trajectory_id in enumerate(dataset.trajectory_ids)
    }
    train_indices = np.asarray(
        [id_to_index[str(value)] for value in split["training_trajectory_ids"]],
        dtype=int,
    )
    test_indices = np.asarray(
        [id_to_index[str(value)] for value in split["held_out_trajectory_ids"]],
        dtype=int,
    )
    model = load_mechanics_model(model_path)
    certificate = payload["certificate"]
    assert isinstance(certificate, dict)
    assert model.certificate_status == certificate["status"], (
        "model bundle certificate disagrees with manifest"
    )
    assert model.decisive_comparisons == certificate["decisive_comparisons"], (
        "model bundle comparisons disagree with manifest"
    )

    normalized = (dataset.states - model.state_mean) / model.state_scale
    train_coordinates = model.coordinate(dataset.states[train_indices]).mean(axis=1)
    operator_info = payload["operator_family"]
    assert isinstance(operator_info, dict)
    model_info = operator_info["model"]
    assert isinstance(model_info, dict)
    config = payload["config"]
    assert isinstance(config, dict)
    global_edmd = fit_fibered_operator(
        normalized[train_indices],
        train_coordinates,
        dt=dataset.dt,
        family_degree=0,
        observable_degree=int(model_info["observable_degree"]),
        ridge=float(config["ridge"]),
    )
    truth = normalized[test_indices]
    test_coordinates = model.coordinate(dataset.states[test_indices, 0])
    predictions = {
        "fibered": model.operator.rollout(
            truth[:, 0],
            test_coordinates,
            steps=dataset.step_count,
        ),
        "global_edmd": global_edmd.rollout(
            truth[:, 0],
            test_coordinates,
            steps=dataset.step_count,
        ),
        "persistence": np.repeat(truth[:, :1], dataset.step_count, axis=1),
    }
    stored_errors = operator_info["held_out_errors"]
    assert isinstance(stored_errors, dict)
    for name, prediction in predictions.items():
        measured = _errors(
            prediction,
            truth,
            scale=model.state_scale,
            columns=columns,
            dt=dataset.dt,
        )
        if name != "persistence":
            repeated = np.repeat(
                test_coordinates[:, None],
                dataset.step_count - 1,
                axis=1,
            )
            fitted = model.operator if name == "fibered" else global_edmd
            one_step = fitted.predict_one_step(truth[:, :-1], repeated)
            measured["normalized_one_step_rmse"] = float(
                np.sqrt(np.mean(np.square(one_step - truth[:, 1:])))
            )
        stored = stored_errors[name]
        assert isinstance(stored, dict)
        _assert_close_tree(measured, stored, model_name=name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a Koopman mechanics-workbench manifest.",
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    arguments = parser.parse_args()
    payload = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    checks = validate_workbench_manifest(payload)
    source = Path(payload["dataset"]["source"])
    source = source if source.is_absolute() else ROOT / source
    assert source.is_file(), f"trajectory source is unavailable: {source}"
    assert _sha256(source) == payload["dataset"]["source_sha256"], (
        "trajectory source fingerprint is stale"
    )
    checks.append("trajectory source matches its recorded SHA-256")
    artifacts = payload["artifacts"]
    for name in ("model", "overview", "report"):
        artifact = arguments.manifest.parent / artifacts[name]
        assert artifact.is_file(), f"{name} artifact is missing: {artifact}"
        assert _sha256(artifact) == artifacts[f"{name}_sha256"], (
            f"{name} artifact fingerprint is stale"
        )
    checks.append("report, figure, and model match their recorded SHA-256 values")
    model_path = arguments.manifest.parent / artifacts["model"]
    _reconstruct_errors(payload, source=source, model_path=model_path)
    checks.append("held-out metrics reconstruct from source data and model bundle")
    revision = payload["source_revision"]
    if revision["git_commit"] and revision["git_worktree_clean"] is True:
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
            cwd=ROOT,
            check=True,
        )
        checks.append("recorded clean source revision matches current model code")
    if arguments.manifest.resolve() == DEFAULT_MANIFEST.resolve():
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        invariant = payload["invariant"]
        errors = payload["operator_family"]["held_out_errors"]
        claims = (
            f"{invariant['held_out_mean_normalized_drift']:.4f}",
            f"{errors['fibered']['normalized_rollout_rmse']:.4f}",
            f"{errors['global_edmd']['normalized_rollout_rmse']:.4f}",
            f"{errors['persistence']['normalized_rollout_rmse']:.4f}",
        )
        for claim in claims:
            assert claim in readme, f"README workbench claim is stale: {claim}"
        checks.append("README workbench values match the committed manifest")
    print("Mechanics workbench is internally coherent:")
    for check in checks:
        print(f"- {check}")


if __name__ == "__main__":
    main()
