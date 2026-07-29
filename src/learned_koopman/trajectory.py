from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TrajectoryDataset:
    """Uniformly sampled trajectories ready for operator learning."""

    states: np.ndarray
    times: np.ndarray
    trajectory_ids: tuple[str, ...]
    state_columns: tuple[str, ...]
    trajectory_column: str
    time_column: str
    dt: float
    source: str
    source_sha256: str
    original_lengths: tuple[int, ...]
    reference_values: np.ndarray | None = None
    reference_column: str | None = None
    reference_max_relative_drift: float | None = None

    @property
    def trajectory_count(self) -> int:
        return int(self.states.shape[0])

    @property
    def step_count(self) -> int:
        return int(self.states.shape[1])

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[2])


def _parse_finite(row: dict[str, str], column: str, row_number: int) -> float:
    raw = row.get(column)
    if raw is None or raw == "":
        raise ValueError(f"row {row_number} is missing {column!r}")
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(
            f"row {row_number} has non-numeric {column!r}: {raw!r}"
        ) from error
    if not math.isfinite(value):
        raise ValueError(f"row {row_number} has non-finite {column!r}")
    return value


def load_trajectory_csv(
    path: Path,
    *,
    state_columns: tuple[str, ...],
    trajectory_column: str = "trajectory_id",
    time_column: str = "time",
    reference_column: str | None = None,
    minimum_trajectories: int = 6,
    minimum_steps: int = 32,
    dt_relative_tolerance: float = 0.02,
) -> TrajectoryDataset:
    """Load a bundle of complete, uniformly sampled trajectories.

    Trajectories may have different lengths, but they must share a sampling
    interval. Longer runs are truncated to the shortest complete run so every
    training batch preserves trajectory membership without padding or
    imputation.
    """

    if not state_columns:
        raise ValueError("at least one state column is required")
    if len(set(state_columns)) != len(state_columns):
        raise ValueError("state columns must be unique")
    if not path.is_file():
        raise ValueError(f"trajectory CSV does not exist: {path}")

    grouped: dict[str, list[tuple[float, list[float], float | None]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("trajectory CSV is missing a header")
        required = {trajectory_column, time_column, *state_columns}
        if reference_column:
            required.add(reference_column)
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"trajectory CSV is missing columns: {', '.join(missing)}")

        for row_number, row in enumerate(reader, start=2):
            trajectory_id = (row.get(trajectory_column) or "").strip()
            if not trajectory_id:
                raise ValueError(
                    f"row {row_number} is missing {trajectory_column!r}"
                )
            time = _parse_finite(row, time_column, row_number)
            state = [
                _parse_finite(row, column, row_number) for column in state_columns
            ]
            reference = (
                _parse_finite(row, reference_column, row_number)
                if reference_column
                else None
            )
            grouped.setdefault(trajectory_id, []).append((time, state, reference))

    if len(grouped) < minimum_trajectories:
        raise ValueError(
            f"need at least {minimum_trajectories} trajectories, found {len(grouped)}"
        )

    trajectory_ids = tuple(grouped)
    original_lengths = tuple(len(grouped[key]) for key in trajectory_ids)
    common_steps = min(original_lengths)
    if common_steps < minimum_steps:
        raise ValueError(
            f"need at least {minimum_steps} samples per trajectory, found {common_steps}"
        )

    state_runs: list[np.ndarray] = []
    time_runs: list[np.ndarray] = []
    reference_runs: list[np.ndarray] = []
    local_steps: list[float] = []
    for trajectory_id in trajectory_ids:
        ordered = sorted(grouped[trajectory_id], key=lambda sample: sample[0])
        times = np.asarray([sample[0] for sample in ordered], dtype=np.float64)
        differences = np.diff(times)
        if np.any(differences <= 0.0):
            raise ValueError(
                f"trajectory {trajectory_id!r} has duplicate or decreasing times"
            )
        local_dt = float(np.median(differences))
        if not np.allclose(
            differences,
            local_dt,
            rtol=dt_relative_tolerance,
            atol=max(1e-12, abs(local_dt) * 1e-6),
        ):
            raise ValueError(
                f"trajectory {trajectory_id!r} is not uniformly sampled "
                f"within {dt_relative_tolerance:.1%}"
            )
        local_steps.append(local_dt)
        time_runs.append(times[:common_steps])
        state_runs.append(
            np.asarray(
                [sample[1] for sample in ordered[:common_steps]],
                dtype=np.float64,
            )
        )
        if reference_column:
            reference_runs.append(
                np.asarray(
                    [sample[2] for sample in ordered[:common_steps]],
                    dtype=np.float64,
                )
            )

    dt = float(np.median(local_steps))
    if not np.allclose(
        local_steps,
        dt,
        rtol=dt_relative_tolerance,
        atol=max(1e-12, abs(dt) * 1e-6),
    ):
        raise ValueError(
            "trajectories do not share a common sampling interval within "
            f"{dt_relative_tolerance:.1%}"
        )

    references = None
    reference_max_relative_drift = None
    if reference_runs:
        reference_array = np.stack(reference_runs)
        references = reference_array.mean(axis=1)
        reference_scale = max(float(np.std(references)), 1e-12)
        reference_max_relative_drift = float(
            np.max(np.std(reference_array, axis=1) / reference_scale)
        )

    return TrajectoryDataset(
        states=np.stack(state_runs),
        times=np.stack(time_runs),
        trajectory_ids=trajectory_ids,
        state_columns=state_columns,
        trajectory_column=trajectory_column,
        time_column=time_column,
        dt=dt,
        source=str(path),
        source_sha256=_sha256(path),
        original_lengths=original_lengths,
        reference_values=references,
        reference_column=reference_column,
        reference_max_relative_drift=reference_max_relative_drift,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_duffing_example(
    path: Path,
    *,
    trajectories: int = 30,
    steps: int = 360,
    dt: float = 0.025,
    cubic_stiffness: float = 0.22,
) -> Path:
    """Write conservative Duffing-oscillator trajectories as a workbench fixture."""

    if trajectories < 6:
        raise ValueError("the example needs at least six trajectories")
    if steps < 32 or dt <= 0.0:
        raise ValueError("steps must be at least 32 and dt must be positive")
    amplitudes = np.linspace(0.25, 2.1, trajectories, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("trajectory_id", "time", "position", "velocity", "energy"))
        for index, amplitude in enumerate(amplitudes):
            position = float(amplitude)
            velocity = 0.0
            for step in range(steps):
                energy = (
                    0.5 * velocity**2
                    + 0.5 * position**2
                    + 0.25 * cubic_stiffness * position**4
                )
                writer.writerow(
                    (
                        f"run-{index:03d}",
                        f"{step * dt:.12g}",
                        f"{position:.12g}",
                        f"{velocity:.12g}",
                        f"{energy:.12g}",
                    )
                )
                acceleration = -position - cubic_stiffness * position**3
                velocity_half = velocity + 0.5 * dt * acceleration
                position = position + dt * velocity_half
                acceleration_next = -position - cubic_stiffness * position**3
                velocity = velocity_half + 0.5 * dt * acceleration_next
    return path
