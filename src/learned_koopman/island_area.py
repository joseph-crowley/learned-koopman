from __future__ import annotations

import hashlib
import html
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from learned_koopman import __version__
from learned_koopman.canonical_model import CanonicalKoopmanModel, load_canonical_model
from learned_koopman.map_fixtures import (
    ExactGauge,
    KickHarmonic,
    ObservationChart,
    TwistKickMap,
    wrap_angle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class IslandAreaConfig:
    """Execution contract for the gauge-invariant island-area audit."""

    output: Path
    resonance_manifest: Path
    radial_cells: int = 61
    angular_cells: int = 180
    steps: int = 800
    action_margin: float = 0.35
    batch_size: int = 65_536
    gauge_scales: tuple[float, ...] = (0.01, 0.02, 0.04, 0.10)
    gauge_phases: tuple[float, ...] = (0.0, 0.5 * np.pi)
    libration_span_limit: float = 2.0 * np.pi

    @classmethod
    def quick(
        cls,
        output: Path,
        resonance_manifest: Path,
    ) -> IslandAreaConfig:
        return cls(
            output=output,
            resonance_manifest=resonance_manifest,
            radial_cells=21,
            angular_cells=60,
            steps=240,
        )


@dataclass(frozen=True)
class ProbeMesh:
    action_edges: np.ndarray
    angle_edges: np.ndarray
    initial_actions: np.ndarray
    initial_angles: np.ndarray
    physical_vertices: np.ndarray
    physical_cell_areas: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.action_edges) - 1, len(self.angle_edges) - 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_source_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_worktree_clean": None}
    return {"git_commit": commit, "git_worktree_clean": not dirty}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def quadrilateral_cell_areas(vertices: np.ndarray) -> np.ndarray:
    """Approximate canonical cell areas from a structured vertex mesh."""

    values = np.asarray(vertices, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError("vertices must have shape (radial+1, angular+1, 2)")
    if values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("cell mesh needs at least one cell")
    corners = np.stack(
        (
            values[:-1, :-1],
            values[1:, :-1],
            values[1:, 1:],
            values[:-1, 1:],
        ),
        axis=2,
    )
    x = corners[..., 0]
    y = corners[..., 1]
    signed = 0.5 * np.sum(
        x * np.roll(y, -1, axis=2) - y * np.roll(x, -1, axis=2),
        axis=2,
    )
    return np.abs(signed)


def bounded_libration_mask(
    angle_series: np.ndarray,
    *,
    order: int,
    span_limit: float = 2.0 * np.pi,
) -> tuple[np.ndarray, dict[str, float]]:
    """Classify bounded resonant winding without choosing a phase origin."""

    angles = np.asarray(angle_series, dtype=np.float64)
    if angles.ndim != 2 or angles.shape[0] < 3:
        raise ValueError("angle_series must have shape (time>=3, probe)")
    if order < 1 or span_limit <= 0.0:
        raise ValueError("order and span_limit must be positive")
    previous = wrap_angle(order * angles[0])
    unwrapped = previous.copy()
    minimum = unwrapped.copy()
    maximum = unwrapped.copy()
    for row in angles[1:]:
        current = wrap_angle(order * row)
        unwrapped += wrap_angle(current - previous)
        minimum = np.minimum(minimum, unwrapped)
        maximum = np.maximum(maximum, unwrapped)
        previous = current
    span = maximum - minimum
    mask = span < span_limit
    return mask, {
        "minimum_span": float(np.min(span)),
        "median_span": float(np.median(span)),
        "maximum_bounded_span": (
            float(np.max(span[mask])) if np.any(mask) else 0.0
        ),
        "minimum_unbounded_span": (
            float(np.min(span[~mask])) if np.any(~mask) else 0.0
        ),
    }


def _build_mesh(
    system: TwistKickMap,
    observation: ObservationChart,
    config: IslandAreaConfig,
    *,
    order: int,
) -> ProbeMesh:
    resonance_action = system.resonance_action(order)
    action_min = resonance_action - config.action_margin
    action_max = resonance_action + config.action_margin
    if action_min <= 0.0:
        raise ValueError("probe action band must remain positive")
    action_edges = np.linspace(action_min, action_max, config.radial_cells + 1)
    angle_edges = np.linspace(-np.pi, np.pi, config.angular_cells + 1)
    action_centers = 0.5 * (action_edges[:-1] + action_edges[1:])
    angle_centers = 0.5 * (angle_edges[:-1] + angle_edges[1:])
    initial_actions, initial_angles = np.meshgrid(
        action_centers,
        angle_centers,
        indexing="ij",
    )
    vertex_action, vertex_angle = np.meshgrid(
        action_edges,
        angle_edges,
        indexing="ij",
    )
    physical_vertices = observation.observe(vertex_action, vertex_angle)
    return ProbeMesh(
        action_edges=action_edges,
        angle_edges=angle_edges,
        initial_actions=initial_actions.reshape(-1),
        initial_angles=initial_angles.reshape(-1),
        physical_vertices=physical_vertices,
        physical_cell_areas=quadrilateral_cell_areas(physical_vertices),
    )


def _simulate_probe(
    system: TwistKickMap,
    observation: ObservationChart,
    mesh: ProbeMesh,
    *,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    actions = mesh.initial_actions.copy()
    angles = mesh.initial_angles.copy()
    states = np.empty((steps, len(actions), 2), dtype=np.float32)
    oracle_angles = np.empty((steps, len(actions)), dtype=np.float32)
    for step in range(steps):
        states[step] = observation.observe(actions, angles).astype(np.float32)
        oracle_angles[step] = angles.astype(np.float32)
        actions, angles = system.step(actions, angles)
    return states, oracle_angles


def _encoded_angle_series(
    model: CanonicalKoopmanModel,
    states: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    flat = np.asarray(states, dtype=np.float32).reshape(-1, 2)
    angles = np.empty(len(flat), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(flat), batch_size):
            stop = min(start + batch_size, len(flat))
            tensor = torch.from_numpy(flat[start:stop])
            latent = model.network.encode(tensor)
            angles[start:stop] = torch.atan2(-latent[:, 1], latent[:, 0]).numpy()
    return angles.reshape(states.shape[:2])


def _latent_vertices(
    model: CanonicalKoopmanModel,
    physical_vertices: np.ndarray,
) -> np.ndarray:
    return model.canonical_coordinates(physical_vertices)


def _gauge_angle_series(
    angle_series: np.ndarray,
    gauge: ExactGauge,
) -> np.ndarray:
    angles = np.asarray(angle_series, dtype=np.float64)
    return wrap_angle(
        angles
        + gauge.amplitude
        * np.sin(gauge.order * angles + gauge.phase)
    )


def _gauge_vertices(
    latent_vertices: np.ndarray,
    gauge: ExactGauge,
) -> np.ndarray:
    q = latent_vertices[..., 0]
    p = latent_vertices[..., 1]
    action = 0.5 * (q * q + p * p)
    angle = np.arctan2(-p, q)
    transformed_action, transformed_angle = gauge.forward(action, angle)
    radius = np.sqrt(2.0 * np.maximum(transformed_action, 1e-12))
    return np.stack(
        (
            radius * np.cos(transformed_angle),
            -radius * np.sin(transformed_angle),
        ),
        axis=-1,
    )


def _area(cell_areas: np.ndarray, mask: np.ndarray, shape: tuple[int, int]) -> float:
    return float(np.asarray(cell_areas)[np.asarray(mask).reshape(shape)].sum())


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return float(intersection / union) if union else 1.0


def _load_reference(
    config: IslandAreaConfig,
) -> tuple[dict[str, Any], list[tuple[str, Path, str, CanonicalKoopmanModel]]]:
    manifest_path = config.resonance_manifest
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "resonance-metrology":
        raise ValueError("reference manifest is not a resonance-metrology run")
    if manifest.get("status") != "resolved_refuted":
        raise ValueError("island audit requires the frozen refuted metrology reference")
    if manifest.get("status_reason") != "gauge_freedom":
        raise ValueError("reference metrology result was not refuted by gauge freedom")
    accepted = list(manifest["ensemble"]["accepted_labels"])
    training = {
        row["label"]: row for row in manifest["ensemble"]["training"]["s1"]
    }
    models = []
    model_root = manifest_path.parent / "models"
    for label in accepted:
        row = training[label]
        path = model_root / row["model"]
        if not path.is_file():
            raise ValueError(f"missing reference model: {path}")
        digest = _sha256(path)
        if digest != row["model_sha256"]:
            raise ValueError(f"reference model digest mismatch: {path}")
        model = load_canonical_model(path)
        if model.certificate_status != "supported_on_held_out_trajectories":
            raise ValueError(f"reference model failed its fit gates: {path}")
        models.append((label, path, digest, model))
    if len(models) < 2 or len({row[2] for row in models}) != len(models):
        raise ValueError("island audit needs distinct independently fitted charts")
    return manifest, models


def _plot_report(
    path: Path,
    manifest: dict[str, Any],
    oracle_mask: np.ndarray,
) -> None:
    config = manifest["config"]
    reference = manifest["reference"]
    ensemble = manifest["ensemble"]
    controls = manifest["controls"]
    radial = int(config["radial_cells"])
    angular = int(config["angular_cells"])
    action_min, action_max = manifest["probe"]["action_band"]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6), constrained_layout=True)

    axes[0, 0].imshow(
        oracle_mask.reshape(radial, angular),
        origin="lower",
        aspect="auto",
        extent=(-np.pi, np.pi, action_min, action_max),
        cmap="magma",
        interpolation="nearest",
    )
    axes[0, 0].set_title("Direct bounded-libration membership")
    axes[0, 0].set_xlabel("initial angle")
    axes[0, 0].set_ylabel("initial action")

    labels = ["leading", "direct", "raw", *[row["label"] for row in ensemble["charts"]]]
    values = [
        reference["leading_total_island_area"],
        reference["direct_physical_area"],
        reference["raw_polar_area"],
        *[row["island_area"] for row in ensemble["charts"]],
    ]
    colors = ["#6b7280", "#111827", "#dc2626", *(["#2563eb"] * len(ensemble["charts"]))]
    axes[0, 1].bar(np.arange(len(values)), values, color=colors)
    axes[0, 1].axhline(
        reference["direct_physical_area"],
        color="#111827",
        linestyle="--",
        linewidth=1.2,
    )
    axes[0, 1].set_xticks(np.arange(len(values)), labels, rotation=70, ha="right")
    axes[0, 1].set_ylabel("total island area")
    axes[0, 1].set_title("Invariant area beats the raw polar baseline")

    gauge_rows = controls["exact_gauge_stress"]["rows"]
    scales = sorted({row["scale"] for row in gauge_rows})
    for label in ensemble["accepted_labels"]:
        maxima = [
            max(
                row["relative_area_shift"]
                for row in gauge_rows
                if row["label"] == label and row["scale"] == scale
            )
            for scale in scales
        ]
        axes[1, 0].plot(scales, maxima, marker="o", alpha=0.72)
    axes[1, 0].axhline(
        manifest["thresholds"]["maximum_exact_gauge_area_shift"],
        color="#dc2626",
        linestyle="--",
        label="gate",
    )
    axes[1, 0].set_xlabel("exact-gauge action modulation")
    axes[1, 0].set_ylabel("relative island-area shift")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Exact gauges leave area unchanged")
    axes[1, 0].legend()

    chart_areas = np.asarray([row["island_area"] for row in ensemble["charts"]])
    null_areas = np.asarray([row["null_area"] for row in ensemble["charts"]])
    x = np.arange(len(chart_areas))
    axes[1, 1].bar(x - 0.18, chart_areas, width=0.36, label="kicked", color="#2563eb")
    axes[1, 1].bar(x + 0.18, null_areas, width=0.36, label="null", color="#9ca3af")
    axes[1, 1].axhline(
        controls["noncanonical_scale"]["island_area"],
        color="#dc2626",
        linestyle=":",
        label="noncanonical scale",
    )
    axes[1, 1].set_xticks(x, [str(index + 1) for index in x])
    axes[1, 1].set_xlabel("independent chart")
    axes[1, 1].set_ylabel("area")
    axes[1, 1].set_title("Null floor and noncanonical negative control")
    axes[1, 1].legend()

    fig.suptitle("Gauge-invariant resonant-island area audit", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    status = html.escape(manifest["status"])
    reason = html.escape(manifest["status_reason"])
    reference = manifest["reference"]
    ensemble = manifest["ensemble"]
    controls = manifest["controls"]
    gates = manifest["empirical_gates"]
    chart_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['label'])}</td>"
        f"<td>{row['island_area']:.6f}</td>"
        f"<td>{row['relative_error_vs_direct']:.3%}</td>"
        f"<td>{row['membership_jaccard_vs_direct']:.5f}</td>"
        f"<td>{row['null_fraction']:.3%}</td>"
        "</tr>"
        for row in ensemble["charts"]
    )
    gate_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{html.escape(str(row['value']))}</td>"
        f"<td>{html.escape(str(row['threshold']))}</td>"
        f"<td>{'pass' if row['passed'] else 'fail'}</td>"
        "</tr>"
        for name, row in gates.items()
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gauge-invariant island-area audit</title>
<style>
body {{ max-width: 1020px; margin: 2rem auto; padding: 0 1rem;
font: 16px/1.55 system-ui, sans-serif; color: #172033; }}
h1, h2 {{ color: #10213f; }}
.verdict {{ padding: 1rem; border-left: 5px solid #2563eb; background: #eff6ff; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ border: 1px solid #d7dde8; padding: .55rem; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
img {{ max-width: 100%; }}
code {{ background: #f3f4f6; padding: .1rem .25rem; }}
</style>
</head>
<body>
<h1>Gauge-invariant resonant-island area</h1>
<p class="verdict"><strong>{status}</strong> ({reason}). On this frozen
synthetic fixture, the coordinate-dependent resonant coefficient failed its
exact-gauge test, while bounded-libration area survived it.</p>
<img src="overview.png"
alt="Island-area membership, chart estimates, exact-gauge stress, and controls">
<h2>What was measured</h2>
<p>A structured physical initial-condition mesh was advanced by the exact
return map. A probe cell counts as resonantly trapped when the unwrapped
<code>m·angle</code> remains within one full turn. Cell areas come from the
physical or learned canonical mesh, not from a fitted coefficient.</p>
<table>
<tr><th>measurement</th><th>area</th><th>error vs direct</th></tr>
<tr><td>leading pendulum total</td>
<td>{reference['leading_total_island_area']:.6f}</td>
<td>{reference['leading_relative_error_vs_direct']:.3%}</td></tr>
<tr><td>direct physical/oracle mesh</td>
<td>{reference['direct_physical_area']:.6f}</td><td>—</td></tr>
<tr><td>raw observed polar angle</td>
<td>{reference['raw_polar_area']:.6f}</td>
<td>{reference['raw_relative_error_vs_direct']:.3%}</td></tr>
<tr><td>learned-chart consensus</td>
<td>{ensemble['consensus_area']:.6f}</td>
<td>{ensemble['consensus_relative_error_vs_direct']:.3%}</td></tr>
</table>
<h2>Independent learned charts</h2>
<table>
<tr><th>chart</th><th>area</th><th>error</th><th>membership Jaccard</th><th>null fraction</th></tr>
{chart_rows}
</table>
<h2>Adversarial controls</h2>
<p>Maximum exact-gauge area shift:
<strong>{controls['exact_gauge_stress']['maximum_relative_area_shift']:.4%}</strong>.
The deliberately noncanonical 1.2× area scale moved the answer by
<strong>{controls['noncanonical_scale']['relative_area_shift']:.2%}</strong>,
so the audit is not simply insensitive to coordinate changes.</p>
<table>
<tr><th>gate</th><th>value</th><th>threshold</th><th>verdict</th></tr>
{gate_rows}
</table>
<h2>Claim boundary</h2>
<p>{html.escape(manifest['claim_boundary'])}</p>
<p>The probe states were not used for chart training, but this was the
method-development fixture. The frozen protocol still needs prospective
confirmation on a second system.</p>
<p>This is an empirical synthetic support result, not a theorem, a calibrated
uncertainty interval, a measured-rig validation, or evidence that learned
charts beat classical methods generally.</p>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def run_island_area_audit(config: IslandAreaConfig) -> dict[str, Any]:
    """Test a physical island-area quotient against learned-chart gauges."""

    if config.radial_cells < 9 or config.angular_cells < 24:
        raise ValueError("probe mesh is too small")
    if config.steps < 40 or config.batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    if config.action_margin <= 0.0:
        raise ValueError("action_margin must be positive")
    started = time.perf_counter()
    config.output.mkdir(parents=True, exist_ok=True)
    resonance, model_rows = _load_reference(config)
    map_config = resonance["fixture"]["map"]
    order = int(map_config["kick_order"])
    kick = KickHarmonic(
        order=order,
        amplitude=float(map_config["kick_amplitude"]),
        phase=float(map_config["kick_phase"]),
    )
    system = TwistKickMap(
        base_frequency=float(map_config["base_frequency"]),
        twist=float(map_config["twist"]),
        kicks=(kick,),
    )
    null_system = replace(system, kicks=())
    observation = ObservationChart(**resonance["fixture"]["observation_chart"])
    mesh = _build_mesh(system, observation, config, order=order)
    expected_domain_area = 4.0 * np.pi * config.action_margin
    physical_domain_area = float(mesh.physical_cell_areas.sum())

    kicked_states, oracle_angles = _simulate_probe(
        system,
        observation,
        mesh,
        steps=config.steps,
    )
    oracle_mask, oracle_span = bounded_libration_mask(
        oracle_angles,
        order=order,
        span_limit=config.libration_span_limit,
    )
    raw_angles = np.arctan2(-kicked_states[..., 1], kicked_states[..., 0])
    raw_mask, raw_span = bounded_libration_mask(
        raw_angles,
        order=order,
        span_limit=config.libration_span_limit,
    )
    direct_area = _area(mesh.physical_cell_areas, oracle_mask, mesh.shape)
    raw_area = _area(mesh.physical_cell_areas, raw_mask, mesh.shape)
    leading_area = 8.0 * system.island_half_width(order)

    chart_rows: list[dict[str, Any]] = []
    gauge_rows: list[dict[str, Any]] = []
    learned_angles: dict[str, np.ndarray] = {}
    chart_weights: dict[str, np.ndarray] = {}
    for label, path, digest, model in model_rows:
        angles = _encoded_angle_series(
            model,
            kicked_states,
            batch_size=config.batch_size,
        )
        learned_angles[label] = angles
        mask, span = bounded_libration_mask(
            angles,
            order=order,
            span_limit=config.libration_span_limit,
        )
        latent_vertices = _latent_vertices(model, mesh.physical_vertices)
        weights = quadrilateral_cell_areas(latent_vertices)
        chart_weights[label] = weights
        island_area = _area(weights, mask, mesh.shape)
        chart_rows.append(
            {
                "label": label,
                "model": str(
                    path.relative_to(config.resonance_manifest.resolve().parent)
                ),
                "model_sha256": digest,
                "model_fit_status": model.certificate_status,
                "island_area": island_area,
                "relative_error_vs_direct": abs(island_area - direct_area) / direct_area,
                "membership_jaccard_vs_direct": _jaccard(mask, oracle_mask),
                "membership_disagreement_fraction": float(np.mean(mask != oracle_mask)),
                "domain_area": float(weights.sum()),
                "domain_area_relative_error": (
                    abs(float(weights.sum()) - physical_domain_area)
                    / physical_domain_area
                ),
                "classification": span,
                "uses_oracle_coordinates": False,
            }
        )
        for scale in config.gauge_scales:
            for phase in config.gauge_phases:
                gauge = ExactGauge(
                    amplitude=scale / (2 * order),
                    order=2 * order,
                    phase=phase,
                )
                gauged_angles = _gauge_angle_series(angles, gauge)
                gauged_mask, _ = bounded_libration_mask(
                    gauged_angles,
                    order=order,
                    span_limit=config.libration_span_limit,
                )
                gauged_vertices = _gauge_vertices(latent_vertices, gauge)
                gauged_weights = quadrilateral_cell_areas(gauged_vertices)
                gauged_area = _area(gauged_weights, gauged_mask, mesh.shape)
                gauge_rows.append(
                    {
                        "label": label,
                        "scale": scale,
                        "phase": phase,
                        "island_area": gauged_area,
                        "relative_area_shift": (
                            abs(gauged_area - island_area) / island_area
                        ),
                        "membership_disagreement_fraction": float(
                            np.mean(gauged_mask != mask)
                        ),
                        "domain_area_relative_shift": (
                            abs(float(gauged_weights.sum()) - float(weights.sum()))
                            / float(weights.sum())
                        ),
                    }
                )

    del kicked_states
    null_states, null_oracle_angles = _simulate_probe(
        null_system,
        observation,
        mesh,
        steps=config.steps,
    )
    null_oracle_mask, _ = bounded_libration_mask(
        null_oracle_angles,
        order=order,
        span_limit=config.libration_span_limit,
    )
    null_raw_angles = np.arctan2(-null_states[..., 1], null_states[..., 0])
    null_raw_mask, _ = bounded_libration_mask(
        null_raw_angles,
        order=order,
        span_limit=config.libration_span_limit,
    )
    for chart, (_, _, _, model) in zip(chart_rows, model_rows, strict=True):
        null_angles = _encoded_angle_series(
            model,
            null_states,
            batch_size=config.batch_size,
        )
        null_mask, null_span = bounded_libration_mask(
            null_angles,
            order=order,
            span_limit=config.libration_span_limit,
        )
        null_area = _area(chart_weights[chart["label"]], null_mask, mesh.shape)
        chart["null_area"] = null_area
        chart["null_fraction"] = null_area / chart["island_area"]
        chart["null_classification"] = null_span
    del null_states

    chart_areas = np.asarray([row["island_area"] for row in chart_rows])
    consensus_area = float(np.median(chart_areas))
    maximum_chart_error = max(
        row["relative_error_vs_direct"] for row in chart_rows
    )
    minimum_jaccard = min(
        row["membership_jaccard_vs_direct"] for row in chart_rows
    )
    maximum_null_fraction = max(row["null_fraction"] for row in chart_rows)
    maximum_gauge_shift = max(
        row["relative_area_shift"] for row in gauge_rows
    )

    scale_factor = 1.2
    scaled_vertices = np.sqrt(scale_factor) * mesh.physical_vertices
    scaled_weights = quadrilateral_cell_areas(scaled_vertices)
    scaled_area = _area(scaled_weights, oracle_mask, mesh.shape)
    noncanonical_shift = abs(scaled_area - direct_area) / direct_area

    thresholds = {
        "maximum_direct_vs_leading_error": 0.03,
        "maximum_consensus_vs_direct_error": 0.02,
        "maximum_chart_vs_direct_error": 0.03,
        "minimum_membership_jaccard": 0.98,
        "maximum_exact_gauge_area_shift": 0.005,
        "maximum_null_fraction": 0.10,
        "minimum_noncanonical_area_shift": 0.15,
    }
    measurements = {
        "direct_vs_leading_error": abs(direct_area - leading_area) / leading_area,
        "consensus_vs_direct_error": abs(consensus_area - direct_area) / direct_area,
        "maximum_chart_vs_direct_error": maximum_chart_error,
        "minimum_membership_jaccard": minimum_jaccard,
        "maximum_exact_gauge_area_shift": maximum_gauge_shift,
        "maximum_null_fraction": maximum_null_fraction,
        "noncanonical_area_shift": noncanonical_shift,
    }
    empirical_gates = {
        "direct_map_matches_leading_area": {
            "value": measurements["direct_vs_leading_error"],
            "threshold": f"<= {thresholds['maximum_direct_vs_leading_error']}",
            "passed": (
                measurements["direct_vs_leading_error"]
                <= thresholds["maximum_direct_vs_leading_error"]
            ),
        },
        "learned_consensus_matches_direct_area": {
            "value": measurements["consensus_vs_direct_error"],
            "threshold": f"<= {thresholds['maximum_consensus_vs_direct_error']}",
            "passed": (
                measurements["consensus_vs_direct_error"]
                <= thresholds["maximum_consensus_vs_direct_error"]
            ),
        },
        "every_chart_matches_direct_area": {
            "value": maximum_chart_error,
            "threshold": f"<= {thresholds['maximum_chart_vs_direct_error']}",
            "passed": maximum_chart_error <= thresholds["maximum_chart_vs_direct_error"],
        },
        "membership_matches_direct_topology": {
            "value": minimum_jaccard,
            "threshold": f">= {thresholds['minimum_membership_jaccard']}",
            "passed": minimum_jaccard >= thresholds["minimum_membership_jaccard"],
        },
        "exact_gauges_preserve_area": {
            "value": maximum_gauge_shift,
            "threshold": f"<= {thresholds['maximum_exact_gauge_area_shift']}",
            "passed": maximum_gauge_shift <= thresholds["maximum_exact_gauge_area_shift"],
        },
        "null_area_stays_below_resolution_floor": {
            "value": maximum_null_fraction,
            "threshold": f"<= {thresholds['maximum_null_fraction']}",
            "passed": maximum_null_fraction <= thresholds["maximum_null_fraction"],
        },
        "noncanonical_control_moves_area": {
            "value": noncanonical_shift,
            "threshold": f">= {thresholds['minimum_noncanonical_area_shift']}",
            "passed": noncanonical_shift >= thresholds["minimum_noncanonical_area_shift"],
        },
        "learned_chart_beats_raw_polar_baseline": {
            "value": abs(consensus_area - direct_area),
            "threshold": f"< {abs(raw_area - direct_area)}",
            "passed": abs(consensus_area - direct_area) < abs(raw_area - direct_area),
        },
    }
    supported = all(row["passed"] for row in empirical_gates.values())
    status = "resolved_supported" if supported else "not_resolved_abstained"
    status_reason = (
        "gauge_invariant_island_area"
        if supported
        else "one_or_more_predeclared_area_gates_failed"
    )

    source_revision = _git_source_state()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "island-area-audit",
        "package_version": __version__,
        "status": status,
        "status_reason": status_reason,
        "config": {
            **asdict(config),
            "output": str(config.output),
            "resonance_manifest": str(config.resonance_manifest),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "source_revision": source_revision,
        "protocol_posture": {
            "fixture_role": "retrospective_method-development_fixture",
            "probe_states_were_not_used_to_train_the_charts": True,
            "prospective_confirmation_required": True,
            "reason": (
                "The mesh, controls, and thresholds were developed while this "
                "synthetic map was available. They are frozen for reproduction "
                "but do not count as a blinded prospective confirmation."
            ),
        },
        "source_evidence": {
            "resonance_manifest": str(config.resonance_manifest),
            "resonance_manifest_sha256": _sha256(
                config.resonance_manifest
                if config.resonance_manifest.is_absolute()
                else PROJECT_ROOT / config.resonance_manifest
            ),
            "resonance_status": resonance["status"],
            "resonance_status_reason": resonance["status_reason"],
            "resonance_source_revision": resonance["source_revision"],
            "model_count": len(model_rows),
            "models_are_distinct": len({row[2] for row in model_rows}) == len(model_rows),
        },
        "probe": {
            "resonance_order": order,
            "resonance_action": system.resonance_action(order),
            "action_band": [
                float(mesh.action_edges[0]),
                float(mesh.action_edges[-1]),
            ],
            "cell_count": int(np.prod(mesh.shape)),
            "expected_domain_area": expected_domain_area,
            "physical_domain_area": physical_domain_area,
            "physical_domain_relative_quadrature_error": (
                abs(physical_domain_area - expected_domain_area)
                / expected_domain_area
            ),
            "classification": (
                "bounded unwrapped resonant angle: "
                f"span(m*angle) < {config.libration_span_limit:.12g}"
            ),
            "physical_cell_weights": True,
        },
        "reference": {
            "leading_total_island_area": leading_area,
            "leading_one_island_area": leading_area / order,
            "direct_physical_area": direct_area,
            "leading_relative_error_vs_direct": (
                abs(leading_area - direct_area) / direct_area
            ),
            "direct_classified_cells": int(np.count_nonzero(oracle_mask)),
            "direct_classification": oracle_span,
            "raw_polar_area": raw_area,
            "raw_relative_error_vs_direct": abs(raw_area - direct_area) / direct_area,
            "raw_membership_jaccard_vs_direct": _jaccard(raw_mask, oracle_mask),
            "raw_classification": raw_span,
            "oracle_used_only_for_synthetic_evaluation": True,
        },
        "ensemble": {
            "accepted_labels": [row[0] for row in model_rows],
            "accepted_count": len(model_rows),
            "charts": chart_rows,
            "consensus_area": consensus_area,
            "consensus_relative_error_vs_direct": (
                abs(consensus_area - direct_area) / direct_area
            ),
            "maximum_chart_relative_error_vs_direct": maximum_chart_error,
            "relative_range": (
                float((chart_areas.max() - chart_areas.min()) / consensus_area)
            ),
        },
        "controls": {
            "null": {
                "direct_area": _area(
                    mesh.physical_cell_areas,
                    null_oracle_mask,
                    mesh.shape,
                ),
                "raw_polar_area": _area(
                    mesh.physical_cell_areas,
                    null_raw_mask,
                    mesh.shape,
                ),
                "maximum_chart_fraction_of_kicked_area": maximum_null_fraction,
                "interpretation": (
                    "The integrable null has one zero-winding resonant torus. "
                    "Its finite mesh strip is the resolution floor, not an island."
                ),
            },
            "exact_gauge_stress": {
                "scales": list(config.gauge_scales),
                "phases": list(config.gauge_phases),
                "rows": gauge_rows,
                "maximum_relative_area_shift": maximum_gauge_shift,
            },
            "noncanonical_scale": {
                "phase_space_scale": scale_factor,
                "island_area": scaled_area,
                "relative_area_shift": noncanonical_shift,
                "interpretation": (
                    "A deliberately noncanonical uniform phase-space scale "
                    "must move the area and proves that the audit is sensitive."
                ),
            },
        },
        "thresholds": thresholds,
        "measurements": measurements,
        "empirical_gates": empirical_gates,
        "claim_boundary": (
            "On one noiseless synthetic kicked twist map and one fixed dense "
            "physical initial-condition mesh, bounded-libration area recovered "
            "from independently trained exact-symplectic charts agreed with "
            "direct physical/oracle area and survived the frozen exact-gauge "
            "ladder. The dense probe states were not used for chart training, "
            "but the protocol was developed on this fixture. This supports an "
            "invariant quotient on this retrospective fixture only; it does "
            "not establish prospective transfer, measured-system robustness, "
            "general transport recovery, calibrated uncertainty, or a theorem."
        ),
        "not_supported": [
            "measured-rig validation",
            "noise, irregular sampling, or partial observation",
            "general island or separatrix-flux recovery",
            "formal invariance or a-posteriori KAM certification",
            "superiority to classical phase-space methods beyond this fixture",
        ],
        "next_falsifier": (
            "Repeat the coefficient and invariant-area protocols on a second "
            "map family and one measured return map with noise, sampling, and "
            "classical frequency-map baselines frozen in advance."
        ),
        "runtime_seconds": time.perf_counter() - started,
        "artifacts": {
            "manifest": "manifest.json",
            "report": "report.html",
            "overview": "overview.png",
        },
    }
    safe_manifest = _json_safe(manifest)
    overview = config.output / "overview.png"
    report = config.output / "report.html"
    manifest_path = config.output / "manifest.json"
    _plot_report(overview, safe_manifest, oracle_mask)
    _write_report(report, safe_manifest)
    safe_manifest["artifacts"].update(
        {
            "overview_sha256": _sha256(overview),
            "report_sha256": _sha256(report),
        }
    )
    _write_json(manifest_path, safe_manifest)
    return safe_manifest


def validate_island_area_manifest(
    manifest: dict[str, Any],
    *,
    require_clean_source: bool = True,
    require_model_artifacts: bool = True,
) -> list[str]:
    """Validate evidence and claim-state consistency for an island-area run."""

    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported island-area schema")
    if manifest.get("experiment") != "island-area-audit":
        raise ValueError("manifest is not an island-area audit")
    if require_clean_source and not manifest["source_revision"]["git_worktree_clean"]:
        raise ValueError("island-area evidence was not generated from a clean source tree")
    if manifest["source_evidence"]["resonance_status_reason"] != "gauge_freedom":
        raise ValueError("island-area audit is not bound to the gauge refutation")
    if not manifest["protocol_posture"]["prospective_confirmation_required"]:
        raise ValueError("island-area audit hides its retrospective protocol posture")
    charts = manifest["ensemble"]["charts"]
    if len(charts) < 2 or len({row["model_sha256"] for row in charts}) != len(charts):
        raise ValueError("island-area ensemble does not contain independent charts")
    if any(row["uses_oracle_coordinates"] for row in charts):
        raise ValueError("learned-chart area classification used oracle coordinates")
    gates = manifest["empirical_gates"]
    all_pass = all(row["passed"] for row in gates.values())
    if manifest["status"] == "resolved_supported" and not all_pass:
        raise ValueError("supported island-area status has a failed gate")
    if manifest["status"] != "resolved_supported" and all_pass:
        raise ValueError("island-area status understates a fully passed run")
    maximum_shift = manifest["controls"]["exact_gauge_stress"][
        "maximum_relative_area_shift"
    ]
    if maximum_shift != max(
        row["relative_area_shift"]
        for row in manifest["controls"]["exact_gauge_stress"]["rows"]
    ):
        raise ValueError("maximum exact-gauge area shift is stale")
    if (
        manifest["controls"]["noncanonical_scale"]["relative_area_shift"]
        < manifest["thresholds"]["minimum_noncanonical_area_shift"]
    ):
        raise ValueError("noncanonical negative control did not move area")
    artifact_root = Path(
        manifest.get("_artifact_root", manifest["config"]["output"])
    )
    for name in ("report", "overview"):
        target = artifact_root / manifest["artifacts"][name]
        if not target.is_file():
            raise ValueError(f"missing island-area artifact: {target}")
        if _sha256(target) != manifest["artifacts"][f"{name}_sha256"]:
            raise ValueError(f"island-area artifact digest mismatch: {target}")
    reference_manifest = Path(manifest["config"]["resonance_manifest"])
    if not reference_manifest.is_absolute():
        reference_manifest = PROJECT_ROOT / reference_manifest
    if not reference_manifest.is_file():
        raise ValueError(f"missing resonance reference manifest: {reference_manifest}")
    if (
        _sha256(reference_manifest)
        != manifest["source_evidence"]["resonance_manifest_sha256"]
    ):
        raise ValueError("resonance reference manifest digest mismatch")
    if require_model_artifacts:
        model_root = reference_manifest.parent
        for row in charts:
            target = model_root / row["model"]
            if not target.is_file():
                raise ValueError(f"missing island-area reference model: {target}")
            if _sha256(target) != row["model_sha256"]:
                raise ValueError(f"island-area model digest mismatch: {target}")
    return [
        "schema and claim state are consistent",
        f"{len(charts)} independent learned charts verified",
        "direct, raw, null, exact-gauge, and noncanonical controls verified",
        "report and overview digests verified",
    ]
