from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import TensorDataset

from learned_koopman.config import ExperimentConfig
from learned_koopman.physics import pendulum_energy_from_state, pendulum_frequency, simulate


@dataclass(frozen=True)
class EvaluationTrajectory:
    amplitude: float
    states: np.ndarray
    energy: float


def training_dataset(config: ExperimentConfig) -> TensorDataset:
    """Create state and action-angle windows from complete trajectories."""

    amplitude_edges = np.linspace(0.15, 2.85, config.train_amplitudes + 1)
    amplitudes = 0.5 * (amplitude_edges[:-1] + amplitude_edges[1:])
    frequencies = pendulum_frequency(amplitudes)
    states, _, _ = simulate(
        amplitudes,
        np.zeros_like(amplitudes),
        steps=config.train_steps,
        dt=config.dt,
    )
    windows: list[np.ndarray] = []
    phase_windows: list[np.ndarray] = []
    frequency_targets: list[float] = []
    width = config.horizon + 1
    time = np.arange(config.train_steps) * config.dt
    for trajectory, frequency in zip(states, frequencies, strict=True):
        phase = np.stack(
            (np.cos(frequency * time), np.sin(frequency * time)),
            axis=-1,
        )
        for start in range(0, config.train_steps - width, config.window_stride):
            windows.append(trajectory[start : start + width])
            phase_windows.append(phase[start : start + width])
            frequency_targets.append(float(frequency))
    tensor = torch.tensor(np.stack(windows), dtype=torch.float32)
    phase_tensor = torch.tensor(np.stack(phase_windows), dtype=torch.float32)
    frequency_tensor = torch.tensor(frequency_targets, dtype=torch.float32).unsqueeze(-1)
    return TensorDataset(tensor, phase_tensor, frequency_tensor)


def evaluation_trajectories(config: ExperimentConfig) -> list[EvaluationTrajectory]:
    amplitudes = np.array([0.25, 1.0, 2.0, 2.8, 3.05], dtype=np.float64)
    states, _, _ = simulate(
        amplitudes,
        np.zeros_like(amplitudes),
        steps=config.rollout_steps + 1,
        dt=config.dt,
    )
    return [
        EvaluationTrajectory(
            amplitude=float(amplitude),
            states=trajectory,
            energy=float(pendulum_energy_from_state(trajectory[0])),
        )
        for amplitude, trajectory in zip(amplitudes, states, strict=True)
    ]
