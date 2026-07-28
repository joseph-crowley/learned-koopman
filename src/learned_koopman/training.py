from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from learned_koopman.config import ExperimentConfig
from learned_koopman.data import training_dataset
from learned_koopman.models import (
    EnergyConditionedRotation,
    FixedKoopmanAE,
    ResidualMLP,
    SeparatrixAtlas,
)
from learned_koopman.physics import circular_state_error, torch_energy


@dataclass
class TrainedModels:
    mlp: ResidualMLP
    fixed: FixedKoopmanAE
    conditioned: EnergyConditionedRotation
    atlas: SeparatrixAtlas | None
    histories: dict[str, list[float]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _train(
    model: nn.Module,
    loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    loss_function: Callable[
        [nn.Module, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ],
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        total = 0.0
        for sequence, phase_targets, frequency_targets in loader:
            optimizer.zero_grad()
            loss = loss_function(model, sequence, phase_targets, frequency_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total += float(loss.detach())
        history.append(total / len(loader))
    model.eval()
    return history


def _mlp_loss(
    module: nn.Module,
    sequence: torch.Tensor,
    _phase_targets: torch.Tensor,
    _frequency_targets: torch.Tensor,
) -> torch.Tensor:
    model = module
    assert isinstance(model, ResidualMLP)
    state = sequence[:, 0]
    losses = []
    for step in range(1, sequence.shape[1]):
        state = model.step(state)
        losses.append(circular_state_error(state, sequence[:, step]).mean())
    return torch.stack(losses).mean()


def _fixed_loss(
    module: nn.Module,
    sequence: torch.Tensor,
    _phase_targets: torch.Tensor,
    _frequency_targets: torch.Tensor,
) -> torch.Tensor:
    model = module
    assert isinstance(model, FixedKoopmanAE)
    initial = sequence[:, 0]
    latent = model.encode(initial)
    reconstruction = circular_state_error(model.decode(latent), initial).mean()
    rollout_losses = []
    latent_losses = []
    for step in range(1, sequence.shape[1]):
        latent = model.step_latent(latent)
        target = sequence[:, step]
        rollout_losses.append(circular_state_error(model.decode(latent), target).mean())
        latent_losses.append((latent - model.encode(target)).square().mean())
    return (
        reconstruction
        + torch.stack(rollout_losses).mean()
        + 0.5 * torch.stack(latent_losses).mean()
    )


def _conditioned_loss(
    module: nn.Module,
    sequence: torch.Tensor,
    phase_targets: torch.Tensor,
    frequency_targets: torch.Tensor,
) -> torch.Tensor:
    model = module
    assert isinstance(model, EnergyConditionedRotation)
    initial = sequence[:, 0]
    condition = model.normalized_energy(initial)
    physical_energy = torch_energy(initial)
    phase = model.encode_phase(initial)
    flat_sequence = sequence.flatten(0, 1)
    flat_condition = model.normalized_energy(flat_sequence)
    flat_phase = model.encode_phase(flat_sequence)
    reconstruction = circular_state_error(
        model.decode(flat_phase, flat_condition),
        flat_sequence,
    ).mean()
    phase_supervision = (flat_phase - phase_targets.flatten(0, 1)).square().mean()
    frequency_supervision = (model.angular_frequency(condition) - frequency_targets).square().mean()
    rollout_losses = []
    phase_losses = []
    energy_losses = []
    for step in range(1, sequence.shape[1]):
        phase = model.rotate(phase, condition)
        target = sequence[:, step]
        prediction = model.decode(phase, condition)
        rollout_losses.append(circular_state_error(prediction, target).mean())
        phase_losses.append((phase - model.encode_phase(target)).square().mean())
        energy_losses.append((torch_energy(prediction) - physical_energy).square().mean())
    return (
        reconstruction
        + torch.stack(rollout_losses).mean()
        + 0.5 * torch.stack(phase_losses).mean()
        + 0.1 * torch.stack(energy_losses).mean()
        + phase_supervision
        + frequency_supervision
    )


def _conditioned_coordinate_loss(
    module: nn.Module,
    sequence: torch.Tensor,
    phase_targets: torch.Tensor,
    frequency_targets: torch.Tensor,
) -> torch.Tensor:
    """Anchor the supervised coordinates before fitting the physical decoder."""

    model = module
    assert isinstance(model, EnergyConditionedRotation)
    flat_sequence = sequence.flatten(0, 1)
    phase = model.encode_phase(flat_sequence)
    phase_supervision = (phase - phase_targets.flatten(0, 1)).square().mean()
    condition = model.normalized_energy(sequence[:, 0])
    frequency_supervision = (model.angular_frequency(condition) - frequency_targets).square().mean()
    return phase_supervision + frequency_supervision


def _training_loader(
    config: ExperimentConfig,
    dataset: TensorDataset,
) -> DataLoader:
    """Give each model the same deterministic shuffle, independent of training order."""

    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )


def _train_atlas_saddle(
    atlas: SeparatrixAtlas,
    loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
) -> list[float]:
    optimizer = torch.optim.Adam([atlas.raw_saddle_rate], lr=learning_rate)
    history: list[float] = []
    atlas.train()
    for _ in range(epochs):
        total = 0.0
        batches = 0
        for sequence, _, _ in loader:
            state = sequence[:, :-1].flatten(0, 1)
            target = sequence[:, 1:].flatten(0, 1)
            displacement, _ = atlas.saddle_coordinates(state)
            physical_energy = torch_energy(state)
            selected = (physical_energy > atlas.minimum_saddle_energy) & (displacement.abs() < 1.0)
            if not torch.any(selected):
                continue
            optimizer.zero_grad()
            prediction = atlas.saddle_step(state[selected])
            loss = circular_state_error(prediction, target[selected]).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        history.append(total / max(batches, 1))
    atlas.eval()
    return history


def train_models(
    config: ExperimentConfig,
    *,
    include_atlas: bool = False,
) -> TrainedModels:
    set_seed(config.seed)
    dataset = training_dataset(config)

    mlp = ResidualMLP(config.hidden_dim)
    fixed = FixedKoopmanAE(config.hidden_dim, config.latent_dim, config.dt)
    conditioned = EnergyConditionedRotation(config.hidden_dim, config.dt)
    conditioned_pretraining = _train(
        conditioned,
        _training_loader(config, dataset),
        epochs=config.epochs_conditioned_pretrain,
        learning_rate=config.learning_rate,
        loss_function=_conditioned_coordinate_loss,
    )
    histories = {
        "mlp": _train(
            mlp,
            _training_loader(config, dataset),
            epochs=config.epochs_mlp,
            learning_rate=config.learning_rate,
            loss_function=_mlp_loss,
        ),
        "fixed_koopman": _train(
            fixed,
            _training_loader(config, dataset),
            epochs=config.epochs_fixed,
            learning_rate=config.learning_rate,
            loss_function=_fixed_loss,
        ),
        "energy_conditioned": conditioned_pretraining
        + _train(
            conditioned,
            _training_loader(config, dataset),
            epochs=config.epochs_conditioned,
            learning_rate=config.learning_rate,
            loss_function=_conditioned_loss,
        ),
    }
    atlas = None
    if include_atlas:
        atlas = SeparatrixAtlas(
            conditioned,
            config.dt,
        )
        histories["atlas_saddle"] = _train_atlas_saddle(
            atlas,
            _training_loader(config, dataset),
            epochs=config.epochs_atlas_saddle,
            learning_rate=config.learning_rate,
        )
    return TrainedModels(
        mlp=mlp,
        fixed=fixed,
        conditioned=conditioned,
        atlas=atlas,
        histories=histories,
    )
