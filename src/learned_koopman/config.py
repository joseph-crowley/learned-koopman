from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """All knobs required to reproduce a portfolio experiment."""

    seed: int = 7
    dt: float = 0.02
    train_steps: int = 900
    rollout_steps: int = 1200
    horizon: int = 8
    window_stride: int = 12
    train_amplitudes: int = 28
    train_min_amplitude: float = 0.15
    train_max_amplitude: float = 2.85
    evaluation_amplitudes: tuple[float, ...] = (0.25, 1.0, 2.0, 2.8, 3.05)
    showcase_amplitude: float = 2.0
    summary_band_min_amplitude: float = 2.95
    hidden_dim: int = 48
    latent_dim: int = 8
    batch_size: int = 256
    epochs_mlp: int = 100
    epochs_fixed: int = 140
    epochs_conditioned_pretrain: int = 40
    epochs_conditioned: int = 120
    epochs_atlas_saddle: int = 30
    learning_rate: float = 2e-3
    output_dir: Path = Path("results/portfolio")

    @classmethod
    def quick(cls, output_dir: Path = Path("results/quick")) -> ExperimentConfig:
        return cls(
            rollout_steps=500,
            output_dir=output_dir,
        )

    @classmethod
    def atlas(cls, output_dir: Path = Path("results/atlas")) -> ExperimentConfig:
        """Use denser near-separatrix training and held-out high-energy shells."""

        return cls(
            train_amplitudes=40,
            train_max_amplitude=3.12,
            evaluation_amplitudes=(0.25, 1.0, 2.0, 2.8, 2.95, 3.05, 3.10),
            showcase_amplitude=3.05,
            output_dir=output_dir,
        )

    @classmethod
    def quick_atlas(cls, output_dir: Path = Path("results/atlas-quick")) -> ExperimentConfig:
        return replace(cls.atlas(output_dir=output_dir), rollout_steps=500)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload
