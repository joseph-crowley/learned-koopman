from __future__ import annotations

from dataclasses import asdict, dataclass
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
    hidden_dim: int = 48
    latent_dim: int = 8
    batch_size: int = 256
    epochs_mlp: int = 100
    epochs_fixed: int = 140
    epochs_conditioned_pretrain: int = 40
    epochs_conditioned: int = 120
    learning_rate: float = 2e-3
    output_dir: Path = Path("results/portfolio")

    @classmethod
    def quick(cls, output_dir: Path = Path("results/quick")) -> ExperimentConfig:
        return cls(
            rollout_steps=500,
            output_dir=output_dir,
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload
