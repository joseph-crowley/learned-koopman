from pathlib import Path

from learned_koopman.config import ExperimentConfig
from learned_koopman.experiment import run_experiment


def test_one_epoch_experiment_writes_metrics_and_figure(tmp_path: Path) -> None:
    config = ExperimentConfig(
        train_steps=80,
        rollout_steps=80,
        horizon=2,
        window_stride=20,
        train_amplitudes=4,
        hidden_dim=8,
        latent_dim=3,
        batch_size=32,
        epochs_mlp=1,
        epochs_fixed=1,
        epochs_conditioned=1,
        output_dir=tmp_path,
    )
    result = run_experiment(config)
    assert "metrics" in result
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "comparison.png").is_file()
