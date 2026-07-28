from pathlib import Path

from learned_koopman.config import ExperimentConfig
from learned_koopman.experiment import run_experiment, run_robustness_sweep


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
        epochs_conditioned_pretrain=1,
        epochs_conditioned=1,
        output_dir=tmp_path,
    )
    result = run_experiment(config)
    assert "metrics" in result
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "comparison.png").is_file()


def test_robustness_sweep_records_independent_seeds(tmp_path: Path) -> None:
    config = ExperimentConfig(
        train_steps=28,
        rollout_steps=12,
        horizon=2,
        window_stride=4,
        train_amplitudes=3,
        hidden_dim=8,
        latent_dim=3,
        batch_size=32,
        epochs_mlp=1,
        epochs_fixed=1,
        epochs_conditioned_pretrain=1,
        epochs_conditioned=1,
        output_dir=tmp_path,
    )
    result = run_robustness_sweep(config, [3, 5])
    assert result["seeds"] == [3, 5]
    assert result["comparisons"]["seed_count"] == 2
    assert (tmp_path / "robustness.json").is_file()
