import torch

from learned_koopman.models import EnergyConditionedRotation, FixedKoopmanAE, ResidualMLP


def _states(batch: int = 8) -> torch.Tensor:
    theta = torch.linspace(-2.0, 2.0, batch)
    omega = torch.linspace(-0.2, 0.2, batch)
    return torch.stack((torch.sin(theta), torch.cos(theta), omega), dim=-1)


def test_models_preserve_shapes_and_circle_representation() -> None:
    states = _states()
    models = [
        ResidualMLP(16),
        FixedKoopmanAE(16, 4),
        EnergyConditionedRotation(16, 0.02),
    ]
    for model in models:
        prediction = model.step(states)
        assert prediction.shape == states.shape
        torch.testing.assert_close(
            prediction[:, :2].norm(dim=-1),
            torch.ones(len(states)),
            atol=1e-6,
            rtol=1e-6,
        )


def test_energy_conditioned_frequency_is_bounded() -> None:
    model = EnergyConditionedRotation(16, 0.02)
    frequencies = model.angular_frequency(torch.tensor([[0.0], [0.5], [1.0]]))
    assert torch.all(frequencies > 0.0)
    assert torch.all(frequencies < 1.05)


def test_fixed_operator_is_orthogonal_by_construction() -> None:
    model = FixedKoopmanAE(16, 4)
    with torch.no_grad():
        model.generator.normal_()
    operator = model.operator_matrix()
    torch.testing.assert_close(
        operator.T @ operator,
        torch.eye(4),
        atol=1e-5,
        rtol=1e-5,
    )
