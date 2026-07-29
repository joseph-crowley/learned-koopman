from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ScalarShearNet(nn.Module):
    """A scalar potential derivative used inside an exact symplectic shear."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value.unsqueeze(-1)).squeeze(-1)


class SymplecticMap1D(nn.Module):
    """Invertible canonical map built only from translations, scaling, and shears."""

    def __init__(
        self,
        hidden_dim: int,
        shear_layers: int,
        *,
        initial_center: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or shear_layers < 2:
            raise ValueError("canonical map needs positive width and at least two shears")
        self.hidden_dim = hidden_dim
        self.shear_layers = shear_layers
        self.center = nn.Parameter(torch.tensor(initial_center, dtype=torch.float32))
        self.log_scale = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.shears = nn.ModuleList(
            ScalarShearNet(hidden_dim) for _ in range(shear_layers)
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        q, p = (states - self.center).unbind(dim=-1)
        scale = torch.exp(self.log_scale)
        q, p = scale * q, p / scale
        for index, shear in enumerate(self.shears):
            if index % 2 == 0:
                q = q + shear(p)
            else:
                p = p + shear(q)
        return torch.stack((q, p), dim=-1)

    def inverse(self, latent: torch.Tensor) -> torch.Tensor:
        q, p = latent.unbind(dim=-1)
        for index in range(len(self.shears) - 1, -1, -1):
            shear = self.shears[index]
            if index % 2 == 0:
                q = q - shear(p)
            else:
                p = p - shear(q)
        scale = torch.exp(self.log_scale)
        values = torch.stack((q / scale, p * scale), dim=-1)
        return values + self.center


class RadialHamiltonian(nn.Module):
    """Learn h(I) through a polynomial frequency omega(I) = dh/dI."""

    def __init__(self, degree: int = 3, initial_frequency: float = 1.0) -> None:
        super().__init__()
        if degree < 1:
            raise ValueError("Hamiltonian degree must be positive")
        self.degree = degree
        inverse_softplus = float(np.log(np.expm1(initial_frequency)))
        self.raw_base_frequency = nn.Parameter(
            torch.tensor(inverse_softplus, dtype=torch.float32)
        )
        self.higher_frequency_coefficients = nn.Parameter(
            torch.zeros(degree - 1, dtype=torch.float32)
        )

    def frequency_coefficients(self) -> torch.Tensor:
        base = F.softplus(self.raw_base_frequency) + 1e-4
        return torch.cat((base.reshape(1), self.higher_frequency_coefficients))

    def frequency(self, action: torch.Tensor) -> torch.Tensor:
        coefficients = self.frequency_coefficients()
        result = torch.zeros_like(action)
        power = torch.ones_like(action)
        for coefficient in coefficients:
            result = result + coefficient * power
            power = power * action
        return result

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        coefficients = self.frequency_coefficients()
        result = torch.zeros_like(action)
        power = action
        for index, coefficient in enumerate(coefficients, start=1):
            result = result + coefficient * power / index
            power = power * action
        return result


class CanonicalKoopmanNetwork(nn.Module):
    """Exact-symplectic conjugacy to an action-conditioned latent rotation."""

    def __init__(
        self,
        *,
        dt: float,
        hidden_dim: int = 32,
        shear_layers: int = 6,
        hamiltonian_degree: int = 3,
        initial_center: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        super().__init__()
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self.hidden_dim = hidden_dim
        self.shear_layers = shear_layers
        self.hamiltonian_degree = hamiltonian_degree
        self.canonical_map = SymplecticMap1D(
            hidden_dim,
            shear_layers,
            initial_center=initial_center,
        )
        self.hamiltonian = RadialHamiltonian(hamiltonian_degree)

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        return self.canonical_map(states)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.canonical_map.inverse(latent)

    @staticmethod
    def action_from_latent(latent: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.square(latent).sum(dim=-1)

    def action(self, states: torch.Tensor) -> torch.Tensor:
        return self.action_from_latent(self.encode(states))

    def angle(self, states: torch.Tensor) -> torch.Tensor:
        q, p = self.encode(states).unbind(dim=-1)
        return torch.atan2(-p, q)

    def frequency(self, states: torch.Tensor) -> torch.Tensor:
        return self.hamiltonian.frequency(self.action(states))

    def latent_step(self, latent: torch.Tensor) -> torch.Tensor:
        action = self.action_from_latent(latent)
        angle = self.dt * self.hamiltonian.frequency(action)
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        q, p = latent.unbind(dim=-1)
        return torch.stack(
            (cosine * q + sine * p, -sine * q + cosine * p),
            dim=-1,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.decode(self.latent_step(self.encode(states)))

    def rollout(self, initial: torch.Tensor, *, steps: int) -> torch.Tensor:
        if steps < 1:
            raise ValueError("steps must be positive")
        values = [initial]
        state = initial
        for _ in range(steps - 1):
            state = self(state)
            values.append(state)
        return torch.stack(values, dim=-2)


@dataclass
class CanonicalKoopmanModel:
    """Loadable canonical model with support and certificate metadata."""

    network: CanonicalKoopmanNetwork
    state_columns: tuple[str, str]
    action_min: float
    action_max: float
    certificate_status: str

    def coordinate(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=np.float64)
        if values.ndim == 0 or values.shape[-1] != 2:
            raise ValueError("expected canonical state vectors with two values")
        with torch.no_grad():
            action = self.network.action(
                torch.tensor(values, dtype=torch.float32)
            )
        return action.numpy().astype(np.float64)

    def canonical_coordinates(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=np.float64)
        if values.ndim == 0 or values.shape[-1] != 2:
            raise ValueError("expected canonical state vectors with two values")
        with torch.no_grad():
            latent = self.network.encode(
                torch.tensor(values, dtype=torch.float32)
            )
        return latent.numpy().astype(np.float64)

    def support_status(self, states: np.ndarray) -> np.ndarray:
        action = np.atleast_1d(self.coordinate(states))
        supported = (action >= self.action_min) & (action <= self.action_max)
        status = np.where(supported, "supported", "action_extrapolation").astype(
            object
        )
        if self.certificate_status != "supported_on_held_out_trajectories":
            status[:] = "fit_not_certified"
        return status

    def rollout(
        self,
        initial: np.ndarray,
        *,
        steps: int,
        allow_extrapolation: bool = False,
    ) -> np.ndarray:
        values = np.asarray(initial, dtype=np.float64)
        one_state = values.ndim == 1
        if one_state:
            values = values[None, :]
        support = self.support_status(values)
        if np.any(support != "supported") and not allow_extrapolation:
            raise ValueError(
                f"initial state is unsupported ({', '.join(set(support.tolist()))}); "
                "pass allow_extrapolation=True to override"
            )
        with torch.no_grad():
            result = self.network.rollout(
                torch.tensor(values, dtype=torch.float32),
                steps=steps,
            )
        array = result.numpy().astype(np.float64)
        return array[0] if one_state else array


def save_canonical_model(
    path: Path,
    model: CanonicalKoopmanModel,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    network = model.network
    torch.save(
        {
            "schema_version": 1,
            "dt": network.dt,
            "hidden_dim": network.hidden_dim,
            "shear_layers": network.shear_layers,
            "hamiltonian_degree": network.hamiltonian_degree,
            "state_columns": list(model.state_columns),
            "action_min": model.action_min,
            "action_max": model.action_max,
            "certificate_status": model.certificate_status,
            "state_dict": network.state_dict(),
        },
        path,
    )


def load_canonical_model(path: Path) -> CanonicalKoopmanModel:
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported canonical Koopman model schema")
    network = CanonicalKoopmanNetwork(
        dt=float(payload["dt"]),
        hidden_dim=int(payload["hidden_dim"]),
        shear_layers=int(payload["shear_layers"]),
        hamiltonian_degree=int(payload["hamiltonian_degree"]),
    )
    network.load_state_dict(payload["state_dict"])
    network.eval()
    columns = tuple(payload["state_columns"])
    if len(columns) != 2:
        raise ValueError("canonical model export has invalid state columns")
    return CanonicalKoopmanModel(
        network=network,
        state_columns=(str(columns[0]), str(columns[1])),
        action_min=float(payload["action_min"]),
        action_max=float(payload["action_max"]),
        certificate_status=str(payload["certificate_status"]),
    )
