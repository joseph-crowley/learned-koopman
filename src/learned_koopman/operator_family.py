from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def observable_feature_names(
    state_columns: tuple[str, ...],
    *,
    degree: int,
) -> tuple[str, ...]:
    if degree not in {1, 2}:
        raise ValueError("observable degree must be 1 or 2")
    names = ["1", *state_columns]
    if degree == 2:
        for left, left_name in enumerate(state_columns):
            for right_name in state_columns[left:]:
                names.append(f"{left_name}*{right_name}")
    return tuple(names)


def polynomial_observables(states: np.ndarray, *, degree: int) -> np.ndarray:
    """Lift states with a transparent constant/linear/quadratic dictionary."""

    if degree not in {1, 2}:
        raise ValueError("observable degree must be 1 or 2")
    values = np.asarray(states, dtype=np.float64)
    if values.ndim < 1:
        raise ValueError("states must have at least one dimension")
    features = [np.ones(values.shape[:-1] + (1,), dtype=np.float64), values]
    if degree == 2:
        products = [
            values[..., left : left + 1] * values[..., right : right + 1]
            for left in range(values.shape[-1])
            for right in range(left, values.shape[-1])
        ]
        features.extend(products)
    return np.concatenate(features, axis=-1)


@dataclass(frozen=True)
class FiberedKoopmanModel:
    """Polynomial family of finite Koopman regressions indexed by an invariant."""

    matrices: np.ndarray
    invariant_center: float
    invariant_scale: float
    state_dim: int
    observable_degree: int
    dt: float
    ridge: float

    @property
    def family_degree(self) -> int:
        return int(self.matrices.shape[0] - 1)

    @property
    def observable_dim(self) -> int:
        return int(self.matrices.shape[1])

    def normalized_coordinate(self, invariant_value: float) -> float:
        return (float(invariant_value) - self.invariant_center) / self.invariant_scale

    def operator(self, invariant_value: float) -> np.ndarray:
        coordinate = self.normalized_coordinate(invariant_value)
        return sum(
            coordinate**degree * matrix
            for degree, matrix in enumerate(self.matrices)
        )

    def predict_one_step(
        self,
        states: np.ndarray,
        invariant_values: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(states, dtype=np.float64)
        invariants = np.asarray(invariant_values, dtype=np.float64)
        if values.shape[:-1] != invariants.shape:
            raise ValueError("one invariant value is required per state")
        lifted = polynomial_observables(values, degree=self.observable_degree)
        flat_lifted = lifted.reshape(-1, lifted.shape[-1])
        flat_invariants = invariants.reshape(-1)
        predicted = np.stack(
            [
                row @ self.operator(invariant)
                for row, invariant in zip(flat_lifted, flat_invariants, strict=True)
            ]
        )
        state = predicted[:, 1 : 1 + self.state_dim]
        return state.reshape(values.shape)

    def rollout(
        self,
        initial_states: np.ndarray,
        invariant_values: np.ndarray,
        *,
        steps: int,
    ) -> np.ndarray:
        initial = np.asarray(initial_states, dtype=np.float64)
        invariants = np.asarray(invariant_values, dtype=np.float64)
        if initial.ndim != 2 or initial.shape[0] != len(invariants):
            raise ValueError("rollout expects [trajectories, state] initial states")
        lifted = polynomial_observables(initial, degree=self.observable_degree)
        operators = np.stack([self.operator(value) for value in invariants])
        result = [initial]
        for _ in range(steps - 1):
            lifted = np.einsum("nf,nfg->ng", lifted, operators)
            result.append(lifted[:, 1 : 1 + self.state_dim])
        return np.stack(result, axis=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_degree": self.family_degree,
            "observable_degree": self.observable_degree,
            "observable_dim": self.observable_dim,
            "state_dim": self.state_dim,
            "dt": self.dt,
            "ridge": self.ridge,
            "invariant_center": self.invariant_center,
            "invariant_scale": self.invariant_scale,
            "matrices": self.matrices.tolist(),
        }


def fit_fibered_operator(
    trajectories: np.ndarray,
    invariant_values: np.ndarray,
    *,
    dt: float,
    family_degree: int = 2,
    observable_degree: int = 2,
    ridge: float = 1e-6,
) -> FiberedKoopmanModel:
    """Fit ``psi(x_next) = psi(x) K(I)`` by ridge regression."""

    states = np.asarray(trajectories, dtype=np.float64)
    invariants = np.asarray(invariant_values, dtype=np.float64)
    if states.ndim != 3 or states.shape[0] != len(invariants):
        raise ValueError("trajectories must be [trajectory, time, state]")
    if family_degree < 0 or family_degree > 3:
        raise ValueError("family degree must be between 0 and 3")
    if ridge < 0.0 or dt <= 0.0:
        raise ValueError("ridge must be nonnegative and dt must be positive")

    invariant_center = float(invariants.mean())
    invariant_scale = float(invariants.std())
    if family_degree == 0:
        invariant_scale = 1.0
    elif invariant_scale <= 1e-8:
        raise ValueError("invariant coordinate is collapsed")
    coordinates = (invariants - invariant_center) / invariant_scale

    current = polynomial_observables(
        states[:, :-1],
        degree=observable_degree,
    )
    future = polynomial_observables(
        states[:, 1:],
        degree=observable_degree,
    )
    flat_current = current.reshape(-1, current.shape[-1])
    flat_future = future.reshape(-1, future.shape[-1])
    repeated_coordinates = np.repeat(coordinates, states.shape[1] - 1)
    design = np.concatenate(
        [
            repeated_coordinates[:, None] ** degree * flat_current
            for degree in range(family_degree + 1)
        ],
        axis=1,
    )
    gram = design.T @ design
    regularizer = ridge * np.eye(gram.shape[0], dtype=np.float64)
    coefficients = np.linalg.solve(gram + regularizer, design.T @ flat_future)
    observable_dim = current.shape[-1]
    matrices = coefficients.reshape(
        family_degree + 1,
        observable_dim,
        observable_dim,
    )
    return FiberedKoopmanModel(
        matrices=matrices,
        invariant_center=invariant_center,
        invariant_scale=invariant_scale,
        state_dim=states.shape[-1],
        observable_degree=observable_degree,
        dt=dt,
        ridge=ridge,
    )


def spectral_summary(
    model: FiberedKoopmanModel,
    invariant_values: np.ndarray,
) -> list[dict[str, Any]]:
    """Summarize the fitted finite operator using principal-branch frequencies.

    Frequencies are derived from eigenvalue angles in ``[-pi, pi]`` and are
    therefore limited by the sampling Nyquist frequency. They are diagnostics,
    not residual-certified Koopman eigenfrequencies.
    """

    summaries: list[dict[str, Any]] = []
    for invariant in np.asarray(invariant_values, dtype=np.float64):
        eigenvalues = np.linalg.eigvals(model.operator(float(invariant)))
        eigenvalue_rows = []
        positive_frequencies = []
        for value in eigenvalues:
            magnitude = float(abs(value))
            angle = float(np.angle(value))
            frequency = abs(angle) / (2.0 * np.pi * model.dt)
            damping = math_log_magnitude(magnitude) / model.dt
            if abs(angle) > 1e-7 and frequency > 1e-8:
                positive_frequencies.append(frequency)
            eigenvalue_rows.append(
                {
                    "real": float(value.real),
                    "imag": float(value.imag),
                    "magnitude": magnitude,
                    "frequency_hz": frequency,
                    "continuous_damping_rate": damping,
                }
            )
        summaries.append(
            {
                "invariant": float(invariant),
                "spectral_radius": max(row["magnitude"] for row in eigenvalue_rows),
                "lowest_nonzero_principal_frequency_hz": (
                    min(positive_frequencies) if positive_frequencies else None
                ),
                "eigenvalues": sorted(
                    eigenvalue_rows,
                    key=lambda row: (-row["magnitude"], row["frequency_hz"]),
                ),
            }
        )
    return summaries


def math_log_magnitude(magnitude: float) -> float:
    return float(np.log(max(magnitude, 1e-15)))
