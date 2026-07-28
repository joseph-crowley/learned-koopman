# Architecture

## Data flow

```text
initial amplitude and conserved energy
    ↓
velocity-Verlet pendulum simulator
    ↓
(sin θ, cos θ, ω) trajectory
    ↓
complete-trajectory windows ─────────────────────────┐
    ↓                                                │
baseline / fixed KAE / conditioned model / atlas     │
    ↓                                                │
autonomous rollout                                   │
    ↓                                                │
angle · velocity · energy · frequency · chart use ◀──┘
```

Training and evaluation share the simulator and state contract but not
trajectory instances. Evaluation holds out complete amplitudes.

## Fixed operator

`FixedKoopmanAE` has a nonlinear encoder and decoder around an exactly
orthogonal latent step. The operator is constructed as a matrix exponential of
a skew-symmetric generator, avoiding spectral-radius drift by construction.

Its loss combines:

- physical-state reconstruction;
- decoded multistep prediction;
- encoded-next-state latent consistency.

It is the fair version of the original repository's stated model.

## Energy-conditioned operator

`EnergyConditionedRotation` encodes state to a two-dimensional unit phase,
learns \(\omega(H)\), rotates phase exactly, and decodes phase plus energy.

Training uses a short coordinate curriculum before the joint physical loss.
The first stage anchors the phase encoder and frequency network to the exact
libration phase; the second fits reconstruction, rollout, latent consistency,
and energy losses. This prevents a measured phase-collapse failure without
changing the model or selecting a favorable seed.

The frequency target comes from the arithmetic-geometric mean expression for
the complete elliptic integral. That choice makes the learned coordinate
consistent across amplitudes and turns the frequency curve into an inspectable
output.

The decoder is still learned, so physical energy preservation is measured and
penalized rather than assumed.

## Near-separatrix atlas

`SeparatrixAtlas` freezes a trained `EnergyConditionedRotation` as its regular
chart and learns one scalar rate for a local hyperbolic chart around the
unstable upright equilibrium. In local canonical coordinates \(q,p\), its
operator is

\[
\exp\left(
\Delta t
\begin{bmatrix}
0 & 1\\
\lambda^2 & 0
\end{bmatrix}
\right).
\]

The determinant is one for every learned \(\lambda\), so the local update is
symplectic in \(q,p\) by construction. That statement does not extend through
the arbitrary neural decoder to a claim of globally symplectic physical
dynamics.

Routing is an explicit validity rule:

```text
regular chart: all ordinary energies and states away from the upright saddle
saddle chart:  H > 0.8 and |q| < 1.4
```

When a predicted trajectory changes chart, the model converts through its own
predicted physical state and initializes the destination chart there. No
reference state is consulted during rollout. High-energy predictions are
projected onto the known initial energy shell; projection is separately tested
on the single chart, where it does not fix the coordinate failure.

The promoted evidence includes two destructive ablations:

- the saddle chart alone, which initially tracks the slow departure but
  accumulates large energy drift away from its local validity region;
- energy projection on the single chart, which preserves the invariant but
  does not repair its near-separatrix phase representation.

The first implementation used a neural categorical router. A critic run showed
that it exactly reproduced the predeclared validity region without changing
decisions, so the final model removes those parameters and exposes the
geometric rule directly.

## Design choices

- Vanilla PyTorch keeps every model and objective visible.
- `uv.lock` fixes the executable environment.
- Generated datasets stay in memory; no large binary files are required.
- JSON results are the source for README claims.
- Independent seeded data loaders keep one model's training schedule from
  changing another model's minibatch order.
- The robustness command records every seed plus mean, standard deviation,
  minimum, and maximum for each core metric.
- The atlas robustness command first averages the predeclared high-energy
  amplitudes within each seed, then compares those independent seeded runs.
- Held-out local-chart residuals, switch disagreement, chart occupancy, and the
  local operator determinant keep the mechanism inspectable.
- The quick demo is CI-sized; the committed portfolio and atlas artifacts are
  the promoted results.
