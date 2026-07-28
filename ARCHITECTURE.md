# Architecture

## Data flow

```text
initial amplitude
    ↓
velocity-Verlet pendulum simulator
    ↓
(sin θ, cos θ, ω) trajectory
    ↓
complete-trajectory windows ───────────────┐
    ↓                                      │
baseline / fixed KAE / conditioned model   │
    ↓                                      │
autonomous rollout                         │
    ↓                                      │
angle · velocity · energy · frequency ◀────┘
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

## Design choices

- Vanilla PyTorch keeps every model and objective visible.
- `uv.lock` fixes the executable environment.
- Generated datasets stay in memory; no large binary files are required.
- JSON results are the source for README claims.
- Independent seeded data loaders keep one model's training schedule from
  changing another model's minibatch order.
- The robustness command records every seed plus mean, standard deviation,
  minimum, and maximum for each core metric.
- The quick demo is CI-sized; the portfolio benchmark is the promoted result.
