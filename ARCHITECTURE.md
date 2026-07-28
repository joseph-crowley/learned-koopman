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

The model is supervised with the exact libration frequency derived from the
arithmetic-geometric mean expression for the complete elliptic integral. That
choice makes the learned coordinate consistent across amplitudes and turns the
frequency curve into an inspectable output.

The decoder is still learned, so physical energy preservation is measured and
penalized rather than assumed.

## Design choices

- Vanilla PyTorch keeps every model and objective visible.
- `uv.lock` fixes the executable environment.
- Generated datasets stay in memory; no large binary files are required.
- JSON results are the source for README claims.
- The quick demo is CI-sized; the portfolio benchmark is the promoted result.
