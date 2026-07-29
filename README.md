# Learned Koopman

[![CI](https://github.com/joseph-crowley/learned-koopman/actions/workflows/ci.yml/badge.svg)](https://github.com/joseph-crowley/learned-koopman/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Learn a canonical change of coordinates that turns nonlinear mechanical
motion into an action-conditioned rotation.**

This is a PyTorch research workbench for Koopman theory, Hamilton–Jacobi
mechanics, and structure-preserving system identification. From repeated
canonical trajectories \((q,p)\), its flagship model learns

1. an exactly symplectic and analytically invertible map
   \(F_\theta:(q,p)\mapsto(Q,P)\);
2. the action \(I=(Q^2+P^2)/2\);
3. a latent Hamiltonian \(h_\psi(I)\);
4. the nonlinear frequency law \(\omega(I)=dh_\psi/dI\); and
5. the fiberwise Koopman eigenfunctions \(e^{ik\phi}\).

Prediction is analytic in the learned canonical chart—no neural ODE solver and
no unconstrained latent matrix:

\[
x_{k+1}
=F_\theta^{-1}
\left(
R_{\Delta t\,h_\psi'(I)}
F_\theta(x_k)
\right).
\]

![Held-out physical rollout, learned circular canonical chart, physical-action
calibration, and training history.](results/koopman-hj/overview.png)

## Run the canonical model

Install [`uv`](https://docs.astral.sh/uv/) and use the included conservative
Duffing data:

```bash
git clone https://github.com/joseph-crowley/learned-koopman.git
cd learned-koopman
uv sync --extra dev

uv run learned-koopman canonical-train \
  examples/duffing-trajectories.csv \
  --position-column position \
  --momentum-column velocity \
  --reference-column energy \
  --quick \
  --output results/my-koopman-hj
```

Open `results/my-koopman-hj/report.html`. The run writes a loadable model,
machine-readable certificate, overview figure, and a nested canonical-action
audit. The optional `energy` column is excluded from training and used only
afterward to test the learned Hamiltonian.

Use the exported world model:

```bash
uv run learned-koopman canonical-predict \
  results/my-koopman-hj/model.pt \
  --initial 1.1 0.0 \
  --steps 400 \
  --output results/my-koopman-hj/prediction.csv
```

Prediction refuses an uncertified fit or a state outside the observed action
range unless `--allow-unsupported` is explicit.

## What the checked-in experiment establishes

The promoted result uses 22 Duffing trajectories for training and eight
complete trajectories for held-out evaluation. Energy and empirical action are
never optimization targets.

| Held-out or post-fit measurement | Result |
|---|---:|
| Recursive normalized rollout RMSE | **0.0551** |
| Persistence rollout RMSE | 1.5636 |
| Observed Koopman phase residual | **0.00053** |
| Observed normalized action drift | **0.0092** |
| Numerical symplectic defect | **3.58×10⁻⁷** |
| Model-rollout action drift | **2.92×10⁻⁵** |
| Latent action vs. \((2\pi)^{-1}\oint p\,dq\) | **R² 0.9999996** |
| Physical-action calibration slope | **0.9987** |
| Learned \(dh/dI\) vs. measured frequency | **1.77% error** |
| Learned \(h(I)\) vs. energy shape | **0.49% error** |

The [report](results/koopman-hj/report.html),
[manifest](results/koopman-hj/manifest.json), and
[action audit](results/koopman-hj/action-audit/report.html) carry the evidence
and exact claim boundary. This is one synthetic system and one deterministic
split—not yet a statistically powered research result or a hardware
validation.

## Why the Hamilton–Jacobi connection matters

A generic learned invariant is free up to a nonlinear monotone reparameterization.
That is enough to label orbit families, but it is not yet physical action.
Symplecticity fixes the gauge: canonical maps preserve phase-space area, so the
latent circle area \(I\) must agree with

\[
J=\frac{1}{2\pi}\oint p\,dq
\]

when the learned chart truly straightens the periodic orbits. In that chart the
Hamiltonian depends only on action, the angle advances linearly, and the same
coordinates simultaneously expose Hamilton–Jacobi and Koopman structure:

\[
H\circ F_\theta^{-1}=h(I),\qquad
\dot I=0,\qquad
\dot\phi=h'(I),\qquad
\mathcal L e^{ik\phi}=ik\,h'(I)e^{ik\phi}.
\]

The implementation does not merely penalize symplectic error. Its translations,
reciprocal scaling, neural canonical shears, radial Hamiltonian flow, and
analytic inverse are symplectic by construction.

## Bring measured trajectories

The CSV contract is deliberately plain:

```csv
trial_id,time,position,momentum
run-01,0.000,0.800,0.000
run-01,0.010,0.799,-0.021
run-02,0.000,1.100,0.000
run-02,0.010,1.099,-0.030
```

```bash
uv run learned-koopman canonical-train measurements.csv \
  --trajectory-column trial_id \
  --time-column time \
  --position-column position \
  --momentum-column momentum \
  --output results/my-rig
```

The first canonical profile assumes:

- one degree of freedom with correctly paired canonical \(q,p\);
- autonomous, conservative, periodic motion away from a separatrix;
- at least six complete trajectories with a shared near-uniform sample time;
- enough duration to observe complete orbits for the post-fit action audit.

Velocity equals canonical momentum only when the mass convention makes that
true; otherwise convert it before fitting. Missing values, non-finite states,
irregular time, incomplete orbit evidence, stale certificates, and unsupported
prediction are rejected rather than silently repaired.

## Use the action audit independently

The audit is a measurement instrument, not the model architecture:

```bash
uv run learned-koopman hj-audit \
  examples/duffing-trajectories.csv \
  --position-column position \
  --momentum-column velocity \
  --reference-column energy \
  --model results/koopman-hj/model.pt \
  --output results/my-action-audit
```

It measures closed-orbit area and period, tests \(dH/dJ=\omega\), and tells you
whether a learned coordinate is merely monotone in action or actually fixed to
the canonical gauge.

## What this can become useful for

The canonical model is intended as the conservative core of a mechanics
workbench:

- fast long-horizon surrogate simulation with no integration loop;
- nonlinear normal-form identification and backbone/frequency–amplitude curves;
- resonance, detuning, and modal-interaction analysis;
- action-based parameter tracking, anomaly detection, and digital twins;
- perturbation and control models that learn slow action drift around a
  structure-preserving autonomous core;
- reduced Hamilton–Jacobi–Bellman or reachability calculations in learned
  intrinsic coordinates;
- local chart atlases for separatrices, rotations, impacts, and other topology
  changes.

The full design, adjacent SOTA, differentiating hypotheses, experiment matrix,
and publication threshold are in
[`KOOPMAN_HJ_FRONTIER.md`](KOOPMAN_HJ_FRONTIER.md).

## Other research paths in the repository

The earlier invariant-conditioned mechanics workbench remains useful when
states are not known to be canonical:

```bash
uv run learned-koopman analyze examples/duffing-trajectories.csv \
  --state-columns position velocity \
  --reference-column energy \
  --quick \
  --output results/my-invariant-model
```

It learns a label-free invariant and a transparent polynomial family of local
Koopman operators, compares it with global EDMD and persistence on held-out
runs, and exports a support-gated predictor.

Its committed Duffing run reports learned-coordinate drift **0.0024**,
fibered-operator rollout RMSE **0.0755**, global quadratic EDMD RMSE **0.4240**,
and persistence RMSE **1.5636**. Those values describe the older nonsymplectic
family; the canonical result above is the promoted model.

The integrated research lab also contains:

- a two-chart near-separatrix pendulum experiment, with five-seed high-energy
  valid time $3.98\pm0.06$ versus $0.36\pm0.12$ for one chart;
- label-free invariant discovery with held-out energy $R^2=0.979$, rank $=1.000$,
  and drift $=0.0053$;
- a deliberately falsifiable stochastic transfer operator, where stronger baselines falsify the learned propagation;
- controlled actuator gain $0.35\rightarrow1.000$, with 9/12 real crossings recovered.

Run it with `uv run learned-koopman lab --quick`.

## Python API

```python
from pathlib import Path

from learned_koopman.canonical_experiment import (
    CanonicalExperimentConfig,
    run_canonical_experiment,
)
from learned_koopman.trajectory import load_trajectory_csv

data = load_trajectory_csv(
    Path("measurements.csv"),
    state_columns=("position", "momentum"),
    trajectory_column="trial_id",
    time_column="time",
)
manifest = run_canonical_experiment(
    data,
    Path("results/my-system"),
    config=CanonicalExperimentConfig.full(seed=7),
)
print(manifest["certificate"]["status"])
```

## Verify everything

```bash
uv run ruff check .
uv run pytest
uv run python scripts/check_canonical_model.py
uv run python scripts/check_hj_action.py
uv run python scripts/check_workbench.py
uv run python scripts/check_research_lab.py
```

## Read next

- [Koopman + HJ frontier](KOOPMAN_HJ_FRONTIER.md) — research landscape,
  target system, experiments, applications, and publication bar;
- [Architecture](ARCHITECTURE.md) — exact model composition and evidence flow;
- [Scientific scope](SCIENTIFIC_SCOPE.md) — what is and is not established;
- [Physics workbench](PHYSICS_WORKBENCH.md) — the broader mathematical program;
- [Contributing](CONTRIBUTING.md) — reproducible development workflow.

MIT licensed. If you build on the project, cite [`CITATION.cff`](CITATION.cff).
