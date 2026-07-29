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
machine-readable empirical-gate manifest, overview figure, and a nested
canonical-action audit. The optional `energy` column is excluded from training
and used only afterward to test the learned Hamiltonian.

Use the exported world model:

```bash
uv run learned-koopman canonical-predict \
  results/my-koopman-hj/model.pt \
  --initial 1.1 0.0 \
  --steps 400 \
  --output results/my-koopman-hj/prediction.csv
```

Prediction refuses a fit that failed its current-dataset gates or a state
outside the observed action range unless `--allow-unsupported` is explicit.

## Diagnose the chart, then test whether its residual is identifiable

An action-range check on one state cannot establish that a learned canonical
chart is valid. Test complete trajectories against independent geometry,
phase-law, and conjugacy residuals:

```bash
uv run learned-koopman canonical-diagnose \
  results/koopman-hj/model.pt \
  examples/duffing-trajectories.csv \
  --position-column position \
  --momentum-column velocity \
  --output results/koopman-hj/orbit-diagnostics.json
```

Run the closed-form oracle pipeline regression:

```bash
uv run learned-koopman chart-fidelity \
  --output results/chart-fidelity.json
```

That experiment observes a known symplectic twist-kick map through a canonical
shear and reproduces one analytic cohomological cancellation. It is useful as
a convention and pipeline regression, but the result follows from a
closed-form identity and is not a learned-chart falsifier.

The empirical test trains independent exact-symplectic charts, estimates the
residual from held-out trajectory transitions, and attacks apparent agreement
with controlled exact canonical gauges:

```bash
uv run learned-koopman resonance-metrology \
  --profile full \
  --output results/my-resonance-metrology
```

The report returns a resonant generating coefficient and island-width estimate
only when the band crosses the resonance and the ensemble, null, shuffled,
wrong-harmonic, estimator-variant, detection-floor, and exact-gauge controls
permit it. Otherwise it says why the quantity is unresolved. See
[`RESIDUAL_METROLOGY.md`](RESIDUAL_METROLOGY.md) for the mathematics, API, and
predeclared claim boundary. Rotation-law initialization uses only circular
phase increments from the training trajectories; the manifest records that
optimization seed separately from the held-out result.

For an existing measured return-map CSV and two or more independently trained
canonical models, `learned-koopman resonance-estimate` runs the same
trajectory-band estimator without oracle coordinates. It refuses failed model
fits and byte-identical copied charts, records every model digest, and labels
ensemble spread as a lower bound on chart ambiguity rather than calibrated
physical error.

## Checked resonance result: prediction is not identifiability

The frozen full reference run returned **`resolved_refuted
(gauge_freedom)`**. All eight charts were good one-step predictors. Five
supported a conditioned band fit, and their median estimate nearly met the
recovery target. But an exact canonical gauge that stayed inside the same
prediction envelope moved the recovered block by more than twice the allowed
complex and magnitude tolerances.

| Frozen measurement | Result |
|---|---:|
| Prediction-accepted charts | **8 / 8** |
| Charts with an estimable band fit | 5 / 8 |
| Held-out normalized one-step RMSE | **0.0453–0.0508** |
| Consensus complex block error | 19.59% (20% gate passes) |
| Median per-chart complex error | 20.44% (3 / 5 exceed 20%) |
| Median magnitude error | 6.87% (15% gate passes) |
| Recovered / planted island halfwidth | 0.1887 / 0.1826 |
| Empirical floor coverage | 25% of 4 evaluable charts (80% gate fails) |
| Estimable null charts | 5 / 8 (minimum 6 fails) |
| Shuffled-angle level / allowed | 37.23% / 20% |
| Maximum prediction-equivalent exact-gauge shift | **44.16% complex, 43.29% magnitude** |
| Estimator variants | G9 unresolved; quadratic trigger abstained |

![Learned-chart prediction, coefficient recovery, floor coverage, and exact
gauge stress.](results/resonance-metrology/overview.png)

The defensible result is negative and useful: on this fixture, predictive
agreement among exact-symplectic learned charts is insufficient to identify a
resonant normal-form coefficient at the claimed 20% precision. This does not
show that such coefficients are never recoverable. It localizes the missing
ingredient: a chart gauge fixed by external physics, a richer island-capable
surrogate, or an invariant quotient that removes the measured gauge direction.
The recovery sits exactly at the reporting boundary: error of the
componentwise-median coefficient is 19.59%, while the median of the five
per-chart errors is 20.44%. It is near-recovery with shared bias, not a
positive 20%-accuracy claim.
See the [report](results/resonance-metrology/report.html),
[manifest](results/resonance-metrology/manifest.json), and
[decision record](RESEARCH_DECISION.md).

## What the checked-in experiment establishes

The promoted result uses 22 Duffing trajectories for training and eight
complete trajectories for held-out evaluation. Energy and empirical action are
never optimization targets.

| Held-out or post-fit measurement | Result |
|---|---:|
| Recursive normalized rollout RMSE | **0.0270** |
| Persistence rollout RMSE | 1.5636 |
| Observed Koopman phase residual | **0.00024** |
| Observed normalized action drift | **0.0034** |
| Complete-orbit radial coefficient of variation | **0.0026** |
| Complete latent-conjugacy RMSE | **0.00031** |
| Numerical symplectic defect | **2.38×10⁻⁷** |
| Model-rollout action drift | **3.19×10⁻⁵** |
| Latent action vs. \((2\pi)^{-1}\oint p\,dq\) | **R² 0.99999994** |
| Physical-action calibration slope | **0.99983** |
| Learned \(dh/dI\) vs. measured frequency | **0.93% error** |
| Learned \(h(I)\) vs. energy shape | **0.18% error** |

The [report](results/koopman-hj/report.html),
[manifest](results/koopman-hj/manifest.json), and
[orbit diagnostics](results/koopman-hj/orbit-diagnostics.json) carry the model
evidence and exact claim boundary. The independent
[action audit](results/koopman-hj/action-audit/report.html) supplies the
physical ruler. This is one synthetic system and one deterministic split—not
yet a statistically powered research result or a hardware validation. The
resonance-metrology result is reported separately because good integrable
prediction does not by itself make a residual normal-form coefficient
identifiable.

## Why the Hamilton–Jacobi connection matters

A generic learned invariant is free up to a nonlinear monotone
reparameterization. That is enough to label orbit families, but it is not yet
physical action. Symplecticity removes the arbitrary fitted scale: canonical
maps preserve phase-space area, so a chart that truly circularizes the periodic
orbits has mean radial action \(I\) agreeing with

\[
J=\frac{1}{2\pi}\oint p\,dq
\]

The enclosed-area version of this equality is structural for any exact
symplectic map; the nontrivial diagnostic is whether the learned orbit is
circular enough that its mean radial action agrees at that fixed physical
scale. In that chart the Hamiltonian depends only on action, the angle advances
linearly, and the same coordinates simultaneously expose Hamilton–Jacobi and
Koopman structure:

\[
H\circ F_\theta^{-1}=h(I),\qquad
\dot I=0,\qquad
\dot\phi=h'(I),\qquad
\mathcal L e^{ik\phi}=ik\,h'(I)e^{ik\phi}.
\]

The implementation does not merely penalize symplectic error. Its translations,
reciprocal scaling, neural canonical shears, radial Hamiltonian flow, and
analytic inverse are symplectic by construction.

The checked Duffing data comes from a velocity-Verlet map. A dt-versus-dt/10
loop-area comparison found a roughly \(2.4\times10^{-4}\) median discrepancy,
but that number combines integrator, polygon-quadrature, and cycle-closure
effects. It is an empirical systematic at the same scale as the sharpest
action-calibration metrics, not a formal modified-Hamiltonian bound.

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

Today that path is for noiseless, well-sampled trajectories. The phase advance
per sample must remain below the angular Nyquist limit, and the current cycle
detector is not hardened for sensor noise, missing samples, or ambiguous
sections. Those cases require preprocessing and independent section-quality
checks rather than an optimistic fit.

Velocity equals canonical momentum only when the mass convention makes that
true; otherwise convert it before fitting. Missing values, non-finite states,
irregular time, incomplete orbit evidence, stale result manifests, and
unsupported prediction are rejected rather than silently repaired.

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
uv run learned-koopman chart-fidelity --output results/chart-fidelity.json
uv run learned-koopman resonance-metrology \
  --profile ci --output results/ci-resonance-metrology
uv run learned-koopman canonical-diagnose \
  results/koopman-hj/model.pt examples/duffing-trajectories.csv \
  --position-column position --momentum-column velocity \
  --output results/koopman-hj/orbit-diagnostics.json
uv run python scripts/check_chart_fidelity.py
uv run python scripts/check_resonance_metrology.py
uv run python scripts/check_resonance_metrology.py \
  results/ci-resonance-metrology/manifest.json
uv run python scripts/check_canonical_diagnostics.py
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
- [Resonance metrology](RESIDUAL_METROLOGY.md) — the learned-chart residual
  instrument, exact-gauge stress, and abstention rules;
- [Contributing](CONTRIBUTING.md) — reproducible development workflow.

MIT licensed. If you build on the project, cite [`CITATION.cff`](CITATION.cff).
