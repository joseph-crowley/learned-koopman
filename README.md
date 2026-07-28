# Learned Koopman

[![CI](https://github.com/joseph-crowley/learned-koopman/actions/workflows/ci.yml/badge.svg)](https://github.com/joseph-crowley/learned-koopman/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**A reproducible PyTorch study of a simple question with a sharp edge: can a
neural network learn coordinates that make nonlinear pendulum dynamics easy to
evolve?**

The project compares transparent physics and data-driven baselines with two
latent-dynamics models:

- a fixed, orthogonal Koopman autoencoder;
- an energy-conditioned rotation model that learns a different latent frequency
  on each invariant energy shell.

The result is useful but bounded. After a short coordinate-pretraining
curriculum, the energy-conditioned model improves mean valid horizon and angle
error over the residual MLP in a three-seed sensitivity check, but it does not
win every seed. The tested fixed operator underfits the amplitude-dependent
frequency continuum, and every learned model deteriorates near the separatrix.

![Autonomous rollout, valid prediction time, and rollout frequencies](results/portfolio/comparison.png)

## One-command demonstration

Install [`uv`](https://docs.astral.sh/uv/), clone the repository, and run:

```bash
uv sync --extra dev
uv run learned-koopman demo --quick
```

The command generates deterministic trajectories, trains all learned models,
runs autonomous rollouts, and writes a figure plus machine-readable metrics.
It runs on CPU and does not require downloaded data.

Run the longer portfolio experiment with:

```bash
uv run learned-koopman benchmark
```

Probe sensitivity to initialization with three independent trained runs:

```bash
uv run learned-koopman robustness
```

## What the benchmark shows

The representative seed-7 run at initial amplitude \(\theta_0=2.0\), evaluated
over a 24-unit autonomous rollout:

| Model | Parameters | Valid prediction time | Angle RMSE | Maximum energy drift |
|---|---:|---:|---:|---:|
| Persistence | 0 | 0.32 | 1.984 | 0.000 |
| Global DMD | 9 | 0.60 | 1.996 | 2.732 |
| Small-angle physics | 0 | 0.28 | 1.853 | 0.584 |
| Residual MLP | 2,691 | 3.30 | 0.535 | 0.202 |
| Fixed Koopman AE | 1,227 | 0.74 | 1.755 | 1.179 |
| **Energy-conditioned rotation** | **3,054** | **6.04** | **0.211** | **0.153** |

The exact parameter counts are recorded in
[`results/portfolio/metrics.json`](results/portfolio/metrics.json). The table
uses a valid-horizon threshold of
\(\sqrt{\Delta\theta^2 + 0.25\,\Delta\omega^2} > 0.15\).

The same comparison across seeds 7, 17, and 29 gives:

| Model | Mean valid time | Mean angle RMSE | Mean energy drift | Wins over the other neural model |
|---|---:|---:|---:|---:|
| Residual MLP | 5.74 ± 2.50 | 0.338 ± 0.158 | 0.170 ± 0.033 | 1 / 3 |
| Fixed Koopman AE | 0.72 ± 0.04 | 1.937 ± 0.134 | 1.293 ± 0.117 | — |
| **Energy-conditioned rotation** | **7.23 ± 1.86** | **0.218 ± 0.026** | 0.170 ± 0.028 | **2 / 3** |

These are means and population standard deviations over three runs, not
confidence intervals. Per-seed results and the aggregation contract are in
[`results/portfolio/robustness.json`](results/portfolio/robustness.json).

Four conclusions survive the stronger test:

1. Small-angle physics is superb where its assumptions hold and fails quickly
   at large amplitude.
2. The tested eight-dimensional fixed operator consistently underfits the
   continuum of nonlinear pendulum frequencies; this experiment does not prove
   that every fixed finite approximation must perform poorly.
3. Coordinate pretraining keeps the conditioned fit stable across all three
   runs. It improves the three-seed averages but does not beat a strong
   residual MLP on every initialization.
4. The single conditioned phase chart still breaks down near the separatrix. A
   multi-chart model is the natural next experiment.

## Models

### Fixed Koopman autoencoder

The encoder and decoder are trained with reconstruction, latent-consistency,
and multistep rollout losses. Latent evolution is an orthogonal matrix generated
by a learned skew-symmetric matrix:

\[
z_{t+\Delta t} = \exp\!\left(\Delta t(G-G^\top)\right)z_t.
\]

This is a deliberately structured fixed-operator baseline. Orthogonality
prevents spectral-radius blow-up, while its finite fixed spectrum remains a
restrictive approximation to a continuum of pendulum frequencies.

### Energy-conditioned rotation

The structured model learns an action-angle phase coordinate and a frequency
law conditioned on the conserved energy:

\[
\phi_{t+\Delta t}
=R\!\left(\omega(H)\Delta t\right)\phi_t.
\]

Training first anchors the phase encoder and frequency network, then fits the
decoder and multistep physical rollout jointly. It uses the pendulum's exact
elliptic-integral frequency law as direct phase and frequency supervision.
Globally this is a **fibered family of linear operators**, not one finite
state-independent Koopman matrix. That boundary is important.

### Baselines

- persistence;
- the velocity-Verlet map of the linearized pendulum;
- global least-squares DMD on circular state coordinates;
- a matched-data residual MLP without a linear bottleneck.

## Reproducibility

- State: \((\sin\theta,\cos\theta,\dot\theta)\), respecting angle periodicity.
- Simulator: reversible, symplectic velocity Verlet.
- Split: complete evaluation trajectories at amplitudes absent from the
  training grid, never shuffled adjacent state pairs. The 3.05 case is also
  outside the training range and close to the separatrix.
- Seeds: 7 for the representative artifact; 7, 17, and 29 for the committed
  sensitivity check. Each model receives an independent, identically seeded
  data-loader shuffle so training order cannot change another model's result.
- Outputs: environment versions, parameter counts, final losses, and every
  per-amplitude metric are written to JSON.
- Checks:

  ```bash
  uv run ruff check .
  uv run pytest
  uv run python scripts/check_portfolio_results.py
  uv run python scripts/check_run_health.py results/portfolio/metrics.json
  ```

The committed figure and metrics were produced by the repository's
`benchmark` command. See [Scientific scope](SCIENTIFIC_SCOPE.md) for the exact
claim boundary and [Architecture](ARCHITECTURE.md) for the implementation map.

## Project structure

```text
src/learned_koopman/
├── physics.py              # symplectic simulator and analytic frequency
├── data.py                 # trajectory windows and held-out amplitudes
├── models/
│   ├── baselines.py
│   ├── fixed_koopman.py
│   └── energy_conditioned.py
├── training.py             # explicit, model-specific objectives
├── evaluation.py           # autonomous rollouts and physical metrics
├── experiment.py           # reproducible artifacts
└── cli.py                  # one-command entrypoint
```

## Project history

This project began as a compact 2023 experiment around an encoder, learned
linear latent evolution, and decoder for pendulum dynamics. The original
prototype is preserved in
[`legacy/2023-prototype`](legacy/2023-prototype) and at the Git tag
`prototype-2023`.

The current edition returns to that idea with physics-grounded simulation,
autonomous rollout evaluation, reproducible benchmarks, and explicit
scientific scope.

## Prior art and context

Learned Koopman observables and linearly recurrent autoencoders predate this
project. Particularly relevant references are:

- [Takeishi et al.](https://arxiv.org/abs/1710.04340), *Learning Koopman
  Invariant Subspaces for Dynamic Mode Decomposition* (NeurIPS 2017);
- [Lusch, Kutz, and Brunton](https://doi.org/10.1038/s41467-018-07210-0),
  *Deep learning for universal linear embeddings of nonlinear dynamics*
  (Nature Communications 2018);
- [Otto and Rowley](https://doi.org/10.1137/18M1177846), *Linearly Recurrent
  Autoencoder Networks* (SIAM JADS 2019);
- [Azencot et al.](https://proceedings.mlr.press/v119/azencot20a.html),
  *Consistent Koopman Autoencoders* (ICML 2020).

This repository is a polished educational and experimental PyTorch project, not
a claim of a new universal Koopman theorem or state-of-the-art forecasting
system.

## License

[MIT](LICENSE)
