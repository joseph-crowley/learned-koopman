# Learned Koopman

[![CI](https://github.com/joseph-crowley/learned-koopman/actions/workflows/ci.yml/badge.svg)](https://github.com/joseph-crowley/learned-koopman/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Discover the hidden coordinates that organize nonlinear mechanical motion,
then learn a simple evolution law on each coordinate fiber.**

Learned Koopman is a small, inspectable PyTorch workbench for repeated
trajectory data. Give it free-response measurements from a low-dimensional
mechanical system. It learns a candidate conserved coordinate, uses that
coordinate to index a family of local Koopman operators, tests the model on
complete unseen runs, and exports a predictor with an explicit support gate.

![The workbench discovers a trajectory coordinate, beats global quadratic
EDMD on held-out rollouts, and exposes its fitted spectrum.](results/mechanics-workbench/overview.png)

## Try it in five minutes

Install [`uv`](https://docs.astral.sh/uv/), then run the included conservative
Duffing oscillator:

```bash
git clone https://github.com/joseph-crowley/learned-koopman.git
cd learned-koopman
uv sync --extra dev

uv run learned-koopman generate-example \
  --output examples/my-duffing.csv

uv run learned-koopman analyze examples/my-duffing.csv \
  --state-columns position velocity \
  --reference-column energy \
  --quick \
  --output results/my-duffing
```

Open `results/my-duffing/report.html`. The run also writes:

- `overview.png` — the learned coordinate, baseline comparison, held-out
  trajectory, and fitted finite-operator spectrum;
- `manifest.json` — the split, metrics, certificate, provenance, and artifact
  fingerprints;
- `model.pt` — a weights-only bundle for coordinate evaluation and rollout.

Use the exported model:

```bash
uv run learned-koopman predict results/my-duffing/model.pt \
  --initial 1.2 0.0 \
  --steps 300 \
  --output results/my-duffing/prediction.csv
```

The predictor refuses a rejected fit or an initial state outside both the
learned-coordinate range and the sampled training-state neighborhood. An
explicit `--allow-unsupported` override keeps exploratory use possible.

## Bring your own trajectories

The CSV contract is deliberately plain:

```csv
trial_id,time,position,velocity
run-01,0.000,0.800,0.000
run-01,0.010,0.799,-0.021
run-02,0.000,1.100,0.000
run-02,0.010,1.099,-0.030
```

```bash
uv run learned-koopman analyze measurements.csv \
  --trajectory-column trial_id \
  --time-column time \
  --state-columns position velocity \
  --output results/my-rig
```

The first workbench profile expects:

- at least six complete trajectories and 32 samples per trajectory;
- a shared, approximately uniform sampling interval;
- fully observed numerical state columns;
- low-dimensional, autonomous, near-conservative motion.

Trajectories may have different lengths; the loader truncates them to the
shortest complete run and records every original length. A reference column
such as measured energy is optional. It is excluded from training and used
only as a post-fit scientific check.

## The mathematical idea

A conserved quantity is a zero-generator Koopman eigenfunction:

$$
U^t I=I,
\qquad
\mathcal L I=0.
$$

The neural coordinate $I_\theta(x)$ is trained from states and trajectory
membership. It is encouraged to stay constant along each run, vary across
runs, and remain smooth between neighboring trajectory sets. No energy,
amplitude, phase, or frequency label enters this fit.

The workbench then learns a transparent polynomial family:

$$
\psi(x_{k+1})
\approx
\psi(x_k)K(\hat c),
\qquad
K(\hat c)=K_0+\hat cK_1+\hat c^2K_2,
\qquad
c=I_\theta(x_0).
$$

Here $\psi$ is an explicit constant, linear, or quadratic observable
dictionary, and $\hat c$ is the learned coordinate normalized on the training
set. The invariant selects the local operator, and its value is computed from
the initial state only. This fibered construction can represent an
amplitude-dependent frequency law without forcing every trajectory through one
global finite matrix.

The deeper theory and product direction are in
[`PHYSICS_WORKBENCH.md`](PHYSICS_WORKBENCH.md): phase and frequency recovery,
symplectic local maps, chart overlap laws, residual-calibrated validity,
controlled balance laws, and stochastic transfer.

## What the checked-in run establishes

The promoted Duffing result uses 22 training trajectories and eight complete
held-out trajectories in one deterministic seed-7 split.

| Held-out measurement | Result |
|---|---:|
| Learned-coordinate drift | **0.0024** |
| Fibered quadratic EDMD rollout RMSE | **0.0755** |
| Global quadratic EDMD rollout RMSE | 0.4240 |
| Persistence rollout RMSE | 1.5636 |
| Energy rank correlation after training | **1.000** |

The optional energy column is absent from both optimization stages. The
fibered model is evaluated recursively, conditioned only on each unseen
initial state. The matched global quadratic EDMD error is 5.6 times larger on
this split.

The [human report](results/mechanics-workbench/report.html) and
[machine-readable manifest](results/mechanics-workbench/manifest.json) contain
the full evidence. This is a synthetic-mechanics result from one split. The
next decisive test is a measured oscillator with units, sensor noise, and
repeated trials.

## Trust is part of the model

The workbench treats a failed experiment as useful output:

- train and test are separated by complete trajectory ID;
- training normalization is fit on training trajectories only;
- held-out forecasts use $I_\theta(x_0)$, never a future-state average;
- promotion requires every held-out initial state to pass invariant-range and
  sampled-state-distance gates;
- the model bundle carries its certificate and refuses rollout after a
  negative fit;
- `scripts/check_workbench.py` reconstructs held-out errors from the source
  CSV and exported model, then verifies the source, model, report, figure, and
  clean source revision.

## The research lab behind the workbench

The repository also contains four compact experiments that stress different
parts of the larger idea:

- Label-free invariant — held-out energy $R^2=0.979$, rank $=1.000$, and
  drift $=0.0053$ over five seeds. Tests whether grouped trajectories reveal
  a conserved coordinate.
- Two-chart separatrix atlas — valid time $3.98\pm0.06$, versus
  $0.36\pm0.12$ for one chart. Shows why a coordinate singularity needs a
  second local law.
- Stochastic simplex transfer — positive row-stochastic structure passes,
  while stronger baselines falsify the learned propagation. Preserves a
  mathematically valid negative result.
- Controlled crossing — actuator gain $0.35\rightarrow1.000$, with 9/12
  real crossings recovered. A clean grey-box system-identification exercise.

Run the CPU-sized integrated demonstration:

```bash
uv run learned-koopman lab --quick
```

The research cells are auditable probes with separate mathematical contracts,
baselines, and falsifiers. Their full results live in
[`results/research-lab/manifest.json`](results/research-lab/manifest.json),
[`results/atlas/robustness.json`](results/atlas/robustness.json), and
[`SCIENTIFIC_SCOPE.md`](SCIENTIFIC_SCOPE.md).

## Python API

```python
from pathlib import Path

from learned_koopman.trajectory import load_trajectory_csv
from learned_koopman.workbench import WorkbenchConfig, run_mechanics_workbench

data = load_trajectory_csv(
    Path("measurements.csv"),
    trajectory_column="trial_id",
    time_column="time",
    state_columns=("position", "velocity"),
)

manifest = run_mechanics_workbench(
    data,
    Path("results/my-rig"),
    config=WorkbenchConfig.full(seed=7),
)
print(manifest["certificate"]["status"])
```

## Verify the repository

```bash
uv run ruff check .
uv run pytest
uv run python scripts/check_workbench.py
uv run python scripts/check_research_lab.py
uv run python scripts/check_portfolio_results.py
uv run python scripts/check_atlas_results.py
```

The project runs on CPU with Python 3.11+, PyTorch, NumPy, and Matplotlib.

## Read next

- [Physics workbench](PHYSICS_WORKBENCH.md) — mathematical center, engineering
  contract, adjacent research, and build sequence;
- [Scientific scope](SCIENTIFIC_SCOPE.md) — exact claims, supplied structure,
  learned structure, and present limits;
- [Architecture](ARCHITECTURE.md) — implementation and experiment flow;
- [Research roadmap](RESEARCH_ROADMAP.md) — prior art, novelty ladder, and
  falsifiable next experiments;
- [`legacy/2023-prototype`](legacy/2023-prototype) — the original exploratory
  model that started the project.

## Citation and license

If this project contributes to your work, cite the repository using
[`CITATION.cff`](CITATION.cff).

[MIT](LICENSE)
