# Koopman mechanics workbench

## Executive recommendation

Learned Koopman should become a local scientific instrument for nonlinear
mechanics:

> Give it repeated trajectories. It discovers candidate quantities that
> organize the motion, fits simple evolution laws conditioned on those
> quantities, shows where the laws work, and exports a reduced model that
> refuses to hide its unsupported regions.

The highest-leverage path is not a larger global autoencoder. It is a
**gauge-fixed canonical Koopman atlas with residual-calibrated validity**.

The flagship working slice now handles uniformly sampled, canonical
one-degree-of-freedom conservative trajectories. It learns an exactly
symplectic transformation, a physical action, and a Hamiltonian normal form;
compares recursive prediction on complete held-out trials; checks action
against closed-orbit phase-space area; writes an engineering report; and
exports a support-gated predictor. The earlier label-free invariant and
polynomial Koopman family remain useful when the measured state is not known to
be canonical.

The tool should not yet be described as hardware-ready, generally certified,
or suitable for dissipative, forced, stochastic, partially observed, or
high-dimensional systems. Those regimes require different mathematical
contracts.

## First use

Generate a nonlinear oscillator dataset:

```bash
uv run learned-koopman generate-example \
  --output examples/duffing-trajectories.csv
```

Train the canonical model:

```bash
uv run learned-koopman canonical-train examples/duffing-trajectories.csv \
  --position-column position \
  --momentum-column velocity \
  --reference-column energy \
  --quick \
  --output results/my-koopman-hj
```

The `energy` column is optional and is excluded from training. When supplied,
it tests the discovered coordinate only after the model has been fit.

Use the exported model:

```bash
uv run learned-koopman canonical-predict results/my-koopman-hj/model.pt \
  --initial 1.2 0.0 \
  --steps 300 \
  --output results/my-koopman-hj/prediction.csv
```

The Python entrypoint is:

```python
from pathlib import Path

from learned_koopman.canonical_experiment import (
    CanonicalExperimentConfig,
    run_canonical_experiment,
)
from learned_koopman.trajectory import load_trajectory_csv

data = load_trajectory_csv(
    Path("experiment.csv"),
    state_columns=("position", "velocity"),
    trajectory_column="trial_id",
    time_column="time",
)
run_canonical_experiment(
    data,
    Path("results/experiment"),
    config=CanonicalExperimentConfig.full(seed=7),
)
```

See [`KOOPMAN_HJ_FRONTIER.md`](KOOPMAN_HJ_FRONTIER.md) for the exact model,
closest prior art, practical systems, and research threshold.

## Mathematical center

### An invariant is a zero-generator eigenfunction

For a flow \(\Phi^t\), the Koopman operator acts on observables:

\[
U^t g = g\circ\Phi^t.
\]

A conserved quantity \(I\) satisfies

\[
U^t I=I,
\]

or, for the continuous generator \(\mathcal L\),

\[
\mathcal L I=0.
\]

Its level sets

\[
\mathcal M_c=\{x:I(x)=c\}
\]

define a candidate partition into invariant fibers. They form a useful
one-dimensional quotient only when the coordinate separates the relevant
trajectory families. The current workbench learns one candidate
\(I_\theta(x)\) using only state trajectories and their membership:

\[
\mathcal J_I =
\lambda_c\,\mathbb E_j\operatorname{Var}_t I_\theta(x_{j,t})
+\lambda_g\,\mathcal J_{\mathrm{neighbor}}
+\lambda_v(\operatorname{Std}_j\bar I_j-1)^2
+\lambda_0(\mathbb E_j\bar I_j)^2.
\]

The first term makes the coordinate constant inside each run. The variance
term prevents the constant solution. The trajectory-set graph encourages a
smooth candidate coordinate without supplying an energy, amplitude, phase, or
frequency label. A conserved scalar is generally identifiable only up to a
smooth monotone reparameterization; the variance and centering terms choose
one numerical gauge for this run.

This is related to data-driven conservation-law discovery such as
[AI Poincaré](https://doi.org/10.1103/PhysRevLett.126.180604), but the current
loss and test contract are narrower: one scalar, fully observed,
near-conservative trajectories.

### A field of operators, not one matrix

For an integrable one-degree-of-freedom system, action-angle coordinates obey

\[
\dot I=0,\qquad \dot\phi=\omega(I).
\]

On a fixed fiber \(I=I_0\), \(e^{ik\phi}\) is an eigenfunction of the
fiber-restricted generator:

\[
\left.\mathcal L\right|_{I_0}e^{ik\phi}
=ik\omega(I_0)e^{ik\phi}.
\]

If \(\omega\) varies with \(I\), there is no single global eigenvalue here.
The relation is a fiberwise or direct-integral spectral family.

This explains why a single finite matrix can fail across nonlinear orbit
families: the system carries a continuum of frequencies. It also explains why
the original energy-conditioned rotation was useful.

The workbench's first general approximation is deliberately transparent:

\[
\psi(x_{k+1})
\approx
\psi(x_k)K(c),
\qquad
K(c)=\sum_{r=0}^{R}\hat c^rK_r,
\qquad
c=I_\theta(x_0).
\]

Here \(\psi\) is a constant, linear, or quadratic dictionary and \(\hat c\) is
the normalized learned invariant. Training is one ridge regression after the
invariant has been learned. A complete held-out trajectory is rolled out
recursively with the fiber coordinate computed from its initial state only and
then fixed. Future held-out samples never condition the forecast.

This operator family is not automatically symplectic, stable, or exact. Its
advantage is inspectability: every matrix, spectrum, residual, and baseline is
visible. The new canonical backend is exactly symplectic and learns
\(H\circ F^{-1}=h(I)\);
[generating-function neural networks](https://proceedings.mlr.press/v139/chen21r.html)
remain an important matched neighbor and baseline.

### Where an atlas becomes necessary

At a one-degree-of-freedom saddle separatrix, \(\omega(I)\to0\), the period
diverges, and orbit topology changes. Action-angle coordinates become
singular. A mathematically coherent atlas uses local coordinates
\(\psi_\alpha\), local maps \(F_\alpha\), and overlap transitions

\[
T_{\beta\alpha}
=\psi_\beta\circ\psi_\alpha^{-1}.
\]

The important learnable contracts are:

\[
T_{\beta\alpha}\circ F_\alpha
\approx
F_\beta\circ T_{\beta\alpha}
\quad\text{(overlap conjugacy)},
\]

\[
T_{\gamma\alpha}
\approx
T_{\gamma\beta}\circ T_{\beta\alpha}
\quad\text{(cocycle consistency)}.
\]

For Hamiltonian charts, local evolution and transitions should also obey

\[
DF_\alpha^\top JDF_\alpha\approx J,
\qquad
DT_{\beta\alpha}^\top JDT_{\beta\alpha}\approx J.
\]

[CANDyMan](https://www.nature.com/articles/s42256-022-00575-4) establishes the
value of learned manifold charts. The opportunity here is to add mechanical
structure, explicit gluing laws, Koopman residuals, and abstention.

### Residuals should control trust

A small forecast error on one test set is not a universal Koopman
certificate. Credible outputs should separate:

- finite-data spectral or generator residual;
- held-out forecast calibration;
- physical structure defect;
- distance from sampled support;
- overlap disagreement;
- an explicit abstention decision.

[Residual DMD](https://doi.org/10.1017/jfm.2022.1052) gives rigorous residual
machinery for spectral approximations.
[ResKoopNet](https://proceedings.mlr.press/v267/xu25y.html) brings residuals
into representation learning. Recent results on
[when data-driven dynamical learning can and cannot
succeed](https://www.nature.com/articles/s41467-026-74220-8) make the product
lesson unavoidable: assumptions, sampling coverage, and failure to certify
must be visible.

The current workbench therefore uses the word `supported` only for its
empirical held-out trajectory contract. It abstains from silent prediction
unless the fit earned a positive certificate and the initial state passes both
the learned-coordinate range and a nearest-sampled-training-state distance
gate. This is still a sampled-data heuristic, not a formal support estimate.

### Control changes the invariant

For a control-affine plant

\[
\dot x=f_0(x)+\sum_j u_jf_j(x),
\]

the generator has the corresponding structure

\[
\mathcal L_u=\mathcal L_0+\sum_j u_j\mathcal L_j.
\]

If \(I\) is invariant under free motion,

\[
\frac{dI}{dt}
=\mathcal L_u I
=\sum_j u_j\mathcal L_jI.
\]

That equation is a particularly useful bridge for mechanical engineering. A
quantity discovered from free response can become an actuation-efficiency
coordinate, work audit, crossing predictor, or planning state.

For damped or port-Hamiltonian systems, the correct target is a balance law,
for example

\[
\dot H=y^\top u-d(x),\qquad d(x)\ge0,
\]

not exact conservation. Current work on
[Koopman control with error bounds and closed-loop
guarantees](https://arxiv.org/abs/2509.02839) shows why prediction quality alone
is insufficient.

### Stochastic motion is a different operator

For a Markov process,

\[
U_\tau g(x)
=\mathbb E[g(X_{t+\tau})\mid X_t=x].
\]

Its adjoint propagates densities. A positive row-stochastic matrix is a valid
finite probability operator, but validity is not usefulness—the v3 transfer
experiment is a concrete counterexample.

The stochastic backend should use variational slow-process objectives such as
[VAMP](https://deeptime-ml.github.io/latest/notebooks/vamp.html) and require
independent Chapman–Kolmogorov, branching, Ulam, occupancy, and no-operator
comparisons. It remains experimental until those gates pass.

## Engineer-facing contract

### Input

CSV columns:

```text
trajectory_id,time,q1,p1[,q2,p2,...][,reference]
```

Current requirements:

- at least six complete trajectories;
- at least 32 samples per trajectory;
- finite, strictly increasing, approximately uniform timestamps;
- a common sampling interval across trials;
- fully observed numerical state;
- autonomous, near-conservative motion.

Longer trials are truncated to the shortest complete trial. The tool does not
silently resample, smooth, differentiate, fill missing values, or infer units.

### Output

```text
results/my-system/
├── manifest.json   # source, split, method, matrices, metrics, certificate
├── model.pt        # weights-only loadable model bundle
├── overview.png    # invariant, rollouts, held-out trace, fitted spectrum
└── report.html     # human-readable engineering report
```

The loadable bundle contains:

- state names and normalization;
- invariant-network architecture and weights;
- polynomial operator-family matrices;
- sampling interval;
- fitted invariant range.

Loading uses PyTorch's weights-only mode and reconstructs known local classes.

## Current-state architecture

```text
trajectory CSV
  -> TrajectoryDataset validation
  -> complete-run train / held-out split
  -> state normalization from training data
  -> label-free LearnedInvariant
  -> trajectory means c_j
  -> polynomial observables psi(x)
  -> fibered ridge regression K(c)
  -> recursive held-out rollouts
  -> global quadratic EDMD + persistence falsifiers
  -> empirical certificate
  -> report + manifest + model bundle
```

Primary modules:

- `trajectory.py`: data contract, CSV validation, Duffing example;
- `models/invariant.py`: dimension-general scalar network;
- `invariant_experiment.py`: grouped-trajectory objective;
- `operator_family.py`: observable dictionary and operator field;
- `workbench.py`: split, fit, evaluation, certificate, report, export;
- `cli.py`: `generate-example`, `analyze`, and `predict`.

## Adjacent tools and practical fit

| Project or method | Strength | How this project should relate |
|---|---|---|
| [PyKoopman](https://pykoopman.readthedocs.io/) | EDMD, NNDMD, DMDc, observables, examples | Interoperate and benchmark; do not reimplement every estimator |
| [PySINDy](https://pysindy.readthedocs.io/en/stable/) | sparse equations, control, constrained libraries | Add as an interpretable local-law backend |
| [deeptime](https://deeptime-ml.github.io/latest/) | VAMP/TICA and stochastic kinetics | Use for stochastic baselines and scoring |
| [Residual DMD](https://doi.org/10.1017/jfm.2022.1052) | spectral residuals and pseudospectra | Make residual-ranked modes and rejection a core certificate |
| [CANDyMan](https://www.nature.com/articles/s42256-022-00575-4) | learned intrinsic manifold charts | Add exact physical structure and gluing contracts |
| [Spectral submanifolds](https://www.nature.com/articles/s41467-022-28518-y) | nonlinear normal modes for mechanics | Treat as a serious local-model baseline and option |

The product opening is not that these ingredients are absent from research. It
is that engineers still lack a small inspectable workflow joining discovered
invariants, local operator families, structure checks, applicability, and
portable prediction without hiding falsification.

## Build strategy

### Phase 1 — deterministic workbench: working now

- CSV trajectory ingestion;
- complete-run split;
- one scalar candidate invariant;
- quadratic observable dictionary;
- polynomial operator family;
- global quadratic EDMD and persistence baselines;
- empirical support certificate;
- HTML/PNG/JSON report;
- load and predict API;
- Duffing actual run.

### Phase 2 — nonlinear modal instrument

- explicit oscillator schema with positions, velocities, angle topology, and
  units;
- learned phase coordinate on each invariant fiber;
- frequency/backbone curve with empirical-period falsifier;
- multi-invariant discovery with gradient-rank deflation;
- calibrated `supported / caution / abstain` envelope;
- held-out load, geometry, damping, and sampling parameters.

### Phase 3 — structure-preserving atlas

- typed chart and transition interfaces;
- exactly symplectic local-map backend;
- overlap conjugacy and cocycle losses;
- residual-certified routing and chart creation;
- elliptic, saddle, and rotational regimes;
- comparison with SSM, PyKoopman, PySINDy, and direct neural baselines.

### Phase 4 — forced and dissipative mechanics

- controls and plant parameters in the data schema;
- bilinear local generators;
- work/passivity and port-Hamiltonian balance;
- controllability and actuation-efficiency maps;
- MPC-facing export;
- closed-loop evaluation against nonlinear MPC and energy shaping.

### Phase 5 — stochastic and partial observation

- delay/history observation models;
- noise and smoothing sensitivity, never silent preprocessing;
- VAMP/VAC coordinates and transfer models;
- implied timescales, CK, branching, and rare-event calibration;
- sensor-placement and observability diagnostics.

## Evaluation and promotion gates

A workbench model is promoted only when:

1. train and test are disjoint complete trajectories;
2. the invariant is noncollapsed;
3. held-out invariant drift is below the predeclared threshold;
4. every held-out initial state passes the learned-coordinate and
   sampled-state distance gates;
5. recursive fibered rollout beats both global quadratic EDMD and persistence;
6. the certificate is reconstructed from stored evidence;
7. export/load reproduces invariant values and rollout;
8. every optional physical or reference label is absent from optimization.

Future chart models additionally require overlap, cocycle, and physical-form
defects. Future control models require action-blind, work-balance, and
closed-loop falsifiers. Future stochastic models require VAMP, CK, branching,
and Ulam comparisons.

## Red team and rejected paths

- **Generic deep Koopman platform:** too broad, weakly differentiated, and
  likely to hide finite-rank failure behind model capacity.
- **Dashboard first:** a UI cannot supply a scientific data contract or
  trustworthy operator.
- **One global certificate:** mathematically misleading; spectral residual,
  physical defect, forecast calibration, and coverage are different claims.
- **Conservation loss for damped data:** wrong physics; use a balance law.
- **MPC before model applicability:** unsafe research order; prediction and
  error bounds must come first.
- **Stochastic transfer as a promoted backend today:** contradicted by the
  repository's own falsification result.
- **Hardware-useful claim from synthetic Duffing:** premature. The next
  external proof must be one measured oscillator dataset with sensor noise,
  units, and repeated trials.

## Verification posture

Verified locally:

- v3.2 repository and test contracts, with source revision and artifact
  fingerprints recorded in the workbench manifest;
- dimension-general invariant model;
- Duffing CSV ingestion;
- label-free fit with optional reference excluded;
- export/load/predict round trip;
- complete held-out comparison against global quadratic EDMD and persistence;
- manifest reconstruction and report generation.

The committed Duffing result is one deterministic seed-7 split. It is a
working product slice and falsifiable case study, not split/seed robustness or
hardware validation.

Primary-source-backed:

- Koopman observable and generator framing:
  [Modern Koopman Theory](https://doi.org/10.1137/21M1401243);
- conservation-law discovery:
  [AI Poincaré](https://doi.org/10.1103/PhysRevLett.126.180604);
- learned charts:
  [CANDyMan](https://www.nature.com/articles/s42256-022-00575-4);
- exact symplectic maps:
  [Chen and Tao](https://proceedings.mlr.press/v139/chen21r.html);
- residual-certified spectral analysis:
  [ResDMD](https://doi.org/10.1017/jfm.2022.1052) and
  [ResKoopNet](https://proceedings.mlr.press/v267/xu25y.html);
- nonlinear mechanical reduced models:
  [spectral submanifolds](https://www.nature.com/articles/s41467-022-28518-y);
- controlled Koopman guarantees:
  [2025 control overview](https://arxiv.org/abs/2509.02839);
- stochastic variational modeling:
  [VAMPnets](https://www.nature.com/articles/s41467-017-02388-1).

Open gaps:

- no measured hardware data;
- no explicit units or topology schema;
- one invariant only;
- no independent empirical-frequency check;
- no exact local symplecticity;
- no learned transition maps or gluing certificate;
- no calibrated probabilistic uncertainty;
- no forced, damped, stochastic, or partial-observation workbench path.

## Final recommendation

Treat the present workbench as the first real instrument, not the finished
product. Its next decisive proof is:

> repeated measured free-response trials → unlabeled invariant → nonlinear
> backbone curve → invariant-conditioned reduced model → calibrated abstention.

Then build the distinctive research result on top:

> invariant-first, exactly structure-preserving local Koopman atlases with
> residual-certified routing and gluing laws.
