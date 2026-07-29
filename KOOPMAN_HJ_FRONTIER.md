# The Koopman + Hamilton–Jacobi Frontier

**Status:** implementation-backed research program

**Evidence date:** 2026-07-28

**Flagship direction:** gauge-fixed canonical Koopman atlases for useful
mechanical system identification, simulation, and control

## Executive Recommendation

Build this project around a **Canonical Koopman Atlas**, not a generic Koopman
autoencoder and not a standalone neural Hamilton–Jacobi PDE solver.

The useful core is:

\[
(q,p)
\xrightarrow[\text{exact symplectic}]{F_\theta}
(Q,P),
\qquad
I_j=\frac{Q_j^2+P_j^2}{2},
\qquad
H\circ F_\theta^{-1}=h_\psi(I),
\]

\[
\dot I=0,
\qquad
\dot\phi=\nabla_I h_\psi(I),
\qquad
\psi_k=e^{ik\cdot\phi},
\qquad
\mathcal L\psi_k=i\,k\cdot\nabla_Ih_\psi\,\psi_k.
\]

For one degree of freedom, that core is now implemented and works on the
checked-in Duffing experiment. It learns a canonical chart and Hamiltonian
normal form without energy labels, recursively predicts held-out trajectories,
and independently checks the latent action against
\((2\pi)^{-1}\oint p\,dq\).

The high-ceiling research contribution is the full combination:

1. **exact symplectic learned conjugacy**, rather than a symplectic penalty;
2. **Hamiltonian-derived fiberwise Koopman spectrum**, rather than an arbitrary
   latent linear matrix;
3. **canonical gauge fixing against physical action**, rather than reporting
   only correlation with a conserved scalar;
4. **charted topology and calibrated abstention** near separatrices,
   resonances, impacts, and nonintegrable regions;
5. **controlled slow action drift around a trusted conservative core**; and
6. **mechanics-facing outputs**: nonlinear modes, backbone curves, resonance
   maps, fast surrogates, reduced control models, and validity certificates.

The current code is an unusually good PyTorch example and a credible research
seed. It is **not yet a defensible novel research result** because the evidence
is one synthetic system and because Action-Angle Networks, neural canonical
transformations, exact symplectic map learners, and a very recent
Buckingham-Symplectic Network already occupy nearby ground. Novelty must come
from a clearly differentiated method plus matched baselines, multiple systems,
robustness, and preferably measured hardware.

## Work Performed

This pass:

- inspected the repository architecture, mathematical claims, result
  manifests, model exports, tests, and active branches;
- traced the original invariant-conditioned Koopman idea to its stronger
  canonical-mechanics form;
- reviewed primary literature across Koopman spectral theory, action-angle
  learning, exact symplectic networks, neural canonical transformations,
  Hamiltonian networks, chart atlases, continuous spectra, HJ PDE solvers, and
  HJB control;
- implemented an exact-symplectic one-degree-of-freedom canonical Koopman
  model;
- implemented a closed-orbit action and Hamilton–Jacobi audit;
- trained and evaluated the model on complete held-out Duffing trajectories;
- exported a support-gated model and prediction command;
- added human and machine reports, artifact hashes, validators, tests, and CI
  actual runs; and
- converted the findings into the staged research and product program below.

## Evidence Posture

The checked-in result is an `actual_run` on synthetic data:

| Claim | Evidence |
|---|---:|
| Held-out recursive normalized rollout RMSE | 0.0551 |
| Persistence RMSE on the same runs | 1.5636 |
| Held-out observed action drift | 0.0092 |
| Held-out Koopman phase residual | 0.00053 |
| Numerical inverse error | \(4.77\times10^{-7}\) |
| Numerical symplectic defect | \(3.58\times10^{-7}\) |
| Model-rollout action drift | \(2.92\times10^{-5}\) |
| Held-out latent action vs. physical action | \(R^2=0.9999996\) |
| Held-out action slope / intercept | 0.9987 / 0.00015 |
| Held-out \(dh/dI\) vs. measured frequency | 1.77% normalized RMSE |
| Held-out \(h(I)\) vs. reference energy shape | 0.49% normalized RMSE |

Training excludes the reference energy, empirical action, and complete
held-out trajectories. The action and learned-Hamiltonian checks are evaluated
on the held-out trajectory IDs in the top-level certificate. The physical
\(dH/dJ=\omega\) audit also reports the denser all-trajectory numerical ruler.

Evidence artifacts:

- `results/koopman-hj/manifest.json`
- `results/koopman-hj/report.html`
- `results/koopman-hj/model.pt`
- `results/koopman-hj/action-audit/manifest.json`

This evidence supports “the construction works on this problem.” It does not
support “state of the art,” “novel,” “general,” “robust,” “hardware-ready,” or
“safe for control.”

## Delegate Inputs

No external model or subagent verdict is treated as evidence in this pass. The
review uses live repository behavior and primary published or preprint sources.
That is deliberate: novelty and scientific claims here must survive direct
prior-art comparison and executable tests, not model agreement.

## Repo Current-State Map

### Working flagship

`canonical-train` learns:

\[
\Phi_{\Delta t}
=F_\theta^{-1}\circ
R_{\Delta t\,h_\psi'(I)}
\circ F_\theta.
\]

`F_\theta` is composed of canonical translations, reciprocal scaling, and
alternating neural shears. Every layer has an analytic inverse. The latent
rotation is the exact flow of \(h_\psi(I)\), so the complete update is
symplectic by construction.

`canonical-predict` loads the exported model, checks its fit certificate and
observed action range, and writes a physical rollout.

`hj-audit` measures:

\[
J=\frac{1}{2\pi}\oint p\,dq,\qquad
\omega=\frac{2\pi}{T},\qquad
\frac{dH}{dJ}=\omega.
\]

It also distinguishes an arbitrary monotone invariant gauge from canonical
action. This is an evaluator, not the model.

### Useful secondary path

`analyze` learns a general scalar invariant and a polynomial family of local
Koopman operators. It does not require the supplied states to be canonical.
That path is useful for general trajectory organization and as an ablation, but
it is neither exactly symplectic nor gauge-fixed.

### Research probes

The repository also has:

- a two-chart near-separatrix pendulum model;
- label-free invariant discovery;
- a positive stochastic transfer operator with a preserved negative result;
- controlled actuator identification through energy-shell crossings.

These are valuable ingredients for the atlas, uncertainty, and control
extensions, but they are not yet connected to the canonical core.

## SOTA Landscape

### 1. Direct Koopman–Hamilton–Jacobi theory

Vaidya’s 2025 paper
[When Koopman Meets Hamilton and Jacobi](https://arxiv.org/abs/2504.07346)
is the most direct conceptual neighbor. It develops two Koopman-spectral
procedures for recovering invariant Lagrangian submanifolds and HJ solutions,
with a convex eigenfunction approximation framework, convergence analysis, and
an optimal-control demonstration.

**What it owns:** the direct theoretical bridge from Koopman eigenfunctions to
HJ Lagrangian submanifolds and control.

**Opening here:** a trainable canonical mechanics system that learns
action-angle normal forms from trajectory data, exposes physical engineering
quantities, handles charts, and carries empirical validity gates. The current
project should cite Vaidya prominently and should not claim to invent the
Koopman–HJ connection.

### 2. Learned action-angle simulation

[Learning Integrable Dynamics with Action-Angle Networks](https://arxiv.org/abs/2211.15338)
learns a nonlinear transformation into action-angle coordinates where
evolution is linear. It reports efficient prediction without higher-order
integration and good scaling to time jumps.

**What it owns:** action-angle networks as fast learned simulators for
integrable systems.

**Opening here:** exact symplecticity, area-based action gauge fixing,
\(h(I)\)-derived frequency rather than an unconstrained frequency law,
fiberwise Koopman residuals, charted topology, and mechanics-facing
certification.

### 3. Neural canonical transformations

[Neural Canonical Transformation with Symplectic Flows](https://doi.org/10.1103/PhysRevX.10.021020)
identifies the correspondence between canonical transformations and
symplectic normalizing flows, with training from a Hamiltonian or phase-space
samples.

**What it owns:** learned canonical transformations using symplectic flows.

**Opening here:** connect the learned transformation explicitly to Koopman
harmonics, a deployable discrete world model, physical action certification,
and local model validity.

### 4. Exactly symplectic discrete maps

[GFNN](https://proceedings.mlr.press/v139/chen21r.html) learns a generating
function for an exactly symplectic discrete map and gives a long-time error
analysis with at-most-linear growth under its assumptions.

[SympNets](https://arxiv.org/abs/2001.03750) develop intrinsic
structure-preserving symplectic architectures and universal approximation
results. A newer dynamical-systems construction,
[Symplectic Neural Networks Based on Dynamical Systems](https://arxiv.org/abs/2408.09821),
adds representation results and symbolic Hamiltonian regression.

**What they own:** exact symplectic neural map learning and strong
approximation theory.

**Opening here:** constrain the map further to an interpretable integrable
normal form, recover actions/frequencies/Koopman observables, learn chart
validity, and exploit the result for reduced analysis and control. Exact
symplecticity alone is not a novel claim.

### 5. Hamiltonian neural networks and controlled world models

[Hamiltonian Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2019/hash/26cd8ecadce0d4efd6cc8a8725cbd1f8-Abstract.html)
learn a scalar Hamiltonian and use Hamilton’s equations as the inductive bias.

[Action-Conditioned Hamiltonian Generative Networks](https://proceedings.mlr.press/v283/troch25a.html)
extend abstract Hamiltonian dynamics with external actions and demonstrate a
physics-informed model-based reinforcement-learning path.

**What they own:** energy-generating neural vector fields and action-conditioned
Hamiltonian world models.

**Opening here:** use a canonical normal form to separate fast phase from slow
controlled action drift, reducing the model and control problem before applying
MPC, HJB, or policy learning.

### 6. The closest current symbolic-action neighbor

The April 2026 preprint
[Discovery of Symbolic Hamiltonian Expressions with Buckingham-Symplectic Networks](https://arxiv.org/abs/2604.00576)
combines dimensional consistency, a symplectic transformation, latent
action-angle variables, and symbolic Hamiltonian discovery. It reports
harmonic-oscillator and Kepler results.

**What it owns or plausibly contests:** much of the broad “symplectic
action-angle transformation plus interpretable Hamiltonian” territory.

**Consequence:** this project cannot responsibly pitch that broad combination
as novel. A differentiated paper must win on charted validity, explicit
Koopman spectral objects, physical action certification, controlled
perturbations, measured mechanics, or a new theorem/identifiability result.
BuSyNet must be a matched baseline if code becomes available, or reimplemented
faithfully if licensing permits.

### 7. Koopman spectral reliability beyond point spectrum

[Modern Koopman Theory for Dynamical Systems](https://doi.org/10.1137/21M1401243)
is the broad reference for embeddings, spectra, control, and the difficulty of
finite-dimensional invariant representations.

[Rigged DMD](https://doi.org/10.1137/24M1662370) addresses continuous spectra
and generalized eigenfunctions with resolvent methods and convergence results,
including integrable Hamiltonian examples.

**What they own:** the warning that a few learned eigenfunctions are not the
whole operator, especially in mixing, chaotic, or continuous-spectrum regimes.

**Opening here:** an automatic “integrable chart accepted / continuous-spectrum
or nonintegrable behavior detected / abstain or route to Rigged DMD” boundary.
This is more scientifically useful than forcing every dataset through an
action-angle model.

### 8. Manifold charts and topology

[CANDyMan](https://www.nature.com/articles/s42256-022-00575-4) learns atlases of
intrinsic state variables and local dynamics, motivated by the fact that one
global chart may not achieve intrinsic dimension.

**What it owns:** learned multi-chart intrinsic dynamics.

**Opening here:** make every chart canonical, make overlaps symplectic, enforce
action/frequency cocycles, and treat separatrices and resonance zones as
scientific routing boundaries.

### 9. HJ PDE, viscosity, HJB, and reachability

This is adjacent but distinct from classical action-angle mechanics:

- [Finite-difference least square methods for HJ equations using neural networks](https://arxiv.org/abs/2406.10758)
  combines neural approximation with a monotone consistent scheme for
  viscosity solutions.
- [Hamilton-Jacobi Based Policy-Iteration via Deep Operator Learning](https://arxiv.org/abs/2406.10920)
  uses DeepONet for families of HJB optimal-control problems.
- [Solving HJ equations by minimizing residuals of monotone discretizations](https://arxiv.org/abs/2601.21764)
  develops newer well-posedness conditions for optimization-based monotone HJ
  solvers.

**What they own:** nonsmooth value functions, viscosity solutions, terminal
conditions, reachability, and optimal-control PDEs.

**Opening here:** first learn a low-dimensional canonical model and validity
region from data; then solve HJB or reachability on the reduced action-angle
state. Do not conflate a classical generating function for integrable mechanics
with an HJB value function.

## Where the Differentiated Contribution Can Be

### Candidate method: Gauge-Fixed Canonical Koopman Atlas

The method hypothesis is:

> A charted exact-symplectic conjugacy, trained from canonical trajectories and
> constrained by a learned Hamiltonian normal form, yields physically
> identifiable actions, fiberwise Koopman spectra, fast stable simulation, and
> a useful boundary between integrable and nonintegrable behavior.

The differentiating pieces are jointly testable:

1. **Gauge identifiability.** Exact symplecticity plus topology makes latent
   circle area agree with physical action. This is stronger than learning a
   monotone conserved coordinate.
2. **Spectral–mechanical equivalence.** The same \(h(I)\) must explain
   energy shape, frequency, phase evolution, and Koopman multipliers.
3. **Atlas consistency.** Overlap maps must be symplectic, actions must agree,
   and angle changes must satisfy an integer-affine torus transition law.
4. **Residual-calibrated validity.** The system must know when a chart stops
   representing the data.
5. **Controlled perturbation around normal form.** Inputs, damping, and weak
   forcing should appear as learned slow action drift and phase correction,
   preserving the trusted autonomous core.

No single point above should be called novel until a formal prior-art review and
experiments establish the gap. The bundle is, however, a coherent and valuable
research target.

## Target Architecture

### Layer A: data and canonical-sensor contract

Inputs:

- synchronized generalized position and canonical momentum;
- trajectory/trial identity;
- sample time and units;
- optional controls, physical parameters, and post-fit references;
- optional observation model when direct momentum is unavailable.

Required checks:

- time monotonicity and sampling regularity;
- canonical units and mass conversion;
- sensor synchronization;
- complete-orbit coverage;
- conservative-window detection;
- uncertainty and missing-data flags.

### Layer B: exact canonical conjugacy

For \(n\) degrees of freedom:

\[
F_\theta:\mathbb R^{2n}\to\mathbb R^{2n},
\qquad
(DF_\theta)^\top J\,DF_\theta=J.
\]

Candidate parameterizations:

- gradient/shear SympNets;
- generating-function layers;
- Hamiltonian flow layers;
- symplectic normalizing flows;
- Lie-group or cotangent-lift maps for constrained mechanisms.

Architectural symplecticity is preferred over a soft defect penalty. Penalties
remain diagnostics and can enforce additional constraints, but they should not
carry the fundamental guarantee.

### Layer C: integrable Hamiltonian normal form

\[
I_j=\frac{Q_j^2+P_j^2}{2},
\qquad
h_\psi=h_\psi(I,\mu),
\qquad
\Omega(I,\mu)=\nabla_Ih_\psi.
\]

Here \(\mu\) contains physical parameters. The gradient relation is important:
an independently learned frequency network and energy network can disagree,
whereas one \(h_\psi\) makes the HJ identity structural.

Use dimensionally consistent or symbolic heads when interpretability is the
goal. Use monotone or convex parameterizations only when the physics warrants
them; softening systems must be allowed to decrease in frequency.

### Layer D: Koopman spectral surface

For integer multi-index \(k\):

\[
\psi_k=e^{ik\cdot\phi},
\qquad
\lambda_k(I)=i\,k\cdot\Omega(I).
\]

The useful object is a **spectral bundle over action space**, not one global
matrix. Expose:

- eigenfunction residuals;
- frequency surfaces;
- resonant integer relations \(k\cdot\Omega\approx0\);
- spectral uncertainty;
- chart support and transition laws.

### Layer E: HJ generating-function view

Locally recover or parameterize \(S(q,I)\) such that:

\[
p=\partial_q S(q,I),
\qquad
H(q,\partial_qS)=h(I),
\qquad
\phi=\partial_I S.
\]

This surface makes the connection to classical HJ explicit and supplies a
strong PDE residual. It will require multiple local branches around turning
points; one global \(S(q,I)\) is generally the wrong topology.

### Layer F: atlas and nonintegrability boundary

Charts should cover:

- libration;
- rotation;
- saddle/separatrix neighborhoods;
- resonant islands;
- impact/contact modes;
- parameter regimes.

Each overlap must check:

\[
T_{ab}^\ast\omega=\omega,
\qquad
I_b=A I_a+c,
\qquad
\phi_b=A^{-\top}\phi_a+\nabla g(I),
\]

with the appropriate integer matrix \(A\) for torus coordinates.

Route based on calibrated residuals and topology, not a black-box classifier
alone. If no chart is valid, abstain or hand off to a nonintegrable/continuous
spectrum model.

### Layer G: forcing, damping, and control

Use the learned integrable core as the zeroth-order system:

\[
\dot I
=\epsilon a_\eta(I,\phi,\mu)
+B_\eta(I,\phi,\mu)u,
\]

\[
\dot\phi
=\Omega(I,\mu)
+\epsilon b_\eta(I,\phi,\mu)
+C_\eta(I,\phi,\mu)u.
\]

This supports:

- action shaping and resonance avoidance;
- reduced nonlinear MPC;
- parameter estimation and health monitoring;
- averaging and slow-fast control;
- reduced HJB/reachability in intrinsic coordinates;
- passivity or port-Hamiltonian constraints for dissipative systems.

The conservative core stays fixed and auditable; the residual earns only the
extra behavior the data require.

## Practical Systems Worth Building

### 1. Nonlinear vibration workbench

For mechanical engineers:

- ingest free-decay or conservative-window test data;
- identify action, frequency–amplitude backbone, and nonlinear mode shape;
- compare runs, loads, temperatures, or component variants;
- predict long trajectories and phase;
- flag chart/extrapolation failure;
- export a compact digital-twin model.

This is the nearest practical product.

### 2. Resonance and detuning analyzer

For weakly coupled oscillators:

- learn \(\Omega(I,\mu)\);
- search integer relations \(k\cdot\Omega\approx0\);
- map resonance tongues and internal resonances;
- simulate averaged action exchange;
- recommend safe or high-response operating regions.

### 3. Structure-preserving reduced model for control

For robotics and flexible mechanisms:

- identify the autonomous canonical core;
- fit controlled action/phase residuals;
- plan energy transfer in action coordinates;
- compare action-space MPC with direct neural MPC and classical energy
  shaping;
- retain a physical support and residual certificate.

### 4. Mechanics diagnostic and anomaly monitor

Learn a baseline \(F,h\) on healthy data. Track:

- action drift not explained by measured input;
- shifts in \(h(I)\) or \(\Omega(I)\);
- growing Koopman residual;
- chart occupancy changes;
- loss of canonical closure.

Those are physically interpretable condition indicators, not just embedding
distance.

### 5. Reduced coordinate front end for HJB and reachability

Use the canonical atlas to reduce dimension and separate fast phase. Solve
value or reachability problems in the reduced coordinates with a monotone HJ
solver or operator learner. The reduced solver still needs its own viscosity,
boundary-condition, and safety validation.

## Build Strategy

### Stage 0 — working one-degree-of-freedom core

Done:

- exact symplectic neural conjugacy;
- \(h(I)\)-generated latent rotation;
- analytic inverse and model export;
- complete-trajectory train/test split;
- physical action and HJ audit;
- support-gated prediction;
- artifact validation and CI.

### Stage 1 — establish the paper floor

Build matched implementations or adapters for:

- Action-Angle Network;
- GFNN;
- SympNet / Hamiltonian flow network;
- HNN plus symplectic integrator;
- neural ODE;
- EDMD / extended DMD;
- unconstrained invertible conjugacy;
- the existing invariant-conditioned family;
- BuSyNet when reproducible.

Systems:

- hardening and softening Duffing;
- simple pendulum libration;
- pendulum rotation;
- Morse oscillator;
- Hénon–Heiles below and near nonintegrable regimes;
- Toda or FPUT chain;
- one measured nonlinear oscillator or vibration rig.

Run at least five seeds, three sample-time regimes, three noise levels, and
multiple training-set sizes. Predeclare metrics and do not tune baselines only
on their losing configurations.

### Stage 2 — prove the gauge-fixing claim

Ablate:

1. arbitrary scalar invariant;
2. invertible but non-symplectic map;
3. symplectic map with independent \(\omega(I)\);
4. symplectic map with one \(h(I)\);
5. full physical-action audit.

Decisive measurements:

- affine slope and intercept versus physical action;
- Poisson brackets;
- \(dH/dI-\Omega\);
- held-out orbit closure;
- action stability under noise and sparse samples;
- stability under canonical rescaling.

### Stage 3 — topology and validity

Connect the existing atlas work to the canonical model:

- libration and rotation charts;
- local saddle chart;
- overlap symplecticity;
- chart cocycle consistency;
- residual calibration;
- abstention;
- Rigged DMD or direct model fallback in rejected regions.

The decisive demonstration is not merely longer rollout. It is correct routing
through a topology change without a fake global action-angle coordinate.

### Stage 4 — multiple degrees of freedom

Start with integrable coupled examples:

- anisotropic oscillator;
- Toda lattice;
- Kepler;
- weakly coupled nonlinear modes away from resonance.

Tests:

- \(\{I_i,I_j\}=0\);
- torus reconstruction;
- vector frequency accuracy;
- integer resonance recovery;
- GL\((n,\mathbb Z)\) action ambiguity;
- chart transition consistency.

Then approach KAM and mixed phase space. The model must distinguish surviving
tori from resonance layers and chaotic regions instead of declaring the whole
system integrable.

### Stage 5 — controlled useful system

Add known control sequences and damping to a measured or high-fidelity
oscillator. Fit only the nonconservative residual around the frozen canonical
core. Compare:

- direct neural dynamics;
- Koopman with control;
- grey-box physical model;
- energy shaping;
- nonlinear MPC;
- action-space MPC;
- action-conditioned Hamiltonian world model.

Report closed-loop task success, constraint violations, energy/work balance,
support excursions, inference time, and calibration—not just prediction MSE.

## Experiment Matrix

| Experiment | Question | Decisive result |
|---|---|---|
| E1 exactness | Is the implementation structurally symplectic? | inverse, Jacobian, action drift at numerical floor |
| E2 gauge | Does symplecticity recover physical action? | held-out slope near 1; nonlinear gauge ablation fails |
| E3 HJ consistency | Does one \(h(I)\) explain energy and frequency? | low held-out \(H-h(I)-c\) and \(\Omega-\nabla h\) |
| E4 Koopman | Are phase harmonics genuine local eigenfunctions? | low generator/map residual across held-out actions |
| E5 long horizon | Does structure improve useful rollout? | accuracy and bounded invariant error vs matched baselines |
| E6 robustness | Does the result survive seeds, noise, and sampling? | confidence intervals and no threshold fragility |
| E7 topology | Does the atlas handle separatrix/rotation correctly? | valid overlaps, no global-chart fiction, calibrated abstention |
| E8 multi-DOF | Are learned actions in involution? | Poisson brackets, frequencies, resonances, torus recovery |
| E9 measured rig | Does it help an engineer? | useful backbone, prediction, parameter/anomaly result |
| E10 control | Does the reduced model improve closed-loop behavior? | task/constraint wins vs strong control baselines |

## Evaluation and Governance

### Data splits

- split by complete trajectory, operating condition, and—where possible—
  physical run;
- keep a final untouched system/parameter regime;
- never use future states to estimate an initial action during rollout;
- report interpolation and extrapolation separately.

### Required baselines

Every promoted result needs:

- persistence and known linear physics;
- direct residual MLP;
- HNN plus fair integrator;
- Action-Angle Network;
- exact symplectic map learner;
- Koopman/EDMD baseline with matched observables;
- the closest current method for the claimed contribution.

### Required metrics

- one-step and recursive physical error;
- valid prediction time;
- action and energy drift;
- symplectic and inverse defect;
- Koopman eigenfunction residual;
- physical-action calibration;
- HJ residual and frequency consistency;
- runtime and memory;
- seed and dataset-size variation;
- support coverage and calibration;
- topology/routing failure;
- control performance and constraints when applicable.

### Promotion levels

1. **Implemented:** code path and structural tests pass.
2. **Supported on current dataset:** all predeclared current-dataset gates pass.
3. **Robust synthetic result:** systems, seeds, noise, sampling, and baselines
   pass.
4. **Measured mechanics result:** independent physical trials pass.
5. **Research claim:** closest prior art, ablations, and statistical evidence
   support a specific differentiated claim.
6. **Deployable tool:** sensor contract, uncertainty, monitoring, rollback,
   documentation, and user workflow pass.

Never jump levels in prose.

## Red-Team and Rejected Alternatives

### “Just call the current result novel”

Rejected. One Duffing system is too easy: every regular one-degree-of-freedom
autonomous Hamiltonian system is locally integrable. Recent prior art is close,
especially Action-Angle Networks, neural canonical transformations, SympNets,
GFNN, and BuSyNet.

### “Use the closed-orbit action integral as the model”

Rejected as the center. It requires complete periodic orbits and only labels
the family after observation. It is excellent ground truth and calibration,
but it does not create a predictor, canonical transformation, or controller.

### “Learn any invariant and rename it action”

Rejected. A conserved scalar has arbitrary monotone gauge. Perfect rank
correlation is not canonical action. The original learned invariant showed
exactly this: perfect ordering but only 0.847 affine \(R^2\). The symplectic
model fixes the gauge and reaches held-out \(R^2=0.9999996\) with slope 0.9987.

### “Use one unconstrained encoder and decoder”

Rejected for the flagship. A latent rotation can be symplectic while the
physical map is not. The encoder and inverse must themselves be canonical if
the system is to make physical action and HJ claims.

### “Penalize the symplectic defect”

Rejected as the fundamental guarantee. Soft penalties can be useful
regularizers, but a small sampled defect does not prove the map is symplectic
elsewhere. Use an exact architecture and retain the defect as an implementation
check.

### “Force one global action-angle chart”

Rejected. Angle coordinates are singular at equilibria and require topology
changes across separatrices and rotations. Multi-degree systems add torus
ambiguities and resonance. Use an atlas.

### “Call classical HJ and HJB the same problem”

Rejected. The current system learns a canonical normal form and classical HJ
structure for conservative mechanics. HJB value functions, viscosity
solutions, reachability, terminal costs, and optimal control are a separate
layer with separate numerical obligations.

### “Use rollout MSE as the only score”

Rejected. A black box may win a short window while learning no stable
mechanics. Conversely, a beautiful symplectic map can fit the wrong tori.
Require physical error, action, HJ, Koopman, exactness, support, and baseline
checks together.

### “Extend symplectic dynamics directly to damping”

Rejected. Dissipative flow is not symplectic. Preserve a symplectic autonomous
core and model forcing/damping through port-Hamiltonian, contact, conformally
symplectic, or explicit action-drift structure.

## Verification and Gaps

Verified now:

- package and CLI paths run on CPU;
- canonical network has analytic inverse;
- unit tests check exact action preservation and numerical symplecticity;
- complete-run held-out split is recorded;
- reference energy is excluded from training;
- committed model reloads and predicts;
- artifact and source fingerprints validate;
- the canonical result passes its current-dataset certificate.

Still missing:

- multi-seed canonical result;
- matched Action-Angle, GFNN, SympNet, HNN, and BuSyNet comparisons;
- non-Duffing systems;
- noise, partial observation, irregular sampling, and parameter variation;
- direct momentum-estimation path;
- measured apparatus;
- charted canonical overlaps;
- multi-action/involution tests;
- controlled action drift and closed-loop comparison;
- formal identifiability or approximation theorem;
- independent research review and reproducibility run.

## Final Recommendation

Keep and promote the current one-degree-of-freedom model as the **working
canonical core**. It is mathematically cleaner, more practically useful, and
more faithful to the original ambition than the post-hoc action audit or the
unconstrained invariant-conditioned operator alone.

Do not yet market it as a novel research result. Market it as:

> an implementation-backed PyTorch research workbench that learns an
> exact-symplectic canonical Koopman normal form and audits it against
> Hamilton–Jacobi mechanics.

The next decisive move is a compact paper-floor benchmark:

1. hardening and softening Duffing plus pendulum libration/rotation;
2. five seeds, noise and sample-time sweeps;
3. Action-Angle Network, GFNN/SympNet, HNN, EDMD, and unconstrained-conjugacy
   baselines;
4. gauge-fixing and \(h(I)\) ablations; and
5. one measured nonlinear oscillator.

If the exact-symplectic, gauge-fixed model wins on physical action,
frequency/Hamiltonian consistency, long-horizon stability, and useful
mechanics outputs—not merely RMSE—then the project has a defensible paper and a
real tool. The atlas and controlled-action extensions are the larger frontier
after that floor is established.
