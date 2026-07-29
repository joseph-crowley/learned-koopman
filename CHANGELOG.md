# Changelog

## 3.4.0 — 2026-07-29

This release turns chart ambiguity from a roadmap warning into a runnable
resonance-metrology experiment.

### Added

- exact twist-kick, canonical observation-chart, and adversarial-gauge
  fixtures with inverse and area-preservation tests;
- a trajectory-band complex residual estimator with weighted-Birkhoff
  rotation profiles, per-bin support checks, and explicit resonance-crossing
  abstention;
- learned S1/S2 chart ensembles, raw/oracle/null/shuffled/wrong-harmonic
  controls, kick-size detection sweeps, estimator-variant checks, and an exact
  \(2m\) shared-bias stress;
- `resonance-metrology`, a Python API, CLI, JSON/HTML/PNG evidence surface,
  model/data artifacts, validator, and CI smoke;
- `resonance-estimate`, a no-oracle trajectory/model CLI that refuses failed
  or byte-identical copied chart fits, records model digests, and reports
  pairwise chart ambiguity without calibrated claims;
- a public resonance-metrology decision and mathematical convention record;
- contributor guidance for evidence-bearing scientific changes; and
- an observation-only winding-rate initializer that prevents large
  discrete-map rotations from falling into the wrong wrapped-frequency
  optimization basin, with its diagnostics recorded in every run.

### Changed

- demoted `chart-fidelity` from an empirical falsifier to the closed-form
  oracle pipeline regression it actually implements;
- fixed the action-slope sign hole, gated the affine intercept, and clamped
  exported action support to the nonnegative physical range;
- corrected Action-Angle Networks' prior-art description: its G-SympNet
  Cartesian encoder is symplectic; the surviving distinction is canonical
  area action, physical calibration, generated dynamics, and chart-error
  metrology;
- narrowed Duffing discretization and measured-data language to the evidence.

### Result

- the clean full profile accepted all eight learned charts as predictors,
  placed the componentwise-median block at 19.59% complex error while the
  median per-chart error was 20.44%, and returned
  `resolved_refuted (gauge_freedom)` because a prediction-equivalent exact
  gauge moved the block by 44.16%; the failed floor, null-availability, and
  shuffled-angle gates remain visible in the checked manifest.
- post-review hardening makes wrong-harmonic traps per-chart, makes an
  unavailable paired-null floor and an abstained estimator variant explicit,
  validates the committed flagship evidence in CI, and adds the abstention
  and no-formal-guarantee ledgers to the report.

## 3.3.0 — 2026-07-28

This release begins the transition from one integrable chart to a
residual-aware phase-space instrument.

### Added

- independent radial, phase-step, phase-law, and complete-conjugacy diagnostics
  for observed trajectories;
- `canonical-diagnose`, which tests complete new trajectories against a saved
  canonical chart instead of treating an action-range check as sufficient
  support;
- an explicitly documented Fourier convention for truncated residual
  generating functions;
- `chart-fidelity`, a controlled falsification experiment that separates
  physical resonant structure from a deliberately misspecified canonical
  observation chart;
- hard claim boundaries distinguishing a known-gauge oracle experiment from
  the still-unresolved learned-chart identifiability problem.

### Changed

- replaced the proposed one-number chart diagnostic with independent geometric
  and phase-law residuals;
- narrowed the action comparison: exact symplecticity fixes the physical area
  scale, while mean radial action tests circularization quality;
- promoted chart error, rather than ordinary measurement noise, as the first
  research obstacle to resolve.

## 3.2.0 — 2026-07-28

This release turns the Hamilton–Jacobi connection into a runnable,
structure-preserving model rather than a roadmap item.

### Added

- an exact-symplectic neural canonical transformation composed of analytically
  invertible translations, reciprocal scalings, and alternating shears;
- a latent Hamiltonian normal form \(h(I)\), with
  \(I=(Q^2+P^2)/2\), \(\omega(I)=dh/dI\), and analytic rotation instead of
  numerical time integration;
- `canonical-train` / `koopman-hj` and `canonical-predict` commands;
- a canonical-action audit that measures
  \(J=(2\pi)^{-1}\oint p\,dq\), checks \(dH/dJ=\omega\), and distinguishes
  nonlinear invariant gauge from physical action;
- complete-trajectory held-out evaluation of recursive rollout, observed
  action drift, Koopman phase residual, numerical invertibility, symplectic
  defect, and exact model action conservation;
- post-fit checks showing whether the learned \(h(I)\) recovers measured
  frequency and reference-energy shape without using energy in training;
- loadable model and HTML/PNG/JSON evidence surfaces, artifact validators,
  refusal of uncertified or out-of-action-support prediction, and CI smokes;
- a committed 30-trajectory Duffing result plus a source-backed Koopman–HJ
  frontier and research program.

### Changed

- promoted exact canonical mechanics ahead of the unconstrained polynomial
  operator family while retaining the latter as a useful invariant-discovery
  and baseline path;
- reframed empirical closed-orbit integration as a hard evaluation ruler, not
  the product architecture.

## 3.1.0 — 2026-07-28

This release turns the research lab into the first useful local instrument for
nonlinear-mechanics trajectory data.

### Added

- trajectory CSV ingestion with complete-trial identity, strict sampling
  checks, optional post-training reference values, and source fingerprints;
- a dimension-general label-free invariant learner;
- a transparent polynomial observable library and
  invariant-conditioned Koopman operator family;
- complete held-out recursive comparison with global quadratic EDMD and persistence;
- an empirical support certificate that incorporates invariant drift,
  invariant-range coverage, sampled training-state distance, and baseline wins;
- `generate-example`, `analyze`, and `predict` commands;
- a weights-only model bundle with a Python load/rollout API, the fit verdict,
  and default refusal of uncertified fits or unsupported initial states;
- an HTML engineering report, overview figure, manifest validator, and
  committed 30-trajectory Duffing actual run;
- artifact SHA-256 binding and independent held-out metric reconstruction from
  the source CSV plus exported model;
- a source-backed mathematical and product blueprint for invariant-first,
  structure-preserving local Koopman atlases.

### Changed

- broadened `LearnedInvariant` from a fixed three-state input to arbitrary
  measured state dimension while preserving the pendulum default;
- promoted the project from built-in simulations alone to an external-data
  workbench without weakening the v3 scientific boundaries.

## 3.0.0 — 2026-07-28

This release recovers the original project's broader idea and turns it into a
connected nonlinear-dynamics research lab.

### Added

- a label-free scalar-invariant experiment trained without physical energy,
  phase, amplitude, or frequency targets;
- a categorical simplex transfer model with a positive, row-stochastic
  operator, genuine stochastic-process branching evidence, and matched
  no-operator, Ulam, occupancy, and Chapman–Kolmogorov falsifiers;
- a torque-controlled pendulum simulator, a predeclared scalar actuator-gain
  identification model, an exact supplied-equation oracle, and a
  higher-capacity residual ablation for real below-to-above separatrix
  crossings;
- an integrated `lab` command, four experiment-specific commands, an overview
  figure, one machine-readable manifest, and a scientific-coherence validator;
- a research roadmap connecting the working experiments to symplectic atlases,
  residual certification, invariant discovery, stochastic transfer, hybrid
  dynamics, and control.

### Changed

- reframed the project around learning geometry, local laws, and transitions
  rather than one global linearization;
- added stateful hysteresis and a minimum dwell time to autonomous atlas
  routing;
- promoted full route traces, switch locations, rapid reversals, alternations,
  and valid-horizon switching metrics into the atlas evidence;
- made the atlas validators independently reconstruct route truth and reject
  pathological chattering;
- preserved negative results in the promoted evidence: the current stochastic
  transfer profile is falsified by stronger baselines, while the minimal
  controlled model reaches the supplied simulator floor;
- replaced dismissive prototype framing with a project-lineage account that
  preserves the original simplex and latent-operator intuition.

## 2.0.0 — 2026-07-28

This release turns the single-chart separatrix failure into a falsifiable
two-chart experiment.

### Added

- a symplectic hyperbolic chart with a learned local saddle rate;
- explicit autonomous chart transitions through predicted physical state;
- high-energy invariant-shell projection plus projection-only and
  saddle-only ablations;
- denser equal-data training for every learned baseline and held-out
  near-separatrix trajectories;
- a five-seed high-energy-band comparison;
- chart occupancy, transition disagreement, local residual, and operator
  determinant diagnostics;
- `atlas` and `atlas-robustness` commands;
- a validator tying public claims to committed atlas evidence.

### Changed

- promoted the near-separatrix atlas as the primary research result while
  preserving the v1.1 portfolio evidence;
- generalized training and evaluation amplitudes into the experiment config;
- replaced an uninformative neural router with the explicit geometric validity
  rule it had rediscovered.

## 1.1.0 — 2026-07-28

This release turns initialization sensitivity into visible, executable
evidence.

### Added

- a three-seed `robustness` command with per-run and aggregate metrics;
- coordinate pretraining that prevents the conditioned phase encoder from
  collapsing under an unlucky initialization;
- independent deterministic data loaders for order-invariant model
  comparisons;
- a trained-run health check and a second-seed clean-room CI smoke.

### Changed

- tightened fixed-operator and conditioned-model claims to match exactly what
  the experiment establishes;
- promoted three-seed means, variation, and win counts into the README;
- updated the result figure and project-history language to present the work
  confidently without hiding its limits;
- updated GitHub Actions to current Node-based action releases.

## 1.0.0 — 2026-07-28

The portfolio edition turns the original prototype into a complete,
reproducible PyTorch project.

### Added

- installable `src/` package and command-line interface;
- reversible symplectic pendulum simulator;
- circular state representation and complete-trajectory evaluation;
- persistence, linearized-physics, DMD, and residual-MLP baselines;
- orthogonal fixed-operator Koopman autoencoder;
- energy-conditioned action-angle rotation model;
- autonomous rollout, frequency, valid-horizon, and energy-drift metrics;
- deterministic committed benchmark figure and JSON evidence;
- unit tests, linting, wheel build, and GitHub Actions CI;
- explicit architecture and scientific-scope documents.

### Preserved

- the complete 2023 prototype under `legacy/2023-prototype`;
- the original history at Git tag `prototype-2023`.
