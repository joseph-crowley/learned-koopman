# Changelog

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
