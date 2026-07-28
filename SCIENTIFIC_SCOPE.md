# Scientific scope and claim boundary

## What is demonstrated

The repository contains two committed actual-run studies:

- the broad-libration portfolio benchmark at seed 7, with a sensitivity sweep
  at seeds 7, 17, and 29;
- the near-separatrix atlas benchmark at seed 7, with a five-seed sweep at
  seeds 7, 17, 29, 41, and 53.

Together they demonstrate that:

- a compact PyTorch project can learn and evaluate structured latent dynamics
  end to end;
- the tested eight-dimensional fixed orthogonal operator is a poor global
  approximation for pendulum trajectories spanning different amplitudes;
- coordinate pretraining prevents the conditioned phase encoder from
  collapsing under an unlucky initialization;
- the energy-conditioned rotation improves the three-seed broad-libration
  averages over the residual MLP, while the MLP still wins one seed;
- the same single conditioned chart degrades sharply near the separatrix, even
  when every learned baseline receives denser high-energy training;
- a second, locally hyperbolic symplectic chart plus explicit chart transitions
  raises mean valid horizon over held-out amplitudes 2.95, 3.05, and 3.10 from
  0.36 ± 0.12 to 3.94 ± 0.06;
- the atlas beats the residual MLP's per-seed band average in four of five runs,
  while the MLP remains better in one run;
- high-energy shell projection alone does not repair the conditioned chart,
  and the hyperbolic chart alone develops large physical energy drift;
- the atlas retains the single conditioned model exactly at ordinary energies
  because the second chart and projection are inactive there.

This is evidence for a charted representation near a coordinate singularity,
not proof that the separatrix itself has been globally linearized.

## What is not claimed

This project does not claim:

- an exact finite-dimensional Koopman representation of the global pendulum;
- state-of-the-art forecasting performance;
- discovery of the pendulum Hamiltonian from raw data;
- physical symplecticity of an arbitrary learned encoder/decoder;
- calibrated probabilistic uncertainty;
- correctness for rotational, forced, damped, noisy, or controlled regimes;
- publication novelty for encoder–operator–decoder architectures;
- a statistically powered population estimate from five seeds;
- that either structured model beats a residual MLP at every amplitude or seed;
- that physical symplecticity follows from a symplectic operator inside one
  learned chart;
- discovery of chart boundaries from data.

The energy-conditioned model receives the conserved energy and supervision from
the known elliptic frequency law. It is therefore physics-guided, not a claim
of fully unsupervised discovery.

The atlas also receives that conserved initial energy. Its chart index is the
predeclared rule \(H>0.8\) and \(|q|<1.4\), where \(q\) is displacement from the
upright saddle. The local hyperbolic rate is learned, while its symplectic
operator form and high-energy energy-shell projection are supplied physics.
The result tests whether a correct local chart repairs the single-chart
failure; it does not test unsupervised chart discovery.

All evaluation amplitudes are excluded exactly from the corresponding training
grid. In the v1 portfolio experiment, 3.05 is outside the training range. In
the v2 atlas experiment, 2.95, 3.05, and 3.10 are held-out interpolation points
inside a denser training range ending at 3.12. Every v2 trajectory remains a
libration below the separatrix.

## Why “Koopman” remains in the name

The project studies the central Koopman modeling instinct: choose observables
whose evolution is simple and linear. The fixed model uses one global linear
operator. The conditioned model uses a fibered family—one linear rotation per
invariant energy shell. The atlas adds a local hyperbolic operator and explicit
transition through predicted physical state. The latter two are
Koopman-inspired structured operator families, not one global finite matrix.

## Evaluation contract

Every promoted learned model is judged by:

- autonomous rollouts from its own predicted state or carried latent state;
- circular angle error and velocity error;
- a predeclared valid-prediction threshold;
- energy drift;
- recovered oscillation frequency;
- held-out complete trajectories across amplitude;
- multi-seed sensitivity with every seed retained.

One-step teacher forcing is not used as the headline metric. The exact simulator
is the reference and is never presented as a learned competitor.

The atlas additionally reports:

- fraction of autonomous rollout steps in each chart;
- number of chart transitions;
- disagreement between local next-state predictions at a transition;
- held-out local-chart residuals computed on reference transitions;
- determinant of the local saddle operator;
- ablations for energy projection alone and the saddle chart alone.

## Natural next experiments

1. Extend the atlas to clockwise and counter-clockwise rotational charts and
   test actual cross-regime topology.
2. Replace supplied energy and frequency supervision with a calibrated learned
   action coordinate.
3. Add generator-level or ResDMD-style spectral certification beyond the
   current held-out local-transition residual.
4. Revisit the original Gumbel simplex as a stochastic transfer-operator model,
   with a mass-preserving transition and a VAMP/categorical objective.
