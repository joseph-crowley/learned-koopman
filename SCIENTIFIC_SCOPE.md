# Scientific scope and claim boundary

## What is demonstrated

The committed benchmark is an actual run of the repository at seed 7, paired
with a sensitivity sweep at seeds 7, 17, and 29. Together they demonstrate
that:

- a compact PyTorch project can learn and evaluate structured latent dynamics
  end to end;
- the tested eight-dimensional fixed orthogonal operator is a poor global
  approximation for pendulum trajectories spanning different amplitudes;
- coordinate pretraining prevents the conditioned phase encoder from
  collapsing under an unlucky initialization;
- the explicitly energy-conditioned rotation has better mean valid horizon and
  mean angle RMSE than the residual MLP in this three-seed check, while the MLP
  still wins one seed;
- the same single-chart conditioned model degrades sharply near the
  separatrix.

The last point is a result, not an embarrassment. Action-angle coordinates are
singular at the separatrix, so a charted representation or explicit abstention
region is the scientifically appropriate continuation.

## What is not claimed

This project does not claim:

- an exact finite-dimensional Koopman representation of the global pendulum;
- state-of-the-art forecasting performance;
- discovery of the pendulum Hamiltonian from raw data;
- physical symplecticity of an arbitrary learned encoder/decoder;
- calibrated probabilistic uncertainty;
- correctness for rotational, forced, damped, noisy, or controlled regimes;
- publication novelty for encoder–operator–decoder architectures.
- a statistically powered estimate from only three seeds;
- that the conditioned model beats a residual MLP on every initialization.

The energy-conditioned model receives the conserved energy and supervision from
the known elliptic frequency law. It is therefore physics-guided, not a claim
of fully unsupervised discovery.

The evaluation amplitudes are excluded exactly from the training grid, but they
remain interpolation tests except for the 3.05 near-separatrix case, which is
outside the training range.

## Why “Koopman” remains in the name

The project studies the central Koopman modeling instinct: choose observables
whose evolution is simple and linear. The fixed model uses one global linear
operator. The conditioned model uses a fibered family—one linear rotation per
invariant energy shell. The latter is Koopman-inspired but is not one global
finite matrix.

## Evaluation contract

Every promoted learned model is judged by:

- autonomous rollouts from one encoded initial condition;
- circular angle error and velocity error;
- a predeclared valid-prediction threshold;
- energy drift;
- recovered oscillation frequency;
- held-out complete trajectories across amplitude.
- a three-seed sensitivity check for the learned-model comparison.

One-step teacher forcing is not used as the headline metric. The exact simulator
is the reference and is never presented as a learned competitor.

## Natural next experiments

1. Add a near-separatrix hyperbolic chart and inspect transition continuity.
2. Replace supervised energy with a calibrated learned action coordinate.
3. Audit learned observables using held-out generator residuals.
4. Revisit the original Gumbel simplex as a stochastic transfer-operator model,
   with a mass-preserving transition and a VAMP/categorical objective.
