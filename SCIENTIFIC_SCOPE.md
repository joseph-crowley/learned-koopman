# Scientific scope and claim boundary

## What is demonstrated

The repository contains three generations of committed actual-run evidence:

- a broad-libration portfolio benchmark at seed 7, with a sensitivity sweep at
  seeds 7, 17, and 29;
- a near-separatrix atlas benchmark at seed 7, with a five-seed sweep at seeds
  7, 17, 29, 41, and 53;
- a connected v3 research-lab run covering autonomous local charts, label-free
  invariant discovery, stochastic transfer, and controlled crossings.

The deterministic studies demonstrate that:

- the tested eight-dimensional fixed orthogonal operator is a poor global
  approximation for pendulum trajectories spanning different amplitudes;
- an energy-conditioned rotation improves the three-seed broad-libration
  averages over a residual MLP, while the MLP still wins one seed;
- the same single phase chart degrades sharply near the separatrix;
- a locally hyperbolic chart plus explicit transitions raises the five-seed
  high-energy-band valid horizon from 0.36 ± 0.12 to 3.98 ± 0.06;
- the atlas beats the residual MLP's per-seed band average in four of five
  runs, while the MLP remains better in one run;
- energy projection alone does not repair the conditioned chart, and the
  hyperbolic chart alone develops large physical energy drift;
- exit hysteresis and a minimum dwell remove the severe boundary chatter found
  in two earlier seeds without consulting a reference trajectory.

The new research cells demonstrate that:

- a scalar network trained only from state trajectories and trajectory
  membership can learn a noncollapsed quotient coordinate whose held-out
  values are almost perfectly monotone with physical energy;
- soft categorical memberships and a positive row-stochastic matrix can form a
  mathematically coherent transfer model under genuine stochastic process
  noise;
- that learned transition is falsified on the current profile: it loses to no
  propagation at one lag, to Ulam and occupancy at the branching horizon, and
  to empirical Ulam on Chapman–Kolmogorov consistency;
- a grey-box PyTorch model can identify the single unknown actuator gain from
  0.35 to 1.000 and recursively match a supplied exact oracle through real
  \(H<1\) to \(H\ge1\) crossing events;
- replaying those controlled initial conditions without torque produces no
  crossings, and the simulator's energy change agrees with external work.

Together these form a polished PyTorch research example. They do not yet form
one new research method. The invariant result uses five optimization seeds;
the stochastic transfer and controlled identification results currently use
one seed each.

## What is supplied and what is learned

The energy-conditioned model receives conserved energy, exact phase targets,
and the known elliptic frequency law. It is physics-guided rather than an
unsupervised discovery model.

The atlas receives conserved initial energy. Its entry rule is \(H>0.8\) and
\(|q|<1.4\); it exits at \(|q|\ge1.5\) and enforces a 12-step minimum dwell.
The local hyperbolic rate is learned, while the canonical coordinates,
symplectic operator form, guard geometry, and high-energy shell projection are
supplied physics. The current evidence does not isolate the learned rate as
necessary or optimal, so the gain is attributed to the supplied atlas
structure as a whole.

The invariant learner receives only circular state trajectories grouped by
orbit. It does not receive energy, amplitude values, shell ordering, phase, or
frequency during optimization. Exact energy appears only after training for
evaluation. Its current result covers noiseless librations; it has not yet
proved robust recovery under noise, partial observation, or rotation.

The stochastic transfer experiment uses a supplied damped stochastic pendulum
and train-only k-means coarse states. The neural model learns soft memberships
and a categorical transition. This is a finite-state transfer experiment, not
a VAE and not unsupervised metastable-state discovery. Its constraints pass,
but its measured `operator_verdict` is `falsified_by_current_profile`.

The promoted controlled model receives the exact conservative force as a
grey-box prior and identifies one scalar control gain. A supplied unit-gain
oracle defines the numerical floor, and a neural residual is retained as a
worse ablation. Evaluation uses known future controls without future true
states. This is system identification, not a closed-loop policy or a
model-predictive controller.

## What is not claimed

This project does not claim:

- an exact finite-dimensional Koopman representation of the global pendulum;
- state-of-the-art forecasting or control performance;
- a new theorem, a convergent Koopman spectral method, or publication novelty
  for encoder–operator–decoder models;
- globally symplectic physical dynamics from a symplectic latent operator;
- unsupervised discovery of chart boundaries or canonical transformations;
- calibrated rare-event probabilities or a converged stochastic transfer
  discretization;
- a closed-loop swing-up or stabilization result;
- correctness on experimental hardware or measured real-system data;
- statistically powered population conclusions from five optimization seeds;
- that one structured model beats every baseline at every amplitude, seed,
  horizon, or scientific metric.

All deterministic evaluation amplitudes are absent from the corresponding
training grid. The atlas trajectories remain librations below the separatrix.
The controlled experiment crosses the energy threshold because applied torque
does work; it does not yet add learned rotational charts.

## Why “Koopman” remains in the name

The project studies the central Koopman modeling instinct: find observables
whose evolution is simple. The fixed model uses one global linear operator.
The conditioned model uses a fibered family—one rotation per invariant shell.
The atlas uses local structured operators and explicit switching. The
stochastic cell learns a finite transfer operator over categorical
memberships. These are related operator-learning experiments, not a claim that
all four models instantiate one finite Koopman matrix.

## Evaluation contract

The autonomous predictors are judged by free rollout from their own predicted
state, circular state error, valid horizon, energy drift, held-out complete
trajectories, and seed sensitivity. The atlas additionally reports:

- the complete autonomous route trace and exact switch locations;
- switches, alternations, rapid reversals, and maximum switch density;
- the same switching metrics inside the valid-prediction horizon;
- chart occupancy, local disagreement, held-out local residuals, and the
  determinant of the saddle operator;
- projection-only and saddle-only ablations.

Invariant discovery reports within-trajectory drift, between-shell signal,
noncollapse, monotone shell ordering, and post-hoc affine alignment. Transfer
learning reports probability constraints, held-out categorical likelihood,
no-operator and Ulam counterfactuals, Chapman–Kolmogorov error, stationary and
spectral diagnostics, effective state count, independent physical-noise
branching, and a mechanically derived verdict. Controlled prediction reports
one-step and recursive error, crossing-window error, event precision/recall and
timing, external work, an exact oracle, a gain-only learned model, a neural
residual, and direct action-aware/action-blind ablations.

The integrated validator rejects non-finite results, hidden route chatter,
invariant collapse, broken probability constraints, missing stochastic
branching, stale transfer verdicts, false autonomous crossing attribution,
failed actuator identification, and controlled comparisons that omit the
supplied oracle or direct action-blind ablation.

## Research threshold from here

A credible next method should connect at least two cells rather than merely add
another demo. The highest-value routes are:

1. learn the invariant first, then use it in the conditioned model and measure
   how much of the oracle-energy performance survives;
2. learn rotational charts and residual-certified validity regions, with exact
   overlap and symplectic-defect tests;
3. use the controlled atlas in closed-loop swing-up and stabilization against
   EDMDc, direct neural dynamics, energy shaping, and nonlinear MPC;
4. replace supervised coarse states with metastable-state discovery and improve
   Chapman–Kolmogorov consistency across lags and noise regimes;
5. generalize one certified atlas across physical parameters, forcing,
   observation noise, and partial observation.

The broader frontier map and prior art are in
[`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md).
