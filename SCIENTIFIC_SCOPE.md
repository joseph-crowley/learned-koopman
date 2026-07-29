# Scientific scope and claim boundary

## What is demonstrated

The repository contains five generations of committed actual-run evidence:

- a broad-libration portfolio benchmark at seed 7, with a sensitivity sweep at
  seeds 7, 17, and 29;
- a near-separatrix atlas benchmark at seed 7, with a five-seed sweep at seeds
  7, 17, 29, 41, and 53;
- a connected v3 research-lab run covering autonomous local charts, label-free
  invariant discovery, stochastic transfer, and controlled crossings;
- a non-pendulum mechanics-workbench run on 30 conservative Duffing
  trajectories at one deterministic seed-7 split, split by complete run.
- an exact-symplectic canonical Koopman–Hamilton–Jacobi model on the same
  Duffing dataset, again split by complete trajectory.

The canonical Koopman–HJ experiment demonstrates that:

- a neural canonical map composed only of translations, reciprocal scalings,
  and alternating shears is analytically invertible and symplectic by
  construction for every set of learned weights;
- its latent Hamiltonian \(h(I)\), with \(I=(Q^2+P^2)/2\), generates an exact
  action-conditioned rotation, so model action conservation is architectural
  rather than a training penalty;
- the model reaches normalized recursive rollout RMSE 0.0270 on eight complete
  held-out trajectories, versus 1.5636 for persistence;
- the held-out observed Koopman phase residual is 0.00024, observed normalized
  action drift is 0.0034, numerical symplectic defect is
  \(2.38\times10^{-7}\), and model-rollout action drift is
  \(3.19\times10^{-5}\);
- independent complete-orbit diagnostics report mean radial coefficient of
  variation 0.0026, phase-step coefficient of variation 0.0060, and normalized
  conjugacy RMSE 0.00031 on the held-out trajectories;
- the learned latent action agrees with independently integrated physical
  action \(J=(2\pi)^{-1}\oint p\,dq\) at affine \(R^2=0.99999994\), slope
  0.99983, and intercept 0.00051;
- without energy labels in training, \(dh/dI\) matches measured orbit
  frequency at 0.93% normalized RMSE and \(h(I)\), after the physically
  irrelevant additive energy offset, matches the reference-energy shape at
  0.18% normalized RMSE;
- the post-fit data themselves satisfy \(dH/dJ=\omega\) at 0.008% normalized
  RMSE, validating the closed-orbit action ruler used to judge the model.

Version 3.3 adds two narrower instruments:

- independent radial, phase-step, learned phase-law, and complete-conjugacy
  residuals, so no single statistic is asked to stand for both chart geometry
  and time parameterization;
- a controlled known-chart experiment in which a residual harmonic is stable
  at exact resonance and strongly contaminated off resonance under the same
  misspecified canonical shear.

The second result verifies the expected first-order cohomological mechanism in
an oracle setting. It does not show that a learned chart's optimizer error is a
bounded canonical gauge transformation. Learned-chart ensembles and
adversarial chart perturbations remain the decisive falsifier.

These are actual-run results for one synthetic one-degree-of-freedom
integrable system and one deterministic split. They establish that the
implementation realizes the intended canonical normal form on this problem.
They do not establish cross-system SOTA, seed robustness, measured-hardware
performance, multi-degree integrability, or publication novelty.

The mechanics workbench demonstrates that:

- a general two-state trajectory CSV can be validated, fingerprinted, split,
  analyzed, reported, exported, loaded, and recursively predicted without
  pendulum-specific code;
- one label-free coordinate has held-out normalized within-trajectory drift
  0.0024 and perfectly ranks the unseen Duffing energy levels;
- a quadratic observable dictionary evolved by the learned
  invariant-conditioned family reaches normalized held-out rollout RMSE 0.076,
  versus 0.424 for the same dictionary under one global quadratic EDMD model
  and 1.564 for persistence;
- the optional energy reference is excluded from invariant and operator
  training and is used only for post-training evaluation;
- the certificate rejects trajectory leakage, collapse, excessive invariant
  drift, unsupported held-out initial states, stale baseline comparisons, and
  silent prediction outside both the fitted invariant range and sampled
  training-state neighborhood.

This is an actual synthetic-mechanics tool path, not yet evidence on measured
hardware or split/seed robustness. The polynomial operator family is not
exactly symplectic and its
local spectra are diagnostics rather than rigorously certified Koopman
eigenvalues.

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

Together these form a polished PyTorch research example. The canonical
Koopman–HJ construction is now one coherent candidate method; its novelty and
usefulness still require matched prior-art baselines, multiple systems,
multiple seeds, noise and sampling studies, and measured data. The invariant
result uses five optimization seeds; the canonical model, stochastic transfer,
and controlled identification results currently use one seed each.

## What is supplied and what is learned

The canonical model receives ordered canonical position-momentum trajectories,
timestamps, and trajectory membership. It supplies the one-degree-of-freedom
symplectic shear architecture, radial latent normal form, polynomial degree for
\(\omega(I)\), and complete-trajectory split. The canonical transformation and
Hamiltonian coefficients are learned. Reference energy and empirical
closed-orbit action are excluded from optimization and used only after fitting.
The system assumes that the supplied columns are truly canonical; it cannot
infer mass scaling or a Legendre transformation from arbitrary position and
velocity columns.

The mechanics workbench receives numerical state trajectories, timestamps, and
trajectory membership. It supplies state normalization, a constant/linear/
quadratic observable dictionary, and a polynomial degree for the operator
family. The neural scalar and operator matrices are learned. An optional
reference column is never passed to either fit. The support verdict is empirical
and dataset-specific; it is not a formal spectral, uncertainty, or safety
guarantee.

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
- global action-angle coordinates through turning-point topology,
  separatrices, rotations, or chaotic regions;
- Liouville integrability or canonical identification for multiple degrees of
  freedom;
- that numerical success on Duffing proves a new theorem or a novel method;
- a viscosity solution of a nonsmooth HJ PDE, HJB optimal-control solution, or
  safety/reachability guarantee;
- globally symplectic physical dynamics from the older learned-decoder models;
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
The canonical model learns
\(\psi_k=(Q-iP)^k/(Q^2+P^2)^{k/2}\), whose multiplier
\(\exp(ik\omega(I)\Delta t)\) is constant on each action fiber.
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

A credible publication result must now test whether the canonical construction
adds value beyond its closest neighbors rather than merely add another demo.
The highest-value routes are:

1. benchmark the canonical model against Action-Angle Networks, GFNN,
   SympNets, HNN, neural ODE, EDMD, and a matched unconstrained invertible
   conjugacy across hardening and softening Duffing, pendulum
   libration/rotation, Toda/FPUT, and measured oscillators;
2. run seed, sample-time, trajectory-length, noise, missing-state, parameter,
   and long-horizon robustness with preregistered promotion thresholds;
3. learn rotational charts and residual-certified validity regions, with exact
   overlap and symplectic-defect tests;
4. extend the radial normal form to multi-action tori with Poisson-commuting
   actions, frequency vectors, resonance detection, and atlas cocycles;
5. use a controlled canonical core in closed-loop swing-up and stabilization against
   EDMDc, direct neural dynamics, energy shaping, and nonlinear MPC;
6. replace supervised coarse states with metastable-state discovery and improve
   Chapman–Kolmogorov consistency across lags and noise regimes;
7. generalize one certified atlas across physical parameters, forcing,
   observation noise, and partial observation.

The broader frontier map, prior art, differentiated contribution, and
experiment sequence are in
[`KOOPMAN_HJ_FRONTIER.md`](KOOPMAN_HJ_FRONTIER.md).
