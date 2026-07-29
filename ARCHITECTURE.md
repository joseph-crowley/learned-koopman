# Architecture

## Canonical Koopman–HJ flow

```text
canonical trajectory CSV (q,p)
  └─> complete-run train / held-out split
       └─> exact symplectic encoder F_theta
            ├─> translation
            ├─> reciprocal canonical scaling
            └─> alternating neural q- and p-shears
                 └─> latent canonical state (Q,P)
                      ├─> action I = (Q² + P²) / 2
                      ├─> Hamiltonian h_psi(I)
                      ├─> frequency omega(I) = dh_psi/dI
                      └─> exact radial Hamiltonian rotation
                           └─> analytic F_theta inverse
                                └─> physical prediction (q',p')

held-out physical trajectories
  ├─> recursive rollout + persistence comparison
  ├─> observed action drift
  ├─> fiberwise Koopman phase residual
  ├─> numerical inverse and symplectic checks
  └─> empirical J = (2 pi)^-1 integral p dq
       ├─> canonical gauge check I approximately equals J
       ├─> physical HJ identity dH/dJ = omega
       └─> learned h(I), dh/dI comparison
            └─> certificate + report + loadable model
```

The flagship model is

\[
\Phi_{\Delta t}
=F_\theta^{-1}\circ R_{\Delta t\,h_\psi'(I)}\circ F_\theta,
\qquad
I(Q,P)=\frac{Q^2+P^2}{2}.
\]

`SymplecticMap1D` is a composition of maps with analytic inverses:

\[
q\leftarrow q+f(p),\qquad
p\leftarrow p+g(q),\qquad
(q,p)\leftarrow(e^a q,e^{-a}p).
\]

Each component is canonical for every neural-network weight. The latent
rotation is the exact time-\(\Delta t\) flow of the radial Hamiltonian
\(h_\psi(I)\). Their composition is therefore symplectic by construction,
not because a finite penalty happened to become small. The validator still
computes a numerical Jacobian defect so implementation errors remain visible.

The learned complex phase observable

\[
\psi_k(q,p)=
\left(\frac{Q-iP}{\sqrt{Q^2+P^2}}\right)^k
\]

satisfies the fiberwise Koopman law

\[
\psi_k(x_{n+1})
=e^{ik\omega(I)\Delta t}\psi_k(x_n)
\]

when the canonical normal form fits the observed trajectory. The eigenvalue is
constant on an invariant action shell, not globally constant across a
nonisochronous family.

Training uses only ordered state samples and complete trajectory identity.
Reference energy and the empirical action integral are post-fit tests. The
closed-orbit area is decisive because a generic scalar invariant has arbitrary
monotone gauge, whereas a symplectic transformation preserves phase-space
area. Agreement between latent \(I\) and physical
\((2\pi)^{-1}\oint p\,dq\) tests whether the model learned a genuinely
canonical chart.

The exported model carries its certificate and observed action range.
`canonical-predict` refuses a rejected fit or action extrapolation unless the
caller explicitly overrides the gate.

## Mechanics-workbench flow

```text
trajectory CSV
  └─> complete, uniform TrajectoryDataset
       ├─> complete-run train / held-out split
       ├─> training-only state normalization
       └─> label-free scalar invariant I_theta(x)
            ├─> orbit coordinate c_j = mean_t I_theta(x_j,t)
            └─> transparent observables psi(x)
                 ├─> fibered family K(c) = sum_r c^r K_r
                 ├─> global quadratic EDMD + persistence falsifiers
                 └─> recursive held-out rollouts + local spectra
                      └─> certificate + report + loadable model
```

`TrajectoryDataset` preserves trial identity, timestamps, state-column names,
the original run lengths, and a SHA-256 fingerprint of the source CSV. It
rejects missing values, non-finite states, non-monotone time, irregular
sampling, and inconsistent sampling intervals rather than silently cleaning
experimental data.

The split happens by complete trajectory ID. State normalization is fit on the
training runs only. If a reference column such as known energy is supplied, it
is retained solely for post-training evaluation.

`LearnedInvariant` accepts an arbitrary state dimension in the workbench while
retaining the original three-input default. Its scalar trajectory means index
a polynomial field of finite Koopman regressions:

\[
\psi(x_{k+1}) \approx \psi(x_k)
\left(K_0 + \hat c K_1 + \hat c^2 K_2\right).
\]

The observables are explicit constant, state, and optional quadratic terms.
Physical state is always directly decodable from the linear entries. The
workbench compares recursive rollouts with the same observable dictionary fit
as one global quadratic EDMD model and with persistence.

The current certificate is empirical. It requires a noncollapsed,
approximately trajectory-constant coordinate; every held-out initial state
inside the fitted invariant range and near sampled training states; and rollout
wins over both baselines. The loadable bundle also carries the fit certificate
and refuses negative fits or unsupported initial states unless the caller
explicitly overrides it.

This polynomial family is not exactly symplectic and does not claim a rigorous
spectral certificate. It remains the more general path when supplied states
are not known canonical coordinates. The canonical Koopman–HJ model above is
the exact-symplectic one-degree-of-freedom path; residual-calibrated chart
gluing and higher-dimensional tori remain future layers. See
`KOOPMAN_HJ_FRONTIER.md` and `PHYSICS_WORKBENCH.md`.

## Research-lab flow

```text
autonomous trajectories ──┬─> fixed / conditioned / atlas predictors
                          │      └─> free rollout + route truth
                          └─> grouped states
                                 └─> learned invariant + anti-collapse

damped stochastic trajectories ─> simplex memberships
                                  └─> row-stochastic transfer + CK tests

bounded torque sequences ─> controlled kick-drift-kick trajectories
                           └─> actuator-gain identification + residual ablation

all four result dictionaries ─> research-lab validator
                              └─> manifest + overview figure
```

All cells use the circular state \((\sin\theta,\cos\theta,\dot\theta)\), but
they do not pretend to share one objective. Each has its own data split,
mathematical constraint, baseline, and falsifier.

## Fixed operator

`FixedKoopmanAE` has a nonlinear encoder and decoder around an exactly
orthogonal latent step. The operator is constructed as a matrix exponential of
a skew-symmetric generator, avoiding spectral-radius drift by construction.

Its loss combines:

- physical-state reconstruction;
- decoded multistep prediction;
- encoded-next-state latent consistency.

It is the fair version of the original repository's stated model.

## Energy-conditioned operator

`EnergyConditionedRotation` encodes state to a two-dimensional unit phase,
learns \(\omega(H)\), rotates phase exactly, and decodes phase plus energy.

Training uses a short coordinate curriculum before the joint physical loss.
The first stage anchors the phase encoder and frequency network to the exact
libration phase; the second fits reconstruction, rollout, latent consistency,
and energy losses. This prevents a measured phase-collapse failure without
changing the model or selecting a favorable seed.

The frequency target comes from the arithmetic-geometric mean expression for
the complete elliptic integral. That choice makes the learned coordinate
consistent across amplitudes and turns the frequency curve into an inspectable
output.

The decoder is still learned, so physical energy preservation is measured and
penalized rather than assumed.

## Near-separatrix atlas

`SeparatrixAtlas` freezes a trained `EnergyConditionedRotation` as its regular
chart and learns one scalar rate for a local hyperbolic chart around the
unstable upright equilibrium. In local canonical coordinates \(q,p\), its
operator is

\[
\exp\left(
\Delta t
\begin{bmatrix}
0 & 1\\
\lambda^2 & 0
\end{bmatrix}
\right).
\]

The determinant is one for every learned \(\lambda\), so the local update is
symplectic in \(q,p\) by construction. That statement does not extend through
the arbitrary neural decoder to a claim of globally symplectic physical
dynamics.

Routing is an explicit validity rule:

```text
regular chart: all ordinary energies and states away from the upright saddle
saddle entry:  H > 0.8 and |q| < 1.4
saddle exit:   H <= 0.8 or |q| >= 1.5, after a 12-step minimum dwell
```

When a predicted trajectory changes chart, the model converts through its own
predicted physical state and initializes the destination chart there. No
reference state is consulted during rollout. High-energy predictions are
projected onto the known initial energy shell; projection is separately tested
on the single chart, where it does not fix the coordinate failure.

The promoted evidence includes two destructive ablations:

- the saddle chart alone, which initially tracks the slow departure but
  accumulates large energy drift away from its local validity region;
- energy projection on the single chart, which preserves the invariant but
  does not repair its near-separatrix phase representation.

The first implementation used a neural categorical router. A critic run showed
that it exactly reproduced the predeclared validity region without changing
decisions, so the final model removes those parameters and exposes the
geometric rule directly.

The evaluator stores the complete categorical route trace. Validators
independently reconstruct switch locations, alternations, rapid reversals, and
switch density, both globally and within the valid-prediction prefix. This
turned a previously hidden two-seed chatter failure into an executable
contract.

## Label-free invariant

`LearnedInvariant` is a small scalar encoder over the circular state. Training
uses:

- variance within each observed trajectory, minimized for constancy;
- variance across trajectory means, fixed away from zero to prevent collapse;
- centering to fix a gauge;
- smoothness over a neighbor graph built from symmetric trajectory-set
  distances.

The model never receives an energy, amplitude, phase, frequency, trajectory
time, or ordering label. Training segments begin at staggered phases, and
evaluation uses complete interleaved shells. Exact physical energy is imported
only in the post-training evaluator.

## Simplex transfer operator

`SimplexTransferOperator` maps physical state to soft categorical membership.
Both memberships and rows of the transition matrix are softmax-normalized:

```text
state -> encoder logits -> simplex membership
membership @ K -> future membership, with K positive and row stochastic
future membership @ physical prototypes -> diagnostic state expectation
```

The simulator is a damped pendulum with Euler–Maruyama velocity noise.
Train/validation splitting happens by complete stochastic path. The objective
combines coarse-state classification with one- and two-lag categorical
likelihoods. Independent stochastic branches from identical anchor states
verify genuine process uncertainty.

Evaluation always includes membership with no propagation, empirical Ulam,
occupancy, direct two-lag, and branching-horizon counterfactuals. The current
operator passes its simplex constraints but is mechanically labeled
`falsified_by_current_profile`; valid stochastic structure did not make \(K\)
useful on every required comparison.

## Controlled crossing

The controlled simulator uses a kick-drift-kick update for
\(\ddot\theta=-\sin\theta+u\) with piecewise-constant bounded torque. An
external-work calculation checks that energy change is attributable to the
input.

`GainOnlyControlledPendulum` supplies the conservative \(-\sin\theta\) term and
identifies the one unknown actuator gain. `ExactUnitGainOracle` exposes the
supplied simulator equation and numerical floor. The higher-capacity
`ActionConditionedPendulum` adds a bounded neural residual but is retained as a
worse ablation rather than promoted.

Short windows train the identified model; every reported long-horizon result
recursively feeds back its own prediction while receiving only the known
control sequence. Comparisons include the oracle, neural residual, controlled
small-angle physics, and the same identified model with its action channel
zeroed.

## Integrated manifest

`run_research_lab` runs the four cells, validates their scientific contracts,
and writes:

- generated component outputs, with learned weights where available;
- one self-contained `manifest.json`;
- one four-panel `overview.png`.

The validator checks mathematical constraints and anti-cheating boundaries,
not a single composite score. A transfer experiment can therefore pass its
probability contract while visibly losing its CK comparison; the failed edge
is evidence, not a reason to erase the run.

## Design choices

- Vanilla PyTorch keeps every model and objective visible.
- `uv.lock` fixes the executable environment.
- Generated datasets stay in memory; no large binary files are required.
- JSON results are the source for README claims.
- Independent seeded data loaders keep one model's training schedule from
  changing another model's minibatch order.
- The robustness command records every seed plus mean, standard deviation,
  minimum, and maximum for each core metric.
- The atlas robustness command first averages the predeclared high-energy
  amplitudes within each seed, then compares those independent seeded runs.
- Held-out local-chart residuals, the full route trace, probability constraints,
  process-noise branches, crossing attribution, and component baselines keep
  every mechanism inspectable.
- Each research cell remains independently callable; the integrated command is
  orchestration, not a monolithic model.
- The quick demo is CI-sized; the committed lab, portfolio, and atlas evidence
  comes from the full profiles.
