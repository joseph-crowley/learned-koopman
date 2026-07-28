# Research roadmap

## The project-scale idea

A nonlinear system rarely becomes globally simple in one finite latent space.
But different scientific questions can expose different simple learned
objects:

- **local flow charts** for autonomous prediction near a coordinate
  singularity;
- **invariants** for discovering the foliation on which trajectories move;
- **transfer operators** for probability flow under unresolved or stochastic
  dynamics;
- **controlled flow maps** for predicting how interventions move the system
  between regimes.

This repository treats those as connected experiments, not interchangeable
uses of the word “Koopman.” Each object has a different mathematical contract
and should fail a different falsification test.

| Question | Learned object | Required structure | Primary falsifier |
|---|---|---|---|
| Can autonomous motion remain simple near a singular transition? | Local operator atlas | autonomous rollout, explicit routing, transition diagnostics | route chatter or overlap disagreement |
| Can a conserved coordinate be recovered without its labels? | Scalar invariant | trajectory constancy without collapse | constant output or poor shell ordering |
| How does probability move under weak stochasticity? | Simplex transfer operator | positivity and mass preservation | Chapman–Kolmogorov failure |
| What changes when the system is acted on? | Action-conditioned flow map | explicit bounded control and real crossing data | no advantage over an action-blind model |

The pendulum is deliberately small enough that every learned object can be
compared with known physics. It is a microscope for representation choices,
not a claim that pendulum forecasting itself is an unsolved application.

## What is useful now

### 1. A falsifiable separatrix atlas

The current atlas asks whether a locally hyperbolic chart can repair the failure
of one energy-conditioned phase chart near the upright saddle. The strongest
next step is not a larger network. It is a more complete transition contract:
overlap consistency, physical symplectic-defect measurement, residual-certified
routing, and an explicit abstention region.

### 2. Invariant-first discovery

The supplied Hamiltonian is an excellent control variable for the atlas, but it
also makes a clean discovery claim impossible. A separate experiment therefore
learns a trajectory-constant scalar without energy labels and uses the exact
Hamiltonian only as held-out evaluation.

The research continuation is to learn an invariant foliation, then a phase
coordinate inside each shell, and finally detect where the period diverges.
That would allow the separatrix to emerge as a singularity of the learned
coordinates rather than as a supplied threshold.

### 3. Stochastic transfer rather than a mismatched “VAE”

The original prototype's simplex intuition is worth preserving when the
scientific object is probability flow. A defensible version uses soft
memberships and a positive row-stochastic transition matrix. It is evaluated
through predictive probability, stationary behavior, and
Chapman–Kolmogorov consistency—not a Gaussian KL term attached to a categorical
sample.

The working cell proves the probability constraints but falsifies its learned
propagation on the current profile: no propagation wins at one lag, and Ulam
wins on branching and Chapman–Kolmogorov checks. The research continuation is
therefore not just a reversible or detailed-balance-aware operator; it must beat
no-\(K\) and train-only Ulam counterfactuals across seeds, lags, and noise
regimes while resolving rare transitions.

### 4. Controlled crossing

An autonomous conservative pendulum does not cross its separatrix. A torque
input makes crossing a real dynamical event and tests whether a learned
representation remains useful under intervention. This is the shortest path
from an explanatory latent model to a planning-facing system.

The current cell cleanly identifies one scalar actuator gain and shows that a
higher-capacity residual is unnecessary for the known simulator. The research
continuation is a four-chart controlled atlas—libration, saddle, clockwise
rotation, and counter-clockwise rotation—evaluated in model predictive control
against energy shaping, nonlinear dynamics, and generic learned baselines.

## The high-value research frontiers

### Learned symplectic atlases

Learn exact chart-local symplectic maps and canonical transition maps. Test
overlap invertibility, preservation of the symplectic form, and cocycle
consistency around chart loops. Relevant foundations include
[generating-function neural networks](https://proceedings.mlr.press/v139/chen21r.html)
and
[CANDyMan's learned manifold charts](https://www.nature.com/articles/s42256-022-00575-4).

The potential research object is not one preferred latent basis. It is the
atlas plus its gluing laws.

### Residual-certified adaptive charts

Use held-out generator or Koopman residuals and disagreement in chart overlaps
as local defect estimators. A model should route only where an expert is
validated, abstain outside that region, and create a new chart when the
certificate fails. This connects the project to
[ResDMD](https://doi.org/10.1017/jfm.2022.1052),
[ResKoopNet](https://proceedings.mlr.press/v267/xu25y.html), and recent work on
the conditions and limits of certified data-driven dynamics
[by Colbrook, Mezić, and Stepanenko](https://www.nature.com/articles/s41467-026-74220-8).

### Action-angle and invariant discovery

Jointly discover conserved quantities, symmetries, and canonical phase
coordinates without leaking the analytic answer. Important adjacent work
includes [AI Poincaré](https://doi.org/10.1103/PhysRevLett.126.180604),
[geometric conservation-law discovery](https://www.nature.com/articles/s41467-023-40325-7),
and
[Action-Angle Networks](https://arxiv.org/abs/2211.15338).

The sharp experiment is whether a learned invariant-first atlas discovers the
frequency collapse and topology change near the separatrix.

### Hybrid and no-chatter dynamics

Treat charts as a hybrid system with interpretable modes, guard regions,
overlaps, and transitions. Require hysteresis or a no-Zeno argument rather than
accepting rapid expert switching as an implementation detail. Adjacent work
includes
[recurrent switching linear dynamical systems](https://proceedings.mlr.press/v54/linderman17a.html)
and
[neural hybrid automata](https://arxiv.org/abs/2106.04165).

### Controlled operator learning

Prediction quality does not imply controllability or planning value. Learn
action-conditioned local generators or bilinear operators, measure
controllability, and close the loop on swing-up and stabilization. Useful
baselines come from
[Koopman model predictive control](https://doi.org/10.1016/j.automatica.2018.05.033)
and
[bilinear generator learning](https://doi.org/10.1137/22M1523601).

### Weak forcing, manifold splitting, and chaos

Use the autonomous atlas as an unperturbed backbone, then add weak periodic
forcing. Learn stable and unstable manifolds, predict their splitting, and
test long-time phase-space statistics. A neural Melnikov residual would be a
particularly natural bridge between classical perturbation theory and learned
local dynamics.

### Parameterized operator families

Move from one pendulum to a family over gravity, length, damping, forcing,
control authority, sampling interval, and observation noise. Require evolution
to commute with chart changes. This turns a fitted example into a small
physics-operator system.

### Symbolic local equations and gluing laws

Do not force one global symbolic ODE onto incompatible coordinates. Discover
sparse local normal forms, guard conditions, and transition maps. This joins
the atlas idea to [SINDy](https://doi.org/10.1073/pnas.1517384113) and
coordinate-plus-equation discovery
[by Champion et al.](https://doi.org/10.1073/pnas.1906995116).

## A defensible novelty ladder

### Presentable PyTorch project

The bar is executable experiments, visible mathematical constraints, meaningful
baselines, autonomous or intervention-aware evaluation, deterministic seeds,
and machine-readable evidence. This is the bar the repository is designed to
meet now.

### Workshop-quality research result

The next credible claim would require at least one new method—such as
residual-certified routing, exact symplectic gluing, or invariant-first chart
discovery—tested across multiple systems and against the strongest directly
adjacent baselines.

### Strong publication claim

A strong claim needs a method that remains useful beyond the pendulum, ablation
evidence identifying its mechanism, calibrated uncertainty or a posteriori
validity, and a clear prior-art distinction from learned Koopman coordinates,
multi-chart autoencoders, Hamiltonian networks, and hybrid mixtures of experts.

The most promising thesis is:

> Learn simple local dynamics, the geometry that glues them, and a certificate
> of where each local description remains valid.
