# Resonance metrology

## The question

A learned canonical chart can make a nonlinear map look nearly integrable,
but a small chart error can also masquerade as a perturbation. Before using a
residual Fourier coefficient to predict an island width, this project asks a
harder question:

> Do equally predictive exact-symplectic charts recover the same resonant
> normal-form block, and can the disagreement be measured before reporting the
> number?

`resonance-metrology` is the first complete experiment aimed at that question.
It is a numerical instrument and a falsifier, not a formal certificate.

## What the tool does

The reference workflow creates a noiseless exact-symplectic twist-kick return
map, hides it behind a nontrivial canonical observation chart, and trains eight
independent exact-symplectic Koopman charts without giving them the oracle
action, angle, or kick.

Map rotations are periodic optimization variables: an arbitrary 1-radian
initializer can converge to the wrong wrapped-frequency basin even when the
data advances by roughly 2 radians per sample. The tool therefore initializes
each rotation polynomial from the circular mean of raw polar phase increments
on each training orbit, regressed against raw radial action. This is a
data-only optimization seed, not a canonical coordinate or a residual
measurement; its coefficients, circular concentration, and orbit-fit error
are recorded in the manifest. The held-out prediction gates remain unchanged.

On held-out trajectories it then:

1. pools the learned \((\hat I,\hat\phi,\Delta\hat I)\) transitions into fixed
   action bins;
2. estimates complex Fourier coefficients on the angles actually occupied by
   the trajectories;
3. measures the rotation profile with a weighted Birkhoff average;
4. fits the residual across a band that crosses the target resonance,
   separating a constant resonant block from the chart's
   \((e^{im\Omega}-1)\) detuning signature;
5. reports coefficient magnitude, aligned complex coefficient, island
   halfwidth, phase/location error, ensemble spread, and an empirical error
   floor;
6. runs oracle, raw-coordinate, null, shuffled-angle, wrong-harmonic,
   kick-size, estimator-variant, and off-band controls; and
7. composes every accepted chart with controlled exact-symplectic \(2m\)
   gauge errors to test whether ensemble agreement merely hides shared bias.

The gauge stress is deliberately one-directional: it can forbid a positive
verdict but cannot rescue a failure.

## Use it

```bash
uv run learned-koopman resonance-metrology \
  --profile full \
  --output results/my-resonance-metrology
```

Open `results/my-resonance-metrology/report.html`. The `manifest.json` records
every predeclared gate and abstention, all per-chart values, controls,
artifact hashes, source revision, and the exact claim boundary.

For custom programs, the array/model API is:

```python
from learned_koopman.canonical_model import load_canonical_model
from learned_koopman.resonance_metrology import estimate_resonant_block

models = [load_canonical_model(path) for path in model_paths]
result = estimate_resonant_block(
    models,
    trajectories,          # shape: (trial, step, 2)
    order=3,
    band=(0.7, 2.6),
    bins=14,
)
```

Without an oracle or external chart bound, the multiplicative error floor is
only a pairwise lower bound. The API records that fact and does not claim
calibrated physical accuracy.

For a measured return-map CSV and an independently trained chart ensemble:

```bash
uv run learned-koopman resonance-estimate measurements.csv \
  --model results/chart-seed-7/model.pt \
  --model results/chart-seed-17/model.pt \
  --position-column position \
  --momentum-column momentum \
  --order 3 \
  --band 0.7 2.6 \
  --output results/my-resonance-estimate.json
```

This path refuses models that failed their own held-out fit gates and requires
at least two charts. Without an oracle chart or external chart-error bound it
reports only a pairwise chart-spread lower bound; it cannot promote calibrated
physical precision.

## Abstention is part of the result

The band must cross \(m\Omega(I)=2\pi k\). Without that crossing, the
constant physical column and the chart-coboundary column become nearly
collinear; a controlled test produced a large false coefficient at such a
harmonic. The tool therefore refuses values for:

- no resonance crossing in the supplied band;
- ill-conditioned harmonic or band fits;
- inadequate angular coverage;
- coefficients below the measured floor;
- chart disagreement beyond the floor; or
- unstable results under predeclared regression variants.

These are scientific outputs, not software errors.

## Coefficient convention

For

\[
G_m(\phi)=K_m\cos(m\phi+\chi_m),
\]

the action kick is

\[
\Delta I=-\partial_\phi G_m
        =mK_m\sin(m\phi+\chi_m).
\]

The fitted action-kick amplitude is therefore \(mK_m\), and the generating
amplitude is \(K_m\). The leading pendulum halfwidth used here is

\[
\Delta I_{\mathrm{half}}
=2\sqrt{\frac{|K_m|}{|\tau|}}
=2\sqrt{\frac{|\Delta I_m|}{m|\tau|}}.
\]

The implementation carries both fields explicitly so the factor of \(m\)
cannot disappear silently.

## What was selected and what was parked

Selected: learned-chart residual metrology from occupied trajectories, with
classical frequency extraction, exact gauge adversaries, independent
controls, and explicit abstention.

Parked until that result is understood:

- certified tori and monodromy;
- Greene-residue thresholds on learned surrogates;
- fusion Poincaré cartography and transport optimization;
- expressive island-capable map surrogates;
- noisy or irregular measured-rig ingestion; and
- multi-action resonances and involution.

Those are high-value directions, but each consumes a chart whose error must
first be priced or refused.

## Research position

The architecture itself is adjacent to established work on neural canonical
transformations, SympNets, generating-function networks, and
action-angle models. The narrower open hypothesis is whether resonant
normal-form information can be recovered with a useful empirical error budget
under learned-chart ambiguity. One synthetic reference result cannot establish
that as a general research contribution; it can establish a reproducible
protocol, a measured positive or negative boundary, and the next decisive
experiments.

The checked full run establishes the negative boundary. Eight of eight charts
passed held-out prediction, and the median trajectory-band coefficient reached
19.59% complex error, but a prediction-equivalent exact \(2m\) gauge moved the
block by as much as 44.16%. The empirical floor covered only 20% of realized
per-chart errors, the null instrument produced only five estimable charts, and
the shuffled-angle level exceeded its gate. The resulting
`resolved_refuted (gauge_freedom)` status is a refutation of the stated
precision on this fixture, not a universal impossibility result.

That result kills the tempting product claim that an ensemble of accurate
canonical predictors automatically yields a physical residual coefficient.
The next high-value experiment must add information that fixes or quotients
the gauge: measured physical phase/action markers, continuation from a known
normal form, or a gauge-invariant island/transport observable.

Relevant starting points include:

- [Action-Angle Networks](https://arxiv.org/abs/2211.15338)
- [Neural Canonical Transformation](https://arxiv.org/abs/1910.00024)
- [SympNets](https://arxiv.org/abs/2001.03750)
- [Generating Function Neural Networks](https://proceedings.mlr.press/v139/chen21r.html)
- [HénonNet](https://arxiv.org/abs/2007.04496)
- [Residual DMD](https://arxiv.org/abs/2205.09779)
- [Weighted Birkhoff / RRE torus computation](https://arxiv.org/abs/2505.08715)
- [ActionFinder](https://arxiv.org/abs/2012.05250)
