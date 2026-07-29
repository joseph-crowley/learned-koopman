# Gauge-invariant island-area audit

The resonance-metrology result found the boundary a scientific coordinate
system must respect: an exact canonical gauge changed the recovered complex
normal-form block by 44.16% while leaving one-step prediction inside the same
acceptance envelope.

`island-area-audit` asks the constructive follow-up:

> Can the same learned charts recover a physical phase-space quantity after
> quotienting out the gauge direction that moved the coefficient?

On the frozen synthetic fixture, the answer is yes for total
bounded-libration area.

## Observable

For target resonance order \(m\), define the slow phase

\[
\psi_n=m\phi_n .
\]

A mesh cell is classified as resonantly trapped when the unwrapped slow phase
remains bounded within one full turn over the probe:

\[
\max_n\widetilde\psi_n-\min_n\widetilde\psi_n < 2\pi .
\]

No phase origin is fitted. A fixed offset, a smooth circle
reparameterization, and the exact generating-function gauges used by
`resonance-metrology` preserve bounded versus circulating winding
topologically. On a finite trajectory the classifier also requires the
coordinate distortion to remain below the span margin; the audit measures
that margin rather than assuming it.

The total island area is a quadrature over the classified initial-condition
cells,

\[
\widehat{\mathcal A}=\sum_c {\bf 1}_{\mathrm{bounded}}(c)\,|c|.
\]

Each \(|c|\) is computed from the physical or learned canonical cell vertices
with a quadrilateral shoelace rule. It is not inferred from the recovered
Fourier amplitude.

For the leading pendulum normal form, the action half-width is

\[
\Delta I=2\sqrt{\frac{\epsilon/m}{|\tau|}}.
\]

One of the \(m\) islands has leading area \(8\Delta I/m\), so the total
leading area is \(8\Delta I\). This asymptotic value is an independent
classical baseline; the direct discrete-map mesh is the numerical reference.

## Run it

The committed S1 models are the same eight independent learned charts that
produced the gauge refutation:

```bash
uv sync --extra dev

uv run learned-koopman island-area-audit \
  --resonance-manifest results/resonance-metrology/manifest.json \
  --output results/my-island-area

uv run python scripts/check_island_area.py \
  results/my-island-area/manifest.json
```

The command regenerates the kicked and no-kick probes. It verifies the frozen
manifest and every model digest before analysis.

## Frozen result

The reference mesh uses 61 action cells, 180 angle cells, and 800 map steps.

| Measurement | Result |
|---|---:|
| Leading total island area | 1.460593 |
| Direct physical/oracle mesh | 1.454970 |
| Raw observed-polar baseline | 1.382882 |
| Learned-chart median | 1.453772 |
| Learned median error vs. direct | 0.0823% |
| Worst learned-chart error vs. direct | 0.0825% |
| Minimum membership Jaccard vs. direct | 0.99917 |
| Learned-chart relative range | 0.00028% |
| Worst exact-gauge area shift | 0.00440% |
| Gauge-induced membership flips | 0 / 64 chart-gauge rows |
| Largest null area / kicked area | 4.96% |
| Noncanonical 1.2× scale shift | 20.0% |
| Direct-only 123×360 refinement | 1.460938 |

The null map contains one zero-winding resonant torus of zero continuum area.
On a finite mesh it occupies one action strip; that strip is reported as a
resolution floor rather than relabeled as an island.

The noncanonical plumbing check multiplies both phase-space coordinates by
\(\sqrt{1.2}\). Shoelace area must therefore change by exactly 20%. This
catches normalization errors, but it is an identity rather than an empirical
learned-chart stress test.

The refined direct mesh is within 0.024% of the leading pendulum value, while
the frozen 61×180 reference mesh is about 0.4% low. The learned 0.0823% error
is therefore agreement with the direct classifier on the same mesh, where
boundary-discretization error cancels—not a claim of absolute continuum
accuracy.

## Why this is different from the failed coefficient

The complex residual block is a coordinate representation. Under a canonical
change of chart it can mix with chart harmonics, bin membership, alignment,
and the fitted rotation law. The earlier audit measured that operational
response at 44.16%.

Area is a symplectic quantity. The learned charts and adversarial gauge ladder
are canonical, so a properly evaluated physical cell area should survive.
The experiment still has content: independently learned charts can
misclassify winding topology or distort numerical quadrature even when every
layer is formally symplectic. The direct membership comparison, raw baseline,
null, and chart ensemble test those failure modes. The noncanonical scale
checks only the area-scaling plumbing.

## Claim boundary

The checked result is one noiseless exact-symplectic map, one observation
chart, one architecture family, and one dense synthetic initial-condition
mesh. The dense probe states were not used to train the charts, but the mesh,
controls, and thresholds were developed while this fixture was available.
The result is therefore a retrospective method-development result, not a
blinded prospective confirmation. It supports:

> On this fixture, bounded-libration area is a stable invariant quotient of
> the learned canonical charts, even though their recovered resonant
> coefficient is not identifiable at the declared precision.

It does not establish robustness to sensor noise, sparse or irregular
sampling, partial observation, measured machinery, arbitrary islands,
separatrix flux, calibrated uncertainty, or a formal invariance theorem.
Exact symplecticity makes area preservation structural; the empirical content
is that the learned charts also recover the correct winding membership and
numerical quadrature. The raw-polar comparison shows value under this
observation distortion, not universal need for machine learning. Its exact
4.95% value is span-threshold-sensitive because raw bounded and circulating
populations meet near \(2\pi\). It is a weak observation-coordinate baseline,
not a threshold-free classical comparator; frequency-map analysis is required
in the prospective transfer experiment.

The next serious experiment is the same paired coefficient/area protocol on a
second map family and one measured Poincaré return map, with frequency-map and
direct contour/area baselines frozen before seeing the learned result.
