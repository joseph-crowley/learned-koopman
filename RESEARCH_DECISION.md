# Research decision: price the chart before selling the coefficient

## Selected

Build a trajectory-sampled resonance-metrology instrument around independently
trained exact-symplectic charts, classical rotation-number estimation,
resonance-crossing support, paired nulls, independent baselines, estimator
variants, and controlled exact canonical gauges.

The reason was practical and scientific: every proposed downstream tool
(island width, normal-form atlas, transport map, or control surface) consumes a
chart. The first useful question is whether that chart permits the residual
quantity to be identified at a declared precision.

## Killed or parked

- **Killed:** treating the closed-form `chart-fidelity` identity as empirical
  learned-chart evidence.
- **Killed:** claiming that predictive agreement among canonical charts fixes
  a physical residual coefficient.
- **Killed:** presenting a synthetic coefficient as a formal certificate,
  general novelty result, or measured-system validation.
- **Parked:** Greene residues, monodromy, transport optimization, fusion
  Poincaré maps, multi-action resonances, noise hardening, and hardware
  ingestion until chart ambiguity is fixed or quotiented.

## Decisive experiment

The full profile froze 48 trajectories x 400 steps, a 36/12 orbit split, four
seeds, two architectures, paired kicked/null systems, a target third harmonic,
14 action bins, and nine predeclared gates.

Promoted run 1 at `4d6cc24` was invalid: all charts fell into the wrong wrapped
rotation basin. That result was preserved. A data-only circular-winding
initializer repaired the instrument without using oracle coordinates or moving
any residual gate.

The provenance-clean promoted run recorded in
`results/resonance-metrology/manifest.json` produced:

- 8/8 prediction-accepted charts;
- 5/8 estimable trajectory-band blocks;
- 19.59% consensus complex error, while the median per-chart complex error
  was 20.44% and 3/5 charts exceeded 20%;
- 25% empirical-floor coverage among four charts with an available paired
  null floor; one chart's floor was explicitly unavailable;
- 5/8 estimable paired nulls;
- 37.23% shuffled-angle level against a 20% limit;
- a 3.42% maximum among evaluable trigger variants, with G9 unresolved
  because the quadratic trigger abstained; and
- a 44.16% complex / 43.29% magnitude block shift under an exact gauge that
  remained inside the frozen prediction envelope.

Verdict: **`resolved_refuted (gauge_freedom)`**. This refutes the declared 20%
residual precision on the synthetic fixture. It does not refute learned
canonical models, the band estimator in oracle coordinates, or recovery under
an externally fixed gauge.

## Strongest next move

Turn the negative boundary into a constructive instrument by adding one
independent physical ruler that a predictive chart cannot freely gauge away.
The leading candidates are:

1. continuation from a known small-amplitude or perturbative normal form;
2. measured phase/action landmarks from forcing, section timing, or work
   integrals; and
3. gauge-invariant island area, separatrix flux, or transport observables
   reported before coordinate-dependent coefficients.

The next decisive experiment should compare those three gauge-fixing routes on
at least two map families and one measured return map, with the current exact
gauge ladder retained unchanged.
