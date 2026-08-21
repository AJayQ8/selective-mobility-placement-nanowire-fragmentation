# Frozen pre-fragmentation synthesis contract

Decision date: 2026-08-20

Authorizing instruction: Abdul's instruction to perform the final high-value,
zero-simulation precursor improvement before the CMS rewrite

Paper branch: `paper/selective-mobility-scripta`

Paper PR: <https://github.com/aalzanki/aj-physics/pull/22>

## Scientific question

Does an early, pre-fragmentation change in the untreated junction's natural
failure zone track the nonmonotone source-A lifetime response, and does the
same signed morphology recur in the two confirmation sources for the
like-for-like intermediate and outboard placements?

## Scope and authority

- Perform zero field-evolution or solver steps.
- Read only archived one-dimensional profiles and committed compact evidence.
- Treat source A as exploratory discovery and sources B and C as the two
  independent confirmation units.
- Treat arms as correlated diagnostics, never as independent replicates.
- Do not add a radius, angle, interface-width, material, mask-optimization, or
  mobility-law campaign.

## Frozen observables

### Source-A placement sweep

- Cases: `c14p5`, `c18p5`, `c22p5`, `c26p5`, `c30p5`, `c34p5`, `c38p5`, and
  `c64p5`.
- Primary common time: `t=1300`, before every included event.
- Morphology metric: the treated-minus-paired-untreated phase-radius response
  at the fixed untreated first-event distance `|s|=19.25`, averaged over the
  two wires and two signed arms.
- Lifetime response: the corrected modern-detector value in
  `source_data/position_response.csv`, referenced to the source-A untreated
  bracket `[1690,1700]` and midpoint `1695`.
- Primary descriptive association: Pearson correlation across all eight
  placements.
- Same-mode sensitivity: Pearson correlation restricted to `c26p5`, `c30p5`,
  `c34p5`, `c38p5`, and `c64p5`, whose first events remain
  junction-adjacent.
- No correlation p-value, fitted lifetime law, confidence interval, or
  population-inference claim is permitted because this is a post-hoc
  within-one-source association.

### Matched confirmation observable

- Sources: B (`seed=104729`) and C (`seed=130363`).
- Cases: `c26p5` and `c34p5`, each paired to its source-specific untreated
  profile.
- Times: `700,800,...,1500`; primary checks at `1300` and `1500`.
- Metric: the same four-arm mean phase-radius response at `|s|=19.25` used for
  the source-A sweep.
- Expected signs: positive for `c26p5` and negative for `c34p5`.
- The existing frozen three-contour weak-zone audit remains the independent
  robustness check; this synthesis does not rewrite or replace its contract.

## Frozen acceptance checks

The synthesis passes only if all of the following hold:

1. All eight source-A welds are connected at `t=1300` at all three recorded
   contour thresholds.
2. The source-A full-sweep Pearson correlation at `t=1300` is at least `0.90`.
3. The source-A same-failure-mode Pearson correlation at `t=1300` is at least
   `0.90`.
4. The far control `c64p5` has absolute phase-radius response below `0.01` and
   absolute corrected lifetime shift at most one diagnostic interval (`10`).
5. All eight matched B/C sign checks pass: two sources by two placements by
   two primary times.
6. The expected sign is persistent from no later than `t=1300` through
   `t=1500` for every B/C source--placement pair.
7. The existing three-contour held-out audit remains passed at `24/24` frozen
   checks, and its pre-event gate remains passed.

## Claim boundary

A pass supports an early morphological correlate of the placement response
within the selected right-angle, finite-interface model and replicated signed
precursors at two held-out source states. It does not establish a universal
predictor, unique causal variable, statistical population law, sharp-interface
limit, or material calibration.

The old source-A mechanism audit used the superseded untreated lifetime
baseline. Its archived metric values and original hashes are retained, but
all paper-facing lifetime shifts must be rejoined from the corrected compact
event table. Adding a constant baseline correction leaves Pearson and Spearman
correlations unchanged; the synthesis nevertheless recomputes them from the
corrected table rather than reusing the old reported values.
