# Paired repeat panel contract

This panel tests whether the position-scan mechanism survives independent
released-noise source realizations. It is a validation panel, not an adaptive
search.

## Frozen source realizations

- Existing realization: seed 2292 (already complete; not rerun here).
- New realization 2: seed 104729.
- New realization 3: seed 130363.

The new seeds were fixed before either source was generated. They must not be
screened or replaced because of a scientific outcome.

Each new source is initialized with the released binary crossed-wire geometry
and released overlapping-noise semantics, then evolved with the untreated Roy
solver from `t = 0` to `t = 100`. Only the verified `t = 100` field is branched.

## Frozen paired conditions

Every source realization runs these four conditions from the exact same saved
`t = 100` field, in this order:

1. untreated;
2. collar centered at 18.5;
3. collar centered at 26.5;
4. collar centered at 34.5.

The collar width, transition width, mobility factor, lattice, timestep,
diagnostic cadence, first-event detector, and `t = 3000` censoring horizon are
unchanged from the completed position scan.

## Interpretation fixed before the runs

- `c18.5` tests relocation of failure downstream of direct natural-site
  coverage.
- `c26.5` tests the protective crossover at the natural weak zone.
- `c34.5` tests the harmful intermediate placement at that same weak zone.
- Untreated is the within-realization clock and site reference.

Exact event times are not required to reproduce seed 2292. The primary
question is whether the site/mode ordering and the protective-versus-harmful
sign pattern are consistent within each paired source.

Numerical corruption, a failed source checkpoint, insufficient disk, or a
changed execution contract stops the sequence. A scientifically unfavorable
result is retained and does not permit seed replacement, parameter tuning, or
a new simulation.

## Scope boundary

The fixed scope contains two new conditioned sources, eight paired
trajectories, and one factual aggregate report after all trajectories verify.
It excludes a wider position sweep, convergence family, parameter retuning,
and additional simulation.
