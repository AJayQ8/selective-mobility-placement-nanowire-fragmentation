# R6 untreated-versus-c34 spatial-refinement contract

Status: frozen before any fine-grid evolution. This package is a numerical
validation of an existing result, not a new parameter search.

## Question

Does the harmful c34 placement effect survive a factor-two spatial refinement
when the physical domain, radius, diffuse-interface width, mobility field,
source morphology, timestep, and event detector are held fixed? The fine
branches use the same precommitted elapsed horizon of `2000`. The historical
coarse campaign used a horizon of `2900`; equality of the fine and coarse
horizons is neither imposed nor claimed.

The validation compares only two trajectories, in this order:

1. untreated;
2. c34, the tubular collar with support `28.5 <= |s| <= 40.5`, transition
   width `3.0`, and protected mobility factor `0.1`.

No third geometry, changed timestep, horizon extension, threshold change, or
adaptive rescue is authorized by this contract.

## Immutable source and interpolation

The sole source is the accepted R6 branch-aware state at physical time 120:

- coarse shape: `(96, 384, 384)`;
- coarse spacing: `h = 0.5`;
- field SHA-256:
  `69802e10c44be4ad343efbd68f955897d4e068d5407a3f6e5ea2b9b0347264a1`;
- checkpoint metadata SHA-256:
  `965ffb278521ebfb320385662f5f7ad28abdcb029514f1c30cb1cd4fb6043c3b`;
- source summary SHA-256:
  `d927a3287807390ecd4034dc0c8a5a8e6d8ea60588b13e31d3ae06983f8d29e0`;
- array fingerprint:
  `3a410dd120911e2f709c6277a16b546bb6b6ae504469615cc47bd76cbd77a14a`.

The default source directory is the original host path, but `--source-root`
may point at a relocated copy containing the same three frozen filenames. The
field, metadata, summary, and array fingerprint must still match the hashes
above; the override cannot select different science. Raw source data remains
outside Git.

It is interpolated once to `(192, 768, 768)`, `h = 0.25`, by separable exact
periodic Fourier resampling. There is no clipping, smoothing, mass correction,
noise injection, or source re-equilibration. The prolongator must pass analytic
periodic-mode, coarse-Nyquist, mean/mass, and coincident-node tests. The
generated fine source must reproduce every coarse node to absolute error
`<= 2e-11`, preserve the mean to `<= 2e-13`, and preserve physical mass to
relative error `<= 2e-12`.

This design isolates the continuation-grid sensitivity of the observed
placement contrast. It does not claim that the interpolated field is identical
to a source independently evolved at `h = 0.25` from time zero.

## Frozen physics and numerics

- physical domain: `48 x 192 x 192`;
- wire radius: `R = 6` (`24` fine cells);
- first and second wire centers in the thin direction: `x = 0` and `x = 12`;
- `A = kappa = M0 = 1`, stabilizer `alpha = 0.5`;
- diffuse-interface width: `W = sqrt(8)`;
- released variable mobility and periodic boundaries;
- production timestep: `dt = 1`;
- elapsed horizon after the matched source: `2000`;
- diagnostics every `10`, energy every `50`, profiles every `100`;
- regular restart checkpoints at elapsed times `800` and `1600`, plus an
  event, horizon, or interruption checkpoint as needed;
- trajectories stop after the first confirmed event or at the fixed horizon.

The exact legacy solver, collar geometry, event detector, contact diagnostic,
persistence, and checkpoint modules are imported from
`science_lab.papers.nanowire_gb_junction`. Their hashes are checked before a
run.

Checkpoint publication is reconciled without changing the trajectory. A
complete unjournaled field/metadata pair is adopted only when deterministic
replay reaches that exact step and matches the stored field exactly. A
field-only checkpoint (the normal crash window between the two atomic writes)
is completed only after the same exact replay comparison. A metadata-only
checkpoint cannot arise from the writer's ordering; it is preserved and the
queue fails closed. This applies to the fine source and to regular, event,
interruption, and horizon checkpoints. Terminal journal state is verified and
promoted if a crash occurred immediately before summary publication. No
checkpoint is overwritten, tolerated approximately, or used to alter an event.

## Discarded timestep guard

Before either production trajectory, both cases are advanced from the fine
source by one `dt=1` proposal and separately by two `dt=0.5` proposals. These
fields are discarded. The guard passes only if, for each case:

- active-interface RMSE is `<= 0.02`;
- maximum normalized contact-radius difference is `<= 0.01`;
- maximum axial-profile normalized RMSE is `<= 0.02`;
- maximum axial-profile normalized absolute difference is `<= 0.05`.

A failed guard stops the queue. It does not authorize `dt=0.5` production.

## Frozen event and health definitions

The event is the legacy conservative persistent single-arm detector: one
complete transverse slice below `c=0.45`, outside the crossed-junction central
exclusion, nested and site-stable at thresholds `0.45/0.50/0.55`, present in
three consecutive ten-unit diagnostics, with maximum gap displacement `W`.

Numerical health requires finite fields, absolute field magnitude `<= 2`,
relative mass drift `<= 1e-4`, and sampled relative energy rebound `<= 1e-6`.

## A-priori acceptance bounds

The coarse reduced-domain references are untreated `[1600, 1610]` (midpoint
`1605`, natural local gap at `|s|=18.75`) and c34 `[1500, 1510]` (midpoint
`1505`, simultaneous natural gaps on `first_wire_z` at `-18.0` and
`second_wire_y` at `+18.5`), giving `c34 - untreated = -100`. Multiplicity is
reported rather than required to match exactly.

The historical coarse summaries are frozen by SHA-256 as
`ec4b8f79d1a4cefe5793cb126025c7ccfe28537c4074b1dd41feae39e7d403d8`
for untreated and
`aea7466a58b6d9c2de0f8475cd1ba1f11477658015ddf0ac932bc8483591f7d0`
for c34. When that adjacent coarse archive is present, preflight verifies both
files; a partial or hash-mismatched archive is `NO-GO`. A relocatable copy that
contains only the exact source remains valid because the reference values and
expected summary hashes are themselves frozen in this contract and protocol.

The fine-grid validation passes only if all of the following hold:

1. both trajectories are numerically healthy and have a confirmed event by
   2000 with one or two diagnostic-tied persistent gaps;
2. c34's complete event bracket is strictly earlier than untreated's;
3. every persistent gap is on `first_wire_z` or `second_wire_y`, with
   `14 <= |s| <= 22.5`; no remote gap is allowed;
4. each fine midpoint is within 5% of its own coarse midpoint;
5. the fine paired midpoint effect is in `[-125, -75]`.

An event-free horizon is censored and does not pass. More than two tied gaps,
any remote gap, failed health check, failed timestep guard, or failed
provenance check does not pass. One versus two gaps is reported, but exact
multiplicity convergence is not claimed because the coarse c34 reference has
two simultaneous natural gaps. Results outside the numerical bounds are
reported as spatial sensitivity; they are not repaired or relabelled.

## Resource and execution gate

The source build and trajectories are long/large. A zero-evolution preflight
must return `GO` immediately before launch. It separately checks physical RAM,
the actual output filesystem, worst-case remaining persistent output, the
largest atomic write, a 10 GiB untouched reserve, filesystem volatility, the
validated temporary-Git-snapshot fingerprint, competing simulations,
the exact source, legacy hashes, current available RAM, and restart design.
Every runtime/transitive repository file must be tracked, clean, and at a
local HEAD identical to the configured `origin/*` upstream. The preflight
records a complete live SHA-256 manifest and receipt-core hash; every later
stage and resume rechecks that binding. In addition, every stage contract
stores one immutable campaign identity: the exact source root and protocol,
the full frozen-file manifest, resolved Python executable and package/runtime
versions, and Git HEAD/upstream identity. A fresh resource preflight may update
free-space/RAM measurements, but it cannot make a prior stage from a different
code, executable, environment, source root, or Git identity acceptable.
Active snapshot growth, a competing
science/Git process, or insufficient margin is `NO-GO`; this package never
deletes data.

Worst-case persistent output is the bounded field/checkpoint inventory plus a
separate 512 MiB allowance for JSON summaries, receipts, logs, profiles, and
other planned small artifacts. That allowance is planned output, not reserve
and not filesystem volatility. It is included in every preflight and runtime
disk calculation; the volatility allowance remains an additional untouched
term.

Peak RAM is conservatively frozen at 32 GiB. The legacy full-step inventory is
256 bytes per fine cell (27.0 GiB); the discarded guard may additionally hold
the common source, the `dt=1` endpoint, and the treated mobility factor while
a proposal owns its work arrays, giving about 29.6 GiB before rounding up.
Physical RAM must leave 16 GiB above the 32 GiB ceiling. Current
`psutil.available` (which includes reclaimable capacity) must separately leave
8 GiB, so the live gate is 40 GiB. The smaller live margin is deliberate: the
32 GiB ceiling already rounds the explicit 29.6 GiB inventory upward.

The supervisor runs source preparation, the discarded timestep guard,
untreated, c34, and the frozen analysis sequentially. It has no adaptive
follow-on branch. A hard 9.5-hour campaign wall-clock cap sends SIGTERM to the
active runner, allows up to 60 seconds for an interruption checkpoint, then
stops the queue; if graceful preservation cannot finish, the child is killed
and the latest prior valid checkpoint remains the restart point.
