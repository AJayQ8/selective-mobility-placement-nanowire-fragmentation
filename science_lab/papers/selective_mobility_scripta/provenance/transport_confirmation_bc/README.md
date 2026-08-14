# Archived B/C transport confirmation

This package tests whether the weak-zone control-volume signature first found
in exploratory source A recurs in archived confirmation sources B and C.  It
does not run or resume a simulation.

`frozen_contract.json` was written and hashed before any B/C transport rate
was calculated.  It binds the six archived cases, their t=1200/1300/1400
profiles, their t=1300 fields, the untreated-derived control volume, the
source-A implementation, the one-percent continuity limit, and the directional
pass/fail rules.  The study is therefore a hash-bound pre-analysis validation
of existing trajectories, not a preregistered simulation campaign.

Run from the repository root with the historical paired-panel result directory
supplied explicitly:

```sh
.venv/bin/python -m \
  science_lab.papers.selective_mobility_scripta.provenance.transport_confirmation_bc.run_analysis \
  --historical-root /path/to/paired_repeat_panel_v1 \
  --preflight-only

.venv/bin/python -m \
  science_lab.papers.selective_mobility_scripta.provenance.transport_confirmation_bc.run_analysis \
  --historical-root /path/to/paired_repeat_panel_v1
```

The compact result contains no raw absolute input path.  Raw 3-D fields remain
outside the paper repository.

## Result

The frozen all-or-nothing gate did **not** pass.  Both confirmation sources
reproduced the two directional signatures in both estimators:

- source B: intermediate +0.063590 observed / +0.062855 reconstructed;
  outboard -0.060052 observed / -0.058795 reconstructed;
- source C: intermediate +0.064252 observed / +0.063496 reconstructed;
  outboard -0.060494 observed / -0.059134 reconstructed.

All 24 reconstructed arm rates had the same sign as their centered observed
rates.  Twenty-three met the frozen one-percent armwise closure ceiling.  The
single miss was source C, outboard, first-wire-z minus: 1.1575%.  Every
source-level case mean closed within 0.74%.  The correct interpretation is
therefore **directional recurrence with one marginal continuity-gate miss**,
not a passed pre-analysis validation and not a failed sign test.  No analysis
choice was changed after seeing the result.
