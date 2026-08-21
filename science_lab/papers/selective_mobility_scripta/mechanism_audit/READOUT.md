# Time-resolved precursor readout

Status: **PASS**
Readout: 2026-08-20
Simulation steps: **0**

The prospective held-out gate passed 24 of 24 checks.  In sources B and C,
`c26p5` leaves the weakest natural-zone arm thicker than paired untreated,
while `c34p5` leaves it thinner, at times 1300 and 1500 and at all three
contour levels (`c=0.45`, `0.50`, and `0.55`).

## Primary `c=0.50` paired differences

Positive values mean thicker than paired untreated; negative values mean
thinner.

| Source | Time | Intermediate `c26p5` | Outboard `c34p5` |
|---|---:|---:|---:|
| A (discovery) | 1300 | +0.1508 | -0.2427 |
| A (discovery) | 1500 | +0.3831 | -0.5429 |
| B (held out) | 1300 | +0.1450 | -0.2506 |
| B (held out) | 1500 | +0.3839 | -0.5904 |
| C (held out) | 1300 | +0.1565 | -0.2585 |
| C (held out) | 1500 | +0.4312 | -0.7162 |

For all three source fields, the intermediate sign becomes persistent by time
1000 and the outboard sign is persistent from the first audited time, 700.
Time 1500 precedes every included terminal event; the earliest terminal
bracket begins at 1550.

## Interpretation

The result adds a replicated, time-resolved morphological precursor to the
terminal lifetime ordering.  It materially strengthens the statement that the
two placements steer the natural weak zone in opposite directions before
fragmentation.  It does not establish minimum contour radius as a unique
causal variable, make the four arms independent replicates, or demonstrate
geometric/material transfer.

Machine-readable evidence is in `readout_v1/summary.json` and
`readout_v1/weak_zone_time_series.csv`; `readout_v1/manifest.json` records the
output hashes.  The summary also records SHA-256 and byte length for all 90
immutable archive inputs.
