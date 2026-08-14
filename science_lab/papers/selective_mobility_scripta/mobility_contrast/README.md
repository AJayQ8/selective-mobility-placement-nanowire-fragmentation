# Weaker mobility-contrast confirmation

This directory contains the code and compact records for the
`m_min = 0.3` mobility-contrast test reported in the paper. The test used
confirmation sources B (`104729`) and C (`130363`) at two collar positions:

- intermediate: `s_c = 26.5`;
- outboard: `s_c = 34.5`.

The four treated trajectories were paired with the corresponding untreated
source baselines. The solver, detector, grid, time step, simulation horizon,
and numerical-health criteria were unchanged from the primary study.

## Study design

The parameters and evaluation criteria were recorded in
`frozen_contract.json` before the four simulations were run. The prescribed
comparison required a positive conservative paired interval at the
intermediate position, a negative interval at the outboard position,
junction-adjacent first failure in every case, correct source pairing, and
the specified numerical-health checks.

## Included files

- `frozen_contract.json`: parameters, source bindings, and evaluation criteria;
- `preflight.py`: resource and input checks used before the original runs;
- `run_case.py` and `run_campaign.py`: single-case and campaign execution code;
- `analyze.py`: calculation of the paired timing, topology, and health checks;
- `test_mobility_contrast.py`: focused tests for the contract and analysis;
- `provenance/mobility_contrast/analysis_portable.json`: compact result record
  included in the public archive.

The original campaign was run on macOS, so `preflight.py` includes
host-specific resource checks. Those checks are not required for inspection of
the archived results. Large trajectory fields and checkpoints are excluded
from this compact repository; their recorded identities are retained in the
portable provenance data.

## Validation

From the repository root, run:

```sh
make setup
make test-all
```

These commands validate the bundled contract, analysis code, provenance, and
reported compact results without rerunning the production simulations.
