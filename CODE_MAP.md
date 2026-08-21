# Code and data map

## Numerical model

The model lineage is under
`science_lab/papers/nanowire_gb_junction/`:

- `roy_2021_reproduction/model.py` implements the CPU phase-field solver and
  released initializer semantics;
- `roy_gb_junction_sentinel/` provides source conditioning, diagnostics, and
  checkpoint storage;
- `roy_fixed_gb_bridge/` contains geometry and bridge metrics;
- `roy_selective_mobility_protection/` contains the four-arm mobility mask,
  campaign drivers, pairing logic, and transport analyses.

## Study-specific checks

`science_lab/papers/selective_mobility_scripta/` contains:

- `mobility_contrast/` -- the weaker-suppression confirmation;
- `spatial_refinement/` -- the common-source factor-two grid check;
- `precursor_synthesis/` -- the zero-step pre-fragmentation morphology join;
- `precursor_context_audit.py` and `precursor_context_audit/` -- the probe-
  support and early-sign interpretation audit plus its readout;
- `provenance/` -- compact source identities, detector replay, and transport
  recurrence records;
- `source_data/` -- normalized CSV/JSON values, the CMS matrix, failed-gate
  summaries, and morphology projections;
- `figures/` -- direct renderers and checked artwork.

## Entry points

```sh
make setup      # create the pinned Python environment
make test       # release, provenance, and figure checks
make test-all   # every included unit test
make figures    # regenerate artwork under reproduced_artifacts/
make provenance # rebuild the compact path-portable provenance manifest
```

Production campaigns are intentionally not part of the default validation
commands. Their drivers require explicit source paths, resource preflights,
and substantially more storage than the compact archive.
