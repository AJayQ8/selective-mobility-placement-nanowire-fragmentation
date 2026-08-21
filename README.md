# Placement-dependent mobility suppression in a welded nanowire junction

This repository contains the public computational archive for the study
*Placement can reverse the fragmentation-time response to localized mobility
suppression in a welded nanowire junction*.

The central result is a paired, same-failure-class sign reversal. Moving each
of four otherwise identical low-mobility collars outward from
`s_c = 26.5` to `s_c = 34.5` changes the response from delayed to accelerated
fragmentation. The ordering was found in exploratory source A, confirmed in
independent sources B and C, retained at two mobility contrasts, and subjected
to targeted time-step and common-source grid checks.

Release 1.1.0 adds the zero-step pre-fragmentation morphology synthesis used in
the Computational Materials Science manuscript. Its primary descriptive
association uses only the five placements for which the fixed probe lies
outside the collar support and the event mode is common ($r=0.949$, $n=5$).
The all-eight value is retained only as a confounded sensitivity result. The
release also preserves the failed cylinder-dispersion gates and a compact CMS
validation matrix instead of hiding or tuning those negative checks.

## Start here

- Normalized evidence: [`source_data/`](science_lab/papers/selective_mobility_scripta/source_data/)
- Figure assets and renderers: [`figures/`](science_lab/papers/selective_mobility_scripta/figures/)
- Portable provenance: [`provenance/`](science_lab/papers/selective_mobility_scripta/provenance/)
- Precursor synthesis: [`precursor_synthesis/`](science_lab/papers/selective_mobility_scripta/precursor_synthesis/)
- Probe-context audit: [`precursor_context_audit/`](science_lab/papers/selective_mobility_scripta/precursor_context_audit/)
- Preserved negative gates: [`failed_benchmarks/`](science_lab/papers/selective_mobility_scripta/source_data/failed_benchmarks/)
- Code and data map: [`CODE_MAP.md`](CODE_MAP.md)
- Reproduction instructions: [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)
- Exact archive contents: [`ARCHIVE_SCOPE.md`](ARCHIVE_SCOPE.md)

## Quick validation

With Python 3.12:

```sh
make setup
make test
```

After `make setup`, every Make target automatically uses `.venv/bin/python`.
`make test` is compact and does **not** run the large production simulations.
It verifies the normalized evidence, precursor and context audits, CMS matrix,
preserved negative gates, provenance records, figure manifests, licenses, and
release hashes. Run `make test-all` to execute every shipped unit and artifact
test that does not require the excluded production fields.

Artwork regeneration is available through `make figures`. The graphical
abstract uses the Arial font family; the prebuilt, hash-verified output is
included for systems where that family is unavailable.

## Repository scope

This compact archive contains the code, normalized evidence, final artwork,
and portable provenance needed to inspect the reported calculations and
regenerate the figures. Large raw fields and checkpoints are excluded because
of their size; their identities are recorded in the provenance manifests.
See [`ARCHIVE_SCOPE.md`](ARCHIVE_SCOPE.md) for the complete boundary.

## Licenses

- Code is distributed under GPL-3.0-only; see `LICENSE` and
  `THIRD_PARTY_NOTICES.md`.
- Original figures, normalized data, and documentation by Ayas Alzanki are
  distributed under CC BY 4.0; see `LICENSES/CC-BY-4.0.txt`.
