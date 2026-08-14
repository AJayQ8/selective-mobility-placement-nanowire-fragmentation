# Normalized source data

These CSV, JSON, and NPZ files are the compact public evidence layer.
They reproduce the values in the reported tables and figures without the
multi-gigabyte trajectory fields.

- `evidence.json` maps the reported results to their supporting records.
- `provenance.json` binds S01--S16 to original identities and portable copies.
- the CSV files contain the timing, topology, transport, contrast, and
  refinement values reported in the study;
- `morphology_event_projections.*` contains the compact Figure 1 projections;
- large excluded run summaries are represented by original identities rather
  than copied operational metadata.

Run `make test` from the repository root to check the reported arithmetic,
identities, and release boundary.

The optional `build_morphology_projections.py` utility reconstructs the compact
projection archive from separately deposited raw fields. It requires an
explicit `--archive-root`; the raw fields are not included here.
