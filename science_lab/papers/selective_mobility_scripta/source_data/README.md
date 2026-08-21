# Normalized source data

These CSV, JSON, and NPZ files are the compact public evidence layer.
They reproduce the values in the reported tables and figures without the
multi-gigabyte trajectory fields.

- `evidence.json` maps the reported results to their supporting records.
- `provenance.json` binds S01--S18 to original identities and portable copies.
- the CSV files contain the timing, topology, transport, contrast, and
  refinement values reported in the study;
- `morphology_event_projections.*` contains the compact Figure 1 projections;
- `cms_normalized_response.csv`, `cms_validation_matrix.csv`, and
  `cms_analysis_receipt.json` are deterministic zero-step CMS conversion
  products;
- `cms_benchmark_inputs.json` contains only the hash-traceable benchmark fields
  consumed by that builder;
- `failed_benchmarks/` preserves negative validation gates without presenting
  them as successful benchmarks;
- `raw/` contains the small, hash-bound inputs required to rerun the precursor
  synthesis; it does not contain 3-D trajectory fields;
- large excluded run summaries are represented by original identities rather
  than copied operational metadata.

Run `make test` from the repository root to check the reported arithmetic,
identities, and release boundary.

The optional `build_morphology_projections.py` utility reconstructs the compact
projection archive from separately deposited raw fields. It requires an
explicit `--archive-root`; the raw fields are not included here.
