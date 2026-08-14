# Focused spatial-refinement validation

This folder implements the frozen untreated-versus-c34 R6 validation in
[`SCIENTIFIC_CONTRACT.md`](SCIENTIFIC_CONTRACT.md). Raw fields are written only
under ignored `results/` directories.

The preflight and supervisor preserve the original macOS execution safeguards
for historical reproducibility. They require external hash-matched source
fields and are not invoked by the portable repository validation targets.

From the repository root, run the zero-evolution gate with:

```sh
.venv/bin/python -m \
  science_lab.papers.selective_mobility_scripta.spatial_refinement.preflight
```

For a relocated byte-identical source, append
`--source-root /path/to/source`; the override is rejected unless all frozen
hashes match.

After a fresh `GO` and review of the quoted runtime, launch the frozen sequence
with:

```sh
.venv/bin/python -m \
  science_lab.papers.selective_mobility_scripta.spatial_refinement.supervisor \
  --launch-frozen-sequence
```

The supervisor performs exactly: preflight, fine-source interpolation,
discarded timestep guard, untreated, c34, frozen analysis. It has no follow-on
branch. A stopped run is resumed by invoking the same supervisor command; each
stage verifies its stored JSON contract and checkpoint hashes before resuming.
Complete or field-only crash-window checkpoints are reconciled only after an
exact deterministic replay match. Metadata-only or mismatching artifacts are
preserved and stop the queue.

Run focused tests with:

```sh
.venv/bin/python -m unittest discover \
  -s science_lab/papers/selective_mobility_scripta/spatial_refinement \
  -t . -p 'test_*.py' -v
```
