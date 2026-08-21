# Precursor probe-context audit

This is a zero-simulation-step interpretation audit of the frozen precursor
synthesis. It answers two questions that the original synthesis did not make
explicit:

1. Does the fixed phase-radius probe at `|s|=19.25` intersect the imposed
   mobility collar for each source-A placement?
2. How long before the event bracket is the expected sign already present in
   the two representative stored time series?

The audit makes the five outside-support cases with a common first-event mode
the primary post-hoc descriptive association. The all-eight association is
retained only as a confounded sensitivity calculation because its three
remote-event cases are exactly the three cases whose probe intersects collar
support. No p-value, fitted lifetime law, or calibrated predictor is reported.

Regenerate from the repository root with:

```sh
PYTHONPATH=science_lab python3 \
  -m papers.selective_mobility_scripta.precursor_context_audit
```

The generated `readout_v1/` directory contains the probe table, early-sign
lead table, summary, and hash manifest. The script reads only committed compact
evidence and performs zero simulation steps.

The same script can be run from a flattened review/archive package by passing
`--source-a`, `--time-series`, `--precursor-summary`,
`--normalized-response`, and `--output` explicitly. This portable entry point
recomputes the audit rather than merely checking copied output hashes.
