# Preserved negative validation gates

These compact summaries record validation attempts that were stopped under
their frozen criteria. They are evidence about model fidelity and scope, not
passing benchmarks and not inputs that were tuned until agreement appeared.

- `cylinder_dispersion_summary.json` records the failed radius-6 isolated-
  cylinder classical-dispersion gate.
- `conditioned_cylinder_summary.json` records the unsuccessful conditioned-
  source repair attempt.
- `rw_source_screen_summary.json` records the radius-12 source-viability screen
  that improved the finite-interface regime but did not qualify for a full
  classical-dispersion expansion.

The reported welded-junction comparisons remain paired branches launched from
identical checkpoints under an identical conserved diffuse-interface operator.
These negative gates therefore bound transfer to classical sharp-interface
surface diffusion; they do not convert the paired first-passage comparisons
into material-calibrated predictions.
