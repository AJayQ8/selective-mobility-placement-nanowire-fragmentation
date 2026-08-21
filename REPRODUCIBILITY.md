# Reproducibility guide

## What this archive can reproduce directly

The archive directly supports three levels of checking:

1. **Claim audit** — recompute the reported timing shifts, propagated
   diagnostic intervals, source roles, transport signs, numerical checks, and
   refinement outcomes, precursor associations, probe context, and CMS
   validation outcomes from the normalized CSV/JSON layer.
2. **Figure regeneration** — rebuild the reported figures from those normalized
   inputs and the compact morphology projections.
3. **Implementation inspection and unit tests** — inspect and test the CPU
   solver, detector, geometry, campaign protocols, and analysis routines.

The compact archive does not contain the multi-gigabyte 3-D trajectory fields
or checkpoints. Re-running a full production campaign therefore starts from the
published initializer, seeds, and protocols rather than replaying a retained
field. Exact historical field identities remain recorded by SHA-256 in the
provenance manifests.

## Environment

- Python 3.12 (the frozen environment used 3.12.8)
- NumPy 1.26.4
- SciPy 1.13.1
- Matplotlib 3.9.4
- Pillow 12.3.0
- pypdf 6.15.0
- psutil 7.2.2

Create the environment:

```sh
make setup
```

All subsequent Make targets automatically use `.venv/bin/python` when that
environment exists.

## Validate the release

```sh
make test
```

This command performs no production simulation and writes no scientific
result. It verifies shipped artifacts against `MANIFEST.sha256`, runs the
archive audit, and exercises the figure data and layout tests that do not
require the excluded raw fields.

For the complete public test suite:

```sh
make test-all
```

This executes every shipped unit and artifact test without launching a
production campaign.

## Regenerate figures

```sh
make figures
```

The command writes a new set under `reproduced_artifacts/`; it never overwrites
the checked-in artwork. Figures 1--5 and the supporting figures use bundled
normalized inputs. The graphical abstract is regenerated only when Arial is
available; otherwise the script reports that the font is unavailable and
leaves the included hash-verified copy unchanged.

Raster pixels are deterministic under the pinned environment. Scientific
values, geometry, physical sizes, and artifact hashes are checked separately.

Figures 1, 2, 4, and 5 have native 137-mm review artwork. The taller three-panel
Figure 3 has a native 160-mm review version. Full-width artwork is native
190-mm output and is provided separately so it is not downscaled through a
smaller layout.

## Rebuild the archived zero-step analyses

```sh
.venv/bin/python -m science_lab.papers.selective_mobility_scripta.precursor_synthesis.analyze
.venv/bin/python -m science_lab.papers.selective_mobility_scripta.precursor_context_audit
.venv/bin/python -m science_lab.papers.selective_mobility_scripta.source_data.build_cms_analysis
```

These commands read only committed compact evidence. They perform zero model
time steps and do not launch a simulation. The first command can optionally
refresh its compact extraction from a separately available archive, but that
is not required for the shipped readout.

## Large simulations

The production calculations are expensive and can generate hundreds of
gigabytes. Do not launch them casually. The frozen long-run protocols include
zero-step RAM/disk preflights, exact-resume checkpoints, and fail-closed guards.
The source roots for original coarse/refinement checkpoints are command-line
or environment inputs in the public copy; original field hashes remain frozen.
Some retained launch supervisors contain the original macOS-specific process,
power-management, and temporary-storage checks. They document how the reported
runs were guarded, but they are separate from the platform-neutral validation
and figure-regeneration commands above.

## Precision and scope

The released CPU path reproduces the documented NumPy/SciPy mixed-precision
semantics. It is not claimed to be bitwise identical to the original cuFFT
implementation. The factor-two continuation-grid check holds a mature source
fixed; it is not source-formation convergence or a sharp-interface limit.
