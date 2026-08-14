# Figure assets

The renderers have direct, semantic names:

- `figure1.py` -- placement response and terminal morphology;
- `figure2.py` -- source-level confirmation;
- `figure3.py` -- control-volume transport accounting;
- `figure4.py` -- numerical and mobility-contrast robustness;
- `supporting_figures.py` -- overlap and scalar-control plots;
- `graphical_abstract.py` -- 3000 x 1200 graphical abstract;
- `generate_all.py` -- runs the complete set.

`artwork/main/` contains 137-mm Figures 1--4. Its `full_width/` subdirectory
contains native 190-mm alternatives. Supporting artwork and the graphical
abstract have their own directories. JSON manifests bind every image to its
source files and record its physical dimensions.

Run `make figures` at the repository root to generate a separate copy under
`reproduced_artifacts/figures/`. The checked-in artwork is not overwritten.
