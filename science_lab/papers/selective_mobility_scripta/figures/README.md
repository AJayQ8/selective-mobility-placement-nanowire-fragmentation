# Figure assets

The renderers have direct, semantic names:

- `figure1.py` -- placement response and terminal morphology;
- `figure2.py` -- source-level confirmation;
- `figure3.py` -- audited pre-fragmentation morphology evidence;
- `figure4.py` -- control-volume transport accounting;
- `figure5.py` -- numerical and mobility-contrast robustness;
- `supporting_figures.py` -- overlap and scalar-control plots;
- `graphical_abstract.py` -- 3000 x 1200 graphical abstract;
- `generate_all.py` -- runs the complete set.

`artwork/main/` contains the five main figures. Figures 1, 2, 4, and 5 have
137-mm review versions; the taller three-panel Figure 3 has a 160-mm review
version. The `full_width/` subdirectory contains native 190-mm alternatives.
Supporting artwork and the graphical abstract have their own directories.
JSON manifests bind every image to its source files and record its physical
dimensions.

Run `make figures` at the repository root to generate a separate copy under
`reproduced_artifacts/figures/`. The checked-in artwork is not overwritten.
