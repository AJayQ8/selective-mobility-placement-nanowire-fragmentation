# Archive scope and exclusions

## Included

- normalized source tables used by the reported figures;
- final 137-mm and 190-mm artwork;
- graphical abstract and display-size previews;
- compact morphology projections used by Figure 1;
- original source identities and compact path-portable evidence records;
- fixed written/executable protocols and path-free zero-step check extracts;
- compact detector, contrast, transport, time-step, and grid-validation records;
- compact source-A and matched A/B/C pre-fragmentation morphology records;
- the fixed-probe/collar-support audit and earliest stored sign-lead analysis;
- the zero-step CMS normalized-response table and validation matrix;
- failed classical-dispersion and source-qualification gate summaries;
- CPU solver, geometry, detector, campaign, and analysis implementation;
- figure-generation code, tests, licenses, citation metadata, and SHA-256 manifest.

## Intentionally excluded

- article-source files and compiled articles;
- development history and correspondence;
- raw 3-D `.npy` fields and runtime checkpoints;
- trajectory directories, restart state, large diagnostic histories, and logs;
- machine-local absolute paths and host metadata;
- virtual environments, caches, temporary renders, and Git internals;
- tokens, credentials, and external-system metadata;
- superseded figure generators and outputs;
- multi-radius, pulse, kernel, state-mask, and other out-of-scope studies.

The omitted raw arrays are not required to inspect the implementation, verify
the normalized arithmetic, rerun the archived zero-step analyses, or regenerate
the figures. Their content identities remain recorded so a separately deposited
large-data archive can be verified later without changing this release.
