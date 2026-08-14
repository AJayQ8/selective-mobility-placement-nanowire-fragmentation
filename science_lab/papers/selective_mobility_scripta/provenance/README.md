# Portable provenance

This directory contains compact, inspectable evidence for the campaign
contracts, pre-run protocol, detector replay, weaker-contrast confirmation,
B/C transport recurrence, time-step check, and factor-two refinement.

`source_index.json` records the original SHA-256 and byte length for S01--S16.
Compact source records are included where they support direct auditing.
Large runtime summaries are represented by their original identities and the
normalized values derived from them. `scripts/build_public_provenance.py`
recomputes these bindings in `source_data/provenance.json`.

The paired-repeat directory contains a path-free extract of its zero-step
preflight. The executable protocol and public scientific contract are kept
once, together in the runtime package under `nanowire_gb_junction/`.
The extract records separate original and public hashes for those two files.
The public copies retain the scientific definitions while using portable
repository names and wording; neither is represented as a byte-for-byte copy.

Raw trajectory arrays are deliberately excluded. Their identities and the
scientific results derived from them remain bound by the manifests.

The detector-replay contract records the original replay and continuation
runners by path and SHA-256, but those two historical launch files are not
copied into this compact archive. The replay also depends on three excluded
large checkpoints. `detector_replay/result_portable.json` is the inspectable
result record; the contract preserves the identity of the original execution
without implying that the replay can be rerun from this compact tree alone.

Portable records use two non-filesystem URI prefixes: `source://legacy-nanowire`
identifies an object in the original large-data tree, and
`source://study-archive` identifies an object in the private study snapshot.
These URIs preserve provenance labels without exposing machine-local paths;
they are not expected to resolve as public links.

Some copied historical records retain the original campaign role label
`prospective_scan`. In the paper and normalized public tables, source A is
described more precisely as an exploratory scan and is not counted as an
independent confirmation.
