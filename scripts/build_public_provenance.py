#!/usr/bin/env python3
"""Bind original evidence identities to their path-portable public copies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "science_lab" / "papers" / "selective_mobility_scripta"
INDEX = PAPER / "provenance" / "source_index.json"
OUTPUT = PAPER / "source_data" / "provenance.json"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    records = []
    for source in index["sources"]:
        record = {
            "id": source["id"],
            "role": source["role"],
            "original_identity": {
                "archive_path": source["original_archive_path"],
                "sha256": source["original_sha256"],
                "bytes": source["original_bytes"],
            },
        }
        if "public_path" in source:
            path = ROOT / source["public_path"]
            if not path.is_file():
                raise RuntimeError(f"missing public copy for {source['id']}: {path}")
            record["public_copy"] = {
                "path": source["public_path"],
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "path_policy": "portable",
            }
        else:
            record["public_copy"] = None
            record["availability"] = "original identity recorded; compact derived values included"
        if "companion" in source:
            companion = source["companion"]
            companion_path = ROOT / companion["path"]
            if sha256(companion_path) != companion["sha256"]:
                raise RuntimeError(f"companion mismatch for {source['id']}")
            record["companion"] = companion
        records.append(record)
    derived_names = [
        "evidence.json",
        "position_response.csv",
        "paired_panel.csv",
        "equal_budget_controls.csv",
        "primary_mask_budget.csv",
        "transport_response.csv",
        "mobility_contrast.csv",
        "transport_confirmation_bc.csv",
        "timestep_refinement.csv",
        "spatial_refinement.csv",
        "detector_replay.csv",
        "numerical_checks.csv",
        "morphology_event_projections.json",
        "morphology_event_projections.npz",
        "cms_benchmark_inputs.json",
        "cms_normalized_response.csv",
        "cms_validation_matrix.csv",
        "cms_analysis_receipt.json",
    ]
    derived = []
    for name in derived_names:
        path = PAPER / "source_data" / name
        derived.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    for relative in (
        "precursor_synthesis/readout_v1/source_a_placement_precursor.csv",
        "precursor_synthesis/readout_v1/matched_phase_radius_time_series.csv",
        "precursor_synthesis/readout_v1/summary.json",
        "precursor_synthesis/readout_v1/manifest.json",
        "precursor_context_audit/readout_v1/summary.json",
        "precursor_context_audit/readout_v1/probe_context.csv",
        "precursor_context_audit/readout_v1/early_sign_lead.csv",
        "precursor_context_audit/readout_v1/manifest.json",
        "source_data/failed_benchmarks/cylinder_dispersion_summary.json",
        "source_data/failed_benchmarks/conditioned_cylinder_summary.json",
        "source_data/failed_benchmarks/rw_source_screen_summary.json",
    ):
        path = PAPER / relative
        derived.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": index["schema_version"],
        "manifest_id": index["manifest_id"],
        "identity_policy": index["identity_policy"],
        "raw_trajectory_fields_included": False,
        "absolute_machine_paths_included": False,
        "sources": records,
        "derived_artifacts": derived,
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} with {len(records)} sources")


if __name__ == "__main__":
    main()
