#!/usr/bin/env python3
"""Zero-evolution preflight for the seed-104729 paired dt=0.5 audit."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from ..roy_fixed_gb_bridge.provenance import sha256_path
from ..roy_gb_junction_sentinel.storage import (
    atomic_json,
    reserve_output_directory,
)
from . import timestep_refinement_protocol as protocol


SOURCE_PATH = Path(__file__).resolve()
PAIR_RUNNER_PATH = SOURCE_PATH.with_name(
    "run_timestep_refinement_pair.py"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_frozen_source() -> dict[str, Any]:
    """Verify the exact existing float64 field without evolving it."""

    summary = _load_json(protocol.SOURCE_SUMMARY)
    metadata = _load_json(protocol.SOURCE_METADATA)
    checkpoint = summary["checkpoint"]
    hashes = {
        "summary": sha256_path(protocol.SOURCE_SUMMARY),
        "metadata": sha256_path(protocol.SOURCE_METADATA),
        "field": sha256_path(protocol.SOURCE_FIELD),
    }
    checks = {
        "summary_sha256": (
            hashes["summary"] == protocol.SOURCE_SUMMARY_SHA256
        ),
        "metadata_sha256": (
            hashes["metadata"] == protocol.SOURCE_METADATA_SHA256
        ),
        "field_sha256": (
            hashes["field"] == protocol.SOURCE_FIELD_SHA256
        ),
        "status": summary.get("status") == "completed",
        "seed": int(summary.get("seed", -1)) == protocol.SEED,
        "physical_time": (
            int(summary.get("current_step", -1))
            == int(protocol.SOURCE_PHYSICAL_TIME)
            and int(checkpoint.get("step", -1))
            == int(protocol.SOURCE_PHYSICAL_TIME)
        ),
        "float64": (
            checkpoint.get("dtype") == "<f8"
            and metadata.get("dtype") == "<f8"
        ),
        "shape": (
            checkpoint.get("shape") == [96, 768, 768]
            and metadata.get("shape") == [96, 768, 768]
        ),
        "field_bytes": (
            int(checkpoint.get("field_bytes", -1))
            == protocol.SOURCE_FIELD_BYTES
            and protocol.SOURCE_FIELD.stat().st_size
            == protocol.SOURCE_FIELD_BYTES
        ),
        "field_fingerprint": (
            checkpoint.get("field_fingerprint")
            == protocol.SOURCE_FINGERPRINT
            and metadata.get("field_fingerprint")
            == protocol.SOURCE_FINGERPRINT
        ),
        "checkpoint_field_hash": (
            checkpoint.get("field_sha256")
            == protocol.SOURCE_FIELD_SHA256
            and metadata.get("field_sha256")
            == protocol.SOURCE_FIELD_SHA256
        ),
        "checkpoint_metadata_hash": (
            checkpoint.get("metadata_sha256")
            == protocol.SOURCE_METADATA_SHA256
        ),
        "initial_mass": (
            float(summary.get("initial_mass"))
            == protocol.SOURCE_INITIAL_MASS
        ),
        "source_continuation_dt": (
            float(
                summary["contract"]["definition"]["parameters"][
                    "timestep"
                ]
            )
            == 1.0
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen source verification failed: {checks}")
    return {
        "summary_path": str(protocol.SOURCE_SUMMARY.resolve()),
        "summary_sha256": hashes["summary"],
        "metadata_path": str(protocol.SOURCE_METADATA.resolve()),
        "metadata_sha256": hashes["metadata"],
        "field_path": str(protocol.SOURCE_FIELD.resolve()),
        "field_sha256": hashes["field"],
        "field_fingerprint": protocol.SOURCE_FINGERPRINT,
        "dtype": checkpoint["dtype"],
        "shape": checkpoint["shape"],
        "field_bytes": checkpoint["field_bytes"],
        "physical_time": protocol.SOURCE_PHYSICAL_TIME,
        "initial_mass": protocol.SOURCE_INITIAL_MASS,
        "checks": checks,
    }


def verify_dt1_references() -> dict[str, Any]:
    """Verify the like-for-like records that freeze the acceptance targets."""

    reports: dict[str, Any] = {}
    for case_id in protocol.CASE_ORDER:
        path = protocol.DT1_SUMMARIES[case_id]
        summary = _load_json(path)
        digest = sha256_path(path)
        event = summary["event_assessment"]
        bracket = tuple(float(value) for value in event["event_bracket"])
        expected = protocol.DT1_EVENT_BRACKETS[case_id]
        source = summary["contract"]["source_checkpoint"]
        checks = {
            "summary_sha256": (
                digest == protocol.DT1_SUMMARY_SHA256[case_id]
            ),
            "completed": summary.get("status") == "completed",
            "case": summary.get("case") == case_id,
            "event_detected": event.get("detected") is True,
            "event_bracket": bracket == expected,
            "diagnostic_interval": (
                int(event.get("diagnostic_interval", -1))
                == int(protocol.DIAGNOSTIC_PHYSICAL_INTERVAL)
            ),
            "same_source_field": (
                source.get("field_sha256")
                == protocol.SOURCE_FIELD_SHA256
                and source.get("field_fingerprint")
                == protocol.SOURCE_FINGERPRINT
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"dt=1 reference verification failed for {case_id}: "
                f"{checks}"
            )
        reports[case_id] = {
            "summary_path": str(path.resolve()),
            "summary_sha256": digest,
            "event_bracket": list(bracket),
            "event_midpoint": (
                protocol.DT1_EVENT_MIDPOINTS[case_id]
            ),
            "checks": checks,
        }
    return reports


def run_preflight(output: Path) -> dict[str, Any]:
    """Write one versioned, solver-free launch manifest."""

    if not PAIR_RUNNER_PATH.is_file():
        raise FileNotFoundError(f"pair runner is absent: {PAIR_RUNNER_PATH}")
    source = verify_frozen_source()
    references = verify_dt1_references()
    runtime = protocol.runtime_projection()
    storage = protocol.storage_projection()
    free = int(shutil.disk_usage(output.parent).free)
    storage = {
        **storage,
        "free_disk_bytes": free,
        "disk_sufficient": free >= storage["required_free_disk_bytes"],
    }
    checks = {
        "source_verified": all(source["checks"].values()),
        "dt1_references_verified": all(
            all(item["checks"].values())
            for item in references.values()
        ),
        "physical_timestep_is_half": (
            protocol.PHYSICAL_TIMESTEP == 0.5
        ),
        "diagnostics_are_ten_physical_units": (
            protocol.DIAGNOSTIC_PHYSICAL_INTERVAL == 10.0
            and protocol.DIAGNOSTIC_PROPOSAL_INTERVAL == 20
        ),
        "physical_time_is_separate_from_proposals": (
            protocol.physical_time_for_proposal_count(20) == 110.0
        ),
        "case_order_is_sequential_pair": (
            protocol.CASE_ORDER == ("untreated", "c34p5")
            and runtime["sequential"] is True
        ),
        "runtime_is_long_and_bounded": (
            5.5 <= runtime["safeguarded_wall_hours"] <= 6.5
        ),
        "storage_is_about_five_gib": (
            5.0 <= storage["projected_field_gibibytes"] <= 5.2
        ),
        "disk_sufficient": storage["disk_sufficient"],
        "acceptance_is_frozen": (
            protocol.MAXIMUM_INDIVIDUAL_RELATIVE_TIME_ERROR
            == 0.05
            and protocol.MAXIMUM_PAIRED_EFFECT_RELATIVE_ERROR
            == 0.25
            and protocol.DT1_PAIRED_EFFECT == -110.0
        ),
        "zero_simulation_proposals": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"timestep preflight failed: {checks}")
    reserve_output_directory(output)
    report = {
        "schema_version": 1,
        "status": "completed",
        "classification": "paired_dt0p5_zero_evolution_preflight_passed",
        "seed": protocol.SEED,
        "cases": [case.to_dict() for case in protocol.CASES],
        "clock": {
            "source_physical_time": protocol.SOURCE_PHYSICAL_TIME,
            "target_physical_time": protocol.TARGET_PHYSICAL_TIME,
            "physical_timestep": protocol.PHYSICAL_TIMESTEP,
            "target_proposal_count": protocol.TARGET_PROPOSAL_COUNT,
            "diagnostic_physical_interval": (
                protocol.DIAGNOSTIC_PHYSICAL_INTERVAL
            ),
            "diagnostic_proposal_interval": (
                protocol.DIAGNOSTIC_PROPOSAL_INTERVAL
            ),
            "checkpoint_step_semantics": (
                "continuation_proposal_count"
            ),
        },
        "source": source,
        "dt1_references": references,
        "acceptance": {
            "natural_wire": protocol.NATURAL_WIRE,
            "natural_site_absolute_bounds": list(
                protocol.NATURAL_SITE_ABSOLUTE_BOUNDS
            ),
            "maximum_individual_relative_time_error": (
                protocol.MAXIMUM_INDIVIDUAL_RELATIVE_TIME_ERROR
            ),
            "dt1_paired_effect": protocol.DT1_PAIRED_EFFECT,
            "maximum_paired_effect_relative_error": (
                protocol.MAXIMUM_PAIRED_EFFECT_RELATIVE_ERROR
            ),
            "treated_bracket_must_precede_untreated": True,
        },
        "runtime": runtime,
        "storage": storage,
        "checks": checks,
        "simulation_proposals_performed": 0,
        "provenance": {
            "preflight_runner_sha256": sha256_path(SOURCE_PATH),
            "pair_runner_sha256": sha256_path(PAIR_RUNNER_PATH),
            "protocol_sha256": sha256_path(
                Path(protocol.__file__).resolve()
            ),
            "scientific_contract_sha256": sha256_path(
                protocol.CONTRACT_PATH
            ),
            "python": sys.version,
        },
        "scope": {
            "zero_solver_evolution": True,
            "source_field_loaded_into_solver": False,
            "pair_launched": False,
            "follow_on_authorized": False,
        },
    }
    atomic_json(output / "summary.json", report)
    return report


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=protocol.PREFLIGHT_OUTPUT,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    report = run_preflight(arguments.output.resolve())
    print(
        json.dumps(
            {
                "status": report["status"],
                "classification": report["classification"],
                "simulation_proposals_performed": 0,
                "projected_wall_hours": report["runtime"][
                    "safeguarded_wall_hours"
                ],
                "projected_field_gibibytes": report["storage"][
                    "projected_field_gibibytes"
                ],
                "output": str(arguments.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
