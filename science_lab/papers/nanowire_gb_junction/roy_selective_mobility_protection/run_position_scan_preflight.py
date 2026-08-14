#!/usr/bin/env python3
"""Prepare and verify the five-case collar-position scan without evolving it."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..roy_2021_reproduction import model as roy_model
from ..roy_2021_reproduction.model import FROZEN_DEG90
from ..roy_fixed_gb_bridge import (
    run_t100_checkpoint_source_localization_preflight as source_helper,
)
from ..roy_fixed_gb_bridge.provenance import sha256_path
from ..roy_gb_junction_sentinel.storage import (
    atomic_json,
    reserve_output_directory,
)
from . import geometry as selective_geometry
from . import model as selective_model
from . import position_scan_protocol as protocol
from . import run_far_field_placement_sentinel as far_runner
from .geometry import (
    four_arm_tubular_collar_factor,
    surface_weighted_mobility_deficit,
)


SOURCE_PATH = Path(__file__).resolve()
DIRECTORY = SOURCE_PATH.parent
DEFAULT_OUTPUT = (
    DIRECTORY / "results" / "position_scan_preflight_v1"
)
ANCHOR_AUDIT_PATH = (
    DIRECTORY
    / "results"
    / "position_anchor_mechanism_audit_v1"
    / "summary.json"
)
RUNTIME_REFERENCE_PATHS = {
    "thinning_zone": (
        DIRECTORY / "results" / "k1_first_break_t3000_v1" / "summary.json"
    ),
    "adjacent": (
        DIRECTORY
        / "results"
        / "outer_placement_first_break_t3000_v1"
        / "summary.json"
    ),
    "far": (
        DIRECTORY
        / "results"
        / "far_field_placement_sentinel_v1"
        / "summary.json"
    ),
}
WALL_CAP_CHECKPOINT_ALLOWANCE_PER_BATCH_INVOCATION = 2
SIGNAL_OR_CRASH_CHECKPOINT_ALLOWANCE = 1


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verified_reference_budget() -> dict[str, Any]:
    """Load the completed c14.5 resistance budget with provenance checks."""

    audit = _load_json(ANCHOR_AUDIT_PATH)
    checks = {
        "classification": (
            audit.get("classification")
            == "two_competing_failure_modes_supported_as_hypothesis_not_yet_law"
        ),
        "anchor_health": all(
            item.get("passed") is True
            for item in audit["numerical_integrity"].values()
        ),
        "budget_fairness": (
            audit["budget_fairness"][
                "all_positioned_budgets_within_two_percent"
            ]
            is True
        ),
        "reference_center": (
            audit["placement_coordinates"]["thinning_zone"][
                "center_distance"
            ]
            == protocol.REFERENCE_CENTER
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"reference budget failed: {checks}")
    value = float(
        audit["budget_fairness"]["cases"]["thinning_zone"][
            "surface_weighted_mobility_deficit"
        ]
    )
    if not np.isfinite(value) or value <= 0.0:
        raise RuntimeError("reference mobility budget is invalid")
    return {
        "center": protocol.REFERENCE_CENTER,
        "surface_weighted_mobility_deficit": value,
        "path": str(ANCHOR_AUDIT_PATH.resolve()),
        "sha256": sha256_path(ANCHOR_AUDIT_PATH),
        "checks": checks,
    }


def recomputed_reference_budget() -> dict[str, Any]:
    """Independently recompute c14.5 from the verified time-100 field."""

    geometry = protocol.geometry_for_center(
        protocol.REFERENCE_CENTER
    )
    factor = four_arm_tubular_collar_factor(
        FROZEN_DEG90.lattice,
        geometry,
        radius=protocol.RADIUS,
        interface_width=protocol.INTERFACE_WIDTH,
    )
    mapped = np.load(
        source_helper.T100_CHECKPOINT_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )
    try:
        budget = surface_weighted_mobility_deficit(
            mapped,
            factor,
            cell_volume=FROZEN_DEG90.lattice.cell_volume,
        )
    finally:
        del mapped
        del factor
        gc.collect()
    if not np.isfinite(budget) or budget <= 0.0:
        raise RuntimeError("recomputed reference budget is invalid")
    return {
        "center": protocol.REFERENCE_CENTER,
        "geometry": geometry.to_dict(),
        "surface_weighted_mobility_deficit": budget,
        "source_checkpoint_sha256": sha256_path(
            source_helper.T100_CHECKPOINT_PATH
        ),
        "method": (
            "current tubular factor and time-100 diffuse-interface "
            "surface-weighted mobility deficit"
        ),
    }


def derive_case_factor(
    case: protocol.PositionCase,
    *,
    reference_budget: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build one full factor and its exact time-100 budget report."""

    factor = four_arm_tubular_collar_factor(
        FROZEN_DEG90.lattice,
        case.geometry,
        radius=protocol.RADIUS,
        interface_width=protocol.INTERFACE_WIDTH,
    )
    mapped = np.load(
        source_helper.T100_CHECKPOINT_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )
    try:
        budget = surface_weighted_mobility_deficit(
            mapped,
            factor,
            cell_volume=FROZEN_DEG90.lattice.cell_volume,
        )
    finally:
        del mapped
        gc.collect()
    relative = (budget - reference_budget) / reference_budget
    static = protocol.static_case_report(case)
    checks = {
        **static["checks"],
        "factor_shape": factor.shape == FROZEN_DEG90.lattice.shape,
        "factor_finite": bool(np.isfinite(factor).all()),
        "factor_positive": float(np.min(factor)) > 0.0,
        "factor_at_most_one": float(np.max(factor)) <= 1.0,
        "budget_is_finite_positive": (
            np.isfinite(budget) and budget > 0.0
        ),
        "budget_difference_is_near_equal": (
            abs(relative) <= protocol.MAXIMUM_BUDGET_DIFFERENCE
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"factor preflight failed for {case.slug}: {checks}"
        )
    report = {
        **static,
        "surface_weighted_mobility_deficit": budget,
        "relative_budget_vs_thinning_zone": relative,
        "maximum_absolute_budget_difference": (
            protocol.MAXIMUM_BUDGET_DIFFERENCE
        ),
        "factor": {
            "shape": list(factor.shape),
            "minimum": float(np.min(factor)),
            "maximum": float(np.max(factor)),
            "support_cell_fraction": float(
                np.mean(factor < 1.0 - 1.0e-14)
            ),
        },
        "checks": checks,
    }
    return factor, report


def runtime_and_storage_preflight() -> dict[str, Any]:
    """Project aggregate resources from completed production measurements."""

    references: dict[str, Any] = {}
    medians: list[float] = []
    for name, path in RUNTIME_REFERENCE_PATHS.items():
        summary = _load_json(path)
        median = float(summary["execution"]["median_step_wall_seconds"])
        checks = {
            "completed": summary.get("status") == "completed",
            "median_finite_positive": (
                np.isfinite(median) and median > 0.0
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"runtime reference {name} failed: {checks}"
            )
        medians.append(median)
        references[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_path(path),
            "median_step_wall_seconds": median,
            "checks": checks,
        }
    slowest = max(medians)
    case_count = len(protocol.CASES)
    full_steps_per_case = protocol.TARGET_STEP - protocol.START_STEP
    full_steps = case_count * full_steps_per_case
    planning_steps = sum(
        protocol.PLANNING_TERMINAL_STEPS[case.center]
        - protocol.START_STEP
        for case in protocol.CASES
    )
    field_bytes = int(FROZEN_DEG90.lattice.cell_count * 8 + 256)
    checkpoints_per_case = (
        len(protocol.CHECKPOINT_STEPS)
        + protocol.MAXIMUM_EVENT_CHECKPOINTS
        + WALL_CAP_CHECKPOINT_ALLOWANCE_PER_BATCH_INVOCATION
        + SIGNAL_OR_CRASH_CHECKPOINT_ALLOWANCE
    )
    retained_field_bytes = (
        case_count * checkpoints_per_case * field_bytes
    )
    required_free_disk = (
        retained_field_bytes + protocol.MINIMUM_DISK_HEADROOM_BYTES
    )
    free_disk = int(shutil.disk_usage(protocol.RESULTS).free)
    full_raw = full_steps * slowest
    full_safe = full_raw * protocol.RUNTIME_SAFETY_FACTOR
    planning_raw = planning_steps * slowest
    planning_safe = planning_raw * protocol.RUNTIME_SAFETY_FACTOR
    per_case_safe = (
        full_steps_per_case
        * slowest
        * protocol.RUNTIME_SAFETY_FACTOR
    )
    report = {
        "runtime_references": references,
        "slowest_reference_median_step_wall_seconds": slowest,
        "runtime_safety_factor": protocol.RUNTIME_SAFETY_FACTOR,
        "case_count": case_count,
        "full_horizon": {
            "accepted_steps": full_steps,
            "raw_seconds": full_raw,
            "raw_hours": full_raw / 3600.0,
            "safeguarded_seconds": full_safe,
            "safeguarded_hours": full_safe / 3600.0,
        },
        "planning_scenario": {
            "meaning": (
                "one delayed c18p5 event near t2600 and four "
                "untreated-like events near t1720; not a guarantee"
            ),
            "terminal_steps": {
                protocol.center_slug(center): step
                for center, step in protocol.PLANNING_TERMINAL_STEPS.items()
            },
            "accepted_steps": planning_steps,
            "raw_seconds": planning_raw,
            "raw_hours": planning_raw / 3600.0,
            "safeguarded_seconds": planning_safe,
            "safeguarded_hours": planning_safe / 3600.0,
        },
        "per_case_full_horizon_safeguarded_hours": (
            per_case_safe / 3600.0
        ),
        "per_case_below_session_wall_cap": (
            per_case_safe < protocol.MAXIMUM_WALL_SECONDS
        ),
        "storage": {
            "field_checkpoint_bytes": field_bytes,
            "maximum_checkpoints_per_case": checkpoints_per_case,
            "checkpoint_allowance_semantics": (
                "regular + up to three event candidates + two wall-cap "
                "sessions + one signal/crash checkpoint per batch "
                "invocation; later manual resumes are re-preflighted"
            ),
            "retained_field_bytes": retained_field_bytes,
            "minimum_shared_headroom_bytes": (
                protocol.MINIMUM_DISK_HEADROOM_BYTES
            ),
            "required_free_disk_bytes": required_free_disk,
            "free_disk_bytes": free_disk,
            "disk_sufficient": free_disk >= required_free_disk,
        },
        "warning": (
            "The planning scenario is about nine safeguarded hours; "
            "completion within an eight-hour sleep must not be promised."
        ),
    }
    if not report["per_case_below_session_wall_cap"]:
        raise RuntimeError(f"per-case runtime preflight failed: {report}")
    if not report["storage"]["disk_sufficient"]:
        raise RuntimeError(f"aggregate disk preflight failed: {report}")
    return report


def build_report() -> dict[str, Any]:
    """Compute the complete no-evolution preflight."""

    started = time.perf_counter()
    source = far_runner.verified_source_checkpoint_entry()
    recorded_reference = verified_reference_budget()
    recomputed_reference = recomputed_reference_budget()
    reference_matches = np.isclose(
        recomputed_reference[
            "surface_weighted_mobility_deficit"
        ],
        recorded_reference[
            "surface_weighted_mobility_deficit"
        ],
        rtol=0.0,
        atol=1.0e-10,
    )
    if not reference_matches:
        raise RuntimeError(
            "recorded and independently recomputed c14.5 budgets differ"
        )
    reference = {
        **recomputed_reference,
        "recorded_crosscheck": recorded_reference,
        "recorded_crosscheck_matches": bool(reference_matches),
    }
    cases: list[dict[str, Any]] = []
    for case in protocol.CASES:
        factor, report = derive_case_factor(
            case,
            reference_budget=reference[
                "surface_weighted_mobility_deficit"
            ],
        )
        cases.append(report)
        del factor
        gc.collect()
    resources = runtime_and_storage_preflight()
    checks = {
        "contract_exists": protocol.CONTRACT_PATH.is_file(),
        "exact_ordered_centers": (
            tuple(item["center"] for item in cases)
            == protocol.NEW_CENTERS
        ),
        "unique_centers": (
            len({item["center"] for item in cases}) == len(cases)
        ),
        "all_case_checks_pass": all(
            all(item["checks"].values()) for item in cases
        ),
        "all_budgets_within_two_percent": all(
            abs(item["relative_budget_vs_thinning_zone"])
            <= protocol.MAXIMUM_BUDGET_DIFFERENCE
            for item in cases
        ),
        "disk_sufficient": resources["storage"]["disk_sufficient"],
        "per_case_runtime_bounded": (
            resources["per_case_below_session_wall_cap"]
        ),
        "no_simulation_authorized_or_performed": True,
    }
    passed = all(checks.values())
    if not passed:
        raise RuntimeError(f"position-scan preflight failed: {checks}")
    return {
        "schema_version": 1,
        "status": "completed",
        "classification": "position_scan_preflight_passed",
        "passed": passed,
        "checks": checks,
        "source_checkpoint": source,
        "reference_budget": reference,
        "cases": cases,
        "resources": resources,
        "wall_seconds": time.perf_counter() - started,
        "provenance": {
            "preflight_runner_sha256": sha256_path(SOURCE_PATH),
            "scientific_contract_sha256": sha256_path(
                protocol.CONTRACT_PATH
            ),
            "protocol_sha256": sha256_path(
                Path(protocol.__file__).resolve()
            ),
            "geometry_sha256": sha256_path(
                Path(selective_geometry.__file__).resolve()
            ),
            "model_sha256": sha256_path(
                Path(selective_model.__file__).resolve()
            ),
            "roy_model_sha256": sha256_path(
                Path(roy_model.__file__).resolve()
            ),
            "python": sys.version,
            "numpy": np.__version__,
        },
        "scope": {
            "static_preflight_only": True,
            "simulation_steps_performed": 0,
            "position_scan_started": False,
            "adaptive_refinement_authorized": False,
            "seed_ensemble_authorized": False,
            "convergence_family_authorized": False,
        },
    }


def write_report(output: Path) -> dict[str, Any]:
    reserve_output_directory(output)
    report = build_report()
    atomic_json(output / "summary.json", report)
    return report


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parse_arguments(argv)
    report = write_report(DEFAULT_OUTPUT.resolve())
    print(
        json.dumps(
            {
                "status": report["status"],
                "classification": report["classification"],
                "passed": report["passed"],
                "simulation_steps_performed": report["scope"][
                    "simulation_steps_performed"
                ],
                "output": str(DEFAULT_OUTPUT.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
