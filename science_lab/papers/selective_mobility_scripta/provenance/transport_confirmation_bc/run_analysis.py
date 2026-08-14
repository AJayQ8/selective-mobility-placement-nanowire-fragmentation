#!/usr/bin/env python3
"""Confirm the source-A transport signature in archived sources B and C.

The scientific choices and all input hashes live in ``frozen_contract.json``.
This program performs no time integration.  It reads archived t=1200/1300/
1400 profiles and t=1300 fields, applies the source-A control-volume analysis
unchanged, and writes compact JSON/CSV evidence.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from science_lab.papers.nanowire_gb_junction.roy_2021_reproduction.model import (
    FROZEN_DEG90,
    RoyPseudospectralSolver,
)
from science_lab.papers.nanowire_gb_junction.roy_selective_mobility_protection import (
    analyze_position_scan_mechanism as profile_audit,
)
from science_lab.papers.nanowire_gb_junction.roy_selective_mobility_protection import (
    analyze_position_scan_transport_v3 as source_a_transport,
)
from science_lab.papers.nanowire_gb_junction.roy_selective_mobility_protection import (
    position_scan_protocol,
)
from science_lab.papers.nanowire_gb_junction.roy_selective_mobility_protection.geometry import (
    four_arm_tubular_collar_factor,
)
from science_lab.papers.nanowire_gb_junction.roy_selective_mobility_protection.model import (
    SpatialMobilityRoySolver,
)


PACKAGE_DIR = Path(__file__).resolve().parent
CONTRACT_PATH = PACKAGE_DIR / "frozen_contract.json"
DEFAULT_PREFLIGHT = PACKAGE_DIR / "preflight.json"
DEFAULT_OUTPUT = PACKAGE_DIR / "results" / "bc_transport_confirmation_v1"
SEEDS = (104729, 130363)
CASES = ("untreated", "c26p5", "c34p5")
PROFILE_STEPS = (1200, 1300, 1400)
FIELD_STEP = 1300
NATURAL_BOUNDS = (14.0, 22.5)
CONTINUITY_LIMIT = 0.01
WIRES = source_a_transport.WIRES
BRANCHES = source_a_transport.BRANCHES


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def case_directory(root: Path, seed: int, case_id: str) -> Path:
    return root / f"seed-{seed}" / f"case-{case_id}"


def artifact_paths(root: Path, seed: int, case_id: str) -> dict[str, Path]:
    directory = case_directory(root, seed, case_id)
    return {
        "case_contract_sha256": directory / "contract.json",
        "case_summary_sha256": directory / "summary.json",
        "checkpoint_metadata_sha256": (
            directory / "checkpoint-position_scan-step-1300.json"
        ),
        "checkpoint_field_sha256": (
            directory / "checkpoint-position_scan-step-1300.npy"
        ),
        "profile_1200_sha256": directory / "profiles-step-1200.npz",
        "profile_1300_sha256": directory / "profiles-step-1300.npz",
        "profile_1400_sha256": directory / "profiles-step-1400.npz",
    }


def validate_inputs(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    artifact_count = 0
    for seed in SEEDS:
        seed_key = str(seed)
        for case_id in CASES:
            expected = contract["bound_inputs"][seed_key][case_id]
            paths = artifact_paths(root, seed, case_id)
            artifact_checks: dict[str, bool] = {}
            for hash_key, path in paths.items():
                artifact_count += 1
                artifact_checks[f"{hash_key}_exists"] = path.is_file()
                artifact_checks[f"{hash_key}_matches"] = (
                    path.is_file()
                    and sha256_path(path) == expected[hash_key]
                )

            metadata_path = paths["checkpoint_metadata_sha256"]
            metadata = load_json(metadata_path)
            metadata_checks = {
                "step_is_1300": int(metadata["step"]) == FIELD_STEP,
                "shape_matches": metadata["shape"] == [96, 768, 768],
                "dtype_matches": metadata["dtype"] == "<f8",
                "bytes_match": int(metadata["field_bytes"]) == 452984960,
                "recorded_finite": metadata["finite"] is True,
                "recorded_field_hash_matches_contract": (
                    metadata["field_sha256"]
                    == expected["checkpoint_field_sha256"]
                ),
            }
            case_contract = load_json(paths["case_contract_sha256"])
            summary = load_json(paths["case_summary_sha256"])
            metadata_checks.update(
                {
                    "case_contract_seed_matches": (
                        int(case_contract["seed"]) == seed
                    ),
                    "case_contract_case_matches": (
                        case_contract["case"] == case_id
                    ),
                    "case_summary_completed": summary["status"] == "completed",
                    "summary_reaches_analysis_window": (
                        int(summary["completed_step"]) >= 1400
                    ),
                }
            )

            profile_checks: dict[str, bool] = {}
            required_profile_keys = {
                f"{wire}_{suffix}"
                for wire in WIRES
                for suffix in ("coordinate", "phase_radius")
            }
            for step in PROFILE_STEPS:
                path = paths[f"profile_{step}_sha256"]
                with np.load(path, allow_pickle=False) as arrays:
                    profile_checks[f"step_{step}_keys_present"] = (
                        required_profile_keys.issubset(arrays.files)
                    )
                    profile_checks[f"step_{step}_shapes_match"] = all(
                        np.asarray(arrays[key]).shape == (768,)
                        for key in required_profile_keys
                    )
                    profile_checks[f"step_{step}_finite"] = all(
                        bool(np.all(np.isfinite(arrays[key])))
                        for key in required_profile_keys
                    )

            case_checks = {
                **artifact_checks,
                **metadata_checks,
                **profile_checks,
            }
            checks[f"{seed}/{case_id}"] = {
                "passed": all(case_checks.values()),
                "checks": case_checks,
            }

    all_passed = all(item["passed"] for item in checks.values())
    return {
        "all_passed": all_passed,
        "artifact_count": artifact_count,
        "cases": checks,
    }


def build_preflight(root: Path) -> dict[str, Any]:
    contract = load_json(CONTRACT_PATH)
    validation = validate_inputs(root, contract)
    report = {
        "schema_version": 1,
        "status": "preflight_passed" if validation["all_passed"] else "preflight_failed",
        "analysis_steps_performed": 0,
        "operator_evaluations_performed": 0,
        "new_solver_steps": 0,
        "contract_sha256": sha256_path(CONTRACT_PATH),
        "runner_sha256": sha256_path(Path(__file__).resolve()),
        "input_root_recorded": False,
        "input_validation": validation,
        "resource_bound": {
            "field_shape": [96, 768, 768],
            "field_bytes": 452984960,
            "cases_processed_sequentially": 6,
            "estimated_peak_ram_gib_upper_bound": 16.0,
            "expected_persistent_output_mib_upper_bound": 2.0,
            "long_run_gate_required": False,
            "reason": "six operator evaluations, no time integration, compact output",
        },
    }
    return report


def phase_volume(profile_path: Path, wire: str, branch: str) -> float:
    with np.load(profile_path, allow_pickle=False) as arrays:
        coordinate = np.asarray(arrays[f"{wire}_coordinate"], dtype=np.float64)
        radius = np.asarray(arrays[f"{wire}_phase_radius"], dtype=np.float64)
    distance, radial = profile_audit._branch_view(coordinate, radius, branch)
    return float(
        profile_audit._trapz_in_bounds(
            distance,
            np.pi * radial**2,
            NATURAL_BOUNDS,
        )
    )


def observed_rate(root: Path, seed: int, case_id: str, wire: str, branch: str) -> float:
    directory = case_directory(root, seed, case_id)
    lower = phase_volume(directory / "profiles-step-1200.npz", wire, branch)
    upper = phase_volume(directory / "profiles-step-1400.npz", wire, branch)
    return float((upper - lower) / 200.0)


def flux_rows(
    root: Path,
    seed: int,
    case_id: str,
    reconstructed: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for wire in WIRES:
        coordinate = reconstructed[wire]["coordinate"]
        physical_axis_current = -reconstructed[wire][
            "operator_flux_M_grad_mu"
        ]
        for branch in BRANCHES:
            sign = 1.0 if branch == "plus" else -1.0
            distance, selected = profile_audit._branch_view(
                coordinate,
                physical_axis_current,
                branch,
            )
            outward_current = sign * selected
            inner = float(np.interp(NATURAL_BOUNDS[0], distance, outward_current))
            outer = float(np.interp(NATURAL_BOUNDS[1], distance, outward_current))
            control = reconstructed[wire]["control_volume"][branch]
            axial = float(control["spectral_axial_volume_rate"])
            transverse = float(control["spectral_transverse_volume_rate"])
            periodic_x = float(control["spectral_periodic_x_volume_rate"])
            predicted = axial + transverse + periodic_x
            observed = observed_rate(root, seed, case_id, wire, branch)
            rows.append(
                {
                    "seed": seed,
                    "source_role": (
                        "confirmation_source_B"
                        if seed == 104729
                        else "confirmation_source_C"
                    ),
                    "case_id": case_id,
                    "step": FIELD_STEP,
                    "wire": wire,
                    "branch": branch,
                    "physical_outward_current_at_inner_boundary": inner,
                    "physical_outward_current_at_outer_boundary": outer,
                    "spectral_axial_control_volume_rate": axial,
                    "spectral_transverse_control_volume_rate": transverse,
                    "spectral_periodic_x_control_volume_rate": periodic_x,
                    "predicted_natural_zone_volume_rate": predicted,
                    "observed_centered_natural_zone_volume_rate": observed,
                    "continuity_sign_agrees": bool(np.sign(predicted) == np.sign(observed)),
                    "relative_magnitude_error": (
                        abs(predicted - observed) / abs(observed)
                        if observed != 0.0
                        else None
                    ),
                }
            )
    return rows


def factor_for_case(case_id: str) -> np.ndarray:
    center = {"c26p5": 26.5, "c34p5": 34.5}[case_id]
    geometry = position_scan_protocol.geometry_for_center(center)
    return four_arm_tubular_collar_factor(
        FROZEN_DEG90.lattice,
        geometry,
        radius=position_scan_protocol.RADIUS,
        interface_width=position_scan_protocol.INTERFACE_WIDTH,
    )


def source_case_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predicted = np.asarray(
        [row["predicted_natural_zone_volume_rate"] for row in rows],
        dtype=np.float64,
    )
    observed = np.asarray(
        [row["observed_centered_natural_zone_volume_rate"] for row in rows],
        dtype=np.float64,
    )
    errors = [row["relative_magnitude_error"] for row in rows]
    return {
        "arm_count": len(rows),
        "mean_predicted_natural_zone_volume_rate": float(np.mean(predicted)),
        "mean_observed_centered_natural_zone_volume_rate": float(np.mean(observed)),
        "maximum_armwise_relative_magnitude_error": float(max(errors)),
        "all_four_continuity_signs_agree": all(
            row["continuity_sign_agrees"] for row in rows
        ),
        "all_four_close_below_one_percent": all(
            error <= CONTINUITY_LIMIT for error in errors
        ),
        "predicted_arm_range": float(np.ptp(predicted)),
        "observed_arm_range": float(np.ptp(observed)),
    }


def evaluate_frozen_checks(aggregates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, Any] = {}
    for seed in SEEDS:
        key = str(seed)
        source = aggregates[key]
        untreated_predicted = source["untreated"][
            "mean_predicted_natural_zone_volume_rate"
        ]
        untreated_observed = source["untreated"][
            "mean_observed_centered_natural_zone_volume_rate"
        ]
        comparisons: dict[str, Any] = {}
        for case_id in ("c26p5", "c34p5"):
            comparisons[case_id] = {
                "predicted_rate_change_from_untreated": (
                    source[case_id]["mean_predicted_natural_zone_volume_rate"]
                    - untreated_predicted
                ),
                "observed_rate_change_from_untreated": (
                    source[case_id]["mean_observed_centered_natural_zone_volume_rate"]
                    - untreated_observed
                ),
            }
        intermediate = comparisons["c26p5"]
        outboard = comparisons["c34p5"]
        continuity_passed = all(
            source[case_id]["all_four_continuity_signs_agree"]
            and source[case_id]["all_four_close_below_one_percent"]
            for case_id in CASES
        )
        intermediate_passed = (
            intermediate["predicted_rate_change_from_untreated"] > 0.0
            and intermediate["observed_rate_change_from_untreated"] > 0.0
        )
        outboard_passed = (
            outboard["predicted_rate_change_from_untreated"] < 0.0
            and outboard["observed_rate_change_from_untreated"] < 0.0
        )
        by_source[key] = {
            "comparisons_to_untreated": comparisons,
            "continuity_passed": continuity_passed,
            "intermediate_direction_passed": intermediate_passed,
            "outboard_direction_passed": outboard_passed,
            "all_frozen_science_checks_passed": (
                continuity_passed and intermediate_passed and outboard_passed
            ),
        }
    return {
        "sources": by_source,
        "all_frozen_science_checks_passed": all(
            item["all_frozen_science_checks_passed"]
            for item in by_source.values()
        ),
    }


def build_report(root: Path, preflight: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.monotonic()
    all_rows: list[dict[str, Any]] = []
    aggregates: dict[str, dict[str, Any]] = {str(seed): {} for seed in SEEDS}

    for case_id in CASES:
        factor = None if case_id == "untreated" else factor_for_case(case_id)
        for seed in SEEDS:
            directory = case_directory(root, seed, case_id)
            field = np.load(
                directory / "checkpoint-position_scan-step-1300.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            if case_id == "untreated":
                solver = RoyPseudospectralSolver(
                    FROZEN_DEG90.lattice,
                    FROZEN_DEG90.parameters,
                    fft_workers=position_scan_protocol.FFT_WORKERS,
                )
            else:
                solver = SpatialMobilityRoySolver(
                    FROZEN_DEG90.lattice,
                    FROZEN_DEG90.parameters,
                    factor,
                    fft_workers=position_scan_protocol.FFT_WORKERS,
                )
            reconstructed = source_a_transport._operator_control_volume_flux(
                solver,
                field,
                NATURAL_BOUNDS,
            )
            rows = flux_rows(root, seed, case_id, reconstructed)
            all_rows.extend(rows)
            aggregates[str(seed)][case_id] = source_case_aggregate(rows)
            del reconstructed, solver, field
            gc.collect()
        del factor
        gc.collect()

    frozen_checks = evaluate_frozen_checks(aggregates)
    passed = frozen_checks["all_frozen_science_checks_passed"]
    report = {
        "schema_version": 1,
        "status": "complete",
        "classification": (
            "bc_archived_transport_confirmation_passed"
            if passed
            else "bc_archived_transport_confirmation_not_confirmed_or_inconclusive"
        ),
        "scope": {
            "new_solver_steps": 0,
            "operator_evaluations": 6,
            "retrospective_archived_trajectory_analysis": True,
            "sources_are_independent_units": True,
            "arms_are_not_replicates": True,
        },
        "analysis_definition": load_json(CONTRACT_PATH)["analysis_definition"],
        "case_aggregate": aggregates,
        "frozen_checks": frozen_checks,
        "arm_rows": all_rows,
        "runtime_seconds": time.monotonic() - started,
        "provenance": {
            "contract_sha256": sha256_path(CONTRACT_PATH),
            "preflight_sha256": sha256_path(DEFAULT_PREFLIGHT),
            "runner_sha256": sha256_path(Path(__file__).resolve()),
            "input_validation": preflight["input_validation"],
            "raw_input_root_recorded": False,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "claim_boundary": [
            "The result tests the previously defined weak-zone transport signature in two archived confirmation sources; it is not a new simulation campaign.",
            "The reconstructed rate is instantaneous at t=1300 and is compared with a centered t=1200--1400 profile slope.",
            "The result can support source-level recurrence of the signed transport response but cannot establish a universal scalar lifetime law.",
        ],
    }
    return report, all_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "seed",
        "source_role",
        "case_id",
        "step",
        "wire",
        "branch",
        "physical_outward_current_at_inner_boundary",
        "physical_outward_current_at_outer_boundary",
        "spectral_axial_control_volume_rate",
        "spectral_transverse_control_volume_rate",
        "spectral_periodic_x_control_volume_rate",
        "predicted_natural_zone_volume_rate",
        "observed_centered_natural_zone_volume_rate",
        "continuity_sign_agrees",
        "relative_magnitude_error",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    root = arguments.historical_root.resolve()
    if arguments.preflight_only:
        report = build_preflight(root)
        atomic_json(arguments.preflight.resolve(), report)
        print(json.dumps({"status": report["status"], "new_solver_steps": 0}, sort_keys=True))
        return 0 if report["status"] == "preflight_passed" else 2

    preflight_path = arguments.preflight.resolve()
    preflight = load_json(preflight_path)
    runner_hash = sha256_path(Path(__file__).resolve())
    if preflight["status"] != "preflight_passed":
        raise RuntimeError("preflight did not pass")
    if preflight["contract_sha256"] != sha256_path(CONTRACT_PATH):
        raise RuntimeError("contract changed after preflight")
    if preflight["runner_sha256"] != runner_hash:
        raise RuntimeError("runner changed after preflight")
    validation = validate_inputs(root, load_json(CONTRACT_PATH))
    if not validation["all_passed"]:
        raise RuntimeError("input validation changed after preflight")

    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    report, rows = build_report(root, preflight)
    atomic_json(output / "summary.json", report)
    write_csv(output / "arm_rates.csv", rows)
    print(
        json.dumps(
            {
                "status": report["status"],
                "classification": report["classification"],
                "new_solver_steps": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
