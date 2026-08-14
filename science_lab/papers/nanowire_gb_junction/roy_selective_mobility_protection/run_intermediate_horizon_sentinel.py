#!/usr/bin/env python3
"""Run one restartable inward-collar trajectory from global time 100 to 600."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import signal
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from ..roy_2021_reproduction import model as roy_model
from ..roy_2021_reproduction.model import FROZEN_DEG90
from ..roy_2021_reproduction.run_preflight import contact_metrics
from ..roy_fixed_gb_bridge import (
    run_t100_checkpoint_source_localization_preflight as source_helper,
)
from ..roy_fixed_gb_bridge.metrics import array_fingerprint
from ..roy_fixed_gb_bridge.provenance import sha256_path
from ..roy_gb_junction_sentinel import (
    run_t100_conditioned_production as production_helper,
)
from ..roy_gb_junction_sentinel.diagnostics import instantaneous_pinches
from ..roy_gb_junction_sentinel.storage import (
    acquire_output_lock,
    atomic_json,
    atomic_npz,
    load_checkpoint,
    release_output_lock,
    reserve_output_directory,
    write_checkpoint,
)
from . import analysis as selective_analysis
from . import geometry as selective_geometry
from . import model as selective_model
from .analysis import (
    INTERFACE_WIDTH,
    MINIMUM_SEARCH_DISTANCE,
    crossed_contour_radius_profiles,
    phase_excess_radius_profiles,
)
from .geometry import four_arm_tubular_collar_factor
from .model import SpatialMobilityRoySolver
from .run_t100_tubular_position_diagnostic import (
    INWARD_GEOMETRY,
    _factor_checks,
    _treatment_profile_response,
)


SOURCE_PATH = Path(__file__).resolve()
DIRECTORY = SOURCE_PATH.parent
CONTRACT_PATH = DIRECTORY / "INTERMEDIATE_HORIZON_SENTINEL.md"
POSITION_RUNNER_PATH = (
    DIRECTORY / "run_t100_tubular_position_diagnostic.py"
)
POSITION_SUMMARY_PATH = (
    DIRECTORY
    / "results"
    / "t100_tubular_position_diagnostic_v1"
    / "summary.json"
)
EFFECTIVE_RUNNER_PATH = DIRECTORY / "run_effective_time_diagnostic.py"
EFFECTIVE_SUMMARY_PATH = (
    DIRECTORY / "results" / "effective_time_diagnostic_v1" / "summary.json"
)
ROY_RESULTS = (
    DIRECTORY.parent
    / "roy_2021_reproduction"
    / "results"
    / "t1000_source_semantic_seed2292"
)
DEFAULT_OUTPUT = (
    DIRECTORY / "results" / "inward_intermediate_t600_v1"
)

START_STEP = 100
TARGET_STEP = 600
DIAGNOSTIC_INTERVAL = 10
ENERGY_INTERVAL = 50
MILESTONE_INTERVAL = 100
CHECKPOINT_STEPS = (400, 600)
FFT_WORKERS = 12
MAXIMUM_WALL_SECONDS = 45.0 * 60.0
MASS_DRIFT_LIMIT = 1.0e-4
ENERGY_REBOUND_LIMIT = 1.0e-6
FIELD_MAGNITUDE_LIMIT = 10.0
MINIMUM_DISK_HEADROOM_BYTES = 2 << 30
MAXIMUM_EVENT_CHECKPOINTS = 3
CHECKPOINT_STREAM = "inward-intermediate"
LEVELS = ("0.45", "0.50", "0.55")
WIRES = ("first_wire_z", "second_wire_y")
RUNTIME_SAFETY_FACTOR = 1.3


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_summary_artifact(
    summary_path: Path,
    *,
    expected_classification: str,
) -> dict[str, Any]:
    summary = _load_json(summary_path)
    checks = {
        "completed": summary.get("status") == "completed",
        "classification": (
            summary.get("classification") == expected_classification
        ),
        "long_run_not_launched": (
            summary.get("scope", {}).get("long_run_launched") is False
        ),
        "production_not_authorized": (
            summary.get("scope", {}).get("production_authorized") is False
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"invalid prerequisite {summary_path}: {checks}"
        )
    return {
        "path": str(summary_path.resolve()),
        "sha256": sha256_path(summary_path),
        "checks": checks,
    }


def verify_prerequisites() -> dict[str, Any]:
    position = _verify_summary_artifact(
        POSITION_SUMMARY_PATH,
        expected_classification=(
            "tubular_position_short_no_consistent_sign_reversal"
        ),
    )
    position_summary = _load_json(POSITION_SUMMARY_PATH)
    position_checks = {
        "numerical_preflight_passed": (
            position_summary.get("numerical_preflight_passed") is True
        ),
        "sixteen_steps_per_case": (
            position_summary.get("scope", {}).get(
                "accepted_steps_per_case"
            )
            == 16
        ),
        "position_runner_unchanged": (
            position_summary.get("implementation_sha256", {}).get(
                "runner"
            )
            == sha256_path(POSITION_RUNNER_PATH)
        ),
        "selective_model_unchanged": (
            position_summary.get("implementation_sha256", {}).get(
                "selective_model"
            )
            == sha256_path(Path(selective_model.__file__).resolve())
        ),
        "selective_geometry_unchanged": (
            position_summary.get("implementation_sha256", {}).get(
                "selective_geometry"
            )
            == sha256_path(Path(selective_geometry.__file__).resolve())
        ),
    }
    if not all(position_checks.values()):
        raise RuntimeError(
            f"position prerequisite changed: {position_checks}"
        )

    effective = _verify_summary_artifact(
        EFFECTIVE_SUMMARY_PATH,
        expected_classification=(
            "effective_time_diagnostic_completed_without_automatic_gate"
        ),
    )
    effective_summary = _load_json(EFFECTIVE_SUMMARY_PATH)
    effective_checks = {
        "integrity_passed": (
            effective_summary.get("all_integrity_checks_passed") is True
        ),
        "sixteen_untreated_steps": (
            effective_summary.get("scope", {}).get(
                "untreated_accepted_steps"
            )
            == 16
        ),
        "effective_runner_unchanged": (
            effective_summary.get("implementation_sha256", {}).get(
                "runner"
            )
            == sha256_path(EFFECTIVE_RUNNER_PATH)
        ),
    }
    if not all(effective_checks.values()):
        raise RuntimeError(
            f"effective-time prerequisite changed: {effective_checks}"
        )
    source = source_helper.verify_sources()
    if source.get("verified") is not True:
        raise RuntimeError("t=100 source provenance did not verify")
    return {
        "position": {**position, "additional_checks": position_checks},
        "effective_time": {
            **effective,
            "additional_checks": effective_checks,
        },
        "source_verified": True,
    }


def _measured_runtime() -> dict[str, float]:
    position = _load_json(POSITION_SUMMARY_PATH)
    effective = _load_json(EFFECTIVE_SUMMARY_PATH)
    medians = (
        float(position["execution"]["median_proposal_seconds"]),
        float(effective["execution"]["median_proposal_seconds"]),
    )
    conservative_median = max(medians)
    projected = (
        (TARGET_STEP - START_STEP)
        * conservative_median
        * RUNTIME_SAFETY_FACTOR
    )
    return {
        "position_median_proposal_seconds": medians[0],
        "effective_time_median_proposal_seconds": medians[1],
        "conservative_median_proposal_seconds": conservative_median,
        "safety_factor": RUNTIME_SAFETY_FACTOR,
        "projected_wall_seconds": projected,
        "projected_wall_hours": projected / 3600.0,
    }


def resource_preflight(output: Path, *, current_step: int) -> dict[str, Any]:
    field_bytes = int(FROZEN_DEG90.lattice.cell_count * 8 + 256)
    remaining_regular = sum(
        step > current_step for step in CHECKPOINT_STEPS
    )
    required_disk = (
        (
            remaining_regular
            + MAXIMUM_EVENT_CHECKPOINTS
            + 1
        )
        * field_bytes
        + MINIMUM_DISK_HEADROOM_BYTES
    )
    free_disk = int(shutil.disk_usage(output.parent).free)
    runtime = _measured_runtime()
    report = {
        "field_bytes": field_bytes,
        "remaining_regular_checkpoints": remaining_regular,
        "maximum_event_checkpoints": MAXIMUM_EVENT_CHECKPOINTS,
        "emergency_checkpoint_allowance": 1,
        "minimum_disk_headroom_bytes": MINIMUM_DISK_HEADROOM_BYTES,
        "required_free_disk_bytes": required_disk,
        "free_disk_bytes": free_disk,
        "disk_sufficient": free_disk >= required_disk,
        "runtime": runtime,
        "runtime_below_one_hour": (
            runtime["projected_wall_hours"] < 1.0
        ),
    }
    if not report["disk_sufficient"]:
        raise RuntimeError(f"disk preflight failed: {report}")
    if not report["runtime_below_one_hour"]:
        raise RuntimeError(f"runtime preflight exceeded one hour: {report}")
    return report


def _contract(prerequisites: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": "inward_tubular_collar_intermediate_horizon",
        "start_step": START_STEP,
        "target_step": TARGET_STEP,
        "diagnostic_interval": DIAGNOSTIC_INTERVAL,
        "energy_interval": ENERGY_INTERVAL,
        "milestone_interval": MILESTONE_INTERVAL,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "fft_workers": FFT_WORKERS,
        "maximum_wall_seconds": MAXIMUM_WALL_SECONDS,
        "stop_on_robust_event": True,
        "collar": INWARD_GEOMETRY.to_dict(),
        "radial_shell_support": [
            6.0 - 2.0 * INTERFACE_WIDTH,
            6.0 + 2.0 * INTERFACE_WIDTH,
        ],
        "radial_shell_plateau": [
            6.0 - INTERFACE_WIDTH,
            6.0 + INTERFACE_WIDTH,
        ],
        "source_checkpoint": {
            "path": str(source_helper.T100_CHECKPOINT_PATH.resolve()),
            "sha256": sha256_path(source_helper.T100_CHECKPOINT_PATH),
            "fingerprint": source_helper.EXPECTED_T100_FINGERPRINT,
            "global_step": START_STEP,
        },
        "prerequisites": prerequisites,
        "provenance": {
            "runner_sha256": sha256_path(SOURCE_PATH),
            "scientific_contract_sha256": sha256_path(CONTRACT_PATH),
            "selective_model_sha256": sha256_path(
                Path(selective_model.__file__).resolve()
            ),
            "selective_geometry_sha256": sha256_path(
                Path(selective_geometry.__file__).resolve()
            ),
            "selective_analysis_sha256": sha256_path(
                Path(selective_analysis.__file__).resolve()
            ),
            "roy_model_sha256": sha256_path(
                Path(roy_model.__file__).resolve()
            ),
            "diagnostic_helper_sha256": sha256_path(
                Path(production_helper.__file__).resolve()
            ),
            "python": sys.version,
            "numpy": np.__version__,
        },
        "scope": {
            "one_treated_case": True,
            "untreated_simulation_reused": True,
            "parameter_sweep_authorized": False,
            "automatic_extension_authorized": False,
        },
    }


def _verify_resume_contract(
    stored: dict[str, Any],
    current: dict[str, Any],
) -> None:
    if stored != current:
        raise RuntimeError("resume contract differs from launch contract")


def _load_source_field() -> np.ndarray:
    mapped = np.load(
        source_helper.T100_CHECKPOINT_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )
    if array_fingerprint(mapped) != source_helper.EXPECTED_T100_FINGERPRINT:
        raise RuntimeError("immutable t=100 checkpoint fingerprint mismatch")
    return np.array(mapped, dtype=np.float64, copy=True, order="C")


def _make_solver() -> tuple[SpatialMobilityRoySolver, dict[str, Any]]:
    factor = four_arm_tubular_collar_factor(
        FROZEN_DEG90.lattice,
        INWARD_GEOMETRY,
        radius=6.0,
        interface_width=INTERFACE_WIDTH,
    )
    checks = _factor_checks(factor, INWARD_GEOMETRY)
    if not all(checks.values()):
        raise RuntimeError(f"inward factor checks failed: {checks}")
    solver = SpatialMobilityRoySolver(
        FROZEN_DEG90.lattice,
        FROZEN_DEG90.parameters,
        factor,
        fft_workers=FFT_WORKERS,
    )
    report = {
        "minimum": float(np.min(factor)),
        "maximum": float(np.max(factor)),
        "support_cell_fraction": float(
            np.mean(factor < 1.0 - 1.0e-14)
        ),
        "checks": checks,
    }
    del factor
    gc.collect()
    return solver, report


def _untreated_checkpoint(step: int) -> tuple[np.ndarray, dict[str, Any]]:
    metadata_path = ROY_RESULTS / f"checkpoint-step-{step:04d}.json"
    metadata = _load_json(metadata_path)
    field_path = Path(metadata["field_path"])
    checks = {
        "step": int(metadata["step"]) == step,
        "metadata_sha256_recorded": True,
        "field_sha256": sha256_path(field_path) == metadata["field_sha256"],
        "shape": list(metadata["shape"]) == list(FROZEN_DEG90.lattice.shape),
        "dtype": metadata["dtype"] == np.dtype(np.float64).str,
        "finite_recorded": metadata["finite"] is True,
    }
    field = np.load(field_path, mmap_mode="r", allow_pickle=False)
    checks["fingerprint"] = (
        array_fingerprint(field) == metadata["field_fingerprint"]
    )
    checks["finite"] = bool(np.isfinite(field).all())
    if not all(checks.values()):
        raise RuntimeError(
            f"untreated checkpoint {step} failed: {checks}"
        )
    return field, {
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": sha256_path(metadata_path),
        "field_path": str(field_path.resolve()),
        "field_sha256": metadata["field_sha256"],
        "field_fingerprint": metadata["field_fingerprint"],
        "checks": checks,
    }


def _profile_arrays(
    phase: dict[str, Any],
    contour: dict[str, Any],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for wire in WIRES:
        arrays[f"{wire}_coordinate"] = phase["treated"][wire][
            "coordinate"
        ]
        for case in ("treated", "untreated"):
            arrays[f"{wire}_{case}_phase_radius"] = phase[case][wire][
                "radius"
            ]
            for level in LEVELS:
                arrays[
                    f"{wire}_{case}_contour_radius_{level}"
                ] = contour[case][wire]["levels"][level]["mean_radius"]
                arrays[
                    f"{wire}_{case}_contour_valid_{level}"
                ] = contour[case][wire]["levels"][level][
                    "valid_fraction"
                ]
    return arrays


def _site_region(distance: float) -> str:
    if MINIMUM_SEARCH_DISTANCE <= distance < 16.5:
        return "inherited_thinning"
    if 16.5 <= distance < 17.5:
        return "protected_outer_plateau"
    if 17.5 <= distance < 20.5:
        return "outer_transition"
    if 20.5 <= distance <= 60.0:
        return "near_far_arm"
    return "outside_clean_analysis"


def _minimum_sites(contour: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    sigma = INTERFACE_WIDTH / FROZEN_DEG90.lattice.spacing
    for case in ("treated", "untreated"):
        output[case] = {}
        for wire in WIRES:
            coordinate = contour[case][wire]["coordinate"]
            clean = (
                (np.abs(coordinate) >= MINIMUM_SEARCH_DISTANCE)
                & (np.abs(coordinate) <= 60.0)
            )
            output[case][wire] = {}
            for level in LEVELS:
                radius = ndimage.gaussian_filter1d(
                    contour[case][wire]["levels"][level]["mean_radius"],
                    sigma=sigma,
                    mode="wrap",
                )
                valid = (
                    contour[case][wire]["levels"][level][
                        "valid_fraction"
                    ]
                    >= 0.95
                )
                sides: dict[str, Any] = {}
                for side_name, side in (
                    ("negative", coordinate < 0.0),
                    ("positive", coordinate > 0.0),
                ):
                    mask = clean & valid & side
                    indices = np.flatnonzero(mask)
                    local = int(np.argmin(radius[mask]))
                    index = int(indices[local])
                    distance = abs(float(coordinate[index]))
                    sides[side_name] = {
                        "coordinate": float(coordinate[index]),
                        "distance": distance,
                        "radius": float(radius[index]),
                        "region": _site_region(distance),
                    }
                output[case][wire][level] = sides
    return output


def _milestone_comparison(
    output: Path,
    field: np.ndarray,
    *,
    step: int,
) -> dict[str, Any]:
    untreated, untreated_source = _untreated_checkpoint(step)
    phase = {
        "treated": phase_excess_radius_profiles(field),
        "untreated": phase_excess_radius_profiles(untreated),
    }
    contour = {
        "treated": crossed_contour_radius_profiles(field),
        "untreated": crossed_contour_radius_profiles(untreated),
    }
    response = _treatment_profile_response(
        phase,
        contour,
        treated_name="treated",
        geometry=INWARD_GEOMETRY,
    )
    arrays = _profile_arrays(phase, contour)
    artifact_path = output / f"profiles-step-{step:04d}.npz"
    atomic_npz(artifact_path, **arrays)
    report = {
        "step": step,
        "untreated_source": untreated_source,
        "response": response,
        "minimum_sites": _minimum_sites(contour),
        "contact": {
            "treated": contact_metrics(field, FROZEN_DEG90),
            "untreated": contact_metrics(untreated, FROZEN_DEG90),
        },
        "pinches": {
            "treated": instantaneous_pinches(
                field,
                geometry="crossed",
                spacing=FROZEN_DEG90.lattice.spacing,
                radius=6.0,
            ),
            "untreated": instantaneous_pinches(
                untreated,
                geometry="crossed",
                spacing=FROZEN_DEG90.lattice.spacing,
                radius=6.0,
            ),
        },
        "artifact": {
            "path": str(artifact_path.resolve()),
            "sha256": sha256_path(artifact_path),
            "bytes": int(artifact_path.stat().st_size),
        },
    }
    del untreated, phase, contour, arrays
    gc.collect()
    return report


def _terminal_summary(
    output: Path,
    contract: dict[str, Any],
    status: dict[str, Any],
) -> dict[str, Any]:
    terminal = status["status"]
    event = status["event_assessment"]
    if terminal == "completed" and event.get("detected") is True:
        classification = (
            "inward_intermediate_robust_first_break_completed"
        )
    elif terminal == "completed":
        classification = (
            "inward_intermediate_t600_completed_no_robust_break"
        )
    else:
        classification = f"inward_intermediate_{terminal}"
    times = status["step_wall_seconds"]
    summary = {
        "status": terminal,
        "classification": classification,
        "stop_reason": status["stop_reason"],
        "completed_step": status["current_step"],
        "event_assessment": event,
        "milestones": status["milestones"],
        "records": status["records"],
        "checkpoints": status["checkpoints"],
        "execution": {
            "elapsed_wall_seconds": status["elapsed_wall_seconds"],
            "accepted_steps": status["current_step"] - START_STEP,
            "step_wall_seconds_count": len(times),
            "median_step_wall_seconds": (
                float(statistics.median(times)) if times else None
            ),
            "maximum_step_wall_seconds": max(times) if times else None,
        },
        "contract": contract,
        "scope": {
            "one_treated_case_completed": True,
            "automatic_extension_performed": False,
            "scientific_interpretation_pending": True,
        },
    }
    atomic_json(output / "summary.json", summary)
    return summary


def run(
    output: Path,
    *,
    resume: bool,
    stop_requested: dict[str, int | None],
) -> dict[str, Any]:
    prerequisites = verify_prerequisites()
    current_contract = _contract(prerequisites)
    if resume:
        if not output.is_dir():
            raise FileNotFoundError("resume output directory is absent")
    else:
        reserve_output_directory(output)
    lock = acquire_output_lock(output)
    try:
        if resume:
            stored_contract = _load_json(output / "contract.json")
            _verify_resume_contract(stored_contract, current_contract)
            status = _load_json(output / "run_status.json")
            if status["status"] not in {
                "running",
                "interrupted",
                "wall_cap",
            }:
                raise RuntimeError("run status is not resumable")
            checkpoints = list(status["checkpoints"])
            if checkpoints:
                field = load_checkpoint(checkpoints[-1])
                current_step = int(checkpoints[-1]["step"])
            else:
                field = _load_source_field()
                current_step = START_STEP
            records = [
                item
                for item in status["records"]
                if int(item["step"]) <= current_step
            ]
            milestones = [
                item
                for item in status["milestones"]
                if int(item["step"]) <= current_step
            ]
            step_wall_seconds = list(status["step_wall_seconds"])[
                : current_step - START_STEP
            ]
            event = production_helper._event_from_records(
                records, latched=None
            )
            prior_elapsed = float(
                records[-1]["elapsed_wall_seconds"]
            )
            prior_energy = next(
                (
                    item["free_energy"]
                    for item in reversed(records)
                    if item.get("free_energy") is not None
                ),
                None,
            )
            event_checkpoint_count = sum(
                "event_candidate" in checkpoint.get("kinds", [])
                for checkpoint in checkpoints
                if int(checkpoint["step"]) <= current_step
            )
        else:
            atomic_json(output / "contract.json", current_contract)
            field = _load_source_field()
            current_step = START_STEP
            records: list[dict[str, Any]] = []
            milestones: list[dict[str, Any]] = []
            step_wall_seconds: list[float] = []
            checkpoints: list[dict[str, Any]] = []
            event = {
                "detected": False,
                "latched": False,
                "first_candidate_step": None,
                "confirmation_step": None,
                "event_bracket": None,
                "persistent_gaps": [],
            }
            prior_elapsed = 0.0
            prior_energy = None
            event_checkpoint_count = 0

        resource = resource_preflight(
            output, current_step=current_step
        )
        solver, factor_report = _make_solver()
        initial_mass = float(source_helper.EXPECTED_T100_MASS)
        session_started = time.perf_counter()

        if not records:
            initial = production_helper._diagnostic_record(
                field,
                solver,
                step=START_STEP,
                initial_mass=initial_mass,
                elapsed_wall_seconds=prior_elapsed,
                include_energy=True,
            )
            records.append(initial)
            prior_energy = float(initial["free_energy"])
            event = production_helper._event_from_records(
                records, latched=event
            )
            production_helper._write_status(
                output,
                {
                    "status": "running",
                    "current_step": START_STEP,
                    "target_step": TARGET_STEP,
                    "stop_reason": None,
                    "elapsed_wall_seconds": prior_elapsed,
                    "records": records,
                    "milestones": milestones,
                    "step_wall_seconds": step_wall_seconds,
                    "checkpoints": checkpoints,
                    "event_assessment": event,
                    "event_checkpoint_count": event_checkpoint_count,
                    "last_energy": prior_energy,
                    "resource_preflight": resource,
                    "factor": factor_report,
                },
            )

        terminal_status: str | None = None
        stop_reason: str | None = None
        for step in range(current_step + 1, TARGET_STEP + 1):
            step_started = time.perf_counter()
            field = solver.propose_step(field)
            step_wall_seconds.append(
                float(time.perf_counter() - step_started)
            )
            current_step = step
            if not np.isfinite(field).all():
                terminal_status = "numerically_aborted"
                stop_reason = "nonfinite_field"

            candidate_now = False
            if (
                step % DIAGNOSTIC_INTERVAL == 0
                or terminal_status is not None
            ):
                elapsed = (
                    prior_elapsed
                    + time.perf_counter()
                    - session_started
                )
                include_energy = bool(
                    step % ENERGY_INTERVAL == 0
                    or terminal_status is not None
                )
                record = production_helper._diagnostic_record(
                    field,
                    solver,
                    step=step,
                    initial_mass=initial_mass,
                    elapsed_wall_seconds=elapsed,
                    include_energy=include_energy,
                )
                records.append(record)
                health_reason = production_helper._health_stop_reason(
                    record, previous_energy=prior_energy
                )
                if record["free_energy"] is not None:
                    prior_energy = float(record["free_energy"])
                if health_reason is not None:
                    terminal_status = "numerically_aborted"
                    stop_reason = health_reason
                event = production_helper._event_from_records(
                    records, latched=event
                )
                candidate_now = bool(record["pinches"]["candidate"])
                if event.get("detected") is True:
                    terminal_status = "completed"
                    stop_reason = "robust_single_arm_event_confirmed"

            if (
                terminal_status is None
                and step % MILESTONE_INTERVAL == 0
            ):
                milestones.append(
                    _milestone_comparison(output, field, step=step)
                )

            elapsed = (
                prior_elapsed + time.perf_counter() - session_started
            )
            signal_pending = stop_requested["signal"] is not None
            wall_cap_reached = elapsed >= MAXIMUM_WALL_SECONDS
            event_checkpoint_due = bool(
                candidate_now
                and event_checkpoint_count
                < MAXIMUM_EVENT_CHECKPOINTS
            )
            regular_checkpoint_due = step in CHECKPOINT_STEPS
            checkpoint_due = bool(
                regular_checkpoint_due
                or event_checkpoint_due
                or signal_pending
                or wall_cap_reached
                or terminal_status is not None
                or step == TARGET_STEP
            )
            if checkpoint_due:
                kinds: list[str] = []
                if regular_checkpoint_due:
                    kinds.append("regular")
                if event_checkpoint_due:
                    kinds.append("event_candidate")
                if event.get("detected") is True:
                    kinds.append("event_confirmation")
                if signal_pending:
                    kinds.append("signal")
                if wall_cap_reached:
                    kinds.append("wall_cap")
                if terminal_status == "numerically_aborted":
                    kinds.append("numerical_abort")
                if step == TARGET_STEP:
                    kinds.append("target")
                checkpoint = write_checkpoint(
                    output,
                    field,
                    step=step,
                    kinds=tuple(kinds),
                    stream=CHECKPOINT_STREAM,
                )
                checkpoints.append(checkpoint)
                if event_checkpoint_due:
                    event_checkpoint_count += 1

            if step % DIAGNOSTIC_INTERVAL == 0 or checkpoint_due:
                production_helper._write_status(
                    output,
                    {
                        "status": "running",
                        "current_step": step,
                        "target_step": TARGET_STEP,
                        "stop_reason": stop_reason,
                        "elapsed_wall_seconds": elapsed,
                        "records": records,
                        "milestones": milestones,
                        "step_wall_seconds": step_wall_seconds,
                        "checkpoints": checkpoints,
                        "event_assessment": event,
                        "event_checkpoint_count": (
                            event_checkpoint_count
                        ),
                        "last_energy": prior_energy,
                        "resource_preflight": resource,
                        "factor": factor_report,
                    },
                )
                print(
                    json.dumps(
                        {
                            "status": "running",
                            "step": step,
                            "elapsed_wall_seconds": elapsed,
                            "milestone_count": len(milestones),
                            "checkpoint_count": len(checkpoints),
                            "event_detected": event.get("detected"),
                            "stop_reason": stop_reason,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            if terminal_status is not None:
                break
            if signal_pending:
                terminal_status = "interrupted"
                stop_reason = (
                    f"deferred_signal_{stop_requested['signal']}"
                )
                break
            if wall_cap_reached:
                terminal_status = "wall_cap"
                stop_reason = "forty_five_minute_wall_cap_reached"
                break
            if step == TARGET_STEP:
                terminal_status = "completed"
                stop_reason = "fixed_t600_target_reached"
                break

        elapsed_total = (
            prior_elapsed + time.perf_counter() - session_started
        )
        final_status = {
            "status": terminal_status,
            "current_step": current_step,
            "target_step": TARGET_STEP,
            "stop_reason": stop_reason,
            "elapsed_wall_seconds": elapsed_total,
            "records": records,
            "milestones": milestones,
            "step_wall_seconds": step_wall_seconds,
            "checkpoints": checkpoints,
            "event_assessment": event,
            "event_checkpoint_count": event_checkpoint_count,
            "last_energy": prior_energy,
            "resource_preflight": resource,
            "factor": factor_report,
        }
        production_helper._write_status(output, final_status)
        return _terminal_summary(output, current_contract, final_status)
    finally:
        release_output_lock(lock)


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--acknowledge-bounded-runtime",
        action="store_true",
        help="Required for a fresh 32-to-35-minute launch.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    if (
        arguments.launch
        and not arguments.acknowledge_bounded_runtime
    ):
        raise RuntimeError(
            "fresh launch requires --acknowledge-bounded-runtime"
        )
    output = (
        arguments.output
        if arguments.output is not None
        else DEFAULT_OUTPUT
    ).resolve()
    stop_requested: dict[str, int | None] = {"signal": None}

    def request_stop(signal_number: int, _frame: object) -> None:
        stop_requested["signal"] = signal_number

    handled = (signal.SIGINT, signal.SIGTERM)
    prior_handlers = {
        number: signal.getsignal(number) for number in handled
    }
    for number in handled:
        signal.signal(number, request_stop)
    try:
        result = run(
            output,
            resume=arguments.resume,
            stop_requested=stop_requested,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "classification": result["classification"],
                    "completed_step": result["completed_step"],
                    "output": str(output),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if result["status"] == "completed" else 2
    except Exception as error:
        if output.is_dir():
            atomic_json(
                output / "failure.json",
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "pid": os.getpid(),
                    "unix_time": time.time(),
                },
            )
        raise
    finally:
        for number, handler in prior_handlers.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    raise SystemExit(main())
