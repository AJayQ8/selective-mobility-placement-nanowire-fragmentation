#!/usr/bin/env python3
"""Continue the verified inward-collar trajectory from t=600 to event/t=2000."""

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

from ..roy_2021_reproduction import model as roy_model
from ..roy_2021_reproduction.model import FROZEN_DEG90
from ..roy_2021_reproduction.run_preflight import (
    array_fingerprint as roy_array_fingerprint,
    contact_metrics,
)
from ..roy_fixed_gb_bridge import (
    run_t100_checkpoint_source_localization_preflight as source_helper,
)
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
from . import run_intermediate_horizon_sentinel as intermediate
from .analysis import (
    crossed_contour_radius_profiles,
    phase_excess_radius_profiles,
)
from .model import SpatialMobilityRoySolver


SOURCE_PATH = Path(__file__).resolve()
DIRECTORY = SOURCE_PATH.parent
CONTRACT_PATH = DIRECTORY / "FULL_LIFETIME_CONTINUATION.md"
SOURCE_OUTPUT = (
    DIRECTORY / "results" / "inward_intermediate_t600_v1"
)
SOURCE_SUMMARY_PATH = SOURCE_OUTPUT / "summary.json"
SOURCE_CHECKPOINT_METADATA_PATH = (
    SOURCE_OUTPUT
    / "checkpoint-inward-intermediate-step-0600.json"
)
UNTREATED_T1000_OUTPUT = (
    DIRECTORY.parent
    / "roy_2021_reproduction"
    / "results"
    / "t1000_source_semantic_seed2292"
)
UNTREATED_T2000_OUTPUT = (
    DIRECTORY.parent
    / "roy_2021_reproduction"
    / "results"
    / "t2000_continuation_source_semantic_seed2292"
)
DEFAULT_OUTPUT = (
    DIRECTORY / "results" / "inward_full_lifetime_t2000_v1"
)

START_STEP = 600
TARGET_STEP = 2000
DIAGNOSTIC_INTERVAL = 10
ENERGY_INTERVAL = 50
MILESTONE_INTERVAL = 100
CHECKPOINT_STEPS = tuple(range(800, TARGET_STEP + 1, 200))
FFT_WORKERS = 12
MAXIMUM_WALL_SECONDS = 2.0 * 60.0 * 60.0
RUNTIME_SAFETY_FACTOR = 1.3
MASS_DRIFT_LIMIT = 1.0e-4
ENERGY_REBOUND_LIMIT = 1.0e-6
FIELD_MAGNITUDE_LIMIT = 10.0
MINIMUM_DISK_HEADROOM_BYTES = 4 << 30
MAXIMUM_EVENT_CHECKPOINTS = 3
CHECKPOINT_STREAM = "inward-full-lifetime"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_lineage() -> tuple[dict[str, Any], dict[str, Any]]:
    summary = _load_json(SOURCE_SUMMARY_PATH)
    metadata = _load_json(SOURCE_CHECKPOINT_METADATA_PATH)
    source_checkpoint = next(
        (
            item
            for item in summary.get("checkpoints", [])
            if int(item.get("step", -1)) == START_STEP
        ),
        None,
    )
    if source_checkpoint is None:
        raise RuntimeError("t=600 source checkpoint is absent from summary")
    contract = summary.get("contract", {})
    provenance = contract.get("provenance", {})
    checks = {
        "summary_completed": summary.get("status") == "completed",
        "classification": (
            summary.get("classification")
            == "inward_intermediate_t600_completed_no_robust_break"
        ),
        "completed_step": int(summary.get("completed_step", -1))
        == START_STEP,
        "fixed_target_stop": (
            summary.get("stop_reason") == "fixed_t600_target_reached"
        ),
        "no_source_event": (
            summary.get("event_assessment", {}).get("detected") is False
        ),
        "metadata_step": int(metadata.get("step", -1)) == START_STEP,
        "metadata_matches_summary": (
            metadata.get("field_sha256")
            == source_checkpoint.get("field_sha256")
            and metadata.get("field_fingerprint")
            == source_checkpoint.get("field_fingerprint")
        ),
        "intermediate_runner_unchanged": (
            provenance.get("runner_sha256")
            == sha256_path(Path(intermediate.__file__).resolve())
        ),
        "selective_model_unchanged": (
            provenance.get("selective_model_sha256")
            == sha256_path(Path(selective_model.__file__).resolve())
        ),
        "selective_geometry_unchanged": (
            provenance.get("selective_geometry_sha256")
            == sha256_path(Path(selective_geometry.__file__).resolve())
        ),
        "selective_analysis_unchanged": (
            provenance.get("selective_analysis_sha256")
            == sha256_path(Path(selective_analysis.__file__).resolve())
        ),
        "field_exists": Path(str(metadata.get("field_path"))).is_file(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"t=600 source lineage failed: {checks}")
    entry = {
        **metadata,
        "metadata_path": str(SOURCE_CHECKPOINT_METADATA_PATH.resolve()),
        "metadata_sha256": sha256_path(SOURCE_CHECKPOINT_METADATA_PATH),
        "kinds": ["imported_verified_t600_source"],
        "external_source": True,
    }
    lineage = {
        "summary_path": str(SOURCE_SUMMARY_PATH.resolve()),
        "summary_sha256": sha256_path(SOURCE_SUMMARY_PATH),
        "checkpoint_metadata_path": str(
            SOURCE_CHECKPOINT_METADATA_PATH.resolve()
        ),
        "checkpoint_metadata_sha256": sha256_path(
            SOURCE_CHECKPOINT_METADATA_PATH
        ),
        "checkpoint_field_path": str(Path(metadata["field_path"]).resolve()),
        "checkpoint_field_sha256": metadata["field_sha256"],
        "checkpoint_field_fingerprint": metadata["field_fingerprint"],
        "checks": checks,
    }
    return lineage, entry


def _untreated_metadata_path(step: int) -> Path:
    if not START_STEP <= step <= TARGET_STEP:
        raise ValueError("untreated comparison step is outside the horizon")
    root = (
        UNTREATED_T1000_OUTPUT
        if step <= 1000
        else UNTREATED_T2000_OUTPUT
    )
    return root / f"checkpoint-step-{step:04d}.json"


def _untreated_checkpoint(
    step: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    metadata_path = _untreated_metadata_path(step)
    metadata = _load_json(metadata_path)
    field_path = Path(str(metadata["field_path"]))
    field = np.load(
        field_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    report = {
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": sha256_path(metadata_path),
        "field_path": str(field_path.resolve()),
        "field_sha256": metadata["field_sha256"],
        "field_fingerprint": metadata["field_fingerprint"],
        "checks": {
            "step": int(metadata["step"]) == step,
            "field_sha256": (
                sha256_path(field_path) == metadata["field_sha256"]
            ),
            "field_fingerprint": (
                roy_array_fingerprint(field)
                == metadata["field_fingerprint"]
            ),
            "shape": list(field.shape) == list(FROZEN_DEG90.lattice.shape),
            "dtype": field.dtype == np.dtype(np.float64),
            "finite": bool(np.isfinite(field).all()),
        },
    }
    if not all(report["checks"].values()):
        raise RuntimeError(
            f"untreated checkpoint {step} failed: {report['checks']}"
        )
    return field, report


def verify_untreated_reference() -> dict[str, Any]:
    first = _load_json(UNTREATED_T1000_OUTPUT / "summary.json")
    second_path = UNTREATED_T2000_OUTPUT / "summary.json"
    second = _load_json(second_path)
    checks = {
        "t1000_completed": first.get("status") == "completed",
        "t2000_completed": second.get("status") == "completed",
        "robust_event_detected": (
            second.get("breakup_assessment", {}).get(
                "robust_breakup_detected"
            )
            is True
        ),
        "reference_bracket": (
            second.get("breakup_assessment", {}).get(
                "common_threshold_robust_event", {}
            ).get("event_bracket")
            == [1710, 1720]
        ),
        "reference_confirmation": (
            second.get("breakup_assessment", {}).get(
                "common_threshold_robust_event", {}
            ).get("confirmation_step")
            == 1740
        ),
        "all_milestone_metadata_present": all(
            _untreated_metadata_path(step).is_file()
            for step in range(START_STEP, TARGET_STEP + 1, 100)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"untreated reference failed: {checks}")
    return {
        "t1000_summary_path": str(
            (UNTREATED_T1000_OUTPUT / "summary.json").resolve()
        ),
        "t1000_summary_sha256": sha256_path(
            UNTREATED_T1000_OUTPUT / "summary.json"
        ),
        "t2000_summary_path": str(second_path.resolve()),
        "t2000_summary_sha256": sha256_path(second_path),
        "event_bracket": [1710, 1720],
        "confirmation_step": 1740,
        "checks": checks,
    }


def _runtime_projection() -> dict[str, float]:
    source = _load_json(SOURCE_SUMMARY_PATH)
    median = float(source["execution"]["median_step_wall_seconds"])
    projected = (
        (TARGET_STEP - START_STEP)
        * median
        * RUNTIME_SAFETY_FACTOR
    )
    return {
        "source_median_step_wall_seconds": median,
        "remaining_steps": float(TARGET_STEP - START_STEP),
        "safety_factor": RUNTIME_SAFETY_FACTOR,
        "projected_wall_seconds": projected,
        "projected_wall_hours": projected / 3600.0,
    }


def resource_preflight(
    output: Path,
    *,
    current_step: int,
    existing_event_checkpoints: int = 0,
) -> dict[str, Any]:
    metadata = _load_json(SOURCE_CHECKPOINT_METADATA_PATH)
    field_bytes = int(metadata["field_bytes"])
    remaining_regular = sum(
        step > current_step for step in CHECKPOINT_STEPS
    )
    remaining_event = max(
        0, MAXIMUM_EVENT_CHECKPOINTS - existing_event_checkpoints
    )
    required_disk = (
        (remaining_regular + remaining_event + 1) * field_bytes
        + MINIMUM_DISK_HEADROOM_BYTES
    )
    free_disk = int(shutil.disk_usage(output.parent).free)
    runtime = _runtime_projection()
    report = {
        "field_bytes": field_bytes,
        "remaining_regular_checkpoints": remaining_regular,
        "remaining_event_checkpoints": remaining_event,
        "emergency_checkpoint_allowance": 1,
        "minimum_disk_headroom_bytes": MINIMUM_DISK_HEADROOM_BYTES,
        "required_free_disk_bytes": required_disk,
        "free_disk_bytes": free_disk,
        "disk_sufficient": free_disk >= required_disk,
        "runtime": runtime,
        "runtime_below_two_hours": (
            runtime["projected_wall_seconds"] < MAXIMUM_WALL_SECONDS
        ),
    }
    if not report["disk_sufficient"]:
        raise RuntimeError(f"disk preflight failed: {report}")
    if not report["runtime_below_two_hours"]:
        raise RuntimeError(f"runtime preflight failed: {report}")
    return report


def _contract() -> dict[str, Any]:
    source_lineage, _ = _source_lineage()
    untreated = verify_untreated_reference()
    return {
        "schema_version": 1,
        "protocol": "inward_tubular_collar_full_lifetime_continuation",
        "start_step": START_STEP,
        "target_step": TARGET_STEP,
        "diagnostic_interval": DIAGNOSTIC_INTERVAL,
        "energy_interval": ENERGY_INTERVAL,
        "milestone_interval": MILESTONE_INTERVAL,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "fft_workers": FFT_WORKERS,
        "maximum_wall_seconds": MAXIMUM_WALL_SECONDS,
        "maximum_event_checkpoints": MAXIMUM_EVENT_CHECKPOINTS,
        "stop_on_robust_event": True,
        "source_lineage": source_lineage,
        "untreated_reference": untreated,
        "collar": intermediate.INWARD_GEOMETRY.to_dict(),
        "provenance": {
            "runner_sha256": sha256_path(SOURCE_PATH),
            "scientific_contract_sha256": sha256_path(CONTRACT_PATH),
            "intermediate_runner_sha256": sha256_path(
                Path(intermediate.__file__).resolve()
            ),
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
            "one_treated_continuation": True,
            "untreated_simulation_reused": True,
            "parameter_sweep_authorized": False,
            "automatic_follow_on_authorized": False,
        },
    }


def _verify_resume_contract(
    stored: dict[str, Any],
    current: dict[str, Any],
) -> None:
    if stored != current:
        raise RuntimeError("resume contract differs from launch contract")


def _make_solver() -> tuple[SpatialMobilityRoySolver, dict[str, Any]]:
    return intermediate._make_solver()


def _diagnostic_record(
    field: np.ndarray,
    solver: SpatialMobilityRoySolver,
    *,
    step: int,
    initial_mass: float,
    elapsed_wall_seconds: float,
    include_energy: bool,
) -> dict[str, Any]:
    record = production_helper._diagnostic_record(
        field,
        solver,
        step=step,
        initial_mass=initial_mass,
        elapsed_wall_seconds=elapsed_wall_seconds,
        include_energy=include_energy,
    )
    record["elapsed_after_t600"] = step - START_STEP
    return record


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
    response = intermediate._treatment_profile_response(
        phase,
        contour,
        treated_name="treated",
        geometry=intermediate.INWARD_GEOMETRY,
    )
    arrays = intermediate._profile_arrays(phase, contour)
    artifact_path = output / f"profiles-step-{step:04d}.npz"
    atomic_npz(artifact_path, **arrays)
    report = {
        "step": step,
        "untreated_source": untreated_source,
        "response": response,
        "minimum_sites": intermediate._minimum_sites(contour),
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
        classification = "inward_full_lifetime_robust_first_event"
    elif terminal == "completed":
        classification = "inward_full_lifetime_t2000_censored"
    else:
        classification = f"inward_full_lifetime_{terminal}"
    times = status["step_wall_seconds"]
    summary = {
        "status": terminal,
        "classification": classification,
        "stop_reason": status["stop_reason"],
        "completed_step": status["current_step"],
        "event_assessment": event,
        "untreated_reference_event": {
            "event_bracket": [1710, 1720],
            "confirmation_step": 1740,
        },
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
            "one_treated_continuation_completed": True,
            "automatic_follow_on_performed": False,
            "scientific_interpretation_pending": True,
        },
    }
    atomic_json(output / "summary.json", summary)
    return summary


def _write_status(
    output: Path,
    *,
    terminal_status: str,
    current_step: int,
    stop_reason: str | None,
    elapsed: float,
    records: list[dict[str, Any]],
    milestones: list[dict[str, Any]],
    step_wall_seconds: list[float],
    checkpoints: list[dict[str, Any]],
    event: dict[str, Any],
    event_checkpoint_count: int,
    prior_energy: float | None,
    resource: dict[str, Any],
    factor_report: dict[str, Any],
) -> dict[str, Any]:
    status = {
        "status": terminal_status,
        "current_step": current_step,
        "target_step": TARGET_STEP,
        "stop_reason": stop_reason,
        "elapsed_wall_seconds": elapsed,
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
    production_helper._write_status(output, status)
    return status


def run(
    output: Path,
    *,
    resume: bool,
    stop_requested: dict[str, int | None],
) -> dict[str, Any]:
    current_contract = _contract()
    source_lineage, source_entry = _source_lineage()
    del source_lineage
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
            checkpoints = [
                item
                for item in status["checkpoints"]
                if int(item["step"]) <= int(status["current_step"])
            ]
            if not checkpoints:
                raise RuntimeError("no verified continuation checkpoint")
            field = load_checkpoint(checkpoints[-1])
            current_step = int(checkpoints[-1]["step"])
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
            prior_elapsed = float(records[-1]["elapsed_wall_seconds"])
            prior_energy = next(
                (
                    item["free_energy"]
                    for item in reversed(records)
                    if item.get("free_energy") is not None
                ),
                None,
            )
            event_checkpoint_count = sum(
                "event_candidate" in item.get("kinds", [])
                for item in checkpoints
            )
        else:
            atomic_json(output / "contract.json", current_contract)
            field = load_checkpoint(source_entry)
            current_step = START_STEP
            records: list[dict[str, Any]] = []
            milestones: list[dict[str, Any]] = []
            step_wall_seconds: list[float] = []
            checkpoints = [source_entry]
            event = production_helper._event_from_records(
                records, latched=None
            )
            prior_elapsed = 0.0
            prior_energy = None
            event_checkpoint_count = 0

        resource = resource_preflight(
            output,
            current_step=current_step,
            existing_event_checkpoints=event_checkpoint_count,
        )
        solver, factor_report = _make_solver()
        initial_mass = float(source_helper.EXPECTED_T100_MASS)
        session_started = time.perf_counter()

        if not records:
            initial = _diagnostic_record(
                field,
                solver,
                step=START_STEP,
                initial_mass=initial_mass,
                elapsed_wall_seconds=0.0,
                include_energy=True,
            )
            records.append(initial)
            prior_energy = float(initial["free_energy"])
            event = production_helper._event_from_records(
                records, latched=event
            )
            _write_status(
                output,
                terminal_status="running",
                current_step=START_STEP,
                stop_reason=None,
                elapsed=0.0,
                records=records,
                milestones=milestones,
                step_wall_seconds=step_wall_seconds,
                checkpoints=checkpoints,
                event=event,
                event_checkpoint_count=event_checkpoint_count,
                prior_energy=prior_energy,
                resource=resource,
                factor_report=factor_report,
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
                record = _diagnostic_record(
                    field,
                    solver,
                    step=step,
                    initial_mass=initial_mass,
                    elapsed_wall_seconds=elapsed,
                    include_energy=bool(
                        step % ENERGY_INTERVAL == 0
                        or terminal_status is not None
                    ),
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
                and event_checkpoint_count < MAXIMUM_EVENT_CHECKPOINTS
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
                _write_status(
                    output,
                    terminal_status="running",
                    current_step=step,
                    stop_reason=stop_reason,
                    elapsed=elapsed,
                    records=records,
                    milestones=milestones,
                    step_wall_seconds=step_wall_seconds,
                    checkpoints=checkpoints,
                    event=event,
                    event_checkpoint_count=event_checkpoint_count,
                    prior_energy=prior_energy,
                    resource=resource,
                    factor_report=factor_report,
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
                stop_reason = "two_hour_wall_cap_reached"
                break
            if step == TARGET_STEP:
                terminal_status = "completed"
                stop_reason = "fixed_t2000_target_reached"
                break

        elapsed_total = (
            prior_elapsed + time.perf_counter() - session_started
        )
        final_status = _write_status(
            output,
            terminal_status=str(terminal_status),
            current_step=current_step,
            stop_reason=stop_reason,
            elapsed=elapsed_total,
            records=records,
            milestones=milestones,
            step_wall_seconds=step_wall_seconds,
            checkpoints=checkpoints,
            event=event,
            event_checkpoint_count=event_checkpoint_count,
            prior_energy=prior_energy,
            resource=resource,
            factor_report=factor_report,
        )
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
        help="Required for the estimated 70-to-90-minute fresh launch.",
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
