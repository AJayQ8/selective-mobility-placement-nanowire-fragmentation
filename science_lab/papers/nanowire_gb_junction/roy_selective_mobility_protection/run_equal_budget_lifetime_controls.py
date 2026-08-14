#!/usr/bin/env python3
"""Run restartable K2/K3 equal-budget lifetime controls."""

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
from . import equal_budget_controls
from . import model as selective_model
from . import run_equal_budget_control_preflight as preflight_runner
from . import run_full_lifetime_continuation as k1_runner
from . import run_intermediate_horizon_sentinel as intermediate_runner
from .equal_budget_controls import derive_equal_budget_controls
from .model import SpatialMobilityRoySolver
from .run_t100_tubular_position_diagnostic import INWARD_GEOMETRY


SOURCE_PATH = Path(__file__).resolve()
DIRECTORY = SOURCE_PATH.parent
CONTRACT_PATH = DIRECTORY / "EQUAL_BUDGET_CONTROLS.md"
PREFLIGHT_OUTPUT = (
    DIRECTORY / "results" / "equal_budget_control_preflight_v2"
)
PREFLIGHT_SUMMARY_PATH = PREFLIGHT_OUTPUT / "summary.json"
DEFAULT_OUTPUTS = {
    "K2": DIRECTORY / "results" / "k2_junction_cap_lifetime_v1",
    "K3": DIRECTORY / "results" / "k3_uniform_lifetime_v1",
}
SEQUENCE_STATUS_PATH = (
    DIRECTORY / "results" / "equal_budget_controls_sequence_status.json"
)

CASES = ("K2", "K3")
START_STEP = 100
TARGET_STEP = 2000
DIAGNOSTIC_INTERVAL = 10
ENERGY_INTERVAL = 50
MILESTONE_INTERVAL = 100
CHECKPOINT_STEPS = tuple(range(200, TARGET_STEP + 1, 200))
FFT_WORKERS = 12
MAXIMUM_WALL_SECONDS = 2.0 * 60.0 * 60.0
RUNTIME_SAFETY_FACTOR = 1.3
MASS_DRIFT_LIMIT = 1.0e-4
ENERGY_REBOUND_LIMIT = 1.0e-6
FIELD_MAGNITUDE_LIMIT = 10.0
MINIMUM_DISK_HEADROOM_BYTES = 4 << 30
MAXIMUM_EVENT_CHECKPOINTS = 3


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_preflight() -> dict[str, Any]:
    summary = _load_json(PREFLIGHT_SUMMARY_PATH)
    provenance = summary.get("contract", {}).get("provenance", {})
    checks = {
        "completed": summary.get("status") == "completed",
        "classification": (
            summary.get("classification")
            == "equal_budget_control_preflight_passed"
        ),
        "passed": summary.get("passed") is True,
        "production_not_launched": (
            summary.get("scope", {}).get("production_launched") is False
        ),
        "both_cases_passed": all(
            item.get("passed") is True
            for item in summary.get("cases", [])
        )
        and len(summary.get("cases", [])) == 2,
        "preflight_runner_unchanged": (
            provenance.get("runner_sha256")
            == sha256_path(Path(preflight_runner.__file__).resolve())
        ),
        "control_definitions_unchanged": (
            provenance.get("equal_budget_controls_sha256")
            == sha256_path(Path(equal_budget_controls.__file__).resolve())
        ),
        "selective_model_unchanged": (
            provenance.get("selective_model_sha256")
            == sha256_path(Path(selective_model.__file__).resolve())
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"equal-budget preflight failed: {checks}")
    return {
        "path": str(PREFLIGHT_SUMMARY_PATH.resolve()),
        "sha256": sha256_path(PREFLIGHT_SUMMARY_PATH),
        "checks": checks,
        "runtime_projection": summary["runtime_projection"],
        "control_definition": summary["contract"]["control_definition"],
    }


def _load_source_field() -> np.ndarray:
    mapped = np.load(
        source_helper.T100_CHECKPOINT_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )
    checks = {
        "fingerprint": (
            array_fingerprint(mapped)
            == source_helper.EXPECTED_T100_FINGERPRINT
        ),
        "shape": list(mapped.shape) == list(FROZEN_DEG90.lattice.shape),
        "dtype": mapped.dtype == np.dtype(np.float64),
        "finite": bool(np.isfinite(mapped).all()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"time-100 source failed: {checks}")
    return np.array(mapped, dtype=np.float64, copy=True, order="C")


def _derive_case(
    case: str,
) -> tuple[
    equal_budget_controls.EqualBudgetControlDefinition,
    np.ndarray,
    dict[str, Any],
]:
    if case not in CASES:
        raise ValueError(f"unknown equal-budget case: {case}")
    mapped = np.load(
        source_helper.T100_CHECKPOINT_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )
    definition, k1, k2, k3 = derive_equal_budget_controls(
        mapped,
        FROZEN_DEG90.lattice,
        inward_geometry=INWARD_GEOMETRY,
        radius=6.0,
        interface_width=float(np.sqrt(8.0)),
    )
    preflight = verify_preflight()
    if definition.to_dict() != preflight["control_definition"]:
        raise RuntimeError("derived control definition differs from preflight")
    factor = k2 if case == "K2" else k3
    report = {
        "case": case,
        "definition": definition.to_dict(),
        "shape": list(factor.shape),
        "minimum": float(np.min(factor)),
        "maximum": float(np.max(factor)),
        "support_cell_fraction": (
            float(np.mean(factor < 1.0 - 1.0e-14))
            if factor.size > 1
            else 1.0
        ),
        "factor_semantics": (
            "junction_centered_tubular_cap"
            if case == "K2"
            else "uniform_equal_budget_factor"
        ),
    }
    del mapped, k1
    if case == "K2":
        del k3
    else:
        del k2
    gc.collect()
    return definition, factor, report


def _runtime_projection(
    *,
    current_step: int,
) -> dict[str, float]:
    preflight = verify_preflight()
    median = float(
        preflight["runtime_projection"][
            "conservative_median_proposal_seconds"
        ]
    )
    remaining = TARGET_STEP - current_step
    projected = remaining * median * RUNTIME_SAFETY_FACTOR
    return {
        "measured_conservative_median_step_seconds": median,
        "remaining_steps": float(remaining),
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
    field_bytes = int(FROZEN_DEG90.lattice.cell_count * 8 + 256)
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
    runtime = _runtime_projection(current_step=current_step)
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
        "runtime_below_two_hour_cap": (
            runtime["projected_wall_seconds"] < MAXIMUM_WALL_SECONDS
        ),
    }
    if not report["disk_sufficient"]:
        raise RuntimeError(f"disk preflight failed: {report}")
    if not report["runtime_below_two_hour_cap"]:
        raise RuntimeError(f"runtime preflight failed: {report}")
    return report


def _contract(
    case: str,
    definition: equal_budget_controls.EqualBudgetControlDefinition,
    factor_report: dict[str, Any],
) -> dict[str, Any]:
    preflight = verify_preflight()
    untreated = k1_runner.verify_untreated_reference()
    return {
        "schema_version": 1,
        "protocol": "equal_budget_lifetime_control",
        "case": case,
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
        "source_checkpoint": {
            "path": str(source_helper.T100_CHECKPOINT_PATH.resolve()),
            "sha256": sha256_path(source_helper.T100_CHECKPOINT_PATH),
            "fingerprint": source_helper.EXPECTED_T100_FINGERPRINT,
            "global_step": START_STEP,
        },
        "preflight": preflight,
        "control_definition": definition.to_dict(),
        "factor": factor_report,
        "untreated_reference": untreated,
        "k1_reference": {
            "summary_path": str(
                (
                    k1_runner.DEFAULT_OUTPUT / "summary.json"
                ).resolve()
            ),
            "summary_sha256": sha256_path(
                k1_runner.DEFAULT_OUTPUT / "summary.json"
            ),
        },
        "provenance": {
            "runner_sha256": sha256_path(SOURCE_PATH),
            "scientific_contract_sha256": sha256_path(CONTRACT_PATH),
            "equal_budget_controls_sha256": sha256_path(
                Path(equal_budget_controls.__file__).resolve()
            ),
            "selective_model_sha256": sha256_path(
                Path(selective_model.__file__).resolve()
            ),
            "selective_analysis_sha256": sha256_path(
                Path(selective_analysis.__file__).resolve()
            ),
            "roy_model_sha256": sha256_path(
                Path(roy_model.__file__).resolve()
            ),
            "python": sys.version,
            "numpy": np.__version__,
        },
        "scope": {
            "one_equal_budget_control": True,
            "other_controls_launched_inside_case": False,
            "parameter_sweep_authorized": False,
            "part_two_authorized": False,
        },
    }


def _verify_resume_contract(
    stored: dict[str, Any],
    current: dict[str, Any],
) -> None:
    if stored != current:
        raise RuntimeError("resume contract differs from launch contract")


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
    record["elapsed_after_t100"] = step - START_STEP
    return record


def _untreated_checkpoint(
    step: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not START_STEP <= step <= TARGET_STEP:
        raise ValueError("untreated comparison step is outside the horizon")
    root = (
        k1_runner.UNTREATED_T1000_OUTPUT
        if step <= 1000
        else k1_runner.UNTREATED_T2000_OUTPUT
    )
    metadata_path = root / f"checkpoint-step-{step:04d}.json"
    metadata = _load_json(metadata_path)
    field_path = Path(str(metadata["field_path"]))
    field = np.load(field_path, mmap_mode="r", allow_pickle=False)
    checks = {
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
    }
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


def _milestone(
    output: Path,
    field: np.ndarray,
    *,
    step: int,
) -> dict[str, Any]:
    untreated, untreated_source = _untreated_checkpoint(step)
    phase = {
        "treated": selective_analysis.phase_excess_radius_profiles(field),
        "untreated": selective_analysis.phase_excess_radius_profiles(
            untreated
        ),
    }
    contour = {
        "treated": selective_analysis.crossed_contour_radius_profiles(field),
        "untreated": selective_analysis.crossed_contour_radius_profiles(
            untreated
        ),
    }
    arrays = intermediate_runner._profile_arrays(phase, contour)
    artifact_path = output / f"profiles-step-{step:04d}.npz"
    atomic_npz(artifact_path, **arrays)
    report = {
        "step": step,
        "untreated_source": untreated_source,
        "contact": {
            "control": contact_metrics(field, FROZEN_DEG90),
            "untreated": contact_metrics(untreated, FROZEN_DEG90),
        },
        "pinches": {
            "control": instantaneous_pinches(
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


def _terminal_summary(
    output: Path,
    contract: dict[str, Any],
    status: dict[str, Any],
) -> dict[str, Any]:
    terminal = status["status"]
    event = status["event_assessment"]
    case = contract["case"]
    if terminal == "completed" and event.get("detected") is True:
        classification = f"{case}_equal_budget_robust_first_event"
    elif terminal == "completed":
        classification = f"{case}_equal_budget_t2000_censored"
    else:
        classification = f"{case}_equal_budget_{terminal}"
    times = status["step_wall_seconds"]
    summary = {
        "status": terminal,
        "classification": classification,
        "case": case,
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
            "one_control_completed": True,
            "automatic_scientific_follow_on_performed": False,
            "part_two_started": False,
            "scientific_interpretation_pending_pair": True,
        },
    }
    atomic_json(output / "summary.json", summary)
    return summary


def run_case(
    output: Path,
    *,
    case: str,
    resume: bool,
    stop_requested: dict[str, int | None],
) -> dict[str, Any]:
    definition, factor, factor_report = _derive_case(case)
    contract = _contract(case, definition, factor_report)
    if resume:
        if not output.is_dir():
            raise FileNotFoundError("resume output directory is absent")
    else:
        reserve_output_directory(output)
    lock = acquire_output_lock(output)
    try:
        if resume:
            stored_contract = _load_json(output / "contract.json")
            _verify_resume_contract(stored_contract, contract)
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
            prior_elapsed = float(status["elapsed_wall_seconds"])
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
            atomic_json(output / "contract.json", contract)
            field = _load_source_field()
            current_step = START_STEP
            records: list[dict[str, Any]] = []
            milestones: list[dict[str, Any]] = []
            step_wall_seconds: list[float] = []
            checkpoints: list[dict[str, Any]] = []
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
        solver = SpatialMobilityRoySolver(
            FROZEN_DEG90.lattice,
            FROZEN_DEG90.parameters,
            factor,
            fft_workers=FFT_WORKERS,
        )
        del factor
        gc.collect()
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
                milestones.append(_milestone(output, field, step=step))

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
                    stream=(
                        "junction_cap_equal_budget_lifetime"
                        if case == "K2"
                        else "uniform_equal_budget_lifetime"
                    ),
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
                            "case": case,
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
        return _terminal_summary(output, contract, final_status)
    finally:
        release_output_lock(lock)


def _write_sequence_status(
    *,
    status: str,
    current_case: str | None,
    completed_cases: list[dict[str, Any]],
    stop_reason: str | None,
) -> None:
    atomic_json(
        SEQUENCE_STATUS_PATH,
        {
            "status": status,
            "current_case": current_case,
            "completed_cases": completed_cases,
            "stop_reason": stop_reason,
            "last_update_unix_time": time.time(),
            "pid": os.getpid(),
            "part_two_started": False,
        },
    )


def run_sequence(
    *,
    resume: bool,
    stop_requested: dict[str, int | None],
) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    for case in CASES:
        output = DEFAULT_OUTPUTS[case]
        completed_summary = output / "summary.json"
        if resume and completed_summary.is_file():
            prior = _load_json(completed_summary)
            if prior.get("status") == "completed":
                completed.append(
                    {
                        "case": case,
                        "status": prior["status"],
                        "classification": prior["classification"],
                        "completed_step": prior["completed_step"],
                        "summary_path": str(completed_summary.resolve()),
                        "summary_sha256": sha256_path(completed_summary),
                    }
                )
                continue
        case_resume = resume and output.exists()
        _write_sequence_status(
            status="running",
            current_case=case,
            completed_cases=completed,
            stop_reason=None,
        )
        result = run_case(
            output,
            case=case,
            resume=case_resume,
            stop_requested=stop_requested,
        )
        completed.append(
            {
                "case": case,
                "status": result["status"],
                "classification": result["classification"],
                "completed_step": result["completed_step"],
                "summary_path": str((output / "summary.json").resolve()),
                "summary_sha256": sha256_path(output / "summary.json"),
            }
        )
        if result["status"] != "completed":
            _write_sequence_status(
                status="stopped",
                current_case=None,
                completed_cases=completed,
                stop_reason=f"{case}_{result['status']}",
            )
            return completed
        if stop_requested["signal"] is not None:
            _write_sequence_status(
                status="interrupted",
                current_case=None,
                completed_cases=completed,
                stop_reason=(
                    f"deferred_signal_{stop_requested['signal']}"
                ),
            )
            return completed
    _write_sequence_status(
        status="completed",
        current_case=None,
        completed_cases=completed,
        stop_reason=None,
    )
    return completed


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--case",
        choices=(*CASES, "all"),
        required=True,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--acknowledge-bounded-runtime",
        action="store_true",
        help="Required for approximately two hours per control case.",
    )
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
    if arguments.case == "all" and arguments.output is not None:
        raise ValueError("--output is invalid with --case all")

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
        if arguments.case == "all":
            results = run_sequence(
                resume=arguments.resume,
                stop_requested=stop_requested,
            )
            print(
                json.dumps(
                    {
                        "status": (
                            "completed"
                            if len(results) == len(CASES)
                            and all(
                                item["status"] == "completed"
                                for item in results
                            )
                            else "stopped"
                        ),
                        "cases": results,
                        "sequence_status": str(
                            SEQUENCE_STATUS_PATH.resolve()
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            output = (
                arguments.output
                if arguments.output is not None
                else DEFAULT_OUTPUTS[arguments.case]
            ).resolve()
            result = run_case(
                output,
                case=arguments.case,
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
    finally:
        for number, handler in prior_handlers.items():
            signal.signal(number, handler)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
