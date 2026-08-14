#!/usr/bin/env python3
"""Run one restartable mature-t=100 crossed-wire GB sentinel case.

This runner is separate from ``run_production.py`` because that file belongs
to the cancelled equilibrium-product-union protocol.  Here the immutable Roy
source-semantic checkpoint at global step 100 is the common handoff.
"""

from __future__ import annotations

import argparse
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

from ..roy_2021_reproduction.model import FROZEN_DEG90, field_diagnostics
from ..roy_fixed_gb_bridge import (
    run_t100_checkpoint_source_localization_preflight as source_helper,
)
from ..roy_fixed_gb_bridge.metrics import array_fingerprint
from ..roy_fixed_gb_bridge.model import (
    FixedGrainBoundaryParameters,
    RoyFixedGrainBoundarySolver,
    build_periodic_bicrystal_profile,
)
from ..roy_fixed_gb_bridge.provenance import sha256_path
from .diagnostics import instantaneous_pinches, persistent_single_arm_event
from .storage import (
    acquire_output_lock,
    atomic_json,
    load_checkpoint,
    release_output_lock,
    reserve_output_directory,
    write_checkpoint,
)


SOURCE_PATH = Path(__file__).resolve()
DIRECTORY = SOURCE_PATH.parent
PREFLIGHT_RUNNER_PATH = DIRECTORY / "run_t100_conditioned_preflight.py"
PREFLIGHT_CONTRACT_PATH = DIRECTORY / "T100_CONDITIONED_PREFLIGHT.md"
PREFLIGHT_SUMMARY_PATH = (
    DIRECTORY
    / "results"
    / "t100_conditioned_preflight_v2"
    / "summary.json"
)
DEFAULT_RESULTS = DIRECTORY / "results"

CASE_SPACINGS = {
    "groove": 19.0,
    "ridge": 33.0,
    "maximum_separation": 96.0,
}
START_STEP = 100
TARGET_STEP = 2000
DIAGNOSTIC_INTERVAL = 10
ENERGY_INTERVAL = 50
CHECKPOINT_INTERVAL = 200
FFT_WORKERS = 12
ENERGY_RATIO = 0.35
GB_AXIS = 2
MAXIMUM_WALL_SECONDS = 4.0 * 3600.0
MASS_DRIFT_LIMIT = 1.0e-4
ENERGY_REBOUND_LIMIT = 1.0e-6
FIELD_MAGNITUDE_LIMIT = 10.0
MINIMUM_DISK_HEADROOM_BYTES = 2 << 30
MAXIMUM_EVENT_CHECKPOINTS = 3


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_preflight(case: str) -> dict[str, Any]:
    if case not in CASE_SPACINGS:
        raise ValueError(f"unsupported conditioned case: {case}")
    summary = _load_json(PREFLIGHT_SUMMARY_PATH)
    expected_spacing = CASE_SPACINGS[case]
    measured_case = summary.get("cases", {}).get(case, {})
    checks = {
        "preflight_completed": summary.get("status") == "completed",
        "preflight_passed": summary.get("preflight_passed") is True,
        "classification_passed": summary.get("classification")
        == "t100_conditioned_three_placement_preflight_passed",
        "case_passed": measured_case.get("passed") is True,
        "case_spacing_frozen": float(measured_case.get("spacing", np.nan))
        == expected_spacing,
        "zero_accepted_preflight_steps": summary.get("scope", {}).get(
            "accepted_trajectory_steps"
        )
        == 0,
        "production_not_started_by_preflight": summary.get("scope", {}).get(
            "production_started"
        )
        is False,
        "preflight_runner_unchanged": summary.get("provenance", {}).get(
            "runner_sha256"
        )
        == sha256_path(PREFLIGHT_RUNNER_PATH),
        "preflight_contract_unchanged": summary.get("provenance", {}).get(
            "contract_sha256"
        )
        == sha256_path(PREFLIGHT_CONTRACT_PATH),
    }
    if not all(checks.values()):
        raise RuntimeError(f"conditioned preflight verification failed: {checks}")
    source = source_helper.verify_sources()
    if source.get("verified") is not True:
        raise RuntimeError("t=100 source provenance did not verify")
    return {
        "summary_path": str(PREFLIGHT_SUMMARY_PATH.resolve()),
        "summary_sha256": sha256_path(PREFLIGHT_SUMMARY_PATH),
        "checks": checks,
        "source_verified": True,
        "spacing": expected_spacing,
    }


def resource_preflight(output: Path, *, current_step: int) -> dict[str, Any]:
    field_bytes = int(FROZEN_DEG90.lattice.cell_count * 8 + 256)
    remaining_regular = sum(
        step > current_step
        for step in range(
            ((START_STEP // CHECKPOINT_INTERVAL) + 1) * CHECKPOINT_INTERVAL,
            TARGET_STEP + 1,
            CHECKPOINT_INTERVAL,
        )
    )
    required_disk = (
        (remaining_regular + MAXIMUM_EVENT_CHECKPOINTS + 1) * field_bytes
        + MINIMUM_DISK_HEADROOM_BYTES
    )
    free_disk = int(shutil.disk_usage(output.parent).free)
    report = {
        "field_bytes": field_bytes,
        "remaining_regular_checkpoints": remaining_regular,
        "maximum_event_checkpoints": MAXIMUM_EVENT_CHECKPOINTS,
        "emergency_checkpoint_allowance": 1,
        "minimum_disk_headroom_bytes": MINIMUM_DISK_HEADROOM_BYTES,
        "required_free_disk_bytes": required_disk,
        "free_disk_bytes": free_disk,
        "disk_sufficient": free_disk >= required_disk,
    }
    if not report["disk_sufficient"]:
        raise RuntimeError(f"conditioned production disk preflight failed: {report}")
    return report


def _contract(case: str, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": "mature_t100_conditioned_fixed_gb",
        "case": case,
        "spacing": CASE_SPACINGS[case],
        "start_step": START_STEP,
        "target_step": TARGET_STEP,
        "diagnostic_interval": DIAGNOSTIC_INTERVAL,
        "energy_interval": ENERGY_INTERVAL,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "fft_workers": FFT_WORKERS,
        "energy_ratio": ENERGY_RATIO,
        "gb_axis": GB_AXIS,
        "maximum_wall_seconds": MAXIMUM_WALL_SECONDS,
        "stop_on_robust_event": True,
        "preflight": preflight,
        "source_checkpoint": {
            "path": str(source_helper.T100_CHECKPOINT_PATH.resolve()),
            "sha256": sha256_path(source_helper.T100_CHECKPOINT_PATH),
            "fingerprint": source_helper.EXPECTED_T100_FINGERPRINT,
            "global_step": START_STEP,
        },
        "provenance": {
            "runner_sha256": sha256_path(SOURCE_PATH),
            "fixed_gb_model_sha256": sha256_path(
                DIRECTORY.parent / "roy_fixed_gb_bridge" / "model.py"
            ),
            "roy_model_sha256": sha256_path(
                DIRECTORY.parent / "roy_2021_reproduction" / "model.py"
            ),
            "diagnostics_sha256": sha256_path(DIRECTORY / "diagnostics.py"),
            "storage_sha256": sha256_path(DIRECTORY / "storage.py"),
            "python": sys.version,
            "numpy": np.__version__,
        },
        "scope": {
            "raw_t0_initializer_solved": False,
            "raw_t0_initializer_bypassed": True,
            "spacing_sweep_enabled": False,
            "automatic_scientific_claim": False,
        },
    }


def _verify_resume_contract(stored: dict[str, Any], current: dict[str, Any]) -> None:
    if stored != current:
        raise RuntimeError("resume contract differs from the launch contract")


def _load_source_field() -> np.ndarray:
    mapped = np.load(
        source_helper.T100_CHECKPOINT_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )
    if array_fingerprint(mapped) != source_helper.EXPECTED_T100_FINGERPRINT:
        raise RuntimeError("immutable t=100 checkpoint fingerprint mismatch")
    return np.array(mapped, dtype=np.float64, copy=True, order="C")


def _make_solver(case: str) -> tuple[RoyFixedGrainBoundarySolver, dict[str, float]]:
    mapped = FixedGrainBoundaryParameters.matched_to_roy(
        FROZEN_DEG90.parameters,
        energy_ratio=ENERGY_RATIO,
    )
    profile = build_periodic_bicrystal_profile(
        FROZEN_DEG90.lattice,
        mapped,
        axis=GB_AXIS,
        primary_coordinate=CASE_SPACINGS[case],
    )
    solver = RoyFixedGrainBoundarySolver(
        FROZEN_DEG90.lattice,
        FROZEN_DEG90.parameters,
        profile,
        fft_workers=FFT_WORKERS,
    )
    return solver, {
        "primary_coordinate": float(profile.primary_coordinate),
        "image_coordinate": float(profile.image_coordinate),
        "surface_width": float(mapped.surface_width),
    }


def _diagnostic_record(
    field: np.ndarray,
    solver: RoyFixedGrainBoundarySolver,
    *,
    step: int,
    initial_mass: float,
    elapsed_wall_seconds: float,
    include_energy: bool,
) -> dict[str, Any]:
    scalar = field_diagnostics(field, FROZEN_DEG90.lattice).to_dict()
    finite = bool(scalar["finite"])
    mass = scalar["mass"]
    drift = (
        abs(float(mass) - initial_mass) / abs(initial_mass)
        if finite and mass is not None
        else None
    )
    pinches = (
        instantaneous_pinches(
            field,
            geometry="crossed",
            spacing=FROZEN_DEG90.lattice.spacing,
            radius=FROZEN_DEG90.radius_1 * FROZEN_DEG90.lattice.spacing,
        )
        if finite
        else None
    )
    return {
        "step": int(step),
        "global_time": float(step * FROZEN_DEG90.parameters.timestep),
        "elapsed_after_handoff": int(step - START_STEP),
        "elapsed_wall_seconds": float(elapsed_wall_seconds),
        "field": scalar,
        "relative_mass_drift": drift,
        "pinches": pinches,
        "free_energy": solver.free_energy(field) if finite and include_energy else None,
        "last_inverse_imaginary_linf": float(
            solver.last_inverse_imaginary_linf
        ),
    }


def _health_stop_reason(
    record: dict[str, Any],
    *,
    previous_energy: float | None,
) -> str | None:
    field = record["field"]
    if not field["finite"]:
        return "nonfinite_field"
    if (
        abs(float(field["minimum"])) > FIELD_MAGNITUDE_LIMIT
        or abs(float(field["maximum"])) > FIELD_MAGNITUDE_LIMIT
    ):
        return "field_magnitude_limit_exceeded"
    if float(record["relative_mass_drift"]) > MASS_DRIFT_LIMIT:
        return "mass_drift_limit_exceeded"
    energy = record["free_energy"]
    if (
        energy is not None
        and previous_energy is not None
        and (float(energy) - previous_energy) / abs(previous_energy)
        > ENERGY_REBOUND_LIMIT
    ):
        return "energy_rebound_limit_exceeded"
    return None


def _event_from_records(
    records: list[dict[str, Any]],
    *,
    latched: dict[str, Any] | None,
) -> dict[str, Any]:
    return persistent_single_arm_event(
        [
            {"step": record["step"], "pinches": record["pinches"]}
            for record in records
            if record.get("pinches") is not None
        ],
        diagnostic_interval=DIAGNOSTIC_INTERVAL,
        latched_event=latched,
    )


def _write_status(output: Path, status: dict[str, Any]) -> None:
    status["last_update_unix_time"] = time.time()
    status["pid"] = os.getpid()
    atomic_json(output / "run_status.json", status)


def _terminal_summary(
    output: Path,
    contract: dict[str, Any],
    status: dict[str, Any],
) -> dict[str, Any]:
    event = status["event_assessment"]
    terminal = status["status"]
    if terminal == "completed" and event.get("detected") is True:
        classification = f"{contract['case']}_robust_first_break_completed"
    elif terminal == "completed":
        classification = f"{contract['case']}_t2000_censored_no_robust_break"
    else:
        classification = f"{contract['case']}_{terminal}"
    step_times = status["step_wall_seconds"]
    summary = {
        "status": terminal,
        "classification": classification,
        "stop_reason": status["stop_reason"],
        "case": contract["case"],
        "spacing": contract["spacing"],
        "completed_step": status["current_step"],
        "event_assessment": event,
        "records": status["records"],
        "checkpoints": status["checkpoints"],
        "execution": {
            "elapsed_wall_seconds": status["elapsed_wall_seconds"],
            "accepted_steps": status["current_step"] - START_STEP,
            "step_wall_seconds_count": len(step_times),
            "median_step_wall_seconds": (
                float(statistics.median(step_times)) if step_times else None
            ),
            "maximum_step_wall_seconds": max(step_times) if step_times else None,
        },
        "contract": contract,
        "scope": {
            "raw_t0_initializer_bypassed": True,
            "scientific_interpretation_pending_comparison": True,
            "automatic_follow_on_inside_runner": False,
        },
    }
    atomic_json(output / "summary.json", summary)
    return summary


def run(
    output: Path,
    *,
    case: str,
    resume: bool,
    stop_requested: dict[str, int | None],
) -> dict[str, Any]:
    preflight = verify_preflight(case)
    current_contract = _contract(case, preflight)
    if resume:
        if not output.is_dir():
            raise FileNotFoundError("resume output directory does not exist")
    else:
        reserve_output_directory(output)
    lock = acquire_output_lock(output)
    try:
        resource_report: dict[str, Any]
        if resume:
            stored_contract = _load_json(output / "contract.json")
            _verify_resume_contract(stored_contract, current_contract)
            status = _load_json(output / "run_status.json")
            if status["status"] not in {"running", "interrupted", "wall_cap"}:
                raise RuntimeError("run status is not resumable")
            checkpoints = list(status["checkpoints"])
            if checkpoints:
                field = load_checkpoint(checkpoints[-1])
                current_step = int(checkpoints[-1]["step"])
            else:
                field = _load_source_field()
                current_step = START_STEP
            records = [
                record
                for record in status["records"]
                if int(record["step"]) <= current_step
            ]
            step_wall_seconds = list(status["step_wall_seconds"])[
                : current_step - START_STEP
            ]
            event = status["event_assessment"]
            prior_elapsed = float(status["elapsed_wall_seconds"])
            prior_energy = status.get("last_energy")
            event_checkpoint_count = int(status["event_checkpoint_count"])
        else:
            atomic_json(output / "contract.json", current_contract)
            field = _load_source_field()
            current_step = START_STEP
            records = []
            step_wall_seconds = []
            checkpoints = []
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

        resource_report = resource_preflight(
            output, current_step=current_step
        )
        solver, profile = _make_solver(case)
        initial_mass = float(source_helper.EXPECTED_T100_MASS)
        session_started = time.perf_counter()

        if not records:
            initial = _diagnostic_record(
                field,
                solver,
                step=START_STEP,
                initial_mass=initial_mass,
                elapsed_wall_seconds=prior_elapsed,
                include_energy=True,
            )
            records.append(initial)
            prior_energy = float(initial["free_energy"])
            event = _event_from_records(records, latched=event)
            _write_status(
                output,
                {
                    "status": "running",
                    "current_step": START_STEP,
                    "target_step": TARGET_STEP,
                    "stop_reason": None,
                    "elapsed_wall_seconds": prior_elapsed,
                    "records": records,
                    "step_wall_seconds": step_wall_seconds,
                    "checkpoints": checkpoints,
                    "event_assessment": event,
                    "event_checkpoint_count": event_checkpoint_count,
                    "last_energy": prior_energy,
                    "resource_preflight": resource_report,
                    "profile": profile,
                },
            )

        terminal_status: str | None = None
        stop_reason: str | None = None
        for step in range(current_step + 1, TARGET_STEP + 1):
            step_started = time.perf_counter()
            field = solver.propose_step(field)
            step_wall_seconds.append(float(time.perf_counter() - step_started))
            current_step = step
            if not np.isfinite(field).all():
                terminal_status = "numerically_aborted"
                stop_reason = "nonfinite_field"

            candidate_now = False
            record: dict[str, Any] | None = None
            if step % DIAGNOSTIC_INTERVAL == 0 or terminal_status is not None:
                elapsed = prior_elapsed + time.perf_counter() - session_started
                include_energy = bool(
                    step % ENERGY_INTERVAL == 0
                    or terminal_status is not None
                )
                record = _diagnostic_record(
                    field,
                    solver,
                    step=step,
                    initial_mass=initial_mass,
                    elapsed_wall_seconds=elapsed,
                    include_energy=include_energy,
                )
                records.append(record)
                health_reason = _health_stop_reason(
                    record,
                    previous_energy=prior_energy,
                )
                if record["free_energy"] is not None:
                    prior_energy = float(record["free_energy"])
                if health_reason is not None:
                    terminal_status = "numerically_aborted"
                    stop_reason = health_reason
                event = _event_from_records(records, latched=event)
                candidate_now = bool(record["pinches"]["candidate"])
                if event.get("detected") is True:
                    terminal_status = "completed"
                    stop_reason = "robust_single_arm_event_confirmed"

            elapsed = prior_elapsed + time.perf_counter() - session_started
            signal_pending = stop_requested["signal"] is not None
            wall_cap_reached = elapsed >= MAXIMUM_WALL_SECONDS
            event_checkpoint_due = bool(
                candidate_now
                and event_checkpoint_count < MAXIMUM_EVENT_CHECKPOINTS
            )
            regular_checkpoint_due = step % CHECKPOINT_INTERVAL == 0
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
                    stream=case,
                )
                checkpoints.append(checkpoint)
                if event_checkpoint_due:
                    event_checkpoint_count += 1
                _write_status(
                    output,
                    {
                        "status": "running",
                        "current_step": step,
                        "target_step": TARGET_STEP,
                        "stop_reason": stop_reason,
                        "elapsed_wall_seconds": elapsed,
                        "records": records,
                        "step_wall_seconds": step_wall_seconds,
                        "checkpoints": checkpoints,
                        "event_assessment": event,
                        "event_checkpoint_count": event_checkpoint_count,
                        "last_energy": prior_energy,
                        "resource_preflight": resource_report,
                        "profile": profile,
                    },
                )
                print(
                    json.dumps(
                        {
                            "case": case,
                            "status": "running",
                            "step": step,
                            "elapsed_wall_seconds": elapsed,
                            "checkpoint": checkpoint["field_path"],
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
                stop_reason = f"deferred_signal_{stop_requested['signal']}"
                break
            if wall_cap_reached:
                terminal_status = "wall_cap"
                stop_reason = "four_hour_case_wall_cap_reached"
                break
            if step == TARGET_STEP:
                terminal_status = "completed"
                stop_reason = "fixed_t2000_target_reached"
                break

        elapsed_total = prior_elapsed + time.perf_counter() - session_started
        final_status = {
            "status": terminal_status,
            "current_step": current_step,
            "target_step": TARGET_STEP,
            "stop_reason": stop_reason,
            "elapsed_wall_seconds": elapsed_total,
            "records": records,
            "step_wall_seconds": step_wall_seconds,
            "checkpoints": checkpoints,
            "event_assessment": event,
            "event_checkpoint_count": event_checkpoint_count,
            "last_energy": prior_energy,
            "resource_preflight": resource_report,
            "profile": profile,
        }
        _write_status(output, final_status)
        return _terminal_summary(output, current_contract, final_status)
    finally:
        release_output_lock(lock)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument("--case", choices=tuple(CASE_SPACINGS), required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    output = (
        arguments.output
        if arguments.output is not None
        else DEFAULT_RESULTS
        / f"t100_conditioned_production_{arguments.case}_v1"
    ).resolve()
    stop_requested: dict[str, int | None] = {"signal": None}

    def request_stop(signal_number: int, _frame: object) -> None:
        stop_requested["signal"] = signal_number

    handled = (signal.SIGINT, signal.SIGTERM)
    previous_handlers = {
        number: signal.getsignal(number) for number in handled
    }
    for number in handled:
        signal.signal(number, request_stop)
    try:
        result = run(
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
                    "case": result["case"],
                    "completed_step": result["completed_step"],
                    "output": str(output),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if result["status"] == "completed" else 2
    except Exception as exc:
        if output.is_dir():
            atomic_json(
                output / "failure.json",
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "pid": os.getpid(),
                    "unix_time": time.time(),
                },
            )
        raise
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    raise SystemExit(main())
