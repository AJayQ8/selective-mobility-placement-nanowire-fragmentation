#!/usr/bin/env python3
"""Run the preregistered fixed-shape outer-collar lifetime control."""

from __future__ import annotations

import argparse
import gc
import json
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
from ..roy_fixed_gb_bridge import (
    run_t100_checkpoint_source_localization_preflight as source_helper,
)
from ..roy_fixed_gb_bridge.metrics import array_fingerprint
from ..roy_fixed_gb_bridge.provenance import sha256_path
from ..roy_gb_junction_sentinel import (
    run_t100_conditioned_production as production_helper,
)
from ..roy_gb_junction_sentinel.storage import (
    acquire_output_lock,
    atomic_json,
    load_checkpoint,
    release_output_lock,
    reserve_output_directory,
    write_checkpoint,
)
from . import analysis as selective_analysis
from . import geometry as selective_geometry
from . import model as selective_model
from . import run_equal_budget_lifetime_controls as lifetime_runner
from .geometry import (
    four_arm_tubular_collar_factor,
    surface_weighted_mobility_deficit,
)
from .model import SpatialMobilityRoySolver
from .run_t100_tubular_position_diagnostic import (
    INWARD_GEOMETRY,
    OUTER_GEOMETRY,
)


SOURCE_PATH = Path(__file__).resolve()
DIRECTORY = SOURCE_PATH.parent
CONTRACT_PATH = DIRECTORY / "OUTER_PLACEMENT_LIFETIME_CONTROL.md"
READOUT_PATH = DIRECTORY / "LOCALIZED_FIRST_BREAK_READOUT.md"
STATIC_SUMMARY_PATH = (
    DIRECTORY / "results" / "t100_static_preflight_v1" / "summary.json"
)
DEFAULT_OUTPUT = (
    DIRECTORY / "results" / "outer_placement_lifetime_v1"
)
REFERENCE_SUMMARIES = {
    "K1": DIRECTORY / "results" / "k1_first_break_t3000_v1" / "summary.json",
    "K2": DIRECTORY / "results" / "k2_first_break_t3000_v1" / "summary.json",
    "K3": DIRECTORY / "results" / "k3_uniform_lifetime_v1" / "summary.json",
}

START_STEP = 100
TARGET_STEP = 2000
DIAGNOSTIC_INTERVAL = 10
ENERGY_INTERVAL = 50
MILESTONE_INTERVAL = 100
CHECKPOINT_STEPS = (500, 900, 1300, 1700, 2000)
FFT_WORKERS = 12
MAXIMUM_WALL_SECONDS = 2.0 * 60.0 * 60.0
EXPECTED_EVENT_MIDPOINT = 1715
RUNTIME_SAFETY_FACTOR = 1.2
MINIMUM_DISK_HEADROOM_BYTES = 4 << 30
MAXIMUM_EVENT_CHECKPOINTS = 3
MAXIMUM_BUDGET_SURPLUS = 0.02
STREAM_NAME = "outer_placement_lifetime"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def verify_precursors() -> dict[str, Any]:
    reports: dict[str, Any] = {}
    expected = {
        "K1": ("K1_localized_robust_first_event", [2540, 2550]),
        "K2": ("K2_localized_robust_first_event", [2190, 2200]),
        "K3": ("K3_equal_budget_robust_first_event", [1760, 1770]),
    }
    for case, path in REFERENCE_SUMMARIES.items():
        summary = _load_json(path)
        classification, bracket = expected[case]
        checks = {
            "completed": summary.get("status") == "completed",
            "classification": (
                summary.get("classification") == classification
            ),
            "event_detected": (
                summary.get("event_assessment", {}).get("detected") is True
            ),
            "event_bracket": (
                summary.get("event_assessment", {}).get("event_bracket")
                == bracket
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(f"{case} precursor failed: {checks}")
        reports[case] = {
            "summary_path": str(path.resolve()),
            "summary_sha256": sha256_path(path),
            "classification": classification,
            "event_bracket": bracket,
            "checks": checks,
        }
    untreated = lifetime_runner.k1_runner.verify_untreated_reference()
    reports["untreated"] = untreated
    return reports


def derive_outer_factor() -> tuple[np.ndarray, dict[str, Any]]:
    mapped = np.load(
        source_helper.T100_CHECKPOINT_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )
    lattice = FROZEN_DEG90.lattice
    radius = 6.0
    interface_width = float(np.sqrt(8.0))
    inward = four_arm_tubular_collar_factor(
        lattice,
        INWARD_GEOMETRY,
        radius=radius,
        interface_width=interface_width,
    )
    inward_budget = surface_weighted_mobility_deficit(
        mapped,
        inward,
        cell_volume=lattice.cell_volume,
    )
    del inward
    gc.collect()
    outer = four_arm_tubular_collar_factor(
        lattice,
        OUTER_GEOMETRY,
        radius=radius,
        interface_width=interface_width,
    )
    outer_budget = surface_weighted_mobility_deficit(
        mapped,
        outer,
        cell_volume=lattice.cell_volume,
    )
    relative_surplus = (outer_budget - inward_budget) / inward_budget

    static = _load_json(STATIC_SUMMARY_PATH)
    depletion = static["depletion_measurement"]["pooled"]
    maximum_half_depth = float(
        depletion["maximum_outward_half_depth"]
    )
    geometry_checks = {
        "outer_geometry_matches_static_preflight": (
            static["collar_geometry"] == OUTER_GEOMETRY.to_dict()
        ),
        "outer_starts_beyond_measured_half_depth": (
            OUTER_GEOMETRY.inner_support_distance > maximum_half_depth
        ),
        "same_width_as_k1": (
            OUTER_GEOMETRY.support_width == INWARD_GEOMETRY.support_width
        ),
        "same_transition_as_k1": (
            OUTER_GEOMETRY.transition_width
            == INWARD_GEOMETRY.transition_width
        ),
        "same_mobility_contrast_as_k1": (
            OUTER_GEOMETRY.protected_mobility_factor
            == INWARD_GEOMETRY.protected_mobility_factor
        ),
        "budget_surplus_is_positive": relative_surplus > 0.0,
        "budget_surplus_is_bounded": (
            relative_surplus <= MAXIMUM_BUDGET_SURPLUS
        ),
        "factor_is_finite": bool(np.isfinite(outer).all()),
        "factor_is_positive": float(np.min(outer)) > 0.0,
        "factor_is_at_most_one": float(np.max(outer)) <= 1.0,
    }
    if not all(geometry_checks.values()):
        raise RuntimeError(
            f"outer placement definition failed: {geometry_checks}"
        )
    report = {
        "factor_semantics": "fixed_shape_outer_tubular_collars",
        "inward_geometry": INWARD_GEOMETRY.to_dict(),
        "outer_geometry": OUTER_GEOMETRY.to_dict(),
        "outward_center_shift": (
            OUTER_GEOMETRY.center_distance
            - INWARD_GEOMETRY.center_distance
        ),
        "outward_center_shift_over_R": (
            OUTER_GEOMETRY.center_distance
            - INWARD_GEOMETRY.center_distance
        )
        / radius,
        "maximum_measured_depletion_half_depth": maximum_half_depth,
        "outer_start_clearance_from_half_depth": (
            OUTER_GEOMETRY.inner_support_distance - maximum_half_depth
        ),
        "inward_surface_weighted_deficit": inward_budget,
        "outer_surface_weighted_deficit": outer_budget,
        "relative_outer_budget_surplus": relative_surplus,
        "maximum_allowed_budget_surplus": MAXIMUM_BUDGET_SURPLUS,
        "minimum": float(np.min(outer)),
        "maximum": float(np.max(outer)),
        "support_cell_fraction": float(
            np.mean(outer < 1.0 - 1.0e-14)
        ),
        "shape": list(outer.shape),
        "checks": geometry_checks,
        "static_summary": {
            "path": str(STATIC_SUMMARY_PATH.resolve()),
            "sha256": sha256_path(STATIC_SUMMARY_PATH),
        },
    }
    del mapped
    gc.collect()
    return outer, report


def _measured_step_seconds() -> float:
    values = []
    for case in ("K1", "K2"):
        summary = _load_json(REFERENCE_SUMMARIES[case])
        values.append(
            float(summary["execution"]["median_step_wall_seconds"])
        )
    return max(values)


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
    free_disk = int(
        __import__("shutil").disk_usage(output.parent).free
    )
    measured = _measured_step_seconds()
    expected_steps = max(0, EXPECTED_EVENT_MIDPOINT - current_step)
    horizon_steps = max(0, TARGET_STEP - current_step)
    expected_seconds = expected_steps * measured
    conservative_horizon_seconds = (
        horizon_steps * measured * RUNTIME_SAFETY_FACTOR
    )
    report = {
        "field_bytes": field_bytes,
        "remaining_regular_checkpoints": remaining_regular,
        "remaining_event_checkpoints": remaining_event,
        "emergency_checkpoint_allowance": 1,
        "minimum_disk_headroom_bytes": MINIMUM_DISK_HEADROOM_BYTES,
        "required_free_disk_bytes": required_disk,
        "free_disk_bytes": free_disk,
        "disk_sufficient": free_disk >= required_disk,
        "runtime": {
            "measured_median_step_seconds": measured,
            "expected_event_midpoint": EXPECTED_EVENT_MIDPOINT,
            "expected_remaining_steps": expected_steps,
            "expected_wall_seconds": expected_seconds,
            "expected_wall_minutes": expected_seconds / 60.0,
            "horizon_remaining_steps": horizon_steps,
            "safety_factor": RUNTIME_SAFETY_FACTOR,
            "conservative_horizon_wall_seconds": (
                conservative_horizon_seconds
            ),
            "conservative_horizon_wall_minutes": (
                conservative_horizon_seconds / 60.0
            ),
        },
        "runtime_below_two_hour_cap": (
            conservative_horizon_seconds < MAXIMUM_WALL_SECONDS
        ),
    }
    if not report["disk_sufficient"]:
        raise RuntimeError(f"disk preflight failed: {report}")
    if not report["runtime_below_two_hour_cap"]:
        raise RuntimeError(f"runtime preflight failed: {report}")
    return report


def _contract(factor_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": "fixed_shape_outer_placement_lifetime_control",
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
        "factor": factor_report,
        "precursors": verify_precursors(),
        "decision_criteria": {
            "minimum_k1_relative_lifetime_advantage": 0.10,
            "minimum_k1_absolute_lifetime_advantage": 100,
            "outer_has_conservative_budget_surplus": True,
            "no_treatment_edge_artifact_required": True,
        },
        "provenance": {
            "runner_sha256": sha256_path(SOURCE_PATH),
            "scientific_contract_sha256": sha256_path(CONTRACT_PATH),
            "prior_readout_sha256": sha256_path(READOUT_PATH),
            "geometry_sha256": sha256_path(
                Path(selective_geometry.__file__).resolve()
            ),
            "selective_model_sha256": sha256_path(
                Path(selective_model.__file__).resolve()
            ),
            "selective_analysis_sha256": sha256_path(
                Path(selective_analysis.__file__).resolve()
            ),
            "lifetime_runner_sha256": sha256_path(
                Path(lifetime_runner.__file__).resolve()
            ),
            "roy_model_sha256": sha256_path(
                Path(roy_model.__file__).resolve()
            ),
            "python": sys.version,
            "numpy": np.__version__,
        },
        "scope": {
            "one_outer_placement_control": True,
            "placement_sweep_authorized": False,
            "seed_ensemble_authorized": False,
            "convergence_family_authorized": False,
            "part_two_authorized": False,
        },
    }


def _terminal_summary(
    output: Path,
    contract: dict[str, Any],
    status: dict[str, Any],
) -> dict[str, Any]:
    terminal = status["status"]
    event = status["event_assessment"]
    if terminal == "completed" and event.get("detected") is True:
        classification = "outer_fixed_shape_robust_first_event"
    elif terminal == "completed":
        classification = "outer_fixed_shape_t2000_censored"
    else:
        classification = f"outer_fixed_shape_{terminal}"
    times = status["step_wall_seconds"]
    summary = {
        "status": terminal,
        "classification": classification,
        "case": "outer_fixed_shape",
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
            "one_outer_control_completed": True,
            "automatic_follow_on_performed": False,
            "placement_sweep_started": False,
            "seed_ensemble_started": False,
            "part_two_started": False,
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
    factor, factor_report = derive_outer_factor()
    contract = _contract(factor_report)
    if resume:
        if not output.is_dir():
            raise FileNotFoundError("resume output directory is absent")
    else:
        reserve_output_directory(output)
    lock = acquire_output_lock(output)
    try:
        if resume:
            stored_contract = _load_json(output / "contract.json")
            lifetime_runner._verify_resume_contract(
                stored_contract, contract
            )
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
            initial = lifetime_runner._diagnostic_record(
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
            lifetime_runner._write_status(
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
                record = lifetime_runner._diagnostic_record(
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
                    lifetime_runner._milestone(output, field, step=step)
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
                    stream=STREAM_NAME,
                )
                checkpoints.append(checkpoint)
                if event_checkpoint_due:
                    event_checkpoint_count += 1

            if step % DIAGNOSTIC_INTERVAL == 0 or checkpoint_due:
                lifetime_runner._write_status(
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
                            "case": "outer_fixed_shape",
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
        final_status = lifetime_runner._write_status(
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


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--acknowledge-bounded-runtime",
        action="store_true",
        help="Required for the approximately 80--95 minute run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    if arguments.launch and not arguments.acknowledge_bounded_runtime:
        raise RuntimeError(
            "fresh launch requires --acknowledge-bounded-runtime"
        )
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
            arguments.output.resolve(),
            resume=arguments.resume,
            stop_requested=stop_requested,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "classification": result["classification"],
                    "completed_step": result["completed_step"],
                    "output": str(arguments.output.resolve()),
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
