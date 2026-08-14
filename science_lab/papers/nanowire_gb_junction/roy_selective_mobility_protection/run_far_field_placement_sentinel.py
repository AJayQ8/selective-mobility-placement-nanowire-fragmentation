#!/usr/bin/env python3
"""Run one preregistered far-field collar-placement sentinel."""

from __future__ import annotations

import argparse
import gc
import json
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
from ..roy_fixed_gb_bridge import (
    run_t100_checkpoint_source_localization_preflight as source_helper,
)
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
from . import run_localized_first_break_continuation as localized_runner
from . import run_outer_placement_lifetime_control as outer_runner
from .geometry import (
    FourArmCollarGeometry,
    four_arm_tubular_collar_factor,
    surface_weighted_mobility_deficit,
)
from .model import SpatialMobilityRoySolver
from .run_t100_tubular_position_diagnostic import INWARD_GEOMETRY


SOURCE_PATH = Path(__file__).resolve()
DIRECTORY = SOURCE_PATH.parent
CONTRACT_PATH = DIRECTORY / "FAR_FIELD_PLACEMENT_SENTINEL.md"
RESULTS = DIRECTORY / "results"
DEFAULT_OUTPUT = RESULTS / "far_field_placement_sentinel_v1"
REFERENCE_PATHS = {
    "thinning_zone": RESULTS / "k1_first_break_t3000_v1" / "summary.json",
    "adjacent_outer": (
        RESULTS / "outer_placement_first_break_t3000_v1" / "summary.json"
    ),
}

START_STEP = 100
TARGET_STEP = 3000
DIAGNOSTIC_INTERVAL = 10
ENERGY_INTERVAL = 50
MILESTONE_INTERVAL = 100
CHECKPOINT_STEPS = (
    500,
    900,
    1300,
    1700,
    2000,
    2200,
    2400,
    2600,
    2800,
    3000,
)
FFT_WORKERS = 12
MAXIMUM_WALL_SECONDS = 3.0 * 60.0 * 60.0
RUNTIME_SAFETY_FACTOR = 1.2
MINIMUM_DISK_HEADROOM_BYTES = 4 << 30
MAXIMUM_EVENT_CHECKPOINTS = 3
MAXIMUM_BUDGET_SURPLUS = 0.02
STREAM_NAME = "far_field_placement_sentinel"

RADIUS = 6.0
INTERFACE_WIDTH = float(np.sqrt(8.0))
RAYLEIGH_WAVELENGTH = float(2.0 * np.sqrt(2.0) * np.pi * RADIUS)
FURTHEST_OBSERVED_BREAK = 43.0
APPARENT_DOWNSTREAM_OFFSET = 15.0
PREDICTED_TRACKING_SITE = 85.25
FAR_GEOMETRY = FourArmCollarGeometry(58.5, 70.5, 3.0, 0.1)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verified_source_checkpoint_entry() -> dict[str, Any]:
    metadata_path = source_helper.T100_CHECKPOINT_METADATA_PATH
    metadata = _load_json(metadata_path)
    checks = {
        "step": int(metadata.get("step", -1)) == START_STEP,
        "fingerprint": (
            metadata.get("field_fingerprint")
            == source_helper.EXPECTED_T100_FINGERPRINT
        ),
        "field_path": (
            Path(str(metadata.get("field_path", ""))).resolve()
            == source_helper.T100_CHECKPOINT_PATH.resolve()
        ),
        "field_sha256": (
            metadata.get("field_sha256")
            == sha256_path(source_helper.T100_CHECKPOINT_PATH)
        ),
        "metadata_finite": metadata.get("finite") is True,
        "shape": (
            metadata.get("shape") == list(FROZEN_DEG90.lattice.shape)
        ),
        "dtype": metadata.get("dtype") == np.dtype(np.float64).str,
    }
    if not all(checks.values()):
        raise RuntimeError(f"time-100 source failed: {checks}")
    return {
        **metadata,
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": sha256_path(metadata_path),
        "kinds": ["imported_verified_t100_source"],
        "external_source": True,
        "elapsed_wall_seconds": 0.0,
        "source_checks": checks,
    }


def _load_run_checkpoint(entry: dict[str, Any]) -> np.ndarray:
    if entry.get("external_source") is not True:
        return load_checkpoint(entry)
    verified = verified_source_checkpoint_entry()
    stable_keys = (
        "step",
        "metadata_path",
        "metadata_sha256",
        "field_path",
        "field_sha256",
        "field_fingerprint",
    )
    checks = {
        key: entry.get(key) == verified.get(key)
        for key in stable_keys
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"imported time-100 checkpoint changed: {checks}"
        )
    # The immutable Roy checkpoint predates the sentinel storage module and
    # uses its legacy, independently verified fingerprint convention.
    return outer_runner._load_source_field()


def _retained_elapsed(
    checkpoint: dict[str, Any],
    records: list[dict[str, Any]],
) -> float:
    if checkpoint.get("elapsed_wall_seconds") is not None:
        return float(checkpoint["elapsed_wall_seconds"])
    checkpoint_step = int(checkpoint["step"])
    retained = [
        item
        for item in records
        if int(item["step"]) <= checkpoint_step
    ]
    if not retained:
        raise RuntimeError("no elapsed time retained with checkpoint")
    return float(retained[-1]["elapsed_wall_seconds"])


def verify_references() -> dict[str, Any]:
    expected = {
        "thinning_zone": (
            "K1_localized_robust_first_event",
            [2540, 2550],
            35.5,
        ),
        "adjacent_outer": (
            "outer_fixed_shape_continuation_robust_first_event",
            [2370, 2380],
            43.0,
        ),
    }
    reports: dict[str, Any] = {}
    for name, path in REFERENCE_PATHS.items():
        summary = _load_json(path)
        classification, bracket, site = expected[name]
        persistent = summary["event_assessment"]["persistent_gaps"]
        checks = {
            "completed": summary.get("status") == "completed",
            "classification": (
                summary.get("classification") == classification
            ),
            "event_detected": (
                summary["event_assessment"]["detected"] is True
            ),
            "event_bracket": (
                summary["event_assessment"]["event_bracket"] == bracket
            ),
            "one_persistent_gap": len(persistent) == 1,
            "event_site": (
                float(persistent[0]["confirmation_midpoint"]) == site
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(f"{name} reference failed: {checks}")
        reports[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_path(path),
            "event_bracket": bracket,
            "event_site": site,
            "checks": checks,
        }
    reports["untreated"] = (
        outer_runner.lifetime_runner.k1_runner.verify_untreated_reference()
    )
    return reports


def derive_far_factor() -> tuple[np.ndarray, dict[str, Any]]:
    mapped = np.load(
        source_helper.T100_CHECKPOINT_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )
    lattice = FROZEN_DEG90.lattice
    inward = four_arm_tubular_collar_factor(
        lattice,
        INWARD_GEOMETRY,
        radius=RADIUS,
        interface_width=INTERFACE_WIDTH,
    )
    inward_budget = surface_weighted_mobility_deficit(
        mapped,
        inward,
        cell_volume=lattice.cell_volume,
    )
    del inward
    gc.collect()
    far = four_arm_tubular_collar_factor(
        lattice,
        FAR_GEOMETRY,
        radius=RADIUS,
        interface_width=INTERFACE_WIDTH,
    )
    far_budget = surface_weighted_mobility_deficit(
        mapped,
        far,
        cell_volume=lattice.cell_volume,
    )
    relative_surplus = (far_budget - inward_budget) / inward_budget
    required_inner = (
        FURTHEST_OBSERVED_BREAK + APPARENT_DOWNSTREAM_OFFSET
    )
    checks = {
        "same_width_as_thinning_zone": (
            FAR_GEOMETRY.support_width == INWARD_GEOMETRY.support_width
        ),
        "same_transition_as_thinning_zone": (
            FAR_GEOMETRY.transition_width
            == INWARD_GEOMETRY.transition_width
        ),
        "same_mobility_as_thinning_zone": (
            FAR_GEOMETRY.protected_mobility_factor
            == INWARD_GEOMETRY.protected_mobility_factor
        ),
        "inner_edge_beyond_observed_influence": (
            FAR_GEOMETRY.inner_support_distance > required_inner
        ),
        "center_beyond_one_rayleigh_wavelength": (
            FAR_GEOMETRY.center_distance > RAYLEIGH_WAVELENGTH
        ),
        "budget_surplus_positive": relative_surplus > 0.0,
        "budget_surplus_bounded": (
            relative_surplus <= MAXIMUM_BUDGET_SURPLUS
        ),
        "factor_finite": bool(np.isfinite(far).all()),
        "factor_positive": float(np.min(far)) > 0.0,
        "factor_at_most_one": float(np.max(far)) <= 1.0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"far factor definition failed: {checks}")
    report = {
        "factor_semantics": "fixed_shape_far_field_tubular_collars",
        "thinning_zone_geometry": INWARD_GEOMETRY.to_dict(),
        "far_geometry": FAR_GEOMETRY.to_dict(),
        "rayleigh_wavelength": RAYLEIGH_WAVELENGTH,
        "center_over_R": FAR_GEOMETRY.center_distance / RADIUS,
        "center_over_rayleigh_wavelength": (
            FAR_GEOMETRY.center_distance / RAYLEIGH_WAVELENGTH
        ),
        "furthest_observed_break": FURTHEST_OBSERVED_BREAK,
        "apparent_downstream_offset": APPARENT_DOWNSTREAM_OFFSET,
        "required_inner_edge": required_inner,
        "inner_edge_clearance": (
            FAR_GEOMETRY.inner_support_distance - required_inner
        ),
        "predicted_tracking_site": PREDICTED_TRACKING_SITE,
        "thinning_zone_surface_weighted_deficit": inward_budget,
        "far_surface_weighted_deficit": far_budget,
        "relative_far_budget_surplus": relative_surplus,
        "maximum_allowed_budget_surplus": MAXIMUM_BUDGET_SURPLUS,
        "minimum": float(np.min(far)),
        "maximum": float(np.max(far)),
        "support_cell_fraction": float(
            np.mean(far < 1.0 - 1.0e-14)
        ),
        "shape": list(far.shape),
        "checks": checks,
    }
    del mapped
    gc.collect()
    return far, report


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
    source = _load_json(
        outer_runner.REFERENCE_SUMMARIES["K1"]
    )
    median = float(source["execution"]["median_step_wall_seconds"])
    remaining = TARGET_STEP - current_step
    projected = remaining * median * RUNTIME_SAFETY_FACTOR
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
            "reference_median_step_wall_seconds": median,
            "remaining_steps": remaining,
            "safety_factor": RUNTIME_SAFETY_FACTOR,
            "projected_wall_seconds": projected,
            "projected_wall_hours": projected / 3600.0,
        },
        "runtime_below_three_hour_cap": (
            projected < MAXIMUM_WALL_SECONDS
        ),
    }
    if not report["disk_sufficient"]:
        raise RuntimeError(f"disk preflight failed: {report}")
    if not report["runtime_below_three_hour_cap"]:
        raise RuntimeError(f"runtime preflight failed: {report}")
    return report


def _contract(factor_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": "far_field_placement_sentinel",
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
        "references": verify_references(),
        "provenance": {
            "runner_sha256": sha256_path(SOURCE_PATH),
            "scientific_contract_sha256": sha256_path(CONTRACT_PATH),
            "geometry_sha256": sha256_path(
                Path(selective_geometry.__file__).resolve()
            ),
            "selective_model_sha256": sha256_path(
                Path(selective_model.__file__).resolve()
            ),
            "selective_analysis_sha256": sha256_path(
                Path(selective_analysis.__file__).resolve()
            ),
            "localized_runner_sha256": sha256_path(
                Path(localized_runner.__file__).resolve()
            ),
            "roy_model_sha256": sha256_path(
                Path(roy_model.__file__).resolve()
            ),
            "python": sys.version,
            "numpy": np.__version__,
        },
        "scope": {
            "one_far_field_sentinel": True,
            "incremental_series_authorized": False,
            "seed_ensemble_authorized": False,
            "convergence_family_authorized": False,
            "part_two_authorized": False,
        },
    }


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
    if terminal == "completed" and event.get("detected") is True:
        classification = "far_field_robust_first_event"
    elif terminal == "completed":
        classification = "far_field_t3000_censored"
    else:
        classification = f"far_field_{terminal}"
    times = status["step_wall_seconds"]
    summary = {
        "status": terminal,
        "classification": classification,
        "case": "far_field",
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
            "one_far_field_sentinel_completed": True,
            "automatic_follow_on_performed": False,
            "incremental_series_started": False,
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
    factor, factor_report = derive_far_factor()
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
            localized_runner._verify_resume_contract(
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
            field = _load_run_checkpoint(checkpoints[-1])
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
            prior_elapsed = _retained_elapsed(
                checkpoints[-1], records
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
                "event_candidate" in item.get("kinds", [])
                for item in checkpoints
            )
        else:
            atomic_json(output / "contract.json", contract)
            source_checkpoint = verified_source_checkpoint_entry()
            field = _load_run_checkpoint(source_checkpoint)
            current_step = START_STEP
            records: list[dict[str, Any]] = []
            milestones: list[dict[str, Any]] = []
            step_wall_seconds: list[float] = []
            checkpoints: list[dict[str, Any]] = [source_checkpoint]
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
            initial = outer_runner.lifetime_runner._diagnostic_record(
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
                record = outer_runner.lifetime_runner._diagnostic_record(
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
                    localized_runner._milestone(output, field, step=step)
                )

            elapsed = (
                prior_elapsed + time.perf_counter() - session_started
            )
            signal_pending = stop_requested["signal"] is not None
            session_elapsed = time.perf_counter() - session_started
            wall_cap_reached = (
                session_elapsed >= MAXIMUM_WALL_SECONDS
            )
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
                checkpoint["elapsed_wall_seconds"] = elapsed
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
                            "case": "far_field",
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
                stop_reason = "three_hour_wall_cap_reached"
                break
            if step == TARGET_STEP:
                terminal_status = "completed"
                stop_reason = "fixed_t3000_target_reached"
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


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--acknowledge-long-runtime",
        action="store_true",
        help="Required for the approximately 2.4--2.8 hour run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    if arguments.launch and not arguments.acknowledge_long_runtime:
        raise RuntimeError(
            "fresh launch requires --acknowledge-long-runtime"
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
