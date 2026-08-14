#!/usr/bin/env python3
"""Run one frozen fixed-width mobility-collar position-scan case."""

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
import scipy

from ..roy_2021_reproduction import model as roy_model
from ..roy_2021_reproduction.model import FROZEN_DEG90
from ..roy_fixed_gb_bridge import (
    run_t100_checkpoint_source_localization_preflight as source_helper,
)
from ..roy_fixed_gb_bridge.provenance import sha256_path
from ..roy_gb_junction_sentinel import (
    run_t100_conditioned_production as production_helper,
)
from ..roy_gb_junction_sentinel import diagnostics as event_diagnostics
from ..roy_gb_junction_sentinel import storage as storage_module
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
from . import position_scan_protocol as protocol
from . import run_far_field_placement_sentinel as far_runner
from . import run_localized_first_break_continuation as localized_runner
from . import run_outer_placement_lifetime_control as outer_runner
from . import run_position_scan_preflight as scan_preflight
from .model import SpatialMobilityRoySolver


SOURCE_PATH = Path(__file__).resolve()
PREFLIGHT_SUMMARY_PATH = (
    scan_preflight.DEFAULT_OUTPUT / "summary.json"
)
PROTOCOL_NAME = "fixed_width_position_scan_case_v1"
PRODUCTION_MASS_DRIFT_LIMIT = 1.0e-4
FIELD_MAGNITUDE_LIMIT = 2.0
ENERGY_REBOUND_RELATIVE_LIMIT = 1.0e-6


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_summary_allows_resume(output: Path) -> None:
    """Permit session terminals, but never re-evolve a scientific terminal."""

    summary_path = output / "summary.json"
    if not summary_path.exists():
        return
    prior_summary = _load_json(summary_path)
    status = prior_summary.get("status")
    if status in {"completed", "numerically_aborted"}:
        raise RuntimeError(
            "nonresumable terminal summary already exists; "
            "verify/adopt it"
        )
    if status not in {"wall_cap", "interrupted"}:
        raise RuntimeError(
            f"existing summary has unknown resume status: {status}"
        )


def verified_preflight() -> dict[str, Any]:
    """Verify that the no-evolution five-case preflight is present."""

    summary = _load_json(PREFLIGHT_SUMMARY_PATH)
    live_source = far_runner.verified_source_checkpoint_entry()
    checks = {
        "status": summary.get("status") == "completed",
        "classification": (
            summary.get("classification")
            == "position_scan_preflight_passed"
        ),
        "passed": summary.get("passed") is True,
        "zero_steps": (
            summary["scope"].get("simulation_steps_performed") == 0
        ),
        "exact_centers": (
            tuple(item["center"] for item in summary["cases"])
            == protocol.NEW_CENTERS
        ),
        "all_checks": all(summary["checks"].values()),
        "preflight_runner_unchanged": (
            summary["provenance"]["preflight_runner_sha256"]
            == sha256_path(
                Path(scan_preflight.__file__).resolve()
            )
        ),
        "scientific_contract_unchanged": (
            summary["provenance"][
                "scientific_contract_sha256"
            ]
            == sha256_path(protocol.CONTRACT_PATH)
        ),
        "protocol_unchanged": (
            summary["provenance"]["protocol_sha256"]
            == sha256_path(Path(protocol.__file__).resolve())
        ),
        "geometry_unchanged": (
            summary["provenance"]["geometry_sha256"]
            == sha256_path(
                Path(selective_geometry.__file__).resolve()
            )
        ),
        "model_unchanged": (
            summary["provenance"]["model_sha256"]
            == sha256_path(
                Path(selective_model.__file__).resolve()
            )
        ),
        "roy_model_unchanged": (
            summary["provenance"]["roy_model_sha256"]
            == sha256_path(Path(roy_model.__file__).resolve())
        ),
        "reference_audit_unchanged": (
            summary["reference_budget"]["recorded_crosscheck"][
                "sha256"
            ]
            == sha256_path(scan_preflight.ANCHOR_AUDIT_PATH)
        ),
        "source_checkpoint_unchanged": (
            summary["source_checkpoint"]["field_sha256"]
            == live_source["field_sha256"]
            and summary["source_checkpoint"]["field_fingerprint"]
            == live_source["field_fingerprint"]
            and summary["source_checkpoint"]["metadata_sha256"]
            == live_source["metadata_sha256"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"position scan preflight failed: {checks}")
    return {
        "path": str(PREFLIGHT_SUMMARY_PATH.resolve()),
        "sha256": sha256_path(PREFLIGHT_SUMMARY_PATH),
        "checks": checks,
        "summary": summary,
    }


def preflight_case_report(
    case: protocol.PositionCase,
) -> dict[str, Any]:
    verified = verified_preflight()
    matches = [
        item
        for item in verified["summary"]["cases"]
        if float(item["center"]) == case.center
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"preflight has {len(matches)} entries for {case.slug}"
        )
    report = matches[0]
    if not all(report["checks"].values()):
        raise RuntimeError(f"case preflight checks failed: {case.slug}")
    return report


def derive_case_factor(
    case: protocol.PositionCase,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rebuild one factor and require exact agreement with the preflight."""

    preflight = verified_preflight()["summary"]
    factor, report = scan_preflight.derive_case_factor(
        case,
        reference_budget=preflight["reference_budget"][
            "surface_weighted_mobility_deficit"
        ],
    )
    frozen = preflight_case_report(case)
    scalar_checks = {
        "geometry": report["geometry"] == frozen["geometry"],
        "xi": report["xi"] == frozen["xi"],
        "budget": np.isclose(
            report["surface_weighted_mobility_deficit"],
            frozen["surface_weighted_mobility_deficit"],
            rtol=0.0,
            atol=1.0e-10,
        ),
        "relative_budget": np.isclose(
            report["relative_budget_vs_thinning_zone"],
            frozen["relative_budget_vs_thinning_zone"],
            rtol=0.0,
            atol=1.0e-12,
        ),
        "factor": report["factor"] == frozen["factor"],
    }
    if not all(scalar_checks.values()):
        raise RuntimeError(
            f"factor differs from frozen preflight: {scalar_checks}"
        )
    report["preflight_reproduction_checks"] = scalar_checks
    return factor, report


def resource_preflight(
    output: Path,
    *,
    current_step: int,
    existing_event_checkpoints: int = 0,
) -> dict[str, Any]:
    """Check one case while retaining the aggregate storage guard."""

    aggregate = scan_preflight.runtime_and_storage_preflight()
    field_bytes = int(FROZEN_DEG90.lattice.cell_count * 8 + 256)
    remaining_regular = sum(
        step > current_step for step in protocol.CHECKPOINT_STEPS
    )
    remaining_event = max(
        0,
        protocol.MAXIMUM_EVENT_CHECKPOINTS
        - existing_event_checkpoints,
    )
    required_disk = (
        (remaining_regular + remaining_event + 2) * field_bytes
        + protocol.MINIMUM_DISK_HEADROOM_BYTES
    )
    free_disk = int(shutil.disk_usage(output.parent).free)
    median = float(
        aggregate["slowest_reference_median_step_wall_seconds"]
    )
    remaining_steps = protocol.TARGET_STEP - current_step
    projected = (
        remaining_steps
        * median
        * protocol.RUNTIME_SAFETY_FACTOR
    )
    report = {
        "field_bytes": field_bytes,
        "remaining_regular_checkpoints": remaining_regular,
        "remaining_event_checkpoints": remaining_event,
        "wall_cap_or_signal_checkpoint_allowance": 1,
        "emergency_partial-write_headroom": 1,
        "minimum_disk_headroom_bytes": (
            protocol.MINIMUM_DISK_HEADROOM_BYTES
        ),
        "required_free_disk_bytes": required_disk,
        "free_disk_bytes": free_disk,
        "disk_sufficient": free_disk >= required_disk,
        "aggregate_preflight": {
            "path": str(PREFLIGHT_SUMMARY_PATH.resolve()),
            "sha256": sha256_path(PREFLIGHT_SUMMARY_PATH),
            "required_free_disk_bytes": aggregate["storage"][
                "required_free_disk_bytes"
            ],
            "free_disk_bytes_at_case_preflight": aggregate["storage"][
                "free_disk_bytes"
            ],
            "disk_sufficient": aggregate["storage"]["disk_sufficient"],
        },
        "runtime": {
            "reference_median_step_wall_seconds": median,
            "remaining_steps": remaining_steps,
            "safety_factor": protocol.RUNTIME_SAFETY_FACTOR,
            "projected_wall_seconds": projected,
            "projected_wall_hours": projected / 3600.0,
        },
        "runtime_below_session_cap": (
            projected < protocol.MAXIMUM_WALL_SECONDS
        ),
    }
    if not report["disk_sufficient"]:
        raise RuntimeError(f"case disk preflight failed: {report}")
    if not report["aggregate_preflight"]["disk_sufficient"]:
        raise RuntimeError(f"aggregate disk preflight failed: {report}")
    if not report["runtime_below_session_cap"]:
        raise RuntimeError(f"case runtime preflight failed: {report}")
    return report


def _position_contract(
    case: protocol.PositionCase,
    factor_report: dict[str, Any],
) -> dict[str, Any]:
    geometry = case.geometry
    return {
        "case_id": case.slug,
        "center_distance": case.center,
        "inner_support_distance": geometry.inner_support_distance,
        "outer_support_distance": geometry.outer_support_distance,
        "support_width": geometry.support_width,
        "transition_width": geometry.transition_width,
        "protected_mobility_factor": (
            geometry.protected_mobility_factor
        ),
        "xi": case.xi,
        "mobility_factor_at_depletion_minimum": (
            factor_report[
                "mobility_factor_at_depletion_minimum"
            ]
        ),
        "mobility_factor_at_untreated_failure_site": (
            factor_report[
                "mobility_factor_at_untreated_event_site"
            ]
        ),
        "surface_weighted_mobility_deficit": (
            factor_report["surface_weighted_mobility_deficit"]
        ),
        "relative_budget_vs_thinning_zone": (
            factor_report["relative_budget_vs_thinning_zone"]
        ),
        "predicted_mode": case.predicted_mode,
        "predicted_downstream_site": (
            case.predicted_downstream_site
        ),
        "prediction": case.prediction,
    }


def _contract(
    case: protocol.PositionCase,
    factor_report: dict[str, Any],
) -> dict[str, Any]:
    preflight = verified_preflight()
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "case": case.slug,
        "position": _position_contract(case, factor_report),
        "start_step": protocol.START_STEP,
        "target_step": protocol.TARGET_STEP,
        "diagnostic_interval": protocol.DIAGNOSTIC_INTERVAL,
        "energy_interval": protocol.ENERGY_INTERVAL,
        "milestone_interval": protocol.MILESTONE_INTERVAL,
        "checkpoint_steps": list(protocol.CHECKPOINT_STEPS),
        "fft_workers": protocol.FFT_WORKERS,
        "maximum_wall_seconds": protocol.MAXIMUM_WALL_SECONDS,
        "maximum_event_checkpoints": (
            protocol.MAXIMUM_EVENT_CHECKPOINTS
        ),
        "stop_on_robust_event": True,
        "source_checkpoint": {
            "path": str(
                source_helper.T100_CHECKPOINT_PATH.resolve()
            ),
            "sha256": sha256_path(
                source_helper.T100_CHECKPOINT_PATH
            ),
            "fingerprint": (
                source_helper.EXPECTED_T100_FINGERPRINT
            ),
            "global_step": protocol.START_STEP,
        },
        "references": far_runner.verify_references(),
        "preflight": {
            "path": preflight["path"],
            "sha256": preflight["sha256"],
        },
        "provenance": {
            "runner_sha256": sha256_path(SOURCE_PATH),
            "scientific_contract_sha256": sha256_path(
                protocol.CONTRACT_PATH
            ),
            "protocol_sha256": sha256_path(
                Path(protocol.__file__).resolve()
            ),
            "preflight_runner_sha256": sha256_path(
                Path(scan_preflight.__file__).resolve()
            ),
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
            "far_runner_sha256": sha256_path(
                Path(far_runner.__file__).resolve()
            ),
            "outer_runner_sha256": sha256_path(
                Path(outer_runner.__file__).resolve()
            ),
            "lifetime_runner_sha256": sha256_path(
                Path(
                    outer_runner.lifetime_runner.__file__
                ).resolve()
            ),
            "source_helper_sha256": sha256_path(
                Path(source_helper.__file__).resolve()
            ),
            "production_helper_sha256": sha256_path(
                Path(production_helper.__file__).resolve()
            ),
            "event_diagnostics_sha256": sha256_path(
                Path(event_diagnostics.__file__).resolve()
            ),
            "storage_sha256": sha256_path(
                Path(storage_module.__file__).resolve()
            ),
            "roy_model_sha256": sha256_path(
                Path(roy_model.__file__).resolve()
            ),
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "scope": {
            "one_frozen_position_case": True,
            "adaptive_refinement_authorized": False,
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
        "target_step": protocol.TARGET_STEP,
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
        classification = f"{case}_position_scan_robust_first_event"
    elif terminal == "completed":
        classification = f"{case}_position_scan_t3000_censored"
    else:
        classification = f"{case}_position_scan_{terminal}"
    times = status["step_wall_seconds"]
    summary = {
        "status": terminal,
        "classification": classification,
        "case": case,
        "stop_reason": status["stop_reason"],
        "completed_step": status["current_step"],
        "event_assessment": event,
        "milestones": status["milestones"],
        "records": status["records"],
        "checkpoints": status["checkpoints"],
        "execution": {
            "elapsed_wall_seconds": status[
                "elapsed_wall_seconds"
            ],
            "accepted_steps": (
                status["current_step"] - protocol.START_STEP
            ),
            "step_wall_seconds_count": len(times),
            "median_step_wall_seconds": (
                float(statistics.median(times)) if times else None
            ),
            "maximum_step_wall_seconds": max(times) if times else None,
        },
        "contract": contract,
        "scope": {
            "one_position_case_completed": terminal == "completed",
            "automatic_follow_on_performed_inside_case": False,
            "adaptive_refinement_started": False,
            "seed_ensemble_started": False,
            "convergence_family_started": False,
            "part_two_started": False,
            "scientific_interpretation_pending": True,
        },
    }
    atomic_json(output / "summary.json", summary)
    return summary


def terminal_decision(
    *,
    health_reason: str | None,
    event_detected: bool,
) -> tuple[str | None, str | None]:
    """Give numerical health precedence over event completion."""

    if health_reason is not None:
        return "numerically_aborted", health_reason
    if event_detected:
        return "completed", "robust_single_arm_event_confirmed"
    return None, None


def health_stop_reason(
    record: dict[str, Any],
    *,
    previous_energy: float | None,
) -> str | None:
    """Apply the specified health bounds during evolution."""

    base = production_helper._health_stop_reason(
        record, previous_energy=previous_energy
    )
    if base is not None:
        return base
    field = record["field"]
    if field["finite"] is not True:
        return "nonfinite_field"
    magnitude = max(
        abs(float(field["minimum"])),
        abs(float(field["maximum"])),
    )
    if magnitude > FIELD_MAGNITUDE_LIMIT:
        return "field_magnitude_limit_exceeded"
    if (
        abs(float(record["relative_mass_drift"]))
        > PRODUCTION_MASS_DRIFT_LIMIT
    ):
        return "mass_drift_limit_exceeded"
    return None


def records_health_stop_reason(
    records: list[dict[str, Any]],
) -> str | None:
    """Recheck retained records before resuming a possibly terminal run."""

    previous_energy: float | None = None
    for record in records:
        reason = health_stop_reason(
            record, previous_energy=previous_energy
        )
        if reason is not None:
            return reason
        if record.get("free_energy") is not None:
            previous_energy = float(record["free_energy"])
    return None


def preloop_terminal_state(
    *,
    current_step: int,
    records: list[dict[str, Any]],
    event: dict[str, Any],
    retained_terminal_status: str | None,
    retained_stop_reason: str | None,
) -> tuple[str | None, str | None]:
    """Finalize retained health/events/targets without another solver step."""

    if current_step > protocol.TARGET_STEP:
        raise RuntimeError("resume checkpoint lies beyond target")
    terminal_status, stop_reason = terminal_decision(
        health_reason=records_health_stop_reason(records),
        event_detected=bool(event.get("detected")),
    )
    if retained_terminal_status is not None:
        terminal_status = retained_terminal_status
        stop_reason = retained_stop_reason
    if (
        terminal_status is None
        and current_step == protocol.TARGET_STEP
    ):
        terminal_status = "completed"
        stop_reason = "fixed_t3000_target_reached"
    return terminal_status, stop_reason


def quarantine_unlisted_checkpoints(
    output: Path,
    listed: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Move crash-window checkpoint files aside before deterministic replay."""

    referenced = {
        Path(str(item[key])).resolve()
        for item in listed
        for key in ("field_path", "metadata_path")
        if item.get(key) is not None
    }
    metadata_paths = {
        path.resolve()
        for path in output.glob(
            "checkpoint-position_scan-step-*.json"
        )
    }
    field_paths = {
        path.resolve()
        for path in output.glob(
            "checkpoint-position_scan-step-*.npy"
        )
    }
    orphan_paths = (metadata_paths | field_paths) - referenced
    if not orphan_paths:
        return []
    quarantine = output / "orphaned_checkpoints"
    quarantine.mkdir(exist_ok=True)
    reports: list[dict[str, Any]] = []
    stems = sorted({path.stem for path in orphan_paths})
    # Remove the extension-independent suffix generated by Path.stem.
    normalized = sorted(
        {
            stem
            for stem in stems
            if stem.startswith("checkpoint-position_scan-step-")
        }
    )
    for stem in normalized:
        metadata_path = output / f"{stem}.json"
        field_path = output / f"{stem}.npy"
        present = [
            path
            for path in (field_path, metadata_path)
            if path.resolve() in orphan_paths
        ]
        verification = "partial_checkpoint_quarantined"
        if metadata_path in present and field_path in present:
            stored = _load_json(metadata_path)
            checks = {
                "stream": stored.get("stream") == "position_scan",
                "field_path": (
                    Path(str(stored.get("field_path", ""))).resolve()
                    == field_path.resolve()
                ),
                "field_sha256": (
                    stored.get("field_sha256")
                    == sha256_path(field_path)
                ),
            }
            if not all(checks.values()):
                raise RuntimeError(
                    f"orphan checkpoint verification failed: {checks}"
                )
            entry = {
                **stored,
                "metadata_path": str(metadata_path.resolve()),
                "metadata_sha256": sha256_path(metadata_path),
            }
            recovered_field = load_checkpoint(entry)
            del recovered_field
            gc.collect()
            verification = "verified_pair_quarantined_for_replay"
        moved: list[str] = []
        for path in present:
            destination = quarantine / path.name
            if destination.exists():
                raise FileExistsError(
                    f"orphan quarantine collision: {destination}"
                )
            path.replace(destination)
            moved.append(str(destination.resolve()))
        reports.append(
            {
                "stem": stem,
                "verification": verification,
                "moved_paths": moved,
            }
        )
    atomic_json(
        quarantine / "recovery.json",
        {
            "policy": (
                "unlisted crash-window files preserved before replay"
            ),
            "checkpoints": reports,
        },
    )
    return reports


def run(
    case: protocol.PositionCase,
    output: Path,
    *,
    resume: bool,
    stop_requested: dict[str, int | None],
) -> dict[str, Any]:
    """Evolve one frozen case; this function is called only after launch."""

    factor, factor_report = derive_case_factor(case)
    contract = _contract(case, factor_report)
    if resume:
        if not output.is_dir():
            raise FileNotFoundError("resume output directory is absent")
        verify_summary_allows_resume(output)
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
                "completed",
                "numerically_aborted",
            }:
                raise RuntimeError("run status is not resumable")
            checkpoints = [
                item
                for item in status["checkpoints"]
                if int(item["step"]) <= int(status["current_step"])
            ]
            if not checkpoints:
                raise RuntimeError("no verified continuation checkpoint")
            recovery = quarantine_unlisted_checkpoints(
                output, checkpoints
            )
            if recovery:
                factor_report["resume_recovery"] = recovery
            field = far_runner._load_run_checkpoint(checkpoints[-1])
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
                : current_step - protocol.START_STEP
            ]
            event = production_helper._event_from_records(
                records, latched=None
            )
            prior_elapsed = far_runner._retained_elapsed(
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
            retained_terminal_status = (
                status["status"]
                if status["status"]
                in {"completed", "numerically_aborted"}
                else None
            )
            retained_stop_reason = status.get("stop_reason")
        else:
            atomic_json(output / "contract.json", contract)
            source_checkpoint = (
                far_runner.verified_source_checkpoint_entry()
            )
            field = far_runner._load_run_checkpoint(source_checkpoint)
            current_step = protocol.START_STEP
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
            retained_terminal_status = None
            retained_stop_reason = None

        resource = resource_preflight(
            output,
            current_step=current_step,
            existing_event_checkpoints=event_checkpoint_count,
        )
        solver = SpatialMobilityRoySolver(
            FROZEN_DEG90.lattice,
            FROZEN_DEG90.parameters,
            factor,
            fft_workers=protocol.FFT_WORKERS,
        )
        del factor
        gc.collect()
        initial_mass = float(source_helper.EXPECTED_T100_MASS)
        session_started = time.perf_counter()

        if not records:
            initial = outer_runner.lifetime_runner._diagnostic_record(
                field,
                solver,
                step=protocol.START_STEP,
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
                current_step=protocol.START_STEP,
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

        terminal_status, stop_reason = preloop_terminal_state(
            current_step=current_step,
            records=records,
            event=event,
            retained_terminal_status=retained_terminal_status,
            retained_stop_reason=retained_stop_reason,
        )
        steps = (
            ()
            if terminal_status is not None
            else range(
                current_step + 1,
                protocol.TARGET_STEP + 1,
            )
        )
        for step in steps:
            step_started = time.perf_counter()
            field = solver.propose_step(field)
            step_wall_seconds.append(
                float(time.perf_counter() - step_started)
            )
            current_step = step
            nonfinite_reason = (
                None
                if np.isfinite(field).all()
                else "nonfinite_field"
            )

            candidate_now = False
            if (
                step % protocol.DIAGNOSTIC_INTERVAL == 0
                or nonfinite_reason is not None
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
                        step % protocol.ENERGY_INTERVAL == 0
                        or nonfinite_reason is not None
                    ),
                )
                records.append(record)
                health_reason = (
                    nonfinite_reason
                    or health_stop_reason(
                        record, previous_energy=prior_energy
                    )
                )
                if record["free_energy"] is not None:
                    prior_energy = float(record["free_energy"])
                event = production_helper._event_from_records(
                    records, latched=event
                )
                terminal_status, stop_reason = terminal_decision(
                    health_reason=health_reason,
                    event_detected=bool(event.get("detected")),
                )
                candidate_now = bool(record["pinches"]["candidate"])

            if (
                terminal_status is None
                and step % protocol.MILESTONE_INTERVAL == 0
            ):
                milestones.append(
                    localized_runner._milestone(
                        output, field, step=step
                    )
                )

            elapsed = (
                prior_elapsed + time.perf_counter() - session_started
            )
            signal_pending = stop_requested["signal"] is not None
            session_elapsed = time.perf_counter() - session_started
            wall_cap_reached = (
                session_elapsed >= protocol.MAXIMUM_WALL_SECONDS
            )
            event_checkpoint_due = bool(
                candidate_now
                and event_checkpoint_count
                < protocol.MAXIMUM_EVENT_CHECKPOINTS
            )
            regular_checkpoint_due = (
                step in protocol.CHECKPOINT_STEPS
            )
            checkpoint_due = bool(
                regular_checkpoint_due
                or event_checkpoint_due
                or signal_pending
                or wall_cap_reached
                or terminal_status is not None
                or step == protocol.TARGET_STEP
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
                if step == protocol.TARGET_STEP:
                    kinds.append("target")
                checkpoint = write_checkpoint(
                    output,
                    field,
                    step=step,
                    kinds=tuple(kinds),
                    stream=case.stream_name,
                )
                checkpoint["elapsed_wall_seconds"] = elapsed
                checkpoints.append(checkpoint)
                if event_checkpoint_due:
                    event_checkpoint_count += 1

            if (
                step % protocol.DIAGNOSTIC_INTERVAL == 0
                or checkpoint_due
            ):
                _write_status(
                    output,
                    terminal_status=(
                        terminal_status or "running"
                    ),
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
                            "case": case.slug,
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
            if step == protocol.TARGET_STEP:
                terminal_status = "completed"
                stop_reason = "fixed_t3000_target_reached"
                break
            if signal_pending:
                terminal_status = "interrupted"
                stop_reason = (
                    f"deferred_signal_{stop_requested['signal']}"
                )
                break
            if wall_cap_reached:
                terminal_status = "wall_cap"
                stop_reason = "three_hour_session_wall_cap_reached"
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
    parser.add_argument(
        "--center",
        type=float,
        choices=list(protocol.NEW_CENTERS),
        required=True,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--acknowledge-long-runtime",
        action="store_true",
        help="Required for a fresh approximately 1.5--2.9 hour case.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    case = protocol.case_for_center(arguments.center)
    output = (
        arguments.output.resolve()
        if arguments.output is not None
        else case.output.resolve()
    )
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
            case,
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
    finally:
        for number, handler in prior_handlers.items():
            signal.signal(number, handler)
    return {
        "completed": 0,
        "wall_cap": 3,
        "interrupted": 4,
        "numerically_aborted": 5,
    }.get(result["status"], 6)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
