#!/usr/bin/env python3
"""Run the frozen seed-104729 untreated/c34p5 pair at physical dt=0.5."""

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
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy

from ..roy_2021_reproduction import model as roy_model
from ..roy_2021_reproduction.model import FROZEN_DEG90
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
from . import geometry as selective_geometry
from . import model as selective_model
from . import position_scan_protocol
from . import run_position_scan_case as case_engine
from . import run_timestep_refinement_preflight as preflight_runner
from . import timestep_refinement_protocol as protocol
from .geometry import (
    four_arm_tubular_collar_factor,
    surface_weighted_mobility_deficit,
)
from .model import SpatialMobilityRoySolver


SOURCE_PATH = Path(__file__).resolve()
PROTOCOL_NAME = "paired_timestep_refinement_dt0p5_v1"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_preflight() -> dict[str, Any]:
    """Require the immutable zero-evolution preflight before allocation."""

    path = protocol.PREFLIGHT_OUTPUT / "summary.json"
    report = _load_json(path)
    checks = {
        "status": report.get("status") == "completed",
        "classification": (
            report.get("classification")
            == "paired_dt0p5_zero_evolution_preflight_passed"
        ),
        "zero_evolution": (
            report.get("simulation_proposals_performed") == 0
            and report["scope"].get("zero_solver_evolution") is True
        ),
        "all_checks": all(report["checks"].values()),
        "seed": int(report.get("seed", -1)) == protocol.SEED,
        "case_order": tuple(
            item["case_id"] for item in report["cases"]
        )
        == protocol.CASE_ORDER,
        "source_field": (
            report["source"]["field_sha256"]
            == protocol.SOURCE_FIELD_SHA256
        ),
        "source_fingerprint": (
            report["source"]["field_fingerprint"]
            == protocol.SOURCE_FINGERPRINT
        ),
        "preflight_runner": (
            report["provenance"]["preflight_runner_sha256"]
            == sha256_path(Path(preflight_runner.__file__).resolve())
        ),
        "pair_runner": (
            report["provenance"]["pair_runner_sha256"]
            == sha256_path(SOURCE_PATH)
        ),
        "protocol": (
            report["provenance"]["protocol_sha256"]
            == sha256_path(Path(protocol.__file__).resolve())
        ),
        "scientific_contract": (
            report["provenance"]["scientific_contract_sha256"]
            == sha256_path(protocol.CONTRACT_PATH)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"timestep-refinement preflight verification failed: {checks}"
        )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_path(path),
        "checks": checks,
        "report": report,
    }


def source_checkpoint_entry(
    preflight: dict[str, Any],
) -> dict[str, Any]:
    """Build the storage-loader entry from the verified preflight."""

    source = preflight["report"]["source"]
    return {
        "step": int(protocol.SOURCE_PHYSICAL_TIME),
        "stream": "paired_source",
        "kinds": ["conditioned_source"],
        "field_path": source["field_path"],
        "field_bytes": source["field_bytes"],
        "field_sha256": source["field_sha256"],
        "field_fingerprint": source["field_fingerprint"],
        "shape": source["shape"],
        "dtype": source["dtype"],
        "finite": True,
        "metadata_path": source["metadata_path"],
        "metadata_sha256": source["metadata_sha256"],
    }


def build_factor(
    case: protocol.RefinementCase,
    source_field: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the unchanged factor from the common source field."""

    if case.untreated:
        return np.ones((1,), dtype=np.float64), {
            "case_id": case.case_id,
            "untreated": True,
            "center": None,
            "minimum": 1.0,
            "maximum": 1.0,
            "surface_weighted_mobility_deficit": 0.0,
            "fixed_without_retuning": True,
        }
    position = position_scan_protocol.CASES_BY_CENTER[34.5]
    factor = four_arm_tubular_collar_factor(
        FROZEN_DEG90.lattice,
        position.geometry,
        radius=position_scan_protocol.RADIUS,
        interface_width=position_scan_protocol.INTERFACE_WIDTH,
    )
    budget = surface_weighted_mobility_deficit(
        source_field,
        factor,
        cell_volume=FROZEN_DEG90.lattice.cell_volume,
    )
    return factor, {
        **position_scan_protocol.static_case_report(position),
        "case_id": case.case_id,
        "untreated": False,
        "surface_weighted_mobility_deficit": float(budget),
        "factor_shape": list(factor.shape),
        "factor_minimum": float(np.min(factor)),
        "factor_maximum": float(np.max(factor)),
        "fixed_without_retuning": True,
    }


def _diagnostic_record(
    field: np.ndarray,
    solver: SpatialMobilityRoySolver,
    *,
    proposal_count: int,
    initial_mass: float,
    elapsed_wall_seconds: float,
    include_energy: bool,
) -> dict[str, Any]:
    """Record physical time explicitly while reusing the one-arm detector."""

    physical_time = protocol.physical_time_for_proposal_count(
        proposal_count
    )
    rounded = round(physical_time)
    if abs(physical_time - rounded) > 1.0e-12:
        raise RuntimeError(
            "scheduled diagnostic is not on an integer physical time"
        )
    record = production_helper._diagnostic_record(
        field,
        solver,
        step=int(rounded),
        initial_mass=initial_mass,
        elapsed_wall_seconds=elapsed_wall_seconds,
        include_energy=include_energy,
    )
    # ``step`` remains the legacy detector key and denotes physical time.
    record["physical_time"] = physical_time
    record["continuation_proposal_count"] = int(proposal_count)
    record["physical_timestep"] = protocol.PHYSICAL_TIMESTEP
    record["step_semantics"] = "physical_time"
    return record


def event_from_records(
    records: list[dict[str, Any]],
    *,
    latched: dict[str, Any] | None,
) -> dict[str, Any]:
    """Apply the existing persistent single-arm detector at 10 time units."""

    return production_helper._event_from_records(
        records, latched=latched
    )


def _case_contract(
    case: protocol.RefinementCase,
    preflight: dict[str, Any],
    factor_report: dict[str, Any],
) -> dict[str, Any]:
    parameters = replace(
        FROZEN_DEG90.parameters,
        timestep=protocol.PHYSICAL_TIMESTEP,
    )
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "seed": protocol.SEED,
        "case": case.case_id,
        "case_definition": case.to_dict(),
        "factor": factor_report,
        "clock": {
            "source_physical_time": protocol.SOURCE_PHYSICAL_TIME,
            "target_physical_time": protocol.TARGET_PHYSICAL_TIME,
            "physical_timestep": protocol.PHYSICAL_TIMESTEP,
            "target_continuation_proposal_count": (
                protocol.TARGET_PROPOSAL_COUNT
            ),
            "diagnostic_physical_interval": (
                protocol.DIAGNOSTIC_PHYSICAL_INTERVAL
            ),
            "diagnostic_proposal_interval": (
                protocol.DIAGNOSTIC_PROPOSAL_INTERVAL
            ),
            "energy_physical_interval": (
                protocol.ENERGY_PHYSICAL_INTERVAL
            ),
            "checkpoint_step_semantics": (
                "continuation_proposal_count"
            ),
        },
        "parameters": asdict(parameters),
        "source": preflight["report"]["source"],
        "preflight": {
            "path": preflight["path"],
            "sha256": preflight["sha256"],
        },
        "event_detector": {
            "type": "persistent_single_arm_event",
            "required_records": 3,
            "diagnostic_interval_in_physical_time": 10,
            "maximum_gap_displacement": (
                position_scan_protocol.INTERFACE_WIDTH
            ),
            "same_detector_as_dt1_pair": True,
        },
        "regular_checkpoint_physical_times": list(
            protocol.REGULAR_CHECKPOINT_PHYSICAL_TIMES
        ),
        "maximum_event_checkpoints": (
            protocol.MAXIMUM_EVENT_CHECKPOINTS
        ),
        "maximum_case_wall_seconds": (
            protocol.MAXIMUM_CASE_WALL_SECONDS
        ),
        "provenance": {
            "runner_sha256": sha256_path(SOURCE_PATH),
            "preflight_runner_sha256": sha256_path(
                Path(preflight_runner.__file__).resolve()
            ),
            "protocol_sha256": sha256_path(
                Path(protocol.__file__).resolve()
            ),
            "scientific_contract_sha256": sha256_path(
                protocol.CONTRACT_PATH
            ),
            "geometry_sha256": sha256_path(
                Path(selective_geometry.__file__).resolve()
            ),
            "selective_model_sha256": sha256_path(
                Path(selective_model.__file__).resolve()
            ),
            "roy_model_sha256": sha256_path(
                Path(roy_model.__file__).resolve()
            ),
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "scope": {
            "one_preregistered_timestep_refinement_case": True,
            "position_search_authorized": False,
            "parameter_tuning_authorized": False,
            "follow_on_authorized": False,
        },
    }


def _write_case_status(
    output: Path,
    *,
    status: str,
    stop_reason: str | None,
    proposal_count: int,
    records: list[dict[str, Any]],
    proposal_wall_seconds: list[float],
    checkpoints: list[dict[str, Any]],
    event: dict[str, Any],
    prior_energy: float | None,
    elapsed_wall_seconds: float,
    factor_report: dict[str, Any],
) -> dict[str, Any]:
    report = {
        "status": status,
        "stop_reason": stop_reason,
        "continuation_proposal_count": proposal_count,
        "physical_time": protocol.physical_time_for_proposal_count(
            proposal_count
        ),
        "records": records,
        "proposal_wall_seconds": proposal_wall_seconds,
        "checkpoints": checkpoints,
        "event_assessment": event,
        "last_energy": prior_energy,
        "elapsed_wall_seconds": elapsed_wall_seconds,
        "factor": factor_report,
        "pid": os.getpid(),
        "last_update_unix_time": time.time(),
    }
    atomic_json(output / "run_status.json", report)
    return report


def _checkpoint(
    output: Path,
    field: np.ndarray,
    *,
    proposal_count: int,
    kinds: tuple[str, ...],
) -> dict[str, Any]:
    entry = write_checkpoint(
        output,
        field,
        step=proposal_count,
        kinds=kinds,
        stream="dt_refinement",
    )
    entry["proposal_count"] = int(proposal_count)
    entry["physical_time"] = (
        protocol.physical_time_for_proposal_count(proposal_count)
    )
    entry["step_semantics"] = "continuation_proposal_count"
    return entry


def _case_summary(
    output: Path,
    *,
    contract: dict[str, Any],
    status: dict[str, Any],
) -> dict[str, Any]:
    times = status["proposal_wall_seconds"]
    event = status["event_assessment"]
    if status["status"] == "completed" and event.get("detected") is True:
        classification = (
            f"{contract['case']}_dt0p5_robust_single_arm_event"
        )
    elif status["status"] == "completed":
        classification = f"{contract['case']}_dt0p5_censored"
    else:
        classification = f"{contract['case']}_dt0p5_{status['status']}"
    summary = {
        "schema_version": 1,
        "status": status["status"],
        "classification": classification,
        "case": contract["case"],
        "stop_reason": status["stop_reason"],
        "completed_physical_time": status["physical_time"],
        "continuation_proposal_count": status[
            "continuation_proposal_count"
        ],
        "event_assessment": event,
        "records": status["records"],
        "checkpoints": status["checkpoints"],
        "execution": {
            "elapsed_wall_seconds": status["elapsed_wall_seconds"],
            "accepted_continuation_proposals": status[
                "continuation_proposal_count"
            ],
            "median_proposal_wall_seconds": (
                float(statistics.median(times)) if times else None
            ),
            "maximum_proposal_wall_seconds": max(times) if times else None,
            "physical_timestep": protocol.PHYSICAL_TIMESTEP,
        },
        "contract": contract,
        "scope": {
            "scientific_interpretation_pending_pair": True,
            "follow_on_started": False,
        },
    }
    atomic_json(output / "summary.json", summary)
    return summary


def _verify_case_resume_contract(
    stored: dict[str, Any],
    current: dict[str, Any],
) -> None:
    if stored != current:
        raise RuntimeError("refinement resume contract is not identical")


def run_case(
    case: protocol.RefinementCase,
    output: Path,
    *,
    preflight: dict[str, Any],
    resume: bool,
    stop_requested: dict[str, int | None],
) -> dict[str, Any]:
    """Run or resume one member; the pair driver calls this sequentially."""

    source_entry = source_checkpoint_entry(preflight)
    source_field = load_checkpoint(source_entry)
    factor, factor_report = build_factor(case, source_field)
    contract = _case_contract(case, preflight, factor_report)
    if resume:
        if not output.is_dir():
            raise FileNotFoundError(f"case output is absent: {output}")
        _verify_case_resume_contract(
            _load_json(output / "contract.json"), contract
        )
        retained = _load_json(output / "run_status.json")
        if retained["status"] not in {"interrupted", "wall_cap"}:
            raise RuntimeError(
                f"case is not resumable: {retained['status']}"
            )
        checkpoints = list(retained["checkpoints"])
        if not checkpoints:
            raise RuntimeError("resumable case has no checkpoint")
        latest = checkpoints[-1]
        field = load_checkpoint(latest)
        proposal_count = int(latest["proposal_count"])
        physical_time = protocol.physical_time_for_proposal_count(
            proposal_count
        )
        records = [
            item
            for item in retained["records"]
            if float(item["physical_time"]) <= physical_time
        ]
        proposal_wall_seconds = list(
            retained["proposal_wall_seconds"]
        )[:proposal_count]
        event = event_from_records(records, latched=None)
        prior_energy = next(
            (
                float(item["free_energy"])
                for item in reversed(records)
                if item.get("free_energy") is not None
            ),
            None,
        )
        prior_elapsed = float(
            latest.get(
                "elapsed_wall_seconds",
                retained["elapsed_wall_seconds"],
            )
        )
    else:
        reserve_output_directory(output)
        atomic_json(output / "contract.json", contract)
        field = np.asarray(source_field, dtype=np.float64)
        proposal_count = 0
        records: list[dict[str, Any]] = []
        proposal_wall_seconds: list[float] = []
        checkpoints: list[dict[str, Any]] = []
        event = event_from_records(records, latched=None)
        prior_energy = None
        prior_elapsed = 0.0
    del source_field
    gc.collect()

    parameters = replace(
        FROZEN_DEG90.parameters,
        timestep=protocol.PHYSICAL_TIMESTEP,
    )
    solver = SpatialMobilityRoySolver(
        FROZEN_DEG90.lattice,
        parameters,
        factor,
        fft_workers=protocol.FFT_WORKERS,
    )
    del factor
    gc.collect()
    session_started = time.perf_counter()
    initial_mass = protocol.SOURCE_INITIAL_MASS
    terminal_status: str | None = None
    stop_reason: str | None = None

    if not records:
        initial = _diagnostic_record(
            field,
            solver,
            proposal_count=proposal_count,
            initial_mass=initial_mass,
            elapsed_wall_seconds=prior_elapsed,
            include_energy=True,
        )
        records.append(initial)
        prior_energy = float(initial["free_energy"])
        event = event_from_records(records, latched=event)
        _write_case_status(
            output,
            status="running",
            stop_reason=None,
            proposal_count=proposal_count,
            records=records,
            proposal_wall_seconds=proposal_wall_seconds,
            checkpoints=checkpoints,
            event=event,
            prior_energy=prior_energy,
            elapsed_wall_seconds=prior_elapsed,
            factor_report=factor_report,
        )

    event_checkpoint_count = sum(
        "event_candidate" in item.get("kinds", [])
        for item in checkpoints
    )
    while proposal_count < protocol.TARGET_PROPOSAL_COUNT:
        started = time.perf_counter()
        field = solver.propose_step(field)
        proposal_wall_seconds.append(
            float(time.perf_counter() - started)
        )
        proposal_count += 1
        diagnostic_due = (
            proposal_count % protocol.DIAGNOSTIC_PROPOSAL_INTERVAL
            == 0
        )
        candidate_now = False
        health_reason: str | None = None
        if diagnostic_due:
            elapsed = (
                prior_elapsed + time.perf_counter() - session_started
            )
            record = _diagnostic_record(
                field,
                solver,
                proposal_count=proposal_count,
                initial_mass=initial_mass,
                elapsed_wall_seconds=elapsed,
                include_energy=(
                    proposal_count
                    % protocol.ENERGY_PROPOSAL_INTERVAL
                    == 0
                ),
            )
            records.append(record)
            health_reason = case_engine.health_stop_reason(
                record, previous_energy=prior_energy
            )
            if record["free_energy"] is not None:
                prior_energy = float(record["free_energy"])
            event = event_from_records(records, latched=event)
            candidate_now = bool(record["pinches"]["candidate"])
            if health_reason is not None:
                terminal_status = "numerically_aborted"
                stop_reason = health_reason
            elif event.get("detected") is True:
                terminal_status = "completed"
                stop_reason = "robust_single_arm_event_confirmed"

        signal_pending = stop_requested["signal"] is not None
        wall_cap_reached = (
            time.perf_counter() - session_started
            >= protocol.MAXIMUM_CASE_WALL_SECONDS
        )
        regular_due = (
            proposal_count
            in protocol.REGULAR_CHECKPOINT_PROPOSAL_COUNTS
        )
        event_due = (
            candidate_now
            and event_checkpoint_count
            < protocol.MAXIMUM_EVENT_CHECKPOINTS
        )
        target_reached = (
            proposal_count == protocol.TARGET_PROPOSAL_COUNT
        )
        checkpoint_due = bool(
            regular_due
            or event_due
            or signal_pending
            or wall_cap_reached
            or terminal_status is not None
            or target_reached
        )
        elapsed = (
            prior_elapsed + time.perf_counter() - session_started
        )
        if checkpoint_due:
            kinds: list[str] = []
            if regular_due:
                kinds.append("regular")
            if event_due:
                kinds.append("event_candidate")
            if event.get("detected") is True:
                kinds.append("event_confirmation")
            if signal_pending:
                kinds.append("signal")
            if wall_cap_reached:
                kinds.append("wall_cap")
            if terminal_status == "numerically_aborted":
                kinds.append("numerical_abort")
            if target_reached:
                kinds.append("target")
            checkpoint = _checkpoint(
                output,
                field,
                proposal_count=proposal_count,
                kinds=tuple(kinds),
            )
            checkpoint["elapsed_wall_seconds"] = elapsed
            checkpoints.append(checkpoint)
            if event_due:
                event_checkpoint_count += 1

        if (
            diagnostic_due
            or checkpoint_due
        ):
            _write_case_status(
                output,
                status=terminal_status or "running",
                stop_reason=stop_reason,
                proposal_count=proposal_count,
                records=records,
                proposal_wall_seconds=proposal_wall_seconds,
                checkpoints=checkpoints,
                event=event,
                prior_energy=prior_energy,
                elapsed_wall_seconds=elapsed,
                factor_report=factor_report,
            )
        physical_time = protocol.physical_time_for_proposal_count(
            proposal_count
        )
        if (
            diagnostic_due
            and (
                int(round(physical_time)) % 100 == 0
                or candidate_now
            )
        ):
            print(
                json.dumps(
                    {
                        "status": "running",
                        "case": case.case_id,
                        "physical_time": physical_time,
                        "continuation_proposal_count": proposal_count,
                        "event_detected": event.get("detected"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if terminal_status is not None:
            break
        if target_reached:
            terminal_status = "completed"
            stop_reason = "fixed_t2000_target_reached_without_event"
            break
        if signal_pending:
            terminal_status = "interrupted"
            stop_reason = (
                f"deferred_signal_{stop_requested['signal']}"
            )
            break
        if wall_cap_reached:
            terminal_status = "wall_cap"
            stop_reason = "four_hour_case_wall_cap_reached"
            break

    elapsed_total = (
        prior_elapsed + time.perf_counter() - session_started
    )
    final_status = _write_case_status(
        output,
        status=str(terminal_status),
        stop_reason=stop_reason,
        proposal_count=proposal_count,
        records=records,
        proposal_wall_seconds=proposal_wall_seconds,
        checkpoints=checkpoints,
        event=event,
        prior_energy=prior_energy,
        elapsed_wall_seconds=elapsed_total,
        factor_report=factor_report,
    )
    summary = _case_summary(
        output, contract=contract, status=final_status
    )
    del solver, field
    gc.collect()
    return summary


def _event_midpoint(event: dict[str, Any]) -> float:
    bracket = event["event_bracket"]
    return 0.5 * (float(bracket[0]) + float(bracket[1]))


def natural_failure_mode(event: dict[str, Any]) -> dict[str, Any]:
    """Classify the frozen junction-adjacent first-event mode."""

    gaps = event.get("persistent_gaps") or []
    lower, upper = protocol.NATURAL_SITE_ABSOLUTE_BOUNDS
    gap_checks = [
        {
            "wire": gap.get("wire"),
            "first_midpoint": float(gap["first_midpoint"]),
            "wire_is_natural": (
                gap.get("wire") == protocol.NATURAL_WIRE
            ),
            "site_is_natural": (
                lower
                <= abs(float(gap["first_midpoint"]))
                <= upper
            ),
        }
        for gap in gaps
    ]
    passed = bool(
        event.get("detected") is True
        and gap_checks
        and all(
            item["wire_is_natural"] and item["site_is_natural"]
            for item in gap_checks
        )
    )
    return {
        "mode": (
            "natural_junction_adjacent" if passed else "other_or_absent"
        ),
        "passed": passed,
        "gap_checks": gap_checks,
    }


def evaluate_acceptance(
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate only the frozen, pre-result numerical criteria."""

    events = {
        case_id: summaries[case_id]["event_assessment"]
        for case_id in protocol.CASE_ORDER
    }
    modes = {
        case_id: natural_failure_mode(events[case_id])
        for case_id in protocol.CASE_ORDER
    }
    completed = {
        case_id: (
            summaries[case_id].get("status") == "completed"
            and events[case_id].get("detected") is True
        )
        for case_id in protocol.CASE_ORDER
    }
    midpoints = {
        case_id: (
            _event_midpoint(events[case_id])
            if completed[case_id]
            else None
        )
        for case_id in protocol.CASE_ORDER
    }
    individual: dict[str, Any] = {}
    for case_id in protocol.CASE_ORDER:
        midpoint = midpoints[case_id]
        reference = protocol.DT1_EVENT_MIDPOINTS[case_id]
        relative = (
            abs(float(midpoint) - reference) / abs(reference)
            if midpoint is not None
            else None
        )
        individual[case_id] = {
            "dt0p5_event_midpoint": midpoint,
            "dt1_event_midpoint": reference,
            "relative_error": relative,
            "maximum_relative_error": (
                protocol.MAXIMUM_INDIVIDUAL_RELATIVE_TIME_ERROR
            ),
            "passed": (
                relative is not None
                and relative
                <= protocol.MAXIMUM_INDIVIDUAL_RELATIVE_TIME_ERROR
            ),
        }
    paired_effect = (
        float(midpoints["c34p5"]) - float(midpoints["untreated"])
        if all(value is not None for value in midpoints.values())
        else None
    )
    effect_relative_error = (
        abs(paired_effect - protocol.DT1_PAIRED_EFFECT)
        / abs(protocol.DT1_PAIRED_EFFECT)
        if paired_effect is not None
        else None
    )
    bracket_order = bool(
        all(completed.values())
        and float(events["c34p5"]["event_bracket"][1])
        < float(events["untreated"]["event_bracket"][0])
    )
    checks = {
        "both_completed_with_events": all(completed.values()),
        "same_natural_failure_mode": (
            all(item["passed"] for item in modes.values())
            and len({item["mode"] for item in modes.values()}) == 1
        ),
        "c34_earlier_beyond_bracket_uncertainty": bracket_order,
        "individual_times_within_five_percent": all(
            item["passed"] for item in individual.values()
        ),
        "paired_effect_within_twenty_five_percent": (
            effect_relative_error is not None
            and effect_relative_error
            <= protocol.MAXIMUM_PAIRED_EFFECT_RELATIVE_ERROR
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failure_modes": modes,
        "individual_time_refinement": individual,
        "paired_effect": {
            "dt0p5": paired_effect,
            "dt1": protocol.DT1_PAIRED_EFFECT,
            "relative_error": effect_relative_error,
            "maximum_relative_error": (
                protocol.MAXIMUM_PAIRED_EFFECT_RELATIVE_ERROR
            ),
        },
        "event_brackets": {
            case_id: events[case_id].get("event_bracket")
            for case_id in protocol.CASE_ORDER
        },
    }


def _pair_contract(preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "seed": protocol.SEED,
        "case_order": list(protocol.CASE_ORDER),
        "execution": "strictly_sequential",
        "preflight": {
            "path": preflight["path"],
            "sha256": preflight["sha256"],
        },
        "runtime_projection": protocol.runtime_projection(),
        "storage_projection": protocol.storage_projection(),
        "acceptance": preflight["report"]["acceptance"],
        "provenance": {
            "runner_sha256": sha256_path(SOURCE_PATH),
            "protocol_sha256": sha256_path(
                Path(protocol.__file__).resolve()
            ),
            "scientific_contract_sha256": sha256_path(
                protocol.CONTRACT_PATH
            ),
        },
        "scope": {
            "only_two_frozen_cases": True,
            "parallel_case_execution_authorized": False,
            "follow_on_authorized": False,
        },
    }


def _write_pair_status(
    output: Path,
    *,
    status: str,
    next_case_index: int,
    cases: dict[str, Any],
    stop_reason: str | None,
) -> dict[str, Any]:
    report = {
        "status": status,
        "next_case_index": next_case_index,
        "case_order": list(protocol.CASE_ORDER),
        "cases": cases,
        "stop_reason": stop_reason,
        "pid": os.getpid(),
        "last_update_unix_time": time.time(),
    }
    atomic_json(output / "run_status.json", report)
    return report


def run_pair(
    output: Path,
    *,
    resume: bool,
    stop_requested: dict[str, int | None],
    case_executor: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute the pair in its frozen order, stopping on technical failure."""

    preflight = verify_preflight()
    storage = protocol.storage_projection()
    free = int(shutil.disk_usage(output.parent).free)
    if free < int(storage["required_free_disk_bytes"]):
        raise RuntimeError(
            "insufficient free disk for timestep-refinement pair: "
            f"required={storage['required_free_disk_bytes']} free={free}"
        )
    contract = _pair_contract(preflight)
    executor = case_executor or run_case
    if resume:
        if not output.is_dir():
            raise FileNotFoundError(f"pair output is absent: {output}")
        stored_contract = _load_json(output / "contract.json")
        if stored_contract != contract:
            raise RuntimeError("pair resume contract is not identical")
        retained = _load_json(output / "run_status.json")
        if retained["status"] not in {"interrupted", "wall_cap"}:
            raise RuntimeError(
                f"pair is not resumable: {retained['status']}"
            )
        case_reports = dict(retained["cases"])
        next_index = int(retained["next_case_index"])
    else:
        reserve_output_directory(output)
        atomic_json(output / "contract.json", contract)
        case_reports: dict[str, Any] = {}
        next_index = 0
        _write_pair_status(
            output,
            status="running",
            next_case_index=0,
            cases=case_reports,
            stop_reason=None,
        )
    lock = acquire_output_lock(output)
    try:
        for index in range(next_index, len(protocol.CASE_ORDER)):
            case_id = protocol.CASE_ORDER[index]
            case = protocol.validate_case(case_id)
            case_path = protocol.case_output(output, case_id)
            case_resume = case_path.is_dir()
            summary = executor(
                case,
                case_path,
                preflight=preflight,
                resume=case_resume,
                stop_requested=stop_requested,
            )
            case_reports[case_id] = {
                "status": summary["status"],
                "summary_path": str(
                    (case_path / "summary.json").resolve()
                ),
                "summary_sha256": (
                    sha256_path(case_path / "summary.json")
                    if (case_path / "summary.json").is_file()
                    else None
                ),
                "completed_physical_time": summary.get(
                    "completed_physical_time"
                ),
                "event_assessment": summary.get("event_assessment"),
            }
            if summary["status"] != "completed":
                status = _write_pair_status(
                    output,
                    status=summary["status"],
                    next_case_index=index,
                    cases=case_reports,
                    stop_reason=(
                        f"{case_id}:{summary.get('stop_reason')}"
                    ),
                )
                atomic_json(output / "summary.json", status)
                return status
            _write_pair_status(
                output,
                status="running",
                next_case_index=index + 1,
                cases=case_reports,
                stop_reason=None,
            )
            if stop_requested["signal"] is not None:
                status = _write_pair_status(
                    output,
                    status="interrupted",
                    next_case_index=index + 1,
                    cases=case_reports,
                    stop_reason=(
                        f"deferred_signal_{stop_requested['signal']}"
                    ),
                )
                atomic_json(output / "summary.json", status)
                return status

        full_summaries = {
            case_id: _load_json(
                protocol.case_output(output, case_id) / "summary.json"
            )
            for case_id in protocol.CASE_ORDER
        }
        acceptance = evaluate_acceptance(full_summaries)
        atomic_json(output / "acceptance.json", acceptance)
        final = {
            "schema_version": 1,
            "status": "completed",
            "classification": (
                "paired_dt0p5_validation_passed"
                if acceptance["passed"]
                else "paired_dt0p5_validation_failed"
            ),
            "case_order": list(protocol.CASE_ORDER),
            "cases": case_reports,
            "acceptance": acceptance,
            "contract": contract,
            "scope": {
                "follow_on_started": False,
            },
        }
        _write_pair_status(
            output,
            status="completed",
            next_case_index=len(protocol.CASE_ORDER),
            cases=case_reports,
            stop_reason="paired_acceptance_evaluated",
        )
        atomic_json(output / "summary.json", final)
        return final
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
        "--output",
        type=Path,
        default=protocol.DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--acknowledge-long-runtime",
        action="store_true",
        help=(
            "Required for the approximately 6.1 h / 5.1 GiB "
            "sequential validation."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    if not arguments.acknowledge_long_runtime:
        raise RuntimeError(
            "solver launch requires --acknowledge-long-runtime "
            "(~6.1 h sequential runtime; ~5.1 GiB projected fields)"
        )
    stop_requested: dict[str, int | None] = {"signal": None}

    def request_stop(signal_number: int, _frame: object) -> None:
        stop_requested["signal"] = signal_number

    handled = (signal.SIGINT, signal.SIGTERM)
    prior = {number: signal.getsignal(number) for number in handled}
    for number in handled:
        signal.signal(number, request_stop)
    try:
        result = run_pair(
            arguments.output.resolve(),
            resume=arguments.resume,
            stop_requested=stop_requested,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "classification": result.get("classification"),
                    "output": str(arguments.output.resolve()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        for number, handler in prior.items():
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
