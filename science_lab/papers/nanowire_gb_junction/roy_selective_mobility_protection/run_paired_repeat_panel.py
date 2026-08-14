#!/usr/bin/env python3
"""Preflight and run the frozen two-source, eight-trajectory repeat panel."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from ..roy_2021_reproduction.model import FROZEN_DEG90
from ..roy_fixed_gb_bridge.provenance import sha256_path
from ..roy_gb_junction_sentinel.storage import (
    acquire_output_lock,
    atomic_json,
    release_output_lock,
    reserve_output_directory,
)
from . import paired_repeat_protocol as protocol
from . import run_paired_repeat_case as case_runner
from . import run_paired_repeat_source as source_runner
from . import run_position_scan_case as case_engine
from . import run_position_scan_preflight as scan_preflight


SOURCE_PATH = Path(__file__).resolve()
REPO_ROOT = SOURCE_PATH.parents[4]
SOURCE_MODULE = (
    "science_lab.papers.nanowire_gb_junction."
    "roy_selective_mobility_protection.run_paired_repeat_source"
)
CASE_MODULE = (
    "science_lab.papers.nanowire_gb_junction."
    "roy_selective_mobility_protection.run_paired_repeat_case"
)
PANEL_MODULE = (
    "science_lab.papers.nanowire_gb_junction."
    "roy_selective_mobility_protection.run_paired_repeat_panel"
)
ACTIVE_CHILD: subprocess.Popen[Any] | None = None

EXISTING_SEED_PATHS = {
    "c18p5": (
        protocol.RESULTS / "position_scan_c18p5_v1" / "summary.json"
    ),
    "c26p5": (
        protocol.RESULTS / "position_scan_c26p5_v1" / "summary.json"
    ),
    "c34p5": (
        protocol.RESULTS / "position_scan_c34p5_v1" / "summary.json"
    ),
}
EXISTING_ANCHOR_AUDIT = (
    protocol.RESULTS
    / "position_anchor_mechanism_audit_v1"
    / "summary.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _active_simulation_processes() -> list[dict[str, Any]]:
    """Find only relevant nanowire production processes; never kill them."""

    completed = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        check=True,
        capture_output=True,
        text=True,
    )
    markers = (
        "run_position_scan_case",
        "run_paired_repeat_case",
        "run_paired_repeat_source",
    )
    reports: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        if any(marker in command for marker in markers):
            reports.append({"pid": pid, "command": command})
    return reports


def verify_existing_seed_anchor() -> dict[str, Any]:
    audit = _load_json(EXISTING_ANCHOR_AUDIT)
    summaries = {
        case_id: _load_json(path)
        for case_id, path in EXISTING_SEED_PATHS.items()
    }
    checks = {
        "anchor_audit_exists": EXISTING_ANCHOR_AUDIT.is_file(),
        "anchor_untreated_event": (
            audit["events"]["untreated"]["event_midpoint"] == 1715.0
        ),
        "three_position_summaries_completed": all(
            summary.get("status") == "completed"
            for summary in summaries.values()
        ),
        "three_position_events_detected": all(
            summary["event_assessment"].get("detected") is True
            for summary in summaries.values()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"existing seed anchor failed: {checks}")
    return {
        "seed": protocol.EXISTING_SEED,
        "checks": checks,
        "anchor_audit_path": str(EXISTING_ANCHOR_AUDIT.resolve()),
        "anchor_audit_sha256": sha256_path(EXISTING_ANCHOR_AUDIT),
        "position_summary_sha256": {
            case_id: sha256_path(path)
            for case_id, path in EXISTING_SEED_PATHS.items()
        },
    }


def preflight(output: Path) -> dict[str, Any]:
    """Perform no evolution; freeze measured runtime and disk bounds."""

    measured = scan_preflight.runtime_and_storage_preflight()
    slowest = float(
        measured["slowest_reference_median_step_wall_seconds"]
    )
    field_bytes = int(FROZEN_DEG90.lattice.cell_count * 8 + 256)
    checkpoints_per_case = (
        len(protocol.CHECKPOINT_STEPS)
        + protocol.MAXIMUM_EVENT_CHECKPOINTS
        + 2
    )
    retained_fields = (
        len(protocol.NEW_SEEDS)
        * (
            1
            + len(protocol.CASES) * checkpoints_per_case
        )
    )
    required_disk = (
        retained_fields * field_bytes
        + protocol.MINIMUM_DISK_HEADROOM_BYTES
    )
    free_disk = int(shutil.disk_usage(output.parent).free)
    planning_terminal = {
        "untreated": 1720,
        "c18p5": 2600,
        "c26p5": 2020,
        "c34p5": 1620,
    }
    planning_steps_per_seed = sum(
        step - protocol.SOURCE_TARGET_STEP
        for step in planning_terminal.values()
    ) + protocol.SOURCE_TARGET_STEP
    planning_steps = (
        len(protocol.NEW_SEEDS) * planning_steps_per_seed
    )
    full_steps = len(protocol.NEW_SEEDS) * (
        protocol.SOURCE_TARGET_STEP
        + len(protocol.CASES)
        * (protocol.TARGET_STEP - protocol.SOURCE_TARGET_STEP)
    )
    active = _active_simulation_processes()
    anchor = verify_existing_seed_anchor()
    checks = {
        "output_absent": not output.exists(),
        "scientific_contract_exists": protocol.CONTRACT_PATH.is_file(),
        "exact_two_new_seeds": len(protocol.NEW_SEEDS) == 2,
        "seeds_are_distinct": len(set(protocol.NEW_SEEDS)) == 2,
        "seed2292_not_rerun": (
            protocol.EXISTING_SEED not in protocol.NEW_SEEDS
        ),
        "exact_four_cases": (
            tuple(case.case_id for case in protocol.CASES)
            == protocol.CASE_ORDER
        ),
        "disk_sufficient": free_disk >= required_disk,
        "no_competing_nanowire_production": not active,
        "existing_anchor_verified": all(anchor["checks"].values()),
    }
    report = {
        "status": "preflight_passed" if all(checks.values()) else "failed",
        "checks": checks,
        "simulation_steps_performed": 0,
        "output": str(output.resolve()),
        "seeds": list(protocol.NEW_SEEDS),
        "case_order": list(protocol.CASE_ORDER),
        "existing_seed_anchor": anchor,
        "storage": {
            "field_checkpoint_bytes": field_bytes,
            "maximum_retained_field_count": retained_fields,
            "required_free_disk_bytes": required_disk,
            "required_free_disk_gib": required_disk / (1 << 30),
            "free_disk_bytes": free_disk,
            "free_disk_gib": free_disk / (1 << 30),
        },
        "runtime": {
            "reference_median_step_wall_seconds": slowest,
            "runtime_safety_factor": protocol.RUNTIME_SAFETY_FACTOR,
            "planning_terminal_steps": planning_terminal,
            "planning_accepted_steps": planning_steps,
            "planning_raw_hours": planning_steps * slowest / 3600.0,
            "planning_safeguarded_hours": (
                planning_steps
                * slowest
                * protocol.RUNTIME_SAFETY_FACTOR
                / 3600.0
            ),
            "full_horizon_accepted_steps": full_steps,
            "full_horizon_safeguarded_hours": (
                full_steps
                * slowest
                * protocol.RUNTIME_SAFETY_FACTOR
                / 3600.0
            ),
        },
        "active_relevant_processes": active,
        "provenance": {
            "panel_runner_sha256": sha256_path(SOURCE_PATH),
            "source_runner_sha256": sha256_path(
                Path(source_runner.__file__).resolve()
            ),
            "case_runner_sha256": sha256_path(
                Path(case_runner.__file__).resolve()
            ),
            "case_engine_sha256": sha256_path(
                Path(case_engine.__file__).resolve()
            ),
            "protocol_sha256": sha256_path(
                Path(protocol.__file__).resolve()
            ),
            "scientific_contract_sha256": sha256_path(
                protocol.CONTRACT_PATH
            ),
        },
    }
    if not all(checks.values()):
        raise RuntimeError(f"paired repeat preflight failed: {report}")
    return report


def _contract(preflight_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": "paired_repeat_panel_v1",
        "ordered_sources": [
            {
                "seed": seed,
                "source_output": str(
                    protocol.source_output(
                        Path(preflight_report["output"]), seed
                    ).resolve()
                ),
                "ordered_cases": [
                    {
                        "case_id": case.case_id,
                        "output": str(
                            protocol.case_output(
                                Path(preflight_report["output"]),
                                seed,
                                case.case_id,
                            ).resolve()
                        ),
                    }
                    for case in protocol.CASES
                ],
            }
            for seed in protocol.NEW_SEEDS
        ],
        "sequential_only": True,
        "stop_on_numerical_or_contract_failure": True,
        "continue_on_scientifically_unfavorable_result": True,
        "maximum_sessions_per_case": (
            protocol.MAXIMUM_SESSIONS_PER_CASE
        ),
        "preflight": preflight_report,
        "provenance": preflight_report["provenance"],
        "scope": {
            "two_new_sources_authorized": True,
            "eight_new_trajectories_authorized": True,
            "factual_aggregation_authorized": True,
            "wider_position_sweep_authorized": False,
            "convergence_family_authorized": False,
        },
    }


def _write_status(
    output: Path,
    *,
    status: str,
    current_seed: int | None,
    current_case: str | None,
    stop_reason: str | None,
    sources: dict[str, Any],
    cases: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    report = {
        "status": status,
        "current_seed": current_seed,
        "current_case": current_case,
        "stop_reason": stop_reason,
        "sources": sources,
        "cases": cases,
        "elapsed_wall_seconds": time.perf_counter() - started,
        "pid": os.getpid(),
        "last_update_unix_time": time.time(),
    }
    atomic_json(output / "run_status.json", report)
    return report


def _run_child(command: list[str]) -> int:
    global ACTIVE_CHILD
    print(
        json.dumps(
            {
                "event": "child_start",
                "command": command,
                "wall_time": time.time(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        start_new_session=True,
    )
    ACTIVE_CHILD = process
    try:
        return_code = int(process.wait())
    finally:
        ACTIVE_CHILD = None
    print(
        json.dumps(
            {
                "event": "child_exit",
                "return_code": return_code,
                "wall_time": time.time(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return return_code


def source_command(seed: int, output: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        SOURCE_MODULE,
        "--seed",
        str(seed),
        "--output",
        str(output.resolve()),
    ]


def case_command(
    seed: int,
    case_id: str,
    source_output: Path,
    output: Path,
    *,
    resume: bool,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        CASE_MODULE,
        "--resume" if resume else "--launch",
        "--seed",
        str(seed),
        "--case",
        case_id,
        "--source-output",
        str(source_output.resolve()),
        "--output",
        str(output.resolve()),
    ]


def _event_observation(summary: dict[str, Any]) -> dict[str, Any]:
    event = summary["event_assessment"]
    if not event.get("detected"):
        return {
            "detected": False,
            "censored_at": int(summary["completed_step"]),
            "event_midpoint": None,
            "confirmation_step": None,
            "first_site": None,
            "wire": None,
        }
    gap = event["persistent_gaps"][0]
    bracket = event["event_bracket"]
    return {
        "detected": True,
        "censored_at": None,
        "event_midpoint": 0.5 * (bracket[0] + bracket[1]),
        "confirmation_step": int(event["confirmation_step"]),
        "first_site": float(gap["first_midpoint"]),
        "confirmation_site": float(gap["confirmation_midpoint"]),
        "wire": gap["wire"],
        "branch": gap["first_branch"],
    }


def _existing_seed_observations() -> dict[str, Any]:
    audit = _load_json(EXISTING_ANCHOR_AUDIT)
    untreated = audit["events"]["untreated"]
    observations: dict[str, Any] = {
        "untreated": {
            "detected": True,
            "event_midpoint": untreated["event_midpoint"],
            "confirmation_step": untreated["confirmation_step"],
            "first_site": untreated["first_event_site"],
            "confirmation_site": untreated["confirmation_site"],
            "wire": "first_wire_z",
            "branch": "plus",
            "source": str(EXISTING_ANCHOR_AUDIT.resolve()),
        }
    }
    for case_id, path in EXISTING_SEED_PATHS.items():
        observations[case_id] = {
            **_event_observation(_load_json(path)),
            "source": str(path.resolve()),
        }
    return observations


def _paired_pattern(observations: dict[str, Any]) -> dict[str, Any]:
    times = {
        case_id: observations[case_id]["event_midpoint"]
        for case_id in protocol.CASE_ORDER
    }
    sites = {
        case_id: observations[case_id]["first_site"]
        for case_id in protocol.CASE_ORDER
    }
    all_detected = all(
        observations[case_id]["detected"]
        for case_id in protocol.CASE_ORDER
    )
    if not all_detected:
        return {
            "all_four_events_detected": False,
            "c18_relocated_downstream": None,
            "c26_later_than_untreated": None,
            "c34_earlier_than_untreated": None,
            "full_qualitative_pattern": None,
        }
    checks = {
        "all_four_events_detected": True,
        "c18_relocated_downstream": (
            sites["c18p5"] > sites["untreated"] + 10.0
        ),
        "c26_later_than_untreated": (
            times["c26p5"] > times["untreated"]
        ),
        "c34_earlier_than_untreated": (
            times["c34p5"] < times["untreated"]
        ),
    }
    checks["full_qualitative_pattern"] = all(checks.values())
    return checks


def _aggregate(
    output: Path,
    sources: dict[str, Any],
    cases: dict[str, Any],
) -> dict[str, Any]:
    realizations: dict[str, Any] = {
        str(protocol.EXISTING_SEED): {
            "observations": _existing_seed_observations(),
        }
    }
    for seed in protocol.NEW_SEEDS:
        observations = {}
        for case_id in protocol.CASE_ORDER:
            summary_path = protocol.case_output(
                output, seed, case_id
            ) / "summary.json"
            observations[case_id] = _event_observation(
                _load_json(summary_path)
            )
            observations[case_id]["source"] = str(
                summary_path.resolve()
            )
        realizations[str(seed)] = {
            "source": sources[str(seed)],
            "observations": observations,
        }
    for realization in realizations.values():
        realization["paired_pattern"] = _paired_pattern(
            realization["observations"]
        )
    full_flags = [
        realization["paired_pattern"]["full_qualitative_pattern"]
        for realization in realizations.values()
    ]
    aggregate = {
        "status": "completed",
        "classification": "paired_repeat_panel_factually_aggregated",
        "realizations": realizations,
        "panel_assessment": {
            "realization_count": len(realizations),
            "new_trajectory_count": len(cases),
            "full_pattern_flags": full_flags,
            "full_pattern_reproduced_in_all_three": (
                all(flag is True for flag in full_flags)
            ),
            "interpretation_requires_scientific_review": True,
        },
        "scope": {
            "factual_aggregation_only": True,
            "new_simulation_launched_by_aggregation": False,
        },
    }
    atomic_json(output / "aggregate.json", aggregate)
    return aggregate


def run_panel(
    output: Path,
    *,
    preflight_report: dict[str, Any],
    stop_requested: dict[str, int | None],
) -> dict[str, Any]:
    """Run the frozen sequence and stop on technical, never scientific, failure."""

    contract = _contract(preflight_report)
    atomic_json(output / "contract.json", contract)
    lock = acquire_output_lock(output)
    started = time.perf_counter()
    sources: dict[str, Any] = {}
    cases: dict[str, Any] = {}
    current_seed: int | None = None
    current_case: str | None = None
    try:
        _write_status(
            output,
            status="running",
            current_seed=None,
            current_case=None,
            stop_reason=None,
            sources=sources,
            cases=cases,
            started=started,
        )
        for seed in protocol.NEW_SEEDS:
            current_seed = seed
            source_output = protocol.source_output(output, seed)
            if stop_requested["signal"] is not None:
                raise RuntimeError("panel interrupted before source")
            source_return = _run_child(
                source_command(seed, source_output)
            )
            if source_return != 0:
                raise RuntimeError(
                    f"source seed {seed} exited {source_return}"
                )
            source = source_runner.verify_completed_source(
                source_output, seed=seed
            )
            # This is a technical distinctness check, not morphology screening.
            prior_fingerprints = {
                item["initial_fingerprint"]
                for item in sources.values()
            }
            if source["initial_fingerprint"] in prior_fingerprints:
                raise RuntimeError(
                    "two preregistered seeds produced an identical source"
                )
            sources[str(seed)] = source
            _write_status(
                output,
                status="running",
                current_seed=seed,
                current_case=None,
                stop_reason=None,
                sources=sources,
                cases=cases,
                started=started,
            )
            for case in protocol.CASES:
                current_case = case.case_id
                key = f"{seed}:{case.case_id}"
                case_output = protocol.case_output(
                    output, seed, case.case_id
                )
                completed = False
                for session in range(
                    1, protocol.MAXIMUM_SESSIONS_PER_CASE + 1
                ):
                    resume = case_output.exists()
                    return_code = _run_child(
                        case_command(
                            seed,
                            case.case_id,
                            source_output,
                            case_output,
                            resume=resume,
                        )
                    )
                    summary_path = case_output / "summary.json"
                    if not summary_path.is_file():
                        raise RuntimeError(
                            f"{key} produced no terminal summary"
                        )
                    summary = _load_json(summary_path)
                    cases[key] = {
                        "status": summary.get("status"),
                        "completed_step": summary.get(
                            "completed_step"
                        ),
                        "return_code": return_code,
                        "sessions": session,
                        "summary_path": str(
                            summary_path.resolve()
                        ),
                        "summary_sha256": sha256_path(summary_path),
                    }
                    _write_status(
                        output,
                        status="running",
                        current_seed=seed,
                        current_case=case.case_id,
                        stop_reason=None,
                        sources=sources,
                        cases=cases,
                        started=started,
                    )
                    if summary.get("status") == "completed":
                        if return_code != 0:
                            raise RuntimeError(
                                f"{key} completed with exit {return_code}"
                            )
                        verified = case_runner.verify_completed_case(
                            case_output,
                            seed=seed,
                            case_id=case.case_id,
                            source=source,
                        )
                        cases[key]["verification"] = verified
                        completed = True
                        break
                    if (
                        summary.get("status") == "wall_cap"
                        and return_code == 3
                        and session
                        < protocol.MAXIMUM_SESSIONS_PER_CASE
                    ):
                        continue
                    raise RuntimeError(
                        f"{key} stopped status={summary.get('status')} "
                        f"exit={return_code}"
                    )
                if not completed:
                    raise RuntimeError(
                        f"{key} exhausted verified sessions"
                    )
                if stop_requested["signal"] is not None:
                    raise RuntimeError("panel interrupted after case")

        aggregate = _aggregate(output, sources, cases)
        status = _write_status(
            output,
            status="completed",
            current_seed=None,
            current_case=None,
            stop_reason="eight_cases_verified_and_factually_aggregated",
            sources=sources,
            cases=cases,
            started=started,
        )
        summary = {
            **status,
            "classification": "paired_repeat_panel_completed",
            "aggregate": {
                "path": str((output / "aggregate.json").resolve()),
                "sha256": sha256_path(output / "aggregate.json"),
                "assessment": aggregate["panel_assessment"],
            },
            "contract": contract,
            "scope": {
                "two_sources_conditioned": True,
                "eight_new_trajectories_completed": True,
                "automatic_follow_on_simulation_started": False,
                "wider_sweep_started": False,
                "convergence_family_started": False,
            },
        }
        atomic_json(output / "summary.json", summary)
        return summary
    except Exception as error:
        _write_status(
            output,
            status="failed",
            current_seed=current_seed,
            current_case=current_case,
            stop_reason=f"{type(error).__name__}: {error}",
            sources=sources,
            cases=cases,
            started=started,
        )
        atomic_json(
            output / "failure.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "current_seed": current_seed,
                "current_case": current_case,
                "pid": os.getpid(),
                "unix_time": time.time(),
            },
        )
        raise
    finally:
        release_output_lock(lock)


def launch_background(output: Path) -> dict[str, Any]:
    """Reserve the output and detach one quiet sequential controller."""

    report = preflight(output)
    reserve_output_directory(output)
    atomic_json(output / "preflight.json", report)
    log_path = output / "batch.log"
    command = [
        sys.executable,
        "-m",
        PANEL_MODULE,
        "--run-prepared",
        "--output",
        str(output.resolve()),
    ]
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    background = {
        "status": "launched",
        "pid": process.pid,
        "process_group_id": process.pid,
        "command": command,
        "output": str(output.resolve()),
        "log": str(log_path.resolve()),
        "preflight_sha256": sha256_path(output / "preflight.json"),
        "launched_unix_time": time.time(),
    }
    atomic_json(output / "background.json", background)
    return background


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--launch-background", action="store_true")
    mode.add_argument("--run-prepared", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=protocol.DEFAULT_OUTPUT
    )
    parser.add_argument(
        "--acknowledge-long-runtime",
        action="store_true",
        help="Required to launch the approved approximately 12--15 hour panel.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    output = arguments.output.resolve()
    if arguments.preflight:
        print(json.dumps(preflight(output), sort_keys=True), flush=True)
        return 0
    if arguments.launch_background:
        if not arguments.acknowledge_long_runtime:
            raise RuntimeError(
                "background launch requires explicit long-runtime acknowledgement"
            )
        print(
            json.dumps(launch_background(output), sort_keys=True),
            flush=True,
        )
        return 0
    preflight_path = output / "preflight.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("prepared preflight is missing")
    report = _load_json(preflight_path)
    if report.get("status") != "preflight_passed":
        raise RuntimeError("prepared preflight did not pass")
    stop_requested: dict[str, int | None] = {"signal": None}

    def request_stop(signal_number: int, _frame: object) -> None:
        stop_requested["signal"] = signal_number
        active = ACTIVE_CHILD
        if active is not None and active.poll() is None:
            try:
                os.killpg(active.pid, signal_number)
            except ProcessLookupError:
                pass

    handled = (signal.SIGINT, signal.SIGTERM)
    prior_handlers = {
        number: signal.getsignal(number) for number in handled
    }
    for number in handled:
        signal.signal(number, request_stop)
    try:
        result = run_panel(
            output,
            preflight_report=report,
            stop_requested=stop_requested,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "classification": result["classification"],
                    "output": str(output),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        for number, handler in prior_handlers.items():
            signal.signal(number, handler)
    return 0 if result["status"] == "completed" else 6


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
