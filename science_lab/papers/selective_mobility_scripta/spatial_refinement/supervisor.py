#!/usr/bin/env python3
"""Non-adaptive sequential supervisor for the frozen validation pair."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from science_lab.papers.nanowire_gb_junction.roy_gb_junction_sentinel.storage import (
    atomic_json,
)

from . import freeze, protocol


PREFLIGHT_MODULE = (
    "science_lab.papers.selective_mobility_scripta.spatial_refinement.preflight"
)
RUNNER_MODULE = (
    "science_lab.papers.selective_mobility_scripta.spatial_refinement.runner"
)
ANALYZER_MODULE = (
    "science_lab.papers.selective_mobility_scripta.spatial_refinement.analyze"
)
STAGES = ("preflight", "source", "guard", "untreated", "c34", "analysis")
ACTIVE_CHILD: subprocess.Popen[str] | None = None
STOP_SIGNAL: int | None = None


def _signal_handler(signum: int, _frame: Any) -> None:
    global STOP_SIGNAL
    STOP_SIGNAL = int(signum)
    if ACTIVE_CHILD is not None and ACTIVE_CHILD.poll() is None:
        ACTIVE_CHILD.send_signal(signum)


def stage_command(
    stage: str, output: Path, source_root: Path | None = None
) -> list[str]:
    resolved_source = protocol.source_paths(source_root).root
    base = [sys.executable, "-m"]
    if stage == "preflight":
        return [
            *base,
            PREFLIGHT_MODULE,
            "--output",
            str(output),
            "--source-root",
            str(resolved_source),
        ]
    if stage in {"source", "guard", "untreated", "c34"}:
        return [
            *base,
            RUNNER_MODULE,
            stage,
            "--output",
            str(output),
            "--resume",
            "--source-root",
            str(resolved_source),
        ]
    if stage == "analysis":
        return [
            *base,
            ANALYZER_MODULE,
            "--output",
            str(output),
            "--source-root",
            str(resolved_source),
        ]
    raise ValueError(f"unknown stage: {stage}")


def _write_status(output: Path, payload: dict[str, Any]) -> None:
    atomic_json(
        output / "supervisor_status.json",
        {
            "schema_version": protocol.SCHEMA_VERSION,
            "protocol_id": protocol.PROTOCOL_ID,
            "adaptive_follow_on_authorized": False,
            "wall_clock_cap_seconds": protocol.GUARDED_RUNTIME_SECONDS,
            "pid": os.getpid(),
            "updated_unix_time": time.time(),
            **payload,
        },
    )


def _notify(message: str) -> None:
    try:
        subprocess.run(
            [
                "/usr/bin/osascript",
                "-e",
                f'display notification {json.dumps(message)} with title "Spatial refinement"',
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass


def _terminate_at_wall_cap(child: subprocess.Popen[str]) -> int:
    child.send_signal(signal.SIGTERM)
    try:
        return child.wait(timeout=60.0)
    except subprocess.TimeoutExpired:
        child.kill()
        return child.wait()


def run_supervisor(
    output: Path, *, source_root: Path | None = None
) -> dict[str, Any]:
    global ACTIVE_CHILD
    output = output.resolve()
    resolved_source = protocol.source_paths(source_root).root
    output.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    campaign_started = time.monotonic()
    caffeinate: subprocess.Popen[Any] | None = None
    if Path("/usr/bin/caffeinate").is_file():
        caffeinate = subprocess.Popen(
            ["/usr/bin/caffeinate", "-dimsu", "-w", str(os.getpid())]
        )
    try:
        for stage in STAGES:
            if STOP_SIGNAL is not None:
                break
            _write_status(
                output,
                {"status": "running", "current_stage": stage, "completed": completed},
            )
            if stage != "preflight":
                freeze.verify_preflight_binding(
                    output,
                    protocol_id=protocol.PROTOCOL_ID,
                    source_root=resolved_source,
                )
                inventory = freeze.process_inventory()
                if inventory["science"] or inventory["git"]:
                    raise RuntimeError(
                        f"competing process before {stage}: {inventory}"
                    )
            command = stage_command(stage, output, resolved_source)
            ACTIVE_CHILD = subprocess.Popen(command, text=True)
            remaining = protocol.GUARDED_RUNTIME_SECONDS - (
                time.monotonic() - campaign_started
            )
            if remaining <= 0.0:
                return_code = _terminate_at_wall_cap(ACTIVE_CHILD)
                watchdog_expired = True
            else:
                try:
                    return_code = ACTIVE_CHILD.wait(timeout=remaining)
                    watchdog_expired = False
                except subprocess.TimeoutExpired:
                    return_code = _terminate_at_wall_cap(ACTIVE_CHILD)
                    watchdog_expired = True
            ACTIVE_CHILD = None
            if watchdog_expired:
                report = {
                    "status": "stopped",
                    "current_stage": stage,
                    "completed": completed,
                    "return_code": return_code,
                    "signal": int(signal.SIGTERM),
                    "reason": "hard_wall_clock_cap_reached",
                    "elapsed_wall_seconds": time.monotonic() - campaign_started,
                }
                _write_status(output, report)
                _notify("Spatial refinement stopped at hard wall-clock cap")
                return report
            if return_code != 0:
                report = {
                    "status": "stopped",
                    "current_stage": stage,
                    "completed": completed,
                    "return_code": return_code,
                    "signal": STOP_SIGNAL,
                    "reason": (
                        "preflight_no_go"
                        if stage == "preflight"
                        else "frozen_stage_failed"
                    ),
                }
                _write_status(output, report)
                _notify(f"Spatial refinement stopped at {stage}")
                return report
            if stage == "preflight":
                preflight = json.loads(
                    (output / "preflight" / "summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                if preflight.get("decision") != "GO":
                    raise RuntimeError("preflight process returned without GO")
            completed.append(stage)
        status = "completed" if completed == list(STAGES) else "interrupted"
        report = {
            "status": status,
            "current_stage": None,
            "completed": completed,
            "signal": STOP_SIGNAL,
            "reason": "fixed_sequence_complete" if status == "completed" else "signal",
        }
        _write_status(output, report)
        _notify(f"Spatial refinement {status}")
        return report
    finally:
        ACTIVE_CHILD = None
        if caffeinate is not None and caffeinate.poll() is None:
            caffeinate.terminate()


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=protocol.DEFAULT_OUTPUT)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=protocol.EXTERNAL_SOURCE_DIRECTORY,
    )
    parser.add_argument(
        "--launch-frozen-sequence",
        action="store_true",
        help="confirm that this invocation should launch the long run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    if not args.launch_frozen_sequence:
        raise SystemExit(
            "refusing to launch: pass --launch-frozen-sequence after reviewing "
            "the preflight receipt"
        )
    report = run_supervisor(args.output, source_root=args.source_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    for handled_signal in (signal.SIGINT, signal.SIGTERM):
        signal.signal(handled_signal, _signal_handler)
    raise SystemExit(main())
