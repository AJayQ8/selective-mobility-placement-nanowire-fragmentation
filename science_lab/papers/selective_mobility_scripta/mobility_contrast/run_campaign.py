#!/usr/bin/env python3
"""Sequential supervisor for the frozen four-case m=0.3 campaign."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from science_lab.papers.nanowire_gb_junction.roy_gb_junction_sentinel.storage import (
    atomic_json,
)

from .analyze import analyze
from .common import (
    CAMPAIGN_ID,
    CAMPAIGN_STATUS_PATH,
    CASE_ORDER,
    PREFLIGHT_PATH,
    RESULTS_ROOT,
    load_json,
    raw_case_output,
    sha256_path,
    source_output,
)


CASE_MODULE = (
    "science_lab.papers.selective_mobility_scripta."
    "mobility_contrast.run_case"
)
SOURCE_PATH = Path(__file__).resolve()


def _write_status(**values: Any) -> dict[str, Any]:
    current = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "supervisor_pid": os.getpid(),
        "updated_unix_time": time.time(),
        **values,
    }
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(CAMPAIGN_STATUS_PATH, current)
    return current


def _case_status(output: Path) -> dict[str, Any] | None:
    path = output / "run_status.json"
    return load_json(path) if path.is_file() else None


def _completed(output: Path) -> bool:
    summary = output / "summary.json"
    if not summary.is_file():
        return False
    report = load_json(summary)
    return (
        report.get("status") == "completed"
        and report.get("stop_reason") == "robust_single_arm_event_confirmed"
    )


def _command(
    historical_root: Path,
    seed: int,
    case_id: str,
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
        str(source_output(historical_root, seed).resolve()),
        "--output",
        str(raw_case_output(seed, case_id).resolve()),
        "--preflight",
        str(PREFLIGHT_PATH.resolve()),
    ]


def run_campaign(historical_root: Path) -> int:
    preflight = load_json(PREFLIGHT_PATH)
    if preflight.get("status") != "GO" or not all(preflight["checks"].values()):
        raise RuntimeError("campaign cannot launch without the exact GO preflight")
    if preflight["campaign_implementation"]["run_campaign_sha256"] != sha256_path(
        SOURCE_PATH
    ):
        raise RuntimeError("campaign supervisor changed after the GO preflight")
    stop_requested: dict[str, int | None] = {"signal": None}
    current_child: dict[str, subprocess.Popen[Any] | None] = {"process": None}

    def request_stop(signal_number: int, _frame: object) -> None:
        stop_requested["signal"] = signal_number
        child = current_child["process"]
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)

    handled = (signal.SIGINT, signal.SIGTERM)
    prior = {number: signal.getsignal(number) for number in handled}
    for number in handled:
        signal.signal(number, request_stop)
    caffeinate = subprocess.Popen(
        ["/usr/bin/caffeinate", "-dimsu", "-w", str(os.getpid())]
    )
    completed: list[str] = []
    campaign_started = time.time()
    try:
        _write_status(
            status="running",
            completed_cases=completed,
            current_case=None,
            preflight_sha256=sha256_path(PREFLIGHT_PATH),
        )
        for seed, case_id in CASE_ORDER:
            label = f"{seed}/{case_id}_m03"
            output = raw_case_output(seed, case_id)
            if _completed(output):
                completed.append(label)
                continue
            sessions = 0
            while not _completed(output):
                if stop_requested["signal"] is not None:
                    _write_status(
                        status="interrupted",
                        completed_cases=completed,
                        current_case=label,
                        stop_signal=stop_requested["signal"],
                    )
                    return 4
                sessions += 1
                if sessions > 2:
                    raise RuntimeError(f"session bound exceeded for {label}")
                resume = output.is_dir()
                command = _command(
                    historical_root,
                    seed,
                    case_id,
                    resume=resume,
                )
                child = subprocess.Popen(command)
                current_child["process"] = child
                while child.poll() is None:
                    status = _case_status(output)
                    _write_status(
                        status="running",
                        completed_cases=completed,
                        current_case=label,
                        current_session=sessions,
                        child_pid=child.pid,
                        child_status=status,
                        free_disk_bytes=int(shutil.disk_usage(RESULTS_ROOT).free),
                        elapsed_wall_seconds=time.time() - campaign_started,
                    )
                    time.sleep(5.0)
                return_code = int(child.returncode)
                current_child["process"] = None
                status = _case_status(output)
                terminal = status.get("status") if status else None
                if return_code == 0 and _completed(output):
                    break
                if return_code == 3 and terminal == "wall_cap":
                    continue
                if return_code == 4 and terminal == "interrupted":
                    _write_status(
                        status="interrupted",
                        completed_cases=completed,
                        current_case=label,
                        child_status=status,
                    )
                    return 4
                raise RuntimeError(
                    f"case {label} stopped unexpectedly: rc={return_code}, status={terminal}"
                )
            completed.append(label)
            _write_status(
                status="running",
                completed_cases=completed,
                current_case=None,
                elapsed_wall_seconds=time.time() - campaign_started,
            )
        analysis = analyze()
        _write_status(
            status="completed",
            completed_cases=completed,
            current_case=None,
            analysis_path=str((RESULTS_ROOT / "analysis.json").resolve()),
            analysis_sha256=sha256_path(RESULTS_ROOT / "analysis.json"),
            analysis_passed=analysis["passed"],
            elapsed_wall_seconds=time.time() - campaign_started,
        )
        return 0 if analysis["passed"] else 2
    except Exception as error:
        _write_status(
            status="failed",
            completed_cases=completed,
            error=f"{type(error).__name__}: {error}",
            elapsed_wall_seconds=time.time() - campaign_started,
        )
        raise
    finally:
        child = current_child["process"]
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            child.wait(timeout=900)
        if caffeinate.poll() is None:
            caffeinate.terminate()
        for number, handler in prior.items():
            signal.signal(number, handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_campaign(args.historical_root.resolve())
    print(json.dumps(load_json(CAMPAIGN_STATUS_PATH), sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
