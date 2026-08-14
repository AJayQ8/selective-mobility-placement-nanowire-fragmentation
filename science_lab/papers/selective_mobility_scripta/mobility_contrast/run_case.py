#!/usr/bin/env python3
"""Run one frozen m=0.3 treated case with exact resume and disk guards."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import json
import shutil
import signal
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from science_lab.papers.nanowire_gb_junction.roy_2021_reproduction.model import (
    FROZEN_DEG90,
)
from science_lab.papers.nanowire_gb_junction.roy_gb_junction_sentinel.storage import (
    atomic_json,
    load_checkpoint,
)
from science_lab.papers.nanowire_gb_junction.roy_selective_mobility_protection import (
    paired_repeat_protocol as paired_protocol,
    position_scan_protocol,
    run_paired_repeat_case as paired_runner,
    run_paired_repeat_source as source_runner,
    run_position_scan_case as case_engine,
)
from science_lab.papers.nanowire_gb_junction.roy_selective_mobility_protection.geometry import (
    four_arm_tubular_collar_factor,
    surface_weighted_mobility_deficit,
)

from .common import (
    CAMPAIGN_ID,
    CASE_IDS,
    CONTRACT_PATH,
    PARENT_REPEAT_CONTRACT_PATH,
    PREFLIGHT_PATH,
    PROTECTED_FACTOR,
    SEEDS,
    frozen_contract,
    load_json,
    sha256_path,
)


SOURCE_PATH = Path(__file__).resolve()


def derive_factor(
    case: paired_protocol.RepeatCase,
    source: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if case.case_id not in CASE_IDS or case.position_case is None:
        raise RuntimeError("case is outside the frozen m=0.3 campaign")
    geometry = dataclasses.replace(
        case.position_case.geometry,
        protected_mobility_factor=PROTECTED_FACTOR,
    )
    factor = four_arm_tubular_collar_factor(
        FROZEN_DEG90.lattice,
        geometry,
        radius=position_scan_protocol.RADIUS,
        interface_width=position_scan_protocol.INTERFACE_WIDTH,
    )
    field = load_checkpoint(source["checkpoint"])
    try:
        budget = surface_weighted_mobility_deficit(
            field,
            factor,
            cell_volume=FROZEN_DEG90.lattice.cell_volume,
        )
    finally:
        del field
        gc.collect()
    minimum = float(np.min(factor))
    maximum = float(np.max(factor))
    checks = {
        "minimum": np.isclose(minimum, PROTECTED_FACTOR, rtol=0.0, atol=1.0e-15),
        "maximum": maximum == 1.0,
        "center": geometry.center_distance == case.center,
        "support_width": geometry.support_width == 12.0,
        "transition_width": geometry.transition_width == 3.0,
        "finite": bool(np.isfinite(factor).all()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"custom factor failed frozen checks: {checks}")
    return factor, {
        "case_id": f"{case.case_id}_m03",
        "parent_case_id": case.case_id,
        "untreated": False,
        "center_distance": case.center,
        "geometry": geometry.to_dict(),
        "surface_weighted_mobility_deficit": float(budget),
        "factor": {
            "shape": list(factor.shape),
            "minimum": minimum,
            "maximum": maximum,
            "support_cell_fraction": float(np.mean(factor < 1.0 - 1.0e-14)),
        },
        "checks": checks,
        "fixed_without_seed_specific_retuning": True,
    }


def _verified_preflight(path: Path) -> dict[str, Any]:
    report = load_json(path)
    checks = {
        "go": report.get("status") == "GO",
        "campaign": report.get("campaign_id") == CAMPAIGN_ID,
        "zero_steps": report.get("simulation_steps_performed") == 0,
        "all_checks": all(report.get("checks", {}).values()),
        "contract": report["contract"]["sha256"] == sha256_path(CONTRACT_PATH),
        "runner": report["campaign_implementation"]["run_case_sha256"]
        == sha256_path(SOURCE_PATH),
        "common": report["campaign_implementation"]["common_sha256"]
        == sha256_path(SOURCE_PATH.with_name("common.py")),
    }
    if not all(checks.values()):
        raise RuntimeError(f"campaign preflight is not launchable: {checks}")
    return report


def _contract(
    case: paired_protocol.RepeatCase,
    source: dict[str, Any],
    factor_report: dict[str, Any],
    preflight_path: Path,
) -> dict[str, Any]:
    parent = paired_runner._contract(case, source, factor_report)
    parent["protocol"] = "m03_bc_position_sign_case_v1"
    parent["case"] = f"{case.case_id}_m03"
    parent["parent_case"] = case.case_id
    parent["protected_mobility_factor"] = PROTECTED_FACTOR
    parent["campaign_id"] = CAMPAIGN_ID
    parent["provenance"]["contrast_runner_sha256"] = sha256_path(SOURCE_PATH)
    parent["provenance"]["frozen_contrast_contract_sha256"] = sha256_path(
        CONTRACT_PATH
    )
    parent["provenance"]["launch_preflight_sha256"] = sha256_path(preflight_path)
    parent["provenance"]["parent_science_contract_sha256"] = parent[
        "provenance"
    ]["scientific_contract_sha256"]
    parent["provenance"]["scientific_contract_sha256"] = sha256_path(
        CONTRACT_PATH
    )
    parent["scope"] = {
        "one_frozen_m03_case": True,
        "paired_source_shared_exactly": True,
        "seed_specific_retuning_authorized": False,
        "adaptive_rescue_authorized": False,
        "additional_contrast_authorized": False,
        "position_sweep_authorized": False,
    }
    return parent


def _resource_guard(
    output: Path,
    preflight: dict[str, Any],
    *,
    reason: str,
    step: int | None,
) -> dict[str, Any]:
    free = int(shutil.disk_usage(output.parent).free)
    floor = int(preflight["resources"]["required_free_disk_bytes"])
    record = {
        "campaign_id": CAMPAIGN_ID,
        "checked_unix_time": time.time(),
        "reason": reason,
        "step": step,
        "free_disk_bytes": free,
        "conservative_fixed_floor_bytes": floor,
        "passed": free > floor,
        "policy": (
            "fixed launch requirement retained at every large write; this is "
            "stricter than subtracting outputs already published"
        ),
    }
    atomic_json(output / "disk_guard.json", record)
    if not record["passed"]:
        raise RuntimeError(f"disk guard failed before {reason}: {record}")
    return record


@contextlib.contextmanager
def installed_dependencies(
    case: paired_protocol.RepeatCase,
    source: dict[str, Any],
    factor: np.ndarray,
    factor_report: dict[str, Any],
    *,
    output: Path,
    preflight_path: Path,
) -> Iterator[None]:
    preflight = _verified_preflight(preflight_path)
    with paired_runner.installed_case_dependencies(
        case, source, factor, factor_report
    ):
        original_contract = case_engine._contract
        original_resource = case_engine.resource_preflight
        original_write = case_engine.write_checkpoint

        def guarded_write(
            checkpoint_output: Path,
            field: np.ndarray,
            *,
            step: int,
            kinds: tuple[str, ...],
            stream: str,
        ) -> dict[str, Any]:
            _resource_guard(
                output,
                preflight,
                reason="checkpoint_atomic_write",
                step=int(step),
            )
            return original_write(
                checkpoint_output,
                field,
                step=step,
                kinds=kinds,
                stream=stream,
            )

        def resource_report(
            _resource_output: Path,
            *,
            current_step: int,
            existing_event_checkpoints: int = 0,
        ) -> dict[str, Any]:
            guard = _resource_guard(
                output,
                preflight,
                reason="case_start_or_resume",
                step=int(current_step),
            )
            return {
                "campaign_preflight_path": str(preflight_path.resolve()),
                "campaign_preflight_sha256": sha256_path(preflight_path),
                "current_step": int(current_step),
                "existing_event_checkpoints": int(existing_event_checkpoints),
                "guard": guard,
            }

        case_engine._contract = lambda _case, _factor_report: _contract(
            case, source, factor_report, preflight_path
        )
        case_engine.resource_preflight = resource_report
        case_engine.write_checkpoint = guarded_write
        try:
            yield
        finally:
            case_engine._contract = original_contract
            case_engine.resource_preflight = original_resource
            case_engine.write_checkpoint = original_write


def run_case(
    case_id: str,
    *,
    seed: int,
    source_path: Path,
    output: Path,
    preflight_path: Path,
    resume: bool,
    stop_requested: dict[str, int | None],
) -> dict[str, Any]:
    frozen_contract()
    if seed not in SEEDS or case_id not in CASE_IDS:
        raise ValueError("case is outside the frozen campaign")
    case = paired_protocol.validate_case(case_id)
    original_contract_path = paired_protocol.CONTRACT_PATH
    paired_protocol.CONTRACT_PATH = PARENT_REPEAT_CONTRACT_PATH
    try:
        source = source_runner.verify_completed_source(source_path, seed=seed)
        factor, factor_report = derive_factor(case, source)
        with installed_dependencies(
            case,
            source,
            factor,
            factor_report,
            output=output,
            preflight_path=preflight_path,
        ):
            return case_engine.run(
                case,
                output,
                resume=resume,
                stop_requested=stop_requested,
            )
    finally:
        paired_protocol.CONTRACT_PATH = original_contract_path


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--case", choices=CASE_IDS, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, default=PREFLIGHT_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    stop_requested: dict[str, int | None] = {"signal": None}

    def request_stop(signal_number: int, _frame: object) -> None:
        stop_requested["signal"] = signal_number

    handled = (signal.SIGINT, signal.SIGTERM)
    prior = {number: signal.getsignal(number) for number in handled}
    for number in handled:
        signal.signal(number, request_stop)
    try:
        result = run_case(
            args.case,
            seed=args.seed,
            source_path=args.source_output.resolve(),
            output=args.output.resolve(),
            preflight_path=args.preflight.resolve(),
            resume=args.resume,
            stop_requested=stop_requested,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "seed": args.seed,
                    "case": f"{args.case}_m03",
                    "completed_step": result["completed_step"],
                    "output": str(args.output.resolve()),
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
