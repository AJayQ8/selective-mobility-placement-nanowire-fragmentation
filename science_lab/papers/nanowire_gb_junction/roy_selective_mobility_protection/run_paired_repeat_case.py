#!/usr/bin/env python3
"""Run one paired-repeat condition from an explicitly verified t=100 source."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import shutil
import signal
import sys
import traceback
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import scipy

from ..roy_2021_reproduction import model as roy_model
from ..roy_2021_reproduction.model import FROZEN_DEG90
from ..roy_fixed_gb_bridge.provenance import sha256_path
from ..roy_gb_junction_sentinel.storage import load_checkpoint
from . import geometry as selective_geometry
from . import model as selective_model
from . import paired_repeat_protocol as protocol
from . import position_scan_protocol
from . import run_paired_repeat_source as source_runner
from . import run_position_scan_case as case_engine
from . import run_position_scan_preflight as scan_preflight
from .geometry import (
    four_arm_tubular_collar_factor,
    surface_weighted_mobility_deficit,
)


SOURCE_PATH = Path(__file__).resolve()
FIELD_BYTES = int(FROZEN_DEG90.lattice.cell_count * 8 + 256)


def derive_factor(
    case: protocol.RepeatCase,
    source: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the frozen factor; never retune it to the new realization."""

    if case.center is None:
        factor = np.ones((1,), dtype=np.float64)
        return factor, {
            "case_id": case.case_id,
            "untreated": True,
            "center_distance": None,
            "surface_weighted_mobility_deficit": 0.0,
            "factor": {
                "shape": [1],
                "minimum": 1.0,
                "maximum": 1.0,
                "support_cell_fraction": 0.0,
            },
            "fixed_without_seed_specific_retuning": True,
        }

    position_case = case.position_case
    if position_case is None:
        raise RuntimeError("collar case has no position definition")
    factor = four_arm_tubular_collar_factor(
        FROZEN_DEG90.lattice,
        position_case.geometry,
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
    static = position_scan_protocol.static_case_report(position_case)
    return factor, {
        **static,
        "case_id": case.case_id,
        "untreated": False,
        "surface_weighted_mobility_deficit": float(budget),
        "factor": {
            "shape": list(factor.shape),
            "minimum": float(np.min(factor)),
            "maximum": float(np.max(factor)),
            "support_cell_fraction": float(
                np.mean(factor < 1.0 - 1.0e-14)
            ),
        },
        "fixed_without_seed_specific_retuning": True,
    }


def resource_preflight(
    output: Path,
    *,
    current_step: int,
    existing_event_checkpoints: int,
) -> dict[str, Any]:
    """Bound one repeat case using measured production throughput."""

    measured = scan_preflight.runtime_and_storage_preflight()
    median = float(
        measured["slowest_reference_median_step_wall_seconds"]
    )
    remaining_regular = sum(
        step > current_step for step in protocol.CHECKPOINT_STEPS
    )
    remaining_event = max(
        0,
        protocol.MAXIMUM_EVENT_CHECKPOINTS
        - existing_event_checkpoints,
    )
    checkpoint_allowance = remaining_regular + remaining_event + 2
    required = (
        checkpoint_allowance * FIELD_BYTES
        + protocol.MINIMUM_DISK_HEADROOM_BYTES
    )
    free = int(shutil.disk_usage(output.parent).free)
    remaining_steps = protocol.TARGET_STEP - current_step
    projected = (
        remaining_steps
        * median
        * protocol.RUNTIME_SAFETY_FACTOR
    )
    report = {
        "field_bytes": FIELD_BYTES,
        "remaining_regular_checkpoints": remaining_regular,
        "remaining_event_checkpoints": remaining_event,
        "signal_or_wall_checkpoint_allowance": 2,
        "minimum_disk_headroom_bytes": (
            protocol.MINIMUM_DISK_HEADROOM_BYTES
        ),
        "required_free_disk_bytes": required,
        "free_disk_bytes": free,
        "disk_sufficient": free >= required,
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
        raise RuntimeError(f"repeat case disk preflight failed: {report}")
    if not report["runtime_below_session_cap"]:
        raise RuntimeError(
            f"repeat case runtime preflight failed: {report}"
        )
    return report


def _contract(
    case: protocol.RepeatCase,
    source: dict[str, Any],
    factor_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": "paired_repeat_case_v1",
        "seed": source["seed"],
        "case": case.case_id,
        "position": factor_report,
        "start_step": position_scan_protocol.START_STEP,
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
            "summary_path": source["summary_path"],
            "summary_sha256": source["summary_sha256"],
            "metadata_path": source["checkpoint"]["metadata_path"],
            "metadata_sha256": source["checkpoint"][
                "metadata_sha256"
            ],
            "field_path": source["checkpoint"]["field_path"],
            "field_sha256": source["checkpoint"]["field_sha256"],
            "field_fingerprint": source[
                "conditioned_fingerprint"
            ],
            "global_step": position_scan_protocol.START_STEP,
            "initial_mass": source["initial_mass"],
        },
        "provenance": {
            "repeat_runner_sha256": sha256_path(SOURCE_PATH),
            "case_engine_sha256": sha256_path(
                Path(case_engine.__file__).resolve()
            ),
            "source_runner_sha256": sha256_path(
                Path(source_runner.__file__).resolve()
            ),
            "protocol_sha256": sha256_path(
                Path(protocol.__file__).resolve()
            ),
            "position_protocol_sha256": sha256_path(
                Path(position_scan_protocol.__file__).resolve()
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
            "one_preregistered_repeat_case": True,
            "paired_source_shared_exactly": True,
            "seed_specific_retuning_authorized": False,
            "adaptive_position_search_authorized": False,
            "wider_sweep_authorized": False,
            "convergence_family_authorized": False,
        },
    }


@contextlib.contextmanager
def installed_case_dependencies(
    case: protocol.RepeatCase,
    source: dict[str, Any],
    factor: np.ndarray,
    factor_report: dict[str, Any],
) -> Iterator[None]:
    """Inject explicit source/factor dependencies into the verified engine."""

    originals = {
        "derive_case_factor": case_engine.derive_case_factor,
        "contract": case_engine._contract,
        "resource_preflight": case_engine.resource_preflight,
        "verified_source": (
            case_engine.far_runner.verified_source_checkpoint_entry
        ),
        "load_checkpoint": case_engine.far_runner._load_run_checkpoint,
        "expected_mass": (
            case_engine.source_helper.EXPECTED_T100_MASS
        ),
    }
    case_engine.derive_case_factor = lambda _case: (
        factor,
        factor_report,
    )
    case_engine._contract = lambda _case, _report: _contract(
        case, source, factor_report
    )
    case_engine.resource_preflight = (
        lambda output, *, current_step, existing_event_checkpoints=0: (
            resource_preflight(
                output,
                current_step=current_step,
                existing_event_checkpoints=(
                    existing_event_checkpoints
                ),
            )
        )
    )
    case_engine.far_runner.verified_source_checkpoint_entry = (
        lambda: source["checkpoint"]
    )
    case_engine.far_runner._load_run_checkpoint = load_checkpoint
    case_engine.source_helper.EXPECTED_T100_MASS = source[
        "initial_mass"
    ]
    try:
        yield
    finally:
        case_engine.derive_case_factor = originals[
            "derive_case_factor"
        ]
        case_engine._contract = originals["contract"]
        case_engine.resource_preflight = originals[
            "resource_preflight"
        ]
        case_engine.far_runner.verified_source_checkpoint_entry = (
            originals["verified_source"]
        )
        case_engine.far_runner._load_run_checkpoint = originals[
            "load_checkpoint"
        ]
        case_engine.source_helper.EXPECTED_T100_MASS = originals[
            "expected_mass"
        ]


def run_case(
    case: protocol.RepeatCase,
    *,
    seed: int,
    source_output: Path,
    output: Path,
    resume: bool,
    stop_requested: dict[str, int | None],
) -> dict[str, Any]:
    source = source_runner.verify_completed_source(
        source_output, seed=seed
    )
    factor, factor_report = derive_factor(case, source)
    with installed_case_dependencies(
        case, source, factor, factor_report
    ):
        return case_engine.run(
            case,
            output,
            resume=resume,
            stop_requested=stop_requested,
        )


def verify_completed_case(
    output: Path,
    *,
    seed: int,
    case_id: str,
    source: dict[str, Any],
) -> dict[str, Any]:
    case = protocol.validate_case(case_id)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    contract = summary["contract"]
    checks = {
        "status": summary.get("status") == "completed",
        "case": (
            summary.get("case") == case.case_id
            and contract.get("case") == case.case_id
        ),
        "seed": int(contract.get("seed", -1)) == seed,
        "target_or_event": (
            summary["stop_reason"]
            in {
                "robust_single_arm_event_confirmed",
                "fixed_t3000_target_reached",
            }
        ),
        "source_summary": (
            contract["source_checkpoint"]["summary_sha256"]
            == source["summary_sha256"]
        ),
        "source_field": (
            contract["source_checkpoint"]["field_sha256"]
            == source["checkpoint"]["field_sha256"]
        ),
        "source_fingerprint": (
            contract["source_checkpoint"]["field_fingerprint"]
            == source["conditioned_fingerprint"]
        ),
        "runner": (
            contract["provenance"]["repeat_runner_sha256"]
            == sha256_path(SOURCE_PATH)
        ),
        "engine": (
            contract["provenance"]["case_engine_sha256"]
            == sha256_path(Path(case_engine.__file__).resolve())
        ),
        "protocol": (
            contract["provenance"]["protocol_sha256"]
            == sha256_path(Path(protocol.__file__).resolve())
        ),
        "science_contract": (
            contract["provenance"]["scientific_contract_sha256"]
            == sha256_path(protocol.CONTRACT_PATH)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"repeat case verification failed: {checks}")
    return {
        "seed": seed,
        "case_id": case.case_id,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": sha256_path(summary_path),
        "completed_step": int(summary["completed_step"]),
        "stop_reason": summary["stop_reason"],
        "event_assessment": summary["event_assessment"],
        "checks": checks,
    }


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--seed", type=int, choices=protocol.NEW_SEEDS, required=True
    )
    parser.add_argument(
        "--case", choices=protocol.CASE_ORDER, required=True
    )
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    case = protocol.validate_case(arguments.case)
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
        result = run_case(
            case,
            seed=arguments.seed,
            source_output=arguments.source_output.resolve(),
            output=arguments.output.resolve(),
            resume=arguments.resume,
            stop_requested=stop_requested,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "seed": arguments.seed,
                    "case": case.case_id,
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
