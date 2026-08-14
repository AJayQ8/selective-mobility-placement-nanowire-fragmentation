#!/usr/bin/env python3
"""Condition one frozen alternate released-noise source to t=100."""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import statistics
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from ..roy_2021_reproduction import model as roy_model
from ..roy_2021_reproduction import run_t1000 as source_diagnostics
from ..roy_2021_reproduction.model import (
    FROZEN_DEG90,
    RoyPseudospectralSolver,
    apply_released_overlapping_noise,
    field_diagnostics,
    initialize_strict_deg90,
)
from ..roy_fixed_gb_bridge.provenance import sha256_path
from ..roy_gb_junction_sentinel.initializer import array_fingerprint
from ..roy_gb_junction_sentinel.storage import (
    acquire_output_lock,
    atomic_json,
    release_output_lock,
    reserve_output_directory,
    write_checkpoint,
)
from . import paired_repeat_protocol as protocol


SOURCE_PATH = Path(__file__).resolve()
MODEL_PATH = Path(roy_model.__file__).resolve()
MASS_DRIFT_LIMIT = source_diagnostics.MASS_DRIFT_LIMIT
FIELD_MAGNITUDE_LIMIT = (
    source_diagnostics.FIELD_MAGNITUDE_CATASTROPHE_LIMIT
)


def source_definition(seed: int) -> roy_model.RoyDeg90Definition:
    """Change only the preregistered noise seed."""

    return replace(FROZEN_DEG90, noise_seed=protocol.validate_seed(seed))


def _contract(
    seed: int,
    *,
    initial_fingerprint: str,
    initial_mass: float,
    noise_layout: dict[str, Any],
) -> dict[str, Any]:
    definition = source_definition(seed)
    return {
        "schema_version": 1,
        "protocol": "paired_repeat_conditioned_source_v1",
        "seed": seed,
        "target_step": protocol.SOURCE_TARGET_STEP,
        "fft_workers": protocol.FFT_WORKERS,
        "diagnostic_interval": protocol.DIAGNOSTIC_INTERVAL,
        "initial_fingerprint": initial_fingerprint,
        "initial_mass": initial_mass,
        "definition": asdict(definition),
        "noise_layout": noise_layout,
        "released_source_commit": roy_model.RELEASED_SOURCE_COMMIT,
        "provenance": {
            "runner_sha256": sha256_path(SOURCE_PATH),
            "model_sha256": sha256_path(MODEL_PATH),
            "scientific_contract_sha256": sha256_path(
                protocol.CONTRACT_PATH
            ),
            "protocol_sha256": sha256_path(
                Path(protocol.__file__).resolve()
            ),
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "scope": {
            "one_preregistered_source_conditioned": True,
            "seed_screening_authorized": False,
            "production_cases_started_inside_source": False,
            "wider_sweep_authorized": False,
        },
    }


def _write_status(
    output: Path,
    *,
    status: str,
    current_step: int,
    stop_reason: str | None,
    elapsed: float,
    records: list[dict[str, Any]],
    step_wall_seconds: list[float],
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any]:
    report = {
        "status": status,
        "current_step": current_step,
        "target_step": protocol.SOURCE_TARGET_STEP,
        "stop_reason": stop_reason,
        "elapsed_wall_seconds": elapsed,
        "records": records,
        "step_wall_seconds": step_wall_seconds,
        "checkpoint": checkpoint,
        "pid": os.getpid(),
        "last_update_unix_time": time.time(),
    }
    atomic_json(output / "run_status.json", report)
    return report


def run_source(
    seed: int,
    output: Path,
    *,
    stop_requested: dict[str, int | None],
) -> dict[str, Any]:
    """Generate one source without screening its scientific morphology."""

    seed = protocol.validate_seed(seed)
    reserve_output_directory(output)
    lock = acquire_output_lock(output)
    started = time.perf_counter()
    try:
        definition = source_definition(seed)
        field, masks = initialize_strict_deg90(definition)
        noise_layout = apply_released_overlapping_noise(
            field,
            definition.noise_amplitude,
            definition.noise_seed,
        )
        del masks
        gc.collect()
        initial_fingerprint = array_fingerprint(field)
        initial_mass = float(
            field_diagnostics(field, definition.lattice).mass
        )
        contract = _contract(
            seed,
            initial_fingerprint=initial_fingerprint,
            initial_mass=initial_mass,
            noise_layout=asdict(noise_layout),
        )
        atomic_json(output / "contract.json", contract)
        solver = RoyPseudospectralSolver(
            definition.lattice,
            definition.parameters,
            fft_workers=protocol.FFT_WORKERS,
        )
        records = [
            source_diagnostics.make_record(
                field,
                solver,
                step=0,
                initial_mass=initial_mass,
                elapsed_wall_seconds=0.0,
                include_energy=True,
                maximum_discarded_imaginary=0.0,
            )
        ]
        step_wall_seconds: list[float] = []
        maximum_discarded = 0.0
        current_step = 0
        stop_reason: str | None = None
        terminal_status = "running"
        _write_status(
            output,
            status=terminal_status,
            current_step=current_step,
            stop_reason=None,
            elapsed=0.0,
            records=records,
            step_wall_seconds=step_wall_seconds,
            checkpoint=None,
        )
        for step in range(1, protocol.SOURCE_TARGET_STEP + 1):
            step_started = time.perf_counter()
            field = solver.propose_step(field)
            step_wall_seconds.append(
                float(time.perf_counter() - step_started)
            )
            current_step = step
            if np.isfinite(solver.last_inverse_imaginary_linf):
                maximum_discarded = max(
                    maximum_discarded,
                    float(solver.last_inverse_imaginary_linf),
                )
            diagnostic_due = (
                step == 1
                or step % protocol.DIAGNOSTIC_INTERVAL == 0
                or step == protocol.SOURCE_TARGET_STEP
            )
            if diagnostic_due:
                record = source_diagnostics.make_record(
                    field,
                    solver,
                    step=step,
                    initial_mass=initial_mass,
                    elapsed_wall_seconds=(
                        time.perf_counter() - started
                    ),
                    include_energy=bool(
                        step % protocol.ENERGY_INTERVAL == 0
                        or step == protocol.SOURCE_TARGET_STEP
                    ),
                    maximum_discarded_imaginary=maximum_discarded,
                )
                records.append(record)
                stop_reason = (
                    source_diagnostics.numerical_abort_reason(record)
                )
                if stop_reason is not None:
                    terminal_status = "numerically_aborted"
                    break
            if stop_requested["signal"] is not None:
                terminal_status = "interrupted"
                stop_reason = (
                    f"deferred_signal_{stop_requested['signal']}"
                )
                break

        checkpoint = write_checkpoint(
            output,
            field,
            step=current_step,
            kinds=(
                ("conditioned_source",)
                if current_step == protocol.SOURCE_TARGET_STEP
                else ("interrupted_source",)
            ),
            stream="paired_source",
        )
        elapsed = time.perf_counter() - started
        if (
            terminal_status == "running"
            and current_step == protocol.SOURCE_TARGET_STEP
        ):
            terminal_status = "completed"
            stop_reason = "fixed_t100_conditioning_completed"
        status = _write_status(
            output,
            status=terminal_status,
            current_step=current_step,
            stop_reason=stop_reason,
            elapsed=elapsed,
            records=records,
            step_wall_seconds=step_wall_seconds,
            checkpoint=checkpoint,
        )
        summary = {
            **status,
            "classification": (
                "paired_repeat_t100_source_completed"
                if terminal_status == "completed"
                else f"paired_repeat_t100_source_{terminal_status}"
            ),
            "seed": seed,
            "initial_fingerprint": initial_fingerprint,
            "initial_mass": initial_mass,
            "conditioned_mass": records[-1]["field"]["mass"],
            "conditioned_fingerprint": checkpoint[
                "field_fingerprint"
            ],
            "execution": {
                "accepted_steps": current_step,
                "median_step_wall_seconds": (
                    float(statistics.median(step_wall_seconds))
                    if step_wall_seconds
                    else None
                ),
                "maximum_step_wall_seconds": (
                    max(step_wall_seconds)
                    if step_wall_seconds
                    else None
                ),
                "maximum_discarded_inverse_imaginary_linf": (
                    maximum_discarded
                ),
            },
            "contract": contract,
            "scope": {
                "production_cases_started": False,
                "seed_screened_or_replaced": False,
            },
        }
        atomic_json(output / "summary.json", summary)
        return summary
    finally:
        release_output_lock(lock)


def verify_completed_source(
    output: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    """Verify the exact source checkpoint used by all four paired cases."""

    seed = protocol.validate_seed(seed)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = summary["checkpoint"]
    metadata_path = Path(checkpoint["metadata_path"])
    field_path = Path(checkpoint["field_path"])
    checks = {
        "status": summary.get("status") == "completed",
        "classification": (
            summary.get("classification")
            == "paired_repeat_t100_source_completed"
        ),
        "seed": int(summary.get("seed", -1)) == seed,
        "step": (
            int(summary.get("current_step", -1))
            == protocol.SOURCE_TARGET_STEP
            and int(checkpoint.get("step", -1))
            == protocol.SOURCE_TARGET_STEP
        ),
        "contract_seed": int(summary["contract"]["seed"]) == seed,
        "contract_runner": (
            summary["contract"]["provenance"]["runner_sha256"]
            == sha256_path(SOURCE_PATH)
        ),
        "contract_model": (
            summary["contract"]["provenance"]["model_sha256"]
            == sha256_path(MODEL_PATH)
        ),
        "contract_protocol": (
            summary["contract"]["provenance"]["protocol_sha256"]
            == sha256_path(Path(protocol.__file__).resolve())
        ),
        "contract_science": (
            summary["contract"]["provenance"][
                "scientific_contract_sha256"
            ]
            == sha256_path(protocol.CONTRACT_PATH)
        ),
        "metadata_hash": (
            sha256_path(metadata_path)
            == checkpoint["metadata_sha256"]
        ),
        "field_hash": (
            sha256_path(field_path) == checkpoint["field_sha256"]
        ),
        "finite": checkpoint.get("finite") is True,
        "conditioned_fingerprint": (
            summary["conditioned_fingerprint"]
            == checkpoint["field_fingerprint"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"conditioned source verification failed: {checks}")
    return {
        "seed": seed,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": sha256_path(summary_path),
        "checkpoint": checkpoint,
        "initial_mass": float(summary["initial_mass"]),
        "conditioned_mass": float(summary["conditioned_mass"]),
        "initial_fingerprint": summary["initial_fingerprint"],
        "conditioned_fingerprint": summary[
            "conditioned_fingerprint"
        ],
        "checks": checks,
    }


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=protocol.NEW_SEEDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
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
        result = run_source(
            arguments.seed,
            arguments.output.resolve(),
            stop_requested=stop_requested,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "seed": result["seed"],
                    "current_step": result["current_step"],
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
        "interrupted": 4,
        "numerically_aborted": 5,
    }.get(result["status"], 6)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
