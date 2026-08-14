#!/usr/bin/env python3
"""Run the fixed Roy-source trajectory from t=0 to the t=1000 checkpoint.

This runner has exactly one target.  It cannot continue to t=2000 and it
cannot change the frozen source-semantic model from the command line.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import signal
import statistics
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy import ndimage

from .model import (
    FROZEN_DEG90,
    RELEASED_SOURCE_COMMIT,
    RoyDeg90Definition,
    RoyPseudospectralSolver,
    apply_released_overlapping_noise,
    field_diagnostics,
    initialize_strict_deg90,
)
from .run_preflight import (
    MODEL_PATH,
    atomic_json,
    atomic_npy,
    array_fingerprint,
    contact_metrics,
    make_figure,
    peak_rss_bytes,
    reserve_output_directory,
    resource_preflight,
    sha256_path,
)


SOURCE_PATH = Path(__file__).resolve()
DEFAULT_OUTPUT = SOURCE_PATH.parent / "results" / "t1000_source_semantic_seed2292"
TARGET_STEP = 1000
DIAGNOSTIC_INTERVAL = 25
CHECKPOINT_INTERVAL = 100
MAXIMUM_WALL_SECONDS = 2.0 * 3600.0
MASS_DRIFT_LIMIT = 1.0e-4
FIELD_MAGNITUDE_CATASTROPHE_LIMIT = 10.0
EXPECTED_INITIAL_FINGERPRINT = (
    "d0574f7e73c5b2c81c5cd2cb564af4f36481d54dad85ec02ead56738628026e2"
)
MINIMUM_PRODUCTION_DISK_HEADROOM_BYTES = 1 << 30


def diagnostic_steps() -> tuple[int, ...]:
    return tuple(range(DIAGNOSTIC_INTERVAL, TARGET_STEP + 1, DIAGNOSTIC_INTERVAL))


def checkpoint_steps() -> tuple[int, ...]:
    return tuple(range(CHECKPOINT_INTERVAL, TARGET_STEP + 1, CHECKPOINT_INTERVAL))


def production_resource_preflight(
    output_directory: Path,
    *,
    completed_step: int = 0,
) -> dict[str, Any]:
    """Check RAM and remaining checkpoint storage before launch or resume."""

    report = resource_preflight(FROZEN_DEG90, output_directory)
    checkpoint_bytes = int(report["estimated_checkpoint_bytes"])
    remaining_regular = sum(step > completed_step for step in checkpoint_steps())
    required_free_disk = (
        (remaining_regular + 1) * checkpoint_bytes
        + MINIMUM_PRODUCTION_DISK_HEADROOM_BYTES
    )
    free_disk = int(report["free_disk_bytes"])
    report["production"] = {
        "completed_step": completed_step,
        "remaining_regular_checkpoints": remaining_regular,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "required_free_disk_bytes": required_free_disk,
        "free_disk_bytes": free_disk,
        "free_disk_sufficient": bool(free_disk >= required_free_disk),
    }
    report["safe_to_start_t1000"] = bool(
        report["safe_to_start_bounded_preflight"]
        and report["production"]["free_disk_sufficient"]
    )
    if not report["safe_to_start_t1000"]:
        raise RuntimeError(
            "t=1000 resource preflight failed: "
            + json.dumps(report, sort_keys=True)
        )
    return report


def four_arm_metrics(
    field: np.ndarray,
    definition: RoyDeg90Definition = FROZEN_DEG90,
) -> dict[str, Any]:
    """Measure whether the welded center remains attached to all four arms."""

    nx, ny, nz = definition.lattice.shape
    radius = max(definition.radius_1, definition.radius_2)
    first_x = nx // 2
    second_x = first_x + definition.radius_1 + definition.radius_2
    center_y = ny // 2
    center_z = nz // 2
    arm_offset = 3 * radius

    x0 = max(0, first_x - radius)
    x1 = min(nx, second_x + radius + 1)
    y0 = max(0, center_y - arm_offset - radius)
    y1 = min(ny, center_y + arm_offset + radius + 1)
    z0 = max(0, center_z - arm_offset - radius)
    z1 = min(nz, center_z + arm_offset + radius + 1)
    local = field[x0:x1, y0:y1, z0:z1]
    anchors = {
        "first_wire_z_minus": (first_x, center_y, center_z - arm_offset),
        "first_wire_z_plus": (first_x, center_y, center_z + arm_offset),
        "second_wire_y_minus": (second_x, center_y - arm_offset, center_z),
        "second_wire_y_plus": (second_x, center_y + arm_offset, center_z),
    }
    first_core_local = (
        first_x - x0,
        center_y - y0,
        center_z - z0,
    )
    structure = ndimage.generate_binary_structure(3, 1)
    thresholds: dict[str, Any] = {}
    for threshold in (0.45, 0.50, 0.55):
        labels, component_count = ndimage.label(
            local >= threshold,
            structure=structure,
        )
        reference_label = int(labels[first_core_local])
        attached: dict[str, bool] = {}
        anchor_labels: dict[str, int] = {}
        for name, (x_index, y_index, z_index) in anchors.items():
            label = int(
                labels[
                    x_index - x0,
                    y_index - y0,
                    z_index - z0,
                ]
            )
            anchor_labels[name] = label
            attached[name] = bool(
                reference_label != 0 and label == reference_label
            )
        thresholds[f"{threshold:.2f}"] = {
            "local_component_count": int(component_count),
            "reference_label": reference_label,
            "anchor_labels": anchor_labels,
            "arm_attached": attached,
            "attached_arm_count": int(sum(attached.values())),
            "all_four_arms_attached": bool(all(attached.values())),
        }

    central_half_extent = 2 * radius
    central = field[
        x0:x1,
        center_y - central_half_extent : center_y + central_half_extent + 1,
        center_z - central_half_extent : center_z + central_half_extent + 1,
    ]
    return {
        "arm_offset_grid_points": arm_offset,
        "arm_offset_over_R": 3.0,
        "thresholds": thresholds,
        "central_roi_composition_sum": float(np.sum(central, dtype=np.float64)),
        "central_roi_cell_count": int(central.size),
    }


def make_record(
    field: np.ndarray,
    solver: RoyPseudospectralSolver,
    *,
    step: int,
    initial_mass: float,
    elapsed_wall_seconds: float,
    include_energy: bool,
    maximum_discarded_imaginary: float,
) -> dict[str, Any]:
    diagnostics = field_diagnostics(field, FROZEN_DEG90.lattice).to_dict()
    if not bool(diagnostics["finite"]):
        diagnostics = {
            key: (
                None
                if isinstance(value, float) and not np.isfinite(value)
                else value
            )
            for key, value in diagnostics.items()
        }
        return {
            "step": step,
            "time": float(step * FROZEN_DEG90.parameters.timestep),
            "elapsed_wall_seconds": elapsed_wall_seconds,
            "field": diagnostics,
            "relative_mass_drift": None,
            "contact": None,
            "four_arm_topology": None,
            "free_energy": None,
            "last_discarded_inverse_imaginary_linf": None,
            "maximum_discarded_inverse_imaginary_linf": (
                maximum_discarded_imaginary
                if np.isfinite(maximum_discarded_imaginary)
                else None
            ),
        }
    relative_mass_drift = (
        abs(float(diagnostics["mass"]) - initial_mass) / abs(initial_mass)
        if initial_mass != 0.0
        else 0.0
    )
    return {
        "step": step,
        "time": float(step * FROZEN_DEG90.parameters.timestep),
        "elapsed_wall_seconds": elapsed_wall_seconds,
        "field": diagnostics,
        "relative_mass_drift": relative_mass_drift,
        "contact": contact_metrics(field, FROZEN_DEG90),
        "four_arm_topology": four_arm_metrics(field),
        "free_energy": solver.free_energy(field) if include_energy else None,
        "last_discarded_inverse_imaginary_linf": (
            solver.last_inverse_imaginary_linf
        ),
        "maximum_discarded_inverse_imaginary_linf": (
            maximum_discarded_imaginary
        ),
    }


def numerical_abort_reason(record: dict[str, Any]) -> str | None:
    field = record["field"]
    if not bool(field["finite"]):
        return "nonfinite_field"
    maximum_absolute = max(
        abs(float(field["minimum"])),
        abs(float(field["maximum"])),
    )
    if maximum_absolute > FIELD_MAGNITUDE_CATASTROPHE_LIMIT:
        return "catastrophic_field_magnitude"
    if (
        record["relative_mass_drift"] is not None
        and float(record["relative_mass_drift"]) > MASS_DRIFT_LIMIT
    ):
        return "mass_drift_exceeded"
    return None


def checkpoint_paths(output_directory: Path, step: int) -> tuple[Path, Path]:
    stem = f"checkpoint-step-{step:04d}"
    return output_directory / f"{stem}.npy", output_directory / f"{stem}.json"


def write_checkpoint(
    output_directory: Path,
    field: np.ndarray,
    *,
    step: int,
) -> dict[str, Any]:
    field_path, metadata_path = checkpoint_paths(output_directory, step)
    if field_path.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint at step {step}")
    write_seconds = atomic_npy(field_path, field)
    metadata = {
        "step": step,
        "field_path": str(field_path),
        "field_bytes": field_path.stat().st_size,
        "field_sha256": sha256_path(field_path),
        "field_fingerprint": array_fingerprint(field),
        "shape": list(field.shape),
        "dtype": field.dtype.str,
        "finite": bool(np.isfinite(field).all()),
        "write_wall_seconds": write_seconds,
    }
    atomic_json(metadata_path, metadata)
    metadata["metadata_path"] = str(metadata_path)
    metadata["metadata_sha256"] = sha256_path(metadata_path)
    return metadata


def load_checkpoint(metadata: dict[str, Any]) -> np.ndarray:
    metadata_path = Path(str(metadata["metadata_path"]))
    if sha256_path(metadata_path) != metadata["metadata_sha256"]:
        raise RuntimeError("checkpoint metadata hash mismatch")
    stored = json.loads(metadata_path.read_text(encoding="utf-8"))
    field_path = Path(str(stored["field_path"]))
    if sha256_path(field_path) != stored["field_sha256"]:
        raise RuntimeError("checkpoint field hash mismatch")
    field = np.load(field_path, allow_pickle=False)
    if list(field.shape) != stored["shape"]:
        raise RuntimeError("checkpoint shape mismatch")
    if field.dtype.str != stored["dtype"]:
        raise RuntimeError("checkpoint dtype mismatch")
    if array_fingerprint(field) != stored["field_fingerprint"]:
        raise RuntimeError("checkpoint field fingerprint mismatch")
    if not np.isfinite(field).all():
        raise RuntimeError("checkpoint field is nonfinite")
    return np.asarray(field, dtype=np.float64)


def acquire_output_lock(output_directory: Path) -> Any:
    lock_path = output_directory / ".run.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"another process holds the run lock: {lock_path}")
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def contract_identity() -> dict[str, Any]:
    return {
        "target_step": TARGET_STEP,
        "diagnostic_interval": DIAGNOSTIC_INTERVAL,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "maximum_wall_seconds": MAXIMUM_WALL_SECONDS,
        "mass_drift_limit": MASS_DRIFT_LIMIT,
        "field_magnitude_catastrophe_limit": (
            FIELD_MAGNITUDE_CATASTROPHE_LIMIT
        ),
        "expected_initial_fingerprint": EXPECTED_INITIAL_FINGERPRINT,
        "definition": asdict(FROZEN_DEG90),
        "released_source_commit": RELEASED_SOURCE_COMMIT,
        "runner_sha256": sha256_path(SOURCE_PATH),
        "model_sha256": sha256_path(MODEL_PATH),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def verify_resume_contract(
    contract: dict[str, Any],
    *,
    fft_workers: int,
) -> None:
    current = contract_identity()
    for key in (
        "target_step",
        "diagnostic_interval",
        "checkpoint_interval",
        "maximum_wall_seconds",
        "mass_drift_limit",
        "field_magnitude_catastrophe_limit",
        "expected_initial_fingerprint",
        "definition",
        "released_source_commit",
        "runner_sha256",
        "model_sha256",
    ):
        if contract.get(key) != current[key]:
            raise RuntimeError(f"resume contract mismatch: {key}")
    if int(contract.get("fft_workers", 0)) != fft_workers:
        raise RuntimeError("resume contract mismatch: fft_workers")


def descriptive_t1000_assessment(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    by_step = {int(record["step"]): record for record in records}
    persistence_steps = (800, 900, 1000)
    available = all(step in by_step for step in persistence_steps)
    thresholds: dict[str, Any] = {}
    for threshold in ("0.45", "0.50", "0.55"):
        robust_junction = bool(
            available
            and all(
                by_step[step]["contact"]["thresholds"][threshold][
                    "cores_connected_locally"
                ]
                for step in persistence_steps
            )
        )
        four_arms = bool(
            available
            and all(
                by_step[step]["four_arm_topology"]["thresholds"][threshold][
                    "all_four_arms_attached"
                ]
                for step in persistence_steps
            )
        )
        thresholds[threshold] = {
            "junction_persistent_at_800_900_1000": robust_junction,
            "all_four_arms_attached_at_800_900_1000": four_arms,
            "endpoint_contact_plane_cells": (
                by_step[1000]["contact"]["thresholds"][threshold][
                    "contact_plane_cells"
                ]
                if 1000 in by_step
                else None
            ),
        }
    central_growth = None
    if 0 in by_step and 1000 in by_step:
        central_growth = (
            by_step[1000]["four_arm_topology"][
                "central_roi_composition_sum"
            ]
            - by_step[0]["four_arm_topology"][
                "central_roi_composition_sum"
            ]
        )
    return {
        "persistence_steps": list(persistence_steps),
        "required_records_available": available,
        "thresholds": thresholds,
        "central_roi_composition_sum_change_t0_to_t1000": central_growth,
        "paper_has_no_quantitative_t1000_neck_size_target": True,
        "manual_visual_morphology_comparison_required": True,
        "automatic_t2000_continuation_authorized": False,
        "initializer_independence_assessed": False,
    }


def write_status(output_directory: Path, status: dict[str, Any]) -> None:
    atomic_json(output_directory / "run_status.json", status)


def run(
    output_directory: Path,
    *,
    fft_workers: int,
    resume: bool,
    stop_requested: dict[str, int | None],
) -> dict[str, Any]:
    contract_path = output_directory / "contract.json"
    status_path = output_directory / "run_status.json"
    session_started = time.perf_counter()

    if resume:
        if not output_directory.is_dir():
            raise FileNotFoundError(f"resume output is missing: {output_directory}")
        lock_handle = acquire_output_lock(output_directory)
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        verify_resume_contract(contract, fft_workers=fft_workers)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status["status"] in {"completed", "numerically_aborted", "wall_cap"}:
            raise RuntimeError("terminal t=1000 output cannot resume")
        if not status["checkpoints"]:
            raise RuntimeError("no complete checkpoint is available to resume")
        latest = status["checkpoints"][-1]
        field = load_checkpoint(latest)
        current_step = int(latest["step"])
        initial_mass = float(contract["initial_mass"])
        records = list(status["records"])
        step_wall_seconds = list(status["step_wall_seconds"])
        checkpoints = list(status["checkpoints"])
        previous_elapsed = float(status["elapsed_wall_seconds"])
        maximum_discarded = float(
            status["maximum_discarded_inverse_imaginary_linf"]
        )
        initial_plane = np.load(
            output_directory / "initial-contact-plane.npy",
            allow_pickle=False,
        )
        resources = production_resource_preflight(
            output_directory,
            completed_step=current_step,
        )
    else:
        resources = production_resource_preflight(output_directory)
        reserve_output_directory(output_directory)
        lock_handle = acquire_output_lock(output_directory)
        field, masks = initialize_strict_deg90(FROZEN_DEG90)
        apply_released_overlapping_noise(
            field,
            FROZEN_DEG90.noise_amplitude,
            FROZEN_DEG90.noise_seed,
        )
        del masks
        fingerprint = array_fingerprint(field)
        if fingerprint != EXPECTED_INITIAL_FINGERPRINT:
            raise RuntimeError(
                "fresh initial fingerprint does not match the validated exact grid"
            )
        initial_mass = float(
            field_diagnostics(field, FROZEN_DEG90.lattice).mass
        )
        contract = {
            **contract_identity(),
            "initial_mass": initial_mass,
            "fft_workers": fft_workers,
            "resource_preflight": resources,
            "long_run_user_authorized": True,
            "t2000_authorized": False,
        }
        atomic_json(contract_path, contract)
        contact_index = (
            FROZEN_DEG90.lattice.shape[0] // 2 + FROZEN_DEG90.radius_1
        )
        initial_plane = field[contact_index].copy()
        atomic_npy(output_directory / "initial-contact-plane.npy", initial_plane)
        current_step = 0
        records = []
        step_wall_seconds = []
        checkpoints = []
        previous_elapsed = 0.0
        maximum_discarded = 0.0

    try:
        solver = RoyPseudospectralSolver(
            FROZEN_DEG90.lattice,
            FROZEN_DEG90.parameters,
            fft_workers=fft_workers,
        )
        if current_step == 0 and not records:
            records.append(
                make_record(
                    field,
                    solver,
                    step=0,
                    initial_mass=initial_mass,
                    elapsed_wall_seconds=0.0,
                    include_energy=True,
                    maximum_discarded_imaginary=0.0,
                )
            )
        start_elapsed = previous_elapsed + (
            time.perf_counter() - session_started
        )
        write_status(
            output_directory,
            {
                "status": "running",
                "current_step": current_step,
                "target_step": TARGET_STEP,
                "elapsed_wall_seconds": start_elapsed,
                "records": records,
                "step_wall_seconds": step_wall_seconds,
                "checkpoints": checkpoints,
                "maximum_discarded_inverse_imaginary_linf": (
                    maximum_discarded
                ),
                "last_update_unix_time": time.time(),
                "pid": os.getpid(),
            },
        )
        print(
            json.dumps(
                {
                    "status": "running",
                    "step": current_step,
                    "target_step": TARGET_STEP,
                    "output": str(output_directory),
                },
                sort_keys=True,
            ),
            flush=True,
        )

        terminal_status: str | None = None
        stop_reason: str | None = None
        for step in range(current_step + 1, TARGET_STEP + 1):
            step_started = time.perf_counter()
            field = solver.propose_step(field)
            step_wall_seconds.append(float(time.perf_counter() - step_started))
            current_step = step
            if np.isfinite(solver.last_inverse_imaginary_linf):
                maximum_discarded = max(
                    maximum_discarded,
                    solver.last_inverse_imaginary_linf,
                )
            elapsed = previous_elapsed + (
                time.perf_counter() - session_started
            )
            signal_pending = stop_requested["signal"] is not None
            wall_cap_reached = elapsed >= MAXIMUM_WALL_SECONDS
            diagnostic_due = bool(
                step == 1
                or step % DIAGNOSTIC_INTERVAL == 0
                or signal_pending
                or wall_cap_reached
                or step == TARGET_STEP
            )
            record: dict[str, Any] | None = None
            if diagnostic_due:
                record = make_record(
                    field,
                    solver,
                    step=step,
                    initial_mass=initial_mass,
                    elapsed_wall_seconds=elapsed,
                    include_energy=bool(
                        step % CHECKPOINT_INTERVAL == 0
                        or step == TARGET_STEP
                    ),
                    maximum_discarded_imaginary=maximum_discarded,
                )
                records.append(record)
                stop_reason = numerical_abort_reason(record)
                if stop_reason is not None:
                    terminal_status = "numerically_aborted"

            checkpoint_due = bool(
                step % CHECKPOINT_INTERVAL == 0
                or signal_pending
                or wall_cap_reached
                or terminal_status is not None
                or step == TARGET_STEP
            )
            if checkpoint_due:
                checkpoint = write_checkpoint(
                    output_directory,
                    field,
                    step=step,
                )
                checkpoints.append(checkpoint)
                status = {
                    "status": "running",
                    "current_step": step,
                    "target_step": TARGET_STEP,
                    "elapsed_wall_seconds": elapsed,
                    "records": records,
                    "step_wall_seconds": step_wall_seconds,
                    "checkpoints": checkpoints,
                    "maximum_discarded_inverse_imaginary_linf": (
                        maximum_discarded
                    ),
                    "last_update_unix_time": time.time(),
                    "pid": os.getpid(),
                }
                write_status(output_directory, status)
                print(
                    json.dumps(
                        {
                            "status": status["status"],
                            "step": step,
                            "elapsed_wall_seconds": elapsed,
                            "checkpoint": checkpoint["field_path"],
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
                stop_reason = f"deferred_signal_{stop_requested['signal']}"
                break
            if wall_cap_reached:
                terminal_status = "wall_cap"
                stop_reason = "two_hour_wall_cap_reached"
                break
            if step == TARGET_STEP:
                terminal_status = "completed"
                stop_reason = "fixed_t1000_target_reached"
                break

        elapsed_total = previous_elapsed + (
            time.perf_counter() - session_started
        )
        status = {
            "status": terminal_status,
            "current_step": current_step,
            "target_step": TARGET_STEP,
            "stop_reason": stop_reason,
            "elapsed_wall_seconds": elapsed_total,
            "records": records,
            "step_wall_seconds": step_wall_seconds,
            "checkpoints": checkpoints,
            "maximum_discarded_inverse_imaginary_linf": maximum_discarded,
            "last_update_unix_time": time.time(),
            "pid": os.getpid(),
        }
        write_status(output_directory, status)

        if terminal_status == "interrupted":
            return status

        figure_path = output_directory / f"contact-plane-step-{current_step:04d}.png"
        contact_index = (
            FROZEN_DEG90.lattice.shape[0] // 2 + FROZEN_DEG90.radius_1
        )
        make_figure(
            figure_path,
            initial_plane,
            field[contact_index],
            FROZEN_DEG90,
            current_step,
        )
        assessment = (
            descriptive_t1000_assessment(records)
            if terminal_status == "completed"
            else None
        )
        summary = {
            "run": "Roy et al. 2021 source-semantic t=1000 trajectory",
            "status": terminal_status,
            "classification": (
                "roy_t1000_completed_pending_science_audit"
                if terminal_status == "completed"
                else f"roy_t1000_{terminal_status}"
            ),
            "stop_reason": stop_reason,
            "contract": contract,
            "execution": {
                "completed_step": current_step,
                "elapsed_wall_seconds": elapsed_total,
                "step_wall_seconds_count": len(step_wall_seconds),
                "median_step_wall_seconds": (
                    float(statistics.median(step_wall_seconds))
                    if step_wall_seconds
                    else None
                ),
                "peak_rss_bytes": peak_rss_bytes(),
                "maximum_discarded_inverse_imaginary_linf": (
                    maximum_discarded
                ),
            },
            "records": records,
            "checkpoints": checkpoints,
            "descriptive_t1000_assessment": assessment,
            "outputs": {
                "run_status": str(status_path),
                "figure": str(figure_path),
                "figure_sha256": sha256_path(figure_path),
            },
            "interpretation_boundary": {
                "t1000_is_intermediate_morphology_checkpoint": True,
                "breakup_timing_assessed": False,
                "initializer_independence_assessed": False,
                "automatic_t2000_continuation_authorized": False,
            },
        }
        summary_path = output_directory / "summary.json"
        summary["outputs"]["summary"] = str(summary_path)
        atomic_json(summary_path, summary)
        return summary
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fft-workers", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    output = arguments.output.resolve()
    stop_requested: dict[str, int | None] = {"signal": None}

    def request_stop(signal_number: int, _frame: object) -> None:
        stop_requested["signal"] = signal_number

    handled = (signal.SIGINT, signal.SIGTERM)
    previous_handlers = {
        number: signal.getsignal(number) for number in handled
    }
    for number in handled:
        signal.signal(number, request_stop)
    try:
        result = run(
            output,
            fft_workers=arguments.fft_workers,
            resume=arguments.resume,
            stop_requested=stop_requested,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "classification": result.get("classification"),
                    "current_step": result.get(
                        "execution", result
                    ).get("completed_step", result.get("current_step")),
                    "output": str(output),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if result["status"] in {"completed", "interrupted"} else 2
    except Exception as exc:
        if output.is_dir():
            failure = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "pid": os.getpid(),
                "unix_time": time.time(),
            }
            try:
                atomic_json(output / "failure.json", failure)
            except Exception:
                pass
        raise
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    raise SystemExit(main())
