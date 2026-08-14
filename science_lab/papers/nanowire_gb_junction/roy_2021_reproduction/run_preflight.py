#!/usr/bin/env python3
"""Run one bounded Roy-source CPU-port preflight.

Two modes are deliberately separate:

``reduced``
    Twenty steps on ``96 x 128 x 768``.  This preserves the released thin/long
    stride ratio and contact resolution while shortening one periodic wire.
    It is an operator and determinism check, not a physical reproduction.

``exact-timing``
    Five steps on the published ``96 x 768 x 768`` lattice, with the first two
    treated as FFT warmup, followed by one atomic checkpoint write.  It
    estimates the cost of a later 2,000-step reproduction and cannot launch
    that trajectory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import shutil
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage

from .model import (
    FROZEN_DEG90,
    RELEASED_SOURCE_COMMIT,
    PeriodicLattice,
    RoyDeg90Definition,
    RoyPseudospectralSolver,
    apply_released_overlapping_noise,
    digital_gap_diagnostics,
    field_diagnostics,
    initialize_strict_deg90,
)


SOURCE_PATH = Path(__file__).resolve()
MODEL_PATH = SOURCE_PATH.with_name("model.py")
DIRECTORY = SOURCE_PATH.parent
DEFAULT_RESULTS_DIRECTORY = DIRECTORY / "results"
REDUCED_SHAPE = (96, 128, 768)
REDUCED_STEPS = 20
EXACT_TIMING_STEPS = 5
EXACT_TIMING_WARMUP_STEPS = 2
MASS_DRIFT_LIMIT = 1.0e-4
FIELD_MAGNITUDE_CATASTROPHE_LIMIT = 10.0
ESTIMATED_PEAK_BYTES_PER_CELL = 256
MINIMUM_DISK_HEADROOM_BYTES = 1 << 30


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_fingerprint(field: np.ndarray) -> str:
    if not field.flags.c_contiguous:
        field = np.ascontiguousarray(field)
    digest = hashlib.sha256()
    digest.update(str(field.shape).encode("ascii"))
    digest.update(field.dtype.str.encode("ascii"))
    digest.update(memoryview(field).cast("B"))
    return digest.hexdigest()


def reserve_output_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"output directory already exists; refusing to overwrite: {path}"
        ) from exc


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_npy(path: Path, field: np.ndarray) -> float:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    started = time.perf_counter()
    try:
        with temporary.open("wb") as handle:
            np.save(handle, field, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return float(time.perf_counter() - started)


def peak_rss_bytes() -> int:
    measured = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform.startswith("linux"):
        return measured * 1024
    return measured


def resource_preflight(
    definition: RoyDeg90Definition,
    output_directory: Path,
) -> dict[str, Any]:
    """Conservatively check RAM and disk before allocating the full lattice."""

    cell_count = definition.lattice.cell_count
    estimated_peak_bytes = cell_count * ESTIMATED_PEAK_BYTES_PER_CELL
    estimated_checkpoint_bytes = cell_count * np.dtype(np.float64).itemsize + 256
    required_free_disk_bytes = (
        2 * estimated_checkpoint_bytes + MINIMUM_DISK_HEADROOM_BYTES
    )

    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
    physical_memory_bytes = page_size * physical_pages
    try:
        import psutil

        available_memory_bytes: int | None = int(
            psutil.virtual_memory().available
        )
    except (ImportError, AttributeError):
        available_memory_bytes = None

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    free_disk_bytes = int(shutil.disk_usage(output_directory.parent).free)
    checks = {
        "physical_memory_sufficient": bool(
            physical_memory_bytes >= estimated_peak_bytes
        ),
        "available_memory_sufficient_if_measurable": bool(
            available_memory_bytes is None
            or available_memory_bytes >= estimated_peak_bytes
        ),
        "free_disk_sufficient": bool(
            free_disk_bytes >= required_free_disk_bytes
        ),
    }
    report = {
        "cell_count": cell_count,
        "estimated_peak_bytes": estimated_peak_bytes,
        "estimate_bytes_per_cell": ESTIMATED_PEAK_BYTES_PER_CELL,
        "physical_memory_bytes": physical_memory_bytes,
        "available_memory_bytes": available_memory_bytes,
        "estimated_checkpoint_bytes": estimated_checkpoint_bytes,
        "required_free_disk_bytes": required_free_disk_bytes,
        "free_disk_bytes": free_disk_bytes,
        "checks": checks,
        "safe_to_start_bounded_preflight": bool(all(checks.values())),
        "estimate_is_conservative_not_measured": True,
    }
    if not report["safe_to_start_bounded_preflight"]:
        raise RuntimeError(
            "resource preflight failed before lattice allocation: "
            + json.dumps(report, sort_keys=True)
        )
    return report


def definition_for_mode(mode: str) -> tuple[RoyDeg90Definition, int]:
    if mode == "reduced":
        return (
            RoyDeg90Definition(
                lattice=PeriodicLattice(
                    REDUCED_SHAPE, FROZEN_DEG90.lattice.spacing
                ),
                parameters=FROZEN_DEG90.parameters,
                radius_1=FROZEN_DEG90.radius_1,
                radius_2=FROZEN_DEG90.radius_2,
                solid_composition=FROZEN_DEG90.solid_composition,
                noise_amplitude=FROZEN_DEG90.noise_amplitude,
                noise_seed=FROZEN_DEG90.noise_seed,
            ),
            REDUCED_STEPS,
        )
    if mode == "exact-timing":
        return FROZEN_DEG90, EXACT_TIMING_STEPS
    raise ValueError(f"unsupported mode: {mode}")


def contact_metrics(
    field: np.ndarray,
    definition: RoyDeg90Definition,
) -> dict[str, Any]:
    """Measure local connection using Roy's published c=0.5 convention."""

    nx, ny, nz = definition.lattice.shape
    first_center = nx // 2
    second_center = first_center + definition.radius_1 + definition.radius_2
    contact_index = first_center + definition.radius_1
    center_y = ny // 2
    center_z = nz // 2
    physical_radius = (
        0.5
        * (definition.radius_1 + definition.radius_2)
        * definition.lattice.spacing
    )

    half_extent = 2 * max(definition.radius_1, definition.radius_2)
    x0 = max(0, first_center - definition.radius_1)
    x1 = min(nx, second_center + definition.radius_2 + 1)
    y0 = max(0, center_y - half_extent)
    y1 = min(ny, center_y + half_extent + 1)
    z0 = max(0, center_z - half_extent)
    z1 = min(nz, center_z + half_extent + 1)
    local = field[x0:x1, y0:y1, z0:z1]
    plane = field[contact_index]
    structure = ndimage.generate_binary_structure(3, 1)

    thresholds: dict[str, Any] = {}
    for threshold in (0.45, 0.50, 0.55):
        binary = local >= threshold
        labels, component_count = ndimage.label(binary, structure=structure)
        first_label = int(
            labels[
                first_center - x0,
                center_y - y0,
                center_z - z0,
            ]
        )
        second_label = int(
            labels[
                second_center - x0,
                center_y - y0,
                center_z - z0,
            ]
        )
        plane_count = int(np.count_nonzero(plane >= threshold))
        equivalent_radius = float(
            np.sqrt(
                plane_count * definition.lattice.spacing**2 / np.pi
            )
        )
        thresholds[f"{threshold:.2f}"] = {
            "local_component_count": int(component_count),
            "first_core_label": first_label,
            "second_core_label": second_label,
            "cores_connected_locally": bool(
                first_label != 0 and first_label == second_label
            ),
            "contact_plane_cells": plane_count,
            "contact_plane_equivalent_radius": equivalent_radius,
            "contact_plane_equivalent_radius_over_R": float(
                equivalent_radius / physical_radius
            ),
        }

    return {
        "contact_index": contact_index,
        "center_gap_value": float(
            field[contact_index, center_y, center_z]
        ),
        "contact_plane_minimum": float(np.min(plane)),
        "contact_plane_maximum": float(np.max(plane)),
        "first_wire_core_value": float(
            field[first_center, center_y, center_z]
        ),
        "second_wire_core_value": float(
            field[second_center, center_y, center_z]
        ),
        "thresholds": thresholds,
    }


def make_figure(
    output_path: Path,
    initial_plane: np.ndarray,
    endpoint_plane: np.ndarray,
    definition: RoyDeg90Definition,
    endpoint_step: int,
) -> None:
    extent = (
        -0.5 * definition.lattice.physical_lengths[2],
        0.5 * definition.lattice.physical_lengths[2],
        -0.5 * definition.lattice.physical_lengths[1],
        0.5 * definition.lattice.physical_lengths[1],
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), constrained_layout=True)
    for axis, plane, title in (
        (axes[0], initial_plane, "Released source field, t=0"),
        (axes[1], endpoint_plane, f"CPU port, t={endpoint_step}"),
    ):
        image = axis.imshow(
            plane,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=-0.1,
            vmax=1.1,
            interpolation="nearest",
            aspect="equal",
        )
        if float(np.min(plane)) <= 0.5 <= float(np.max(plane)):
            axis.contour(
                plane,
                levels=[0.5],
                colors=["white"],
                linewidths=0.8,
                origin="lower",
                extent=extent,
            )
        axis.set_title(title)
        axis.set_xlabel("z")
        axis.set_ylabel("y")
    colorbar = figure.colorbar(image, ax=axes, shrink=0.86)
    colorbar.set_label("composition c")
    figure.suptitle("Nominal tangent plane (thin-axis index 60)")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_preflight(
    mode: str,
    output_directory: Path,
    *,
    fft_workers: int,
) -> dict[str, Any]:
    definition, step_limit = definition_for_mode(mode)
    resources = resource_preflight(definition, output_directory)
    reserve_output_directory(output_directory)
    run_started = time.perf_counter()

    binary_field, masks = initialize_strict_deg90(definition)
    gap = digital_gap_diagnostics(masks, definition.lattice)
    binary_diagnostics = field_diagnostics(
        binary_field, definition.lattice, masks
    ).to_dict()
    noise_started = time.perf_counter()
    noise_layout = apply_released_overlapping_noise(
        binary_field,
        definition.noise_amplitude,
        definition.noise_seed,
    )
    noise_wall_seconds = float(time.perf_counter() - noise_started)
    initial_diagnostics = field_diagnostics(
        binary_field, definition.lattice, masks
    ).to_dict()
    initial_contact = contact_metrics(binary_field, definition)
    initial_fingerprint = array_fingerprint(binary_field)
    initial_mass = float(initial_diagnostics["mass"])
    contact_index = int(initial_contact["contact_index"])
    initial_plane = binary_field[contact_index].copy()
    del masks
    gc.collect()

    solver = RoyPseudospectralSolver(
        definition.lattice,
        definition.parameters,
        fft_workers=fft_workers,
    )
    initial_energy = solver.free_energy(binary_field)
    pre_step_setup_wall_seconds = float(time.perf_counter() - run_started)
    records: list[dict[str, Any]] = [
        {
            "step": 0,
            "time": 0.0,
            "field": initial_diagnostics,
            "free_energy": initial_energy,
            "contact": initial_contact,
            "relative_mass_drift": 0.0,
            "inverse_imaginary_linf": 0.0,
        }
    ]
    step_wall_seconds: list[float] = []
    maximum_relative_mass_drift = 0.0
    maximum_inverse_imaginary_linf = 0.0
    maximum_absolute_field = max(
        abs(float(initial_diagnostics["minimum"])),
        abs(float(initial_diagnostics["maximum"])),
    )
    deterministic_repeat_equal: bool | None = None
    deterministic_repeat_wall_seconds: float | None = None
    field = binary_field
    completed_steps = 0

    for step in range(1, step_limit + 1):
        step_started = time.perf_counter()
        proposal = solver.propose_step(field)
        step_wall = float(time.perf_counter() - step_started)
        step_wall_seconds.append(step_wall)

        if mode == "reduced" and step == 1:
            repeat_started = time.perf_counter()
            repeated = solver.propose_step(field)
            deterministic_repeat_wall_seconds = float(
                time.perf_counter() - repeat_started
            )
            deterministic_repeat_equal = bool(
                np.array_equal(proposal, repeated)
            )
            del repeated

        field = proposal
        completed_steps = step
        diagnostics = field_diagnostics(field, definition.lattice).to_dict()
        current_mass = diagnostics["mass"]
        relative_mass_drift = (
            abs(float(current_mass) - initial_mass) / abs(initial_mass)
            if initial_mass != 0.0
            else 0.0
        )
        maximum_relative_mass_drift = max(
            maximum_relative_mass_drift, relative_mass_drift
        )
        maximum_inverse_imaginary_linf = max(
            maximum_inverse_imaginary_linf,
            solver.last_inverse_imaginary_linf,
        )
        if diagnostics["finite"]:
            maximum_absolute_field = max(
                maximum_absolute_field,
                abs(float(diagnostics["minimum"])),
                abs(float(diagnostics["maximum"])),
            )
        records.append(
            {
                "step": step,
                "time": float(step * definition.parameters.timestep),
                "field": diagnostics,
                "contact": contact_metrics(field, definition),
                "relative_mass_drift": relative_mass_drift,
                "inverse_imaginary_linf": (
                    solver.last_inverse_imaginary_linf
                ),
                "step_wall_seconds": step_wall,
            }
        )
        if not diagnostics["finite"]:
            break

    final_energy = (
        solver.free_energy(field)
        if bool(records[-1]["field"]["finite"])
        else None
    )
    records[-1]["free_energy"] = final_energy
    endpoint_fingerprint = array_fingerprint(field)
    checkpoint_path = output_directory / "endpoint.npy"
    checkpoint_write_wall_seconds = atomic_npy(checkpoint_path, field)
    checkpoint_sha256 = sha256_path(checkpoint_path)
    endpoint_plane = field[contact_index].copy()
    figure_path = output_directory / "contact-plane.png"
    make_figure(
        figure_path,
        initial_plane,
        endpoint_plane,
        definition,
        completed_steps,
    )

    all_finite = bool(all(record["field"]["finite"] for record in records))
    checks = {
        "source_commit_frozen": bool(
            RELEASED_SOURCE_COMMIT
            == "a275dc0639c8f1e3dad795234e4a5ea7a7aeb5b3"
        ),
        "strict_mask_has_one_empty_contact_node": bool(
            gap["empty_indices_between_wires"] == [60]
        ),
        "released_noise_stride_mismatch_preserved": bool(
            noise_layout.stride_mismatch_present
        ),
        "all_fields_finite": all_finite,
        "mass_drift_within_1e_4": bool(
            maximum_relative_mass_drift <= MASS_DRIFT_LIMIT
        ),
        "no_catastrophic_field_growth": bool(
            maximum_absolute_field <= FIELD_MAGNITUDE_CATASTROPHE_LIMIT
        ),
        "requested_steps_completed": bool(completed_steps == step_limit),
        "reduced_repeat_is_bitwise_deterministic": (
            deterministic_repeat_equal if mode == "reduced" else None
        ),
        "published_full_grid_used": bool(
            definition.lattice.shape == FROZEN_DEG90.lattice.shape
        ),
    }
    required_checks = [
        checks["source_commit_frozen"],
        checks["strict_mask_has_one_empty_contact_node"],
        checks["released_noise_stride_mismatch_preserved"],
        checks["all_fields_finite"],
        checks["mass_drift_within_1e_4"],
        checks["no_catastrophic_field_growth"],
        checks["requested_steps_completed"],
    ]
    if mode == "reduced":
        required_checks.append(
            bool(checks["reduced_repeat_is_bitwise_deterministic"])
        )
    preflight_passed = bool(all(required_checks))
    classification = (
        f"roy_{mode.replace('-', '_')}_cpu_port_preflight_passed"
        if preflight_passed
        else f"roy_{mode.replace('-', '_')}_cpu_port_preflight_failed"
    )

    runtime_projection: dict[str, Any] | None = None
    if mode == "exact-timing" and step_wall_seconds:
        timed_steps = step_wall_seconds[EXACT_TIMING_WARMUP_STEPS:]
        median_step = float(statistics.median(timed_steps))
        checkpoint_count_t1000 = 10
        checkpoint_count_t2000 = 20
        safety_factor = 1.3
        checkpoint_bytes = checkpoint_path.stat().st_size
        projected_t1000 = safety_factor * (
            pre_step_setup_wall_seconds
            + median_step * 1000
            + checkpoint_write_wall_seconds * checkpoint_count_t1000
        )
        projected_t2000 = safety_factor * (
            pre_step_setup_wall_seconds
            + median_step * 2000
            + checkpoint_write_wall_seconds * checkpoint_count_t2000
        )
        runtime_projection = {
            "median_measured_step_wall_seconds": median_step,
            "measured_step_wall_seconds_after_warmup": timed_steps,
            "warmup_steps_excluded": EXACT_TIMING_WARMUP_STEPS,
            "one_time_pre_step_setup_wall_seconds": (
                pre_step_setup_wall_seconds
            ),
            "one_time_noise_generation_wall_seconds": noise_wall_seconds,
            "checkpoint_write_wall_seconds": checkpoint_write_wall_seconds,
            "checkpoint_interval_steps": 100,
            "projected_checkpoint_storage_t1000_bytes": (
                checkpoint_bytes * checkpoint_count_t1000
            ),
            "projected_checkpoint_storage_t2000_bytes": (
                checkpoint_bytes * checkpoint_count_t2000
            ),
            "safety_factor": safety_factor,
            "projected_t1000_wall_seconds": projected_t1000,
            "projected_t1000_wall_hours": projected_t1000 / 3600.0,
            "projected_t2000_wall_seconds": projected_t2000,
            "projected_t2000_wall_hours": projected_t2000 / 3600.0,
            "warning_required_before_t1000": bool(
                projected_t1000 > 2.0 * 3600.0
            ),
            "warning_required_before_t2000": bool(
                projected_t2000 > 2.0 * 3600.0
            ),
            "projection_is_not_a_simulation_result": True,
            "projection_is_provisional_from_three_post_warmup_steps": True,
            "production_diagnostic_overhead_beyond_checkpointing_measured": (
                False
            ),
            "long_run_requires_explicit_launch_flag": True,
        }

    summary = {
        "run": "Roy et al. 2021 source-semantic CPU-port preflight",
        "status": "completed",
        "mode": mode,
        "classification": classification,
        "preflight_passed": preflight_passed,
        "source_contract": {
            "released_source_commit": RELEASED_SOURCE_COMMIT,
            "source_url": (
                "https://github.com/abhinavroy1999/"
                f"nanowire-fragmentation-code/tree/{RELEASED_SOURCE_COMMIT}"
            ),
            "paper": "Roy et al., J. Appl. Phys. 130, 194301 (2021)",
            "paper_url": "https://arxiv.org/abs/2107.01801",
            "cpu_fft_is_not_bitwise_cufft": True,
            "released_noise_index_behavior_preserved": True,
            "corrected_whole_field_noise_used": False,
            "released_complex_to_real_discards_imaginary_component": True,
            "q0p9_gate_used": False,
        },
        "definition": {
            **asdict(definition),
            "step_limit": step_limit,
            "mode_is_physically_interpretable": False,
        },
        "geometry": gap,
        "noise_layout": asdict(noise_layout),
        "resource_preflight": resources,
        "binary_field_before_noise": binary_diagnostics,
        "initial_field_fingerprint": initial_fingerprint,
        "endpoint_field_fingerprint": endpoint_fingerprint,
        "records": records,
        "checks": checks,
        "execution": {
            "completed_steps": completed_steps,
            "pre_step_setup_wall_seconds": pre_step_setup_wall_seconds,
            "noise_generation_wall_seconds": noise_wall_seconds,
            "step_wall_seconds": step_wall_seconds,
            "median_step_wall_seconds": (
                float(statistics.median(step_wall_seconds))
                if step_wall_seconds
                else None
            ),
            "deterministic_repeat_wall_seconds": (
                deterministic_repeat_wall_seconds
            ),
            "checkpoint_write_wall_seconds": checkpoint_write_wall_seconds,
            "total_wall_seconds": float(time.perf_counter() - run_started),
            "maximum_relative_mass_drift": (
                maximum_relative_mass_drift
            ),
            "maximum_inverse_imaginary_linf": (
                maximum_inverse_imaginary_linf
            ),
            "maximum_discarded_imaginary_over_maximum_absolute_field": (
                maximum_inverse_imaginary_linf / maximum_absolute_field
                if maximum_absolute_field > 0.0
                else None
            ),
            "maximum_absolute_field": maximum_absolute_field,
            "peak_rss_bytes": peak_rss_bytes(),
            "fft_workers": fft_workers,
        },
        "runtime_projection": runtime_projection,
        "outputs": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "checkpoint_sha256": checkpoint_sha256,
            "figure": str(figure_path),
            "figure_sha256": sha256_path(figure_path),
        },
        "reproducibility": {
            "runner_sha256": sha256_path(SOURCE_PATH),
            "model_sha256": sha256_path(MODEL_PATH),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "interpretation_boundary": {
            "physical_reproduction_assessed": False,
            "junction_at_t1000_assessed": False,
            "breakup_timing_assessed": False,
            "long_run_started": False,
            "discarded_inverse_imaginary_is_diagnostic_not_gate": True,
            "this_run_can_only_validate_the_cpu_port_preflight": True,
        },
    }
    summary_path = output_directory / "summary.json"
    summary["outputs"]["summary"] = str(summary_path)
    atomic_json(summary_path, summary)
    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("reduced", "exact-timing"),
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="new output directory; existing directories are never overwritten",
    )
    parser.add_argument(
        "--fft-workers",
        type=int,
        default=12,
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    output = arguments.output
    if output is None:
        output = DEFAULT_RESULTS_DIRECTORY / arguments.mode.replace("-", "_")
    summary = run_preflight(
        arguments.mode,
        output.resolve(),
        fft_workers=arguments.fft_workers,
    )
    print(
        json.dumps(
            {
                "classification": summary["classification"],
                "preflight_passed": summary["preflight_passed"],
                "output": summary["outputs"]["summary"],
                "runtime_projection": summary["runtime_projection"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
