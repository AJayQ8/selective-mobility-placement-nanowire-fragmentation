#!/usr/bin/env python3
"""Three-step full-grid localization preflight for the fixed-GB bridge.

The exact frozen Roy field is constructed once and copied into three paths:
the frozen solver, the bridge factory with G=0, and the bridge with G=0.35.
Each path advances exactly three steps.  The run is non-resumable and cannot
launch a continuation, production trajectory, parameter sweep, or report
workflow.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy import ndimage

from ..roy_2021_reproduction.model import (
    FROZEN_DEG90,
    PeriodicLattice,
    RoyPseudospectralSolver,
    apply_released_overlapping_noise,
    initialize_strict_deg90,
    released_noise_layout,
)
from .geometry import centered_coordinates
from .metrics import array_fingerprint
from .model import (
    FixedGrainBoundaryParameters,
    RoyFixedGrainBoundarySolver,
    build_periodic_bicrystal_profile,
    build_solver,
)
from .provenance import (
    MANIFEST_PATH,
    sha256_path,
    verify_frozen_baseline,
)


SOURCE_PATH = Path(__file__).resolve()
DIRECTORY = SOURCE_PATH.parent
DEFAULT_OUTPUT_DIRECTORY = (
    DIRECTORY / "results" / "full_grid_localization_preflight"
)
HORIZON_RESULT_PATH = (
    DIRECTORY / "results" / "planar_groove_horizon_preflight" / "summary.json"
)
EXPECTED_SOURCE_SHA256 = {
    "baseline_manifest.json": "369d20f99fe15f70edb497ee868b1683cbddd5532ae558821823b57040683962",
    "model.py": "0d22e4fef00e427821d586dce87a85ff93924e5fa2c9ef6a18913843d996a64f",
    "geometry.py": "ae2ac05939076ddc9964b75df9244dc0287d07d20f26eee151d388af281c5f0e",
    "metrics.py": "fc699eada0f17f34c87c4a654f08ccaab607172445deacc3785d6e04c6d65963",
    "provenance.py": "bcce214240d0c070a186d789e11545eeae7a4c9fbc7aaeb1488feef6a9315aa3",
    "run_planar_groove_preflight.py": "fdcb07ceba4fd29e409f8d90eeca328aeec053685311ffbd7baae9174f0c8944",
    "run_planar_groove_horizon_preflight.py": "a0a864cf1dac42fe8862ebd4e3d1682f6c73803923731698b3adb59c30888011",
    "planar_groove_horizon_summary.json": "47b32349cf5ef2a4624c6cd3bee18c86a4c4beabefcfc7c9510f2b0e5f8e6976",
}
EXPECTED_INITIAL_FINGERPRINT = (
    "d0574f7e73c5b2c81c5cd2cb564af4f36481d54dad85ec02ead56738628026e2"
)
FFT_WORKERS = 12
STEPS = 3
ENERGY_RATIO = 0.35
GB_OFFSET_OVER_RADIUS = 2.6
WALL_SECONDS_CAP = 180.0
ESTIMATED_BYTES_PER_CELL = 384
MINIMUM_DISK_BYTES = 1 << 30
FIELD_MAGNITUDE_LIMIT = 2.0
MASS_DRIFT_LIMIT = 1.0e-6
LEGACY_PRODUCTION_MASS_DRIFT_LIMIT = 1.0e-4
ENERGY_RISE_LIMIT = 1.0e-6
Q0_OUTSIDE_PREFIX_MAXIMUM = 1.0e-14
Q0_NOISE_ALLOWANCE_FRACTION_MINIMUM = 0.999
Q1_CORE_RMS_MINIMUM = 1.0e-12
Q1_ALLOWED_FRACTION_MINIMUM = 0.95
Q1_LEAKAGE_RMS_RATIO_MAXIMUM = 0.01
INCREMENT_CORE_DOMINANCE_FACTOR = 3.0
PRIMARY_IMAGE_RMS_RATIO_MINIMUM = 0.25
PRIMARY_IMAGE_RMS_RATIO_MAXIMUM = 4.0
ALLOWED_INCREMENT_FRACTION_MINIMUM = 0.50
LEAKAGE_RMS_RATIO_MAXIMUM = 0.25
FAR_ARM_RADIUS_CHANGE_OVER_R_LIMIT = 1.0e-4
REGION_CHUNK = 32


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        measured = float(value)
        return measured if np.isfinite(measured) else None
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                _json_safe(payload),
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


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def peak_rss_bytes() -> int:
    measured = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return measured * 1024 if sys.platform.startswith("linux") else measured


def verify_full_grid_sources() -> dict[str, Any]:
    provenance = verify_frozen_baseline()
    paths = {
        "baseline_manifest.json": MANIFEST_PATH,
        "model.py": DIRECTORY / "model.py",
        "geometry.py": DIRECTORY / "geometry.py",
        "metrics.py": DIRECTORY / "metrics.py",
        "provenance.py": DIRECTORY / "provenance.py",
        "run_planar_groove_preflight.py": (
            DIRECTORY / "run_planar_groove_preflight.py"
        ),
        "run_planar_groove_horizon_preflight.py": (
            DIRECTORY / "run_planar_groove_horizon_preflight.py"
        ),
        "planar_groove_horizon_summary.json": HORIZON_RESULT_PATH,
    }
    measured = {name: sha256_path(path) for name, path in paths.items()}
    mismatches = {
        name: {
            "expected": EXPECTED_SOURCE_SHA256[name],
            "actual": measured[name],
        }
        for name in paths
        if measured[name] != EXPECTED_SOURCE_SHA256[name]
    }
    with HORIZON_RESULT_PATH.open("r", encoding="utf-8") as handle:
        horizon = json.load(handle)
    if horizon.get("preflight_passed") is not False:
        mismatches["horizon_formal_status"] = {
            "expected": "false",
            "actual": str(horizon.get("preflight_passed")),
        }
    if mismatches:
        raise RuntimeError(
            "full-grid source contract mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        **provenance,
        "bridge_source_sha256": measured,
        "horizon_formal_failure_preserved": True,
        "verified": True,
    }


def resource_preflight(
    lattice: PeriodicLattice,
    output_directory: Path,
) -> dict[str, Any]:
    estimated_peak_bytes = (
        lattice.cell_count * ESTIMATED_BYTES_PER_CELL
    )
    physical_memory_bytes = int(
        os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    )
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
        "disk_headroom_sufficient": bool(
            free_disk_bytes >= MINIMUM_DISK_BYTES
        ),
    }
    return {
        "cell_count": lattice.cell_count,
        "estimated_bytes_per_cell": ESTIMATED_BYTES_PER_CELL,
        "estimated_peak_bytes": estimated_peak_bytes,
        "physical_memory_bytes": physical_memory_bytes,
        "available_memory_bytes": available_memory_bytes,
        "free_disk_bytes": free_disk_bytes,
        "required_disk_bytes": MINIMUM_DISK_BYTES,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _periodic_distance(
    coordinate: np.ndarray,
    plane: float,
    length: float,
) -> np.ndarray:
    return np.abs(
        (coordinate - plane + 0.5 * length) % length - 0.5 * length
    )


def _empty_accumulator() -> dict[str, float | int]:
    return {"count": 0, "sum_squared": 0.0, "maximum_absolute": 0.0}


def localization_stats(
    field: np.ndarray,
    *,
    primary_coordinate: float,
    image_coordinate: float,
    radius: float,
    width: float,
    noise_prefix_length: int,
) -> dict[str, Any]:
    """Accumulate exact ROI reductions without storing full Boolean masks."""

    lattice = FROZEN_DEG90.lattice
    if field.shape != lattice.shape:
        raise ValueError("localization field has the wrong shape")
    nx, ny, nz = lattice.shape
    x = centered_coordinates(lattice, 0)
    y = centered_coordinates(lattice, 1)
    z = centered_coordinates(lattice, 2)
    r1 = np.sqrt(x[:, None] ** 2 + y[None, :] ** 2)
    core_ring = np.abs(r1 - radius) <= width
    halo_ring = np.abs(r1 - radius) <= 2.0 * width
    dense_disk = r1 <= radius - width
    base_flat_index = (
        (
            np.arange(nx, dtype=np.int64)[:, None]
            * ny
            + np.arange(ny, dtype=np.int64)[None, :]
        )
        * nz
    )
    y_distal = np.abs(y) >= 2.0 * radius
    region_names = (
        "all",
        "primary_core",
        "image_core",
        "combined_core",
        "intended_halo",
        "noise_prefix",
        "noise_allowance",
        "allowed",
        "junction",
        "second_wire",
        "far_background",
        "far_arm",
        "dense_primary",
        "dense_image",
        "outside_noise_prefix",
    )
    accumulators = {
        name: _empty_accumulator() for name in region_names
    }
    length_z = lattice.physical_lengths[2]

    for start in range(0, nz, REGION_CHUNK):
        stop = min(nz, start + REGION_CHUNK)
        values = field[:, :, start:stop]
        z_block = z[start:stop]
        dp = _periodic_distance(
            z_block, primary_coordinate, length_z
        )
        di = _periodic_distance(
            z_block, image_coordinate, length_z
        )
        dg = np.minimum(dp, di)
        primary_core = (
            core_ring[:, :, None] & (dp[None, None, :] <= width)
        )
        image_core = (
            core_ring[:, :, None] & (di[None, None, :] <= width)
        )
        combined_core = primary_core | image_core
        intended_halo = (
            halo_ring[:, :, None]
            & (dg[None, None, :] <= 2.0 * width)
        )
        flat_index = (
            base_flat_index[:, :, None]
            + np.arange(start, stop, dtype=np.int64)[None, None, :]
        )
        noise_prefix = flat_index < noise_prefix_length
        noise_allowance = (
            noise_prefix & (dg[None, None, :] <= 2.0 * width)
        )
        allowed = intended_halo | noise_allowance
        junction = (
            (x[:, None, None] - radius) ** 2
            + y[None, :, None] ** 2
            + z_block[None, None, :] ** 2
            <= radius**2
        )
        r2 = np.sqrt(
            (x[:, None] - 2.0 * radius) ** 2
            + z_block[None, :] ** 2
        )
        second_wire = (
            (r2[:, None, :] <= radius + 2.0 * width)
            & y_distal[None, :, None]
            & ~junction
        )
        far_background = ~(allowed | junction | second_wire)
        far_arm = (
            core_ring[:, :, None]
            & (
                _periodic_distance(
                    z_block,
                    -primary_coordinate,
                    length_z,
                )[None, None, :]
                <= width
            )
        )
        dense_primary = (
            dense_disk[:, :, None] & (dp[None, None, :] <= width)
        )
        dense_image = (
            dense_disk[:, :, None] & (di[None, None, :] <= width)
        )
        masks = {
            "all": np.ones(values.shape, dtype=bool),
            "primary_core": primary_core,
            "image_core": image_core,
            "combined_core": combined_core,
            "intended_halo": intended_halo,
            "noise_prefix": noise_prefix,
            "noise_allowance": noise_allowance,
            "allowed": allowed,
            "junction": junction,
            "second_wire": second_wire,
            "far_background": far_background,
            "far_arm": far_arm,
            "dense_primary": dense_primary,
            "dense_image": dense_image,
            "outside_noise_prefix": ~noise_prefix,
        }
        for name, mask in masks.items():
            selected = values[np.broadcast_to(mask, values.shape)]
            if not selected.size:
                continue
            accumulator = accumulators[name]
            accumulator["count"] = int(accumulator["count"]) + int(
                selected.size
            )
            accumulator["sum_squared"] = float(
                accumulator["sum_squared"]
            ) + float(np.sum(selected * selected, dtype=np.float64))
            accumulator["maximum_absolute"] = max(
                float(accumulator["maximum_absolute"]),
                float(np.max(np.abs(selected))),
            )

    reports: dict[str, Any] = {}
    total_squared = float(accumulators["all"]["sum_squared"])
    for name, accumulator in accumulators.items():
        count = int(accumulator["count"])
        sum_squared = float(accumulator["sum_squared"])
        reports[name] = {
            "cell_count": count,
            "sum_squared": sum_squared,
            "rms": float(np.sqrt(sum_squared / count)) if count else 0.0,
            "maximum_absolute": float(
                accumulator["maximum_absolute"]
            ),
            "fraction_of_total_squared_norm": (
                float(sum_squared / total_squared)
                if total_squared > 0.0
                else 0.0
            ),
        }
    return reports


def source_localization_checks(
    q0_stats: dict[str, Any],
    q1_stats: dict[str, Any],
) -> dict[str, bool]:
    """Hard checks for the noise-only origin and first physical GB source."""

    q0_total = q0_stats["all"]["sum_squared"]
    q1_minimum_core_rms = min(
        q1_stats["primary_core"]["rms"],
        q1_stats["image_core"]["rms"],
    )
    return {
        "q0_outside_noise_prefix_is_zero": bool(
            q0_stats["outside_noise_prefix"]["maximum_absolute"]
            <= Q0_OUTSIDE_PREFIX_MAXIMUM
        ),
        "q0_noise_allowance_contains_99p9_percent": bool(
            q0_total == 0.0
            or q0_stats["noise_allowance"][
                "fraction_of_total_squared_norm"
            ]
            >= Q0_NOISE_ALLOWANCE_FRACTION_MINIMUM
        ),
        "q0_junction_is_zero": bool(
            q0_stats["junction"]["maximum_absolute"]
            <= Q0_OUTSIDE_PREFIX_MAXIMUM
        ),
        "q0_second_wire_is_zero": bool(
            q0_stats["second_wire"]["maximum_absolute"]
            <= Q0_OUTSIDE_PREFIX_MAXIMUM
        ),
        "q0_primary_core_is_zero": bool(
            q0_stats["primary_core"]["maximum_absolute"]
            <= Q0_OUTSIDE_PREFIX_MAXIMUM
        ),
        "q0_image_core_is_zero": bool(
            q0_stats["image_core"]["maximum_absolute"]
            <= Q0_OUTSIDE_PREFIX_MAXIMUM
        ),
        "q1_primary_and_image_cores_are_active": bool(
            q1_minimum_core_rms >= Q1_CORE_RMS_MINIMUM
        ),
        "q1_allowed_regions_contain_95_percent": bool(
            q1_stats["allowed"]["fraction_of_total_squared_norm"]
            >= Q1_ALLOWED_FRACTION_MINIMUM
        ),
        "q1_junction_rms_below_one_percent_core": bool(
            q1_stats["junction"]["rms"]
            <= Q1_LEAKAGE_RMS_RATIO_MAXIMUM * q1_minimum_core_rms
        ),
        "q1_second_wire_rms_below_one_percent_core": bool(
            q1_stats["second_wire"]["rms"]
            <= Q1_LEAKAGE_RMS_RATIO_MAXIMUM * q1_minimum_core_rms
        ),
    }


def contact_connectivity(field: np.ndarray) -> dict[str, Any]:
    """Roy-local topology without assuming the early junction is connected."""

    definition = FROZEN_DEG90
    nx, ny, nz = definition.lattice.shape
    first_center = nx // 2
    second_center = first_center + definition.radius_1 + definition.radius_2
    center_y = ny // 2
    center_z = nz // 2
    half_extent = 2 * max(definition.radius_1, definition.radius_2)
    x0 = first_center - definition.radius_1
    x1 = second_center + definition.radius_2 + 1
    y0 = center_y - half_extent
    y1 = center_y + half_extent + 1
    z0 = center_z - half_extent
    z1 = center_z + half_extent + 1
    local = field[x0:x1, y0:y1, z0:z1]
    structure = ndimage.generate_binary_structure(3, 1)
    reports: dict[str, Any] = {}
    for threshold in (0.45, 0.50, 0.55):
        labels, component_count = ndimage.label(
            local >= threshold,
            structure=structure,
        )
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
        reports[f"{threshold:.2f}"] = {
            "component_count": int(component_count),
            "first_core_label": first_label,
            "second_core_label": second_label,
            "cores_connected": bool(
                first_label > 0 and first_label == second_label
            ),
        }
    return reports


def _topology_status_equal(
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool:
    return all(
        first[key]["cores_connected"] == second[key]["cores_connected"]
        for key in first
    )


def _field_record(
    field: np.ndarray,
    *,
    energy: float,
    initial_mass: float,
    lattice: PeriodicLattice,
    inverse_imaginary_linf: float,
    step: int,
) -> dict[str, Any]:
    mass = float(
        np.sum(field, dtype=np.float64) * lattice.cell_volume
    )
    return {
        "step": step,
        "time": float(step * FROZEN_DEG90.parameters.timestep),
        "finite": bool(np.isfinite(field).all()),
        "minimum": float(np.min(field)),
        "maximum": float(np.max(field)),
        "maximum_absolute": float(np.max(np.abs(field))),
        "mass": mass,
        "relative_mass_drift": abs(mass - initial_mass) / abs(initial_mass),
        "energy": energy,
        "field_fingerprint": array_fingerprint(field),
        "inverse_imaginary_linf": inverse_imaginary_linf,
        "topology": contact_connectivity(field),
    }


def _maximum_energy_rise(records: list[dict[str, Any]]) -> float:
    scale = max(abs(float(records[0]["energy"])), np.finfo(float).tiny)
    return float(
        max(
            [0.0]
            + [
                max(
                    0.0,
                    float(current["energy"]) - float(previous["energy"]),
                )
                / scale
                for previous, current in zip(records, records[1:])
            ]
        )
    )


def _far_arm_equivalent_radius(
    field: np.ndarray,
    *,
    center: float,
    width: float,
    radius: float,
) -> float:
    lattice = FROZEN_DEG90.lattice
    z = centered_coordinates(lattice, 2)
    selected = _periodic_distance(
        z,
        center,
        lattice.physical_lengths[2],
    ) <= width
    radii: list[float] = []
    for index in np.flatnonzero(selected):
        area = (
            int(np.count_nonzero(field[:, :, index] >= 0.5))
            * lattice.spacing**2
        )
        radii.append(float(np.sqrt(area / np.pi)))
    return float(np.mean(radii))


def _check_wall(started: float) -> None:
    if time.perf_counter() - started > WALL_SECONDS_CAP:
        raise TimeoutError("full-grid preflight exceeded its hard wall cap")


def run(output_directory: Path) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output: {output_directory}"
        )
    output_directory.mkdir(parents=True)
    started = time.perf_counter()
    provenance = verify_full_grid_sources()
    resources = resource_preflight(
        FROZEN_DEG90.lattice,
        output_directory,
    )
    contract = {
        "schema_version": 1,
        "run": "Roy fixed-GB three-step full-grid localization preflight",
        "shape": FROZEN_DEG90.lattice.shape,
        "spacing": FROZEN_DEG90.lattice.spacing,
        "fft_workers": FFT_WORKERS,
        "steps_per_path": STEPS,
        "total_propose_step_calls": 3 * STEPS,
        "energy_ratio": ENERGY_RATIO,
        "gb_axis": 2,
        "gb_offset_over_radius": GB_OFFSET_OVER_RADIUS,
        "wall_seconds_cap": WALL_SECONDS_CAP,
        "acceptance_thresholds": {
            "q0_outside_prefix_maximum": Q0_OUTSIDE_PREFIX_MAXIMUM,
            "q0_noise_allowance_fraction_minimum": (
                Q0_NOISE_ALLOWANCE_FRACTION_MINIMUM
            ),
            "q1_core_rms_minimum": Q1_CORE_RMS_MINIMUM,
            "q1_allowed_fraction_minimum": (
                Q1_ALLOWED_FRACTION_MINIMUM
            ),
            "q1_leakage_rms_ratio_maximum": (
                Q1_LEAKAGE_RMS_RATIO_MAXIMUM
            ),
            "increment_core_dominance_factor": (
                INCREMENT_CORE_DOMINANCE_FACTOR
            ),
            "allowed_increment_fraction_minimum": (
                ALLOWED_INCREMENT_FRACTION_MINIMUM
            ),
            "cumulative_leakage_rms_ratio_maximum": (
                LEAKAGE_RMS_RATIO_MAXIMUM
            ),
            "far_arm_radius_change_over_r_limit": (
                FAR_ARM_RADIUS_CHANGE_OVER_R_LIMIT
            ),
        },
        "scope": {
            "production_authorized": False,
            "continuation_authorized": False,
            "resume_authorized": False,
            "parameter_sweep_authorized": False,
            "automatic_follow_on": False,
        },
        "provenance": provenance,
        "resource_preflight": resources,
    }
    _atomic_json(output_directory / "contract.json", contract)
    _atomic_json(
        output_directory / "run_status.json",
        {
            "status": "running",
            "started_unix": time.time(),
            "completed_steps": {
                "roy": 0,
                "factory_g0": 0,
                "fixed_gb": 0,
            },
        },
    )
    if not resources["passed"]:
        raise RuntimeError("resource preflight failed before allocation")

    definition = FROZEN_DEG90
    lattice = definition.lattice
    parameters = definition.parameters
    physical_radius = definition.radius_1 * lattice.spacing
    mapped = FixedGrainBoundaryParameters.matched_to_roy(
        parameters,
        energy_ratio=ENERGY_RATIO,
    )
    primary_coordinate = GB_OFFSET_OVER_RADIUS * physical_radius
    profile = build_periodic_bicrystal_profile(
        lattice,
        mapped,
        axis=2,
        primary_coordinate=primary_coordinate,
    )
    noise_layout = released_noise_layout(lattice)

    setup_started = time.perf_counter()
    initial, masks = initialize_strict_deg90(definition)
    apply_released_overlapping_noise(
        initial,
        definition.noise_amplitude,
        definition.noise_seed,
    )
    setup_seconds = float(time.perf_counter() - setup_started)
    initial_fingerprint = array_fingerprint(initial)
    if initial_fingerprint != EXPECTED_INITIAL_FINGERPRINT:
        raise RuntimeError(
            "full-grid initializer fingerprint does not match frozen Roy field"
        )
    del masks
    gc.collect()
    _check_wall(started)

    initial_mass = float(
        np.sum(initial, dtype=np.float64) * lattice.cell_volume
    )
    initial_topology = contact_connectivity(initial)

    zero_parameters = FixedGrainBoundaryParameters.matched_to_roy(
        parameters,
        energy_ratio=0.0,
    )
    zero_profile = build_periodic_bicrystal_profile(
        lattice,
        zero_parameters,
        axis=2,
        primary_coordinate=primary_coordinate,
    )

    # Frozen Roy reference trajectory.
    reference_solver = RoyPseudospectralSolver(
        lattice,
        parameters,
        fft_workers=FFT_WORKERS,
    )
    reference = initial.copy()
    reference_records = [
        _field_record(
            reference,
            energy=reference_solver.free_energy(reference),
            initial_mass=initial_mass,
            lattice=lattice,
            inverse_imaginary_linf=0.0,
            step=0,
        )
    ]
    reference_snapshots: list[np.ndarray] = []
    reference_step_seconds: list[float] = []
    for step in range(1, STEPS + 1):
        step_started = time.perf_counter()
        reference = reference_solver.propose_step(reference)
        reference_step_seconds.append(
            float(time.perf_counter() - step_started)
        )
        reference_snapshots.append(reference.copy())
        reference_records.append(
            _field_record(
                reference,
                energy=reference_solver.free_energy(reference),
                initial_mass=initial_mass,
                lattice=lattice,
                inverse_imaginary_linf=(
                    reference_solver.last_inverse_imaginary_linf
                ),
                step=step,
            )
        )
        _check_wall(started)
    del reference, reference_solver
    gc.collect()

    # Explicit G=0 factory trajectory: it must be the exact frozen class.
    factory_zero_solver = build_solver(
        lattice,
        parameters,
        grain_boundary=zero_profile,
        fft_workers=FFT_WORKERS,
    )
    factory_zero_is_frozen_class = bool(
        type(factory_zero_solver) is RoyPseudospectralSolver
    )
    factory_zero = initial.copy()
    zero_records = [
        _field_record(
            factory_zero,
            energy=factory_zero_solver.free_energy(factory_zero),
            initial_mass=initial_mass,
            lattice=lattice,
            inverse_imaginary_linf=0.0,
            step=0,
        )
    ]
    zero_array_equal: list[bool] = []
    zero_step_seconds: list[float] = []
    for step in range(1, STEPS + 1):
        step_started = time.perf_counter()
        factory_zero = factory_zero_solver.propose_step(factory_zero)
        zero_step_seconds.append(float(time.perf_counter() - step_started))
        zero_array_equal.append(
            bool(np.array_equal(factory_zero, reference_snapshots[step - 1]))
        )
        zero_records.append(
            _field_record(
                factory_zero,
                energy=factory_zero_solver.free_energy(factory_zero),
                initial_mass=initial_mass,
                lattice=lattice,
                inverse_imaginary_linf=(
                    factory_zero_solver.last_inverse_imaginary_linf
                ),
                step=step,
            )
        )
        _check_wall(started)
    del factory_zero, factory_zero_solver
    gc.collect()

    # Positive-G path plus origin/localization reductions.
    bridge_solver = RoyFixedGrainBoundarySolver(
        lattice,
        parameters,
        profile,
        fft_workers=FFT_WORKERS,
    )
    q0 = bridge_solver.grain_coupling_potential(initial)
    q0_stats = localization_stats(
        q0,
        primary_coordinate=profile.primary_coordinate,
        image_coordinate=profile.image_coordinate,
        radius=physical_radius,
        width=mapped.surface_width,
        noise_prefix_length=noise_layout.unique_target_prefix_length,
    )
    center_y = lattice.shape[1] // 2
    slices: dict[str, np.ndarray] = {
        "q0_y_center": np.asarray(
            q0[:, center_y, :],
            dtype=np.float32,
        )
    }
    del q0
    gc.collect()

    q1_reference = bridge_solver.grain_coupling_potential(
        reference_snapshots[0]
    )
    q1_stats = localization_stats(
        q1_reference,
        primary_coordinate=profile.primary_coordinate,
        image_coordinate=profile.image_coordinate,
        radius=physical_radius,
        width=mapped.surface_width,
        noise_prefix_length=noise_layout.unique_target_prefix_length,
    )
    slices["q1_reference_y_center"] = np.asarray(
        q1_reference[:, center_y, :],
        dtype=np.float32,
    )
    del q1_reference
    gc.collect()

    bridge = initial.copy()
    bridge_initial_mass = initial_mass
    bridge_records = [
        _field_record(
            bridge,
            energy=bridge_solver.free_energy(bridge),
            initial_mass=bridge_initial_mass,
            lattice=lattice,
            inverse_imaginary_linf=0.0,
            step=0,
        )
    ]
    delta_reports: list[dict[str, Any]] = []
    bridge_step_seconds: list[float] = []
    previous_delta = np.zeros_like(initial)
    delta_one_stats: dict[str, Any] | None = None
    delta_three: np.ndarray | None = None
    increment_three: np.ndarray | None = None
    for step in range(1, STEPS + 1):
        step_started = time.perf_counter()
        bridge = bridge_solver.propose_step(bridge)
        bridge_step_seconds.append(float(time.perf_counter() - step_started))
        bridge_records.append(
            _field_record(
                bridge,
                energy=bridge_solver.free_energy(bridge),
                initial_mass=bridge_initial_mass,
                lattice=lattice,
                inverse_imaginary_linf=(
                    bridge_solver.last_inverse_imaginary_linf
                ),
                step=step,
            )
        )
        delta = bridge - reference_snapshots[step - 1]
        increment = delta - previous_delta
        delta_stats = localization_stats(
            delta,
            primary_coordinate=profile.primary_coordinate,
            image_coordinate=profile.image_coordinate,
            radius=physical_radius,
            width=mapped.surface_width,
            noise_prefix_length=noise_layout.unique_target_prefix_length,
        )
        increment_stats = localization_stats(
            increment,
            primary_coordinate=profile.primary_coordinate,
            image_coordinate=profile.image_coordinate,
            radius=physical_radius,
            width=mapped.surface_width,
            noise_prefix_length=noise_layout.unique_target_prefix_length,
        )
        delta_reports.append(
            {
                "step": step,
                "cumulative_delta": delta_stats,
                "incremental_delta": increment_stats,
            }
        )
        if step == 1:
            delta_one_stats = delta_stats
        if step == STEPS:
            delta_three = delta.copy()
            increment_three = increment.copy()
        previous_delta = delta
        _check_wall(started)

    assert delta_one_stats is not None
    assert delta_three is not None
    assert increment_three is not None
    slices["delta3_y_center"] = np.asarray(
        delta_three[:, center_y, :],
        dtype=np.float32,
    )
    slices["increment3_y_center"] = np.asarray(
        increment_three[:, center_y, :],
        dtype=np.float32,
    )
    slices["reference3_y_center"] = np.asarray(
        reference_snapshots[-1][:, center_y, :],
        dtype=np.float32,
    )
    slices["bridge3_y_center"] = np.asarray(
        bridge[:, center_y, :],
        dtype=np.float32,
    )
    increment_three_stats = delta_reports[-1]["incremental_delta"]
    delta_three_stats = delta_reports[-1]["cumulative_delta"]

    reference_far_radius = _far_arm_equivalent_radius(
        reference_snapshots[-1],
        center=-primary_coordinate,
        width=mapped.surface_width,
        radius=physical_radius,
    )
    bridge_far_radius = _far_arm_equivalent_radius(
        bridge,
        center=-primary_coordinate,
        width=mapped.surface_width,
        radius=physical_radius,
    )
    far_radius_change_over_r = (
        abs(bridge_far_radius - reference_far_radius) / physical_radius
    )

    health_checks = {
        "resource_preflight_passed": resources["passed"],
        "initializer_fingerprint_matches": (
            initial_fingerprint == EXPECTED_INITIAL_FINGERPRINT
        ),
        "exactly_three_steps_each": bool(
            len(reference_records) == STEPS + 1
            and len(zero_records) == STEPS + 1
            and len(bridge_records) == STEPS + 1
        ),
        "all_fields_finite": bool(
            all(
                record["finite"]
                for records in (
                    reference_records,
                    zero_records,
                    bridge_records,
                )
                for record in records
            )
        ),
        "all_fields_below_magnitude_limit": bool(
            all(
                record["maximum_absolute"] <= FIELD_MAGNITUDE_LIMIT
                for records in (
                    reference_records,
                    zero_records,
                    bridge_records,
                )
                for record in records
            )
        ),
        "reference_mass_drift_below_limit": bool(
            max(r["relative_mass_drift"] for r in reference_records)
            <= MASS_DRIFT_LIMIT
        ),
        "factory_g0_mass_drift_below_limit": bool(
            max(r["relative_mass_drift"] for r in zero_records)
            <= MASS_DRIFT_LIMIT
        ),
        "bridge_mass_drift_below_limit": bool(
            max(r["relative_mass_drift"] for r in bridge_records)
            <= MASS_DRIFT_LIMIT
        ),
        "reference_energy_healthy": bool(
            reference_records[-1]["energy"] < reference_records[0]["energy"]
            and _maximum_energy_rise(reference_records)
            <= ENERGY_RISE_LIMIT
        ),
        "factory_g0_energy_healthy": bool(
            zero_records[-1]["energy"] < zero_records[0]["energy"]
            and _maximum_energy_rise(zero_records) <= ENERGY_RISE_LIMIT
        ),
        "bridge_energy_healthy": bool(
            bridge_records[-1]["energy"] < bridge_records[0]["energy"]
            and _maximum_energy_rise(bridge_records) <= ENERGY_RISE_LIMIT
        ),
        "wall_cap_not_reached": bool(
            time.perf_counter() - started <= WALL_SECONDS_CAP
        ),
    }
    exact_zero_checks = {
        "factory_returns_frozen_class": factory_zero_is_frozen_class,
        "every_step_array_equal": bool(all(zero_array_equal)),
        "every_step_fingerprint_equal": bool(
            all(
                zero_records[step]["field_fingerprint"]
                == reference_records[step]["field_fingerprint"]
                for step in range(STEPS + 1)
            )
        ),
        "every_step_energy_float_equal": bool(
            all(
                zero_records[step]["energy"]
                == reference_records[step]["energy"]
                for step in range(STEPS + 1)
            )
        ),
    }
    source_checks = source_localization_checks(q0_stats, q1_stats)
    i3_primary = increment_three_stats["primary_core"]["rms"]
    i3_image = increment_three_stats["image_core"]["rms"]
    i3_noise = increment_three_stats["noise_allowance"]["rms"]
    i3_background = max(
        increment_three_stats[name]["rms"]
        for name in ("far_background", "junction", "second_wire")
    )
    d1_primary = delta_one_stats["primary_core"]["rms"]
    d1_image = delta_one_stats["image_core"]["rms"]
    minimum_i3_core = min(i3_primary, i3_image)
    core_ratio = (
        i3_primary / i3_image
        if i3_image > 0.0
        else float("inf")
    )
    activation_checks = {
        "primary_increment_dominates_background_and_step1": bool(
            i3_primary
            >= INCREMENT_CORE_DOMINANCE_FACTOR
            * max(i3_background, d1_primary)
        ),
        "image_increment_dominates_background_and_step1": bool(
            i3_image
            >= INCREMENT_CORE_DOMINANCE_FACTOR
            * max(i3_background, d1_image)
        ),
        "primary_image_rms_ratio_is_bounded": bool(
            PRIMARY_IMAGE_RMS_RATIO_MINIMUM
            <= core_ratio
            <= PRIMARY_IMAGE_RMS_RATIO_MAXIMUM
        ),
        "combined_core_increment_dominates_noise_allowance": bool(
            increment_three_stats["combined_core"]["rms"] >= i3_noise
        ),
        "allowed_regions_contain_half_increment_norm": bool(
            increment_three_stats["allowed"][
                "fraction_of_total_squared_norm"
            ]
            >= ALLOWED_INCREMENT_FRACTION_MINIMUM
        ),
    }
    leakage_checks = {
        "cumulative_junction_rms_below_quarter_core": bool(
            delta_three_stats["junction"]["rms"]
            <= LEAKAGE_RMS_RATIO_MAXIMUM * minimum_i3_core
        ),
        "cumulative_second_wire_rms_below_quarter_core": bool(
            delta_three_stats["second_wire"]["rms"]
            <= LEAKAGE_RMS_RATIO_MAXIMUM * minimum_i3_core
        ),
        "far_arm_radius_change_below_limit": bool(
            far_radius_change_over_r
            <= FAR_ARM_RADIUS_CHANGE_OVER_R_LIMIT
        ),
    }
    topology_checks = {
        "initial_statuses_identical": _topology_status_equal(
            initial_topology,
            bridge_records[0]["topology"],
        ),
        "statuses_match_reference_at_every_step": bool(
            all(
                _topology_status_equal(
                    reference_records[step]["topology"],
                    bridge_records[step]["topology"],
                )
                for step in range(STEPS + 1)
            )
        ),
    }

    hard_failure = not all(
        list(health_checks.values())
        + list(exact_zero_checks.values())
        + list(source_checks.values())
        + list(leakage_checks.values())
        + list(topology_checks.values())
    )
    activation_passed = bool(all(activation_checks.values()))
    if hard_failure:
        classification = "roy_fixed_gb_full_grid_localization_failed"
        passed = False
    elif not activation_passed:
        classification = (
            "roy_fixed_gb_full_grid_localization_inconclusive_three_steps"
        )
        passed = False
    else:
        classification = "roy_fixed_gb_full_grid_localization_passed"
        passed = True

    elapsed_seconds = float(time.perf_counter() - started)
    summary = {
        "schema_version": 1,
        "classification": classification,
        "preflight_passed": passed,
        "scope_boundary": contract["scope"],
        "contract": contract,
        "geometry": {
            "physical_radius": physical_radius,
            "surface_width": mapped.surface_width,
            "gb_axis": 2,
            "primary_coordinate": profile.primary_coordinate,
            "primary_fractional_index": (
                profile.primary_coordinate / lattice.spacing
                + lattice.shape[2] // 2
            ),
            "periodic_image_coordinate": profile.image_coordinate,
            "periodic_image_fractional_index": (
                profile.image_coordinate / lattice.spacing
                + lattice.shape[2] // 2
            ),
            "second_wire_axis_coordinate_x": 2.0 * physical_radius,
            "junction_coordinate": [physical_radius, 0.0, 0.0],
            "per_plane_energy_audit": profile.per_plane_energy_audit(),
        },
        "initializer": {
            "field_fingerprint": initial_fingerprint,
            "noise_generation_seconds": setup_seconds,
            "released_noise_layout": {
                "unique_target_prefix_length": (
                    noise_layout.unique_target_prefix_length
                ),
                "unique_target_fraction": (
                    noise_layout.unique_target_fraction
                ),
                "released_stride": noise_layout.released_stride,
                "standard_geometry_stride": (
                    noise_layout.standard_geometry_stride
                ),
                "stride_mismatch_present": (
                    noise_layout.stride_mismatch_present
                ),
            },
            "topology": initial_topology,
        },
        "execution": {
            "completed_steps": {
                "roy": STEPS,
                "factory_g0": STEPS,
                "fixed_gb": STEPS,
            },
            "total_propose_step_calls": 3 * STEPS,
            "elapsed_seconds": elapsed_seconds,
            "peak_rss_bytes": peak_rss_bytes(),
            "reference_step_seconds": reference_step_seconds,
            "factory_g0_step_seconds": zero_step_seconds,
            "bridge_step_seconds": bridge_step_seconds,
        },
        "trajectories": {
            "roy": reference_records,
            "factory_g0": zero_records,
            "fixed_gb": bridge_records,
        },
        "localization": {
            "q0_origin": q0_stats,
            "q1_reference_source": q1_stats,
            "paired_deltas": delta_reports,
            "increment3_primary_image_rms_ratio": core_ratio,
            "far_arm_reference_equivalent_radius": (
                reference_far_radius
            ),
            "far_arm_bridge_equivalent_radius": bridge_far_radius,
            "far_arm_radius_change_over_r": far_radius_change_over_r,
        },
        "checks": {
            "health": health_checks,
            "exact_zero": exact_zero_checks,
            "source_localization": source_checks,
            "activation": activation_checks,
            "leakage": leakage_checks,
            "topology": topology_checks,
        },
        "interpretation_boundary": {
            "step1_is_noise_prefix_conditioning": True,
            "intended_wire_response_judged_from_increment3": True,
            "periodic_image_is_intended_physics": True,
            "initial_disconnected_junction_is_expected": True,
            "inverse_imaginary_linf_is_descriptive_only": True,
            "legacy_production_mass_drift_limit": (
                LEGACY_PRODUCTION_MASS_DRIFT_LIMIT
            ),
            "result_does_not_authorize_production": True,
        },
        "provenance": {
            **provenance,
            "runner_sha256": sha256_path(SOURCE_PATH),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "outputs": {
            "compact_slices": "diagnostic-slices.npz",
            "full_field_checkpoint_saved": False,
            "restart_possible": False,
        },
    }
    _atomic_npz(output_directory / "diagnostic-slices.npz", slices)
    summary["outputs"]["compact_slices_sha256"] = sha256_path(
        output_directory / "diagnostic-slices.npz"
    )
    _atomic_json(output_directory / "summary.json", summary)
    _atomic_json(
        output_directory / "run_status.json",
        {
            "status": "completed",
            "classification": classification,
            "preflight_passed": passed,
            "completed_steps": {
                "roy": STEPS,
                "factory_g0": STEPS,
                "fixed_gb": STEPS,
            },
            "elapsed_seconds": elapsed_seconds,
            "summary_sha256": sha256_path(
                output_directory / "summary.json"
            ),
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        summary = run(output)
    except Exception as exc:
        if output.exists():
            _atomic_json(
                output / "run_status.json",
                {
                    "status": "failed_or_incomplete",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "automatic_resume": False,
                },
            )
        raise
    print(
        json.dumps(
            {
                "classification": summary["classification"],
                "elapsed_seconds": summary["execution"]["elapsed_seconds"],
                "preflight_passed": summary["preflight_passed"],
                "summary": str(output / "summary.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not summary["preflight_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
