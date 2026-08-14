#!/usr/bin/env python3
"""Static GB-source localization audit on the immutable Roy t=100 field."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..roy_2021_reproduction.model import (
    FROZEN_DEG90,
    released_noise_layout,
)
from . import common_state_response_metrics as response_metrics
from .geometry import centered_coordinates
from .metrics import array_fingerprint
from .model import (
    FixedGrainBoundaryParameters,
    build_periodic_bicrystal_profile,
    interpolation_derivative,
)
from .provenance import sha256_path, verify_frozen_baseline
from .run_full_grid_localization_preflight import (
    _atomic_json,
    _atomic_npz,
    peak_rss_bytes,
)


SOURCE_PATH = Path(__file__).resolve()
DIRECTORY = SOURCE_PATH.parent
BASELINE_DIRECTORY = DIRECTORY.parent / "roy_2021_reproduction"
T1000_RESULT_DIRECTORY = (
    BASELINE_DIRECTORY / "results" / "t1000_source_semantic_seed2292"
)
T1000_CONTRACT_PATH = T1000_RESULT_DIRECTORY / "contract.json"
T1000_STATUS_PATH = T1000_RESULT_DIRECTORY / "run_status.json"
T1000_SUMMARY_PATH = T1000_RESULT_DIRECTORY / "summary.json"
T100_CHECKPOINT_METADATA_PATH = (
    T1000_RESULT_DIRECTORY / "checkpoint-step-0100.json"
)
T100_CHECKPOINT_PATH = T1000_RESULT_DIRECTORY / "checkpoint-step-0100.npy"
T1000_RUNNER_PATH = BASELINE_DIRECTORY / "run_t1000.py"
ROY_MODEL_PATH = BASELINE_DIRECTORY / "model.py"
BRIDGE_MODEL_PATH = DIRECTORY / "model.py"
METRIC_SOURCE_PATH = DIRECTORY / "common_state_response_metrics.py"
GEOMETRY_SOURCE_PATH = DIRECTORY / "geometry.py"
FINGERPRINT_SOURCE_PATH = DIRECTORY / "metrics.py"
PROVENANCE_SOURCE_PATH = DIRECTORY / "provenance.py"
FULL_GRID_HELPER_SOURCE_PATH = (
    DIRECTORY / "run_full_grid_localization_preflight.py"
)
NOISY_T1_SUMMARY_PATH = (
    DIRECTORY / "results" / "common_state_response_preflight" / "summary.json"
)
CLEAN_T1_SUMMARY_PATH = (
    DIRECTORY
    / "results"
    / "noise_free_common_state_response_preflight"
    / "summary.json"
)
DEFAULT_OUTPUT_DIRECTORY = (
    DIRECTORY / "results" / "t100_checkpoint_source_localization_preflight"
)

EXPECTED_SHA256 = {
    "t1000_contract": (
        "b81d7651196bdbd03296b383625bd326d13bfc8215f75258b5f66dcb3e93e124"
    ),
    "t1000_status": (
        "abda121f4059a169f08ca0049ffda94d979fdc25e4dc5868f51bcc4a6da9f146"
    ),
    "t1000_summary": (
        "1629c71b5b1ec3c8c253b318da9031ac8d91689d9d2104905d287abab6a96976"
    ),
    "t100_checkpoint_metadata": (
        "b8f43d662ed54ae2d7879e3b0854271923d097f4b1e9c9d8ccee4c813afdb175"
    ),
    "t100_checkpoint": (
        "ee6a292c73170d1834c38f775ebc59798e6dc3d50e82d443d3a2f27a2ebaa360"
    ),
    "t1000_runner": (
        "f6b336bc8c7db99b7271c1cb42f70e6b9d45a3d0349963bb10c693f91ed41dff"
    ),
    "roy_model": (
        "3d335de207dabbb11c670e5fafea453348e2323f6740088325f100f93f062106"
    ),
    "bridge_model": (
        "0d22e4fef00e427821d586dce87a85ff93924e5fa2c9ef6a18913843d996a64f"
    ),
    "metric_source": (
        "330806d5327497d7334d8249a8aa746e04daf2ffee382d23f6b2d154bb5966ea"
    ),
    "geometry_source": (
        "ae2ac05939076ddc9964b75df9244dc0287d07d20f26eee151d388af281c5f0e"
    ),
    "fingerprint_source": (
        "fc699eada0f17f34c87c4a654f08ccaab607172445deacc3785d6e04c6d65963"
    ),
    "provenance_source": (
        "bcce214240d0c070a186d789e11545eeae7a4c9fbc7aaeb1488feef6a9315aa3"
    ),
    "full_grid_helper_source": (
        "1e1b235643871321e72fda92cd70f4ce8d002d3783df7d82a6d103a7c7003c7d"
    ),
    "noisy_t1_summary": (
        "dac6dec0baf0614dcf69ef842ce94c6b1ad760e04f278b61de02492b291a1f6b"
    ),
    "clean_t1_summary": (
        "03ac6e2dd010b95b99ceda4f2f85bf30c038adfdfaa56efc42763613a092a0ca"
    ),
}
EXPECTED_T100_FINGERPRINT = (
    "32e0c092f3e1471a8138f6084d9072fd3b68b47716a21ab4bb65583a569fe902"
)
EXPECTED_T100_FIELD_BYTES = 452_984_960
EXPECTED_T100_STEP = 100
EXPECTED_T100_TIME = 100.0
EXPECTED_T100_MINIMUM = -0.01921772211790085
EXPECTED_T100_MAXIMUM = 1.02471125125885
EXPECTED_T100_MASS = 83_903.5479192928
EXPECTED_T100_RELATIVE_MASS_DRIFT = 3.0053403513051563e-7
EXPECTED_T100_ENERGY = 6_651.300387067587
EXPECTED_RELEASED_PREFIX_LENGTH = 7_078_560

ENERGY_RATIO = 0.35
GB_AXIS = 2
GB_OFFSET_OVER_RADIUS = 2.6
TOTAL_PROPOSE_CALLS = 0
TOTAL_FFT_CALLS = 0
WALL_SECONDS_CAP = 120.0
SOURCE_CORE_RMS_MINIMUM = 1.0e-12
SOURCE_INTENDED_FRACTION_MINIMUM = 0.95
SOURCE_LEAKAGE_RMS_RATIO_MAXIMUM = 0.01
PRIMARY_IMAGE_RMS_RATIO_MINIMUM = 0.25
PRIMARY_IMAGE_RMS_RATIO_MAXIMUM = 4.0
MORPHOLOGY_LEVELS = (0.45, 0.50, 0.55)
PHASE_VAPOR_CUTOFF = 0.05
PHASE_SOLID_CUTOFF = 0.95
SOURCE_CHUNK = 16
MINIMUM_DISK_HEADROOM_BYTES = 1 << 30
CONSERVATIVE_BYTES_PER_CELL = 128


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return float(numerator / denominator)


def _periodic_distance(
    coordinate: np.ndarray,
    plane: float,
    length: float,
) -> np.ndarray:
    return np.abs(
        (coordinate - plane + 0.5 * length) % length - 0.5 * length
    )


def _file_state(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_path(path),
    }


def _records_match(
    first: dict[str, Any],
    second: dict[str, Any],
    names: tuple[str, ...],
) -> bool:
    return all(first.get(name) == second.get(name) for name in names)


def verify_sources() -> dict[str, Any]:
    baseline = verify_frozen_baseline()
    paths = {
        "t1000_contract": T1000_CONTRACT_PATH,
        "t1000_status": T1000_STATUS_PATH,
        "t1000_summary": T1000_SUMMARY_PATH,
        "t100_checkpoint_metadata": T100_CHECKPOINT_METADATA_PATH,
        "t100_checkpoint": T100_CHECKPOINT_PATH,
        "t1000_runner": T1000_RUNNER_PATH,
        "roy_model": ROY_MODEL_PATH,
        "bridge_model": BRIDGE_MODEL_PATH,
        "metric_source": METRIC_SOURCE_PATH,
        "geometry_source": GEOMETRY_SOURCE_PATH,
        "fingerprint_source": FINGERPRINT_SOURCE_PATH,
        "provenance_source": PROVENANCE_SOURCE_PATH,
        "full_grid_helper_source": FULL_GRID_HELPER_SOURCE_PATH,
        "noisy_t1_summary": NOISY_T1_SUMMARY_PATH,
        "clean_t1_summary": CLEAN_T1_SUMMARY_PATH,
    }
    measured = {name: sha256_path(path) for name, path in paths.items()}
    mismatches = {
        name: {
            "expected": EXPECTED_SHA256[name],
            "actual": actual,
        }
        for name, actual in measured.items()
        if actual != EXPECTED_SHA256[name]
    }

    metadata = _load_json(T100_CHECKPOINT_METADATA_PATH)
    contract = _load_json(T1000_CONTRACT_PATH)
    status = _load_json(T1000_STATUS_PATH)
    summary = _load_json(T1000_SUMMARY_PATH)
    noisy = _load_json(NOISY_T1_SUMMARY_PATH)
    clean = _load_json(CLEAN_T1_SUMMARY_PATH)
    checkpoint_names = (
        "step",
        "shape",
        "dtype",
        "field_bytes",
        "field_sha256",
        "field_fingerprint",
        "finite",
    )
    summary_checkpoint = next(
        (
            item
            for item in summary.get("checkpoints", [])
            if item.get("step") == EXPECTED_T100_STEP
        ),
        None,
    )
    status_checkpoint = next(
        (
            item
            for item in status.get("checkpoints", [])
            if item.get("step") == EXPECTED_T100_STEP
        ),
        None,
    )
    contract_checks = {
        "metadata_step": metadata.get("step") == EXPECTED_T100_STEP,
        "metadata_shape": tuple(metadata.get("shape", ()))
        == FROZEN_DEG90.lattice.shape,
        "metadata_dtype": metadata.get("dtype") == "<f8",
        "metadata_bytes": metadata.get("field_bytes")
        == EXPECTED_T100_FIELD_BYTES,
        "metadata_fingerprint": metadata.get("field_fingerprint")
        == EXPECTED_T100_FINGERPRINT,
        "metadata_field_sha": metadata.get("field_sha256")
        == EXPECTED_SHA256["t100_checkpoint"],
        "metadata_finite": metadata.get("finite") is True,
        "summary_completed": summary.get("status") == "completed",
        "summary_target": summary.get("contract", {}).get("target_step")
        == 1000,
        "summary_runner": summary.get("contract", {}).get("runner_sha256")
        == EXPECTED_SHA256["t1000_runner"],
        "summary_model": summary.get("contract", {}).get("model_sha256")
        == EXPECTED_SHA256["roy_model"],
        "status_completed": status.get("status") == "completed",
        "status_current_step": status.get("current_step") == 1000,
        "contract_runner": contract.get("runner_sha256")
        == EXPECTED_SHA256["t1000_runner"],
        "contract_model": contract.get("model_sha256")
        == EXPECTED_SHA256["roy_model"],
        "summary_checkpoint_present": summary_checkpoint is not None,
        "status_checkpoint_present": status_checkpoint is not None,
        "summary_checkpoint_matches_metadata": bool(
            summary_checkpoint is not None
            and _records_match(metadata, summary_checkpoint, checkpoint_names)
        ),
        "status_checkpoint_matches_metadata": bool(
            status_checkpoint is not None
            and _records_match(metadata, status_checkpoint, checkpoint_names)
        ),
        "noisy_t1_classification": noisy.get("classification")
        == (
            "common_state_response_inconclusive_"
            "released_noise_source_dominant_broad_by_frozen_roi_checks"
        ),
        "clean_t1_classification": clean.get("classification")
        == (
            "noise_free_common_state_inconclusive_"
            "clean_source_not_interpretable_or_localized"
        ),
    }
    failed_contract_checks = {
        name: passed for name, passed in contract_checks.items() if not passed
    }
    if mismatches or failed_contract_checks:
        raise RuntimeError(
            "t100 source audit provenance mismatch: "
            + json.dumps(
                {
                    "hash_mismatches": mismatches,
                    "contract_failures": failed_contract_checks,
                },
                sort_keys=True,
            )
        )
    return {
        **baseline,
        "measured_sha256": measured,
        "checkpoint_contract_checks": contract_checks,
        "verified": True,
    }


def resource_preflight(output_directory: Path) -> dict[str, Any]:
    cell_count = FROZEN_DEG90.lattice.cell_count
    estimated_peak_bytes = cell_count * CONSERVATIVE_BYTES_PER_CELL
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
    free_disk_bytes = int(shutil.disk_usage(output_directory.parent).free)
    checks = {
        "physical_memory_sufficient": (
            physical_memory_bytes >= estimated_peak_bytes
        ),
        "available_memory_sufficient_if_measurable": (
            available_memory_bytes is None
            or available_memory_bytes >= estimated_peak_bytes
        ),
        "disk_headroom_sufficient": (
            free_disk_bytes >= MINIMUM_DISK_HEADROOM_BYTES
        ),
    }
    return {
        "cell_count": cell_count,
        "conservative_bytes_per_cell": CONSERVATIVE_BYTES_PER_CELL,
        "estimated_peak_bytes": estimated_peak_bytes,
        "checkpoint_file_bytes": T100_CHECKPOINT_PATH.stat().st_size,
        "physical_memory_bytes": physical_memory_bytes,
        "available_memory_bytes": available_memory_bytes,
        "free_disk_bytes": free_disk_bytes,
        "minimum_disk_headroom_bytes": MINIMUM_DISK_HEADROOM_BYTES,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _state_record_from_summary() -> dict[str, Any]:
    summary = _load_json(T1000_SUMMARY_PATH)
    record = next(
        item
        for item in summary["records"]
        if item["step"] == EXPECTED_T100_STEP
    )
    topology = record["four_arm_topology"]["thresholds"]
    checks = {
        "step_is_100": record["step"] == EXPECTED_T100_STEP,
        "time_is_100": record["time"] == EXPECTED_T100_TIME,
        "field_is_finite": record["field"]["finite"] is True,
        "field_shape_matches": tuple(record["field"]["shape"])
        == FROZEN_DEG90.lattice.shape,
        "field_dtype_is_float64": record["field"]["dtype"] == "float64",
        "minimum_matches": record["field"]["minimum"]
        == EXPECTED_T100_MINIMUM,
        "maximum_matches": record["field"]["maximum"]
        == EXPECTED_T100_MAXIMUM,
        "mass_matches": record["field"]["mass"] == EXPECTED_T100_MASS,
        "mass_drift_matches": record["relative_mass_drift"]
        == EXPECTED_T100_RELATIVE_MASS_DRIFT,
        "mass_drift_below_roy_limit": record["relative_mass_drift"]
        <= summary["contract"]["mass_drift_limit"],
        "energy_matches": record["free_energy"] == EXPECTED_T100_ENERGY,
        "energy_below_initial": record["free_energy"]
        < summary["records"][0]["free_energy"],
        "all_four_arms_attached": all(
            item["all_four_arms_attached"] for item in topology.values()
        ),
    }
    return {
        "record": record,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def morphology_preflight(
    field: np.ndarray,
    *,
    primary_coordinate: float,
    image_coordinate: float,
    radius: float,
    surface_width: float,
) -> dict[str, Any]:
    lattice = FROZEN_DEG90.lattice
    x = centered_coordinates(lattice, 0)
    y = centered_coordinates(lattice, 1)
    z = centered_coordinates(lattice, 2)
    r1 = np.sqrt(x[:, None] ** 2 + y[None, :] ** 2)
    length_z = lattice.physical_lengths[2]
    dp = _periodic_distance(z, primary_coordinate, length_z)
    di = _periodic_distance(z, image_coordinate, length_z)
    dg = np.minimum(dp, di)
    band_indices = np.flatnonzero(dg <= 2.0 * surface_width)
    slices: list[dict[str, Any]] = []
    maximum_radius_error = 0.0
    interface_cells = 0
    strict_escaped_interface_cells = 0
    grid_resolved_escaped_interface_cells = 0
    maximum_strict_escape = 0.0
    grid_radial_tolerance = lattice.spacing / np.sqrt(2.0)
    for index in band_indices:
        plane = np.asarray(field[:, :, index])
        equivalent: dict[str, float] = {}
        for level in MORPHOLOGY_LEVELS:
            count = int(np.count_nonzero(plane >= level))
            measured_radius = float(
                np.sqrt(count * lattice.spacing**2 / np.pi)
            )
            equivalent[f"{level:.2f}"] = measured_radius
            maximum_radius_error = max(
                maximum_radius_error,
                abs(measured_radius - radius),
            )
        interface = (
            (plane >= PHASE_VAPOR_CUTOFF)
            & (plane <= PHASE_SOLID_CUTOFF)
        )
        interface_count = int(np.count_nonzero(interface))
        interface_cells += interface_count
        if interface_count:
            radii = r1[interface]
            radial_minimum: float | None = float(np.min(radii))
            radial_maximum: float | None = float(np.max(radii))
            radial_excess = (
                np.abs(radii - radius) - surface_width
            )
            strict_escaped = int(
                np.count_nonzero(radial_excess > 0.0)
            )
            grid_resolved_escaped = int(
                np.count_nonzero(
                    radial_excess > grid_radial_tolerance
                )
            )
            maximum_strict_escape = max(
                maximum_strict_escape,
                float(max(0.0, np.max(radial_excess))),
            )
        else:
            radial_minimum = None
            radial_maximum = None
            strict_escaped = 0
            grid_resolved_escaped = 0
        strict_escaped_interface_cells += strict_escaped
        grid_resolved_escaped_interface_cells += (
            grid_resolved_escaped
        )
        plane_name = "primary" if dp[index] <= di[index] else "image"
        slices.append(
            {
                "z_index": int(index),
                "z_coordinate": float(z[index]),
                "nearest_plane": plane_name,
                "distance_to_nearest_plane": float(dg[index]),
                "equivalent_radius_by_level": equivalent,
                "center_composition": float(
                    field[
                        lattice.shape[0] // 2,
                        lattice.shape[1] // 2,
                        index,
                    ]
                ),
                "interface_cell_count": interface_count,
                "interface_radial_minimum": radial_minimum,
                "interface_radial_maximum": radial_maximum,
                "strict_interface_escape_count": strict_escaped,
                "grid_resolved_interface_escape_count": (
                    grid_resolved_escaped
                ),
            }
        )
    plane_counts = {
        name: sum(item["nearest_plane"] == name for item in slices)
        for name in ("primary", "image")
    }
    checks = {
        "both_gb_bands_have_grid_planes": all(
            count > 0 for count in plane_counts.values()
        ),
        "all_equivalent_radii_within_one_width": (
            maximum_radius_error <= surface_width
        ),
        "interface_band_is_nonempty": interface_cells > 0,
        "all_interface_cells_inside_grid_resolved_frozen_ring": (
            grid_resolved_escaped_interface_cells == 0
        ),
    }
    return {
        "composition_interface_band": [
            PHASE_VAPOR_CUTOFF,
            PHASE_SOLID_CUTOFF,
        ],
        "gb_band_distance_limit_over_width": 2.0,
        "frozen_radius": radius,
        "surface_width": surface_width,
        "plane_slice_counts": plane_counts,
        "maximum_equivalent_radius_error": maximum_radius_error,
        "maximum_equivalent_radius_error_over_width": (
            maximum_radius_error / surface_width
        ),
        "interface_cell_count": interface_cells,
        "strict_interface_escape_count": strict_escaped_interface_cells,
        "strict_interface_escape_fraction": (
            strict_escaped_interface_cells / interface_cells
            if interface_cells
            else None
        ),
        "maximum_strict_escape_physical": maximum_strict_escape,
        "maximum_strict_escape_over_width": (
            maximum_strict_escape / surface_width
        ),
        "grid_radial_tolerance_definition": (
            "Cartesian cell half-diagonal h/sqrt(2)"
        ),
        "grid_radial_tolerance_physical": grid_radial_tolerance,
        "grid_radial_tolerance_over_width": (
            grid_radial_tolerance / surface_width
        ),
        "grid_resolved_interface_escape_count": (
            grid_resolved_escaped_interface_cells
        ),
        "grid_resolved_interface_escape_fraction": (
            grid_resolved_escaped_interface_cells / interface_cells
            if interface_cells
            else None
        ),
        "slices": slices,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def compute_source_and_representation_floor(
    field: np.ndarray,
    density: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.empty(field.shape, dtype=np.float64)
    floor = np.empty_like(source)
    for start in range(0, field.shape[2], SOURCE_CHUNK):
        stop = min(field.shape[2], start + SOURCE_CHUNK)
        values = np.asarray(field[:, :, start:stop], dtype=np.float64)
        density_block = density[start:stop][None, None, :]
        source[:, :, start:stop] = (
            interpolation_derivative(values) * density_block
        )
        values32 = np.asarray(values, dtype=np.float32)
        upward = np.nextafter(values32, np.float32(np.inf)).astype(
            np.float64
        )
        downward = np.nextafter(
            values32, np.float32(-np.inf)
        ).astype(np.float64)
        center = values32.astype(np.float64)
        q_center = interpolation_derivative(center) * density_block
        q_upward = interpolation_derivative(upward) * density_block
        q_downward = interpolation_derivative(downward) * density_block
        floor[:, :, start:stop] = np.maximum(
            np.abs(q_upward - q_center),
            np.abs(q_downward - q_center),
        )
    return source, floor


def compute_source_after_morphology_preflight(
    field: np.ndarray,
    density: np.ndarray,
    morphology: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Refuse any source evaluation when the analytic ROI is unsuitable."""
    if not morphology.get("passed", False):
        raise RuntimeError(
            "analytic ROI morphology preflight failed before source evaluation"
        )
    return compute_source_and_representation_floor(field, density)


def _empty_accumulator() -> dict[str, Any]:
    return {
        "cell_count": 0,
        "sum_squared": 0.0,
        "maximum_absolute": 0.0,
        "floor_sum_squared": 0.0,
        "floor_maximum": 0.0,
        "above_floor_sum_squared": {
            "1x": 0.0,
            "2x": 0.0,
            "4x": 0.0,
        },
    }


def _accumulate_region(
    accumulator: dict[str, Any],
    source: np.ndarray,
    floor: np.ndarray,
    mask: np.ndarray,
) -> None:
    selected_source = source[mask]
    selected_floor = floor[mask]
    if not selected_source.size:
        return
    squared = selected_source * selected_source
    accumulator["cell_count"] += int(selected_source.size)
    accumulator["sum_squared"] += float(
        np.sum(squared, dtype=np.float64)
    )
    accumulator["maximum_absolute"] = max(
        accumulator["maximum_absolute"],
        float(np.max(np.abs(selected_source))),
    )
    accumulator["floor_sum_squared"] += float(
        np.sum(selected_floor * selected_floor, dtype=np.float64)
    )
    accumulator["floor_maximum"] = max(
        accumulator["floor_maximum"],
        float(np.max(selected_floor)),
    )
    for multiple, name in ((1.0, "1x"), (2.0, "2x"), (4.0, "4x")):
        active = np.abs(selected_source) > multiple * selected_floor
        accumulator["above_floor_sum_squared"][name] += float(
            np.sum(squared[active], dtype=np.float64)
        )


def _finalize_accumulator(
    accumulator: dict[str, Any],
    *,
    total_source_squared: float,
) -> dict[str, Any]:
    count = int(accumulator["cell_count"])
    source_squared = float(accumulator["sum_squared"])
    floor_squared = float(accumulator["floor_sum_squared"])
    source_rms = float(np.sqrt(source_squared / count)) if count else 0.0
    floor_rms = float(np.sqrt(floor_squared / count)) if count else 0.0
    maximum = float(accumulator["maximum_absolute"])
    floor_maximum = float(accumulator["floor_maximum"])
    rms_ratio = _safe_ratio(source_rms, floor_rms)
    maximum_ratio = _safe_ratio(maximum, floor_maximum)
    resolved = bool(
        (
            (floor_rms == 0.0 and source_rms > 0.0)
            or (rms_ratio is not None and rms_ratio > 1.0)
        )
        and (
            (floor_maximum == 0.0 and maximum > 0.0)
            or (maximum_ratio is not None and maximum_ratio > 1.0)
        )
    )
    return {
        "cell_count": count,
        "sum_squared": source_squared,
        "fraction_of_total_squared_norm": (
            source_squared / total_source_squared
            if total_source_squared > 0.0
            else 0.0
        ),
        "rms": source_rms,
        "maximum_absolute": maximum,
        "representation_floor_rms": floor_rms,
        "representation_floor_maximum": floor_maximum,
        "rms_over_representation_floor": rms_ratio,
        "maximum_over_representation_floor": maximum_ratio,
        "resolved_above_representation_floor": resolved,
        "source_squared_norm_fraction_above_floor": {
            name: (
                value / source_squared if source_squared > 0.0 else 0.0
            )
            for name, value in accumulator[
                "above_floor_sum_squared"
            ].items()
        },
    }


def partition_source(
    field: np.ndarray,
    source: np.ndarray,
    floor: np.ndarray,
    *,
    primary_coordinate: float,
    image_coordinate: float,
    radius: float,
    surface_width: float,
    noise_prefix_length: int,
) -> dict[str, Any]:
    lattice = FROZEN_DEG90.lattice
    nx, ny, nz = lattice.shape
    x = centered_coordinates(lattice, 0)
    y = centered_coordinates(lattice, 1)
    z = centered_coordinates(lattice, 2)
    r1 = np.sqrt(x[:, None] ** 2 + y[None, :] ** 2)
    surface_distance = np.abs(r1 - radius)
    base_flat_index = (
        (
            np.arange(nx, dtype=np.int64)[:, None]
            * ny
            + np.arange(ny, dtype=np.int64)[None, :]
        )
        * nz
    )
    length_z = lattice.physical_lengths[2]
    component_accumulators = {
        name: _empty_accumulator()
        for name in (
            "former_released_prefix_region_source",
            "intended_intersection_source",
            "residual_source",
        )
    }
    phase_accumulators = {
        f"{phase}_{location}": _empty_accumulator()
        for phase in ("vapor", "diffuse", "dense")
        for location in ("gb_band", "far_background")
    }
    maximum_recomposition_error = 0.0
    mask_count = 0
    for start in range(0, nz, SOURCE_CHUNK):
        stop = min(nz, start + SOURCE_CHUNK)
        z_block = z[start:stop]
        dp = _periodic_distance(
            z_block, primary_coordinate, length_z
        )
        di = _periodic_distance(z_block, image_coordinate, length_z)
        plane_distance = np.minimum(dp, di)
        rho = np.hypot(
            surface_distance[:, :, None],
            plane_distance[None, None, :],
        )
        flat_index = (
            base_flat_index[:, :, None]
            + np.arange(start, stop, dtype=np.int64)[None, None, :]
        )
        prefix = flat_index < noise_prefix_length
        intended = ~prefix & (rho <= 2.0 * surface_width)
        residual = ~(prefix | intended)
        source_block = source[:, :, start:stop]
        floor_block = floor[:, :, start:stop]
        masks = {
            "former_released_prefix_region_source": prefix,
            "intended_intersection_source": intended,
            "residual_source": residual,
        }
        mask_count += sum(int(np.count_nonzero(mask)) for mask in masks.values())
        reconstructed = np.zeros_like(source_block)
        for name, mask in masks.items():
            _accumulate_region(
                component_accumulators[name],
                source_block,
                floor_block,
                mask,
            )
            reconstructed[mask] = source_block[mask]
        maximum_recomposition_error = max(
            maximum_recomposition_error,
            float(np.max(np.abs(source_block - reconstructed))),
        )

        values = np.asarray(field[:, :, start:stop])
        phase_masks = {
            "vapor": values < PHASE_VAPOR_CUTOFF,
            "diffuse": (
                (values >= PHASE_VAPOR_CUTOFF)
                & (values <= PHASE_SOLID_CUTOFF)
            ),
            "dense": values > PHASE_SOLID_CUTOFF,
        }
        gb_band = np.broadcast_to(
            plane_distance[None, None, :] <= 2.0 * surface_width,
            source_block.shape,
        )
        for phase_name, phase_mask in phase_masks.items():
            _accumulate_region(
                phase_accumulators[f"{phase_name}_gb_band"],
                source_block,
                floor_block,
                phase_mask & gb_band,
            )
            _accumulate_region(
                phase_accumulators[
                    f"{phase_name}_far_background"
                ],
                source_block,
                floor_block,
                phase_mask & ~gb_band,
            )

    total_squared = float(
        np.sum(source * source, dtype=np.float64)
    )
    component_reports = {
        name: _finalize_accumulator(
            accumulator,
            total_source_squared=total_squared,
        )
        for name, accumulator in component_accumulators.items()
    }
    phase_reports = {
        name: _finalize_accumulator(
            accumulator,
            total_source_squared=total_squared,
        )
        for name, accumulator in phase_accumulators.items()
    }
    component_squared = sum(
        item["sum_squared"] for item in component_reports.values()
    )
    return {
        "components": component_reports,
        "phase_and_plane_partitions": phase_reports,
        "total_sum_squared": total_squared,
        "mask_cell_count": mask_count,
        "mask_count_is_exhaustive": mask_count == lattice.cell_count,
        "maximum_recomposition_error": maximum_recomposition_error,
        "relative_squared_norm_accounting_error": (
            abs(component_squared - total_squared)
            / max(total_squared, np.finfo(float).tiny)
        ),
    }


def _representation_record(
    source_statistics: dict[str, Any],
    floor_statistics: dict[str, Any],
) -> dict[str, Any]:
    source_rms = float(source_statistics["rms"])
    floor_rms = float(floor_statistics["rms"])
    source_maximum = float(source_statistics["maximum_absolute"])
    floor_maximum = float(floor_statistics["maximum_absolute"])
    rms_ratio = _safe_ratio(source_rms, floor_rms)
    maximum_ratio = _safe_ratio(source_maximum, floor_maximum)
    return {
        "source_rms": source_rms,
        "source_maximum": source_maximum,
        "floor_rms": floor_rms,
        "floor_maximum": floor_maximum,
        "rms_ratio": rms_ratio,
        "maximum_ratio": maximum_ratio,
        "resolved": bool(
            (
                (floor_rms == 0.0 and source_rms > 0.0)
                or (rms_ratio is not None and rms_ratio > 1.0)
            )
            and (
                (floor_maximum == 0.0 and source_maximum > 0.0)
                or (
                    maximum_ratio is not None
                    and maximum_ratio > 1.0
                )
            )
        ),
    }


def classify(
    *,
    hard_checks: dict[str, bool],
    state_checks: dict[str, bool],
    morphology_checks: dict[str, bool],
    core_resolution_checks: dict[str, bool],
    source_checks: dict[str, bool],
    unresolved_failed_checks: dict[str, Any],
) -> tuple[str, bool]:
    if not all(hard_checks.values()):
        return "t100_source_localization_failed_integrity", False
    if not all(state_checks.values()):
        return "t100_source_localization_state_suitability_inconclusive", False
    if not all(morphology_checks.values()):
        return "t100_source_localization_analytic_roi_invalid", False
    if not all(core_resolution_checks.values()):
        return "t100_source_at_checkpoint_representation_floor", False
    if not source_checks["primary_and_image_source_cores_are_active"]:
        return "t100_source_below_activation_threshold", False
    if not source_checks["primary_image_source_ratio_is_bounded"]:
        return "t100_source_primary_image_asymmetry_inconclusive", False
    if not all(source_checks.values()):
        failed_source_checks = {
            name for name, passed in source_checks.items() if not passed
        }
        if failed_source_checks.issubset(unresolved_failed_checks):
            return (
                "t100_source_localization_failure_at_"
                "checkpoint_representation_floor",
                False,
            )
        return "t100_gb_source_not_localized_on_mature_roy_checkpoint", False
    return "t100_gb_source_localized_on_mature_roy_checkpoint", True


def _quantile_upper(
    analysis: dict[str, Any],
    *,
    field_name: str,
    profile_name: str,
    quantile_name: str,
) -> float | None:
    item = analysis["profiles"][field_name][profile_name][
        "quantiles"
    ][quantile_name]
    if item is None:
        return None
    return float(item["grid_resolved_upper_bound_over_width"])


def _prior_source_scalars(
    summary: dict[str, Any],
    *,
    field_name: str,
) -> dict[str, Any]:
    sums = summary["response"]["source_component_sum_squared"]
    if "total_source" in sums:
        mapped = {
            "total": sums["total_source"],
            "former_prefix_region": sums["released_prefix_source"],
            "intended_intersection": sums[
                "intended_intersection_source"
            ],
            "residual": sums["residual_source"],
        }
    else:
        mapped = {
            "total": sums["total_clean_source"],
            "former_prefix_region": sums[
                "former_released_prefix_region_source"
            ],
            "intended_intersection": sums[
                "intended_intersection_source"
            ],
            "residual": sums["residual_source"],
        }
    roi = summary["response"]["rois"][field_name]
    return {
        "sum_squared": mapped,
        "fractions": {
            name: value / mapped["total"]
            for name, value in mapped.items()
            if name != "total"
        },
        "rms": roi["all"]["rms"],
        "maximum_absolute": roi["all"]["maximum_absolute"],
        "primary_core_rms": roi["primary_core"]["rms"],
        "image_core_rms": roi["image_core"]["rms"],
        "primary_image_core_rms_ratio": _safe_ratio(
            roi["primary_core"]["rms"],
            roi["image_core"]["rms"],
        ),
        "junction_rms": roi["junction"]["rms"],
        "second_wire_rms": roi["second_wire"]["rms"],
        "far_arm_rms": roi["far_arm"]["rms"],
        "quantile_upper_bounds_over_width": {
            profile_name: {
                quantile_name: _quantile_upper(
                    summary["response"],
                    field_name=field_name,
                    profile_name=profile_name,
                    quantile_name=quantile_name,
                )
                for quantile_name in ("w50", "w90")
            }
            for profile_name in ("rho_all", "plane_all", "surface_all")
        },
    }


def run(output_directory: Path) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output: {output_directory}"
        )
    output_directory.mkdir(parents=True)
    started = time.perf_counter()
    provenance = verify_sources()
    resources = resource_preflight(output_directory)
    state_contract = _state_record_from_summary()
    checkpoint_file_before = _file_state(T100_CHECKPOINT_PATH)
    scope = {
        "solver_instances_constructed": 0,
        "propose_step_calls": TOTAL_PROPOSE_CALLS,
        "fft_calls": TOTAL_FFT_CALLS,
        "accepted_trajectory_steps": 0,
        "physical_time_before": EXPECTED_T100_TIME,
        "physical_time_after": EXPECTED_T100_TIME,
        "source_only_static_audit": True,
        "initializer_adopted": False,
        "initializer_pivot_authorized": False,
        "production_authorized": False,
        "continuation_authorized": False,
        "resume_authorized": False,
        "parameter_sweep_authorized": False,
        "automatic_follow_on": False,
    }
    contract = {
        "schema_version": 1,
        "run": "t100 Roy checkpoint static fixed-GB source localization",
        "checkpoint_step": EXPECTED_T100_STEP,
        "checkpoint_time": EXPECTED_T100_TIME,
        "shape": FROZEN_DEG90.lattice.shape,
        "spacing": FROZEN_DEG90.lattice.spacing,
        "energy_ratio": ENERGY_RATIO,
        "gb_axis": GB_AXIS,
        "gb_offset_over_radius": GB_OFFSET_OVER_RADIUS,
        "total_propose_calls": TOTAL_PROPOSE_CALLS,
        "total_fft_calls": TOTAL_FFT_CALLS,
        "wall_seconds_cap": WALL_SECONDS_CAP,
        "phase_partition_cutoffs": [
            PHASE_VAPOR_CUTOFF,
            PHASE_SOLID_CUTOFF,
        ],
        "acceptance_thresholds": {
            "source_core_rms_minimum": SOURCE_CORE_RMS_MINIMUM,
            "source_intended_fraction_minimum": (
                SOURCE_INTENDED_FRACTION_MINIMUM
            ),
            "source_leakage_rms_ratio_maximum": (
                SOURCE_LEAKAGE_RMS_RATIO_MAXIMUM
            ),
            "primary_image_rms_ratio": [
                PRIMARY_IMAGE_RMS_RATIO_MINIMUM,
                PRIMARY_IMAGE_RMS_RATIO_MAXIMUM,
            ],
            "morphology_radius_error_maximum_over_width": 1.0,
            "morphology_grid_radial_tolerance": (
                "h/sqrt(2), the Cartesian cell half-diagonal"
            ),
            "morphology_grid_radial_tolerance_physical": (
                FROZEN_DEG90.lattice.spacing / np.sqrt(2.0)
            ),
            "morphology_grid_resolved_escape_cells_maximum": 0,
            "representation_floor_ratio_boundary": 1.0,
        },
        "frozen_gate_note": (
            "Only source gates already frozen for the t1 common-state "
            "diagnostics decide localization; far-arm, phase partitions, "
            "symmetry details, and W50/W90 remain descriptive."
        ),
        "scope": scope,
        "provenance": provenance,
        "resource_preflight": resources,
    }
    _atomic_json(output_directory / "contract.json", contract)
    _atomic_json(
        output_directory / "run_status.json",
        {
            "status": "running",
            "started_unix": time.time(),
            "propose_step_calls": 0,
            "fft_calls": 0,
        },
    )
    if not resources["passed"]:
        raise RuntimeError("resource preflight failed")
    if not state_contract["passed"]:
        raise RuntimeError("stored t100 state contract failed")

    checkpoint = np.load(
        T100_CHECKPOINT_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )
    checkpoint_fingerprint_before = array_fingerprint(checkpoint)
    checkpoint_checks = {
        "mmap_is_read_only": not checkpoint.flags.writeable,
        "shape_matches": checkpoint.shape == FROZEN_DEG90.lattice.shape,
        "dtype_is_float64": checkpoint.dtype == np.dtype("<f8"),
        "c_contiguous": checkpoint.flags.c_contiguous,
        "finite": bool(np.isfinite(checkpoint).all()),
        "fingerprint_matches": (
            checkpoint_fingerprint_before
            == EXPECTED_T100_FINGERPRINT
        ),
        "minimum_matches": float(np.min(checkpoint))
        == EXPECTED_T100_MINIMUM,
        "maximum_matches": float(np.max(checkpoint))
        == EXPECTED_T100_MAXIMUM,
        "mass_matches": float(
            np.sum(checkpoint, dtype=np.float64)
            * FROZEN_DEG90.lattice.cell_volume
        )
        == EXPECTED_T100_MASS,
    }
    if not all(checkpoint_checks.values()):
        raise RuntimeError(
            "loaded checkpoint contract failed: "
            + json.dumps(checkpoint_checks, sort_keys=True)
        )

    radius = FROZEN_DEG90.radius_1 * FROZEN_DEG90.lattice.spacing
    mapped = FixedGrainBoundaryParameters.matched_to_roy(
        FROZEN_DEG90.parameters,
        energy_ratio=ENERGY_RATIO,
    )
    profile = build_periodic_bicrystal_profile(
        FROZEN_DEG90.lattice,
        mapped,
        axis=GB_AXIS,
        primary_coordinate=GB_OFFSET_OVER_RADIUS * radius,
    )
    plane_energy = profile.per_plane_energy_audit()
    noise_layout = released_noise_layout(FROZEN_DEG90.lattice)
    if (
        noise_layout.unique_target_prefix_length
        != EXPECTED_RELEASED_PREFIX_LENGTH
    ):
        raise RuntimeError("released prefix length contract mismatch")
    morphology = morphology_preflight(
        checkpoint,
        primary_coordinate=profile.primary_coordinate,
        image_coordinate=profile.image_coordinate,
        radius=radius,
        surface_width=mapped.surface_width,
    )
    if time.perf_counter() - started > WALL_SECONDS_CAP:
        raise TimeoutError("wall cap reached before source calculation")

    source, representation_floor = (
        compute_source_after_morphology_preflight(
            checkpoint,
            profile.grain_energy_density,
            morphology,
        )
    )
    analysis = response_metrics.analyze_fields(
        {
            "conditioned_t100_source": source,
            "checkpoint_composition_representation_floor": (
                representation_floor
            ),
        },
        primary_coordinate=profile.primary_coordinate,
        image_coordinate=profile.image_coordinate,
        radius=radius,
        surface_width=mapped.surface_width,
        noise_prefix_length=EXPECTED_RELEASED_PREFIX_LENGTH,
    )
    partitions = partition_source(
        checkpoint,
        source,
        representation_floor,
        primary_coordinate=profile.primary_coordinate,
        image_coordinate=profile.image_coordinate,
        radius=radius,
        surface_width=mapped.surface_width,
        noise_prefix_length=EXPECTED_RELEASED_PREFIX_LENGTH,
    )
    if time.perf_counter() - started > WALL_SECONDS_CAP:
        raise TimeoutError("wall cap reached during source analysis")

    rois = analysis["rois"]
    source_rois = rois["conditioned_t100_source"]
    floor_rois = rois[
        "checkpoint_composition_representation_floor"
    ]
    representation_by_roi = {
        name: _representation_record(source_rois[name], floor_rois[name])
        for name in (
            "all",
            "primary_core",
            "image_core",
            "intended_halo",
            "junction",
            "second_wire",
            "far_arm",
            "released_prefix",
        )
    }
    primary_rms = source_rois["primary_core"]["rms"]
    image_rms = source_rois["image_core"]["rms"]
    weaker_core = min(primary_rms, image_rms)
    primary_image_ratio = (
        primary_rms / image_rms
        if image_rms > 0.0
        else float("inf")
    )
    components = partitions["components"]
    intended_fraction = components["intended_intersection_source"][
        "fraction_of_total_squared_norm"
    ]
    strict_majority = bool(
        components["intended_intersection_source"]["sum_squared"]
        > components["former_released_prefix_region_source"][
            "sum_squared"
        ]
        + components["residual_source"]["sum_squared"]
    )
    source_checks = {
        "primary_and_image_source_cores_are_active": (
            weaker_core >= SOURCE_CORE_RMS_MINIMUM
        ),
        "primary_image_source_ratio_is_bounded": (
            PRIMARY_IMAGE_RMS_RATIO_MINIMUM
            <= primary_image_ratio
            <= PRIMARY_IMAGE_RMS_RATIO_MAXIMUM
        ),
        "intended_intersection_contains_95_percent_source_norm": (
            intended_fraction >= SOURCE_INTENDED_FRACTION_MINIMUM
        ),
        "intended_intersection_is_strict_source_majority": (
            strict_majority
        ),
        "junction_source_below_one_percent_core": (
            source_rois["junction"]["rms"]
            <= SOURCE_LEAKAGE_RMS_RATIO_MAXIMUM * weaker_core
        ),
        "second_wire_source_below_one_percent_core": (
            source_rois["second_wire"]["rms"]
            <= SOURCE_LEAKAGE_RMS_RATIO_MAXIMUM * weaker_core
        ),
    }
    core_resolution_checks = {
        "primary_core_resolved": representation_by_roi[
            "primary_core"
        ]["resolved"],
        "image_core_resolved": representation_by_roi[
            "image_core"
        ]["resolved"],
    }
    unresolved_failed_checks: dict[str, Any] = {}
    localization_gate_names = (
        "intended_intersection_contains_95_percent_source_norm",
        "intended_intersection_is_strict_source_majority",
    )
    if any(not source_checks[name] for name in localization_gate_names):
        nonintended_resolved = bool(
            components["former_released_prefix_region_source"][
                "resolved_above_representation_floor"
            ]
            or components["residual_source"][
                "resolved_above_representation_floor"
            ]
        )
        if not nonintended_resolved:
            for check_name in localization_gate_names:
                if not source_checks[check_name]:
                    unresolved_failed_checks[check_name] = [
                        "former_released_prefix_region_source",
                        "residual_source",
                    ]
    for check_name, roi_name in (
        ("junction_source_below_one_percent_core", "junction"),
        ("second_wire_source_below_one_percent_core", "second_wire"),
    ):
        if (
            not source_checks[check_name]
            and not representation_by_roi[roi_name]["resolved"]
        ):
            unresolved_failed_checks[check_name] = [roi_name]

    checkpoint_fingerprint_after = array_fingerprint(checkpoint)
    checkpoint_file_after = _file_state(T100_CHECKPOINT_PATH)
    hard_checks = {
        "provenance_verified": provenance["verified"],
        "resource_preflight_passed": resources["passed"],
        "checkpoint_contract_passed": all(checkpoint_checks.values()),
        "zero_solver_instances": (
            scope["solver_instances_constructed"] == 0
        ),
        "zero_propose_step_calls": scope["propose_step_calls"] == 0,
        "zero_fft_calls": scope["fft_calls"] == 0,
        "zero_accepted_trajectory_steps": (
            scope["accepted_trajectory_steps"] == 0
        ),
        "physical_time_unchanged": (
            scope["physical_time_before"]
            == scope["physical_time_after"]
            == EXPECTED_T100_TIME
        ),
        "checkpoint_fingerprint_unchanged": (
            checkpoint_fingerprint_before
            == checkpoint_fingerprint_after
            == EXPECTED_T100_FINGERPRINT
        ),
        "checkpoint_file_unchanged": (
            checkpoint_file_before == checkpoint_file_after
        ),
        "source_and_floor_finite": bool(
            np.isfinite(source).all()
            and np.isfinite(representation_floor).all()
        ),
        "source_masks_exhaustive": partitions[
            "mask_count_is_exhaustive"
        ],
        "source_recomposition_exact": (
            partitions["maximum_recomposition_error"] == 0.0
        ),
        "source_squared_norm_accounting": (
            partitions["relative_squared_norm_accounting_error"]
            <= 1.0e-12
        ),
        "histogram_and_partition_accounting": all(
            analysis["accounting_checks"].values()
        ),
        "gb_plane_energy_audit_passed": (
            plane_energy["primary_relative_error"] <= 5.0e-3
            and plane_energy["image_relative_error"] <= 5.0e-3
            and plane_energy["image_mismatch_relative"] <= 1.0e-12
        ),
    }

    current_scalars = {
        "sum_squared": {
            "total": partitions["total_sum_squared"],
            "former_prefix_region": components[
                "former_released_prefix_region_source"
            ]["sum_squared"],
            "intended_intersection": components[
                "intended_intersection_source"
            ]["sum_squared"],
            "residual": components["residual_source"]["sum_squared"],
        },
        "fractions": {
            "former_prefix_region": components[
                "former_released_prefix_region_source"
            ]["fraction_of_total_squared_norm"],
            "intended_intersection": components[
                "intended_intersection_source"
            ]["fraction_of_total_squared_norm"],
            "residual": components["residual_source"][
                "fraction_of_total_squared_norm"
            ],
        },
        "rms": source_rois["all"]["rms"],
        "maximum_absolute": source_rois["all"]["maximum_absolute"],
        "primary_core_rms": primary_rms,
        "image_core_rms": image_rms,
        "primary_image_core_rms_ratio": primary_image_ratio,
        "junction_rms": source_rois["junction"]["rms"],
        "second_wire_rms": source_rois["second_wire"]["rms"],
        "far_arm_rms": source_rois["far_arm"]["rms"],
        "quantile_upper_bounds_over_width": {
            profile_name: {
                quantile_name: _quantile_upper(
                    analysis,
                    field_name="conditioned_t100_source",
                    profile_name=profile_name,
                    quantile_name=quantile_name,
                )
                for quantile_name in ("w50", "w90")
            }
            for profile_name in ("rho_all", "plane_all", "surface_all")
        },
    }
    noisy_summary = _load_json(NOISY_T1_SUMMARY_PATH)
    clean_summary = _load_json(CLEAN_T1_SUMMARY_PATH)
    prior_scalars = {
        "noisy_t1": _prior_source_scalars(
            noisy_summary,
            field_name="total_source",
        ),
        "clean_binary_t1": _prior_source_scalars(
            clean_summary,
            field_name="total_clean_source",
        ),
    }
    comparisons: dict[str, Any] = {
        "comparison_is_scalar_only_not_field_subtraction": True,
        "current_t100": current_scalars,
        **prior_scalars,
        "t100_over_prior_sum_squared": {
            prior_name: {
                name: _safe_ratio(
                    current_scalars["sum_squared"][name],
                    prior["sum_squared"][name],
                )
                for name in current_scalars["sum_squared"]
            }
            for prior_name, prior in prior_scalars.items()
        },
        "t100_over_prior_rms": {
            prior_name: _safe_ratio(
                current_scalars["rms"],
                prior["rms"],
            )
            for prior_name, prior in prior_scalars.items()
        },
    }

    center_y = FROZEN_DEG90.lattice.shape[1] // 2
    q_slice = np.asarray(source[:, center_y, :], dtype=np.float32)
    x = centered_coordinates(FROZEN_DEG90.lattice, 0)
    z = centered_coordinates(FROZEN_DEG90.lattice, 2)
    surface_distance_slice = np.abs(np.abs(x) - radius)
    dp = _periodic_distance(
        z,
        profile.primary_coordinate,
        FROZEN_DEG90.lattice.physical_lengths[2],
    )
    di = _periodic_distance(
        z,
        profile.image_coordinate,
        FROZEN_DEG90.lattice.physical_lengths[2],
    )
    rho_slice = np.hypot(
        surface_distance_slice[:, None],
        np.minimum(dp, di)[None, :],
    )
    flat_index_slice = (
        (
            np.arange(FROZEN_DEG90.lattice.shape[0])[:, None]
            * FROZEN_DEG90.lattice.shape[1]
            + center_y
        )
        * FROZEN_DEG90.lattice.shape[2]
        + np.arange(FROZEN_DEG90.lattice.shape[2])[None, :]
    )
    prefix_slice = (
        flat_index_slice < EXPECTED_RELEASED_PREFIX_LENGTH
    )
    intended_slice = ~prefix_slice & (
        rho_slice <= 2.0 * mapped.surface_width
    )
    residual_slice = ~(prefix_slice | intended_slice)
    slices = {
        "checkpoint_t100_y_center": np.asarray(
            checkpoint[:, center_y, :],
            dtype=np.float32,
        ),
        "total_t100_gb_source_y_center": q_slice,
        "former_prefix_region_source_y_center": np.where(
            prefix_slice,
            q_slice,
            np.float32(0.0),
        ),
        "intended_intersection_source_y_center": np.where(
            intended_slice,
            q_slice,
            np.float32(0.0),
        ),
        "residual_source_y_center": np.where(
            residual_slice,
            q_slice,
            np.float32(0.0),
        ),
    }
    _atomic_npz(output_directory / "diagnostic-slices.npz", slices)
    compact_slices_sha256 = sha256_path(
        output_directory / "diagnostic-slices.npz"
    )
    elapsed_seconds = float(time.perf_counter() - started)
    hard_checks["wall_cap_not_reached"] = (
        elapsed_seconds <= WALL_SECONDS_CAP
    )
    classification, passed = classify(
        hard_checks=hard_checks,
        state_checks=state_contract["checks"],
        morphology_checks=morphology["checks"],
        core_resolution_checks=core_resolution_checks,
        source_checks=source_checks,
        unresolved_failed_checks=unresolved_failed_checks,
    )
    summary = {
        "schema_version": 1,
        "classification": classification,
        "preflight_passed": passed,
        "scope_boundary": scope,
        "contract": contract,
        "execution": {
            "elapsed_seconds": elapsed_seconds,
            "peak_rss_bytes": peak_rss_bytes(),
            "solver_instances_constructed": 0,
            "propose_step_calls": 0,
            "fft_calls": 0,
            "accepted_trajectory_steps": 0,
        },
        "checkpoint": {
            "path": str(T100_CHECKPOINT_PATH),
            "metadata_path": str(T100_CHECKPOINT_METADATA_PATH),
            "file_state_before": checkpoint_file_before,
            "file_state_after": checkpoint_file_after,
            "fingerprint_before": checkpoint_fingerprint_before,
            "fingerprint_after": checkpoint_fingerprint_after,
            "mmap_writeable": bool(checkpoint.flags.writeable),
            "loaded_checks": checkpoint_checks,
            "stored_t100_record": state_contract["record"],
            "stored_t100_record_checks": state_contract["checks"],
        },
        "geometry": {
            "physical_radius": radius,
            "surface_width": mapped.surface_width,
            "gb_axis": GB_AXIS,
            "primary_coordinate": profile.primary_coordinate,
            "periodic_image_coordinate": profile.image_coordinate,
            "per_plane_energy_audit": plane_energy,
        },
        "morphology_preflight": morphology,
        "source": {
            "definition": "q_GB(c100) = N'(c100) * g_GB",
            "source_only_no_response_field": True,
            "former_released_prefix_is_spatial_region_only": True,
            "partitions": partitions,
            "primary_image_core_rms_ratio": primary_image_ratio,
            "representation_by_roi": representation_by_roi,
            "anatomical_partitions": analysis[
                "anatomical_partitions"
            ]["conditioned_t100_source"],
            "rois": source_rois,
            "distance_definition": analysis["distance_definition"],
            "profiles": analysis["profiles"][
                "conditioned_t100_source"
            ],
        },
        "comparisons_to_t1": comparisons,
        "checks": {
            "hard_integrity": hard_checks,
            "stored_state_suitability": state_contract["checks"],
            "analytic_roi_morphology": morphology["checks"],
            "core_representation_resolution": core_resolution_checks,
            "source_localization": source_checks,
            "unresolved_failed_source_checks": (
                unresolved_failed_checks
            ),
        },
        "interpretation_boundary": {
            "static_source_compatibility_only": True,
            "no_solver_response_measured": True,
            "no_operator_or_timestep_diagnosis": True,
            "t100_is_not_equilibrium": True,
            "t100_is_not_initializer_independent": True,
            "released_noise_is_not_validated_as_physical": True,
            "w50_w90_are_descriptive_only": True,
            "far_arm_and_phase_partitions_are_descriptive_only": True,
            "result_does_not_authorize_initializer_change": True,
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
        },
        "outputs": {
            "compact_slices": "diagnostic-slices.npz",
            "compact_slices_sha256": compact_slices_sha256,
            "full_field_checkpoint_saved": False,
            "restart_possible": False,
        },
    }
    _atomic_json(output_directory / "summary.json", summary)
    _atomic_json(
        output_directory / "run_status.json",
        {
            "status": "completed",
            "classification": classification,
            "preflight_passed": passed,
            "elapsed_seconds": elapsed_seconds,
            "propose_step_calls": 0,
            "fft_calls": 0,
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
                "elapsed_seconds": summary["execution"][
                    "elapsed_seconds"
                ],
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
