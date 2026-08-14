#!/usr/bin/env python3
"""Run one bounded full-grid Roy crossed-initializer validation.

The runner uses one passed equilibrium R=6 cross-section, constructs two
smooth fuzzy unions of the same tangent cylinders, and evolves GB-off/GB-on
pairs from byte-identical states for exactly 100 Roy steps. It has no retry,
parameter sweep, continuation, sentinel, or report-writing capability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from ..roy_2021_reproduction.model import (
    PeriodicLattice,
    RoyModelParameters,
    RoyPseudospectralSolver,
)
from ..roy_fixed_gb_bridge.model import (
    FixedGrainBoundaryParameters,
    RoyFixedGrainBoundarySolver,
    build_periodic_bicrystal_profile,
)


SOURCE_PATH = Path(__file__).resolve()
DIRECTORY = SOURCE_PATH.parent
CONTRACT_PATH = DIRECTORY / "INITIALIZER_VALIDATION_CONTRACT.md"
DEFAULT_OUTPUT = DIRECTORY / "results" / "full_grid_validation"
SOURCE_PROBE = (
    DIRECTORY.parent
    / "roy_isolated_gb_ridge"
    / "results"
    / "production_ratio_probe"
)
SOURCE_SUMMARY = SOURCE_PROBE / "summary.json"
SOURCE_ARCHIVE = SOURCE_PROBE / "initializer.npz"
EXPECTED_SOURCE_SHA256 = (
    "6c496a189c20ae07844869a0db781a6bfa532bf7c7dd73518fd3d1a177fe711a"
)
EXPECTED_PROFILE_FINGERPRINT = (
    "b80e24245e358c9c9ffc3aed1d9c3f752809a434c84c0ac3fdd5b301d466e156"
)

SHAPE = (96, 768, 768)
SPACING = 0.5
RADIUS = 6.0
RADIUS_CELLS = 12
TIMESTEP = 1.0
ENERGY_RATIO = 0.35
GB_AXIS = 2
GB_OFFSET_OVER_RADIUS = 2.6
FFT_WORKERS = 12
STEPS = 100
SAMPLE_STEPS = (0, 1, 10, 25, 50, 100)
PROJECTION_MARGIN = 1.5
MAXIMUM_PROJECTED_WALL_SECONDS = 3600.0
WALL_SECONDS_CAP = 3600.0
FIELD_MINIMUM = -0.08
FIELD_MAXIMUM = 1.08
MAXIMUM_RELATIVE_MASS_DRIFT = 1.0e-4
MAXIMUM_RELATIVE_ENERGY_REBOUND = 1.0e-6
MINIMUM_JUNCTION_DIFFERENCE_FRACTION = 0.99
MINIMUM_INITIAL_CONTACT_RADIUS_DIFFERENCE_OVER_R = 0.05
MINIMUM_T1_GB_BAND_RESPONSE_FRACTION = 0.95
MAXIMUM_T1_JUNCTION_TO_GB_RMS = 0.25
MAXIMUM_ARM_RMS_DIFFERENCE_OVER_R = 0.01
MAXIMUM_ARM_LINF_DIFFERENCE_OVER_R = 0.05
MAXIMUM_CONTACT_RADIUS_DIFFERENCE_OVER_R = 0.05
MAXIMUM_CONTACT_EXCESS_RELATIVE_DIFFERENCE = 0.10
MAXIMUM_CENTRAL_RMS_GROWTH_FACTOR = 1.10
MINIMUM_GB_RESPONSE_COSINE = 0.95
MAXIMUM_GB_RESPONSE_RELATIVE_L2_DIFFERENCE = 0.15
MINIMUM_AVAILABLE_MEMORY_BYTES = 24 * 1024**3
MINIMUM_DISK_BYTES = 8 * 1024**3
CONTACT_THRESHOLD = 0.60
JUNCTION_HALF_WIDTH_X = 2.0 * np.sqrt(8.0)
JUNCTION_HALF_WIDTH_AXIAL = 2.0 * RADIUS + 2.0 * np.sqrt(8.0)
ARM_EXCLUSION_HALF_WIDTH = 2.0 * RADIUS + np.sqrt(8.0)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def _array_fingerprint(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
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


def _atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _reserve_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)


def centered_coordinates(cells: int, spacing: float = SPACING) -> np.ndarray:
    """Coordinates centered on the half-cell symmetry plane."""

    return (
        np.arange(cells, dtype=np.float64)
        - 0.5 * (cells - 1)
    ) * spacing


def embed_profile(
    profile: np.ndarray,
    target_shape: tuple[int, int],
    *,
    row_shift: int,
    column_shift: int,
) -> np.ndarray:
    """Embed a compact 2-D profile with integer shifts and zero exterior."""

    source = np.asarray(profile, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError("profile must be two-dimensional")
    target = np.zeros(target_shape, dtype=np.float64)

    def overlap(
        source_cells: int,
        target_cells: int,
        shift: int,
    ) -> tuple[slice, slice]:
        target_start = max(0, shift)
        target_stop = min(target_cells, shift + source_cells)
        if target_stop <= target_start:
            raise ValueError("profile does not overlap target")
        source_start = target_start - shift
        source_stop = source_start + (target_stop - target_start)
        return (
            slice(target_start, target_stop),
            slice(source_start, source_stop),
        )

    target_rows, source_rows = overlap(
        source.shape[0], target_shape[0], row_shift
    )
    target_columns, source_columns = overlap(
        source.shape[1], target_shape[1], column_shift
    )
    target[target_rows, target_columns] = source[
        source_rows, source_columns
    ]
    return target


def product_union(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Probabilistic product t-conorm."""

    return q1 + q2 - q1 * q2


def additive_odds_union(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Smooth union obtained by adding the two phase odds."""

    numerator = q1 + q2 - 2.0 * q1 * q2
    denominator = 1.0 - q1 * q2
    result = np.ones(np.broadcast_shapes(q1.shape, q2.shape), dtype=np.float64)
    np.divide(
        numerator,
        denominator,
        out=result,
        where=np.abs(denominator) > 32.0 * np.finfo(float).eps,
    )
    return result


def load_source_profile() -> tuple[np.ndarray, float, float, dict[str, Any]]:
    """Load and verify the passed production-ratio equilibrium profile."""

    if _sha256_path(SOURCE_ARCHIVE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("equilibrium initializer archive hash changed")
    with SOURCE_SUMMARY.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if (
        summary.get("classification") != "ridge_probe_passed"
        or summary.get("probe_passed") is not True
    ):
        raise RuntimeError("source ridge probe is not passed")
    with np.load(SOURCE_ARCHIVE, allow_pickle=False) as archive:
        profile = np.array(archive["cross_section"], dtype=np.float64)
    if profile.shape != (96, 96):
        raise RuntimeError(f"unexpected source profile shape: {profile.shape}")
    if _array_fingerprint(profile) != EXPECTED_PROFILE_FINGERPRINT:
        raise RuntimeError("equilibrium profile fingerprint changed")
    initializer = summary["initializer"]
    vapor = float(initializer["homogeneous_vapor"])
    solid = float(initializer["homogeneous_solid"])
    if not solid > vapor:
        raise RuntimeError("source homogeneous phases are invalid")
    q_raw = (profile - vapor) / (solid - vapor)
    clipping = {
        "raw_minimum_q": float(np.min(q_raw)),
        "raw_maximum_q": float(np.max(q_raw)),
        "maximum_below_zero": max(0.0, -float(np.min(q_raw))),
        "maximum_above_one": max(0.0, float(np.max(q_raw)) - 1.0),
        "clipped_cell_count": int(np.count_nonzero((q_raw < 0.0) | (q_raw > 1.0))),
    }
    if (
        clipping["maximum_below_zero"] > 1.0e-6
        or clipping["maximum_above_one"] > 1.0e-6
    ):
        raise RuntimeError(f"source phase normalization is unsafe: {clipping}")
    q = np.clip(q_raw, 0.0, 1.0)
    return q, vapor, solid, clipping


def build_wire_templates(
    source_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build the two tangent equilibrium-cylinder templates."""

    nx, ny, nz = SHAPE
    if source_q.shape != (nx, nx):
        raise ValueError("source profile does not match thin lattice dimension")
    half_separation_cells = RADIUS_CELLS
    first_xy = embed_profile(
        source_q,
        (nx, ny),
        row_shift=-half_separation_cells,
        column_shift=(ny - source_q.shape[1]) // 2,
    )
    second_xz = embed_profile(
        source_q,
        (nx, nz),
        row_shift=half_separation_cells,
        column_shift=(nz - source_q.shape[1]) // 2,
    )
    source_center = 0.5 * (source_q.shape[0] - 1)
    first_center = source_center - half_separation_cells
    second_center = source_center + half_separation_cells
    metadata = {
        "first_wire_axis": 2,
        "second_wire_axis": 1,
        "first_center_x_index": first_center,
        "second_center_x_index": second_center,
        "contact_midplane_x_index": 0.5 * (first_center + second_center),
        "center_separation_cells": second_center - first_center,
        "center_separation": (
            second_center - first_center
        ) * SPACING,
        "first_template_fingerprint": _array_fingerprint(first_xy),
        "second_template_fingerprint": _array_fingerprint(second_xz),
    }
    return first_xy, second_xz, metadata


def construct_union_field(
    first_xy: np.ndarray,
    second_xz: np.ndarray,
    *,
    union: str,
    vapor: float,
    solid: float,
    chunk_depth: int = 16,
) -> np.ndarray:
    """Construct one full field without allocating full q1 and q2 arrays."""

    if first_xy.shape != SHAPE[:2] or second_xz.shape != (
        SHAPE[0],
        SHAPE[2],
    ):
        raise ValueError("wire templates do not match the production shape")
    if union not in {"product", "additive_odds"}:
        raise ValueError(f"unknown union: {union}")
    result = np.empty(SHAPE, dtype=np.float64)
    q1 = first_xy[:, :, None]
    span = solid - vapor
    for start in range(0, SHAPE[2], chunk_depth):
        stop = min(start + chunk_depth, SHAPE[2])
        q2 = second_xz[:, None, start:stop]
        if union == "product":
            q = product_union(q1, q2)
        else:
            q = additive_odds_union(q1, q2)
        result[:, :, start:stop] = vapor + span * q
    return result


def _phase(field: np.ndarray, vapor: float, solid: float) -> np.ndarray:
    return (field - vapor) / (solid - vapor)


def contact_metrics(
    field: np.ndarray,
    first_xy: np.ndarray,
    second_xz: np.ndarray,
    vapor: float,
    solid: float,
) -> dict[str, float]:
    """Measure the central high-phase contact core and bridge excess."""

    q = _phase(field, vapor, solid)
    midpoint = 0.5 * (
        0.5 * (SOURCE_PROFILE_CELLS - 1) - RADIUS_CELLS
        + 0.5 * (SOURCE_PROFILE_CELLS - 1) + RADIUS_CELLS
    )
    lower = int(np.floor(midpoint))
    upper = int(np.ceil(midpoint))
    plane = 0.5 * (q[lower] + q[upper])
    hard = 0.5 * (
        np.maximum(
            first_xy[lower, :, None],
            second_xz[lower, None, :],
        )
        + np.maximum(
            first_xy[upper, :, None],
            second_xz[upper, None, :],
        )
    )
    y = centered_coordinates(SHAPE[1])
    z = centered_coordinates(SHAPE[2])
    central = (
        (np.abs(y)[:, None] <= JUNCTION_HALF_WIDTH_AXIAL)
        & (np.abs(z)[None, :] <= JUNCTION_HALF_WIDTH_AXIAL)
    )
    high = (plane >= CONTACT_THRESHOLD) & central
    area = float(np.count_nonzero(high) * SPACING**2)
    bridge_excess = float(
        np.sum(np.maximum(plane - hard, 0.0)[central], dtype=np.float64)
        * SPACING**2
    )
    return {
        "threshold": CONTACT_THRESHOLD,
        "high_phase_area": area,
        "high_phase_equivalent_radius": float(
            np.sqrt(max(area, 0.0) / np.pi)
        ),
        "bridge_excess_integral": bridge_excess,
        "central_plane_maximum_q": float(np.max(plane[central])),
    }


SOURCE_PROFILE_CELLS = 96


def arm_radius_profile(
    field: np.ndarray,
    vapor: float,
    solid: float,
) -> np.ndarray:
    """Composition-area radius of the first (z-directed) arm."""

    q = np.clip(_phase(field, vapor, solid), 0.0, 1.0)
    area = np.sum(q, axis=(0, 1), dtype=np.float64) * SPACING**2
    return np.sqrt(np.maximum(area, 0.0) / np.pi)


def field_health(
    field: np.ndarray,
    *,
    initial_mass: float,
    lattice: PeriodicLattice,
) -> dict[str, Any]:
    finite = bool(np.all(np.isfinite(field)))
    minimum = float(np.min(field)) if finite else None
    maximum = float(np.max(field)) if finite else None
    mass = (
        float(np.sum(field, dtype=np.float64) * lattice.cell_volume)
        if finite
        else None
    )
    drift = (
        abs(mass - initial_mass) / max(abs(initial_mass), np.finfo(float).tiny)
        if mass is not None
        else None
    )
    return {
        "finite": finite,
        "minimum": minimum,
        "maximum": maximum,
        "mass": mass,
        "relative_mass_drift": drift,
        "within_bounds": bool(
            finite
            and minimum is not None
            and maximum is not None
            and minimum >= FIELD_MINIMUM
            and maximum <= FIELD_MAXIMUM
        ),
        "mass_healthy": bool(
            drift is not None and drift <= MAXIMUM_RELATIVE_MASS_DRIFT
        ),
    }


def junction_difference_localization(
    first: np.ndarray,
    second: np.ndarray,
) -> dict[str, float]:
    """Squared-norm localization of the initializer difference."""

    difference = first - second
    x = centered_coordinates(SHAPE[0])
    y = centered_coordinates(SHAPE[1])
    z = centered_coordinates(SHAPE[2])
    central = (
        (np.abs(x)[:, None, None] <= JUNCTION_HALF_WIDTH_X)
        & (np.abs(y)[None, :, None] <= JUNCTION_HALF_WIDTH_AXIAL)
        & (np.abs(z)[None, None, :] <= JUNCTION_HALF_WIDTH_AXIAL)
    )
    total = float(np.sum(difference**2, dtype=np.float64))
    inside = float(np.sum(difference[central] ** 2, dtype=np.float64))
    return {
        "total_squared_norm": total,
        "junction_squared_norm": inside,
        "junction_fraction": inside / total if total > 0.0 else 0.0,
        "central_rms": float(
            np.sqrt(np.mean(difference[central] ** 2, dtype=np.float64))
        ),
    }


def gb_response_localization(
    control: np.ndarray,
    gb: np.ndarray,
    grain_density: np.ndarray,
) -> dict[str, float]:
    """Axial localization and protected-junction response at one step."""

    difference = gb - control
    support_1d = grain_density >= 0.01 * float(np.max(grain_density))
    support = np.broadcast_to(
        support_1d.reshape((1, 1, SHAPE[2])),
        SHAPE,
    )
    z = centered_coordinates(SHAPE[2])
    junction_1d = np.abs(z) <= RADIUS
    gb_values = difference[support]
    junction_values = difference[:, :, junction_1d]
    total = float(np.sum(difference**2, dtype=np.float64))
    supported = float(np.sum(gb_values**2, dtype=np.float64))
    gb_rms = float(np.sqrt(np.mean(gb_values**2, dtype=np.float64)))
    junction_rms = float(
        np.sqrt(np.mean(junction_values**2, dtype=np.float64))
    )
    return {
        "total_squared_norm": total,
        "gb_band_squared_norm": supported,
        "gb_band_squared_norm_fraction": (
            supported / total if total > 0.0 else 0.0
        ),
        "gb_band_rms": gb_rms,
        "junction_rms": junction_rms,
        "junction_to_gb_rms_ratio": (
            junction_rms / gb_rms if gb_rms > 0.0 else float("inf")
        ),
    }


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    numerator = float(np.dot(first, second))
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return numerator / denominator if denominator > 0.0 else 0.0


def endpoint_agreement(
    states: dict[str, np.ndarray],
    first_xy: np.ndarray,
    second_xz: np.ndarray,
    vapor: float,
    solid: float,
    initial_metrics: dict[str, dict[str, float]],
) -> tuple[dict[str, Any], dict[str, bool], dict[str, np.ndarray]]:
    """Compare morphology and the differential GB response at t=100."""

    profiles = {
        name: arm_radius_profile(field, vapor, solid)
        for name, field in states.items()
    }
    z = centered_coordinates(SHAPE[2])
    arm = np.abs(z) >= ARM_EXCLUSION_HALF_WIDTH
    off_difference = (
        profiles["product_off"][arm] - profiles["odds_off"][arm]
    )
    contact = {
        name: contact_metrics(
            field, first_xy, second_xz, vapor, solid
        )
        for name, field in states.items()
    }
    product_contact = contact["product_off"]
    odds_contact = contact["odds_off"]
    contact_radius_difference = abs(
        product_contact["high_phase_equivalent_radius"]
        - odds_contact["high_phase_equivalent_radius"]
    )
    contact_excess_difference = abs(
        product_contact["bridge_excess_integral"]
        - odds_contact["bridge_excess_integral"]
    )
    contact_excess_scale = max(
        abs(product_contact["bridge_excess_integral"]),
        abs(odds_contact["bridge_excess_integral"]),
        np.finfo(float).tiny,
    )
    final_localization = junction_difference_localization(
        states["product_off"], states["odds_off"]
    )
    initial_central_rms = float(
        initial_metrics["difference_localization"]["central_rms"]
    )
    final_central_rms = float(final_localization["central_rms"])
    product_response = (
        profiles["product_on"] - profiles["product_off"]
    )[arm]
    odds_response = (
        profiles["odds_on"] - profiles["odds_off"]
    )[arm]
    response_scale = max(
        float(np.linalg.norm(product_response)),
        float(np.linalg.norm(odds_response)),
        np.finfo(float).tiny,
    )
    metrics = {
        "arm_coordinate_count": int(np.count_nonzero(arm)),
        "off_arm_rms_difference": float(
            np.sqrt(np.mean(off_difference**2, dtype=np.float64))
        ),
        "off_arm_rms_difference_over_R": float(
            np.sqrt(np.mean(off_difference**2, dtype=np.float64)) / RADIUS
        ),
        "off_arm_linf_difference": float(np.max(np.abs(off_difference))),
        "off_arm_linf_difference_over_R": float(
            np.max(np.abs(off_difference)) / RADIUS
        ),
        "contact": contact,
        "off_contact_radius_difference": contact_radius_difference,
        "off_contact_radius_difference_over_R": (
            contact_radius_difference / RADIUS
        ),
        "off_contact_excess_relative_difference": (
            contact_excess_difference / contact_excess_scale
        ),
        "initial_central_rms_difference": initial_central_rms,
        "final_central_rms_difference": final_central_rms,
        "central_rms_growth_factor": (
            final_central_rms / initial_central_rms
            if initial_central_rms > 0.0
            else float("inf")
        ),
        "gb_radius_response_cosine": _cosine(
            product_response, odds_response
        ),
        "gb_radius_response_relative_l2_difference": float(
            np.linalg.norm(product_response - odds_response)
            / response_scale
        ),
        "gb_product_response_l2": float(np.linalg.norm(product_response)),
        "gb_odds_response_l2": float(np.linalg.norm(odds_response)),
    }
    checks = {
        "off_arm_rms_agreement": (
            metrics["off_arm_rms_difference_over_R"]
            <= MAXIMUM_ARM_RMS_DIFFERENCE_OVER_R
        ),
        "off_arm_linf_agreement": (
            metrics["off_arm_linf_difference_over_R"]
            <= MAXIMUM_ARM_LINF_DIFFERENCE_OVER_R
        ),
        "off_contact_radius_agreement": (
            metrics["off_contact_radius_difference_over_R"]
            <= MAXIMUM_CONTACT_RADIUS_DIFFERENCE_OVER_R
        ),
        "off_contact_excess_agreement": (
            metrics["off_contact_excess_relative_difference"]
            <= MAXIMUM_CONTACT_EXCESS_RELATIVE_DIFFERENCE
        ),
        "central_initializer_difference_did_not_grow": (
            metrics["central_rms_growth_factor"]
            <= MAXIMUM_CENTRAL_RMS_GROWTH_FACTOR
        ),
        "gb_radius_response_direction_agrees": (
            metrics["gb_radius_response_cosine"]
            >= MINIMUM_GB_RESPONSE_COSINE
        ),
        "gb_radius_response_magnitude_agrees": (
            metrics["gb_radius_response_relative_l2_difference"]
            <= MAXIMUM_GB_RESPONSE_RELATIVE_L2_DIFFERENCE
        ),
    }
    return metrics, checks, profiles


def classify(
    health_checks: dict[str, bool],
    validity_checks: dict[str, bool],
    localization_checks: dict[str, bool],
    agreement_checks: dict[str, bool],
) -> str:
    """Apply the preregistered decision order."""

    if not all(health_checks.values()):
        return "crossed_initializer_validation_numerically_unhealthy"
    if not all(validity_checks.values()):
        return "crossed_initializer_validation_inconclusive_invalid_sensitivity"
    if not all(localization_checks.values()):
        return "crossed_initializer_validation_inconclusive_gb_startup_not_localized"
    if not all(agreement_checks.values()):
        return "crossed_initializer_validation_failed_dependence_persists"
    return "crossed_initializer_cleared_for_bounded_startup"


def _contract_payload(profile: Any) -> dict[str, Any]:
    return {
        "shape": SHAPE,
        "spacing": SPACING,
        "radius": RADIUS,
        "surface_width": profile.parameters.surface_width,
        "radius_over_width": RADIUS / profile.parameters.surface_width,
        "timestep": TIMESTEP,
        "steps": STEPS,
        "sample_steps": SAMPLE_STEPS,
        "fft_workers": FFT_WORKERS,
        "mobility": "sqrt(abs(c-c^2))",
        "noise": None,
        "deterministic_mode": None,
        "constant_mobility_preprocessing": False,
        "gb_active_from_t0": True,
        "energy_ratio": ENERGY_RATIO,
        "gb_axis": GB_AXIS,
        "gb_offset_over_radius": GB_OFFSET_OVER_RADIUS,
        "gb_primary_coordinate": profile.primary_coordinate,
        "gb_image_coordinate": profile.image_coordinate,
        "unions": {
            "product": "q1 + q2 - q1*q2",
            "additive_odds": (
                "(q1 + q2 - 2*q1*q2)/(1 - q1*q2)"
            ),
        },
        "limits": {
            "field": [FIELD_MINIMUM, FIELD_MAXIMUM],
            "relative_mass_drift": MAXIMUM_RELATIVE_MASS_DRIFT,
            "relative_energy_rebound": MAXIMUM_RELATIVE_ENERGY_REBOUND,
            "projected_wall_seconds": MAXIMUM_PROJECTED_WALL_SECONDS,
            "wall_seconds": WALL_SECONDS_CAP,
            "junction_difference_fraction": (
                MINIMUM_JUNCTION_DIFFERENCE_FRACTION
            ),
            "initial_contact_radius_difference_over_R": (
                MINIMUM_INITIAL_CONTACT_RADIUS_DIFFERENCE_OVER_R
            ),
            "t1_gb_band_response_fraction": (
                MINIMUM_T1_GB_BAND_RESPONSE_FRACTION
            ),
            "t1_junction_to_gb_rms": MAXIMUM_T1_JUNCTION_TO_GB_RMS,
            "arm_rms_difference_over_R": (
                MAXIMUM_ARM_RMS_DIFFERENCE_OVER_R
            ),
            "arm_linf_difference_over_R": (
                MAXIMUM_ARM_LINF_DIFFERENCE_OVER_R
            ),
            "contact_radius_difference_over_R": (
                MAXIMUM_CONTACT_RADIUS_DIFFERENCE_OVER_R
            ),
            "contact_excess_relative_difference": (
                MAXIMUM_CONTACT_EXCESS_RELATIVE_DIFFERENCE
            ),
            "central_rms_growth_factor": (
                MAXIMUM_CENTRAL_RMS_GROWTH_FACTOR
            ),
            "gb_response_cosine": MINIMUM_GB_RESPONSE_COSINE,
            "gb_response_relative_l2_difference": (
                MAXIMUM_GB_RESPONSE_RELATIVE_L2_DIFFERENCE
            ),
        },
    }


def run(output: Path) -> dict[str, Any]:
    """Execute the discarded timing probe and, if safe, the fixed run."""

    _reserve_output(output)
    started = time.perf_counter()
    lattice = PeriodicLattice(SHAPE, SPACING)
    parameters = RoyModelParameters(timestep=TIMESTEP)
    mapped = FixedGrainBoundaryParameters.matched_to_roy(
        parameters, energy_ratio=ENERGY_RATIO
    )
    profile = build_periodic_bicrystal_profile(
        lattice,
        mapped,
        axis=GB_AXIS,
        primary_coordinate=GB_OFFSET_OVER_RADIUS * RADIUS,
    )
    contract = _contract_payload(profile)
    resource_report = {
        "available_memory_bytes": int(psutil.virtual_memory().available),
        "physical_memory_bytes": int(psutil.virtual_memory().total),
        "free_disk_bytes": int(shutil.disk_usage(output.parent).free),
        "one_field_bytes": int(
            np.prod(SHAPE, dtype=np.int64) * np.dtype(np.float64).itemsize
        ),
    }
    resource_checks = {
        "available_memory_sufficient": (
            resource_report["available_memory_bytes"]
            >= MINIMUM_AVAILABLE_MEMORY_BYTES
        ),
        "disk_sufficient": (
            resource_report["free_disk_bytes"] >= MINIMUM_DISK_BYTES
        ),
    }
    if not all(resource_checks.values()):
        summary = {
            "status": "stopped",
            "classification": "crossed_initializer_validation_resource_stop",
            "contract": contract,
            "resources": {
                "report": resource_report,
                "checks": resource_checks,
            },
            "automatic_follow_on_authorized": False,
        }
        _atomic_json(output / "summary.json", summary)
        return summary

    source_q, vapor, solid, clipping = load_source_profile()
    first_xy, second_xz, geometry = build_wire_templates(source_q)
    product = construct_union_field(
        first_xy,
        second_xz,
        union="product",
        vapor=vapor,
        solid=solid,
    )
    odds = construct_union_field(
        first_xy,
        second_xz,
        union="additive_odds",
        vapor=vapor,
        solid=solid,
    )
    initial_contact = {
        "product": contact_metrics(
            product, first_xy, second_xz, vapor, solid
        ),
        "additive_odds": contact_metrics(
            odds, first_xy, second_xz, vapor, solid
        ),
    }
    initial_difference = junction_difference_localization(product, odds)
    contact_radius_difference = abs(
        initial_contact["product"]["high_phase_equivalent_radius"]
        - initial_contact["additive_odds"]["high_phase_equivalent_radius"]
    )
    initial_metrics: dict[str, Any] = {
        "contact": initial_contact,
        "contact_radius_difference": contact_radius_difference,
        "contact_radius_difference_over_R": (
            contact_radius_difference / RADIUS
        ),
        "difference_localization": initial_difference,
        "product_mass": float(
            np.sum(product, dtype=np.float64) * lattice.cell_volume
        ),
        "additive_odds_mass": float(
            np.sum(odds, dtype=np.float64) * lattice.cell_volume
        ),
        "relative_mass_difference": abs(
            float(np.sum(product - odds, dtype=np.float64))
            * lattice.cell_volume
        )
        / max(
            abs(float(np.sum(product, dtype=np.float64) * lattice.cell_volume)),
            np.finfo(float).tiny,
        ),
    }
    validity_checks = {
        "initializer_difference_is_junction_localized": (
            initial_difference["junction_fraction"]
            >= MINIMUM_JUNCTION_DIFFERENCE_FRACTION
        ),
        "initializer_contact_radii_are_meaningfully_distinct": (
            initial_metrics["contact_radius_difference_over_R"]
            >= MINIMUM_INITIAL_CONTACT_RADIUS_DIFFERENCE_OVER_R
        ),
        "product_field_is_finite": bool(np.all(np.isfinite(product))),
        "additive_odds_field_is_finite": bool(np.all(np.isfinite(odds))),
    }

    control_solver = RoyPseudospectralSolver(
        lattice, parameters, fft_workers=FFT_WORKERS
    )
    gb_solver = RoyFixedGrainBoundarySolver(
        lattice, parameters, profile, fft_workers=FFT_WORKERS
    )
    probe_started = time.perf_counter()
    product_control_probe = control_solver.propose_step(product)
    control_seconds = time.perf_counter() - probe_started
    probe_started = time.perf_counter()
    product_gb_probe = gb_solver.propose_step(product)
    gb_seconds = time.perf_counter() - probe_started
    projected_seconds = (
        PROJECTION_MARGIN
        * 2.0
        * STEPS
        * (control_seconds + gb_seconds)
    )
    probe = {
        "discarded_control_seconds": control_seconds,
        "discarded_gb_seconds": gb_seconds,
        "projected_four_path_seconds_with_margin": projected_seconds,
        "projection_margin": PROJECTION_MARGIN,
        "within_one_hour_cap": (
            projected_seconds <= MAXIMUM_PROJECTED_WALL_SECONDS
        ),
        "discarded_proposals": 2,
        "accepted_steps": 0,
    }
    del product_control_probe, product_gb_probe
    if projected_seconds > MAXIMUM_PROJECTED_WALL_SECONDS:
        summary = {
            "status": "stopped",
            "classification": (
                "crossed_initializer_validation_projected_over_one_hour"
            ),
            "contract": contract,
            "resources": {
                "report": resource_report,
                "checks": resource_checks,
            },
            "source": {
                "archive_sha256": EXPECTED_SOURCE_SHA256,
                "profile_fingerprint": EXPECTED_PROFILE_FINGERPRINT,
                "normalization": clipping,
            },
            "geometry": geometry,
            "initial_metrics": initial_metrics,
            "validity_checks": validity_checks,
            "timing_probe": probe,
            "automatic_follow_on_authorized": False,
        }
        _atomic_json(output / "summary.json", summary)
        return summary

    states = {
        "product_off": np.array(product, copy=True),
        "product_on": np.array(product, copy=True),
        "odds_off": np.array(odds, copy=True),
        "odds_on": np.array(odds, copy=True),
    }
    initial_fingerprints = {
        name: _array_fingerprint(field)
        for name, field in states.items()
    }
    pair_identity_checks = {
        "product_pair_identical": (
            initial_fingerprints["product_off"]
            == initial_fingerprints["product_on"]
        ),
        "odds_pair_identical": (
            initial_fingerprints["odds_off"]
            == initial_fingerprints["odds_on"]
        ),
    }
    solvers = {
        "product_off": control_solver,
        "product_on": gb_solver,
        "odds_off": control_solver,
        "odds_on": gb_solver,
    }
    initial_masses = {
        name: float(
            np.sum(field, dtype=np.float64) * lattice.cell_volume
        )
        for name, field in states.items()
    }
    previous_energies: dict[str, float] = {}
    initial_energies: dict[str, float] = {}
    maximum_rebounds = {name: 0.0 for name in states}
    history: list[dict[str, Any]] = []
    t1_localization: dict[str, dict[str, float]] = {}
    trajectory_stopped_early = False
    stop_reason: str | None = None

    def record(step: int) -> bool:
        healthy = True
        for name, field in states.items():
            solver = solvers[name]
            energy = solver.free_energy(field)
            if name not in initial_energies:
                initial_energies[name] = energy
                previous_energies[name] = energy
            rebound = (
                energy - previous_energies[name]
            ) / max(abs(initial_energies[name]), np.finfo(float).tiny)
            maximum_rebounds[name] = max(maximum_rebounds[name], rebound)
            previous_energies[name] = energy
            health = field_health(
                field,
                initial_mass=initial_masses[name],
                lattice=lattice,
            )
            contact = contact_metrics(
                field, first_xy, second_xz, vapor, solid
            )
            history.append(
                {
                    "step": step,
                    "time": step * TIMESTEP,
                    "state": name,
                    "energy": energy,
                    "sample_relative_energy_rebound": rebound,
                    **health,
                    **{
                        f"contact_{key}": value
                        for key, value in contact.items()
                    },
                }
            )
            healthy = healthy and bool(
                health["finite"]
                and health["within_bounds"]
                and health["mass_healthy"]
            )
        return healthy

    if not record(0):
        trajectory_stopped_early = True
        stop_reason = "initial field health failure"
    run_started = time.perf_counter()
    if not trajectory_stopped_early:
        for step in range(1, STEPS + 1):
            if time.perf_counter() - run_started > WALL_SECONDS_CAP:
                trajectory_stopped_early = True
                stop_reason = "wall cap reached"
                break
            for name in states:
                states[name] = solvers[name].propose_step(states[name])
                field = states[name]
                current_mass = float(
                    np.sum(field, dtype=np.float64) * lattice.cell_volume
                )
                relative_mass_drift = abs(
                    current_mass - initial_masses[name]
                ) / max(
                    abs(initial_masses[name]),
                    np.finfo(float).tiny,
                )
                if (
                    not np.all(np.isfinite(field))
                    or float(np.min(field)) < FIELD_MINIMUM
                    or float(np.max(field)) > FIELD_MAXIMUM
                    or relative_mass_drift > MAXIMUM_RELATIVE_MASS_DRIFT
                ):
                    trajectory_stopped_early = True
                    stop_reason = f"{name} field health failed at step {step}"
                    break
            if trajectory_stopped_early:
                break
            if step == 1:
                t1_localization = {
                    "product": gb_response_localization(
                        states["product_off"],
                        states["product_on"],
                        profile.grain_energy_density,
                    ),
                    "additive_odds": gb_response_localization(
                        states["odds_off"],
                        states["odds_on"],
                        profile.grain_energy_density,
                    ),
                }
            if step in SAMPLE_STEPS and not record(step):
                trajectory_stopped_early = True
                stop_reason = f"sampled health failure at step {step}"
                break
    trajectory_wall_seconds = time.perf_counter() - run_started

    history_path = output / "history.csv"
    if history:
        with history_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

    health_checks = {
        **pair_identity_checks,
        "trajectory_completed_100_steps": (
            not trajectory_stopped_early
            and all(
                any(
                    record["step"] == STEPS and record["state"] == name
                    for record in history
                )
                for name in states
            )
        ),
        "all_sampled_fields_healthy": bool(
            history
            and all(
                record["finite"]
                and record["within_bounds"]
                and record["mass_healthy"]
                for record in history
            )
        ),
        "all_sampled_energies_dissipative": bool(
            maximum_rebounds
            and all(
                rebound <= MAXIMUM_RELATIVE_ENERGY_REBOUND
                for rebound in maximum_rebounds.values()
            )
        ),
        "wall_cap_not_reached": stop_reason != "wall cap reached",
    }
    localization_checks = {
        "product_t1_response_in_gb_band": bool(
            t1_localization
            and t1_localization["product"][
                "gb_band_squared_norm_fraction"
            ]
            >= MINIMUM_T1_GB_BAND_RESPONSE_FRACTION
        ),
        "odds_t1_response_in_gb_band": bool(
            t1_localization
            and t1_localization["additive_odds"][
                "gb_band_squared_norm_fraction"
            ]
            >= MINIMUM_T1_GB_BAND_RESPONSE_FRACTION
        ),
        "product_t1_junction_protected": bool(
            t1_localization
            and t1_localization["product"][
                "junction_to_gb_rms_ratio"
            ]
            <= MAXIMUM_T1_JUNCTION_TO_GB_RMS
        ),
        "odds_t1_junction_protected": bool(
            t1_localization
            and t1_localization["additive_odds"][
                "junction_to_gb_rms_ratio"
            ]
            <= MAXIMUM_T1_JUNCTION_TO_GB_RMS
        ),
    }
    if health_checks["trajectory_completed_100_steps"]:
        agreement_metrics, agreement_checks, profiles = endpoint_agreement(
            states,
            first_xy,
            second_xz,
            vapor,
            solid,
            initial_metrics,
        )
        slices_path = output / "diagnostic-slices.npz"
        midpoint_lower = SHAPE[0] // 2 - 1
        midpoint_upper = SHAPE[0] // 2
        _atomic_npz(
            slices_path,
            z=centered_coordinates(SHAPE[2]),
            product_initial_contact=0.5
            * (
                product[midpoint_lower].astype(np.float32)
                + product[midpoint_upper].astype(np.float32)
            ),
            odds_initial_contact=0.5
            * (
                odds[midpoint_lower].astype(np.float32)
                + odds[midpoint_upper].astype(np.float32)
            ),
            product_final_contact=0.5
            * (
                states["product_off"][midpoint_lower].astype(np.float32)
                + states["product_off"][midpoint_upper].astype(np.float32)
            ),
            odds_final_contact=0.5
            * (
                states["odds_off"][midpoint_lower].astype(np.float32)
                + states["odds_off"][midpoint_upper].astype(np.float32)
            ),
            **{
                f"radius_{name}": values
                for name, values in profiles.items()
            },
        )
        output_files = {
            "history": history_path.name,
            "history_sha256": _sha256_path(history_path),
            "diagnostic_slices": slices_path.name,
            "diagnostic_slices_sha256": _sha256_path(slices_path),
        }
    else:
        agreement_metrics = {}
        agreement_checks = {
            "endpoint_agreement_not_evaluated": False,
        }
        output_files = {
            "history": history_path.name if history else None,
            "history_sha256": (
                _sha256_path(history_path) if history else None
            ),
        }

    classification = classify(
        health_checks,
        validity_checks,
        localization_checks,
        agreement_checks,
    )
    summary = {
        "status": (
            "completed"
            if health_checks["trajectory_completed_100_steps"]
            else "stopped"
        ),
        "classification": classification,
        "contract": contract,
        "checks": {
            "health": health_checks,
            "sensitivity_validity": validity_checks,
            "t1_gb_localization": localization_checks,
            "t100_initializer_agreement": agreement_checks,
        },
        "resources": {
            "report": resource_report,
            "checks": resource_checks,
        },
        "source": {
            "archive_sha256": EXPECTED_SOURCE_SHA256,
            "profile_fingerprint": EXPECTED_PROFILE_FINGERPRINT,
            "normalization": clipping,
        },
        "geometry": geometry,
        "initial_fingerprints": initial_fingerprints,
        "initial_metrics": initial_metrics,
        "timing_probe": probe,
        "trajectory": {
            "stopped_early": trajectory_stopped_early,
            "stop_reason": stop_reason,
            "wall_seconds": trajectory_wall_seconds,
            "maximum_sampled_relative_energy_rebound": maximum_rebounds,
            "history_record_count": len(history),
        },
        "t1_gb_localization": t1_localization,
        "t100_agreement": agreement_metrics,
        "outputs": output_files,
        "provenance": {
            "runner_sha256": _sha256_path(SOURCE_PATH),
            "contract_sha256": _sha256_path(CONTRACT_PATH),
            "roy_model_sha256": _sha256_path(
                DIRECTORY.parent / "roy_2021_reproduction" / "model.py"
            ),
            "fixed_gb_model_sha256": _sha256_path(
                DIRECTORY.parent / "roy_fixed_gb_bridge" / "model.py"
            ),
        },
        "scientific_scope": {
            "bounded_startup_only": True,
            "long_time_initializer_independence_established": False,
            "sentinel_started": False,
            "spacing_sweep_started": False,
            "constant_mobility_used": False,
            "gb_delayed_switch_used": False,
            "random_noise_used": False,
            "automatic_follow_on_authorized": False,
        },
        "wall_seconds": time.perf_counter() - started,
        "automatic_follow_on_authorized": False,
    }
    _atomic_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run(args.output)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "classification": summary["classification"],
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
