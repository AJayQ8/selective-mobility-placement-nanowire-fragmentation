"""Geometry and localization metrics shared by the bounded bridge runners."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from scipy import ndimage

from ..roy_2021_reproduction.model import Array, PeriodicLattice
from .geometry import centered_coordinates


def array_fingerprint(field: Array) -> str:
    contiguous = (
        field if field.flags.c_contiguous else np.ascontiguousarray(field)
    )
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def signed_periodic_distance(
    coordinate: Array,
    plane: float,
    length: float,
) -> Array:
    return (coordinate - plane + 0.5 * length) % length - 0.5 * length


def _interpolated_crossing(
    coordinate_0: float,
    value_0: float,
    value_1: float,
    spacing: float,
    level: float,
) -> float:
    if value_1 == value_0:
        return float(coordinate_0)
    fraction = (level - value_0) / (value_1 - value_0)
    return float(coordinate_0 + fraction * spacing)


def slab_surface_profiles(
    field: Array,
    lattice: PeriodicLattice,
    *,
    tangent_axis: int,
    normal_axis: int,
    extruded_axis: int,
    level: float,
) -> tuple[Array, Array]:
    """Interpolate bottom and top level sets for each tangent coordinate."""

    if sorted((tangent_axis, normal_axis, extruded_axis)) != [0, 1, 2]:
        raise ValueError("the three axes must be distinct")
    ordered = np.moveaxis(
        field,
        (tangent_axis, normal_axis, extruded_axis),
        (0, 1, 2),
    )
    plane = np.mean(ordered, axis=2)
    normal = centered_coordinates(lattice, normal_axis)
    center = lattice.shape[normal_axis] // 2
    bottom = np.full(lattice.shape[tangent_axis], np.nan, dtype=np.float64)
    top = np.full_like(bottom, np.nan)

    for tangent_index in range(plane.shape[0]):
        values = plane[tangent_index]
        bottom_candidates = np.flatnonzero(
            (values[:center] < level)
            & (values[1 : center + 1] >= level)
        )
        top_candidates = np.flatnonzero(
            (values[center:-1] >= level)
            & (values[center + 1 :] < level)
        )
        if bottom_candidates.size:
            lower = int(bottom_candidates[-1])
            bottom[tangent_index] = _interpolated_crossing(
                normal[lower],
                float(values[lower]),
                float(values[lower + 1]),
                lattice.spacing,
                level,
            )
        if top_candidates.size:
            lower = int(center + top_candidates[0])
            top[tangent_index] = _interpolated_crossing(
                normal[lower],
                float(values[lower]),
                float(values[lower + 1]),
                lattice.spacing,
                level,
            )
    return bottom, top


def _branch_angle(
    profile: Array,
    tangent: Array,
    *,
    plane: float,
    length: float,
    width: float,
    window: tuple[float, float],
) -> float:
    signed = signed_periodic_distance(tangent, plane, length)
    branch_angles: list[float] = []
    for sign in (-1.0, 1.0):
        local = sign * signed
        mask = (
            (local >= window[0] * width)
            & (local <= window[1] * width)
            & np.isfinite(profile)
        )
        if int(np.count_nonzero(mask)) < 6:
            return float("nan")
        coefficients = np.polyfit(local[mask], profile[mask], deg=3)
        branch_angles.append(float(np.arctan(abs(coefficients[-2]))))
    return float(np.degrees(np.pi - sum(branch_angles)))


def _surface_depth(
    profile: Array,
    tangent: Array,
    *,
    plane: float,
    length: float,
    top: bool,
) -> float:
    signed = signed_periodic_distance(tangent, plane, length)
    far = (
        (np.abs(signed) >= 0.18 * length)
        & (np.abs(signed) <= 0.24 * length)
        & np.isfinite(profile)
    )
    root = int(np.argmin(np.abs(signed)))
    if not np.any(far) or not np.isfinite(profile[root]):
        return float("nan")
    far_level = float(np.median(profile[far]))
    if top:
        return float(far_level - profile[root])
    return float(profile[root] - far_level)


def topology_checks(field: Array, thresholds: tuple[float, ...]) -> dict[str, Any]:
    """Detect slab disconnection or a vapor cavity away from normal boundaries."""

    structure = ndimage.generate_binary_structure(3, 1)
    reports: dict[str, Any] = {}
    for threshold in thresholds:
        solid = field >= threshold
        _, solid_components = ndimage.label(solid, structure=structure)
        vapor_labels, vapor_components = ndimage.label(
            ~solid,
            structure=structure,
        )
        boundary_labels = set(
            np.unique(vapor_labels[:, 0, :]).tolist()
            + np.unique(vapor_labels[:, -1, :]).tolist()
        )
        enclosed = [
            label
            for label in range(1, vapor_components + 1)
            if label not in boundary_labels
        ]
        reports[f"{threshold:.2f}"] = {
            "solid_component_count": int(solid_components),
            "solid_connected": bool(solid_components == 1),
            "enclosed_vapor_component_count": len(enclosed),
            "no_enclosed_vapor_cavity": bool(not enclosed),
        }
    return reports


def planar_groove_metrics(
    field: Array,
    lattice: PeriodicLattice,
    *,
    tangent_axis: int,
    normal_axis: int,
    extruded_axis: int,
    planes: tuple[float, float],
    width: float,
) -> dict[str, Any]:
    """Measure four triple junctions and their threshold/window robustness."""

    tangent = centered_coordinates(lattice, tangent_axis)
    length = lattice.physical_lengths[tangent_axis]
    thresholds = (0.45, 0.50, 0.55)
    windows = ((1.0, 3.0), (1.25, 3.25))
    measurements: dict[str, Any] = {}
    for level in thresholds:
        bottom, top = slab_surface_profiles(
            field,
            lattice,
            tangent_axis=tangent_axis,
            normal_axis=normal_axis,
            extruded_axis=extruded_axis,
            level=level,
        )
        for window in windows:
            key = f"level_{level:.2f}_window_{window[0]:.2f}_{window[1]:.2f}"
            angles: list[float] = []
            for plane in planes:
                angles.append(
                    _branch_angle(
                        bottom,
                        tangent,
                        plane=plane,
                        length=length,
                        width=width,
                        window=window,
                    )
                )
                angles.append(
                    _branch_angle(
                        top,
                        tangent,
                        plane=plane,
                        length=length,
                        width=width,
                        window=window,
                    )
                )
            measurements[key] = {
                "angles_deg": angles,
                "mean_angle_deg": float(np.mean(angles)),
                "spread_deg": float(np.max(angles) - np.min(angles)),
            }

    bottom, top = slab_surface_profiles(
        field,
        lattice,
        tangent_axis=tangent_axis,
        normal_axis=normal_axis,
        extruded_axis=extruded_axis,
        level=0.5,
    )
    depths: list[float] = []
    for plane in planes:
        depths.append(
            _surface_depth(
                bottom,
                tangent,
                plane=plane,
                length=length,
                top=False,
            )
        )
        depths.append(
            _surface_depth(
                top,
                tangent,
                plane=plane,
                length=length,
                top=True,
            )
        )
    primary_key = "level_0.50_window_1.00_3.00"
    primary = measurements[primary_key]
    all_robust_angles = [
        angle
        for report in measurements.values()
        for angle in report["angles_deg"]
    ]
    return {
        "depths": depths,
        "depth_over_width": [float(value / width) for value in depths],
        "depth_relative_spread": float(
            (np.max(depths) - np.min(depths)) / np.mean(depths)
        )
        if float(np.mean(depths)) > 0.0
        else float("inf"),
        "primary_angles_deg": primary["angles_deg"],
        "primary_mean_angle_deg": primary["mean_angle_deg"],
        "primary_spread_deg": primary["spread_deg"],
        "robust_angle_min_deg": float(np.min(all_robust_angles)),
        "robust_angle_max_deg": float(np.max(all_robust_angles)),
        "robust_angle_spread_deg": float(
            np.max(all_robust_angles) - np.min(all_robust_angles)
        ),
        "measurements": measurements,
        "topology": topology_checks(field, thresholds),
    }


def extrusion_error(
    field: Array,
    *,
    extruded_axis: int,
) -> float:
    reference = np.mean(field, axis=extruded_axis, keepdims=True)
    return float(np.max(np.abs(field - reference)))

