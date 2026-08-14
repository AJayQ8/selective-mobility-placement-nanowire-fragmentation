"""Checkpoint-derived collar placement and compact response diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from ..roy_2021_reproduction.model import (
    FROZEN_DEG90,
    Array,
    PeriodicLattice,
    RoyDeg90Definition,
)
from ..roy_isolated_gb_ridge.analysis import radial_level_contours
from .geometry import FourArmCollarGeometry, centered_coordinates


RADIUS = 6.0
INTERFACE_WIDTH = float(np.sqrt(8.0))
RAYLEIGH_WAVELENGTH = float(
    2.0 * np.sqrt(2.0) * np.pi * RADIUS
)
CONTOUR_LEVELS = (0.45, 0.50, 0.55)
CONTOUR_ANGLE_COUNT = 128
CONTOUR_RADIAL_STEP = 0.125
MINIMUM_SEARCH_DISTANCE = RADIUS + 2.0 * INTERFACE_WIDTH
MAXIMUM_SEARCH_DISTANCE = 1.25 * RAYLEIGH_WAVELENGTH
BASELINE_MINIMUM_DISTANCE = 1.5 * RAYLEIGH_WAVELENGTH
BASELINE_MAXIMUM_DISTANCE = 2.0 * RAYLEIGH_WAVELENGTH


def _quadratic_minimum(
    coordinates: Array,
    values: Array,
    index: int,
) -> tuple[float, float]:
    if index <= 0 or index >= values.size - 1:
        return float(coordinates[index]), float(values[index])
    x = coordinates[index - 1 : index + 2]
    y = values[index - 1 : index + 2]
    coefficients = np.polyfit(x, y, 2)
    if coefficients[0] <= 0.0:
        return float(coordinates[index]), float(values[index])
    vertex = float(-coefficients[1] / (2.0 * coefficients[0]))
    if vertex < x[0] or vertex > x[-1]:
        return float(coordinates[index]), float(values[index])
    return vertex, float(np.polyval(coefficients, vertex))


def _outward_crossing(
    coordinates: Array,
    values: Array,
    target: float,
    *,
    start: int,
) -> float:
    for index in range(start, values.size - 1):
        first = values[index] - target
        second = values[index + 1] - target
        if first == 0.0:
            return float(coordinates[index])
        if first * second <= 0.0 and values[index + 1] >= target:
            denominator = values[index + 1] - values[index]
            if denominator == 0.0:
                return float(coordinates[index + 1])
            fraction = (target - values[index]) / denominator
            return float(
                coordinates[index]
                + fraction * (coordinates[index + 1] - coordinates[index])
            )
    raise RuntimeError("outward half-depth crossing is unresolved")


def _side_depletion(distance: Array, radius: Array) -> dict[str, float]:
    search = (
        (distance >= MINIMUM_SEARCH_DISTANCE)
        & (distance <= MAXIMUM_SEARCH_DISTANCE)
        & np.isfinite(radius)
    )
    baseline_mask = (
        (distance >= BASELINE_MINIMUM_DISTANCE)
        & (distance <= BASELINE_MAXIMUM_DISTANCE)
        & np.isfinite(radius)
    )
    if np.count_nonzero(search) < 5 or np.count_nonzero(baseline_mask) < 5:
        raise RuntimeError("depletion search or baseline is unresolved")
    baseline = float(np.median(radius[baseline_mask]))
    search_indices = np.flatnonzero(search)
    local = int(np.argmin(radius[search]))
    index = int(search_indices[local])
    minimum_position, minimum_radius = _quadratic_minimum(
        distance, radius, index
    )
    depth = baseline - minimum_radius
    if depth <= 0.0:
        raise RuntimeError("depletion minimum is not below its baseline")
    half_level = baseline - 0.5 * depth
    outward = _outward_crossing(
        distance,
        radius,
        half_level,
        start=index,
    )
    return {
        "minimum_position": minimum_position,
        "minimum_radius": minimum_radius,
        "baseline_radius": baseline,
        "depletion_depth": depth,
        "outward_half_depth": outward,
        "minimum_clearance_from_exclusion": (
            minimum_position - MINIMUM_SEARCH_DISTANCE
        ),
    }


def _wire_contours(
    field: Array,
    *,
    lattice: PeriodicLattice,
    axial_axis: int,
    center: tuple[float, float],
    maximum_radius: float,
) -> dict[float, Any]:
    coordinates = tuple(
        centered_coordinates(cells, lattice.spacing)
        for cells in lattice.shape
    )
    transverse_axes = tuple(
        axis for axis in range(3) if axis != axial_axis
    )
    contours = radial_level_contours(
        field,
        axial_axis=axial_axis,
        transverse_coordinates=(
            coordinates[transverse_axes[0]],
            coordinates[transverse_axes[1]],
        ),
        center=center,
        levels=CONTOUR_LEVELS,
        angle_count=CONTOUR_ANGLE_COUNT,
        radial_step=CONTOUR_RADIAL_STEP,
        maximum_radius=maximum_radius,
    )
    axial = coordinates[axial_axis]
    reports: dict[float, Any] = {}
    for level, contour in contours.items():
        smoothed = ndimage.gaussian_filter1d(
            contour.mean_radius,
            sigma=INTERFACE_WIDTH / lattice.spacing,
            mode="wrap",
        )
        positive = axial >= 0.0
        negative = axial <= 0.0
        reports[level] = {
            "positive": _side_depletion(
                axial[positive], smoothed[positive]
            ),
            "negative": _side_depletion(
                -axial[negative][::-1], smoothed[negative][::-1]
            ),
            "minimum_valid_fraction": float(
                np.min(contour.valid_fraction)
            ),
        }
    return reports


def measure_t100_depletion(
    field: Array,
    definition: RoyDeg90Definition = FROZEN_DEG90,
) -> dict[str, Any]:
    """Measure all four source-semantic arms at three contour levels."""

    lattice = definition.lattice
    if field.shape != lattice.shape:
        raise ValueError("field shape does not match the frozen Roy lattice")
    radius = definition.radius_1 * lattice.spacing
    second_center_x = (
        definition.radius_1 + definition.radius_2
    ) * lattice.spacing
    first = _wire_contours(
        field,
        lattice=lattice,
        axial_axis=2,
        center=(0.0, 0.0),
        maximum_radius=radius + INTERFACE_WIDTH,
    )
    # The second wire sits near the positive thin-domain boundary, so its
    # contour rays must stay within that inscribed radius.
    x = centered_coordinates(lattice.shape[0], lattice.spacing)
    second_inscribed = min(
        second_center_x - x[0],
        x[-1] - second_center_x,
    )
    second = _wire_contours(
        field,
        lattice=lattice,
        axial_axis=1,
        center=(second_center_x, 0.0),
        maximum_radius=min(
            radius + INTERFACE_WIDTH,
            second_inscribed - 0.25 * lattice.spacing,
        ),
    )
    outward = []
    minima = []
    clearances = []
    for wire in (first, second):
        for level in CONTOUR_LEVELS:
            for side in ("positive", "negative"):
                record = wire[level][side]
                outward.append(record["outward_half_depth"])
                minima.append(record["minimum_position"])
                clearances.append(
                    record["minimum_clearance_from_exclusion"]
                )
    return {
        "coordinate_convention": (
            "Roy source-semantic integer-centered; junction y=z=0"
        ),
        "levels": list(CONTOUR_LEVELS),
        "first_wire_z": first,
        "second_wire_y": second,
        "pooled": {
            "minimum_position_range": [
                float(np.min(minima)),
                float(np.max(minima)),
            ],
            "minimum_clearance_range": [
                float(np.min(clearances)),
                float(np.max(clearances)),
            ],
            "outward_half_depth_range": [
                float(np.min(outward)),
                float(np.max(outward)),
            ],
            "maximum_outward_half_depth": float(np.max(outward)),
        },
    }


def derive_four_arm_geometry(
    measurement: dict[str, Any],
    *,
    spacing: float,
    radius: float = RADIUS,
    protected_mobility_factor: float = 0.1,
) -> FourArmCollarGeometry:
    """Place the first smooth ramp outside every measured recovery flank."""

    maximum = float(
        measurement["pooled"]["maximum_outward_half_depth"]
    )
    inner = float(np.ceil(maximum / spacing) * spacing)
    return FourArmCollarGeometry(
        inner_support_distance=inner,
        outer_support_distance=inner + 2.0 * radius,
        transition_width=0.5 * radius,
        protected_mobility_factor=protected_mobility_factor,
    )


def phase_excess_radius_profiles(
    field: Array,
    definition: RoyDeg90Definition = FROZEN_DEG90,
) -> dict[str, dict[str, Array]]:
    """Return compact area-equivalent profiles for the two crossed wires."""

    lattice = definition.lattice
    if field.shape != lattice.shape:
        raise ValueError("field shape does not match the frozen Roy lattice")
    _, ny, nz = lattice.shape
    half = 48
    y0 = ny // 2 - half
    z0 = nz // 2 - half
    definitions = (
        ("first_wire_z", field[:, y0 : y0 + 96, :], (0, 1), nz),
        ("second_wire_y", field[:, :, z0 : z0 + 96], (0, 2), ny),
    )
    output: dict[str, dict[str, Array]] = {}
    for name, local, transverse_axes, cells in definitions:
        area = (
            np.sum(local, axis=transverse_axes, dtype=np.float64)
            * lattice.spacing**2
        )
        if np.min(area) < 0.0:
            raise RuntimeError("phase-excess area became negative")
        output[name] = {
            "coordinate": centered_coordinates(cells, lattice.spacing),
            "area": area,
            "radius": np.sqrt(area / np.pi),
        }
    return output


def crossed_contour_radius_profiles(
    field: Array,
    definition: RoyDeg90Definition = FROZEN_DEG90,
) -> dict[str, dict[str, Any]]:
    """Return raw three-level contour-radius profiles for both wires."""

    lattice = definition.lattice
    if field.shape != lattice.shape:
        raise ValueError("field shape does not match the frozen Roy lattice")
    coordinates = tuple(
        centered_coordinates(cells, lattice.spacing)
        for cells in lattice.shape
    )
    radius = definition.radius_1 * lattice.spacing
    second_center_x = (
        definition.radius_1 + definition.radius_2
    ) * lattice.spacing
    x = coordinates[0]
    second_inscribed = min(
        second_center_x - x[0],
        x[-1] - second_center_x,
    )
    definitions = (
        (
            "first_wire_z",
            2,
            (coordinates[0], coordinates[1]),
            (0.0, 0.0),
            radius + INTERFACE_WIDTH,
        ),
        (
            "second_wire_y",
            1,
            (coordinates[0], coordinates[2]),
            (second_center_x, 0.0),
            min(
                radius + INTERFACE_WIDTH,
                second_inscribed - 0.25 * lattice.spacing,
            ),
        ),
    )
    output: dict[str, dict[str, Any]] = {}
    for name, axis, transverse, center, maximum_radius in definitions:
        contours = radial_level_contours(
            field,
            axial_axis=axis,
            transverse_coordinates=transverse,
            center=center,
            levels=CONTOUR_LEVELS,
            angle_count=CONTOUR_ANGLE_COUNT,
            radial_step=CONTOUR_RADIAL_STEP,
            maximum_radius=maximum_radius,
        )
        output[name] = {
            "coordinate": coordinates[axis],
            "levels": {
                f"{level:.2f}": {
                    "mean_radius": contours[level].mean_radius,
                    "valid_fraction": contours[level].valid_fraction,
                    "angular_std": contours[level].angular_std,
                }
                for level in CONTOUR_LEVELS
            },
        }
    return output


def cross_arm_relative_l2_asymmetry(field: Array) -> float:
    """Compare a field with the source-semantic wire-exchange transform."""

    if field.ndim != 3:
        raise ValueError("field must be three-dimensional")
    nx = field.shape[0]
    if nx != FROZEN_DEG90.lattice.shape[0]:
        raise ValueError("wire-exchange transform requires the frozen thin axis")
    indices = (
        2 * (nx // 2 + FROZEN_DEG90.radius_1) - np.arange(nx)
    ) % nx
    transformed = np.take(
        np.swapaxes(field, 1, 2),
        indices.astype(np.int64),
        axis=0,
    )
    numerator = float(
        np.linalg.norm(
            np.asarray(field, dtype=np.float64) - transformed
        )
    )
    denominator = float(np.linalg.norm(field))
    return numerator / denominator if denominator else 0.0
