"""Reusable measurements for the isolated-GB cylinder feasibility check.

The conserved primary observable is the *full-cross-section* paired phase
excess,

``Delta A(z) = dx * dy * sum_xy(c_gb - c_control) / phase_jump``.

No clipping, contouring, or radial region of interest enters that quantity.
Area-equivalent and contour radii are complementary geometric estimators.  In
particular, a radius-space numerical floor is never compared directly with an
area-space signal: area floors are first converted with the local ``2*pi*R``
Jacobian.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


Array = np.ndarray
BranchKey = tuple[int, int]


@dataclass(frozen=True)
class CrossSectionProfiles:
    """Cross-sectional phase-excess areas and area-equivalent radii."""

    area: Array
    radius: Array
    roi_cell_count: int
    transverse_cell_area: float
    phase_jump: float


@dataclass(frozen=True)
class RadialContourProfile:
    """Subcell ray crossings for one concentration level.

    ``radii`` has shape ``(n_axial, n_angles)``.  The remaining arrays have
    length ``n_axial`` and summarize only finite ray crossings.
    """

    level: float
    angles: Array
    radii: Array
    mean_radius: Array
    median_radius: Array
    minimum_radius: Array
    maximum_radius: Array
    angular_std: Array
    valid_fraction: Array


@dataclass(frozen=True)
class FoldedBranch:
    """One unaveraged side of one periodic grain-boundary plane."""

    plane_index: int
    plane: float
    side: int
    distance: Array
    values: Array


@dataclass(frozen=True)
class BranchFeature:
    """Groove and first positive-ridge measurements for one raw branch."""

    plane_index: int
    plane: float
    side: int
    groove_value: float
    groove_depth: float
    groove_offset: float
    ridge_value: float
    ridge_height: float
    ridge_offset_from_plane: float
    ridge_offset_from_root: float
    height_to_depth_ratio: float
    has_negative_groove: bool
    has_positive_ridge: bool


@dataclass(frozen=True)
class SectorLobeBalance:
    """Positive and negative corrected area integrals in one GB-side sector."""

    plane_index: int
    side: int
    positive_integral: float
    negative_magnitude_integral: float
    net_integral: float
    positive_to_negative_ratio: float


@dataclass(frozen=True)
class LobeBalance:
    """Global and four-sector lobe accounting for a periodic paired profile."""

    raw_total_integral: float
    uniform_residual: float
    removed_uniform_integral: float
    corrected_total_integral: float
    positive_integral: float
    negative_magnitude_integral: float
    positive_to_negative_ratio: float
    corrected_profile: Array
    sectors: Mapping[BranchKey, SectorLobeBalance]
    sector_net_sum: float
    sector_partition_error: float


@dataclass(frozen=True)
class InternalFloorReport:
    """All internal floors expressed in radius units."""

    components: Mapping[str, float]
    largest_component: str
    largest_floor: float
    contour_area_consistent: bool | None
    contour_area_maximum_difference: float | None


def _validated_axis(axis: int, ndim: int) -> int:
    normalized = int(axis)
    if normalized < 0:
        normalized += ndim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"axis {axis} is invalid for a {ndim}-D field")
    return normalized


def _transverse_spacings(
    spacing: float | Sequence[float],
) -> tuple[float, float]:
    if np.isscalar(spacing):
        values = (float(spacing), float(spacing))
    else:
        values = tuple(float(value) for value in spacing)
        if len(values) != 2:
            raise ValueError("transverse_spacing must be a scalar or length two")
    if not all(np.isfinite(values)) or min(values) <= 0.0:
        raise ValueError("transverse spacings must be finite and positive")
    return values


def _finite_scale(values: Array | Sequence[float]) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = np.abs(finite[np.isfinite(finite)])
    if finite.size == 0:
        return 0.0
    return float(np.quantile(finite, 0.95))


def robust_internal_floor(
    values: Array | Sequence[float],
    *,
    absolute_floor: float = 0.0,
    relative_floor: float = 64.0 * np.finfo(np.float64).eps,
) -> float:
    """Return a finite absolute-plus-relative floor in ``values``' units."""

    if not np.isfinite(absolute_floor) or absolute_floor < 0.0:
        raise ValueError("absolute_floor must be finite and nonnegative")
    if not np.isfinite(relative_floor) or relative_floor < 0.0:
        raise ValueError("relative_floor must be finite and nonnegative")
    return float(max(absolute_floor, relative_floor * _finite_scale(values)))


def area_equivalent_radius(
    area: Array | Sequence[float] | float,
    *,
    negative_tolerance: float | None = None,
) -> Array:
    """Convert cross-sectional area to an equivalent circular radius.

    Tiny negative values within ``negative_tolerance`` are treated as roundoff
    and clamped to zero.  Materially negative values yield ``nan`` so that a
    caller's numerical-health check can classify them explicitly.
    """

    values = np.asarray(area, dtype=np.float64)
    if negative_tolerance is None:
        negative_tolerance = robust_internal_floor(
            values,
            relative_floor=256.0 * np.finfo(np.float64).eps,
        )
    if not np.isfinite(negative_tolerance) or negative_tolerance < 0.0:
        raise ValueError("negative_tolerance must be finite and nonnegative")
    safe = values.copy()
    safe[(safe < 0.0) & (safe >= -negative_tolerance)] = 0.0
    safe[safe < -negative_tolerance] = np.nan
    return np.sqrt(safe / np.pi)


def _centered_coordinates(cells: int, spacing: float) -> Array:
    return (
        np.arange(cells, dtype=np.float64) - cells // 2
    ) * float(spacing)


def _radial_roi_mask(
    shape: tuple[int, int],
    spacing: tuple[float, float],
    radial_roi: float | Array | None,
    transverse_coordinates: tuple[Array, Array] | None,
    center: tuple[float, float] | None,
) -> Array:
    if radial_roi is None:
        return np.ones(shape, dtype=bool)
    if not np.isscalar(radial_roi):
        mask = np.asarray(radial_roi, dtype=bool)
        if mask.shape != shape:
            raise ValueError("radial_roi mask must match the transverse shape")
        return mask

    cutoff = float(radial_roi)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("a scalar radial_roi must be finite and positive")
    if transverse_coordinates is None:
        coordinates = (
            _centered_coordinates(shape[0], spacing[0]),
            _centered_coordinates(shape[1], spacing[1]),
        )
    else:
        coordinates = _validate_transverse_coordinates(
            transverse_coordinates,
            shape,
        )
    if center is None:
        center = (0.0, 0.0)
    if len(center) != 2 or not np.all(np.isfinite(center)):
        raise ValueError("center must contain two finite coordinates")
    first = coordinates[0][:, None] - float(center[0])
    second = coordinates[1][None, :] - float(center[1])
    return first * first + second * second <= cutoff * cutoff


def cross_section_profiles(
    field: Array,
    *,
    axial_axis: int,
    transverse_spacing: float | Sequence[float],
    solid_value: float = 1.0,
    vapor_value: float = 0.0,
    radial_roi: float | Array | None = None,
    transverse_coordinates: tuple[Array, Array] | None = None,
    center: tuple[float, float] | None = None,
) -> CrossSectionProfiles:
    """Measure phase-excess area and equivalent radius on every axial slice.

    ``radial_roi`` may be a physical cutoff radius or a Boolean transverse
    mask.  The field is neither clipped nor thresholded.  Explicit
    ``solid_value`` and ``vapor_value`` define the phase jump.  A radial ROI is
    appropriate for an individual isolated cylinder; it must not be used for
    the primary paired ``Delta A`` measurement.
    """

    values = np.asarray(field)
    if values.ndim != 3:
        raise ValueError("field must be three-dimensional")
    axis = _validated_axis(axial_axis, values.ndim)
    if not np.all(np.isfinite(values)):
        raise ValueError("field must contain only finite values")
    spacing = _transverse_spacings(transverse_spacing)
    if not np.isfinite(solid_value) or not np.isfinite(vapor_value):
        raise ValueError("phase reference values must be finite")
    phase_jump = float(solid_value - vapor_value)
    if phase_jump <= 0.0:
        raise ValueError("solid_value must exceed vapor_value")

    ordered = np.moveaxis(values, axis, -1)
    transverse_shape = (ordered.shape[0], ordered.shape[1])
    mask = _radial_roi_mask(
        transverse_shape,
        spacing,
        radial_roi,
        transverse_coordinates,
        center,
    )
    normalized = (ordered - vapor_value) / phase_jump
    area = (
        np.sum(normalized * mask[..., None], axis=(0, 1), dtype=np.float64)
        * spacing[0]
        * spacing[1]
    )
    return CrossSectionProfiles(
        area=area,
        radius=area_equivalent_radius(area),
        roi_cell_count=int(np.count_nonzero(mask)),
        transverse_cell_area=float(spacing[0] * spacing[1]),
        phase_jump=phase_jump,
    )


def paired_cross_section_delta_area(
    gb_field: Array,
    control_field: Array,
    *,
    axial_axis: int,
    transverse_spacing: float | Sequence[float],
    phase_jump: float = 1.0,
) -> Array:
    """Return the exact full-cross-section paired ``Delta A`` profile.

    This function intentionally exposes no ROI or clipping option.  Therefore
    ``sum(Delta A) * dz`` is exactly the paired phase-volume difference up to
    floating-point summation order.
    """

    gb = np.asarray(gb_field)
    control = np.asarray(control_field)
    if gb.shape != control.shape or gb.ndim != 3:
        raise ValueError("paired fields must have the same three-dimensional shape")
    if not np.all(np.isfinite(gb)) or not np.all(np.isfinite(control)):
        raise ValueError("paired fields must contain only finite values")
    if not np.isfinite(phase_jump) or phase_jump <= 0.0:
        raise ValueError("phase_jump must be finite and positive")
    axis = _validated_axis(axial_axis, gb.ndim)
    spacing = _transverse_spacings(transverse_spacing)
    ordered = np.moveaxis(gb - control, axis, -1)
    return (
        np.sum(ordered, axis=(0, 1), dtype=np.float64)
        * spacing[0]
        * spacing[1]
        / float(phase_jump)
    )


def _validate_transverse_coordinates(
    coordinates: tuple[Array, Array],
    shape: tuple[int, int],
) -> tuple[Array, Array]:
    if len(coordinates) != 2:
        raise ValueError("transverse_coordinates must contain two arrays")
    validated: list[Array] = []
    for coordinate, cells in zip(coordinates, shape, strict=True):
        values = np.asarray(coordinate, dtype=np.float64)
        if values.ndim != 1 or values.size != cells:
            raise ValueError("each transverse coordinate must match its axis")
        differences = np.diff(values)
        if (
            not np.all(np.isfinite(values))
            or np.any(differences <= 0.0)
            or not np.allclose(
                differences,
                differences[0],
                rtol=1.0e-10,
                atol=1.0e-13,
            )
        ):
            raise ValueError(
                "transverse coordinates must be finite, increasing, and uniform"
            )
        validated.append(values)
    return validated[0], validated[1]


def _ray_interpolation_geometry(
    coordinates: tuple[Array, Array],
    center: tuple[float, float],
    angles: Array,
    radii: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    first, second = coordinates
    first_points = center[0] + np.cos(angles)[:, None] * radii[None, :]
    second_points = center[1] + np.sin(angles)[:, None] * radii[None, :]
    first_fractional = (first_points - first[0]) / (first[1] - first[0])
    second_fractional = (second_points - second[0]) / (second[1] - second[0])
    valid = (
        (first_fractional >= 0.0)
        & (first_fractional <= first.size - 1)
        & (second_fractional >= 0.0)
        & (second_fractional <= second.size - 1)
    )
    first_lower = np.clip(
        np.floor(first_fractional).astype(np.int64),
        0,
        first.size - 2,
    )
    second_lower = np.clip(
        np.floor(second_fractional).astype(np.int64),
        0,
        second.size - 2,
    )
    first_weight = np.clip(first_fractional - first_lower, 0.0, 1.0)
    second_weight = np.clip(second_fractional - second_lower, 0.0, 1.0)
    return (
        first_lower,
        second_lower,
        first_weight,
        second_weight,
        valid,
    )


def _bilinear_ray_values(
    section: Array,
    geometry: tuple[Array, Array, Array, Array, Array],
) -> Array:
    first, second, first_weight, second_weight, valid = geometry
    lower_lower = section[first, second]
    upper_lower = section[first + 1, second]
    lower_upper = section[first, second + 1]
    upper_upper = section[first + 1, second + 1]
    sampled = (
        (1.0 - first_weight) * (1.0 - second_weight) * lower_lower
        + first_weight * (1.0 - second_weight) * upper_lower
        + (1.0 - first_weight) * second_weight * lower_upper
        + first_weight * second_weight * upper_upper
    )
    return np.where(valid, sampled, np.nan)


def _descending_crossings(
    sampled: Array,
    radii: Array,
    level: float,
) -> Array:
    result = np.full(sampled.shape[0], np.nan, dtype=np.float64)
    for angle_index, values in enumerate(sampled):
        crossings = np.flatnonzero(
            np.isfinite(values[:-1])
            & np.isfinite(values[1:])
            & (values[:-1] >= level)
            & (values[1:] < level)
        )
        if crossings.size == 0:
            continue
        lower = int(crossings[-1])
        value_0 = float(values[lower])
        value_1 = float(values[lower + 1])
        if value_1 == value_0:
            result[angle_index] = radii[lower]
        else:
            fraction = (level - value_0) / (value_1 - value_0)
            result[angle_index] = (
                radii[lower]
                + fraction * (radii[lower + 1] - radii[lower])
            )
    return result


def _nan_summary(values: Array, operation) -> Array:
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    for index, row in enumerate(values):
        finite = row[np.isfinite(row)]
        if finite.size:
            result[index] = float(operation(finite))
    return result


def radial_level_contours(
    field: Array,
    *,
    axial_axis: int,
    transverse_coordinates: tuple[Array, Array],
    center: tuple[float, float],
    levels: Sequence[float] = (0.45, 0.50, 0.55),
    angle_count: int = 128,
    radial_step: float | None = None,
    maximum_radius: float | None = None,
) -> dict[float, RadialContourProfile]:
    """Find outer descending concentration contours along azimuthal rays.

    Bilinear interpolation in each transverse section plus linear
    interpolation across the bracketing radial samples gives subcell contour
    radii.  All ray values are retained so circularity and angular scatter can
    be audited instead of being hidden by an average.
    """

    values = np.asarray(field)
    if values.ndim != 3:
        raise ValueError("field must be three-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("field must contain only finite values")
    axis = _validated_axis(axial_axis, values.ndim)
    ordered = np.moveaxis(values, axis, -1)
    coordinates = _validate_transverse_coordinates(
        transverse_coordinates,
        (ordered.shape[0], ordered.shape[1]),
    )
    if len(center) != 2 or not np.all(np.isfinite(center)):
        raise ValueError("center must contain two finite coordinates")
    center = (float(center[0]), float(center[1]))
    if angle_count < 8:
        raise ValueError("angle_count must be at least eight")

    level_values = tuple(float(level) for level in levels)
    if (
        not level_values
        or len(set(level_values)) != len(level_values)
        or not all(np.isfinite(level) for level in level_values)
    ):
        raise ValueError("levels must be distinct and finite")
    spacings = (
        coordinates[0][1] - coordinates[0][0],
        coordinates[1][1] - coordinates[1][0],
    )
    if radial_step is None:
        radial_step = 0.25 * min(spacings)
    if not np.isfinite(radial_step) or radial_step <= 0.0:
        raise ValueError("radial_step must be finite and positive")
    inscribed = min(
        center[0] - coordinates[0][0],
        coordinates[0][-1] - center[0],
        center[1] - coordinates[1][0],
        coordinates[1][-1] - center[1],
    )
    if maximum_radius is None:
        maximum_radius = float(inscribed)
    if (
        not np.isfinite(maximum_radius)
        or maximum_radius <= radial_step
        or maximum_radius > inscribed + 1.0e-12
    ):
        raise ValueError(
            "maximum_radius must be positive and fit inside the sampled section"
        )

    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        int(angle_count),
        endpoint=False,
        dtype=np.float64,
    )
    radial_count = int(np.ceil(maximum_radius / radial_step)) + 1
    radii = np.linspace(
        0.0,
        float(maximum_radius),
        radial_count,
        dtype=np.float64,
    )
    geometry = _ray_interpolation_geometry(
        coordinates,
        center,
        angles,
        radii,
    )
    all_crossings = {
        level: np.full(
            (ordered.shape[-1], angle_count),
            np.nan,
            dtype=np.float64,
        )
        for level in level_values
    }
    for axial_index in range(ordered.shape[-1]):
        sampled = _bilinear_ray_values(ordered[..., axial_index], geometry)
        for level in level_values:
            all_crossings[level][axial_index] = _descending_crossings(
                sampled,
                radii,
                level,
            )

    reports: dict[float, RadialContourProfile] = {}
    for level in level_values:
        crossings = all_crossings[level]
        valid_fraction = np.mean(np.isfinite(crossings), axis=1)
        reports[level] = RadialContourProfile(
            level=level,
            angles=angles.copy(),
            radii=crossings,
            mean_radius=_nan_summary(crossings, np.mean),
            median_radius=_nan_summary(crossings, np.median),
            minimum_radius=_nan_summary(crossings, np.min),
            maximum_radius=_nan_summary(crossings, np.max),
            angular_std=_nan_summary(crossings, np.std),
            valid_fraction=valid_fraction,
        )
    return reports


def signed_periodic_distance(
    coordinate: Array | Sequence[float] | float,
    plane: float,
    period: float,
) -> Array:
    """Signed minimum-image distance in ``[-period/2, period/2)``."""

    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("period must be finite and positive")
    return (
        np.asarray(coordinate, dtype=np.float64)
        - float(plane)
        + 0.5 * period
    ) % period - 0.5 * period


def _validated_periodic_profile(
    coordinate: Array | Sequence[float],
    profile: Array | Sequence[float],
    period: float,
) -> tuple[Array, Array, float]:
    z = np.asarray(coordinate, dtype=np.float64)
    values = np.asarray(profile, dtype=np.float64)
    if z.ndim != 1 or values.ndim != 1 or z.size != values.size or z.size < 4:
        raise ValueError("coordinate and profile must be equal one-dimensional arrays")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(values)):
        raise ValueError("coordinate and profile must contain only finite values")
    order = np.argsort(z)
    z = z[order]
    values = values[order]
    steps = np.diff(z)
    if np.any(steps <= 0.0) or not np.allclose(
        steps,
        steps[0],
        rtol=1.0e-10,
        atol=1.0e-13,
    ):
        raise ValueError("coordinate must be a uniform periodic lattice")
    spacing = float(steps[0])
    if not np.isclose(z.size * spacing, period, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("coordinate lattice extent must equal period")
    return z, values, spacing


def _periodic_linear_value(
    coordinate: Array,
    values: Array,
    location: float,
    period: float,
) -> float:
    origin = coordinate[0]
    wrapped = (float(location) - origin) % period + origin
    extended_coordinate = np.concatenate(
        (coordinate, np.asarray([coordinate[0] + period])),
    )
    extended_values = np.concatenate((values, values[:1]))
    return float(np.interp(wrapped, extended_coordinate, extended_values))


def _default_branch_limit(
    plane_index: int,
    planes: Array,
    period: float,
) -> float:
    if planes.size == 1:
        return 0.5 * period
    distances = np.abs(
        signed_periodic_distance(
            np.delete(planes, plane_index),
            planes[plane_index],
            period,
        )
    )
    return 0.5 * float(np.min(distances))


def fold_periodic_profile(
    coordinate: Array | Sequence[float],
    profile: Array | Sequence[float],
    *,
    planes: Sequence[float],
    period: float,
    maximum_distance: float | None = None,
) -> dict[BranchKey, FoldedBranch]:
    """Return every plane/side branch without averaging any of them."""

    z, values, spacing = _validated_periodic_profile(
        coordinate,
        profile,
        period,
    )
    plane_values = np.asarray(planes, dtype=np.float64)
    if (
        plane_values.ndim != 1
        or plane_values.size == 0
        or not np.all(np.isfinite(plane_values))
    ):
        raise ValueError("planes must be a nonempty finite sequence")
    tolerance = 64.0 * np.finfo(np.float64).eps * max(period, 1.0)
    for first in range(plane_values.size):
        separations = np.abs(
            signed_periodic_distance(
                plane_values[first + 1 :],
                plane_values[first],
                period,
            )
        )
        if np.any(separations <= tolerance):
            raise ValueError("periodic GB planes must be distinct")

    branches: dict[BranchKey, FoldedBranch] = {}
    for plane_index, plane in enumerate(plane_values):
        branch_limit = (
            _default_branch_limit(plane_index, plane_values, period)
            if maximum_distance is None
            else float(maximum_distance)
        )
        if not np.isfinite(branch_limit) or branch_limit <= spacing:
            raise ValueError("maximum branch distance must exceed one cell")
        signed = signed_periodic_distance(z, plane, period)
        plane_value = _periodic_linear_value(z, values, plane, period)
        for side in (-1, 1):
            outward = side * signed
            selected = (outward > tolerance) & (
                outward <= branch_limit + tolerance
            )
            order = np.argsort(outward[selected])
            distance = np.concatenate(
                (
                    np.asarray([0.0]),
                    outward[selected][order],
                )
            )
            branch_values = np.concatenate(
                (
                    np.asarray([plane_value]),
                    values[selected][order],
                )
            )
            branches[(plane_index, side)] = FoldedBranch(
                plane_index=plane_index,
                plane=float(plane),
                side=side,
                distance=distance,
                values=branch_values,
            )
    return branches


def periodic_sector_weights(
    coordinate: Array | Sequence[float],
    *,
    planes: Sequence[float],
    period: float,
) -> Array:
    """Partition every axial cell exactly among nearest plane/side sectors.

    The result has shape ``(n_planes, 2, n_axial)``; side index zero is ``-1``
    and side index one is ``+1``.  Cells exactly on a plane are split between
    its two sides, while cells exactly equidistant from two planes are split
    between the tied planes.  The weights sum to one for every axial cell.
    """

    z = np.asarray(coordinate, dtype=np.float64)
    plane_values = np.asarray(planes, dtype=np.float64)
    if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)):
        raise ValueError("coordinate must be a nonempty finite vector")
    if (
        plane_values.ndim != 1
        or plane_values.size == 0
        or not np.all(np.isfinite(plane_values))
    ):
        raise ValueError("planes must be a nonempty finite sequence")
    tolerance = 128.0 * np.finfo(np.float64).eps * max(period, 1.0)
    signed = np.stack(
        [
            signed_periodic_distance(z, plane, period)
            for plane in plane_values
        ],
        axis=0,
    )
    absolute = np.abs(signed)
    minimum = np.min(absolute, axis=0)
    tied = np.isclose(absolute, minimum[None, :], rtol=0.0, atol=tolerance)
    tie_count = np.sum(tied, axis=0)
    weights = np.zeros((plane_values.size, 2, z.size), dtype=np.float64)
    for plane_index in range(plane_values.size):
        share = np.where(tied[plane_index], 1.0 / tie_count, 0.0)
        negative = signed[plane_index] < -tolerance
        positive = signed[plane_index] > tolerance
        on_plane = ~(negative | positive)
        weights[plane_index, 0] = share * (
            negative.astype(np.float64) + 0.5 * on_plane
        )
        weights[plane_index, 1] = share * (
            positive.astype(np.float64) + 0.5 * on_plane
        )
    normalization = np.sum(weights, axis=(0, 1))
    if np.any(normalization <= 0.0):
        raise RuntimeError("periodic sector partition left an unassigned cell")
    weights /= normalization[None, None, :]
    return weights


def _subcell_extremum(
    coordinate: Array,
    values: Array,
    *,
    kind: str,
) -> tuple[float, float]:
    if coordinate.size == 0:
        return float("nan"), float("nan")
    index = int(np.argmin(values) if kind == "minimum" else np.argmax(values))
    best_x = float(coordinate[index])
    best_y = float(values[index])
    if coordinate.size < 3:
        return best_x, best_y
    start = min(max(index - 1, 0), coordinate.size - 3)
    x = coordinate[start : start + 3]
    y = values[start : start + 3]
    origin = float(x[1])
    quadratic, linear, constant = np.polyfit(x - origin, y, 2)
    correct_curvature = (
        quadratic > 0.0 if kind == "minimum" else quadratic < 0.0
    )
    if not correct_curvature or quadratic == 0.0:
        return best_x, best_y
    vertex_relative = -linear / (2.0 * quadratic)
    vertex = origin + vertex_relative
    if vertex < x[0] or vertex > x[-1]:
        return best_x, best_y
    vertex_value = (
        quadratic * vertex_relative * vertex_relative
        + linear * vertex_relative
        + constant
    )
    return float(vertex), float(vertex_value)


def _windowed_extremum(
    branch: FoldedBranch,
    window: tuple[float, float],
    *,
    kind: str,
) -> tuple[float, float]:
    lower, upper = (float(window[0]), float(window[1]))
    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower < 0.0
        or upper <= lower
    ):
        raise ValueError("feature windows must be finite and increasing")
    selected = (
        (branch.distance >= lower)
        & (branch.distance <= upper)
        & np.isfinite(branch.values)
    )
    if np.count_nonzero(selected) < 2:
        return float("nan"), float("nan")
    return _subcell_extremum(
        branch.distance[selected],
        branch.values[selected],
        kind=kind,
    )


def _first_positive_lobe_maximum(
    branch: FoldedBranch,
    *,
    groove_offset: float,
    groove_value: float,
    ridge_window: tuple[float, float],
    floor: float,
) -> tuple[float, float]:
    """Return the first resolved positive-lobe maximum after a groove root.

    The ridge window constrains the accepted maximum; it does not erase the
    path from the groove root to that window.  Consequently, a positive lobe
    before the window is not silently skipped in favor of a later, taller
    lobe.  A candidate must also have samples on both sides and be a local
    maximum, which rejects a monotone compensation signal whose largest value
    occurs only at the outward search boundary.
    """

    lower, upper = (float(ridge_window[0]), float(ridge_window[1]))
    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower < 0.0
        or upper <= lower
    ):
        raise ValueError("feature windows must be finite and increasing")
    if (
        not np.isfinite(groove_offset)
        or not np.isfinite(groove_value)
        or groove_value >= -floor
    ):
        return float("nan"), float("nan")

    scale = max(abs(groove_offset), abs(upper), 1.0)
    tolerance = 128.0 * np.finfo(np.float64).eps * scale
    selected = (
        (branch.distance > groove_offset + tolerance)
        & (branch.distance <= upper + tolerance)
        & np.isfinite(branch.values)
    )
    outward_distance = branch.distance[selected]
    outward_values = branch.values[selected]
    if outward_distance.size < 2:
        return float("nan"), float("nan")

    # Include the subcell groove root itself so that the first positive sample
    # necessarily represents an outward negative-to-positive sign crossing.
    distance = np.concatenate(
        (np.asarray([groove_offset]), outward_distance),
    )
    values = np.concatenate(
        (np.asarray([groove_value]), outward_values),
    )
    positive = np.flatnonzero(values[1:] > floor)
    if positive.size == 0:
        return float("nan"), float("nan")
    lobe_start = int(positive[0] + 1)
    nonpositive_after = np.flatnonzero(
        values[lobe_start + 1 :] <= floor
    )
    lobe_stop = (
        int(lobe_start + 1 + nonpositive_after[0])
        if nonpositive_after.size
        else int(values.size)
    )

    lobe_indices = np.arange(lobe_start, lobe_stop, dtype=np.int64)
    if lobe_indices.size == 0:
        return float("nan"), float("nan")
    peak_index = int(lobe_indices[np.argmax(values[lobe_indices])])

    # A maximum at either sampled search boundary has no resolved descending
    # and ascending neighborhood.  It is not evidence of a flanking ridge.
    if peak_index <= 0 or peak_index >= values.size - 1:
        return float("nan"), float("nan")
    peak_value = values[peak_index]
    if not (
        peak_value >= values[peak_index - 1]
        and peak_value >= values[peak_index + 1]
        and (
            peak_value > values[peak_index - 1]
            or peak_value > values[peak_index + 1]
        )
    ):
        return float("nan"), float("nan")
    if not (
        distance[peak_index] > lower + tolerance
        and distance[peak_index] < upper - tolerance
    ):
        return float("nan"), float("nan")

    return _subcell_extremum(
        distance[peak_index - 1 : peak_index + 2],
        values[peak_index - 1 : peak_index + 2],
        kind="maximum",
    )


def measure_groove_ridge_branches(
    coordinate: Array | Sequence[float],
    delta_radius: Array | Sequence[float],
    *,
    planes: Sequence[float],
    period: float,
    groove_window: tuple[float, float],
    ridge_window: tuple[float, float],
    floor: float = 0.0,
    maximum_distance: float | None = None,
) -> dict[BranchKey, BranchFeature]:
    """Measure a negative root and following positive ridge on every branch."""

    if not np.isfinite(floor) or floor < 0.0:
        raise ValueError("floor must be finite and nonnegative")
    if ridge_window[0] < groove_window[0]:
        raise ValueError("ridge search must not begin before the groove search")
    branches = fold_periodic_profile(
        coordinate,
        delta_radius,
        planes=planes,
        period=period,
        maximum_distance=maximum_distance,
    )
    features: dict[BranchKey, BranchFeature] = {}
    for key, branch in branches.items():
        groove_offset, groove_value = _windowed_extremum(
            branch,
            groove_window,
            kind="minimum",
        )
        ridge_offset, ridge_value = _first_positive_lobe_maximum(
            branch,
            groove_offset=groove_offset,
            groove_value=groove_value,
            ridge_window=ridge_window,
            floor=floor,
        )
        groove_depth = max(-groove_value, 0.0)
        ridge_height = max(ridge_value, 0.0)
        ratio = (
            float(ridge_height / groove_depth)
            if groove_depth > floor
            else float("nan")
        )
        features[key] = BranchFeature(
            plane_index=branch.plane_index,
            plane=branch.plane,
            side=branch.side,
            groove_value=groove_value,
            groove_depth=groove_depth,
            groove_offset=groove_offset,
            ridge_value=ridge_value,
            ridge_height=ridge_height,
            ridge_offset_from_plane=ridge_offset,
            ridge_offset_from_root=ridge_offset - groove_offset,
            height_to_depth_ratio=ratio,
            has_negative_groove=bool(groove_value < -floor),
            has_positive_ridge=bool(ridge_value > floor),
        )
    return features


def lobe_mass_balance(
    delta_area: Array | Sequence[float],
    axial_spacing: float,
    *,
    sector_weights: Array,
    remove_uniform_residual: bool = True,
) -> LobeBalance:
    """Integrate positive/negative lobes while preserving raw mass accounting."""

    profile = np.asarray(delta_area, dtype=np.float64)
    weights = np.asarray(sector_weights, dtype=np.float64)
    if profile.ndim != 1 or profile.size == 0 or not np.all(np.isfinite(profile)):
        raise ValueError("delta_area must be a nonempty finite vector")
    if (
        weights.ndim != 3
        or weights.shape[1] != 2
        or weights.shape[2] != profile.size
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
    ):
        raise ValueError(
            "sector_weights must have finite shape (n_planes, 2, n_axial)"
        )
    if not np.allclose(
        np.sum(weights, axis=(0, 1)),
        1.0,
        rtol=0.0,
        atol=2.0e-15,
    ):
        raise ValueError("sector weights must sum to one at every axial cell")
    if not np.isfinite(axial_spacing) or axial_spacing <= 0.0:
        raise ValueError("axial_spacing must be finite and positive")

    raw_total = float(np.sum(profile, dtype=np.float64) * axial_spacing)
    uniform = float(np.mean(profile)) if remove_uniform_residual else 0.0
    corrected = profile - uniform
    removed_integral = float(uniform * profile.size * axial_spacing)
    positive_density = np.maximum(corrected, 0.0)
    negative_density = np.maximum(-corrected, 0.0)
    positive = float(np.sum(positive_density) * axial_spacing)
    negative = float(np.sum(negative_density) * axial_spacing)
    ratio_floor = robust_internal_floor(
        np.asarray([positive, negative]),
        relative_floor=256.0 * np.finfo(np.float64).eps,
    )
    ratio = float(positive / negative) if negative > ratio_floor else float("nan")

    sector_reports: dict[BranchKey, SectorLobeBalance] = {}
    sector_net_sum = 0.0
    for plane_index in range(weights.shape[0]):
        for side_index, side in enumerate((-1, 1)):
            weight = weights[plane_index, side_index]
            sector_positive = float(
                np.sum(weight * positive_density) * axial_spacing
            )
            sector_negative = float(
                np.sum(weight * negative_density) * axial_spacing
            )
            sector_ratio = (
                float(sector_positive / sector_negative)
                if sector_negative > ratio_floor
                else float("nan")
            )
            sector_net = sector_positive - sector_negative
            sector_net_sum += sector_net
            sector_reports[(plane_index, side)] = SectorLobeBalance(
                plane_index=plane_index,
                side=side,
                positive_integral=sector_positive,
                negative_magnitude_integral=sector_negative,
                net_integral=sector_net,
                positive_to_negative_ratio=sector_ratio,
            )
    corrected_total = float(np.sum(corrected) * axial_spacing)
    return LobeBalance(
        raw_total_integral=raw_total,
        uniform_residual=uniform,
        removed_uniform_integral=removed_integral,
        corrected_total_integral=corrected_total,
        positive_integral=positive,
        negative_magnitude_integral=negative,
        positive_to_negative_ratio=ratio,
        corrected_profile=corrected,
        sectors=sector_reports,
        sector_net_sum=float(sector_net_sum),
        sector_partition_error=float(sector_net_sum - corrected_total),
    )


def area_floor_to_radius(
    area_floor: float | Array,
    reference_radius: float | Array,
) -> Array:
    """Convert an area perturbation floor to radius units via ``2*pi*R``."""

    area = np.asarray(area_floor, dtype=np.float64)
    radius = np.asarray(reference_radius, dtype=np.float64)
    if np.any(~np.isfinite(area)) or np.any(area < 0.0):
        raise ValueError("area_floor must be finite and nonnegative")
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("reference_radius must be finite and positive")
    return area / (2.0 * np.pi * radius)


def _maximum_median_deviation(values: Array | Sequence[float]) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size <= 1:
        return 0.0
    return float(np.max(np.abs(finite - np.median(finite))))


def _contour_level_spread(values: Array | Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        finite = array[np.isfinite(array)]
        return float(np.ptp(finite)) if finite.size > 1 else 0.0
    if array.ndim != 2:
        raise ValueError(
            "contour ridge heights must have shape (n_levels, n_branches)"
        )
    spreads: list[float] = []
    for branch in array.T:
        finite = branch[np.isfinite(branch)]
        if finite.size > 1:
            spreads.append(float(np.ptp(finite)))
    return max(spreads, default=0.0)


def internal_radius_floor(
    *,
    control_radius: Array | Sequence[float],
    plane_side_ridge_heights: Array | Sequence[float],
    contour_level_ridge_heights: Array | Sequence[float],
    uniform_area_residual: float,
    reference_radius: float,
    representation_scale: float | None = None,
    representation_radius_floor: float | None = None,
    contour_radius_signal: Array | Sequence[float] | None = None,
    area_radius_signal: Array | Sequence[float] | None = None,
    consistency_tolerance: float | None = None,
) -> InternalFloorReport:
    """Combine independently measured floors without mixing physical units.

    ``contour_level_ridge_heights`` is already in radius units.  The paired
    uniform area residual is converted with ``2*pi*reference_radius`` before
    comparison.  A measured mixed-precision floor can be supplied directly as
    ``representation_radius_floor``; otherwise a conservative float64
    arithmetic floor is used.  If contour- and conserved-area radius signals
    are supplied, their consistency is reported separately rather than folded
    into an area metric.
    """

    if not np.isfinite(reference_radius) or reference_radius <= 0.0:
        raise ValueError("reference_radius must be finite and positive")
    if not np.isfinite(uniform_area_residual):
        raise ValueError("uniform_area_residual must be finite")
    if representation_scale is None:
        representation_scale = reference_radius
    if not np.isfinite(representation_scale) or representation_scale <= 0.0:
        raise ValueError("representation_scale must be finite and positive")
    if representation_radius_floor is not None and (
        not np.isfinite(representation_radius_floor)
        or representation_radius_floor < 0.0
    ):
        raise ValueError(
            "representation_radius_floor must be finite and nonnegative"
        )

    components = {
        "control_roughness": _maximum_median_deviation(control_radius),
        "plane_side_scatter": _maximum_median_deviation(
            plane_side_ridge_heights
        ),
        "contour_level_spread": _contour_level_spread(
            contour_level_ridge_heights
        ),
        "uniform_mass_residual": float(
            area_floor_to_radius(
                abs(uniform_area_residual),
                reference_radius,
            )
        ),
        "floating_point_representation": (
            float(representation_radius_floor)
            if representation_radius_floor is not None
            else float(
                64.0
                * np.finfo(np.float64).eps
                * max(reference_radius, representation_scale)
            )
        ),
    }
    largest_component = max(components, key=components.__getitem__)
    largest_floor = float(components[largest_component])

    consistent: bool | None = None
    maximum_difference: float | None = None
    if (contour_radius_signal is None) != (area_radius_signal is None):
        raise ValueError(
            "both contour_radius_signal and area_radius_signal are required"
        )
    if contour_radius_signal is not None and area_radius_signal is not None:
        contour = np.asarray(contour_radius_signal, dtype=np.float64)
        area = np.asarray(area_radius_signal, dtype=np.float64)
        if contour.shape != area.shape:
            raise ValueError("contour and area radius signals must have equal shape")
        finite = np.isfinite(contour) & np.isfinite(area)
        maximum_difference = (
            float(np.max(np.abs(contour[finite] - area[finite])))
            if np.any(finite)
            else float("inf")
        )
        tolerance = (
            max(largest_floor, 0.0)
            if consistency_tolerance is None
            else float(consistency_tolerance)
        )
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("consistency_tolerance must be finite and nonnegative")
        consistent = bool(maximum_difference <= tolerance)
    return InternalFloorReport(
        components=components,
        largest_component=largest_component,
        largest_floor=largest_floor,
        contour_area_consistent=consistent,
        contour_area_maximum_difference=maximum_difference,
    )
