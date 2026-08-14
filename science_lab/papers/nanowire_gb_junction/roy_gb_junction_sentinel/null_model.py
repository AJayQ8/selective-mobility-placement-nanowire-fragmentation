"""Preregistered linear-profile null for the S0--S4 sentinel.

The primary radius observable is always the *raw* phase-excess-area radius.
Gaussian smoothing is used only to select an axial site.  It is never used in
the additive profile, the residual, or the dimensionless breakup coordinate
``rho = r_raw(site) / r_far_raw``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

from .contract import sha256_path


GRID_SPACING = 0.5
AXIAL_PERIOD = 384.0
RADIUS = 6.0
INTERFACE_WIDTH = float(np.sqrt(8.0))
RAYLEIGH_WAVELENGTH = float(2.0 * np.sqrt(2.0) * np.pi * RADIUS)
SITE_SMOOTHING_SIGMA_CELLS = INTERFACE_WIDTH / GRID_SPACING
SITE_SEARCH_BOUNDS = (
    RADIUS + 2.0 * INTERFACE_WIDTH,
    1.25 * RAYLEIGH_WAVELENGTH,
)
RHO_SEARCH_BOUNDS = (
    RADIUS + 2.0 * INTERFACE_WIDTH,
    0.5 * AXIAL_PERIOD - 0.5 * GRID_SPACING,
)
FAR_BASELINE_BOUNDS = (
    1.5 * RAYLEIGH_WAVELENGTH,
    2.0 * RAYLEIGH_WAVELENGTH,
)

S0_PROFILE_KEYS = {
    "step",
    "junction_relative_z",
    "contour_radius_c0p5_raw",
    "contour_valid_fraction_c0p5",
    "raw_phase_excess_area_radius",
}
S4_PAIR_PROFILE_KEYS = {
    "step",
    "gb_relative_z",
    "gb_off_raw_phase_excess_area_radius",
    "gb_on_raw_phase_excess_area_radius",
}


@dataclass(frozen=True)
class S0RadiusProfile:
    """One S0 diagnostic profile in the junction-centred coordinate frame."""

    step: int
    z: np.ndarray
    contour_radius_c0p5_raw: np.ndarray
    contour_valid_fraction_c0p5: np.ndarray
    raw_phase_excess_area_radius: np.ndarray
    path: Path
    sha256: str


@dataclass(frozen=True)
class PairedGBRadiusProfile:
    """Same-time isolated-wire S4 GB-off and GB-on raw radius profiles."""

    step: int
    z: np.ndarray
    gb_off_raw_phase_excess_area_radius: np.ndarray
    gb_on_raw_phase_excess_area_radius: np.ndarray
    path: Path
    sha256: str

    @property
    def delta_raw_phase_excess_area_radius(self) -> np.ndarray:
        return (
            self.gb_on_raw_phase_excess_area_radius
            - self.gb_off_raw_phase_excess_area_radius
        )


def _scalar_step(archive: Mapping[str, np.ndarray]) -> int:
    value = np.asarray(archive["step"])
    if value.size != 1:
        raise ValueError("profile step must be a scalar")
    measured = float(value.reshape(-1)[0])
    if not np.isfinite(measured) or measured < 0.0 or not measured.is_integer():
        raise ValueError("profile step must be a nonnegative integer")
    return int(measured)


def _validate_periodic_coordinate(
    coordinate: np.ndarray,
    *,
    period: float = AXIAL_PERIOD,
    spacing: float = GRID_SPACING,
) -> np.ndarray:
    values = np.asarray(coordinate, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size < 4
        or not np.all(np.isfinite(values))
        or not np.all(np.diff(values) > 0.0)
    ):
        raise ValueError("axial coordinate must be finite and strictly increasing")
    if not np.allclose(np.diff(values), spacing, rtol=0.0, atol=1.0e-13):
        raise ValueError("axial coordinate is not on the frozen 0.5 grid")
    if not np.isclose(values.size * spacing, period, rtol=0.0, atol=1.0e-12):
        raise ValueError("axial coordinate does not span the frozen period")
    return values


def _validate_radius(
    name: str,
    values: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    radius = np.asarray(values, dtype=np.float64)
    if radius.shape != shape or not np.all(np.isfinite(radius)):
        raise ValueError(f"{name} must be a finite axial profile")
    if np.any(radius < 0.0):
        raise ValueError(f"{name} contains a negative area-equivalent radius")
    return radius


def load_s0_profile(
    path: Path,
    *,
    expected_sha256: str | None = None,
    require_resolved_contour: bool = True,
) -> S0RadiusProfile:
    """Load a hash-pinned S0 profile artifact without clipping its radii."""

    path = path.resolve()
    measured_hash = sha256_path(path)
    if expected_sha256 is not None and measured_hash != expected_sha256:
        raise ValueError("S0 profile SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(S0_PROFILE_KEYS - set(archive.files))
        if missing:
            raise ValueError(f"S0 profile artifact is missing keys: {missing}")
        step = _scalar_step(archive)
        z = _validate_periodic_coordinate(archive["junction_relative_z"])
        if not np.isclose(z[0] + z[-1], 0.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("S0 coordinate must be centred on the junction")
        raw_radius = _validate_radius(
            "S0 raw phase-excess-area radius",
            archive["raw_phase_excess_area_radius"],
            z.shape,
        )
        contour = np.asarray(archive["contour_radius_c0p5_raw"], dtype=np.float64)
        valid = np.asarray(
            archive["contour_valid_fraction_c0p5"], dtype=np.float64
        )
    if contour.shape != z.shape or valid.shape != z.shape:
        raise ValueError("S0 contour arrays do not match the axial coordinate")
    if not np.all(np.isfinite(valid)) or np.any((valid < 0.0) | (valid > 1.0)):
        raise ValueError("S0 contour valid fractions are invalid")
    if require_resolved_contour:
        required = (
            (z >= SITE_SEARCH_BOUNDS[0])
            & (z <= FAR_BASELINE_BOUNDS[1])
        )
        if (
            np.count_nonzero(required) < 5
            or not np.all(np.isfinite(contour[required]))
            or np.any(contour[required] <= 0.0)
            or np.min(valid[required]) < 0.95
        ):
            raise ValueError(
                "S0 +z c=0.5 contour analysis window is not fully resolved"
            )
    return S0RadiusProfile(
        step=step,
        z=z,
        contour_radius_c0p5_raw=contour,
        contour_valid_fraction_c0p5=valid,
        raw_phase_excess_area_radius=raw_radius,
        path=path,
        sha256=measured_hash,
    )


def load_paired_gb_profile(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> PairedGBRadiusProfile:
    """Load a hash-pinned, same-initializer S4 GB-on/GB-off profile pair."""

    path = path.resolve()
    measured_hash = sha256_path(path)
    if expected_sha256 is not None and measured_hash != expected_sha256:
        raise ValueError("paired S4 profile SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(S4_PAIR_PROFILE_KEYS - set(archive.files))
        if missing:
            raise ValueError(f"paired S4 profile artifact is missing keys: {missing}")
        step = _scalar_step(archive)
        z = _validate_periodic_coordinate(archive["gb_relative_z"])
        if float(np.min(np.abs(z))) > 1.0e-13:
            raise ValueError("paired S4 coordinate must contain the primary GB at zero")
        off = _validate_radius(
            "S4 GB-off raw phase-excess-area radius",
            archive["gb_off_raw_phase_excess_area_radius"],
            z.shape,
        )
        on = _validate_radius(
            "S4 GB-on raw phase-excess-area radius",
            archive["gb_on_raw_phase_excess_area_radius"],
            z.shape,
        )
    return PairedGBRadiusProfile(
        step=step,
        z=z,
        gb_off_raw_phase_excess_area_radius=off,
        gb_on_raw_phase_excess_area_radius=on,
        path=path,
        sha256=measured_hash,
    )


def fixed_site_smoothing(values: np.ndarray) -> np.ndarray:
    """Apply the only smoothing authorized by the null contract."""

    profile = np.asarray(values, dtype=np.float64)
    if profile.ndim != 1 or not np.all(np.isfinite(profile)):
        raise ValueError("site-selection profile must be finite and one-dimensional")
    return np.asarray(
        ndimage.gaussian_filter1d(
            profile,
            sigma=SITE_SMOOTHING_SIGMA_CELLS,
            mode="wrap",
        ),
        dtype=np.float64,
    )


def _quadratic_extremum(
    coordinate: np.ndarray,
    values: np.ndarray,
    index: int,
    *,
    maximum: bool,
) -> tuple[float, float]:
    if index <= 0 or index >= values.size - 1:
        return float(coordinate[index]), float(values[index])
    x = coordinate[index - 1 : index + 2]
    y = values[index - 1 : index + 2]
    coefficients = np.polyfit(x, y, 2)
    curvature = float(coefficients[0])
    if (maximum and curvature >= 0.0) or (not maximum and curvature <= 0.0):
        return float(coordinate[index]), float(values[index])
    vertex = float(-coefficients[1] / (2.0 * curvature))
    if vertex < x[0] or vertex > x[-1]:
        return float(coordinate[index]), float(values[index])
    return vertex, float(np.polyval(coefficients, vertex))


def _side_arrays(
    z: np.ndarray,
    values: np.ndarray,
    *,
    side: int,
) -> tuple[np.ndarray, np.ndarray]:
    if side not in (-1, 1):
        raise ValueError("side must be -1 or +1")
    if side > 0:
        mask = z > 0.0
        return z[mask], values[mask]
    mask = z < 0.0
    return -z[mask][::-1], values[mask][::-1]


def side_rho_measurement(
    z: np.ndarray,
    raw_radius: np.ndarray,
    *,
    side: int,
    allow_negative: bool = False,
) -> dict[str, float | int]:
    """Select a side minimum after smoothing, then evaluate ``rho`` raw."""

    coordinate = _validate_periodic_coordinate(z)
    raw = np.asarray(raw_radius, dtype=np.float64)
    if raw.shape != coordinate.shape or not np.all(np.isfinite(raw)):
        raise ValueError("raw radius must be a finite axial profile")
    if not allow_negative and np.any(raw < 0.0):
        raise ValueError("physical raw radius cannot be negative")
    smoothed = fixed_site_smoothing(raw)
    distance, smooth_side = _side_arrays(coordinate, smoothed, side=side)
    _, raw_side = _side_arrays(coordinate, raw, side=side)
    search = (
        (distance >= RHO_SEARCH_BOUNDS[0])
        & (distance <= RHO_SEARCH_BOUNDS[1])
    )
    far = (
        (distance >= FAR_BASELINE_BOUNDS[0])
        & (distance <= FAR_BASELINE_BOUNDS[1])
    )
    if np.count_nonzero(search) < 5 or np.count_nonzero(far) < 5:
        raise ValueError("rho search or far-baseline window is unresolved")
    indices = np.flatnonzero(search)
    index = int(indices[int(np.argmin(smooth_side[search]))])
    position, smoothed_minimum = _quadratic_extremum(
        distance, smooth_side, index, maximum=False
    )
    raw_at_site = float(np.interp(position, distance, raw_side))
    far_raw = float(np.median(raw_side[far]))
    if not far_raw > 0.0:
        raise ValueError("raw far-arm radius is not positive")
    return {
        "side": int(side),
        "site_distance": position,
        "site_smoothed_radius_for_selection_only": smoothed_minimum,
        "site_raw_phase_excess_area_radius": raw_at_site,
        "far_raw_phase_excess_area_radius": far_raw,
        "rho_raw": raw_at_site / far_raw,
    }


def contour_thinning_measurement(profile: S0RadiusProfile) -> dict[str, float]:
    """Measure the preregistered +z c=0.5 thinning minimum."""

    required = (
        (profile.z >= SITE_SEARCH_BOUNDS[0])
        & (profile.z <= FAR_BASELINE_BOUNDS[1])
    )
    distance = profile.z[required]
    raw_side = profile.contour_radius_c0p5_raw[required]
    validity = profile.contour_valid_fraction_c0p5[required]
    if (
        distance.size < 5
        or not np.all(np.isfinite(raw_side))
        or np.any(raw_side <= 0.0)
        or not np.all(np.isfinite(validity))
        or np.min(validity) < 0.95
    ):
        raise ValueError("alignment +z c=0.5 contour window is unresolved")
    side = np.asarray(
        ndimage.gaussian_filter1d(
            raw_side,
            sigma=SITE_SMOOTHING_SIGMA_CELLS,
            mode="nearest",
        ),
        dtype=np.float64,
    )
    search = (
        (distance >= SITE_SEARCH_BOUNDS[0])
        & (distance <= SITE_SEARCH_BOUNDS[1])
    )
    far = (
        (distance >= FAR_BASELINE_BOUNDS[0])
        & (distance <= FAR_BASELINE_BOUNDS[1])
    )
    if np.count_nonzero(search) < 5 or np.count_nonzero(far) < 5:
        raise ValueError("contour thinning windows are unresolved")
    indices = np.flatnonzero(search)
    index = int(indices[int(np.argmin(side[search]))])
    position, minimum = _quadratic_extremum(
        distance, side, index, maximum=False
    )
    baseline = float(np.median(side[far]))
    depth = baseline - minimum
    if not depth > 0.0:
        raise ValueError("c=0.5 thinning minimum is not below the far arm")
    half_level = baseline - 0.5 * depth

    def crossing(direction: int) -> float:
        current = index
        while 0 <= current + direction < side.size:
            other = current + direction
            first = side[current] - half_level
            second = side[other] - half_level
            if first * second <= 0.0 and side[other] >= half_level:
                denominator = side[other] - side[current]
                if denominator == 0.0:
                    return float(distance[other])
                fraction = (half_level - side[current]) / denominator
                return float(
                    distance[current]
                    + fraction * (distance[other] - distance[current])
                )
            current = other
        raise ValueError("c=0.5 thinning half-depth crossing is unresolved")

    left = crossing(-1)
    right = crossing(1)
    minimum_valid = float(np.min(validity))
    return {
        "z_min": position,
        "minimum_smoothed_contour_radius": minimum,
        "far_smoothed_contour_radius": baseline,
        "depletion_depth": depth,
        "half_depth_left": left,
        "half_depth_right": right,
        "minimum_contour_valid_fraction": minimum_valid,
        "smoothing_sigma_cells": SITE_SMOOTHING_SIGMA_CELLS,
        "smoothing_boundary_mode": "nearest_on_resolved_plus_arm_window",
        "resolved_plus_arm_window": [
            SITE_SEARCH_BOUNDS[0],
            FAR_BASELINE_BOUNDS[1],
        ],
    }


def calibrate_rho_criterion(
    last_no_gap: S0RadiusProfile,
    first_complete_gap_candidate: S0RadiusProfile,
) -> dict[str, Any]:
    """Freeze ``rho_c`` midway between the two S0 event-bracketing records."""

    if first_complete_gap_candidate.step <= last_no_gap.step:
        raise ValueError("S0 complete-gap candidate must follow the no-gap record")
    if not np.array_equal(last_no_gap.z, first_complete_gap_candidate.z):
        raise ValueError("S0 event-bracketing coordinates differ")

    records: dict[str, Any] = {}
    minima: list[float] = []
    for name, profile in (
        ("last_no_gap", last_no_gap),
        ("first_complete_gap_candidate", first_complete_gap_candidate),
    ):
        sides = {
            "minus": side_rho_measurement(
                profile.z,
                profile.raw_phase_excess_area_radius,
                side=-1,
            ),
            "plus": side_rho_measurement(
                profile.z,
                profile.raw_phase_excess_area_radius,
                side=1,
            ),
        }
        rho_values = [float(item["rho_raw"]) for item in sides.values()]
        symmetry_mean = float(np.mean(rho_values))
        minima.append(symmetry_mean)
        records[name] = {
            "step": profile.step,
            "sides": sides,
            "symmetry_mean_minimum_rho_raw": symmetry_mean,
            "lower_rho_side": (
                "minus" if rho_values[0] <= rho_values[1] else "plus"
            ),
            "branch_symmetry_spread": abs(rho_values[0] - rho_values[1]),
        }
    last_rho, candidate_rho = minima
    if not candidate_rho < last_rho:
        raise ValueError("S0 rho did not decrease across the complete-gap bracket")
    threshold = 0.5 * (last_rho + candidate_rho)
    return {
        "definition": "rho = r_min_raw_phase_excess_area / r_far_raw_phase_excess_area",
        "site_selection": (
            "fixed periodic Gaussian smoothing, sigma=sqrt(8)/0.5 cells; "
            "smoothing selects position only"
        ),
        "threshold_rule": (
            "midpoint of the two-arm symmetry-mean S0 minimum rho at "
            "last-no-gap and first complete-gap candidate records"
        ),
        "search_interval": list(RHO_SEARCH_BOUNDS),
        "far_window": list(FAR_BASELINE_BOUNDS),
        "records": records,
        "rho_critical": threshold,
        "maximum_branch_symmetry_spread": max(
            float(records["last_no_gap"]["branch_symmetry_spread"]),
            float(
                records["first_complete_gap_candidate"][
                    "branch_symmetry_spread"
                ]
            ),
        ),
    }


def periodic_linear_interpolate(
    coordinate: np.ndarray,
    values: np.ndarray,
    query: np.ndarray | float,
    *,
    period: float = AXIAL_PERIOD,
) -> np.ndarray:
    """Linearly interpolate a complete uniform periodic profile."""

    x = _validate_periodic_coordinate(coordinate, period=period)
    y = np.asarray(values, dtype=np.float64)
    if y.shape != x.shape or not np.all(np.isfinite(y)):
        raise ValueError("periodic profile values are invalid")
    locations = np.asarray(query, dtype=np.float64)
    origin = float(x[0])
    wrapped = (locations - origin) % period + origin
    extended_x = np.concatenate((x, np.asarray([origin + period])))
    extended_y = np.concatenate((y, y[:1]))
    return np.asarray(np.interp(wrapped, extended_x, extended_y), dtype=np.float64)


def periodic_gb_planes(
    spacing: float,
    *,
    period: float = AXIAL_PERIOD,
) -> tuple[float, float]:
    """Return the primary and image GB coordinates relative to the junction."""

    if not np.isfinite(spacing):
        raise ValueError("GB spacing must be finite")

    def wrapped(value: float) -> float:
        return float((value + 0.5 * period) % period - 0.5 * period)

    return wrapped(float(spacing)), wrapped(float(spacing) + 0.5 * period)


def build_null_profile(
    s0: S0RadiusProfile,
    paired_gb: PairedGBRadiusProfile,
    *,
    gb_spacing: float,
) -> dict[str, Any]:
    """Build ``r_null = r_S0 + (r_S4,on - r_S4,off)`` exactly in radius space."""

    if s0.step != paired_gb.step:
        raise ValueError("S0 and paired S4 profiles must be at the exact same step")
    delta = paired_gb.delta_raw_phase_excess_area_radius
    shifted = periodic_linear_interpolate(
        paired_gb.z,
        delta,
        s0.z - float(gb_spacing),
    )
    raw_null = s0.raw_phase_excess_area_radius + shifted
    if not np.all(np.isfinite(raw_null)):
        raise ValueError("linear null produced a nonfinite raw radius")
    primary, image = periodic_gb_planes(gb_spacing)
    return {
        "step": s0.step,
        "junction_relative_z": s0.z,
        "s0_raw_phase_excess_area_radius": s0.raw_phase_excess_area_radius,
        "shifted_paired_s4_delta_raw_radius": shifted,
        "null_raw_phase_excess_area_radius": raw_null,
        "minimum_null_raw_radius": float(np.min(raw_null)),
        "null_profile_passes_zero": bool(np.any(raw_null <= 0.0)),
        "primary_gb_coordinate": primary,
        "image_gb_coordinate": image,
        "equation": "r_null(z,t;s)=r_S0(z,t)+r_S4_on(z-s,t)-r_S4_off(z-s,t)",
    }


def evaluate_null_profile(
    s0: S0RadiusProfile,
    paired_gb: PairedGBRadiusProfile,
    *,
    gb_spacing: float,
    rho_critical: float,
    coupled_raw_phase_excess_area_radius: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate one preregistered null profile and an optional coupled residual."""

    if not np.isfinite(rho_critical) or not 0.0 < rho_critical < 1.0:
        raise ValueError("rho_critical must lie strictly between zero and one")
    built = build_null_profile(s0, paired_gb, gb_spacing=gb_spacing)
    null_radius = np.asarray(
        built["null_raw_phase_excess_area_radius"], dtype=np.float64
    )
    sides = {
        "minus": side_rho_measurement(
            s0.z, null_radius, side=-1, allow_negative=True
        ),
        "plus": side_rho_measurement(
            s0.z, null_radius, side=1, allow_negative=True
        ),
    }
    minimum_rho = min(float(item["rho_raw"]) for item in sides.values())
    result: dict[str, Any] = {
        "step": s0.step,
        "gb_spacing": float(gb_spacing),
        "primary_gb_coordinate": built["primary_gb_coordinate"],
        "image_gb_coordinate": built["image_gb_coordinate"],
        "sides": sides,
        "minimum_null_rho_raw": minimum_rho,
        "rho_critical": float(rho_critical),
        "null_predicts_complete_gap": minimum_rho <= rho_critical,
        "minimum_null_raw_radius": built["minimum_null_raw_radius"],
        "null_profile_passes_zero": built["null_profile_passes_zero"],
        "residual_definition": "r_coupled_raw-r_null_raw",
        "smoothing_used_for_residual": False,
    }
    if coupled_raw_phase_excess_area_radius is not None:
        coupled = _validate_radius(
            "coupled raw phase-excess-area radius",
            coupled_raw_phase_excess_area_radius,
            s0.z.shape,
        )
        residual = coupled - null_radius
        delta = np.asarray(
            built["shifted_paired_s4_delta_raw_radius"], dtype=np.float64
        )
        delta_l2 = float(np.linalg.norm(delta))
        delta_linf = float(np.max(np.abs(delta)))
        far_mask = (
            (np.abs(s0.z) >= FAR_BASELINE_BOUNDS[0])
            & (np.abs(s0.z) <= FAR_BASELINE_BOUNDS[1])
        )
        far_radius = float(
            np.median(s0.raw_phase_excess_area_radius[far_mask])
        )
        depletion_weight = np.maximum(
            far_radius - s0.raw_phase_excess_area_radius,
            0.0,
        )
        weight_sum = float(np.sum(depletion_weight, dtype=np.float64))
        result["coupled_minus_null_raw_radius"] = residual
        result["residual_l2"] = float(np.linalg.norm(residual))
        result["residual_linf"] = float(np.max(np.abs(residual)))
        result["shifted_paired_s4_delta_l2"] = delta_l2
        result["shifted_paired_s4_delta_linf"] = delta_linf
        result["residual_normalization_defined"] = bool(
            delta_l2 > 0.0 and delta_linf > 0.0
        )
        result["residual_l2_over_shifted_delta_l2"] = (
            float(np.linalg.norm(residual) / delta_l2)
            if delta_l2 > 0.0
            else None
        )
        result["residual_linf_over_shifted_delta_linf"] = (
            float(np.max(np.abs(residual)) / delta_linf)
            if delta_linf > 0.0
            else None
        )
        result["signed_depletion_weighted_residual"] = (
            float(
                np.sum(
                    depletion_weight * residual,
                    dtype=np.float64,
                )
                / weight_sum
            )
            if weight_sum > 0.0
            else None
        )
        result["depletion_weight_definition"] = (
            "max(S0_far_raw-S0_raw(z),0)"
        )
        result["smoothing_used_for_depletion_weight"] = False
    return result


def earliest_threshold_crossing(
    times: Sequence[float],
    rho_values: Sequence[float],
    *,
    rho_critical: float,
) -> float | None:
    """Return the first downward ``rho_c`` crossing by linear interpolation."""

    t = np.asarray(times, dtype=np.float64)
    rho = np.asarray(rho_values, dtype=np.float64)
    if (
        t.ndim != 1
        or rho.shape != t.shape
        or t.size == 0
        or not np.all(np.isfinite(t))
        or not np.all(np.isfinite(rho))
        or not np.all(np.diff(t) > 0.0)
    ):
        raise ValueError("threshold-crossing samples are invalid")
    if not np.isfinite(rho_critical):
        raise ValueError("rho_critical must be finite")
    if rho[0] <= rho_critical:
        return float(t[0])
    for index in range(1, t.size):
        if rho[index] <= rho_critical < rho[index - 1]:
            fraction = (
                (rho[index - 1] - rho_critical)
                / (rho[index - 1] - rho[index])
            )
            return float(t[index - 1] + fraction * (t[index] - t[index - 1]))
    return None


def evaluate_null_trajectory(
    s0_profiles: Sequence[S0RadiusProfile],
    paired_gb_profiles: Sequence[PairedGBRadiusProfile],
    *,
    gb_spacing: float,
    rho_critical: float,
    coupled_raw_profiles: Mapping[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Evaluate exact common-time null profiles and freeze the predicted crossing."""

    s0_by_step = {item.step: item for item in s0_profiles}
    gb_by_step = {item.step: item for item in paired_gb_profiles}
    if len(s0_by_step) != len(s0_profiles) or len(gb_by_step) != len(
        paired_gb_profiles
    ):
        raise ValueError("null trajectory contains duplicate steps")
    if set(s0_by_step) != set(gb_by_step):
        raise ValueError("S0 and paired S4 null trajectories need exact common steps")
    steps = sorted(s0_by_step)
    records: list[dict[str, Any]] = []
    for step in steps:
        coupled = (
            None
            if coupled_raw_profiles is None
            else coupled_raw_profiles.get(step)
        )
        records.append(
            evaluate_null_profile(
                s0_by_step[step],
                gb_by_step[step],
                gb_spacing=gb_spacing,
                rho_critical=rho_critical,
                coupled_raw_phase_excess_area_radius=coupled,
            )
        )
    crossing = earliest_threshold_crossing(
        steps,
        [float(record["minimum_null_rho_raw"]) for record in records],
        rho_critical=rho_critical,
    )
    return {
        "definition": {
            "equation": (
                "r_null(z,t;s)=r_S0(z,t)+"
                "[r_S4_on(z-s,t)-r_S4_off(z-s,t)]"
            ),
            "radius_observable": "raw_phase_excess_area_radius",
            "residual": "raw_coupled_radius_minus_raw_null_radius",
            "site_selection_smoothing_sigma_cells": SITE_SMOOTHING_SIGMA_CELLS,
            "smoothing_enters_radius_or_residual": False,
            "periodic_plane_accounting": "shifted full periodic paired-S4 profile",
            "threshold_crossing_interpolation": "earliest_linear",
        },
        "gb_spacing": float(gb_spacing),
        "rho_critical": float(rho_critical),
        "records": records,
        "earliest_predicted_crossing_time": crossing,
    }
