#!/usr/bin/env python3
"""Audit position-scan mechanisms from saved one-dimensional profiles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

from ..roy_fixed_gb_bridge.provenance import sha256_path
from ..roy_gb_junction_sentinel.storage import atomic_json
from .geometry import signed_arm_window
from . import position_scan_protocol as protocol


SOURCE_PATH = Path(__file__).resolve()
DIRECTORY = SOURCE_PATH.parent
RESULTS = DIRECTORY / "results"
DEFAULT_OUTPUT = RESULTS / "position_scan_mechanism_profile_audit_v1"
AGGREGATE_PATH = RESULTS / "position_scan_response_v1" / "summary.json"

COMMON_STEPS = (500, 700, 900, 1300, 1500)
PRECURSOR_STEPS = (700, 900, 1300, 1500)
NATURAL_ZONE_DEFINITION_STEP = 1300
NATURAL_ZONE_BACKGROUND_BOUNDS = (60.0, 100.0)
NATURAL_ZONE_SEARCH_END = 100.0
NATURAL_EVENT_SITE = protocol.UNTREATED_EVENT_SITE
WIRES = ("first_wire_z", "second_wire_y")
BRANCHES = ("minus", "plus")
CONTOUR_LEVELS = ("0.45", "0.50", "0.55")
DOWNSTREAM_RESPONSE_LENGTH = 30.0
PROFILE_GRID_SPACING = 0.5
MASK_GRID_TOLERANCE = PROFILE_GRID_SPACING / np.sqrt(2.0)

CASE_CENTERS = {
    "c14p5": 14.5,
    "c18p5": 18.5,
    "c22p5": 22.5,
    "c26p5": 26.5,
    "c30p5": 30.5,
    "c34p5": 34.5,
    "c38p5": 38.5,
    "c64p5": 64.5,
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _paired_directory(step: int) -> Path:
    if step <= 600:
        return RESULTS / "inward_intermediate_t600_v1"
    return RESULTS / "inward_full_lifetime_t2000_v1"


def _profile_source(case_id: str, step: int) -> tuple[Path, str]:
    """Return one treated profile path and its array-name prefix."""

    filename = f"profiles-step-{step:04d}.npz"
    if case_id == "c14p5":
        if step <= 2000:
            return _paired_directory(step) / filename, "treated_"
        return RESULTS / "k1_first_break_t3000_v1" / filename, ""
    if case_id == "c22p5":
        if step <= 2000:
            return (
                RESULTS / "outer_placement_lifetime_v1" / filename,
                "treated_",
            )
        return (
            RESULTS
            / "outer_placement_first_break_t3000_v1"
            / filename,
            "",
        )
    if case_id == "c64p5":
        return (
            RESULTS / "far_field_placement_sentinel_v1" / filename,
            "",
        )
    if case_id not in CASE_CENTERS:
        raise ValueError(f"unknown case: {case_id}")
    return RESULTS / f"position_scan_{case_id}_v1" / filename, ""


def _untreated_source(step: int) -> Path:
    return _paired_directory(step) / f"profiles-step-{step:04d}.npz"


def _summary_source(case_id: str, step: int) -> Path:
    if case_id == "c14p5":
        if step <= 2000:
            return (
                RESULTS
                / "inward_full_lifetime_t2000_v1"
                / "summary.json"
            )
        return RESULTS / "k1_first_break_t3000_v1" / "summary.json"
    if case_id == "c22p5":
        if step <= 2000:
            return (
                RESULTS
                / "outer_placement_lifetime_v1"
                / "summary.json"
            )
        return (
            RESULTS
            / "outer_placement_first_break_t3000_v1"
            / "summary.json"
        )
    if case_id == "c64p5":
        return (
            RESULTS / "far_field_placement_sentinel_v1" / "summary.json"
        )
    return RESULTS / f"position_scan_{case_id}_v1" / "summary.json"


def _branch_view(
    coordinate: np.ndarray,
    values: np.ndarray,
    branch: str,
) -> tuple[np.ndarray, np.ndarray]:
    sign = 1.0 if branch == "plus" else -1.0
    selection = coordinate * sign >= 0.0
    distance = coordinate[selection] * sign
    selected = values[selection]
    order = np.argsort(distance)
    return (
        np.asarray(distance[order], dtype=np.float64),
        np.asarray(selected[order], dtype=np.float64),
    )


def _component_half_depth_zone(
    distance: np.ndarray,
    radius: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    search = (
        (distance >= protocol.CENTRAL_EXCLUSION_DISTANCE)
        & (distance <= NATURAL_ZONE_SEARCH_END)
        & (valid >= 0.999)
    )
    background = (
        (distance >= NATURAL_ZONE_BACKGROUND_BOUNDS[0])
        & (distance <= NATURAL_ZONE_BACKGROUND_BOUNDS[1])
        & (valid >= 0.999)
    )
    if not np.any(search) or not np.any(background):
        raise RuntimeError("natural-zone profile support is incomplete")
    indices = np.flatnonzero(search)
    minimum_index = int(indices[np.argmin(radius[search])])
    background_radius = float(np.median(radius[background]))
    minimum_radius = float(radius[minimum_index])
    half_depth_radius = float(
        minimum_radius + 0.5 * (background_radius - minimum_radius)
    )
    inside = search & (radius <= half_depth_radius)
    left = minimum_index
    right = minimum_index
    while left > 0 and inside[left - 1]:
        left -= 1
    while right + 1 < inside.size and inside[right + 1]:
        right += 1

    def crossing(i0: int, i1: int) -> float:
        x0 = float(distance[i0])
        x1 = float(distance[i1])
        y0 = float(radius[i0])
        y1 = float(radius[i1])
        if y1 == y0:
            return 0.5 * (x0 + x1)
        fraction = (half_depth_radius - y0) / (y1 - y0)
        return float(x0 + fraction * (x1 - x0))

    interpolated_inner = (
        crossing(left - 1, left) if left > 0 else float(distance[left])
    )
    interpolated_outer = (
        crossing(right, right + 1)
        if right + 1 < distance.size
        else float(distance[right])
    )
    return {
        "background_radius": background_radius,
        "minimum_radius": minimum_radius,
        "minimum_position": float(distance[minimum_index]),
        "half_depth_radius": half_depth_radius,
        "grid_bounds": [
            float(distance[left]),
            float(distance[right]),
        ],
        "interpolated_bounds": [
            interpolated_inner,
            interpolated_outer,
        ],
    }


def derive_natural_zone() -> dict[str, Any]:
    source = _untreated_source(NATURAL_ZONE_DEFINITION_STEP)
    records: list[dict[str, Any]] = []
    with np.load(source) as arrays:
        for wire in WIRES:
            coordinate = np.asarray(
                arrays[f"{wire}_coordinate"], dtype=np.float64
            )
            for branch in BRANCHES:
                for level in CONTOUR_LEVELS:
                    distance, radius = _branch_view(
                        coordinate,
                        arrays[
                            f"{wire}_untreated_contour_radius_{level}"
                        ],
                        branch,
                    )
                    valid_distance, valid = _branch_view(
                        coordinate,
                        arrays[
                            f"{wire}_untreated_contour_valid_{level}"
                        ],
                        branch,
                    )
                    if not np.array_equal(distance, valid_distance):
                        raise RuntimeError("contour coordinates differ")
                    record = _component_half_depth_zone(
                        distance, radius, valid
                    )
                    record.update(
                        {
                            "wire": wire,
                            "branch": branch,
                            "level": level,
                        }
                    )
                    records.append(record)
    grid_bounds = {tuple(item["grid_bounds"]) for item in records}
    if len(grid_bounds) != 1:
        raise RuntimeError(
            f"natural-zone grid bounds disagree: {sorted(grid_bounds)}"
        )
    frozen = list(next(iter(grid_bounds)))
    interpolated = np.asarray(
        [item["interpolated_bounds"] for item in records],
        dtype=np.float64,
    )
    return {
        "definition_step": NATURAL_ZONE_DEFINITION_STEP,
        "observable": "untreated c=0.45/0.50/0.55 contour radius",
        "algorithm": (
            "Connected grid component containing each arm minimum at or "
            "below half depth relative to the median radius over "
            "distance 60--100."
        ),
        "search_bounds": [
            protocol.CENTRAL_EXCLUSION_DISTANCE,
            NATURAL_ZONE_SEARCH_END,
        ],
        "background_bounds": list(NATURAL_ZONE_BACKGROUND_BOUNDS),
        "frozen_grid_bounds": frozen,
        "all_twelve_arm_level_definitions_agree": True,
        "interpolated_bounds_median": [
            float(value) for value in np.median(interpolated, axis=0)
        ],
        "interpolated_bounds_range": [
            [
                float(np.min(interpolated[:, index])),
                float(np.max(interpolated[:, index])),
            ]
            for index in range(2)
        ],
        "records": records,
        "source": str(source.resolve()),
        "source_sha256": sha256_path(source),
    }


def _trapz_in_bounds(
    distance: np.ndarray,
    values: np.ndarray,
    bounds: tuple[float, float],
) -> float:
    selection = (
        (distance >= bounds[0]) & (distance <= bounds[1])
    )
    if np.count_nonzero(selection) < 2:
        return 0.0
    return float(np.trapz(values[selection], distance[selection]))


def _interpolate(
    distance: np.ndarray,
    values: np.ndarray,
    position: float,
) -> float:
    return float(np.interp(position, distance, values))


def _summary_statistics(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "range": float(np.ptp(array)),
        "all_positive": bool(np.all(array > 0.0)),
        "all_negative": bool(np.all(array < 0.0)),
        "values": [float(value) for value in array],
    }


def _direct_exposure(
    case_id: str,
    natural_bounds: tuple[float, float],
) -> dict[str, float]:
    geometry = protocol.geometry_for_center(CASE_CENTERS[case_id])
    nodes = np.arange(
        natural_bounds[0],
        natural_bounds[1] + 0.5 * PROFILE_GRID_SPACING,
        PROFILE_GRID_SPACING,
        dtype=np.float64,
    )
    node_values = signed_arm_window(nodes, geometry)
    fine = np.linspace(
        natural_bounds[0], natural_bounds[1], 10_001
    )
    continuous = signed_arm_window(fine, geometry)
    return {
        "definition": (
            "Normalized axial mobility-deficit overlap with the frozen "
            "grid-resolved natural zone."
        ),
        "grid_node_mean": float(np.mean(node_values)),
        "continuous_interval_mean": float(
            np.trapz(continuous, fine)
            / (natural_bounds[1] - natural_bounds[0])
        ),
        "maximum": float(np.max(node_values)),
    }


def _load_profiles(
    case_id: str,
    step: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    treated_path, prefix = _profile_source(case_id, step)
    untreated_path = _untreated_source(step)
    treated: dict[str, dict[str, np.ndarray]] = {}
    untreated: dict[str, dict[str, np.ndarray]] = {}
    with np.load(treated_path) as case_arrays, np.load(
        untreated_path
    ) as base_arrays:
        for wire in WIRES:
            case_coordinate = np.asarray(
                case_arrays[f"{wire}_coordinate"], dtype=np.float64
            )
            base_coordinate = np.asarray(
                base_arrays[f"{wire}_coordinate"], dtype=np.float64
            )
            if not np.array_equal(case_coordinate, base_coordinate):
                raise RuntimeError(
                    f"{case_id} {wire} coordinate differs at {step}"
                )
            treated[wire] = {
                "coordinate": case_coordinate,
                "phase_radius": np.asarray(
                    case_arrays[f"{wire}_{prefix}phase_radius"],
                    dtype=np.float64,
                ),
            }
            untreated[wire] = {
                "coordinate": base_coordinate,
                "phase_radius": np.asarray(
                    base_arrays[
                        f"{wire}_untreated_phase_radius"
                    ],
                    dtype=np.float64,
                ),
            }
            for level in CONTOUR_LEVELS:
                treated[wire][f"contour_radius_{level}"] = np.asarray(
                    case_arrays[
                        f"{wire}_{prefix}contour_radius_{level}"
                    ],
                    dtype=np.float64,
                )
                treated[wire][f"contour_valid_{level}"] = np.asarray(
                    case_arrays[
                        f"{wire}_{prefix}contour_valid_{level}"
                    ],
                    dtype=np.float64,
                )
                untreated[wire][f"contour_radius_{level}"] = np.asarray(
                    base_arrays[
                        f"{wire}_untreated_contour_radius_{level}"
                    ],
                    dtype=np.float64,
                )
                untreated[wire][f"contour_valid_{level}"] = np.asarray(
                    base_arrays[
                        f"{wire}_untreated_contour_valid_{level}"
                    ],
                    dtype=np.float64,
                )
    return treated, untreated


def _branch_metrics(
    case_id: str,
    step: int,
    wire: str,
    branch: str,
    treated: dict[str, np.ndarray],
    untreated: dict[str, np.ndarray],
    natural_bounds: tuple[float, float],
) -> dict[str, Any]:
    geometry = protocol.geometry_for_center(CASE_CENTERS[case_id])
    coordinate = treated["coordinate"]
    distance, treated_radius = _branch_view(
        coordinate, treated["phase_radius"], branch
    )
    base_distance, base_radius = _branch_view(
        coordinate, untreated["phase_radius"], branch
    )
    if not np.array_equal(distance, base_distance):
        raise RuntimeError("treated and untreated distance grids differ")
    area_response = np.pi * (
        treated_radius**2 - base_radius**2
    )
    natural_zone_volume = _trapz_in_bounds(
        distance, area_response, natural_bounds
    )
    collar_bounds = (
        geometry.inner_support_distance,
        geometry.outer_support_distance,
    )
    downstream_bounds = (
        geometry.outer_support_distance,
        geometry.outer_support_distance + DOWNSTREAM_RESPONSE_LENGTH,
    )
    corridor_bounds = (
        natural_bounds[1],
        geometry.inner_support_distance,
    )
    contour_response: dict[str, Any] = {}
    contour_values_in_collar: list[float] = []
    for level in CONTOUR_LEVELS:
        level_distance, case_contour = _branch_view(
            coordinate,
            treated[f"contour_radius_{level}"],
            branch,
        )
        _, case_valid = _branch_view(
            coordinate,
            treated[f"contour_valid_{level}"],
            branch,
        )
        _, base_contour = _branch_view(
            coordinate,
            untreated[f"contour_radius_{level}"],
            branch,
        )
        _, base_valid = _branch_view(
            coordinate,
            untreated[f"contour_valid_{level}"],
            branch,
        )
        validity = min(
            _interpolate(level_distance, case_valid, NATURAL_EVENT_SITE),
            _interpolate(level_distance, base_valid, NATURAL_EVENT_SITE),
        )
        contour_response[level] = {
            "radius_response_at_natural_site": (
                _interpolate(
                    level_distance, case_contour, NATURAL_EVENT_SITE
                )
                - _interpolate(
                    level_distance, base_contour, NATURAL_EVENT_SITE
                )
            ),
            "minimum_interpolated_valid_fraction": validity,
        }
        collar = (
            (level_distance >= collar_bounds[0])
            & (level_distance <= collar_bounds[1])
            & (case_valid >= 0.999)
        )
        contour_values_in_collar.extend(
            float(value) for value in case_contour[collar]
        )
    contour_array = np.asarray(
        contour_values_in_collar, dtype=np.float64
    )
    plateau_bounds = (
        protocol.RADIUS - protocol.INTERFACE_WIDTH,
        protocol.RADIUS + protocol.INTERFACE_WIDTH,
    )
    support_bounds = (
        protocol.RADIUS - 2.0 * protocol.INTERFACE_WIDTH,
        protocol.RADIUS + 2.0 * protocol.INTERFACE_WIDTH,
    )
    return {
        "case_id": case_id,
        "step": step,
        "wire": wire,
        "branch": branch,
        "natural_site_phase_radius_response": (
            _interpolate(distance, treated_radius, NATURAL_EVENT_SITE)
            - _interpolate(distance, base_radius, NATURAL_EVENT_SITE)
        ),
        "natural_zone_volume_response": natural_zone_volume,
        "collar_volume_response": _trapz_in_bounds(
            distance, area_response, collar_bounds
        ),
        "junction_to_collar_corridor_volume_response": (
            _trapz_in_bounds(
                distance, area_response, corridor_bounds
            )
            if corridor_bounds[1] > corridor_bounds[0]
            else 0.0
        ),
        "downstream_volume_response": _trapz_in_bounds(
            distance, area_response, downstream_bounds
        ),
        "contour_response": contour_response,
        "collar_contour_mask_screen": {
            "sample_count": int(contour_array.size),
            "minimum_radius": float(np.min(contour_array)),
            "maximum_radius": float(np.max(contour_array)),
            "fraction_inside_radial_plateau": float(
                np.mean(
                    (contour_array >= plateau_bounds[0])
                    & (contour_array <= plateau_bounds[1])
                )
            ),
            "fraction_inside_radial_support_with_grid_tolerance": float(
                np.mean(
                    (
                        contour_array
                        >= support_bounds[0] - MASK_GRID_TOLERANCE
                    )
                    & (
                        contour_array
                        <= support_bounds[1] + MASK_GRID_TOLERANCE
                    )
                )
            ),
        },
    }


def _aggregate_arm_metrics(
    arm_records: list[dict[str, Any]],
) -> dict[str, Any]:
    scalar_names = (
        "natural_site_phase_radius_response",
        "natural_zone_volume_response",
        "collar_volume_response",
        "junction_to_collar_corridor_volume_response",
        "downstream_volume_response",
    )
    output = {
        name: _summary_statistics(
            [float(item[name]) for item in arm_records]
        )
        for name in scalar_names
    }
    output["contour_response_at_natural_site"] = {
        level: _summary_statistics(
            [
                float(
                    item["contour_response"][level][
                        "radius_response_at_natural_site"
                    ]
                )
                for item in arm_records
            ]
        )
        for level in CONTOUR_LEVELS
    }
    contour_samples = sum(
        item["collar_contour_mask_screen"]["sample_count"]
        for item in arm_records
    )
    output["collar_contour_mask_screen"] = {
        "sample_count": contour_samples,
        "minimum_radius": min(
            item["collar_contour_mask_screen"]["minimum_radius"]
            for item in arm_records
        ),
        "maximum_radius": max(
            item["collar_contour_mask_screen"]["maximum_radius"]
            for item in arm_records
        ),
        "all_arm_level_means_inside_radial_plateau": all(
            item["collar_contour_mask_screen"][
                "fraction_inside_radial_plateau"
            ]
            == 1.0
            for item in arm_records
        ),
        "all_arm_level_means_inside_radial_support": all(
            item["collar_contour_mask_screen"][
                "fraction_inside_radial_support_with_grid_tolerance"
            ]
            == 1.0
            for item in arm_records
        ),
        "limitation": (
            "Mean angular contour radii screen gross radial detachment but "
            "cannot exclude an angular sector escaping the fixed shell."
        ),
    }
    return output


def _weld_connectivity_at_step(
    case_id: str, step: int
) -> dict[str, Any]:
    summary = _load_json(_summary_source(case_id, step))
    matches = [
        item for item in summary["milestones"] if int(item["step"]) == step
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{case_id} has {len(matches)} milestones at {step}"
        )
    contact = matches[0]["contact"]
    if "thresholds" not in contact:
        treated_keys = [
            key for key in ("treated", "control") if key in contact
        ]
        if len(treated_keys) != 1:
            raise RuntimeError(
                f"{case_id} treated contact record is ambiguous"
            )
        contact = contact[treated_keys[0]]
    thresholds = contact["thresholds"]
    connected = {
        level: bool(record["cores_connected_locally"])
        for level, record in thresholds.items()
    }
    return {
        "step": step,
        "connected_by_threshold": connected,
        "all_three_thresholds_connected": all(connected.values()),
    }


def _correlation(
    rows: list[dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    response = np.asarray(
        [row["aggregate"][metric]["mean"] for row in rows],
        dtype=np.float64,
    )
    lifetime = np.asarray(
        [row["delta_time_vs_untreated"] for row in rows],
        dtype=np.float64,
    )
    pearson = pearsonr(response, lifetime)
    spearman = spearmanr(response, lifetime)
    return {
        "metric": metric,
        "pearson_r": float(pearson.statistic),
        "pearson_two_sided_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_two_sided_p": float(spearman.pvalue),
        "case_order": [row["case_id"] for row in rows],
        "metric_values": [float(value) for value in response],
        "lifetime_shifts": [float(value) for value in lifetime],
        "interpretation": (
            "Post-hoc within-one-seed association, not independent "
            "validation or a calibrated lifetime law."
        ),
    }


def build_report() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    aggregate = _load_json(AGGREGATE_PATH)
    if (
        aggregate.get("status") != "complete"
        or aggregate["numerical_integrity"].get("all_passed") is not True
    ):
        raise RuntimeError("position scan aggregate is not healthy")
    event_by_case = {
        item["case_id"]: item for item in aggregate["cases"]
    }
    if set(event_by_case) != set(CASE_CENTERS):
        raise RuntimeError("aggregate case set differs from audit case set")

    natural_zone = derive_natural_zone()
    natural_bounds = tuple(
        float(value)
        for value in natural_zone["frozen_grid_bounds"]
    )
    direct_exposure = {
        case_id: _direct_exposure(case_id, natural_bounds)
        for case_id in CASE_CENTERS
    }
    profile_rows: list[dict[str, Any]] = []
    arm_rows: list[dict[str, Any]] = []
    for step in COMMON_STEPS:
        for case_id in CASE_CENTERS:
            treated, untreated = _load_profiles(case_id, step)
            case_arm_rows = []
            for wire in WIRES:
                for branch in BRANCHES:
                    row = _branch_metrics(
                        case_id,
                        step,
                        wire,
                        branch,
                        treated[wire],
                        untreated[wire],
                        natural_bounds,
                    )
                    case_arm_rows.append(row)
                    arm_rows.append(row)
            event = event_by_case[case_id]
            profile_rows.append(
                {
                    "case_id": case_id,
                    "center": CASE_CENTERS[case_id],
                    "step": step,
                    "mode_label": event["mode_label"],
                    "event_time_midpoint": event[
                        "event_time_midpoint"
                    ],
                    "delta_time_vs_untreated": event[
                        "delta_time_vs_untreated"
                    ],
                    "direct_natural_zone_exposure": direct_exposure[
                        case_id
                    ],
                    "aggregate": _aggregate_arm_metrics(
                        case_arm_rows
                    ),
                }
            )

    rows_by_step = {
        step: [
            row for row in profile_rows if int(row["step"]) == step
        ]
        for step in COMMON_STEPS
    }
    correlations = {
        str(step): {
            "natural_zone_volume_response": _correlation(
                rows_by_step[step], "natural_zone_volume_response"
            ),
            "natural_site_phase_radius_response": _correlation(
                rows_by_step[step],
                "natural_site_phase_radius_response",
            ),
        }
        for step in COMMON_STEPS
    }
    indexed = {
        (row["case_id"], int(row["step"])): row
        for row in profile_rows
    }
    far_case = "c64p5"
    harmful_cases = ("c34p5", "c38p5")
    harmful_case_checks: dict[str, Any] = {}
    for case_id in harmful_cases:
        checks_by_step: dict[str, Any] = {}
        for step in PRECURSOR_STEPS:
            row = indexed[(case_id, step)]
            far = indexed[(far_case, step)]
            volume = row["aggregate"][
                "natural_zone_volume_response"
            ]
            far_volume = far["aggregate"][
                "natural_zone_volume_response"
            ]
            contour = row["aggregate"][
                "contour_response_at_natural_site"
            ]
            checks_by_step[str(step)] = {
                "all_four_arms_have_natural_zone_deficit": volume[
                    "all_negative"
                ],
                "deficit_exceeds_three_times_arm_range": (
                    abs(volume["mean"]) > 3.0 * volume["range"]
                ),
                "deficit_exceeds_three_times_far_null": (
                    abs(volume["mean"])
                    > 3.0 * abs(far_volume["mean"])
                ),
                "all_three_contour_levels_thin": all(
                    contour[level]["all_negative"]
                    for level in CONTOUR_LEVELS
                ),
                "collar_region_accumulates_phase": row["aggregate"][
                    "collar_volume_response"
                ]["all_positive"],
            }
        checks = {
            "zero_direct_natural_zone_exposure": (
                direct_exposure[case_id]["maximum"] == 0.0
            ),
            "all_common_time_checks_pass": all(
                all(item.values())
                for item in checks_by_step.values()
            ),
            "precursor_last_step_precedes_first_event": (
                PRECURSOR_STEPS[-1]
                < event_by_case[case_id]["event_bracket"][0]
            ),
        }
        harmful_case_checks[case_id] = {
            "checks": checks,
            "checks_by_step": checks_by_step,
            "passed": all(checks.values()),
        }
    mask_screen_passed = all(
        row["aggregate"]["collar_contour_mask_screen"][
            "all_arm_level_means_inside_radial_plateau"
        ]
        for row in profile_rows
    )
    weld = {
        case_id: _weld_connectivity_at_step(case_id, 1300)
        for case_id in CASE_CENTERS
    }
    profile_nonlocal_precursor_passed = (
        all(
            item["passed"] for item in harmful_case_checks.values()
        )
        and mask_screen_passed
        and all(
            item["all_three_thresholds_connected"]
            for item in weld.values()
        )
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "classification": (
            "common_time_nonlocal_precursor_supported_within_one_seed"
            if profile_nonlocal_precursor_passed
            else "profile_mechanism_audit_inconclusive"
        ),
        "scope": {
            "saved_profiles_only": True,
            "new_solver_steps": 0,
            "common_global_times_only": list(COMMON_STEPS),
            "four_arms_are_correlated_mechanistic_observations": True,
        },
        "natural_zone": natural_zone,
        "direct_natural_zone_exposure": direct_exposure,
        "profile_rows": profile_rows,
        "lifetime_association": correlations,
        "harmful_window_precursor": {
            "cases": harmful_case_checks,
            "passed": profile_nonlocal_precursor_passed,
            "interpretation": (
                "The intermediate collars have zero direct axial overlap "
                "with the frozen natural zone yet create an early, growing "
                "four-arm mass deficit there while phase accumulates in "
                "the collar region. The far collar supplies the null."
            ),
        },
        "weld_connectivity_t1300": weld,
        "fixed_eulerian_mask_profile_screen": {
            "all_saved_arm_level_mean_contours_inside_radial_plateau": (
                mask_screen_passed
            ),
            "radial_plateau_bounds": [
                protocol.RADIUS - protocol.INTERFACE_WIDTH,
                protocol.RADIUS + protocol.INTERFACE_WIDTH,
            ],
            "radial_support_bounds": [
                protocol.RADIUS - 2.0 * protocol.INTERFACE_WIDTH,
                protocol.RADIUS + 2.0 * protocol.INTERFACE_WIDTH,
            ],
            "limitation": (
                "This compact-profile screen cannot exclude angular "
                "interface escape or quantify the evolving resistance "
                "budget; selected full-field checkpoints are required."
            ),
        },
        "decision": {
            "event_detector_jitter_as_explanation": "rejected",
            "nonlocal_transport_redistribution": (
                "supported_as_a_within-one-seed_mechanism_hypothesis"
                if profile_nonlocal_precursor_passed
                else "not_supported"
            ),
            "harmful_placement_window": (
                "mechanistically_supported_but_not_seed_validated"
                if profile_nonlocal_precursor_passed
                else "inconclusive"
            ),
            "direct_overlap_only_explanation": (
                "insufficient_for_c34p5_and_c38p5"
                if profile_nonlocal_precursor_passed
                else "not_resolved"
            ),
        },
        "claim_boundary": [
            (
                "All four arms share one conditioned source and are not "
                "four independent realizations."
            ),
            (
                "The audit can reject an event-detector artifact but cannot "
                "establish seed-independent timing or a universal window."
            ),
            (
                "The correlation with lifetime is post-hoc and is not an "
                "independently validated predictive law."
            ),
            (
                "The treatment is a fixed Eulerian mobility landscape, not "
                "yet demonstrated to be a material-following coating."
            ),
            (
                "Sparse saved checkpoints can support instantaneous flux "
                "evidence but not a continuously integrated transport path."
            ),
        ],
        "next_required_audit": (
            "selected_full_field_dynamic_budget_mask_and_flux"
        ),
        "provenance": {
            "analysis_script": str(SOURCE_PATH),
            "analysis_script_sha256": sha256_path(SOURCE_PATH),
            "aggregate": str(AGGREGATE_PATH.resolve()),
            "aggregate_sha256": sha256_path(AGGREGATE_PATH),
            "profile_file_count": len(
                {
                    str(_profile_source(case_id, step)[0])
                    for case_id in CASE_CENTERS
                    for step in COMMON_STEPS
                }
                | {
                    str(_untreated_source(step))
                    for step in COMMON_STEPS
                }
            ),
        },
    }
    return report, arm_rows


def _write_csv_rows(
    path: Path,
    rows: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_report(output: Path) -> dict[str, Any]:
    report, arm_rows = build_report()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "summary.json", report)
    profile_csv = []
    for row in report["profile_rows"]:
        profile_csv.append(
            {
                "case_id": row["case_id"],
                "center": row["center"],
                "step": row["step"],
                "mode_label": row["mode_label"],
                "delta_time_vs_untreated": row[
                    "delta_time_vs_untreated"
                ],
                "direct_exposure": row[
                    "direct_natural_zone_exposure"
                ]["grid_node_mean"],
                "natural_site_radius_response": row["aggregate"][
                    "natural_site_phase_radius_response"
                ]["mean"],
                "natural_zone_volume_response": row["aggregate"][
                    "natural_zone_volume_response"
                ]["mean"],
                "collar_volume_response": row["aggregate"][
                    "collar_volume_response"
                ]["mean"],
                "downstream_volume_response": row["aggregate"][
                    "downstream_volume_response"
                ]["mean"],
            }
        )
    _write_csv_rows(
        output / "profile_metrics.csv",
        profile_csv,
        (
            "case_id",
            "center",
            "step",
            "mode_label",
            "delta_time_vs_untreated",
            "direct_exposure",
            "natural_site_radius_response",
            "natural_zone_volume_response",
            "collar_volume_response",
            "downstream_volume_response",
        ),
    )
    flat_arm_rows = []
    for row in arm_rows:
        flat_arm_rows.append(
            {
                "case_id": row["case_id"],
                "step": row["step"],
                "wire": row["wire"],
                "branch": row["branch"],
                "natural_site_phase_radius_response": row[
                    "natural_site_phase_radius_response"
                ],
                "natural_zone_volume_response": row[
                    "natural_zone_volume_response"
                ],
                "collar_volume_response": row[
                    "collar_volume_response"
                ],
                "downstream_volume_response": row[
                    "downstream_volume_response"
                ],
            }
        )
    _write_csv_rows(
        output / "arm_metrics.csv",
        flat_arm_rows,
        (
            "case_id",
            "step",
            "wire",
            "branch",
            "natural_site_phase_radius_response",
            "natural_zone_volume_response",
            "collar_volume_response",
            "downstream_volume_response",
        ),
    )
    return report


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    report = write_report(arguments.output.resolve())
    print(
        json.dumps(
            {
                "status": report["status"],
                "classification": report["classification"],
                "harmful_window_precursor_passed": report[
                    "harmful_window_precursor"
                ]["passed"],
                "output": str(arguments.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
