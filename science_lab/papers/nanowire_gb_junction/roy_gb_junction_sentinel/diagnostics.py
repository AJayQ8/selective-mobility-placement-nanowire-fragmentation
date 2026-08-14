"""Conservative first-pinch and axial-fragment diagnostics.

The sentinel's primary event is deliberately topological and one-sided: one
complete axial slice of a wire must contain no material at ``c >= 0.45``.
There is no minimum-radius fallback.  This detects a single-arm pinch that
the older two-detached-arm Roy diagnostic could miss.

All axial run finding is periodic.  Crossed-wire event candidates exclude
the central interval occupied by the orthogonal wire, while fragment
topology is always measured on the complete periodic axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


Array = np.ndarray
DEFAULT_THRESHOLDS = (0.45, 0.50, 0.55)
PRIMARY_PINCH_THRESHOLD = 0.45
MORPHOLOGY_THRESHOLD = 0.50
DEFAULT_PERSISTENCE_RECORDS = 3
DEFAULT_DIAGNOSTIC_INTERVAL = 10


def _threshold_key(value: float) -> str:
    return f"{float(value):.2f}"


def _validate_thresholds(
    thresholds: Sequence[float],
) -> tuple[float, ...]:
    values = tuple(float(value) for value in thresholds)
    if (
        not values
        or len(set(values)) != len(values)
        or not all(np.isfinite(value) for value in values)
        or tuple(sorted(values)) != values
    ):
        raise ValueError("thresholds must be distinct, finite, and increasing")
    if PRIMARY_PINCH_THRESHOLD not in values:
        raise ValueError("the conservative c=0.45 pinch threshold is required")
    if MORPHOLOGY_THRESHOLD not in values:
        raise ValueError("the c=0.50 morphology threshold is required")
    return values


def centered_coordinates(cells: int, spacing: float) -> Array:
    """Return half-cell-centered coordinates for an even periodic lattice."""

    if int(cells) != cells or cells < 2 or cells % 2:
        raise ValueError("the axial lattice must have an even number of cells")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be finite and positive")
    return (
        np.arange(int(cells), dtype=np.float64) - 0.5 * (int(cells) - 1)
    ) * float(spacing)


def full_slice_areas(
    field: Array,
    *,
    axial_axis: int,
    spacing: float,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Array]:
    """Return thresholded full-transverse-plane area on every axial slice."""

    values = np.asarray(field)
    if values.ndim != 3:
        raise ValueError("field must be three-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("field must contain only finite values")
    axis = int(axial_axis)
    if axis not in (0, 1, 2):
        raise ValueError("axial_axis must be 0, 1, or 2")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be finite and positive")
    levels = _validate_thresholds(thresholds)
    transverse_axes = tuple(index for index in range(3) if index != axis)
    cell_area = float(spacing) ** 2
    return {
        _threshold_key(level): (
            np.count_nonzero(values >= level, axis=transverse_axes).astype(
                np.float64
            )
            * cell_area
        )
        for level in levels
    }


@dataclass(frozen=True)
class _CircularRun:
    start_index: int
    cell_count: int
    total_cells: int

    @property
    def crosses_periodic_seam(self) -> bool:
        return self.cell_count == self.total_cells or (
            self.start_index + self.cell_count > self.total_cells
        )

    def indices(self) -> Array:
        return (
            self.start_index
            + np.arange(self.cell_count, dtype=np.int64)
        ) % self.total_cells


def _circular_true_runs(mask: Array) -> list[_CircularRun]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("periodic run mask must be one-dimensional")
    if not np.any(values):
        return []
    if np.all(values):
        return [_CircularRun(0, int(values.size), int(values.size))]
    starts = np.flatnonzero(values & ~np.roll(values, 1))
    runs: list[_CircularRun] = []
    for raw_start in starts:
        start = int(raw_start)
        count = 1
        while count < values.size and values[(start + count) % values.size]:
            count += 1
        runs.append(_CircularRun(start, count, int(values.size)))
    return runs


def _wrap_coordinate(value: float, length: float) -> float:
    wrapped = (float(value) + 0.5 * length) % length - 0.5 * length
    # Use the negative representative at the periodic antipode.
    if wrapped >= 0.5 * length:
        wrapped -= length
    return float(wrapped)


def _run_report(
    run: _CircularRun,
    coordinates: Array,
    spacing: float,
) -> dict[str, Any]:
    length = float(coordinates.size * spacing)
    start_center = float(coordinates[run.start_index])
    unwrapped_end_center = start_center + (run.cell_count - 1) * spacing
    unwrapped_midpoint = 0.5 * (start_center + unwrapped_end_center)
    midpoint = _wrap_coordinate(unwrapped_midpoint, length)
    indices = run.indices()
    signed = coordinates[indices]
    if run.cell_count == run.total_cells:
        branch = "entire_periodic_axis"
    elif run.crosses_periodic_seam:
        branch = "periodic_antipode"
    elif np.all(signed > 0.0):
        branch = "plus"
    elif np.all(signed < 0.0):
        branch = "minus"
    else:
        branch = "junction_center"
    return {
        "start_index": run.start_index,
        "cell_count": run.cell_count,
        "width": float(run.cell_count * spacing),
        "start_center_coordinate": start_center,
        "end_center_coordinate_unwrapped": unwrapped_end_center,
        "start_edge_coordinate_unwrapped": float(
            start_center - 0.5 * spacing
        ),
        "end_edge_coordinate_unwrapped": float(
            start_center + (run.cell_count - 0.5) * spacing
        ),
        "midpoint": midpoint,
        "crosses_periodic_seam": run.crosses_periodic_seam,
        "branch": branch,
    }


def _minimum_image_distance(
    first: float,
    second: float,
    length: float,
) -> float:
    return float(
        abs((float(second) - float(first) + 0.5 * length) % length - 0.5 * length)
    )


def periodic_fragment_topology(
    slice_areas: Array,
    *,
    spacing: float,
) -> dict[str, Any]:
    """Convert periodic zero-area cuts into axial material-run lengths."""

    areas = np.asarray(slice_areas, dtype=np.float64)
    if (
        areas.ndim != 1
        or areas.size < 2
        or not np.all(np.isfinite(areas))
        or np.any(areas < 0.0)
    ):
        raise ValueError("slice areas must be finite, nonnegative, and 1-D")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be finite and positive")
    coordinates = centered_coordinates(areas.size, spacing)
    zero = areas == 0.0
    gap_runs = _circular_true_runs(zero)
    material_runs = _circular_true_runs(~zero)
    gaps = [
        _run_report(run, coordinates, spacing)
        for run in gap_runs
    ]
    fragments = [
        {
            **_run_report(run, coordinates, spacing),
            "axial_length": float(run.cell_count * spacing),
        }
        for run in material_runs
    ]
    material_absent = bool(np.all(zero))
    continuous_periodic_wire = bool(not np.any(zero))
    return {
        "valid": not material_absent,
        "invalid_reason": "all_axial_slices_are_empty" if material_absent else None,
        "periodic_axis_length": float(areas.size * spacing),
        "cut_count": len(gaps),
        "fragment_count": len(fragments),
        "continuous_periodic_wire": continuous_periodic_wire,
        "gaps": gaps,
        "fragments": fragments,
        "fragment_axial_lengths": [
            float(fragment["axial_length"]) for fragment in fragments
        ],
        "total_material_axial_length": float(
            sum(float(fragment["axial_length"]) for fragment in fragments)
        ),
        "edge_runs_merged_periodically": True,
    }


def _run_contains(
    container: _CircularRun,
    contained: _CircularRun,
) -> bool:
    return set(contained.indices().tolist()).issubset(
        set(container.indices().tolist())
    )


def _wire_report(
    areas_by_threshold: dict[str, Array],
    *,
    wire_name: str,
    axial_axis: int,
    spacing: float,
    central_exclusion: float | None,
    site_tolerance: float,
) -> dict[str, Any]:
    first = next(iter(areas_by_threshold.values()))
    coordinates = centered_coordinates(first.size, spacing)
    length = float(first.size * spacing)
    runs_by_threshold: dict[str, list[_CircularRun]] = {}
    threshold_reports: dict[str, Any] = {}
    for key, areas in areas_by_threshold.items():
        runs = _circular_true_runs(np.asarray(areas) == 0.0)
        runs_by_threshold[key] = runs
        positive = np.asarray(areas)[np.asarray(areas) > 0.0]
        threshold_reports[key] = {
            "minimum_area": float(np.min(areas)),
            "minimum_positive_area": (
                float(np.min(positive)) if positive.size else None
            ),
            "zero_slice_count": int(np.count_nonzero(np.asarray(areas) == 0.0)),
            "gaps": [
                _run_report(run, coordinates, spacing)
                for run in runs
            ],
            "fragment_topology": periodic_fragment_topology(
                areas, spacing=spacing
            ),
        }

    primary_key = _threshold_key(PRIMARY_PINCH_THRESHOLD)
    candidates: list[dict[str, Any]] = []
    for primary in runs_by_threshold[primary_key]:
        # An empty wire is an invalid morphology, not a resolved pinch gap.
        if primary.cell_count == primary.total_cells:
            continue
        primary_indices = primary.indices()
        primary_coordinates = coordinates[primary_indices]
        outside_central = bool(
            central_exclusion is None
            or np.all(
                np.abs(primary_coordinates)
                >= float(central_exclusion) - 32.0 * np.finfo(float).eps
            )
        )
        if not outside_central:
            continue

        nested: dict[str, Any] = {}
        nested_valid = True
        for key, runs in runs_by_threshold.items():
            containing = [
                run for run in runs if _run_contains(run, primary)
            ]
            if len(containing) != 1:
                nested_valid = False
                continue
            nested[key] = _run_report(
                containing[0], coordinates, spacing
            )
        primary_report = _run_report(primary, coordinates, spacing)
        nested_midpoints = [
            float(report["midpoint"]) for report in nested.values()
        ]
        midpoint_spread = (
            max(
                _minimum_image_distance(first, second, length)
                for index, first in enumerate(nested_midpoints)
                for second in nested_midpoints[index + 1 :]
            )
            if len(nested_midpoints) >= 2
            else (0.0 if nested_valid else None)
        )
        site_robust = bool(
            nested_valid
            and midpoint_spread is not None
            and midpoint_spread
            <= site_tolerance + 32.0 * np.finfo(float).eps
        )
        candidates.append(
            {
                **primary_report,
                "wire": wire_name,
                "axial_axis": axial_axis,
                "axial_length": length,
                "event_threshold": PRIMARY_PINCH_THRESHOLD,
                "threshold_gaps": nested,
                "threshold_nested": nested_valid,
                "threshold_gap_midpoint_spread": midpoint_spread,
                "site_tolerance": site_tolerance,
                "site_robust": site_robust,
                "eligible_outside_crossed_junction": outside_central,
                "heuristic_fallback_used": False,
            }
        )

    return {
        "wire": wire_name,
        "axial_axis": axial_axis,
        "axial_length": length,
        "central_exclusion": central_exclusion,
        "thresholds": threshold_reports,
        "candidate_gaps": candidates,
        "candidate": any(bool(gap["site_robust"]) for gap in candidates),
    }


def instantaneous_pinches(
    field: Array,
    *,
    geometry: str,
    spacing: float = 0.5,
    radius: float = 6.0,
    width: float = float(np.sqrt(8.0)),
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Measure complete-slice pinches in a crossed or isolated sentinel state."""

    values = np.asarray(field)
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ValueError("field must be a finite three-dimensional array")
    if geometry not in {"crossed", "isolated"}:
        raise ValueError("geometry must be 'crossed' or 'isolated'")
    if (
        not np.isfinite(radius)
        or radius <= 0.0
        or not np.isfinite(width)
        or width <= 0.0
    ):
        raise ValueError("radius and width must be finite and positive")
    levels = _validate_thresholds(thresholds)
    central_exclusion = float(radius + 2.0 * width)
    axes = (
        (("first_wire_z", 2), ("second_wire_y", 1))
        if geometry == "crossed"
        else (("isolated_wire_z", 2),)
    )
    wires: dict[str, Any] = {}
    all_candidates: list[dict[str, Any]] = []
    for wire_name, axis in axes:
        axial_length = values.shape[axis] * spacing
        exclusion = central_exclusion if geometry == "crossed" else None
        if exclusion is not None and exclusion >= 0.5 * axial_length:
            raise ValueError("crossed-junction exclusion consumes the half-arm")
        areas = full_slice_areas(
            values,
            axial_axis=axis,
            spacing=spacing,
            thresholds=levels,
        )
        report = _wire_report(
            areas,
            wire_name=wire_name,
            axial_axis=axis,
            spacing=spacing,
            central_exclusion=exclusion,
            site_tolerance=float(width),
        )
        wires[wire_name] = report
        all_candidates.extend(report["candidate_gaps"])

    return {
        "geometry": geometry,
        "spacing": float(spacing),
        "radius": float(radius),
        "width": float(width),
        "crossed_central_exclusion": (
            central_exclusion if geometry == "crossed" else None
        ),
        "thresholds": list(levels),
        "primary_event_threshold": PRIMARY_PINCH_THRESHOLD,
        "event_definition": (
            "one complete full-transverse axial slice with c < 0.45 "
            "outside the crossed-junction exclusion; no heuristic fallback"
        ),
        "wires": wires,
        "candidate_gaps": all_candidates,
        "candidate": any(
            bool(gap["site_robust"]) for gap in all_candidates
        ),
        "heuristic_fallback_allowed": False,
    }


def _pinch_payload(observation: dict[str, Any]) -> dict[str, Any]:
    if "candidate_gaps" in observation:
        return observation
    for key in ("pinches", "instantaneous_pinches", "pinch"):
        candidate = observation.get(key)
        if isinstance(candidate, dict) and "candidate_gaps" in candidate:
            return candidate
    raise ValueError("observation has no instantaneous pinch payload")


def _gap_matches(
    first: dict[str, Any],
    second: dict[str, Any],
    maximum_displacement: float,
) -> bool:
    if first["wire"] != second["wire"]:
        return False
    first_length = float(first["axial_length"])
    second_length = float(second["axial_length"])
    if not np.isclose(
        first_length, second_length, rtol=0.0, atol=1.0e-12
    ):
        return False
    return (
        _minimum_image_distance(
            float(first["midpoint"]),
            float(second["midpoint"]),
            first_length,
        )
        <= maximum_displacement + 32.0 * np.finfo(float).eps
    )


def _compact_gap(gap: dict[str, Any]) -> dict[str, Any]:
    return {
        key: gap[key]
        for key in (
            "wire",
            "branch",
            "axial_axis",
            "axial_length",
            "start_index",
            "cell_count",
            "width",
            "midpoint",
            "crosses_periodic_seam",
        )
    }


def persistent_single_arm_event(
    observations: Sequence[dict[str, Any]],
    *,
    required_records: int = DEFAULT_PERSISTENCE_RECORDS,
    diagnostic_interval: int = DEFAULT_DIAGNOSTIC_INTERVAL,
    maximum_gap_displacement: float = float(np.sqrt(8.0)),
    latched_event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Latch the first same-gap chain present in consecutive diagnostics."""

    if latched_event is not None and bool(latched_event.get("detected")):
        return dict(latched_event)
    if required_records < 2:
        raise ValueError("required_records must be at least two")
    if diagnostic_interval <= 0:
        raise ValueError("diagnostic_interval must be positive")
    if (
        not np.isfinite(maximum_gap_displacement)
        or maximum_gap_displacement < 0.0
    ):
        raise ValueError("maximum_gap_displacement must be finite and nonnegative")

    records = list(observations)
    steps = [int(record["step"]) for record in records]
    if any(later <= earlier for earlier, later in zip(steps, steps[1:])):
        raise ValueError("observation steps must be strictly increasing")
    payloads = [_pinch_payload(record) for record in records]
    earliest_chains: list[list[dict[str, Any]]] | None = None
    earliest_index: int | None = None

    for start in range(0, len(records) - required_records + 1):
        window_steps = steps[start : start + required_records]
        if any(
            later - earlier != diagnostic_interval
            for earlier, later in zip(window_steps, window_steps[1:])
        ):
            continue
        chains = [
            [gap]
            for gap in payloads[start]["candidate_gaps"]
            if bool(gap.get("site_robust", False))
        ]
        for offset in range(1, required_records):
            next_gaps = [
                gap
                for gap in payloads[start + offset]["candidate_gaps"]
                if bool(gap.get("site_robust", False))
            ]
            extended: list[list[dict[str, Any]]] = []
            for chain in chains:
                for gap in next_gaps:
                    if _gap_matches(
                        chain[-1], gap, maximum_gap_displacement
                    ):
                        extended.append([*chain, gap])
            chains = extended
            if not chains:
                break
        if chains:
            earliest_chains = chains
            earliest_index = start
            break

    if earliest_chains is None or earliest_index is None:
        return {
            "detected": False,
            "latched": False,
            "required_records": required_records,
            "diagnostic_interval": diagnostic_interval,
            "maximum_gap_displacement": maximum_gap_displacement,
            "first_candidate_step": None,
            "confirmation_step": None,
            "event_bracket": None,
            "persistent_gaps": [],
            "heuristic_fallback_used": False,
        }

    # Distinct physical gaps can occasionally produce multiple equivalent
    # matching paths.  Deduplicate by their complete index/count trajectory.
    unique: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for chain in earliest_chains:
        signature = tuple(
            (
                gap["wire"],
                int(gap["start_index"]),
                int(gap["cell_count"]),
            )
            for gap in chain
        )
        unique[signature] = chain
    first_step = steps[earliest_index]
    confirmation_step = steps[earliest_index + required_records - 1]
    previous_step = steps[earliest_index - 1] if earliest_index > 0 else None
    persistent: list[dict[str, Any]] = []
    for chain in unique.values():
        displacements = [
            _minimum_image_distance(
                float(first["midpoint"]),
                float(second["midpoint"]),
                float(first["axial_length"]),
            )
            for first, second in zip(chain, chain[1:])
        ]
        persistent.append(
            {
                "wire": chain[0]["wire"],
                "first_branch": chain[0]["branch"],
                "branch_history": [gap["branch"] for gap in chain],
                "first_midpoint": float(chain[0]["midpoint"]),
                "confirmation_midpoint": float(chain[-1]["midpoint"]),
                "maximum_step_displacement": (
                    max(displacements) if displacements else 0.0
                ),
                "gap_history": [_compact_gap(gap) for gap in chain],
            }
        )
    persistent.sort(
        key=lambda item: (str(item["wire"]), float(item["first_midpoint"]))
    )
    return {
        "detected": True,
        "latched": True,
        "required_records": required_records,
        "diagnostic_interval": diagnostic_interval,
        "maximum_gap_displacement": maximum_gap_displacement,
        "last_nonmatching_observation_step": previous_step,
        "first_candidate_step": first_step,
        "confirmation_step": confirmation_step,
        "event_bracket": [previous_step, first_step],
        "persistent_gaps": persistent,
        "simultaneous_persistent_gap_count": len(persistent),
        "heuristic_fallback_used": False,
    }
