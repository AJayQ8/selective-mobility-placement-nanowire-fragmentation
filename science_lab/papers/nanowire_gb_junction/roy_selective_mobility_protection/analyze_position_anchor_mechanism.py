#!/usr/bin/env python3
"""Audit the completed collar-position anchors without running a solver."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..roy_gb_junction_sentinel.diagnostics import instantaneous_pinches
from ..roy_gb_junction_sentinel.storage import atomic_json
from .geometry import FourArmCollarGeometry, signed_arm_window
from .run_far_field_placement_sentinel import FAR_GEOMETRY
from .run_t100_tubular_position_diagnostic import (
    INWARD_GEOMETRY,
    OUTER_GEOMETRY,
)


SOURCE_PATH = Path(__file__).resolve()
DIRECTORY = SOURCE_PATH.parent
RESULTS = DIRECTORY / "results"
DEFAULT_OUTPUT = RESULTS / "position_anchor_mechanism_audit_v1"

UNTREATED_T1000 = (
    DIRECTORY.parent
    / "roy_2021_reproduction"
    / "results"
    / "t1000_source_semantic_seed2292"
)
UNTREATED_T2000 = (
    DIRECTORY.parent
    / "roy_2021_reproduction"
    / "results"
    / "t2000_continuation_source_semantic_seed2292"
)
T100_STATIC_SUMMARY = RESULTS / "t100_static_preflight_v1" / "summary.json"

CASE_SOURCES = {
    "untreated": (
        UNTREATED_T1000 / "summary.json",
        UNTREATED_T2000 / "summary.json",
    ),
    "far": (
        RESULTS / "far_field_placement_sentinel_v1" / "summary.json",
    ),
    "adjacent": (
        RESULTS / "outer_placement_lifetime_v1" / "summary.json",
        RESULTS
        / "outer_placement_first_break_t3000_v1"
        / "summary.json",
    ),
    "thinning_zone": (
        RESULTS / "inward_full_lifetime_t2000_v1" / "summary.json",
        RESULTS / "k1_first_break_t3000_v1" / "summary.json",
    ),
}
EVENT_SUMMARIES = {
    "far": CASE_SOURCES["far"][-1],
    "adjacent": CASE_SOURCES["adjacent"][-1],
    "thinning_zone": CASE_SOURCES["thinning_zone"][-1],
}
PROFILE_PATHS = {
    "far": (
        RESULTS
        / "far_field_placement_sentinel_v1"
        / "profiles-step-1700.npz"
    ),
    "adjacent": (
        RESULTS
        / "outer_placement_lifetime_v1"
        / "profiles-step-1700.npz"
    ),
    "thinning_zone": (
        RESULTS
        / "inward_full_lifetime_t2000_v1"
        / "profiles-step-1700.npz"
    ),
}
GEOMETRIES = {
    "far": FAR_GEOMETRY,
    "adjacent": OUTER_GEOMETRY,
    "thinning_zone": INWARD_GEOMETRY,
}

RADIUS = 6.0
CENTRAL_EXCLUSION = float(RADIUS + 2.0 * np.sqrt(8.0))
NEAR_REGION_END = 40.0
PROFILE_STEP = 1700
PRODUCTION_MASS_DRIFT_LIMIT = 1.0e-4
FIELD_MAGNITUDE_LIMIT = 2.0
ENERGY_REBOUND_RELATIVE_LIMIT = 1.0e-6
UNTREATED_EVENT_BRACKET = [1710, 1720]
UNTREATED_CONFIRMATION_STEP = 1740
UNTREATED_EVENT_CHECKPOINTS = (1720, 1730, 1740)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _deduplicated_records(
    paths: tuple[Path, ...],
) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for path in paths:
        for record in _load_json(path)["records"]:
            by_step[int(record["step"])] = record
    return [by_step[step] for step in sorted(by_step)]


def _health_audit(paths: tuple[Path, ...]) -> dict[str, Any]:
    records = _deduplicated_records(paths)
    drifts = np.asarray(
        [record["relative_mass_drift"] for record in records],
        dtype=np.float64,
    )
    energies = [
        (int(record["step"]), float(record["free_energy"]))
        for record in records
        if record.get("free_energy") is not None
    ]
    energy_values = np.asarray(
        [value for _, value in energies], dtype=np.float64
    )
    relative_rebounds = (
        np.diff(energy_values) / np.abs(energy_values[:-1])
        if energy_values.size > 1
        else np.asarray([], dtype=np.float64)
    )
    imaginary_values = []
    for record in records:
        for key in (
            "last_inverse_imaginary_linf",
            "maximum_discarded_inverse_imaginary_linf",
            "last_discarded_inverse_imaginary_linf",
        ):
            if record.get(key) is not None:
                imaginary_values.append(float(record[key]))
    checks = {
        "all_fields_finite": all(
            record["field"]["finite"] is True for record in records
        ),
        "field_magnitude_bounded": all(
            abs(float(record["field"]["minimum"]))
            <= FIELD_MAGNITUDE_LIMIT
            and abs(float(record["field"]["maximum"]))
            <= FIELD_MAGNITUDE_LIMIT
            for record in records
        ),
        "mass_drift_below_production_limit": (
            float(np.max(np.abs(drifts)))
            <= PRODUCTION_MASS_DRIFT_LIMIT
        ),
        "free_energy_nonincreasing": (
            relative_rebounds.size == 0
            or float(np.max(relative_rebounds))
            <= ENERGY_REBOUND_RELATIVE_LIMIT
        ),
    }
    return {
        "record_step_range": [
            int(records[0]["step"]),
            int(records[-1]["step"]),
        ],
        "record_count": len(records),
        "maximum_absolute_relative_mass_drift": float(
            np.max(np.abs(drifts))
        ),
        "final_relative_mass_drift": float(drifts[-1]),
        "energy_sample_count": len(energies),
        "maximum_relative_energy_rebound": (
            float(np.max(relative_rebounds))
            if relative_rebounds.size
            else 0.0
        ),
        "energy_change": float(
            energy_values[-1] - energy_values[0]
        ),
        "maximum_recorded_inverse_imaginary_residue": (
            max(imaginary_values) if imaginary_values else None
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _untreated_event_site() -> dict[str, Any]:
    history = []
    for step in UNTREATED_EVENT_CHECKPOINTS:
        field_path = UNTREATED_T2000 / f"checkpoint-step-{step:04d}.npy"
        field = np.load(field_path, mmap_mode="r", allow_pickle=False)
        pinches = instantaneous_pinches(
            field,
            geometry="crossed",
            spacing=0.5,
            radius=RADIUS,
        )
        matches = [
            gap
            for gap in pinches["candidate_gaps"]
            if gap["wire"] == "first_wire_z"
            and gap["branch"] == "plus"
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"untreated positive-arm event unresolved at {step}"
            )
        history.append(
            {
                "step": step,
                "midpoint": float(matches[0]["midpoint"]),
                "width": float(matches[0]["width"]),
            }
        )
        del field, pinches
        gc.collect()
    midpoints = np.asarray(
        [item["midpoint"] for item in history], dtype=np.float64
    )
    return {
        "first_candidate_site": float(midpoints[0]),
        "confirmation_site": float(midpoints[-1]),
        "site_history": history,
        "maximum_site_displacement": float(
            np.max(np.abs(np.diff(midpoints)))
        ),
        "site_stable_within_grid_cell": (
            float(np.max(np.abs(np.diff(midpoints)))) <= 0.5
        ),
    }


def _event_record(
    case: str,
    untreated_site: dict[str, Any],
) -> dict[str, Any]:
    if case == "untreated":
        bracket = UNTREATED_EVENT_BRACKET
        return {
            "event_bracket": bracket,
            "event_midpoint": float(np.mean(bracket)),
            "confirmation_step": UNTREATED_CONFIRMATION_STEP,
            "first_event_site": untreated_site[
                "first_candidate_site"
            ],
            "confirmation_site": untreated_site[
                "confirmation_site"
            ],
        }
    summary = _load_json(EVENT_SUMMARIES[case])
    event = summary["event_assessment"]
    gaps = event["persistent_gaps"]
    if len(gaps) != 1:
        raise RuntimeError(f"{case} persistent event is not unique")
    bracket = [int(value) for value in event["event_bracket"]]
    return {
        "event_bracket": bracket,
        "event_midpoint": float(np.mean(bracket)),
        "confirmation_step": int(event["confirmation_step"]),
        "first_event_site": float(gaps[0]["first_midpoint"]),
        "confirmation_site": float(
            gaps[0]["confirmation_midpoint"]
        ),
        "wire": gaps[0]["wire"],
        "branch": gaps[0]["first_branch"],
    }


def _profile_arrays() -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
]:
    thinning = np.load(PROFILE_PATHS["thinning_zone"])
    coordinate = np.asarray(
        thinning["first_wire_z_coordinate"], dtype=np.float64
    )
    untreated = np.asarray(
        thinning["first_wire_z_untreated_phase_radius"],
        dtype=np.float64,
    )
    treated: dict[str, np.ndarray] = {
        "thinning_zone": np.asarray(
            thinning["first_wire_z_treated_phase_radius"],
            dtype=np.float64,
        )
    }
    adjacent = np.load(PROFILE_PATHS["adjacent"])
    if not np.array_equal(
        coordinate, adjacent["first_wire_z_coordinate"]
    ):
        raise RuntimeError("adjacent coordinate differs")
    if not np.array_equal(
        untreated,
        adjacent["first_wire_z_untreated_phase_radius"],
    ):
        raise RuntimeError("stored untreated profiles differ")
    treated["adjacent"] = np.asarray(
        adjacent["first_wire_z_treated_phase_radius"],
        dtype=np.float64,
    )
    far = np.load(PROFILE_PATHS["far"])
    if not np.array_equal(
        coordinate, far["first_wire_z_coordinate"]
    ):
        raise RuntimeError("far coordinate differs")
    treated["far"] = np.asarray(
        far["first_wire_z_phase_radius"], dtype=np.float64
    )
    return coordinate, untreated, treated


def _value_at(
    coordinate: np.ndarray,
    values: np.ndarray,
    position: float,
) -> float:
    return float(values[int(np.argmin(np.abs(coordinate - position)))])


def _profile_response(
    case: str,
    coordinate: np.ndarray,
    untreated: np.ndarray,
    treated: np.ndarray,
    natural_failure_site: float,
) -> dict[str, Any]:
    geometry = GEOMETRIES[case]
    delta = treated - untreated
    positive = (
        (coordinate >= CENTRAL_EXCLUSION)
        & (coordinate <= 100.0)
    )
    near = (
        (coordinate >= CENTRAL_EXCLUSION)
        & (coordinate <= NEAR_REGION_END)
    )
    collar = (
        (coordinate >= geometry.inner_support_distance)
        & (coordinate <= geometry.outer_support_distance)
    )
    downstream = (
        (coordinate > geometry.outer_support_distance)
        & (
            coordinate
            <= geometry.outer_support_distance + 30.0
        )
    )
    downstream_indices = np.flatnonzero(downstream)
    minimum_downstream_index = int(
        downstream_indices[np.argmin(delta[downstream])]
    )
    positive_indices = np.flatnonzero(positive)
    minimum_radius_index = int(
        positive_indices[np.argmin(treated[positive])]
    )
    return {
        "profile_step": PROFILE_STEP,
        "first_wire_positive_arm": {
            "minimum_treated_radius": float(
                treated[minimum_radius_index]
            ),
            "minimum_treated_radius_position": float(
                coordinate[minimum_radius_index]
            ),
            "response_at_untreated_failure_site": _value_at(
                coordinate, delta, natural_failure_site
            ),
            "mean_response_inside_collar": float(
                np.mean(delta[collar])
            ),
            "mean_response_first_30_after_outer_edge": float(
                np.mean(delta[downstream])
            ),
            "minimum_response_first_30_after_outer_edge": float(
                delta[minimum_downstream_index]
            ),
            "minimum_response_position_after_outer_edge": float(
                coordinate[minimum_downstream_index]
            ),
            "near_junction_response_rms": float(
                np.sqrt(np.mean(delta[near] ** 2))
            ),
            "positive_arm_response_rms": float(
                np.sqrt(np.mean(delta[positive] ** 2))
            ),
        },
    }


def _depletion_reference() -> dict[str, float]:
    measurement = _load_json(T100_STATIC_SUMMARY)[
        "depletion_measurement"
    ]
    minima = []
    half_depths = []
    for wire in ("first_wire_z", "second_wire_y"):
        for level in ("0.45", "0.5", "0.55"):
            for side in ("negative", "positive"):
                record = measurement[wire][level][side]
                minima.append(float(record["minimum_position"]))
                half_depths.append(float(record["outward_half_depth"]))
    return {
        "median_depletion_minimum": float(np.median(minima)),
        "minimum_position_minimum": float(np.min(minima)),
        "minimum_position_maximum": float(np.max(minima)),
        "maximum_outward_half_depth": float(np.max(half_depths)),
    }


def _point_mobility_factor(
    geometry: FourArmCollarGeometry,
    position: float,
) -> float:
    window = float(
        signed_arm_window(
            np.asarray([position], dtype=np.float64), geometry
        )[0]
    )
    return float(
        1.0
        - (1.0 - geometry.protected_mobility_factor) * window
    )


def build_report() -> dict[str, Any]:
    untreated_site = _untreated_event_site()
    events = {
        case: _event_record(case, untreated_site)
        for case in (
            "untreated",
            "far",
            "adjacent",
            "thinning_zone",
        )
    }
    health = {
        case: _health_audit(paths)
        for case, paths in CASE_SOURCES.items()
    }
    coordinate, untreated, treated = _profile_arrays()
    natural_site = float(events["untreated"]["first_event_site"])
    responses = {
        case: _profile_response(
            case,
            coordinate,
            untreated,
            treated[case],
            natural_site,
        )
        for case in ("far", "adjacent", "thinning_zone")
    }

    far_summary = _load_json(EVENT_SUMMARIES["far"])
    far_factor = far_summary["contract"]["factor"]
    adjacent_summary = _load_json(EVENT_SUMMARIES["adjacent"])
    adjacent_factor = adjacent_summary["contract"]["factor"]
    base_budget = float(
        far_factor["thinning_zone_surface_weighted_deficit"]
    )
    budgets = {
        "thinning_zone": {
            "surface_weighted_mobility_deficit": base_budget,
            "relative_to_thinning_zone": 0.0,
        },
        "adjacent": {
            "surface_weighted_mobility_deficit": float(
                adjacent_factor["outer_surface_weighted_deficit"]
            ),
            "relative_to_thinning_zone": float(
                adjacent_factor["relative_outer_budget_surplus"]
            ),
        },
        "far": {
            "surface_weighted_mobility_deficit": float(
                far_factor["far_surface_weighted_deficit"]
            ),
            "relative_to_thinning_zone": float(
                far_factor["relative_far_budget_surplus"]
            ),
        },
    }

    untreated_time = float(events["untreated"]["event_midpoint"])
    for case, event in events.items():
        event["time_change_from_untreated"] = float(
            event["event_midpoint"] - untreated_time
        )
        event["relative_time_change_from_untreated"] = float(
            (event["event_midpoint"] - untreated_time)
            / untreated_time
        )

    relocation_cases = ("thinning_zone", "adjacent")
    relocation_offsets = {
        case: float(
            events[case]["first_event_site"]
            - GEOMETRIES[case].outer_support_distance
        )
        for case in relocation_cases
    }
    offset_values = np.asarray(
        list(relocation_offsets.values()), dtype=np.float64
    )

    depletion = _depletion_reference()
    placement_coordinates = {}
    for case, geometry in GEOMETRIES.items():
        placement_coordinates[case] = {
            "center_distance": geometry.center_distance,
            "center_minus_depletion_minimum_over_R": float(
                (
                    geometry.center_distance
                    - depletion["median_depletion_minimum"]
                )
                / RADIUS
            ),
            "inner_edge_minus_depletion_minimum_over_R": float(
                (
                    geometry.inner_support_distance
                    - depletion["median_depletion_minimum"]
                )
                / RADIUS
            ),
            "mobility_factor_at_initial_depletion_minimum": (
                _point_mobility_factor(
                    geometry,
                    depletion["median_depletion_minimum"],
                )
            ),
            "mobility_factor_at_untreated_failure_site": (
                _point_mobility_factor(geometry, natural_site)
            ),
            "event_site_minus_outer_edge_over_R": float(
                (
                    events[case]["first_event_site"]
                    - geometry.outer_support_distance
                )
                / RADIUS
            ),
        }

    mechanism_checks = {
        "far_lifetime_within_two_percent_of_untreated": (
            abs(events["far"]["relative_time_change_from_untreated"])
            <= 0.02
        ),
        "far_breaks_at_untreated_site_within_one_grid_cell": (
            abs(events["far"]["first_event_site"] - natural_site)
            <= 0.5
        ),
        "far_near_junction_profile_is_untreated_like": (
            responses["far"]["first_wire_positive_arm"][
                "near_junction_response_rms"
            ]
            <= 0.05
        ),
        "both_near_collars_thicken_the_natural_failure_site": all(
            responses[case]["first_wire_positive_arm"][
                "response_at_untreated_failure_site"
            ]
            >= 1.0
            for case in relocation_cases
        ),
        "both_near_collars_create_a_downstream_radius_deficit": all(
            responses[case]["first_wire_positive_arm"][
                "minimum_response_first_30_after_outer_edge"
            ]
            <= -0.3
            for case in relocation_cases
        ),
        "relocated_sites_share_outer_edge_offset_within_one_grid_cell": (
            float(np.ptp(offset_values)) <= 0.5
        ),
        "far_case_falsifies_unconditional_edge_tracking": (
            abs(
                events["far"]["first_event_site"]
                - (
                    GEOMETRIES["far"].outer_support_distance
                    + float(np.mean(offset_values))
                )
            )
            > RADIUS
        ),
    }

    return {
        "schema_version": 1,
        "classification": (
            "two_competing_failure_modes_supported_as_hypothesis_not_yet_law"
        ),
        "scope": {
            "completed_runs_only": True,
            "new_solver_steps": 0,
            "profile_comparison_step": PROFILE_STEP,
            "position_scan_started": False,
        },
        "numerical_integrity": health,
        "budget_fairness": {
            "definition": (
                "t100 surface-weighted integral of 1-m over the "
                "diffuse interface"
            ),
            "cases": budgets,
            "far_is_conservative_against_thinning_zone": (
                budgets["far"]["relative_to_thinning_zone"] > 0.0
            ),
            "all_positioned_budgets_within_two_percent": all(
                abs(item["relative_to_thinning_zone"]) <= 0.02
                for item in budgets.values()
            ),
        },
        "events": events,
        "untreated_site_reanalysis": untreated_site,
        "t1700_profile_response": responses,
        "depletion_reference": depletion,
        "placement_coordinates": placement_coordinates,
        "conditional_relocation": {
            "event_site_minus_outer_edge": relocation_offsets,
            "mean_offset": float(np.mean(offset_values)),
            "offset_range": [
                float(np.min(offset_values)),
                float(np.max(offset_values)),
            ],
            "evidence_count": len(relocation_cases),
            "interpretation": (
                "When the junction-adjacent thinning mode is suppressed, "
                "the two available cases fail about 14.5--15.0 units "
                "beyond the collar outer edge. The far case breaks first "
                "at the untreated site, so edge tracking is conditional."
            ),
        },
        "mechanism_checks": mechanism_checks,
        "mechanism_checks_passed": all(mechanism_checks.values()),
        "mechanism_interpretation": {
            "supported": (
                "The current anchors support competition between a "
                "natural junction-adjacent failure mode and a "
                "collar-conditioned downstream mode."
            ),
            "not_yet_supported": (
                "Three positions with one trajectory each do not establish "
                "a continuous design law, a sharp transition, or a "
                "universal 14.75-unit relocation length."
            ),
        },
        "scan_coordinate_recommendation": {
            "primary_axis": (
                "xi=(collar_center-x_depletion_minimum_t100)/R"
            ),
            "why_primary": (
                "It locates a fixed-width collar relative to the measured "
                "initial thinning minimum rather than an arbitrary domain "
                "origin."
            ),
            "mandatory_derived_coordinates": [
                "mobility factor at the t100 depletion minimum",
                "mobility factor at the untreated first-break site",
                "(first-break site-collar outer edge)/R",
            ],
            "future_width_study": (
                "Once width varies, report inner and outer edges separately "
                "in units of R; center alone is no longer sufficient."
            ),
            "positions_frozen_by_this_audit": False,
        },
        "limits": [
            "one trajectory per position",
            "one radius and one diffuse-interface width",
            "fixed collar width and mobility contrast",
            "long-time flux was not saved at a common checkpoint",
            "the relocation offset currently has only two supporting cases",
        ],
    }


def write_report(output: Path) -> dict[str, Any]:
    report = build_report()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "summary.json", report)
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
                "classification": report["classification"],
                "mechanism_checks_passed": report[
                    "mechanism_checks_passed"
                ],
                "output": str(arguments.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
