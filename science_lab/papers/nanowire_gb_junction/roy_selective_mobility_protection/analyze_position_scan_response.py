#!/usr/bin/env python3
"""Aggregate the frozen position scan after all five new cases complete."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..roy_fixed_gb_bridge.provenance import sha256_path
from ..roy_gb_junction_sentinel import (
    run_t100_conditioned_production as production_helper,
)
from ..roy_gb_junction_sentinel.storage import atomic_json
from . import analyze_position_anchor_mechanism as anchor_analysis
from . import position_scan_protocol as protocol
from . import run_position_scan_case as case_runner


SOURCE_PATH = Path(__file__).resolve()
DIRECTORY = SOURCE_PATH.parent
DEFAULT_OUTPUT = DIRECTORY / "results" / "position_scan_response_v1"
ANCHOR_AUDIT_PATH = (
    DIRECTORY
    / "results"
    / "position_anchor_mechanism_audit_v1"
    / "summary.json"
)
ANCHOR_SUMMARY_PATHS = {
    "c14p5": (
        DIRECTORY / "results" / "k1_first_break_t3000_v1" / "summary.json"
    ),
    "c22p5": (
        DIRECTORY
        / "results"
        / "outer_placement_first_break_t3000_v1"
        / "summary.json"
    ),
    "c64p5": (
        DIRECTORY
        / "results"
        / "far_field_placement_sentinel_v1"
        / "summary.json"
    ),
}
ANCHOR_CASES = {
    "c14p5": {
        "audit_key": "thinning_zone",
        "center": 14.5,
    },
    "c22p5": {
        "audit_key": "adjacent",
        "center": 22.5,
    },
    "c64p5": {
        "audit_key": "far",
        "center": 64.5,
    },
}
WIRES = ("first_wire_z", "second_wire_y")
LEVELS = ("0.45", "0.50", "0.55")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _record_index(
    records: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    steps = [int(item["step"]) for item in records]
    if steps != sorted(steps):
        raise RuntimeError("diagnostic records are not sorted")
    if len(steps) != len(set(steps)):
        raise RuntimeError("diagnostic records contain duplicate steps")
    return {
        int(item["step"]): item
        for item in records
    }


def _health_audit(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise RuntimeError("trajectory contains no diagnostic records")
    initial_mass = float(case_runner.source_helper.EXPECTED_T100_MASS)
    expected_shape = list(case_runner.FROZEN_DEG90.lattice.shape)
    expected_mean_denominator = (
        case_runner.FROZEN_DEG90.lattice.cell_count
        * case_runner.FROZEN_DEG90.lattice.cell_volume
    )
    reported_mass_drifts: list[float] = []
    reconstructed_mass_drifts: list[float] = []
    mass_records_coherent = True
    for item in records:
        field = item["field"]
        mass = float(field["mass"])
        mean = float(field["mean"])
        reported_drift = float(item["relative_mass_drift"])
        reconstructed_drift = abs(mass - initial_mass) / abs(
            initial_mass
        )
        expected_mean = mass / expected_mean_denominator
        values_finite = bool(
            np.isfinite(mass)
            and np.isfinite(mean)
            and np.isfinite(reported_drift)
            and np.isfinite(reconstructed_drift)
        )
        coherent = bool(
            values_finite
            and field["shape"] == expected_shape
            and np.isclose(
                mean,
                expected_mean,
                rtol=0.0,
                atol=1.0e-15,
            )
            and np.isclose(
                reported_drift,
                reconstructed_drift,
                rtol=0.0,
                atol=1.0e-15,
            )
        )
        mass_records_coherent = mass_records_coherent and coherent
        reported_mass_drifts.append(abs(reported_drift))
        reconstructed_mass_drifts.append(reconstructed_drift)
    maximum_mass_drift = max(reconstructed_mass_drifts)
    all_finite = all(
        item["field"]["finite"] is True for item in records
    )
    maximum_magnitude = max(
        max(
            abs(float(item["field"]["minimum"])),
            abs(float(item["field"]["maximum"])),
        )
        for item in records
    )
    energies = [
        float(item["free_energy"])
        for item in records
        if item.get("free_energy") is not None
    ]
    rebounds = [
        (second - first) / max(abs(first), np.finfo(float).tiny)
        for first, second in zip(energies, energies[1:])
    ]
    maximum_rebound = max(rebounds, default=float("-inf"))
    checks = {
        "all_fields_finite": all_finite,
        "mass_shape_mean_and_drift_coherent": (
            mass_records_coherent
        ),
        "field_magnitude_bounded": (
            maximum_magnitude <= case_runner.FIELD_MAGNITUDE_LIMIT
        ),
        "mass_drift_below_production_limit": (
            maximum_mass_drift
            <= case_runner.PRODUCTION_MASS_DRIFT_LIMIT
        ),
        "free_energy_nonincreasing": (
            maximum_rebound
            <= case_runner.ENERGY_REBOUND_RELATIVE_LIMIT
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "record_count": len(records),
        "record_step_range": [
            int(records[0]["step"]),
            int(records[-1]["step"]),
        ],
        "maximum_absolute_relative_mass_drift": (
            maximum_mass_drift
        ),
        "maximum_reported_absolute_relative_mass_drift": max(
            reported_mass_drifts
        ),
        "maximum_field_magnitude": maximum_magnitude,
        "energy_sample_count": len(energies),
        "maximum_relative_energy_rebound": (
            maximum_rebound if rebounds else None
        ),
    }


def _compact_topology(
    record: dict[str, Any],
) -> dict[str, Any]:
    pinches = record["pinches"]
    if bool(pinches["candidate"]) != any(
        bool(gap["site_robust"])
        for gap in pinches["candidate_gaps"]
    ):
        raise RuntimeError("top-level candidate flag is incoherent")
    result: dict[str, Any] = {}
    for wire in WIRES:
        wire_report = pinches["wires"][wire]
        if bool(wire_report["candidate"]) != any(
            bool(gap["site_robust"])
            for gap in wire_report["candidate_gaps"]
        ):
            raise RuntimeError(
                f"candidate flag is incoherent for {wire}"
            )
        thresholds: dict[str, Any] = {}
        for level in LEVELS:
            topology = wire_report["thresholds"][level][
                "fragment_topology"
            ]
            if (
                topology["valid"] is not True
                or topology["invalid_reason"] is not None
            ):
                raise RuntimeError(
                    f"invalid topology for {wire} at {level}"
                )
            gaps = topology["gaps"]
            fragments = topology["fragments"]
            lengths = [
                float(value)
                for value in topology["fragment_axial_lengths"]
            ]
            gap_width = sum(float(gap["width"]) for gap in gaps)
            fragment_length = sum(
                float(fragment["axial_length"])
                for fragment in fragments
            )
            periodic_length = float(
                topology["periodic_axis_length"]
            )
            topology_checks = {
                "cut_count_matches_gaps": (
                    int(topology["cut_count"]) == len(gaps)
                ),
                "fragment_count_matches_fragments": (
                    int(topology["fragment_count"])
                    == len(fragments)
                ),
                "length_array_matches_fragments": (
                    len(lengths) == len(fragments)
                    and all(
                        np.isclose(
                            length,
                            float(fragment["axial_length"]),
                            rtol=0.0,
                            atol=1.0e-12,
                        )
                        for length, fragment in zip(
                            lengths, fragments
                        )
                    )
                ),
                "total_material_matches_fragments": np.isclose(
                    float(topology["total_material_axial_length"]),
                    fragment_length,
                    rtol=0.0,
                    atol=1.0e-12,
                ),
                "material_plus_gaps_closes_axis": np.isclose(
                    fragment_length + gap_width,
                    periodic_length,
                    rtol=0.0,
                    atol=FROZEN_SPACING,
                ),
                "gap_widths_match_cells": all(
                    np.isclose(
                        float(gap["width"]),
                        int(gap["cell_count"]) * FROZEN_SPACING,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                    for gap in gaps
                ),
                "fragment_lengths_match_cells": all(
                    np.isclose(
                        float(fragment["axial_length"]),
                        int(fragment["cell_count"])
                        * FROZEN_SPACING,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                    for fragment in fragments
                ),
                "periodic_cut_fragment_relation": (
                    (
                        int(topology["cut_count"]) == 0
                        and topology["continuous_periodic_wire"]
                        is True
                        and int(topology["fragment_count"]) == 1
                    )
                    or (
                        int(topology["cut_count"]) > 0
                        and topology["continuous_periodic_wire"]
                        is False
                        and int(topology["fragment_count"])
                        == int(topology["cut_count"])
                    )
                ),
            }
            if not all(topology_checks.values()):
                raise RuntimeError(
                    f"incoherent topology for {wire} at {level}: "
                    f"{topology_checks}"
                )
            thresholds[level] = {
                "valid": True,
                "continuous_periodic_wire": topology[
                    "continuous_periodic_wire"
                ],
                "cut_count": int(topology["cut_count"]),
                "fragment_count": int(topology["fragment_count"]),
                "fragment_axial_lengths": lengths,
                "gaps": [
                    {
                        "branch": gap["branch"],
                        "midpoint": float(gap["midpoint"]),
                        "width": float(gap["width"]),
                        "cell_count": int(gap["cell_count"]),
                        "crosses_periodic_seam": gap[
                            "crosses_periodic_seam"
                        ],
                    }
                    for gap in gaps
                ],
                "fragments": [
                    {
                        "branch": fragment["branch"],
                        "midpoint": float(fragment["midpoint"]),
                        "axial_length": float(
                            fragment["axial_length"]
                        ),
                        "crosses_periodic_seam": fragment[
                            "crosses_periodic_seam"
                        ],
                    }
                    for fragment in fragments
                ],
                "total_material_axial_length": float(
                    topology["total_material_axial_length"]
                ),
                "periodic_axis_length": float(
                    topology["periodic_axis_length"]
                ),
            }
        result[wire] = {
            "candidate": bool(wire_report["candidate"]),
            "thresholds": thresholds,
        }
    return result


def _validate_detected_event(
    summary: dict[str, Any],
    record_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    event = summary["event_assessment"]
    bracket = [int(value) for value in event["event_bracket"]]
    first_step = int(event["first_candidate_step"])
    confirmation_step = int(event["confirmation_step"])
    persistent = event["persistent_gaps"]
    checks = {
        "detected": event.get("detected") is True,
        "latched": event.get("latched") is True,
        "no_heuristic_fallback": (
            event.get("heuristic_fallback_used") is False
        ),
        "bracket_shape": (
            len(bracket) == 2 and bracket[0] < bracket[1]
        ),
        "first_candidate_matches_bracket": (
            first_step == bracket[1]
        ),
        "diagnostic_bracket": (
            bracket[1] - bracket[0]
            == int(event["diagnostic_interval"])
        ),
        "three_record_persistence": (
            int(event["required_records"]) == 3
        ),
        "confirmation_after_candidate": (
            confirmation_step
            == first_step
            + 2 * int(event["diagnostic_interval"])
        ),
        "candidate_record_present": first_step in record_index,
        "confirmation_record_present": (
            confirmation_step in record_index
        ),
        "persistent_gap_present": len(persistent) >= 1,
        "simultaneous_count_coherent": (
            int(event["simultaneous_persistent_gap_count"])
            == len(persistent)
        ),
        "stable_site": (
            float(event["maximum_gap_displacement"])
            <= protocol.INTERFACE_WIDTH
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"robust event validation failed: {checks}")
    first_record = record_index[first_step]
    confirmation_record = record_index[confirmation_step]
    first_topology = _compact_topology(first_record)
    confirmation_topology = _compact_topology(
        confirmation_record
    )
    sites: list[dict[str, Any]] = []
    for item in persistent:
        wire = item["wire"]
        branch = item["first_branch"]
        site = float(item["first_midpoint"])
        candidate_matches = [
            gap
            for gap in first_record["pinches"]["candidate_gaps"]
            if (
                gap["wire"] == wire
                and gap["branch"] == branch
                and abs(float(gap["midpoint"]) - site)
                <= FROZEN_SPACING
            )
        ]
        if len(candidate_matches) != 1:
            raise RuntimeError(
                "persistent site is absent from primary candidate topology"
            )
        confirmation_site = float(item["confirmation_midpoint"])
        confirmation_matches = [
            gap
            for gap in confirmation_record["pinches"][
                "candidate_gaps"
            ]
            if (
                gap["wire"] == wire
                and gap["branch"] == branch
                and abs(
                    float(gap["midpoint"]) - confirmation_site
                )
                <= FROZEN_SPACING
            )
        ]
        if len(confirmation_matches) != 1:
            raise RuntimeError(
                "persistent site is absent from confirmation candidates"
            )
        first_candidate = candidate_matches[0]
        confirmation_candidate = confirmation_matches[0]
        primary = first_topology[wire]["thresholds"]["0.45"]
        if primary["cut_count"] < 1:
            raise RuntimeError("primary event wire has no topological cut")
        for level in LEVELS:
            topology_gaps = first_topology[wire]["thresholds"][
                level
            ]["gaps"]
            first_threshold_site = float(
                first_candidate["threshold_gaps"][level][
                    "midpoint"
                ]
            )
            if not any(
                gap["branch"] == branch
                and np.isclose(
                    float(gap["midpoint"]),
                    first_threshold_site,
                    rtol=0.0,
                    atol=1.0e-12,
                )
                for gap in topology_gaps
            ):
                raise RuntimeError(
                    f"event site is not robust at threshold {level}"
                )
            confirmation_topology_gaps = confirmation_topology[wire][
                "thresholds"
            ][level]["gaps"]
            confirmation_threshold_site = float(
                confirmation_candidate["threshold_gaps"][level][
                    "midpoint"
                ]
            )
            if not any(
                gap["branch"] == branch
                and np.isclose(
                    float(gap["midpoint"]),
                    confirmation_threshold_site,
                    rtol=0.0,
                    atol=1.0e-12,
                )
                for gap in confirmation_topology_gaps
            ):
                raise RuntimeError(
                    "confirmation site is not robust at threshold "
                    f"{level}"
                )
        sites.append(
            {
                "wire": wire,
                "branch": branch,
                "site_signed": site,
                "site_abs": abs(site),
                "confirmation_site_signed": float(
                    confirmation_site
                ),
                "maximum_step_displacement": float(
                    item["maximum_step_displacement"]
                ),
            }
        )
    return {
        "event_state": "observed",
        "event_bracket": bracket,
        "event_time_midpoint": 0.5 * sum(bracket),
        "event_time_half_width": 0.5 * (bracket[1] - bracket[0]),
        "first_candidate_step": first_step,
        "confirmation_step": confirmation_step,
        "sites": sites,
        "topology_at_first_candidate": first_topology,
        "topology_at_confirmation": confirmation_topology,
        "topology_at_censoring": None,
        "event_time_lower_bound": None,
        "checks": checks,
    }


FROZEN_SPACING = 0.5


def _event_report(
    summary: dict[str, Any],
    record_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    event = summary["event_assessment"]
    reconstructed = production_helper._event_from_records(
        summary["records"], latched=None
    )
    if reconstructed != event:
        raise RuntimeError(
            "stored event assessment differs from record reconstruction"
        )
    if event.get("detected") is True:
        return _validate_detected_event(summary, record_index)
    checks = {
        "completed": summary.get("status") == "completed",
        "at_target": (
            int(summary["completed_step"]) == protocol.TARGET_STEP
        ),
        "target_stop_reason": (
            summary.get("stop_reason")
            == "fixed_t3000_target_reached"
        ),
        "not_detected": event.get("detected") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"invalid right-censored trajectory: {checks}"
        )
    terminal_topology = _compact_topology(
        record_index[protocol.TARGET_STEP]
    )
    horizon_checks = {
        "no_pending_candidate": (
            record_index[protocol.TARGET_STEP]["pinches"][
                "candidate"
            ]
            is False
        ),
        "no_threshold_cut_at_horizon": all(
            terminal_topology[wire]["thresholds"][level][
                "cut_count"
            ]
            == 0
            for wire in WIRES
            for level in LEVELS
        ),
    }
    if not all(horizon_checks.values()):
        raise RuntimeError(
            f"right-censoring is unresolved at horizon: "
            f"{horizon_checks}"
        )
    checks.update(horizon_checks)
    return {
        "event_state": "right_censored",
        "event_bracket": None,
        "event_time_midpoint": None,
        "event_time_half_width": None,
        "first_candidate_step": None,
        "confirmation_step": None,
        "sites": [],
        "topology_at_first_candidate": None,
        "topology_at_confirmation": None,
        "topology_at_censoring": terminal_topology,
        "event_time_lower_bound": protocol.TARGET_STEP,
        "checks": checks,
    }


def _mode_label(
    sites: list[dict[str, Any]],
    *,
    outer_edge: float,
) -> tuple[str, list[dict[str, Any]]]:
    if not sites:
        return "not_observed", []
    downstream_target = (
        outer_edge + protocol.CONDITIONAL_DOWNSTREAM_OFFSET
    )
    scored: list[dict[str, Any]] = []
    for item in sites:
        absolute = float(item["site_abs"])
        natural_residual = absolute - protocol.UNTREATED_EVENT_SITE
        downstream_residual = absolute - downstream_target
        natural = (
            abs(natural_residual)
            <= protocol.MODE_COMPATIBILITY_TOLERANCE
        )
        downstream = (
            abs(downstream_residual)
            <= protocol.MODE_COMPATIBILITY_TOLERANCE
        )
        scored.append(
            {
                **item,
                "natural_site_residual": natural_residual,
                "downstream_site_residual": downstream_residual,
                "natural_compatible": natural,
                "downstream_compatible": downstream,
            }
        )
    identities = {
        (item["wire"], item["branch"]) for item in scored
    }
    if len(identities) > 1:
        return "mixed_or_simultaneous", scored
    if all(item["natural_compatible"] for item in scored) and not any(
        item["downstream_compatible"] for item in scored
    ):
        return "natural", scored
    if all(item["downstream_compatible"] for item in scored) and not any(
        item["natural_compatible"] for item in scored
    ):
        return "downstream", scored
    if any(item["natural_compatible"] for item in scored) and any(
        item["downstream_compatible"] for item in scored
    ):
        return "ambiguous_or_mixed", scored
    return "other", scored


def _new_case_row(
    case: protocol.PositionCase,
    path: Path,
) -> dict[str, Any]:
    summary = _load_json(path)
    factor_report = case_runner.preflight_case_report(case)
    expected_contract = case_runner._contract(case, factor_report)
    expected_position = expected_contract["position"]
    contract = summary["contract"]
    contract_checks = {
        "exact_current_contract": contract == expected_contract,
        "status_completed": summary.get("status") == "completed",
        "case_id": summary.get("case") == case.slug,
        "classification": summary.get("classification") in {
            f"{case.slug}_position_scan_robust_first_event",
            f"{case.slug}_position_scan_t3000_censored",
        },
        "protocol": contract.get("protocol") == case_runner.PROTOCOL_NAME,
        "position": contract.get("position") == expected_position,
        "start_step": contract.get("start_step") == protocol.START_STEP,
        "target_step": contract.get("target_step") == protocol.TARGET_STEP,
        "diagnostic_interval": (
            contract.get("diagnostic_interval")
            == protocol.DIAGNOSTIC_INTERVAL
        ),
        "energy_interval": (
            contract.get("energy_interval")
            == protocol.ENERGY_INTERVAL
        ),
        "source": (
            contract.get("source_checkpoint", {}).get("sha256")
            == sha256_path(
                case_runner.source_helper.T100_CHECKPOINT_PATH
            )
        ),
        "runner_hash": (
            contract.get("provenance", {}).get("runner_sha256")
            == sha256_path(Path(case_runner.__file__).resolve())
        ),
        "contract_hash": (
            contract.get("provenance", {}).get(
                "scientific_contract_sha256"
            )
            == sha256_path(protocol.CONTRACT_PATH)
        ),
        "preflight_hash": (
            contract.get("preflight", {}).get("sha256")
            == case_runner.verified_preflight()["sha256"]
        ),
        "budget": (
            abs(
                float(
                    contract["position"][
                        "relative_budget_vs_thinning_zone"
                    ]
                )
            )
            <= protocol.MAXIMUM_BUDGET_DIFFERENCE
        ),
    }
    if not all(contract_checks.values()):
        raise RuntimeError(
            f"{case.slug} contract validation failed: {contract_checks}"
        )
    records = summary["records"]
    record_index = _record_index(records)
    expected_steps = list(
        range(
            protocol.START_STEP,
            int(summary["completed_step"]) + 1,
            protocol.DIAGNOSTIC_INTERVAL,
        )
    )
    if list(record_index) != expected_steps:
        raise RuntimeError(
            f"{case.slug} diagnostic record cadence is incomplete"
        )
    if int(summary["completed_step"]) not in record_index:
        raise RuntimeError(
            f"{case.slug} lacks a terminal diagnostic record"
        )
    health = _health_audit(records)
    if not health["passed"]:
        raise RuntimeError(
            f"{case.slug} numerical health failed: {health}"
        )
    event = _event_report(summary, record_index)
    expected_classification = (
        f"{case.slug}_position_scan_robust_first_event"
        if event["event_state"] == "observed"
        else f"{case.slug}_position_scan_t3000_censored"
    )
    expected_stop_reason = (
        "robust_single_arm_event_confirmed"
        if event["event_state"] == "observed"
        else "fixed_t3000_target_reached"
    )
    if (
        summary["classification"] != expected_classification
        or summary["stop_reason"] != expected_stop_reason
    ):
        raise RuntimeError(
            f"{case.slug} terminal classification is incoherent"
        )
    execution_checks = {
        "accepted_steps": (
            int(summary["execution"]["accepted_steps"])
            == int(summary["completed_step"]) - protocol.START_STEP
        ),
        "step_time_count": (
            int(summary["execution"]["step_wall_seconds_count"])
            == int(summary["completed_step"]) - protocol.START_STEP
        ),
        "energy_cadence": all(
            (
                record.get("free_energy") is not None
            )
            == (
                int(record["step"]) % protocol.ENERGY_INTERVAL
                == 0
            )
            for record in records
        ),
        "energy_samples_present": (
            sum(
                record.get("free_energy") is not None
                for record in records
            )
            == len(
                range(
                    protocol.START_STEP,
                    int(summary["completed_step"]) + 1,
                    protocol.ENERGY_INTERVAL,
                )
            )
        ),
        "event_stops_at_confirmation": (
            event["event_state"] != "observed"
            or int(summary["completed_step"])
            == int(event["confirmation_step"])
        ),
        "no_automatic_follow_on": (
            summary["scope"].get(
                "automatic_follow_on_performed_inside_case"
            )
            is False
        ),
        "no_refinement": (
            summary["scope"].get("adaptive_refinement_started") is False
        ),
        "no_seed_ensemble": (
            summary["scope"].get("seed_ensemble_started") is False
        ),
        "no_convergence_family": (
            summary["scope"].get("convergence_family_started") is False
        ),
        "no_part_two": (
            summary["scope"].get("part_two_started") is False
        ),
    }
    if not all(execution_checks.values()):
        raise RuntimeError(
            f"{case.slug} execution/scope failed: {execution_checks}"
        )
    mode, scored_sites = _mode_label(
        event["sites"],
        outer_edge=case.geometry.outer_support_distance,
    )
    primary_site = scored_sites[0] if len(scored_sites) == 1 else None
    midpoint = event["event_time_midpoint"]
    return {
        "case_id": case.slug,
        "role": "prospective_scan",
        "center": case.center,
        "inner_edge": case.geometry.inner_support_distance,
        "outer_edge": case.geometry.outer_support_distance,
        "xi": case.xi,
        "mobility_at_depletion_minimum": (
            expected_position[
                "mobility_factor_at_depletion_minimum"
            ]
        ),
        "mobility_at_natural_site": expected_position[
            "mobility_factor_at_untreated_failure_site"
        ],
        "surface_weighted_mobility_deficit": expected_position[
            "surface_weighted_mobility_deficit"
        ],
        "relative_budget_vs_thinning_zone": expected_position[
            "relative_budget_vs_thinning_zone"
        ],
        "predicted_mode": case.predicted_mode,
        "event_state": event["event_state"],
        "event_bracket": event["event_bracket"],
        "event_time_midpoint": midpoint,
        "event_time_half_width": event["event_time_half_width"],
        "delta_time_vs_untreated": (
            None
            if midpoint is None
            else midpoint - protocol.UNTREATED_EVENT_MIDPOINT
        ),
        "relative_delta_time_vs_untreated": (
            None
            if midpoint is None
            else (
                midpoint - protocol.UNTREATED_EVENT_MIDPOINT
            )
            / protocol.UNTREATED_EVENT_MIDPOINT
        ),
        "event_sites": scored_sites,
        "event_wire": (
            primary_site["wire"] if primary_site else None
        ),
        "event_branch": (
            primary_site["branch"] if primary_site else None
        ),
        "event_site_signed": (
            primary_site["site_signed"] if primary_site else None
        ),
        "event_site_abs": (
            primary_site["site_abs"] if primary_site else None
        ),
        "event_site_minus_outer_edge": (
            None
            if primary_site is None
            else (
                primary_site["site_abs"]
                - case.geometry.outer_support_distance
            )
        ),
        "event_site_minus_outer_edge_over_R": (
            None
            if primary_site is None
            else (
                primary_site["site_abs"]
                - case.geometry.outer_support_distance
            )
            / protocol.RADIUS
        ),
        "natural_site_residual": (
            primary_site["natural_site_residual"]
            if primary_site
            else None
        ),
        "downstream_site_residual": (
            primary_site["downstream_site_residual"]
            if primary_site
            else None
        ),
        "mode_label": mode,
        "first_candidate_step": event["first_candidate_step"],
        "confirmation_step": event["confirmation_step"],
        "topology_at_first_candidate": event[
            "topology_at_first_candidate"
        ],
        "topology_at_confirmation": event[
            "topology_at_confirmation"
        ],
        "topology_at_censoring": event[
            "topology_at_censoring"
        ],
        "event_time_lower_bound": event[
            "event_time_lower_bound"
        ],
        "health": health,
        "validation": {
            "contract_checks": contract_checks,
            "execution_checks": execution_checks,
            "event_checks": event["checks"],
        },
        "input_path": str(path.resolve()),
        "input_sha256": sha256_path(path),
    }


def verify_completed_case_summary(
    case: protocol.PositionCase,
    path: Path | None = None,
) -> dict[str, Any]:
    """Public batch guard for one completed new summary."""

    return _new_case_row(
        case,
        (path or (case.output / "summary.json")).resolve(),
    )


def _anchor_row(
    case_id: str,
    audit: dict[str, Any],
) -> dict[str, Any]:
    spec = ANCHOR_CASES[case_id]
    center = float(spec["center"])
    geometry = protocol.geometry_for_center(center)
    path = ANCHOR_SUMMARY_PATHS[case_id]
    summary = _load_json(path)
    records = summary["records"]
    record_index = _record_index(records)
    segment_health = _health_audit(records)
    if not segment_health["passed"]:
        raise RuntimeError(f"anchor {case_id} health failed")
    event = _event_report(summary, record_index)
    mode, sites = _mode_label(
        event["sites"],
        outer_edge=geometry.outer_support_distance,
    )
    primary = sites[0] if len(sites) == 1 else None
    audit_key = spec["audit_key"]
    health = audit["numerical_integrity"][audit_key]
    if not health["passed"]:
        raise RuntimeError(f"anchor {case_id} full health failed")
    coordinate = audit["placement_coordinates"][audit_key]
    budget = audit["budget_fairness"]["cases"][audit_key]
    midpoint = event["event_time_midpoint"]
    return {
        "case_id": case_id,
        "role": "calibration_anchor",
        "center": center,
        "inner_edge": geometry.inner_support_distance,
        "outer_edge": geometry.outer_support_distance,
        "xi": coordinate[
            "center_minus_depletion_minimum_over_R"
        ],
        "mobility_at_depletion_minimum": coordinate[
            "mobility_factor_at_initial_depletion_minimum"
        ],
        "mobility_at_natural_site": coordinate[
            "mobility_factor_at_untreated_failure_site"
        ],
        "surface_weighted_mobility_deficit": budget[
            "surface_weighted_mobility_deficit"
        ],
        "relative_budget_vs_thinning_zone": budget[
            "relative_to_thinning_zone"
        ],
        "predicted_mode": None,
        "event_state": event["event_state"],
        "event_bracket": event["event_bracket"],
        "event_time_midpoint": midpoint,
        "event_time_half_width": event["event_time_half_width"],
        "delta_time_vs_untreated": (
            midpoint - protocol.UNTREATED_EVENT_MIDPOINT
        ),
        "relative_delta_time_vs_untreated": (
            midpoint - protocol.UNTREATED_EVENT_MIDPOINT
        )
        / protocol.UNTREATED_EVENT_MIDPOINT,
        "event_sites": sites,
        "event_wire": primary["wire"] if primary else None,
        "event_branch": primary["branch"] if primary else None,
        "event_site_signed": (
            primary["site_signed"] if primary else None
        ),
        "event_site_abs": (
            primary["site_abs"] if primary else None
        ),
        "event_site_minus_outer_edge": (
            None
            if primary is None
            else primary["site_abs"] - geometry.outer_support_distance
        ),
        "event_site_minus_outer_edge_over_R": (
            None
            if primary is None
            else (
                primary["site_abs"]
                - geometry.outer_support_distance
            )
            / protocol.RADIUS
        ),
        "natural_site_residual": (
            primary["natural_site_residual"] if primary else None
        ),
        "downstream_site_residual": (
            primary["downstream_site_residual"] if primary else None
        ),
        "mode_label": mode,
        "first_candidate_step": event["first_candidate_step"],
        "confirmation_step": event["confirmation_step"],
        "topology_at_first_candidate": event[
            "topology_at_first_candidate"
        ],
        "topology_at_confirmation": event[
            "topology_at_confirmation"
        ],
        "topology_at_censoring": event[
            "topology_at_censoring"
        ],
        "event_time_lower_bound": event[
            "event_time_lower_bound"
        ],
        "health": health,
        "validation": {
            "event_checks": event["checks"],
            "event_summary_segment_health": segment_health,
            "full_health_source": str(ANCHOR_AUDIT_PATH.resolve()),
            "full_health_source_sha256": sha256_path(
                ANCHOR_AUDIT_PATH
            ),
        },
        "input_path": str(path.resolve()),
        "input_sha256": sha256_path(path),
    }


def _common_new_provenance(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summaries = [_load_json(Path(row["input_path"])) for row in rows]
    comparable = tuple(
        sorted(
            key
            for key in summaries[0]["contract"]["provenance"]
            if key not in {"python", "numpy"}
        )
    )
    checks: dict[str, bool] = {}
    values: dict[str, Any] = {}
    for key in comparable:
        entries = {
            item["contract"]["provenance"][key]
            for item in summaries
        }
        checks[f"common_{key}"] = len(entries) == 1
        values[key] = next(iter(entries)) if len(entries) == 1 else None
    source_hashes = {
        item["contract"]["source_checkpoint"]["sha256"]
        for item in summaries
    }
    checks["common_source_checkpoint"] = len(source_hashes) == 1
    values["source_checkpoint_sha256"] = (
        next(iter(source_hashes)) if len(source_hashes) == 1 else None
    )
    if not all(checks.values()):
        raise RuntimeError(
            f"new-case provenance mismatch: {checks}"
        )
    return {"checks": checks, "values": values}


def build_report(
    new_summary_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Validate all inputs and build, but do not write, the response table."""

    paths = (
        new_summary_paths
        if new_summary_paths is not None
        else {
            case.slug: case.output / "summary.json"
            for case in protocol.CASES
        }
    )
    if set(paths) != set(protocol.CASES_BY_SLUG):
        raise RuntimeError(
            "new summary paths must contain the exact five frozen cases"
        )
    audit = _load_json(ANCHOR_AUDIT_PATH)
    live_audit = anchor_analysis.build_report()
    if audit != live_audit:
        raise RuntimeError(
            "stored anchor audit differs from live reconstruction"
        )
    if (
        audit.get("classification")
        != "two_competing_failure_modes_supported_as_hypothesis_not_yet_law"
    ):
        raise RuntimeError("anchor audit classification changed")
    anchor_rows = [
        _anchor_row(case_id, audit)
        for case_id in ("c14p5", "c22p5", "c64p5")
    ]
    new_rows = [
        _new_case_row(case, paths[case.slug].resolve())
        for case in protocol.CASES
    ]
    common = _common_new_provenance(new_rows)
    rows = sorted(
        [*anchor_rows, *new_rows],
        key=lambda item: float(item["center"]),
    )
    mode_counts: dict[str, int] = {}
    for row in new_rows:
        mode = row["mode_label"]
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    all_health = all(row["health"]["passed"] for row in rows)
    if not all_health:
        raise RuntimeError("one or more aggregate rows are unhealthy")
    return {
        "schema_version": 1,
        "status": "complete",
        "classification": "position_scan_response_aggregated",
        "contract": {
            "expected_new_centers": list(protocol.NEW_CENTERS),
            "radius": protocol.RADIUS,
            "spacing": FROZEN_SPACING,
            "interface_width": protocol.INTERFACE_WIDTH,
            "classification_rules": {
                "absolute_site_used": True,
                "natural_target": protocol.UNTREATED_EVENT_SITE,
                "downstream_target": (
                    "collar_outer_edge+14.75"
                ),
                "downstream_offset": (
                    protocol.CONDITIONAL_DOWNSTREAM_OFFSET
                ),
                "common_tolerance": (
                    protocol.MODE_COMPATIBILITY_TOLERANCE
                ),
            },
        },
        "provenance": {
            "aggregator_sha256": sha256_path(SOURCE_PATH),
            "scientific_contract_sha256": sha256_path(
                protocol.CONTRACT_PATH
            ),
            "anchor_audit": {
                "path": str(ANCHOR_AUDIT_PATH.resolve()),
                "sha256": sha256_path(ANCHOR_AUDIT_PATH),
            },
            "common_new_case_provenance": common,
            "input_sha256": {
                row["case_id"]: row["input_sha256"] for row in rows
            },
        },
        "references": {
            "untreated": {
                "event_bracket": [1710, 1720],
                "event_time_midpoint": (
                    protocol.UNTREATED_EVENT_MIDPOINT
                ),
                "event_site_abs": protocol.UNTREATED_EVENT_SITE,
            },
            "depletion": {
                "time_100_median_minimum": (
                    protocol.DEPLETION_MINIMUM
                )
            },
        },
        "numerical_integrity": {
            "all_passed": all_health,
            "cases": {
                row["case_id"]: row["health"] for row in rows
            },
        },
        "cases": rows,
        "groups": {
            "calibration_case_ids": [
                "c14p5",
                "c22p5",
                "c64p5",
            ],
            "prospective_case_ids": [
                case.slug for case in protocol.CASES
            ],
        },
        "descriptive_summary": {
            "mode_counts_new_only": mode_counts,
            "mode_sequence_new_only": [
                {
                    "case_id": row["case_id"],
                    "center": row["center"],
                    "mode_label": row["mode_label"],
                    "event_time_midpoint": row[
                        "event_time_midpoint"
                    ],
                }
                for row in new_rows
            ],
        },
        "claim_boundary": [
            (
                "The c14.5 and c22.5 anchors defined the provisional "
                "14.75 offset and therefore cannot validate it."
            ),
            (
                "The five new cases form a preregistered interpolation/"
                "position scan, not independent external validation."
            ),
            (
                "One realization per position cannot establish seed "
                "independence, a universal transition, or a final design law."
            ),
            (
                "First-event midpoint is a diagnostic bracket midpoint, "
                "not an exact breakup time."
            ),
        ],
        "scope": {
            "simulation_steps_performed_by_aggregator": 0,
            "adaptive_refinement_performed": False,
        },
    }


CSV_COLUMNS = (
    "case_id",
    "role",
    "center",
    "inner_edge",
    "outer_edge",
    "xi",
    "mobility_at_depletion_minimum",
    "mobility_at_natural_site",
    "surface_weighted_mobility_deficit",
    "relative_budget_vs_thinning_zone",
    "event_state",
    "bracket_lo",
    "bracket_hi",
    "event_time_midpoint",
    "event_time_half_width",
    "event_time_lower_bound",
    "relative_delta_time_vs_untreated",
    "event_wire",
    "event_branch",
    "event_site_signed",
    "event_site_abs",
    "event_site_minus_outer_edge",
    "event_site_minus_outer_edge_over_R",
    "natural_site_residual",
    "downstream_site_residual",
    "mode_label",
    "primary_wire_cut_count_t045",
    "primary_wire_fragment_count_t045",
    "primary_wire_fragment_lengths_t045",
    "other_wire_cut_count_t045",
    "maximum_absolute_relative_mass_drift",
    "maximum_relative_energy_rebound",
    "health_pass",
    "input_sha256",
)


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    bracket = row["event_bracket"]
    topology = row["topology_at_first_candidate"]
    primary_wire = row["event_wire"]
    other_wire = (
        None
        if primary_wire is None
        else next(wire for wire in WIRES if wire != primary_wire)
    )
    primary = (
        None
        if topology is None or primary_wire is None
        else topology[primary_wire]["thresholds"]["0.45"]
    )
    other = (
        None
        if topology is None or other_wire is None
        else topology[other_wire]["thresholds"]["0.45"]
    )
    return {
        "case_id": row["case_id"],
        "role": row["role"],
        "center": row["center"],
        "inner_edge": row["inner_edge"],
        "outer_edge": row["outer_edge"],
        "xi": row["xi"],
        "mobility_at_depletion_minimum": row[
            "mobility_at_depletion_minimum"
        ],
        "mobility_at_natural_site": row[
            "mobility_at_natural_site"
        ],
        "surface_weighted_mobility_deficit": row[
            "surface_weighted_mobility_deficit"
        ],
        "relative_budget_vs_thinning_zone": row[
            "relative_budget_vs_thinning_zone"
        ],
        "event_state": row["event_state"],
        "bracket_lo": bracket[0] if bracket else None,
        "bracket_hi": bracket[1] if bracket else None,
        "event_time_midpoint": row["event_time_midpoint"],
        "event_time_half_width": row["event_time_half_width"],
        "event_time_lower_bound": row["event_time_lower_bound"],
        "relative_delta_time_vs_untreated": row[
            "relative_delta_time_vs_untreated"
        ],
        "event_wire": row["event_wire"],
        "event_branch": row["event_branch"],
        "event_site_signed": row["event_site_signed"],
        "event_site_abs": row["event_site_abs"],
        "event_site_minus_outer_edge": row[
            "event_site_minus_outer_edge"
        ],
        "event_site_minus_outer_edge_over_R": row[
            "event_site_minus_outer_edge_over_R"
        ],
        "natural_site_residual": row["natural_site_residual"],
        "downstream_site_residual": row[
            "downstream_site_residual"
        ],
        "mode_label": row["mode_label"],
        "primary_wire_cut_count_t045": (
            primary["cut_count"] if primary else None
        ),
        "primary_wire_fragment_count_t045": (
            primary["fragment_count"] if primary else None
        ),
        "primary_wire_fragment_lengths_t045": (
            ";".join(
                str(value)
                for value in primary["fragment_axial_lengths"]
            )
            if primary
            else None
        ),
        "other_wire_cut_count_t045": (
            other["cut_count"] if other else None
        ),
        "maximum_absolute_relative_mass_drift": row["health"][
            "maximum_absolute_relative_mass_drift"
        ],
        "maximum_relative_energy_rebound": row["health"][
            "maximum_relative_energy_rebound"
        ],
        "health_pass": row["health"]["passed"],
        "input_sha256": row["input_sha256"],
    }


def _atomic_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(_csv_row(row))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_text(rows: Iterable[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow(_csv_row(row))
    return buffer.getvalue()


def verify_written_report(
    output: Path,
    new_summary_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Adopt an atomically complete aggregate after exact regeneration."""

    summary_path = output / "summary.json"
    csv_path = output / "positions.csv"
    if not summary_path.is_file() or not csv_path.is_file():
        raise RuntimeError("aggregate directory is incomplete")
    expected = build_report(new_summary_paths)
    stored = _load_json(summary_path)
    checks = {
        "summary_exact": stored == expected,
        "csv_exact": (
            csv_path.read_bytes()
            == _csv_text(expected["cases"]).encode("utf-8")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"existing aggregate verification failed: {checks}"
        )
    return stored


def write_report(
    output: Path,
    new_summary_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Validate first, then atomically publish JSON and CSV."""

    report = build_report(new_summary_paths)
    if output.exists():
        raise FileExistsError(f"aggregate output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.mkdir(exist_ok=False)
    try:
        atomic_json(temporary / "summary.json", report)
        _atomic_csv(temporary / "positions.csv", report["cases"])
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return report


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", action="store_true", required=True)
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
                "case_count": len(report["cases"]),
                "output": str(arguments.output.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
