#!/usr/bin/env python3
"""Create and verify the S0/S4-derived sentinel alignment lock.

This module runs no solver.  It refuses to create a launch lock unless:

* S0 has a completed, persistent robust event;
* the preregistered last fully resolved prebreak +z c=0.5 profile exists;
* paired S4 GB-on/GB-off data exist at that *exact* S0 reference step;
* the paired S4 profile resolves a negative groove and inward positive ridge;
* grid-rounded S1/S2 placements still expose the S0 thinning minimum to
  opposite, at-least-half-peak signs; and
* the prior M1/M3 and B4 validation artifacts retain their pinned hashes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contract import (
    MAXIMUM_PERIODIC_SEPARATION,
    canonical_sha256,
    sha256_path,
)
from .null_model import (
    GRID_SPACING,
    INTERFACE_WIDTH,
    PairedGBRadiusProfile,
    calibrate_rho_criterion,
    contour_thinning_measurement,
    load_paired_gb_profile,
    load_s0_profile,
    periodic_gb_planes,
    periodic_linear_interpolate,
)
from .storage import atomic_json


DIRECTORY = Path(__file__).resolve().parent
NULL_MODEL_PATH = DIRECTORY / "null_model.py"
PAPER_DIRECTORY = DIRECTORY.parent
M1_RESULT = (
    PAPER_DIRECTORY
    / "roy_ridge_alignment_measurements"
    / "results"
    / "m1_m3_measurements_t1600"
)
M3_AUDIT_RESULT = (
    PAPER_DIRECTORY
    / "roy_ridge_alignment_measurements"
    / "results"
    / "m3_sign_lobe_overlap_audit"
)
B4_RESULT = (
    PAPER_DIRECTORY
    / "roy_isolated_gb_ridge"
    / "b4_mullins_validation"
    / "results"
    / "fixed_grid_analysis"
)

PREREQUISITE_SOURCES = {
    "m1_summary": {
        "path": M1_RESULT / "summary.json",
        "sha256": "42eac89b0315709484df4e614e343cd2c4ca168c9d8689faaaad526c76928d80",
    },
    "m1_profiles": {
        "path": M1_RESULT / "m1-ridge-profiles.npz",
        "sha256": "f5f6d34ffd035599173b4195dcbce09521afb0faa6d97eefa8b9a761c85f28ef",
    },
    "m3_sign_lobe_summary": {
        "path": M3_AUDIT_RESULT / "summary.json",
        "sha256": "54c804937d14234d4f19d260087fc087b56cdebe4c706fa910c03ff9ad8664c6",
    },
    "b4_summary": {
        "path": B4_RESULT / "summary.json",
        "sha256": "5897f9d8f9b28e035e9164666d2f9c5fe0a8d032a85768baaed3ff48910f19f6",
    },
    "b4_completion_audit": {
        "path": B4_RESULT / "completion-audit.json",
        "sha256": "2546a442decf5777dcb376e7c27f17086688a398b909c7eabdf7f01d2e764374",
    },
}

EXPECTED_M3_CLASSIFICATION = (
    "m1_m3_complete_fixed_spacing_opposite_sign_lobe_overlap_observed_"
    "peak_tracking_not_holdable"
)
EXPECTED_B4_CLASSIFICATION = (
    "b4_fixed_grid_finite_slope_mullins_shape_agreement"
)
ALIGNMENT_SELECTION_RULE = (
    "latest diagnostic before the first complete-gap candidate whose +z "
    "c=0.5 contour is explicitly marked fully resolved"
)
RIDGE_BRANCH_LIMIT = 32.0
RIDGE_SEARCH_MINIMUM = 2.0 * GRID_SPACING
RIDGE_SEARCH_MAXIMUM = RIDGE_BRANCH_LIMIT - 2.0 * INTERFACE_WIDTH
GROOVE_SEARCH_MAXIMUM = 2.0 * INTERFACE_WIDTH
HALF_PEAK_GATE = 0.5


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _resolved_artifact_path(summary_path: Path, declared: Any) -> Path:
    path = Path(str(declared))
    if not path.is_absolute():
        path = summary_path.parent / path
    return path.resolve()


def verify_validation_prerequisites() -> dict[str, Any]:
    """Verify immutable M1/M3 and B4 evidence before any alignment lock."""

    measured: dict[str, Any] = {}
    for name, source in PREREQUISITE_SOURCES.items():
        path = Path(source["path"]).resolve()
        expected = str(source["sha256"])
        if not path.is_file():
            raise FileNotFoundError(f"missing validation prerequisite: {path}")
        digest = sha256_path(path)
        if digest != expected:
            raise ValueError(f"validation prerequisite hash changed: {name}")
        measured[name] = {
            "path": str(path),
            "sha256": digest,
        }

    m1 = _read_json(Path(PREREQUISITE_SOURCES["m1_summary"]["path"]))
    m3 = _read_json(
        Path(PREREQUISITE_SOURCES["m3_sign_lobe_summary"]["path"])
    )
    b4 = _read_json(Path(PREREQUISITE_SOURCES["b4_summary"]["path"]))
    completion = _read_json(
        Path(PREREQUISITE_SOURCES["b4_completion_audit"]["path"])
    )
    checks = {
        "m1_completed": m1.get("status") == "completed",
        "m3_opposite_sign_lobe_audit_passed": (
            m3.get("classification") == EXPECTED_M3_CLASSIFICATION
            and all(bool(value) for value in m3.get("checks", {}).values())
            and all(
                bool(value)
                for value in m3.get("classification_criteria", {}).values()
            )
        ),
        "b4_fixed_grid_shape_validation_passed": (
            b4.get("status") == "completed"
            and b4.get("classification") == EXPECTED_B4_CLASSIFICATION
            and b4.get("b4_passed") is True
        ),
        "b4_completion_audit_matches": (
            completion.get("status") == "completed"
            and completion.get("classification")
            == EXPECTED_B4_CLASSIFICATION
            and completion.get("b4_passed") is True
            and completion.get("summary_json_sha256")
            == PREREQUISITE_SOURCES["b4_summary"]["sha256"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"alignment validation prerequisite failed: {checks}")
    return {
        "sources": measured,
        "checks": checks,
        "scope": (
            "M1/M3 and B4 are prerequisite validation only; their profile is "
            "not extrapolated into the production spacing lock"
        ),
    }


def _profile_entries(
    summary: Mapping[str, Any],
    *,
    owner: str,
) -> dict[int, dict[str, Any]]:
    declared = summary.get("profile_artifacts")
    if not isinstance(declared, Mapping) or not declared:
        raise ValueError(f"{owner} summary has no profile_artifacts mapping")
    entries: dict[int, dict[str, Any]] = {}
    for key, value in declared.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"{owner} profile entry {key!r} is not an object")
        entry = dict(value)
        try:
            key_step = int(str(key))
            step = int(entry.get("step", key_step))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{owner} profile key/step is not an integer: {key!r}"
            ) from error
        if key_step != step:
            raise ValueError(f"{owner} profile key/step mismatch at {key!r}")
        if step in entries:
            raise ValueError(f"{owner} profile step is duplicated: {step}")
        if not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64:
            raise ValueError(f"{owner} profile entry lacks SHA-256 at step {step}")
        if "path" not in entry:
            raise ValueError(f"{owner} profile entry lacks a path at step {step}")
        entries[step] = entry
    return entries


def _load_entry_s0(
    summary_path: Path,
    entry: Mapping[str, Any],
    *,
    require_resolved_contour: bool,
):
    path = _resolved_artifact_path(summary_path, entry["path"])
    profile = load_s0_profile(
        path,
        expected_sha256=str(entry["sha256"]),
        require_resolved_contour=require_resolved_contour,
    )
    if profile.step != int(entry["step"]):
        raise ValueError("S0 profile artifact step differs from its manifest")
    return profile


def _load_entry_s4(
    summary_path: Path,
    entry: Mapping[str, Any],
) -> PairedGBRadiusProfile:
    path = _resolved_artifact_path(summary_path, entry["path"])
    profile = load_paired_gb_profile(
        path,
        expected_sha256=str(entry["sha256"]),
    )
    if profile.step != int(entry["step"]):
        raise ValueError("paired S4 profile artifact step differs from its manifest")
    return profile


def _s0_sources(
    summary_path: Path,
) -> tuple[
    dict[str, Any],
    Any,
    Any,
    Any,
    dict[str, Any],
]:
    summary = _read_json(summary_path)
    if summary.get("status") != "completed" or summary.get("run_id") != "S0":
        raise ValueError("alignment source must be a completed S0 run")
    event = summary.get("event_assessment", {}).get("robust_event", {})
    if event.get("detected") is not True:
        raise ValueError("S0 has no completed robust event")
    bracket = event.get("event_bracket")
    if (
        not isinstance(bracket, list)
        or len(bracket) != 2
        or not all(isinstance(value, int) for value in bracket)
        or bracket[1] <= bracket[0]
    ):
        raise ValueError("S0 robust event has no valid event bracket")
    last_no_gap_step, first_candidate_step = bracket
    if (
        event.get("first_candidate_step") != first_candidate_step
        or not isinstance(event.get("confirmation_step"), int)
        or event["confirmation_step"] < first_candidate_step
        or not event.get("persistent_gaps")
    ):
        raise ValueError("S0 robust event is not persistent and confirmed")

    entries = _profile_entries(summary, owner="S0")
    if last_no_gap_step not in entries or first_candidate_step not in entries:
        raise ValueError("S0 rho-calibration event profiles are missing")
    if entries[last_no_gap_step].get("complete_gap_candidate") is not False:
        raise ValueError(
            "S0 rho-calibration bracket does not start at a no-gap record"
        )
    if entries[first_candidate_step].get("complete_gap_candidate") is not True:
        raise ValueError(
            "S0 rho-calibration bracket does not end at a candidate record"
        )
    resolved_prebreak_steps = sorted(
        step
        for step, entry in entries.items()
        if step < first_candidate_step
        and entry.get("plus_contour_c0p5_fully_resolved") is True
        and entry.get("complete_gap_candidate") is False
    )
    if not resolved_prebreak_steps:
        raise ValueError("S0 has no fully resolved prebreak +z c=0.5 profile")
    reference_step = resolved_prebreak_steps[-1]
    declared_alignment = summary.get("alignment_reference", {})
    if (
        declared_alignment.get("selection_rule") != ALIGNMENT_SELECTION_RULE
        or declared_alignment.get(
            "last_fully_resolved_prebreak_plus_contour_step"
        )
        != reference_step
    ):
        raise ValueError("S0 did not preregister the frozen alignment selection rule")

    reference = _load_entry_s0(
        summary_path,
        entries[reference_step],
        require_resolved_contour=True,
    )
    last_no_gap = _load_entry_s0(
        summary_path,
        entries[last_no_gap_step],
        require_resolved_contour=False,
    )
    first_candidate = _load_entry_s0(
        summary_path,
        entries[first_candidate_step],
        require_resolved_contour=False,
    )
    provenance = {
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": sha256_path(summary_path),
        "event_bracket": bracket,
        "confirmation_step": event["confirmation_step"],
        "reference_step": reference_step,
        "reference_profile_path": str(reference.path),
        "reference_profile_sha256": reference.sha256,
        "last_no_gap_profile_path": str(last_no_gap.path),
        "last_no_gap_profile_sha256": last_no_gap.sha256,
        "first_candidate_profile_path": str(first_candidate.path),
        "first_candidate_profile_sha256": first_candidate.sha256,
    }
    return summary, reference, last_no_gap, first_candidate, provenance


def _s4_source(
    summary_path: Path,
    *,
    required_step: int,
) -> tuple[dict[str, Any], PairedGBRadiusProfile, dict[str, Any]]:
    summary = _read_json(summary_path)
    if summary.get("status") != "completed" or summary.get("run_id") != "S4":
        raise ValueError("paired null source must be a completed S4 reference")
    paired = summary.get("paired_reference")
    if not isinstance(paired, Mapping):
        raise ValueError("S4 summary has no paired_reference object")
    on_fingerprint = paired.get("gb_on_initializer_fingerprint")
    off_fingerprint = paired.get("gb_off_initializer_fingerprint")
    if (
        paired.get("gb_on_run_id") != "S4"
        or paired.get("gb_off_run_id") != "S4_OFF"
        or paired.get("same_initializer_fingerprint") is not True
        or not isinstance(on_fingerprint, str)
        or on_fingerprint != off_fingerprint
    ):
        raise ValueError("S4 on/off pair does not share one initializer")
    entries = _profile_entries(summary, owner="paired S4")
    if required_step not in entries:
        raise ValueError(
            "paired S4 has no profile at the exact S0 alignment reference step"
        )
    profile = _load_entry_s4(summary_path, entries[required_step])
    return summary, profile, {
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": sha256_path(summary_path),
        "profile_path": str(profile.path),
        "profile_sha256": profile.sha256,
        "profile_step": profile.step,
        "shared_initializer_fingerprint": on_fingerprint,
    }


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


def measure_inward_gb_features(
    profile: PairedGBRadiusProfile,
) -> dict[str, float | bool]:
    """Measure the primary plane's branch pointing toward the junction."""

    delta = profile.delta_raw_phase_excess_area_radius
    mask = profile.z <= 0.0
    distance = -profile.z[mask][::-1]
    values = delta[mask][::-1]
    groove_mask = (distance >= 0.0) & (distance <= GROOVE_SEARCH_MAXIMUM)
    ridge_mask = (
        (distance >= RIDGE_SEARCH_MINIMUM)
        & (distance <= RIDGE_SEARCH_MAXIMUM)
    )
    if np.count_nonzero(groove_mask) < 3 or np.count_nonzero(ridge_mask) < 3:
        raise ValueError("paired S4 inward groove/ridge windows are unresolved")
    groove_indices = np.flatnonzero(groove_mask)
    groove_index = int(groove_indices[int(np.argmin(values[groove_mask]))])
    groove_offset, groove_value = _quadratic_extremum(
        distance, values, groove_index, maximum=False
    )
    ridge_indices = np.flatnonzero(ridge_mask)
    ridge_index = int(ridge_indices[int(np.argmax(values[ridge_mask]))])
    ridge_offset, ridge_value = _quadratic_extremum(
        distance, values, ridge_index, maximum=True
    )
    resolved = bool(
        groove_value < 0.0
        and ridge_value > 0.0
        and ridge_offset > groove_offset
    )
    if not resolved:
        raise ValueError(
            "paired S4 exact-time profile has no resolved inward groove/ridge"
        )
    return {
        "branch": "primary_plane_toward_junction",
        "groove_offset_from_plane": groove_offset,
        "groove_delta_raw_radius": groove_value,
        "ridge_offset_from_plane": ridge_offset,
        "ridge_delta_raw_radius": ridge_value,
        "resolved_negative_groove_and_positive_ridge": resolved,
    }


def round_to_production_grid(value: float) -> float:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("alignment distance must be finite and nonnegative")
    return float(np.floor(value / GRID_SPACING + 0.5) * GRID_SPACING)


def _signed_overlap_case(
    *,
    case: str,
    spacing: float,
    target_sign: int,
    target_peak: float,
    z_min: float,
    half_depth_left: float,
    half_depth_right: float,
    paired_gb: PairedGBRadiusProfile,
) -> dict[str, Any]:
    delta = paired_gb.delta_raw_phase_excess_area_radius
    at_minimum = float(
        periodic_linear_interpolate(
            paired_gb.z,
            delta,
            z_min - spacing,
        )
    )
    score = float(target_sign * at_minimum / target_peak)
    sample = np.linspace(half_depth_left, half_depth_right, 2049)
    interval_delta = periodic_linear_interpolate(
        paired_gb.z,
        delta,
        sample - spacing,
    )
    target_fraction = float(
        np.mean(target_sign * interval_delta > 0.0)
    )
    primary, image = periodic_gb_planes(spacing)
    checks = {
        "point_has_target_sign": target_sign * at_minimum > 0.0,
        "point_reaches_half_target_peak": score >= HALF_PEAK_GATE,
        "spacing_is_in_unique_periodic_range": (
            0.0 <= spacing <= MAXIMUM_PERIODIC_SEPARATION
        ),
    }
    return {
        "case": case,
        "spacing": spacing,
        "primary_gb_coordinate": primary,
        "image_gb_coordinate": image,
        "delta_raw_radius_at_s0_z_min": at_minimum,
        "target_sign": target_sign,
        "target_peak_magnitude": target_peak,
        "point_target_normalized_score": score,
        "half_depth_target_sign_fraction": target_fraction,
        "half_depth_minimum_delta_raw_radius": float(np.min(interval_delta)),
        "half_depth_maximum_delta_raw_radius": float(np.max(interval_delta)),
        "checks": checks,
    }


def compute_alignment_lock_payload(
    s0_summary_path: Path,
    s4_summary_path: Path,
) -> dict[str, Any]:
    """Compute a deterministic lock payload; no simulation is executed."""

    s0_summary_path = s0_summary_path.resolve()
    s4_summary_path = s4_summary_path.resolve()
    prerequisites = verify_validation_prerequisites()
    (
        _,
        reference,
        last_no_gap,
        first_candidate,
        s0_provenance,
    ) = _s0_sources(s0_summary_path)
    _, paired_gb, s4_provenance = _s4_source(
        s4_summary_path,
        required_step=reference.step,
    )
    thinning = contour_thinning_measurement(reference)
    features = measure_inward_gb_features(paired_gb)
    z_min = float(thinning["z_min"])
    ridge_offset = float(features["ridge_offset_from_plane"])
    unrounded = {
        "s1_groove": z_min,
        "s2_inward_ridge": z_min + ridge_offset,
    }
    rounded = {
        name: round_to_production_grid(value)
        for name, value in unrounded.items()
    }
    s1 = rounded["s1_groove"]
    s2 = rounded["s2_inward_ridge"]
    if s1 == s2:
        raise ValueError("grid alignment collapsed S1 and S2 to one spacing")
    if max(s1, s2) > MAXIMUM_PERIODIC_SEPARATION:
        raise ValueError("alignment spacing exceeds the unique bicrystal range")
    groove_peak = abs(float(features["groove_delta_raw_radius"]))
    ridge_peak = float(features["ridge_delta_raw_radius"])
    overlap = {
        "S1": _signed_overlap_case(
            case="negative_groove_lobe",
            spacing=s1,
            target_sign=-1,
            target_peak=groove_peak,
            z_min=z_min,
            half_depth_left=float(thinning["half_depth_left"]),
            half_depth_right=float(thinning["half_depth_right"]),
            paired_gb=paired_gb,
        ),
        "S2": _signed_overlap_case(
            case="positive_inward_ridge_lobe",
            spacing=s2,
            target_sign=1,
            target_peak=ridge_peak,
            z_min=z_min,
            half_depth_left=float(thinning["half_depth_left"]),
            half_depth_right=float(thinning["half_depth_right"]),
            paired_gb=paired_gb,
        ),
    }
    overlap_passed = all(
        all(bool(value) for value in case["checks"].values())
        for case in overlap.values()
    )
    if not overlap_passed:
        raise RuntimeError(
            "grid-rounded S1/S2 placements failed opposite sign-lobe overlap"
        )
    rho = calibrate_rho_criterion(last_no_gap, first_candidate)
    payload: dict[str, Any] = {
        "lock_version": 1,
        "frozen": True,
        "source_run_id": "S0",
        "s0_summary_path": str(s0_summary_path),
        "s0_summary_sha256": sha256_path(s0_summary_path),
        "s4_summary_path": str(s4_summary_path),
        "s4_summary_sha256": sha256_path(s4_summary_path),
        "s0_z_min": z_min,
        "s1_spacing": s1,
        "s2_spacing": s2,
        "calculation_provenance": (
            "S0 last fully resolved prebreak +z c=0.5 minimum; same-time "
            "paired S4 raw phase-excess-area GB-on-minus-off ridge; 0.5-grid "
            "rounding; direct post-rounding sign-lobe audit"
        ),
        "alignment": {
            "selection_rule": ALIGNMENT_SELECTION_RULE,
            "reference_step": reference.step,
            "s0_thinning": thinning,
            "paired_s4_inward_features": features,
            "unrounded_spacings": unrounded,
            "rounded_spacings": rounded,
            "rounding_grid": GRID_SPACING,
            "opposite_sign_lobe_overlap": overlap,
            "opposite_sign_lobe_overlap_passed": overlap_passed,
        },
        "rho_calibration": rho,
        "null_model_preregistration": {
            "equation": (
                "r_null(z,t;s)=r_S0(z,t)+"
                "[r_S4_on(z-s,t)-r_S4_off(z-s,t)]"
            ),
            "profile_observable": "raw_phase_excess_area_radius",
            "residual": "r_coupled_raw-r_null_raw",
            "periodic_planes": (
                "full periodic paired-S4 profile shifted by declared spacing"
            ),
            "site_selection_only_smoothing_sigma_cells": (
                INTERFACE_WIDTH / GRID_SPACING
            ),
            "smoothing_enters_additive_profile_or_residual": False,
            "rho_critical": rho["rho_critical"],
            "crossing_interpolation": "earliest linear crossing",
            "must_be_frozen_before_S1_or_S2": True,
        },
        "provenance": {
            "s0": s0_provenance,
            "paired_s4": s4_provenance,
            "validation_prerequisites": prerequisites,
            "alignment_module_path": str(Path(__file__).resolve()),
            "alignment_module_sha256": sha256_path(Path(__file__).resolve()),
            "null_model_module_path": str(NULL_MODEL_PATH.resolve()),
            "null_model_module_sha256": sha256_path(NULL_MODEL_PATH),
        },
        "execution_scope": {
            "simulation_steps_executed": 0,
            "S1_S2_launch_enabled_by_this_lock": True,
            "spacing_sweep_enabled": False,
        },
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def verify_alignment_lock(path: Path) -> dict[str, Any]:
    """Recompute a lock from its hash-pinned sources and require exact equality."""

    path = path.resolve()
    stored = _read_json(path)
    if stored.get("lock_version") != 1 or stored.get("frozen") is not True:
        raise ValueError("alignment lock must be frozen version 1")
    expected_digest = stored.get("payload_sha256")
    unsigned = dict(stored)
    unsigned.pop("payload_sha256", None)
    if expected_digest != canonical_sha256(unsigned):
        raise ValueError("alignment lock payload digest mismatch")
    recomputed = compute_alignment_lock_payload(
        Path(str(stored["s0_summary_path"])),
        Path(str(stored["s4_summary_path"])),
    )
    if stored != recomputed:
        raise ValueError("alignment lock no longer reproduces from its sources")
    return stored


def create_alignment_lock(
    s0_summary_path: Path,
    s4_summary_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Atomically create and independently reload one non-overwriting lock."""

    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite alignment lock: {output}")
    payload = compute_alignment_lock_payload(s0_summary_path, s4_summary_path)
    atomic_json(output, payload)
    return verify_alignment_lock(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("s0_summary", type=Path)
    parser.add_argument("s4_summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.verify_only:
        payload = verify_alignment_lock(arguments.output)
    else:
        payload = create_alignment_lock(
            arguments.s0_summary,
            arguments.s4_summary,
            arguments.output,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
