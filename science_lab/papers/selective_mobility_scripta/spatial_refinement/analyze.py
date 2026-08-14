#!/usr/bin/env python3
"""Frozen paired acceptance analysis for the two fine-grid trajectories."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from science_lab.papers.nanowire_gb_junction.roy_fixed_gb_bridge.provenance import (
    sha256_path,
)
from science_lab.papers.nanowire_gb_junction.roy_gb_junction_sentinel.storage import (
    atomic_json,
)

from . import freeze, protocol, runner


def _event_row(summary: dict[str, Any]) -> dict[str, Any]:
    event = summary.get("event_assessment", {})
    bracket = event.get("event_bracket")
    gaps = event.get("persistent_gaps", [])
    midpoint = (
        0.5 * (float(bracket[0]) + float(bracket[1]))
        if isinstance(bracket, list) and len(bracket) == 2
        else None
    )
    gap_rows = [
        {
            "wire": gap.get("wire"),
            "site": float(gap["first_midpoint"]),
            "site_abs": abs(float(gap["first_midpoint"])),
        }
        for gap in gaps
    ]
    return {
        "detected": event.get("detected") is True,
        "event_bracket": bracket,
        "event_midpoint": midpoint,
        "persistent_gap_count": len(gaps),
        "persistent_gaps": gap_rows,
        "confirmation_step": event.get("confirmation_step"),
    }


def _health_row(summary: dict[str, Any]) -> dict[str, Any]:
    records = summary.get("records", [])
    mass = [
        float(row["relative_mass_drift"])
        for row in records
        if row.get("relative_mass_drift") is not None
    ]
    rebound = [
        float(row["relative_energy_rebound"])
        for row in records
        if row.get("relative_energy_rebound") is not None
    ]
    return {
        "health_passed": summary.get("health_passed") is True,
        "maximum_relative_mass_drift": max(mass, default=None),
        "maximum_sampled_relative_energy_rebound": max(rebound, default=None),
    }


def analyze_summaries(
    summaries: dict[str, dict[str, Any]],
    *,
    guard_passed: bool,
) -> dict[str, Any]:
    if set(summaries) != {"untreated", "c34"}:
        raise ValueError("analysis requires exactly untreated and c34")
    events = {key: _event_row(value) for key, value in summaries.items()}
    health = {key: _health_row(value) for key, value in summaries.items()}
    both_events = all(row["detected"] for row in events.values())
    bounded_multiplicity = all(
        protocol.PERSISTENT_GAP_COUNT_BOUNDS[0]
        <= row["persistent_gap_count"]
        <= protocol.PERSISTENT_GAP_COUNT_BOUNDS[1]
        for row in events.values()
    )
    natural = all(
        row["persistent_gaps"]
        and all(
            gap["wire"] in protocol.NATURAL_WIRES
            and protocol.NATURAL_SITE_ABS_BOUNDS[0]
            <= float(gap["site_abs"])
            <= protocol.NATURAL_SITE_ABS_BOUNDS[1]
            for gap in row["persistent_gaps"]
        )
        for row in events.values()
    )
    strictly_earlier = bool(
        both_events
        and float(events["c34"]["event_bracket"][1])
        < float(events["untreated"]["event_bracket"][0])
    )
    midpoint_checks = {
        key: bool(
            row["event_midpoint"] is not None
            and abs(
                float(row["event_midpoint"])
                - protocol.COARSE_EVENT_MIDPOINTS[key]
            )
            / protocol.COARSE_EVENT_MIDPOINTS[key]
            <= protocol.EVENT_MIDPOINT_RELATIVE_LIMIT
        )
        for key, row in events.items()
    }
    paired_effect = (
        float(events["c34"]["event_midpoint"])
        - float(events["untreated"]["event_midpoint"])
        if both_events
        else None
    )
    effect_check = bool(
        paired_effect is not None
        and protocol.PAIRED_EFFECT_BOUNDS[0]
        <= paired_effect
        <= protocol.PAIRED_EFFECT_BOUNDS[1]
    )
    checks = {
        "discarded_timestep_guard_passed": bool(guard_passed),
        "both_trajectories_terminal_and_healthy": all(
            value.get("status") in {"completed_event", "completed_horizon"}
            and health[key]["health_passed"]
            for key, value in summaries.items()
        ),
        "both_events_detected_by_horizon": both_events,
        "both_event_gap_multiplicities_between_one_and_two": bounded_multiplicity,
        "c34_complete_bracket_strictly_earlier": strictly_earlier,
        "all_event_gaps_natural_local_sites": natural,
        "untreated_midpoint_within_five_percent": midpoint_checks["untreated"],
        "c34_midpoint_within_five_percent": midpoint_checks["c34"],
        "paired_effect_within_minus_125_minus_75": effect_check,
    }
    passed = all(checks.values())
    if passed:
        classification = "spatial_refinement_validation_passed"
    elif not checks["both_trajectories_terminal_and_healthy"]:
        classification = "spatial_refinement_validation_invalid_numerics"
    elif not both_events:
        classification = "spatial_refinement_validation_censored_or_event_absent"
    else:
        classification = "spatial_refinement_validation_failed_frozen_bounds"
    return {
        "schema_version": protocol.SCHEMA_VERSION,
        "protocol_id": protocol.PROTOCOL_ID,
        "status": "completed",
        "classification": classification,
        "passed": passed,
        "checks": checks,
        "events": events,
        "health": health,
        "paired_effect_c34_minus_untreated": paired_effect,
        "coarse_reference": {
            "event_brackets": {
                key: list(value)
                for key, value in protocol.COARSE_EVENT_BRACKETS.items()
            },
            "event_midpoints": dict(protocol.COARSE_EVENT_MIDPOINTS),
            "event_gaps": protocol.COARSE_EVENT_GAPS,
            "paired_effect": -100.0,
            "exact_gap_multiplicity_convergence_claimed": False,
        },
        "interpretation": (
            "factor_two_spatial_refinement_supports_the_coarse_harmful_effect"
            if passed
            else "frozen_validation_did_not_pass;_no_adaptive_follow_on_authorized"
        ),
        "adaptive_follow_on_authorized": False,
    }


def run_analysis(
    output_root: Path, *, source_root: Path | None = None
) -> dict[str, Any]:
    output_root = output_root.resolve()
    resolved_source = protocol.source_paths(source_root).root
    freeze.verify_preflight_binding(
        output_root,
        protocol_id=protocol.PROTOCOL_ID,
        source_root=resolved_source,
    )
    guard = runner.verify_timestep_guard(
        output_root, source_root=resolved_source
    )
    summaries = {
        case_id: runner.verify_trajectory(
            output_root, case_id, source_root=resolved_source
        )
        for case_id in ("untreated", "c34")
    }
    report = analyze_summaries(summaries, guard_passed=guard["passed"] is True)
    report["inputs"] = {
        "timestep_guard_summary": {
            "path": str((output_root / "timestep-guard" / "summary.json").resolve()),
            "sha256": sha256_path(output_root / "timestep-guard" / "summary.json"),
        },
        "trajectories": {
            key: {
                "path": str((output_root / f"case-{key}" / "summary.json").resolve()),
                "sha256": sha256_path(
                    output_root / f"case-{key}" / "summary.json"
                ),
            }
            for key in summaries
        },
    }
    report["generated_unix_time"] = time.time()
    analysis_output = output_root / "analysis"
    analysis_output.mkdir(exist_ok=True)
    atomic_json(analysis_output / "summary.json", report)
    return report


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=protocol.DEFAULT_OUTPUT)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=protocol.EXTERNAL_SOURCE_DIRECTORY,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    report = run_analysis(args.output, source_root=args.source_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    # A frozen scientific fail is still a successfully completed analysis.
    # Operational exceptions retain a nonzero exit through the traceback.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
