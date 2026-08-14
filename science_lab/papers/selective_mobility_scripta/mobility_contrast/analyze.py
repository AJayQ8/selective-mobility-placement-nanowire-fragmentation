#!/usr/bin/env python3
"""Audit the four frozen m=0.3 results against the pre-run criteria."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from science_lab.papers.nanowire_gb_junction.roy_gb_junction_sentinel.storage import (
    atomic_json,
)

from .common import (
    ANALYSIS_PATH,
    CAMPAIGN_ID,
    CASE_ORDER,
    CONTRACT_PATH,
    FROZEN_IMPLEMENTATION_SNAPSHOT,
    PREFLIGHT_PATH,
    frozen_contract,
    load_json,
    raw_case_output,
    sha256_path,
)


def _health(summary: dict[str, Any]) -> dict[str, Any]:
    records = summary["records"]
    finite = all(record["field"]["finite"] is True for record in records)
    mass = max(abs(float(record["relative_mass_drift"])) for record in records)
    energies = [
        float(record["free_energy"])
        for record in records
        if record.get("free_energy") is not None
    ]
    rebounds = [
        current - previous
        for previous, current in zip(energies, energies[1:])
        if current > previous * (1.0 + 1.0e-6)
    ]
    checks = {
        "finite": finite,
        "mass_drift": mass <= 1.0e-4,
        "sampled_energy": not rebounds,
        "terminal_not_numerical_abort": summary.get("status") != "numerically_aborted",
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "maximum_absolute_relative_mass_drift": mass,
        "sampled_energy_rebound_count": len(rebounds),
    }


def _case_observation(
    seed: int,
    case_id: str,
    *,
    preflight_sha256: str,
) -> dict[str, Any]:
    output = raw_case_output(seed, case_id)
    summary_path = output / "summary.json"
    summary = load_json(summary_path)
    event = summary["event_assessment"]
    contract = summary["contract"]
    frozen = frozen_contract()
    source_bound = frozen["bound_sources"][str(seed)]
    gaps = event.get("persistent_gaps", [])
    sites = [abs(float(gap["first_midpoint"])) for gap in gaps]
    topology_checks = {
        "at_least_one_persistent_gap": bool(sites),
        "all_within_frozen_junction_adjacent_interval": bool(sites)
        and all(14.0 <= site <= 22.5 for site in sites),
    }
    provenance_checks = {
        "campaign": contract.get("campaign_id") == CAMPAIGN_ID,
        "case": contract.get("case") == f"{case_id}_m03",
        "parent_case": contract.get("parent_case") == case_id,
        "seed": int(contract.get("seed", -1)) == seed,
        "factor": math.isclose(
            float(contract.get("protected_mobility_factor", -1.0)),
            0.3,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "science_contract": contract["provenance"]["scientific_contract_sha256"]
        == sha256_path(CONTRACT_PATH),
        "preflight": contract["provenance"]["launch_preflight_sha256"]
        == preflight_sha256,
        "source_field": contract["source_checkpoint"]["field_sha256"]
        == source_bound["source_checkpoint_field_sha256"],
        "source_metadata": contract["source_checkpoint"]["metadata_sha256"]
        == source_bound["source_checkpoint_metadata_sha256"],
    }
    completion_checks = {
        "completed": summary.get("status") == "completed",
        "robust_event": summary.get("stop_reason")
        == "robust_single_arm_event_confirmed",
        "detected": event.get("detected") is True,
        "before_or_at_target": int(summary["completed_step"]) <= 3000,
        "modern_detector": event.get("heuristic_fallback_used") is False,
    }
    bracket = [int(value) for value in event["event_bracket"]]
    return {
        "seed": seed,
        "case_id": f"{case_id}_m03",
        "role": "intermediate" if case_id == "c26p5" else "outboard",
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": sha256_path(summary_path),
        "completed_step": int(summary["completed_step"]),
        "event_bracket": bracket,
        "event_midpoint": 0.5 * sum(bracket),
        "confirmation_step": int(event["confirmation_step"]),
        "absolute_first_sites": sites,
        "failure_mode": (
            "junction_adjacent" if all(topology_checks.values()) else "other"
        ),
        "completion_checks": completion_checks,
        "provenance_checks": provenance_checks,
        "topology_checks": topology_checks,
        "health": _health(summary),
    }


def analyze(output: Path = ANALYSIS_PATH) -> dict[str, Any]:
    preflight = load_json(PREFLIGHT_PATH)
    if preflight.get("status") != "GO":
        raise RuntimeError("analysis requires the exact GO preflight")
    recorded_snapshot = preflight.get("frozen_implementation_snapshot")
    preflight_sha = sha256_path(PREFLIGHT_PATH)
    if preflight["campaign_implementation"]["analyze_sha256"] != sha256_path(
        Path(__file__).resolve()
    ):
        raise RuntimeError("analysis code changed after the GO preflight")
    observations = [
        _case_observation(seed, case_id, preflight_sha256=preflight_sha)
        for seed, case_id in CASE_ORDER
    ]
    baselines = frozen_contract()["scope"]["existing_untreated_baselines"]
    for item in observations:
        untreated = baselines[str(item["seed"])]["event_bracket"]
        treated = item["event_bracket"]
        paired = [treated[0] - untreated[1], treated[1] - untreated[0]]
        midpoint = item["event_midpoint"] - 0.5 * sum(untreated)
        item["untreated_event_bracket"] = untreated
        item["paired_shift_midpoint"] = midpoint
        item["conservative_paired_interval"] = paired
        item["sign_check"] = (
            paired[0] > 0
            if item["role"] == "intermediate"
            else paired[1] < 0
        )
        item["passed"] = all(
            (
                all(item["completion_checks"].values()),
                all(item["provenance_checks"].values()),
                all(item["topology_checks"].values()),
                item["health"]["passed"],
                item["sign_check"],
            )
        )
    checks = {
        "all_four_completed_with_robust_event": all(
            all(item["completion_checks"].values()) for item in observations
        ),
        "all_four_exactly_paired": all(
            all(item["provenance_checks"].values()) for item in observations
        ),
        "both_intermediate_intervals_strictly_positive": all(
            item["sign_check"]
            for item in observations
            if item["role"] == "intermediate"
        ),
        "both_outboard_intervals_strictly_negative": all(
            item["sign_check"]
            for item in observations
            if item["role"] == "outboard"
        ),
        "all_four_junction_adjacent": all(
            all(item["topology_checks"].values()) for item in observations
        ),
        "all_four_numerically_healthy": all(
            item["health"]["passed"] for item in observations
        ),
        "no_extra_case": sorted(
            (item["seed"], item["case_id"].removesuffix("_m03"))
            for item in observations
        )
        == sorted(CASE_ORDER),
        "frozen_implementation_snapshot_bound": recorded_snapshot
        == FROZEN_IMPLEMENTATION_SNAPSHOT,
    }
    report = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "status": "completed",
        "classification": (
            "weaker_contrast_sign_reversal_confirmed"
            if all(checks.values())
            else "weaker_contrast_frozen_gate_not_met"
        ),
        "passed": all(checks.values()),
        "checks": checks,
        "observations": observations,
        "provenance": {
            "frozen_contract_sha256": sha256_path(CONTRACT_PATH),
            "preflight_sha256": preflight_sha,
        },
        "scope": {
            "four_frozen_treated_cases_only": True,
            "no_adaptive_rescue": True,
            "no_adaptive_study_change": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ANALYSIS_PATH)
    args = parser.parse_args(argv)
    report = analyze(args.output.resolve())
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
