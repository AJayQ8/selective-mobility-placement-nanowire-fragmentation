#!/usr/bin/env python3
"""Reanalyze the completed repeat panel with one-arm event semantics.

This module never evolves a field.  It preserves the immutable v1 execution
aggregate and writes a separately versioned analysis with complete input
provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..roy_2021_reproduction.model import FROZEN_DEG90
from ..roy_fixed_gb_bridge.provenance import sha256_path
from ..roy_gb_junction_sentinel import diagnostics as event_diagnostics
from ..roy_gb_junction_sentinel.diagnostics import instantaneous_pinches
from ..roy_gb_junction_sentinel.storage import atomic_json
from . import analyze_position_scan_response as response_analysis
from . import paired_repeat_protocol as protocol
from . import position_scan_protocol as position_protocol
from . import run_paired_repeat_case as case_runner
from . import run_paired_repeat_panel as frozen_panel_runner
from . import run_paired_repeat_source as source_runner
from . import run_position_scan_case as case_engine


SOURCE_PATH = Path(__file__).resolve()
DIRECTORY = SOURCE_PATH.parent
PANEL_OUTPUT = protocol.RESULTS / "paired_repeat_panel_v1"
DEFAULT_OUTPUT = protocol.RESULTS / "paired_repeat_panel_analysis_v2"
LEGACY_UNTREATED_SUMMARY = (
    DIRECTORY.parent
    / "roy_2021_reproduction"
    / "results"
    / "t2000_continuation_source_semantic_seed2292"
    / "summary.json"
)
LEGACY_THRESHOLDS = ("0.45", "0.50", "0.55")
REQUIRED_RECORDS = 3
DIAGNOSTIC_INTERVAL = 10
RELOCATION_MARGIN = 10.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _robust_legacy_arms(record: dict[str, Any]) -> set[str]:
    topology = record["distal_four_arm_topology"]
    detached = [
        set(topology["thresholds"][level]["detached_arms"])
        for level in LEGACY_THRESHOLDS
    ]
    return set.intersection(*detached)


def legacy_single_arm_timing(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Derive the persistent one-arm clock from legacy connectivity records."""

    ordered = sorted(records, key=lambda item: int(item["step"]))
    steps = [int(item["step"]) for item in ordered]
    if any(later <= earlier for earlier, later in zip(steps, steps[1:])):
        raise ValueError("legacy record steps must be strictly increasing")
    robust = [_robust_legacy_arms(item) for item in ordered]
    for start in range(0, len(ordered) - REQUIRED_RECORDS + 1):
        window_steps = steps[start : start + REQUIRED_RECORDS]
        if any(
            later - earlier != DIAGNOSTIC_INTERVAL
            for earlier, later in zip(window_steps, window_steps[1:])
        ):
            continue
        persistent = set.intersection(
            *robust[start : start + REQUIRED_RECORDS]
        )
        if not persistent:
            continue
        if len(persistent) != 1:
            raise RuntimeError(
                "legacy first-arm event is not unique: "
                f"{sorted(persistent)}"
            )
        if start == 0:
            raise RuntimeError(
                "legacy event lacks a preceding diagnostic bracket"
            )
        return {
            "detected": True,
            "arm": next(iter(persistent)),
            "event_bracket": [steps[start - 1], window_steps[0]],
            "first_candidate_step": window_steps[0],
            "confirmation_step": window_steps[-1],
            "required_records": REQUIRED_RECORDS,
            "diagnostic_interval": DIAGNOSTIC_INTERVAL,
        }
    return {
        "detected": False,
        "arm": None,
        "event_bracket": None,
        "first_candidate_step": None,
        "confirmation_step": None,
        "required_records": REQUIRED_RECORDS,
        "diagnostic_interval": DIAGNOSTIC_INTERVAL,
    }


def _split_arm(arm: str) -> tuple[str, str]:
    for branch in ("plus", "minus"):
        suffix = f"_{branch}"
        if arm.endswith(suffix):
            return arm[: -len(suffix)], branch
    raise ValueError(f"legacy arm has no signed branch: {arm}")


def _resolve_recorded_path(
    recorded: str,
    *,
    summary_directory: Path,
) -> Path:
    """Resolve an absolute recorded path, with a relocatable local fallback."""

    direct = Path(recorded)
    if direct.is_file():
        return direct.resolve()
    fallback = summary_directory / direct.name
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(
        f"recorded artifact is absent: {direct}; fallback: {fallback}"
    )


def _checkpoint_entry(
    summary: dict[str, Any], step: int
) -> dict[str, Any]:
    matches = [
        item
        for item in summary["checkpoints"]
        if int(item["step"]) == step
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one checkpoint at {step}, found {len(matches)}"
        )
    return matches[0]


def _verified_checkpoint(
    summary: dict[str, Any],
    summary_path: Path,
    *,
    step: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    entry = _checkpoint_entry(summary, step)
    directory = summary_path.parent
    field_path = _resolve_recorded_path(
        entry["field_path"], summary_directory=directory
    )
    metadata_path = _resolve_recorded_path(
        entry["metadata_path"], summary_directory=directory
    )
    field_hash = sha256_path(field_path)
    metadata_hash = sha256_path(metadata_path)
    if field_hash != entry["field_sha256"]:
        raise RuntimeError(f"field hash mismatch at {step}")
    if metadata_hash != entry["metadata_sha256"]:
        raise RuntimeError(f"metadata hash mismatch at {step}")
    metadata = _load_json(metadata_path)
    field = np.load(field_path, mmap_mode="r", allow_pickle=False)
    checks = {
        "field_sha256": field_hash == entry["field_sha256"],
        "metadata_sha256": metadata_hash == entry["metadata_sha256"],
        "shape": list(field.shape) == list(entry["shape"]),
        "dtype": field.dtype.str == entry["dtype"],
        "metadata_field_sha256": (
            metadata["field_sha256"] == entry["field_sha256"]
        ),
        "metadata_shape": list(metadata["shape"]) == list(entry["shape"]),
        "metadata_dtype": metadata["dtype"] == entry["dtype"],
        "step": int(metadata["step"]) == step,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"checkpoint validation failed at {step}: {checks}"
        )
    return field, {
        "step": step,
        "field_path": str(field_path),
        "field_sha256": field_hash,
        "metadata_path": str(metadata_path),
        "metadata_sha256": metadata_hash,
        "shape": list(field.shape),
        "dtype": field.dtype.str,
        "checks": checks,
    }


def _legacy_site(
    field: np.ndarray,
    *,
    wire: str,
    branch: str,
) -> float:
    pinches = instantaneous_pinches(
        field,
        geometry="crossed",
        spacing=FROZEN_DEG90.lattice.spacing,
        radius=(
            FROZEN_DEG90.radius_1
            * FROZEN_DEG90.lattice.spacing
        ),
    )
    matches = [
        gap
        for gap in pinches["candidate_gaps"]
        if gap["wire"] == wire
        and gap["branch"] == branch
        and gap["site_robust"] is True
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"legacy site is not unique for {wire}/{branch}: {matches}"
        )
    return float(matches[0]["midpoint"])


def _legacy_untreated_observation() -> tuple[
    dict[str, Any], list[dict[str, Any]]
]:
    """Compatibility reconstruction, not a bit-identical detector replay.

    Timing is derived from stored three-threshold connectivity at t=1690,
    1700, 1710, and 1720.  The newer full-transverse diagnostic is verified
    on the saved t=1700 and t=1720 fields.  No t=1710 full field exists, so
    the adapter must remain explicitly labelled as a compatibility result.
    """

    summary = _load_json(LEGACY_UNTREATED_SUMMARY)
    timing = legacy_single_arm_timing(summary["records"])
    if not timing["detected"]:
        raise RuntimeError("legacy untreated event was not detected")
    wire, branch = _split_arm(timing["arm"])
    first_step = int(timing["first_candidate_step"])
    confirmation_step = int(timing["confirmation_step"])
    first_field, first_manifest = _verified_checkpoint(
        summary,
        LEGACY_UNTREATED_SUMMARY,
        step=first_step,
    )
    confirmation_field, confirmation_manifest = _verified_checkpoint(
        summary,
        LEGACY_UNTREATED_SUMMARY,
        step=confirmation_step,
    )
    first_site = _legacy_site(
        first_field, wire=wire, branch=branch
    )
    confirmation_site = _legacy_site(
        confirmation_field, wire=wire, branch=branch
    )
    bracket = timing["event_bracket"]
    return (
        {
            "detected": True,
            "censored_at": None,
            "event_midpoint": 0.5 * (bracket[0] + bracket[1]),
            "event_bracket": bracket,
            "first_candidate_step": first_step,
            "confirmation_step": confirmation_step,
            "sites": [
                {
                    "wire": wire,
                    "branch": branch,
                    "first_site": first_site,
                    "confirmation_site": confirmation_site,
                    "first_site_abs": abs(first_site),
                    "confirmation_site_abs": abs(confirmation_site),
                }
            ],
            "detector": {
                "name": (
                    "persistent_single_arm_three_threshold_"
                    "legacy_compatibility_adapter_v1"
                ),
                "required_records": timing["required_records"],
                "diagnostic_interval": timing["diagnostic_interval"],
                "thresholds": list(LEGACY_THRESHOLDS),
                "full_transverse_checks_at": [
                    first_step,
                    confirmation_step,
                ],
                "missing_full_field_at_intermediate_step": 1710,
                "derived_not_pinned": True,
                "literally_identical_to_modern_detector": False,
            },
            "source": str(LEGACY_UNTREATED_SUMMARY.resolve()),
            "source_sha256": sha256_path(LEGACY_UNTREATED_SUMMARY),
        },
        [first_manifest, confirmation_manifest],
    )


def _modern_observation(summary: dict[str, Any]) -> dict[str, Any]:
    event = case_engine.production_helper._event_from_records(
        summary["records"], latched=None
    )
    if event != summary["event_assessment"]:
        raise RuntimeError(
            "stored event differs from analysis-time reconstruction"
        )
    if not event.get("detected"):
        return {
            "detected": False,
            "censored_at": int(summary["completed_step"]),
            "event_midpoint": None,
            "event_bracket": None,
            "first_candidate_step": None,
            "confirmation_step": None,
            "sites": [],
            "detector": {
                "name": "persistent_single_arm_event_v1",
                "derived_not_pinned": True,
            },
        }
    bracket = [int(value) for value in event["event_bracket"]]
    sites = [
        {
            "wire": gap["wire"],
            "branch": gap["first_branch"],
            "first_site": float(gap["first_midpoint"]),
            "confirmation_site": float(gap["confirmation_midpoint"]),
            "first_site_abs": abs(float(gap["first_midpoint"])),
            "confirmation_site_abs": abs(
                float(gap["confirmation_midpoint"])
            ),
        }
        for gap in event["persistent_gaps"]
    ]
    return {
        "detected": True,
        "censored_at": None,
        "event_midpoint": 0.5 * sum(bracket),
        "event_bracket": bracket,
        "first_candidate_step": int(event["first_candidate_step"]),
        "confirmation_step": int(event["confirmation_step"]),
        "sites": sites,
        "detector": {
            "name": "persistent_single_arm_event_v1",
            "required_records": event["required_records"],
            "diagnostic_interval": event["diagnostic_interval"],
            "maximum_gap_displacement": event[
                "maximum_gap_displacement"
            ],
            "derived_not_pinned": True,
            "literally_identical_to_modern_detector": True,
        },
    }


def paired_pattern(
    observations: dict[str, Any],
) -> dict[str, Any]:
    cases = protocol.CASE_ORDER
    all_detected = all(
        observations[case_id]["detected"] for case_id in cases
    )
    if not all_detected:
        return {
            "all_four_events_detected": False,
            "c18_relocated_downstream": None,
            "c26_later_than_untreated": None,
            "c34_earlier_than_untreated": None,
            "full_qualitative_pattern": None,
        }
    untreated_abs = [
        site["first_site_abs"]
        for site in observations["untreated"]["sites"]
    ]
    relocated_abs = [
        site["first_site_abs"]
        for site in observations["c18p5"]["sites"]
    ]
    times = {
        case_id: float(observations[case_id]["event_midpoint"])
        for case_id in cases
    }
    checks = {
        "all_four_events_detected": True,
        "c18_relocated_downstream": (
            min(relocated_abs)
            > max(untreated_abs) + RELOCATION_MARGIN
        ),
        "c26_later_than_untreated": (
            times["c26p5"] > times["untreated"]
        ),
        "c34_earlier_than_untreated": (
            times["c34p5"] < times["untreated"]
        ),
    }
    checks["full_qualitative_pattern"] = all(checks.values())
    checks["site_comparison"] = {
        "coordinate": "absolute_distance_from_junction",
        "all_simultaneous_persistent_sites_used": True,
        "untreated_maximum_abs_site": max(untreated_abs),
        "c18_minimum_abs_site": min(relocated_abs),
        "required_margin": RELOCATION_MARGIN,
    }
    return checks


def _paired_effects(
    observations: dict[str, Any],
) -> dict[str, dict[str, float]]:
    baseline = float(observations["untreated"]["event_midpoint"])
    return {
        case_id: {
            "delta_time_vs_paired_untreated": (
                float(observations[case_id]["event_midpoint"])
                - baseline
            ),
            "relative_delta_time_vs_paired_untreated": (
                float(observations[case_id]["event_midpoint"])
                - baseline
            )
            / baseline,
        }
        for case_id in ("c18p5", "c26p5", "c34p5")
    }


def _verify_v1_immutable() -> dict[str, Any]:
    summary_path = PANEL_OUTPUT / "summary.json"
    contract_path = PANEL_OUTPUT / "contract.json"
    aggregate_path = PANEL_OUTPUT / "aggregate.json"
    summary = _load_json(summary_path)
    contract = _load_json(contract_path)
    checks = {
        "status": summary["status"] == "completed",
        "classification": (
            summary["classification"] == "paired_repeat_panel_completed"
        ),
        "aggregate_sha256": (
            sha256_path(aggregate_path)
            == summary["aggregate"]["sha256"]
        ),
        "runner_sha256": (
            sha256_path(Path(frozen_panel_runner.__file__).resolve())
            == summary["contract"]["provenance"][
                "panel_runner_sha256"
            ]
        ),
        "contract_content": contract == summary["contract"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"v1 immutability check failed: {checks}")
    return {
        "checks": checks,
        "summary": {
            "path": str(summary_path.resolve()),
            "sha256": sha256_path(summary_path),
        },
        "contract": {
            "path": str(contract_path.resolve()),
            "sha256": sha256_path(contract_path),
        },
        "aggregate": {
            "path": str(aggregate_path.resolve()),
            "sha256": sha256_path(aggregate_path),
        },
    }


def _verify_existing_treated_case(
    case_id: str, path: Path
) -> dict[str, Any]:
    case = position_protocol.CASES_BY_SLUG[case_id]
    row = response_analysis.verify_completed_case_summary(case, path)
    return {
        "passed": all(row["validation"]["contract_checks"].values())
        and all(row["validation"]["execution_checks"].values())
        and all(row["validation"]["event_checks"].values())
        and row["health"]["passed"],
        "input_sha256": row["input_sha256"],
    }


def build_report() -> dict[str, Any]:
    v1 = _verify_v1_immutable()
    legacy, checkpoint_manifest = _legacy_untreated_observation()
    realizations: dict[str, Any] = {
        str(protocol.EXISTING_SEED): {
            "observations": {"untreated": legacy},
            "validation": {
                "untreated_legacy_compatibility": True,
            },
        }
    }
    input_manifest: dict[str, Any] = {
        "analysis_code": {
            "analyzer": {
                "path": str(SOURCE_PATH),
                "sha256": sha256_path(SOURCE_PATH),
            },
            "modern_detector": {
                "path": str(
                    Path(event_diagnostics.__file__).resolve()
                ),
                "sha256": sha256_path(
                    Path(event_diagnostics.__file__).resolve()
                ),
            },
            "frozen_panel_runner": {
                "path": str(
                    Path(frozen_panel_runner.__file__).resolve()
                ),
                "sha256": sha256_path(
                    Path(frozen_panel_runner.__file__).resolve()
                ),
            },
        },
        "immutable_panel_v1": v1,
        "case_summaries": {
            str(protocol.EXISTING_SEED): {
                "untreated": {
                    "path": str(LEGACY_UNTREATED_SUMMARY.resolve()),
                    "sha256": sha256_path(LEGACY_UNTREATED_SUMMARY),
                }
            }
        },
        "legacy_event_checkpoints": checkpoint_manifest,
    }

    for case_id, path in frozen_panel_runner.EXISTING_SEED_PATHS.items():
        validation = _verify_existing_treated_case(case_id, path)
        if not validation["passed"]:
            raise RuntimeError(
                f"existing treated case failed validation: {case_id}"
            )
        summary = _load_json(path)
        realizations[str(protocol.EXISTING_SEED)]["observations"][
            case_id
        ] = _modern_observation(summary)
        realizations[str(protocol.EXISTING_SEED)]["observations"][
            case_id
        ]["source"] = str(path.resolve())
        realizations[str(protocol.EXISTING_SEED)]["validation"][
            case_id
        ] = validation
        input_manifest["case_summaries"][
            str(protocol.EXISTING_SEED)
        ][case_id] = {
            "path": str(path.resolve()),
            "sha256": validation["input_sha256"],
        }

    for seed in protocol.NEW_SEEDS:
        source_output = protocol.source_output(PANEL_OUTPUT, seed)
        source = source_runner.verify_completed_source(
            source_output, seed=seed
        )
        observations: dict[str, Any] = {}
        validations: dict[str, Any] = {}
        input_manifest["case_summaries"][str(seed)] = {}
        for case_id in protocol.CASE_ORDER:
            output = protocol.case_output(PANEL_OUTPUT, seed, case_id)
            validation = case_runner.verify_completed_case(
                output,
                seed=seed,
                case_id=case_id,
                source=source,
            )
            if not all(validation["checks"].values()):
                raise RuntimeError(
                    f"new case failed validation: {seed}/{case_id}"
                )
            path = output / "summary.json"
            observation = _modern_observation(_load_json(path))
            observation["source"] = str(path.resolve())
            observations[case_id] = observation
            validations[case_id] = validation
            input_manifest["case_summaries"][str(seed)][case_id] = {
                "path": str(path.resolve()),
                "sha256": sha256_path(path),
            }
        realizations[str(seed)] = {
            "source": source,
            "observations": observations,
            "validation": validations,
        }

    for realization in realizations.values():
        realization["paired_effects"] = _paired_effects(
            realization["observations"]
        )
        realization["paired_pattern"] = paired_pattern(
            realization["observations"]
        )
    expected_effects = {
        "2292": {"c18p5": 750.0, "c26p5": 290.0, "c34p5": -110.0},
        "104729": {
            "c18p5": 760.0,
            "c26p5": 270.0,
            "c34p5": -110.0,
        },
        "130363": {
            "c18p5": 770.0,
            "c26p5": 260.0,
            "c34p5": -100.0,
        },
    }
    effect_checks = {
        seed: {
            case_id: (
                realizations[seed]["paired_effects"][case_id][
                    "delta_time_vs_paired_untreated"
                ]
                == expected
            )
            for case_id, expected in cases.items()
        }
        for seed, cases in expected_effects.items()
    }
    if not all(
        passed
        for cases in effect_checks.values()
        for passed in cases.values()
    ):
        raise RuntimeError(
            f"paired effect regression failed: {effect_checks}"
        )

    legacy_summary = _load_json(LEGACY_UNTREATED_SUMMARY)
    two_arm = legacy_summary["breakup_assessment"][
        "common_threshold_robust_event"
    ]
    two_arm_bracket = two_arm["event_bracket"]
    full_flags = [
        item["paired_pattern"]["full_qualitative_pattern"]
        for item in realizations.values()
    ]
    return {
        "schema_version": 2,
        "status": "completed",
        "classification": (
            "paired_repeat_panel_single_arm_reanalysis_complete"
        ),
        "realizations": realizations,
        "validation_clocks": {
            "production_comparison": {
                "definition": "first persistent single-arm failure",
                "seed_2292_event_midpoint": legacy["event_midpoint"],
                "derived_at_analysis_time": True,
            },
            "roy_reproduction_only": {
                "definition": (
                    "at least two of the same arms detached at all three "
                    "thresholds for three consecutive observations"
                ),
                "event_bracket": two_arm_bracket,
                "event_midpoint": 0.5 * sum(two_arm_bracket),
                "confirmation_step": two_arm["confirmation_step"],
                "roy_example_time": legacy_summary[
                    "breakup_assessment"
                ]["paper_example_trajectory_breakup_time"],
                "excluded_from_production_differences": True,
            },
        },
        "panel_assessment": {
            "realization_count": len(realizations),
            "trajectory_count": 12,
            "full_pattern_flags": full_flags,
            "full_pattern_reproduced_in_all_three": all(
                flag is True for flag in full_flags
            ),
            "paired_effect_regression_checks": effect_checks,
            "interpretation_requires_scientific_review": True,
        },
        "input_manifest": input_manifest,
        "limitations": [
            (
                "Seed 2292 uses a compatibility adapter because its legacy "
                "records contain robust distal-arm connectivity, not the "
                "modern pinch payload."
            ),
            (
                "The modern full-transverse detector is checked on saved "
                "seed-2292 fields at t=1700 and 1720; no t=1710 full field "
                "was retained."
            ),
            (
                "The original v1 aggregate remains immutable and mixes "
                "the older two-arm seed-2292 clock with modern one-arm "
                "clocks; it is execution history, not the corrected table."
            ),
        ],
        "scope": {
            "stored_data_reanalysis_only": True,
            "simulation_steps_performed": 0,
        },
    }


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = build_report()
    atomic_json(output / "summary.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "classification": report["classification"],
                "output": str((output / "summary.json").resolve()),
                "full_pattern_reproduced_in_all_three": report[
                    "panel_assessment"
                ]["full_pattern_reproduced_in_all_three"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
