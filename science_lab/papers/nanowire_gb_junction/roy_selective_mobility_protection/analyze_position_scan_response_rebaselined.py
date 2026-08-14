#!/usr/bin/env python3
"""Rebaseline the frozen position scan to the one-arm untreated clock.

The original response analysis remains immutable execution history.  This
stored-data-only analysis rebuilds and validates it, derives the seed-2292
one-arm clock through the compatibility adapter, and recomputes every
difference against that clock.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..roy_fixed_gb_bridge.provenance import sha256_path
from ..roy_gb_junction_sentinel.storage import atomic_json
from . import analyze_paired_repeat_panel as paired_analysis
from . import analyze_position_scan_response as frozen_analysis


SOURCE_PATH = Path(__file__).resolve()
RESULTS = SOURCE_PATH.parent / "results"
FROZEN_OUTPUT = RESULTS / "position_scan_response_v1"
DEFAULT_OUTPUT = RESULTS / "position_scan_response_rebaselined_v2"
EXPECTED_DELTAS = {
    "c14p5": 850.0,
    "c18p5": 750.0,
    "c22p5": 680.0,
    "c26p5": 290.0,
    "c30p5": -10.0,
    "c34p5": -110.0,
    "c38p5": -100.0,
    "c64p5": 0.0,
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report() -> dict[str, Any]:
    frozen_path = FROZEN_OUTPUT / "summary.json"
    frozen_stored = _load_json(frozen_path)
    frozen_live = frozen_analysis.build_report()
    if frozen_stored != frozen_live:
        raise RuntimeError(
            "stored position response differs from live reconstruction"
        )
    paired = paired_analysis.build_report()
    untreated = paired["realizations"]["2292"]["observations"][
        "untreated"
    ]
    baseline = float(untreated["event_midpoint"])
    rows: list[dict[str, Any]] = []
    regression: dict[str, bool] = {}
    for frozen_row in frozen_live["cases"]:
        row = dict(frozen_row)
        midpoint = row["event_time_midpoint"]
        if midpoint is None:
            delta = None
            relative = None
        else:
            delta = float(midpoint) - baseline
            relative = delta / baseline
        row["legacy_two_arm_delta_time_vs_untreated"] = row[
            "delta_time_vs_untreated"
        ]
        row[
            "legacy_two_arm_relative_delta_time_vs_untreated"
        ] = row["relative_delta_time_vs_untreated"]
        row["delta_time_vs_untreated"] = delta
        row["relative_delta_time_vs_untreated"] = relative
        row["production_baseline_definition"] = (
            "first persistent single-arm failure"
        )
        rows.append(row)
        expected = EXPECTED_DELTAS[row["case_id"]]
        regression[row["case_id"]] = delta == expected
    if not all(regression.values()):
        raise RuntimeError(
            f"corrected response regression failed: {regression}"
        )

    legacy_reference = frozen_live["references"]["untreated"]
    return {
        "schema_version": 2,
        "status": "complete",
        "classification": (
            "position_scan_response_single_arm_rebaselined"
        ),
        "contract": frozen_live["contract"],
        "references": {
            "production_untreated": {
                "definition": "first persistent single-arm failure",
                "event_bracket": untreated["event_bracket"],
                "event_time_midpoint": baseline,
                "event_sites": untreated["sites"],
                "detector": untreated["detector"],
            },
            "roy_validation_only": {
                "definition": (
                    "persistent two-arm failure retained only for the "
                    "like-for-like Roy reproduction"
                ),
                "event_bracket": legacy_reference["event_bracket"],
                "event_time_midpoint": legacy_reference[
                    "event_time_midpoint"
                ],
                "event_site_abs": legacy_reference["event_site_abs"],
                "excluded_from_scan_differences": True,
            },
            "depletion": frozen_live["references"]["depletion"],
        },
        "cases": rows,
        "groups": frozen_live["groups"],
        "descriptive_summary": {
            **frozen_live["descriptive_summary"],
            "corrected_position_response": [
                {
                    "case_id": row["case_id"],
                    "center": row["center"],
                    "event_time_midpoint": row[
                        "event_time_midpoint"
                    ],
                    "delta_time_vs_untreated": row[
                        "delta_time_vs_untreated"
                    ],
                    "event_site_abs": row["event_site_abs"],
                    "mode_label": row["mode_label"],
                }
                for row in rows
            ],
            "delta_regression_checks": regression,
        },
        "numerical_integrity": frozen_live["numerical_integrity"],
        "provenance": {
            "analyzer": {
                "path": str(SOURCE_PATH),
                "sha256": sha256_path(SOURCE_PATH),
            },
            "frozen_response_analyzer": {
                "path": str(
                    Path(frozen_analysis.__file__).resolve()
                ),
                "sha256": sha256_path(
                    Path(frozen_analysis.__file__).resolve()
                ),
            },
            "frozen_response": {
                "path": str(frozen_path.resolve()),
                "sha256": sha256_path(frozen_path),
            },
            "paired_reanalysis_analyzer": {
                "path": str(
                    Path(paired_analysis.__file__).resolve()
                ),
                "sha256": sha256_path(
                    Path(paired_analysis.__file__).resolve()
                ),
            },
            "paired_reanalysis_inputs": paired["input_manifest"],
            "case_input_sha256": frozen_live["provenance"][
                "input_sha256"
            ],
        },
        "claim_boundary": [
            *frozen_live["claim_boundary"],
            (
                "The corrected untreated clock is a legacy compatibility "
                "reconstruction: connectivity records supply persistence, "
                "while saved full fields verify the modern transverse-gap "
                "criterion at the first and confirmation times."
            ),
            (
                "Changing the baseline corrects time differences only; it "
                "does not add placements, seeds, or convergence evidence."
            ),
        ],
        "scope": {
            "stored_data_reanalysis_only": True,
            "simulation_steps_performed": 0,
            "frozen_v1_artifact_modified": False,
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
                "corrected_deltas": {
                    row["case_id"]: row[
                        "delta_time_vs_untreated"
                    ]
                    for row in report["cases"]
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
