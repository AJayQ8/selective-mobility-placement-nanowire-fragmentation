#!/usr/bin/env python3
"""Audit probe/collar geometry and early-time separation without simulation.

The original precursor synthesis is preserved as a frozen v1 analysis.  This
follow-on audit addresses interpretation: it determines whether the fixed
phase-radius probe lies inside each collar, makes the outside-support/common-
event-mode subset primary, and measures how far the first stored signed
separation precedes the corresponding event bracket.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


SOURCE_PATH = Path(__file__).resolve()
PAPER_ROOT = SOURCE_PATH.parent
REPOSITORY_ROOT = PAPER_ROOT.parents[2]
PRECURSOR_ROOT = PAPER_ROOT / "precursor_synthesis" / "readout_v1"
SOURCE_A_PATH = PRECURSOR_ROOT / "source_a_placement_precursor.csv"
TIME_SERIES_PATH = PRECURSOR_ROOT / "matched_phase_radius_time_series.csv"
SUMMARY_PATH = PRECURSOR_ROOT / "summary.json"
NORMALIZED_PATH = PAPER_ROOT / "source_data" / "cms_normalized_response.csv"
DEFAULT_OUTPUT = PAPER_ROOT / "precursor_context_audit" / "readout_v1"

PROBE_DISTANCE = 19.25
SUPPORT_HALF_WIDTH = 6.0
PLATEAU_HALF_WIDTH = 3.0
TRANSITION_WIDTH = 3.0
MINIMUM_MULTIPLIER = 0.1
PRIMARY_CASES = ("c26p5", "c30p5", "c34p5", "c38p5", "c64p5")
OVERLAP_CASES = ("c14p5", "c18p5", "c22p5")
REPRESENTATIVE_CASES = ("c26p5", "c34p5")
EXPECTED_SIGN = {"c26p5": 1, "c34p5": -1}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    from io import StringIO

    handle = StringIO(newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=list(rows[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, handle.getvalue())


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, indent=2) + "\n")


def _pearson(x_values: Iterable[float], y_values: Iterable[float]) -> float:
    x = tuple(float(value) for value in x_values)
    y = tuple(float(value) for value in y_values)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("correlation inputs must have equal nontrivial length")
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x, y)
    )
    x_sum = sum((value - x_mean) ** 2 for value in x)
    y_sum = sum((value - y_mean) ** 2 for value in y)
    if x_sum == 0.0 or y_sum == 0.0:
        raise ValueError("correlation input is constant")
    return numerator / math.sqrt(x_sum * y_sum)


def _ranks(values: Iterable[float]) -> list[float]:
    data = tuple(float(value) for value in values)
    order = sorted(range(len(data)), key=data.__getitem__)
    ranks = [0.0] * len(data)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and data[order[stop]] == data[order[start]]:
            stop += 1
        rank = 0.5 * (start + stop - 1) + 1.0
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def _spearman(x_values: Iterable[float], y_values: Iterable[float]) -> float:
    x = tuple(float(value) for value in x_values)
    y = tuple(float(value) for value in y_values)
    return _pearson(_ranks(x), _ranks(y))


def _axial_coverage(distance: float, center: float) -> float:
    lower_support = center - SUPPORT_HALF_WIDTH
    lower_plateau = center - PLATEAU_HALF_WIDTH
    upper_plateau = center + PLATEAU_HALF_WIDTH
    upper_support = center + SUPPORT_HALF_WIDTH
    if distance < lower_support or distance > upper_support:
        return 0.0
    if lower_plateau <= distance <= upper_plateau:
        return 1.0
    if distance < lower_plateau:
        argument = math.pi * (distance - lower_plateau) / TRANSITION_WIDTH
    else:
        argument = math.pi * (distance - upper_plateau) / TRANSITION_WIDTH
    return 0.5 * (1.0 + math.cos(argument))


def _region(coverage: float) -> str:
    if math.isclose(coverage, 0.0, abs_tol=1e-15):
        return "outside_support"
    if math.isclose(coverage, 1.0, abs_tol=1e-15):
        return "plateau"
    return "transition"


def _persistent_onset(rows: list[dict[str, str]], sign: int) -> int:
    ordered = sorted(
        (int(row["time"]), float(row["mean_phase_radius_response"]))
        for row in rows
    )
    for index, (time, _) in enumerate(ordered):
        if all(sign * value > 0.0 for _, value in ordered[index:]):
            return time
    raise RuntimeError("expected sign never becomes persistent")


def build(
    *,
    source_a_path: Path = SOURCE_A_PATH,
    time_series_path: Path = TIME_SERIES_PATH,
    precursor_summary_path: Path = SUMMARY_PATH,
    normalized_path: Path = NORMALIZED_PATH,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    source_a = _read_csv(source_a_path)
    time_series = _read_csv(time_series_path)
    precursor_summary = _read_json(precursor_summary_path)
    normalized = _read_csv(normalized_path)
    if len(source_a) != 8:
        raise RuntimeError(f"expected eight source-A rows, found {len(source_a)}")
    if len(time_series) != 54:
        raise RuntimeError(f"expected 54 time-series rows, found {len(time_series)}")

    probe_rows: list[dict[str, Any]] = []
    for row in source_a:
        center = float(row["center"])
        coverage = _axial_coverage(PROBE_DISTANCE, center)
        region = _region(coverage)
        multiplier = 1.0 - (1.0 - MINIMUM_MULTIPLIER) * coverage
        primary = (
            region == "outside_support" and row["mode_label"] == "natural"
        )
        probe_rows.append(
            {
                "case_id": row["case_id"],
                "center": f"{center:.1f}",
                "probe_distance": f"{PROBE_DISTANCE:.2f}",
                "support_lower": f"{center - SUPPORT_HALF_WIDTH:.1f}",
                "support_upper": f"{center + SUPPORT_HALF_WIDTH:.1f}",
                "probe_region": region,
                "axial_coverage": f"{coverage:.12g}",
                "radial_plateau_multiplier_at_probe": f"{multiplier:.12g}",
                "event_mode": row["mode_label"],
                "phase_radius_response_t1300": row[
                    "phase_radius_response_t1300"
                ],
                "corrected_lifetime_shift": row["corrected_lifetime_shift"],
                "primary_outside_support_common_mode_subset": primary,
            }
        )

    primary_rows = [
        row
        for row in probe_rows
        if row["primary_outside_support_common_mode_subset"] is True
    ]
    overlap_rows = [
        row for row in probe_rows if row["probe_region"] != "outside_support"
    ]
    if tuple(row["case_id"] for row in primary_rows) != PRIMARY_CASES:
        raise RuntimeError("unexpected outside-support/common-mode subset")
    if tuple(row["case_id"] for row in overlap_rows) != OVERLAP_CASES:
        raise RuntimeError("unexpected probe/collar-overlap subset")

    def association(rows: list[dict[str, Any]]) -> dict[str, Any]:
        radius = [float(row["phase_radius_response_t1300"]) for row in rows]
        lifetime = [float(row["corrected_lifetime_shift"]) for row in rows]
        return {
            "n": len(rows),
            "case_order": [row["case_id"] for row in rows],
            "pearson_r": _pearson(radius, lifetime),
            "spearman_rho": _spearman(radius, lifetime),
            "inferential_statistics_reported": False,
        }

    all_association = association(probe_rows)
    primary_association = association(primary_rows)
    frozen_same = precursor_summary["source_a_placement_sweep"][
        "same_failure_mode_sensitivity"
    ]
    if not math.isclose(
        primary_association["pearson_r"],
        float(frozen_same["pearson_r"]),
        abs_tol=1e-14,
    ):
        raise RuntimeError("primary association differs from frozen same-mode value")

    overlap_metric = [
        float(row["phase_radius_response_t1300"]) for row in overlap_rows
    ]
    primary_metric = [
        float(row["phase_radius_response_t1300"]) for row in primary_rows
    ]
    cluster_summary = {
        "overlap_metric_min": min(overlap_metric),
        "overlap_metric_max": max(overlap_metric),
        "overlap_metric_spread": max(overlap_metric) - min(overlap_metric),
        "outside_support_metric_min": min(primary_metric),
        "outside_support_metric_max": max(primary_metric),
        "outside_support_metric_spread": max(primary_metric) - min(primary_metric),
        "between_cluster_gap": min(overlap_metric) - max(primary_metric),
    }

    events = {
        (row["source_label"], row["case_id"]): row
        for row in normalized
        if float(row["m_min"]) == MINIMUM_MULTIPLIER
        and row["case_id"] in REPRESENTATIVE_CASES
    }
    lead_rows: list[dict[str, Any]] = []
    for source in ("A", "B", "C"):
        for case in REPRESENTATIVE_CASES:
            selected = [
                row
                for row in time_series
                if row["source"] == source and row["case"] == case
            ]
            onset = _persistent_onset(selected, EXPECTED_SIGN[case])
            event_lower = int(float(events[(source, case)]["event_lower"]))
            lead_rows.append(
                {
                    "source": source,
                    "evidence_role": selected[0]["evidence_role"],
                    "case_id": case,
                    "expected_sign": EXPECTED_SIGN[case],
                    "first_stored_persistent_signed_time": onset,
                    "event_bracket_lower": event_lower,
                    "lead_to_event_bracket_lower": event_lower - onset,
                    "all_stored_times_from_onset_have_expected_sign": True,
                }
            )

    minimum_lead = min(
        row["lead_to_event_bracket_lower"] for row in lead_rows
    )
    checks = {
        "zero_simulation_steps": True,
        "three_overlap_cases_identified": len(overlap_rows) == 3,
        "five_primary_cases_outside_support": len(primary_rows) == 5,
        "overlap_and_event_mode_are_confounded": all(
            row["event_mode"] == "downstream" for row in overlap_rows
        ),
        "primary_pearson_matches_frozen_same_mode_value": math.isclose(
            primary_association["pearson_r"],
            0.9493485905736291,
            abs_tol=1e-14,
        ),
        "all_representative_series_signed_from_first_stored_time": all(
            row["first_stored_persistent_signed_time"] == 700
            for row in lead_rows
        ),
        "minimum_early_sign_lead_at_least_850": minimum_lead >= 850,
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise RuntimeError(f"precursor context audit failed: {failed}")

    report = {
        "schema_version": 1,
        "analysis_id": "precursor_context_audit_v1",
        "status": "complete",
        "scope": {
            "simulation_steps_performed": 0,
            "probe_distance": PROBE_DISTANCE,
            "support_half_width": SUPPORT_HALF_WIDTH,
            "plateau_half_width": PLATEAU_HALF_WIDTH,
            "minimum_multiplier": MINIMUM_MULTIPLIER,
            "source_a_role": "exploratory_discovery",
            "arms_are_independent_replicates": False,
        },
        "probe_context": {
            "overlap_cases": list(OVERLAP_CASES),
            "primary_outside_support_common_mode_cases": list(PRIMARY_CASES),
            "overlap_case_count": len(overlap_rows),
            "primary_case_count": len(primary_rows),
            "full_eight_point_association": {
                **all_association,
                "interpretation": (
                    "Confounded sensitivity: the three remote-event cases are "
                    "also the three cases whose fixed probe intersects collar "
                    "support."
                ),
            },
            "primary_outside_support_common_mode_association": {
                **primary_association,
                "interpretation": (
                    "Primary post-hoc descriptive association. The probe is "
                    "outside collar support and the first-event mode is common "
                    "for all five cases."
                ),
            },
            "cluster_summary": cluster_summary,
        },
        "early_time_separation": {
            "first_stored_time": 700,
            "representative_source_case_count": len(lead_rows),
            "minimum_lead_to_event_bracket_lower": minimum_lead,
            "interpretation": (
                "Both representative signs are already present at the first "
                "stored diagnostic in all three sources. This supports early "
                "mechanistic separation, not a calibrated lifetime predictor."
            ),
        },
        "acceptance_checks": checks,
        "claim_boundary": [
            "The full eight-point correlation is not used as primary evidence.",
            "The five-point association is post-hoc and descriptive.",
            "The early-time series cover two representative placements.",
            "No p-value, confidence interval, fitted law, or population claim is made.",
        ],
        "provenance": {
            "analysis_sha256": _sha256(SOURCE_PATH),
            "source_a_precursor_sha256": _sha256(source_a_path),
            "time_series_sha256": _sha256(time_series_path),
            "precursor_summary_sha256": _sha256(precursor_summary_path),
            "normalized_response_sha256": _sha256(normalized_path),
        },
    }
    return report, probe_rows, lead_rows


def write(
    output: Path,
    report: dict[str, Any],
    probe_rows: list[dict[str, Any]],
    lead_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    probe_path = output / "probe_context.csv"
    lead_path = output / "early_sign_lead.csv"
    _write_json(summary_path, report)
    _write_csv(probe_path, probe_rows)
    _write_csv(lead_path, lead_rows)
    manifest = {
        "schema_version": 1,
        "analysis": {
            "path": SOURCE_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": _sha256(SOURCE_PATH),
        },
        "outputs": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (summary_path, probe_path, lead_path)
        ],
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-a", type=Path, default=SOURCE_A_PATH)
    parser.add_argument("--time-series", type=Path, default=TIME_SERIES_PATH)
    parser.add_argument(
        "--precursor-summary",
        type=Path,
        default=SUMMARY_PATH,
    )
    parser.add_argument(
        "--normalized-response",
        type=Path,
        default=NORMALIZED_PATH,
    )
    arguments = parser.parse_args(argv)
    report, probe_rows, lead_rows = build(
        source_a_path=arguments.source_a,
        time_series_path=arguments.time_series,
        precursor_summary_path=arguments.precursor_summary,
        normalized_path=arguments.normalized_response,
    )
    write(arguments.output, report, probe_rows, lead_rows)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
