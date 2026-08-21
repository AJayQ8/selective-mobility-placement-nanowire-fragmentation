#!/usr/bin/env python3
"""Build the zero-step pre-fragmentation synthesis for the CMS paper.

Archive extraction is optional and read-only.  Once the compact matched-profile
table has been committed, the paper-facing synthesis regenerates without the
large sibling archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SOURCE_PATH = Path(__file__).resolve()
DIRECTORY = SOURCE_PATH.parent
CONTRACT_PATH = DIRECTORY / "SCIENTIFIC_CONTRACT.md"
PAPER_ROOT = DIRECTORY.parent
REPOSITORY_ROOT = PAPER_ROOT.parents[2]
SOURCE_DATA = PAPER_ROOT / "source_data"
RAW_DIRECTORY = SOURCE_DATA / "raw"
IMPORTED_SCAN_SUMMARY = (
    RAW_DIRECTORY / "position_scan_mechanism_profile_audit_v1.json"
)
IMPORTED_SCAN_METRICS = (
    RAW_DIRECTORY / "position_scan_mechanism_profile_metrics_v1.csv"
)
POSITION_RESPONSE = SOURCE_DATA / "position_response.csv"
MATCHED_RAW = RAW_DIRECTORY / "precursor_phase_radius_time_series_v1.csv"
MATCHED_RAW_MANIFEST = (
    RAW_DIRECTORY / "precursor_phase_radius_time_series_v1_manifest.json"
)
CONTOUR_SUMMARY = PAPER_ROOT / "mechanism_audit/readout_v1/summary.json"
DEFAULT_OUTPUT = DIRECTORY / "readout_v1"

EXPECTED_SCAN_SUMMARY_SHA256 = (
    "a0fd8a4b7f36392d255b6df5bc14aecdb538d15575277fb908c2a39a75ac2b6e"
)
EXPECTED_SCAN_METRICS_SHA256 = (
    "5ad0bb6f7c45a9f824fa47a758774de221e5dd5711fa01694137fb8b65220c96"
)

SITE = 19.25
TIMES = tuple(range(700, 1501, 100))
PRIMARY_TIMES = (1300, 1500)
WIRES = ("first_wire_z", "second_wire_y")
BRANCHES = (-1, 1)
CASES = ("c26p5", "c34p5")
EXPECTED_SIGN = {"c26p5": 1, "c34p5": -1}
SOURCE_DEFINITIONS = (
    ("A", 2292, "exploratory_discovery"),
    ("B", 104729, "independent_confirmation"),
    ("C", 130363, "independent_confirmation"),
)
SCAN_CASES = (
    "c14p5",
    "c18p5",
    "c22p5",
    "c26p5",
    "c30p5",
    "c34p5",
    "c38p5",
    "c64p5",
)
SAME_MODE_CASES = (
    "c26p5",
    "c30p5",
    "c34p5",
    "c38p5",
    "c64p5",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    fieldnames = list(rows[0])
    from io import StringIO

    handle = StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, handle.getvalue())


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _canonical_archive_uri(path: Path, archive_root: Path) -> str:
    relative = path.resolve().relative_to(archive_root.resolve())
    return f"archive://aj-physics-nanowire-gb-junction/{relative.as_posix()}"


def _profile_paths(
    archive_root: Path,
    source: str,
    seed: int,
    case: str,
    time: int,
) -> tuple[Path, str, Path, str]:
    results = (
        archive_root
        / "science_lab/papers/nanowire_gb_junction/"
        "roy_selective_mobility_protection/results"
    )
    filename = f"profiles-step-{time:04d}.npz"
    if source == "A":
        treated = results / f"position_scan_{case}_v1" / filename
        untreated = results / "inward_full_lifetime_t2000_v1" / filename
        return treated, "", untreated, "untreated_"
    root = results / "paired_repeat_panel_v1" / f"seed-{seed}"
    treated = root / f"case-{case}" / filename
    untreated = root / "case-untreated" / filename
    return treated, "", untreated, ""


def _phase_radius_values(path: Path, prefix: str) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as arrays:
        for wire in WIRES:
            coordinate = np.asarray(
                arrays[f"{wire}_coordinate"], dtype=np.float64
            )
            radius = np.asarray(
                arrays[f"{wire}_{prefix}phase_radius"], dtype=np.float64
            )
            if coordinate.shape != radius.shape:
                raise ValueError(f"coordinate/radius shape mismatch in {path}")
            if not np.all(np.diff(coordinate) > 0.0):
                raise ValueError(f"coordinate is not strictly increasing in {path}")
            if not np.isfinite(coordinate).all() or not np.isfinite(radius).all():
                raise ValueError(f"nonfinite phase-radius data in {path}")
            values[f"{wire}_coordinate"] = coordinate
            values[f"{wire}_radius"] = radius
    return values


def _matched_response(
    treated_path: Path,
    treated_prefix: str,
    untreated_path: Path,
    untreated_prefix: str,
) -> tuple[float, list[float]]:
    treated = _phase_radius_values(treated_path, treated_prefix)
    untreated = _phase_radius_values(untreated_path, untreated_prefix)
    differences: list[float] = []
    for wire in WIRES:
        treated_coordinate = treated[f"{wire}_coordinate"]
        untreated_coordinate = untreated[f"{wire}_coordinate"]
        if not np.array_equal(treated_coordinate, untreated_coordinate):
            raise ValueError("paired phase-radius coordinate grids differ")
        for branch in BRANCHES:
            coordinate = branch * SITE
            treated_value = np.interp(
                coordinate, treated_coordinate, treated[f"{wire}_radius"]
            )
            untreated_value = np.interp(
                coordinate, untreated_coordinate, untreated[f"{wire}_radius"]
            )
            differences.append(float(treated_value - untreated_value))
    return float(np.mean(differences)), differences


def extract_archive(archive_root: Path) -> dict[str, Any]:
    """Extract the frozen matched observable without evolving any field."""

    archive_root = archive_root.resolve()
    records: list[dict[str, Any]] = []
    inputs: dict[Path, dict[str, Any]] = {}
    for source, seed, role in SOURCE_DEFINITIONS:
        for case in CASES:
            for time in TIMES:
                treated, treated_prefix, untreated, untreated_prefix = (
                    _profile_paths(archive_root, source, seed, case, time)
                )
                for path in (treated, untreated):
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    inputs[path] = {
                        "uri": _canonical_archive_uri(path, archive_root),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_path(path),
                    }
                mean_response, arm_differences = _matched_response(
                    treated,
                    treated_prefix,
                    untreated,
                    untreated_prefix,
                )
                records.append(
                    {
                        "source": source,
                        "seed": seed,
                        "evidence_role": role,
                        "case": case,
                        "time": time,
                        "natural_site": SITE,
                        "mean_phase_radius_response": f"{mean_response:.17g}",
                        "minimum_arm_response": f"{min(arm_differences):.17g}",
                        "maximum_arm_response": f"{max(arm_differences):.17g}",
                        "arm_count": len(arm_differences),
                        "all_arms_expected_sign": all(
                            EXPECTED_SIGN[case] * value > 0.0
                            for value in arm_differences
                        ),
                        "treated_profile_uri": inputs[treated]["uri"],
                        "treated_profile_sha256": inputs[treated]["sha256"],
                        "untreated_profile_uri": inputs[untreated]["uri"],
                        "untreated_profile_sha256": inputs[untreated]["sha256"],
                    }
                )
    _write_csv(MATCHED_RAW, records)
    manifest = {
        "schema_version": 1,
        "extractor": {
            "path": SOURCE_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": sha256_path(SOURCE_PATH),
        },
        "contract": {
            "path": CONTRACT_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": sha256_path(CONTRACT_PATH),
        },
        "scope": {
            "archive_read_only": True,
            "simulation_steps_performed": 0,
            "source_count": 3,
            "treated_case_count": 2,
            "time_count": len(TIMES),
            "row_count": len(records),
            "arm_values_are_correlated_diagnostics": True,
        },
        "inputs": [inputs[path] for path in sorted(inputs)],
        "output": {
            "path": MATCHED_RAW.relative_to(REPOSITORY_ROOT).as_posix(),
            "bytes": MATCHED_RAW.stat().st_size,
            "sha256": sha256_path(MATCHED_RAW),
        },
    }
    _write_json(MATCHED_RAW_MANIFEST, manifest)
    return manifest


def _pearson(x: Iterable[float], y: Iterable[float]) -> float:
    x_array = np.asarray(tuple(x), dtype=np.float64)
    y_array = np.asarray(tuple(y), dtype=np.float64)
    if x_array.shape != y_array.shape or x_array.size < 2:
        raise ValueError("correlation inputs must have matching nontrivial shapes")
    if np.ptp(x_array) == 0.0 or np.ptp(y_array) == 0.0:
        raise ValueError("correlation input is constant")
    return float(np.corrcoef(x_array, y_array)[0, 1])


def _rankdata(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _spearman(x: Iterable[float], y: Iterable[float]) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _one(rows: Iterable[dict[str, str]], **matches: str) -> dict[str, str]:
    selected = [
        row
        for row in rows
        if all(row[key] == value for key, value in matches.items())
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {matches}, found {len(selected)}")
    return selected[0]


def _association(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric = [float(row["phase_radius_response_t1300"]) for row in rows]
    lifetime = [float(row["corrected_lifetime_shift"]) for row in rows]
    return {
        "case_order": [row["case_id"] for row in rows],
        "metric_values": metric,
        "corrected_lifetime_shifts": lifetime,
        "pearson_r": _pearson(metric, lifetime),
        "spearman_rho": _spearman(metric, lifetime),
        "inferential_p_value_reported": False,
        "interpretation": (
            "Post-hoc within-one-source descriptive association, not an "
            "independently calibrated predictor or population law."
        ),
    }


def _persistent_onset(rows: list[dict[str, str]], expected_sign: int) -> int | None:
    ordered = sorted(
        (
            int(row["time"]),
            float(row["mean_phase_radius_response"]),
        )
        for row in rows
    )
    for index, (time, _) in enumerate(ordered):
        if all(expected_sign * value > 0.0 for _, value in ordered[index:]):
            return time
    return None


def build_report() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if sha256_path(IMPORTED_SCAN_SUMMARY) != EXPECTED_SCAN_SUMMARY_SHA256:
        raise RuntimeError("imported source-A mechanism summary hash changed")
    if sha256_path(IMPORTED_SCAN_METRICS) != EXPECTED_SCAN_METRICS_SHA256:
        raise RuntimeError("imported source-A mechanism metrics hash changed")

    raw_manifest = _read_json(MATCHED_RAW_MANIFEST)
    if raw_manifest["scope"]["simulation_steps_performed"] != 0:
        raise RuntimeError("matched-profile extraction is not zero-step")
    if raw_manifest["output"]["sha256"] != sha256_path(MATCHED_RAW):
        raise RuntimeError("matched-profile raw table hash differs from manifest")

    scan_summary = _read_json(IMPORTED_SCAN_SUMMARY)
    scan_metrics = _read_csv(IMPORTED_SCAN_METRICS)
    position = _read_csv(POSITION_RESPONSE)
    contour = _read_json(CONTOUR_SUMMARY)
    matched = _read_csv(MATCHED_RAW)

    if len(matched) != 54:
        raise RuntimeError(f"expected 54 matched-profile rows, found {len(matched)}")
    if len(scan_metrics) != 40:
        raise RuntimeError(f"expected 40 scan metric rows, found {len(scan_metrics)}")

    t1300 = {
        row["case_id"]: row
        for row in scan_metrics
        if row["step"] == "1300"
    }
    if tuple(t1300) != SCAN_CASES:
        raise RuntimeError(f"unexpected source-A case order: {tuple(t1300)}")

    source_a_rows: list[dict[str, Any]] = []
    for case in SCAN_CASES:
        metric = t1300[case]
        event = _one(position, case_id=case)
        if metric["mode_label"] != event["mode_label"]:
            raise RuntimeError(f"failure-mode mismatch for {case}")
        source_a_rows.append(
            {
                "case_id": case,
                "center": float(event["center"]),
                "mode_label": event["mode_label"],
                "phase_radius_response_t1300": float(
                    metric["natural_site_radius_response"]
                ),
                "corrected_lifetime_shift": float(event["delta_vs_untreated"]),
                "event_midpoint": float(event["event_midpoint"]),
                "event_lower": float(event["event_lower"]),
                "event_upper": float(event["event_upper"]),
                "weld_connected_all_three_thresholds_t1300": bool(
                    scan_summary["weld_connectivity_t1300"][case][
                        "all_three_thresholds_connected"
                    ]
                ),
                "source_ids": event["source_ids"],
            }
        )

    full_association = _association(source_a_rows)
    same_mode_rows = [
        row for row in source_a_rows if row["case_id"] in SAME_MODE_CASES
    ]
    same_mode_association = _association(same_mode_rows)
    far = _one(source_a_rows, case_id="c64p5")
    all_connected = all(
        row["weld_connected_all_three_thresholds_t1300"]
        for row in source_a_rows
    )

    matched_checks: list[dict[str, Any]] = []
    onset_rows: list[dict[str, Any]] = []
    for source, seed, role in SOURCE_DEFINITIONS:
        source_rows = [row for row in matched if row["source"] == source]
        for case in CASES:
            case_rows = [row for row in source_rows if row["case"] == case]
            onset = _persistent_onset(case_rows, EXPECTED_SIGN[case])
            onset_rows.append(
                {
                    "source": source,
                    "seed": seed,
                    "evidence_role": role,
                    "case": case,
                    "expected_sign": EXPECTED_SIGN[case],
                    "persistent_expected_sign_onset": onset,
                    "onset_no_later_than_t1300": (
                        onset is not None and onset <= 1300
                    ),
                }
            )
            if source in ("B", "C"):
                for time in PRIMARY_TIMES:
                    row = _one(case_rows, time=str(time))
                    value = float(row["mean_phase_radius_response"])
                    matched_checks.append(
                        {
                            "source": source,
                            "seed": seed,
                            "case": case,
                            "time": time,
                            "mean_phase_radius_response": value,
                            "expected_sign": EXPECTED_SIGN[case],
                            "passed": EXPECTED_SIGN[case] * value > 0.0,
                        }
                    )

    held_out_onsets = [
        row for row in onset_rows if row["source"] in ("B", "C")
    ]
    contour_gate = contour["primary_held_out_gate"]
    contour_pass = (
        contour_gate["passed"] is True
        and contour_gate["required_check_count"] == 24
        and contour_gate["passed_count"] == 24
        and contour["pre_event_gate"]["passed"] is True
    )

    checks = {
        "all_source_a_welds_connected_t1300": all_connected,
        "full_sweep_pearson_at_least_0p90": (
            full_association["pearson_r"] >= 0.90
        ),
        "same_mode_pearson_at_least_0p90": (
            same_mode_association["pearson_r"] >= 0.90
        ),
        "far_phase_radius_response_below_0p01": (
            abs(float(far["phase_radius_response_t1300"])) < 0.01
        ),
        "far_lifetime_shift_at_most_one_interval": (
            abs(float(far["corrected_lifetime_shift"])) <= 10.0
        ),
        "all_eight_matched_bc_checks_pass": (
            len(matched_checks) == 8
            and all(row["passed"] for row in matched_checks)
        ),
        "all_bc_signs_persistent_by_t1300": all(
            row["onset_no_later_than_t1300"] for row in held_out_onsets
        ),
        "existing_three_contour_gate_remains_24_of_24": contour_pass,
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "status": "complete",
        "classification": (
            "exploratory_full_scan_association_with_matched_held_out_"
            "precursor_passed"
            if passed
            else "precursor_synthesis_failed"
        ),
        "scope": {
            "simulation_steps_performed": 0,
            "source_a_role": "exploratory_discovery",
            "independent_confirmation_sources": ["B", "C"],
            "independent_source_count": 3,
            "arms_are_independent_replicates": False,
            "primary_time": 1300,
            "natural_event_site": SITE,
        },
        "source_a_placement_sweep": {
            "full_sweep": full_association,
            "same_failure_mode_sensitivity": same_mode_association,
            "far_control": far,
            "all_welds_connected_t1300": all_connected,
        },
        "matched_phase_radius_confirmation": {
            "primary_checks": matched_checks,
            "passed_count": sum(row["passed"] for row in matched_checks),
            "required_count": 8,
            "persistent_onsets": onset_rows,
        },
        "existing_three_contour_confirmation": {
            "passed": contour_pass,
            "passed_count": contour_gate["passed_count"],
            "required_count": contour_gate["required_check_count"],
            "pre_event_gate_passed": contour["pre_event_gate"]["passed"],
        },
        "acceptance_checks": checks,
        "decision": {
            "passed": passed,
            "interpretation": (
                "An early source-A morphology response tracks the corrected "
                "placement-lifetime curve, including within the common "
                "junction-adjacent failure mode, and the same signed metric "
                "recurs in B/C. This is not a universal predictive law."
            ),
        },
        "provenance": {
            "contract_sha256": sha256_path(CONTRACT_PATH),
            "analysis_sha256": sha256_path(SOURCE_PATH),
            "imported_scan_summary_sha256": sha256_path(IMPORTED_SCAN_SUMMARY),
            "imported_scan_metrics_sha256": sha256_path(IMPORTED_SCAN_METRICS),
            "position_response_sha256": sha256_path(POSITION_RESPONSE),
            "matched_raw_sha256": sha256_path(MATCHED_RAW),
            "matched_raw_manifest_sha256": sha256_path(MATCHED_RAW_MANIFEST),
            "contour_summary_sha256": sha256_path(CONTOUR_SUMMARY),
        },
        "claim_boundary": [
            "The eight-position correlation is post-hoc and within one source.",
            "B/C confirm two frozen placement signs, not a continuous law.",
            "Arms are correlated diagnostics, not independent replicates.",
            "No radius, angle, material, or sharp-interface transfer is shown.",
        ],
    }
    return report, source_a_rows, [
        {
            "source": row["source"],
            "seed": int(row["seed"]),
            "evidence_role": row["evidence_role"],
            "case": row["case"],
            "time": int(row["time"]),
            "mean_phase_radius_response": float(
                row["mean_phase_radius_response"]
            ),
            "minimum_arm_response": float(row["minimum_arm_response"]),
            "maximum_arm_response": float(row["maximum_arm_response"]),
            "all_arms_expected_sign": row["all_arms_expected_sign"] == "True",
        }
        for row in matched
    ]


def write_readout(
    output: Path,
    report: dict[str, Any],
    source_a_rows: list[dict[str, Any]],
    matched_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    source_a_path = output / "source_a_placement_precursor.csv"
    matched_path = output / "matched_phase_radius_time_series.csv"
    _write_json(summary_path, report)
    _write_csv(source_a_path, source_a_rows)
    _write_csv(matched_path, matched_rows)
    manifest = {
        "schema_version": 1,
        "analysis": {
            "path": SOURCE_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": sha256_path(SOURCE_PATH),
        },
        "contract": {
            "path": CONTRACT_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": sha256_path(CONTRACT_PATH),
        },
        "outputs": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
            for path in (summary_path, source_a_path, matched_path)
        ],
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root",
        type=Path,
        help="read-only sibling archive root; refreshes the compact extraction",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    if arguments.archive_root is not None:
        extract_archive(arguments.archive_root)
    if not MATCHED_RAW.is_file() or not MATCHED_RAW_MANIFEST.is_file():
        raise FileNotFoundError(
            "compact matched-profile extraction is absent; provide --archive-root"
        )
    report, source_a_rows, matched_rows = build_report()
    write_readout(arguments.output, report, source_a_rows, matched_rows)
    print(json.dumps(report, indent=2))
    if not report["decision"]["passed"]:
        raise SystemExit("pre-fragmentation synthesis failed")
    return report


if __name__ == "__main__":
    main()
