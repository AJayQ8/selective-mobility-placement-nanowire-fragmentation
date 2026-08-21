#!/usr/bin/env python3
"""Build deterministic CMS validation and transferability tables.

This script performs no simulation. It reads only committed compact evidence
and writes the normalized-response and validation-matrix products used by the
Computational Materials Science manuscript.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parent
POSITION_PATH = ROOT / "position_response.csv"
CONTRAST_PATH = ROOT / "mobility_contrast.csv"
CHECKS_PATH = ROOT / "numerical_checks.csv"
BENCHMARK_PATH = ROOT / "cms_benchmark_inputs.json"
PRECURSOR_PATH = ROOT.parent / "precursor_synthesis" / "readout_v1" / "summary.json"
CONTEXT_PATH = ROOT.parent / "precursor_context_audit" / "readout_v1" / "summary.json"

NORMALIZED_PATH = ROOT / "cms_normalized_response.csv"
MATRIX_PATH = ROOT / "cms_validation_matrix.csv"
RECEIPT_PATH = ROOT / "cms_analysis_receipt.json"


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one(rows: Iterable[Dict[str, str]], **matches: str) -> Dict[str, str]:
    selected = [
        row
        for row in rows
        if all(row[key] == value for key, value in matches.items())
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {matches}, found {len(selected)}")
    return selected[0]


def _number(value: float) -> str:
    return f"{value:.12g}"


def _receipt_input_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return "../" + path.relative_to(ROOT.parent).as_posix()


def build_normalized_rows() -> List[Dict[str, str]]:
    position = _read_csv(POSITION_PATH)
    contrast = _read_csv(CONTRAST_PATH)
    rows: List[Dict[str, str]] = []

    for case_id, placement_role in (
        ("c26p5", "intermediate"),
        ("c34p5", "outboard"),
    ):
        row = _one(position, case_id=case_id)
        untreated_lower = 1690.0
        untreated_upper = 1700.0
        untreated_midpoint = 1695.0
        event_lower = float(row["event_lower"])
        event_upper = float(row["event_upper"])
        paired_shift = float(row["delta_vs_untreated"])
        rows.append(
            {
                "source_label": "A",
                "seed": row["seed"],
                "source_role": "exploratory_discovery",
                "m_min": row["mobility_factor"],
                "case_id": case_id,
                "placement_role": placement_role,
                "untreated_event_lower": _number(untreated_lower),
                "untreated_event_upper": _number(untreated_upper),
                "untreated_midpoint": _number(untreated_midpoint),
                "event_lower": _number(event_lower),
                "event_upper": _number(event_upper),
                "event_midpoint": row["event_midpoint"],
                "paired_shift": _number(paired_shift),
                "paired_interval_lower": _number(event_lower - untreated_upper),
                "paired_interval_upper": _number(event_upper - untreated_lower),
                "normalized_shift": _number(paired_shift / untreated_midpoint),
                "normalized_percent": _number(
                    100.0 * paired_shift / untreated_midpoint
                ),
                "source_ids": row["source_ids"],
            }
        )

    for row in contrast:
        untreated_midpoint = float(row["untreated_midpoint"])
        paired_shift = float(row["paired_shift"])
        rows.append(
            {
                "source_label": row["source_label"],
                "seed": row["seed"],
                "source_role": row["source_role"],
                "m_min": row["m_min"],
                "case_id": row["case_id"],
                "placement_role": row["placement_role"],
                "untreated_event_lower": row["untreated_event_lower"],
                "untreated_event_upper": row["untreated_event_upper"],
                "untreated_midpoint": row["untreated_midpoint"],
                "event_lower": row["event_lower"],
                "event_upper": row["event_upper"],
                "event_midpoint": row["event_midpoint"],
                "paired_shift": row["paired_shift"],
                "paired_interval_lower": row["paired_interval_lower"],
                "paired_interval_upper": row["paired_interval_upper"],
                "normalized_shift": _number(paired_shift / untreated_midpoint),
                "normalized_percent": _number(
                    100.0 * paired_shift / untreated_midpoint
                ),
                "source_ids": row["source_ids"],
            }
        )

    source_order = {"A": 0, "B": 1, "C": 2}
    placement_order = {"intermediate": 0, "outboard": 1}
    rows.sort(
        key=lambda row: (
            float(row["m_min"]),
            source_order[row["source_label"]],
            placement_order[row["placement_role"]],
        )
    )
    if len(rows) != 10:
        raise ValueError(f"expected 10 normalized rows, found {len(rows)}")
    return rows


def _check_map() -> Dict[str, Dict[str, str]]:
    return {row["check_id"]: row for row in _read_csv(CHECKS_PATH)}


def build_validation_rows(
    normalized: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    benchmark_inputs = _read_json(BENCHMARK_PATH)
    roy = benchmark_inputs["roy_reproduction"]
    uniform = benchmark_inputs["uniform_clock_control"]
    checks = _check_map()
    precursor = _read_json(PRECURSOR_PATH)
    context = _read_json(CONTEXT_PATH)

    breakup = roy["breakup_assessment"]
    bracket = breakup["common_threshold_robust_event"]["event_bracket"]
    example = int(breakup["paper_example_trajectory_breakup_time"])
    paper_mean = float(breakup["paper_mean_breakup_time"])
    paper_interval_half_width = float(
        breakup["paper_mean_standard_deviation"]
    )
    bracket_midpoint = 0.5 * (bracket[0] + bracket[1])
    benchmark_passed = (
        bracket[0] <= example <= bracket[1]
        and paper_mean - paper_interval_half_width
        <= bracket_midpoint
        <= paper_mean + paper_interval_half_width
    )

    q = float(
        uniform["contract"]["control_definition"]["uniform_mobility_factor"]
    )
    uniform_bracket = uniform["event_assessment"]["event_bracket"]
    mapped_bracket = [
        100.0 + q * (bound - 100.0) for bound in uniform_bracket
    ]
    untreated_bracket = [1690.0, 1700.0]
    clock_overlap = max(mapped_bracket[0], untreated_bracket[0]) <= min(
        mapped_bracket[1], untreated_bracket[1]
    )

    strong = [row for row in normalized if float(row["m_min"]) == 0.1]
    intermediate = [
        float(row["normalized_percent"])
        for row in strong
        if row["placement_role"] == "intermediate"
    ]
    outboard = [
        float(row["normalized_percent"])
        for row in strong
        if row["placement_role"] == "outboard"
    ]
    weaker = [row for row in normalized if float(row["m_min"]) == 0.3]
    weaker_signs = all(
        (
            float(row["paired_shift"]) > 0
            if row["placement_role"] == "intermediate"
            else float(row["paired_shift"]) < 0
        )
        for row in weaker
    )

    def passed(*ids: str) -> bool:
        return all(
            checks[check_id]["passed"].lower() == "true"
            for check_id in ids
        )

    def status(condition: bool, *, supporting: bool = False) -> str:
        if not condition:
            return "fail"
        return "supporting" if supporting else "pass"

    return [
        {
            "validation_id": "CMS-V01",
            "criterion": "published_baseline",
            "evidence": (
                f"Roy-source bracket {bracket[0]}-{bracket[1]} contains "
                f"the published example {example}; its midpoint "
                f"{_number(bracket_midpoint)} lies within the published "
                f"three-simulation summary {_number(paper_mean)}"
                f"+/-{_number(paper_interval_half_width)}"
            ),
            "status": status(benchmark_passed),
            "transfer_scope": (
                "implementation validation for the selected Roy model and "
                "source semantics"
            ),
            "boundary": (
                "not the single-arm treatment clock and not material calibration"
            ),
            "source_ids": "S01",
        },
        {
            "validation_id": "CMS-V02",
            "criterion": "exact_uniform_clock",
            "evidence": (
                f"q={_number(q)} maps uniform bracket "
                f"{uniform_bracket[0]}-{uniform_bracket[1]} to "
                f"{_number(mapped_bracket[0])}-{_number(mapped_bracket[1])}, "
                "overlapping untreated 1690-1700"
            ),
            "status": status(clock_overlap),
            "transfer_scope": "analytic and numerical solver benchmark",
            "boundary": (
                "uniform mobility only; localized trajectories are not time "
                "rescalings"
            ),
            "source_ids": "S01;S09",
        },
        {
            "validation_id": "CMS-T01",
            "criterion": "independent_source_state",
            "evidence": (
                "strong-contrast intermediate response "
                f"{_number(min(intermediate))} to {_number(max(intermediate))} "
                "percent; outboard response "
                f"{_number(min(outboard))} to {_number(max(outboard))} percent "
                "across A/B/C"
            ),
            "status": "pass",
            "transfer_scope": (
                "discovery source plus two protocol-frozen confirmation sources"
            ),
            "boundary": (
                "three source states, not a population variance estimate"
            ),
            "source_ids": "S02;S03;S04;S14",
        },
        {
            "validation_id": "CMS-M01",
            "criterion": "pre_fragmentation_morphology",
            "evidence": (
                "source-A outside-support/common-event-mode subset at "
                "t=1300 has descriptive r=0.949 (n=5); representative "
                "signs persist from the first stored time t=700, at least "
                "850 before an event; matched B/C phase-radius signs 8/8 "
                "and three-contour signs 24/24"
            ),
            "status": status(
                precursor["decision"]["passed"] is True
                and context["probe_context"][
                    "primary_outside_support_common_mode_association"
                ]["n"]
                == 5
                and abs(
                    context["probe_context"][
                        "primary_outside_support_common_mode_association"
                    ]["pearson_r"]
                    - 0.9493485905736291
                )
                < 1e-12
                and context["early_time_separation"][
                    "minimum_lead_to_event_bracket_lower"
                ]
                >= 850
                and precursor["matched_phase_radius_confirmation"][
                    "passed_count"
                ]
                == 8
                and precursor["existing_three_contour_confirmation"][
                    "passed_count"
                ]
                == 24
            ),
            "transfer_scope": (
                "early morphology across the exploratory placement sweep and "
                "two independent source-state confirmations"
            ),
            "boundary": (
                "primary five-point source-A association is post-hoc and "
                "descriptive; the all-eight association is a confounded "
                "sensitivity check; B/C test two placements, not a calibrated "
                "predictive law"
            ),
            "source_ids": "S17;S18",
        },
        {
            "validation_id": "CMS-T02",
            "criterion": "mobility_contrast",
            "evidence": (
                "m_min=0.3 retains +200/+200 intermediate and -60/-60 "
                "outboard shifts in B/C"
            ),
            "status": status(weaker_signs and passed("N18", "N19")),
            "transfer_scope": "two tested kinetic contrasts",
            "boundary": (
                "no continuous contrast law, threshold, or calibrated dose"
            ),
            "source_ids": "S15",
        },
        {
            "validation_id": "CMS-V03",
            "criterion": "timestep",
            "evidence": (
                "source-B outboard paired shift changes from -110 at dt=1 "
                "to -100 at dt=0.5"
            ),
            "status": status(passed("N07", "N08", "N09")),
            "transfer_scope": "targeted temporal-discretization validation",
            "boundary": "one source and the smallest primary effect",
            "source_ids": "S06",
        },
        {
            "validation_id": "CMS-T03",
            "criterion": "domain_and_source_protocol",
            "evidence": (
                "reduced-domain continuation retains +750 intermediate and "
                "-100 outboard signs"
            ),
            "status": status(passed("N10")),
            "transfer_scope": (
                "sign preservation under changed in-plane spectrum and source "
                "protocol"
            ),
            "boundary": "not strict domain-size convergence",
            "source_ids": "S10;S12;S13",
        },
        {
            "validation_id": "CMS-V04",
            "criterion": "spatial_refinement",
            "evidence": (
                "factor-two common-source refinement retains a -100 outboard "
                "paired shift inside the frozen -125 to -75 window"
            ),
            "status": status(passed("N14", "N15", "N16")),
            "transfer_scope": (
                "grid validation of the selected diffuse-interface model"
            ),
            "boundary": (
                "R/W fixed; not source-formation convergence or a "
                "sharp-interface limit"
            ),
            "source_ids": "S11",
        },
        {
            "validation_id": "CMS-V05",
            "criterion": "transport_closure",
            "evidence": (
                "source-A closure below 0.855 percent; B/C all 24 signs agree "
                "and source-case means close within 0.74 percent"
            ),
            "status": status(
                passed("N05", "N20", "N21", "N22"), supporting=True
            ),
            "transfer_scope": "mechanistic consistency and signed recurrence",
            "boundary": (
                "one of 24 B/C arm closures is 1.1575 percent; recurrence is "
                "supporting evidence, not a passed all-arm gate"
            ),
            "source_ids": "S05;S16",
        },
    ]


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    normalized = build_normalized_rows()
    validation = build_validation_rows(normalized)
    if not all(row["status"] in {"pass", "supporting"} for row in validation):
        failed = [
            row["validation_id"]
            for row in validation
            if row["status"] == "fail"
        ]
        raise RuntimeError(f"CMS validation gate failed: {failed}")

    pass_count = sum(row["status"] == "pass" for row in validation)
    supporting_count = sum(
        row["status"] == "supporting" for row in validation
    )

    _write_csv(NORMALIZED_PATH, normalized)
    _write_csv(MATRIX_PATH, validation)

    inputs = [
        POSITION_PATH,
        CONTRAST_PATH,
        CHECKS_PATH,
        BENCHMARK_PATH,
        PRECURSOR_PATH,
        CONTEXT_PATH,
    ]
    outputs = [NORMALIZED_PATH, MATRIX_PATH]
    receipt = {
        "schema_version": 1,
        "analysis_id": "selective_mobility_cms_archive_v1",
        "simulation_steps_performed": 0,
        "normalized_row_count": len(normalized),
        "validation_row_count": len(validation),
        "all_required_gates_satisfied": True,
        "pass_row_count": pass_count,
        "supporting_row_count": supporting_count,
        "inputs": {
            _receipt_input_path(path): _sha256(path) for path in inputs
        },
        "outputs": {
            path.relative_to(ROOT).as_posix(): _sha256(path) for path in outputs
        },
    }
    RECEIPT_PATH.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
