"""Generate Figure 3: audited pre-fragmentation morphology evidence.

Panel A makes the outside-collar/common-event-mode subset primary and visually
subordinates the three probe/collar-overlap cases. Panel B keeps exploratory
source A separate from confirmation sources B and C. Panel C shows the eight
predeclared B/C sign checks individually.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from . import figure_common as common
except ImportError:  # Direct execution from this directory.
    import figure_common as common

FIGURE_DIR = Path(__file__).resolve().parent
PAPER_ROOT = FIGURE_DIR.parent
READOUT_ROOT = PAPER_ROOT / "precursor_synthesis" / "readout_v1"
SWEEP_SOURCE = READOUT_ROOT / "source_a_placement_precursor.csv"
TIME_SOURCE = READOUT_ROOT / "matched_phase_radius_time_series.csv"
SUMMARY_SOURCE = READOUT_ROOT / "summary.json"
CONTEXT_ROOT = PAPER_ROOT / "precursor_context_audit" / "readout_v1"
CONTEXT_SUMMARY = CONTEXT_ROOT / "summary.json"
PROBE_CONTEXT = CONTEXT_ROOT / "probe_context.csv"
DEFAULT_OUTPUT_DIR = FIGURE_DIR / "artwork" / "main"
REVIEW_WIDTH_MM = 160.0
REVIEW_HEIGHT_MM = 170.0
JOURNAL_WIDTH_MM = 190.0
COLORS = {
    **common.COLORS,
    "downstream": "#009E73",
    "natural": "#50545B",
    "far": "#7A4EAB",
}


def _event_style(row: dict[str, str]) -> tuple[str, str, str]:
    if row["case_id"] == "c64p5":
        return "D", COLORS["far"], "white"
    if row["mode_label"] == "downstream":
        return "s", COLORS["downstream"], COLORS["downstream"]
    return "o", COLORS["natural"], COLORS["natural"]


def load_and_validate() -> dict[str, Any]:
    sweep = common.read_csv(SWEEP_SOURCE)
    time_series = common.read_csv(TIME_SOURCE)
    summary = common.read_json(SUMMARY_SOURCE)
    if len(sweep) != 8:
        raise RuntimeError(f"expected eight source-A sweep rows, found {len(sweep)}")
    if len(time_series) != 54:
        raise RuntimeError(f"expected 54 time-series rows, found {len(time_series)}")
    expected_shifts = {
        "c14p5": 850.0,
        "c18p5": 750.0,
        "c22p5": 680.0,
        "c26p5": 290.0,
        "c30p5": -10.0,
        "c34p5": -110.0,
        "c38p5": -100.0,
        "c64p5": 0.0,
    }
    actual_shifts = {
        row["case_id"]: float(row["corrected_lifetime_shift"])
        for row in sweep
    }
    if actual_shifts != expected_shifts:
        raise RuntimeError(f"corrected lifetime shifts changed: {actual_shifts}")
    if not all(
        row["weld_connected_all_three_thresholds_t1300"] == "True"
        for row in sweep
    ):
        raise RuntimeError("one or more source-A welds are disconnected at t=1300")
    full = summary["source_a_placement_sweep"]["full_sweep"]
    same = summary["source_a_placement_sweep"]["same_failure_mode_sensitivity"]
    if abs(float(full["pearson_r"]) - 0.9802650393895559) > 1e-12:
        raise RuntimeError("full-sweep sensitivity association changed")
    if abs(float(same["pearson_r"]) - 0.9493485905736291) > 1e-12:
        raise RuntimeError("outside-support association changed")
    confirmation = summary["matched_phase_radius_confirmation"]
    if confirmation["passed_count"] != 8 or confirmation["required_count"] != 8:
        raise RuntimeError("matched B/C confirmation is not 8/8")
    if summary["existing_three_contour_confirmation"]["passed_count"] != 24:
        raise RuntimeError("three-contour confirmation is not 24/24")

    context = common.read_json(CONTEXT_SUMMARY)
    probe_rows = common.read_csv(PROBE_CONTEXT)
    primary = context["probe_context"][
        "primary_outside_support_common_mode_association"
    ]
    if primary["n"] != 5:
        raise RuntimeError("expected five cases in the primary subset")
    if abs(primary["pearson_r"] - 0.9493485905736291) > 1e-12:
        raise RuntimeError("primary outside-support association changed")
    overlap = {
        row["case_id"]
        for row in probe_rows
        if row["probe_region"] != "outside_support"
    }
    if overlap != {"c14p5", "c18p5", "c22p5"}:
        raise RuntimeError(f"probe/collar overlap cases changed: {overlap}")
    return {
        "sweep": sweep,
        "time_series": time_series,
        "summary": summary,
        "context": context,
        "probe_context": probe_rows,
    }


def _draw_panel_a(axis: plt.Axes, data: dict[str, Any]) -> None:
    overlap_cases = set(data["context"]["probe_context"]["overlap_cases"])
    label_offsets = {
        "c14p5": (-7, 8),
        "c18p5": (8, -6),
        "c22p5": (-7, 8),
        "c26p5": (7, 8),
        "c30p5": (6, 10),
        "c34p5": (-8, 2),
        "c38p5": (-6, 12),
        "c64p5": (6, 10),
    }
    alignment = {
        "c14p5": ("right", "bottom"),
        "c18p5": ("left", "top"),
        "c22p5": ("right", "bottom"),
        "c26p5": ("left", "bottom"),
        "c30p5": ("left", "bottom"),
        "c34p5": ("right", "bottom"),
        "c38p5": ("right", "bottom"),
        "c64p5": ("left", "bottom"),
    }
    leader_style = {
        "arrowstyle": "-",
        "color": COLORS["muted"],
        "linewidth": 0.55,
        "shrinkA": 2.0,
        "shrinkB": 4.0,
    }
    label_box = {
        "facecolor": "white",
        "edgecolor": "none",
        "boxstyle": "square,pad=0.12",
        "alpha": 0.95,
    }
    for row in data["sweep"]:
        case = row["case_id"]
        x = float(row["phase_radius_response_t1300"])
        y = float(row["corrected_lifetime_shift"])
        marker, color, facecolor = _event_style(row)
        overlaps_collar = case in overlap_cases
        if overlaps_collar:
            color = COLORS["muted"]
            facecolor = "white"
        axis.errorbar(
            x,
            y,
            yerr=10.0,
            fmt=marker,
            markersize=6.8,
            color=color,
            markerfacecolor=facecolor,
            markeredgecolor=color,
            markeredgewidth=1.0,
            elinewidth=0.8,
            capsize=2.4,
            capthick=0.8,
            alpha=0.52 if overlaps_collar else 1.0,
            zorder=4,
        )
        horizontal, vertical = alignment[case]
        axis.annotate(
            rf"{float(row['center']):.1f}",
            xy=(x, y),
            xytext=label_offsets[case],
            textcoords="offset points",
            fontsize=8.1,
            ha=horizontal,
            va=vertical,
            color=COLORS["muted"] if overlaps_collar else COLORS["ink"],
            zorder=5,
            arrowprops=leader_style,
            bbox=label_box,
        )

    axis.axhline(0.0, color=COLORS["ink"], linewidth=0.85, zorder=1)
    axis.axvline(0.0, color=COLORS["muted"], linewidth=0.75, zorder=1)
    axis.set_xlim(-0.31, 0.91)
    axis.set_ylim(-185.0, 940.0)
    axis.set_xticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
    axis.set_yticks([0, 200, 400, 600, 800])
    axis.set_xlabel(
        r"phase-radius response, $\Delta r_\phi$ "
        r"($t=1300$; $|s|=19.25$)"
    )
    axis.set_ylabel(r"lifetime shift, $\Delta T$")
    axis.grid(color=COLORS["grid"], linewidth=0.60, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    primary = data["context"]["probe_context"][
        "primary_outside_support_common_mode_association"
    ]
    axis.text(
        0.025,
        0.955,
        "outside support; common event mode\n"
        + rf"Pearson $r={primary['pearson_r']:.2f}$ ($n={primary['n']}$)",
        transform=axis.transAxes,
        fontsize=7.9,
        ha="left",
        va="top",
        color=COLORS["ink"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90},
    )
    event_handles = [
        Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            markersize=6.0,
            markerfacecolor="white",
            markeredgecolor=COLORS["muted"],
            alpha=0.60,
            label="probe intersects collar support",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=6.0,
            markerfacecolor=COLORS["natural"],
            markeredgecolor=COLORS["natural"],
            label="outside support; junction event",
        ),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            markersize=5.6,
            markerfacecolor="white",
            markeredgecolor=COLORS["far"],
            label="outside support; far control",
        ),
    ]
    axis.legend(
        handles=event_handles,
        loc="center right",
        bbox_to_anchor=(0.995, 0.46),
        frameon=False,
        ncol=1,
        handletextpad=0.35,
        labelspacing=0.30,
        borderaxespad=0.35,
    )


def _draw_panel_b(axis: plt.Axes, rows: list[dict[str, str]]) -> None:
    placement_colors = {
        "c26p5": COLORS["intermediate"],
        "c34p5": COLORS["outboard"],
    }
    placement_labels = {
        "c26p5": "intermediate",
        "c34p5": "outboard",
    }
    for case in ("c26p5", "c34p5"):
        times = sorted(
            {int(row["time"]) for row in rows if row["case"] == case}
        )
        source_a = []
        confirmation_mean = []
        confirmation_minimum = []
        confirmation_maximum = []
        for time in times:
            by_source = {
                row["source"]: float(row["mean_phase_radius_response"])
                for row in rows
                if row["case"] == case and int(row["time"]) == time
            }
            if set(by_source) != {"A", "B", "C"}:
                raise RuntimeError(
                    f"expected sources A-C at t={time} for {case}"
                )
            source_a.append(by_source["A"])
            confirmation = [by_source["B"], by_source["C"]]
            confirmation_mean.append(fmean(confirmation))
            confirmation_minimum.append(min(confirmation))
            confirmation_maximum.append(max(confirmation))

        color = placement_colors[case]
        band = axis.fill_between(
            times,
            confirmation_minimum,
            confirmation_maximum,
            color=color,
            alpha=0.13,
            linewidth=0.0,
            zorder=1,
        )
        band.set_gid(f"{case}-bc-range")
        confirmation_line = axis.plot(
            times,
            confirmation_mean,
            color=color,
            linewidth=2.0,
            zorder=4,
        )[0]
        confirmation_line.set_gid(f"{case}-bc-mean")
        source_a_line = axis.plot(
            times,
            source_a,
            color=color,
            linewidth=1.15,
            linestyle=(0, (3.2, 2.4)),
            alpha=0.82,
            zorder=3,
        )[0]
        source_a_line.set_gid(f"{case}-source-a")

    axis.axhline(0.0, color=COLORS["ink"], linewidth=0.85, zorder=0)
    for check_time in (1300, 1500):
        axis.axvline(
            check_time,
            color=COLORS["grid"],
            linewidth=0.8,
            linestyle="--",
            zorder=0,
        )
    axis.set_xlim(675, 1525)
    axis.set_ylim(-0.32, 0.255)
    axis.set_xticks([700, 900, 1100, 1300, 1500])
    axis.set_yticks([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2])
    axis.set_xlabel("model time, $t$")
    axis.set_ylabel(r"mean response, $\Delta r_\phi$")
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.60, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    handles = [
        Line2D(
            [],
            [],
            color=placement_colors[case],
            linewidth=2.0,
            label=placement_labels[case],
        )
        for case in ("c26p5", "c34p5")
    ]
    handles.extend(
        [
            Line2D(
                [],
                [],
                color=COLORS["ink"],
                linewidth=2.0,
                label="B/C confirmation mean",
            ),
            Line2D(
                [],
                [],
                color=COLORS["ink"],
                linewidth=1.15,
                linestyle=(0, (3.2, 2.4)),
                label="source A exploratory",
            ),
        ]
    )
    legend = axis.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.52, 0.49),
        ncol=2,
        frameon=True,
        handlelength=2.1,
        handletextpad=0.42,
        columnspacing=1.0,
        borderpad=0.35,
        fontsize=7.6,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(0.92)


def _panel_heading(
    figure: plt.Figure,
    x: float,
    y: float,
    label: str,
    title: str,
) -> None:
    figure.text(
        x,
        y,
        f"({label})",
        fontsize=11.2,
        fontweight="bold",
        ha="left",
        va="center",
        color=COLORS["ink"],
    )
    figure.text(
        x + 0.056,
        y,
        title,
        fontsize=9.9,
        fontweight="bold",
        ha="left",
        va="center",
        color=COLORS["ink"],
    )


def _draw_panel_c(axis: plt.Axes, rows: list[dict[str, str]]) -> None:
    base_positions = {
        ("c26p5", 1300): 0.0,
        ("c26p5", 1500): 1.0,
        ("c34p5", 1300): 3.0,
        ("c34p5", 1500): 4.0,
    }
    source_offsets = {"B": -0.11, "C": 0.11}
    source_markers = {"B": "s", "C": "^"}
    placement_colors = {
        "c26p5": COLORS["intermediate"],
        "c34p5": COLORS["outboard"],
    }
    for case in ("c26p5", "c34p5"):
        for source in ("B", "C"):
            selected = sorted(
                (
                    int(row["time"]),
                    float(row["mean_phase_radius_response"]),
                )
                for row in rows
                if row["case"] == case
                and row["source"] == source
                and int(row["time"]) in (1300, 1500)
            )
            if len(selected) != 2:
                raise RuntimeError(
                    f"expected two frozen checks for source {source}, {case}"
                )
            axis.plot(
                [
                    base_positions[(case, time)] + source_offsets[source]
                    for time, _ in selected
                ],
                [value for _, value in selected],
                linestyle="none",
                marker=source_markers[source],
                markersize=6.2,
                markerfacecolor="white",
                markeredgecolor=placement_colors[case],
                markeredgewidth=1.05,
                zorder=4,
            )

    axis.axhline(0.0, color=COLORS["ink"], linewidth=0.85, zorder=1)
    axis.axvline(2.0, color=COLORS["grid"], linewidth=0.8, zorder=0)
    axis.set_xlim(-0.55, 4.55)
    axis.set_ylim(-0.325, 0.36)
    axis.set_xticks([0.0, 1.0, 3.0, 4.0])
    axis.set_xticklabels(["1300", "1500", "1300", "1500"])
    axis.set_yticks([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2])
    axis.set_xlabel("predeclared model time, $t$")
    axis.set_ylabel(r"phase-radius response, $\Delta r_\phi$")
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.60, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.text(
        0.5,
        0.325,
        r"intermediate, $s_c=26.5$",
        fontsize=8.2,
        ha="center",
        va="top",
        color=COLORS["intermediate"],
    )
    axis.text(
        3.5,
        0.325,
        r"outboard, $s_c=34.5$",
        fontsize=8.2,
        ha="center",
        va="top",
        color=COLORS["outboard"],
    )
    source_handles = [
        Line2D(
            [],
            [],
            color=COLORS["ink"],
            linestyle="none",
            marker="s",
            markersize=5.7,
            markerfacecolor="white",
            markeredgecolor=COLORS["ink"],
            label="source B",
        ),
        Line2D(
            [],
            [],
            color=COLORS["ink"],
            linestyle="none",
            marker="^",
            markersize=5.7,
            markerfacecolor="white",
            markeredgecolor=COLORS["ink"],
            label="source C",
        ),
    ]
    legend = axis.legend(
        handles=source_handles,
        loc="center",
        bbox_to_anchor=(0.50, 0.50),
        ncol=2,
        frameon=True,
        handletextpad=0.35,
        columnspacing=0.9,
        borderpad=0.25,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(0.92)


def build_figure(data: dict[str, Any], width_mm: float) -> plt.Figure:
    common.style()
    plt.rcParams.update(
        {
            "font.size": 9.6,
            "axes.labelsize": 9.8,
            "xtick.labelsize": 8.9,
            "ytick.labelsize": 8.9,
            "legend.fontsize": 8.0,
        }
    )
    height_mm = REVIEW_HEIGHT_MM * width_mm / REVIEW_WIDTH_MM
    figure = plt.figure(
        figsize=(width_mm / common.MM_PER_INCH, height_mm / common.MM_PER_INCH)
    )
    axis_a = figure.add_axes([0.140, 0.735, 0.830, 0.190])
    axis_b = figure.add_axes([0.215, 0.375, 0.600, 0.220])
    axis_c = figure.add_axes([0.140, 0.065, 0.830, 0.190])
    _panel_heading(
        figure,
        0.020,
        0.968,
        "a",
        "placement-lifetime association (source A)",
    )
    _panel_heading(
        figure,
        0.020,
        0.635,
        "b",
        "time evolution: confirmation B/C with exploratory A",
    )
    _panel_heading(
        figure,
        0.020,
        0.305,
        "c",
        "confirmation at predeclared times (sources B and C)",
    )
    _draw_panel_a(axis_a, data)
    _draw_panel_b(axis_b, data["time_series"])
    _draw_panel_c(axis_c, data["time_series"])
    return figure


def generate(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    data = load_and_validate()
    review = build_figure(data, REVIEW_WIDTH_MM)
    journal = build_figure(data, JOURNAL_WIDTH_MM)
    try:
        review_outputs = common.save_exact(
            review,
            output_dir / "figure3_precursor",
            REVIEW_WIDTH_MM,
        )
        full_width_outputs = common.save_exact(
            journal,
            output_dir / "full_width" / "figure3_precursor",
            JOURNAL_WIDTH_MM,
        )
    finally:
        plt.close(review)
        plt.close(journal)

    summary = data["summary"]
    manifest = {
        "schema_version": 1,
        "figure": 3,
        "generator": common.portable(Path(__file__)),
        "scope": (
            "reader-facing placement association, compact time summary, and "
            "separated confirmation-source values"
        ),
        "sources": [
            {
                "path": common.portable(path),
                "sha256": common.sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in (
                SWEEP_SOURCE,
                TIME_SOURCE,
                SUMMARY_SOURCE,
                CONTEXT_SUMMARY,
                PROBE_CONTEXT,
            )
        ],
        "science": {
            "primary_time": 1300,
            "natural_event_site": 19.25,
            "full_sweep_pearson_r": summary["source_a_placement_sweep"][
                "full_sweep"
            ]["pearson_r"],
            "full_sweep_role": "confounded_sensitivity_not_drawn",
            "primary_outside_support_common_mode_pearson_r": data["context"][
                "probe_context"
            ]["primary_outside_support_common_mode_association"]["pearson_r"],
            "primary_subset_n": 5,
            "matched_bc_checks_passed": 8,
            "matched_bc_checks_required": 8,
            "predictive_law_claimed": False,
        },
        "presentation": {
            "panel_count": 3,
            "panel_a_point_label_definition_drawn": False,
            "panel_a_all_labels_have_leaders": True,
            "panel_a_leader_style": (
                "short angled neutral lines without arrowheads"
            ),
            "panel_a_cluster_leaders_cross": False,
            "panel_b_width_fraction": 0.60,
            "panel_b_confirmation_mean_drawn": True,
            "panel_b_confirmation_range_drawn": True,
            "panel_b_source_a_drawn_separately": True,
            "panel_b_all_source_mean_drawn": False,
            "panel_b_individual_markers_drawn": False,
            "panel_c_individual_points": 8,
            "internal_sign_shorthand_drawn": False,
        },
        "caption_notes_for_manuscript_integration": [
            (
                "Numerals adjacent to symbols indicate the collar-center "
                "position, s_c."
            ),
            (
                "Panel (a) uses the five outside-support/common-event-mode "
                "placements for its primary descriptive association; pale "
                "squares identify probe/collar-overlap cases excluded from it."
            ),
            (
                "Panel (b) solid curves and bands show the B/C confirmation "
                "mean and range; dashed curves show exploratory source A."
            ),
            (
                "All eight confirmation comparisons agreed with the lifetime "
                "ordering."
            ),
        ],
        "geometry": {
            "review_width_mm": REVIEW_WIDTH_MM,
            "review_height_mm": REVIEW_HEIGHT_MM,
            "journal_width_mm": JOURNAL_WIDTH_MM,
            "journal_height_mm": (
                REVIEW_HEIGHT_MM * JOURNAL_WIDTH_MM / REVIEW_WIDTH_MM
            ),
            "minimum_semantic_text_pt": 7.9,
            "panel_count": 3,
        },
        "review_outputs": review_outputs,
        "full_width_outputs": full_width_outputs,
    }
    manifest_path = output_dir / "figure3_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()
    print(json.dumps(generate(arguments.output_dir), indent=2))


if __name__ == "__main__":
    main()
