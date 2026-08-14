"""Generate Figure 2: source-level confirmation of the placement ordering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

try:
    from . import figure_common as common
except ImportError:  # Direct execution from this directory.
    import figure_common as common


DEFAULT_OUTPUT_DIR = common.FIGURE_DIR / "artwork" / "main"
FIGURE_HEIGHT_IN = 5.8
FIGURE_HEIGHT_MM = FIGURE_HEIGHT_IN * common.MM_PER_INCH
CATEGORY_CASES = ["untreated", "c18p5", "c26p5", "c34p5"]
CATEGORY_LABELS = [
    "untreated\nreference",
    "near\n18.5",
    "intermediate\n26.5",
    "outboard\n34.5",
]
OFFSETS = {"2292": -0.22, "104729": 0.0, "130363": 0.22}
PANEL_A_HEIGHT_IN = 2.30
PANEL_B_HEIGHT_IN = 1.313
PANEL_A_BOTTOM_IN = 2.614
PANEL_B_BOTTOM_IN = 0.405
LABEL_OFFSET_IN = 2.5 / common.MM_PER_INCH


def build_figure(paired: list[dict[str, str]]) -> plt.Figure:
    common.style()
    figure, (ax_a, ax_b) = plt.subplots(
        2,
        1,
        figsize=(common.COMPACT_WIDTH_MM / common.MM_PER_INCH, FIGURE_HEIGHT_IN),
        constrained_layout=False,
        gridspec_kw={"height_ratios": (1.04, 1.0)},
    )
    ax_a.set_position(
        [0.18, PANEL_A_BOTTOM_IN / FIGURE_HEIGHT_IN, 0.805, PANEL_A_HEIGHT_IN / FIGURE_HEIGHT_IN]
    )
    ax_b.set_position(
        [0.18, PANEL_B_BOTTOM_IN / FIGURE_HEIGHT_IN, 0.805, PANEL_B_HEIGHT_IN / FIGURE_HEIGHT_IN]
    )
    rows = {(row["seed"], row["case_id"]): row for row in paired}
    xpos = np.arange(len(CATEGORY_CASES), dtype=float)

    for seed, marker in common.SOURCE_MARKERS.items():
        for x, case in zip(xpos[1:], CATEGORY_CASES[1:]):
            row = rows[(seed, case)]
            ax_a.plot(
                x + OFFSETS[seed],
                common.number(row, "delta_vs_paired_untreated"),
                marker=marker,
                ms=common.PRIMARY_MARKER_PT,
                color=common.CASE_COLORS[case],
                mfc=common.CASE_COLORS[case],
                mec="white",
                mew=0.75,
                ls="none",
                zorder=4,
            )
    for x, y, label, case, va in (
        (1.0, 815.0, "+750 to +770", "c18p5", "bottom"),
        (2.0, 340.0, "+260 to +290", "c26p5", "bottom"),
        (3.0, -230.0, "-110 to -100", "c34p5", "center"),
    ):
        text = ax_a.text(
            x,
            y,
            label,
            ha="center",
            va=va,
            fontsize=common.MIN_TEXT_PT,
            color=common.CASE_COLORS[case],
            fontweight="bold",
            zorder=6,
        )
        text.set_gid(f"range-label-{case}")
    ax_a.axhline(0, color=common.COLORS["ink"], lw=0.9)
    ax_a.set_xticks(xpos, CATEGORY_LABELS)
    ax_a.set_xlim(-0.5, 3.5)
    ax_a.set_ylim(-300, 900)
    ax_a.set_yticks([-100, 0, 250, 500, 750])
    ax_a.set_ylabel("paired shift, " + r"$\Delta T$" + "\n(model time units)")
    common.clean_axis(ax_a, "y")
    source_labels = {
        "2292": "A (exploratory)",
        "104729": "B (confirmation)",
        "130363": "C (confirmation)",
    }
    legend = ax_a.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=common.SOURCE_MARKERS[seed],
                ls="none",
                ms=9.0,
                mfc="white",
                mec="#3F4A58",
                mew=1.35,
                label=source_labels[seed],
            )
            for seed in common.SOURCE_MARKERS
        ],
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.15),
        handletextpad=0.45,
        columnspacing=1.25,
        borderpad=0.15,
    )
    legend.set_gid("source-shape-legend")

    band = ax_b.axhspan(14.0, 22.5, color=common.COLORS["local_band"], alpha=0.72, lw=0)
    band.set_gid("junction-adjacent-band")
    band_label = ax_b.text(
        1.0,
        16.5,
        "junction-adjacent",
        ha="center",
        va="center",
        fontsize=common.MIN_TEXT_PT,
        color="#775B3B",
        zorder=3,
    )
    band_label.set_gid("junction-adjacent-label")
    for seed, marker in common.SOURCE_MARKERS.items():
        for x, case in zip(xpos, CATEGORY_CASES):
            row = rows[(seed, case)]
            low = common.number(row, "minimum_abs_site")
            high = common.number(row, "maximum_abs_site")
            ax_b.plot(
                x + OFFSETS[seed],
                0.5 * (low + high),
                marker=marker,
                ms=common.PRIMARY_MARKER_PT,
                color=common.CASE_COLORS[case],
                mfc=common.CASE_COLORS[case],
                mec="white",
                mew=0.75,
                ls="none",
                zorder=4,
            )
    ax_b.set_xticks(xpos, CATEGORY_LABELS)
    ax_b.set_xlim(-0.5, 3.5)
    ax_b.set_ylim(10, 44)
    ax_b.set_ylabel(r"first-site distance, $|s_f|$" + "\n(model units)")
    common.clean_axis(ax_b, "y")

    for axis, label, height in (
        (ax_a, "a", PANEL_A_HEIGHT_IN),
        (ax_b, "b", PANEL_B_HEIGHT_IN),
    ):
        text = axis.text(
            0.0,
            1.0 + LABEL_OFFSET_IN / height,
            f"({label})",
            transform=axis.transAxes,
            fontsize=common.PANEL_TEXT_PT,
            fontweight="bold",
            ha="left",
            va="bottom",
            color=common.COLORS["ink"],
            clip_on=False,
            zorder=30,
        )
        text.set_gid(f"panel-label-{label}")
    return figure


def generate(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    data = common.load_core_sources()
    validation = common.validate_sources(data)
    if not validation["passed"]:
        raise RuntimeError(f"source validation failed: {validation['failures']}")
    compact = build_figure(data["paired"])
    compact_outputs = common.save_exact(
        compact, output_dir / "figure2_confirmation", common.COMPACT_WIDTH_MM
    )
    full = build_figure(data["paired"])
    scale = common.FULL_WIDTH_MM / common.COMPACT_WIDTH_MM
    full.set_size_inches(
        common.FULL_WIDTH_MM / common.MM_PER_INCH,
        FIGURE_HEIGHT_IN * scale,
        forward=True,
    )
    full_outputs = common.save_exact(
        full,
        output_dir / "full_width" / "figure2_confirmation",
        common.FULL_WIDTH_MM,
    )
    manifest = {
        "schema_version": 1,
        "figure": 2,
        "generator": common.portable(Path(__file__)),
        "scope": "source-level confirmation of the placement ordering",
        "source_validation": validation,
        "science_values_changed": False,
        "compact_physical_size": common.physical_record(common.COMPACT_WIDTH_MM, FIGURE_HEIGHT_MM),
        "compact_outputs": compact_outputs,
        "full_width_physical_size": common.physical_record(
            common.FULL_WIDTH_MM, FIGURE_HEIGHT_MM * scale
        ),
        "full_width_outputs": full_outputs,
        "sources": [
            {"path": common.portable(path), "sha256": common.sha256(path)}
            for path in common.source_paths()
        ],
    }
    (output_dir / "figure2_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
