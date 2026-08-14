"""Generate the overlap guide and scalar-control supporting figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    from . import figure_common as common
except ImportError:  # Direct execution from this directory.
    import figure_common as common


DEFAULT_OUTPUT_DIR = common.FIGURE_DIR / "artwork" / "supporting"
LAYOUT_SCALE = common.COMPACT_WIDTH_MM / 180.0
OVERLAP_HEIGHT_IN = 5.35 * LAYOUT_SCALE
CONTROLS_HEIGHT_IN = 3.45 * LAYOUT_SCALE


def build_overlap_guide(position: list[dict[str, str]]) -> plt.Figure:
    figure, (ax_a, ax_b) = plt.subplots(
        2,
        1,
        figsize=(common.COMPACT_WIDTH_MM / common.MM_PER_INCH, OVERLAP_HEIGHT_IN),
        constrained_layout=False,
        sharex=True,
    )
    figure.subplots_adjust(left=0.19, right=0.985, bottom=0.12, top=0.95, hspace=0.30)
    rows = {row["case_id"]: row for row in position}
    untreated = rows["untreated"]
    ordered = [
        "c14p5",
        "c18p5",
        "c22p5",
        "c26p5",
        "c30p5",
        "c34p5",
        "c38p5",
        "c64p5",
    ]
    for case in ordered:
        row = rows[case]
        center = common.number(row, "center")
        effect = common.number(row, "delta_vs_untreated")
        low, high = common.paired_interval(row, untreated)
        marker = "D" if case == "c64p5" else "o"
        face = "white" if case == "c64p5" else common.COLORS["intermediate"]
        ax_a.errorbar(
            center,
            effect,
            yerr=[[effect - low], [high - effect]],
            fmt=marker,
            ms=7.2,
            mfc=face,
            mec=(
                common.COLORS["intermediate"]
                if case != "c64p5"
                else common.COLORS["far"]
            ),
            mew=1.0,
            ecolor=(
                common.COLORS["intermediate"]
                if case != "c64p5"
                else common.COLORS["far"]
            ),
            elinewidth=1.1,
            capsize=3.0,
            zorder=4,
        )
    ax_a.axhline(0, color=common.COLORS["ink"], lw=0.9)
    ax_a.set_ylim(-210, 930)
    ax_a.set_ylabel("paired shift, " + r"$\Delta T$" + "\n(model time units)")
    common.clean_axis(ax_a, "y")
    common.panel_label(ax_a, "a", x=0.008, y=0.92)

    weak_low, weak_high = 14.0, 22.5
    centers = np.linspace(7.0, 68.0, 1221)

    def overlap_length(center: np.ndarray, half_width: float) -> np.ndarray:
        lower = np.maximum(center - half_width, weak_low)
        upper = np.minimum(center + half_width, weak_high)
        return np.maximum(0.0, upper - lower)

    support = overlap_length(centers, 6.0)
    plateau = overlap_length(centers, 3.0)
    ax_b.plot(
        centers,
        support,
        color=common.COLORS["intermediate"],
        lw=1.9,
        label="collar support",
    )
    ax_b.plot(
        centers,
        plateau,
        color=common.COLORS["near"],
        lw=1.9,
        ls="--",
        label="low-mobility plateau",
    )
    evaluated = np.array([common.number(rows[case], "center") for case in ordered])
    ax_b.plot(
        evaluated,
        overlap_length(evaluated, 6.0),
        marker="o",
        ls="none",
        ms=7.2,
        mfc=common.COLORS["intermediate"],
        mec="white",
        mew=0.7,
    )
    ax_b.plot(
        evaluated,
        overlap_length(evaluated, 3.0),
        marker="s",
        ls="none",
        ms=7.2,
        mfc=common.COLORS["near"],
        mec="white",
        mew=0.7,
    )
    ax_b.set_xlim(7, 68)
    ax_b.set_ylim(-0.45, 9.1)
    ax_b.set_xlabel(r"collar center, $s_c$")
    ax_b.set_ylabel("overlap with untreated\nweak zone (model units)")
    ax_b.legend(
        frameon=False,
        ncol=2,
        loc="upper right",
        handlelength=2.2,
        handletextpad=0.5,
        columnspacing=1.2,
        borderpad=0.15,
    )
    common.clean_axis(ax_b, "y")
    common.panel_label(ax_b, "b", x=0.008, y=0.90)
    return figure


def build_scalar_controls(
    position: list[dict[str, str]], budget: list[dict[str, str]]
) -> plt.Figure:
    figure = plt.figure(
        figsize=(common.COMPACT_WIDTH_MM / common.MM_PER_INCH, CONTROLS_HEIGHT_IN),
        constrained_layout=False,
    )
    grid = figure.add_gridspec(
        1,
        2,
        left=0.165,
        right=0.985,
        bottom=0.22,
        top=0.88,
        wspace=0.40,
        width_ratios=(1.0, 1.12),
    )
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    position_rows = {row["case_id"]: row for row in position}
    budget_rows = {row["case_id"]: row for row in budget}
    untreated = position_rows["untreated"]
    q = common.number(budget_rows["K3"], "mobility_factor")
    mapped_lower = 100.0 + q * (common.number(budget_rows["K3"], "event_lower") - 100.0)
    mapped_upper = 100.0 + q * (common.number(budget_rows["K3"], "event_upper") - 100.0)
    intervals = [
        (
            common.number(untreated, "event_lower"),
            common.number(untreated, "event_upper"),
            common.COLORS["untreated"],
        ),
        (mapped_lower, mapped_upper, common.COLORS["intermediate"]),
    ]
    for y, (low, high, color) in zip([1, 0], intervals):
        ax_a.plot([low, high], [y, y], color=color, lw=3.0, solid_capstyle="butt")
        ax_a.plot([low, high], [y, y], marker="|", color=color, ms=9.0, ls="none")
        ax_a.plot(0.5 * (low + high), y, marker="o", color=color, ms=7.2)
    ax_a.set_yticks([1, 0], ["untreated", "mapped"])
    ax_a.set_xlim(1688, 1703)
    ax_a.set_ylim(-0.6, 1.6)
    ax_a.set_xlabel("production-clock event bracket")
    common.clean_axis(ax_a, "x")
    common.panel_label(ax_a, "a", x=-0.14, y=1.02)

    controls = ["K1", "K2", "K3"]
    labels = ["off-\njunction\ncollar", "junction\ncap", "uniform"]
    colors = [
        common.COLORS["near"],
        common.COLORS["intermediate"],
        common.COLORS["untreated"],
    ]
    x = np.arange(3, dtype=float)
    for x_value, case, color in zip(x, controls, colors):
        row = budget_rows[case]
        low, high = common.number(row, "event_lower"), common.number(row, "event_upper")
        midpoint = common.number(row, "event_midpoint")
        ax_b.errorbar(
            x_value,
            midpoint,
            yerr=[[midpoint - low], [high - midpoint]],
            fmt="o",
            ms=8.0,
            color=color,
            capsize=3.3,
            elinewidth=1.2,
        )
    ax_b.set_xticks(x, labels)
    ax_b.set_xlim(-0.45, 2.45)
    ax_b.set_ylim(1650, 2640)
    ax_b.set_ylabel(r"event time, $T$ (model units)")
    common.clean_axis(ax_b, "y")
    common.panel_label(ax_b, "b", x=-0.12, y=1.02)
    return figure


def _save_pair(
    builder: Any,
    arguments: tuple[Any, ...],
    stem: str,
    height_in: float,
    output_dir: Path,
) -> dict[str, Any]:
    compact = common.save_exact(
        builder(*arguments), output_dir / stem, common.COMPACT_WIDTH_MM
    )
    scale = common.FULL_WIDTH_MM / common.COMPACT_WIDTH_MM
    full = builder(*arguments)
    full.set_size_inches(
        common.FULL_WIDTH_MM / common.MM_PER_INCH,
        height_in * scale,
        forward=True,
    )
    full_outputs = common.save_exact(
        full,
        output_dir / "full_width" / stem,
        common.FULL_WIDTH_MM,
    )
    return {
        "compact_outputs": compact,
        "full_width_outputs": full_outputs,
        "compact_physical_size": common.physical_record(
            common.COMPACT_WIDTH_MM, height_in * common.MM_PER_INCH
        ),
        "full_width_physical_size": common.physical_record(
            common.FULL_WIDTH_MM, height_in * common.MM_PER_INCH * scale
        ),
    }


def generate(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    common.style()
    data = common.load_core_sources()
    validation = common.validate_sources(data)
    if not validation["passed"]:
        raise RuntimeError(f"source validation failed: {validation['failures']}")
    figures = {
        "overlap_guide": _save_pair(
            build_overlap_guide,
            (data["position"],),
            "supporting_overlap_guide",
            OVERLAP_HEIGHT_IN,
            output_dir,
        ),
        "scalar_controls": _save_pair(
            build_scalar_controls,
            (data["position"], data["budget"]),
            "supporting_scalar_controls",
            CONTROLS_HEIGHT_IN,
            output_dir,
        ),
    }
    manifest = {
        "schema_version": 1,
        "generator": common.portable(Path(__file__)),
        "scope": "overlap guide and scalar controls",
        "source_validation": validation,
        "figures": figures,
        "sources": [
            {"path": common.portable(path), "sha256": common.sha256(path)}
            for path in common.source_paths()
        ],
    }
    (output_dir / "supporting_figures_manifest.json").write_text(
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
