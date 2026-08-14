"""Generate Figure 1: placement response and terminal morphology."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

try:
    from . import figure_common as common
except ImportError:  # Direct execution from this directory.
    import figure_common as common


DEFAULT_OUTPUT_DIR = common.FIGURE_DIR / "artwork" / "main"
FIGURE_HEIGHT_MM = 153.0
OUTER_MARGIN_MM = 2.5
COLUMN_WIDTH_MM = 62.0
COLUMN_GUTTER_MM = 8.0
LEFT_COLUMN_X = OUTER_MARGIN_MM / common.COMPACT_WIDTH_MM
RIGHT_COLUMN_X = (
    OUTER_MARGIN_MM + COLUMN_WIDTH_MM + COLUMN_GUTTER_MM
) / common.COMPACT_WIDTH_MM
COLUMN_WIDTH = COLUMN_WIDTH_MM / common.COMPACT_WIDTH_MM
TOP_AXIS_INSET_MM = 13.5
TOP_AXIS_WIDTH_MM = 48.5
TOP_X_LEFT = (OUTER_MARGIN_MM + TOP_AXIS_INSET_MM) / common.COMPACT_WIDTH_MM
TOP_X_RIGHT = (
    OUTER_MARGIN_MM
    + COLUMN_WIDTH_MM
    + COLUMN_GUTTER_MM
    + TOP_AXIS_INSET_MM
) / common.COMPACT_WIDTH_MM
TOP_WIDTH = TOP_AXIS_WIDTH_MM / common.COMPACT_WIDTH_MM
TOP_Y_MM = 94.0
TOP_HEIGHT_MM = 43.0
LOWER_Y_MM = 7.0
LOWER_SIZE_MM = 62.0
TOP_LABEL_Y_MM = 143.0
LOWER_LABEL_Y_MM = 74.0


def _morphology_panel(
    axis: plt.Axes,
    arrays: Any,
    manifest: dict[str, Any],
    case: str,
    color: str,
) -> None:
    y = arrays["y"]
    z = arrays["z"]
    projection = arrays[f"projection_{case}"]
    extent = [float(z.min()), float(z.max()), float(y.min()), float(y.max())]
    axis.imshow(
        projection,
        origin="lower",
        extent=extent,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        rasterized=True,
    )
    axis.contour(
        z,
        y,
        projection,
        levels=[0.45],
        colors=[common.COLORS["ink"]],
        linewidths=0.55,
    )
    event_site = float(manifest["cases"][case]["reported_event_site"])
    marker = axis.plot(
        event_site,
        0.0,
        marker="v",
        ms=8.4,
        color=color,
        mec="white",
        mew=0.8,
        zorder=10,
    )[0]
    marker.set_gid(f"first-site-{case}")

    x0, x1, bar_y = -38.0, -18.0, -40.5
    for x, ydata in (
        ([x0, x1], [bar_y, bar_y]),
        ([x0, x0], [bar_y - 1.35, bar_y + 1.35]),
        ([x1, x1], [bar_y - 1.35, bar_y + 1.35]),
    ):
        line = axis.plot(x, ydata, color=common.COLORS["ink"], lw=2.0 if len(set(ydata)) == 1 else 1.1, zorder=25)[0]
        line.set_gid("scale-bar")
    label = axis.text(
        0.5 * (x0 + x1),
        -37.5,
        "20 model\nunits",
        ha="center",
        va="bottom",
        fontsize=common.MIN_TEXT_PT,
        linespacing=0.88,
        color=common.COLORS["ink"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 0.65},
        zorder=26,
    )
    label.set_gid("scale-label")
    axis.set_xlim(-48, 48)
    axis.set_ylim(-48, 48)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#8B95A4")


def build_figure(
    position: list[dict[str, str]],
    morphology_manifest: dict[str, Any],
    morphology_arrays: Any,
) -> plt.Figure:
    common.style()
    figure = plt.figure(
        figsize=(
            common.COMPACT_WIDTH_MM / common.MM_PER_INCH,
            FIGURE_HEIGHT_MM / common.MM_PER_INCH,
        ),
        constrained_layout=False,
    )
    ax_a = figure.add_axes(
        [TOP_X_LEFT, TOP_Y_MM / FIGURE_HEIGHT_MM, TOP_WIDTH, TOP_HEIGHT_MM / FIGURE_HEIGHT_MM]
    )
    ax_b = figure.add_axes(
        [TOP_X_RIGHT, TOP_Y_MM / FIGURE_HEIGHT_MM, TOP_WIDTH, TOP_HEIGHT_MM / FIGURE_HEIGHT_MM]
    )
    ax_c = figure.add_axes(
        [LEFT_COLUMN_X, LOWER_Y_MM / FIGURE_HEIGHT_MM, COLUMN_WIDTH, LOWER_SIZE_MM / FIGURE_HEIGHT_MM]
    )
    ax_d = figure.add_axes(
        [RIGHT_COLUMN_X, LOWER_Y_MM / FIGURE_HEIGHT_MM, COLUMN_WIDTH, LOWER_SIZE_MM / FIGURE_HEIGHT_MM]
    )

    distance = np.linspace(0.0, 48.0, 961)
    for center, color, label, linestyle in (
        (18.5, common.COLORS["near"], "18.5", "-"),
        (26.5, common.COLORS["intermediate"], "26.5", "--"),
        (34.5, common.COLORS["outboard"], "34.5", "-."),
    ):
        ax_a.plot(
            distance,
            common.mask_profile(distance, center),
            color=color,
            lw=1.75,
            ls=linestyle,
            label=label,
        )
    ax_a.axvspan(14.0, 22.5, color=common.COLORS["local_band"], alpha=0.68, lw=0)
    ax_a.set_xlim(5, 47)
    ax_a.set_ylim(0.0, 1.05)
    ax_a.set_xlabel(r"distance from weld, $|s|$")
    ax_a.set_ylabel(r"mobility multiplier, $m$")
    ax_a.set_yticks([0.1, 0.5, 1.0])
    ax_a.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.075),
        ncol=3,
        handlelength=1.35,
        handletextpad=0.30,
        columnspacing=0.65,
        borderpad=0.10,
    )
    common.clean_axis(ax_a, "y")

    rows = {row["case_id"]: row for row in position}
    untreated = rows["untreated"]
    for case in ("c14p5", "c18p5", "c22p5", "c26p5", "c30p5", "c34p5", "c38p5"):
        row = rows[case]
        center = common.number(row, "center")
        effect = common.number(row, "delta_vs_untreated")
        low, high = common.paired_interval(row, untreated)
        if effect > 25:
            color = common.COLORS["near"] if center <= 22.5 else common.COLORS["intermediate"]
        elif effect < -25:
            color = common.COLORS["outboard"]
        else:
            color = common.COLORS["far"]
        ax_b.errorbar(
            center,
            effect,
            yerr=[[effect - low], [high - effect]],
            fmt="o",
            ms=common.PRIMARY_MARKER_PT,
            mfc=color,
            mec="white",
            mew=0.75,
            ecolor=color,
            elinewidth=1.1,
            capsize=3.0,
            zorder=4,
        )
    far = rows["c64p5"]
    far_effect = common.number(far, "delta_vs_untreated")
    far_low, far_high = common.paired_interval(far, untreated)
    ax_b.errorbar(
        common.number(far, "center"),
        far_effect,
        yerr=[[far_effect - far_low], [far_high - far_effect]],
        fmt="D",
        ms=common.PRIMARY_MARKER_PT,
        mfc="white",
        mec=common.COLORS["far"],
        mew=1.2,
        ecolor=common.COLORS["far"],
        elinewidth=1.1,
        capsize=3.0,
        zorder=4,
    )
    ax_b.annotate(
        "far control",
        (common.number(far, "center"), far_effect),
        xytext=(-8, 10),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=common.MIN_TEXT_PT,
        color=common.COLORS["far"],
    )
    ax_b.axhline(0.0, color=common.COLORS["ink"], lw=0.9)
    ax_b.set_xlim(7, 68)
    ax_b.set_ylim(-210, 930)
    ax_b.set_yticks([-100, 0, 250, 500, 750])
    ax_b.set_xlabel(r"collar center, $s_c$")
    ax_b.set_ylabel("")
    ax_b.text(
        0.20,
        1.035,
        r"$\Delta T$ (model time units)",
        transform=ax_b.transAxes,
        ha="left",
        va="bottom",
        fontsize=common.MIN_TEXT_PT,
        color=common.COLORS["ink"],
        clip_on=False,
        zorder=45,
    )
    ax_b.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                ls="none",
                ms=common.PRIMARY_MARKER_PT,
                mfc="white",
                mec=common.COLORS["far"],
                mew=1.1,
                label="scan",
            )
        ],
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.075),
        handletextpad=0.35,
        borderpad=0.10,
    )
    common.clean_axis(ax_b, "y")

    _morphology_panel(ax_c, morphology_arrays, morphology_manifest, "near", common.COLORS["near"])
    _morphology_panel(ax_d, morphology_arrays, morphology_manifest, "outboard", common.COLORS["outboard"])

    top_y = TOP_LABEL_Y_MM / FIGURE_HEIGHT_MM
    lower_y = LOWER_LABEL_Y_MM / FIGURE_HEIGHT_MM
    for x, label in ((LEFT_COLUMN_X, "a"), (RIGHT_COLUMN_X, "b")):
        common.figure_panel_label(figure, x, top_y, label)
    for x, label in ((LEFT_COLUMN_X, "c"), (RIGHT_COLUMN_X, "d")):
        common.figure_panel_label(figure, x, lower_y, label)
    descriptor_offset = 11.0 / common.COMPACT_WIDTH_MM
    near = figure.text(
        LEFT_COLUMN_X + descriptor_offset,
        lower_y,
        r"Near, $s_c=18.5$: remote",
        ha="left",
        va="center",
        fontsize=9.8,
        fontweight="bold",
        color=common.COLORS["near"],
    )
    near.set_gid("descriptor-near")
    outboard = figure.text(
        RIGHT_COLUMN_X + descriptor_offset,
        lower_y,
        r"Outboard, $s_c=34.5$: local",
        ha="left",
        va="center",
        fontsize=9.8,
        fontweight="bold",
        color=common.COLORS["outboard"],
    )
    outboard.set_gid("descriptor-outboard")
    return figure


def generate(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    data = common.load_core_sources()
    validation = common.validate_sources(data)
    if not validation["passed"]:
        raise RuntimeError(f"source validation failed: {validation['failures']}")
    full_width_dir = output_dir / "full_width"
    with np.load(common.MORPHOLOGY_SOURCE, allow_pickle=False) as arrays:
        compact = build_figure(data["position"], data["morphology_manifest"], arrays)
        compact_outputs = common.save_exact(
            compact, output_dir / "figure1_phenomenon", common.COMPACT_WIDTH_MM
        )
        full = build_figure(data["position"], data["morphology_manifest"], arrays)
        scale = common.FULL_WIDTH_MM / common.COMPACT_WIDTH_MM
        full.set_size_inches(
            common.FULL_WIDTH_MM / common.MM_PER_INCH,
            FIGURE_HEIGHT_MM * scale / common.MM_PER_INCH,
            forward=True,
        )
        full_outputs = common.save_exact(
            full, full_width_dir / "figure1_phenomenon", common.FULL_WIDTH_MM
        )
    manifest = {
        "schema_version": 1,
        "figure": 1,
        "generator": common.portable(Path(__file__)),
        "scope": "placement response and terminal morphology",
        "source_validation": validation,
        "science_values_changed": False,
        "compact_physical_size": common.physical_record(common.COMPACT_WIDTH_MM, FIGURE_HEIGHT_MM),
        "compact_outputs": compact_outputs,
        "full_width_physical_size": common.physical_record(
            common.FULL_WIDTH_MM,
            FIGURE_HEIGHT_MM * common.FULL_WIDTH_MM / common.COMPACT_WIDTH_MM,
        ),
        "full_width_outputs": full_outputs,
        "sources": [
            {"path": common.portable(path), "sha256": common.sha256(path)}
            for path in common.source_paths()
        ],
    }
    (output_dir / "figure1_manifest.json").write_text(
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
