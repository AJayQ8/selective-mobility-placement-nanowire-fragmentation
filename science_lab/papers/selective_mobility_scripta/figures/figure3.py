"""Generate Figure 3: control-volume transport accounting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
import numpy as np

try:
    from . import figure_common as common
except ImportError:  # Direct execution from this directory.
    import figure_common as common


DEFAULT_OUTPUT_DIR = common.FIGURE_DIR / "artwork" / "main"
FIGURE_HEIGHT_MM = 125.0
FIGURE_HEIGHT_IN = FIGURE_HEIGHT_MM / common.MM_PER_INCH
AXIAL_COLOR = common.COLORS["intermediate"]
TRANSVERSE_COLOR = common.COLORS["transverse"]
WIRE_FILL = "#DCEAF2"
WIRE_EDGE = "#55758A"
ZONE_EDGE = TRANSVERSE_COLOR
PLACEMENT_LABELS = [
    "intermediate 26.5",
    "outboard 34.5",
    "farther-out 38.5",
    "far control 64.5",
]


def _draw_control_volumes(axis: plt.Axes) -> None:
    axis.set_xlim(-1.24, 1.30)
    axis.set_ylim(-1.02, 0.90)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")

    axis.add_patch(
        FancyBboxPatch(
            (-1.12, -0.15),
            2.24,
            0.30,
            boxstyle="round,pad=0,rounding_size=0.15",
            facecolor=WIRE_FILL,
            edgecolor=WIRE_EDGE,
            linewidth=1.25,
            zorder=1,
        )
    )
    axis.add_patch(
        FancyBboxPatch(
            (-0.15, -0.82),
            0.30,
            1.64,
            boxstyle="round,pad=0,rounding_size=0.15",
            facecolor=WIRE_FILL,
            edgecolor=WIRE_EDGE,
            linewidth=1.25,
            zorder=1,
        )
    )
    axis.add_patch(
        Circle(
            (0.0, 0.0),
            0.21,
            facecolor="#C7DDE9",
            edgecolor=WIRE_EDGE,
            linewidth=1.25,
            zorder=2,
        )
    )

    for arm, (x, y, width, height) in {
        "right": (0.43, -0.215, 0.33, 0.43),
        "left": (-0.76, -0.215, 0.33, 0.43),
        "top": (-0.215, 0.43, 0.43, 0.33),
        "bottom": (-0.215, -0.76, 0.43, 0.33),
    }.items():
        highlighted = arm == "right"
        volume = Rectangle(
            (x, y),
            width,
            height,
            facecolor=(0.957, 0.847, 0.776, 0.58 if highlighted else 0.24),
            edgecolor=ZONE_EDGE,
            linewidth=1.75 if highlighted else 1.45,
            linestyle=(0, (3.6, 1.7)),
            zorder=4,
        )
        volume.set_gid(f"control-volume-{arm}")
        axis.add_patch(volume)

    for start, end, color, name in (
        ((0.595, 0.0), (0.445, 0.0), AXIAL_COLOR, "axial-inner"),
        ((0.595, 0.0), (0.745, 0.0), AXIAL_COLOR, "axial-outer"),
        ((0.595, 0.0), (0.595, 0.195), TRANSVERSE_COLOR, "transverse-upper"),
        ((0.595, 0.0), (0.595, -0.195), TRANSVERSE_COLOR, "transverse-lower"),
    ):
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9.0,
            linewidth=1.65,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=12,
        )
        arrow.set_gid(f"outward-{name}")
        axis.add_patch(arrow)


def _draw_key(axis: plt.Axes) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    heading = axis.text(
        0.02,
        0.99,
        "outward-normal\nconvention",
        ha="left",
        va="top",
        fontsize=common.MIN_TEXT_PT,
        fontweight="bold",
        color=common.COLORS["ink"],
    )
    heading.set_gid("outward-normal-heading")
    for y, color, label, gid in (
        (0.70, AXIAL_COLOR, "axial faces (2)", "axial"),
        (0.49, TRANSVERSE_COLOR, "transverse crop\nfaces (2)", "transverse"),
    ):
        arrow = FancyArrowPatch(
            (0.03, y),
            (0.30, y),
            arrowstyle="-|>",
            mutation_scale=13.5,
            linewidth=1.45,
            color=color,
            transform=axis.transAxes,
            zorder=5,
        )
        arrow.set_gid(f"key-{gid}-arrow")
        axis.add_patch(arrow)
        text = axis.text(
            0.37,
            y,
            label,
            ha="left",
            va="center",
            fontsize=common.MIN_TEXT_PT,
            color=color,
            linespacing=0.95,
        )
        text.set_gid(f"key-{gid}-label")
    periodic = axis.text(
        0.03,
        0.29,
        r"full periodic $x$ span: net 0",
        ha="left",
        va="center",
        fontsize=common.MIN_TEXT_PT,
        color=common.COLORS["ink"],
    )
    periodic.set_gid("periodic-x-net-zero")
    sample = Rectangle(
        (0.04, 0.03),
        0.24,
        0.12,
        transform=axis.transAxes,
        facecolor=(0.957, 0.847, 0.776, 0.32),
        edgecolor=ZONE_EDGE,
        linewidth=1.45,
        linestyle=(0, (3.6, 1.7)),
    )
    sample.set_gid("key-control-volume-sample")
    axis.add_patch(sample)
    interval = axis.text(
        0.37,
        0.09,
        "all four arms\n$14 \\leq |s| \\leq 22.5$",
        ha="left",
        va="center",
        fontsize=common.MIN_TEXT_PT,
        color=common.COLORS["ink"],
        linespacing=0.95,
    )
    interval.set_gid("key-control-volume-interval")


def build_figure(transport: list[dict[str, str]]) -> plt.Figure:
    common.style()
    figure = plt.figure(
        figsize=(common.COMPACT_WIDTH_MM / common.MM_PER_INCH, FIGURE_HEIGHT_IN),
        constrained_layout=False,
    )
    plot = figure.add_axes([0.285, 0.096, 0.685, 0.310])
    diagram = figure.add_axes([0.055, 0.535, 0.585, 0.385])
    key = figure.add_axes([0.655, 0.525, 0.325, 0.335])
    _draw_control_volumes(diagram)
    _draw_key(key)

    rows = {row["case_id"]: row for row in transport}
    untreated = rows["untreated"]
    cases = ["c26p5", "c34p5", "c38p5", "c64p5"]
    y_positions = np.arange(len(cases), dtype=float)[::-1]
    for y, case in zip(y_positions, cases):
        row = rows[case]
        axial = common.number(row, "axial_rate") - common.number(untreated, "axial_rate")
        transverse = common.number(row, "transverse_rate") - common.number(
            untreated, "transverse_rate"
        )
        observed = common.number(row, "observed_rate_change")
        axial_bar = plot.barh(
            y,
            axial,
            height=0.55,
            color=AXIAL_COLOR,
            alpha=0.86,
            edgecolor="white",
            lw=0.4,
            zorder=2,
        )[0]
        axial_bar.set_gid(f"axial-{case}")
        transverse_bar = plot.barh(
            y,
            transverse,
            left=axial,
            height=0.55,
            color=TRANSVERSE_COLOR,
            alpha=0.96,
            edgecolor="white",
            lw=0.4,
            zorder=3,
        )[0]
        transverse_bar.set_gid(f"transverse-{case}")
        marker = plot.plot(
            observed,
            y,
            marker="D",
            ms=common.PRIMARY_MARKER_PT,
            mfc="white",
            mec=common.COLORS["ink"],
            mew=1.1,
            ls="none",
            zorder=5,
        )[0]
        marker.set_gid(f"measured-{case}")

    null_label = plot.text(
        0.004,
        0.0,
        r"$\approx 0$",
        ha="left",
        va="center",
        fontsize=common.MIN_TEXT_PT,
        color=common.COLORS["ink"],
        zorder=8,
    )
    null_label.set_gid("far-control-near-zero")
    plot.axvline(0, color=common.COLORS["ink"], lw=0.9)
    plot.set_yticks(y_positions, PLACEMENT_LABELS)
    plot.set_ylim(-0.72, 3.72)
    plot.set_xlim(-0.073, 0.075)
    ticks = [-0.06, -0.03, 0.00, 0.03, 0.06]
    plot.set_xticks(ticks, [f"{value:.2f}" for value in ticks])
    plot.set_xlabel(r"change in weak-zone rate, $\Delta\dot{V}$")
    plot.legend(
        handles=[
            Patch(facecolor=AXIAL_COLOR, alpha=0.86, label="axial"),
            Patch(facecolor=TRANSVERSE_COLOR, label="transverse"),
            Line2D(
                [0],
                [0],
                marker="D",
                ls="none",
                ms=common.PRIMARY_MARKER_PT,
                mfc="white",
                mec=common.COLORS["ink"],
                label="measured",
            ),
        ],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        handlelength=1.25,
        handletextpad=0.45,
        columnspacing=1.1,
        borderpad=0.15,
    )
    common.clean_axis(plot, "x")

    title = figure.text(
        0.35,
        0.935,
        "four-arm weak-zone control volumes",
        ha="center",
        va="center",
        fontsize=common.MIN_TEXT_PT,
        fontweight="bold",
        color=common.COLORS["ink"],
    )
    title.set_gid("control-volume-title")
    common.figure_panel_label(figure, 0.025, 0.935, "a")
    common.figure_panel_label(figure, 0.025, 0.452, "b")
    return figure


def generate(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    data = common.load_core_sources()
    validation = common.validate_sources(data)
    if not validation["passed"]:
        raise RuntimeError(f"source validation failed: {validation['failures']}")
    compact_outputs = common.save_exact(
        build_figure(data["transport"]),
        output_dir / "figure3_transport",
        common.COMPACT_WIDTH_MM,
    )
    scale = common.FULL_WIDTH_MM / common.COMPACT_WIDTH_MM
    full = build_figure(data["transport"])
    full.set_size_inches(
        common.FULL_WIDTH_MM / common.MM_PER_INCH,
        FIGURE_HEIGHT_IN * scale,
        forward=True,
    )
    full_outputs = common.save_exact(
        full,
        output_dir / "full_width" / "figure3_transport",
        common.FULL_WIDTH_MM,
    )
    manifest = {
        "schema_version": 1,
        "figure": 3,
        "generator": common.portable(Path(__file__)),
        "scope": "control-volume transport accounting",
        "source_validation": validation,
        "science_values_changed": False,
        "compact_physical_size": common.physical_record(
            common.COMPACT_WIDTH_MM, FIGURE_HEIGHT_MM
        ),
        "compact_outputs": compact_outputs,
        "full_width_physical_size": common.physical_record(
            common.FULL_WIDTH_MM, FIGURE_HEIGHT_MM * scale
        ),
        "full_width_outputs": full_outputs,
        "geometry": {
            "finite_axial_faces_per_arm": 2,
            "finite_in_plane_transverse_faces_per_arm": 2,
            "full_periodic_x_span": True,
            "periodic_x_net_contribution": 0.0,
            "control_volumes_shown": 4,
            "weak_zone_interval": [14.0, 22.5],
        },
        "sources": [
            {"path": common.portable(path), "sha256": common.sha256(path)}
            for path in common.source_paths()
        ],
    }
    (output_dir / "figure3_manifest.json").write_text(
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
