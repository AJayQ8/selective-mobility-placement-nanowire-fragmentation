"""Generate Figure 4: numerical and mobility-contrast robustness."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


FIGURE_DIR = Path(__file__).resolve().parent
PAPER_ROOT = FIGURE_DIR.parent
REPOSITORY_ROOT = PAPER_ROOT.parents[2]
SOURCE_DIR = PAPER_ROOT / "source_data"
TIMESTEP_SOURCE = SOURCE_DIR / "timestep_refinement.csv"
SPATIAL_SOURCE = SOURCE_DIR / "spatial_refinement.csv"
CONTRAST_SOURCE = SOURCE_DIR / "mobility_contrast.csv"
DEFAULT_OUTPUT_DIR = FIGURE_DIR / "artwork" / "main"

MM_PER_INCH = 25.4
POINTS_PER_INCH = 72.0
COMPACT_WIDTH_MM = 137.0
COMPACT_HEIGHT_MM = 145.0
FULL_WIDTH_MM = 190.0
EXPORT_DPI = 600

COLORS = {
    "ink": "#202124",
    "grid": "#D9DEE5",
    "band": "#E5E8ED",
    "intermediate": "#0072B2",
    "outboard": "#D55E00",
}
SOURCE_MARKERS = {"B": "s", "C": "^"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.4,
            "axes.labelsize": 10.8,
            "xtick.labelsize": 9.8,
            "ytick.labelsize": 9.8,
            "legend.fontsize": 9.4,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.3,
            "ytick.major.size": 3.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def load_and_validate() -> dict[str, list[dict[str, str]]]:
    timestep = _read_csv(TIMESTEP_SOURCE)
    spatial = _read_csv(SPATIAL_SOURCE)
    contrast = _read_csv(CONTRAST_SOURCE)

    timestep_by_case = {row["case_id"]: row for row in timestep}
    if set(timestep_by_case) != {"untreated", "c34p5"}:
        raise RuntimeError("unexpected timestep cases")
    treated_dt = timestep_by_case["c34p5"]
    if (float(treated_dt["coarse_paired_effect"]), float(treated_dt["refined_paired_effect"])) != (-110.0, -100.0):
        raise RuntimeError("unexpected timestep paired effects")

    spatial_by_case = {(row["grid_level"], row["case_id"]): row for row in spatial}
    for level in ("R", "F"):
        row = spatial_by_case[(level, "c34p5")]
        if float(row["paired_effect_c34_minus_untreated"]) != -100.0:
            raise RuntimeError(f"unexpected {level} paired effect")
        if (float(row["paired_interval_lower"]), float(row["paired_interval_upper"])) != (-110.0, -90.0):
            raise RuntimeError(f"unexpected {level} paired interval")
    fine = spatial_by_case[("F", "c34p5")]
    if fine["passed"] != "true" or fine["all_frozen_checks_passed"] != "true":
        raise RuntimeError("fine-grid validation is not complete")

    expected_contrast = {
        ("B", 0.1, "intermediate"): 270.0,
        ("C", 0.1, "intermediate"): 260.0,
        ("B", 0.1, "outboard"): -110.0,
        ("C", 0.1, "outboard"): -100.0,
        ("B", 0.3, "intermediate"): 200.0,
        ("C", 0.3, "intermediate"): 200.0,
        ("B", 0.3, "outboard"): -60.0,
        ("C", 0.3, "outboard"): -60.0,
    }
    actual_contrast = {
        (row["source_label"], float(row["m_min"]), row["placement_role"]): float(row["paired_shift"])
        for row in contrast
    }
    if actual_contrast != expected_contrast:
        raise RuntimeError(f"unexpected contrast values: {actual_contrast}")
    for row in contrast:
        shift = float(row["paired_shift"])
        if (float(row["paired_interval_lower"]), float(row["paired_interval_upper"])) != (shift - 10.0, shift + 10.0):
            raise RuntimeError(f"unexpected contrast interval: {row}")
        if row["failure_mode"] != "junction_adjacent":
            raise RuntimeError(f"unexpected contrast topology: {row}")

    return {"timestep": timestep, "spatial": spatial, "contrast": contrast}


def _panel_label(figure: plt.Figure, x: float, y: float, label: str, title: str) -> None:
    figure.text(x, y, f"({label})", fontsize=12.0, fontweight="bold", ha="left", va="center", color=COLORS["ink"])
    figure.text(x + 0.065, y, title, fontsize=10.4, fontweight="bold", ha="left", va="center", color=COLORS["ink"])


def _paired_interval(treated: dict[str, str], untreated: dict[str, str], prefix: str) -> tuple[float, float]:
    return (
        float(treated[f"{prefix}_event_lower"]) - float(untreated[f"{prefix}_event_upper"]),
        float(treated[f"{prefix}_event_upper"]) - float(untreated[f"{prefix}_event_lower"]),
    )


def build_figure(data: dict[str, list[dict[str, str]]], width_mm: float) -> plt.Figure:
    _style()
    height_mm = COMPACT_HEIGHT_MM * width_mm / COMPACT_WIDTH_MM
    figure = plt.figure(figsize=(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH))
    ax_a = figure.add_axes([0.180, 0.630, 0.800, 0.230])
    ax_b = figure.add_axes([0.180, 0.080, 0.800, 0.280])
    _panel_label(figure, 0.020, 0.955, "a", "numerical validation")
    _panel_label(figure, 0.020, 0.490, "b", "mobility-contrast robustness")

    timestep = {row["case_id"]: row for row in data["timestep"]}
    treated_dt = timestep["c34p5"]
    untreated_dt = timestep["untreated"]
    top_points = [
        (-0.10, -110.0, _paired_interval(treated_dt, untreated_dt, "coarse"), "o", True, r"$\Delta t=1$"),
        (0.10, -100.0, _paired_interval(treated_dt, untreated_dt, "refined"), "D", False, r"$\Delta t=0.5$"),
        (1.00, -100.0, (-110.0, -90.0), "o", True, None),
        (2.00, -100.0, (-110.0, -90.0), "o", True, None),
    ]
    ax_a.axhspan(-125.0, -75.0, xmin=0.73, xmax=0.98, color=COLORS["band"], alpha=0.95, zorder=0)
    for x, value, interval, marker, filled, label in top_points:
        low, high = interval
        ax_a.errorbar(
            x,
            value,
            yerr=[[value - low], [high - value]],
            fmt=marker,
            color=COLORS["outboard"],
            markerfacecolor=COLORS["outboard"] if filled else "white",
            markeredgecolor=COLORS["outboard"],
            markersize=8.2,
            markeredgewidth=1.0,
            elinewidth=1.15,
            capsize=3.4,
            capthick=1.15,
            label=label,
            zorder=4,
        )
    ax_a.axhline(0.0, color=COLORS["ink"], linewidth=1.0, zorder=2)
    ax_a.set_xlim(-0.42, 2.42)
    ax_a.set_ylim(-155.0, 15.0)
    ax_a.set_yticks([-150, -100, -50, 0])
    ax_a.set_xticks([0.0, 1.0, 2.0], ["P: full\n$h=0.5$", "R: reduced\n$h=0.5$", "F: refined\n$h=0.25$"])
    ax_a.set_ylabel("paired shift, $\\Delta T$\n(model time units)")
    ax_a.grid(axis="y", color=COLORS["grid"], linewidth=0.7, zorder=0)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    top_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=7.5, markerfacecolor=COLORS["outboard"], markeredgecolor=COLORS["outboard"], label=r"$\Delta t=1$"),
        Line2D([], [], marker="D", linestyle="none", markersize=7.5, markerfacecolor="white", markeredgecolor=COLORS["outboard"], label=r"$\Delta t=0.5$"),
        Patch(facecolor=COLORS["band"], edgecolor="none", label="predefined interval"),
    ]
    ax_a.legend(handles=top_handles, loc="lower center", bbox_to_anchor=(0.57, 1.015), ncol=3, frameon=False, handletextpad=0.45, columnspacing=1.25)

    contrast_x = {0.1: 0.0, 0.3: 1.0}
    role_offset = {"intermediate": -0.115, "outboard": 0.115}
    source_offset = {"B": -0.026, "C": 0.026}
    role_color = {"intermediate": COLORS["intermediate"], "outboard": COLORS["outboard"]}
    for row in data["contrast"]:
        source = row["source_label"]
        role = row["placement_role"]
        factor = float(row["m_min"])
        shift = float(row["paired_shift"])
        x = contrast_x[factor] + role_offset[role] + source_offset[source]
        ax_b.errorbar(
            x,
            shift,
            yerr=[[shift - float(row["paired_interval_lower"])], [float(row["paired_interval_upper"]) - shift]],
            fmt=SOURCE_MARKERS[source],
            markersize=8.2,
            markerfacecolor=role_color[role],
            markeredgecolor="white",
            markeredgewidth=0.9,
            ecolor=role_color[role],
            elinewidth=1.15,
            capsize=3.4,
            capthick=1.15,
            linestyle="none",
            zorder=5,
        )
    ax_b.axhline(0.0, color=COLORS["ink"], linewidth=1.0, zorder=2)
    ax_b.set_xlim(-0.42, 1.42)
    ax_b.set_ylim(-150.0, 315.0)
    ax_b.set_xticks([0.0, 1.0], ["0.1", "0.3"])
    ax_b.set_yticks([-100, 0, 100, 200, 300])
    ax_b.set_xlabel(r"minimum mobility multiplier, $m_{\min}$")
    ax_b.set_ylabel("paired shift, $\\Delta T$\n(model time units)")
    ax_b.grid(axis="y", color=COLORS["grid"], linewidth=0.7, zorder=0)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    bottom_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=7.2, markerfacecolor=COLORS["intermediate"], markeredgecolor="white", label=r"intermediate, $s_c=26.5$"),
        Line2D([], [], marker="o", linestyle="none", markersize=7.2, markerfacecolor=COLORS["outboard"], markeredgecolor="white", label=r"outboard, $s_c=34.5$"),
        Line2D([], [], marker="s", linestyle="none", markersize=7.0, markerfacecolor="white", markeredgecolor=COLORS["ink"], label="source B"),
        Line2D([], [], marker="^", linestyle="none", markersize=7.0, markerfacecolor="white", markeredgecolor=COLORS["ink"], label="source C"),
    ]
    treatment_legend = ax_b.legend(
        handles=bottom_handles[:2],
        loc="lower center",
        bbox_to_anchor=(0.50, 1.205),
        ncol=2,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.20,
    )
    ax_b.add_artist(treatment_legend)
    ax_b.legend(
        handles=bottom_handles[2:],
        loc="lower center",
        bbox_to_anchor=(0.50, 1.065),
        ncol=2,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.45,
    )
    return figure


def _save(figure: plt.Figure, directory: Path, expected_width_mm: float) -> list[dict[str, Any]]:
    from PIL import Image
    from pypdf import PdfReader

    directory.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    for suffix in ("pdf", "png", "svg"):
        path = directory / f"figure4_robustness.{suffix}"
        kwargs: dict[str, Any] = {"facecolor": "white"}
        if suffix == "pdf":
            kwargs["metadata"] = {
                "CreationDate": None,
                "Creator": None,
                "Producer": None,
            }
        elif suffix == "png":
            kwargs.update({"dpi": EXPORT_DPI, "metadata": {"Software": None}})
        else:
            kwargs["metadata"] = {"Date": None, "Creator": None}
        figure.savefig(path, **kwargs)
        if suffix == "svg":
            normalized = "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
            path.write_text(normalized + "\n", encoding="utf-8")
        outputs.append({"path": _portable(path), "sha256": _sha256(path), "bytes": path.stat().st_size})

    pdf = directory / "figure4_robustness.pdf"
    page = PdfReader(str(pdf)).pages[0]
    width_mm = float(page.mediabox.width) / POINTS_PER_INCH * MM_PER_INCH
    if abs(width_mm - expected_width_mm) > 0.03:
        raise RuntimeError(f"unexpected PDF width: {width_mm:.4f} mm")
    png = directory / "figure4_robustness.png"
    with Image.open(png) as image:
        expected_px = round(expected_width_mm / MM_PER_INCH * EXPORT_DPI)
        if abs(image.width - expected_px) > 1:
            raise RuntimeError(f"unexpected PNG width: {image.width} px")
    return outputs


def generate(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    data = load_and_validate()
    review = build_figure(data, COMPACT_WIDTH_MM)
    full = build_figure(data, FULL_WIDTH_MM)
    try:
        compact_outputs = _save(review, output_dir, COMPACT_WIDTH_MM)
        full_outputs = _save(full, output_dir / "full_width", FULL_WIDTH_MM)
    finally:
        plt.close(review)
        plt.close(full)

    manifest = {
        "schema_version": 1,
        "figure": 4,
        "generator": _portable(Path(__file__)),
        "scope": "targeted numerical validation and two-factor mobility-contrast robustness",
        "sources": [
            {"path": _portable(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (TIMESTEP_SOURCE, SPATIAL_SOURCE, CONTRAST_SOURCE)
        ],
        "science": {
            "production_dt1_shift": -110.0,
            "production_dt05_shift": -100.0,
            "reduced_shift": -100.0,
            "refined_shift": -100.0,
            "fine_acceptance_interval": [-125.0, -75.0],
            "contrast_factors": [0.1, 0.3],
            "m03_intermediate_shifts": {"B": 200.0, "C": 200.0},
            "m03_outboard_shifts": {"B": -60.0, "C": -60.0},
            "all_contrast_events_junction_adjacent": True,
            "continuous_contrast_law_claimed": False,
        },
        "geometry": {
            "compact_width_mm": COMPACT_WIDTH_MM,
            "compact_height_mm": COMPACT_HEIGHT_MM,
            "full_width_mm": FULL_WIDTH_MM,
            "full_height_mm": COMPACT_HEIGHT_MM * FULL_WIDTH_MM / COMPACT_WIDTH_MM,
            "minimum_semantic_text_pt": 9.4,
            "marker_size_pt": 8.2,
            "panel_count": 2,
        },
        "compact_outputs": compact_outputs,
        "full_width_outputs": full_outputs,
    }
    manifest_path = output_dir / "figure4_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
