"""Generate the graphical abstract from the paired source-level response table.

The artwork isolates the study's strongest causal comparison: moving the
same four-arm mask from the intermediate to the outboard placement reverses the
timing response while the failure class remains junction-adjacent.  It does not
reuse the near/remote morphology comparison or introduce a mechanism claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image
from pypdf import PdfReader


FIGURE_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = FIGURE_DIR.parent
SOURCE = PACKAGE_DIR / "source_data" / "paired_panel.csv"
OUTPUT_DIR = FIGURE_DIR / "artwork" / "graphical_abstract"

FONT_FAMILY = "Arial"
FONT_PATH: Path | None = None
BLUE = "#247BA0"
ORANGE = "#D55E00"
INK = "#111111"
MUTED = "#667085"
GRID = "#D8DEE7"
MASTER_WIDTH_PX = 3000
MASTER_HEIGHT_PX = 1200
DISPLAY_WIDTH_PX = 500
DISPLAY_HEIGHT_PX = 200
PRINT_WIDTH_CM = 13.0
FIGURE_WIDTH_IN = PRINT_WIDTH_CM / 2.54
FIGURE_HEIGHT_IN = FIGURE_WIDTH_IN / 2.5
MASTER_DPI = MASTER_WIDTH_PX / FIGURE_WIDTH_IN


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable(path: Path) -> str:
    try:
        return str(path.relative_to(PACKAGE_DIR))
    except ValueError:
        return path.name


def _read_source() -> dict[str, list[dict[str, str]]]:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected: dict[str, list[dict[str, str]]] = {}
    for case in ("c26p5", "c34p5"):
        case_rows = sorted(
            (row for row in rows if row["case_id"] == case),
            key=lambda row: int(row["seed"]),
        )
        if len(case_rows) != 3:
            raise RuntimeError(f"expected three source rows for {case}")
        selected[case] = case_rows
    expected = {
        "c26p5": ([290.0, 270.0, 260.0], "junction_adjacent"),
        "c34p5": ([-110.0, -110.0, -100.0], "junction_adjacent"),
    }
    for case, (effects, failure_mode) in expected.items():
        actual = [float(row["delta_vs_paired_untreated"]) for row in selected[case]]
        if actual != effects:
            raise RuntimeError(f"unexpected paired effects for {case}: {actual}")
        if {row["failure_mode"] for row in selected[case]} != {failure_mode}:
            raise RuntimeError(f"unexpected failure class for {case}")
    return selected


def _build_figure(data: dict[str, list[dict[str, str]]]) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.sans-serif": [FONT_FAMILY],
            "font.size": 8.2,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    figure = plt.figure(
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        constrained_layout=False,
        facecolor="white",
    )
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.06, 0.72, 1.06),
        left=0.018,
        right=0.982,
        bottom=0.06,
        top=0.94,
        wspace=0.055,
    )
    left = figure.add_subplot(grid[0, 0])
    middle = figure.add_subplot(grid[0, 1])
    right = figure.add_subplot(grid[0, 2])

    for axis, case, color, title, center in (
        (left, "c26p5", BLUE, "INTERMEDIATE", "26.5"),
        (right, "c34p5", ORANGE, "OUTBOARD", "34.5"),
    ):
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.axis("off")
        axis.add_patch(
            Rectangle(
                (0.015, 0.02),
                0.97,
                0.96,
                transform=axis.transAxes,
                facecolor="#F8FAFC",
                edgecolor=color,
                linewidth=1.25,
            )
        )
        axis.text(
            0.5,
            0.84,
            title,
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=9.0,
            fontweight="bold",
            color=color,
        )
        axis.text(
            0.5,
            0.70,
            rf"$s_c={center}$",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=8.3,
            color=INK,
        )
        values = [int(float(row["delta_vs_paired_untreated"])) for row in data[case]]
        value_text = "+260 to +290" if case == "c26p5" else "-100 to -110"
        axis.text(
            0.5,
            0.50,
            value_text,
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=12.0,
            fontweight="bold",
            color=color,
        )
        axis.text(
            0.5,
            0.36,
            r"paired shift, $\Delta T$",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=8.0,
            color=MUTED,
        )
        axis.text(
            0.5,
            0.21,
            "DELAYED" if case == "c26p5" else "ADVANCED",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color=color,
        )
        axis.text(
            0.5,
            0.095,
            "junction-adjacent failure",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=8.0,
            color=INK,
        )
        expected_values = (
            [290, 270, 260] if case == "c26p5" else [-110, -110, -100]
        )
        if values != expected_values:
            raise RuntimeError(f"unexpected plotted values for {case}: {values}")

    middle.axis("off")
    middle.text(
        0.5,
        0.82,
        "SAME FOUR\nCOLLARS",
        transform=middle.transAxes,
        ha="center",
        va="center",
        fontsize=8.2,
        linespacing=0.95,
        fontweight="bold",
        color=INK,
    )
    middle.plot(
        [0.12, 0.88],
        [0.53, 0.53],
        transform=middle.transAxes,
        color="#8B95A4",
        lw=2.0,
        solid_capstyle="round",
    )
    middle.add_patch(
        Rectangle(
            (0.15, 0.43), 0.20, 0.20,
            transform=middle.transAxes, fill=False, ec=BLUE,
            lw=1.2, ls=(0, (3, 1.5)),
        )
    )
    middle.add_patch(
        Rectangle(
            (0.65, 0.43), 0.20, 0.20,
            transform=middle.transAxes, fill=False, ec=ORANGE,
            lw=1.2, ls=(0, (3, 1.5)),
        )
    )
    middle.add_patch(
        FancyArrowPatch(
            (0.38, 0.53), (0.62, 0.53),
            transform=middle.transAxes, arrowstyle="-|>",
            mutation_scale=10.0, lw=1.15, color=INK,
        )
    )
    middle.text(
        0.5,
        0.31,
        "SHIFT EACH\nCOLLAR OUTWARD",
        transform=middle.transAxes,
        ha="center",
        va="center",
        fontsize=8.0,
        linespacing=0.95,
        fontweight="bold",
        color=INK,
    )
    middle.text(
        0.5,
        0.16,
        "8 model units",
        transform=middle.transAxes,
        ha="center",
        va="center",
        fontsize=8.0,
        color=MUTED,
    )

    middle.text(
        0.5,
        0.06,
        "RESPONSE REVERSES",
        transform=middle.transAxes,
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color=INK,
    )
    return figure


def generate(output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    global FONT_PATH
    try:
        FONT_PATH = Path(font_manager.findfont(FONT_FAMILY, fallback_to_default=False))
    except ValueError:
        FONT_PATH = None
    if FONT_PATH is None or FONT_PATH.name.lower() not in {"arial.ttf", "arialmt.ttf"}:
        raise RuntimeError(f"Arial did not resolve to an Arial font file: {FONT_PATH}")
    data = _read_source()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = _build_figure(data)
    paths = {
        "pdf": output_dir / "graphical_abstract.pdf",
        "png": output_dir / "graphical_abstract.png",
        "svg": output_dir / "graphical_abstract.svg",
    }
    figure.savefig(
        paths["pdf"],
        metadata={"CreationDate": None, "Creator": None, "Producer": None},
    )
    figure.savefig(paths["png"], dpi=MASTER_DPI, metadata={"Software": None})
    figure.savefig(
        paths["svg"],
        metadata={"Date": None, "Creator": None},
    )
    plt.close(figure)

    with Image.open(paths["png"]) as master:
        if master.size != (MASTER_WIDTH_PX, MASTER_HEIGHT_PX):
            raise RuntimeError(f"unexpected master size: {master.size}")
        display = master.resize((DISPLAY_WIDTH_PX, DISPLAY_HEIGHT_PX), Image.Resampling.LANCZOS)
        display_path = output_dir / "graphical_abstract_500x200.png"
        display.save(display_path, dpi=(100, 100), pnginfo=None)
        print_width = round(PRINT_WIDTH_CM / 2.54 * 96)
        print_height = round(print_width / 2.5)
        print_preview = master.resize((print_width, print_height), Image.Resampling.LANCZOS)
        print_path = output_dir / "graphical_abstract_13cm_96dpi.png"
        print_preview.save(print_path, dpi=(96, 96), pnginfo=None)

    page = PdfReader(str(paths["pdf"])).pages[0]
    pdf_size_cm = [
        float(page.mediabox.width) / 72.0 * 2.54,
        float(page.mediabox.height) / 72.0 * 2.54,
    ]
    if abs(pdf_size_cm[0] - PRINT_WIDTH_CM) > 0.01:
        raise RuntimeError(f"unexpected PDF width: {pdf_size_cm[0]:.3f} cm")
    outputs = []
    for path in [*paths.values(), display_path, print_path]:
        outputs.append(
            {"path": _portable(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
        )
    manifest = {
        "schema_version": 1,
        "generator": _portable(Path(__file__).resolve()),
        "scope": "same-failure-class intermediate-versus-outboard graphical abstract",
        "source_csv": _portable(SOURCE),
        "source_csv_sha256": _sha256(SOURCE),
        "science_values_changed": False,
        "cases": {
            "intermediate": {
                "center": 26.5,
                "effects": [290, 270, 260],
                "failure_class": "junction_adjacent",
            },
            "outboard": {
                "center": 34.5,
                "effects": [-110, -110, -100],
                "failure_class": "junction_adjacent",
            },
        },
        "rigid_translation_claim_present": False,
        "near_remote_comparison_present": False,
        "font_family": FONT_FAMILY,
        "master_pixel_size": [MASTER_WIDTH_PX, MASTER_HEIGHT_PX],
        "science_direct_preview_pixel_size": [DISPLAY_WIDTH_PX, DISPLAY_HEIGHT_PX],
        "print_preview_pixel_size_at_96dpi": [print_width, print_height],
        "pdf_physical_size_cm": pdf_size_cm,
        "contains_graphical_abstract_heading": False,
        "outputs": outputs,
    }
    (output_dir / "graphical_abstract_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()
    print(json.dumps(generate(arguments.output_dir), indent=2))


if __name__ == "__main__":
    main()
