"""Shared data, style, validation, and export helpers for the figure set."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pypdf import PdfReader


FIGURE_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = FIGURE_DIR.parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[2]
SOURCE_DIR = PACKAGE_DIR / "source_data"

POSITION_SOURCE = SOURCE_DIR / "position_response.csv"
PAIRED_SOURCE = SOURCE_DIR / "paired_panel.csv"
TRANSPORT_SOURCE = SOURCE_DIR / "transport_response.csv"
BUDGET_SOURCE = SOURCE_DIR / "equal_budget_controls.csv"
TIMESTEP_SOURCE = SOURCE_DIR / "timestep_refinement.csv"
SPATIAL_SOURCE = SOURCE_DIR / "spatial_refinement.csv"
CONTRAST_SOURCE = SOURCE_DIR / "mobility_contrast.csv"
MORPHOLOGY_SOURCE = SOURCE_DIR / "morphology_event_projections.npz"
MORPHOLOGY_MANIFEST = SOURCE_DIR / "morphology_event_projections.json"

MM_PER_INCH = 25.4
POINTS_PER_INCH = 72.0
COMPACT_WIDTH_MM = 137.0
FULL_WIDTH_MM = 190.0
MIN_TEXT_PT = 9.4
PANEL_TEXT_PT = 12.0
MIN_MARKER_PT = 7.2
PRIMARY_MARKER_PT = 8.6
EXPORT_DPI = 600

COLORS = {
    "ink": "#202124",
    "muted": "#667085",
    "grid": "#D9DEE5",
    "untreated": "#4B5563",
    "near": "#009E73",
    "intermediate": "#0072B2",
    "outboard": "#D55E00",
    "far": "#7A7A7A",
    "transverse": "#E69F00",
    "local_band": "#F4D8C6",
    "phase": "#3E6B8A",
}
CASE_COLORS = {
    "untreated": COLORS["untreated"],
    "c18p5": COLORS["near"],
    "c26p5": COLORS["intermediate"],
    "c34p5": COLORS["outboard"],
    "c38p5": "#CC79A7",
    "c64p5": COLORS["far"],
}
SOURCE_MARKERS = {"2292": "o", "104729": "s", "130363": "^"}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.4,
            "axes.labelsize": 10.8,
            "axes.titlesize": 10.6,
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def number(row: dict[str, str], key: str) -> float:
    if row[key] == "":
        raise ValueError(f"missing {key!r}: {row}")
    return float(row[key])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def portable(path: Path) -> str:
    resolved = path.resolve()
    for root in (REPOSITORY_ROOT.resolve(), PACKAGE_DIR.resolve()):
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            pass
    return path.name


def clean_axis(axis: plt.Axes, grid: str | None = None) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid:
        axis.grid(axis=grid, color=COLORS["grid"], linewidth=0.6, zorder=0)
        axis.set_axisbelow(True)


def panel_label(
    axis: plt.Axes,
    label: str,
    *,
    x: float,
    y: float = 1.02,
) -> plt.Text:
    text = axis.text(
        x,
        y,
        f"({label})",
        transform=axis.transAxes,
        fontsize=PANEL_TEXT_PT,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=COLORS["ink"],
        clip_on=False,
        zorder=30,
    )
    text.set_gid(f"panel-label-{label}")
    return text


def figure_panel_label(
    figure: plt.Figure,
    x: float,
    y: float,
    label: str,
) -> plt.Text:
    text = figure.text(
        x,
        y,
        f"({label})",
        ha="left",
        va="center",
        fontsize=PANEL_TEXT_PT,
        fontweight="bold",
        color="#111111",
        zorder=50,
    )
    text.set_gid(f"panel-label-{label}")
    return text


def mask_profile(
    coordinate: np.ndarray,
    center: float,
    width: float = 12.0,
    transition: float = 3.0,
    minimum: float = 0.1,
) -> np.ndarray:
    inner = center - width / 2.0
    outer = center + width / 2.0
    coverage = np.zeros_like(coordinate, dtype=np.float64)
    rising = (coordinate >= inner) & (coordinate < inner + transition)
    coverage[rising] = 0.5 * (
        1.0 - np.cos(np.pi * (coordinate[rising] - inner) / transition)
    )
    plateau = (coordinate >= inner + transition) & (
        coordinate <= outer - transition
    )
    coverage[plateau] = 1.0
    falling = (coordinate > outer - transition) & (coordinate <= outer)
    coverage[falling] = 0.5 * (
        1.0
        + np.cos(
            np.pi
            * (coordinate[falling] - (outer - transition))
            / transition
        )
    )
    return 1.0 - (1.0 - minimum) * coverage


def paired_interval(
    treated: dict[str, str], untreated: dict[str, str]
) -> tuple[float, float]:
    return (
        number(treated, "event_lower") - number(untreated, "event_upper"),
        number(treated, "event_upper") - number(untreated, "event_lower"),
    )


def physical_record(width_mm: float, height_mm: float) -> dict[str, float | bool]:
    return {
        "width_mm": width_mm,
        "height_mm": height_mm,
        "native_scale_factor": 1.0,
        "minimum_semantic_text_pt": MIN_TEXT_PT,
        "minimum_data_marker_pt": MIN_MARKER_PT,
        "minimum_data_marker_mm": MIN_MARKER_PT * MM_PER_INCH / POINTS_PER_INCH,
        "pdf_mediabox_asserted": True,
    }


def source_paths() -> list[Path]:
    return [
        POSITION_SOURCE,
        PAIRED_SOURCE,
        TRANSPORT_SOURCE,
        BUDGET_SOURCE,
        TIMESTEP_SOURCE,
        SPATIAL_SOURCE,
        MORPHOLOGY_SOURCE,
        MORPHOLOGY_MANIFEST,
    ]


def load_core_sources() -> dict[str, Any]:
    return {
        "position": read_csv(POSITION_SOURCE),
        "paired": read_csv(PAIRED_SOURCE),
        "transport": read_csv(TRANSPORT_SOURCE),
        "budget": read_csv(BUDGET_SOURCE),
        "timestep": read_csv(TIMESTEP_SOURCE),
        "spatial": read_csv(SPATIAL_SOURCE),
        "morphology_manifest": read_json(MORPHOLOGY_MANIFEST),
    }


def validate_sources(data: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on the numerical values used by Figures 1--3."""

    failures: list[str] = []
    position = {row["case_id"]: row for row in data["position"]}
    expected_position = {
        "untreated": (1690, 1700, 1695, 0, 19.25),
        "c14p5": (2540, 2550, 2545, 850, 35.5),
        "c18p5": (2440, 2450, 2445, 750, 39.5),
        "c22p5": (2370, 2380, 2375, 680, 43.0),
        "c26p5": (1980, 1990, 1985, 290, 17.5),
        "c30p5": (1680, 1690, 1685, -10, 18.0),
        "c34p5": (1580, 1590, 1585, -110, 18.75),
        "c38p5": (1590, 1600, 1595, -100, 19.25),
        "c64p5": (1690, 1700, 1695, 0, 19.25),
    }
    if set(position) != set(expected_position):
        failures.append("position case set")
    else:
        for case, expected in expected_position.items():
            row = position[case]
            observed = (
                int(row["event_lower"]),
                int(row["event_upper"]),
                int(float(row["event_midpoint"])),
                int(float(row["delta_vs_untreated"])),
                float(row["first_site_abs"]),
            )
            if observed != expected:
                failures.append(f"position values {case}: {observed}")

    paired = {(row["seed"], row["case_id"]): row for row in data["paired"]}
    expected_effects = {
        "c18p5": [750.0, 760.0, 770.0],
        "c26p5": [290.0, 270.0, 260.0],
        "c34p5": [-110.0, -110.0, -100.0],
    }
    expected_keys = {
        (seed, case)
        for seed in SOURCE_MARKERS
        for case in ("untreated", "c18p5", "c26p5", "c34p5")
    }
    if set(paired) != expected_keys:
        failures.append("paired key set")
    else:
        for case, expected in expected_effects.items():
            observed = [
                number(paired[(seed, case)], "delta_vs_paired_untreated")
                for seed in SOURCE_MARKERS
            ]
            if observed != expected:
                failures.append(f"paired effects {case}: {observed}")

    transport = {row["case_id"]: row for row in data["transport"]}
    if set(transport) != {"untreated", "c26p5", "c34p5", "c38p5", "c64p5"}:
        failures.append("transport case set")
    else:
        for case, row in transport.items():
            components = number(row, "axial_rate") + number(row, "transverse_rate")
            if not np.isclose(number(row, "predicted_volume_rate"), components, atol=1e-14):
                failures.append(f"transport component sum {case}")
            if number(row, "max_corrected_relative_error") >= 0.01:
                failures.append(f"transport closure {case}")

    spatial = {
        (row["grid_level"], row["case_id"]): row for row in data["spatial"]
    }
    fine = spatial.get(("F", "c34p5"), {})
    fine_complete = (
        fine.get("status") == "completed"
        and fine.get("passed") == "true"
        and fine.get("classification") == "spatial_refinement_validation_passed"
        and fine.get("all_frozen_checks_passed") == "true"
        and float(fine.get("paired_effect_c34_minus_untreated", "nan")) == -100.0
    )
    if not fine_complete:
        failures.append("fine-grid validation")

    morphology_manifest = data["morphology_manifest"]
    if morphology_manifest.get("artifact_sha256") != sha256(MORPHOLOGY_SOURCE):
        failures.append("morphology artifact hash")
    expected_morphology = {
        "untreated": (1700, 19.25),
        "near": (2450, 39.5),
        "outboard": (1590, 18.75),
    }
    for case, (step, site) in expected_morphology.items():
        record = morphology_manifest["cases"][case]
        if int(record["step"]) != step or float(record["reported_event_site"]) != site:
            failures.append(f"morphology values {case}")
        if Path(record["source_path"]).is_absolute():
            failures.append(f"absolute morphology path {case}")

    return {
        "passed": not failures,
        "failures": failures,
        "independent_source_count": 3,
        "fine_grid_result_included": fine_complete,
        "science_validation_complete": fine_complete,
    }


def save_exact(
    figure: plt.Figure,
    stem: Path,
    expected_width_mm: float,
) -> list[dict[str, Any]]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    for suffix, kwargs in (
        (
            "pdf",
            {"metadata": {"CreationDate": None, "Creator": None, "Producer": None}},
        ),
        ("png", {"dpi": EXPORT_DPI, "metadata": {"Software": None}}),
        ("svg", {"metadata": {"Date": None, "Creator": None}}),
    ):
        path = stem.with_suffix(f".{suffix}")
        figure.savefig(path, **kwargs)
        if suffix == "svg":
            normalized = "\n".join(
                line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()
            )
            path.write_text(normalized + "\n", encoding="utf-8")
        outputs.append(
            {"path": portable(path), "sha256": sha256(path), "bytes": path.stat().st_size}
        )
    page = PdfReader(str(stem.with_suffix(".pdf"))).pages[0]
    width_mm = float(page.mediabox.width) / POINTS_PER_INCH * MM_PER_INCH
    if abs(width_mm - expected_width_mm) > 0.03:
        raise RuntimeError(f"unexpected PDF width: {width_mm:.4f} mm")
    with Image.open(stem.with_suffix(".png")) as image:
        expected_px = round(expected_width_mm / MM_PER_INCH * EXPORT_DPI)
        if abs(image.width - expected_px) > 1:
            raise RuntimeError(f"unexpected PNG width: {image.width} px")
    plt.close(figure)
    return outputs
