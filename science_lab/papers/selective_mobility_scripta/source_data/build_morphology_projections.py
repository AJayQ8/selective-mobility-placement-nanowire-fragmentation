"""Build compact, hash-bound event projections from archived 3-D fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_NPZ = Path(__file__).resolve().parent / "morphology_event_projections.npz"
OUTPUT_JSON = Path(__file__).resolve().parent / "morphology_event_projections.json"

SPACING = 0.5
CROP_LIMIT = 52.0
THRESHOLD = 0.45

ARCHIVE_FIELDS: dict[str, dict[str, Any]] = {
    "untreated": {
        "label": "Untreated",
        "step": 1700,
        "event_site": 19.25,
        "path": Path(
            "science_lab/papers/nanowire_gb_junction/roy_2021_reproduction/"
            "results/t2000_continuation_source_semantic_seed2292/"
            "checkpoint-step-1700.npy"
        ),
        "sha256": "012f54db686f23577e0e66886438a15cbd16b0cfdeef3c4109b201a78c6ab331",
    },
    "near": {
        "label": "Near collar (18.5)",
        "step": 2450,
        "event_site": 39.5,
        "path": Path(
            "science_lab/papers/nanowire_gb_junction/"
            "roy_selective_mobility_protection/results/"
            "position_scan_c18p5_v1/checkpoint-position_scan-step-2450.npy"
        ),
        "sha256": "0b85554285f3190229ea6f87605c0c817c3615585985dfee064adfda8cea1490",
    },
    "outboard": {
        "label": "Outboard collar (34.5)",
        "step": 1590,
        "event_site": 18.75,
        "path": Path(
            "science_lab/papers/nanowire_gb_junction/"
            "roy_selective_mobility_protection/results/"
            "position_scan_c34p5_v1/checkpoint-position_scan-step-1590.npy"
        ),
        "sha256": "c54aeb555315af55551a48da488d80fce547352efb2ff5122c15c2ecf7652c25",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coordinates(cells: int) -> np.ndarray:
    return (np.arange(cells, dtype=np.float64) - cells // 2) * SPACING


def _gap_runs(coordinate: np.ndarray, axial_maximum: np.ndarray) -> list[dict[str, float]]:
    indices = np.flatnonzero(axial_maximum < THRESHOLD)
    if indices.size == 0:
        return []
    splits = np.flatnonzero(np.diff(indices) > 1) + 1
    runs: list[dict[str, float]] = []
    for group in np.split(indices, splits):
        runs.append(
            {
                "lower": float(coordinate[group[0]] - 0.5 * SPACING),
                "upper": float(coordinate[group[-1]] + 0.5 * SPACING),
                "midpoint": float(0.5 * (coordinate[group[0]] + coordinate[group[-1]])),
            }
        )
    return runs


def build(archive_root: Path) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    records: dict[str, Any] = {}
    common_y: np.ndarray | None = None
    common_z: np.ndarray | None = None

    for case, definition in ARCHIVE_FIELDS.items():
        relative_path = definition["path"]
        if relative_path.is_absolute():
            raise RuntimeError(f"archive path must be relative: {relative_path}")
        path = archive_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        measured_sha = _sha256(path)
        if measured_sha != definition["sha256"]:
            raise RuntimeError(f"field hash mismatch for {case}: {measured_sha}")

        field = np.load(path, mmap_mode="r", allow_pickle=False)
        if field.shape != (96, 768, 768) or field.dtype != np.dtype("<f8"):
            raise RuntimeError(f"unexpected field contract for {case}: {field.shape}, {field.dtype}")
        y_full = _coordinates(field.shape[1])
        z_full = _coordinates(field.shape[2])
        y_indices = np.flatnonzero(np.abs(y_full) <= CROP_LIMIT)
        z_indices = np.flatnonzero(np.abs(z_full) <= CROP_LIMIT)
        y_slice = slice(int(y_indices[0]), int(y_indices[-1]) + 1)
        z_slice = slice(int(z_indices[0]), int(z_indices[-1]) + 1)
        y = y_full[y_slice]
        z = z_full[z_slice]
        projection = np.max(field[:, y_slice, z_slice], axis=0).astype(np.float32)
        axial_maximum = np.max(projection, axis=0).astype(np.float32)

        if common_y is None:
            common_y = y
            common_z = z
        elif not np.array_equal(common_y, y) or not np.array_equal(common_z, z):
            raise RuntimeError("projection coordinate mismatch")

        arrays[f"projection_{case}"] = projection
        arrays[f"axial_maximum_{case}"] = axial_maximum
        records[case] = {
            "label": definition["label"],
            "step": definition["step"],
            "reported_event_site": definition["event_site"],
            "source_path": str(relative_path),
            "source_sha256": measured_sha,
            "projection_shape": list(projection.shape),
            "projection_minimum": float(projection.min()),
            "projection_maximum": float(projection.max()),
            "threshold_gap_runs_in_crop": _gap_runs(z, axial_maximum),
        }

    assert common_y is not None and common_z is not None
    arrays["y"] = common_y
    arrays["z"] = common_z
    np.savez_compressed(OUTPUT_NPZ, **arrays)
    manifest = {
        "schema_version": 1,
        "artifact": str(OUTPUT_NPZ.relative_to(PACKAGE_DIR)),
        "artifact_sha256": _sha256(OUTPUT_NPZ),
        "construction": "maximum concentration over the short x axis on the integer-centered grid",
        "spacing": SPACING,
        "crop_abs_limit": CROP_LIMIT,
        "event_threshold": THRESHOLD,
        "cases": records,
    }
    OUTPUT_JSON.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root",
        type=Path,
        required=True,
        help="root of the separately archived nanowire evidence tree",
    )
    arguments = parser.parse_args(argv)
    print(json.dumps(build(arguments.archive_root.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
