"""Immutable provenance checks for the Roy baseline used by this bridge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DIRECTORY = Path(__file__).resolve().parent
MANIFEST_PATH = DIRECTORY / "baseline_manifest.json"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported baseline manifest schema")
    if manifest.get("immutable_inputs") is not True:
        raise RuntimeError("baseline manifest must declare immutable inputs")
    return manifest


def baseline_directory(manifest: dict[str, Any] | None = None) -> Path:
    frozen = load_manifest() if manifest is None else manifest
    candidate = (DIRECTORY / frozen["baseline_directory"]).resolve()
    expected = (DIRECTORY.parent / "roy_2021_reproduction").resolve()
    if candidate != expected:
        raise RuntimeError("baseline path escaped the frozen sibling directory")
    return candidate


def verify_frozen_baseline() -> dict[str, Any]:
    """Verify every pinned source file before bridge code touches the model."""

    manifest = load_manifest()
    root = baseline_directory(manifest)
    measured: dict[str, str] = {}
    mismatches: dict[str, dict[str, str]] = {}
    for relative_name, expected_hash in manifest["code"].items():
        path = root / relative_name
        actual_hash = sha256_path(path)
        measured[relative_name] = actual_hash
        if actual_hash != expected_hash:
            mismatches[relative_name] = {
                "expected": expected_hash,
                "actual": actual_hash,
            }
    if mismatches:
        raise RuntimeError(
            "frozen Roy source mismatch: " + json.dumps(mismatches, sort_keys=True)
        )
    return {
        "baseline_id": manifest["baseline_id"],
        "manifest_sha256": sha256_path(MANIFEST_PATH),
        "source_sha256": measured,
        "verified": True,
    }

