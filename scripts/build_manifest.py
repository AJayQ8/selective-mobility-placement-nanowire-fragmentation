#!/usr/bin/env python3
"""Write the deterministic SHA-256 manifest for the public archive."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST.sha256"
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "reproduced_artifacts",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def included_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != OUTPUT
        and not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
        and path.name != ".DS_Store"
        and path.suffix not in {".pyc", ".pyo"}
    )


def render() -> str:
    return "".join(
        f"{digest(path)}  {path.relative_to(ROOT).as_posix()}\n"
        for path in included_files()
    )


def main() -> None:
    OUTPUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} with {len(included_files())} files")


if __name__ == "__main__":
    main()
