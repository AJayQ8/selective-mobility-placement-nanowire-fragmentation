"""Regenerate every figure from the normalized source package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from . import (
        figure1,
        figure2,
        figure3,
        figure4,
        figure5,
        graphical_abstract,
        supporting_figures,
    )
except ImportError:  # Direct execution from this directory.
    import figure1
    import figure2
    import figure3
    import figure4
    import figure5
    import graphical_abstract
    import supporting_figures


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "artwork"


def generate(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    manifests = {
        "figure1": figure1.generate(output_dir / "main"),
        "figure2": figure2.generate(output_dir / "main"),
        "figure3": figure3.generate(output_dir / "main"),
        "figure4": figure4.generate(output_dir / "main"),
        "figure5": figure5.generate(output_dir / "main"),
        "supporting": supporting_figures.generate(output_dir / "supporting"),
    }
    graphical_status: dict[str, Any]
    try:
        graphical_status = graphical_abstract.generate(output_dir / "graphical_abstract")
    except RuntimeError as error:
        if "Arial did not resolve" not in str(error):
            raise
        graphical_status = {"generated": False, "reason": str(error)}
    manifests["graphical_abstract"] = graphical_status
    receipt = {
        "schema_version": 1,
        "output_directory": output_dir.as_posix(),
        "generated": {
            name: data.get("generated", True) for name, data in manifests.items()
        },
    }
    (output_dir / "generation_receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
