#!/usr/bin/env python3
"""Regenerate all artwork in a separate output directory."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from science_lab.papers.selective_mobility_scripta.figures import generate_all  # noqa: E402


OUTPUT = ROOT / "reproduced_artifacts" / "figures"


def main() -> None:
    receipt = generate_all.generate(OUTPUT)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
