"""Shared constants and small helpers for the frozen m=0.3 campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).resolve().parent
PAPER_ROOT = PACKAGE.parent
CONTRACT_PATH = PACKAGE / "frozen_contract.json"
PARENT_REPEAT_CONTRACT_PATH = (
    PAPER_ROOT.parent
    / "nanowire_gb_junction"
    / "roy_selective_mobility_protection"
    / "paired_repeat_contract.md"
)
PARENT_REPEAT_CONTRACT_ORIGINAL_SHA256 = (
    "60c9b8023b124e2a3e809e7cd595c3fc2c01b2b2b8d14e40d72d31e8074ec495"
)
PARENT_REPEAT_CONTRACT_PUBLIC_SHA256 = (
    "447c7dd92bd7022337a1a3c166dda854dd88ab7d08f156cab9086ae337ce5f7c"
)
PUBLIC_PARENT_IMPLEMENTATION_SHA256 = {
    "paired_case_runner": "e3d6824eb972807cabbdf25a51f63b2d929b22c88336085c12f10024c58e4cdc",
    "case_engine": "25f9402da479f2315e3fb3ba96f0e63e9911c4937ba54d90b5356a2bcc19ed2f",
    "paired_protocol": "478d1f3ba9439fe4156e347bdfd505383043aa91805b4d8bd93f51a7bed42132",
    "paired_source_runner": "a5477dc8c6ed912484cf9e0fb45e9094abd50add82673d16d134a1bd9780bdb8",
}
RAW_ROOT = PACKAGE / "raw_results" / "m03_bc_v1"
RESULTS_ROOT = PACKAGE / "results" / "m03_bc_v1"
PREFLIGHT_PATH = RESULTS_ROOT / "preflight.json"
CAMPAIGN_STATUS_PATH = RESULTS_ROOT / "campaign_status.json"
ANALYSIS_PATH = RESULTS_ROOT / "analysis.json"

CAMPAIGN_ID = "m03_bc_position_sign_validation_v1"
FROZEN_IMPLEMENTATION_SNAPSHOT = {
    "file_count": 16,
    "sha256": "99fce3a3be5ac69f902534f072c09680ee7d2b5bd9e0fc6fef102d825346fd0b",
}
PROTECTED_FACTOR = 0.3
SEEDS = (104729, 130363)
CASE_IDS = ("c26p5", "c34p5")
CASE_ORDER = tuple((seed, case_id) for seed in SEEDS for case_id in CASE_IDS)

# A field has 96*768*768 float64 entries plus a conservative .npy header.
FIELD_BYTES_UPPER_BOUND = 96 * 768 * 768 * 8 + 256
MAX_REGULAR_CHECKPOINTS_PER_CASE = 10
MAX_EVENT_CHECKPOINTS_PER_CASE = 3
MAX_SIGNAL_OR_WALL_CHECKPOINTS_PER_CASE = 2
MAX_CHECKPOINTS_PER_CASE = (
    MAX_REGULAR_CHECKPOINTS_PER_CASE
    + MAX_EVENT_CHECKPOINTS_PER_CASE
    + MAX_SIGNAL_OR_WALL_CHECKPOINTS_PER_CASE
)
NON_FIELD_OUTPUT_ALLOWANCE_PER_CASE = 128 << 20
PERSISTENT_OUTPUT_UPPER_BOUND = (
    len(CASE_ORDER) * MAX_CHECKPOINTS_PER_CASE * FIELD_BYTES_UPPER_BOUND
    + len(CASE_ORDER) * NON_FIELD_OUTPUT_ALLOWANCE_PER_CASE
)
LARGEST_ATOMIC_WRITE = FIELD_BYTES_UPPER_BOUND
UNTOUCHED_RESERVE = 10 << 30
PEAK_RAM_UPPER_BOUND = 32 << 30
RUNTIME_UPPER_BOUND_SECONDS = 8 * 3600


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_contract() -> dict[str, Any]:
    contract = load_json(CONTRACT_PATH)
    if contract.get("campaign_id") != CAMPAIGN_ID:
        raise RuntimeError("unexpected mobility-contrast campaign contract")
    if contract.get("status") != "frozen_before_simulation":
        raise RuntimeError("campaign contract was not frozen before simulation")
    return contract


def raw_case_output(seed: int, case_id: str) -> Path:
    if seed not in SEEDS or case_id not in CASE_IDS:
        raise ValueError("case is outside the frozen campaign")
    return RAW_ROOT / f"seed-{seed}" / f"case-{case_id}_m03"


def source_output(historical_root: Path, seed: int) -> Path:
    if seed not in SEEDS:
        raise ValueError("source is outside the frozen campaign")
    return historical_root / f"seed-{seed}" / "source-t0100"
