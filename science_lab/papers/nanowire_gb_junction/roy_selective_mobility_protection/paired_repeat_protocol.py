"""Frozen definitions for the two-realization paired repeat panel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import position_scan_protocol as position_protocol


DIRECTORY = Path(__file__).resolve().parent
RESULTS = DIRECTORY / "results"
DEFAULT_OUTPUT = RESULTS / "paired_repeat_panel_v1"
CONTRACT_PATH = DIRECTORY / "paired_repeat_contract.md"

NEW_SEEDS = (104729, 130363)
EXISTING_SEED = 2292
CASE_ORDER = ("untreated", "c18p5", "c26p5", "c34p5")
CASE_CENTERS = {
    "untreated": None,
    "c18p5": 18.5,
    "c26p5": 26.5,
    "c34p5": 34.5,
}
FFT_WORKERS = position_protocol.FFT_WORKERS
SOURCE_TARGET_STEP = position_protocol.START_STEP
TARGET_STEP = position_protocol.TARGET_STEP
DIAGNOSTIC_INTERVAL = position_protocol.DIAGNOSTIC_INTERVAL
ENERGY_INTERVAL = position_protocol.ENERGY_INTERVAL
MILESTONE_INTERVAL = position_protocol.MILESTONE_INTERVAL
CHECKPOINT_STEPS = position_protocol.CHECKPOINT_STEPS
MAXIMUM_EVENT_CHECKPOINTS = position_protocol.MAXIMUM_EVENT_CHECKPOINTS
MAXIMUM_WALL_SECONDS = position_protocol.MAXIMUM_WALL_SECONDS
RUNTIME_SAFETY_FACTOR = position_protocol.RUNTIME_SAFETY_FACTOR
MINIMUM_DISK_HEADROOM_BYTES = 8 << 30
MAXIMUM_SESSIONS_PER_CASE = 2


@dataclass(frozen=True)
class RepeatCase:
    """One nonadaptive member of the paired panel."""

    case_id: str

    def __post_init__(self) -> None:
        if self.case_id not in CASE_ORDER:
            raise ValueError(f"unknown paired-repeat case: {self.case_id}")

    @property
    def slug(self) -> str:
        return self.case_id

    @property
    def center(self) -> float | None:
        return CASE_CENTERS[self.case_id]

    @property
    def stream_name(self) -> str:
        # The verified case engine's crash-recovery quarantine recognizes this
        # stream name. Each case has its own directory, so reuse is unambiguous.
        return "position_scan"

    @property
    def position_case(self) -> position_protocol.PositionCase | None:
        if self.center is None:
            return None
        return position_protocol.CASES_BY_CENTER[self.center]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "center": self.center,
            "untreated": self.center is None,
            "stream_name": self.stream_name,
        }


CASES = tuple(RepeatCase(case_id) for case_id in CASE_ORDER)
CASES_BY_ID = {case.case_id: case for case in CASES}


def validate_seed(seed: int) -> int:
    value = int(seed)
    if value not in NEW_SEEDS:
        raise ValueError(f"seed must be one of {list(NEW_SEEDS)}")
    return value


def validate_case(case_id: str) -> RepeatCase:
    try:
        return CASES_BY_ID[case_id]
    except KeyError as error:
        raise ValueError(f"case must be one of {list(CASE_ORDER)}") from error


def seed_directory(root: Path, seed: int) -> Path:
    return root / f"seed-{validate_seed(seed)}"


def source_output(root: Path, seed: int) -> Path:
    return seed_directory(root, seed) / "source-t0100"


def case_output(root: Path, seed: int, case_id: str) -> Path:
    case = validate_case(case_id)
    return seed_directory(root, seed) / f"case-{case.case_id}"
