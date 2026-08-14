"""Frozen definitions for the seed-104729 paired dt=0.5 validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


DIRECTORY = Path(__file__).resolve().parent
RESULTS = DIRECTORY / "results"
CONTRACT_PATH = DIRECTORY / "TIMESTEP_REFINEMENT_CONTRACT.md"
PREFLIGHT_OUTPUT = RESULTS / "timestep_refinement_preflight_dt0p5_v1"
DEFAULT_OUTPUT = RESULTS / "timestep_refinement_pair_dt0p5_v1"

SEED = 104729
CASE_ORDER = ("untreated", "c34p5")
PHYSICAL_TIMESTEP = 0.5
SOURCE_PHYSICAL_TIME = 100.0
TARGET_PHYSICAL_TIME = 2000.0
DIAGNOSTIC_PHYSICAL_INTERVAL = 10.0
ENERGY_PHYSICAL_INTERVAL = 50.0
REGULAR_CHECKPOINT_PHYSICAL_TIMES = (500.0, 900.0, 1300.0)
MAXIMUM_EVENT_CHECKPOINTS = 3
FFT_WORKERS = 12
RUNTIME_SAFETY_FACTOR = 1.2
MAXIMUM_CASE_WALL_SECONDS = 4.0 * 60.0 * 60.0
MINIMUM_DISK_HEADROOM_BYTES = 8 << 30

SOURCE_DIRECTORY = (
    RESULTS
    / "paired_repeat_panel_v1"
    / "seed-104729"
    / "source-t0100"
)
SOURCE_SUMMARY = SOURCE_DIRECTORY / "summary.json"
SOURCE_METADATA = (
    SOURCE_DIRECTORY / "checkpoint-paired_source-step-0100.json"
)
SOURCE_FIELD = (
    SOURCE_DIRECTORY / "checkpoint-paired_source-step-0100.npy"
)
SOURCE_SUMMARY_SHA256 = (
    "80ba42da99e09e8c7b3f385f90bac752436df463cb5afc85e2d2b0e61e7c54ee"
)
SOURCE_METADATA_SHA256 = (
    "762256c5cc66083c5fae871e814e120a1f50a1cc8cb9d698ff44cc59771915ae"
)
SOURCE_FIELD_SHA256 = (
    "dbee74afc8a04568f72cb25f40a84d6f3bc939fc1b55802ac7aabac17c6f720b"
)
SOURCE_FINGERPRINT = (
    "6cb149fc5d9ca697c2ddcee191f1320a81d9fce8ad98f807c108fb07cefde918"
)
SOURCE_INITIAL_MASS = 83903.9453973702
SOURCE_FIELD_BYTES = 452_984_960

DT1_SUMMARIES = {
    "untreated": (
        RESULTS
        / "paired_repeat_panel_v1"
        / "seed-104729"
        / "case-untreated"
        / "summary.json"
    ),
    "c34p5": (
        RESULTS
        / "paired_repeat_panel_v1"
        / "seed-104729"
        / "case-c34p5"
        / "summary.json"
    ),
}
DT1_SUMMARY_SHA256 = {
    "untreated": (
        "0808c2307328b2d1d3b3af3dc5934283159b3ef1b789a7e972c335620c75f16f"
    ),
    "c34p5": (
        "3be497fe9efe9e4fe1bca0144da2e05030d7e5e5406e557a6054c119f1d79670"
    ),
}
DT1_EVENT_BRACKETS = {
    "untreated": (1680.0, 1690.0),
    "c34p5": (1570.0, 1580.0),
}
DT1_EVENT_MIDPOINTS = {
    case: 0.5 * (bracket[0] + bracket[1])
    for case, bracket in DT1_EVENT_BRACKETS.items()
}
DT1_PAIRED_EFFECT = (
    DT1_EVENT_MIDPOINTS["c34p5"]
    - DT1_EVENT_MIDPOINTS["untreated"]
)
MAXIMUM_INDIVIDUAL_RELATIVE_TIME_ERROR = 0.05
MAXIMUM_PAIRED_EFFECT_RELATIVE_ERROR = 0.25
NATURAL_SITE_ABSOLUTE_BOUNDS = (14.0, 22.5)
NATURAL_WIRE = "first_wire_z"

# The measured dt=1 terminal confirmations imply these planning proposal
# counts after doubling for dt=0.5. They do not stop either trajectory.
PLANNING_CONFIRMATION_PHYSICAL_TIMES = {
    "untreated": 1710.0,
    "c34p5": 1600.0,
}
REFERENCE_MEDIAN_PROPOSAL_SECONDS = {
    "untreated": 2.9476353749923874,
    "c34p5": 2.926448416983476,
}
PROJECTED_CHECKPOINTS_PER_CASE = (
    len(REGULAR_CHECKPOINT_PHYSICAL_TIMES)
    + MAXIMUM_EVENT_CHECKPOINTS
)


def proposals_for_elapsed_physical_time(elapsed: float) -> int:
    """Convert an exactly representable physical duration to proposals."""

    raw = float(elapsed) / PHYSICAL_TIMESTEP
    rounded = round(raw)
    if abs(raw - rounded) > 1.0e-12:
        raise ValueError(
            f"duration {elapsed} is not aligned to dt={PHYSICAL_TIMESTEP}"
        )
    return int(rounded)


def physical_time_for_proposal_count(proposal_count: int) -> float:
    """Map continuation proposal count to global physical time."""

    count = int(proposal_count)
    if count < 0:
        raise ValueError("proposal count must be nonnegative")
    return SOURCE_PHYSICAL_TIME + count * PHYSICAL_TIMESTEP


DIAGNOSTIC_PROPOSAL_INTERVAL = proposals_for_elapsed_physical_time(
    DIAGNOSTIC_PHYSICAL_INTERVAL
)
ENERGY_PROPOSAL_INTERVAL = proposals_for_elapsed_physical_time(
    ENERGY_PHYSICAL_INTERVAL
)
TARGET_PROPOSAL_COUNT = proposals_for_elapsed_physical_time(
    TARGET_PHYSICAL_TIME - SOURCE_PHYSICAL_TIME
)
REGULAR_CHECKPOINT_PROPOSAL_COUNTS = tuple(
    proposals_for_elapsed_physical_time(time - SOURCE_PHYSICAL_TIME)
    for time in REGULAR_CHECKPOINT_PHYSICAL_TIMES
)


@dataclass(frozen=True)
class RefinementCase:
    """One member of the fixed paired validation."""

    case_id: str

    def __post_init__(self) -> None:
        if self.case_id not in CASE_ORDER:
            raise ValueError(f"unknown refinement case: {self.case_id}")

    @property
    def untreated(self) -> bool:
        return self.case_id == "untreated"

    @property
    def center(self) -> float | None:
        return None if self.untreated else 34.5

    @property
    def output_name(self) -> str:
        return f"case-{self.case_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "untreated": self.untreated,
            "center": self.center,
        }


CASES = tuple(RefinementCase(case_id) for case_id in CASE_ORDER)
CASES_BY_ID = {case.case_id: case for case in CASES}


def validate_case(case_id: str) -> RefinementCase:
    try:
        return CASES_BY_ID[case_id]
    except KeyError as error:
        raise ValueError(f"case must be one of {list(CASE_ORDER)}") from error


def case_output(root: Path, case_id: str) -> Path:
    return root / validate_case(case_id).output_name


def runtime_projection() -> dict[str, Any]:
    """Return the preregistered measured-throughput planning estimate."""

    cases: dict[str, Any] = {}
    total_seconds = 0.0
    total_proposals = 0
    for case_id in CASE_ORDER:
        terminal = PLANNING_CONFIRMATION_PHYSICAL_TIMES[case_id]
        proposals = proposals_for_elapsed_physical_time(
            terminal - SOURCE_PHYSICAL_TIME
        )
        seconds = (
            proposals * REFERENCE_MEDIAN_PROPOSAL_SECONDS[case_id]
        )
        total_seconds += seconds
        total_proposals += proposals
        cases[case_id] = {
            "planning_confirmation_physical_time": terminal,
            "projected_proposals": proposals,
            "reference_median_proposal_seconds": (
                REFERENCE_MEDIAN_PROPOSAL_SECONDS[case_id]
            ),
            "unsafeguarded_wall_seconds": seconds,
        }
    safeguarded = total_seconds * RUNTIME_SAFETY_FACTOR
    return {
        "cases": cases,
        "sequential": True,
        "total_projected_proposals": total_proposals,
        "unsafeguarded_wall_seconds": total_seconds,
        "unsafeguarded_wall_hours": total_seconds / 3600.0,
        "safety_factor": RUNTIME_SAFETY_FACTOR,
        "safeguarded_wall_seconds": safeguarded,
        "safeguarded_wall_hours": safeguarded / 3600.0,
    }


def storage_projection() -> dict[str, Any]:
    """Return the twelve-field scientific-output projection."""

    checkpoint_count = PROJECTED_CHECKPOINTS_PER_CASE * len(CASE_ORDER)
    field_bytes = checkpoint_count * SOURCE_FIELD_BYTES
    return {
        "sequential": True,
        "projected_checkpoint_count": checkpoint_count,
        "projected_field_bytes": field_bytes,
        "projected_field_gibibytes": field_bytes / float(1 << 30),
        "minimum_disk_headroom_bytes": MINIMUM_DISK_HEADROOM_BYTES,
        "required_free_disk_bytes": (
            field_bytes + MINIMUM_DISK_HEADROOM_BYTES
        ),
        "note": (
            "metadata and compact JSON are negligible; an interruption "
            "checkpoint is contingency output above the 5.1 GiB projection"
        ),
    }
