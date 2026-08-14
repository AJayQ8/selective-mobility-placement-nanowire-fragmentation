"""Frozen definitions shared by the mobility-collar position scan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..roy_2021_reproduction.model import FROZEN_DEG90
from .geometry import FourArmCollarGeometry, signed_arm_window


DIRECTORY = Path(__file__).resolve().parent
CONTRACT_PATH = DIRECTORY / "POSITION_SCAN_CONTRACT.md"
RESULTS = DIRECTORY / "results"

RADIUS = 6.0
INTERFACE_WIDTH = float(np.sqrt(8.0))
START_STEP = 100
TARGET_STEP = 3000
DIAGNOSTIC_INTERVAL = 10
ENERGY_INTERVAL = 50
MILESTONE_INTERVAL = 100
CHECKPOINT_STEPS = (
    500,
    900,
    1300,
    1700,
    2000,
    2200,
    2400,
    2600,
    2800,
    3000,
)
FFT_WORKERS = 12
MAXIMUM_WALL_SECONDS = 3.0 * 60.0 * 60.0
RUNTIME_SAFETY_FACTOR = 1.2
MINIMUM_DISK_HEADROOM_BYTES = 4 << 30
MAXIMUM_EVENT_CHECKPOINTS = 3
MAXIMUM_BUDGET_DIFFERENCE = 0.02

COLLAR_SUPPORT_WIDTH = 12.0
COLLAR_TRANSITION_WIDTH = 3.0
PROTECTED_MOBILITY_FACTOR = 0.1
REFERENCE_CENTER = 14.5
ANCHOR_CENTERS = (14.5, 22.5, 64.5)
NEW_CENTERS = (18.5, 26.5, 30.5, 34.5, 38.5)

DEPLETION_MINIMUM = 11.664937524146215
UNTREATED_EVENT_SITE = 19.25
UNTREATED_EVENT_MIDPOINT = 1715.0
CONDITIONAL_DOWNSTREAM_OFFSET = 14.75
MODE_COMPATIBILITY_TOLERANCE = INTERFACE_WIDTH
CENTRAL_EXCLUSION_DISTANCE = RADIUS + 2.0 * INTERFACE_WIDTH

# This is a planning forecast, not a scientific prediction.  It allows the
# preflight to distinguish the likely overnight duration from the strict
# all-cases-to-t3000 upper bound.
PLANNING_TERMINAL_STEPS = {
    18.5: 2600,
    26.5: 1720,
    30.5: 1720,
    34.5: 1720,
    38.5: 1720,
}


def center_slug(center: float) -> str:
    """Return the frozen filesystem-safe identifier for one center."""

    if center not in NEW_CENTERS:
        raise ValueError(f"center is not in the frozen scan: {center}")
    return f"c{center:.1f}".replace(".", "p")


def geometry_for_center(center: float) -> FourArmCollarGeometry:
    """Return the fixed-width collar geometry for a frozen center."""

    if center not in (*NEW_CENTERS, *ANCHOR_CENTERS):
        raise ValueError(f"unsupported collar center: {center}")
    half_width = 0.5 * COLLAR_SUPPORT_WIDTH
    return FourArmCollarGeometry(
        inner_support_distance=center - half_width,
        outer_support_distance=center + half_width,
        transition_width=COLLAR_TRANSITION_WIDTH,
        protected_mobility_factor=PROTECTED_MOBILITY_FACTOR,
    )


def point_mobility_factor(
    coordinate: float,
    geometry: FourArmCollarGeometry,
) -> float:
    """Evaluate the on-surface axial factor at one absolute coordinate."""

    coverage = float(
        signed_arm_window(
            np.asarray([coordinate], dtype=np.float64),
            geometry,
        )[0]
    )
    return float(
        1.0
        - (1.0 - geometry.protected_mobility_factor) * coverage
    )


@dataclass(frozen=True)
class PositionCase:
    """One frozen scan position and its pre-result hypothesis."""

    center: float
    predicted_mode: str
    prediction: str

    @property
    def slug(self) -> str:
        return center_slug(self.center)

    @property
    def output(self) -> Path:
        return RESULTS / f"position_scan_{self.slug}_v1"

    @property
    def stream_name(self) -> str:
        # Checkpoint stream names intentionally exclude digits.  Every case
        # has a separate output directory, so one common stream is unambiguous.
        return "position_scan"

    @property
    def geometry(self) -> FourArmCollarGeometry:
        return geometry_for_center(self.center)

    @property
    def xi(self) -> float:
        return (self.center - DEPLETION_MINIMUM) / RADIUS

    @property
    def predicted_downstream_site(self) -> float:
        return (
            self.geometry.outer_support_distance
            + CONDITIONAL_DOWNSTREAM_OFFSET
        )

    def to_dict(self) -> dict[str, Any]:
        geometry = self.geometry
        return {
            "slug": self.slug,
            "center": self.center,
            "geometry": geometry.to_dict(),
            "xi": self.xi,
            "mobility_factor_at_depletion_minimum": (
                point_mobility_factor(DEPLETION_MINIMUM, geometry)
            ),
            "mobility_factor_at_untreated_event_site": (
                point_mobility_factor(UNTREATED_EVENT_SITE, geometry)
            ),
            "predicted_mode": self.predicted_mode,
            "predicted_downstream_site": self.predicted_downstream_site,
            "prediction": self.prediction,
            "planning_terminal_step": PLANNING_TERMINAL_STEPS[self.center],
            "output": str(self.output.resolve()),
        }


CASES = (
    PositionCase(
        18.5,
        "downstream",
        (
            "Direct plateau coverage of the untreated event site should "
            "suppress the natural mode and delay/relocate first failure."
        ),
    ),
    PositionCase(
        26.5,
        "crossover_unassigned",
        (
            "No direct coverage remains, but the inner edge is only 1.25 "
            "beyond the natural site; this case discriminates local return "
            "from finite-range downstream relocation."
        ),
    ),
    PositionCase(
        30.5,
        "natural",
        (
            "The natural junction-adjacent mode should precede any "
            "collar-conditioned downstream mode."
        ),
    ),
    PositionCase(
        34.5,
        "natural",
        (
            "The natural junction-adjacent mode should precede any "
            "collar-conditioned downstream mode."
        ),
    ),
    PositionCase(
        38.5,
        "natural",
        (
            "The natural junction-adjacent mode should precede any "
            "collar-conditioned downstream mode."
        ),
    ),
)
CASES_BY_SLUG = {case.slug: case for case in CASES}
CASES_BY_CENTER = {case.center: case for case in CASES}


def case_for_center(center: float) -> PositionCase:
    """Resolve a numeric CLI center without accepting an adaptive value."""

    try:
        return CASES_BY_CENTER[float(center)]
    except KeyError as error:
        raise ValueError(
            f"center must be one of {list(NEW_CENTERS)}"
        ) from error


def static_case_report(case: PositionCase) -> dict[str, Any]:
    """Return geometry-derived quantities that require no field allocation."""

    report = case.to_dict()
    geometry = case.geometry
    spacing = FROZEN_DEG90.lattice.spacing
    grid_values = (
        geometry.inner_support_distance,
        geometry.outer_support_distance,
        geometry.plateau_bounds[0],
        geometry.plateau_bounds[1],
        geometry.center_distance,
    )
    checks = {
        "center_is_frozen": case.center in NEW_CENTERS,
        "support_width_is_frozen": (
            geometry.support_width == COLLAR_SUPPORT_WIDTH
        ),
        "transition_width_is_frozen": (
            geometry.transition_width == COLLAR_TRANSITION_WIDTH
        ),
        "mobility_factor_is_frozen": (
            geometry.protected_mobility_factor
            == PROTECTED_MOBILITY_FACTOR
        ),
        "geometry_is_grid_aligned": all(
            np.isclose(value / spacing, round(value / spacing))
            for value in grid_values
        ),
        "support_stays_outside_central_exclusion": (
            geometry.inner_support_distance
            > CENTRAL_EXCLUSION_DISTANCE
        ),
        "point_factors_are_bounded": all(
            0.0 < value <= 1.0
            for value in (
                report["mobility_factor_at_depletion_minimum"],
                report["mobility_factor_at_untreated_event_site"],
            )
        ),
    }
    report["central_exclusion_clearance"] = (
        geometry.inner_support_distance - CENTRAL_EXCLUSION_DISTANCE
    )
    report["checks"] = checks
    if not all(checks.values()):
        raise RuntimeError(
            f"static geometry checks failed for {case.slug}: {checks}"
        )
    return report
