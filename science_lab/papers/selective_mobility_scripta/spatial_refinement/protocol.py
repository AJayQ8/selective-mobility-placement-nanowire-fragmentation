"""Frozen definitions for the focused R6 factor-two spatial refinement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np

from science_lab.papers.nanowire_gb_junction.roy_2021_reproduction.model import (
    PeriodicLattice,
    RoyDeg90Definition,
    RoyModelParameters,
)
from science_lab.papers.nanowire_gb_junction.roy_selective_mobility_protection.geometry import (
    FourArmCollarGeometry,
)


PROTOCOL_ID = "selective_mobility_scripta_r6_spatial_refinement_v1"
SCHEMA_VERSION = 1
DIRECTORY = Path(__file__).resolve().parent
CONTRACT_PATH = DIRECTORY / "SCIENTIFIC_CONTRACT.md"
DEFAULT_OUTPUT = DIRECTORY / "results" / "r6_h0p25_untreated_c34_v1"

EXTERNAL_SOURCE_DIRECTORY = Path(
    os.environ.get(
        "SELECTIVE_MOBILITY_REFINEMENT_SOURCE",
        DIRECTORY / "external_inputs" / "r6_branch_aware_source_v3",
    )
)
EXTERNAL_SOURCE_FIELD = (
    EXTERNAL_SOURCE_DIRECTORY
    / "checkpoint-final_bracket_current-step-0120.npy"
)
EXTERNAL_SOURCE_METADATA = (
    EXTERNAL_SOURCE_DIRECTORY
    / "checkpoint-final_bracket_current-step-0120.json"
)
EXTERNAL_SOURCE_SUMMARY = EXTERNAL_SOURCE_DIRECTORY / "summary.json"
EXTERNAL_SOURCE_FIELD_SHA256 = (
    "69802e10c44be4ad343efbd68f955897d4e068d5407a3f6e5ea2b9b0347264a1"
)
EXTERNAL_SOURCE_METADATA_SHA256 = (
    "965ffb278521ebfb320385662f5f7ad28abdcb029514f1c30cb1cd4fb6043c3b"
)
EXTERNAL_SOURCE_SUMMARY_SHA256 = (
    "d927a3287807390ecd4034dc0c8a5a8e6d8ea60588b13e31d3ae06983f8d29e0"
)
EXTERNAL_SOURCE_FINGERPRINT = (
    "3a410dd120911e2f709c6277a16b546bb6b6ae504469615cc47bd76cbd77a14a"
)
EXTERNAL_SOURCE_TIME = 120.0
SOURCE_FIELD_NAME = "checkpoint-final_bracket_current-step-0120.npy"
SOURCE_METADATA_NAME = "checkpoint-final_bracket_current-step-0120.json"
SOURCE_SUMMARY_NAME = "summary.json"

COARSE_REFERENCE_CAMPAIGN_NAME = (
    "radius_scaling_validation_v3_campaign_20260731_run1"
)
COARSE_REFERENCE_SUMMARY_RELATIVE_PATHS = {
    "untreated": Path("r6/case-untreated/summary.json"),
    "c34": Path("r6/case-harmful/summary.json"),
}
COARSE_REFERENCE_SUMMARY_SHA256 = {
    "untreated": (
        "ec4b8f79d1a4cefe5793cb126025c7ccfe28537c4074b1dd41feae39e7d403d8"
    ),
    "c34": (
        "aea7466a58b6d9c2de0f8475cd1ba1f11477658015ddf0ac932bc8483591f7d0"
    ),
}
COARSE_REFERENCE_HORIZON = 2900

COARSE_SHAPE = (96, 384, 384)
FINE_SHAPE = (192, 768, 768)
COARSE_SPACING = 0.5
FINE_SPACING = 0.25
PHYSICAL_LENGTHS = (48.0, 192.0, 192.0)
RADIUS = 6.0
FINE_RADIUS_CELLS = 24
FIRST_WIRE_CENTER_X = 0.0
SECOND_WIRE_CENTER_X = 12.0
WIDTH = float(np.sqrt(8.0))
MOBILITY_FACTOR = 0.1
NOISE_SEED = 2292

TIMESTEP = 1.0
HORIZON = 2000
DIAGNOSTIC_INTERVAL = 10
ENERGY_INTERVAL = 50
PROFILE_INTERVAL = 100
PERSISTENCE_RECORDS = 3
CHECKPOINT_STEPS = (800, 1600)
FFT_WORKERS = 12

MASS_DRIFT_LIMIT = 1.0e-4
ENERGY_REBOUND_LIMIT = 1.0e-6
FIELD_MAGNITUDE_LIMIT = 2.0

GUARD_INTERFACE_RMSE_LIMIT = 0.02
GUARD_CONTACT_NORMALIZED_LIMIT = 0.01
GUARD_PROFILE_NORMALIZED_RMSE_LIMIT = 0.02
GUARD_PROFILE_NORMALIZED_MAX_LIMIT = 0.05

SOURCE_COINCIDENT_ATOL = 2.0e-11
SOURCE_MEAN_ATOL = 2.0e-13
SOURCE_MASS_RELATIVE_LIMIT = 2.0e-12

COARSE_EVENT_BRACKETS = {
    "untreated": (1600, 1610),
    "c34": (1500, 1510),
}
COARSE_EVENT_MIDPOINTS = {"untreated": 1605.0, "c34": 1505.0}
COARSE_EVENT_GAPS = {
    "untreated": [{"wire": "first_wire_z", "site": -18.75}],
    "c34": [
        {"wire": "first_wire_z", "site": -18.0},
        {"wire": "second_wire_y", "site": 18.5},
    ],
}
EVENT_MIDPOINT_RELATIVE_LIMIT = 0.05
PAIRED_EFFECT_BOUNDS = (-125.0, -75.0)
NATURAL_SITE_ABS_BOUNDS = (14.0, 22.5)
NATURAL_WIRES = ("first_wire_z", "second_wire_y")
PERSISTENT_GAP_COUNT_BOUNDS = (1, 2)

EXPECTED_RUNTIME_HOURS = (7.0, 9.5)
GUARDED_RUNTIME_SECONDS = int(EXPECTED_RUNTIME_HOURS[1] * 3600)
# Legacy full-step accounting uses 256 B/cell (27.0 GiB here). The discarded
# guard can also retain the common source, the dt=1 endpoint, and the treated
# mobility factor while a proposal owns its work arrays. That totals about
# 29.6 GiB; 32 GiB is the frozen conservative ceiling.
SOLVER_PEAK_BYTES_PER_CELL = 256
SOLVER_PEAK_ESTIMATE_BYTES = int(np.prod(FINE_SHAPE, dtype=np.int64)) * (
    SOLVER_PEAK_BYTES_PER_CELL
)
PEAK_RAM_ESTIMATE_BYTES = 32 << 30
MINIMUM_PHYSICAL_RAM_MARGIN_BYTES = 16 << 30
MINIMUM_AVAILABLE_RAM_MARGIN_BYTES = 8 << 30
UNTOUCHED_RESERVE_BYTES = 10 << 30
# One shared source plus, for each of two cases, two regular checkpoints, one
# terminal checkpoint, and at most one retained interruption checkpoint.
MAXIMUM_PERSISTENT_FIELDS = 9
FIELD_RAW_BYTES = int(np.prod(FINE_SHAPE, dtype=np.int64)) * 8
FIELD_ARTIFACT_UPPER_BOUND_BYTES = FIELD_RAW_BYTES + 4096
WORST_CASE_FIELD_PERSISTENT_BYTES = (
    MAXIMUM_PERSISTENT_FIELDS * FIELD_ARTIFACT_UPPER_BOUND_BYTES
)
PLANNED_SMALL_OUTPUT_ALLOWANCE_BYTES = 512 << 20
WORST_CASE_PERSISTENT_BYTES = (
    WORST_CASE_FIELD_PERSISTENT_BYTES
    + PLANNED_SMALL_OUTPUT_ALLOWANCE_BYTES
)
LARGEST_ATOMIC_WRITE_BYTES = FIELD_ARTIFACT_UPPER_BOUND_BYTES
TEMPORARY_GIT_SNAPSHOT_CLEANUP_THRESHOLD_BYTES = 20 << 30
DEFAULT_DRIFT_WINDOW_SECONDS = 15.0
DEFAULT_DRIFT_SAMPLE_SECONDS = 5.0

LEGACY_MODULE_SHA256 = {
    "roy_model": "3d335de207dabbb11c670e5fafea453348e2323f6740088325f100f93f062106",
    "selective_model": "38ea302bec8456d645c5f212082a92efbc4d3e97dc9ca198c292ef76cade2a7a",
    "geometry": "d146ece555168ed06733d888a76e0298ead2ed38617be8e895c47fd0380e90e5",
    "event_diagnostics": "ce5dfd6e022a57049ace1a8622bfcd0d2fb095208b0563adfa2038811604fcde",
    "storage": "b745c180d1271cf3f384626eae49735256c92e0754e898b1696b3499b7ee88ef",
    # Hash of the source helper included in this archive.
    "contact_helper": "6d3d67edc4f7d51d940dcc56f18b03696dac197a16aed07d2012d0984b0f0bff",
}


@dataclass(frozen=True)
class CaseDefinition:
    case_id: str
    support: tuple[float, float] | None

    @property
    def untreated(self) -> bool:
        return self.support is None

    @property
    def geometry(self) -> FourArmCollarGeometry | None:
        if self.support is None:
            return None
        return FourArmCollarGeometry(
            inner_support_distance=self.support[0],
            outer_support_distance=self.support[1],
            transition_width=3.0,
            protected_mobility_factor=MOBILITY_FACTOR,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "untreated": self.untreated,
            "geometry": None if self.geometry is None else self.geometry.to_dict(),
        }


UNTREATED = CaseDefinition("untreated", None)
C34 = CaseDefinition("c34", (28.5, 40.5))
CASES = (UNTREATED, C34)
CASES_BY_ID = {case.case_id: case for case in CASES}


@dataclass(frozen=True)
class SourcePaths:
    root: Path
    field: Path
    metadata: Path
    summary: Path


@dataclass(frozen=True)
class CoarseReferencePaths:
    root: Path
    summaries: dict[str, Path]


def source_paths(source_root: Path | None = None) -> SourcePaths:
    """Resolve a relocatable source root while keeping byte hashes frozen."""

    root = (source_root or EXTERNAL_SOURCE_DIRECTORY).expanduser().resolve()
    return SourcePaths(
        root=root,
        field=root / SOURCE_FIELD_NAME,
        metadata=root / SOURCE_METADATA_NAME,
        summary=root / SOURCE_SUMMARY_NAME,
    )


def coarse_reference_paths(
    source_root: Path | None = None,
) -> CoarseReferencePaths:
    """Locate optional hash-bound coarse evidence beside the source campaign."""

    source = source_paths(source_root)
    results_root = source.root.parent.parent
    root = (results_root / COARSE_REFERENCE_CAMPAIGN_NAME).resolve()
    return CoarseReferencePaths(
        root=root,
        summaries={
            case_id: root / relative
            for case_id, relative in COARSE_REFERENCE_SUMMARY_RELATIVE_PATHS.items()
        },
    )


def fine_definition() -> RoyDeg90Definition:
    return RoyDeg90Definition(
        lattice=PeriodicLattice(FINE_SHAPE, FINE_SPACING),
        parameters=RoyModelParameters(
            mobility_prefactor=1.0,
            barrier=1.0,
            kappa=1.0,
            stabilizer_alpha=0.5,
            timestep=TIMESTEP,
        ),
        radius_1=FINE_RADIUS_CELLS,
        radius_2=FINE_RADIUS_CELLS,
        solid_composition=1.0,
        noise_amplitude=1.0e-3,
        noise_seed=NOISE_SEED,
    )


def case_for(case_id: str) -> CaseDefinition:
    try:
        return CASES_BY_ID[case_id]
    except KeyError as error:
        raise ValueError(f"unknown frozen case: {case_id}") from error


def protocol_payload() -> dict[str, Any]:
    definition = fine_definition()
    return {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "source": {
            "field_sha256": EXTERNAL_SOURCE_FIELD_SHA256,
            "metadata_sha256": EXTERNAL_SOURCE_METADATA_SHA256,
            "summary_sha256": EXTERNAL_SOURCE_SUMMARY_SHA256,
            "field_fingerprint": EXTERNAL_SOURCE_FINGERPRINT,
            "physical_time": EXTERNAL_SOURCE_TIME,
            "coarse_shape": list(COARSE_SHAPE),
            "coarse_spacing": COARSE_SPACING,
            "prolongation": "separable_periodic_fourier_factor_2_no_clipping",
        },
        "definition": {
            "shape": list(FINE_SHAPE),
            "spacing": FINE_SPACING,
            "physical_lengths": list(definition.lattice.physical_lengths),
            "radius": RADIUS,
            "radius_cells": FINE_RADIUS_CELLS,
            "first_wire_center_x": FIRST_WIRE_CENTER_X,
            "second_wire_center_x": SECOND_WIRE_CENTER_X,
            "width": WIDTH,
            "parameters": asdict(definition.parameters),
        },
        "cases_in_order": [case.to_dict() for case in CASES],
        "timestep_guard": {
            "comparison": "one_dt1_vs_two_dt0p5_from_identical_fine_source",
            "fields_discarded": True,
            "interface_rmse_limit": GUARD_INTERFACE_RMSE_LIMIT,
            "contact_normalized_limit": GUARD_CONTACT_NORMALIZED_LIMIT,
            "profile_normalized_rmse_limit": GUARD_PROFILE_NORMALIZED_RMSE_LIMIT,
            "profile_normalized_max_limit": GUARD_PROFILE_NORMALIZED_MAX_LIMIT,
        },
        "trajectory": {
            "timestep": TIMESTEP,
            "horizon": HORIZON,
            "diagnostic_interval": DIAGNOSTIC_INTERVAL,
            "energy_interval": ENERGY_INTERVAL,
            "profile_interval": PROFILE_INTERVAL,
            "persistence_records": PERSISTENCE_RECORDS,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "stop_on_event": True,
        },
        "acceptance": {
            "coarse_reference_horizon": COARSE_REFERENCE_HORIZON,
            "fine_horizon": HORIZON,
            "fine_horizon_identical_between_branches": True,
            "fine_horizon_held_fixed_relative_to_coarse": False,
            "coarse_reference_summary_sha256": (
                COARSE_REFERENCE_SUMMARY_SHA256
            ),
            "coarse_event_brackets": {
                key: list(value) for key, value in COARSE_EVENT_BRACKETS.items()
            },
            "coarse_event_midpoints": COARSE_EVENT_MIDPOINTS,
            "coarse_event_gaps": COARSE_EVENT_GAPS,
            "midpoint_relative_limit": EVENT_MIDPOINT_RELATIVE_LIMIT,
            "paired_effect_bounds": list(PAIRED_EFFECT_BOUNDS),
            "natural_wires": list(NATURAL_WIRES),
            "natural_site_abs_bounds": list(NATURAL_SITE_ABS_BOUNDS),
            "persistent_gap_count_bounds": list(PERSISTENT_GAP_COUNT_BOUNDS),
            "exact_gap_multiplicity_convergence_claimed": False,
        },
        "scope": {
            "adaptive_follow_on_authorized": False,
            "horizon_extension_authorized": False,
            "production_dt_change_authorized": False,
            "additional_case_authorized": False,
        },
        "operations": {
            "planned_small_output_allowance_bytes": (
                PLANNED_SMALL_OUTPUT_ALLOWANCE_BYTES
            ),
            "checkpoint_reconciliation": (
                "exact_replay_adopts_complete_or_field_only_orphan"
            ),
            "metadata_only_checkpoint_policy": "preserve_and_fail_closed",
        },
    }
