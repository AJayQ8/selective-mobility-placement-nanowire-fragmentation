"""Load and enforce the frozen S0--S4 sentinel configurations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


DIRECTORY = Path(__file__).resolve().parent
CONFIG_DIRECTORY = DIRECTORY / "configs"
CONTRACT_PATH = DIRECTORY / "SENTINEL_CONTRACT.md"
CONFIG_VERSION = 1
RUN_IDS = ("S0", "S1", "S2", "S3", "S4")
CROSSED_SHAPE = (96, 768, 768)
ISOLATED_SHAPE = (96, 96, 768)
GRID_SPACING = 0.5
RADIUS = 6.0
MAXIMUM_PERIODIC_SEPARATION = 96.0
ALIGNMENT_RUN_IDS = ("S1", "S2")
CROSSED_JUNCTION_SOLVER_COORDINATE = -0.25


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SentinelConfig:
    config_version: int
    run_id: str
    description: str
    geometry: str
    shape: tuple[int, int, int]
    spacing: float
    radius: float
    initializer: str
    grain_boundary: bool
    paired_gb_off_control: bool
    gb_separation_from_junction: float | None
    placement_role: str
    launch_state: str
    target_step: int
    diagnostic_interval: int
    checkpoint_interval: int
    wall_cap_seconds: float
    noise: None
    deterministic_mode: None
    constant_mobility_preparation: bool
    gb_active_from_t0: bool

    @property
    def axial_length(self) -> float:
        return float(self.shape[2] * self.spacing)

    @property
    def field_bytes(self) -> int:
        return int(self.shape[0] * self.shape[1] * self.shape[2] * 8)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "shape": list(self.shape),
        }


def _construct(payload: dict[str, Any]) -> SentinelConfig:
    copied = dict(payload)
    copied["shape"] = tuple(int(value) for value in copied["shape"])
    return SentinelConfig(**copied)


def nearest_bicrystal_plane_distance(
    coordinate: float,
    *,
    axial_length: float,
) -> float:
    """Distance from zero to the nearer of a periodic bicrystal's two planes."""

    half = 0.5 * axial_length

    def wrapped(value: float) -> float:
        return float((value + half) % axial_length - half)

    primary = wrapped(float(coordinate))
    image = wrapped(primary + half)
    return min(abs(primary), abs(image))


def solver_gb_primary_coordinate(config: SentinelConfig) -> float | None:
    """Convert declared junction-relative spacing to the solver coordinate."""

    separation = config.gb_separation_from_junction
    if separation is None:
        return None
    origin = (
        CROSSED_JUNCTION_SOLVER_COORDINATE
        if config.geometry == "crossed"
        else 0.0
    )
    length = config.axial_length
    return float((origin + separation + 0.5 * length) % length - 0.5 * length)


def validate_config(config: SentinelConfig, *, finalized: bool = False) -> None:
    if config.config_version != CONFIG_VERSION:
        raise ValueError("unsupported sentinel config version")
    if config.run_id not in RUN_IDS:
        raise ValueError("unknown sentinel run ID")
    if config.geometry not in {"crossed", "isolated"}:
        raise ValueError("unknown sentinel geometry")
    expected_shape = CROSSED_SHAPE if config.geometry == "crossed" else ISOLATED_SHAPE
    if config.shape != expected_shape:
        raise ValueError(f"{config.run_id} has the wrong frozen shape")
    if config.spacing != GRID_SPACING or config.radius != RADIUS:
        raise ValueError("grid spacing or radius differs from the frozen model")
    if config.noise is not None or config.deterministic_mode is not None:
        raise ValueError("sentinel perturbations are prohibited")
    if config.constant_mobility_preparation:
        raise ValueError("constant-mobility preparation is prohibited")
    if config.target_step != 2000:
        raise ValueError("sentinel target must remain t=2000")
    if config.diagnostic_interval != 10 or config.checkpoint_interval != 200:
        raise ValueError("sentinel cadence differs from the frozen contract")
    if config.wall_cap_seconds <= 0.0:
        raise ValueError("wall cap must be positive")
    if config.geometry == "crossed":
        if config.initializer != "validated_equilibrium_product_union":
            raise ValueError("crossed cases require the validated product union")
    elif config.initializer != "validated_equilibrium_single_wire":
        raise ValueError("isolated case requires the validated single wire")
    if config.grain_boundary != config.gb_active_from_t0:
        raise ValueError("GB state and t=0 activation must agree")
    if config.paired_gb_off_control != (config.run_id == "S4"):
        raise ValueError("only S4 may be the paired GB-on/GB-off reference")
    if (
        not config.grain_boundary
        and config.gb_separation_from_junction is not None
    ):
        raise ValueError("GB-off case cannot declare a GB coordinate")
    if (
        config.grain_boundary
        and finalized
        and config.gb_separation_from_junction is None
    ):
        raise ValueError("finalized GB-on case requires a coordinate")

    if config.run_id == "S0":
        if config.grain_boundary or config.placement_role != "junction_only_reference":
            raise ValueError("S0 must be the junction-only reference")
    elif config.run_id in ALIGNMENT_RUN_IDS:
        if not config.grain_boundary:
            raise ValueError(f"{config.run_id} must have a GB")
        if not finalized:
            if config.gb_separation_from_junction is not None:
                raise ValueError(f"{config.run_id} template must retain a null spacing")
            if config.launch_state != "blocked_pending_s0_alignment_lock":
                raise ValueError(f"{config.run_id} must remain launch-blocked")
    elif config.run_id == "S3":
        if config.gb_separation_from_junction != MAXIMUM_PERIODIC_SEPARATION:
            raise ValueError("S3 must use s=96")
        distance = nearest_bicrystal_plane_distance(
            config.gb_separation_from_junction,
            axial_length=config.axial_length,
        )
        if abs(distance - MAXIMUM_PERIODIC_SEPARATION) > 1.0e-12:
            raise ValueError("S3 is not the maximum-separation control")
        if solver_gb_primary_coordinate(config) != 95.75:
            raise ValueError("S3 solver-coordinate conversion is not frozen")
    elif config.run_id == "S4":
        if (
            config.geometry != "isolated"
            or config.gb_separation_from_junction != 0.0
        ):
            raise ValueError("S4 must be the centered isolated-GB reference")


def load_template(run_id: str) -> SentinelConfig:
    normalized = run_id.upper()
    if normalized not in RUN_IDS:
        raise ValueError(f"unknown run ID: {run_id}")
    path = CONFIG_DIRECTORY / f"{normalized}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = _construct(payload)
    validate_config(config, finalized=False)
    return config


def load_alignment_lock(path: Path) -> dict[str, Any]:
    # Imported lazily because alignment imports the low-level contract
    # constants. This verifier recomputes the lock from hash-pinned S0/S4
    # sources and requires exact equality; schema-only acceptance is forbidden.
    from .alignment import verify_alignment_lock

    return verify_alignment_lock(path)


def load_s4_equivalence(path: Path) -> dict[str, Any]:
    """Verify the separately produced thin/full isolated-cell equivalence audit."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "artifact_version",
        "frozen",
        "classification",
        "thin_shape",
        "full_shape",
        "source_path",
        "source_sha256",
        "checks",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"S4 equivalence artifact is missing: {missing}")
    if payload["artifact_version"] != 1 or payload["frozen"] is not True:
        raise ValueError("S4 equivalence artifact must be version 1 and frozen")
    if payload["classification"] != "s4_thin_full_equivalence_passed":
        raise ValueError("S4 thin/full equivalence did not pass")
    if payload["thin_shape"] != list(ISOLATED_SHAPE):
        raise ValueError("S4 equivalence thin shape mismatch")
    if payload["full_shape"] != list(CROSSED_SHAPE):
        raise ValueError("S4 equivalence full shape mismatch")
    source = Path(str(payload["source_path"]))
    if not source.is_file() or sha256_path(source) != payload["source_sha256"]:
        raise ValueError("S4 equivalence source hash mismatch")
    checks = payload["checks"]
    if not isinstance(checks, dict) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise ValueError("S4 equivalence checks are not all passed")
    return payload


def resolve_config(
    run_id: str,
    *,
    alignment_lock: Path | None = None,
    s4_equivalence: Path | None = None,
) -> SentinelConfig:
    config = load_template(run_id)
    if config.run_id in ALIGNMENT_RUN_IDS:
        if alignment_lock is None:
            raise RuntimeError(
                f"{config.run_id} is blocked until a frozen S0 alignment lock exists"
            )
        lock = load_alignment_lock(alignment_lock)
        coordinate = float(lock[f"{config.run_id.lower()}_spacing"])
        config = replace(
            config,
            gb_separation_from_junction=coordinate,
            launch_state="ready_from_frozen_s0_alignment_lock",
        )
    if config.run_id == "S4":
        if s4_equivalence is None:
            raise RuntimeError(
                "S4 is blocked until a hashed thin/full equivalence artifact passes"
            )
        load_s4_equivalence(s4_equivalence)
        config = replace(
            config,
            launch_state="ready_after_s0_and_thin_cell_equivalence_audit",
        )
    validate_config(config, finalized=True)
    return config


def config_manifest() -> dict[str, Any]:
    configs = {run_id: load_template(run_id).to_dict() for run_id in RUN_IDS}
    return {
        "contract_sha256": sha256_path(CONTRACT_PATH),
        "configs": configs,
        "config_sha256": {
            run_id: sha256_path(CONFIG_DIRECTORY / f"{run_id}.json")
            for run_id in RUN_IDS
        },
        "manifest_sha256": canonical_sha256(configs),
    }
