#!/usr/bin/env python3
"""Two discarded full-grid proposals for the selective-mobility operator."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..roy_2021_reproduction.model import (
    FROZEN_DEG90,
    RoyPseudospectralSolver,
)
from ..roy_2021_reproduction.run_preflight import contact_metrics
from ..roy_2021_reproduction.run_t1000 import four_arm_metrics
from ..roy_fixed_gb_bridge import (
    run_t100_checkpoint_source_localization_preflight as source_helper,
)
from ..roy_fixed_gb_bridge.metrics import array_fingerprint
from .analysis import (
    cross_arm_relative_l2_asymmetry,
    derive_four_arm_geometry,
    measure_t100_depletion,
)
from .geometry import (
    four_arm_collar_factor,
    surface_weighted_mobility_deficit,
)
from .model import SpatialMobilityRoySolver


SOURCE_PATH = Path(__file__).resolve()
DIRECTORY = SOURCE_PATH.parent
DEFAULT_OUTPUT = DIRECTORY / "results" / "t100_static_preflight_v1"
FFT_WORKERS = 12
WALL_CAP_SECONDS = 300.0
MASS_DRIFT_LIMIT = 1.0e-6
ENERGY_REBOUND_LIMIT = 1.0e-6
FIELD_MAGNITUDE_LIMIT = 2.0
IMAGINARY_RESIDUE_LIMIT = 1.0e-5
ASYMMETRY_AMPLIFICATION_LIMIT = 1.0e-4
RESPONSE_TO_ULP_RMS_MINIMUM = 100.0
RUNTIME_SAFETY_FACTOR = 1.3


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        measured = float(value)
        return measured if np.isfinite(measured) else None
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                _json_safe(payload),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _field_record(
    field: np.ndarray,
    *,
    solver: RoyPseudospectralSolver,
    reference_mass: float,
) -> dict[str, Any]:
    mass = float(
        np.sum(field, dtype=np.float64)
        * FROZEN_DEG90.lattice.cell_volume
    )
    return {
        "finite": bool(np.isfinite(field).all()),
        "minimum": float(np.min(field)),
        "maximum": float(np.max(field)),
        "mass": mass,
        "relative_mass_drift": abs(mass - reference_mass)
        / abs(reference_mass),
        "free_energy": solver.free_energy(field),
        "last_inverse_imaginary_linf": float(
            solver.last_inverse_imaginary_linf
        ),
        "contact": contact_metrics(field, FROZEN_DEG90),
        "four_arm_topology": four_arm_metrics(field, FROZEN_DEG90),
    }


def _topology_intact(record: dict[str, Any]) -> bool:
    contact = record["contact"]["thresholds"]
    arms = record["four_arm_topology"]["thresholds"]
    return all(
        contact[f"{level:.2f}"]["cores_connected_locally"]
        and arms[f"{level:.2f}"]["all_four_arms_attached"]
        for level in (0.45, 0.50, 0.55)
    )


def _proposal_health(
    record: dict[str, Any],
    *,
    input_energy: float,
) -> dict[str, bool]:
    return {
        "finite": record["finite"],
        "field_magnitude": (
            abs(record["minimum"]) <= FIELD_MAGNITUDE_LIMIT
            and abs(record["maximum"]) <= FIELD_MAGNITUDE_LIMIT
        ),
        "mass_drift": record["relative_mass_drift"] <= MASS_DRIFT_LIMIT,
        "energy_nonincreasing_with_tolerance": (
            (record["free_energy"] - input_energy) / abs(input_energy)
            <= ENERGY_REBOUND_LIMIT
        ),
        "imaginary_residue": (
            record["last_inverse_imaginary_linf"]
            <= IMAGINARY_RESIDUE_LIMIT
        ),
        "contact_and_four_arms_intact": _topology_intact(record),
    }


def _paired_response_metrics(
    untreated: np.ndarray,
    treated: np.ndarray,
    *,
    reference_mass: float,
) -> dict[str, float]:
    squared_response = 0.0
    squared_floor = 0.0
    response_sum = 0.0
    response_linf = 0.0
    count = 0
    for start in range(0, untreated.shape[2], 16):
        stop = min(untreated.shape[2], start + 16)
        first = untreated[:, :, start:stop]
        second = treated[:, :, start:stop]
        response = second - first
        first32 = np.asarray(first, dtype=np.float32)
        second32 = np.asarray(second, dtype=np.float32)
        floor = np.maximum.reduce(
            (
                np.abs(
                    np.nextafter(first32, np.float32(np.inf)) - first32
                ),
                np.abs(
                    np.nextafter(first32, np.float32(-np.inf)) - first32
                ),
                np.abs(
                    np.nextafter(second32, np.float32(np.inf)) - second32
                ),
                np.abs(
                    np.nextafter(second32, np.float32(-np.inf)) - second32
                ),
            )
        )
        squared_response += float(
            np.sum(response * response, dtype=np.float64)
        )
        squared_floor += float(
            np.sum(
                floor.astype(np.float64) ** 2,
                dtype=np.float64,
            )
        )
        response_sum += float(np.sum(response, dtype=np.float64))
        response_linf = max(
            response_linf, float(np.max(np.abs(response)))
        )
        count += response.size
    response_rms = float(np.sqrt(squared_response / count))
    floor_rms = float(np.sqrt(squared_floor / count))
    return {
        "response_rms": response_rms,
        "output_ulp_floor_rms": floor_rms,
        "response_to_ulp_floor_rms_ratio": (
            response_rms / floor_rms if floor_rms else float("inf")
        ),
        "response_linf": response_linf,
        "differential_mass_fraction": (
            abs(response_sum * FROZEN_DEG90.lattice.cell_volume)
            / abs(reference_mass)
        ),
    }


def _relative_asymmetry(
    field: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray],
) -> float:
    transformed = transform(field)
    numerator = float(np.linalg.norm(field - transformed))
    denominator = float(np.linalg.norm(field))
    return numerator / denominator if denominator else 0.0


def _asymmetry_metrics(field: np.ndarray) -> dict[str, float]:
    indices = (-np.arange(field.shape[1])) % field.shape[1]
    return {
        "wire_exchange": cross_arm_relative_l2_asymmetry(field),
        "y_reflection": _relative_asymmetry(
            field, lambda value: value[:, indices, :]
        ),
        "z_reflection": _relative_asymmetry(
            field, lambda value: value[:, :, indices]
        ),
    }


def _mask_checks(factor: np.ndarray) -> dict[str, bool]:
    indices = (-np.arange(factor.shape[1])) % factor.shape[1]
    return {
        "shape_is_broadcastable": factor.shape
        == (1, FROZEN_DEG90.lattice.shape[1], FROZEN_DEG90.lattice.shape[2]),
        "finite": bool(np.isfinite(factor).all()),
        "bounded": bool(
            np.min(factor) >= 0.1 - 1.0e-15
            and np.max(factor) <= 1.0
        ),
        "read_only": not factor.flags.writeable,
        "y_reflection_exact": bool(
            np.array_equal(factor, factor[:, indices, :])
        ),
        "z_reflection_exact": bool(
            np.array_equal(factor, factor[:, :, indices])
        ),
        "wire_exchange_exact": bool(
            np.array_equal(factor, np.swapaxes(factor, 1, 2))
        ),
        "junction_untreated": bool(
            factor[
                0,
                factor.shape[1] // 2,
                factor.shape[2] // 2,
            ]
            == 1.0
        ),
    }


def _long_run_projection(proposal_seconds: list[float]) -> dict[str, Any]:
    seconds_per_step = float(np.median(proposal_seconds))
    remaining_steps = 2000 - source_helper.EXPECTED_T100_STEP
    projected_seconds = (
        RUNTIME_SAFETY_FACTOR * seconds_per_step * remaining_steps
    )
    checkpoint_count = len(range(200, 2001, 200))
    checkpoint_bytes = (
        checkpoint_count * source_helper.EXPECTED_T100_FIELD_BYTES
    )
    return {
        "median_measured_proposal_seconds": seconds_per_step,
        "runtime_safety_factor": RUNTIME_SAFETY_FACTOR,
        "remaining_steps_to_t2000": remaining_steps,
        "projected_seconds_per_case": projected_seconds,
        "projected_hours_per_case": projected_seconds / 3600.0,
        "checkpoint_interval": 200,
        "projected_checkpoint_count_per_case": checkpoint_count,
        "projected_checkpoint_bytes_per_case": checkpoint_bytes,
        "explicit_notice_required_before_launch": (
            projected_seconds > 3600.0
        ),
    }


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    resource = source_helper.resource_preflight(output)
    if not resource["passed"]:
        raise RuntimeError(f"resource preflight failed: {resource}")
    output.mkdir()
    started = time.perf_counter()

    source = source_helper.verify_sources()
    checkpoint_path = source_helper.T100_CHECKPOINT_PATH
    checkpoint_file_before = source_helper._file_state(checkpoint_path)
    checkpoint = np.load(
        checkpoint_path, mmap_mode="r", allow_pickle=False
    )
    input_fingerprint = array_fingerprint(checkpoint)
    if input_fingerprint != source_helper.EXPECTED_T100_FINGERPRINT:
        raise RuntimeError("immutable t=100 checkpoint fingerprint mismatch")
    if time.perf_counter() - started > WALL_CAP_SECONDS:
        raise TimeoutError("static preflight wall cap reached")

    depletion = measure_t100_depletion(checkpoint)
    geometry = derive_four_arm_geometry(
        depletion,
        spacing=FROZEN_DEG90.lattice.spacing,
    )
    factor = four_arm_collar_factor(
        FROZEN_DEG90.lattice, geometry
    )
    mask_checks = _mask_checks(factor)
    if not all(mask_checks.values()):
        raise RuntimeError(f"mobility mask checks failed: {mask_checks}")
    mobility_deficit = surface_weighted_mobility_deficit(
        checkpoint,
        factor,
        cell_volume=FROZEN_DEG90.lattice.cell_volume,
    )

    parameters = FROZEN_DEG90.parameters
    lattice = FROZEN_DEG90.lattice
    reference_mass = source_helper.EXPECTED_T100_MASS
    untreated_solver = RoyPseudospectralSolver(
        lattice, parameters, fft_workers=FFT_WORKERS
    )
    input_energy = untreated_solver.free_energy(checkpoint)
    proposal_started = time.perf_counter()
    untreated = untreated_solver.propose_step(checkpoint)
    untreated_seconds = float(time.perf_counter() - proposal_started)
    untreated_record = _field_record(
        untreated,
        solver=untreated_solver,
        reference_mass=reference_mass,
    )
    untreated_health = _proposal_health(
        untreated_record, input_energy=input_energy
    )
    del untreated_solver
    gc.collect()

    if time.perf_counter() - started > WALL_CAP_SECONDS:
        raise TimeoutError("static preflight wall cap reached")
    treated_solver = SpatialMobilityRoySolver(
        lattice,
        parameters,
        factor,
        fft_workers=FFT_WORKERS,
    )
    proposal_started = time.perf_counter()
    treated = treated_solver.propose_step(checkpoint)
    treated_seconds = float(time.perf_counter() - proposal_started)
    treated_record = _field_record(
        treated,
        solver=treated_solver,
        reference_mass=reference_mass,
    )
    treated_health = _proposal_health(
        treated_record, input_energy=input_energy
    )
    del treated_solver
    gc.collect()

    paired = _paired_response_metrics(
        untreated, treated, reference_mass=reference_mass
    )
    asymmetry = {
        "input": _asymmetry_metrics(checkpoint),
        "untreated": _asymmetry_metrics(untreated),
        "treated": _asymmetry_metrics(treated),
    }
    amplification = {
        name: asymmetry["treated"][name] - asymmetry["untreated"][name]
        for name in asymmetry["untreated"]
    }
    asymmetry_checks = {
        name: increase <= ASYMMETRY_AMPLIFICATION_LIMIT
        for name, increase in amplification.items()
    }

    checkpoint_fingerprint_after = array_fingerprint(checkpoint)
    checkpoint_file_after = source_helper._file_state(checkpoint_path)
    checks = {
        "source_verified": source["verified"] is True,
        "resource_preflight": resource["passed"] is True,
        "input_is_read_only": not checkpoint.flags.writeable,
        "input_fingerprint": (
            input_fingerprint
            == source_helper.EXPECTED_T100_FINGERPRINT
        ),
        "input_energy_matches": bool(
            np.isclose(
                input_energy,
                source_helper.EXPECTED_T100_ENERGY,
                rtol=0.0,
                atol=1.0e-9,
            )
        ),
        "geometry_inner_edge_is_16p5": bool(
            geometry.inner_support_distance == 16.5
        ),
        "geometry_center_is_22p5": bool(
            geometry.center_distance == 22.5
        ),
        "mask_checks": all(mask_checks.values()),
        "untreated_health": all(untreated_health.values()),
        "treated_health": all(treated_health.values()),
        "paired_response_resolved": (
            paired["response_to_ulp_floor_rms_ratio"]
            >= RESPONSE_TO_ULP_RMS_MINIMUM
        ),
        "paired_differential_mass": (
            paired["differential_mass_fraction"] <= MASS_DRIFT_LIMIT
        ),
        "asymmetry_not_amplified": all(asymmetry_checks.values()),
        "checkpoint_array_unchanged": (
            checkpoint_fingerprint_after == input_fingerprint
        ),
        "checkpoint_file_unchanged": (
            checkpoint_file_after == checkpoint_file_before
        ),
        "exactly_two_discarded_proposals": True,
        "zero_accepted_trajectory_steps": True,
        "wall_cap": time.perf_counter() - started <= WALL_CAP_SECONDS,
    }
    passed = bool(all(checks.values()))
    report = {
        "schema_version": 1,
        "status": "completed",
        "classification": (
            "selective_mobility_static_preflight_passed"
            if passed
            else "selective_mobility_static_preflight_failed"
        ),
        "passed": passed,
        "scope": {
            "discarded_full_grid_proposals": 2,
            "accepted_trajectory_steps": 0,
            "production_authorized": False,
        },
        "source": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_fingerprint": input_fingerprint,
            "checkpoint_sha256": checkpoint_file_before["sha256"],
            "global_step": source_helper.EXPECTED_T100_STEP,
            "input_energy": input_energy,
            "reference_mass": reference_mass,
        },
        "depletion_measurement": depletion,
        "collar_geometry": geometry.to_dict(),
        "surface_weighted_mobility_deficit": mobility_deficit,
        "mask": {
            "shape": list(factor.shape),
            "minimum": float(np.min(factor)),
            "maximum": float(np.max(factor)),
            "checks": mask_checks,
        },
        "untreated": {
            "proposal_seconds": untreated_seconds,
            "record": untreated_record,
            "health": untreated_health,
        },
        "treated": {
            "proposal_seconds": treated_seconds,
            "record": treated_record,
            "health": treated_health,
        },
        "paired_response": paired,
        "asymmetry": {
            **asymmetry,
            "treated_minus_untreated": amplification,
            "checks": asymmetry_checks,
        },
        "runtime_projection": _long_run_projection(
            [untreated_seconds, treated_seconds]
        ),
        "resource_preflight": resource,
        "checks": checks,
        "wall_seconds": float(time.perf_counter() - started),
    }
    _atomic_json(output / "summary.json", report)
    if not passed:
        raise RuntimeError(
            "selective-mobility static preflight failed: "
            + json.dumps(
                {
                    name: value
                    for name, value in checks.items()
                    if not value
                },
                sort_keys=True,
            )
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    report = run(arguments.output.resolve())
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "passed": report["passed"],
                "collar_geometry": report["collar_geometry"],
                "paired_response": report["paired_response"],
                "runtime_projection": report["runtime_projection"],
                "wall_seconds": report["wall_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
