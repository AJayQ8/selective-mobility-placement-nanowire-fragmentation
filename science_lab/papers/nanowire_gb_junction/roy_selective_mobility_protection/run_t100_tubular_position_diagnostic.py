#!/usr/bin/env python3
"""Bounded untreated/outer/inward tubular-collar position diagnostic."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy import fft as scipy_fft

from ..roy_2021_reproduction import model as roy_model
from ..roy_2021_reproduction.model import (
    FROZEN_DEG90,
    RoyPseudospectralSolver,
)
from ..roy_fixed_gb_bridge import (
    run_t100_checkpoint_source_localization_preflight as source_helper,
)
from ..roy_fixed_gb_bridge.metrics import array_fingerprint
from ..roy_gb_junction_sentinel.diagnostics import instantaneous_pinches
from . import analysis as selective_analysis
from . import geometry as selective_geometry
from . import model as selective_model
from .analysis import (
    INTERFACE_WIDTH,
    MINIMUM_SEARCH_DISTANCE,
    crossed_contour_radius_profiles,
    phase_excess_radius_profiles,
)
from .geometry import (
    FourArmCollarGeometry,
    four_arm_tubular_collar_factor,
)
from .model import SpatialMobilityRoySolver
from .run_t100_static_preflight import (
    FFT_WORKERS,
    MASS_DRIFT_LIMIT,
    RESPONSE_TO_ULP_RMS_MINIMUM,
    _atomic_json,
    _field_record,
    _paired_response_metrics,
    _proposal_health,
)


SOURCE_PATH = Path(__file__).resolve()
DIRECTORY = SOURCE_PATH.parent
CONTRACT_PATH = DIRECTORY / "INWARD_PLACEMENT_DIAGNOSTIC.md"
EXPLORATION_CONTRACT_PATH = DIRECTORY / "EXPLORATION_CONTRACT.md"
STATIC_SUMMARY = (
    DIRECTORY / "results" / "t100_static_preflight_v1" / "summary.json"
)
SHORT_SUMMARY = (
    DIRECTORY / "results" / "t100_short_preflight_v1" / "summary.json"
)
DEFAULT_OUTPUT = (
    DIRECTORY / "results" / "t100_tubular_position_diagnostic_v1"
)
ACCEPTED_STEPS = 16
PINCH_MILESTONES = (1, 2, 4, 8, 16)
WALL_CAP_SECONDS = 360.0
REMOTE_RESPONSE_SQUARED_FRACTION_LIMIT = 0.10
TRANSITION_DEFICIT_LIMIT = -0.01 * 6.0
OUTER_GEOMETRY = FourArmCollarGeometry(16.5, 28.5, 3.0, 0.1)
INWARD_GEOMETRY = FourArmCollarGeometry(8.5, 20.5, 3.0, 0.1)
CASE_NAMES = ("untreated", "outer", "inward")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_prerequisites() -> dict[str, Any]:
    static = json.loads(STATIC_SUMMARY.read_text(encoding="utf-8"))
    short = json.loads(SHORT_SUMMARY.read_text(encoding="utf-8"))
    if static.get("passed") is not True or short.get("passed") is not True:
        raise RuntimeError("prior numerical preflights are not complete")
    adverse = float(
        short["profile_response"]["pooled"][
            "mean_treated_minus_untreated_radius_in_inherited_thinning_zone"
        ]
    )
    if adverse >= 0.0:
        raise RuntimeError("prior outer-collar warning no longer has its sign")
    return {
        "static": static,
        "short": short,
        "static_sha256": _sha256_path(STATIC_SUMMARY),
        "short_sha256": _sha256_path(SHORT_SUMMARY),
        "recorded_planar_outer_pooled_phase_radius_response": adverse,
    }


def _implementation_hashes() -> dict[str, str]:
    paths = {
        "runner": SOURCE_PATH,
        "diagnostic_contract": CONTRACT_PATH,
        "exploration_contract": EXPLORATION_CONTRACT_PATH,
        "selective_model": Path(selective_model.__file__).resolve(),
        "selective_geometry": Path(selective_geometry.__file__).resolve(),
        "selective_analysis": Path(selective_analysis.__file__).resolve(),
        "roy_model": Path(roy_model.__file__).resolve(),
    }
    return {name: _sha256_path(path) for name, path in paths.items()}


def _wire_exchange(field: np.ndarray) -> np.ndarray:
    nx = field.shape[0]
    indices = (
        2 * (nx // 2 + FROZEN_DEG90.radius_1) - np.arange(nx)
    ) % nx
    return np.take(
        np.swapaxes(field, 1, 2),
        indices.astype(np.int64),
        axis=0,
    )


def _factor_checks(
    factor: np.ndarray,
    geometry: FourArmCollarGeometry,
) -> dict[str, bool]:
    y_indices = (-np.arange(factor.shape[1])) % factor.shape[1]
    z_indices = (-np.arange(factor.shape[2])) % factor.shape[2]
    return {
        "shape": factor.shape == FROZEN_DEG90.lattice.shape,
        "finite": bool(np.isfinite(factor).all()),
        "bounded": bool(
            np.min(factor) >= geometry.protected_mobility_factor - 1.0e-15
            and np.max(factor) <= 1.0
        ),
        "read_only": not factor.flags.writeable,
        "junction_identity": bool(
            factor[
                FROZEN_DEG90.lattice.shape[0] // 2,
                FROZEN_DEG90.lattice.shape[1] // 2,
                FROZEN_DEG90.lattice.shape[2] // 2,
            ]
            == 1.0
        ),
        "y_reflection_exact": bool(
            np.array_equal(factor, factor[:, y_indices, :])
        ),
        "z_reflection_exact": bool(
            np.array_equal(factor, factor[:, :, z_indices])
        ),
        "wire_exchange_exact": bool(
            np.array_equal(factor, _wire_exchange(factor))
        ),
        "has_exact_remote_identity": bool(
            np.count_nonzero(factor == 1.0) > 0
        ),
    }


def _squared_fraction(values: np.ndarray, mask: np.ndarray) -> float:
    squared = values * values
    total = float(np.sum(squared, dtype=np.float64))
    if total == 0.0:
        return 0.0
    return float(np.sum(squared[mask], dtype=np.float64) / total)


def _absolute_fraction(values: np.ndarray, mask: np.ndarray) -> float:
    absolute = np.abs(values)
    total = float(np.sum(absolute, dtype=np.float64))
    if total == 0.0:
        return 0.0
    return float(np.sum(absolute[mask], dtype=np.float64) / total)


def _localization_metrics(
    checkpoint: np.ndarray,
    factor: np.ndarray,
    untreated: np.ndarray,
    treated: np.ndarray,
) -> dict[str, Any]:
    support = factor < 1.0 - 1.0e-14
    interface = (checkpoint > 0.05) & (checkpoint < 0.95)
    vapor = checkpoint <= 0.05
    solid = checkpoint >= 0.95
    remote = ~support
    base_mobility = np.sqrt(
        np.abs(checkpoint - checkpoint * checkpoint)
    )
    mobility_change = base_mobility * (1.0 - factor)
    response = treated - untreated
    masks = {
        "support": support,
        "interface": interface,
        "support_interface": support & interface,
        "support_vapor": support & vapor,
        "support_solid": support & solid,
        "remote": remote,
        "remote_vapor": remote & vapor,
        "remote_solid": remote & solid,
    }
    mobility = {
        name: {
            "squared_fraction": _squared_fraction(
                mobility_change, mask
            ),
            "absolute_fraction": _absolute_fraction(
                mobility_change, mask
            ),
        }
        for name, mask in masks.items()
    }
    state_response = {
        name: {
            "squared_fraction": _squared_fraction(response, mask),
            "absolute_fraction": _absolute_fraction(response, mask),
        }
        for name, mask in masks.items()
    }
    return {
        "phase_cutoffs": {"vapor_maximum": 0.05, "solid_minimum": 0.95},
        "support_cell_fraction": float(np.mean(support)),
        "mobility_modification": mobility,
        "one_step_state_response": state_response,
        "checks": {
            "mobility_change_exactly_zero_remote": bool(
                np.count_nonzero(mobility_change[remote]) == 0
            ),
            "remote_state_response_squared_fraction": (
                state_response["remote"]["squared_fraction"]
                <= REMOTE_RESPONSE_SQUARED_FRACTION_LIMIT
            ),
        },
    }


def _operator_flux_profiles(
    solver: RoyPseudospectralSolver,
    field: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Return cross-section-integrated ``M grad(mu)`` on both wire axes."""

    mu_hat = solver.chemical_potential_spectrum(field)
    mobility = solver.mobility(field)
    _, ny, nz = field.shape
    half = 48
    y0 = ny // 2 - half
    z0 = nz // 2 - half
    definitions = (
        ("first_wire_z", 2, (0, 1), (slice(None), slice(y0, y0 + 96), slice(None))),
        ("second_wire_y", 1, (0, 2), (slice(None), slice(None), slice(z0, z0 + 96))),
    )
    output: dict[str, dict[str, np.ndarray]] = {}
    for name, axis, transverse_axes, selection in definitions:
        gradient_hat = np.empty_like(mu_hat)
        np.multiply(
            mu_hat,
            solver._wave_numbers[axis],
            out=gradient_hat,
            casting="unsafe",
        )
        gradient_hat *= np.complex64(1j)
        gradient = scipy_fft.ifftn(
            gradient_hat,
            workers=solver.fft_workers,
            overwrite_x=True,
        )
        operator_flux = np.empty(field.shape, dtype=np.float32)
        np.multiply(
            mobility,
            gradient.real,
            out=operator_flux,
            casting="unsafe",
        )
        local = operator_flux[selection]
        integrated = (
            np.sum(local, axis=transverse_axes, dtype=np.float64)
            * FROZEN_DEG90.lattice.spacing**2
        )
        cells = field.shape[axis]
        coordinate = (
            np.arange(cells, dtype=np.float64) - cells // 2
        ) * FROZEN_DEG90.lattice.spacing
        output[name] = {
            "coordinate": coordinate,
            "operator_flux_M_grad_mu": integrated,
        }
        del gradient, operator_flux
        gc.collect()
    return output


def _run_case(
    checkpoint: np.ndarray,
    factor: np.ndarray | None,
    *,
    reference_mass: float,
    input_energy: float,
    started: float,
) -> tuple[np.ndarray, dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    solver: RoyPseudospectralSolver
    if factor is None:
        solver = RoyPseudospectralSolver(
            FROZEN_DEG90.lattice,
            FROZEN_DEG90.parameters,
            fft_workers=FFT_WORKERS,
        )
    else:
        solver = SpatialMobilityRoySolver(
            FROZEN_DEG90.lattice,
            FROZEN_DEG90.parameters,
            factor,
            fft_workers=FFT_WORKERS,
        )
    state: np.ndarray = checkpoint
    records: list[dict[str, Any]] = []
    proposal_seconds: list[float] = []
    previous_energy = input_energy
    all_health = True
    no_pinch = True
    for elapsed_step in range(1, ACCEPTED_STEPS + 1):
        if time.perf_counter() - started > WALL_CAP_SECONDS:
            raise TimeoutError("tubular position diagnostic wall cap reached")
        proposal_started = time.perf_counter()
        state = solver.propose_step(state)
        proposal_seconds.append(
            float(time.perf_counter() - proposal_started)
        )
        record = _field_record(
            state,
            solver=solver,
            reference_mass=reference_mass,
        )
        health = _proposal_health(
            record, input_energy=previous_energy
        )
        all_health = bool(all_health and all(health.values()))
        pinch = None
        if elapsed_step in PINCH_MILESTONES:
            pinch = instantaneous_pinches(
                state,
                geometry="crossed",
                spacing=FROZEN_DEG90.lattice.spacing,
                radius=6.0,
            )
            no_pinch = bool(no_pinch and not pinch["candidate"])
        records.append(
            {
                "elapsed_step": elapsed_step,
                "global_step": source_helper.EXPECTED_T100_STEP + elapsed_step,
                "state": record,
                "health": health,
                "pinch_at_milestone": pinch,
            }
        )
        previous_energy = record["free_energy"]
    flux = _operator_flux_profiles(solver, state)
    run = {
        "records": records,
        "proposal_seconds": proposal_seconds,
        "all_step_health": all_health,
        "no_milestone_pinch": no_pinch,
        "final_fingerprint": array_fingerprint(state),
    }
    del solver
    gc.collect()
    return state, run, flux


def _static_proposal(
    checkpoint: np.ndarray,
    factor: np.ndarray | None,
    *,
    reference_mass: float,
    input_energy: float,
) -> tuple[np.ndarray, dict[str, Any], float]:
    solver: RoyPseudospectralSolver
    if factor is None:
        solver = RoyPseudospectralSolver(
            FROZEN_DEG90.lattice,
            FROZEN_DEG90.parameters,
            fft_workers=FFT_WORKERS,
        )
    else:
        solver = SpatialMobilityRoySolver(
            FROZEN_DEG90.lattice,
            FROZEN_DEG90.parameters,
            factor,
            fft_workers=FFT_WORKERS,
        )
    proposal_started = time.perf_counter()
    output = solver.propose_step(checkpoint)
    proposal_seconds = float(time.perf_counter() - proposal_started)
    record = _field_record(
        output,
        solver=solver,
        reference_mass=reference_mass,
    )
    health = _proposal_health(record, input_energy=input_energy)
    del solver
    gc.collect()
    return output, {
        "record": record,
        "health": health,
    }, proposal_seconds


def _profile_arrays(
    fields: dict[str, np.ndarray],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, np.ndarray],
]:
    phase = {
        name: phase_excess_radius_profiles(field)
        for name, field in fields.items()
    }
    contour = {
        name: crossed_contour_radius_profiles(field)
        for name, field in fields.items()
    }
    arrays: dict[str, np.ndarray] = {}
    for wire in ("first_wire_z", "second_wire_y"):
        arrays[f"{wire}_coordinate"] = phase["untreated"][wire][
            "coordinate"
        ]
        for case in CASE_NAMES:
            arrays[f"{wire}_{case}_phase_radius"] = phase[case][wire][
                "radius"
            ]
            for level in ("0.45", "0.50", "0.55"):
                arrays[
                    f"{wire}_{case}_contour_radius_{level}"
                ] = contour[case][wire]["levels"][level]["mean_radius"]
                arrays[
                    f"{wire}_{case}_contour_valid_{level}"
                ] = contour[case][wire]["levels"][level][
                    "valid_fraction"
                ]
    return phase, contour, arrays


def _side_means(
    coordinate: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    return {
        "negative": float(np.mean(values[mask & (coordinate < 0.0)])),
        "positive": float(np.mean(values[mask & (coordinate > 0.0)])),
    }


def _treatment_profile_response(
    phase: dict[str, Any],
    contour: dict[str, Any],
    *,
    treated_name: str,
    geometry: FourArmCollarGeometry,
) -> dict[str, Any]:
    wires: dict[str, Any] = {}
    pooled_phase: list[float] = []
    pooled_contours: dict[str, list[float]] = {
        level: [] for level in ("0.45", "0.50", "0.55")
    }
    arm_phase_means: list[float] = []
    arm_contour_means: dict[str, list[float]] = {
        level: [] for level in ("0.45", "0.50", "0.55")
    }
    transition_minima: list[float] = []
    contour_valid = True
    material_balance: dict[str, Any] = {}
    for wire in ("first_wire_z", "second_wire_y"):
        coordinate = phase["untreated"][wire]["coordinate"]
        distance = np.abs(coordinate)
        inherited = (
            (distance >= MINIMUM_SEARCH_DISTANCE)
            & (distance <= OUTER_GEOMETRY.inner_support_distance)
        )
        phase_difference = (
            phase[treated_name][wire]["radius"]
            - phase["untreated"][wire]["radius"]
        )
        phase_sides = _side_means(
            coordinate, phase_difference, inherited
        )
        arm_phase_means.extend(phase_sides.values())
        pooled_phase.extend(phase_difference[inherited].tolist())

        contour_levels: dict[str, Any] = {}
        for level in ("0.45", "0.50", "0.55"):
            difference = (
                contour[treated_name][wire]["levels"][level]["mean_radius"]
                - contour["untreated"][wire]["levels"][level]["mean_radius"]
            )
            valid = (
                contour[treated_name][wire]["levels"][level][
                    "valid_fraction"
                ]
                >= 0.95
            ) & (
                contour["untreated"][wire]["levels"][level][
                    "valid_fraction"
                ]
                >= 0.95
            )
            contour_valid = bool(
                contour_valid and np.all(valid[inherited])
            )
            sides = _side_means(
                coordinate, difference, inherited & valid
            )
            arm_contour_means[level].extend(sides.values())
            pooled_contours[level].extend(
                difference[inherited & valid].tolist()
            )
            contour_levels[level] = {
                "mean": float(
                    np.mean(difference[inherited & valid])
                ),
                "minimum": float(
                    np.min(difference[inherited & valid])
                ),
                "side_means": sides,
                "valid": bool(np.all(valid[inherited])),
            }

        inner_transition = (
            (distance >= geometry.inner_support_distance)
            & (
                distance
                <= geometry.inner_support_distance
                + geometry.transition_width
            )
        )
        outer_transition = (
            (
                distance
                >= geometry.outer_support_distance
                - geometry.transition_width
            )
            & (distance <= geometry.outer_support_distance)
        )
        transition_minima.extend(
            [
                float(np.min(phase_difference[inner_transition])),
                float(np.min(phase_difference[outer_transition])),
            ]
        )
        area_difference = np.pi * (
            phase[treated_name][wire]["radius"] ** 2
            - phase["untreated"][wire]["radius"] ** 2
        )
        bins = {
            "junction_core": (0.0, geometry.inner_support_distance),
            "inner_transition": (
                geometry.inner_support_distance,
                geometry.inner_support_distance + geometry.transition_width,
            ),
            "plateau": geometry.plateau_bounds,
            "outer_transition": (
                geometry.outer_support_distance - geometry.transition_width,
                geometry.outer_support_distance,
            ),
            "near_far_arm": (geometry.outer_support_distance, 60.0),
        }
        material_balance[wire] = {
            name: float(
                np.sum(
                    area_difference[
                        (distance >= lower) & (distance < upper)
                    ],
                    dtype=np.float64,
                )
                * FROZEN_DEG90.lattice.spacing
            )
            for name, (lower, upper) in bins.items()
        }
        wires[wire] = {
            "phase_radius": {
                "mean": float(np.mean(phase_difference[inherited])),
                "minimum": float(np.min(phase_difference[inherited])),
                "side_means": phase_sides,
            },
            "contour_radius": contour_levels,
            "true_transition_minima": {
                "inner": float(
                    np.min(phase_difference[inner_transition])
                ),
                "outer": float(
                    np.min(phase_difference[outer_transition])
                ),
            },
        }
    pooled_contour_summary = {
        level: {
            "mean": float(np.mean(values)),
            "minimum": float(np.min(values)),
            "all_four_arm_means_positive": bool(
                all(value > 0.0 for value in arm_contour_means[level])
            ),
            "four_arm_means": arm_contour_means[level],
        }
        for level, values in pooled_contours.items()
    }
    return {
        "wires": wires,
        "pooled": {
            "phase_radius_mean": float(np.mean(pooled_phase)),
            "phase_radius_minimum": float(np.min(pooled_phase)),
            "phase_all_four_arm_means_positive": bool(
                all(value > 0.0 for value in arm_phase_means)
            ),
            "phase_four_arm_means": arm_phase_means,
            "contour_radius": pooled_contour_summary,
            "minimum_true_transition_response": float(
                np.min(transition_minima)
            ),
        },
        "material_balance_volume_difference": material_balance,
        "contour_valid": contour_valid,
    }


def _flux_boundary_report(
    fluxes: dict[str, Any],
    geometry: FourArmCollarGeometry,
) -> dict[str, Any]:
    coordinates_to_sample = (
        geometry.inner_support_distance,
        geometry.inner_support_distance + geometry.transition_width,
        geometry.outer_support_distance - geometry.transition_width,
        geometry.outer_support_distance,
    )
    output: dict[str, Any] = {}
    for wire in ("first_wire_z", "second_wire_y"):
        coordinate = fluxes["untreated"][wire]["coordinate"]
        output[wire] = {}
        for distance in coordinates_to_sample:
            for sign in (-1.0, 1.0):
                target = sign * distance
                index = int(np.argmin(np.abs(coordinate - target)))
                key = f"{target:+.1f}"
                output[wire][key] = {
                    "sample_coordinate": float(coordinate[index]),
                    "untreated_M_grad_mu": float(
                        fluxes["untreated"][wire][
                            "operator_flux_M_grad_mu"
                        ][index]
                    ),
                    "outer_M_grad_mu": float(
                        fluxes["outer"][wire][
                            "operator_flux_M_grad_mu"
                        ][index]
                    ),
                    "inward_M_grad_mu": float(
                        fluxes["inward"][wire][
                            "operator_flux_M_grad_mu"
                        ][index]
                    ),
                }
    return {
        "sign_convention": (
            "reported operator flux is M*grad(mu); physical material current "
            "is its negative because dc/dt=div(M*grad(mu))"
        ),
        "samples": output,
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

    prerequisites = _load_prerequisites()
    implementation_sha256 = _implementation_hashes()
    source_helper.verify_sources()
    checkpoint_path = source_helper.T100_CHECKPOINT_PATH
    checkpoint_file_before = source_helper._file_state(checkpoint_path)
    checkpoint = np.load(
        checkpoint_path, mmap_mode="r", allow_pickle=False
    )
    checkpoint_fingerprint = array_fingerprint(checkpoint)
    if checkpoint_fingerprint != source_helper.EXPECTED_T100_FINGERPRINT:
        raise RuntimeError("immutable t=100 checkpoint fingerprint mismatch")
    reference_mass = source_helper.EXPECTED_T100_MASS

    base_solver = RoyPseudospectralSolver(
        FROZEN_DEG90.lattice,
        FROZEN_DEG90.parameters,
        fft_workers=FFT_WORKERS,
    )
    input_energy = base_solver.free_energy(checkpoint)
    del base_solver
    gc.collect()

    untreated_static, untreated_static_record, untreated_static_seconds = (
        _static_proposal(
            checkpoint,
            None,
            reference_mass=reference_mass,
            input_energy=input_energy,
        )
    )
    static_cases: dict[str, Any] = {
        "untreated": {
            **untreated_static_record,
            "proposal_seconds": untreated_static_seconds,
        }
    }
    static_passed = all(untreated_static_record["health"].values())
    for name, geometry in (
        ("outer", OUTER_GEOMETRY),
        ("inward", INWARD_GEOMETRY),
    ):
        factor = four_arm_tubular_collar_factor(
            FROZEN_DEG90.lattice,
            geometry,
            radius=6.0,
            interface_width=INTERFACE_WIDTH,
        )
        factor_checks = _factor_checks(factor, geometry)
        proposal, record, seconds = _static_proposal(
            checkpoint,
            factor,
            reference_mass=reference_mass,
            input_energy=input_energy,
        )
        paired = _paired_response_metrics(
            untreated_static,
            proposal,
            reference_mass=reference_mass,
        )
        localization = _localization_metrics(
            checkpoint, factor, untreated_static, proposal
        )
        case_checks = {
            "factor": all(factor_checks.values()),
            "proposal_health": all(record["health"].values()),
            "response_resolved": (
                paired["response_to_ulp_floor_rms_ratio"]
                >= RESPONSE_TO_ULP_RMS_MINIMUM
            ),
            "paired_mass": (
                paired["differential_mass_fraction"] <= MASS_DRIFT_LIMIT
            ),
            "localization": all(localization["checks"].values()),
        }
        static_cases[name] = {
            **record,
            "proposal_seconds": seconds,
            "factor": {
                "minimum": float(np.min(factor)),
                "maximum": float(np.max(factor)),
                "support_cell_fraction": float(
                    np.mean(factor < 1.0 - 1.0e-14)
                ),
                "checks": factor_checks,
            },
            "paired_response": paired,
            "localization": localization,
            "checks": case_checks,
        }
        static_passed = bool(static_passed and all(case_checks.values()))
        del proposal, factor
        gc.collect()
    del untreated_static
    gc.collect()
    if not static_passed:
        report = {
            "schema_version": 1,
            "status": "stopped_after_static_failure",
            "classification": "tubular_position_static_preflight_failed",
            "static_cases": static_cases,
            "accepted_steps_per_case": 0,
            "implementation_sha256": implementation_sha256,
            "wall_seconds": float(time.perf_counter() - started),
        }
        _atomic_json(output / "summary.json", report)
        return report

    fields: dict[str, np.ndarray] = {}
    runs: dict[str, Any] = {}
    fluxes: dict[str, Any] = {}
    for name, geometry in (
        ("untreated", None),
        ("outer", OUTER_GEOMETRY),
        ("inward", INWARD_GEOMETRY),
    ):
        factor = (
            None
            if geometry is None
            else four_arm_tubular_collar_factor(
                FROZEN_DEG90.lattice,
                geometry,
                radius=6.0,
                interface_width=INTERFACE_WIDTH,
            )
        )
        field, case_run, case_flux = _run_case(
            checkpoint,
            factor,
            reference_mass=reference_mass,
            input_energy=input_energy,
            started=started,
        )
        fields[name] = field
        runs[name] = case_run
        fluxes[name] = case_flux
        del factor
        gc.collect()

    phase, contour, profile_arrays = _profile_arrays(fields)
    outer_response = _treatment_profile_response(
        phase,
        contour,
        treated_name="outer",
        geometry=OUTER_GEOMETRY,
    )
    inward_response = _treatment_profile_response(
        phase,
        contour,
        treated_name="inward",
        geometry=INWARD_GEOMETRY,
    )
    flux_report = _flux_boundary_report(fluxes, INWARD_GEOMETRY)

    profile_path = output / "profiles.npz"
    _atomic_npz(profile_path, **profile_arrays)
    flux_arrays: dict[str, np.ndarray] = {}
    for wire in ("first_wire_z", "second_wire_y"):
        flux_arrays[f"{wire}_coordinate"] = fluxes["untreated"][wire][
            "coordinate"
        ]
        for case in CASE_NAMES:
            flux_arrays[f"{wire}_{case}_M_grad_mu"] = fluxes[case][wire][
                "operator_flux_M_grad_mu"
            ]
    flux_path = output / "flux-profiles.npz"
    _atomic_npz(flux_path, **flux_arrays)

    inward_pooled = inward_response["pooled"]
    outer_pooled = outer_response["pooled"]
    inward_contour_positive = all(
        inward_pooled["contour_radius"][level][
            "all_four_arm_means_positive"
        ]
        and inward_pooled["contour_radius"][level]["mean"] > 0.0
        for level in ("0.45", "0.50", "0.55")
    )
    outer_contour_negative = all(
        outer_pooled["contour_radius"][level]["mean"] < 0.0
        for level in ("0.45", "0.50", "0.55")
    )
    numerical_checks = {
        "static_preflight": static_passed,
        "all_case_step_counts": all(
            len(runs[name]["proposal_seconds"]) == ACCEPTED_STEPS
            for name in CASE_NAMES
        ),
        "all_case_health": all(
            runs[name]["all_step_health"] for name in CASE_NAMES
        ),
        "no_case_milestone_pinch": all(
            runs[name]["no_milestone_pinch"] for name in CASE_NAMES
        ),
        "outer_contours_valid": outer_response["contour_valid"],
        "inward_contours_valid": inward_response["contour_valid"],
        "wall_cap": time.perf_counter() - started <= WALL_CAP_SECONDS,
    }
    sign_checks = {
        "outer_phase_response_negative": (
            outer_pooled["phase_radius_mean"] < 0.0
        ),
        "outer_contour_responses_negative": outer_contour_negative,
        "inward_phase_response_positive": (
            inward_pooled["phase_radius_mean"] > 0.0
        ),
        "inward_phase_all_four_arms_positive": (
            inward_pooled["phase_all_four_arm_means_positive"]
        ),
        "inward_contours_all_levels_and_arms_positive": (
            inward_contour_positive
        ),
        "inward_true_transitions_no_pronounced_deficit": (
            inward_pooled["minimum_true_transition_response"]
            >= TRANSITION_DEFICIT_LIMIT
        ),
    }
    sign_reversal = bool(
        all(numerical_checks.values()) and all(sign_checks.values())
    )

    checkpoint_fingerprint_after = array_fingerprint(checkpoint)
    checkpoint_file_after = source_helper._file_state(checkpoint_path)
    integrity_checks = {
        "checkpoint_array_unchanged": (
            checkpoint_fingerprint_after == checkpoint_fingerprint
        ),
        "checkpoint_file_unchanged": (
            checkpoint_file_after == checkpoint_file_before
        ),
        "implementation_files_unchanged_during_run": (
            _implementation_hashes() == implementation_sha256
        ),
    }
    all_proposal_seconds = [
        seconds
        for case in CASE_NAMES
        for seconds in runs[case]["proposal_seconds"]
    ]
    report = {
        "schema_version": 1,
        "status": "completed",
        "classification": (
            "tubular_position_short_sign_reversal_observed"
            if sign_reversal
            else "tubular_position_short_no_consistent_sign_reversal"
        ),
        "numerical_preflight_passed": bool(
            all(numerical_checks.values())
            and all(integrity_checks.values())
        ),
        "positional_sign_reversal_observed": sign_reversal,
        "scope": {
            "discarded_static_proposals": 3,
            "accepted_steps_per_case": ACCEPTED_STEPS,
            "accepted_proposals_total": (
                ACCEPTED_STEPS * len(CASE_NAMES)
            ),
            "production_authorized": False,
            "long_run_launched": False,
        },
        "prerequisites": {
            "static_summary_sha256": prerequisites["static_sha256"],
            "short_summary_sha256": prerequisites["short_sha256"],
            "recorded_planar_outer_pooled_phase_radius_response": (
                prerequisites[
                    "recorded_planar_outer_pooled_phase_radius_response"
                ]
            ),
        },
        "source": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_file_before["sha256"],
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "input_energy": input_energy,
            "reference_mass": reference_mass,
        },
        "geometry": {
            "representation": (
                "fixed Eulerian tubular surface shell; axial raised cosine"
            ),
            "radial_shell_support": [
                6.0 - 2.0 * INTERFACE_WIDTH,
                6.0 + 2.0 * INTERFACE_WIDTH,
            ],
            "radial_shell_plateau": [
                6.0 - INTERFACE_WIDTH,
                6.0 + INTERFACE_WIDTH,
            ],
            "outer": OUTER_GEOMETRY.to_dict(),
            "inward": INWARD_GEOMETRY.to_dict(),
        },
        "static_cases": static_cases,
        "runs": runs,
        "profile_response": {
            "outer": outer_response,
            "inward": inward_response,
        },
        "flux_boundary_report": flux_report,
        "decision": {
            "numerical_checks": numerical_checks,
            "sign_checks": sign_checks,
            "positional_sign_reversal_observed": sign_reversal,
            "interpretation_limit": (
                "16-step mechanistic diagnostic; no lifetime claim"
            ),
        },
        "artifacts": {
            "profiles_npz": {
                "path": str(profile_path),
                "sha256": _sha256_path(profile_path),
            },
            "flux_profiles_npz": {
                "path": str(flux_path),
                "sha256": _sha256_path(flux_path),
            },
        },
        "implementation_sha256": implementation_sha256,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "resource_preflight": resource,
        "execution": {
            "median_proposal_seconds": float(
                np.median(all_proposal_seconds)
            ),
            "maximum_proposal_seconds": float(
                np.max(all_proposal_seconds)
            ),
            "accepted_proposal_count": len(all_proposal_seconds),
            "wall_seconds": float(time.perf_counter() - started),
        },
        "integrity_checks": integrity_checks,
    }
    _atomic_json(output / "summary.json", report)
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
                "numerical_preflight_passed": report.get(
                    "numerical_preflight_passed", False
                ),
                "positional_sign_reversal_observed": report.get(
                    "positional_sign_reversal_observed", False
                ),
                "decision": report.get("decision"),
                "execution": report.get("execution"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
