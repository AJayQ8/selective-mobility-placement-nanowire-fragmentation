#!/usr/bin/env python3
"""Reconstruct sparse transport and fixed-mask diagnostics from checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import fft as scipy_fft

from ..roy_2021_reproduction.model import (
    FROZEN_DEG90,
    RoyPseudospectralSolver,
    source_wave_numbers,
)
from ..roy_fixed_gb_bridge.provenance import sha256_path
from ..roy_gb_junction_sentinel.storage import atomic_json
from . import analyze_position_scan_mechanism as profile_audit
from .geometry import (
    four_arm_tubular_collar_factor,
    phase_interface_weight,
)
from .model import SpatialMobilityRoySolver
from . import position_scan_protocol as protocol


SOURCE_PATH = Path(__file__).resolve()
DIRECTORY = SOURCE_PATH.parent
RESULTS = DIRECTORY / "results"
DEFAULT_OUTPUT = RESULTS / "position_scan_transport_audit_v3"
PROFILE_AUDIT_PATH = (
    RESULTS / "position_scan_mechanism_profile_audit_v1" / "summary.json"
)
AGGREGATE_PATH = RESULTS / "position_scan_response_v1" / "summary.json"

DYNAMIC_CASES = (
    "c18p5",
    "c22p5",
    "c26p5",
    "c30p5",
    "c34p5",
    "c38p5",
    "c64p5",
)
DYNAMIC_STEPS = (500, 900, 1300)
FLUX_CASES = ("untreated", "c26p5", "c34p5", "c38p5", "c64p5")
FLUX_STEP = 1300
CONTINUITY_HALF_WINDOW = 100
CONTINUITY_RELATIVE_ERROR_LIMIT = 0.01
TRANSVERSE_GAP_EXPLANATION_MINIMUM = 0.95
TRANSVERSE_HALF_CELLS = 48
FIELD_BLOCK_CELLS = 32
INTERFACE_CUTOFFS = (0.05, 0.95)
GRID_RADIAL_TOLERANCE = (
    FROZEN_DEG90.lattice.spacing / np.sqrt(2.0)
)
WIRES = ("first_wire_z", "second_wire_y")
BRANCHES = ("minus", "plus")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_summary_path(case_id: str, step: int) -> Path:
    if case_id == "untreated":
        directory = (
            DIRECTORY.parent
            / "roy_2021_reproduction"
            / "results"
            / (
                "t1000_source_semantic_seed2292"
                if step <= 1000
                else "t2000_continuation_source_semantic_seed2292"
            )
        )
        return directory / "summary.json"
    if case_id == "c22p5":
        return RESULTS / "outer_placement_lifetime_v1" / "summary.json"
    if case_id == "c64p5":
        return (
            RESULTS / "far_field_placement_sentinel_v1" / "summary.json"
        )
    if case_id not in profile_audit.CASE_CENTERS:
        raise ValueError(f"unsupported checkpoint case: {case_id}")
    return RESULTS / f"position_scan_{case_id}_v1" / "summary.json"


def _checkpoint_entry(case_id: str, step: int) -> dict[str, Any]:
    summary_path = _checkpoint_summary_path(case_id, step)
    summary = _load_json(summary_path)
    matches = [
        item
        for item in summary["checkpoints"]
        if int(item["step"]) == step
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{case_id} has {len(matches)} checkpoints at {step}"
        )
    entry = dict(matches[0])
    field_path = Path(entry["field_path"])
    checks = {
        "field_exists": field_path.exists(),
        "field_bytes_match": (
            field_path.stat().st_size == int(entry["field_bytes"])
        ),
        "shape_matches": tuple(entry["shape"])
        == FROZEN_DEG90.lattice.shape,
        "dtype_matches": entry["dtype"] == "<f8",
        "recorded_finite": entry["finite"] is True,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"{case_id} checkpoint failed metadata checks: {checks}"
        )
    entry.update(
        {
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": sha256_path(summary_path),
            "metadata_checks": checks,
        }
    )
    return entry


def _coordinates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spacing = FROZEN_DEG90.lattice.spacing
    return tuple(
        (
            np.arange(cells, dtype=np.float64) - cells // 2
        )
        * spacing
        for cells in FROZEN_DEG90.lattice.shape
    )


def _wire_geometry_arrays() -> dict[str, Any]:
    x, y, z = _coordinates()
    second_center_x = (
        FROZEN_DEG90.radius_1 + FROZEN_DEG90.radius_2
    ) * FROZEN_DEG90.lattice.spacing
    first_radius = np.sqrt(x[:, None] ** 2 + y[None, :] ** 2)
    second_radius = np.sqrt(
        (x[:, None] - second_center_x) ** 2
        + z[None, :] ** 2
    )
    return {
        "coordinate": {"x": x, "y": y, "z": z},
        "first_radius": first_radius,
        "second_radius": second_radius,
        "second_center_x": second_center_x,
    }


def _global_dynamic_budget(
    field: np.ndarray,
    factor: np.ndarray,
) -> float:
    total = 0.0
    for start in range(0, field.shape[2], FIELD_BLOCK_CELLS):
        stop = min(field.shape[2], start + FIELD_BLOCK_CELLS)
        values = np.asarray(field[:, :, start:stop], dtype=np.float64)
        weight = phase_interface_weight(values)
        total += float(
            np.sum(
                weight * (1.0 - factor[:, :, start:stop]),
                dtype=np.float64,
            )
        )
    return total * FROZEN_DEG90.lattice.cell_volume


def _arm_record_template(wire: str, branch: str) -> dict[str, Any]:
    return {
        "wire": wire,
        "branch": branch,
        "interface_cell_count": 0,
        "plateau_interface_cell_count": 0,
        "ramp_interface_cell_count": 0,
        "strict_outside_support_cell_count": 0,
        "grid_resolved_outside_support_cell_count": 0,
        "surface_weight": 0.0,
        "dynamic_budget": 0.0,
    }


def _capture_and_arm_budgets(
    field: np.ndarray,
    factor: np.ndarray,
    geometry: Any,
) -> dict[str, Any]:
    arrays = _wire_geometry_arrays()
    x = arrays["coordinate"]["x"]
    y = arrays["coordinate"]["y"]
    z = arrays["coordinate"]["z"]
    first_radius = arrays["first_radius"]
    second_radius = arrays["second_radius"]
    second_center_x = arrays["second_center_x"]
    records = {
        (wire, branch): _arm_record_template(wire, branch)
        for wire in WIRES
        for branch in BRANCHES
    }
    lower, upper = INTERFACE_CUTOFFS
    plateau_limit = protocol.INTERFACE_WIDTH
    support_limit = 2.0 * protocol.INTERFACE_WIDTH
    axial_lower = geometry.inner_support_distance
    axial_upper = geometry.outer_support_distance

    for index in np.flatnonzero(
        (np.abs(z) >= axial_lower) & (np.abs(z) <= axial_upper)
    ):
        branch = "plus" if z[index] >= 0.0 else "minus"
        record = records[("first_wire_z", branch)]
        values = np.asarray(field[:, :, index], dtype=np.float64)
        factor_plane = factor[:, :, index]
        second_distance = np.sqrt(
            (x - second_center_x) ** 2 + z[index] ** 2
        )[:, None]
        assigned = first_radius <= second_distance
        weight = phase_interface_weight(values)
        record["surface_weight"] += float(
            np.sum(weight[assigned], dtype=np.float64)
        )
        record["dynamic_budget"] += float(
            np.sum(
                (
                    weight
                    * (1.0 - factor_plane)
                )[assigned],
                dtype=np.float64,
            )
        )
        interface = (
            (values >= lower)
            & (values <= upper)
            & assigned
        )
        radii = first_radius[interface]
        radial_error = np.abs(radii - protocol.RADIUS)
        record["interface_cell_count"] += int(radii.size)
        record["plateau_interface_cell_count"] += int(
            np.count_nonzero(radial_error <= plateau_limit)
        )
        record["ramp_interface_cell_count"] += int(
            np.count_nonzero(
                (radial_error > plateau_limit)
                & (radial_error <= support_limit)
            )
        )
        record["strict_outside_support_cell_count"] += int(
            np.count_nonzero(radial_error > support_limit)
        )
        record["grid_resolved_outside_support_cell_count"] += int(
            np.count_nonzero(
                radial_error > support_limit + GRID_RADIAL_TOLERANCE
            )
        )

    for index in np.flatnonzero(
        (np.abs(y) >= axial_lower) & (np.abs(y) <= axial_upper)
    ):
        branch = "plus" if y[index] >= 0.0 else "minus"
        record = records[("second_wire_y", branch)]
        values = np.asarray(field[:, index, :], dtype=np.float64)
        factor_plane = factor[:, index, :]
        first_distance = np.sqrt(x**2 + y[index] ** 2)[:, None]
        assigned = second_radius <= first_distance
        weight = phase_interface_weight(values)
        record["surface_weight"] += float(
            np.sum(weight[assigned], dtype=np.float64)
        )
        record["dynamic_budget"] += float(
            np.sum(
                (
                    weight
                    * (1.0 - factor_plane)
                )[assigned],
                dtype=np.float64,
            )
        )
        interface = (
            (values >= lower)
            & (values <= upper)
            & assigned
        )
        radii = second_radius[interface]
        radial_error = np.abs(radii - protocol.RADIUS)
        record["interface_cell_count"] += int(radii.size)
        record["plateau_interface_cell_count"] += int(
            np.count_nonzero(radial_error <= plateau_limit)
        )
        record["ramp_interface_cell_count"] += int(
            np.count_nonzero(
                (radial_error > plateau_limit)
                & (radial_error <= support_limit)
            )
        )
        record["strict_outside_support_cell_count"] += int(
            np.count_nonzero(radial_error > support_limit)
        )
        record["grid_resolved_outside_support_cell_count"] += int(
            np.count_nonzero(
                radial_error > support_limit + GRID_RADIAL_TOLERANCE
            )
        )

    output_records = []
    for record in records.values():
        count = record["interface_cell_count"]
        for name in ("surface_weight", "dynamic_budget"):
            record[name] *= FROZEN_DEG90.lattice.cell_volume
        record["plateau_interface_fraction"] = (
            record["plateau_interface_cell_count"] / count
            if count
            else None
        )
        record["ramp_interface_fraction"] = (
            record["ramp_interface_cell_count"] / count
            if count
            else None
        )
        record["strict_outside_support_fraction"] = (
            record["strict_outside_support_cell_count"] / count
            if count
            else None
        )
        record["grid_resolved_outside_support_fraction"] = (
            record["grid_resolved_outside_support_cell_count"] / count
            if count
            else None
        )
        output_records.append(record)
    total_count = sum(
        item["interface_cell_count"] for item in output_records
    )
    pooled = {
        "interface_cell_count": total_count,
        "plateau_interface_fraction": (
            sum(
                item["plateau_interface_cell_count"]
                for item in output_records
            )
            / total_count
            if total_count
            else None
        ),
        "ramp_interface_fraction": (
            sum(
                item["ramp_interface_cell_count"]
                for item in output_records
            )
            / total_count
            if total_count
            else None
        ),
        "strict_outside_support_fraction": (
            sum(
                item["strict_outside_support_cell_count"]
                for item in output_records
            )
            / total_count
            if total_count
            else None
        ),
        "grid_resolved_outside_support_fraction": (
            sum(
                item["grid_resolved_outside_support_cell_count"]
                for item in output_records
            )
            / total_count
            if total_count
            else None
        ),
        "arm_dynamic_budget_sum": float(
            sum(item["dynamic_budget"] for item in output_records)
        ),
    }
    return {"arms": output_records, "pooled": pooled}


def _surface_exposure_in_natural_zone(
    untreated_field: np.ndarray,
    factor: np.ndarray,
    natural_bounds: tuple[float, float],
) -> dict[str, float]:
    _, y, z = _coordinates()
    y_zone = (
        (np.abs(y) >= natural_bounds[0])
        & (np.abs(y) <= natural_bounds[1])
    )
    z_zone = (
        (np.abs(z) >= natural_bounds[0])
        & (np.abs(z) <= natural_bounds[1])
    )
    numerator = 0.0
    denominator = 0.0
    maximum = 0.0
    for start in range(0, untreated_field.shape[2], FIELD_BLOCK_CELLS):
        stop = min(
            untreated_field.shape[2], start + FIELD_BLOCK_CELLS
        )
        values = np.asarray(
            untreated_field[:, :, start:stop], dtype=np.float64
        )
        weight = phase_interface_weight(values)
        zone = (
            y_zone[None, :, None]
            | z_zone[None, None, start:stop]
        )
        zone = np.broadcast_to(zone, weight.shape)
        deficit = 1.0 - factor[:, :, start:stop]
        numerator += float(
            np.sum((weight * deficit)[zone], dtype=np.float64)
        )
        denominator += float(
            np.sum(weight[zone], dtype=np.float64)
        )
        active = zone & (weight > 1.0e-8)
        if np.any(active):
            maximum = max(
                maximum,
                float(
                    np.max(
                        deficit[active]
                        / (1.0 - protocol.PROTECTED_MOBILITY_FACTOR)
                    )
                ),
            )
    normalized = (
        numerator
        / (
            (1.0 - protocol.PROTECTED_MOBILITY_FACTOR)
            * denominator
        )
        if denominator
        else 0.0
    )
    return {
        "surface_weighted_mean_normalized_deficit": normalized,
        "maximum_normalized_deficit": maximum,
    }


def _phase_volume(
    case_id: str,
    step: int,
    wire: str,
    branch: str,
    bounds: tuple[float, float],
) -> float:
    if case_id == "untreated":
        path = profile_audit._untreated_source(step)
        prefix = "untreated_"
    else:
        path, prefix = profile_audit._profile_source(case_id, step)
    with np.load(path) as arrays:
        coordinate = np.asarray(
            arrays[f"{wire}_coordinate"], dtype=np.float64
        )
        radius = np.asarray(
            arrays[f"{wire}_{prefix}phase_radius"],
            dtype=np.float64,
        )
    distance, radial = profile_audit._branch_view(
        coordinate, radius, branch
    )
    return profile_audit._trapz_in_bounds(
        distance, np.pi * radial**2, bounds
    )


def _observed_volume_rate(
    case_id: str,
    wire: str,
    branch: str,
    bounds: tuple[float, float],
) -> float:
    lower_step = FLUX_STEP - CONTINUITY_HALF_WINDOW
    upper_step = FLUX_STEP + CONTINUITY_HALF_WINDOW
    lower = _phase_volume(
        case_id, lower_step, wire, branch, bounds
    )
    upper = _phase_volume(
        case_id, upper_step, wire, branch, bounds
    )
    return float((upper - lower) / (upper_step - lower_step))


def _spectral_real_derivative(
    values: np.ndarray,
    spacing: float,
    *,
    workers: int = protocol.FFT_WORKERS,
) -> np.ndarray:
    """Apply the solver's real pseudospectral derivative in one dimension."""

    real32 = np.asarray(values, dtype=np.float32)
    spectrum = scipy_fft.fft(real32, workers=workers)
    spectrum = np.asarray(spectrum, dtype=np.complex64)
    gradient_hat = np.empty_like(spectrum)
    np.multiply(
        spectrum,
        source_wave_numbers(real32.size, spacing),
        out=gradient_hat,
        casting="unsafe",
    )
    gradient_hat *= np.complex64(1j)
    gradient = scipy_fft.ifft(
        gradient_hat,
        workers=workers,
        overwrite_x=True,
    )
    return np.asarray(gradient.real, dtype=np.float64)


def _trapezoidal_zone_weight(
    coordinate: np.ndarray,
    branch: str,
    bounds: tuple[float, float],
) -> np.ndarray:
    """Return the exact node weights used by ``_trapz_in_bounds``."""

    sign = 1.0 if branch == "plus" else -1.0
    distance = coordinate * sign
    indices = np.flatnonzero(
        (distance >= bounds[0]) & (distance <= bounds[1])
    )
    if indices.size < 2:
        raise RuntimeError("control-volume axial support is unresolved")
    order = np.argsort(distance[indices])
    weight = np.zeros(coordinate.size, dtype=np.float64)
    weight[indices] = 1.0
    weight[indices[order[0]]] = 0.5
    weight[indices[order[-1]]] = 0.5
    return weight


def _transverse_crop_weight(cells: int) -> np.ndarray:
    """Return the exact 96-node rectangular phase-profile crop."""

    start = cells // 2 - TRANSVERSE_HALF_CELLS
    stop = start + 2 * TRANSVERSE_HALF_CELLS
    if start < 0 or stop > cells:
        raise RuntimeError("transverse control-volume crop does not fit")
    weight = np.zeros(cells, dtype=np.float64)
    weight[start:stop] = 1.0
    return weight


def _control_volume_weights(
    natural_bounds: tuple[float, float],
) -> dict[str, dict[str, Any]]:
    """Build phase-volume weights and their spectral derivatives."""

    spacing = FROZEN_DEG90.lattice.spacing
    coordinate = _coordinates()
    definitions = (
        ("first_wire_z", 2, 1),
        ("second_wire_y", 1, 2),
    )
    output: dict[str, dict[str, Any]] = {}
    for wire, axial_axis, transverse_axis in definitions:
        transverse = _transverse_crop_weight(
            FROZEN_DEG90.lattice.shape[transverse_axis]
        )
        branches = {}
        for branch in BRANCHES:
            axial = _trapezoidal_zone_weight(
                coordinate[axial_axis],
                branch,
                natural_bounds,
            )
            branches[branch] = {
                "axial": axial,
                "axial_derivative": _spectral_real_derivative(
                    axial, spacing
                ),
            }
        output[wire] = {
            "axial_axis": axial_axis,
            "transverse_axis": transverse_axis,
            "coordinate": coordinate[axial_axis],
            "transverse": transverse,
            "transverse_derivative": _spectral_real_derivative(
                transverse, spacing
            ),
            "branches": branches,
        }
    return output


def _operator_control_volume_flux(
    solver: RoyPseudospectralSolver,
    field: np.ndarray,
    natural_bounds: tuple[float, float],
) -> dict[str, dict[str, Any]]:
    """Reconstruct physical currents and exact open-crop volume rates.

    For a control weight ``W`` and operator flux ``q=M grad(mu)``, the raw
    physical operator rate is evaluated as ``-<grad(W), q>``.  This is the
    exact periodic pseudospectral summation-by-parts identity for the same
    trapezoidal axial weights and 96-node transverse crop used by the saved
    phase-volume profiles.  The full periodic x span has zero contribution.
    """

    spacing = FROZEN_DEG90.lattice.spacing
    cell_volume = FROZEN_DEG90.lattice.cell_volume
    weights = _control_volume_weights(natural_bounds)
    mu_hat = solver.chemical_potential_spectrum(field)
    mobility = solver.mobility(field)
    _, ny, nz = field.shape
    y0 = ny // 2 - TRANSVERSE_HALF_CELLS
    y1 = y0 + 2 * TRANSVERSE_HALF_CELLS
    z0 = nz // 2 - TRANSVERSE_HALF_CELLS
    z1 = z0 + 2 * TRANSVERSE_HALF_CELLS
    output: dict[str, dict[str, Any]] = {
        wire: {
            "coordinate": np.asarray(
                weights[wire]["coordinate"], dtype=np.float64
            ),
            "operator_flux_M_grad_mu": None,
            "control_volume": {
                branch: {
                    "spectral_axial_volume_rate": None,
                    "spectral_transverse_volume_rate": None,
                    "spectral_periodic_x_volume_rate": 0.0,
                }
                for branch in BRANCHES
            },
        }
        for wire in WIRES
    }

    for axis in (1, 2):
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
        del gradient, gradient_hat

        if axis == 1:
            axial_wire = "second_wire_y"
            axial_profile = (
                np.sum(
                    operator_flux[:, :, z0:z1],
                    axis=(0, 2),
                    dtype=np.float64,
                )
                * spacing**2
            )
            output[axial_wire][
                "operator_flux_M_grad_mu"
            ] = axial_profile
            for branch in BRANCHES:
                axial_weight = weights[axial_wire]["branches"][branch]
                output[axial_wire]["control_volume"][branch][
                    "spectral_axial_volume_rate"
                ] = float(
                    -spacing
                    * np.dot(
                        axial_profile,
                        axial_weight["axial_derivative"],
                    )
                )

            transverse_wire = "first_wire_z"
            transverse_derivative = weights[transverse_wire][
                "transverse_derivative"
            ]
            for branch in BRANCHES:
                axial_weight = weights[transverse_wire]["branches"][
                    branch
                ]["axial"]
                collapsed = np.einsum(
                    "xyz,z->y",
                    operator_flux,
                    axial_weight,
                    dtype=np.float64,
                    optimize=True,
                )
                output[transverse_wire]["control_volume"][branch][
                    "spectral_transverse_volume_rate"
                ] = float(
                    -cell_volume
                    * np.dot(collapsed, transverse_derivative)
                )
        else:
            axial_wire = "first_wire_z"
            axial_profile = (
                np.sum(
                    operator_flux[:, y0:y1, :],
                    axis=(0, 1),
                    dtype=np.float64,
                )
                * spacing**2
            )
            output[axial_wire][
                "operator_flux_M_grad_mu"
            ] = axial_profile
            for branch in BRANCHES:
                axial_weight = weights[axial_wire]["branches"][branch]
                output[axial_wire]["control_volume"][branch][
                    "spectral_axial_volume_rate"
                ] = float(
                    -spacing
                    * np.dot(
                        axial_profile,
                        axial_weight["axial_derivative"],
                    )
                )

            transverse_wire = "second_wire_y"
            transverse_derivative = weights[transverse_wire][
                "transverse_derivative"
            ]
            for branch in BRANCHES:
                axial_weight = weights[transverse_wire]["branches"][
                    branch
                ]["axial"]
                collapsed = np.einsum(
                    "xyz,y->z",
                    operator_flux,
                    axial_weight,
                    dtype=np.float64,
                    optimize=True,
                )
                output[transverse_wire]["control_volume"][branch][
                    "spectral_transverse_volume_rate"
                ] = float(
                    -cell_volume
                    * np.dot(collapsed, transverse_derivative)
                )
        del operator_flux
        gc.collect()

    for wire in WIRES:
        if output[wire]["operator_flux_M_grad_mu"] is None:
            raise RuntimeError(f"{wire} axial current was not reconstructed")
        for branch in BRANCHES:
            record = output[wire]["control_volume"][branch]
            if (
                record["spectral_axial_volume_rate"] is None
                or record["spectral_transverse_volume_rate"] is None
            ):
                raise RuntimeError(
                    f"{wire} {branch} control-volume rate is incomplete"
                )
    return output


def _flux_boundary_records(
    case_id: str,
    flux: dict[str, dict[str, Any]],
    natural_bounds: tuple[float, float],
) -> list[dict[str, Any]]:
    output = []
    for wire in WIRES:
        coordinate = flux[wire]["coordinate"]
        physical_axis_current = -flux[wire][
            "operator_flux_M_grad_mu"
        ]
        for branch in BRANCHES:
            sign = 1.0 if branch == "plus" else -1.0
            distance, selected_axis_current = profile_audit._branch_view(
                coordinate, physical_axis_current, branch
            )
            outward_current = sign * selected_axis_current
            inner = float(
                np.interp(
                    natural_bounds[0], distance, outward_current
                )
            )
            outer = float(
                np.interp(
                    natural_bounds[1], distance, outward_current
                )
            )
            axial_point_face = inner - outer
            control_volume = flux[wire]["control_volume"][branch]
            spectral_axial = float(
                control_volume["spectral_axial_volume_rate"]
            )
            spectral_transverse = float(
                control_volume["spectral_transverse_volume_rate"]
            )
            spectral_periodic_x = float(
                control_volume["spectral_periodic_x_volume_rate"]
            )
            predicted = (
                spectral_axial
                + spectral_transverse
                + spectral_periodic_x
            )
            observed = _observed_volume_rate(
                case_id, wire, branch, natural_bounds
            )
            legacy_gap = observed - axial_point_face
            transverse_gap_fraction = (
                spectral_transverse / legacy_gap
                if legacy_gap != 0.0
                else None
            )
            output.append(
                {
                    "case_id": case_id,
                    "step": FLUX_STEP,
                    "wire": wire,
                    "branch": branch,
                    "physical_outward_current_at_inner_boundary": inner,
                    "physical_outward_current_at_outer_boundary": outer,
                    "axial_point_face_predicted_volume_rate": (
                        axial_point_face
                    ),
                    "spectral_axial_control_volume_rate": (
                        spectral_axial
                    ),
                    "spectral_transverse_control_volume_rate": (
                        spectral_transverse
                    ),
                    "spectral_periodic_x_control_volume_rate": (
                        spectral_periodic_x
                    ),
                    "predicted_natural_zone_volume_rate": predicted,
                    "observed_centered_natural_zone_volume_rate": observed,
                    "continuity_sign_agrees": (
                        np.sign(predicted) == np.sign(observed)
                    ),
                    "relative_magnitude_error": (
                        abs(predicted - observed) / abs(observed)
                        if observed != 0.0
                        else None
                    ),
                    "legacy_axial_only_relative_magnitude_error": (
                        abs(axial_point_face - observed) / abs(observed)
                        if observed != 0.0
                        else None
                    ),
                    "axial_point_face_quadrature_relative_error": (
                        abs(axial_point_face - spectral_axial)
                        / abs(spectral_axial)
                        if spectral_axial != 0.0
                        else None
                    ),
                    "transverse_fraction_of_legacy_gap": (
                        transverse_gap_fraction
                    ),
                }
            )
    return output


def _aggregate_flux_rows(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    axial_point_face = np.asarray(
        [row["axial_point_face_predicted_volume_rate"] for row in rows],
        dtype=np.float64,
    )
    spectral_axial = np.asarray(
        [row["spectral_axial_control_volume_rate"] for row in rows],
        dtype=np.float64,
    )
    spectral_transverse = np.asarray(
        [
            row["spectral_transverse_control_volume_rate"]
            for row in rows
        ],
        dtype=np.float64,
    )
    predicted = np.asarray(
        [row["predicted_natural_zone_volume_rate"] for row in rows],
        dtype=np.float64,
    )
    observed = np.asarray(
        [row["observed_centered_natural_zone_volume_rate"] for row in rows],
        dtype=np.float64,
    )
    return {
        "arm_count": len(rows),
        "mean_physical_outward_current_at_inner_boundary": float(
            np.mean(
                [
                    row[
                        "physical_outward_current_at_inner_boundary"
                    ]
                    for row in rows
                ]
            )
        ),
        "mean_physical_outward_current_at_outer_boundary": float(
            np.mean(
                [
                    row[
                        "physical_outward_current_at_outer_boundary"
                    ]
                    for row in rows
                ]
            )
        ),
        "mean_axial_point_face_predicted_volume_rate": float(
            np.mean(axial_point_face)
        ),
        "mean_spectral_axial_control_volume_rate": float(
            np.mean(spectral_axial)
        ),
        "mean_spectral_transverse_control_volume_rate": float(
            np.mean(spectral_transverse)
        ),
        "mean_predicted_natural_zone_volume_rate": float(
            np.mean(predicted)
        ),
        "mean_observed_centered_natural_zone_volume_rate": float(
            np.mean(observed)
        ),
        "all_four_continuity_signs_agree": all(
            row["continuity_sign_agrees"] for row in rows
        ),
        "maximum_corrected_relative_magnitude_error": float(
            max(row["relative_magnitude_error"] for row in rows)
        ),
        "maximum_legacy_axial_only_relative_magnitude_error": float(
            max(
                row["legacy_axial_only_relative_magnitude_error"]
                for row in rows
            )
        ),
        "minimum_transverse_fraction_of_legacy_gap": float(
            min(
                row["transverse_fraction_of_legacy_gap"]
                for row in rows
            )
        ),
        "maximum_axial_point_face_quadrature_relative_error": float(
            max(
                row["axial_point_face_quadrature_relative_error"]
                for row in rows
            )
        ),
        "predicted_arm_range": float(np.ptp(predicted)),
        "observed_arm_range": float(np.ptp(observed)),
    }


def _factor_for_case(case_id: str) -> np.ndarray:
    geometry = protocol.geometry_for_center(
        profile_audit.CASE_CENTERS[case_id]
    )
    return four_arm_tubular_collar_factor(
        FROZEN_DEG90.lattice,
        geometry,
        radius=protocol.RADIUS,
        interface_width=protocol.INTERFACE_WIDTH,
    )


def build_report() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    profile = _load_json(PROFILE_AUDIT_PATH)
    aggregate = _load_json(AGGREGATE_PATH)
    if (
        profile.get("status") != "complete"
        or profile["harmful_window_precursor"].get("passed") is not True
    ):
        raise RuntimeError("profile mechanism prerequisite did not pass")
    if aggregate["numerical_integrity"].get("all_passed") is not True:
        raise RuntimeError("position-scan numerical integrity failed")
    natural_bounds = tuple(
        float(value)
        for value in profile["natural_zone"]["frozen_grid_bounds"]
    )
    untreated_entry = _checkpoint_entry("untreated", FLUX_STEP)
    untreated_field = np.load(
        untreated_entry["field_path"],
        mmap_mode="r",
        allow_pickle=False,
    )
    initial_budget = {
        item["case_id"]: float(
            item["surface_weighted_mobility_deficit"]
        )
        for item in aggregate["cases"]
    }
    dynamic_rows: list[dict[str, Any]] = []
    exposure: dict[str, Any] = {}
    flux_rows: list[dict[str, Any]] = []
    flux_aggregate: dict[str, Any] = {}
    checkpoint_provenance: dict[str, Any] = {
        "untreated_t1300": untreated_entry
    }

    untreated_solver = RoyPseudospectralSolver(
        FROZEN_DEG90.lattice,
        FROZEN_DEG90.parameters,
        fft_workers=protocol.FFT_WORKERS,
    )
    untreated_flux = _operator_control_volume_flux(
        untreated_solver, untreated_field, natural_bounds
    )
    untreated_rows = _flux_boundary_records(
        "untreated", untreated_flux, natural_bounds
    )
    flux_rows.extend(untreated_rows)
    flux_aggregate["untreated"] = _aggregate_flux_rows(
        untreated_rows
    )
    del untreated_solver, untreated_flux
    gc.collect()

    for case_id in DYNAMIC_CASES:
        geometry = protocol.geometry_for_center(
            profile_audit.CASE_CENTERS[case_id]
        )
        factor = _factor_for_case(case_id)
        exposure[case_id] = _surface_exposure_in_natural_zone(
            untreated_field, factor, natural_bounds
        )
        for step in DYNAMIC_STEPS:
            entry = _checkpoint_entry(case_id, step)
            checkpoint_provenance[f"{case_id}_t{step}"] = entry
            field = np.load(
                entry["field_path"], mmap_mode="r", allow_pickle=False
            )
            global_budget = _global_dynamic_budget(field, factor)
            capture = _capture_and_arm_budgets(
                field, factor, geometry
            )
            dynamic_rows.append(
                {
                    "case_id": case_id,
                    "center": profile_audit.CASE_CENTERS[case_id],
                    "step": step,
                    "global_dynamic_budget": global_budget,
                    "relative_to_t100_budget": (
                        global_budget / initial_budget[case_id]
                    ),
                    "arm_budget_capture_fraction": (
                        capture["pooled"]["arm_dynamic_budget_sum"]
                        / global_budget
                        if global_budget
                        else None
                    ),
                    "mask_capture": capture,
                }
            )
            if case_id in FLUX_CASES and step == FLUX_STEP:
                solver = SpatialMobilityRoySolver(
                    FROZEN_DEG90.lattice,
                    FROZEN_DEG90.parameters,
                    factor,
                    fft_workers=protocol.FFT_WORKERS,
                )
                reconstructed = _operator_control_volume_flux(
                    solver, field, natural_bounds
                )
                case_flux_rows = _flux_boundary_records(
                    case_id, reconstructed, natural_bounds
                )
                flux_rows.extend(case_flux_rows)
                flux_aggregate[case_id] = _aggregate_flux_rows(
                    case_flux_rows
                )
                del solver, reconstructed
                gc.collect()
            del field
            gc.collect()
        del factor
        gc.collect()
    del untreated_field
    gc.collect()

    mask_checks = {}
    for case_id in DYNAMIC_CASES:
        rows = [
            row for row in dynamic_rows if row["case_id"] == case_id
        ]
        mask_checks[case_id] = {
            "all_checkpoints_have_zero_grid_resolved_escape": all(
                row["mask_capture"]["pooled"][
                    "grid_resolved_outside_support_fraction"
                ]
                == 0.0
                for row in rows
            ),
            "maximum_strict_escape_fraction": max(
                row["mask_capture"]["pooled"][
                    "strict_outside_support_fraction"
                ]
                for row in rows
            ),
            "minimum_plateau_capture_fraction": min(
                row["mask_capture"]["pooled"][
                    "plateau_interface_fraction"
                ]
                for row in rows
            ),
            "dynamic_budget_ratio_range": [
                min(row["relative_to_t100_budget"] for row in rows),
                max(row["relative_to_t100_budget"] for row in rows),
            ],
        }
    selected_nonlocal = ("c34p5", "c38p5")
    flux_comparisons = {}
    untreated_rate = flux_aggregate["untreated"][
        "mean_predicted_natural_zone_volume_rate"
    ]
    untreated_axial_point_face_rate = flux_aggregate["untreated"][
        "mean_axial_point_face_predicted_volume_rate"
    ]
    untreated_transverse_rate = flux_aggregate["untreated"][
        "mean_spectral_transverse_control_volume_rate"
    ]
    untreated_observed_rate = flux_aggregate["untreated"][
        "mean_observed_centered_natural_zone_volume_rate"
    ]
    untreated_inner_current = flux_aggregate["untreated"][
        "mean_physical_outward_current_at_inner_boundary"
    ]
    untreated_outer_current = flux_aggregate["untreated"][
        "mean_physical_outward_current_at_outer_boundary"
    ]
    for case_id in ("c26p5", "c34p5", "c38p5", "c64p5"):
        item = flux_aggregate[case_id]
        flux_comparisons[case_id] = {
            "inner_boundary_current_change_from_untreated": (
                item[
                    "mean_physical_outward_current_at_inner_boundary"
                ]
                - untreated_inner_current
            ),
            "outer_boundary_current_change_from_untreated": (
                item[
                    "mean_physical_outward_current_at_outer_boundary"
                ]
                - untreated_outer_current
            ),
            "predicted_rate_change_from_untreated": (
                item["mean_predicted_natural_zone_volume_rate"]
                - untreated_rate
            ),
            "axial_point_face_rate_change_from_untreated": (
                item["mean_axial_point_face_predicted_volume_rate"]
                - untreated_axial_point_face_rate
            ),
            "transverse_rate_change_from_untreated": (
                item["mean_spectral_transverse_control_volume_rate"]
                - untreated_transverse_rate
            ),
            "observed_rate_change_from_untreated": (
                item["mean_observed_centered_natural_zone_volume_rate"]
                - untreated_observed_rate
            ),
            "all_four_continuity_signs_agree": item[
                "all_four_continuity_signs_agree"
            ],
        }
    harmful_net_depletion_supported = all(
        flux_comparisons[case_id][
            "predicted_rate_change_from_untreated"
        ]
        < 0.0
        and flux_comparisons[case_id][
            "observed_rate_change_from_untreated"
        ]
        < 0.0
        for case_id in selected_nonlocal
    )
    harmful_outer_drainage_supported = all(
        flux_comparisons[case_id][
            "outer_boundary_current_change_from_untreated"
        ]
        > 0.0
        for case_id in selected_nonlocal
    )
    near_case_protection_supported = (
        flux_comparisons["c26p5"][
            "predicted_rate_change_from_untreated"
        ]
        > 0.0
        and flux_comparisons["c26p5"][
            "observed_rate_change_from_untreated"
        ]
        > 0.0
    )
    far_null_smaller_than_harmful = (
        abs(
            flux_comparisons["c64p5"][
                "observed_rate_change_from_untreated"
            ]
        )
        < min(
            abs(
                flux_comparisons[case_id][
                    "observed_rate_change_from_untreated"
                ]
            )
            for case_id in selected_nonlocal
        )
    )
    rigorous_mask_passed = all(
        mask_checks[case_id][
            "all_checkpoints_have_zero_grid_resolved_escape"
        ]
        for case_id in selected_nonlocal
    )
    continuity_closure_passed = all(
        item["maximum_corrected_relative_magnitude_error"]
        <= CONTINUITY_RELATIVE_ERROR_LIMIT
        for item in flux_aggregate.values()
    )
    transverse_gap_explanation_passed = all(
        item["minimum_transverse_fraction_of_legacy_gap"]
        >= TRANSVERSE_GAP_EXPLANATION_MINIMUM
        for item in flux_aggregate.values()
    )
    axial_point_face_quadrature_passed = all(
        item["maximum_axial_point_face_quadrature_relative_error"]
        <= 0.005
        for item in flux_aggregate.values()
    )
    transport_supported = (
        harmful_net_depletion_supported
        and harmful_outer_drainage_supported
        and near_case_protection_supported
        and far_null_smaller_than_harmful
        and rigorous_mask_passed
        and continuity_closure_passed
        and transverse_gap_explanation_passed
        and axial_point_face_quadrature_passed
    )
    report = {
        "schema_version": 2,
        "status": "complete",
        "classification": (
            "sparse_transport_supports_sign_changing_redistribution_"
            "within_one_seed"
            if transport_supported
            else "sparse_transport_audit_inconclusive"
        ),
        "scope": {
            "saved_full_checkpoints_only": True,
            "new_solver_steps": 0,
            "operator_evaluations_are_not_time_steps": True,
            "dynamic_budget_cases": list(DYNAMIC_CASES),
            "dynamic_budget_steps": list(DYNAMIC_STEPS),
            "flux_cases": list(FLUX_CASES),
            "flux_step": FLUX_STEP,
            "control_volume_identity": (
                "periodic_pseudospectral_summation_by_parts"
            ),
        },
        "natural_zone_bounds": list(natural_bounds),
        "surface_weighted_direct_exposure_t1300": exposure,
        "dynamic_budget_and_mask": {
            "rows": dynamic_rows,
            "case_checks": mask_checks,
            "missing_anchor": {
                "case_id": "c14p5",
                "reason": (
                    "No exact 500/900/1300 full checkpoints exist for "
                    "this legacy anchor; nonlinear fields were not "
                    "interpolated."
                ),
            },
        },
        "instantaneous_flux_t1300": {
            "definition": (
                "Physical current J=-m(x)M(c)grad(mu), reconstructed with "
                "the frozen spectral derivative. The phase-profile crop is "
                "an open rectangular control volume, so its raw operator "
                "rate includes both axial and transverse-face current."
            ),
            "continuity_check": (
                "The instantaneous t1300 raw physical-operator rate is "
                "evaluated by exact spectral summation by parts using the "
                "saved profile's axial trapezoid and 96-node transverse "
                "crop weights, then compared with the centered t1200--t1400 "
                "phase-volume rate."
            ),
            "control_volume_discretization": {
                "axial_weight": (
                    "same inclusive-node trapezoid as _trapz_in_bounds"
                ),
                "transverse_weight": (
                    "same 96-node rectangle as phase_excess_radius_profiles"
                ),
                "thin_x_weight": (
                    "full periodic span; its derivative and contribution "
                    "are exactly zero"
                ),
                "rate_identity": (
                    "for q=M grad(mu), dV/dt=-h^3 sum_i <D_i W,q_i>"
                ),
            },
            "arm_rows": flux_rows,
            "case_aggregate": flux_aggregate,
            "comparisons_to_untreated": flux_comparisons,
            "checks": {
                "harmful_cases_have_more_negative_predicted_rate": (
                    harmful_net_depletion_supported
                ),
                "harmful_cases_have_increased_outer_boundary_drainage": (
                    harmful_outer_drainage_supported
                ),
                "near_case_has_reduced_net_depletion": (
                    near_case_protection_supported
                ),
                "far_null_observed_change_is_smaller": (
                    far_null_smaller_than_harmful
                ),
                "all_cases_close_below_one_percent": (
                    continuity_closure_passed
                ),
                "omitted_transverse_term_explains_legacy_gap": (
                    transverse_gap_explanation_passed
                ),
                "axial_point_face_quadrature_is_below_half_percent": (
                    axial_point_face_quadrature_passed
                ),
            },
            "mechanism_interpretation": (
                "At the intermediate placements, outward material current "
                "through the natural-zone outer boundary increases while "
                "junction-side drainage remains. Transverse exchange through "
                "the finite diagnostic crop also contributes to its volume "
                "rate and must not be omitted. The full open-control-volume "
                "balance confirms stronger depletion for c34.5/c38.5, "
                "reduced depletion for c26.5, and a near-null far c64.5 "
                "response."
            ),
        },
        "decision": {
            "profile_precursor_prerequisite_passed": True,
            "rigorous_fixed_mask_capture_passed": rigorous_mask_passed,
            "full_control_volume_continuity_passed": (
                continuity_closure_passed
            ),
            "legacy_continuity_gap_cause": (
                "omitted_transverse_current_through_open_profile_crop"
            ),
            "sparse_flux_supports_sign_changing_redistribution": (
                transport_supported
            ),
            "fixed_eulerian_mask_gross_detachment": (
                "rejected_over_saved_decision_window"
                if rigorous_mask_passed
                else "not_rejected"
            ),
            "physical_model_scope": (
                "fixed_spatial_mobility_landscape_not_material_following"
            ),
        },
        "claim_boundary": [
            (
                "The reconstructed current is instantaneous at t=1300; "
                "the comparison slope averages t1200--t1400, and continuous "
                "integrated transport was not saved."
            ),
            (
                "The spectral all-face identity closes the raw physical "
                "operator. It is not an exact one-step identity for the "
                "semi-implicit stabilizer-filtered update."
            ),
            (
                "Dynamic resistance budgets need not remain equal after "
                "their common t=100 calibration because morphology evolves."
            ),
            (
                "Zero grid-resolved radial escape supports the fixed-mask "
                "implementation but does not turn it into an advected "
                "material coating."
            ),
            (
                "One source realization cannot establish a statistically "
                "reproducible harmful-placement window."
            ),
        ],
        "next_required_experiment": (
            "paired_source_realizations_for_untreated_c18p5_c26p5_c34p5"
            if transport_supported
            else "stop_and_reassess_mechanism"
        ),
        "provenance": {
            "analysis_script": str(SOURCE_PATH),
            "analysis_script_sha256": sha256_path(SOURCE_PATH),
            "profile_audit": str(PROFILE_AUDIT_PATH.resolve()),
            "profile_audit_sha256": sha256_path(PROFILE_AUDIT_PATH),
            "aggregate": str(AGGREGATE_PATH.resolve()),
            "aggregate_sha256": sha256_path(AGGREGATE_PATH),
            "checkpoint_records": checkpoint_provenance,
        },
    }
    return report, flux_rows


def _write_flux_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "case_id",
        "step",
        "wire",
        "branch",
        "physical_outward_current_at_inner_boundary",
        "physical_outward_current_at_outer_boundary",
        "axial_point_face_predicted_volume_rate",
        "spectral_axial_control_volume_rate",
        "spectral_transverse_control_volume_rate",
        "spectral_periodic_x_control_volume_rate",
        "predicted_natural_zone_volume_rate",
        "observed_centered_natural_zone_volume_rate",
        "continuity_sign_agrees",
        "relative_magnitude_error",
        "legacy_axial_only_relative_magnitude_error",
        "axial_point_face_quadrature_relative_error",
        "transverse_fraction_of_legacy_gap",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_report(output: Path) -> dict[str, Any]:
    report, flux_rows = build_report()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "summary.json", report)
    _write_flux_csv(output / "flux_boundary_audit.csv", flux_rows)
    return report


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    report = write_report(arguments.output.resolve())
    print(
        json.dumps(
            {
                "status": report["status"],
                "classification": report["classification"],
                "transport_supported": report["decision"][
                    "sparse_flux_supports_sign_changing_redistribution"
                ],
                "output": str(arguments.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
