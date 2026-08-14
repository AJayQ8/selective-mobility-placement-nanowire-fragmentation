"""Chunked metrics for the full-grid common-state response diagnostic."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..roy_2021_reproduction.model import FROZEN_DEG90
from .geometry import centered_coordinates


HISTOGRAM_BIN_PHYSICAL = FROZEN_DEG90.lattice.spacing
REGION_CHUNK = 32


def _periodic_distance(
    coordinate: np.ndarray,
    plane: float,
    length: float,
) -> np.ndarray:
    return np.abs(
        (coordinate - plane + 0.5 * length) % length - 0.5 * length
    )


def _empty_region() -> dict[str, float | int]:
    return {
        "cell_count": 0,
        "sum_squared": 0.0,
        "maximum_absolute": 0.0,
    }


def _accumulate(
    accumulator: dict[str, float | int],
    values: np.ndarray,
    mask: np.ndarray,
) -> None:
    selected = values[np.broadcast_to(mask, values.shape)]
    if not selected.size:
        return
    accumulator["cell_count"] = (
        int(accumulator["cell_count"]) + int(selected.size)
    )
    accumulator["sum_squared"] = (
        float(accumulator["sum_squared"])
        + float(np.sum(selected * selected, dtype=np.float64))
    )
    accumulator["maximum_absolute"] = max(
        float(accumulator["maximum_absolute"]),
        float(np.max(np.abs(selected))),
    )


def _finalize_partition(
    accumulators: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    total = float(
        sum(float(item["sum_squared"]) for item in accumulators.values())
    )
    result: dict[str, dict[str, float | int]] = {}
    for name, item in accumulators.items():
        count = int(item["cell_count"])
        squared = float(item["sum_squared"])
        result[name] = {
            "cell_count": count,
            "sum_squared": squared,
            "rms": float(np.sqrt(squared / count)) if count else 0.0,
            "maximum_absolute": float(item["maximum_absolute"]),
            "fraction_of_partition_squared_norm": (
                float(squared / total) if total > 0.0 else 0.0
            ),
        }
    return result


def _new_histogram(bin_count: int) -> dict[str, np.ndarray]:
    return {
        "cell_count": np.zeros(bin_count, dtype=np.int64),
        "sum_squared": np.zeros(bin_count, dtype=np.float64),
    }


def _add_histogram_counts(
    histogram: dict[str, np.ndarray],
    indices: np.ndarray,
    mask: np.ndarray | None,
) -> None:
    selected = indices.ravel()
    if mask is not None:
        selected = selected[np.broadcast_to(mask, indices.shape).ravel()]
    histogram["cell_count"] += np.bincount(
        selected,
        minlength=histogram["cell_count"].size,
    )


def _add_histogram_values(
    histogram: dict[str, np.ndarray],
    indices: np.ndarray,
    squared: np.ndarray,
    mask: np.ndarray | None,
) -> None:
    selected_indices = indices.ravel()
    selected_squared = squared.ravel()
    if mask is not None:
        flat_mask = np.broadcast_to(mask, indices.shape).ravel()
        selected_indices = selected_indices[flat_mask]
        selected_squared = selected_squared[flat_mask]
    histogram["sum_squared"] += np.bincount(
        selected_indices,
        weights=selected_squared,
        minlength=histogram["sum_squared"].size,
    )


def finalize_histogram(
    histogram: dict[str, np.ndarray],
    *,
    bin_width: float,
    surface_width: float,
) -> dict[str, Any]:
    """Return grid-resolved squared-norm profiles and W50/W90 bounds."""

    counts = histogram["cell_count"]
    squared = histogram["sum_squared"]
    total = float(np.sum(squared, dtype=np.float64))
    cumulative = (
        np.cumsum(squared, dtype=np.float64) / total
        if total > 0.0
        else np.zeros_like(squared)
    )
    bins: list[dict[str, Any]] = []
    for index, (count, value, cumulative_fraction) in enumerate(
        zip(counts, squared, cumulative)
    ):
        if count == 0 and value == 0.0:
            continue
        lower = index * bin_width
        upper = (index + 1) * bin_width
        bins.append(
            {
                "index": index,
                "interval_convention": "[lower, upper)",
                "lower_physical": lower,
                "upper_physical": upper,
                "lower_over_width": lower / surface_width,
                "upper_over_width": upper / surface_width,
                "cell_count": int(count),
                "sum_squared": float(value),
                "rms": (
                    float(np.sqrt(value / count)) if count else 0.0
                ),
                "fraction_of_total_squared_norm": (
                    float(value / total) if total > 0.0 else 0.0
                ),
                "cumulative_fraction": float(cumulative_fraction),
            }
        )

    quantiles: dict[str, Any] = {}
    for label, target in (("w50", 0.50), ("w90", 0.90)):
        if total <= 0.0:
            quantiles[label] = None
            continue
        index = int(np.searchsorted(cumulative, target, side="left"))
        lower = index * bin_width
        upper = (index + 1) * bin_width
        quantiles[label] = {
            "target_fraction": target,
            "bin_index": index,
            "interval_convention": "[lower, upper)",
            "interval_physical": [lower, upper],
            "interval_over_width": [
                lower / surface_width,
                upper / surface_width,
            ],
            "grid_resolved_upper_bound_over_width": (
                upper / surface_width
            ),
        }

    conservative_fractions: dict[str, float] = {}
    for multiple in (1.0, 2.0, 4.0, 8.0):
        last_complete = int(
            np.floor(multiple * surface_width / bin_width)
        ) - 1
        if total <= 0.0 or last_complete < 0:
            fraction = 0.0
        else:
            last_complete = min(last_complete, cumulative.size - 1)
            fraction = float(cumulative[last_complete])
        conservative_fractions[f"{multiple:g}W"] = fraction

    return {
        "bin_width_physical": bin_width,
        "total_cell_count": int(np.sum(counts, dtype=np.int64)),
        "total_sum_squared": total,
        "quantiles": quantiles,
        "fractions_in_complete_bins_within_distance": (
            conservative_fractions
        ),
        "bins": bins,
    }


def decompose_source(
    source: np.ndarray,
    *,
    primary_coordinate: float,
    image_coordinate: float,
    radius: float,
    surface_width: float,
    noise_prefix_length: int,
) -> dict[str, np.ndarray]:
    """Split q into exact prefix, intended-intersection, and residual fields."""

    lattice = FROZEN_DEG90.lattice
    if source.shape != lattice.shape:
        raise ValueError("source has the wrong shape")
    nx, ny, nz = lattice.shape
    x = centered_coordinates(lattice, 0)
    y = centered_coordinates(lattice, 1)
    z = centered_coordinates(lattice, 2)
    r1 = np.sqrt(x[:, None] ** 2 + y[None, :] ** 2)
    surface_distance = np.abs(r1 - radius)
    base_flat_index = (
        (
            np.arange(nx, dtype=np.int64)[:, None]
            * ny
            + np.arange(ny, dtype=np.int64)[None, :]
        )
        * nz
    )
    length_z = lattice.physical_lengths[2]
    prefix = np.zeros_like(source)
    intended = np.zeros_like(source)
    for start in range(0, nz, REGION_CHUNK):
        stop = min(nz, start + REGION_CHUNK)
        z_block = z[start:stop]
        dp = _periodic_distance(z_block, primary_coordinate, length_z)
        di = _periodic_distance(z_block, image_coordinate, length_z)
        plane_distance = np.minimum(dp, di)
        rho = np.hypot(
            surface_distance[:, :, None],
            plane_distance[None, None, :],
        )
        flat_index = (
            base_flat_index[:, :, None]
            + np.arange(start, stop, dtype=np.int64)[None, None, :]
        )
        prefix_mask = flat_index < noise_prefix_length
        intended_mask = ~prefix_mask & (
            rho <= 2.0 * surface_width
        )
        block = source[:, :, start:stop]
        prefix_block = prefix[:, :, start:stop]
        intended_block = intended[:, :, start:stop]
        prefix_block[prefix_mask] = block[prefix_mask]
        intended_block[intended_mask] = block[intended_mask]
    residual = source - prefix - intended
    return {
        "released_prefix_source": prefix,
        "intended_intersection_source": intended,
        "residual_source": residual,
    }


def analyze_fields(
    fields: dict[str, np.ndarray],
    *,
    primary_coordinate: float,
    image_coordinate: float,
    radius: float,
    surface_width: float,
    noise_prefix_length: int,
) -> dict[str, Any]:
    """Analyze several fields in one shared geometry pass."""

    lattice = FROZEN_DEG90.lattice
    if any(field.shape != lattice.shape for field in fields.values()):
        raise ValueError("analysis field has the wrong shape")
    nx, ny, nz = lattice.shape
    x = centered_coordinates(lattice, 0)
    y = centered_coordinates(lattice, 1)
    z = centered_coordinates(lattice, 2)
    r1 = np.sqrt(x[:, None] ** 2 + y[None, :] ** 2)
    surface_distance = np.abs(r1 - radius)
    length_z = lattice.physical_lengths[2]
    maximum_plane = 0.25 * length_z
    maximum_surface = float(np.max(surface_distance))
    maximum_rho = float(np.hypot(maximum_surface, maximum_plane))
    bin_counts = {
        "rho": int(np.floor(maximum_rho / HISTOGRAM_BIN_PHYSICAL)) + 2,
        "plane": int(np.floor(maximum_plane / HISTOGRAM_BIN_PHYSICAL)) + 2,
        "surface": int(np.floor(maximum_surface / HISTOGRAM_BIN_PHYSICAL)) + 2,
    }
    histogram_dimensions = {
        "rho_all": "rho",
        "plane_all": "plane",
        "surface_all": "surface",
        "rho_primary_half": "rho",
        "rho_image_half": "rho",
        "rho_released_prefix": "rho",
        "rho_outside_released_prefix": "rho",
    }
    geometry_histograms = {
        key: _new_histogram(bin_counts[dimension])
        for key, dimension in histogram_dimensions.items()
    }
    field_histograms = {
        field_name: {
            key: _new_histogram(bin_counts[dimension])
            for key, dimension in histogram_dimensions.items()
        }
        for field_name in fields
    }
    anatomy_names = (
        "released_prefix",
        "intended_intersection",
        "junction",
        "second_wire",
        "remainder",
    )
    roi_names = (
        "all",
        "primary_core",
        "image_core",
        "combined_core",
        "intended_halo",
        "junction",
        "second_wire",
        "far_arm",
        "released_prefix",
    )
    anatomy = {
        field_name: {
            region: _empty_region() for region in anatomy_names
        }
        for field_name in fields
    }
    rois = {
        field_name: {region: _empty_region() for region in roi_names}
        for field_name in fields
    }
    anatomy_counts = {name: 0 for name in anatomy_names}
    roi_counts = {name: 0 for name in roi_names}
    base_flat_index = (
        (
            np.arange(nx, dtype=np.int64)[:, None]
            * ny
            + np.arange(ny, dtype=np.int64)[None, :]
        )
        * nz
    )
    core_ring = surface_distance <= surface_width
    halo_ring = surface_distance <= 2.0 * surface_width
    y_distal = np.abs(y) >= 2.0 * radius

    for start in range(0, nz, REGION_CHUNK):
        stop = min(nz, start + REGION_CHUNK)
        z_block = z[start:stop]
        dp = _periodic_distance(z_block, primary_coordinate, length_z)
        di = _periodic_distance(z_block, image_coordinate, length_z)
        dg = np.minimum(dp, di)
        shape = (nx, ny, stop - start)
        plane_distance = np.broadcast_to(
            dg[None, None, :], shape
        )
        surface_block = np.broadcast_to(
            surface_distance[:, :, None], shape
        )
        rho = np.hypot(surface_block, plane_distance)
        rho_index = np.minimum(
            np.floor(rho / HISTOGRAM_BIN_PHYSICAL).astype(np.int64),
            bin_counts["rho"] - 1,
        )
        plane_index = np.minimum(
            np.floor(
                plane_distance / HISTOGRAM_BIN_PHYSICAL
            ).astype(np.int64),
            bin_counts["plane"] - 1,
        )
        surface_index = np.minimum(
            np.floor(
                surface_block / HISTOGRAM_BIN_PHYSICAL
            ).astype(np.int64),
            bin_counts["surface"] - 1,
        )
        primary_half = np.broadcast_to(
            (dp <= di)[None, None, :], shape
        )
        image_half = ~primary_half
        flat_index = (
            base_flat_index[:, :, None]
            + np.arange(start, stop, dtype=np.int64)[None, None, :]
        )
        prefix = flat_index < noise_prefix_length
        outside_prefix = ~prefix
        histogram_inputs = {
            "rho_all": (rho_index, None),
            "plane_all": (plane_index, None),
            "surface_all": (surface_index, None),
            "rho_primary_half": (rho_index, primary_half),
            "rho_image_half": (rho_index, image_half),
            "rho_released_prefix": (rho_index, prefix),
            "rho_outside_released_prefix": (
                rho_index,
                outside_prefix,
            ),
        }
        for key, (indices, mask) in histogram_inputs.items():
            _add_histogram_counts(
                geometry_histograms[key], indices, mask
            )

        intended = outside_prefix & (rho <= 2.0 * surface_width)
        junction_geometry = (
            (x[:, None, None] - radius) ** 2
            + y[None, :, None] ** 2
            + z_block[None, None, :] ** 2
            <= radius**2
        )
        assigned = prefix | intended
        junction = ~assigned & junction_geometry
        r2 = np.sqrt(
            (x[:, None] - 2.0 * radius) ** 2
            + z_block[None, :] ** 2
        )
        second_geometry = np.broadcast_to(
            r2[:, None, :] <= radius + 2.0 * surface_width,
            shape,
        )
        second = ~(assigned | junction) & second_geometry
        remainder = ~(assigned | junction | second)
        anatomy_masks = {
            "released_prefix": prefix,
            "intended_intersection": intended,
            "junction": junction,
            "second_wire": second,
            "remainder": remainder,
        }

        primary_core = (
            core_ring[:, :, None] & (dp[None, None, :] <= surface_width)
        )
        image_core = (
            core_ring[:, :, None] & (di[None, None, :] <= surface_width)
        )
        combined_core = primary_core | image_core
        intended_halo = (
            halo_ring[:, :, None]
            & (dg[None, None, :] <= 2.0 * surface_width)
        )
        roi_second = (
            (r2[:, None, :] <= radius + 2.0 * surface_width)
            & y_distal[None, :, None]
            & ~junction_geometry
        )
        far_arm = (
            core_ring[:, :, None]
            & (
                _periodic_distance(
                    z_block,
                    -primary_coordinate,
                    length_z,
                )[None, None, :]
                <= surface_width
            )
        )
        roi_masks = {
            "all": np.ones(shape, dtype=bool),
            "primary_core": primary_core,
            "image_core": image_core,
            "combined_core": combined_core,
            "intended_halo": intended_halo,
            "junction": junction_geometry,
            "second_wire": roi_second,
            "far_arm": far_arm,
            "released_prefix": prefix,
        }
        for region, mask in anatomy_masks.items():
            anatomy_counts[region] += int(np.count_nonzero(mask))
        for region, mask in roi_masks.items():
            roi_counts[region] += int(np.count_nonzero(mask))

        for field_name, field in fields.items():
            values = field[:, :, start:stop]
            squared = values * values
            for key, (indices, mask) in histogram_inputs.items():
                _add_histogram_values(
                    field_histograms[field_name][key],
                    indices,
                    squared,
                    mask,
                )
            for region, mask in anatomy_masks.items():
                _accumulate(anatomy[field_name][region], values, mask)
            for region, mask in roi_masks.items():
                _accumulate(rois[field_name][region], values, mask)

    for key in histogram_dimensions:
        counts = geometry_histograms[key]["cell_count"]
        for field_name in fields:
            field_histograms[field_name][key]["cell_count"] = counts.copy()
    for field_name in fields:
        for region in anatomy_names:
            anatomy[field_name][region]["cell_count"] = (
                anatomy_counts[region]
            )
        for region in roi_names:
            rois[field_name][region]["cell_count"] = roi_counts[region]

    profiles = {
        field_name: {
            key: finalize_histogram(
                histogram,
                bin_width=HISTOGRAM_BIN_PHYSICAL,
                surface_width=surface_width,
            )
            for key, histogram in histograms.items()
        }
        for field_name, histograms in field_histograms.items()
    }
    anatomy_reports = {
        field_name: _finalize_partition(regions)
        for field_name, regions in anatomy.items()
    }
    roi_reports: dict[str, Any] = {}
    for field_name, regions in rois.items():
        total = float(regions["all"]["sum_squared"])
        report: dict[str, Any] = {}
        for region, item in regions.items():
            count = int(item["cell_count"])
            squared = float(item["sum_squared"])
            report[region] = {
                "cell_count": count,
                "sum_squared": squared,
                "rms": float(np.sqrt(squared / count)) if count else 0.0,
                "maximum_absolute": float(item["maximum_absolute"]),
                "fraction_of_total_squared_norm": (
                    float(squared / total) if total > 0.0 else 0.0
                ),
            }
        roi_reports[field_name] = report

    checks: dict[str, bool] = {}
    for field_name in fields:
        anatomy_count = sum(
            int(item["cell_count"])
            for item in anatomy_reports[field_name].values()
        )
        anatomy_fraction = sum(
            float(item["fraction_of_partition_squared_norm"])
            for item in anatomy_reports[field_name].values()
        )
        rho_total = profiles[field_name]["rho_all"]["total_sum_squared"]
        scale = max(abs(rho_total), np.finfo(float).tiny)
        plane_total = profiles[field_name]["plane_all"][
            "total_sum_squared"
        ]
        surface_total = profiles[field_name]["surface_all"][
            "total_sum_squared"
        ]
        primary_total = profiles[field_name]["rho_primary_half"][
            "total_sum_squared"
        ]
        image_total = profiles[field_name]["rho_image_half"][
            "total_sum_squared"
        ]
        prefix_total = profiles[field_name]["rho_released_prefix"][
            "total_sum_squared"
        ]
        outside_total = profiles[field_name][
            "rho_outside_released_prefix"
        ]["total_sum_squared"]
        checks[f"{field_name}_anatomy_counts_exhaustive"] = (
            anatomy_count == lattice.cell_count
        )
        checks[f"{field_name}_anatomy_norm_exhaustive"] = (
            abs(anatomy_fraction - 1.0) <= 1.0e-12
        )
        checks[f"{field_name}_distance_totals_agree"] = (
            abs(plane_total - rho_total) / scale <= 1.0e-12
            and abs(surface_total - rho_total) / scale <= 1.0e-12
        )
        checks[f"{field_name}_plane_halves_recompose"] = (
            abs(primary_total + image_total - rho_total) / scale
            <= 1.0e-12
        )
        checks[f"{field_name}_prefix_halves_recompose"] = (
            abs(prefix_total + outside_total - rho_total) / scale
            <= 1.0e-12
        )

    return {
        "distance_definition": {
            "rho_physical": "sqrt(abs(r1-R)^2 + min(dp,di)^2)",
            "rho_over_width": "rho_physical / W",
            "plane_distance": "min(dp,di)",
            "surface_distance": "abs(r1-R)",
            "bin_width_physical": HISTOGRAM_BIN_PHYSICAL,
            "w50_w90_are_grid_resolved_upper_bounds": True,
            "w50_w90_are_descriptive_only": True,
        },
        "profiles": profiles,
        "anatomical_partitions": anatomy_reports,
        "rois": roi_reports,
        "accounting_checks": checks,
    }
