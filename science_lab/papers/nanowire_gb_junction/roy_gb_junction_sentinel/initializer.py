"""Exact validated product-union and isolated-wire initializers."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from ..roy_crossed_initializer_validation.run_validation import (
    EXPECTED_PROFILE_FINGERPRINT,
    EXPECTED_SOURCE_SHA256,
    embed_profile,
    load_source_profile,
    product_union,
)
from .contract import SentinelConfig


RADIUS_CELLS = 12
SOURCE_PROFILE_SHAPE = (96, 96)
EXPECTED_CROSSED_T0_FINGERPRINT = (
    "ee793415bb02121d268fa3ba6f40c13ec51007f120672a171cc1fa590d300a6a"
)
EXPECTED_ISOLATED_T0_FINGERPRINT = (
    "b09bd99f8d15f35b7b355b4aa134a77410b5a1bed3ddcefbda7a73fc6c556b52"
)


def array_fingerprint(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def build_crossed_templates(
    source_q: np.ndarray,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Embed the same equilibrated section as two tangent perpendicular arms."""

    nx, ny, nz = shape
    if source_q.shape != SOURCE_PROFILE_SHAPE or nx != SOURCE_PROFILE_SHAPE[0]:
        raise ValueError("source profile or thin dimension is incompatible")
    first_xy = embed_profile(
        source_q,
        (nx, ny),
        row_shift=-RADIUS_CELLS,
        column_shift=(ny - source_q.shape[1]) // 2,
    )
    second_xz = embed_profile(
        source_q,
        (nx, nz),
        row_shift=RADIUS_CELLS,
        column_shift=(nz - source_q.shape[1]) // 2,
    )
    source_center = 0.5 * (source_q.shape[0] - 1)
    first_center = source_center - RADIUS_CELLS
    second_center = source_center + RADIUS_CELLS
    return first_xy, second_xz, {
        "first_wire_axis": 2,
        "second_wire_axis": 1,
        "first_center_x_index": first_center,
        "second_center_x_index": second_center,
        "contact_midplane_x_index": 0.5 * (first_center + second_center),
        "center_y_index": 0.5 * (ny - 1),
        "center_z_index": 0.5 * (nz - 1),
        "center_separation_cells": second_center - first_center,
        "first_template_fingerprint": array_fingerprint(first_xy),
        "second_template_fingerprint": array_fingerprint(second_xz),
    }


def construct_product_union(
    first_xy: np.ndarray,
    second_xz: np.ndarray,
    *,
    vapor: float,
    solid: float,
    chunk_depth: int = 16,
) -> np.ndarray:
    """Construct exactly q1 + q2 - q1*q2 without full q1/q2 copies."""

    shape = (first_xy.shape[0], first_xy.shape[1], second_xz.shape[1])
    if second_xz.shape[0] != shape[0]:
        raise ValueError("crossed templates have inconsistent thin dimensions")
    result = np.empty(shape, dtype=np.float64)
    q1 = first_xy[:, :, None]
    span = solid - vapor
    for start in range(0, shape[2], chunk_depth):
        stop = min(shape[2], start + chunk_depth)
        q2 = second_xz[:, None, start:stop]
        q = product_union(q1, q2)
        result[:, :, start:stop] = vapor + span * q
    return result


def construct_isolated_wire(
    source_q: np.ndarray,
    shape: tuple[int, int, int],
    *,
    vapor: float,
    solid: float,
) -> np.ndarray:
    if source_q.shape != shape[:2]:
        raise ValueError("isolated shape must preserve the source cross-section")
    section = vapor + (solid - vapor) * source_q
    return np.array(
        np.broadcast_to(section[:, :, None], shape),
        dtype=np.float64,
        copy=True,
        order="C",
    )


def construct_initial_field(
    config: SentinelConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Construct the hash-pinned noise-free initial field for one condition."""

    source_q, vapor, solid, clipping = load_source_profile()
    if config.geometry == "crossed":
        first_xy, second_xz, geometry = build_crossed_templates(
            source_q, config.shape
        )
        field = construct_product_union(
            first_xy,
            second_xz,
            vapor=vapor,
            solid=solid,
        )
    else:
        geometry = {
            "wire_axis": 2,
            "center_x_index": 0.5 * (config.shape[0] - 1),
            "center_y_index": 0.5 * (config.shape[1] - 1),
            "center_z_index": 0.5 * (config.shape[2] - 1),
        }
        field = construct_isolated_wire(
            source_q,
            config.shape,
            vapor=vapor,
            solid=solid,
        )
    metadata = {
        "method": config.initializer,
        "union_formula": (
            "q1 + q2 - q1*q2" if config.geometry == "crossed" else None
        ),
        "noise": None,
        "deterministic_mode": None,
        "constant_mobility_preparation": False,
        "source_archive_sha256": EXPECTED_SOURCE_SHA256,
        "source_profile_fingerprint": EXPECTED_PROFILE_FINGERPRINT,
        "source_normalization": clipping,
        "homogeneous_vapor": vapor,
        "homogeneous_solid": solid,
        "field_fingerprint": array_fingerprint(field),
        "geometry": geometry,
    }
    if (
        config.geometry == "crossed"
        and metadata["field_fingerprint"] != EXPECTED_CROSSED_T0_FINGERPRINT
    ):
        raise RuntimeError("crossed t=0 field differs from the frozen validated field")
    if (
        config.geometry == "isolated"
        and metadata["field_fingerprint"] != EXPECTED_ISOLATED_T0_FINGERPRINT
    ):
        raise RuntimeError("isolated t=0 field differs from the frozen validated field")
    return field, metadata
