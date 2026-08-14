"""Small deterministic geometries used by the bridge preflights."""

from __future__ import annotations

import numpy as np

from ..roy_2021_reproduction.model import Array, PeriodicLattice, RoyModelParameters


def centered_coordinates(
    lattice: PeriodicLattice,
    axis: int,
) -> Array:
    return (
        np.arange(lattice.shape[axis], dtype=np.float64)
        - lattice.shape[axis] // 2
    ) * lattice.spacing


def initialize_equilibrium_slab(
    lattice: PeriodicLattice,
    parameters: RoyModelParameters,
    *,
    normal_axis: int,
    half_height: float,
) -> Array:
    """Two flat equilibrium Roy solid/vapor interfaces in a periodic box."""

    if normal_axis not in (0, 1, 2):
        raise ValueError("normal_axis must be 0, 1, or 2")
    normal_length = lattice.physical_lengths[normal_axis]
    if not 0.0 < half_height < 0.5 * normal_length:
        raise ValueError("half_height must fit strictly inside the normal box")
    coordinate = centered_coordinates(lattice, normal_axis)
    signed_inside = half_height - np.abs(coordinate)
    argument = np.sqrt(
        2.0 * parameters.barrier / parameters.kappa
    ) * signed_inside
    profile = 1.0 / (1.0 + np.exp(-np.clip(argument, -60.0, 60.0)))
    shape = [1, 1, 1]
    shape[normal_axis] = lattice.shape[normal_axis]
    return np.array(
        np.broadcast_to(profile.reshape(shape), lattice.shape),
        dtype=np.float64,
        copy=True,
        order="C",
    )


def full_grid_coordinates(
    lattice: PeriodicLattice,
) -> tuple[Array, Array, Array]:
    """Broadcastable centered coordinate vectors for the three Roy axes."""

    return (
        centered_coordinates(lattice, 0)[:, None, None],
        centered_coordinates(lattice, 1)[None, :, None],
        centered_coordinates(lattice, 2)[None, None, :],
    )

