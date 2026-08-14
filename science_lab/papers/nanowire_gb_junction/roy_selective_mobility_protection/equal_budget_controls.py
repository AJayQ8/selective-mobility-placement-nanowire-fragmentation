"""Frozen definitions for the K2/K3 equal-budget causal controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..roy_2021_reproduction.model import PeriodicLattice
from .geometry import (
    FourArmCollarGeometry,
    centered_coordinates,
    four_arm_tubular_collar_factor,
    periodic_distance,
    phase_interface_weight,
    radial_surface_shell,
    raised_cosine_step,
    surface_weighted_mobility_deficit,
)


Array = np.ndarray


@dataclass(frozen=True)
class EqualBudgetControlDefinition:
    """Numerical definition and accounting for K1, K2, and K3."""

    target_budget: float
    interface_weight_integral: float
    junction_half_support_distance: float
    transition_width: float
    protected_mobility_factor: float
    uniform_mobility_factor: float
    junction_budget: float
    uniform_budget: float
    junction_relative_mismatch: float
    uniform_relative_mismatch: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def centered_cap_window(
    coordinate: Array,
    *,
    half_support_distance: float,
    transition_width: float,
) -> Array:
    """Return a compact central window with a raised-cosine outer edge."""

    values = (half_support_distance, transition_width)
    if not all(np.isfinite(values)):
        raise ValueError("cap distances must be finite")
    if transition_width <= 0.0:
        raise ValueError("transition width must be positive")
    if half_support_distance <= transition_width:
        raise ValueError("cap must contain a nonzero plateau")
    distance = np.abs(np.asarray(coordinate, dtype=np.float64))
    return raised_cosine_step(
        (half_support_distance - distance) / transition_width
    )


def junction_centered_tubular_cap_factor(
    lattice: PeriodicLattice,
    *,
    half_support_distance: float,
    transition_width: float,
    protected_mobility_factor: float,
    radius: float,
    interface_width: float,
    first_wire_center_x: float = 0.0,
    second_wire_center_x: float = 12.0,
) -> Array:
    """Return a smooth junction-centered cap on both crossed-wire surfaces."""

    if (
        not np.isfinite(protected_mobility_factor)
        or not 0.0 < protected_mobility_factor <= 1.0
    ):
        raise ValueError("protected mobility factor must lie in (0, 1]")
    nx, ny, nz = lattice.shape
    x = centered_coordinates(nx, lattice.spacing)
    y = centered_coordinates(ny, lattice.spacing)
    z = centered_coordinates(nz, lattice.spacing)
    lengths = lattice.physical_lengths
    first_radial = np.sqrt(
        periodic_distance(
            x, first_wire_center_x, lengths[0]
        )[:, None]
        ** 2
        + periodic_distance(y, 0.0, lengths[1])[None, :] ** 2
    )
    second_radial = np.sqrt(
        periodic_distance(
            x, second_wire_center_x, lengths[0]
        )[:, None]
        ** 2
        + periodic_distance(z, 0.0, lengths[2])[None, :] ** 2
    )
    first_shell = radial_surface_shell(
        first_radial,
        radius=radius,
        interface_width=interface_width,
    )
    second_shell = radial_surface_shell(
        second_radial,
        radius=radius,
        interface_width=interface_width,
    )
    first_coverage = (
        first_shell[:, :, None]
        * centered_cap_window(
            z,
            half_support_distance=half_support_distance,
            transition_width=transition_width,
        )[None, None, :]
    )
    second_coverage = (
        second_shell[:, None, :]
        * centered_cap_window(
            y,
            half_support_distance=half_support_distance,
            transition_width=transition_width,
        )[None, :, None]
    )
    coverage = 1.0 - (1.0 - first_coverage) * (1.0 - second_coverage)
    factor = 1.0 - (1.0 - protected_mobility_factor) * coverage
    factor = np.asarray(factor, dtype=np.float64, order="C")
    if factor.shape != lattice.shape:
        raise RuntimeError("junction cap factor has an invalid shape")
    factor.setflags(write=False)
    return factor


def _budget(
    field: Array,
    factor: Array,
    *,
    lattice: PeriodicLattice,
) -> float:
    return surface_weighted_mobility_deficit(
        field,
        factor,
        cell_volume=lattice.cell_volume,
    )


def derive_equal_budget_controls(
    field: Array,
    lattice: PeriodicLattice,
    *,
    inward_geometry: FourArmCollarGeometry,
    radius: float,
    interface_width: float,
    transition_width: float = 3.0,
    relative_tolerance: float = 1.0e-11,
    maximum_iterations: int = 80,
) -> tuple[EqualBudgetControlDefinition, Array, Array, Array]:
    """Derive exact K1, matched K2, and matched K3 mobility factors."""

    values = (
        radius,
        interface_width,
        transition_width,
        relative_tolerance,
    )
    if not all(np.isfinite(values)) or min(values) <= 0.0:
        raise ValueError("positive finite control parameters are required")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")

    k1 = four_arm_tubular_collar_factor(
        lattice,
        inward_geometry,
        radius=radius,
        interface_width=interface_width,
    )
    target = _budget(field, k1, lattice=lattice)
    if not np.isfinite(target) or target <= 0.0:
        raise RuntimeError("K1 target budget is not positive and finite")

    weight_integral = float(
        np.sum(phase_interface_weight(field), dtype=np.float64)
        * lattice.cell_volume
    )
    uniform_factor = 1.0 - target / weight_integral
    if not 0.0 < uniform_factor < 1.0:
        raise RuntimeError("equal-budget uniform factor lies outside (0, 1)")
    k3 = np.full((1, 1, 1), uniform_factor, dtype=np.float64)
    k3.setflags(write=False)

    lower = transition_width * (1.0 + 1.0e-6)
    upper = min(lattice.physical_lengths[1:]) / 4.0

    def make_cap(half_support: float) -> Array:
        return junction_centered_tubular_cap_factor(
            lattice,
            half_support_distance=half_support,
            transition_width=transition_width,
            protected_mobility_factor=(
                inward_geometry.protected_mobility_factor
            ),
            radius=radius,
            interface_width=interface_width,
        )

    lower_budget = _budget(field, make_cap(lower), lattice=lattice)
    upper_budget = _budget(field, make_cap(upper), lattice=lattice)
    if lower_budget >= target or upper_budget <= target:
        raise RuntimeError(
            "junction cap budget is not bracketed: "
            f"{lower_budget} < {target} < {upper_budget}"
        )

    half_support = 0.5 * (lower + upper)
    k2 = make_cap(half_support)
    junction_budget = _budget(field, k2, lattice=lattice)
    for _ in range(maximum_iterations):
        half_support = 0.5 * (lower + upper)
        candidate = make_cap(half_support)
        candidate_budget = _budget(field, candidate, lattice=lattice)
        mismatch = abs(candidate_budget - target) / target
        k2 = candidate
        junction_budget = candidate_budget
        if mismatch <= relative_tolerance:
            break
        if candidate_budget < target:
            lower = half_support
        else:
            upper = half_support
    else:
        raise RuntimeError("junction cap budget match did not converge")

    uniform_budget = _budget(field, k3, lattice=lattice)
    junction_mismatch = abs(junction_budget - target) / target
    uniform_mismatch = abs(uniform_budget - target) / target
    definition = EqualBudgetControlDefinition(
        target_budget=target,
        interface_weight_integral=weight_integral,
        junction_half_support_distance=half_support,
        transition_width=transition_width,
        protected_mobility_factor=(
            inward_geometry.protected_mobility_factor
        ),
        uniform_mobility_factor=uniform_factor,
        junction_budget=junction_budget,
        uniform_budget=uniform_budget,
        junction_relative_mismatch=junction_mismatch,
        uniform_relative_mismatch=uniform_mismatch,
    )
    return definition, k1, k2, k3
