"""Smooth, finite-contrast mobility landscapes for crossed nanowires."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..roy_2021_reproduction.model import PeriodicLattice


Array = np.ndarray


@dataclass(frozen=True)
class FourArmCollarGeometry:
    """Source-semantic geometry of four symmetric arm collars."""

    inner_support_distance: float
    outer_support_distance: float
    transition_width: float
    protected_mobility_factor: float

    def __post_init__(self) -> None:
        values = (
            self.inner_support_distance,
            self.outer_support_distance,
            self.transition_width,
            self.protected_mobility_factor,
        )
        if not all(np.isfinite(values)):
            raise ValueError("collar parameters must be finite")
        if self.inner_support_distance <= 0.0:
            raise ValueError("inner support distance must be positive")
        if self.outer_support_distance <= self.inner_support_distance:
            raise ValueError("outer support must lie beyond inner support")
        if self.transition_width <= 0.0:
            raise ValueError("transition width must be positive")
        if (
            self.outer_support_distance - self.inner_support_distance
            <= 2.0 * self.transition_width
        ):
            raise ValueError("collar must contain a nonzero plateau")
        if not 0.0 < self.protected_mobility_factor <= 1.0:
            raise ValueError("protected mobility factor must lie in (0, 1]")

    @property
    def center_distance(self) -> float:
        return 0.5 * (
            self.inner_support_distance + self.outer_support_distance
        )

    @property
    def support_width(self) -> float:
        return self.outer_support_distance - self.inner_support_distance

    @property
    def plateau_bounds(self) -> tuple[float, float]:
        return (
            self.inner_support_distance + self.transition_width,
            self.outer_support_distance - self.transition_width,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            **asdict(self),
            "center_distance": self.center_distance,
            "support_width": self.support_width,
            "plateau_bounds": list(self.plateau_bounds),
        }


def centered_coordinates(cells: int, spacing: float) -> Array:
    """Return Roy's integer-centered periodic coordinate convention."""

    if cells < 2 or cells % 2:
        raise ValueError("cells must be an even integer >= 2")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be finite and positive")
    return (
        np.arange(cells, dtype=np.float64) - cells // 2
    ) * float(spacing)


def periodic_distance(
    coordinate: Array,
    center: float,
    period: float,
) -> Array:
    """Return an unsigned minimum-image distance."""

    if not np.isfinite(center):
        raise ValueError("center must be finite")
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("period must be finite and positive")
    values = np.asarray(coordinate, dtype=np.float64)
    return np.abs((values - center + 0.5 * period) % period - 0.5 * period)


def raised_cosine_step(value: Array) -> Array:
    """Compact C1 step: zero below 0, one above 1."""

    u = np.asarray(value, dtype=np.float64)
    clipped = np.clip(u, 0.0, 1.0)
    stepped = 0.5 * (1.0 - np.cos(np.pi * clipped))
    return np.where(u <= 0.0, 0.0, np.where(u >= 1.0, 1.0, stepped))


def signed_arm_window(
    coordinate: Array,
    geometry: FourArmCollarGeometry,
) -> Array:
    """Return the symmetric pair of compact collars on one wire axis."""

    distance = np.abs(np.asarray(coordinate, dtype=np.float64))
    rising = raised_cosine_step(
        (distance - geometry.inner_support_distance)
        / geometry.transition_width
    )
    falling = raised_cosine_step(
        (geometry.outer_support_distance - distance)
        / geometry.transition_width
    )
    return rising * falling


def radial_surface_shell(
    radial_distance: Array,
    *,
    radius: float,
    interface_width: float,
) -> Array:
    """Compact shell with a plateau over ``R-W <= r <= R+W``.

    The shell is exactly zero outside ``R-2W < r < R+2W``. It is a fixed
    Eulerian geometric support, not a composition-dependent mobility rewrite.
    """

    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    if not np.isfinite(interface_width) or interface_width <= 0.0:
        raise ValueError("interface width must be finite and positive")
    inner_support = radius - 2.0 * interface_width
    if inner_support < 0.0:
        raise ValueError("surface shell requires radius >= 2*interface_width")
    distance = np.asarray(radial_distance, dtype=np.float64)
    rising = raised_cosine_step(
        (distance - inner_support) / interface_width
    )
    falling = raised_cosine_step(
        (radius + 2.0 * interface_width - distance)
        / interface_width
    )
    return rising * falling


def four_arm_collar_factor(
    lattice: PeriodicLattice,
    geometry: FourArmCollarGeometry,
) -> Array:
    """Return a broadcastable ``(1, Ny, Nz)`` four-arm mobility factor."""

    _, ny, nz = lattice.shape
    y = centered_coordinates(ny, lattice.spacing)
    z = centered_coordinates(nz, lattice.spacing)
    y_window = signed_arm_window(y, geometry)[:, None]
    z_window = signed_arm_window(z, geometry)[None, :]
    # Smooth union prevents accidental m_p**2 suppression where the two
    # planar collar systems cross in vapor.
    coverage = 1.0 - (1.0 - y_window) * (1.0 - z_window)
    factor = (
        1.0
        - (1.0 - geometry.protected_mobility_factor) * coverage
    )[None, :, :]
    factor = np.asarray(factor, dtype=np.float64)
    factor.setflags(write=False)
    return factor


def four_arm_tubular_collar_factor(
    lattice: PeriodicLattice,
    geometry: FourArmCollarGeometry,
    *,
    radius: float,
    interface_width: float,
    first_wire_center_x: float = 0.0,
    second_wire_center_x: float = 12.0,
) -> Array:
    """Return a full 3-D factor confined to four tubular surface collars."""

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
        * signed_arm_window(z, geometry)[None, None, :]
    )
    second_coverage = (
        second_shell[:, None, :]
        * signed_arm_window(y, geometry)[None, :, None]
    )
    coverage = 1.0 - (1.0 - first_coverage) * (1.0 - second_coverage)
    factor = (
        1.0
        - (1.0 - geometry.protected_mobility_factor) * coverage
    )
    factor = np.asarray(factor, dtype=np.float64, order="C")
    if factor.shape != lattice.shape:
        raise RuntimeError("tubular collar factor has an invalid shape")
    factor.setflags(write=False)
    return factor


def phase_interface_weight(c: Array) -> Array:
    """Bounded surface indicator used only for equal-budget accounting."""

    q = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return 16.0 * q**2 * (1.0 - q) ** 2


def surface_weighted_mobility_deficit(
    c: Array,
    mobility_factor: Array,
    *,
    cell_volume: float,
) -> float:
    """Integrate ``surface_indicator * (1-m)`` over the fixed grid."""

    if not np.isfinite(cell_volume) or cell_volume <= 0.0:
        raise ValueError("cell volume must be finite and positive")
    factor = np.asarray(mobility_factor, dtype=np.float64)
    try:
        broadcast = np.broadcast_to(factor, np.shape(c))
    except ValueError as error:
        raise ValueError("mobility factor is not broadcastable to c") from error
    if not np.isfinite(broadcast).all():
        raise ValueError("mobility factor must be finite")
    if np.min(broadcast) <= 0.0 or np.max(broadcast) > 1.0:
        raise ValueError("mobility factor must lie in (0, 1]")
    weighted = phase_interface_weight(c) * (1.0 - broadcast)
    return float(np.sum(weighted, dtype=np.float64) * cell_volume)
