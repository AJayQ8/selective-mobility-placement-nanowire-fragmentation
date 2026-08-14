"""Roy-compatible conserved dynamics with an optional fixed Model-II GB.

The frozen Roy solver remains the only implementation used when the
grain-boundary energy ratio is zero.  For a positive ratio, this module adds
the fixed-orientation Model-II term to the real-space chemical potential before
the inherited complex64 FFT.  The conserved-field mobility, mixed-precision
spectral derivatives, stabilizer, and timestep are otherwise unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..roy_2021_reproduction.model import (
    Array,
    PeriodicLattice,
    RoyModelParameters,
    RoyPseudospectralSolver,
)


INTERPOLATION_EPSILON = 3.1


def interpolation(c: Array | float, epsilon: float = INTERPOLATION_EPSILON) -> Array:
    """Model-II grain-energy interpolation N(c)."""

    values = np.asarray(c)
    return values**2 * (
        1.0 + 2.0 * (1.0 - values) + epsilon * (1.0 - values) ** 2
    )




def interpolation_derivative(
    c: Array | float,
    epsilon: float = INTERPOLATION_EPSILON,
) -> Array:
    """Analytical derivative N'(c), evaluated without clipping c."""

    values = np.asarray(c)
    return (
        2.0
        * values
        * (1.0 - values)
        * (3.0 + epsilon * (1.0 - 2.0 * values))
    )


@dataclass(frozen=True)
class FixedGrainBoundaryParameters:
    """Model-II coefficients mapped exactly onto one Roy free energy."""

    energy_ratio: float
    surface_width: float
    grain_boundary_width: float
    gamma_surface: float
    gamma_grain_boundary: float
    beta: float
    kappa_eta: float
    interpolation_epsilon: float = INTERPOLATION_EPSILON

    def __post_init__(self) -> None:
        if not 0.0 <= self.energy_ratio < 1.0:
            raise ValueError("energy_ratio must satisfy 0 <= G < 1")
        if self.surface_width <= 0.0 or self.grain_boundary_width <= 0.0:
            raise ValueError("interface widths must be positive")
        if self.gamma_surface <= 0.0:
            raise ValueError("gamma_surface must be positive")
        if self.gamma_grain_boundary < 0.0:
            raise ValueError("gamma_grain_boundary cannot be negative")
        if self.beta < 0.0 or self.kappa_eta < 0.0:
            raise ValueError("grain-boundary coefficients cannot be negative")
        if self.interpolation_epsilon < 0.0:
            raise ValueError("interpolation_epsilon cannot be negative")

    @classmethod
    def matched_to_roy(
        cls,
        roy: RoyModelParameters,
        *,
        energy_ratio: float = 0.35,
        interpolation_epsilon: float = INTERPOLATION_EPSILON,
    ) -> "FixedGrainBoundaryParameters":
        """Map Roy A and kappa to the Model-II width and energy convention."""

        surface_width = float(np.sqrt(8.0 * roy.kappa / roy.barrier))
        gamma_surface = float(roy.barrier * surface_width / 12.0)
        gamma_grain_boundary = float(
            2.0 * energy_ratio * gamma_surface
        )
        grain_boundary_width = surface_width
        beta = float(gamma_grain_boundary / grain_boundary_width)
        kappa_eta = float(
            0.75 * gamma_grain_boundary * grain_boundary_width
        )
        return cls(
            energy_ratio=energy_ratio,
            surface_width=surface_width,
            grain_boundary_width=grain_boundary_width,
            gamma_surface=gamma_surface,
            gamma_grain_boundary=gamma_grain_boundary,
            beta=beta,
            kappa_eta=kappa_eta,
            interpolation_epsilon=interpolation_epsilon,
        )

    @property
    def predicted_dihedral_angle_deg(self) -> float:
        return float(np.degrees(2.0 * np.arccos(self.energy_ratio)))

    def to_dict(self) -> dict[str, float]:
        return {
            "energy_ratio": self.energy_ratio,
            "surface_width": self.surface_width,
            "grain_boundary_width": self.grain_boundary_width,
            "gamma_surface": self.gamma_surface,
            "gamma_grain_boundary": self.gamma_grain_boundary,
            "beta": self.beta,
            "kappa_eta": self.kappa_eta,
            "interpolation_epsilon": self.interpolation_epsilon,
            "predicted_dihedral_angle_deg": self.predicted_dihedral_angle_deg,
        }


@dataclass(frozen=True)
class PeriodicBicrystalProfile:
    """One-dimensional fixed orientation field with two periodic GB planes."""

    lattice_shape: tuple[int, int, int]
    spacing: float
    axis: int
    primary_coordinate: float
    image_coordinate: float
    eta_1: Array
    eta_2: Array
    grain_energy_density: Array
    parameters: FixedGrainBoundaryParameters

    def __post_init__(self) -> None:
        if self.axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")
        expected = self.lattice_shape[self.axis]
        for name, values in (
            ("eta_1", self.eta_1),
            ("eta_2", self.eta_2),
            ("grain_energy_density", self.grain_energy_density),
        ):
            if values.shape != (expected,):
                raise ValueError(f"{name} must be one-dimensional along axis")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} must be finite")
        if float(np.min(self.grain_energy_density)) < -1.0e-14:
            raise ValueError("grain energy density must be nonnegative")

    @property
    def physical_length(self) -> float:
        return float(self.lattice_shape[self.axis] * self.spacing)

    @property
    def density_view(self) -> Array:
        shape = [1, 1, 1]
        shape[self.axis] = self.lattice_shape[self.axis]
        return np.broadcast_to(
            self.grain_energy_density.reshape(shape),
            self.lattice_shape,
        )

    def periodic_distance(self, coordinate: Array, plane: float) -> Array:
        length = self.physical_length
        return np.abs((coordinate - plane + 0.5 * length) % length - 0.5 * length)

    def per_plane_energy_audit(self) -> dict[str, float]:
        coordinates = (
            np.arange(self.lattice_shape[self.axis], dtype=np.float64)
            - self.lattice_shape[self.axis] // 2
        ) * self.spacing
        distance_primary = self.periodic_distance(
            coordinates, self.primary_coordinate
        )
        distance_image = self.periodic_distance(
            coordinates, self.image_coordinate
        )
        primary_mask = distance_primary <= distance_image
        primary = float(
            self.spacing * np.sum(self.grain_energy_density[primary_mask])
        )
        image = float(
            self.spacing * np.sum(self.grain_energy_density[~primary_mask])
        )
        target = self.parameters.gamma_grain_boundary
        return {
            "primary_energy": primary,
            "image_energy": image,
            "target_energy": target,
            "primary_relative_error": abs(primary - target) / target,
            "image_relative_error": abs(image - target) / target,
            "image_mismatch_relative": abs(primary - image) / target,
        }


def build_periodic_bicrystal_profile(
    lattice: PeriodicLattice,
    parameters: FixedGrainBoundaryParameters,
    *,
    axis: int,
    primary_coordinate: float,
) -> PeriodicBicrystalProfile:
    """Construct the cancellation-free analytical two-grain profile."""

    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    cells = lattice.shape[axis]
    length = cells * lattice.spacing
    coordinates = (
        np.arange(cells, dtype=np.float64) - cells // 2
    ) * lattice.spacing
    wrapped_primary = float(
        (primary_coordinate + 0.5 * length) % length - 0.5 * length
    )
    image_coordinate = float(
        (wrapped_primary + 0.5 * length + 0.5 * length) % length
        - 0.5 * length
    )
    theta = 2.0 * np.pi * (coordinates - wrapped_primary) / length
    argument = (
        length
        / (np.pi * parameters.grain_boundary_width)
        * np.sin(theta)
    )
    eta_1 = 0.5 * (1.0 - np.tanh(argument))
    eta_2 = 1.0 - eta_1
    sech_squared = 1.0 / np.cosh(argument) ** 2
    eta_gradient = (
        -np.cos(theta)
        / parameters.grain_boundary_width
        * sech_squared
    )
    orientation_density = 12.0 * eta_1**2 * eta_2**2
    grain_energy_density = (
        parameters.beta * orientation_density
        + parameters.kappa_eta * eta_gradient**2
    )
    return PeriodicBicrystalProfile(
        lattice_shape=lattice.shape,
        spacing=lattice.spacing,
        axis=axis,
        primary_coordinate=wrapped_primary,
        image_coordinate=image_coordinate,
        eta_1=eta_1,
        eta_2=eta_2,
        grain_energy_density=grain_energy_density,
        parameters=parameters,
    )


class RoyFixedGrainBoundarySolver(RoyPseudospectralSolver):
    """Positive-G extension that preserves the Roy conserved-field operator."""

    def __init__(
        self,
        lattice: PeriodicLattice,
        parameters: RoyModelParameters,
        grain_boundary: PeriodicBicrystalProfile,
        *,
        fft_workers: int = 1,
    ) -> None:
        if grain_boundary.parameters.energy_ratio <= 0.0:
            raise ValueError("positive-G solver requires energy_ratio > 0")
        if grain_boundary.lattice_shape != lattice.shape:
            raise ValueError("grain-boundary profile does not match lattice")
        if grain_boundary.spacing != lattice.spacing:
            raise ValueError("grain-boundary spacing does not match lattice")
        super().__init__(lattice, parameters, fft_workers=fft_workers)
        self.grain_boundary = grain_boundary

    def grain_coupling_potential(self, c: Array) -> Array:
        if c.shape != self.lattice.shape:
            raise ValueError("field shape does not match lattice")
        return interpolation_derivative(
            c,
            self.grain_boundary.parameters.interpolation_epsilon,
        ) * self.grain_boundary.density_view

    def _local_potential(self, c: Array) -> Array:
        """Build bulk plus GB potential in bounded axial chunks."""

        p = self.parameters
        gb = self.grain_boundary
        moved_c = np.moveaxis(c, gb.axis, 0)
        local = np.empty_like(c, dtype=np.float64)
        moved_local = np.moveaxis(local, gb.axis, 0)
        reshape = (1,) * (moved_c.ndim - 1)
        chunk = 16
        for start in range(0, moved_c.shape[0], chunk):
            stop = min(moved_c.shape[0], start + chunk)
            values = moved_c[start:stop]
            density = gb.grain_energy_density[start:stop].reshape(
                (stop - start,) + reshape
            )
            moved_local[start:stop] = (
                2.0
                * p.barrier
                * values
                * (1.0 - values)
                * (1.0 - 2.0 * values)
                + interpolation_derivative(
                    values,
                    gb.parameters.interpolation_epsilon,
                )
                * density
            )
        return local

    def _composition_and_potential_spectra(
        self, c: Array
    ) -> tuple[Array, Array]:
        """Roy compute_r semantics with N'(c)g_GB included before the FFT."""

        if c.shape != self.lattice.shape:
            raise ValueError("field shape does not match lattice")
        if not np.isfinite(c).all():
            raise ValueError("field must be finite")
        p = self.parameters
        c_hat = self._forward_complex64(c)
        mu_hat = self._forward_complex64(self._local_potential(c))

        flat_c = c_hat.ravel()
        flat_mu = mu_hat.ravel()
        flat_k2 = self._k2.ravel()
        chunk_size = 1_048_576
        for start in range(0, flat_c.size, chunk_size):
            stop = min(flat_c.size, start + chunk_size)
            scaled_k2 = p.kappa * flat_k2[start:stop]
            real = scaled_k2 * flat_c[start:stop].real
            real += flat_mu[start:stop].real
            flat_mu[start:stop].real = real
            imaginary = scaled_k2 * flat_c[start:stop].imag
            imaginary += flat_mu[start:stop].imag
            flat_mu[start:stop].imag = imaginary
        return c_hat, mu_hat

    def _grain_energy(self, c: Array) -> float:
        gb = self.grain_boundary
        moved_c = np.moveaxis(c, gb.axis, 0)
        reshape = (1,) * (moved_c.ndim - 1)
        total = 0.0
        chunk = 16
        for start in range(0, moved_c.shape[0], chunk):
            stop = min(moved_c.shape[0], start + chunk)
            values = moved_c[start:stop]
            density = gb.grain_energy_density[start:stop].reshape(
                (stop - start,) + reshape
            )
            total += float(
                np.sum(
                    interpolation(
                        values,
                        gb.parameters.interpolation_epsilon,
                    )
                    * density,
                    dtype=np.float64,
                )
            )
        return float(self.lattice.cell_volume * total)

    def free_energy(self, c: Array) -> float:
        return float(super().free_energy(c) + self._grain_energy(c))

    def bridge_metadata(self) -> dict[str, Any]:
        return {
            "fixed_orientation_fields": True,
            "grain_boundary_axis": self.grain_boundary.axis,
            "primary_coordinate": self.grain_boundary.primary_coordinate,
            "periodic_image_coordinate": self.grain_boundary.image_coordinate,
            "parameters": self.grain_boundary.parameters.to_dict(),
            "per_plane_energy_audit": (
                self.grain_boundary.per_plane_energy_audit()
            ),
        }


def build_solver(
    lattice: PeriodicLattice,
    roy_parameters: RoyModelParameters,
    *,
    grain_boundary: PeriodicBicrystalProfile | None,
    fft_workers: int = 1,
) -> RoyPseudospectralSolver:
    """Hard-dispatch G=0 to the frozen Roy class with no new arithmetic."""

    if (
        grain_boundary is None
        or grain_boundary.parameters.energy_ratio == 0.0
    ):
        return RoyPseudospectralSolver(
            lattice,
            roy_parameters,
            fft_workers=fft_workers,
        )
    return RoyFixedGrainBoundarySolver(
        lattice,
        roy_parameters,
        grain_boundary,
        fft_workers=fft_workers,
    )
