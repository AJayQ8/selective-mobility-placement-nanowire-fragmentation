"""Numerical contract tests for the spatial-mobility operator."""

from __future__ import annotations

import unittest

import numpy as np
from scipy import fft as scipy_fft

from ..roy_2021_reproduction.model import (
    PeriodicLattice,
    RoyModelParameters,
    RoyPseudospectralSolver,
    source_wave_numbers,
)
from .model import SpatialMobilityRoySolver


class SpatialMobilityRoySolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lattice = PeriodicLattice((8, 8, 8), 0.5)
        self.parameters = RoyModelParameters()
        self.rng = np.random.default_rng(831)

    @staticmethod
    def _independent_spatial_step(
        field: np.ndarray,
        factor: np.ndarray,
        *,
        timestep: float,
    ) -> np.ndarray:
        cells = field.shape[0]
        modes = source_wave_numbers(cells, 0.5)
        wave_numbers = (
            modes[:, None, None],
            modes[None, :, None],
            modes[None, None, :],
        )
        k2 = sum(wave_number**2 for wave_number in wave_numbers)
        c_hat = scipy_fft.fftn(
            field.astype(np.float32)
        ).astype(np.complex64)
        bulk = 2.0 * field * (1.0 - field) * (1.0 - 2.0 * field)
        bulk_hat = scipy_fft.fftn(
            bulk.astype(np.float32)
        ).astype(np.complex64)
        mu_hat = (
            bulk_hat.astype(np.complex128)
            + k2 * c_hat.astype(np.complex128)
        ).astype(np.complex64)
        mobility = np.sqrt(np.abs(field - field * field)) * factor
        real_increment = np.zeros(field.shape, dtype=np.float64)
        imaginary_increment = np.zeros(field.shape, dtype=np.float64)
        for wave_number in wave_numbers:
            gradient_hat = (
                1j * wave_number * mu_hat.astype(np.complex128)
            ).astype(np.complex64)
            gradient = scipy_fft.ifftn(gradient_hat).real
            flux_hat = scipy_fft.fftn(
                (mobility * gradient).astype(np.float32)
            ).astype(np.complex64)
            real_increment -= wave_number * flux_hat.imag
            imaginary_increment += wave_number * flux_hat.real
        denominator = 1.0 + 0.5 * timestep * k2**2
        updated_hat = np.empty_like(c_hat)
        updated_hat.real = (
            c_hat.real + timestep * real_increment / denominator
        )
        updated_hat.imag = (
            c_hat.imag + timestep * imaginary_increment / denominator
        )
        return scipy_fft.ifftn(updated_hat).real.astype(np.float64)

    def test_all_ones_path_is_bitwise_identical(self) -> None:
        field = 0.1 + 0.8 * self.rng.random(self.lattice.shape)
        baseline = RoyPseudospectralSolver(
            self.lattice, self.parameters, fft_workers=1
        ).propose_step(field)
        treated = SpatialMobilityRoySolver(
            self.lattice,
            self.parameters,
            np.ones((1, 8, 8)),
            fft_workers=1,
        ).propose_step(field)
        np.testing.assert_array_equal(treated, baseline)

    def test_spatial_factor_is_applied_to_mobility(self) -> None:
        field = 0.1 + 0.8 * self.rng.random(self.lattice.shape)
        factor = np.linspace(0.1, 1.0, 8)[None, None, :]
        solver = SpatialMobilityRoySolver(
            self.lattice,
            self.parameters,
            factor,
            fft_workers=1,
        )
        expected = (
            self.parameters.mobility_prefactor
            * np.sqrt(np.abs(field - field * field))
            * factor
        )
        np.testing.assert_array_equal(solver.mobility(field), expected)

    def test_one_step_matches_independent_inside_divergence_oracle(
        self,
    ) -> None:
        field = 0.1 + 0.8 * self.rng.random(self.lattice.shape)
        i, j, k = np.indices(self.lattice.shape, dtype=np.float64)
        factor = 0.1 + 0.9 * (
            0.2 * i / 7.0 + 0.3 * j / 7.0 + 0.5 * k / 7.0
        )
        timestep = 0.03
        expected = self._independent_spatial_step(
            field,
            factor,
            timestep=timestep,
        )
        measured = SpatialMobilityRoySolver(
            self.lattice,
            self.parameters,
            factor,
            fft_workers=1,
        ).propose_step(field, timestep=timestep)
        np.testing.assert_allclose(
            measured, expected, rtol=2.0e-6, atol=2.0e-6
        )

    def test_constant_chemical_potential_null_with_spatial_factor(self) -> None:
        factor = 0.1 + 0.9 * self.rng.random((1, 8, 8))
        solver = SpatialMobilityRoySolver(
            self.lattice,
            self.parameters,
            factor,
            fft_workers=1,
        )
        field = np.full(self.lattice.shape, 0.37, dtype=np.float64)
        measured = solver.propose_step(field, timestep=0.05)
        np.testing.assert_allclose(measured, field, atol=2.0e-7, rtol=0.0)

    def test_uniform_factor_time_rescaling_converges_under_refinement(
        self,
    ) -> None:
        lattice = PeriodicLattice((16, 16, 16), 0.5)
        coordinates = np.arange(16, dtype=np.float64)
        x = np.cos(2.0 * np.pi * coordinates / 16.0)[:, None, None]
        y = np.cos(2.0 * np.pi * coordinates / 16.0)[None, :, None]
        z = np.cos(2.0 * np.pi * coordinates / 16.0)[None, None, :]
        initial = 0.45 + (0.05 / 3.0) * (x + y + z)
        q = 0.25
        baseline = RoyPseudospectralSolver(
            lattice, self.parameters, fft_workers=1
        )
        treated = SpatialMobilityRoySolver(
            lattice,
            self.parameters,
            np.full((1, 1, 1), q),
            fft_workers=1,
        )
        errors = []
        for timestep in (0.2, 0.1, 0.05):
            expected = initial.copy()
            measured = initial.copy()
            for _ in range(int(round(1.0 / timestep))):
                expected = baseline.propose_step(
                    expected, timestep=q * timestep
                )
                measured = treated.propose_step(
                    measured, timestep=timestep
                )
            errors.append(
                float(
                    np.linalg.norm(measured - expected)
                    / np.linalg.norm(expected)
                )
            )
        self.assertLessEqual(errors[1], 0.65 * errors[0])
        self.assertLessEqual(errors[2], 0.65 * errors[1])
        self.assertLess(errors[2], 2.0e-5)

    def test_one_step_remains_nearly_conservative(self) -> None:
        field = 0.2 + 0.6 * self.rng.random(self.lattice.shape)
        factor = 0.1 + 0.9 * self.rng.random((1, 8, 8))
        solver = SpatialMobilityRoySolver(
            self.lattice,
            self.parameters,
            factor,
            fft_workers=1,
        )
        measured = solver.propose_step(field, timestep=0.01)
        relative = abs(
            float(np.sum(measured) - np.sum(field))
        ) / abs(float(np.sum(field)))
        self.assertLess(relative, 1.0e-6)

    def test_smooth_spatial_run_dissipates_energy_and_preserves_mass(
        self,
    ) -> None:
        lattice = PeriodicLattice((16, 16, 16), 0.5)
        coordinates = np.arange(16, dtype=np.float64)
        x = np.cos(2.0 * np.pi * coordinates / 16.0)[:, None, None]
        y = np.cos(2.0 * np.pi * coordinates / 16.0)[None, :, None]
        z = np.cos(2.0 * np.pi * coordinates / 16.0)[None, None, :]
        field = 0.45 + (0.05 / 3.0) * (x + y + z)
        factor = 0.55 + 0.35 * x**2
        solver = SpatialMobilityRoySolver(
            lattice,
            self.parameters,
            factor,
            fft_workers=1,
        )
        initial_mass = float(np.sum(field))
        previous_energy = solver.free_energy(field)
        initial_energy = previous_energy
        for _ in range(20):
            field = solver.propose_step(field, timestep=0.05)
            energy = solver.free_energy(field)
            relative_rise = (
                energy - previous_energy
            ) / abs(previous_energy)
            self.assertLessEqual(relative_rise, 1.0e-7)
            previous_energy = energy
        relative_mass_drift = abs(
            float(np.sum(field)) - initial_mass
        ) / abs(initial_mass)
        self.assertLess(relative_mass_drift, 1.0e-6)
        self.assertLess(previous_energy, initial_energy)
        self.assertTrue(np.isfinite(field).all())
        self.assertLess(float(np.max(np.abs(field))), 2.0)

    def test_symmetric_operator_is_equivariant_under_axis_exchange(
        self,
    ) -> None:
        coordinates = np.arange(8, dtype=np.float64)
        x = np.cos(2.0 * np.pi * coordinates / 8.0)[:, None, None]
        y = np.cos(2.0 * np.pi * coordinates / 8.0)[None, :, None]
        z = np.cos(2.0 * np.pi * coordinates / 8.0)[None, None, :]
        field = 0.45 + 0.01 * x + 0.02 * (y + z)
        factor = 0.4 + 0.6 * (
            0.5 * np.cos(2.0 * np.pi * coordinates / 8.0)[None, :, None] ** 2
            + 0.5 * np.cos(2.0 * np.pi * coordinates / 8.0)[None, None, :] ** 2
        )
        solver = SpatialMobilityRoySolver(
            self.lattice,
            self.parameters,
            factor,
            fft_workers=1,
        )
        measured = solver.propose_step(field, timestep=0.1)
        exchanged = np.swapaxes(measured, 1, 2)
        relative = float(
            np.linalg.norm(measured - exchanged)
            / np.linalg.norm(measured)
        )
        self.assertLess(relative, 5.0e-7)
        self.assertLess(float(np.max(np.abs(measured - exchanged))), 5.0e-7)


if __name__ == "__main__":
    unittest.main()
