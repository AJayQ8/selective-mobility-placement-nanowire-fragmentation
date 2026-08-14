"""Focused tests for the independent Roy et al. CPU port."""

from __future__ import annotations

import unittest

import numpy as np
from scipy import fft as scipy_fft

from .model import (
    GslTaus2,
    PeriodicLattice,
    RoyDeg90Definition,
    RoyModelParameters,
    RoyPseudospectralSolver,
    apply_released_overlapping_noise,
    digital_gap_diagnostics,
    initialize_strict_deg90,
    released_noise_layout,
    source_wave_numbers,
)


class GslTaus2Tests(unittest.TestCase):
    def test_known_gsl_taus2_vector(self) -> None:
        generator = GslTaus2(1)
        value = 0
        for _ in range(10_000):
            value = generator.get_uint32()
        self.assertEqual(value, 2_733_957_125)

    def test_uniform_pos_excludes_endpoints(self) -> None:
        values = GslTaus2(2292).uniform_pos_array(1000)
        self.assertTrue(np.all(values > 0.0))
        self.assertTrue(np.all(values < 1.0))


class ReleasedGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.definition = RoyDeg90Definition(
            lattice=PeriodicLattice((96, 32, 32), 0.5),
            noise_amplitude=0.0,
        )

    def test_strict_masks_leave_the_released_digital_gap(self) -> None:
        _, masks = initialize_strict_deg90(self.definition)
        measured = digital_gap_diagnostics(
            masks, self.definition.lattice
        )
        self.assertEqual(measured["first_wire_last_solid_index"], 59)
        self.assertEqual(measured["empty_indices_between_wires"], [60])
        self.assertEqual(measured["second_wire_first_solid_index"], 61)
        self.assertFalse(measured["center_line_union_connected_at_t0"])

    def test_source_uses_positive_nyquist_mode(self) -> None:
        measured = source_wave_numbers(8, 0.5)
        scale = 2.0 * np.pi / (8 * 0.5)
        np.testing.assert_allclose(
            measured / scale,
            np.array([0, 1, 2, 3, 4, -3, -2, -1]),
        )

    def test_positive_nyquist_derivative_has_discarded_imaginary_part(
        self,
    ) -> None:
        alternating = (-1.0) ** np.arange(8, dtype=float)
        spectrum = scipy_fft.fft(alternating.astype(np.float32))
        derivative = scipy_fft.ifft(
            1j * source_wave_numbers(8, 0.5) * spectrum
        )
        self.assertLess(float(np.max(np.abs(derivative.real))), 1.0e-12)
        self.assertGreater(float(np.max(np.abs(derivative.imag))), 1.0)


class ReleasedNoiseTests(unittest.TestCase):
    @staticmethod
    def _literal_released_noise(
        field: np.ndarray,
        amplitude: float,
        seed: int,
    ) -> None:
        generator = GslTaus2(seed)
        nx, ny, nz = field.shape
        flat = field.ravel()
        for i1 in range(nx):
            for i2 in range(ny):
                for i3 in range(nz):
                    released_index = i3 + nx * (i2 + ny * i1)
                    flat[released_index] += amplitude * (
                        0.5 - generator.uniform_pos()
                    )

    def test_released_stride_modifies_only_the_prefix(self) -> None:
        field = np.zeros((8, 4, 16), dtype=np.float64)
        layout = apply_released_overlapping_noise(
            field, 1.0e-3, 2292, chunk_size=64
        )
        expected = released_noise_layout(PeriodicLattice((8, 4, 16)))
        self.assertEqual(layout, expected)
        self.assertEqual(layout.maximum_write_overlap, 2)
        flat = field.ravel()
        self.assertGreater(np.count_nonzero(flat[: layout.unique_target_prefix_length]), 0)
        self.assertTrue(
            np.array_equal(
                flat[layout.unique_target_prefix_length :],
                np.zeros_like(flat[layout.unique_target_prefix_length :]),
            )
        )

    def test_noise_application_is_deterministic(self) -> None:
        first = np.zeros((8, 4, 16), dtype=np.float64)
        second = np.zeros_like(first)
        apply_released_overlapping_noise(first, 1.0e-3, 2292)
        apply_released_overlapping_noise(second, 1.0e-3, 2292)
        np.testing.assert_array_equal(first, second)

    def test_chunked_noise_exactly_matches_literal_source_order(self) -> None:
        initial = np.linspace(
            -0.2,
            0.2,
            num=4 * 4 * 8,
            dtype=np.float64,
        ).reshape(4, 4, 8)
        expected = initial.copy()
        self._literal_released_noise(expected, 1.0e-3, 2292)
        for chunk_size in (8, 37):
            measured = initial.copy()
            apply_released_overlapping_noise(
                measured,
                1.0e-3,
                2292,
                chunk_size=chunk_size,
            )
            np.testing.assert_array_equal(measured, expected)


class PseudospectralStepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lattice = PeriodicLattice((8, 8, 8), 0.5)
        self.parameters = RoyModelParameters()
        self.solver = RoyPseudospectralSolver(
            self.lattice, self.parameters, fft_workers=1
        )

    def _independent_step(self, c: np.ndarray) -> np.ndarray:
        kx = source_wave_numbers(8, 0.5)[:, None, None]
        ky = source_wave_numbers(8, 0.5)[None, :, None]
        kz = source_wave_numbers(8, 0.5)[None, None, :]
        k2 = kx**2 + ky**2 + kz**2
        c_hat = scipy_fft.fftn(c.astype(np.float32)).astype(np.complex64)
        bulk = 2.0 * c * (1.0 - c) * (1.0 - 2.0 * c)
        g_hat = scipy_fft.fftn(bulk.astype(np.float32)).astype(np.complex64)
        mu_hat = (
            g_hat.astype(np.complex128)
            + k2 * c_hat.astype(np.complex128)
        ).astype(np.complex64)
        mobility = np.sqrt(np.abs(c - c * c))
        real_increment = np.zeros(c.shape, dtype=np.float64)
        imag_increment = np.zeros(c.shape, dtype=np.float64)
        for wave_number in (kx, ky, kz):
            gradient_hat = (
                1j * wave_number * mu_hat.astype(np.complex128)
            ).astype(np.complex64)
            gradient = scipy_fft.ifftn(gradient_hat).real
            flux_hat = scipy_fft.fftn(
                (mobility * gradient).astype(np.float32)
            ).astype(np.complex64)
            real_increment -= wave_number * flux_hat.imag
            imag_increment += wave_number * flux_hat.real
        denominator = 1.0 + 0.5 * k2**2
        updated_hat = np.empty_like(c_hat)
        updated_hat.real = c_hat.real + real_increment / denominator
        updated_hat.imag = c_hat.imag + imag_increment / denominator
        return scipy_fft.ifftn(updated_hat).real.astype(np.float64)

    def test_one_step_matches_independent_equation_11_translation(self) -> None:
        rng = np.random.default_rng(72)
        c = 0.1 + 0.8 * rng.random(self.lattice.shape)
        expected = self._independent_step(c)
        measured = self.solver.propose_step(c)
        np.testing.assert_allclose(
            measured, expected, rtol=2.0e-6, atol=2.0e-6
        )

    def test_compute_r_uses_one_float32_rounding_like_cuda(self) -> None:
        rng = np.random.default_rng(17)
        c = 0.1 + 0.8 * rng.random(self.lattice.shape)
        modes = source_wave_numbers(8, 0.5)
        k2 = (
            modes[:, None, None] ** 2
            + modes[None, :, None] ** 2
            + modes[None, None, :] ** 2
        )
        c_hat = scipy_fft.fftn(c.astype(np.float32)).astype(np.complex64)
        bulk = 2.0 * c * (1.0 - c) * (1.0 - 2.0 * c)
        g_hat = scipy_fft.fftn(bulk.astype(np.float32)).astype(np.complex64)
        expected = (
            g_hat.astype(np.complex128)
            + k2 * c_hat.astype(np.complex128)
        ).astype(np.complex64)
        measured = self.solver.chemical_potential_spectrum(c)
        np.testing.assert_array_equal(measured, expected)

    def test_constant_field_is_stationary_and_mass_mode_is_preserved(self) -> None:
        c = np.full(self.lattice.shape, 0.37, dtype=np.float64)
        measured = self.solver.propose_step(c)
        np.testing.assert_allclose(measured, c, atol=2.0e-7, rtol=0.0)
        self.assertLess(
            abs(float(np.sum(measured) - np.sum(c))) / float(np.sum(c)),
            1.0e-6,
        )

    def test_noisy_field_remains_finite_and_nearly_conservative(self) -> None:
        rng = np.random.default_rng(10)
        c = rng.random(self.lattice.shape)
        measured = self.solver.propose_step(c)
        self.assertTrue(np.isfinite(measured).all())
        relative_mass_error = abs(
            float(np.sum(measured) - np.sum(c))
        ) / abs(float(np.sum(c)))
        self.assertLess(relative_mass_error, 1.0e-5)


if __name__ == "__main__":
    unittest.main()
