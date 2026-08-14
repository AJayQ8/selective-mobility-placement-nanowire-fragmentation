from __future__ import annotations

import unittest

import numpy as np

from .prolongation import (
    periodic_fourier_prolong_factor2,
    prolongation_metrics,
)


class PeriodicFourierProlongationTests(unittest.TestCase):
    def test_analytic_periodic_modes(self) -> None:
        shape = (8, 10, 12)
        i, j, k = np.meshgrid(
            *(np.arange(size, dtype=np.float64) for size in shape),
            indexing="ij",
        )
        coarse = (
            0.7
            + 0.2 * np.cos(2.0 * np.pi * 2.0 * i / shape[0])
            - 0.11 * np.sin(2.0 * np.pi * 3.0 * j / shape[1])
            + 0.09
            * np.cos(
                2.0 * np.pi * i / shape[0]
                + 2.0 * np.pi * 2.0 * k / shape[2]
            )
        )
        fine = periodic_fourier_prolong_factor2(coarse)
        fi, fj, fk = np.meshgrid(
            *(np.arange(2 * size, dtype=np.float64) for size in shape),
            indexing="ij",
        )
        expected = (
            0.7
            + 0.2 * np.cos(2.0 * np.pi * 2.0 * (fi / 2.0) / shape[0])
            - 0.11 * np.sin(2.0 * np.pi * 3.0 * (fj / 2.0) / shape[1])
            + 0.09
            * np.cos(
                2.0 * np.pi * (fi / 2.0) / shape[0]
                + 2.0 * np.pi * 2.0 * (fk / 2.0) / shape[2]
            )
        )
        np.testing.assert_allclose(fine, expected, rtol=0.0, atol=2.0e-13)

    def test_even_grid_nyquist_modes_are_split_correctly(self) -> None:
        shape = (8, 10, 12)
        i, j, k = np.meshgrid(
            *(np.arange(size, dtype=np.float64) for size in shape),
            indexing="ij",
        )
        coarse = (-1.0) ** i + 0.4 * (-1.0) ** j - 0.2 * (-1.0) ** k
        fine = periodic_fourier_prolong_factor2(coarse)
        fi, fj, fk = np.meshgrid(
            *(np.arange(2 * size, dtype=np.float64) for size in shape),
            indexing="ij",
        )
        expected = (
            np.cos(np.pi * fi / 2.0)
            + 0.4 * np.cos(np.pi * fj / 2.0)
            - 0.2 * np.cos(np.pi * fk / 2.0)
        )
        np.testing.assert_allclose(fine, expected, rtol=0.0, atol=2.0e-13)

    def test_mixed_nyquist_and_resolved_mode(self) -> None:
        shape = (8, 10, 12)
        i, j, k = np.meshgrid(
            *(np.arange(size, dtype=np.float64) for size in shape),
            indexing="ij",
        )
        coarse = (-1.0) ** i * np.cos(2.0 * np.pi * 2.0 * k / shape[2])
        fine = periodic_fourier_prolong_factor2(coarse)
        fi, _, fk = np.meshgrid(
            *(np.arange(2 * size, dtype=np.float64) for size in shape),
            indexing="ij",
        )
        expected = np.cos(np.pi * fi / 2.0) * np.cos(
            2.0 * np.pi * 2.0 * (fk / 2.0) / shape[2]
        )
        np.testing.assert_allclose(fine, expected, rtol=0.0, atol=2.0e-13)

    def test_three_axis_nyquist_corner(self) -> None:
        shape = (8, 10, 12)
        i, j, k = np.meshgrid(
            *(np.arange(size, dtype=np.float64) for size in shape),
            indexing="ij",
        )
        coarse = (-1.0) ** (i + j + k)
        fine = periodic_fourier_prolong_factor2(coarse)
        fi, fj, fk = np.meshgrid(
            *(np.arange(2 * size, dtype=np.float64) for size in shape),
            indexing="ij",
        )
        expected = (
            np.cos(np.pi * fi / 2.0)
            * np.cos(np.pi * fj / 2.0)
            * np.cos(np.pi * fk / 2.0)
        )
        np.testing.assert_allclose(fine, expected, rtol=0.0, atol=2.0e-13)

    def test_random_field_recovers_all_coincident_nodes(self) -> None:
        coarse = np.random.default_rng(2292).normal(size=(8, 10, 12))
        fine = periodic_fourier_prolong_factor2(coarse)
        np.testing.assert_allclose(
            fine[::2, ::2, ::2], coarse, rtol=0.0, atol=2.0e-13
        )

    def test_mean_and_physical_mass_are_preserved(self) -> None:
        coarse = np.random.default_rng(130363).normal(size=(8, 10, 12)) + 0.3
        fine = periodic_fourier_prolong_factor2(coarse)
        metrics = prolongation_metrics(
            coarse,
            fine,
            coarse_spacing=0.5,
            fine_spacing=0.25,
        )
        self.assertLess(metrics["mean_abs_difference"], 2.0e-15)
        self.assertLess(metrics["mass_relative_difference"], 2.0e-15)

    def test_rejects_non_factor_two_spacing(self) -> None:
        coarse = np.ones((4, 4, 4))
        fine = periodic_fourier_prolong_factor2(coarse)
        with self.assertRaisesRegex(ValueError, "exactly half"):
            prolongation_metrics(
                coarse, fine, coarse_spacing=0.5, fine_spacing=0.3
            )


if __name__ == "__main__":
    unittest.main()
