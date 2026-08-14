"""Tests for the K2/K3 equal-budget control definitions."""

from __future__ import annotations

import unittest

import numpy as np

from ..roy_2021_reproduction.model import PeriodicLattice
from .equal_budget_controls import (
    centered_cap_window,
    derive_equal_budget_controls,
    junction_centered_tubular_cap_factor,
)
from .geometry import FourArmCollarGeometry


class EqualBudgetControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lattice = PeriodicLattice((16, 64, 64), 0.5)
        self.geometry = FourArmCollarGeometry(
            inner_support_distance=4.0,
            outer_support_distance=12.0,
            transition_width=2.0,
            protected_mobility_factor=0.1,
        )

    def test_centered_cap_window_is_symmetric_and_compact(self) -> None:
        coordinate = (
            np.arange(64, dtype=np.float64) - 32
        ) * self.lattice.spacing
        window = centered_cap_window(
            coordinate,
            half_support_distance=8.0,
            transition_width=2.0,
        )
        self.assertTrue(np.all(window[np.abs(coordinate) <= 6.0] == 1.0))
        self.assertTrue(np.all(window[np.abs(coordinate) >= 8.0] == 0.0))
        indices = (-np.arange(64)) % 64
        np.testing.assert_array_equal(window, window[indices])

    def test_junction_cap_is_bounded_symmetric_and_read_only(self) -> None:
        factor = junction_centered_tubular_cap_factor(
            self.lattice,
            half_support_distance=8.0,
            transition_width=2.0,
            protected_mobility_factor=0.1,
            radius=1.5,
            interface_width=0.5,
            first_wire_center_x=-2.0,
            second_wire_center_x=2.0,
        )
        self.assertEqual(factor.shape, self.lattice.shape)
        self.assertFalse(factor.flags.writeable)
        self.assertGreaterEqual(float(np.min(factor)), 0.1 - 1.0e-15)
        self.assertEqual(float(np.max(factor)), 1.0)
        indices = (-np.arange(64)) % 64
        np.testing.assert_array_equal(factor, factor[:, indices, :])
        np.testing.assert_array_equal(factor, factor[:, :, indices])

    def test_derived_controls_match_budget(self) -> None:
        nx, ny, nz = self.lattice.shape
        x = (
            np.arange(nx, dtype=np.float64) - nx // 2
        ) * self.lattice.spacing
        y = (
            np.arange(ny, dtype=np.float64) - ny // 2
        ) * self.lattice.spacing
        z = (
            np.arange(nz, dtype=np.float64) - nz // 2
        ) * self.lattice.spacing
        first_r = np.sqrt((x[:, None] + 2.0) ** 2 + y[None, :] ** 2)
        second_r = np.sqrt((x[:, None] - 2.0) ** 2 + z[None, :] ** 2)
        first = 0.5 * (1.0 - np.tanh((first_r - 1.5) / 0.5))
        second = 0.5 * (1.0 - np.tanh((second_r - 1.5) / 0.5))
        field = 1.0 - (
            1.0 - first[:, :, None]
        ) * (1.0 - second[:, None, :])
        definition, k1, k2, k3 = derive_equal_budget_controls(
            field,
            self.lattice,
            inward_geometry=self.geometry,
            radius=1.5,
            interface_width=0.5,
            transition_width=2.0,
        )
        self.assertEqual(k1.shape, self.lattice.shape)
        self.assertEqual(k2.shape, self.lattice.shape)
        self.assertEqual(k3.shape, (1, 1, 1))
        self.assertLess(definition.junction_relative_mismatch, 1.0e-10)
        self.assertLess(definition.uniform_relative_mismatch, 1.0e-12)
        self.assertGreater(definition.uniform_mobility_factor, 0.1)
        self.assertLess(definition.uniform_mobility_factor, 1.0)


if __name__ == "__main__":
    unittest.main()
