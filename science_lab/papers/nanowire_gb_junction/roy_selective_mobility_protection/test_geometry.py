"""Focused tests for the selective-mobility geometry."""

from __future__ import annotations

import unittest

import numpy as np

from ..roy_2021_reproduction.model import PeriodicLattice
from .geometry import (
    FourArmCollarGeometry,
    four_arm_collar_factor,
    four_arm_tubular_collar_factor,
    periodic_distance,
    radial_surface_shell,
    signed_arm_window,
)


class FourArmCollarGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lattice = PeriodicLattice((16, 64, 64), 0.5)
        self.geometry = FourArmCollarGeometry(
            inner_support_distance=4.0,
            outer_support_distance=12.0,
            transition_width=2.0,
            protected_mobility_factor=0.1,
        )

    def test_window_has_compact_support_and_finite_plateau(self) -> None:
        coordinate = (
            np.arange(64, dtype=np.float64) - 32
        ) * self.lattice.spacing
        window = signed_arm_window(coordinate, self.geometry)
        self.assertTrue(np.all(window[np.abs(coordinate) <= 4.0] == 0.0))
        self.assertTrue(np.all(window[np.abs(coordinate) >= 12.0] == 0.0))
        plateau = (
            (np.abs(coordinate) >= 6.0)
            & (np.abs(coordinate) <= 10.0)
        )
        self.assertTrue(np.all(window[plateau] == 1.0))

    def test_factor_is_bounded_symmetric_and_read_only(self) -> None:
        factor = four_arm_collar_factor(self.lattice, self.geometry)
        self.assertEqual(factor.shape, (1, 64, 64))
        self.assertFalse(factor.flags.writeable)
        self.assertAlmostEqual(float(np.min(factor)), 0.1)
        self.assertEqual(float(np.max(factor)), 1.0)
        indices = (-np.arange(64)) % 64
        np.testing.assert_array_equal(factor, factor[:, indices, :])
        np.testing.assert_array_equal(factor, factor[:, :, indices])
        np.testing.assert_array_equal(factor, np.swapaxes(factor, 1, 2))

    def test_smooth_union_never_double_suppresses(self) -> None:
        factor = four_arm_collar_factor(self.lattice, self.geometry)
        self.assertGreaterEqual(float(np.min(factor)), 0.1 - 1.0e-15)

    def test_periodic_distance_uses_minimum_image(self) -> None:
        coordinate = np.array([-4.0, -3.5, 0.0, 3.5])
        measured = periodic_distance(coordinate, 3.5, 8.0)
        np.testing.assert_array_equal(
            measured, np.array([0.5, 1.0, 3.5, 0.0])
        )

    def test_radial_shell_is_compact_with_interface_plateau(self) -> None:
        distance = np.linspace(0.0, 4.0, 17)
        shell = radial_surface_shell(
            distance, radius=2.0, interface_width=0.5
        )
        self.assertTrue(np.all(shell[distance <= 1.0] == 0.0))
        self.assertTrue(np.all(shell[distance >= 3.0] == 0.0))
        plateau = (distance >= 1.5) & (distance <= 2.5)
        self.assertTrue(np.all(shell[plateau] == 1.0))

    def test_tubular_factor_is_identity_far_from_wire_surfaces(self) -> None:
        geometry = FourArmCollarGeometry(
            inner_support_distance=4.0,
            outer_support_distance=12.0,
            transition_width=2.0,
            protected_mobility_factor=0.1,
        )
        factor = four_arm_tubular_collar_factor(
            self.lattice,
            geometry,
            radius=1.5,
            interface_width=0.5,
            first_wire_center_x=-2.0,
            second_wire_center_x=2.0,
        )
        self.assertEqual(factor.shape, self.lattice.shape)
        self.assertFalse(factor.flags.writeable)
        self.assertGreaterEqual(float(np.min(factor)), 0.1 - 1.0e-15)
        self.assertEqual(float(np.max(factor)), 1.0)
        # The domain corner is well outside either compact radial shell.
        self.assertEqual(float(factor[0, 0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
