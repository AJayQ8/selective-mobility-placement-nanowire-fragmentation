"""Small deterministic tests for bridge geometry measurements."""

from __future__ import annotations

import unittest

import numpy as np

from ..roy_2021_reproduction.model import PeriodicLattice, RoyModelParameters
from .geometry import initialize_equilibrium_slab
from .metrics import (
    extrusion_error,
    planar_groove_metrics,
    slab_surface_profiles,
)


class SlabMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lattice = PeriodicLattice((256, 96, 4), 0.5)
        self.field = initialize_equilibrium_slab(
            self.lattice,
            RoyModelParameters(),
            normal_axis=1,
            half_height=12.0,
        )

    def test_flat_surface_is_resolved_and_extrusion_invariant(self) -> None:
        bottom, top = slab_surface_profiles(
            self.field,
            self.lattice,
            tangent_axis=0,
            normal_axis=1,
            extruded_axis=2,
            level=0.5,
        )
        self.assertTrue(np.isfinite(bottom).all())
        self.assertTrue(np.isfinite(top).all())
        self.assertLess(float(np.ptp(bottom)), 1.0e-14)
        self.assertLess(float(np.ptp(top)), 1.0e-14)
        self.assertEqual(extrusion_error(self.field, extruded_axis=2), 0.0)

    def test_flat_slab_reports_no_groove_and_straight_angle(self) -> None:
        width = 2.0 * np.sqrt(2.0)
        report = planar_groove_metrics(
            self.field,
            self.lattice,
            tangent_axis=0,
            normal_axis=1,
            extruded_axis=2,
            planes=(0.0, -64.0),
            width=width,
        )
        np.testing.assert_allclose(report["depths"], 0.0, atol=1.0e-14)
        np.testing.assert_allclose(
            report["primary_angles_deg"],
            180.0,
            atol=1.0e-10,
        )
        for threshold_report in report["topology"].values():
            self.assertTrue(threshold_report["solid_connected"])
            self.assertTrue(threshold_report["no_enclosed_vapor_cavity"])


if __name__ == "__main__":
    unittest.main()

