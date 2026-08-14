"""Focused mathematical and exact-bypass tests for the fixed-GB bridge."""

from __future__ import annotations

import unittest

import numpy as np

from ..roy_2021_reproduction.model import (
    PeriodicLattice,
    RoyModelParameters,
    RoyPseudospectralSolver,
)
from .model import (
    FixedGrainBoundaryParameters,
    RoyFixedGrainBoundarySolver,
    build_periodic_bicrystal_profile,
    build_solver,
    interpolation,
    interpolation_derivative,
)


class ParameterMappingTests(unittest.TestCase):
    def test_roy_mapping_recovers_frozen_coefficients(self) -> None:
        mapped = FixedGrainBoundaryParameters.matched_to_roy(
            RoyModelParameters(),
            energy_ratio=0.35,
        )
        self.assertAlmostEqual(mapped.surface_width, 2.0 * np.sqrt(2.0))
        self.assertAlmostEqual(mapped.gamma_surface, np.sqrt(2.0) / 6.0)
        self.assertAlmostEqual(mapped.gamma_grain_boundary, 7.0 * np.sqrt(2.0) / 60.0)
        self.assertAlmostEqual(mapped.beta, 7.0 / 120.0)
        self.assertAlmostEqual(mapped.kappa_eta, 7.0 / 20.0)
        self.assertAlmostEqual(
            mapped.predicted_dihedral_angle_deg,
            float(np.degrees(2.0 * np.arccos(0.35))),
        )

    def test_interpolation_endpoints_have_the_model_ii_nulls(self) -> None:
        endpoints = np.array([0.0, 1.0])
        np.testing.assert_array_equal(
            interpolation(endpoints),
            np.array([0.0, 1.0]),
        )
        np.testing.assert_array_equal(
            interpolation_derivative(endpoints),
            np.zeros(2),
        )


class ProfileTests(unittest.TestCase):
    def test_periodic_profile_has_two_balanced_well_resolved_planes(self) -> None:
        lattice = PeriodicLattice((256, 4, 4), 0.5)
        parameters = FixedGrainBoundaryParameters.matched_to_roy(
            RoyModelParameters()
        )
        profile = build_periodic_bicrystal_profile(
            lattice,
            parameters,
            axis=0,
            primary_coordinate=0.0,
        )
        np.testing.assert_allclose(
            profile.eta_1 + profile.eta_2,
            1.0,
            rtol=0.0,
            atol=0.0,
        )
        self.assertGreaterEqual(float(np.min(profile.grain_energy_density)), 0.0)
        audit = profile.per_plane_energy_audit()
        self.assertLess(audit["primary_relative_error"], 5.0e-3)
        self.assertLess(audit["image_relative_error"], 5.0e-3)
        self.assertLess(audit["image_mismatch_relative"], 1.0e-12)


class SolverBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lattice = PeriodicLattice((8, 8, 8), 0.5)
        self.roy = RoyModelParameters()

    def test_disabled_factory_returns_the_frozen_class_itself(self) -> None:
        solver = build_solver(
            self.lattice,
            self.roy,
            grain_boundary=None,
        )
        self.assertIs(type(solver), RoyPseudospectralSolver)

        zero = FixedGrainBoundaryParameters.matched_to_roy(
            self.roy,
            energy_ratio=0.0,
        )
        zero_profile = build_periodic_bicrystal_profile(
            self.lattice,
            zero,
            axis=0,
            primary_coordinate=0.0,
        )
        zero_solver = build_solver(
            self.lattice,
            self.roy,
            grain_boundary=zero_profile,
        )
        self.assertIs(type(zero_solver), RoyPseudospectralSolver)

    def test_disabled_path_is_bitwise_identical(self) -> None:
        rng = np.random.default_rng(404)
        field = 0.05 + 0.9 * rng.random(self.lattice.shape)
        legacy = RoyPseudospectralSolver(self.lattice, self.roy)
        bridge_off = build_solver(
            self.lattice,
            self.roy,
            grain_boundary=None,
        )
        np.testing.assert_array_equal(
            bridge_off.chemical_potential_spectrum(field),
            legacy.chemical_potential_spectrum(field),
        )
        np.testing.assert_array_equal(
            bridge_off.propose_step(field),
            legacy.propose_step(field),
        )
        self.assertEqual(bridge_off.free_energy(field), legacy.free_energy(field))
        self.assertEqual(
            bridge_off.last_inverse_imaginary_linf,
            legacy.last_inverse_imaginary_linf,
        )

    def test_dense_bicrystal_has_exactly_zero_coupling_force(self) -> None:
        lattice = PeriodicLattice((256, 4, 4), 0.5)
        mapped = FixedGrainBoundaryParameters.matched_to_roy(self.roy)
        profile = build_periodic_bicrystal_profile(
            lattice,
            mapped,
            axis=0,
            primary_coordinate=0.0,
        )
        solver = RoyFixedGrainBoundarySolver(
            lattice,
            self.roy,
            profile,
        )
        field = np.ones(lattice.shape, dtype=np.float64)
        np.testing.assert_array_equal(
            solver.grain_coupling_potential(field),
            np.zeros_like(field),
        )
        np.testing.assert_array_equal(
            solver.chemical_potential_spectrum(field),
            np.zeros(lattice.shape, dtype=np.complex64),
        )
        updated = solver.propose_step(field)
        np.testing.assert_array_equal(updated, field)
        self.assertEqual(float(np.sum(updated)), float(np.sum(field)))

    def test_positive_g_adds_finite_force_and_energy(self) -> None:
        mapped = FixedGrainBoundaryParameters.matched_to_roy(self.roy)
        profile = build_periodic_bicrystal_profile(
            self.lattice,
            mapped,
            axis=0,
            primary_coordinate=0.0,
        )
        solver = RoyFixedGrainBoundarySolver(
            self.lattice,
            self.roy,
            profile,
        )
        field = np.full(self.lattice.shape, 0.5, dtype=np.float64)
        coupling = solver.grain_coupling_potential(field)
        self.assertTrue(np.isfinite(coupling).all())
        self.assertGreater(float(np.max(coupling)), 0.0)
        legacy = RoyPseudospectralSolver(self.lattice, self.roy)
        self.assertGreater(solver.free_energy(field), legacy.free_energy(field))


if __name__ == "__main__":
    unittest.main()
