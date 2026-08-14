"""Synthetic, deterministic checks for isolated-cylinder ridge measurements."""

from __future__ import annotations

import unittest

import numpy as np

from .analysis import (
    area_equivalent_radius,
    area_floor_to_radius,
    cross_section_profiles,
    fold_periodic_profile,
    internal_radius_floor,
    lobe_mass_balance,
    measure_groove_ridge_branches,
    paired_cross_section_delta_area,
    periodic_sector_weights,
    radial_level_contours,
    robust_internal_floor,
    signed_periodic_distance,
)


class CrossSectionProfileTests(unittest.TestCase):
    def test_full_paired_area_matches_phase_volume_exactly(self) -> None:
        rng = np.random.default_rng(221)
        control = rng.normal(0.2, 0.01, size=(9, 11, 13))
        difference = rng.normal(0.0, 2.0e-4, size=control.shape)
        gb = control + difference
        spacing = (0.25, 0.4)
        delta_area = paired_cross_section_delta_area(
            gb,
            control,
            axial_axis=2,
            transverse_spacing=spacing,
        )
        expected = np.sum(gb - control, axis=(0, 1)) * np.prod(spacing)
        np.testing.assert_array_equal(delta_area, expected)
        self.assertAlmostEqual(
            float(np.sum(delta_area) * 0.3),
            float(np.sum(gb - control) * np.prod(spacing) * 0.3),
            places=16,
        )

    def test_area_radius_supports_explicit_phase_values_and_roi(self) -> None:
        field = np.full((21, 21, 4), 0.1)
        first = (np.arange(21) - 10) * 0.5
        second = first.copy()
        roi = first[:, None] ** 2 + second[None, :] ** 2 <= 2.1**2
        normalized = np.linspace(0.7, 1.0, 4)
        for index, value in enumerate(normalized):
            field[..., index][roi] = 0.1 + 0.8 * value
        report = cross_section_profiles(
            field,
            axial_axis=2,
            transverse_spacing=0.5,
            solid_value=0.9,
            vapor_value=0.1,
            radial_roi=roi,
        )
        expected_area = normalized * np.count_nonzero(roi) * 0.25
        np.testing.assert_allclose(report.area, expected_area)
        np.testing.assert_allclose(
            report.radius,
            np.sqrt(expected_area / np.pi),
        )
        self.assertEqual(report.roi_cell_count, int(np.count_nonzero(roi)))

    def test_none_and_scalar_roi_paths(self) -> None:
        field = np.ones((9, 11, 3), dtype=np.float64)
        full = cross_section_profiles(
            field,
            axial_axis=2,
            transverse_spacing=(0.5, 0.25),
            radial_roi=None,
        )
        np.testing.assert_allclose(full.area, 9 * 11 * 0.5 * 0.25)
        self.assertEqual(full.roi_cell_count, 9 * 11)

        first = (np.arange(9) - 4) * 0.5
        second = (np.arange(11) - 5) * 0.25
        expected_mask = (
            first[:, None] ** 2 + second[None, :] ** 2 <= 0.8**2
        )
        scalar = cross_section_profiles(
            field,
            axial_axis=2,
            transverse_spacing=(0.5, 0.25),
            radial_roi=0.8,
            transverse_coordinates=(first, second),
            center=(0.0, 0.0),
        )
        expected_area = np.count_nonzero(expected_mask) * 0.5 * 0.25
        np.testing.assert_allclose(scalar.area, expected_area)
        self.assertEqual(
            scalar.roi_cell_count,
            int(np.count_nonzero(expected_mask)),
        )

    def test_material_negative_area_is_visible_as_nan(self) -> None:
        values = area_equivalent_radius([np.pi, -1.0e-15, -0.1])
        self.assertAlmostEqual(values[0], 1.0)
        self.assertEqual(values[1], 0.0)
        self.assertTrue(np.isnan(values[2]))


class RadialContourTests(unittest.TestCase):
    def test_diffuse_cylinder_contours_recover_analytic_subcell_radii(self) -> None:
        spacing = 0.2
        coordinate = (np.arange(81) - 40) * spacing
        z_count = 7
        axial_radius = 4.0 + np.linspace(-0.15, 0.20, z_count)
        first = coordinate[:, None, None]
        second = coordinate[None, :, None]
        radial = np.sqrt(first * first + second * second)
        width = 0.28
        field = 1.0 / (
            1.0
            + np.exp(
                (radial - axial_radius[None, None, :])
                / width
            )
        )
        reports = radial_level_contours(
            field,
            axial_axis=2,
            transverse_coordinates=(coordinate, coordinate),
            center=(0.0, 0.0),
            levels=(0.45, 0.50, 0.55),
            angle_count=96,
            radial_step=0.025,
            maximum_radius=6.0,
        )
        for level, report in reports.items():
            expected = axial_radius + width * np.log(1.0 / level - 1.0)
            np.testing.assert_allclose(
                report.mean_radius,
                expected,
                atol=1.2e-2,
                rtol=0.0,
            )
            self.assertTrue(np.all(report.valid_fraction == 1.0))
            self.assertLess(float(np.max(report.angular_std)), 4.0e-3)
            self.assertEqual(report.radii.shape, (z_count, 96))

    def test_outermost_descending_crossing_is_used(self) -> None:
        coordinate = np.linspace(-5.0, 5.0, 101)
        radial = np.sqrt(
            coordinate[:, None, None] ** 2
            + coordinate[None, :, None] ** 2
        )
        field = (radial < 3.0).astype(np.float64)
        field[(radial > 0.8) & (radial < 1.1)] = 0.2
        report = radial_level_contours(
            field,
            axial_axis=2,
            transverse_coordinates=(coordinate, coordinate),
            center=(0.0, 0.0),
            levels=(0.5,),
            angle_count=64,
            radial_step=0.025,
            maximum_radius=4.5,
        )[0.5]
        self.assertGreater(float(np.min(report.radii)), 2.9)


class PeriodicBranchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.period = 40.0
        self.spacing = 0.1
        self.z = (np.arange(400) - 200) * self.spacing
        self.planes = (-10.0, 10.0)

    def _asymmetric_signal(self) -> np.ndarray:
        signal = np.zeros_like(self.z)
        ridge_amplitudes = {
            (0, -1): 0.030,
            (0, 1): 0.035,
            (1, -1): 0.040,
            (1, 1): 0.045,
        }
        for plane_index, plane in enumerate(self.planes):
            signed = signed_periodic_distance(self.z, plane, self.period)
            signal += -0.20 * np.exp(-(signed / 0.45) ** 2)
            for side in (-1, 1):
                outward = side * signed
                signal += ridge_amplitudes[(plane_index, side)] * np.exp(
                    -((outward - 2.0) / 0.35) ** 2
                )
        return signal

    def test_folding_retains_two_planes_and_both_sides(self) -> None:
        signal = self._asymmetric_signal()
        branches = fold_periodic_profile(
            self.z,
            signal,
            planes=self.planes,
            period=self.period,
        )
        self.assertEqual(set(branches), {(0, -1), (0, 1), (1, -1), (1, 1)})
        for branch in branches.values():
            self.assertEqual(branch.distance[0], 0.0)
            self.assertTrue(np.all(np.diff(branch.distance) > 0.0))
            self.assertLessEqual(branch.distance[-1], 10.0 + 1.0e-12)

    def test_subcell_feature_measurements_recover_raw_asymmetry(self) -> None:
        signal = self._asymmetric_signal()
        features = measure_groove_ridge_branches(
            self.z,
            signal,
            planes=self.planes,
            period=self.period,
            groove_window=(0.0, 0.8),
            ridge_window=(1.0, 3.2),
            floor=1.0e-6,
        )
        expected_heights = {
            (0, -1): 0.030,
            (0, 1): 0.035,
            (1, -1): 0.040,
            (1, 1): 0.045,
        }
        for key, feature in features.items():
            self.assertTrue(feature.has_negative_groove)
            self.assertTrue(feature.has_positive_ridge)
            self.assertAlmostEqual(feature.groove_depth, 0.20, delta=2.0e-5)
            self.assertAlmostEqual(
                feature.ridge_height,
                expected_heights[key],
                delta=2.0e-5,
            )
            self.assertAlmostEqual(
                feature.ridge_offset_from_plane,
                2.0,
                delta=2.0e-3,
            )

    def test_first_positive_lobe_wins_over_taller_second_lobe(self) -> None:
        distance = np.abs(
            signed_periodic_distance(self.z, 0.0, self.period)
        )
        signal = -0.20 * np.maximum(1.0 - distance / 0.8, 0.0)
        signal += 0.03 * np.maximum(
            1.0 - np.abs(distance - 2.0) / 0.45,
            0.0,
        )
        signal += 0.20 * np.maximum(
            1.0 - np.abs(distance - 5.0) / 0.45,
            0.0,
        )
        features = measure_groove_ridge_branches(
            self.z,
            signal,
            planes=(0.0,),
            period=self.period,
            groove_window=(0.0, 0.9),
            ridge_window=(0.8, 6.0),
            floor=1.0e-6,
        )
        for feature in features.values():
            self.assertTrue(feature.has_positive_ridge)
            self.assertAlmostEqual(feature.ridge_height, 0.03, places=12)
            self.assertAlmostEqual(
                feature.ridge_offset_from_plane,
                2.0,
                places=12,
            )
            self.assertAlmostEqual(
                feature.ridge_offset_from_root,
                2.0,
                places=12,
            )

    def test_subfloor_wiggle_does_not_consume_first_resolved_lobe(self) -> None:
        distance = np.abs(
            signed_periodic_distance(self.z, 0.0, self.period)
        )
        signal = -0.20 * np.maximum(1.0 - distance / 0.8, 0.0)
        signal += 2.0e-7 * np.maximum(
            1.0 - np.abs(distance - 1.2) / 0.25,
            0.0,
        )
        signal += 0.03 * np.maximum(
            1.0 - np.abs(distance - 2.0) / 0.45,
            0.0,
        )
        features = measure_groove_ridge_branches(
            self.z,
            signal,
            planes=(0.0,),
            period=self.period,
            groove_window=(0.0, 0.9),
            ridge_window=(0.8, 3.0),
            floor=1.0e-6,
        )
        for feature in features.values():
            self.assertTrue(feature.has_positive_ridge)
            self.assertAlmostEqual(feature.ridge_height, 0.03, places=12)
            self.assertAlmostEqual(
                feature.ridge_offset_from_plane,
                2.0,
                places=12,
            )

    def test_boundary_only_distant_compensation_is_not_a_ridge(self) -> None:
        distance = np.abs(
            signed_periodic_distance(self.z, 0.0, self.period)
        )
        signal = -0.20 * np.maximum(1.0 - distance / 0.8, 0.0)
        signal += 0.03 * np.clip((distance - 5.0) / 2.0, 0.0, 1.0)
        features = measure_groove_ridge_branches(
            self.z,
            signal,
            planes=(0.0,),
            period=self.period,
            groove_window=(0.0, 0.9),
            ridge_window=(0.8, 7.0),
            floor=1.0e-6,
            maximum_distance=7.0,
        )
        for feature in features.values():
            self.assertFalse(feature.has_positive_ridge)
            self.assertTrue(np.isnan(feature.ridge_value))
            self.assertTrue(np.isnan(feature.ridge_offset_from_plane))
            self.assertTrue(np.isnan(feature.ridge_offset_from_root))

    def test_ridge_before_outward_shifted_groove_is_rejected(self) -> None:
        distance = np.abs(
            signed_periodic_distance(self.z, 0.0, self.period)
        )
        signal = 0.04 * np.maximum(
            1.0 - np.abs(distance - 1.3) / 0.3,
            0.0,
        )
        signal -= 0.20 * np.maximum(
            1.0 - np.abs(distance - 2.0) / 0.4,
            0.0,
        )
        features = measure_groove_ridge_branches(
            self.z,
            signal,
            planes=(0.0,),
            period=self.period,
            groove_window=(1.0, 2.5),
            ridge_window=(1.0, 3.5),
            floor=1.0e-6,
        )
        for feature in features.values():
            self.assertTrue(feature.has_negative_groove)
            self.assertAlmostEqual(feature.groove_offset, 2.0, places=12)
            self.assertFalse(feature.has_positive_ridge)
            self.assertTrue(np.isnan(feature.ridge_value))

    def test_sector_weights_partition_ties_and_lobes_exactly(self) -> None:
        weights = periodic_sector_weights(
            self.z,
            planes=self.planes,
            period=self.period,
        )
        np.testing.assert_allclose(
            np.sum(weights, axis=(0, 1)),
            1.0,
            rtol=0.0,
            atol=0.0,
        )
        delta_area = self._asymmetric_signal()
        delta_area += 1.7e-7
        report = lobe_mass_balance(
            delta_area,
            self.spacing,
            sector_weights=weights,
        )
        self.assertAlmostEqual(
            report.raw_total_integral,
            float(np.sum(delta_area) * self.spacing),
            places=15,
        )
        self.assertAlmostEqual(report.corrected_total_integral, 0.0, places=14)
        self.assertAlmostEqual(report.sector_partition_error, 0.0, places=14)
        self.assertAlmostEqual(
            sum(value.net_integral for value in report.sectors.values()),
            report.corrected_total_integral,
            places=14,
        )
        self.assertEqual(len(report.sectors), 4)


class InternalFloorTests(unittest.TestCase):
    def test_area_floor_is_converted_before_combination(self) -> None:
        converted = float(area_floor_to_radius(0.02, 5.0))
        self.assertAlmostEqual(converted, 0.02 / (10.0 * np.pi))
        report = internal_radius_floor(
            control_radius=[5.0, 5.0002, 4.9999],
            plane_side_ridge_heights=[0.020, 0.021, 0.019, 0.020],
            contour_level_ridge_heights=[
                [0.019, 0.020, 0.018, 0.019],
                [0.020, 0.021, 0.019, 0.020],
                [0.021, 0.022, 0.020, 0.021],
            ],
            uniform_area_residual=0.02,
            reference_radius=5.0,
            representation_radius_floor=7.0e-4,
            contour_radius_signal=[0.020, 0.021],
            area_radius_signal=[0.0202, 0.0209],
            consistency_tolerance=3.0e-4,
        )
        self.assertEqual(report.largest_component, "contour_level_spread")
        self.assertAlmostEqual(
            report.components["uniform_mass_residual"],
            converted,
        )
        self.assertAlmostEqual(
            report.components["floating_point_representation"],
            7.0e-4,
        )
        self.assertTrue(report.contour_area_consistent)
        self.assertAlmostEqual(
            report.contour_area_maximum_difference,
            2.0e-4,
        )

    def test_robust_floor_ignores_nonfinite_values_and_preserves_units(self) -> None:
        floor = robust_internal_floor(
            [np.nan, np.inf, -2.0, 1.0],
            absolute_floor=1.0e-9,
            relative_floor=1.0e-3,
        )
        self.assertGreater(floor, 1.0e-3)
        self.assertLessEqual(floor, 2.0e-3)


if __name__ == "__main__":
    unittest.main()
