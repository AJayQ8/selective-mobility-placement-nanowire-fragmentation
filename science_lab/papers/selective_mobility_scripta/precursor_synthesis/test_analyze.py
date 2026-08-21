"""Fail-closed checks for the zero-step pre-fragmentation synthesis."""

from __future__ import annotations

import csv
import json
import unittest

from . import analyze


class PrecursorSynthesisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = analyze.main([])
        with (
            analyze.DEFAULT_OUTPUT / "source_a_placement_precursor.csv"
        ).open(newline="", encoding="utf-8") as handle:
            cls.source_a = list(csv.DictReader(handle))
        with (
            analyze.DEFAULT_OUTPUT / "matched_phase_radius_time_series.csv"
        ).open(newline="", encoding="utf-8") as handle:
            cls.matched = list(csv.DictReader(handle))

    def test_zero_step_scope_and_row_counts(self) -> None:
        self.assertEqual(self.report["scope"]["simulation_steps_performed"], 0)
        self.assertEqual(len(self.source_a), 8)
        self.assertEqual(len(self.matched), 54)
        self.assertFalse(self.report["scope"]["arms_are_independent_replicates"])

    def test_corrected_lifetime_join(self) -> None:
        shifts = {
            row["case_id"]: float(row["corrected_lifetime_shift"])
            for row in self.source_a
        }
        self.assertEqual(
            shifts,
            {
                "c14p5": 850.0,
                "c18p5": 750.0,
                "c22p5": 680.0,
                "c26p5": 290.0,
                "c30p5": -10.0,
                "c34p5": -110.0,
                "c38p5": -100.0,
                "c64p5": 0.0,
            },
        )

    def test_full_and_same_mode_associations_pass(self) -> None:
        sweep = self.report["source_a_placement_sweep"]
        self.assertAlmostEqual(
            sweep["full_sweep"]["pearson_r"], 0.980265039389556
        )
        self.assertAlmostEqual(
            sweep["same_failure_mode_sensitivity"]["pearson_r"],
            0.9493485905736291,
        )
        self.assertGreaterEqual(sweep["full_sweep"]["pearson_r"], 0.90)
        self.assertGreaterEqual(
            sweep["same_failure_mode_sensitivity"]["pearson_r"], 0.90
        )
        self.assertFalse(
            sweep["full_sweep"]["inferential_p_value_reported"]
        )

    def test_far_control_is_null_and_welds_are_connected(self) -> None:
        sweep = self.report["source_a_placement_sweep"]
        far = sweep["far_control"]
        self.assertLess(abs(float(far["phase_radius_response_t1300"])), 0.01)
        self.assertEqual(float(far["corrected_lifetime_shift"]), 0.0)
        self.assertTrue(sweep["all_welds_connected_t1300"])

    def test_matched_confirmation_and_contour_robustness_pass(self) -> None:
        confirmation = self.report["matched_phase_radius_confirmation"]
        self.assertEqual(confirmation["required_count"], 8)
        self.assertEqual(confirmation["passed_count"], 8)
        self.assertTrue(
            all(row["passed"] for row in confirmation["primary_checks"])
        )
        contour = self.report["existing_three_contour_confirmation"]
        self.assertTrue(contour["passed"])
        self.assertEqual(contour["passed_count"], 24)
        self.assertEqual(contour["required_count"], 24)

    def test_imported_historical_hashes_are_retained(self) -> None:
        provenance = self.report["provenance"]
        self.assertEqual(
            provenance["imported_scan_summary_sha256"],
            analyze.EXPECTED_SCAN_SUMMARY_SHA256,
        )
        self.assertEqual(
            provenance["imported_scan_metrics_sha256"],
            analyze.EXPECTED_SCAN_METRICS_SHA256,
        )
        manifest = json.loads(
            analyze.MATCHED_RAW_MANIFEST.read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["scope"]["simulation_steps_performed"], 0)
        self.assertEqual(manifest["scope"]["row_count"], 54)


if __name__ == "__main__":
    unittest.main()
