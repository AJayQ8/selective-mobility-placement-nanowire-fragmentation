"""Fail-closed tests for the deterministic CMS conversion analysis."""

from __future__ import annotations

import csv
import json
import unittest

from . import build_cms_analysis


class CmsAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        build_cms_analysis.main()
        with build_cms_analysis.NORMALIZED_PATH.open(
            newline="", encoding="utf-8"
        ) as handle:
            cls.normalized = list(csv.DictReader(handle))
        with build_cms_analysis.MATRIX_PATH.open(
            newline="", encoding="utf-8"
        ) as handle:
            cls.matrix = list(csv.DictReader(handle))
        cls.receipt = json.loads(
            build_cms_analysis.RECEIPT_PATH.read_text(encoding="utf-8")
        )

    def test_exact_row_counts_and_zero_simulation(self) -> None:
        self.assertEqual(len(self.normalized), 10)
        self.assertEqual(len(self.matrix), 9)
        self.assertEqual(self.receipt["simulation_steps_performed"], 0)
        self.assertTrue(self.receipt["all_required_gates_satisfied"])
        self.assertEqual(self.receipt["pass_row_count"], 8)
        self.assertEqual(self.receipt["supporting_row_count"], 1)

    def test_strong_contrast_normalized_ranges(self) -> None:
        strong = [
            row for row in self.normalized if float(row["m_min"]) == 0.1
        ]
        intermediate = [
            float(row["normalized_percent"])
            for row in strong
            if row["placement_role"] == "intermediate"
        ]
        outboard = [
            float(row["normalized_percent"])
            for row in strong
            if row["placement_role"] == "outboard"
        ]
        self.assertAlmostEqual(min(intermediate), 15.7099697885)
        self.assertAlmostEqual(max(intermediate), 17.1091445428)
        self.assertAlmostEqual(min(outboard), -6.528189911)
        self.assertAlmostEqual(max(outboard), -6.04229607251)

    def test_weaker_contrast_signs_and_values(self) -> None:
        weaker = [
            row for row in self.normalized if float(row["m_min"]) == 0.3
        ]
        self.assertEqual(
            [float(row["paired_shift"]) for row in weaker],
            [200.0, -60.0, 200.0, -60.0],
        )

    def test_roy_and_clock_rows_pass(self) -> None:
        by_id = {row["validation_id"]: row for row in self.matrix}
        self.assertEqual(by_id["CMS-V01"]["status"], "pass")
        self.assertIn("1710-1720", by_id["CMS-V01"]["evidence"])
        self.assertIn("published example 1719", by_id["CMS-V01"]["evidence"])
        self.assertIn("three-simulation summary 1706+/-12", by_id["CMS-V01"]["evidence"])
        self.assertNotIn("standard deviations", by_id["CMS-V01"]["evidence"])
        self.assertEqual(by_id["CMS-V02"]["status"], "pass")
        self.assertIn(
            "overlapping untreated 1690-1700",
            by_id["CMS-V02"]["evidence"],
        )

    def test_every_boundary_is_nonempty(self) -> None:
        statuses = {}
        for row in self.matrix:
            self.assertIn(row["status"], {"pass", "supporting"})
            self.assertTrue(row["transfer_scope"])
            self.assertTrue(row["boundary"])
            statuses[row["validation_id"]] = row["status"]
        self.assertEqual(statuses["CMS-V05"], "supporting")
        self.assertEqual(
            [key for key, value in statuses.items() if value == "supporting"],
            ["CMS-V05"],
        )

    def test_precursor_row_is_passed_and_bounded(self) -> None:
        by_id = {row["validation_id"]: row for row in self.matrix}
        row = by_id["CMS-M01"]
        self.assertEqual(row["status"], "pass")
        self.assertIn("r=0.949 (n=5)", row["evidence"])
        self.assertNotIn("r=0.980", row["evidence"])
        self.assertIn("t=700", row["evidence"])
        self.assertIn("at least 850", row["evidence"])
        self.assertIn("8/8", row["evidence"])
        self.assertIn("24/24", row["evidence"])
        self.assertIn("post-hoc", row["boundary"])
        self.assertIn("confounded sensitivity", row["boundary"])
        self.assertIn("not a calibrated predictive law", row["boundary"])

    def test_context_audit_is_receipt_bound(self) -> None:
        self.assertEqual(
            self.receipt["analysis_id"], "selective_mobility_cms_archive_v1"
        )
        self.assertIn(
            "../precursor_context_audit/readout_v1/summary.json",
            self.receipt["inputs"],
        )


if __name__ == "__main__":
    unittest.main()
