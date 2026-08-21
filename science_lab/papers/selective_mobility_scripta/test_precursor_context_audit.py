"""Fail-closed tests for the zero-step precursor-context audit."""

from __future__ import annotations

import csv
import json
import unittest

from . import precursor_context_audit as audit


class PrecursorContextAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = audit.main([])
        with (audit.DEFAULT_OUTPUT / "probe_context.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            cls.probe_rows = list(csv.DictReader(handle))
        with (audit.DEFAULT_OUTPUT / "early_sign_lead.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            cls.lead_rows = list(csv.DictReader(handle))

    def test_zero_step_and_expected_subsets(self) -> None:
        self.assertEqual(self.report["scope"]["simulation_steps_performed"], 0)
        context = self.report["probe_context"]
        self.assertEqual(
            context["overlap_cases"], ["c14p5", "c18p5", "c22p5"]
        )
        self.assertEqual(
            context["primary_outside_support_common_mode_cases"],
            ["c26p5", "c30p5", "c34p5", "c38p5", "c64p5"],
        )

    def test_probe_regions_and_multipliers(self) -> None:
        by_case = {row["case_id"]: row for row in self.probe_rows}
        self.assertEqual(by_case["c18p5"]["probe_region"], "plateau")
        self.assertAlmostEqual(
            float(by_case["c18p5"]["radial_plateau_multiplier_at_probe"]),
            0.1,
        )
        self.assertEqual(by_case["c14p5"]["probe_region"], "transition")
        self.assertEqual(by_case["c22p5"]["probe_region"], "transition")
        for case in ("c26p5", "c30p5", "c34p5", "c38p5", "c64p5"):
            self.assertEqual(by_case[case]["probe_region"], "outside_support")
            self.assertEqual(
                by_case[case]["primary_outside_support_common_mode_subset"],
                "True",
            )

    def test_primary_and_confounded_associations(self) -> None:
        context = self.report["probe_context"]
        self.assertAlmostEqual(
            context["primary_outside_support_common_mode_association"][
                "pearson_r"
            ],
            0.9493485905736291,
        )
        self.assertAlmostEqual(
            context["full_eight_point_association"]["pearson_r"],
            0.9802650393895559,
        )
        self.assertIn(
            "Confounded sensitivity",
            context["full_eight_point_association"]["interpretation"],
        )

    def test_early_signs_precede_events_by_at_least_850(self) -> None:
        self.assertEqual(len(self.lead_rows), 6)
        self.assertTrue(
            all(
                int(row["first_stored_persistent_signed_time"]) == 700
                for row in self.lead_rows
            )
        )
        self.assertEqual(
            min(int(row["lead_to_event_bracket_lower"]) for row in self.lead_rows),
            850,
        )
        self.assertEqual(
            self.report["early_time_separation"][
                "minimum_lead_to_event_bracket_lower"
            ],
            850,
        )

    def test_manifest_binds_all_outputs(self) -> None:
        manifest = json.loads(
            (audit.DEFAULT_OUTPUT / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            {record["path"] for record in manifest["outputs"]},
            {"summary.json", "probe_context.csv", "early_sign_lead.csv"},
        )


if __name__ == "__main__":
    unittest.main()
