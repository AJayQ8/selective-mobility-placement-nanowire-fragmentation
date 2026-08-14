"""Tests for the versioned single-arm repeat-panel reanalysis."""

from __future__ import annotations

import unittest

from . import analyze_paired_repeat_panel as analysis


class PairedRepeatReanalysisTests(unittest.TestCase):
    @staticmethod
    def _legacy_record(
        step: int, arms: tuple[str, ...]
    ) -> dict[str, object]:
        return {
            "step": step,
            "distal_four_arm_topology": {
                "thresholds": {
                    level: {"detached_arms": list(arms)}
                    for level in analysis.LEGACY_THRESHOLDS
                }
            },
        }

    def test_legacy_clock_is_derived_not_pinned(self) -> None:
        arm = "first_wire_z_plus"
        report = analysis.legacy_single_arm_timing(
            [
                self._legacy_record(1680, ()),
                self._legacy_record(1690, ()),
                self._legacy_record(1700, (arm,)),
                self._legacy_record(1710, (arm,)),
                self._legacy_record(1720, (arm,)),
            ]
        )
        self.assertTrue(report["detected"])
        self.assertEqual(report["event_bracket"], [1690, 1700])
        self.assertEqual(report["confirmation_step"], 1720)

    def test_relocation_uses_absolute_sites_and_all_gaps(self) -> None:
        def observed(
            time: float, sites: tuple[float, ...]
        ) -> dict[str, object]:
            return {
                "detected": True,
                "event_midpoint": time,
                "sites": [
                    {"first_site_abs": abs(site)} for site in sites
                ],
            }

        observations = {
            "untreated": observed(1685.0, (-18.5, 19.25)),
            "c18p5": observed(2445.0, (-39.5, 39.5)),
            "c26p5": observed(1955.0, (17.5,)),
            "c34p5": observed(1575.0, (18.75,)),
        }
        report = analysis.paired_pattern(observations)
        self.assertTrue(report["c18_relocated_downstream"])
        observations["c18p5"] = observed(
            2445.0, (-39.5, 19.0)
        )
        report = analysis.paired_pattern(observations)
        self.assertFalse(report["c18_relocated_downstream"])

    def test_paired_effects_are_derived_from_each_baseline(self) -> None:
        observations = {
            "untreated": {"event_midpoint": 1695.0},
            "c18p5": {"event_midpoint": 2445.0},
            "c26p5": {"event_midpoint": 1985.0},
            "c34p5": {"event_midpoint": 1585.0},
        }
        effects = analysis._paired_effects(observations)
        self.assertEqual(
            effects["c18p5"]["delta_time_vs_paired_untreated"],
            750.0,
        )
        self.assertEqual(
            effects["c26p5"]["delta_time_vs_paired_untreated"],
            290.0,
        )
        self.assertEqual(
            effects["c34p5"]["delta_time_vs_paired_untreated"],
            -110.0,
        )


if __name__ == "__main__":
    unittest.main()
