from __future__ import annotations

import unittest

from .analyze import analyze_summaries


def _summary(
    bracket: list[int] | None,
    *,
    site: float = 18.0,
    wire: str = "first_wire_z",
    status: str = "completed_event",
    gaps: list[dict] | None = None,
) -> dict:
    event_gaps = (
        gaps
        if gaps is not None
        else []
        if bracket is None
        else [{"wire": wire, "first_midpoint": site}]
    )
    return {
        "status": status,
        "health_passed": True,
        "event_assessment": {
            "detected": bracket is not None,
            "event_bracket": bracket,
            "persistent_gaps": event_gaps,
            "confirmation_step": None if bracket is None else bracket[1] + 20,
        },
        "records": [
            {"relative_mass_drift": 1.0e-6, "relative_energy_rebound": 0.0}
        ],
    }


class FrozenAnalysisTests(unittest.TestCase):
    def test_frozen_reference_like_pair_passes(self) -> None:
        report = analyze_summaries(
            {
                "untreated": _summary([1600, 1610], site=18.75),
                "c34": _summary([1500, 1510], site=18.0),
            },
            guard_passed=True,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["paired_effect_c34_minus_untreated"], -100.0)

    def test_overlapping_brackets_fail_even_if_midpoints_order(self) -> None:
        report = analyze_summaries(
            {
                "untreated": _summary([1600, 1610]),
                "c34": _summary([1590, 1600]),
            },
            guard_passed=True,
        )
        self.assertFalse(report["checks"]["c34_complete_bracket_strictly_earlier"])
        self.assertFalse(report["passed"])

    def test_event_free_horizon_is_censored_not_passed(self) -> None:
        report = analyze_summaries(
            {
                "untreated": _summary(
                    None, status="completed_horizon"
                ),
                "c34": _summary([1500, 1510]),
            },
            guard_passed=True,
        )
        self.assertFalse(report["passed"])
        self.assertEqual(
            report["classification"],
            "spatial_refinement_validation_censored_or_event_absent",
        )

    def test_remote_or_wrong_wire_topology_fails(self) -> None:
        report = analyze_summaries(
            {
                "untreated": _summary([1600, 1610]),
                "c34": _summary([1500, 1510], site=40.0, wire="second_wire_y"),
            },
            guard_passed=True,
        )
        self.assertFalse(report["checks"]["all_event_gaps_natural_local_sites"])

    def test_two_simultaneous_natural_gaps_are_accepted(self) -> None:
        report = analyze_summaries(
            {
                "untreated": _summary([1600, 1610], site=18.75),
                "c34": _summary(
                    [1500, 1510],
                    gaps=[
                        {"wire": "first_wire_z", "first_midpoint": -18.0},
                        {"wire": "second_wire_y", "first_midpoint": 18.5},
                    ],
                ),
            },
            guard_passed=True,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["events"]["c34"]["persistent_gap_count"], 2)

    def test_more_than_two_tied_gaps_fail(self) -> None:
        report = analyze_summaries(
            {
                "untreated": _summary([1600, 1610]),
                "c34": _summary(
                    [1500, 1510],
                    gaps=[
                        {"wire": "first_wire_z", "first_midpoint": -18.0},
                        {"wire": "second_wire_y", "first_midpoint": 18.5},
                        {"wire": "first_wire_z", "first_midpoint": 19.0},
                    ],
                ),
            },
            guard_passed=True,
        )
        self.assertFalse(
            report["checks"][
                "both_event_gap_multiplicities_between_one_and_two"
            ]
        )


if __name__ == "__main__":
    unittest.main()
