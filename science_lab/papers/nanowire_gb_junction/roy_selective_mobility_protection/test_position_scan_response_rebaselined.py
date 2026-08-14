"""Tests for the one-arm position-response rebaseline."""

from __future__ import annotations

import unittest

from . import analyze_position_scan_response_rebaselined as analysis


class PositionResponseRebaselineTests(unittest.TestCase):
    def test_expected_curve_has_single_far_null(self) -> None:
        ordered = [
            analysis.EXPECTED_DELTAS[case_id]
            for case_id in (
                "c14p5",
                "c18p5",
                "c22p5",
                "c26p5",
                "c30p5",
                "c34p5",
                "c38p5",
                "c64p5",
            )
        ]
        self.assertEqual(
            ordered,
            [850.0, 750.0, 680.0, 290.0, -10.0, -110.0, -100.0, 0.0],
        )


if __name__ == "__main__":
    unittest.main()
