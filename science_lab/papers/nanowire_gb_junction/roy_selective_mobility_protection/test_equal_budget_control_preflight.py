"""Static scope tests for the bounded equal-budget preflight."""

from __future__ import annotations

import unittest

from . import run_equal_budget_control_preflight as runner


class EqualBudgetControlPreflightTests(unittest.TestCase):
    def test_scope_is_exactly_two_short_cases(self) -> None:
        self.assertEqual(runner.START_STEP, 100)
        self.assertEqual(runner.ACCEPTED_STEPS_PER_CASE, 16)
        self.assertEqual(
            runner.CASES,
            ("K2_junction_cap", "K3_uniform"),
        )

    def test_no_long_work_is_in_preflight_scope(self) -> None:
        self.assertEqual(runner.TARGET_STEP, 2000)
        self.assertGreater(runner.RUNTIME_SAFETY_FACTOR, 1.0)
        self.assertLessEqual(runner.MASS_DRIFT_LIMIT, 1.0e-4)

    def test_default_output_is_separate_from_prior_results(self) -> None:
        self.assertEqual(
            runner.DEFAULT_OUTPUT.name,
            "equal_budget_control_preflight_v1",
        )


if __name__ == "__main__":
    unittest.main()
