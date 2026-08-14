"""Contract tests for the fixed Roy t=1000 production runner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from .model import (
    PeriodicLattice,
    RoyDeg90Definition,
    RoyModelParameters,
    RoyPseudospectralSolver,
    initialize_strict_deg90,
)
from .run_t1000 import (
    CHECKPOINT_INTERVAL,
    DIAGNOSTIC_INTERVAL,
    TARGET_STEP,
    checkpoint_steps,
    descriptive_t1000_assessment,
    diagnostic_steps,
    four_arm_metrics,
    load_checkpoint,
    parse_arguments,
    write_checkpoint,
)


class FrozenProductionContractTests(unittest.TestCase):
    def test_target_and_cadences_are_fixed(self) -> None:
        self.assertEqual(TARGET_STEP, 1000)
        self.assertEqual(DIAGNOSTIC_INTERVAL, 25)
        self.assertEqual(CHECKPOINT_INTERVAL, 100)
        self.assertEqual(diagnostic_steps()[-1], 1000)
        self.assertEqual(checkpoint_steps(), tuple(range(100, 1001, 100)))

    def test_cli_exposes_no_target_or_t2000_option(self) -> None:
        arguments = parse_arguments(["--launch"])
        self.assertTrue(arguments.launch)
        with self.assertRaises(SystemExit):
            parse_arguments(["--launch", "--target-step", "2000"])


class FourArmMetricTests(unittest.TestCase):
    def test_gap_fill_connects_both_cores_and_all_four_arms(self) -> None:
        definition = RoyDeg90Definition(
            lattice=PeriodicLattice((96, 128, 128), 0.5),
            noise_amplitude=0.0,
        )
        field, _ = initialize_strict_deg90(definition)
        before = four_arm_metrics(field, definition)
        self.assertFalse(
            before["thresholds"]["0.50"]["all_four_arms_attached"]
        )
        field[60, 64, 64] = 1.0
        after = four_arm_metrics(field, definition)
        self.assertTrue(
            after["thresholds"]["0.50"]["all_four_arms_attached"]
        )
        self.assertEqual(
            after["thresholds"]["0.50"]["attached_arm_count"],
            4,
        )


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_is_bitwise_exact(self) -> None:
        field = np.arange(8 * 8 * 8, dtype=np.float64).reshape(8, 8, 8)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            metadata = write_checkpoint(output, field, step=100)
            loaded = load_checkpoint(metadata)
        np.testing.assert_array_equal(loaded, field)

    def test_checkpoint_resume_preserves_next_solver_step(self) -> None:
        lattice = PeriodicLattice((8, 8, 8), 0.5)
        solver = RoyPseudospectralSolver(
            lattice,
            RoyModelParameters(),
            fft_workers=1,
        )
        rng = np.random.default_rng(44)
        initial = 0.1 + 0.8 * rng.random(lattice.shape)
        first = solver.propose_step(initial)
        expected = solver.propose_step(first)
        with tempfile.TemporaryDirectory() as temporary:
            metadata = write_checkpoint(
                Path(temporary),
                first,
                step=100,
            )
            resumed = load_checkpoint(metadata)
        measured = solver.propose_step(resumed)
        np.testing.assert_array_equal(measured, expected)


class DescriptiveAssessmentTests(unittest.TestCase):
    @staticmethod
    def record(step: int, connected: bool, attached: bool) -> dict:
        threshold_contact = {
            "cores_connected_locally": connected,
            "contact_plane_cells": 100,
        }
        threshold_arms = {"all_four_arms_attached": attached}
        return {
            "step": step,
            "contact": {
                "thresholds": {
                    key: dict(threshold_contact)
                    for key in ("0.45", "0.50", "0.55")
                }
            },
            "four_arm_topology": {
                "central_roi_composition_sum": float(step),
                "thresholds": {
                    key: dict(threshold_arms)
                    for key in ("0.45", "0.50", "0.55")
                },
            },
        }

    def test_assessment_reports_persistence_without_authorizing_continuation(
        self,
    ) -> None:
        records = [self.record(0, False, False)]
        records.extend(
            self.record(step, True, True) for step in (800, 900, 1000)
        )
        assessment = descriptive_t1000_assessment(records)
        self.assertTrue(
            assessment["thresholds"]["0.50"][
                "junction_persistent_at_800_900_1000"
            ]
        )
        self.assertTrue(
            assessment["thresholds"]["0.50"][
                "all_four_arms_attached_at_800_900_1000"
            ]
        )
        self.assertFalse(
            assessment["automatic_t2000_continuation_authorized"]
        )


if __name__ == "__main__":
    unittest.main()
