"""Contract tests for the bounded Roy reproduction preflight runner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from .model import (
    FROZEN_DEG90,
    PeriodicLattice,
    RoyDeg90Definition,
    initialize_strict_deg90,
)
from .run_preflight import (
    EXACT_TIMING_STEPS,
    EXACT_TIMING_WARMUP_STEPS,
    REDUCED_SHAPE,
    REDUCED_STEPS,
    atomic_npy,
    contact_metrics,
    definition_for_mode,
    resource_preflight,
    reserve_output_directory,
)


class PreflightDefinitionTests(unittest.TestCase):
    def test_reduced_case_is_bounded_and_not_the_published_grid(self) -> None:
        definition, steps = definition_for_mode("reduced")
        self.assertEqual(definition.lattice.shape, REDUCED_SHAPE)
        self.assertEqual(steps, REDUCED_STEPS)
        self.assertNotEqual(
            definition.lattice.shape, FROZEN_DEG90.lattice.shape
        )
        self.assertEqual(
            definition.lattice.shape[2],
            FROZEN_DEG90.lattice.shape[2],
        )

    def test_exact_timing_case_is_only_five_published_grid_steps(self) -> None:
        definition, steps = definition_for_mode("exact-timing")
        self.assertEqual(definition, FROZEN_DEG90)
        self.assertEqual(steps, EXACT_TIMING_STEPS)
        self.assertEqual(steps, 5)
        self.assertEqual(EXACT_TIMING_WARMUP_STEPS, 2)


class ContactMetricTests(unittest.TestCase):
    def test_one_empty_node_is_disconnected_then_a_fill_connects_cores(self) -> None:
        definition = RoyDeg90Definition(
            lattice=PeriodicLattice((96, 32, 32), 0.5),
            noise_amplitude=0.0,
        )
        field, _ = initialize_strict_deg90(definition)
        initial = contact_metrics(field, definition)
        self.assertFalse(
            initial["thresholds"]["0.50"]["cores_connected_locally"]
        )
        field[60, 16, 16] = 1.0
        joined = contact_metrics(field, definition)
        self.assertTrue(
            joined["thresholds"]["0.50"]["cores_connected_locally"]
        )


class OutputSafetyTests(unittest.TestCase):
    def test_resource_preflight_reports_a_small_case_as_safe(self) -> None:
        definition = RoyDeg90Definition(
            lattice=PeriodicLattice((96, 32, 32), 0.5),
            noise_amplitude=0.0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            report = resource_preflight(
                definition,
                Path(temporary) / "result",
            )
        self.assertTrue(report["safe_to_start_bounded_preflight"])
        self.assertTrue(all(report["checks"].values()))

    def test_output_directory_is_never_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "new"
            reserve_output_directory(path)
            with self.assertRaises(FileExistsError):
                reserve_output_directory(path)

    def test_atomic_npy_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "field.npy"
            field = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
            elapsed = atomic_npy(path, field)
            self.assertGreaterEqual(elapsed, 0.0)
            np.testing.assert_array_equal(
                np.load(path, allow_pickle=False),
                field,
            )


if __name__ == "__main__":
    unittest.main()
