"""Synthetic tests for conservative sentinel pinch diagnostics."""

from __future__ import annotations

import unittest

import numpy as np

from . import diagnostics


def _fake_gap(
    midpoint: float,
    *,
    wire: str = "first_wire_z",
    start_index: int = 1,
    length: float = 16.0,
) -> dict[str, object]:
    return {
        "wire": wire,
        "branch": "plus" if midpoint >= 0.0 else "minus",
        "axial_axis": 2,
        "axial_length": length,
        "start_index": start_index,
        "cell_count": 1,
        "width": 1.0,
        "midpoint": midpoint,
        "crosses_periodic_seam": False,
        "threshold_nested": True,
        "site_robust": True,
    }


def _observation(step: int, gaps: list[dict[str, object]]) -> dict[str, object]:
    return {
        "step": step,
        "pinches": {
            "candidate_gaps": gaps,
        },
    }


class SliceAreaTests(unittest.TestCase):
    def test_full_slice_areas_use_the_complete_transverse_plane(self) -> None:
        field = np.ones((3, 4, 8), dtype=np.float64)
        field[:, :, 6] = 0.44
        areas = diagnostics.full_slice_areas(
            field,
            axial_axis=2,
            spacing=0.5,
        )
        self.assertEqual(areas["0.45"][0], 3.0)
        self.assertEqual(areas["0.45"][6], 0.0)
        self.assertEqual(areas["0.55"][6], 0.0)

    def test_higher_threshold_gap_is_not_promoted_without_c045_gap(self) -> None:
        field = np.ones((4, 16, 16), dtype=np.float64)
        field[:, :, 13] = 0.46
        report = diagnostics.instantaneous_pinches(
            field,
            geometry="crossed",
            spacing=1.0,
            radius=1.0,
            width=1.0,
        )
        self.assertFalse(report["candidate"])
        self.assertFalse(report["heuristic_fallback_allowed"])
        first = report["wires"]["first_wire_z"]
        self.assertEqual(first["thresholds"]["0.45"]["zero_slice_count"], 0)
        self.assertEqual(first["thresholds"]["0.50"]["zero_slice_count"], 1)

    def test_crossed_central_gap_is_excluded_but_outer_gap_is_candidate(self) -> None:
        field = np.ones((4, 16, 16), dtype=np.float64)
        # Half-cell coordinates are ...,-0.5,+0.5,...; both are excluded.
        field[:, :, 8] = 0.0
        report = diagnostics.instantaneous_pinches(
            field,
            geometry="crossed",
            spacing=1.0,
            radius=1.0,
            width=1.0,
        )
        self.assertFalse(report["wires"]["first_wire_z"]["candidate"])

        field[:, :, 13] = 0.0  # z=+5.5, outside R+2W=3.
        report = diagnostics.instantaneous_pinches(
            field,
            geometry="crossed",
            spacing=1.0,
            radius=1.0,
            width=1.0,
        )
        candidates = report["wires"]["first_wire_z"]["candidate_gaps"]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["branch"], "plus")
        self.assertTrue(candidates[0]["threshold_nested"])
        self.assertTrue(candidates[0]["site_robust"])
        self.assertEqual(
            candidates[0]["threshold_gap_midpoint_spread"], 0.0
        )
        self.assertFalse(candidates[0]["heuristic_fallback_used"])

    def test_threshold_midpoint_spread_controls_candidate_status(self) -> None:
        failed_field = np.ones((4, 4, 16), dtype=np.float64)
        failed_field[:, :, 12] = 0.0
        failed_field[:, :, 13:16] = 0.47
        failed_field[:, :, 11] = 0.52
        failed = diagnostics.instantaneous_pinches(
            failed_field,
            geometry="isolated",
            spacing=1.0,
            radius=1.0,
            width=0.5,
        )
        failed_gap = failed["candidate_gaps"][0]
        self.assertTrue(failed_gap["threshold_nested"])
        self.assertGreater(
            failed_gap["threshold_gap_midpoint_spread"],
            failed_gap["site_tolerance"],
        )
        self.assertFalse(failed_gap["site_robust"])
        self.assertFalse(failed["candidate"])

        passed_field = np.ones((4, 4, 16), dtype=np.float64)
        passed_field[:, :, 12:14] = 0.0
        passed_field[:, :, (11, 14)] = 0.47
        passed_field[:, :, (10, 15)] = 0.52
        passed = diagnostics.instantaneous_pinches(
            passed_field,
            geometry="isolated",
            spacing=1.0,
            radius=1.0,
            width=0.5,
        )
        passed_gap = passed["candidate_gaps"][0]
        self.assertEqual(
            passed_gap["threshold_gap_midpoint_spread"], 0.0
        )
        self.assertTrue(passed_gap["site_robust"])
        self.assertTrue(passed["candidate"])


class PeriodicRunTests(unittest.TestCase):
    def test_isolated_gap_merges_across_periodic_array_edge(self) -> None:
        field = np.ones((4, 4, 8), dtype=np.float64)
        field[:, :, (7, 0)] = 0.0
        report = diagnostics.instantaneous_pinches(
            field,
            geometry="isolated",
            spacing=1.0,
            radius=1.0,
            width=1.0,
        )
        gaps = report["wires"]["isolated_wire_z"]["candidate_gaps"]
        self.assertEqual(len(gaps), 1)
        self.assertTrue(gaps[0]["crosses_periodic_seam"])
        self.assertEqual(gaps[0]["cell_count"], 2)
        self.assertEqual(gaps[0]["branch"], "periodic_antipode")
        topology = report["wires"]["isolated_wire_z"]["thresholds"]["0.50"][
            "fragment_topology"
        ]
        self.assertEqual(topology["cut_count"], 1)
        self.assertEqual(topology["fragment_count"], 1)
        self.assertEqual(topology["fragment_axial_lengths"], [6.0])

    def test_periodic_fragments_are_positive_runs_between_zero_gaps(self) -> None:
        areas = np.array([0.0, 0.0, 5.0, 5.0, 0.0, 5.0, 5.0, 5.0])
        topology = diagnostics.periodic_fragment_topology(
            areas, spacing=1.0
        )
        self.assertTrue(topology["valid"])
        self.assertEqual(topology["cut_count"], 2)
        self.assertEqual(topology["fragment_count"], 2)
        self.assertEqual(
            sorted(topology["fragment_axial_lengths"]), [2.0, 3.0]
        )

    def test_continuous_and_absent_periodic_material_are_distinguished(self) -> None:
        continuous = diagnostics.periodic_fragment_topology(
            np.ones(8), spacing=0.5
        )
        self.assertTrue(continuous["continuous_periodic_wire"])
        self.assertEqual(continuous["fragment_count"], 1)
        self.assertEqual(continuous["fragment_axial_lengths"], [4.0])

        absent = diagnostics.periodic_fragment_topology(
            np.zeros(8), spacing=0.5
        )
        self.assertFalse(absent["valid"])
        self.assertEqual(absent["fragment_count"], 0)
        absent_field = diagnostics.instantaneous_pinches(
            np.zeros((4, 4, 8)),
            geometry="isolated",
            spacing=0.5,
            radius=1.0,
            width=1.0,
        )
        self.assertFalse(absent_field["candidate"])


class PersistenceTests(unittest.TestCase):
    def test_same_gap_persists_and_remains_latched_after_disappearance(self) -> None:
        records = [
            _observation(0, []),
            _observation(10, [_fake_gap(2.0, start_index=10)]),
            _observation(20, [_fake_gap(2.5, start_index=11)]),
            _observation(30, [_fake_gap(3.0, start_index=12)]),
        ]
        event = diagnostics.persistent_single_arm_event(
            records,
            maximum_gap_displacement=0.75,
        )
        self.assertTrue(event["detected"])
        self.assertEqual(event["event_bracket"], [0, 10])
        self.assertEqual(event["confirmation_step"], 30)

        records.append(_observation(40, []))
        later = diagnostics.persistent_single_arm_event(
            records,
            maximum_gap_displacement=0.75,
        )
        self.assertEqual(later, event)
        explicit_latch = diagnostics.persistent_single_arm_event(
            [],
            latched_event=event,
        )
        self.assertEqual(explicit_latch, event)

    def test_gap_jump_larger_than_width_does_not_persist(self) -> None:
        records = [
            _observation(0, [_fake_gap(0.0)]),
            _observation(10, [_fake_gap(1.1)]),
            _observation(20, [_fake_gap(2.2)]),
        ]
        event = diagnostics.persistent_single_arm_event(
            records,
            maximum_gap_displacement=1.0,
        )
        self.assertFalse(event["detected"])

    def test_nonconsecutive_records_do_not_confirm(self) -> None:
        records = [
            _observation(0, [_fake_gap(1.0)]),
            _observation(10, [_fake_gap(1.0)]),
            _observation(30, [_fake_gap(1.0)]),
        ]
        event = diagnostics.persistent_single_arm_event(records)
        self.assertFalse(event["detected"])

    def test_non_site_robust_gap_cannot_persist_as_an_event(self) -> None:
        gaps = []
        for midpoint in (1.0, 1.1, 1.2):
            gap = _fake_gap(midpoint)
            gap["site_robust"] = False
            gaps.append(gap)
        records = [
            _observation(step, [gap])
            for step, gap in zip((0, 10, 20), gaps, strict=True)
        ]
        event = diagnostics.persistent_single_arm_event(records)
        self.assertFalse(event["detected"])

    def test_simultaneous_persistent_gaps_are_all_preserved(self) -> None:
        records = [_observation(0, [])]
        for step in (10, 20, 30):
            records.append(
                _observation(
                    step,
                    [
                        _fake_gap(
                            4.0 + 0.1 * (step // 10),
                            start_index=10,
                        ),
                        _fake_gap(
                            -4.0 - 0.1 * (step // 10),
                            start_index=2,
                        ),
                    ],
                )
            )
        event = diagnostics.persistent_single_arm_event(
            records,
            maximum_gap_displacement=0.5,
        )
        self.assertTrue(event["detected"])
        self.assertEqual(event["simultaneous_persistent_gap_count"], 2)
        self.assertEqual(len(event["persistent_gaps"]), 2)


if __name__ == "__main__":
    unittest.main()
