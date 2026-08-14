from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from . import protocol


class FrozenProtocolTests(unittest.TestCase):
    def test_physical_domain_is_unchanged_by_refinement(self) -> None:
        self.assertEqual(
            tuple(protocol.fine_definition().lattice.physical_lengths),
            protocol.PHYSICAL_LENGTHS,
        )
        self.assertEqual(protocol.FINE_RADIUS_CELLS * protocol.FINE_SPACING, 6.0)

    def test_only_untreated_and_c34_are_authorized(self) -> None:
        self.assertEqual([case.case_id for case in protocol.CASES], ["untreated", "c34"])
        geometry = protocol.C34.geometry
        assert geometry is not None
        self.assertEqual(
            (geometry.inner_support_distance, geometry.outer_support_distance),
            (28.5, 40.5),
        )
        self.assertEqual(geometry.transition_width, 3.0)
        self.assertEqual(geometry.protected_mobility_factor, 0.1)

    def test_time_detector_and_acceptance_are_frozen(self) -> None:
        self.assertEqual(protocol.TIMESTEP, 1.0)
        self.assertEqual(protocol.HORIZON, 2000)
        self.assertEqual(protocol.DIAGNOSTIC_INTERVAL, 10)
        self.assertEqual(protocol.PERSISTENCE_RECORDS, 3)
        self.assertEqual(protocol.PAIRED_EFFECT_BOUNDS, (-125.0, -75.0))
        self.assertEqual(protocol.NATURAL_SITE_ABS_BOUNDS, (14.0, 22.5))
        self.assertEqual(
            protocol.NATURAL_WIRES, ("first_wire_z", "second_wire_y")
        )
        self.assertEqual(protocol.PERSISTENT_GAP_COUNT_BOUNDS, (1, 2))
        self.assertEqual(protocol.COARSE_REFERENCE_HORIZON, 2900)
        self.assertEqual(
            protocol.COARSE_EVENT_GAPS["untreated"],
            [{"wire": "first_wire_z", "site": -18.75}],
        )
        self.assertEqual(
            protocol.COARSE_EVENT_GAPS["c34"],
            [
                {"wire": "first_wire_z", "site": -18.0},
                {"wire": "second_wire_y", "site": 18.5},
            ],
        )

    def test_payload_forbids_adaptive_follow_on(self) -> None:
        scope = protocol.protocol_payload()["scope"]
        self.assertFalse(scope["adaptive_follow_on_authorized"])
        self.assertFalse(scope["horizon_extension_authorized"])
        self.assertFalse(scope["production_dt_change_authorized"])
        self.assertFalse(scope["additional_case_authorized"])

    def test_storage_upper_bound_includes_bounded_interruptions(self) -> None:
        self.assertEqual(protocol.MAXIMUM_PERSISTENT_FIELDS, 9)
        self.assertEqual(
            protocol.WORST_CASE_PERSISTENT_BYTES,
            9 * protocol.FIELD_ARTIFACT_UPPER_BOUND_BYTES + (512 << 20),
        )
        self.assertEqual(
            protocol.WORST_CASE_FIELD_PERSISTENT_BYTES,
            9 * protocol.FIELD_ARTIFACT_UPPER_BOUND_BYTES,
        )
        self.assertEqual(protocol.PLANNED_SMALL_OUTPUT_ALLOWANCE_BYTES, 512 << 20)
        self.assertEqual(protocol.PEAK_RAM_ESTIMATE_BYTES, 32 << 30)
        self.assertGreater(
            protocol.PEAK_RAM_ESTIMATE_BYTES,
            protocol.SOLVER_PEAK_ESTIMATE_BYTES
            + 3 * protocol.FIELD_RAW_BYTES,
        )

    def test_source_root_is_relocatable_but_filenames_are_frozen(self) -> None:
        source_root = Path(tempfile.gettempdir()) / "relocated-source"
        paths = protocol.source_paths(source_root)
        self.assertEqual(paths.field.name, protocol.SOURCE_FIELD_NAME)
        self.assertEqual(paths.metadata.name, protocol.SOURCE_METADATA_NAME)
        self.assertEqual(paths.summary.name, protocol.SOURCE_SUMMARY_NAME)
        references = protocol.coarse_reference_paths(source_root)
        self.assertEqual(
            references.root.name, protocol.COARSE_REFERENCE_CAMPAIGN_NAME
        )
        self.assertEqual(
            references.summaries["c34"].name,
            "summary.json",
        )


if __name__ == "__main__":
    unittest.main()
