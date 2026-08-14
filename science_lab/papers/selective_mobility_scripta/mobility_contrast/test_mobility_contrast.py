"""Fast fail-closed checks for the frozen m=0.3 campaign wrapper."""

from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from . import analyze as campaign_analysis
from . import common


class FrozenCampaignTests(unittest.TestCase):
    def test_exact_case_scope(self) -> None:
        self.assertEqual(
            common.CASE_ORDER,
            (
                (104729, "c26p5"),
                (104729, "c34p5"),
                (130363, "c26p5"),
                (130363, "c34p5"),
            ),
        )

    def test_contract_is_frozen_before_simulation(self) -> None:
        contract = common.frozen_contract()
        self.assertEqual(contract["status"], "frozen_before_simulation")
        self.assertEqual(
            contract["frozen_model"]["protected_mobility_factor"], 0.3
        )
        self.assertTrue(contract["scope"]["no_source_A_case"])
        self.assertTrue(contract["scope"]["no_additional_contrast"])
        self.assertTrue(contract["scope"]["no_position_sweep"])
        self.assertTrue(contract["scope"]["no_rescue_run"])

    def test_acceptance_signs_are_strict_and_per_source(self) -> None:
        criteria = common.frozen_contract()["frozen_acceptance_criteria"]
        self.assertIn("for each source", criteria["intermediate_sign"])
        self.assertIn("strictly above zero", criteria["intermediate_sign"])
        self.assertIn("for each source", criteria["outboard_sign"])
        self.assertIn("strictly below zero", criteria["outboard_sign"])

    def test_disk_bound_includes_all_four_cases(self) -> None:
        field_total = (
            len(common.CASE_ORDER)
            * common.MAX_CHECKPOINTS_PER_CASE
            * common.FIELD_BYTES_UPPER_BOUND
        )
        self.assertGreater(common.PERSISTENT_OUTPUT_UPPER_BOUND, field_total)
        self.assertEqual(common.UNTOUCHED_RESERVE, 10 << 30)
        self.assertEqual(common.LARGEST_ATOMIC_WRITE, common.FIELD_BYTES_UPPER_BOUND)

    def test_output_paths_are_case_specific(self) -> None:
        paths = {
            common.raw_case_output(seed, case_id)
            for seed, case_id in common.CASE_ORDER
        }
        self.assertEqual(len(paths), 4)
        self.assertTrue(all("_m03" in path.name for path in paths))

    def test_contract_sha_is_stable_hex(self) -> None:
        digest = common.sha256_path(common.CONTRACT_PATH)
        self.assertEqual(len(digest), 64)
        int(digest, 16)

    def test_study_snapshot_is_hash_only(self) -> None:
        self.assertEqual(common.FROZEN_IMPLEMENTATION_SNAPSHOT["file_count"], 16)
        digest = common.FROZEN_IMPLEMENTATION_SNAPSHOT["sha256"]
        self.assertEqual(len(digest), 64)
        int(digest, 16)

    def test_raw_results_are_locally_ignored(self) -> None:
        repository_root = Path(__file__).resolve().parents[4]
        exclude = repository_root / ".gitignore"
        self.assertIn(
            "raw_results/",
            exclude.read_text(encoding="utf-8"),
        )

    def test_contract_json_is_canonical_json(self) -> None:
        parsed = json.loads(common.CONTRACT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(parsed["campaign_id"], common.CAMPAIGN_ID)

    def test_public_parent_contract_is_bound(self) -> None:
        self.assertTrue(common.PARENT_REPEAT_CONTRACT_PATH.is_file())
        self.assertEqual(
            common.sha256_path(common.PARENT_REPEAT_CONTRACT_PATH),
            common.PARENT_REPEAT_CONTRACT_PUBLIC_SHA256,
        )

    def test_public_parent_implementation_bindings_are_hex(self) -> None:
        self.assertEqual(
            set(common.PUBLIC_PARENT_IMPLEMENTATION_SHA256),
            {
                "paired_case_runner",
                "case_engine",
                "paired_protocol",
                "paired_source_runner",
            },
        )
        for digest in common.PUBLIC_PARENT_IMPLEMENTATION_SHA256.values():
            self.assertEqual(len(digest), 64)
            int(digest, 16)

    def test_health_summary_is_independent_of_preflight_state(self) -> None:
        health = campaign_analysis._health(
            {
                "status": "completed",
                "records": [
                    {
                        "field": {"finite": True},
                        "relative_mass_drift": 2.0e-6,
                        "free_energy": -1.0,
                    },
                    {
                        "field": {"finite": True},
                        "relative_mass_drift": 2.5e-6,
                        "free_energy": -1.1,
                    },
                ],
            }
        )

        self.assertTrue(health["passed"])
        self.assertEqual(health["sampled_energy_rebound_count"], 0)

    def test_analysis_binds_the_preflight_implementation_snapshot(self) -> None:
        preflight = {
            "status": "GO",
            "campaign_implementation": {"analyze_sha256": "test-digest"},
            "frozen_implementation_snapshot": deepcopy(
                common.FROZEN_IMPLEMENTATION_SNAPSHOT
            ),
        }
        baselines = {
            "104729": {"event_bracket": [1680, 1690]},
            "130363": {"event_bracket": [1650, 1660]},
        }
        observations = []
        for seed, case_id in common.CASE_ORDER:
            untreated = baselines[str(seed)]["event_bracket"]
            if case_id == "c26p5":
                bracket = [untreated[0] + 200, untreated[1] + 200]
                role = "intermediate"
            else:
                bracket = [untreated[0] - 60, untreated[1] - 60]
                role = "outboard"
            observations.append(
                {
                    "seed": seed,
                    "case_id": f"{case_id}_m03",
                    "role": role,
                    "event_bracket": bracket,
                    "event_midpoint": 0.5 * sum(bracket),
                    "completion_checks": {"completed": True},
                    "provenance_checks": {"paired": True},
                    "topology_checks": {"junction_adjacent": True},
                    "health": {"passed": True},
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "analysis.json"
            with (
                patch.object(campaign_analysis, "load_json", return_value=preflight),
                patch.object(
                    campaign_analysis,
                    "sha256_path",
                    return_value="test-digest",
                ),
                patch.object(
                    campaign_analysis,
                    "frozen_contract",
                    return_value={
                        "scope": {"existing_untreated_baselines": baselines}
                    },
                ),
                patch.object(
                    campaign_analysis,
                    "_case_observation",
                    side_effect=deepcopy(observations),
                ),
                patch.object(campaign_analysis, "atomic_json") as write_json,
            ):
                report = campaign_analysis.analyze(output)

        self.assertTrue(report["passed"])
        self.assertTrue(report["checks"]["frozen_implementation_snapshot_bound"])
        write_json.assert_called_once_with(output, report)


if __name__ == "__main__":
    unittest.main()
