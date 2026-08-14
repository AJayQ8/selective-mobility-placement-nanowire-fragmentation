from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from . import run_analysis as analysis


class TransportConfirmationTests(unittest.TestCase):
    def test_contract_is_frozen_and_scoped(self) -> None:
        contract = json.loads(analysis.CONTRACT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(contract["status"], "frozen_before_bc_transport_calculation")
        self.assertEqual(contract["scope"]["new_solver_steps"], 0)
        self.assertEqual(contract["scope"]["cases"], list(analysis.CASES))
        self.assertEqual(
            contract["analysis_definition"]["natural_zone_bounds_model_units_inclusive"],
            [14.0, 22.5],
        )
        self.assertIn("<= 0.01", contract["frozen_acceptance_criteria"]["continuity"])

    def test_artifact_mapping_is_exact(self) -> None:
        root = Path("raw")
        paths = analysis.artifact_paths(root, 104729, "c26p5")
        self.assertEqual(len(paths), 7)
        self.assertEqual(
            paths["checkpoint_field_sha256"],
            root
            / "seed-104729"
            / "case-c26p5"
            / "checkpoint-position_scan-step-1300.npy",
        )

    def test_frozen_check_logic_passes_only_both_directions(self) -> None:
        aggregates = {}
        for seed in analysis.SEEDS:
            aggregates[str(seed)] = {
                "untreated": self._aggregate(-0.20),
                "c26p5": self._aggregate(-0.14),
                "c34p5": self._aggregate(-0.26),
            }
        result = analysis.evaluate_frozen_checks(aggregates)
        self.assertTrue(result["all_frozen_science_checks_passed"])
        aggregates["104729"]["c34p5"] = self._aggregate(-0.19)
        result = analysis.evaluate_frozen_checks(aggregates)
        self.assertFalse(result["all_frozen_science_checks_passed"])

    def test_atomic_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            analysis.atomic_json(path, {"status": "ok"})
            self.assertEqual(json.loads(path.read_text()), {"status": "ok"})

    @staticmethod
    def _aggregate(rate: float) -> dict[str, object]:
        return {
            "mean_predicted_natural_zone_volume_rate": rate,
            "mean_observed_centered_natural_zone_volume_rate": rate,
            "all_four_continuity_signs_agree": True,
            "all_four_close_below_one_percent": True,
        }


if __name__ == "__main__":
    unittest.main()

