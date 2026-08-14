from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from . import supervisor


class SupervisorTests(unittest.TestCase):
    def test_sequence_is_fixed_and_nonadaptive(self) -> None:
        self.assertEqual(
            supervisor.STAGES,
            ("preflight", "source", "guard", "untreated", "c34", "analysis"),
        )

    def test_commands_are_sequential_module_invocations(self) -> None:
        output = Path(tempfile.gettempdir()) / "frozen-output"
        preflight = supervisor.stage_command("preflight", output)
        untreated = supervisor.stage_command("untreated", output)
        analysis = supervisor.stage_command("analysis", output)
        self.assertEqual(preflight[:3], [sys.executable, "-m", supervisor.PREFLIGHT_MODULE])
        self.assertIn("--resume", untreated)
        self.assertIn("--source-root", untreated)
        self.assertIn("untreated", untreated)
        self.assertEqual(analysis[:3], [sys.executable, "-m", supervisor.ANALYZER_MODULE])

    def test_unknown_stage_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown stage"):
            supervisor.stage_command(
                "adaptive_rescue", Path(tempfile.gettempdir()) / "output"
            )

    def test_hard_campaign_cap_is_nine_and_a_half_hours(self) -> None:
        self.assertEqual(
            supervisor.protocol.GUARDED_RUNTIME_SECONDS, int(9.5 * 3600)
        )


if __name__ == "__main__":
    unittest.main()
