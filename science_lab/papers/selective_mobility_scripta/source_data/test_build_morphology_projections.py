"""Command-line tests for the optional raw-field projection builder."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

from . import build_morphology_projections


class ProjectionBuilderCliTests(unittest.TestCase):
    def test_help_does_not_require_an_external_archive(self) -> None:
        standard_output = io.StringIO()
        standard_error = io.StringIO()
        with self.assertRaises(SystemExit) as exit_context:
            with redirect_stdout(standard_output), redirect_stderr(standard_error):
                build_morphology_projections.main(["--help"])

        self.assertEqual(exit_context.exception.code, 0)
        self.assertIn("--archive-root", standard_output.getvalue())
        self.assertEqual(standard_error.getvalue(), "")

    def test_archive_root_is_explicitly_required(self) -> None:
        standard_error = io.StringIO()
        with self.assertRaises(SystemExit) as exit_context:
            with redirect_stderr(standard_error):
                build_morphology_projections.main([])

        self.assertEqual(exit_context.exception.code, 2)
        self.assertIn("--archive-root", standard_error.getvalue())


if __name__ == "__main__":
    unittest.main()
