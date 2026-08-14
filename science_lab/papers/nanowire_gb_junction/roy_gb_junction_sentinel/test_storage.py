"""Tests for atomic checkpoint persistence and integrity rejection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from .storage import load_checkpoint, write_checkpoint


class SentinelStorageTests(unittest.TestCase):
    def test_checkpoint_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            field = np.arange(120, dtype=np.float64).reshape(4, 5, 6)
            entry = write_checkpoint(
                output,
                field,
                step=20,
                kinds=("regular", "signal", "regular"),
            )
            self.assertEqual(entry["kinds"], ["regular", "signal"])
            np.testing.assert_array_equal(load_checkpoint(entry), field)
            with self.assertRaises(FileExistsError):
                write_checkpoint(
                    output,
                    field,
                    step=20,
                    kinds=("regular",),
                )

    def test_checkpoint_corruption_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            field = np.ones((4, 4, 4), dtype=np.float64)
            entry = write_checkpoint(
                output,
                field,
                step=0,
                kinds=("initial",),
            )
            Path(entry["field_path"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                load_checkpoint(entry)


if __name__ == "__main__":
    unittest.main()

