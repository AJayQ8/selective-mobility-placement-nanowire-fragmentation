from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import freeze, protocol


class RuntimeFreezeTests(unittest.TestCase):
    def test_frozen_paths_are_unique_and_include_all_stage_entrypoints(self) -> None:
        self.assertEqual(
            len(freeze.FROZEN_RELATIVE_PATHS),
            len(set(freeze.FROZEN_RELATIVE_PATHS)),
        )
        for name in (
            "preflight.py",
            "runner.py",
            "analyze.py",
            "supervisor.py",
            "SCIENTIFIC_CONTRACT.md",
        ):
            self.assertTrue(
                any(path.endswith(name) for path in freeze.FROZEN_RELATIVE_PATHS)
            )

    def test_receipt_core_hash_survives_json_round_trip(self) -> None:
        core = {
            "protocol_id": protocol.PROTOCOL_ID,
            "decision": "GO",
            "checks": {"frozen": True},
            "source_root": str(Path(tempfile.gettempdir()) / "source"),
            "runtime_manifest": {"a.py": "123"},
            "git": {"head": "abc", "upstream": "origin/test"},
        }
        restored = json.loads(json.dumps(core))
        self.assertEqual(
            freeze.canonical_sha256(core), freeze.canonical_sha256(restored)
        )

    def test_live_binding_rechecks_manifest_git_and_source_root(self) -> None:
        source_root = (Path(tempfile.gettempdir()) / "exact-source").resolve()
        core = {
            "protocol_id": protocol.PROTOCOL_ID,
            "decision": "GO",
            "checks": {"frozen": True},
            "source_root": str(source_root),
            "runtime_manifest": {"runtime.py": "hash"},
            "runtime_environment": {"python": "test"},
            "git": {
                "head": "abc",
                "upstream": "origin/test",
                "upstream_head": "abc",
            },
        }
        report = {
            "decision": "GO",
            "checks": {"frozen": True},
            "protocol_id": protocol.PROTOCOL_ID,
            "receipt_core": core,
            "receipt_core_sha256": freeze.canonical_sha256(core),
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "preflight").mkdir()
            (output / "preflight" / "summary.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            with mock.patch.object(
                freeze,
                "runtime_manifest",
                return_value={"runtime.py": "hash"},
            ), mock.patch.object(
                freeze,
                "runtime_environment",
                return_value={"python": "test"},
            ), mock.patch.object(
                freeze,
                "git_freeze_state",
                return_value={
                    "passed": True,
                    "head": "abc",
                    "upstream": "origin/test",
                    "upstream_head": "abc",
                },
            ):
                verified = freeze.verify_preflight_binding(
                    output,
                    protocol_id=protocol.PROTOCOL_ID,
                    source_root=source_root,
                )
        self.assertEqual(verified["decision"], "GO")

    def test_manifest_drift_blocks_stage(self) -> None:
        source_root = (Path(tempfile.gettempdir()) / "exact-source").resolve()
        core = {
            "protocol_id": protocol.PROTOCOL_ID,
            "decision": "GO",
            "checks": {"frozen": True},
            "source_root": str(source_root),
            "runtime_manifest": {"runtime.py": "old"},
            "runtime_environment": {"python": "test"},
            "git": {
                "head": "abc",
                "upstream": "origin/test",
                "upstream_head": "abc",
            },
        }
        report = {
            "decision": "GO",
            "checks": {"frozen": True},
            "protocol_id": protocol.PROTOCOL_ID,
            "receipt_core": core,
            "receipt_core_sha256": freeze.canonical_sha256(core),
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "preflight").mkdir()
            (output / "preflight" / "summary.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            with mock.patch.object(
                freeze,
                "runtime_manifest",
                return_value={"runtime.py": "new"},
            ), mock.patch.object(
                freeze,
                "runtime_environment",
                return_value={"python": "test"},
            ), mock.patch.object(
                freeze,
                "git_freeze_state",
                return_value={
                    "passed": True,
                    "head": "abc",
                    "upstream": "origin/test",
                    "upstream_head": "abc",
                },
            ):
                with self.assertRaisesRegex(RuntimeError, "differs from GO"):
                    freeze.verify_preflight_binding(
                        output,
                        protocol_id=protocol.PROTOCOL_ID,
                        source_root=source_root,
                    )


if __name__ == "__main__":
    unittest.main()
