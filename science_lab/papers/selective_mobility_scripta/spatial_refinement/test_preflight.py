from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import preflight, protocol


def _snapshot(root: Path, name: str = "tmp.Abc123") -> Path:
    candidate = root / name
    candidate.mkdir(mode=0o700)
    os.chmod(candidate, 0o700)
    (candidate / "index").write_bytes(b"DIRC" + b"\0" * 8)
    (candidate / "index.lock").write_bytes(b"")
    (candidate / "objects").mkdir()
    return candidate


class TemporaryGitSnapshotAuditTests(unittest.TestCase):
    def test_exact_temporary_git_fingerprint_is_counted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = _snapshot(root)
            report = preflight.audit_temporary_git_snapshots(root)
            self.assertTrue(report["audit_valid"])
            self.assertEqual(report["validated_count"], 1)
            self.assertEqual(
                report["validated"][0]["path"], str(candidate.resolve())
            )
            self.assertGreater(report["logical_bytes"], 0)

    def test_nonmatching_temp_directory_is_preserved_and_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unrelated = root / "tmp.Unrelated"
            unrelated.mkdir(mode=0o700)
            (unrelated / "science.npy").write_bytes(b"keep")
            report = preflight.audit_temporary_git_snapshots(root)
            self.assertTrue(report["audit_valid"])
            self.assertEqual(report["validated_count"], 0)
            self.assertTrue((unrelated / "science.npy").is_file())

    def test_validated_looking_symlink_is_uncertain_and_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = _snapshot(root)
            (candidate / "objects" / "ab").symlink_to(candidate / "index")
            report = preflight.audit_temporary_git_snapshots(root)
            self.assertFalse(report["audit_valid"])
            self.assertEqual(len(report["uncertain"]), 1)


class ResourcePreflightTests(unittest.TestCase):
    def test_live_legacy_hash_report_has_required_shape_and_passes(self) -> None:
        report = preflight._legacy_hash_report()
        self.assertEqual(
            set(report["checks"]), set(protocol.LEGACY_MODULE_SHA256)
        )
        self.assertEqual(
            set(report["measured_sha256"]), set(protocol.LEGACY_MODULE_SHA256)
        )
        self.assertTrue(report["passed"])

    def test_coarse_reference_archive_is_hash_bound_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "results" / "source-campaign" / "source"
            source.mkdir(parents=True)
            references = protocol.coarse_reference_paths(source)
            for case_id, path in references.summaries.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(case_id.encode("utf-8"))
            expected = {
                case_id: preflight.sha256_path(path)
                for case_id, path in references.summaries.items()
            }
            with mock.patch.object(
                protocol, "COARSE_REFERENCE_SUMMARY_SHA256", expected
            ):
                report = preflight._coarse_reference_hash_report(source)
            self.assertTrue(report["available"])
            self.assertTrue(report["passed"])

            references.summaries["c34"].write_bytes(b"changed")
            with mock.patch.object(
                protocol, "COARSE_REFERENCE_SUMMARY_SHA256", expected
            ):
                report = preflight._coarse_reference_hash_report(source)
            self.assertFalse(report["passed"])

    def test_absent_coarse_reference_archive_does_not_bind_source_location(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "results" / "source-campaign" / "source"
            source.mkdir(parents=True)
            report = preflight._coarse_reference_hash_report(source)
        self.assertFalse(report["available"])
        self.assertTrue(report["passed"])

    def test_go_uses_outputs_transient_reserve_and_separate_volatility(self) -> None:
        snapshot = {
            "audit_valid": True,
            "logical_bytes": 0,
            "validated_count": 0,
            "validated": [],
            "uncertain": [],
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            preflight.runner,
            "verify_external_source",
            return_value={"field": object(), "checks": {"ok": True}},
        ), mock.patch.object(
            preflight,
            "_legacy_hash_report",
            return_value={"passed": True},
        ), mock.patch.object(
            preflight, "audit_temporary_git_snapshots", return_value=snapshot
        ), mock.patch.object(
            preflight, "_competing_science_processes", return_value=[]
        ), mock.patch.object(
            preflight, "_competing_git_processes", return_value=[]
        ), mock.patch.object(
            preflight, "_physical_memory_bytes", return_value=64 << 30
        ), mock.patch.object(
            preflight, "_available_memory_bytes", return_value=64 << 30
        ), mock.patch.object(
            preflight.freeze,
            "runtime_manifest",
            return_value={"runtime.py": "abc"},
        ), mock.patch.object(
            preflight.freeze,
            "git_freeze_state",
            return_value={
                "passed": True,
                "head": "abc",
                "upstream": "origin/test",
                "upstream_head": "abc",
            },
        ):
            report = preflight.run_preflight(
                Path(temporary) / "output",
                drift_window_seconds=0.0,
                disk_free_fn=lambda _path: 100 << 30,
            )
        self.assertEqual(report["decision"], "GO")
        self.assertTrue(report["checks"]["ram_margin"])
        self.assertTrue(report["checks"]["available_ram_margin"])
        self.assertEqual(report["ram"]["peak_plus_margin_bytes"], 40 << 30)
        disk = report["disk"]
        self.assertEqual(
            disk["filesystem_volatility_allowance_bytes"],
            protocol.WORST_CASE_PERSISTENT_BYTES,
        )
        self.assertEqual(
            disk["worst_case_field_persistent_bytes"],
            protocol.WORST_CASE_FIELD_PERSISTENT_BYTES,
        )
        self.assertEqual(
            disk["planned_small_output_allowance_bytes"],
            512 << 20,
        )
        self.assertEqual(
            disk["required_disk_bytes"],
            protocol.WORST_CASE_PERSISTENT_BYTES
            + protocol.LARGEST_ATOMIC_WRITE_BYTES
            + protocol.UNTOUCHED_RESERVE_BYTES
            + protocol.WORST_CASE_PERSISTENT_BYTES,
        )
        receipt = preflight.launch_receipt(report)
        self.assertIn(
            f"planned small-output allowance within persistent output: "
            f"{512 << 20} bytes",
            receipt,
        )

    def test_snapshot_cleanup_threshold_is_no_go_not_auto_delete(self) -> None:
        snapshot = {
            "audit_valid": True,
            "logical_bytes": protocol.TEMPORARY_GIT_SNAPSHOT_CLEANUP_THRESHOLD_BYTES,
            "validated_count": 1,
            "validated": [
                {"path": str(Path(tempfile.gettempdir()) / "fake"), "logical_bytes": 1}
            ],
            "uncertain": [],
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            preflight.runner,
            "verify_external_source",
            return_value={"field": object(), "checks": {"ok": True}},
        ), mock.patch.object(
            preflight,
            "_legacy_hash_report",
            return_value={"passed": True},
        ), mock.patch.object(
            preflight, "audit_temporary_git_snapshots", return_value=snapshot
        ), mock.patch.object(
            preflight, "_competing_science_processes", return_value=[]
        ), mock.patch.object(
            preflight, "_competing_git_processes", return_value=[]
        ), mock.patch.object(
            preflight, "_physical_memory_bytes", return_value=64 << 30
        ), mock.patch.object(
            preflight, "_available_memory_bytes", return_value=64 << 30
        ), mock.patch.object(
            preflight.freeze,
            "runtime_manifest",
            return_value={"runtime.py": "abc"},
        ), mock.patch.object(
            preflight.freeze,
            "git_freeze_state",
            return_value={
                "passed": True,
                "head": "abc",
                "upstream": "origin/test",
                "upstream_head": "abc",
            },
        ):
            report = preflight.run_preflight(
                Path(temporary) / "output",
                drift_window_seconds=0.0,
                disk_free_fn=lambda _path: 100 << 30,
            )
        self.assertEqual(report["decision"], "NO-GO")
        self.assertFalse(
            report["checks"]["temporary_git_snapshots_below_cleanup_threshold"]
        )
        self.assertFalse(report["temporary_git_snapshots"]["cleanup_performed"])

    def test_available_ram_is_a_separate_hard_gate(self) -> None:
        snapshot = {
            "audit_valid": True,
            "logical_bytes": 0,
            "validated_count": 0,
            "validated": [],
            "uncertain": [],
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            preflight.runner,
            "verify_external_source",
            return_value={"field": object(), "checks": {"ok": True}},
        ), mock.patch.object(
            preflight,
            "_legacy_hash_report",
            return_value={"passed": True},
        ), mock.patch.object(
            preflight, "audit_temporary_git_snapshots", return_value=snapshot
        ), mock.patch.object(
            preflight, "_competing_science_processes", return_value=[]
        ), mock.patch.object(
            preflight, "_competing_git_processes", return_value=[]
        ), mock.patch.object(
            preflight, "_physical_memory_bytes", return_value=64 << 30
        ), mock.patch.object(
            preflight, "_available_memory_bytes", return_value=(40 << 30) - 1
        ), mock.patch.object(
            preflight.freeze,
            "runtime_manifest",
            return_value={"runtime.py": "abc"},
        ), mock.patch.object(
            preflight.freeze,
            "git_freeze_state",
            return_value={
                "passed": True,
                "head": "abc",
                "upstream": "origin/test",
                "upstream_head": "abc",
            },
        ):
            report = preflight.run_preflight(
                Path(temporary) / "output",
                drift_window_seconds=0.0,
                disk_free_fn=lambda _path: 100 << 30,
            )
        self.assertTrue(report["checks"]["ram_margin"])
        self.assertFalse(report["checks"]["available_ram_margin"])
        self.assertEqual(report["decision"], "NO-GO")


if __name__ == "__main__":
    unittest.main()
