from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from . import protocol, runner


class RunnerContractTests(unittest.TestCase):
    def test_base_contract_survives_json_round_trip_exactly(self) -> None:
        contract = runner._base_contract("unit_test")
        restored = json.loads(json.dumps(contract))
        runner._verify_contract(restored, contract)

    def test_contract_change_blocks_resume(self) -> None:
        contract = runner._base_contract("unit_test")
        changed = json.loads(json.dumps(contract))
        changed["protocol"]["trajectory"]["horizon"] = 2001
        with self.assertRaisesRegex(RuntimeError, "refusing to resume"):
            runner._verify_contract(changed, contract)

    def test_dead_contract_temp_only_crash_window_recovers_for_every_stage(
        self,
    ) -> None:
        contract = {"stage": "frozen", "contract": "exact"}
        for stage_label in ("fine-source", "guard", "trajectory"):
            with self.subTest(stage=stage_label), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                (output / ".run.lock").touch()
                stale_temp = output / ".contract.json.tmp-99999999"
                stale_temp.write_text("partial", encoding="utf-8")
                with mock.patch.object(runner, "_pid_is_live", return_value=False):
                    runner._recover_or_verify_contract(
                        output,
                        contract,
                        resume=True,
                        output_was_created=False,
                        stage_label=stage_label,
                    )
                self.assertEqual(
                    json.loads(
                        (output / "contract.json").read_text(encoding="utf-8")
                    ),
                    contract,
                )
                self.assertEqual(stale_temp.read_text(encoding="utf-8"), "partial")

    def test_active_contract_temp_is_preserved_and_rejected(self) -> None:
        contract = {"stage": "frozen", "contract": "exact"}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            stale_temp = output / ".contract.json.tmp-99999999"
            stale_temp.write_text("partial", encoding="utf-8")
            with mock.patch.object(runner, "_pid_is_live", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "contract is missing"):
                    runner._recover_or_verify_contract(
                        output,
                        contract,
                        resume=True,
                        output_was_created=False,
                        stage_label="trajectory",
                    )
            self.assertTrue(stale_temp.is_file())
            self.assertFalse((output / "contract.json").exists())

    def test_campaign_identity_rejects_manifest_environment_and_git_drift(
        self,
    ) -> None:
        identity_a = {
            "manifest": {"runtime.py": "aaa", "transitive.py": "bbb"},
            "environment": {
                "python_executable": "/frozen/python",
                "python_version": "3.test",
                "distributions": {"numpy": "test"},
            },
            "git": {
                "head": "head-a",
                "branch": "paper",
                "upstream": "origin/paper",
                "upstream_head": "head-a",
            },
        }

        def contract(identity: dict[str, object]) -> dict[str, object]:
            with mock.patch.object(
                runner.freeze,
                "runtime_manifest",
                return_value=identity["manifest"],
            ), mock.patch.object(
                runner.freeze,
                "runtime_environment",
                return_value=identity["environment"],
            ), mock.patch.object(
                runner.freeze,
                "git_freeze_state",
                return_value=identity["git"],
            ):
                return runner._base_contract(
                    "unit_test", Path(tempfile.gettempdir()) / "source"
                )

        stored = contract(identity_a)
        runner._verify_contract(json.loads(json.dumps(stored)), stored)
        mutations = (
            {
                **identity_a,
                "manifest": {
                    **identity_a["manifest"],
                    "transitive.py": "changed",
                },
            },
            {
                **identity_a,
                "environment": {
                    **identity_a["environment"],
                    "python_executable": "/different/python",
                },
            },
            {
                **identity_a,
                "git": {**identity_a["git"], "head": "head-b"},
            },
        )
        for changed_identity in mutations:
            with self.subTest(identity=changed_identity):
                with self.assertRaisesRegex(RuntimeError, "refusing to resume"):
                    runner._verify_contract(
                        json.loads(json.dumps(stored)),
                        contract(changed_identity),
                    )

    def test_remaining_writes_include_terminal_checkpoint(self) -> None:
        self.assertEqual(runner._remaining_write_count(0, []), 3)
        self.assertEqual(
            runner._remaining_write_count(900, [{"step": 800}]), 2
        )
        self.assertEqual(
            runner._remaining_write_count(
                1700, [{"step": 800}, {"step": 1600}]
            ),
            1,
        )

    def test_runtime_disk_guard_keeps_small_outputs_separate_from_volatility(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            runner,
            "_resource_report",
            return_value={
                "disk": {"filesystem_volatility_allowance_bytes": 1234}
            },
        ), mock.patch.object(
            runner.shutil,
            "disk_usage",
            return_value=SimpleNamespace(free=1 << 50),
        ), mock.patch.object(
            runner.freeze,
            "process_inventory",
            return_value={"science": [], "git": []},
        ):
            report = runner._runtime_disk_guard(
                Path(temporary), current_step=0, checkpoints=[]
            )
        self.assertEqual(
            report["planned_small_output_allowance_bytes"], 512 << 20
        )
        self.assertEqual(
            report["worst_case_persistent_bytes"],
            protocol.WORST_CASE_FIELD_PERSISTENT_BYTES + (512 << 20),
        )
        self.assertEqual(report["filesystem_volatility_allowance_bytes"], 1234)

    def test_complete_orphan_pair_is_adopted_after_exact_replay(self) -> None:
        field = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            protocol, "FINE_SHAPE", field.shape
        ):
            output = Path(temporary)
            runner.write_checkpoint(
                output,
                field,
                step=20,
                kinds=("event_confirmation",),
                stream="event",
            )
            entry = runner._write_or_adopt_checkpoint(
                output,
                field.copy(),
                step=20,
                kinds=("event_confirmation",),
                stream="event",
            )
            restored = runner.load_checkpoint(entry)
        self.assertEqual(
            entry["reconciliation"]["action"],
            "adopted_complete_orphan_pair",
        )
        np.testing.assert_array_equal(restored, field)

    def test_field_only_orphan_is_completed_only_after_exact_replay(self) -> None:
        field = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            protocol, "FINE_SHAPE", field.shape
        ):
            output = Path(temporary)
            field_path, metadata_path = runner.checkpoint_paths(
                output, 800, stream="regular"
            )
            np.save(field_path, field, allow_pickle=False)
            entry = runner._write_or_adopt_checkpoint(
                output,
                field.copy(),
                step=800,
                kinds=("regular",),
                stream="regular",
            )
            self.assertTrue(metadata_path.is_file())
            np.testing.assert_array_equal(runner.load_checkpoint(entry), field)
        self.assertEqual(
            entry["reconciliation"]["action"],
            "completed_field_only_orphan",
        )

    def test_interruption_orphan_at_resume_step_is_adopted(self) -> None:
        field = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            protocol, "FINE_SHAPE", field.shape
        ):
            output = Path(temporary)
            runner.write_checkpoint(
                output,
                field,
                step=0,
                kinds=("interrupted",),
                stream="interrupted",
            )
            artifact = runner._checkpoint_artifact_inventory(output)[
                "artifacts"
            ][0]
            checkpoints: list[dict[str, object]] = []
            entry = runner._adopt_pending_checkpoint_at_current(
                output,
                field.copy(),
                current_step=0,
                event={"detected": False},
                checkpoints=checkpoints,
                artifact=artifact,
            )
        self.assertIsNotNone(entry)
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(
            entry["reconciliation"]["action"],
            "adopted_complete_orphan_pair",
        )

    def test_resume_source_requires_full_entry_equality(self) -> None:
        source = {
            "stream": "source",
            "step": 0,
            "field_sha256": "correct",
        }
        status = {
            "resume_checkpoint": {
                "stream": "source",
                "step": 0,
                "field_sha256": "different",
            },
            "resume_proposal": 0,
            "current_proposal": 0,
        }
        with self.assertRaisesRegex(RuntimeError, "differs from frozen source"):
            runner._validated_resume_checkpoint(status, source, [])

    def test_mismatched_field_only_orphan_is_preserved_and_rejected(self) -> None:
        field = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            protocol, "FINE_SHAPE", field.shape
        ):
            output = Path(temporary)
            field_path, metadata_path = runner.checkpoint_paths(
                output, 800, stream="regular"
            )
            np.save(field_path, field, allow_pickle=False)
            with self.assertRaisesRegex(RuntimeError, "differs from replayed"):
                runner._write_or_adopt_checkpoint(
                    output,
                    field + 1.0,
                    step=800,
                    kinds=("regular",),
                    stream="regular",
                )
            self.assertTrue(field_path.is_file())
            self.assertFalse(metadata_path.exists())

    def test_metadata_only_orphan_is_preserved_and_rejected(self) -> None:
        field = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            protocol, "FINE_SHAPE", field.shape
        ):
            output = Path(temporary)
            runner.write_checkpoint(
                output,
                field,
                step=800,
                kinds=("regular",),
                stream="regular",
            )
            field_path, metadata_path = runner.checkpoint_paths(
                output, 800, stream="regular"
            )
            field_path.unlink()
            with self.assertRaisesRegex(RuntimeError, "metadata-only"):
                runner._write_or_adopt_checkpoint(
                    output,
                    field,
                    step=800,
                    kinds=("regular",),
                    stream="regular",
                )
            self.assertTrue(metadata_path.is_file())

    def test_checkpoint_entry_round_trip_and_corruption_detection(self) -> None:
        field = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            protocol, "FINE_SHAPE", field.shape
        ):
            output = Path(temporary)
            entry = runner.write_checkpoint(
                output,
                field,
                step=20,
                kinds=("event_confirmation",),
                stream="event",
            )
            restored_entry = json.loads(json.dumps(entry))
            validated = runner._validate_recorded_checkpoints(
                output,
                [restored_entry],
                require_complete_inventory=True,
            )
            self.assertEqual(validated, [restored_entry])
            field_path = Path(entry["field_path"])
            np.save(field_path, field + 1.0, allow_pickle=False)
            with self.assertRaisesRegex(RuntimeError, "verification failed"):
                runner._validate_recorded_checkpoints(
                    output,
                    [restored_entry],
                    require_complete_inventory=True,
                )

    def test_terminal_run_status_is_promoted_after_summary_crash_window(self) -> None:
        field = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
        contract = {"contract": "frozen"}
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            protocol, "FINE_SHAPE", field.shape
        ), mock.patch.object(
            runner,
            "verify_trajectory",
            return_value={"status": "completed_event", "verified": True},
        ):
            output_root = Path(temporary)
            output = output_root / "case-untreated"
            output.mkdir()
            checkpoint = runner.write_checkpoint(
                output,
                field,
                step=20,
                kinds=("event_confirmation",),
                stream="event",
            )
            status = {
                "status": "completed_event",
                "case_id": "untreated",
                "stop_reason": "persistent_single_arm_event",
                "current_proposal": 20,
                "event_assessment": {"detected": True},
                "records": [{"step": 0}, {"step": 20}],
                "checkpoints": [checkpoint],
                "health_passed": True,
                "contract": contract,
            }
            runner.atomic_json(output / "run_status.json", status)
            result = runner._promote_terminal_run_status(
                output_root,
                output,
                protocol.UNTREATED,
                contract,
                None,
            )
            committed = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
        self.assertTrue(result["verified"])
        self.assertEqual(committed["status"], "completed_event")

    def test_lock_failure_precedes_fine_source_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            runner, "_verify_stage_binding"
        ), mock.patch.object(
            runner, "_assert_process_guard"
        ), mock.patch.object(
            runner,
            "acquire_output_lock",
            side_effect=RuntimeError("locked"),
        ):
            output_root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "locked"):
                runner.prepare_fine_source(output_root, resume=True)
            output = output_root / "fine-source"
            self.assertTrue(output.is_dir())
            self.assertFalse((output / "contract.json").exists())
            self.assertFalse((output / "run_status.json").exists())

    def test_contract_only_guard_crash_window_restarts_in_same_root(self) -> None:
        source = np.zeros((2, 2, 2), dtype=np.float64)
        frozen_contract = {"stage": "discarded_guard", "frozen": True}

        class GuardSolver:
            def propose_step(
                self, field: np.ndarray, *, timestep: float
            ) -> np.ndarray:
                del timestep
                return field.copy()

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            runner, "_verify_stage_binding"
        ), mock.patch.object(
            runner, "_assert_process_guard"
        ), mock.patch.object(
            runner,
            "verify_fine_source",
            return_value={"field": source.copy()},
        ), mock.patch.object(
            runner, "_guard_contract", return_value=frozen_contract
        ), mock.patch.object(
            runner, "_solver", return_value=GuardSolver()
        ), mock.patch.object(
            runner,
            "_guard_metrics",
            return_value={"passed": True, "checks": {"all": True}},
        ):
            output_root = Path(temporary)
            output = output_root / "timestep-guard"
            output.mkdir()
            (output / ".run.lock").touch()
            summary = runner._run_timestep_guard_with_lock_held(
                output_root,
                resume=True,
                output_was_created=False,
            )
        self.assertTrue(summary["passed"])

    def test_fine_source_complete_pair_is_reconciled_before_summary(self) -> None:
        coarse = np.arange(64, dtype=np.float64).reshape(4, 4, 4)
        fine = coarse.copy()
        metrics = {
            "coincident_node_max_abs": 0.0,
            "mean_abs_difference": 0.0,
            "mass_relative_difference": 0.0,
        }
        frozen_contract = {"stage": "fine_source", "frozen": True}
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            protocol, "FINE_SHAPE", fine.shape
        ), mock.patch.object(
            runner, "_fine_source_contract", return_value=frozen_contract
        ), mock.patch.object(
            runner, "_verify_stage_binding"
        ), mock.patch.object(
            runner, "_assert_process_guard"
        ), mock.patch.object(
            runner,
            "verify_external_source",
            return_value={"field": coarse, "source_root": "frozen"},
        ), mock.patch.object(
            runner.prolongation,
            "periodic_fourier_prolong_factor2",
            return_value=fine.copy(),
        ), mock.patch.object(
            runner.prolongation,
            "prolongation_metrics",
            return_value=metrics,
        ), mock.patch.object(
            runner,
            "_runtime_disk_guard",
            return_value={"passed": True},
        ):
            output_root = Path(temporary)
            output = output_root / "fine-source"
            output.mkdir()
            runner.atomic_json(output / "contract.json", frozen_contract)
            runner.write_checkpoint(
                output,
                fine,
                step=0,
                kinds=("periodic_fourier_prolongated_t120_source",),
                stream="source",
            )
            summary = runner._prepare_fine_source_with_lock_held(
                output_root,
                resume=True,
                output_was_created=False,
            )
            self.assertTrue((output / "summary.json").is_file())
            fields = list(output.glob("*.npy"))
        self.assertEqual(len(fields), 1)
        self.assertEqual(
            summary["checkpoint"]["reconciliation"]["action"],
            "adopted_complete_orphan_pair",
        )

    def test_retained_event_is_terminal_before_another_proposal(self) -> None:
        self.assertFalse(
            runner._trajectory_should_advance(1500, {"detected": True})
        )
        self.assertTrue(
            runner._trajectory_should_advance(1500, {"detected": False})
        )
        self.assertFalse(
            runner._trajectory_should_advance(protocol.HORIZON, {"detected": False})
        )

    def test_event_orphan_replay_adopts_and_makes_no_post_event_proposal(
        self,
    ) -> None:
        shape = (2, 2, 2)
        source_field = np.zeros(shape, dtype=np.float64)
        event_field = np.full(shape, 20.0, dtype=np.float64)
        frozen_contract = {"stage": "trajectory_untreated", "frozen": True}

        class FakeSolver:
            def __init__(self) -> None:
                self.proposal_count = 0

            def propose_step(self, field: np.ndarray) -> np.ndarray:
                self.proposal_count += 1
                return field + 1.0

            def free_energy(self, field: np.ndarray) -> float:
                return float(np.sum(field))

        fake_solver = FakeSolver()

        def record(
            _field: np.ndarray,
            _solver: object,
            *,
            step: int,
            initial_mass: float,
            include_energy: bool,
            previous_energy: float | None,
            include_profiles: bool,
        ) -> dict[str, object]:
            del initial_mass, previous_energy, include_profiles
            return {
                "step": step,
                "field": {
                    "finite": True,
                    "minimum": 0.0,
                    "maximum": 0.0,
                    "mass": 0.0,
                },
                "relative_mass_drift": 0.0,
                "relative_energy_rebound": 0.0,
                "free_energy": float(step) if include_energy else None,
                "pinches": [],
            }

        def event(records: list[dict[str, object]], _event=None) -> dict[str, object]:
            last = int(records[-1]["step"])
            return {
                "detected": last >= 20,
                "event_bracket": [10, 20] if last >= 20 else None,
                "confirmation_step": 20 if last >= 20 else None,
                "persistent_gaps": [],
            }

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            protocol, "FINE_SHAPE", shape
        ), mock.patch.object(
            protocol, "HORIZON", 30
        ), mock.patch.object(
            protocol, "CHECKPOINT_STEPS", ()
        ), mock.patch.object(
            protocol, "DIAGNOSTIC_INTERVAL", 10
        ), mock.patch.object(
            protocol, "ENERGY_INTERVAL", 10
        ), mock.patch.object(
            runner, "_verify_stage_binding"
        ), mock.patch.object(
            runner, "_assert_process_guard"
        ), mock.patch.object(
            runner, "_resource_report", return_value={}
        ), mock.patch.object(
            runner, "verify_timestep_guard", return_value={"passed": True}
        ), mock.patch.object(
            runner, "_trajectory_contract", return_value=frozen_contract
        ), mock.patch.object(
            runner, "_solver", return_value=fake_solver
        ), mock.patch.object(
            runner, "_field_scalars", return_value={"mass": 0.0}
        ), mock.patch.object(
            runner, "_trajectory_record", side_effect=record
        ), mock.patch.object(
            runner, "_event_from_records", side_effect=event
        ), mock.patch.object(
            runner,
            "_runtime_disk_guard",
            return_value={"passed": True},
        ):
            output_root = Path(temporary)
            source_output = output_root / "fine-source"
            source_output.mkdir()
            source_checkpoint = runner.write_checkpoint(
                source_output,
                source_field,
                step=0,
                kinds=("periodic_fourier_prolongated_t120_source",),
                stream="source",
            )
            case_output = output_root / "case-untreated"
            case_output.mkdir()
            runner.atomic_json(case_output / "contract.json", frozen_contract)
            runner.write_checkpoint(
                case_output,
                event_field,
                step=20,
                kinds=("event_confirmation",),
                stream="event",
            )
            initial_record = record(
                source_field,
                fake_solver,
                step=0,
                initial_mass=0.0,
                include_energy=True,
                previous_energy=None,
                include_profiles=True,
            )
            runner.atomic_json(
                case_output / "run_status.json",
                {
                    "status": "running",
                    "case_id": "untreated",
                    "current_proposal": 0,
                    "initial_mass": 0.0,
                    "records": [initial_record],
                    "checkpoints": [],
                    "resume_checkpoint": source_checkpoint,
                    "resume_proposal": 0,
                },
            )
            with mock.patch.object(
                runner,
                "verify_fine_source",
                return_value={"checkpoint": source_checkpoint},
            ):
                summary = runner._run_trajectory_with_lock_held(
                    output_root,
                    protocol.UNTREATED,
                    resume=True,
                    output_was_created=False,
                )

        self.assertEqual(fake_solver.proposal_count, 20)
        self.assertEqual(summary["status"], "completed_event")
        self.assertEqual(
            summary["checkpoints"][-1]["reconciliation"]["action"],
            "adopted_complete_orphan_pair",
        )

    def test_guard_metrics_use_all_frozen_bounds(self) -> None:
        source = np.full((2, 2, 2), 0.5)
        dt1 = source + 1.0e-3
        two_half = source.copy()
        profiles = {
            "first_wire_z": {
                "coordinate": [0.0, 1.0],
                "equivalent_radius": [6.0, 5.9],
            },
            "second_wire_y": {
                "coordinate": [0.0, 1.0],
                "equivalent_radius": [6.0, 5.9],
            },
        }
        with mock.patch.object(
            runner,
            "_contact_normalized",
            side_effect=[{"0.45": 1.0}, {"0.45": 1.001}],
        ), mock.patch.object(runner, "_radius_profiles", return_value=profiles):
            report = runner._guard_metrics(source, dt1, two_half)
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["checks"].values()))


if __name__ == "__main__":
    unittest.main()
