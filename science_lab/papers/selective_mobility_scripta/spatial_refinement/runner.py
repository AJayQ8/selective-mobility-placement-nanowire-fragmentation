#!/usr/bin/env python3
"""Source preparation, discarded timestep guard, and frozen trajectories."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from science_lab.papers.nanowire_gb_junction.roy_2021_reproduction import (
    model as roy_model,
)
from science_lab.papers.nanowire_gb_junction.roy_2021_reproduction.model import (
    RoyPseudospectralSolver,
)
from science_lab.papers.nanowire_gb_junction.roy_2021_reproduction.run_preflight import (
    contact_metrics,
)
from science_lab.papers.nanowire_gb_junction.roy_fixed_gb_bridge.provenance import (
    sha256_path,
)
from science_lab.papers.nanowire_gb_junction.roy_gb_junction_sentinel.diagnostics import (
    instantaneous_pinches,
    persistent_single_arm_event,
)
from science_lab.papers.nanowire_gb_junction.roy_gb_junction_sentinel.initializer import (
    array_fingerprint,
)
from science_lab.papers.nanowire_gb_junction.roy_gb_junction_sentinel.storage import (
    acquire_output_lock,
    atomic_json,
    checkpoint_paths,
    load_checkpoint,
    release_output_lock,
    reserve_output_directory,
    write_checkpoint,
)
from science_lab.papers.nanowire_gb_junction.roy_selective_mobility_protection.geometry import (
    four_arm_tubular_collar_factor,
)
from science_lab.papers.nanowire_gb_junction.roy_selective_mobility_protection.model import (
    SpatialMobilityRoySolver,
)

from . import freeze, prolongation, protocol


SOURCE_PATH = Path(__file__).resolve()
STOP_REQUESTED: dict[str, int | None] = {"signal": None}
CHECKPOINT_ARTIFACT_PATTERN = re.compile(
    r"^checkpoint-(?P<stream>[a-z_-]+)-step-(?P<step>[0-9]+)"
    r"(?P<suffix>\.npy|\.json)$"
)
CONTRACT_TEMP_PATTERN = re.compile(r"^\.contract\.json\.tmp-(?P<pid>[0-9]+)$")
CHECKPOINT_KINDS_BY_STREAM = {
    "source": ("periodic_fourier_prolongated_t120_source",),
    "regular": ("regular",),
    "event": ("event_confirmation",),
    "interrupted": ("interrupted",),
    "horizon": ("fixed_horizon",),
}


def _signal_handler(signum: int, _frame: Any) -> None:
    STOP_REQUESTED["signal"] = int(signum)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _provenance() -> dict[str, Any]:
    return {
        "runner_sha256": sha256_path(SOURCE_PATH),
        "protocol_sha256": sha256_path(Path(protocol.__file__).resolve()),
        "prolongation_sha256": sha256_path(
            Path(prolongation.__file__).resolve()
        ),
        "scientific_contract_sha256": sha256_path(protocol.CONTRACT_PATH),
        "legacy_expected_sha256": dict(protocol.LEGACY_MODULE_SHA256),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "runtime_environment": freeze.runtime_environment(),
    }


def _campaign_identity(source_root: Path | None = None) -> dict[str, Any]:
    git_state = freeze.git_freeze_state()
    identity = {
        "protocol_id": protocol.PROTOCOL_ID,
        "source_root": str(protocol.source_paths(source_root).root),
        "runtime_manifest": freeze.runtime_manifest(),
        "runtime_environment": freeze.runtime_environment(),
        "git": {
            "head": git_state.get("head"),
            "branch": git_state.get("branch"),
            "upstream": git_state.get("upstream"),
            "upstream_head": git_state.get("upstream_head"),
        },
    }
    identity["sha256"] = freeze.canonical_sha256(identity)
    return identity


def _base_contract(
    stage: str, source_root: Path | None = None
) -> dict[str, Any]:
    payload = {
        "schema_version": protocol.SCHEMA_VERSION,
        "protocol": protocol.protocol_payload(),
        "stage": stage,
        "provenance": _provenance(),
        "campaign_identity": _campaign_identity(source_root),
    }
    payload["contract_payload_sha256"] = _canonical_hash(payload)
    return payload


def _verify_contract(stored: dict[str, Any], expected: dict[str, Any]) -> None:
    if stored != expected:
        raise RuntimeError(
            "stored contract differs from the frozen live contract; refusing "
            "to resume"
        )


def _verify_stage_binding(
    output_root: Path, source_root: Path | None
) -> dict[str, Any]:
    resolved_source = protocol.source_paths(source_root).root
    return freeze.verify_preflight_binding(
        output_root,
        protocol_id=protocol.PROTOCOL_ID,
        source_root=resolved_source,
    )


def _assert_process_guard() -> dict[str, Any]:
    inventory = freeze.process_inventory()
    if inventory["science"] or inventory["git"]:
        raise RuntimeError(
            "competing science/Git process blocks the frozen stage: "
            f"{inventory}"
        )
    return inventory


def _pid_is_live(pid: int) -> bool:
    if pid <= 1:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _recover_or_verify_contract(
    output: Path,
    contract: dict[str, Any],
    *,
    resume: bool,
    output_was_created: bool,
    stage_label: str,
) -> None:
    """Recover only a normal dead atomic-contract temp crash window."""

    contract_path = output / "contract.json"
    if output_was_created:
        atomic_json(contract_path, contract)
        return
    if not resume:
        raise FileExistsError(f"{stage_label} output exists; use --resume: {output}")
    if contract_path.is_file():
        _verify_contract(_load_json(contract_path), contract)
        return
    unexpected: list[str] = []
    for path in output.iterdir():
        if path.name == ".run.lock":
            continue
        match = CONTRACT_TEMP_PATTERN.fullmatch(path.name)
        valid_temp = bool(
            match is not None
            and path.is_file()
            and not path.is_symlink()
            and path.stat().st_uid == os.getuid()
            and not _pid_is_live(int(match.group("pid")))
        )
        if not valid_temp:
            unexpected.append(path.name)
    if unexpected:
        raise RuntimeError(
            f"{stage_label} contract is missing beside preserved artifacts: "
            f"{sorted(unexpected)}"
        )
    # The exact dead-writer temps are deliberately preserved for provenance;
    # the 512 MiB small-output allowance covers these tiny crash remnants.
    atomic_json(contract_path, contract)
    _verify_contract(_load_json(contract_path), contract)


def _checkpoint_artifact_inventory(output: Path) -> dict[str, Any]:
    """Inventory committed checkpoint names without modifying crash debris."""

    artifacts: dict[tuple[str, int], dict[str, Any]] = {}
    unknown: list[str] = []
    if not output.is_dir():
        return {"artifacts": [], "unknown": []}
    for path in sorted(output.glob("checkpoint-*")):
        match = CHECKPOINT_ARTIFACT_PATTERN.fullmatch(path.name)
        if match is None:
            unknown.append(str(path.resolve()))
            continue
        stream = match.group("stream")
        step = int(match.group("step"))
        canonical_paths = checkpoint_paths(output, step, stream=stream)
        expected_path = (
            canonical_paths[0]
            if match.group("suffix") == ".npy"
            else canonical_paths[1]
        )
        if path != expected_path:
            unknown.append(str(path.resolve()))
            continue
        key = (stream, step)
        row = artifacts.setdefault(
            key,
            {
                "stream": stream,
                "step": step,
                "field_path": None,
                "metadata_path": None,
                "has_symlink": False,
            },
        )
        member = (
            "field_path" if match.group("suffix") == ".npy" else "metadata_path"
        )
        if row[member] is not None:
            unknown.append(str(path.resolve()))
        if path.is_symlink():
            row["has_symlink"] = True
        row[member] = str(path.resolve())
    return {
        "artifacts": [artifacts[key] for key in sorted(artifacts)],
        "unknown": unknown,
    }


def _checkpoint_array(path: Path) -> np.ndarray:
    try:
        field = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as error:
        raise RuntimeError(f"checkpoint field is unreadable: {path}") from error
    if tuple(field.shape) != protocol.FINE_SHAPE:
        raise RuntimeError(f"checkpoint field shape mismatch: {path}")
    if field.dtype.str != "<f8":
        raise RuntimeError(f"checkpoint field dtype mismatch: {path}")
    if not np.isfinite(field).all():
        raise RuntimeError(f"checkpoint field is nonfinite: {path}")
    return field


def _assert_checkpoint_matches_field(
    checkpoint_field: np.ndarray, field: np.ndarray
) -> None:
    """Compare a mapped orphan to replayed state exactly with bounded memory."""

    if checkpoint_field.shape != field.shape or checkpoint_field.dtype != field.dtype:
        raise RuntimeError("orphan checkpoint differs from replayed field metadata")
    for first_axis_index in range(field.shape[0]):
        stored_bytes = np.ascontiguousarray(
            checkpoint_field[first_axis_index]
        ).view(np.uint8)
        replayed_bytes = np.ascontiguousarray(field[first_axis_index]).view(
            np.uint8
        )
        if not np.array_equal(
            stored_bytes, replayed_bytes
        ):
            raise RuntimeError("orphan checkpoint differs from replayed field")


def _checkpoint_entry_from_pair(
    output: Path,
    *,
    step: int,
    stream: str,
    expected_kinds: tuple[str, ...],
    field: np.ndarray | None = None,
) -> dict[str, Any]:
    """Validate and reconstruct the full entry for a committed artifact pair."""

    field_path, metadata_path = checkpoint_paths(output, step, stream=stream)
    if not field_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(
            f"checkpoint pair is incomplete for {stream} step {step}"
        )
    if field_path.is_symlink() or metadata_path.is_symlink():
        raise RuntimeError("checkpoint artifacts must not be symbolic links")
    metadata = _load_json(metadata_path)
    checkpoint_field = _checkpoint_array(field_path)
    checks = {
        "step": int(metadata.get("step", -1)) == int(step),
        "stream": metadata.get("stream") == stream,
        "kinds": metadata.get("kinds") == list(dict.fromkeys(expected_kinds)),
        "field_path": Path(str(metadata.get("field_path", ""))).resolve()
        == field_path.resolve(),
        "field_bytes": int(metadata.get("field_bytes", -1))
        == int(field_path.stat().st_size),
        "field_sha256": metadata.get("field_sha256") == sha256_path(field_path),
        "field_fingerprint": metadata.get("field_fingerprint")
        == array_fingerprint(checkpoint_field),
        "shape": tuple(metadata.get("shape", ())) == protocol.FINE_SHAPE,
        "dtype": metadata.get("dtype") == "<f8",
        "finite": metadata.get("finite") is True,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"checkpoint metadata verification failed for {stream} step "
            f"{step}: {checks}"
        )
    if field is not None:
        _assert_checkpoint_matches_field(checkpoint_field, field)
    return {
        **metadata,
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": sha256_path(metadata_path),
        "elapsed_physical_time": float(step),
    }


def _complete_field_only_checkpoint(
    output: Path,
    field: np.ndarray,
    *,
    step: int,
    stream: str,
    kinds: tuple[str, ...],
) -> None:
    """Complete only an atomically committed field proven equal to replay."""

    field_path, metadata_path = checkpoint_paths(output, step, stream=stream)
    if not field_path.is_file() or metadata_path.exists():
        raise RuntimeError("field-only recovery precondition failed")
    if field_path.is_symlink():
        raise RuntimeError("field-only checkpoint must not be a symbolic link")
    checkpoint_field = _checkpoint_array(field_path)
    _assert_checkpoint_matches_field(checkpoint_field, field)
    metadata = {
        "step": int(step),
        "stream": stream,
        "kinds": list(dict.fromkeys(kinds)),
        "field_path": str(field_path.resolve()),
        "field_bytes": int(field_path.stat().st_size),
        "field_sha256": sha256_path(field_path),
        "field_fingerprint": array_fingerprint(checkpoint_field),
        "shape": list(checkpoint_field.shape),
        "dtype": checkpoint_field.dtype.str,
        "finite": True,
    }
    atomic_json(metadata_path, metadata)


def _write_or_adopt_checkpoint(
    output: Path,
    field: np.ndarray,
    *,
    step: int,
    stream: str,
    kinds: tuple[str, ...],
) -> dict[str, Any]:
    """Write new, adopt a complete orphan, or complete a proven field orphan."""

    field_path, metadata_path = checkpoint_paths(output, step, stream=stream)
    field_exists = field_path.exists()
    metadata_exists = metadata_path.exists()
    if metadata_exists and not field_exists:
        raise RuntimeError(
            f"metadata-only checkpoint orphan is unsafe and preserved: "
            f"{metadata_path}"
        )
    if not field_exists and not metadata_exists:
        entry = write_checkpoint(
            output,
            field,
            step=step,
            kinds=kinds,
            stream=stream,
        )
        action = "written_new"
    else:
        if field_exists and not metadata_exists:
            _complete_field_only_checkpoint(
                output,
                field,
                step=step,
                stream=stream,
                kinds=kinds,
            )
            action = "completed_field_only_orphan"
        else:
            action = "adopted_complete_orphan_pair"
        entry = _checkpoint_entry_from_pair(
            output,
            step=step,
            stream=stream,
            expected_kinds=kinds,
            field=field,
        )
    entry["reconciliation"] = {
        "action": action,
        "validated_against_replayed_field": action != "written_new",
    }
    return entry


def _adopt_pending_checkpoint_at_current(
    output: Path,
    field: np.ndarray,
    *,
    current_step: int,
    event: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    artifact: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Reconcile the one orphan exactly at the authoritative resume state."""

    if artifact is None:
        return None
    stream = str(artifact["stream"])
    step = int(artifact["step"])
    if step != current_step:
        raise RuntimeError("pending checkpoint is not at the resume step")
    if any(int(row["step"]) == current_step for row in checkpoints):
        raise RuntimeError("parallel unrecorded checkpoint exists at resume step")
    event_detected = event.get("detected") is True
    semantics = {
        "interrupted": True,
        "regular": current_step in protocol.CHECKPOINT_STEPS
        and not event_detected,
        "event": event_detected,
        "horizon": current_step == protocol.HORIZON and not event_detected,
    }
    if semantics.get(stream) is not True:
        raise RuntimeError(
            f"pending {stream} checkpoint contradicts replayed resume state"
        )
    entry = _write_or_adopt_checkpoint(
        output,
        field,
        step=current_step,
        stream=stream,
        kinds=CHECKPOINT_KINDS_BY_STREAM[stream],
    )
    checkpoints.append(entry)
    return entry


def _validate_checkpoint_inventory(
    output: Path,
    *,
    stage: str,
) -> dict[str, Any]:
    """Fail closed on malformed/unsafe halves; validate every complete pair."""

    inventory = _checkpoint_artifact_inventory(output)
    if inventory["unknown"]:
        raise RuntimeError(
            f"unrecognized checkpoint artifacts are preserved: {inventory['unknown']}"
        )
    terminal_count = 0
    interrupted_count = 0
    for artifact in inventory["artifacts"]:
        stream = str(artifact["stream"])
        step = int(artifact["step"])
        field_exists = artifact["field_path"] is not None
        metadata_exists = artifact["metadata_path"] is not None
        if artifact["has_symlink"]:
            raise RuntimeError("checkpoint artifacts must not be symbolic links")
        if metadata_exists and not field_exists:
            raise RuntimeError(
                "metadata-only checkpoint orphan is unsafe and preserved: "
                f"{artifact['metadata_path']}"
            )
        if stage == "fine_source":
            allowed = stream == "source" and step == 0
        else:
            allowed = (
                (stream == "regular" and step in protocol.CHECKPOINT_STEPS)
                or (
                    stream == "event"
                    and 0 < step <= protocol.HORIZON
                    and step % protocol.DIAGNOSTIC_INTERVAL == 0
                )
                or (stream == "interrupted" and 0 <= step <= protocol.HORIZON)
                or (stream == "horizon" and step == protocol.HORIZON)
            )
        if not allowed:
            raise RuntimeError(
                f"checkpoint artifact is outside frozen {stage} semantics: "
                f"{stream} step {step}"
            )
        if stream in {"event", "horizon"}:
            terminal_count += 1
        if stream == "interrupted":
            interrupted_count += 1
        if field_exists and metadata_exists:
            _checkpoint_entry_from_pair(
                output,
                step=step,
                stream=stream,
                expected_kinds=CHECKPOINT_KINDS_BY_STREAM[stream],
            )
        elif field_exists:
            _checkpoint_array(Path(str(artifact["field_path"])))
    if terminal_count > 1:
        raise RuntimeError("more than one terminal checkpoint artifact is present")
    if interrupted_count > 1:
        raise RuntimeError("more than one interruption checkpoint artifact is present")
    return inventory


def _checkpoint_identity(entry: dict[str, Any]) -> tuple[str, int]:
    return str(entry["stream"]), int(entry["step"])


def _validate_recorded_checkpoints(
    output: Path,
    checkpoints: list[dict[str, Any]],
    *,
    require_complete_inventory: bool,
) -> list[dict[str, Any]]:
    """Revalidate recorded entries and optionally reject every orphan pair."""

    inventory = _validate_checkpoint_inventory(output, stage="trajectory")
    validated: list[dict[str, Any]] = []
    recorded_identities: set[tuple[str, int]] = set()
    recorded_steps = [int(row["step"]) for row in checkpoints]
    if recorded_steps != sorted(recorded_steps):
        raise RuntimeError("recorded checkpoints are not chronological")
    for recorded in checkpoints:
        stream, step = _checkpoint_identity(recorded)
        if (stream, step) in recorded_identities:
            raise RuntimeError("duplicate recorded checkpoint identity")
        expected_kinds = CHECKPOINT_KINDS_BY_STREAM.get(stream)
        if expected_kinds is None:
            raise RuntimeError(f"unknown recorded checkpoint stream: {stream}")
        measured = _checkpoint_entry_from_pair(
            output,
            step=step,
            stream=stream,
            expected_kinds=expected_kinds,
        )
        for key in (
            "step",
            "stream",
            "kinds",
            "field_path",
            "field_bytes",
            "field_sha256",
            "field_fingerprint",
            "shape",
            "dtype",
            "finite",
            "metadata_path",
            "metadata_sha256",
        ):
            if recorded.get(key) != measured.get(key):
                raise RuntimeError(
                    f"recorded checkpoint differs from artifacts for {stream} "
                    f"step {step}: {key}"
                )
        validated.append(recorded)
        recorded_identities.add((stream, step))
    if require_complete_inventory:
        artifact_identities = {
            (str(row["stream"]), int(row["step"]))
            for row in inventory["artifacts"]
        }
        if artifact_identities != recorded_identities:
            raise RuntimeError(
                "terminal checkpoint inventory contains unrecorded or missing "
                f"artifacts: recorded={sorted(recorded_identities)}, "
                f"artifacts={sorted(artifact_identities)}"
            )
    return validated


def _terminal_status_checks(status: dict[str, Any]) -> dict[str, bool]:
    terminal = status.get("status")
    current_step = int(status.get("current_proposal", -1))
    checkpoints = list(status.get("checkpoints", []))
    record_steps = [int(row.get("step", -1)) for row in status.get("records", [])]
    current = [
        row for row in checkpoints if int(row.get("step", -1)) == current_step
    ]
    if terminal == "completed_event":
        terminal_checkpoint = any(row.get("stream") == "event" for row in current)
        terminal_semantics = (
            status.get("stop_reason") == "persistent_single_arm_event"
            and status.get("event_assessment", {}).get("detected") is True
        )
    elif terminal == "completed_horizon":
        terminal_checkpoint = any(
            row.get("stream") == "horizon" for row in current
        )
        terminal_semantics = (
            status.get("stop_reason") == "fixed_elapsed_horizon"
            and current_step == protocol.HORIZON
        )
    else:
        terminal_checkpoint = False
        terminal_semantics = False
    return {
        "terminal_status": terminal in {"completed_event", "completed_horizon"},
        "terminal_checkpoint": terminal_checkpoint,
        "terminal_semantics": terminal_semantics,
        "terminal_record": bool(record_steps and record_steps[-1] == current_step),
        "health": status.get("health_passed") is True,
    }


def _validate_record_sequence(records: list[dict[str, Any]]) -> list[int]:
    steps = [int(row["step"]) for row in records]
    if not steps or steps[0] != 0 or steps != sorted(set(steps)):
        raise RuntimeError("diagnostic records are not complete, unique, and ordered")
    if any(
        step != 0 and step % protocol.DIAGNOSTIC_INTERVAL != 0
        for step in steps
    ):
        raise RuntimeError("diagnostic record is off frozen cadence")
    return steps


def _validated_resume_checkpoint(
    status: dict[str, Any],
    source_checkpoint: dict[str, Any],
    recorded_checkpoints: list[dict[str, Any]],
) -> dict[str, Any]:
    checkpoint = status["resume_checkpoint"]
    resume_identity = _checkpoint_identity(checkpoint)
    if resume_identity == _checkpoint_identity(source_checkpoint):
        if checkpoint != source_checkpoint:
            raise RuntimeError("resume source checkpoint differs from frozen source")
    else:
        matching_recorded = [
            row
            for row in recorded_checkpoints
            if _checkpoint_identity(row) == resume_identity
        ]
        if len(matching_recorded) != 1 or matching_recorded[0] != checkpoint:
            raise RuntimeError(
                "resume checkpoint is neither the exact source nor one "
                "recorded checkpoint"
            )
    if int(status.get("resume_proposal", -1)) != int(checkpoint["step"]):
        raise RuntimeError("resume proposal does not match checkpoint step")
    if int(checkpoint["step"]) > int(status.get("current_proposal", -1)):
        raise RuntimeError("resume checkpoint is beyond journal progress")
    return checkpoint


def _promote_terminal_run_status(
    output_root: Path,
    output: Path,
    case: protocol.CaseDefinition,
    contract: dict[str, Any],
    source_root: Path | None,
) -> dict[str, Any] | None:
    """Commit a terminal run_status left just before summary publication."""

    status_path = output / "run_status.json"
    if not status_path.is_file():
        return None
    status = _load_json(status_path)
    if status.get("status") not in {"completed_event", "completed_horizon"}:
        return None
    _verify_contract(status["contract"], contract)
    _validate_record_sequence(list(status.get("records", [])))
    _validate_recorded_checkpoints(
        output,
        list(status.get("checkpoints", [])),
        require_complete_inventory=True,
    )
    checks = _terminal_status_checks(status)
    if not all(checks.values()):
        raise RuntimeError(
            f"terminal run-status promotion checks failed: {checks}"
        )
    if status.get("case_id") != case.case_id:
        raise RuntimeError("terminal run-status case mismatch")
    atomic_json(output / "summary.json", status)
    return verify_trajectory(
        output_root,
        case.case_id,
        source_root=source_root,
    )


def verify_external_source(
    *, load_field: bool, source_root: Path | None = None
) -> dict[str, Any]:
    source_paths = protocol.source_paths(source_root)
    paths = {
        "field": source_paths.field,
        "metadata": source_paths.metadata,
        "summary": source_paths.summary,
    }
    expected_hashes = {
        "field": protocol.EXTERNAL_SOURCE_FIELD_SHA256,
        "metadata": protocol.EXTERNAL_SOURCE_METADATA_SHA256,
        "summary": protocol.EXTERNAL_SOURCE_SUMMARY_SHA256,
    }
    hashes = {
        name: sha256_path(path) if path.is_file() else None
        for name, path in paths.items()
    }
    if hashes != expected_hashes:
        raise RuntimeError(
            f"immutable coarse source verification failed: {hashes}"
        )
    metadata = _load_json(paths["metadata"])
    summary = _load_json(paths["summary"])
    checks = {
        "metadata_field_hash": metadata.get("field_sha256")
        == protocol.EXTERNAL_SOURCE_FIELD_SHA256,
        "metadata_fingerprint": metadata.get("field_fingerprint")
        == protocol.EXTERNAL_SOURCE_FINGERPRINT,
        "metadata_shape": tuple(metadata.get("shape", ()))
        == protocol.COARSE_SHAPE,
        "metadata_dtype": metadata.get("dtype") == "<f8",
        "metadata_step": int(metadata.get("step", -1)) == 120,
        "summary_accepted": summary.get("acceptance", {}).get("passed") is True,
        "summary_time": float(summary.get("selected", {}).get("physical_time", -1))
        == protocol.EXTERNAL_SOURCE_TIME,
        "summary_checkpoint_hash": summary.get("checkpoint", {}).get(
            "field_sha256"
        )
        == protocol.EXTERNAL_SOURCE_FIELD_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"coarse source metadata failed: {checks}")
    report: dict[str, Any] = {
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
        "sha256": hashes,
        "checks": checks,
        "field_fingerprint": protocol.EXTERNAL_SOURCE_FINGERPRINT,
        "source_root": str(source_paths.root),
    }
    if load_field:
        field = np.load(paths["field"], allow_pickle=False, mmap_mode="r")
        if tuple(field.shape) != protocol.COARSE_SHAPE or field.dtype.str != "<f8":
            raise RuntimeError("coarse source array shape/dtype mismatch")
        if not np.isfinite(field).all():
            raise RuntimeError("coarse source array is nonfinite")
        fingerprint = array_fingerprint(field)
        if fingerprint != protocol.EXTERNAL_SOURCE_FINGERPRINT:
            raise RuntimeError("coarse source array fingerprint mismatch")
        report["field"] = field
    return report


def _fine_source_contract(source_root: Path | None = None) -> dict[str, Any]:
    source_paths = protocol.source_paths(source_root)
    contract = _base_contract("fine_source", source_root)
    contract["external_source"] = {
        "source_root": str(source_paths.root),
        "field_path": str(source_paths.field),
        "field_sha256": protocol.EXTERNAL_SOURCE_FIELD_SHA256,
        "metadata_path": str(source_paths.metadata),
        "metadata_sha256": protocol.EXTERNAL_SOURCE_METADATA_SHA256,
        "summary_path": str(source_paths.summary),
        "summary_sha256": protocol.EXTERNAL_SOURCE_SUMMARY_SHA256,
    }
    contract["contract_payload_sha256"] = _canonical_hash(
        {key: value for key, value in contract.items() if key != "contract_payload_sha256"}
    )
    return contract


def verify_fine_source(
    output_root: Path,
    *,
    load_field: bool,
    source_root: Path | None = None,
) -> dict[str, Any]:
    source_output = output_root / "fine-source"
    inventory = _validate_checkpoint_inventory(source_output, stage="fine_source")
    summary = _load_json(source_output / "summary.json")
    _verify_contract(summary["contract"], _fine_source_contract(source_root))
    checks = {
        "status": summary.get("status") == "completed",
        "classification": summary.get("classification")
        == "fine_source_fourier_prolongation_passed",
        "accepted": summary.get("acceptance", {}).get("passed") is True,
        "field_shape": tuple(summary.get("checkpoint", {}).get("shape", ()))
        == protocol.FINE_SHAPE,
    }
    if not all(checks.values()):
        raise RuntimeError(f"fine source verification failed: {checks}")
    checkpoint = summary["checkpoint"]
    measured = _checkpoint_entry_from_pair(
        source_output,
        step=0,
        stream="source",
        expected_kinds=CHECKPOINT_KINDS_BY_STREAM["source"],
    )
    matching_keys = (
        "step",
        "stream",
        "kinds",
        "field_path",
        "field_bytes",
        "field_sha256",
        "field_fingerprint",
        "shape",
        "dtype",
        "finite",
        "metadata_path",
        "metadata_sha256",
    )
    checks.update(
        {
            "single_source_checkpoint": len(inventory["artifacts"]) == 1,
            "checkpoint_entry": all(
                checkpoint.get(key) == measured.get(key) for key in matching_keys
            ),
        }
    )
    if not all(checks.values()):
        raise RuntimeError(f"fine source artifact verification failed: {checks}")
    if load_field:
        summary["field"] = load_checkpoint(checkpoint)
    return summary


def _prepare_fine_source_with_lock_held(
    output_root: Path,
    *,
    resume: bool,
    source_root: Path | None = None,
    output_was_created: bool,
) -> dict[str, Any]:
    """Verify and Fourier-prolong the sole immutable coarse source."""

    _verify_stage_binding(output_root, source_root)
    _assert_process_guard()
    output = output_root / "fine-source"
    if (output / "summary.json").is_file():
        return verify_fine_source(
            output_root, load_field=False, source_root=source_root
        )
    contract = _fine_source_contract(source_root)
    _recover_or_verify_contract(
        output,
        contract,
        resume=resume,
        output_was_created=output_was_created,
        stage_label="fine-source",
    )
    if not output_was_created:
        _validate_checkpoint_inventory(output, stage="fine_source")

    try:
        external = verify_external_source(
            load_field=True, source_root=source_root
        )
        coarse = external.pop("field")
        fine = prolongation.periodic_fourier_prolong_factor2(
            coarse, workers=protocol.FFT_WORKERS
        )
        if STOP_REQUESTED["signal"] is not None:
            interrupted = {
                "status": "interrupted",
                "classification": "fine_source_interrupted_before_checkpoint",
                "signal": STOP_REQUESTED["signal"],
                "contract": contract,
            }
            atomic_json(output / "run_status.json", interrupted)
            return interrupted
        metrics = prolongation.prolongation_metrics(
            coarse,
            fine,
            coarse_spacing=protocol.COARSE_SPACING,
            fine_spacing=protocol.FINE_SPACING,
        )
        checks = {
            "shape": tuple(fine.shape) == protocol.FINE_SHAPE,
            "dtype": fine.dtype == np.float64,
            "finite": bool(np.isfinite(fine).all()),
            "coincident_nodes": metrics["coincident_node_max_abs"]
            <= protocol.SOURCE_COINCIDENT_ATOL,
            "mean": metrics["mean_abs_difference"]
            <= protocol.SOURCE_MEAN_ATOL,
            "physical_mass": metrics["mass_relative_difference"]
            <= protocol.SOURCE_MASS_RELATIVE_LIMIT,
        }
        if not all(checks.values()):
            raise RuntimeError(f"fine source interpolation checks failed: {checks}")
        disk_guard = _runtime_disk_guard(
            output_root, current_step=0, checkpoints=[]
        )
        if not disk_guard["passed"]:
            raise RuntimeError(
                f"fine-source checkpoint resource guard failed: {disk_guard}"
            )
        checkpoint = _write_or_adopt_checkpoint(
            output,
            fine,
            step=0,
            kinds=("periodic_fourier_prolongated_t120_source",),
            stream="source",
        )
        summary = {
            "schema_version": protocol.SCHEMA_VERSION,
            "status": "completed",
            "classification": "fine_source_fourier_prolongation_passed",
            "external_source": external,
            "prolongation_metrics": metrics,
            "acceptance": {"passed": True, "checks": checks},
            "checkpoint": checkpoint,
            "contract": contract,
        }
        atomic_json(output / "summary.json", summary)
        atomic_json(output / "run_status.json", summary)
        return summary
    finally:
        gc.collect()


def prepare_fine_source(
    output_root: Path,
    *,
    resume: bool,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Prepare/reconcile the fine source with all mutable state under lock."""

    _verify_stage_binding(output_root, source_root)
    _assert_process_guard()
    output = output_root / "fine-source"
    output_was_created = False
    if not output.exists():
        reserve_output_directory(output)
        output_was_created = True
    lock = acquire_output_lock(output)
    try:
        return _prepare_fine_source_with_lock_held(
            output_root,
            resume=resume,
            source_root=source_root,
            output_was_created=output_was_created,
        )
    finally:
        release_output_lock(lock)


def _definition() -> roy_model.RoyDeg90Definition:
    return protocol.fine_definition()


def _mobility_factor(case: protocol.CaseDefinition) -> np.ndarray:
    if case.untreated:
        result = np.ones((1,), dtype=np.float64)
        result.setflags(write=False)
        return result
    geometry = case.geometry
    if geometry is None:
        raise RuntimeError("treated case is missing its frozen geometry")
    return four_arm_tubular_collar_factor(
        _definition().lattice,
        geometry,
        radius=protocol.RADIUS,
        interface_width=protocol.WIDTH,
        first_wire_center_x=protocol.FIRST_WIRE_CENTER_X,
        second_wire_center_x=protocol.SECOND_WIRE_CENTER_X,
    )


def _solver(case: protocol.CaseDefinition) -> RoyPseudospectralSolver:
    definition = _definition()
    if case.untreated:
        return RoyPseudospectralSolver(
            definition.lattice,
            definition.parameters,
            fft_workers=protocol.FFT_WORKERS,
        )
    factor = _mobility_factor(case)
    solver = SpatialMobilityRoySolver(
        definition.lattice,
        definition.parameters,
        factor,
        fft_workers=protocol.FFT_WORKERS,
    )
    del factor
    gc.collect()
    return solver


def _field_scalars(field: np.ndarray) -> dict[str, Any]:
    definition = _definition()
    finite = bool(np.isfinite(field).all())
    report: dict[str, Any] = {
        "shape": list(field.shape),
        "dtype": field.dtype.str,
        "finite": finite,
        "minimum": None,
        "maximum": None,
        "mean": None,
        "mass": None,
    }
    if finite:
        report.update(
            {
                "minimum": float(np.min(field)),
                "maximum": float(np.max(field)),
                "mean": float(np.mean(field, dtype=np.float64)),
                "mass": float(
                    np.sum(field, dtype=np.float64)
                    * definition.lattice.cell_volume
                ),
            }
        )
    return report


def _radius_profiles(field: np.ndarray) -> dict[str, dict[str, list[float]]]:
    spacing = protocol.FINE_SPACING
    output: dict[str, dict[str, list[float]]] = {}
    for name, axis in (("first_wire_z", 2), ("second_wire_y", 1)):
        transverse = tuple(index for index in range(3) if index != axis)
        area = (
            np.count_nonzero(field >= 0.50, axis=transverse).astype(np.float64)
            * spacing**2
        )
        cells = field.shape[axis]
        coordinate = (
            np.arange(cells, dtype=np.float64) - cells // 2
        ) * spacing
        output[name] = {
            "coordinate": coordinate.tolist(),
            "equivalent_radius": np.sqrt(area / np.pi).tolist(),
        }
    return output


def _contact_normalized(field: np.ndarray) -> dict[str, float]:
    measured = contact_metrics(field, _definition())
    return {
        key: float(value["contact_plane_equivalent_radius_over_R"])
        for key, value in measured["thresholds"].items()
    }


def _guard_metrics(
    source: np.ndarray,
    dt1: np.ndarray,
    two_half: np.ndarray,
) -> dict[str, Any]:
    active = (source > 0.05) & (source < 0.95)
    if not np.any(active):
        raise RuntimeError("fine source has no active diffuse-interface cells")
    difference = dt1 - two_half
    interface_rmse = float(np.sqrt(np.mean(difference[active] ** 2)))
    contact_a = _contact_normalized(dt1)
    contact_b = _contact_normalized(two_half)
    contact_max = max(abs(contact_a[key] - contact_b[key]) for key in contact_a)
    profiles_a = _radius_profiles(dt1)
    profiles_b = _radius_profiles(two_half)
    profile_rows: dict[str, Any] = {}
    for name in profiles_a:
        a = np.asarray(profiles_a[name]["equivalent_radius"])
        b = np.asarray(profiles_b[name]["equivalent_radius"])
        normalized = (a - b) / protocol.RADIUS
        profile_rows[name] = {
            "normalized_rmse": float(np.sqrt(np.mean(normalized**2))),
            "normalized_max_abs": float(np.max(np.abs(normalized))),
        }
    profile_rmse = max(row["normalized_rmse"] for row in profile_rows.values())
    profile_max = max(row["normalized_max_abs"] for row in profile_rows.values())
    checks = {
        "finite": bool(np.isfinite(dt1).all() and np.isfinite(two_half).all()),
        "interface_rmse": interface_rmse
        <= protocol.GUARD_INTERFACE_RMSE_LIMIT,
        "contact_normalized": contact_max
        <= protocol.GUARD_CONTACT_NORMALIZED_LIMIT,
        "profile_normalized_rmse": profile_rmse
        <= protocol.GUARD_PROFILE_NORMALIZED_RMSE_LIMIT,
        "profile_normalized_max": profile_max
        <= protocol.GUARD_PROFILE_NORMALIZED_MAX_LIMIT,
    }
    return {
        "active_interface_cell_count": int(np.count_nonzero(active)),
        "interface_rmse": interface_rmse,
        "contact_normalized_by_path": {"dt1": contact_a, "two_dt0p5": contact_b},
        "contact_normalized_max_abs_difference": contact_max,
        "profiles": profile_rows,
        "profile_normalized_rmse_max": profile_rmse,
        "profile_normalized_abs_max": profile_max,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _guard_contract(source_root: Path | None = None) -> dict[str, Any]:
    contract = _base_contract("discarded_timestep_guard", source_root)
    contract["cases"] = [case.to_dict() for case in protocol.CASES]
    contract["source_summary"] = str(
        (protocol.DEFAULT_OUTPUT / "fine-source" / "summary.json").resolve()
    )
    contract["contract_payload_sha256"] = _canonical_hash(
        {key: value for key, value in contract.items() if key != "contract_payload_sha256"}
    )
    return contract


def _resolved_guard_contract(
    output_root: Path, source_root: Path | None = None
) -> dict[str, Any]:
    contract = _guard_contract(source_root)
    contract["source_summary"] = str(
        (output_root / "fine-source" / "summary.json").resolve()
    )
    contract["contract_payload_sha256"] = _canonical_hash(
        {
            key: value
            for key, value in contract.items()
            if key != "contract_payload_sha256"
        }
    )
    return contract


def _run_timestep_guard_with_lock_held(
    output_root: Path,
    *,
    resume: bool,
    source_root: Path | None = None,
    output_was_created: bool,
) -> dict[str, Any]:
    _verify_stage_binding(output_root, source_root)
    _assert_process_guard()
    output = output_root / "timestep-guard"
    if (output / "summary.json").is_file():
        summary = _load_json(output / "summary.json")
        _verify_contract(
            summary["contract"],
            _resolved_guard_contract(output_root, source_root),
        )
        if summary.get("passed") is not True:
            raise RuntimeError("production is blocked by the timestep guard")
        return summary
    fine_source = verify_fine_source(
        output_root, load_field=True, source_root=source_root
    )
    source = fine_source.pop("field")
    contract = _resolved_guard_contract(output_root, source_root)
    _recover_or_verify_contract(
        output,
        contract,
        resume=resume,
        output_was_created=output_was_created,
        stage_label="guard",
    )
    rows: dict[str, Any] = {}
    for case in protocol.CASES:
        solver = _solver(case)
        dt1 = solver.propose_step(source, timestep=1.0)
        half = solver.propose_step(source, timestep=0.5)
        two_half = solver.propose_step(half, timestep=0.5)
        del half
        rows[case.case_id] = _guard_metrics(source, dt1, two_half)
        del dt1, two_half, solver
        gc.collect()
    passed = all(row["passed"] for row in rows.values())
    summary = {
        "schema_version": protocol.SCHEMA_VERSION,
        "status": "completed" if passed else "failed",
        "classification": (
            "discarded_timestep_guard_passed"
            if passed
            else "discarded_timestep_guard_failed"
        ),
        "passed": passed,
        "cases": rows,
        "fields_retained": False,
        "production_timestep_remains": protocol.TIMESTEP,
        "contract": contract,
    }
    atomic_json(output / "summary.json", summary)
    if not passed:
        raise RuntimeError("discarded timestep guard failed; queue stops")
    return summary


def run_timestep_guard(
    output_root: Path,
    *,
    resume: bool,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Run/reconcile the discarded guard with mutable state under lock."""

    _verify_stage_binding(output_root, source_root)
    _assert_process_guard()
    output = output_root / "timestep-guard"
    output_was_created = False
    if not output.exists():
        reserve_output_directory(output)
        output_was_created = True
    lock = acquire_output_lock(output)
    try:
        return _run_timestep_guard_with_lock_held(
            output_root,
            resume=resume,
            source_root=source_root,
            output_was_created=output_was_created,
        )
    finally:
        release_output_lock(lock)


def verify_timestep_guard(
    output_root: Path, *, source_root: Path | None = None
) -> dict[str, Any]:
    summary = _load_json(output_root / "timestep-guard" / "summary.json")
    _verify_contract(
        summary["contract"],
        _resolved_guard_contract(output_root, source_root),
    )
    if summary.get("passed") is not True:
        raise RuntimeError("production is blocked by the timestep guard")
    return summary


def _trajectory_contract(
    output_root: Path,
    case: protocol.CaseDefinition,
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    source_summary = verify_fine_source(
        output_root, load_field=False, source_root=source_root
    )
    guard_summary = verify_timestep_guard(output_root, source_root=source_root)
    contract = _base_contract(f"trajectory_{case.case_id}", source_root)
    contract.update(
        {
            "case": case.to_dict(),
            "fine_source": {
                "summary_path": str(
                    (output_root / "fine-source" / "summary.json").resolve()
                ),
                "summary_sha256": sha256_path(
                    output_root / "fine-source" / "summary.json"
                ),
                "field_path": source_summary["checkpoint"]["field_path"],
                "field_sha256": source_summary["checkpoint"]["field_sha256"],
                "field_fingerprint": source_summary["checkpoint"][
                    "field_fingerprint"
                ],
            },
            "timestep_guard": {
                "summary_path": str(
                    (output_root / "timestep-guard" / "summary.json").resolve()
                ),
                "summary_sha256": sha256_path(
                    output_root / "timestep-guard" / "summary.json"
                ),
                "classification": guard_summary["classification"],
            },
        }
    )
    contract["contract_payload_sha256"] = _canonical_hash(
        {key: value for key, value in contract.items() if key != "contract_payload_sha256"}
    )
    return contract


def _trajectory_record(
    field: np.ndarray,
    solver: RoyPseudospectralSolver,
    *,
    step: int,
    initial_mass: float,
    include_energy: bool,
    previous_energy: float | None,
    include_profiles: bool,
) -> dict[str, Any]:
    scalars = _field_scalars(field)
    mass = scalars["mass"]
    drift = (
        abs(float(mass) - initial_mass) / max(abs(initial_mass), np.finfo(float).tiny)
        if scalars["finite"] and mass is not None
        else None
    )
    energy = solver.free_energy(field) if include_energy and scalars["finite"] else None
    rebound = (
        max(0.0, float(energy) - float(previous_energy))
        / max(abs(float(previous_energy)), np.finfo(float).tiny)
        if energy is not None and previous_energy is not None
        else 0.0
    )
    pinches = (
        instantaneous_pinches(
            field,
            geometry="crossed",
            spacing=protocol.FINE_SPACING,
            radius=protocol.RADIUS,
            width=protocol.WIDTH,
        )
        if scalars["finite"]
        else None
    )
    return {
        "proposal": step,
        "step": step,
        "elapsed_physical_time": float(step),
        "field": scalars,
        "relative_mass_drift": drift,
        "free_energy": energy,
        "relative_energy_rebound": rebound,
        "contact": contact_metrics(field, _definition())
        if scalars["finite"]
        else None,
        "pinches": pinches,
        "profiles": _radius_profiles(field)
        if scalars["finite"] and include_profiles
        else None,
    }


def _health_reason(record: dict[str, Any]) -> str | None:
    field = record["field"]
    if not field["finite"]:
        return "nonfinite_field"
    if max(abs(float(field["minimum"])), abs(float(field["maximum"]))) > (
        protocol.FIELD_MAGNITUDE_LIMIT
    ):
        return "field_magnitude_limit"
    if float(record["relative_mass_drift"]) > protocol.MASS_DRIFT_LIMIT:
        return "mass_drift_limit"
    if float(record["relative_energy_rebound"]) > protocol.ENERGY_REBOUND_LIMIT:
        return "energy_rebound_limit"
    return None


def _event_from_records(
    records: list[dict[str, Any]], event: dict[str, Any] | None = None
) -> dict[str, Any]:
    return persistent_single_arm_event(
        [
            {"step": int(record["step"]), "pinches": record["pinches"]}
            for record in records
            if record.get("pinches") is not None
        ],
        required_records=protocol.PERSISTENCE_RECORDS,
        diagnostic_interval=protocol.DIAGNOSTIC_INTERVAL,
        maximum_gap_displacement=protocol.WIDTH,
        latched_event=event,
    )


def _trajectory_should_advance(
    current_step: int, event: dict[str, Any]
) -> bool:
    """Never advance a retained terminal event or fixed-horizon state."""

    return (
        current_step < protocol.HORIZON
        and event.get("detected") is not True
    )


def _resource_report(output_root: Path) -> dict[str, Any]:
    path = output_root / "preflight" / "summary.json"
    report = _load_json(path)
    source_root = Path(str(report.get("receipt_core", {}).get("source_root", "")))
    return freeze.verify_preflight_binding(
        output_root,
        protocol_id=protocol.PROTOCOL_ID,
        source_root=source_root,
    )


def _remaining_write_count(
    current_step: int,
    checkpoints: list[dict[str, Any]],
) -> int:
    existing_steps = {int(item["step"]) for item in checkpoints}
    scheduled = sum(
        step > current_step and step not in existing_steps
        for step in protocol.CHECKPOINT_STEPS
    )
    return scheduled + 1  # event, horizon, or interruption terminal write


def _runtime_disk_guard(
    output_root: Path,
    *,
    current_step: int,
    checkpoints: list[dict[str, Any]],
) -> dict[str, Any]:
    preflight = _resource_report(output_root)
    free = int(shutil.disk_usage(output_root).free)
    processes = freeze.process_inventory()
    remaining_writes = _remaining_write_count(current_step, checkpoints)
    # Keep the campaign-wide upper bound at every write. This deliberately
    # includes the not-yet-started second branch during the first branch.
    required = (
        protocol.WORST_CASE_PERSISTENT_BYTES
        + protocol.LARGEST_ATOMIC_WRITE_BYTES
        + protocol.UNTOUCHED_RESERVE_BYTES
        + int(preflight["disk"]["filesystem_volatility_allowance_bytes"])
    )
    return {
        "step": current_step,
        "free_disk_bytes": free,
        "required_disk_bytes": required,
        "remaining_margin_bytes": free - required,
        "remaining_field_writes": remaining_writes,
        "worst_case_field_persistent_bytes": (
            protocol.WORST_CASE_FIELD_PERSISTENT_BYTES
        ),
        "planned_small_output_allowance_bytes": (
            protocol.PLANNED_SMALL_OUTPUT_ALLOWANCE_BYTES
        ),
        "worst_case_persistent_bytes": protocol.WORST_CASE_PERSISTENT_BYTES,
        "filesystem_volatility_allowance_bytes": int(
            preflight["disk"]["filesystem_volatility_allowance_bytes"]
        ),
        "process_inventory": processes,
        "passed": bool(
            free > required
            and not processes["science"]
            and not processes["git"]
        ),
        "sampled_unix_time": time.time(),
    }


def _heartbeat(
    output: Path,
    *,
    case: protocol.CaseDefinition,
    current_step: int,
    initial_mass: float,
    records: list[dict[str, Any]],
    event: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    source_checkpoint: dict[str, Any],
    disk_sample: dict[str, Any],
    status: str = "running",
) -> None:
    resume_checkpoint = checkpoints[-1] if checkpoints else source_checkpoint
    atomic_json(
        output / "run_status.json",
        {
            "status": status,
            "case_id": case.case_id,
            "current_proposal": current_step,
            "elapsed_physical_time": float(current_step),
            "initial_mass": initial_mass,
            "records": records,
            "event_assessment": event,
            "checkpoints": checkpoints,
            "resume_checkpoint": resume_checkpoint,
            "resume_proposal": int(resume_checkpoint["step"]),
            "latest_disk_sample": disk_sample,
            "pid": os.getpid(),
            "signal": STOP_REQUESTED["signal"],
            "last_update_unix_time": time.time(),
        },
    )


def verify_trajectory(
    output_root: Path,
    case_id: str,
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    case = protocol.case_for(case_id)
    output = output_root / f"case-{case.case_id}"
    summary = _load_json(output / "summary.json")
    _verify_contract(
        summary["contract"],
        _trajectory_contract(output_root, case, source_root=source_root),
    )
    checks = {
        "terminal": summary.get("status") in {
            "completed_event",
            "completed_horizon",
        },
        "health": summary.get("health_passed") is True,
        "case": summary.get("case_id") == case.case_id,
    }
    _validate_record_sequence(list(summary.get("records", [])))
    _validate_recorded_checkpoints(
        output,
        list(summary.get("checkpoints", [])),
        require_complete_inventory=True,
    )
    checks.update(_terminal_status_checks(summary))
    if not all(checks.values()):
        raise RuntimeError(f"trajectory verification failed: {checks}")
    return summary


def _run_trajectory_with_lock_held(
    output_root: Path,
    case: protocol.CaseDefinition,
    *,
    resume: bool,
    source_root: Path | None = None,
    output_was_created: bool,
) -> dict[str, Any]:
    _verify_stage_binding(output_root, source_root)
    _assert_process_guard()
    output = output_root / f"case-{case.case_id}"
    if (output / "summary.json").is_file():
        return verify_trajectory(
            output_root, case.case_id, source_root=source_root
        )
    _resource_report(output_root)
    fine_source = verify_fine_source(
        output_root, load_field=False, source_root=source_root
    )
    verify_timestep_guard(output_root, source_root=source_root)
    contract = _trajectory_contract(
        output_root, case, source_root=source_root
    )
    source_checkpoint = fine_source["checkpoint"]

    initialize_from_source = False
    if output_was_created:
        _recover_or_verify_contract(
            output,
            contract,
            resume=resume,
            output_was_created=True,
            stage_label="trajectory",
        )
        initialize_from_source = True
        artifact_inventory = _validate_checkpoint_inventory(
            output, stage="trajectory"
        )
    else:
        _recover_or_verify_contract(
            output,
            contract,
            resume=resume,
            output_was_created=False,
            stage_label="trajectory",
        )
        artifact_inventory = _validate_checkpoint_inventory(
            output, stage="trajectory"
        )
        promoted = _promote_terminal_run_status(
            output_root,
            output,
            case,
            contract,
            source_root,
        )
        if promoted is not None:
            return promoted
        initialize_from_source = not (output / "run_status.json").is_file()

    if initialize_from_source:
        if len(artifact_inventory["artifacts"]) > 1:
            raise RuntimeError(
                "more than one unjournaled checkpoint artifact is present"
            )
        field = load_checkpoint(source_checkpoint)
        current_step = 0
        initial_mass = float(_field_scalars(field)["mass"])
        solver = _solver(case)
        initial_energy = solver.free_energy(field)
        records = [
            _trajectory_record(
                field,
                solver,
                step=0,
                initial_mass=initial_mass,
                include_energy=True,
                previous_energy=None,
                include_profiles=True,
            )
        ]
        previous_energy = initial_energy
        event = _event_from_records(records)
        checkpoints: list[dict[str, Any]] = []
        pending_at_resume = next(
            (
                row
                for row in artifact_inventory["artifacts"]
                if int(row["step"]) == current_step
            ),
            None,
        )
    else:
        status = _load_json(output / "run_status.json")
        if status.get("status") not in {
            "running",
            "interrupted",
            "resource_guard_failed",
        }:
            raise RuntimeError("trajectory status is not resumable")
        recorded_checkpoints = list(status.get("checkpoints", []))
        _validate_recorded_checkpoints(
            output,
            recorded_checkpoints,
            require_complete_inventory=False,
        )
        checkpoint = _validated_resume_checkpoint(
            status,
            source_checkpoint,
            recorded_checkpoints,
        )
        _validate_record_sequence(list(status["records"]))
        field = load_checkpoint(checkpoint)
        current_step = int(checkpoint["step"])
        initial_mass = float(status["initial_mass"])
        records = [
            row for row in status["records"] if int(row["step"]) <= current_step
        ]
        checkpoints = [
            row
            for row in recorded_checkpoints
            if int(row["step"]) <= current_step
        ]
        recorded_identities = {
            _checkpoint_identity(row) for row in recorded_checkpoints
        }
        inventory = _checkpoint_artifact_inventory(output)
        pending_orphans = [
            (str(row["stream"]), int(row["step"]))
            for row in inventory["artifacts"]
            if (str(row["stream"]), int(row["step"]))
            not in recorded_identities
        ]
        if len(pending_orphans) > 1:
            raise RuntimeError(
                "more than one unjournaled checkpoint artifact is present: "
                f"{pending_orphans}"
            )
        unreachable_orphans = [
            (str(row["stream"]), int(row["step"]))
            for row in inventory["artifacts"]
            if (str(row["stream"]), int(row["step"]))
            in pending_orphans
            and int(row["step"]) < current_step
        ]
        if unreachable_orphans:
            raise RuntimeError(
                "unrecorded checkpoint artifacts are behind the resume point: "
                f"{unreachable_orphans}"
            )
        event = _event_from_records(records)
        previous_energies = [
            row["free_energy"]
            for row in records
            if row.get("free_energy") is not None
        ]
        previous_energy = float(previous_energies[-1])
        solver = _solver(case)
        pending_at_resume = next(
            (
                row
                for row in inventory["artifacts"]
                if (str(row["stream"]), int(row["step"]))
                in pending_orphans
                and int(row["step"]) == current_step
            ),
            None,
        )

    _adopt_pending_checkpoint_at_current(
        output,
        field,
        current_step=current_step,
        event=event,
        checkpoints=checkpoints,
        artifact=pending_at_resume,
    )

    initial_disk = _runtime_disk_guard(
        output_root, current_step=current_step, checkpoints=checkpoints
    )
    if not initial_disk["passed"]:
        raise RuntimeError(f"trajectory disk guard failed: {initial_disk}")
    _heartbeat(
        output,
        case=case,
        current_step=current_step,
        initial_mass=initial_mass,
        records=records,
        event=event,
        checkpoints=checkpoints,
        source_checkpoint=source_checkpoint,
        disk_sample=initial_disk,
    )

    started = time.perf_counter()
    step_times: list[float] = []
    health_reason: str | None = None
    resource_failure: dict[str, Any] | None = None
    try:
        while _trajectory_should_advance(current_step, event):
            if STOP_REQUESTED["signal"] is not None:
                break
            tick = time.perf_counter()
            field = solver.propose_step(field)
            step_times.append(time.perf_counter() - tick)
            current_step += 1
            if current_step % protocol.DIAGNOSTIC_INTERVAL == 0:
                include_energy = current_step % protocol.ENERGY_INTERVAL == 0
                record = _trajectory_record(
                    field,
                    solver,
                    step=current_step,
                    initial_mass=initial_mass,
                    include_energy=include_energy,
                    previous_energy=previous_energy,
                    include_profiles=current_step % protocol.PROFILE_INTERVAL == 0,
                )
                records.append(record)
                if record["free_energy"] is not None:
                    previous_energy = float(record["free_energy"])
                health_reason = _health_reason(record)
                if health_reason is not None:
                    break
                event = _event_from_records(records, event)

            needs_regular = current_step in protocol.CHECKPOINT_STEPS
            event_detected = event.get("detected") is True
            if needs_regular or event_detected:
                disk_sample = _runtime_disk_guard(
                    output_root,
                    current_step=current_step,
                    checkpoints=checkpoints,
                )
                if not disk_sample["passed"]:
                    resource_failure = disk_sample
                    break
                checkpoint = _write_or_adopt_checkpoint(
                    output,
                    field,
                    step=current_step,
                    kinds=("event_confirmation",)
                    if event_detected
                    else ("regular",),
                    stream="event" if event_detected else "regular",
                )
                checkpoint["elapsed_physical_time"] = float(current_step)
                checkpoints.append(checkpoint)
                if event_detected:
                    break

            interrupted_paths = checkpoint_paths(
                output, current_step, stream="interrupted"
            )
            has_orphan_interruption = any(path.exists() for path in interrupted_paths)
            if has_orphan_interruption and not any(
                row.get("stream") == "interrupted"
                and int(row["step"]) == current_step
                for row in checkpoints
            ):
                disk_sample = _runtime_disk_guard(
                    output_root,
                    current_step=current_step,
                    checkpoints=checkpoints,
                )
                if not disk_sample["passed"]:
                    resource_failure = disk_sample
                    break
                checkpoint = _write_or_adopt_checkpoint(
                    output,
                    field,
                    step=current_step,
                    kinds=CHECKPOINT_KINDS_BY_STREAM["interrupted"],
                    stream="interrupted",
                )
                checkpoints.append(checkpoint)
                _heartbeat(
                    output,
                    case=case,
                    current_step=current_step,
                    initial_mass=initial_mass,
                    records=records,
                    event=event,
                    checkpoints=checkpoints,
                    source_checkpoint=source_checkpoint,
                    disk_sample=disk_sample,
                )

            if current_step % protocol.ENERGY_INTERVAL == 0:
                disk_sample = _runtime_disk_guard(
                    output_root,
                    current_step=current_step,
                    checkpoints=checkpoints,
                )
                if not disk_sample["passed"]:
                    resource_failure = disk_sample
                    break
                _heartbeat(
                    output,
                    case=case,
                    current_step=current_step,
                    initial_mass=initial_mass,
                    records=records,
                    event=event,
                    checkpoints=checkpoints,
                    source_checkpoint=source_checkpoint,
                    disk_sample=disk_sample,
                )

        if resource_failure is not None:
            _heartbeat(
                output,
                case=case,
                current_step=current_step,
                initial_mass=initial_mass,
                records=records,
                event=event,
                checkpoints=checkpoints,
                source_checkpoint=source_checkpoint,
                disk_sample=resource_failure,
                status="resource_guard_failed",
            )
            raise RuntimeError(
                "runtime disk guard failed; latest prior checkpoint preserved"
            )

        if (
            STOP_REQUESTED["signal"] is not None
            and event.get("detected") is not True
            and current_step < protocol.HORIZON
        ):
            recorded_identities = {
                _checkpoint_identity(row) for row in checkpoints
            }
            pending_artifacts = [
                row
                for row in _checkpoint_artifact_inventory(output)["artifacts"]
                if (str(row["stream"]), int(row["step"]))
                not in recorded_identities
            ]
            matching = [
                item for item in checkpoints if int(item["step"]) == current_step
            ]
            if matching:
                resume_checkpoint = matching[-1]
            elif pending_artifacts:
                # Preserve the single pre-existing crash artifact for replay;
                # writing another interruption field would exceed the frozen
                # one-pending-artifact and storage bounds.
                resume_checkpoint = (
                    checkpoints[-1] if checkpoints else source_checkpoint
                )
            elif not any(
                "interrupted" in item.get("kinds", ()) for item in checkpoints
            ):
                disk_sample = _runtime_disk_guard(
                    output_root,
                    current_step=current_step,
                    checkpoints=checkpoints,
                )
                if disk_sample["passed"]:
                    resume_checkpoint = _write_or_adopt_checkpoint(
                        output,
                        field,
                        step=current_step,
                        kinds=("interrupted",),
                        stream="interrupted",
                    )
                    resume_checkpoint["elapsed_physical_time"] = float(current_step)
                    checkpoints.append(resume_checkpoint)
                else:
                    resume_checkpoint = checkpoints[-1] if checkpoints else source_checkpoint
            else:
                # Bound restart growth: a second interruption reuses the latest
                # already-valid checkpoint instead of retaining another field.
                resume_checkpoint = checkpoints[-1] if checkpoints else source_checkpoint
            status = {
                "status": "interrupted",
                "case_id": case.case_id,
                "current_proposal": current_step,
                "elapsed_physical_time": float(current_step),
                "initial_mass": initial_mass,
                "records": records,
                "event_assessment": event,
                "checkpoints": checkpoints,
                "resume_checkpoint": resume_checkpoint,
                "resume_proposal": int(resume_checkpoint["step"]),
                "signal": STOP_REQUESTED["signal"],
                "contract": contract,
            }
            atomic_json(output / "run_status.json", status)
            return status

        if health_reason is not None:
            terminal = "numerically_failed"
            stop_reason = health_reason
        elif event.get("detected") is True:
            terminal = "completed_event"
            stop_reason = "persistent_single_arm_event"
        else:
            terminal = "completed_horizon"
            stop_reason = "fixed_elapsed_horizon"
            if not any(int(item["step"]) == current_step for item in checkpoints):
                disk_sample = _runtime_disk_guard(
                    output_root,
                    current_step=current_step,
                    checkpoints=checkpoints,
                )
                if not disk_sample["passed"]:
                    raise RuntimeError(
                        "disk guard failed before fixed-horizon checkpoint"
                    )
                checkpoint = _write_or_adopt_checkpoint(
                    output,
                    field,
                    step=current_step,
                    kinds=("fixed_horizon",),
                    stream="horizon",
                )
                checkpoint["elapsed_physical_time"] = float(current_step)
                checkpoints.append(checkpoint)
        _validate_recorded_checkpoints(
            output,
            checkpoints,
            require_complete_inventory=True,
        )
        summary = {
            "schema_version": protocol.SCHEMA_VERSION,
            "status": terminal,
            "classification": f"h0p25_{case.case_id}_{terminal}",
            "case_id": case.case_id,
            "stop_reason": stop_reason,
            "current_proposal": current_step,
            "elapsed_physical_time": float(current_step),
            "initial_mass": initial_mass,
            "records": records,
            "event_assessment": event,
            "checkpoints": checkpoints,
            "health_passed": terminal != "numerically_failed",
            "execution": {
                "elapsed_wall_seconds": time.perf_counter() - started,
                "proposal_count_this_session": len(step_times),
                "median_proposal_wall_seconds": statistics.median(step_times)
                if step_times
                else None,
            },
            "contract": contract,
        }
        atomic_json(output / "summary.json", summary)
        atomic_json(output / "run_status.json", summary)
        if terminal == "numerically_failed":
            raise RuntimeError(f"trajectory numerical failure: {stop_reason}")
        return summary
    finally:
        gc.collect()


def run_trajectory(
    output_root: Path,
    case: protocol.CaseDefinition,
    *,
    resume: bool,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Run or reconcile one branch with the lock held for all mutable state."""

    _verify_stage_binding(output_root, source_root)
    _assert_process_guard()
    output = output_root / f"case-{case.case_id}"
    output_was_created = False
    if not output.exists():
        reserve_output_directory(output)
        output_was_created = True
    lock = acquire_output_lock(output)
    try:
        return _run_trajectory_with_lock_held(
            output_root,
            case,
            resume=resume,
            source_root=source_root,
            output_was_created=output_was_created,
        )
    finally:
        release_output_lock(lock)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("source", "guard", "untreated", "c34")
    )
    parser.add_argument("--output", type=Path, default=protocol.DEFAULT_OUTPUT)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=protocol.EXTERNAL_SOURCE_DIRECTORY,
        help="directory containing the exact hash-bound t=120 source files",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    output = args.output.resolve()
    if args.stage == "source":
        result = prepare_fine_source(
            output, resume=args.resume, source_root=args.source_root
        )
    elif args.stage == "guard":
        result = run_timestep_guard(
            output, resume=args.resume, source_root=args.source_root
        )
    else:
        result = run_trajectory(
            output,
            protocol.case_for(args.stage),
            resume=args.resume,
            source_root=args.source_root,
        )
    print(json.dumps({"stage": args.stage, "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    for handled_signal in (signal.SIGINT, signal.SIGTERM):
        signal.signal(handled_signal, _signal_handler)
    raise SystemExit(main())
