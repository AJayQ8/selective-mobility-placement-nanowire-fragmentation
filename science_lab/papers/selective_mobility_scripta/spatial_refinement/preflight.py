#!/usr/bin/env python3
"""Zero-evolution provenance, RAM, disk, and temporary-Git launch gate."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import psutil

from science_lab.papers.nanowire_gb_junction.roy_fixed_gb_bridge.provenance import (
    sha256_path,
)
from science_lab.papers.nanowire_gb_junction.roy_gb_junction_sentinel.storage import (
    atomic_json,
)

from . import freeze, protocol, runner


PACKAGE_ROOT = Path(__file__).resolve().parent
LEGACY_PATHS = {
    "roy_model": PACKAGE_ROOT.parents[1]
    / "nanowire_gb_junction/roy_2021_reproduction/model.py",
    "selective_model": PACKAGE_ROOT.parents[1]
    / "nanowire_gb_junction/roy_selective_mobility_protection/model.py",
    "geometry": PACKAGE_ROOT.parents[1]
    / "nanowire_gb_junction/roy_selective_mobility_protection/geometry.py",
    "event_diagnostics": PACKAGE_ROOT.parents[1]
    / "nanowire_gb_junction/roy_gb_junction_sentinel/diagnostics.py",
    "storage": PACKAGE_ROOT.parents[1]
    / "nanowire_gb_junction/roy_gb_junction_sentinel/storage.py",
    "contact_helper": PACKAGE_ROOT.parents[1]
    / "nanowire_gb_junction/roy_2021_reproduction/run_preflight.py",
}
SNAPSHOT_BASENAME = re.compile(r"^tmp\.[A-Za-z0-9]+$")
ALLOWED_OBJECT_CHILD = re.compile(r"^[0-9a-f]{2}$")


def _physical_memory_bytes() -> int | None:
    try:
        completed = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(completed.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        try:
            return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, TypeError):
            return None


def _available_memory_bytes() -> int | None:
    try:
        return int(psutil.virtual_memory().available)
    except (AttributeError, OSError, ValueError):
        return None


def _darwin_temp_root() -> Path | None:
    try:
        completed = subprocess.run(
            ["/usr/bin/getconf", "DARWIN_USER_TEMP_DIR"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    raw = completed.stdout.strip()
    return Path(raw).resolve() if raw else None


def _logical_tree_bytes(root: Path) -> int:
    total = int(root.lstat().st_size)
    for directory, directories, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in directories:
            total += int((base / name).lstat().st_size)
        for name in files:
            total += int((base / name).lstat().st_size)
    return total


def _contains_symlink(root: Path) -> bool:
    for directory, directories, files in os.walk(root, followlinks=False):
        base = Path(directory)
        if any((base / name).is_symlink() for name in [*directories, *files]):
            return True
    return False


def audit_temporary_git_snapshots(
    temp_root: Path | None = None,
    *,
    expected_uid: int | None = None,
) -> dict[str, Any]:
    """Read-only reproduction of the guarded temporary-Git fingerprint."""

    root = (temp_root or _darwin_temp_root())
    uid = os.getuid() if expected_uid is None else int(expected_uid)
    if root is None or not root.is_dir():
        return {
            "audit_valid": False,
            "reason": "darwin_user_temp_directory_missing",
            "temp_root": None if root is None else str(root),
            "validated": [],
            "uncertain": [],
            "logical_bytes": 0,
        }
    canonical = root.resolve()
    root_pattern_ok = bool(
        str(canonical).startswith("/private/var/folders/")
        and canonical.name == "T"
    ) or temp_root is not None
    try:
        root_owner_ok = canonical.stat().st_uid == uid
    except OSError:
        root_owner_ok = False
    validated: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    for candidate in sorted(canonical.glob("tmp.*")):
        if not candidate.is_dir() or candidate.is_symlink():
            continue
        if not SNAPSHOT_BASENAME.fullmatch(candidate.name):
            continue
        try:
            candidate_stat = candidate.stat()
            top = sorted(child.name for child in candidate.iterdir())
        except OSError as error:
            uncertain.append({"path": str(candidate), "reason": str(error)})
            continue
        if top != ["index", "index.lock", "objects"]:
            continue
        reasons: list[str] = []
        if candidate.resolve().parent != canonical:
            reasons.append("escaped_canonical_temp_root")
        if candidate_stat.st_uid != uid:
            reasons.append("foreign_owner")
        if stat.S_IMODE(candidate_stat.st_mode) != 0o700:
            reasons.append("unexpected_mode")
        index = candidate / "index"
        lock = candidate / "index.lock"
        objects = candidate / "objects"
        if not index.is_file() or index.is_symlink():
            reasons.append("invalid_index")
        if not lock.is_file() or lock.is_symlink() or lock.stat().st_size != 0:
            reasons.append("invalid_index_lock")
        if not objects.is_dir() or objects.is_symlink():
            reasons.append("invalid_objects")
        if not reasons:
            try:
                with index.open("rb") as handle:
                    if handle.read(4) != b"DIRC":
                        reasons.append("invalid_index_header")
                if _contains_symlink(candidate):
                    reasons.append("symlink_inside_candidate")
                for directory, directories, files in os.walk(
                    candidate, followlinks=False
                ):
                    base = Path(directory)
                    for name in [*directories, *files]:
                        if (base / name).lstat().st_uid != uid:
                            reasons.append("foreign_owned_entry")
                            break
                for child in objects.iterdir():
                    if child.name not in {"info", "pack"} and not (
                        ALLOWED_OBJECT_CHILD.fullmatch(child.name)
                    ):
                        reasons.append("unexpected_git_object_entry")
                        break
            except OSError as error:
                reasons.append(f"audit_error:{error}")
        if reasons:
            uncertain.append({"path": str(candidate), "reasons": sorted(set(reasons))})
        else:
            validated.append(
                {
                    "path": str(candidate),
                    "logical_bytes": _logical_tree_bytes(candidate),
                }
            )
    return {
        "audit_valid": bool(root_pattern_ok and root_owner_ok and not uncertain),
        "temp_root": str(canonical),
        "root_pattern_valid": root_pattern_ok,
        "root_owned_by_expected_uid": root_owner_ok,
        "validated": validated,
        "validated_count": len(validated),
        "uncertain": uncertain,
        "logical_bytes": sum(item["logical_bytes"] for item in validated),
        "cleanup_threshold_bytes": protocol.TEMPORARY_GIT_SNAPSHOT_CLEANUP_THRESHOLD_BYTES,
    }


def _competing_science_processes() -> list[dict[str, Any]]:
    return freeze.process_inventory()["science"]


def _competing_git_processes() -> list[dict[str, Any]]:
    return freeze.process_inventory()["git"]


def _legacy_hash_report() -> dict[str, Any]:
    measured = {
        name: sha256_path(path) if path.is_file() else None
        for name, path in LEGACY_PATHS.items()
    }
    checks = {
        name: measured[name] == expected
        for name, expected in protocol.LEGACY_MODULE_SHA256.items()
    }
    return {
        "paths": {name: str(path) for name, path in LEGACY_PATHS.items()},
        "measured_sha256": measured,
        "expected_sha256": dict(protocol.LEGACY_MODULE_SHA256),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _coarse_reference_hash_report(source_root: Path | None) -> dict[str, Any]:
    """Verify the historical case summaries when the adjacent archive exists."""

    references = protocol.coarse_reference_paths(source_root)
    exists = {
        case_id: path.is_file()
        for case_id, path in references.summaries.items()
    }
    any_available = any(exists.values())
    all_available = all(exists.values())
    measured = {
        case_id: sha256_path(path) if exists[case_id] else None
        for case_id, path in references.summaries.items()
    }
    checks = {
        case_id: measured[case_id] == expected
        for case_id, expected in protocol.COARSE_REFERENCE_SUMMARY_SHA256.items()
    }
    # A relocatable source-only archive remains usable. A partially present or
    # altered coarse archive does not: that would silently weaken the frozen
    # evidence binding on a host that claims to carry the historical results.
    passed = (not any_available) or (all_available and all(checks.values()))
    return {
        "root": str(references.root),
        "paths": {
            case_id: str(path)
            for case_id, path in references.summaries.items()
        },
        "available": all_available,
        "partially_available": any_available and not all_available,
        "measured_sha256": measured,
        "expected_sha256": dict(protocol.COARSE_REFERENCE_SUMMARY_SHA256),
        "checks": checks,
        "passed": passed,
    }


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def launch_receipt(report: dict[str, Any]) -> str:
    disk = report["disk"]
    ram = report["ram"]
    runtime = report["runtime"]
    return "\n".join(
        (
            f"LONG-RUN LAUNCH RECEIPT: {report['decision']}",
            "runtime and expected finish window: "
            f"{runtime['expected_hours'][0]:.1f}-{runtime['expected_hours'][1]:.1f} "
            "hours after authorized launch",
            "free disk / required disk / remaining margin: "
            f"{disk['free_disk_bytes']} / {disk['required_disk_bytes']} / "
            f"{disk['remaining_margin_bytes']} bytes",
            "worst-case new output and largest atomic write: "
            f"{disk['worst_case_remaining_persistent_bytes']} / "
            f"{disk['largest_atomic_write_bytes']} bytes",
            "planned small-output allowance within persistent output: "
            f"{disk['planned_small_output_allowance_bytes']} bytes",
            "untouched reserve and filesystem-volatility allowance: "
            f"{disk['untouched_reserve_bytes']} / "
            f"{disk['filesystem_volatility_allowance_bytes']} bytes",
            "peak RAM estimate / physical RAM: "
            f"{ram['peak_estimate_bytes']} / {ram['physical_memory_bytes']} bytes",
            "peak plus margin / currently available RAM: "
            f"{ram['peak_plus_margin_bytes']} / "
            f"{ram['available_memory_bytes']} bytes",
            "checkpoint and exact-resume plan: shared verified fine source; "
            "per-case checkpoints at 800/1600 and event, horizon, or signal; "
            "contract/hash equality required on resume",
            "watcher, completion analysis, and notification plan: sequential "
            "supervisor heartbeat, frozen analyzer, terminal/macOS notification",
            "stop behavior if a guard fails: preserve latest valid checkpoint, "
            "stop the queue, and make no adaptive change; the supervisor hard "
            f"cap is {runtime['wall_clock_cap_seconds']} seconds",
            f"receipt core SHA-256: {report['receipt_core_sha256']}",
            "",
        )
    )


def run_preflight(
    output_root: Path,
    *,
    drift_window_seconds: float = protocol.DEFAULT_DRIFT_WINDOW_SECONDS,
    drift_sample_seconds: float = protocol.DEFAULT_DRIFT_SAMPLE_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
    disk_free_fn: Callable[[Path], int] | None = None,
    temp_root: Path | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Perform no solver proposal and emit a complete GO/NO-GO receipt."""

    if drift_window_seconds < 0.0 or drift_sample_seconds <= 0.0:
        raise ValueError("invalid disk-drift sampling interval")
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight_output = output_root / "preflight"
    preflight_output.mkdir(exist_ok=True)
    disk_free = disk_free_fn or (lambda path: int(shutil.disk_usage(path).free))

    resolved_source = protocol.source_paths(source_root)
    source_error: str | None = None
    try:
        source = runner.verify_external_source(
            load_field=True, source_root=resolved_source.root
        )
        source.pop("field", None)
        source_passed = True
    except Exception as error:  # recorded as NO-GO, never relaxed
        source = None
        source_passed = False
        source_error = f"{type(error).__name__}: {error}"
    legacy = _legacy_hash_report()
    coarse_references = _coarse_reference_hash_report(resolved_source.root)
    manifest = freeze.runtime_manifest()
    git_state = freeze.git_freeze_state()
    snapshots_before = audit_temporary_git_snapshots(temp_root)
    competing = _competing_science_processes()
    competing_git = _competing_git_processes()
    physical_memory = _physical_memory_bytes()
    available_memory = _available_memory_bytes()

    samples: list[dict[str, Any]] = []
    started = time.monotonic()
    samples.append({"elapsed_seconds": 0.0, "free_bytes": disk_free(output_root)})
    while time.monotonic() - started < drift_window_seconds:
        remaining = drift_window_seconds - (time.monotonic() - started)
        sleep_fn(min(drift_sample_seconds, max(0.0, remaining)))
        elapsed = time.monotonic() - started
        samples.append({"elapsed_seconds": elapsed, "free_bytes": disk_free(output_root)})
        if elapsed >= drift_window_seconds:
            break
    snapshots_after = audit_temporary_git_snapshots(temp_root)

    loss_rates = []
    for first, second in zip(samples, samples[1:]):
        elapsed = float(second["elapsed_seconds"] - first["elapsed_seconds"])
        if elapsed > 0.0:
            loss_rates.append(
                max(0.0, float(first["free_bytes"] - second["free_bytes"]))
                / elapsed
            )
    unexplained_loss_rate = max(loss_rates, default=0.0)
    by_rate = int(
        2.0 * unexplained_loss_rate * protocol.GUARDED_RUNTIME_SECONDS
    )
    volatility = max(protocol.WORST_CASE_PERSISTENT_BYTES, by_rate)
    free = int(samples[-1]["free_bytes"])
    required = (
        protocol.WORST_CASE_PERSISTENT_BYTES
        + protocol.LARGEST_ATOMIC_WRITE_BYTES
        + protocol.UNTOUCHED_RESERVE_BYTES
        + volatility
    )
    snapshot_growth = int(
        snapshots_after.get("logical_bytes", 0)
        - snapshots_before.get("logical_bytes", 0)
    )
    ram_margin = (
        None
        if physical_memory is None
        else physical_memory - protocol.PEAK_RAM_ESTIMATE_BYTES
    )
    checks = {
        "immutable_source": source_passed,
        "legacy_module_hashes": legacy["passed"],
        "coarse_reference_hashes_if_available": coarse_references["passed"],
        "physical_lengths": tuple(protocol.fine_definition().lattice.physical_lengths)
        == protocol.PHYSICAL_LENGTHS,
        "ram_margin": bool(
            ram_margin is not None
            and ram_margin >= protocol.MINIMUM_PHYSICAL_RAM_MARGIN_BYTES
        ),
        "available_ram_margin": bool(
            available_memory is not None
            and available_memory
            >= protocol.PEAK_RAM_ESTIMATE_BYTES
            + protocol.MINIMUM_AVAILABLE_RAM_MARGIN_BYTES
        ),
        "disk_margin_positive": free > required,
        "temporary_git_snapshot_audit_valid": bool(
            snapshots_before.get("audit_valid")
            and snapshots_after.get("audit_valid")
        ),
        "temporary_git_snapshots_below_cleanup_threshold": int(
            snapshots_after.get("logical_bytes", 0)
        )
        < protocol.TEMPORARY_GIT_SNAPSHOT_CLEANUP_THRESHOLD_BYTES,
        "temporary_git_snapshot_population_not_growing": snapshot_growth <= 0,
        "no_competing_large_simulation": not competing,
        "no_competing_git_process": not competing_git,
        "runtime_manifest_complete": all(value is not None for value in manifest.values()),
        "frozen_paths_tracked_clean_and_remote_head": git_state["passed"] is True,
        "restart_source_exists": source_passed,
        "fixed_checkpoint_plan": protocol.CHECKPOINT_STEPS == (800, 1600),
        "no_adaptive_follow_on": protocol.protocol_payload()["scope"][
            "adaptive_follow_on_authorized"
        ]
        is False,
    }
    decision = "GO" if all(checks.values()) else "NO-GO"
    receipt_core = {
        "protocol_id": protocol.PROTOCOL_ID,
        "decision": decision,
        "checks": checks,
        "source_root": str(resolved_source.root),
        "source_sha256": None if source is None else source.get("sha256"),
        "runtime_manifest": manifest,
        "runtime_environment": freeze.runtime_environment(),
        "git": git_state,
        "legacy_measured_sha256": legacy.get("measured_sha256"),
        "coarse_references": coarse_references,
        "disk_contract": {
            "worst_case_field_persistent_bytes": (
                protocol.WORST_CASE_FIELD_PERSISTENT_BYTES
            ),
            "planned_small_output_allowance_bytes": (
                protocol.PLANNED_SMALL_OUTPUT_ALLOWANCE_BYTES
            ),
            "worst_case_remaining_persistent_bytes": protocol.WORST_CASE_PERSISTENT_BYTES,
            "largest_atomic_write_bytes": protocol.LARGEST_ATOMIC_WRITE_BYTES,
            "untouched_reserve_bytes": protocol.UNTOUCHED_RESERVE_BYTES,
            "filesystem_volatility_allowance_bytes": volatility,
            "free_disk_bytes": free,
            "required_disk_bytes": required,
            "remaining_margin_bytes": free - required,
        },
        "ram_contract": {
            "peak_estimate_bytes": protocol.PEAK_RAM_ESTIMATE_BYTES,
            "required_physical_margin_bytes": (
                protocol.MINIMUM_PHYSICAL_RAM_MARGIN_BYTES
            ),
            "required_available_margin_bytes": (
                protocol.MINIMUM_AVAILABLE_RAM_MARGIN_BYTES
            ),
            "physical_memory_bytes": physical_memory,
            "available_memory_bytes": available_memory,
        },
        "temporary_git_snapshot_logical_bytes": snapshots_after.get("logical_bytes"),
        "temporary_git_snapshot_growth_bytes": snapshot_growth,
        "competing_science_processes": competing,
        "competing_git_processes": competing_git,
        "wall_clock_cap_seconds": protocol.GUARDED_RUNTIME_SECONDS,
    }
    report = {
        "schema_version": protocol.SCHEMA_VERSION,
        "protocol_id": protocol.PROTOCOL_ID,
        "decision": decision,
        "receipt_core": receipt_core,
        "receipt_core_sha256": freeze.canonical_sha256(receipt_core),
        "zero_solver_proposals": True,
        "checks": checks,
        "source": source,
        "source_error": source_error,
        "legacy": legacy,
        "coarse_references": coarse_references,
        "ram": {
            "peak_estimate_bytes": protocol.PEAK_RAM_ESTIMATE_BYTES,
            "solver_peak_bytes_per_cell": protocol.SOLVER_PEAK_BYTES_PER_CELL,
            "solver_peak_estimate_bytes": protocol.SOLVER_PEAK_ESTIMATE_BYTES,
            "guard_additional_live_fields": [
                "common_source_outside_active_proposal",
                "retained_dt1_endpoint",
                "treated_spatial_mobility_factor",
            ],
            "guard_additional_live_field_bytes": (
                3 * protocol.FIELD_RAW_BYTES
            ),
            "physical_memory_bytes": physical_memory,
            "available_memory_bytes": available_memory,
            "peak_plus_margin_bytes": (
                protocol.PEAK_RAM_ESTIMATE_BYTES
                + protocol.MINIMUM_AVAILABLE_RAM_MARGIN_BYTES
            ),
            "required_physical_margin_bytes": (
                protocol.MINIMUM_PHYSICAL_RAM_MARGIN_BYTES
            ),
            "required_available_margin_bytes": (
                protocol.MINIMUM_AVAILABLE_RAM_MARGIN_BYTES
            ),
            "remaining_margin_bytes": ram_margin,
        },
        "disk": {
            "output_filesystem_path": str(output_root),
            "filesystem_device": int(output_root.stat().st_dev),
            "samples": samples,
            "recent_unexplained_loss_rate_bytes_per_second": unexplained_loss_rate,
            "worst_case_field_persistent_bytes": (
                protocol.WORST_CASE_FIELD_PERSISTENT_BYTES
            ),
            "planned_small_output_allowance_bytes": (
                protocol.PLANNED_SMALL_OUTPUT_ALLOWANCE_BYTES
            ),
            "worst_case_remaining_persistent_bytes": protocol.WORST_CASE_PERSISTENT_BYTES,
            "largest_atomic_write_bytes": protocol.LARGEST_ATOMIC_WRITE_BYTES,
            "untouched_reserve_bytes": protocol.UNTOUCHED_RESERVE_BYTES,
            "volatility_from_rate_bytes": by_rate,
            "filesystem_volatility_allowance_bytes": volatility,
            "free_disk_bytes": free,
            "required_disk_bytes": required,
            "remaining_margin_bytes": free - required,
        },
        "temporary_git_snapshots": {
            "before": snapshots_before,
            "after": snapshots_after,
            "growth_bytes": snapshot_growth,
            "cleanup_performed": False,
        },
        "competing_science_processes": competing,
        "competing_git_processes": competing_git,
        "git_freeze": git_state,
        "runtime_manifest": manifest,
        "runtime": {
            "expected_hours": list(protocol.EXPECTED_RUNTIME_HOURS),
            "guarded_runtime_seconds": protocol.GUARDED_RUNTIME_SECONDS,
            "wall_clock_cap_seconds": protocol.GUARDED_RUNTIME_SECONDS,
        },
        "stop_behavior": "preserve_latest_valid_checkpoint_and_stop_without_adaptation",
        "generated_unix_time": time.time(),
    }
    atomic_json(preflight_output / "summary.json", report)
    receipt_text = launch_receipt(report)
    _atomic_text(preflight_output / "launch_receipt.txt", receipt_text)
    receipt_stem = (
        f"receipt-{time.time_ns()}-{report['receipt_core_sha256'][:12]}"
    )
    atomic_json(preflight_output / f"{receipt_stem}.json", report)
    _atomic_text(preflight_output / f"{receipt_stem}.txt", receipt_text)
    return report


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=protocol.DEFAULT_OUTPUT)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=protocol.EXTERNAL_SOURCE_DIRECTORY,
        help="directory containing the exact hash-bound t=120 source files",
    )
    parser.add_argument(
        "--drift-window-seconds",
        type=float,
        default=protocol.DEFAULT_DRIFT_WINDOW_SECONDS,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    report = run_preflight(
        args.output,
        drift_window_seconds=args.drift_window_seconds,
        source_root=args.source_root,
    )
    print(launch_receipt(report), end="")
    return 0 if report["decision"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
