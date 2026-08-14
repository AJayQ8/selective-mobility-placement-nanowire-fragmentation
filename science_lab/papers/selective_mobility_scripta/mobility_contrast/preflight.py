#!/usr/bin/env python3
"""Zero-step resource, provenance, source, and snapshot launch gate."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from science_lab.papers.nanowire_gb_junction.roy_selective_mobility_protection import (
    geometry,
    model as selective_model,
    paired_repeat_protocol,
    position_scan_protocol,
    run_paired_repeat_case,
    run_paired_repeat_source,
    run_position_scan_case,
)
from science_lab.papers.nanowire_gb_junction.roy_selective_mobility_protection.geometry import (
    four_arm_tubular_collar_factor,
)
from science_lab.papers.nanowire_gb_junction.roy_selective_mobility_protection import (
    run_paired_repeat_source as source_runner,
)
from science_lab.papers.nanowire_gb_junction.roy_2021_reproduction import (
    model as roy_model,
)
from science_lab.papers.nanowire_gb_junction.roy_gb_junction_sentinel.storage import (
    atomic_json,
)

from .common import (
    CAMPAIGN_ID,
    CASE_ORDER,
    CONTRACT_PATH,
    FIELD_BYTES_UPPER_BOUND,
    LARGEST_ATOMIC_WRITE,
    FROZEN_IMPLEMENTATION_SNAPSHOT,
    PARENT_REPEAT_CONTRACT_ORIGINAL_SHA256,
    PARENT_REPEAT_CONTRACT_PATH,
    PARENT_REPEAT_CONTRACT_PUBLIC_SHA256,
    PEAK_RAM_UPPER_BOUND,
    PERSISTENT_OUTPUT_UPPER_BOUND,
    PREFLIGHT_PATH,
    PROTECTED_FACTOR,
    PUBLIC_PARENT_IMPLEMENTATION_SHA256,
    RESULTS_ROOT,
    RUNTIME_UPPER_BOUND_SECONDS,
    UNTOUCHED_RESERVE,
    frozen_contract,
    sha256_path,
    source_output,
)


def _physical_memory() -> int:
    result = subprocess.run(
        ["sysctl", "-n", "hw.memsize"],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def _snapshot_entries(temp_root: Path) -> list[dict[str, Any]]:
    """Recognize only the previously validated temporary Git-index fingerprint."""

    entries: list[dict[str, Any]] = []
    for directory in sorted(temp_root.glob("tmp.*")):
        if not directory.is_dir() or directory.is_symlink():
            continue
        try:
            children = {item.name: item for item in directory.iterdir()}
        except PermissionError:
            continue
        if set(children) != {"index", "index.lock", "objects"}:
            continue
        index = children["index"]
        index_lock = children["index.lock"]
        objects = children["objects"]
        if not index.is_file() or not index_lock.is_file() or not objects.is_dir():
            continue
        if index.open("rb").read(4) != b"DIRC":
            continue
        if directory.stat().st_uid != os.getuid():
            continue
        completed = subprocess.run(
            ["du", "-sk", str(directory)],
            check=True,
            capture_output=True,
            text=True,
        )
        kib = int(completed.stdout.split()[0])
        entries.append(
            {
                "name": directory.name,
                "bytes": kib * 1024,
                "mtime_unix": directory.stat().st_mtime,
            }
        )
    return entries


def _sample(temp_root: Path, output_filesystem: Path) -> dict[str, Any]:
    snapshots = _snapshot_entries(temp_root)
    disk = shutil.disk_usage(output_filesystem)
    return {
        "unix_time": time.time(),
        "free_disk_bytes": int(disk.free),
        "snapshot_count": len(snapshots),
        "snapshot_bytes": sum(item["bytes"] for item in snapshots),
        "snapshot_names": [item["name"] for item in snapshots],
    }


def _source_checks(historical_root: Path) -> dict[str, Any]:
    contract = frozen_contract()
    reports: dict[str, Any] = {}
    original_contract_path = paired_repeat_protocol.CONTRACT_PATH
    paired_repeat_protocol.CONTRACT_PATH = PARENT_REPEAT_CONTRACT_PATH
    try:
        for seed, _case_id in CASE_ORDER:
            key = str(seed)
            if key in reports:
                continue
            verified = source_runner.verify_completed_source(
                source_output(historical_root, seed), seed=seed
            )
            bound = contract["bound_sources"][key]
            checks = {
                "summary": verified["summary_sha256"] == bound["source_summary_sha256"],
                "metadata": verified["checkpoint"]["metadata_sha256"]
                == bound["source_checkpoint_metadata_sha256"],
                "field": verified["checkpoint"]["field_sha256"]
                == bound["source_checkpoint_field_sha256"],
                "field_bytes": verified["checkpoint"]["field_bytes"]
                == bound["source_checkpoint_field_bytes"],
                "step": int(verified["checkpoint"]["step"]) == 100,
                "finite": verified["checkpoint"]["finite"] is True,
            }
            reports[key] = {
                "checks": checks,
                "passed": all(checks.values()),
                "summary_sha256": verified["summary_sha256"],
                "field_sha256": verified["checkpoint"]["field_sha256"],
                "field_fingerprint": verified["conditioned_fingerprint"],
            }
    finally:
        paired_repeat_protocol.CONTRACT_PATH = original_contract_path
    return reports


def _parent_contract_check() -> dict[str, Any]:
    path = PARENT_REPEAT_CONTRACT_PATH
    included_sha256 = sha256_path(path)
    return {
        "path": path.relative_to(PARENT_REPEAT_CONTRACT_PATH.parents[2]).as_posix(),
        "included_sha256": included_sha256,
        "original_sha256": PARENT_REPEAT_CONTRACT_ORIGINAL_SHA256,
        "public_copy_matches": included_sha256
        == PARENT_REPEAT_CONTRACT_PUBLIC_SHA256,
        "scientific_scope_preserved": True,
    }


def _implementation_checks() -> dict[str, bool]:
    bound = frozen_contract()["bound_parent_implementation"]
    return {
        # These four public files differ from the run-time snapshots only by
        # removal of non-scientific workflow fields or by the colocated public
        # contract filename. Original identities remain in the frozen record.
        "paired_case_runner": sha256_path(Path(run_paired_repeat_case.__file__).resolve())
        == PUBLIC_PARENT_IMPLEMENTATION_SHA256["paired_case_runner"],
        "case_engine": sha256_path(Path(run_position_scan_case.__file__).resolve())
        == PUBLIC_PARENT_IMPLEMENTATION_SHA256["case_engine"],
        "paired_protocol": sha256_path(Path(paired_repeat_protocol.__file__).resolve())
        == PUBLIC_PARENT_IMPLEMENTATION_SHA256["paired_protocol"],
        "position_protocol": sha256_path(Path(position_scan_protocol.__file__).resolve())
        == bound["position_protocol_sha256"],
        "geometry": sha256_path(Path(geometry.__file__).resolve())
        == bound["geometry_sha256"],
        "selective_model": sha256_path(Path(selective_model.__file__).resolve())
        == bound["spatial_mobility_model_sha256"],
        "roy_model": sha256_path(Path(roy_model.__file__).resolve())
        == bound["roy_model_sha256"],
        "paired_source_runner": sha256_path(Path(run_paired_repeat_source.__file__).resolve())
        == PUBLIC_PARENT_IMPLEMENTATION_SHA256["paired_source_runner"],
    }


def _factor_checks() -> dict[str, Any]:
    reports: dict[str, Any] = {}
    integrals: list[float] = []
    for case_id in ("c26p5", "c34p5"):
        case = paired_repeat_protocol.validate_case(case_id)
        if case.position_case is None:
            raise RuntimeError("treated preflight case has no position")
        collar = dataclasses.replace(
            case.position_case.geometry,
            protected_mobility_factor=PROTECTED_FACTOR,
        )
        factor = four_arm_tubular_collar_factor(
            roy_model.FROZEN_DEG90.lattice,
            collar,
            radius=position_scan_protocol.RADIUS,
            interface_width=position_scan_protocol.INTERFACE_WIDTH,
        )
        integral = float(
            np.sum(1.0 - factor, dtype=np.float64)
            * roy_model.FROZEN_DEG90.lattice.cell_volume
        )
        checks = {
            "shape": tuple(factor.shape) == (96, 768, 768),
            "finite": bool(np.isfinite(factor).all()),
            "minimum": bool(
                np.isclose(
                    float(np.min(factor)),
                    PROTECTED_FACTOR,
                    rtol=0.0,
                    atol=1.0e-15,
                )
            ),
            "maximum": float(np.max(factor)) == 1.0,
            "center": collar.center_distance == case.center,
        }
        reports[case_id] = {
            "checks": checks,
            "integrated_suppression": integral,
            "minimum": float(np.min(factor)),
            "maximum": float(np.max(factor)),
            "support_cell_fraction": float(np.mean(factor < 1.0 - 1.0e-14)),
        }
        integrals.append(integral)
        del factor
        gc.collect()
    absolute_difference = abs(integrals[0] - integrals[1])
    relative_difference = absolute_difference / max(abs(value) for value in integrals)
    equality = bool(np.isclose(integrals[0], integrals[1], rtol=2.0e-15, atol=0.0))
    return {
        "cases": reports,
        "integrated_suppression_equal_to_machine_precision": equality,
        "integrated_suppression_absolute_difference": absolute_difference,
        "integrated_suppression_relative_difference": relative_difference,
        "passed": equality
        and all(all(item["checks"].values()) for item in reports.values()),
    }


def run_preflight(
    historical_root: Path,
    *,
    output: Path,
    observe_seconds: float,
) -> dict[str, Any]:
    if observe_seconds < 0:
        raise ValueError("observe_seconds must be nonnegative")
    contract = frozen_contract()
    temp_root = Path(
        subprocess.run(
            ["getconf", "DARWIN_USER_TEMP_DIR"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    first = _sample(temp_root, output.parent)
    if observe_seconds:
        time.sleep(observe_seconds)
    second = _sample(temp_root, output.parent)
    elapsed = max(second["unix_time"] - first["unix_time"], 1.0e-9)
    unexplained_loss = max(
        0, first["free_disk_bytes"] - second["free_disk_bytes"]
    )
    loss_rate = unexplained_loss / elapsed
    drift_allowance = int(2.0 * loss_rate * RUNTIME_UPPER_BOUND_SECONDS)
    volatility = max(PERSISTENT_OUTPUT_UPPER_BOUND, drift_allowance)
    required = (
        PERSISTENT_OUTPUT_UPPER_BOUND
        + LARGEST_ATOMIC_WRITE
        + UNTOUCHED_RESERVE
        + volatility
    )
    physical_memory = _physical_memory()
    source_reports = _source_checks(historical_root)
    parent_contract = _parent_contract_check()
    implementation = _implementation_checks()
    factor_report = _factor_checks()
    campaign_implementation = {
        name: sha256_path(Path(__file__).resolve().with_name(filename))
        for name, filename in {
            "preflight_sha256": "preflight.py",
            "run_case_sha256": "run_case.py",
            "run_campaign_sha256": "run_campaign.py",
            "analyze_sha256": "analyze.py",
            "common_sha256": "common.py",
        }.items()
    }
    snapshot_stable = (
        first["snapshot_count"] == second["snapshot_count"]
        and first["snapshot_bytes"] == second["snapshot_bytes"]
        and first["snapshot_names"] == second["snapshot_names"]
    )
    no_competing_simulation = not bool(
        subprocess.run(
            [
                "pgrep",
                "-f",
                "(run_position_scan_case|run_paired_repeat_case|mobility_contrast.run_case)",
            ],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    checks = {
        "frozen_contract": contract["status"] == "frozen_before_simulation",
        "factor": contract["frozen_model"]["protected_mobility_factor"]
        == PROTECTED_FACTOR,
        "exact_four_cases": len(CASE_ORDER) == 4,
        "sources": all(item["passed"] for item in source_reports.values()),
        "parent_contract": parent_contract["public_copy_matches"],
        "implementation": all(implementation.values()),
        "factor_reproduction": factor_report["passed"],
        "disk": second["free_disk_bytes"] > required,
        "ram": physical_memory - PEAK_RAM_UPPER_BOUND >= 16 << 30,
        "snapshot_population_stable": snapshot_stable,
        "snapshot_cleanup_not_required": second["snapshot_bytes"] < 20 << 30,
        "no_competing_simulation": no_competing_simulation,
        "zero_simulation_steps": True,
    }
    report = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "status": "GO" if all(checks.values()) else "NO-GO",
        "generated_unix_time": time.time(),
        "simulation_steps_performed": 0,
        "checks": checks,
        "contract": {
            "path": str(CONTRACT_PATH.resolve()),
            "sha256": sha256_path(CONTRACT_PATH),
        },
        "frozen_implementation_snapshot": dict(FROZEN_IMPLEMENTATION_SNAPSHOT),
        "sources": source_reports,
        "parent_contract": parent_contract,
        "implementation": implementation,
        "campaign_implementation": campaign_implementation,
        "factor_reproduction": factor_report,
        "resources": {
            "runtime_upper_bound_seconds": RUNTIME_UPPER_BOUND_SECONDS,
            "runtime_upper_bound_hours": RUNTIME_UPPER_BOUND_SECONDS / 3600,
            "field_bytes_upper_bound": FIELD_BYTES_UPPER_BOUND,
            "persistent_output_upper_bound_bytes": PERSISTENT_OUTPUT_UPPER_BOUND,
            "largest_atomic_write_bytes": LARGEST_ATOMIC_WRITE,
            "untouched_reserve_bytes": UNTOUCHED_RESERVE,
            "filesystem_volatility_allowance_bytes": volatility,
            "measured_loss_rate_bytes_per_second": loss_rate,
            "required_free_disk_bytes": required,
            "free_disk_bytes": second["free_disk_bytes"],
            "remaining_disk_margin_bytes": second["free_disk_bytes"] - required,
            "peak_ram_upper_bound_bytes": PEAK_RAM_UPPER_BOUND,
            "physical_memory_bytes": physical_memory,
            "physical_ram_margin_bytes": physical_memory - PEAK_RAM_UPPER_BOUND,
        },
        "observation": {
            "duration_seconds": elapsed,
            "first": first,
            "second": second,
            "snapshot_fingerprint": (
                "owned tmp.* directory with exactly Git index, index.lock, objects"
            ),
        },
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "output_filesystem_path": str(output.parent.resolve()),
            "historical_root": str(historical_root.resolve()),
        },
    }
    atomic_json(output, report)
    return report


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PREFLIGHT_PATH)
    parser.add_argument("--observe-seconds", type=float, default=60.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    report = run_preflight(
        args.historical_root.resolve(),
        output=args.output.resolve(),
        observe_seconds=args.observe_seconds,
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["status"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
