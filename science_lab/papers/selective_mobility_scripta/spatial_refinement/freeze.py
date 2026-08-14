"""Git/runtime freeze manifest and preflight-binding verification."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


PACKAGE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIRECTORY.parents[3]

# Every repository file imported on the source/guard/trajectory/analyzer/
# supervisor paths, including eager transitive imports of legacy storage.
FROZEN_RELATIVE_PATHS = (
    "science_lab/papers/__init__.py",
    "science_lab/papers/selective_mobility_scripta/__init__.py",
    "science_lab/papers/selective_mobility_scripta/spatial_refinement/__init__.py",
    "science_lab/papers/selective_mobility_scripta/spatial_refinement/SCIENTIFIC_CONTRACT.md",
    "science_lab/papers/selective_mobility_scripta/spatial_refinement/protocol.py",
    "science_lab/papers/selective_mobility_scripta/spatial_refinement/prolongation.py",
    "science_lab/papers/selective_mobility_scripta/spatial_refinement/freeze.py",
    "science_lab/papers/selective_mobility_scripta/spatial_refinement/preflight.py",
    "science_lab/papers/selective_mobility_scripta/spatial_refinement/runner.py",
    "science_lab/papers/selective_mobility_scripta/spatial_refinement/analyze.py",
    "science_lab/papers/selective_mobility_scripta/spatial_refinement/supervisor.py",
    "science_lab/papers/nanowire_gb_junction/__init__.py",
    "science_lab/papers/nanowire_gb_junction/roy_2021_reproduction/__init__.py",
    "science_lab/papers/nanowire_gb_junction/roy_2021_reproduction/model.py",
    "science_lab/papers/nanowire_gb_junction/roy_2021_reproduction/run_preflight.py",
    "science_lab/papers/nanowire_gb_junction/roy_selective_mobility_protection/__init__.py",
    "science_lab/papers/nanowire_gb_junction/roy_selective_mobility_protection/model.py",
    "science_lab/papers/nanowire_gb_junction/roy_selective_mobility_protection/geometry.py",
    "science_lab/papers/nanowire_gb_junction/roy_gb_junction_sentinel/__init__.py",
    "science_lab/papers/nanowire_gb_junction/roy_gb_junction_sentinel/contract.py",
    "science_lab/papers/nanowire_gb_junction/roy_gb_junction_sentinel/diagnostics.py",
    "science_lab/papers/nanowire_gb_junction/roy_gb_junction_sentinel/initializer.py",
    "science_lab/papers/nanowire_gb_junction/roy_gb_junction_sentinel/storage.py",
    "science_lab/papers/nanowire_gb_junction/roy_crossed_initializer_validation/__init__.py",
    "science_lab/papers/nanowire_gb_junction/roy_crossed_initializer_validation/run_validation.py",
    "science_lab/papers/nanowire_gb_junction/roy_fixed_gb_bridge/__init__.py",
    "science_lab/papers/nanowire_gb_junction/roy_fixed_gb_bridge/model.py",
    "science_lab/papers/nanowire_gb_junction/roy_fixed_gb_bridge/provenance.py",
)
FROZEN_PATHS = tuple(REPOSITORY_ROOT / path for path in FROZEN_RELATIVE_PATHS)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def runtime_manifest() -> dict[str, str | None]:
    return {
        relative: sha256_path(REPOSITORY_ROOT / relative)
        if (REPOSITORY_ROOT / relative).is_file()
        else None
        for relative in FROZEN_RELATIVE_PATHS
    }


def runtime_environment() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for distribution in ("numpy", "scipy", "psutil", "matplotlib"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "distributions": versions,
    }


def _git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def git_freeze_state() -> dict[str, Any]:
    try:
        head = _git("rev-parse", "HEAD").stdout.strip()
        branch = _git("symbolic-ref", "--short", "HEAD").stdout.strip()
        upstream = _git(
            "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"
        ).stdout.strip()
        upstream_head = _git("rev-parse", "@{u}").stdout.strip()
        tracked_output = _git("ls-files", "--", *FROZEN_RELATIVE_PATHS).stdout
        tracked = set(tracked_output.splitlines())
        status = _git(
            "status", "--porcelain=v1", "--", *FROZEN_RELATIVE_PATHS
        ).stdout.splitlines()
        missing_tracked = sorted(set(FROZEN_RELATIVE_PATHS) - tracked)
        checks = {
            "all_frozen_paths_exist": all(path.is_file() for path in FROZEN_PATHS),
            "all_frozen_paths_tracked": not missing_tracked,
            "all_frozen_paths_clean": not status,
            "upstream_is_origin": upstream.startswith("origin/"),
            "head_equals_upstream": bool(head and head == upstream_head),
        }
        return {
            "head": head,
            "branch": branch,
            "upstream": upstream,
            "upstream_head": upstream_head,
            "dirty_entries": status,
            "missing_tracked_paths": missing_tracked,
            "checks": checks,
            "passed": all(checks.values()),
        }
    except (OSError, subprocess.SubprocessError) as error:
        return {
            "head": None,
            "branch": None,
            "upstream": None,
            "upstream_head": None,
            "dirty_entries": [],
            "missing_tracked_paths": list(FROZEN_RELATIVE_PATHS),
            "checks": {"git_inventory_available": False},
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
        }


def process_inventory() -> dict[str, list[dict[str, Any]]]:
    try:
        output = subprocess.run(
            ["/bin/ps", "-axo", "pid=,ppid=,comm=,command="],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {
            "science": [{"pid": None, "command": "process_inventory_unavailable"}],
            "git": [{"pid": None, "command": "process_inventory_unavailable"}],
        }
    ignored = {os.getpid(), os.getppid()}
    science: list[dict[str, Any]] = []
    git: list[dict[str, Any]] = []
    for raw in output.splitlines():
        fields = raw.strip().split(maxsplit=3)
        if len(fields) != 4:
            continue
        pid, ppid = int(fields[0]), int(fields[1])
        command_name, command = fields[2], fields[3]
        if pid in ignored:
            continue
        if Path(command_name).name == "git":
            git.append({"pid": pid, "ppid": ppid, "command": command})
            continue
        lowered = command.lower()
        likely_python = "python" in command_name.lower() or "python" in lowered
        likely_science = any(
            token in lowered
            for token in (
                "run_r12",
                "roy_selective_mobility_protection",
                "nanowire_kinetic_order_sensitivity",
                "spatial_refinement.runner",
                "phase-field",
                "phase_field",
            )
        )
        self_preflight = "spatial_refinement.preflight" in lowered
        if likely_python and likely_science and not self_preflight:
            science.append({"pid": pid, "ppid": ppid, "command": command})
    return {"science": science, "git": git}


def verify_preflight_binding(
    output_root: Path,
    *,
    protocol_id: str,
    source_root: Path,
) -> dict[str, Any]:
    summary_path = output_root.resolve() / "preflight" / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    core = report.get("receipt_core")
    if not isinstance(core, dict):
        raise RuntimeError("preflight has no receipt core")
    stored_hash = report.get("receipt_core_sha256")
    checks = {
        "decision": report.get("decision") == "GO",
        "all_preflight_checks": all(report.get("checks", {}).values()),
        "protocol": report.get("protocol_id") == protocol_id,
        "core_decision": core.get("decision") == "GO",
        "core_checks": all(core.get("checks", {}).values()),
        "core_hash": canonical_sha256(core) == stored_hash,
        "manifest": runtime_manifest() == core.get("runtime_manifest"),
        "environment": runtime_environment() == core.get("runtime_environment"),
        "source_root": str(source_root.resolve()) == core.get("source_root"),
    }
    live_git = git_freeze_state()
    frozen_git = core.get("git", {})
    checks["git_live_passed"] = live_git.get("passed") is True
    checks["git_head"] = live_git.get("head") == frozen_git.get("head")
    checks["git_upstream"] = live_git.get("upstream") == frozen_git.get("upstream")
    checks["git_upstream_head"] = live_git.get("upstream_head") == frozen_git.get(
        "upstream_head"
    )
    if not all(checks.values()):
        raise RuntimeError(f"live runtime differs from GO preflight: {checks}")
    return report
