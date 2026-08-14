"""Atomic, fingerprinted persistence for sentinel preflights and trajectories."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any, IO

import numpy as np

from .contract import sha256_path
from .initializer import array_fingerprint


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        measured = float(value)
        return measured if np.isfinite(measured) else None
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                _json_safe(payload),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_npy(path: Path, field: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, field, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def reserve_output_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)


def acquire_output_lock(output: Path) -> IO[str]:
    handle = (output / ".run.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"another process holds the output lock: {output}")
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def release_output_lock(handle: IO[str]) -> None:
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def checkpoint_paths(
    output: Path,
    step: int,
    *,
    stream: str = "primary",
) -> tuple[Path, Path]:
    if not stream or any(character not in "abcdefghijklmnopqrstuvwxyz_-" for character in stream):
        raise ValueError("checkpoint stream name is unsafe")
    stem = f"checkpoint-{stream}-step-{step:04d}"
    return output / f"{stem}.npy", output / f"{stem}.json"


def write_checkpoint(
    output: Path,
    field: np.ndarray,
    *,
    step: int,
    kinds: tuple[str, ...],
    stream: str = "primary",
) -> dict[str, Any]:
    field_path, metadata_path = checkpoint_paths(output, step, stream=stream)
    if field_path.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint at step {step}")
    atomic_npy(field_path, field)
    metadata = {
        "step": int(step),
        "stream": stream,
        "kinds": list(dict.fromkeys(kinds)),
        "field_path": str(field_path.resolve()),
        "field_bytes": int(field_path.stat().st_size),
        "field_sha256": sha256_path(field_path),
        "field_fingerprint": array_fingerprint(field),
        "shape": list(field.shape),
        "dtype": field.dtype.str,
        "finite": bool(np.isfinite(field).all()),
    }
    atomic_json(metadata_path, metadata)
    metadata["metadata_path"] = str(metadata_path.resolve())
    metadata["metadata_sha256"] = sha256_path(metadata_path)
    return metadata


def load_checkpoint(entry: dict[str, Any]) -> np.ndarray:
    metadata_path = Path(str(entry["metadata_path"]))
    if sha256_path(metadata_path) != entry["metadata_sha256"]:
        raise RuntimeError("checkpoint metadata hash mismatch")
    stored = json.loads(metadata_path.read_text(encoding="utf-8"))
    field_path = Path(str(stored["field_path"]))
    if sha256_path(field_path) != stored["field_sha256"]:
        raise RuntimeError("checkpoint field hash mismatch")
    field = np.load(field_path, allow_pickle=False)
    if list(field.shape) != stored["shape"]:
        raise RuntimeError("checkpoint shape mismatch")
    if field.dtype.str != stored["dtype"]:
        raise RuntimeError("checkpoint dtype mismatch")
    if array_fingerprint(field) != stored["field_fingerprint"]:
        raise RuntimeError("checkpoint field fingerprint mismatch")
    if not np.isfinite(field).all():
        raise RuntimeError("checkpoint field is nonfinite")
    return np.asarray(field, dtype=np.float64)
