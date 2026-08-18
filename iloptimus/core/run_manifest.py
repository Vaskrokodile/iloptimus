"""Reproducibility metadata for training, RSI, and test-time-compute runs."""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


_TRACKED_DISTRIBUTIONS = (
    "iloptimus",
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "vllm",
    "mlx",
    "mlx-lm",
    "safetensors",
)


def _git_metadata() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            ).stdout.strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"revision": "unknown", "dirty": None}


def _installed_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in _TRACKED_DISTRIBUTIONS:
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            versions[distribution] = None
    return versions


def _public_dataclass(value: object) -> dict | None:
    return asdict(value) if is_dataclass(value) else None


def build_run_manifest(
    config: object,
    *,
    hardware: object | None = None,
    model: object | None = None,
    taskset: object | None = None,
) -> dict:
    """Build a local-only manifest without importing optional ML runtimes."""
    config_payload = _public_dataclass(config) or {}
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "git": _git_metadata(),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": _installed_versions(),
        },
        "config": config_payload,
        "hardware": _public_dataclass(hardware),
        "model": _public_dataclass(model),
        "taskset": _public_dataclass(taskset),
    }
