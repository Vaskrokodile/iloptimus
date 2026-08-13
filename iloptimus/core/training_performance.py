"""Persist sustained local training throughput for honest time-budget selection."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from .storage import app_home, atomic_write_json


def _path() -> Path:
    return app_home() / "training-performance.json"


def training_profile_key(
    model_id: str, *, sequence_length: int, rank: int, layers: int, backend: str
) -> str:
    return f"{backend}:{model_id}:seq{sequence_length}:r{rank}:l{layers}"


def load_training_seconds_per_iteration(profile_key: str) -> float | None:
    profile = load_training_profile(profile_key)
    return float(profile["seconds_per_iteration"]) if profile else None


def load_training_profile(profile_key: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(_path().read_text(encoding="utf-8"))
        profile = dict(payload.get("profiles", {}).get(profile_key, {}))
        value = float(profile.get("seconds_per_iteration"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return profile if math.isfinite(value) and value > 0 else None


def record_training_throughput(
    profile_key: str,
    reports: Iterable[dict[str, Any]],
    *,
    run_id: str,
    fixed_overhead_seconds: float = 0.0,
) -> dict[str, Any] | None:
    """Record a conservative sustained rate, ignoring warm-up and invalid samples."""
    rates = [
        float(report.get("iterations_per_second") or 0.0)
        for report in reports
        if float(report.get("iterations_per_second") or 0.0) > 0
    ]
    if not rates:
        return None
    sustained = rates[max(1, len(rates) // 5) :] if len(rates) >= 5 else rates
    seconds = [1.0 / rate for rate in sustained]
    # The 75th-percentile step time includes observed thermal/system slowdown
    # without sizing every run to one pathological outlier.
    ordered = sorted(seconds)
    index = min(len(ordered) - 1, math.ceil(0.75 * len(ordered)) - 1)
    conservative = ordered[index]
    path = _path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {"version": 1, "profiles": {}}
    prior = payload.setdefault("profiles", {}).get(profile_key, {})
    prior_seconds = float(prior.get("seconds_per_iteration") or conservative)
    samples = int(prior.get("samples") or 0)
    blended = conservative if not samples else 0.65 * prior_seconds + 0.35 * conservative
    profile = {
        "seconds_per_iteration": round(blended, 4),
        "latest_seconds_per_iteration": round(conservative, 4),
        "median_seconds_per_iteration": round(statistics.median(seconds), 4),
        "samples": samples + len(rates),
        "run_id": run_id,
        "fixed_overhead_seconds": round(max(0.0, fixed_overhead_seconds), 3),
    }
    payload["profiles"][profile_key] = profile
    atomic_write_json(path, payload)
    return profile
