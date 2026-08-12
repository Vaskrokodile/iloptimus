"""Hardware-aware context capacity and decode-speed estimates."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .hardware import HardwareInfo
from .models import ModelInfo, check_compatibility
from .storage import app_home, atomic_write_json


@dataclass(frozen=True)
class ContextEstimate:
    context_window: int
    max_model_context: int
    max_safe_context: int
    estimated_tps: float
    low_tps: float
    high_tps: float
    kv_cache_gb: float
    model_memory_gb: float
    available_memory_gb: float
    fits_in_memory: bool
    basis: str

    def public(self) -> dict[str, Any]:
        return asdict(self)


def _memory_bandwidth_gbps(hw: HardwareInfo) -> float:
    name = f"{hw.cpu_name} {hw.gpu.name}".lower()
    apple = {
        "m1 ultra": 800,
        "m1 max": 400,
        "m1 pro": 200,
        "m1": 68,
        "m2 ultra": 800,
        "m2 max": 400,
        "m2 pro": 200,
        "m2": 100,
        "m3 ultra": 819,
        "m3 max": 400,
        "m3 pro": 150,
        "m3": 100,
        "m4 max": 546,
        "m4 pro": 273,
        "m4": 120,
    }
    for chip, bandwidth in apple.items():
        if chip in name:
            return float(bandwidth)
    if hw.gpu.type == "apple-silicon":
        return 100.0
    if hw.gpu.type == "cuda":
        return max(200.0, min(1_500.0, hw.gpu.vram_gb * 40.0))
    return max(15.0, min(100.0, hw.cpu_cores * 3.0))


def _architecture(model: ModelInfo) -> tuple[int, int, float]:
    layers = max(16, round(10 * math.sqrt(model.params_b)))
    hidden = max(1024, round(math.sqrt(model.params_b * 1e9 / (12 * layers)) / 128) * 128)
    gqa_ratio = 0.25 if model.family in {"qwen2.5", "llama3.2", "mistral", "deepseek-r1-distill"} else 0.5
    return layers, hidden, gqa_ratio


def _kv_gb(model: ModelInfo, context: int) -> float:
    layers, hidden, gqa_ratio = _architecture(model)
    return 2 * layers * hidden * context * 2 * gqa_ratio / 1024**3


def _samples_path() -> Path:
    return app_home() / "performance.json"


def _samples() -> dict[str, list[dict[str, float]]]:
    try:
        payload = json.loads(_samples_path().read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def record_chat_performance(model_id: str, context_tokens: int, tokens_per_sec: float) -> None:
    if tokens_per_sec <= 0:
        return
    payload = _samples()
    values = payload.setdefault(model_id, [])
    values.append({"context_tokens": float(context_tokens), "tokens_per_sec": float(tokens_per_sec)})
    payload[model_id] = values[-20:]
    try:
        atomic_write_json(_samples_path(), payload)
    except OSError:
        # Calibration is optional telemetry and must never make chat fail.
        return


def estimate_context_performance(model: ModelInfo, hw: HardwareInfo, requested_context: int) -> ContextEstimate:
    compatibility = check_compatibility(model, hw)
    model_memory = compatibility.best_precision_gb
    available = hw.total_memory_gb
    bytes_per_token_gb = max(_kv_gb(model, 1), 1e-9)
    memory_budget = max(0.0, available * 0.88 - model_memory)
    safe_by_memory = int(memory_budget / bytes_per_token_gb)
    max_safe = max(2_048, min(model.context_length, safe_by_memory))
    context = max(1_024, min(int(requested_context), model.context_length))
    kv_cache = _kv_gb(model, context)
    fits = model_memory + kv_cache <= available * 0.9

    bandwidth = _memory_bandwidth_gbps(hw)
    backend_efficiency = 0.36 if hw.recommended_backend == "mlx" else 0.31 if hw.gpu.type == "cuda" else 0.12
    base_tps = bandwidth * backend_efficiency / max(model_memory, 0.25)
    architecture_factor = 0.90 if "deepseek" in model.family else 1.0
    context_penalty = 1.0 / (1.0 + (context / 16_384) * (0.06 + 0.008 * math.sqrt(model.params_b)))
    estimate = base_tps * architecture_factor * context_penalty
    basis = "hardware/model estimate"

    samples = _samples().get(model.id, [])
    if samples:
        normalized = [
            sample["tokens_per_sec"]
            / (1.0 / (1.0 + (sample["context_tokens"] / 16_384) * (0.06 + 0.008 * math.sqrt(model.params_b))))
            for sample in samples
            if sample.get("tokens_per_sec", 0) > 0
        ]
        if normalized:
            estimate = statistics.median(normalized) * context_penalty
            basis = f"calibrated from {len(normalized)} local run{'s' if len(normalized) != 1 else ''}"

    if not fits:
        estimate *= 0.35
    estimate = max(0.1, estimate)
    spread = 0.16 if samples else 0.32
    return ContextEstimate(
        context_window=context,
        max_model_context=model.context_length,
        max_safe_context=max_safe,
        estimated_tps=round(estimate, 1),
        low_tps=round(estimate * (1 - spread), 1),
        high_tps=round(estimate * (1 + spread), 1),
        kv_cache_gb=round(kv_cache, 2),
        model_memory_gb=round(model_memory, 2),
        available_memory_gb=round(available, 2),
        fits_in_memory=fits,
        basis=basis,
    )
