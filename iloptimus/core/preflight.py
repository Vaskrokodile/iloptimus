"""Run validation before loading models or starting accelerator work."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .hardware import HardwareInfo
from .models import ModelInfo, check_compatibility
from .tasksets import TasksetInfo

CheckStatus = Literal["pass", "warn", "block"]
SUPPORTED_BACKENDS = {"mlx", "vllm"}
SUPPORTED_PRECISIONS = {"fp16", "int8", "int4"}


@dataclass(frozen=True)
class PreflightCheck:
    id: str
    label: str
    status: CheckStatus
    detail: str


@dataclass(frozen=True)
class RunPreflight:
    ready: bool
    model_id: str
    taskset_id: str
    backend: str
    precision: str
    checks: list[PreflightCheck]

    def public(self) -> dict:
        return {
            "ready": self.ready,
            "model_id": self.model_id,
            "taskset_id": self.taskset_id,
            "backend": self.backend,
            "precision": self.precision,
            "checks": [asdict(check) for check in self.checks],
        }


def evaluate_run_preflight(
    *,
    model_id: str,
    taskset_id: str,
    backend: str,
    precision: str,
    benchmark_batch_size: int,
    hardware: HardwareInfo,
    model: ModelInfo | None,
    taskset: TasksetInfo | None,
    source_available: bool,
) -> RunPreflight:
    checks: list[PreflightCheck] = []

    if model is None:
        checks.append(PreflightCheck("model", "Model", "block", f"Unknown model: {model_id}"))
    else:
        checks.append(PreflightCheck("model", "Model", "pass", f"{model.name} is registered."))

    if taskset is None:
        checks.append(PreflightCheck("taskset", "Taskset", "block", f"Unknown taskset: {taskset_id}"))
    else:
        sandbox_note = " Requires sandboxed execution." if taskset.needs_sandbox else ""
        checks.append(
            PreflightCheck(
                "taskset",
                "Taskset",
                "pass",
                f"{taskset.name} has {taskset.num_tasks} tasks.{sandbox_note}",
            )
        )

    if backend not in SUPPORTED_BACKENDS:
        checks.append(
            PreflightCheck(
                "backend",
                "Backend",
                "block",
                f"{backend} is not a supported local training backend. Choose MLX or vLLM.",
            )
        )
    elif model is not None and backend not in model.backends:
        checks.append(
            PreflightCheck(
                "backend-model",
                "Backend support",
                "block",
                f"{model.name} does not declare support for the {backend} backend.",
            )
        )
    elif backend == "mlx" and hardware.gpu.type != "apple-silicon":
        checks.append(
            PreflightCheck(
                "backend-hardware",
                "Backend hardware",
                "block",
                "MLX training requires Apple Silicon hardware.",
            )
        )
    elif backend == "vllm" and hardware.gpu.type != "cuda":
        checks.append(
            PreflightCheck(
                "backend-hardware",
                "Backend hardware",
                "block",
                "vLLM training requires an NVIDIA CUDA GPU.",
            )
        )
    elif backend == "vllm" and not hardware.vllm_available:
        checks.append(
            PreflightCheck(
                "backend-runtime",
                "Backend runtime",
                "warn",
                "vLLM is unavailable; inference will use the slower HF Transformers fallback.",
            )
        )
    else:
        checks.append(PreflightCheck("backend", "Backend", "pass", f"{backend} is compatible with this hardware."))

    if precision not in SUPPORTED_PRECISIONS:
        checks.append(
            PreflightCheck(
                "precision",
                "Precision",
                "block",
                f"Unsupported precision: {precision}. Choose fp16, int8, or int4.",
            )
        )
    elif model is not None:
        compatibility = check_compatibility(model, hardware)
        if compatibility.status == "not-recommended":
            checks.append(
                PreflightCheck(
                    "memory",
                    "Memory fit",
                    "block",
                    compatibility.reason,
                )
            )
        elif compatibility.status == "tight":
            checks.append(
                PreflightCheck(
                    "memory",
                    "Memory fit",
                    "warn",
                    compatibility.reason,
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    "memory",
                    "Memory fit",
                    "pass",
                    f"Best available precision is {compatibility.best_precision} ({compatibility.best_precision_gb:.1f} GB).",
                )
            )

    if not source_available:
        checks.append(
            PreflightCheck(
                "model-source",
                "Model files",
                "block",
                "Download the selected model in Model Library before starting training.",
            )
        )
    else:
        checks.append(PreflightCheck("model-source", "Model files", "pass", "A local model source is available."))

    if not 1 <= benchmark_batch_size <= 64:
        checks.append(
            PreflightCheck(
                "benchmark-batch",
                "Benchmark batch",
                "block",
                "Benchmark batch size must be between 1 and 64.",
            )
        )
    elif benchmark_batch_size > 1 and backend != "vllm":
        checks.append(
            PreflightCheck(
                "benchmark-batch",
                "Benchmark batch",
                "warn",
                "This backend uses the safe sequential fallback for benchmark inference.",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "benchmark-batch",
                "Benchmark batch",
                "pass",
                f"Up to {benchmark_batch_size} prompts can be submitted together.",
            )
        )

    return RunPreflight(
        ready=not any(check.status == "block" for check in checks),
        model_id=model_id,
        taskset_id=taskset_id,
        backend=backend,
        precision=precision,
        checks=checks,
    )
