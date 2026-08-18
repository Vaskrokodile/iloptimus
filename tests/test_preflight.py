from __future__ import annotations

from iloptimus.core.hardware import GPUInfo, HardwareInfo
from iloptimus.core.models import get_model
from iloptimus.core.preflight import evaluate_run_preflight
from iloptimus.core.tasksets import get_taskset


def cuda_hardware(*, vllm_available: bool = True) -> HardwareInfo:
    return HardwareInfo(
        cpu_name="Test CPU",
        cpu_cores=16,
        ram_gb=64.0,
        os="Linux test",
        arch="x86_64",
        gpu=GPUInfo(name="Test CUDA", vram_gb=24.0, type="cuda"),
        python_version="3.12.0",
        vllm_available=vllm_available,
        torch_available=True,
        recommended_backend="vllm",
        labels=["CUDA GPU", "PyTorch"],
    )


def test_preflight_passes_for_downloaded_cuda_run():
    result = evaluate_run_preflight(
        model_id="qwen2.5-0.5b",
        taskset_id="gsm8k-v1",
        backend="vllm",
        precision="int4",
        benchmark_batch_size=4,
        hardware=cuda_hardware(),
        model=get_model("qwen2.5-0.5b"),
        taskset=get_taskset("gsm8k-v1"),
        source_available=True,
    )

    assert result.ready is True
    assert result.backend == "vllm"
    assert {check.status for check in result.checks} == {"pass"}


def test_preflight_blocks_missing_model_source():
    result = evaluate_run_preflight(
        model_id="qwen2.5-0.5b",
        taskset_id="gsm8k-v1",
        backend="vllm",
        precision="int4",
        benchmark_batch_size=4,
        hardware=cuda_hardware(),
        model=get_model("qwen2.5-0.5b"),
        taskset=get_taskset("gsm8k-v1"),
        source_available=False,
    )

    assert result.ready is False
    assert next(check for check in result.checks if check.id == "model-source").status == "block"


def test_preflight_blocks_backend_that_does_not_match_hardware():
    result = evaluate_run_preflight(
        model_id="qwen2.5-0.5b",
        taskset_id="gsm8k-v1",
        backend="mlx",
        precision="int4",
        benchmark_batch_size=4,
        hardware=cuda_hardware(),
        model=get_model("qwen2.5-0.5b"),
        taskset=get_taskset("gsm8k-v1"),
        source_available=True,
    )

    assert result.ready is False
    assert next(check for check in result.checks if check.id == "backend-hardware").status == "block"


def test_preflight_warns_for_hf_fallback_and_blocks_invalid_batch():
    result = evaluate_run_preflight(
        model_id="qwen2.5-0.5b",
        taskset_id="gsm8k-v1",
        backend="vllm",
        precision="int4",
        benchmark_batch_size=65,
        hardware=cuda_hardware(vllm_available=False),
        model=get_model("qwen2.5-0.5b"),
        taskset=get_taskset("gsm8k-v1"),
        source_available=True,
    )

    assert result.ready is False
    assert next(check for check in result.checks if check.id == "backend-runtime").status == "warn"
    assert next(check for check in result.checks if check.id == "benchmark-batch").status == "block"
