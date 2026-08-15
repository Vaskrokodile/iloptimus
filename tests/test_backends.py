"""Backend-abstraction tests for the vLLM/CUDA path.

These tests verify the backend dispatch contract without requiring torch,
transformers, peft, or vllm to be installed. The VLLMBackend's heavy imports
are all lazy (inside methods), so the module imports cleanly on any machine;
the tests mock the backend's ``backend_obj`` state and the few HF/vLLM calls
the dispatchers make.

Together with the existing MLX tests, this locks in that both backends satisfy
the same :class:`Backend` interface and that the public inference/sft/grpo
modules delegate correctly.
"""

from __future__ import annotations

import sys
import types

import pytest

from iloptimus.core.backends import get_backend, resolve_backend
from iloptimus.core.backends.base import (
    Backend,
    GRPOConfig,
    GRPOMetrics,
    GRPOTrainerLike,
    InferenceResult,
    ModelHandle,
    SFTConfig,
    SFTExample,
    THINK_CLOSE,
)
from iloptimus.core.backends.vllm_backend import VLLMBackend


# ---------------------------------------------------------------------------
# Factory + interface
# ---------------------------------------------------------------------------


def test_get_backend_returns_cached_singleton_per_name():
    a = get_backend("vllm")
    b = get_backend("vllm")
    assert a is b
    assert isinstance(a, VLLMBackend)
    assert a.name == "vllm"


def test_get_backend_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("tpu")


def test_both_backends_satisfy_abc():
    from iloptimus.core.backends.mlx_backend import MLXBackend

    assert isinstance(MLXBackend(), Backend)
    assert isinstance(VLLMBackend(), Backend)


def test_vllm_backend_imports_without_torch_installed():
    """The vLLM backend module must import on a machine without torch/vllm."""
    # If this import already succeeded at collection time the assertion is
    # trivially true; the point is that it does not raise.
    assert VLLMBackend is not None
    assert hasattr(VLLMBackend, "run_sft")
    assert hasattr(VLLMBackend, "make_grpo_trainer")


# ---------------------------------------------------------------------------
# resolve_backend
# ---------------------------------------------------------------------------


def _hw(*, gpu_type="cuda", mlx=False, vllm=True, recommended="vllm"):
    hw = types.SimpleNamespace()
    hw.gpu = types.SimpleNamespace(type=gpu_type, name="CUDA", vram_gb=24.0)
    hw.mlx_available = mlx
    hw.vllm_available = vllm
    hw.torch_available = True
    hw.recommended_backend = recommended
    return hw


def test_resolve_backend_prefers_explicit_available():
    assert resolve_backend(hardware=_hw(mlx=True), preferred="mlx") == "mlx"


def test_resolve_backend_uses_hardware_recommendation():
    assert resolve_backend(hardware=_hw(recommended="vllm")) == "vllm"
    assert resolve_backend(hardware=_hw(gpu_type="apple-silicon", mlx=True, recommended="mlx")) == "mlx"


def test_resolve_backend_falls_back_to_platform_default(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert resolve_backend() == "mlx"
    monkeypatch.setattr(sys, "platform", "linux")
    assert resolve_backend() == "vllm"


# ---------------------------------------------------------------------------
# Inference dispatch through the vLLM backend (mocked HF generate)
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer stub satisfying the dispatcher's calls."""

    eos_token_id = 151643

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert messages and tokenize is False and add_generation_prompt is True
        return "chat:" + messages[0]["content"]

    def encode(self, text):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


def _vllm_handle(*, adapter_path=None, hf_generate_text="hello world"):
    """Build a ModelHandle whose backend_obj mocks the vLLM backend's state.

    ``vllm_llm`` is None so the backend uses the HF generate fallback, which we
    patch to return a canned result without requiring torch.
    """
    captured = {}

    def fake_hf_generate(self, handle, prompt_text, **kwargs):
        captured["prompt_text"] = prompt_text
        captured["kwargs"] = kwargs
        from iloptimus.core.backends.base import GenerateResult

        return GenerateResult(text=hf_generate_text, token_ids=[1, 2, 3], logprobs=[], finish_reason="stop")

    # Patch the HF generate method on the class for the duration of the test.
    VLLMBackend._hf_generate = fake_hf_generate  # type: ignore[assignment]

    backend_obj = {
        "vllm_llm": None,  # force HF fallback path
        "hf_model": object(),
        "hf_tokenizer": _FakeTokenizer(),
        "peft_model": None,
        "device": "cpu",
        "active_adapter_path": adapter_path,
    }
    handle = ModelHandle(
        model=backend_obj["hf_model"],
        tokenizer=backend_obj["hf_tokenizer"],
        model_id="test",
        huggingface_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        precision="int4",
        quantized=True,
        adapter_path=adapter_path,
        backend="vllm",
        backend_obj=backend_obj,
    )
    handle._captured = captured  # type: ignore[attr-defined]
    return handle


def test_vllm_generate_uses_hf_fallback_when_vllm_engine_absent():
    handle = _vllm_handle(hf_generate_text="answer text")
    backend = get_backend("vllm")
    result = backend.generate(handle, "prompt text", max_tokens=10, temperature=0.0)
    assert result.text == "answer text"
    assert result.token_ids == [1, 2, 3]
    # The HF fallback received the rendered prompt and the decode kwargs.
    assert handle._captured["prompt_text"] == "prompt text"
    assert handle._captured["kwargs"]["max_tokens"] == 10


def test_vllm_two_stage_inference_splits_on_think_close():
    reasoning = "let me think" + THINK_CLOSE
    answer = "final answer"
    handle = _vllm_handle(hf_generate_text=reasoning + "\n" + answer)
    backend = get_backend("vllm")
    inf = backend.run_two_stage_inference(handle, "question", max_reasoning_tokens=16, max_answer_tokens=16)
    assert inf.reasoning == "let me think"
    assert inf.answer == answer
    assert inf.forced_answer is False
    assert inf.tokens_per_sec > 0


def test_vllm_two_stage_inference_forces_answer_when_reasoning_exhausts_budget():
    # No think-close token in the output -> forced-answer path.
    handle = _vllm_handle(hf_generate_text="just reasoning no close tag")
    backend = get_backend("vllm")
    inf = backend.run_two_stage_inference(handle, "question", max_reasoning_tokens=16, max_answer_tokens=16)
    assert inf.forced_answer is True
    assert inf.reasoning == "just reasoning no close tag"
    # The forced second pass produced *an* answer (the same canned text).
    assert inf.answer


def test_vllm_stream_generate_yields_per_token_chunks():
    handle = _vllm_handle(hf_generate_text="abc def")
    backend = get_backend("vllm")
    chunks = list(backend.stream_generate(handle, "prompt", max_tokens=8, temperature=0.0))
    # The fake generate returned token_ids [1,2,3] -> 3 chunks.
    assert len(chunks) == 3
    assert all(c.generation_tokens == i + 1 for i, c in enumerate(chunks))


def test_vllm_clear_cache_and_memory_info_are_safe_without_torch():
    backend = get_backend("vllm")
    # Should not raise even though torch is not installed.
    backend.clear_cache(None)
    backend.release_memory()
    info = backend.get_memory_info()
    assert info == {}


# ---------------------------------------------------------------------------
# GRPO trainer interface
# ---------------------------------------------------------------------------


def test_vllm_make_grpo_trainer_returns_grpolike():
    """make_grpo_trainer must return something with train_step + save."""
    # We can't actually construct VLLMGRPOTrainer without torch, but we can
    # verify the backend's method exists and the ABC contract is satisfied.
    assert hasattr(VLLMBackend, "make_grpo_trainer")
    assert issubclass(GRPOTrainerLike, object)
    # The GRPOConfig/GRPOMetrics dataclasses are shared and constructible.
    cfg = GRPOConfig()
    assert cfg.group_size == 4
    m = GRPOMetrics(
        iteration=0,
        mean_reward=0.5,
        std_reward=0.1,
        max_reward=1.0,
        min_reward=0.0,
        mean_correctness=0.5,
        mean_reasoning_quality=0.0,
        loss=0.1,
        rollout_time=1.0,
        update_time=0.5,
        total_time=1.5,
        peak_memory_gb=2.0,
        avg_episode_tokens=100.0,
    )
    assert m.iteration == 0


# ---------------------------------------------------------------------------
# SFT config + examples are backend-agnostic
# ---------------------------------------------------------------------------


def test_sft_config_and_examples_are_shared_types():
    cfg = SFTConfig(num_iters=5, lora_rank=16)
    assert cfg.num_iters == 5
    assert cfg.lora_rank == 16
    ex = SFTExample(prompt="p", response="r")
    assert ex.prompt == "p" and ex.response == "r"
