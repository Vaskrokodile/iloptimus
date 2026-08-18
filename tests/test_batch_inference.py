from __future__ import annotations

from dataclasses import dataclass

from iloptimus.core import benchmark, inference
from iloptimus.core.backends.base import (
    THINK_CLOSE,
    Backend,
    GenerateResult,
    GRPOConfig,
    InferenceResult,
    ModelHandle,
    SFTConfig,
    SFTExample,
)
from iloptimus.core.backends.vllm_backend import VLLMBackend


@dataclass
class _FakeBackend:
    calls: list[list[str]]

    def run_batch_inference(self, handle, prompts, **kwargs):
        self.calls.append(list(prompts))
        return [
            InferenceResult(
                text=f"answer-{index}",
                reasoning="",
                answer=f"answer-{index}",
                elapsed=0.1,
                tokens_generated=2,
                tokens_per_sec=20.0,
            )
            for index, _ in enumerate(prompts)
        ]


def test_run_inference_batch_preserves_empty_and_nonempty_contract(monkeypatch):
    fake = _FakeBackend([])
    monkeypatch.setattr(inference, "_backend_for", lambda _handle: fake)

    handle = object()
    assert inference.run_inference_batch(handle, []) == []
    result = inference.run_inference_batch(handle, ["first", "second"])

    assert [item.answer for item in result] == ["answer-0", "answer-1"]
    assert fake.calls == [["first", "second"]]


def test_benchmark_batches_prompts_and_keeps_task_order(monkeypatch):
    prompts = []
    batches = []
    callbacks = []

    monkeypatch.setattr(benchmark, "get_num_tasks", lambda _domain: 5)
    monkeypatch.setattr(benchmark, "build_prompt", lambda _domain, index: f"prompt-{index}")
    monkeypatch.setattr(benchmark, "clear_cache", lambda: None)
    monkeypatch.setattr(benchmark, "get_memory_info", lambda: {"peak_memory_gb": 2.5})
    monkeypatch.setattr(
        benchmark,
        "grade_response",
        lambda _domain, index, text: type(
            "Grade",
            (),
            {
                "score": float(index) / 10,
                "correctness": float(index % 2),
                "reasoning_quality": 0.5,
            },
        )(),
    )

    def fake_batch(_handle, prompt_batch, **_kwargs):
        batches.append(list(prompt_batch))
        prompts.extend(prompt_batch)
        return [
            InferenceResult(
                text=f"answer-{prompt}",
                reasoning="reasoning",
                answer=f"answer-{prompt}",
                elapsed=0.05,
                tokens_generated=4,
                tokens_per_sec=80.0,
            )
            for prompt in prompt_batch
        ]

    monkeypatch.setattr(benchmark, "run_inference_batch", fake_batch)

    result = benchmark.run_benchmark(
        object(),
        "reasoning",
        num_tasks=5,
        batch_size=2,
        on_task_complete=lambda index, total, task: callbacks.append((index, total, task.task_idx)),
    )

    assert batches == [["prompt-0", "prompt-1"], ["prompt-2", "prompt-3"], ["prompt-4"]]
    assert prompts == [f"prompt-{index}" for index in range(5)]
    assert [task.task_idx for task in result.task_results] == list(range(5))
    assert callbacks == [(index, 5, index) for index in range(5)]
    assert result.total_tokens == 20
    assert result.peak_memory_gb == 2.5


def test_vllm_batch_path_batches_forced_answers_and_preserves_order(monkeypatch):
    calls = []

    class Tokenizer:
        eos_token_id = 99

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is True
            return f"chat:{messages[0]['content']}"

    def fake_generate_batch(_handle, prompts, **kwargs):
        calls.append((list(prompts), kwargs))
        if len(prompts) == 2:
            return [
                GenerateResult(
                    text=f"reason-0{THINK_CLOSE}answer-0",
                    token_ids=[1, 2, 3],
                ),
                GenerateResult(text="reason-1", token_ids=[4, 5]),
            ]
        return [GenerateResult(text="answer-1", token_ids=[6, 7])]

    backend = VLLMBackend()
    monkeypatch.setattr(backend, "_vllm_generate_batch", fake_generate_batch)
    handle = ModelHandle(
        model=object(),
        tokenizer=Tokenizer(),
        model_id="model",
        huggingface_id="example/model",
        precision="fp16",
        quantized=False,
        backend="vllm",
        backend_obj={"vllm_llm": object(), "active_adapter_path": None},
    )

    result = backend.run_batch_inference(
        handle,
        ["prompt-0", "prompt-1"],
        max_reasoning_tokens=8,
        max_answer_tokens=4,
    )

    assert [item.answer for item in result] == ["answer-0", "answer-1"]
    assert [item.forced_answer for item in result] == [False, True]
    assert len(calls) == 2
    assert len(calls[0][0]) == 2
    assert len(calls[1][0]) == 1
    assert calls[1][0][0].startswith("chat:prompt-1reason-1")


def test_default_backend_batch_fallback_clears_each_prompt():
    class FakeBackend(Backend):
        name = "fake"

        def __init__(self):
            self.prompts = []
            self.clears = 0

        def load(self, **_kwargs):
            raise NotImplementedError

        def generate(self, *_args, **_kwargs):
            raise NotImplementedError

        def stream_generate(self, *_args, **_kwargs):
            raise NotImplementedError

        def run_two_stage_inference(self, _handle, prompt, **_kwargs):
            self.prompts.append(prompt)
            return InferenceResult(prompt, "", prompt, 0.1, 1, 10.0)

        def clear_cache(self, _handle=None):
            self.clears += 1

        def release_memory(self):
            return None

        def get_memory_info(self):
            return {}

        def swap_adapters(self, handle, adapter_path):
            return handle

        def run_sft(self, handle, examples: list[SFTExample], config: SFTConfig, adapter_path: str, on_metrics=None):
            raise NotImplementedError

        def make_grpo_trainer(self, handle, config: GRPOConfig, adapter_path: str):
            raise NotImplementedError

    backend = FakeBackend()
    results = backend.run_batch_inference(object(), ["a", "b"])

    assert [result.answer for result in results] == ["a", "b"]
    assert backend.prompts == ["a", "b"]
    assert backend.clears == 2
