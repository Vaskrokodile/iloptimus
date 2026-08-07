"""Integration tests for the IL Optimus pipeline.

Tests the full pipeline orchestration end-to-end with a mock model, so we
don't need to download a real model from HuggingFace. The ML modules
(inference, grader, benchmark, sft, grpo) are verified via imports + grader
unit tests separately.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from iloptimus.core import (
    RunConfig,
    create_run,
    get_run,
    run_pipeline,
)
from iloptimus.core.grader import GradedResult, grade_response, build_prompt, get_num_tasks
from iloptimus.core.hardware import detect_hardware
from iloptimus.core.inference import InferenceResult, ModelHandle
from iloptimus.core.benchmark import BenchmarkResult, TaskResult


# ---------------------------------------------------------------------------
# Grader tests (real, no mocking)
# ---------------------------------------------------------------------------

def test_grader_reasoning_wrong_answer():
    """A wrong answer should get score 0 with low reasoning quality."""
    result = grade_response("reasoning", 0, "<reasoning>blah</reasoning><answer>42</answer>")
    assert result.correctness == 0.0
    assert result.score == 0.0
    assert result.reasoning_quality < 0.5


def test_grader_coding_bad_code():
    """A bad code response should get low correctness via sandbox verification."""
    result = grade_response(
        "coding", 0,
        "<reasoning>I will write a function</reasoning>"
        "<answer>```python\ndef foo(): pass\n```</answer>",
    )
    assert result.correctness == 0.0
    assert "test_result" in result.info


def test_grader_all_domains_load():
    """All 4 taskset domains should load tasks and build prompts."""
    for domain in ["coding", "reasoning", "agentic-reasoning", "agentic-coding"]:
        n = get_num_tasks(domain)
        assert n > 0, f"{domain} has no tasks"
        prompt = build_prompt(domain, 0)
        assert len(prompt) > 50, f"{domain} prompt too short"


# ---------------------------------------------------------------------------
# Pipeline orchestration test (mocked model)
# ---------------------------------------------------------------------------

@pytest.fixture
def hw():
    return detect_hardware()


@pytest.fixture
def mock_handle():
    """A fake ModelHandle that doesn't require a real model."""
    return ModelHandle(
        model=None,
        tokenizer=None,
        model_id="mock-model",
        huggingface_id="mock/model",
        precision="int4",
        quantized=True,
    )


@pytest.fixture
def mock_inference_result():
    return InferenceResult(
        text="<reasoning>Let me think about this.</reasoning><answer>42</answer>",
        reasoning="Let me think about this.",
        answer="42",
        elapsed=0.1,
        tokens_generated=20,
        tokens_per_sec=200.0,
        forced_answer=False,
    )


@pytest.fixture
def mock_benchmark_result(mock_inference_result):
    return BenchmarkResult(
        accuracy=0.5,
        mean_score=0.3,
        mean_reasoning_quality=0.4,
        total_elapsed=1.0,
        total_tokens=100,
        mean_tokens_per_sec=100.0,
        forced_answer_rate=0.0,
        peak_memory_gb=1.0,
        task_results=[
            TaskResult(
                task_idx=0, score=0.3, correctness=0.5, reasoning_quality=0.4,
                elapsed=0.1, tokens_generated=20, tokens_per_sec=200.0,
                forced_answer=False, response_preview="...",
            ),
        ],
    )


def test_pipeline_runs_all_stages(hw, mock_handle, mock_benchmark_result):
    """The full pipeline should run all 7 stages and emit events, with a mocked model."""
    config = RunConfig(
        model_id="deepseek-r1-distill-qwen-1.5b",
        taskset_id="il-reasoning-v1",
        sft_iters=2,
        grpo_iters=2,
        benchmark_tasks=2,
        max_reasoning_tokens=64,
        max_answer_tokens=64,
    )
    state = create_run(config)

    # Mock the ML-heavy functions so we test orchestration, not inference
    sft_adapters = []

    def fake_load_model(**kwargs):
        return mock_handle

    def fake_run_benchmark(handle, **kwargs):
        # Call on_task_complete if provided
        on_task_complete = kwargs.get("on_task_complete")
        if on_task_complete:
            for i in range(kwargs.get("num_tasks", 2) or 2):
                on_task_complete(i, kwargs.get("num_tasks", 2) or 2, mock_benchmark_result.task_results[0])
        return mock_benchmark_result

    def fake_generate_sft_data(handle, **kwargs):
        from iloptimus.core.sft import SFTExample
        on_progress = kwargs.get("on_progress")
        if on_progress:
            on_progress(1, 1)
        return [SFTExample(prompt="test", response="<reasoning>x</reasoning><answer>42</answer>")]

    def fake_run_sft(handle, examples, **kwargs):
        on_metrics = kwargs.get("on_metrics")
        from iloptimus.core.sft import SFTMetrics
        if on_metrics:
            for i in range(kwargs.get("config").num_iters):
                on_metrics(SFTMetrics(
                    iteration=i, loss=1.0 - i * 0.1, learning_rate=1e-4,
                    elapsed=0.01, peak_memory_gb=1.0,
                ))
        adapter_path = kwargs.get("adapter_path", "il_sft_adapters_test")
        sft_adapters.append(adapter_path)
        return adapter_path

    def fake_grpo_train_step(prompt, grade_fn, on_metrics=None, **kwargs):
        from iloptimus.core.grpo import GRPOMetrics
        metrics = GRPOMetrics(
            iteration=0, mean_reward=0.5, std_reward=0.1, max_reward=0.6,
            min_reward=0.4, mean_correctness=0.5, mean_reasoning_quality=0.4,
            loss=0.3, rollout_time=0.1, update_time=0.1, total_time=0.2,
            peak_memory_gb=1.0, avg_episode_tokens=100.0,
        )
        if on_metrics:
            on_metrics(metrics)
        return metrics

    with patch("iloptimus.core.inference.load_model", side_effect=fake_load_model), \
         patch("iloptimus.core.benchmark.run_benchmark", side_effect=fake_run_benchmark), \
         patch("iloptimus.core.sft.generate_sft_data", side_effect=fake_generate_sft_data), \
         patch("iloptimus.core.sft.run_sft", side_effect=fake_run_sft), \
         patch("iloptimus.core.grpo.GRPOTrainer") as MockTrainer, \
         patch("iloptimus.core.grader.build_prompt", return_value="test prompt"), \
         patch("iloptimus.core.grader.grade_response", return_value=GradedResult(
             score=0.5, correctness=0.5, reasoning_quality=0.4,
         )):
        # Mock GRPOTrainer instance
        trainer_instance = MockTrainer.return_value
        trainer_instance.train_step = fake_grpo_train_step
        trainer_instance.save = lambda *a, **kw: None

        asyncio.run(run_pipeline(state.id, config, hw))

    # Verify the pipeline completed
    assert state.status == "completed", f"Pipeline should complete, got {state.status}"
    assert state.progress == 1.0
    assert state.stage == "done"
    assert state.baseline_accuracy == 0.5
    assert state.post_sft_accuracy == 0.5
    assert state.post_grpo_accuracy == 0.5
    assert len(state.sft_loss_history) == 2
    assert len(state.grpo_reward_history) == 2
    assert state.metrics["total_improvement"] == 0.0

    # Verify events were emitted for each stage
    stages_seen = {e["stage"] for e in state.events}
    assert "initializing" in stages_seen
    assert "loading-model" in stages_seen
    assert "benchmarking-baseline" in stages_seen
    assert "sft-training" in stages_seen
    assert "benchmarking-post-sft" in stages_seen
    assert "grpo-training" in stages_seen
    assert "benchmarking-post-grpo" in stages_seen
    assert "done" in stages_seen


def test_pipeline_handles_model_load_failure(hw):
    """If model loading fails, the pipeline should mark the run as failed."""
    config = RunConfig(
        model_id="deepseek-r1-distill-qwen-1.5b",
        taskset_id="il-reasoning-v1",
        benchmark_tasks=1,
    )
    state = create_run(config)

    with patch("iloptimus.core.inference.load_model", side_effect=RuntimeError("Network error")):
        asyncio.run(run_pipeline(state.id, config, hw))

    assert state.status == "failed"
    assert any(e["level"] == "error" for e in state.events)


# ---------------------------------------------------------------------------
# Server tests
# ---------------------------------------------------------------------------

def test_server_app_creates():
    from iloptimus.server import create_app
    app = create_app()
    paths = {r.path for r in app.routes if hasattr(r, "path")}
    assert "/api/health" in paths
    assert "/api/models" in paths
    assert "/api/tasksets" in paths
    assert "/api/runs" in paths
