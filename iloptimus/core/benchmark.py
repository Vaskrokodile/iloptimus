"""Real benchmark runner — runs inference on each task and grades responses.

For each task in a taskset:
1. Build the prompt
2. Run two-stage inference (reasoning + answer)
3. Grade the response (correctness + reasoning quality)
4. Compute aggregate accuracy

Returns real accuracy numbers, not simulated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .grader import build_prompt, get_num_tasks, grade_response
from .inference import ModelHandle, clear_cache, get_memory_info, run_inference_batch


@dataclass
class TaskResult:
    task_idx: int
    score: float  # IL final score (0.0 to 1.0)
    correctness: float  # raw correctness
    reasoning_quality: float
    elapsed: float
    tokens_generated: int
    tokens_per_sec: float
    forced_answer: bool
    response_preview: str  # first 200 chars of response
    reasoning: str = ""  # full reasoning trace
    answer: str = ""  # full answer text


@dataclass
class BenchmarkResult:
    accuracy: float  # mean correctness across all tasks
    mean_score: float  # mean IL final score
    mean_reasoning_quality: float
    total_elapsed: float
    total_tokens: int
    mean_tokens_per_sec: float
    forced_answer_rate: float
    peak_memory_gb: float
    task_results: list[TaskResult] = field(default_factory=list)


def run_benchmark(
    handle: ModelHandle,
    domain: str,
    num_tasks: int | None = None,
    max_reasoning_tokens: int = 512,
    max_answer_tokens: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.9,
    batch_size: int = 4,
    on_task_complete: Callable[[int, int, TaskResult], None] | None = None,
) -> BenchmarkResult:
    """Run a real benchmark on a taskset.

    Args:
        handle: loaded ModelHandle
        domain: taskset domain ("coding", "reasoning", etc.)
        num_tasks: how many tasks to run (None = all)
        max_reasoning_tokens: token budget for reasoning phase
        max_answer_tokens: token budget for answer phase
        temperature: sampling temperature
        top_p: nucleus sampling threshold
        batch_size: number of prompts to submit together when the backend supports it
        on_task_complete: callback called after each task with (idx, total, result)

    Returns:
        BenchmarkResult with real accuracy and per-task details
    """
    total_available = get_num_tasks(domain)
    n = min(num_tasks or total_available, total_available)
    batch_size = max(1, int(batch_size))

    results: list[TaskResult] = []
    prompts = [build_prompt(domain, index) for index in range(n)]
    t0 = time.perf_counter()

    for start in range(0, n, batch_size):
        prompt_batch = prompts[start : start + batch_size]
        inference_batch = run_inference_batch(
            handle,
            prompt_batch,
            max_reasoning_tokens=max_reasoning_tokens,
            max_answer_tokens=max_answer_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        if len(inference_batch) != len(prompt_batch):
            raise RuntimeError(
                "Batch inference returned a different number of results than prompts"
            )

        for offset, inf in enumerate(inference_batch):
            task_index = start + offset
            graded = grade_response(domain, task_index, inf.text)
            task_result = TaskResult(
                task_idx=task_index,
                score=graded.score,
                correctness=graded.correctness,
                reasoning_quality=graded.reasoning_quality,
                elapsed=inf.elapsed,
                tokens_generated=inf.tokens_generated,
                tokens_per_sec=inf.tokens_per_sec,
                forced_answer=inf.forced_answer,
                response_preview=inf.text[:200],
                reasoning=inf.reasoning,
                answer=inf.answer,
            )
            results.append(task_result)

            if on_task_complete:
                on_task_complete(task_index, n, task_result)

        # Clear backend caches between batches. Sequential fallback backends
        # clear per prompt inside their default batch implementation.
        clear_cache()

    total_elapsed = time.perf_counter() - t0
    total_tokens = sum(r.tokens_generated for r in results)

    accuracy = sum(r.correctness for r in results) / n if n > 0 else 0.0
    mean_score = sum(r.score for r in results) / n if n > 0 else 0.0
    mean_rq = sum(r.reasoning_quality for r in results) / n if n > 0 else 0.0
    forced_rate = sum(1 for r in results if r.forced_answer) / n if n > 0 else 0.0
    mean_tps = total_tokens / total_elapsed if total_elapsed > 0 else 0.0

    mem_info = get_memory_info()
    peak_mem = mem_info.get("peak_memory_gb", 0.0)

    return BenchmarkResult(
        accuracy=accuracy,
        mean_score=mean_score,
        mean_reasoning_quality=mean_rq,
        total_elapsed=total_elapsed,
        total_tokens=total_tokens,
        mean_tokens_per_sec=mean_tps,
        forced_answer_rate=forced_rate,
        peak_memory_gb=peak_mem,
        task_results=results,
    )
