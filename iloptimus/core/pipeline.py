"""IL pipeline runner — orchestrates real SFT + GRPO with live SSE streaming.

Runs the full IL pipeline:
1. Load model (mlx_lm with quantization)
2. Baseline benchmark (real inference + grading)
3. SFT training (LoRA fine-tuning on benchmark traces)
4. Post-SFT benchmark
5. GRPO RL training (group-relative policy optimization)
6. Post-GRPO benchmark

All MLX operations run in a thread pool to avoid blocking the async event loop.
Events are streamed via asyncio queues for the frontend to consume.
"""

from __future__ import annotations

import asyncio
import functools
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import AsyncGenerator, Optional

from .hardware import HardwareInfo
from .models import ModelInfo, get_model
from .tasksets import TasksetInfo, get_taskset


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunStage(str, Enum):
    INITIALIZING = "initializing"
    LOADING_MODEL = "loading-model"
    BENCHMARKING_BASELINE = "benchmarking-baseline"
    SFT_TRAINING = "sft-training"
    BENCHMARKING_POST_SFT = "benchmarking-post-sft"
    GRPO_TRAINING = "grpo-training"
    BENCHMARKING_POST_GRPO = "benchmarking-post-grpo"
    DONE = "done"


@dataclass
class RunConfig:
    model_id: str
    taskset_id: str
    backend: str = "mlx"
    precision: str = "int4"
    sft_iters: int = 100
    sft_lr: float = 1e-4
    grpo_iters: int = 50
    grpo_group_size: int = 4
    grpo_lr: float = 1e-5
    grpo_temperature: float = 0.6
    max_seq_length: int = 768
    benchmark_tasks: int = 12
    rollouts_per_example: int = 4
    max_reasoning_tokens: int = 512
    max_answer_tokens: int = 512


@dataclass
class LogEvent:
    timestamp: float
    stage: str
    level: str
    message: str
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunState:
    id: str
    config: RunConfig
    status: str = "pending"
    stage: str = "initializing"
    progress: float = 0.0
    started_at: float = 0.0
    elapsed_seconds: float = 0.0
    events: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    baseline_accuracy: float = 0.0
    post_sft_accuracy: float = 0.0
    post_grpo_accuracy: float = 0.0
    sft_loss_history: list[float] = field(default_factory=list)
    grpo_reward_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "started_at": self.started_at,
            "elapsed_seconds": self.elapsed_seconds,
            "metrics": self.metrics,
            "baseline_accuracy": self.baseline_accuracy,
            "post_sft_accuracy": self.post_sft_accuracy,
            "post_grpo_accuracy": self.post_grpo_accuracy,
            "sft_loss_history": self.sft_loss_history,
            "grpo_reward_history": self.grpo_reward_history,
            "config": asdict(self.config),
        }


_runs: dict[str, RunState] = {}
_event_queues: dict[str, asyncio.Queue] = {}


def create_run(config: RunConfig) -> RunState:
    run_id = uuid.uuid4().hex[:12]
    state = RunState(id=run_id, config=config, started_at=time.time())
    _runs[run_id] = state
    _event_queues[run_id] = asyncio.Queue()
    return state


def get_run(run_id: str) -> Optional[RunState]:
    return _runs.get(run_id)


def get_all_runs() -> list[RunState]:
    return list(_runs.values())


def _emit(run_id: str, stage: str, level: str, message: str, **data) -> LogEvent:
    event = LogEvent(timestamp=time.time(), stage=stage, level=level, message=message, data=data)
    state = _runs.get(run_id)
    if state:
        state.events.append(event.to_dict())
    queue = _event_queues.get(run_id)
    if queue:
        queue.put_nowait(event.to_dict())
    return event


async def _stream_events(run_id: str) -> AsyncGenerator[dict, None]:
    queue = _event_queues.get(run_id)
    if not queue:
        return
    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=30.0)
            yield event
            state = _runs.get(run_id)
            if state and state.status in ("completed", "failed", "cancelled"):
                while not queue.empty():
                    yield queue.get_nowait()
                return
        except asyncio.TimeoutError:
            yield {"timestamp": time.time(), "stage": "heartbeat", "level": "info", "message": "heartbeat", "data": {}}


async def _update_progress(run_id: str, progress: float, stage: str = ""):
    state = _runs.get(run_id)
    if state:
        state.progress = progress
        if stage:
            state.stage = stage
        state.elapsed_seconds = time.time() - state.started_at


async def _run_in_executor(func, *args, **kwargs):
    """Run a sync function in the thread pool to avoid blocking the event loop."""
    loop = asyncio.get_running_loop()
    if kwargs:
        func = functools.partial(func, **kwargs)
    return await loop.run_in_executor(None, func, *args)


# ---------------------------------------------------------------------------
# Real pipeline stages
# ---------------------------------------------------------------------------

async def _load_model_stage(run_id: str, config: RunConfig, model: ModelInfo) -> object:
    """Stage 2: Load the model via mlx_lm."""
    from .inference import load_model

    _emit(run_id, "loading-model", "info", f"Loading {model.huggingface_id} ({config.precision})...")
    await _update_progress(run_id, 0.05, "loading-model")

    handle = await _run_in_executor(
        load_model,
        huggingface_id=model.huggingface_id,
        precision=config.precision,
    )

    _emit(
        run_id, "loading-model", "success",
        f"Model loaded: {model.name} ({config.precision}, ~{model.int4_gb if config.precision == 'int4' else model.fp16_gb:.1f}GB)",
    )
    await _update_progress(run_id, 0.1, "loading-model")
    return handle


async def _benchmark_stage(
    run_id: str, config: RunConfig, handle, domain: str, phase: str, progress_start: float, progress_end: float,
) -> float:
    """Run a real benchmark: inference + grading on each task."""
    from .benchmark import run_benchmark

    n = config.benchmark_tasks
    _emit(run_id, f"benchmarking-{phase}", "info", f"Running {phase} benchmark on {n} tasks...")
    await _update_progress(run_id, progress_start, f"benchmarking-{phase}")

    # Capture the running event loop so the thread-pool callback can schedule back onto it
    loop = asyncio.get_running_loop()

    # Callback for per-task progress (called from the thread pool)
    task_results_seen = []
    def on_task_complete(idx, total, result):
        task_results_seen.append(result)
        acc = sum(r.correctness for r in task_results_seen) / len(task_results_seen)
        asyncio.run_coroutine_threadsafe(
            _emit_and_progress(run_id, f"benchmarking-{phase}", idx, total, result, acc, progress_start, progress_end),
            loop,
        )

    result = await _run_in_executor(
        run_benchmark,
        handle,
        domain=domain,
        num_tasks=n,
        max_reasoning_tokens=config.max_reasoning_tokens,
        max_answer_tokens=config.max_answer_tokens,
        on_task_complete=on_task_complete,
    )

    _emit(
        run_id, f"benchmarking-{phase}", "success",
        f"[{phase}] Accuracy: {result.accuracy:.1%} | Mean score: {result.mean_score:.3f} | Tokens/s: {result.mean_tokens_per_sec:.1f}",
        accuracy=result.accuracy,
        mean_score=result.mean_score,
        tokens_per_sec=result.mean_tokens_per_sec,
        peak_memory_gb=result.peak_memory_gb,
    )
    await _update_progress(run_id, progress_end, f"benchmarking-{phase}")
    return result.accuracy


async def _emit_and_progress(run_id, stage, idx, total, result, acc, p_start, p_end):
    """Emit a benchmark task completion event and update progress."""
    _emit(
        run_id, stage, "metric",
        f"Task {idx+1}/{total}: score={result.score:.3f} correctness={result.correctness:.1%} ({result.tokens_per_sec:.0f} tok/s)",
        task=idx + 1, total=total, score=result.score, correctness=result.correctness,
        accuracy=acc, tokens_per_sec=result.tokens_per_sec,
    )
    progress = p_start + (p_end - p_start) * (idx + 1) / total
    await _update_progress(run_id, progress, stage)


async def _sft_stage(run_id: str, config: RunConfig, handle, domain: str) -> tuple[list[float], str]:
    """Stage 4: Run real SFT training."""
    from .sft import generate_sft_data, run_sft, SFTConfig

    _emit(run_id, "sft-training", "info", "Generating SFT training data from benchmark traces...")
    await _update_progress(run_id, 0.2, "sft-training")

    loop = asyncio.get_running_loop()

    # Generate SFT data from benchmark
    def on_data_progress(done, total):
        _emit(run_id, "sft-training", "info", f"Generating SFT data: {done}/{total} tasks processed")

    examples = await _run_in_executor(
        generate_sft_data,
        handle,
        domain=domain,
        num_tasks=config.benchmark_tasks,
        max_reasoning_tokens=config.max_reasoning_tokens,
        max_answer_tokens=config.max_answer_tokens,
        on_progress=on_data_progress,
    )

    _emit(run_id, "sft-training", "info", f"Generated {len(examples)} SFT examples from correct responses")

    if not examples:
        _emit(run_id, "sft-training", "warn", "No correct responses found for SFT — skipping SFT stage")
        return [0.0], None

    # Run SFT training
    sft_config = SFTConfig(
        learning_rate=config.sft_lr,
        num_iters=config.sft_iters,
        memory_limit_gb=5.0,
    )

    losses: list[float] = []
    total = config.sft_iters

    def on_sft_metrics(metrics):
        losses.append(metrics.loss)
        asyncio.run_coroutine_threadsafe(
            _emit_sft_metrics(run_id, metrics, total),
            loop,
        )

    adapter_path = await _run_in_executor(
        run_sft,
        handle,
        examples,
        config=sft_config,
        adapter_path=f"il_sft_adapters_{run_id}",
        on_metrics=on_sft_metrics,
    )

    _emit(run_id, "sft-training", "success", f"SFT complete. Final loss: {losses[-1]:.4f}", final_loss=losses[-1])
    await _update_progress(run_id, 0.45, "sft-training")
    return losses, adapter_path


async def _emit_sft_metrics(run_id, metrics, total):
    _emit(
        run_id, "sft-training", "metric",
        f"SFT iter {metrics.iteration+1}/{total}: loss={metrics.loss:.4f} | mem={metrics.peak_memory_gb:.1f}GB",
        iter=metrics.iteration + 1, total=total, loss=metrics.loss,
        peak_memory_gb=metrics.peak_memory_gb,
    )
    progress = 0.2 + 0.25 * (metrics.iteration + 1) / total
    await _update_progress(run_id, progress, "sft-training")


async def _grpo_stage(run_id: str, config: RunConfig, handle, domain: str, adapter_path: str | None) -> list[float]:
    """Stage 6: Run real GRPO RL training."""
    from .grpo import GRPOTrainer, GRPOConfig
    from .grader import build_prompt, grade_response, get_num_tasks

    _emit(run_id, "grpo-training", "info", f"Starting GRPO RL training ({config.grpo_iters} iterations)...")
    _emit(run_id, "grpo-training", "info", f"Group size: {config.grpo_group_size} | Temperature: {config.grpo_temperature}")
    await _update_progress(run_id, 0.55, "grpo-training")

    loop = asyncio.get_running_loop()

    grpo_config = GRPOConfig(
        learning_rate=config.grpo_lr,
        group_size=config.grpo_group_size,
        temperature=config.grpo_temperature,
        thinking_tokens=config.max_reasoning_tokens,
        prediction_tokens=config.max_answer_tokens,
    )

    # Create trainer
    trainer = await _run_in_executor(
        GRPOTrainer,
        handle.model,
        handle.tokenizer,
        grpo_config,
        f"il_grpo_adapters_{run_id}",
    )

    rewards: list[float] = []
    total = config.grpo_iters
    num_tasks = get_num_tasks(domain)

    for i in range(total):
        # Pick a task for this iteration (cycle through tasks)
        task_idx = i % num_tasks
        prompt = await _run_in_executor(build_prompt, domain, task_idx)

        def grade_fn(response: str, _domain=domain, _idx=task_idx) -> float:
            return grade_response(_domain, _idx, response).score

        def on_metrics(metrics):
            asyncio.run_coroutine_threadsafe(
                _emit_grpo_metrics(run_id, metrics, total),
                loop,
            )

        metrics = await _run_in_executor(
            trainer.train_step,
            prompt,
            grade_fn,
            on_metrics,
        )
        rewards.append(metrics.mean_reward)

    # Save final adapter
    await _run_in_executor(trainer.save)

    _emit(
        run_id, "grpo-training", "success",
        f"GRPO complete. Final reward: {rewards[-1]:.4f} | Peak mem: {rewards and 'N/A'}",
        final_reward=rewards[-1],
    )
    await _update_progress(run_id, 0.85, "grpo-training")
    return rewards


async def _emit_grpo_metrics(run_id, metrics, total):
    _emit(
        run_id, "grpo-training", "metric",
        f"GRPO iter {metrics.iteration+1}/{total}: reward={metrics.mean_reward:.4f} ± {metrics.std_reward:.4f} | loss={metrics.loss:.4f} | mem={metrics.peak_memory_gb:.1f}GB | {metrics.total_time:.1f}s",
        iter=metrics.iteration + 1, total=total,
        reward=metrics.mean_reward, std_reward=metrics.std_reward,
        loss=metrics.loss, peak_memory_gb=metrics.peak_memory_gb,
        mean_correctness=metrics.mean_correctness,
    )
    progress = 0.55 + 0.30 * (metrics.iteration + 1) / total
    await _update_progress(run_id, progress, "grpo-training")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(run_id: str, config: RunConfig, hw: HardwareInfo):
    """Main pipeline runner. Streams events via the event queue."""
    state = _runs.get(run_id)
    if not state:
        return

    state.status = "running"
    model = get_model(config.model_id)
    taskset = get_taskset(config.taskset_id)

    if not model:
        _emit(run_id, "initializing", "error", f"Model {config.model_id} not found")
        state.status = "failed"
        return
    if not taskset:
        _emit(run_id, "initializing", "error", f"Taskset {config.taskset_id} not found")
        state.status = "failed"
        return

    domain = taskset.domain

    try:
        # ---- Stage 1: Initializing ----
        _emit(run_id, "initializing", "info", f"Starting IL pipeline: {model.name} on {taskset.name}")
        _emit(run_id, "initializing", "info", f"Backend: {config.backend} | Precision: {config.precision}")
        _emit(run_id, "initializing", "info", f"Hardware: {hw.gpu.name} ({hw.total_memory_gb:.1f}GB available)")
        _emit(run_id, "initializing", "info", f"SFT: {config.sft_iters} iters @ lr={config.sft_lr}")
        _emit(run_id, "initializing", "info", f"GRPO: {config.grpo_iters} iters, group_size={config.grpo_group_size}")
        _emit(run_id, "initializing", "info", f"Taskset domain: {domain} ({taskset.num_tasks} tasks)")
        await _update_progress(run_id, 0.0, "initializing")
        await asyncio.sleep(0.3)

        # ---- Stage 2: Load model ----
        handle = await _load_model_stage(run_id, config, model)

        # ---- Stage 3: Baseline benchmark ----
        baseline_acc = await _benchmark_stage(
            run_id, config, handle, domain, "baseline", 0.1, 0.2,
        )
        state.baseline_accuracy = baseline_acc
        state.metrics["baseline_accuracy"] = baseline_acc

        # ---- Stage 4: SFT training ----
        sft_losses, sft_adapter_path = await _sft_stage(run_id, config, handle, domain)
        state.sft_loss_history = sft_losses

        # ---- Stage 5: Post-SFT benchmark ----
        # If we have SFT adapters, reload the model with them
        if sft_adapter_path:
            from .inference import load_model
            _emit(run_id, "benchmarking-post-sft", "info", "Reloading model with SFT adapters...")
            handle = await _run_in_executor(
                load_model,
                huggingface_id=model.huggingface_id,
                precision=config.precision,
                adapter_path=sft_adapter_path,
            )

        post_sft_acc = await _benchmark_stage(
            run_id, config, handle, domain, "post-sft", 0.45, 0.55,
        )
        state.post_sft_accuracy = post_sft_acc
        improvement = post_sft_acc - baseline_acc
        _emit(
            run_id, "benchmarking-post-sft", "success",
            f"Post-SFT accuracy: {post_sft_acc:.1%} ({improvement:+.1%} vs baseline)",
            accuracy=post_sft_acc, improvement=improvement,
        )
        state.metrics["post_sft_accuracy"] = post_sft_acc
        state.metrics["sft_improvement"] = improvement

        # ---- Stage 6: GRPO RL training ----
        grpo_rewards = await _grpo_stage(run_id, config, handle, domain, sft_adapter_path)
        state.grpo_reward_history = grpo_rewards

        # ---- Stage 7: Post-GRPO benchmark ----
        # Reload with GRPO adapters
        grpo_adapter_path = f"il_grpo_adapters_{run_id}"
        from .inference import load_model
        _emit(run_id, "benchmarking-post-grpo", "info", "Reloading model with GRPO adapters...")
        handle = await _run_in_executor(
            load_model,
            huggingface_id=model.huggingface_id,
            precision=config.precision,
            adapter_path=grpo_adapter_path,
        )

        post_grpo_acc = await _benchmark_stage(
            run_id, config, handle, domain, "post-grpo", 0.85, 0.95,
        )
        state.post_grpo_accuracy = post_grpo_acc
        total_improvement = post_grpo_acc - baseline_acc
        _emit(
            run_id, "benchmarking-post-grpo", "success",
            f"Post-GRPO accuracy: {post_grpo_acc:.1%} ({total_improvement:+.1%} vs baseline)",
            accuracy=post_grpo_acc, total_improvement=total_improvement,
        )
        state.metrics["post_grpo_accuracy"] = post_grpo_acc
        state.metrics["total_improvement"] = total_improvement

        # ---- Done ----
        _emit(
            run_id, "done", "success",
            f"IL pipeline complete! {baseline_acc:.1%} -> {post_grpo_acc:.1%} ({total_improvement:+.1%})",
            baseline=baseline_acc, final=post_grpo_acc, improvement=total_improvement,
        )
        state.status = "completed"
        await _update_progress(run_id, 1.0, "done")

    except asyncio.CancelledError:
        state.status = "cancelled"
        _emit(run_id, state.stage, "warn", "Run cancelled by user")
    except Exception as e:
        state.status = "failed"
        _emit(run_id, state.stage, "error", f"Pipeline failed: {e}", error=str(e))
        import traceback
        _emit(run_id, state.stage, "error", traceback.format_exc())
