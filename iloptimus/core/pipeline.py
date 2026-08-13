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
import concurrent.futures
import functools
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import AsyncGenerator, Optional

from .hardware import HardwareInfo
from .models import ModelInfo, get_model
from .storage import atomic_write_json, run_dir, runs_dir
from .tasksets import get_taskset


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
    sft_iters: int = 20  # reduced from 100 — SFT just teaches format, GRPO does the heavy lifting
    sft_lr: float = 1e-3  # SGD with higher LR for visible changes
    sft_task_offset: int = 0
    sft_tasks: int | None = None
    sft_batch_size: int = 1
    sft_grad_accumulation_steps: int = 1
    sft_lora_rank: int = 8
    sft_lora_layers: int = 8
    sft_lora_scale: float = 20.0
    sft_lora_targets: tuple[str, ...] = (
        "self_attn.q_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    )
    sft_optimizer: str = "adamw"
    sft_mask_prompt: bool = True
    sft_grad_checkpoint: bool = False
    sft_memory_limit_gb: float = 3.0
    sft_compile_bucket_size: int = 128
    sft_clear_cache_threshold_gb: float = 1.0
    sft_prefix_cache: bool = False
    sft_seed: int = 0
    grpo_iters: int = 10  # reduced from 50 — research shows diminishing returns after 10-20 iters
    grpo_group_size: int = 2  # reduced from 4 — 2-GRPO matches 16-GRPO per recent research
    grpo_lr: float = 1e-3  # SGD with higher LR for visible changes
    grpo_temperature: float = 0.6
    max_seq_length: int = 512
    benchmark_tasks: int = 12
    rollouts_per_example: int = 4
    max_reasoning_tokens: int = 256  # reduced from 512 — most IL tasks don't need 512 reasoning tokens
    max_answer_tokens: int = 128  # reduced from 512 — answers are short


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
    # Reasoning traces for before/after comparison
    baseline_traces: list[dict] = field(default_factory=list)
    post_sft_traces: list[dict] = field(default_factory=list)
    post_grpo_traces: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        # Sanitize NaN/inf to 0.0 — JSON doesn't support non-finite floats
        def _safe(v):
            if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
                return 0.0
            return v

        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "started_at": self.started_at,
            "elapsed_seconds": self.elapsed_seconds,
            "events": self.events,
            "metrics": self.metrics,
            "baseline_accuracy": _safe(self.baseline_accuracy),
            "post_sft_accuracy": _safe(self.post_sft_accuracy),
            "post_grpo_accuracy": _safe(self.post_grpo_accuracy),
            "sft_loss_history": [_safe(x) for x in self.sft_loss_history],
            "grpo_reward_history": [_safe(x) for x in self.grpo_reward_history],
            "baseline_traces": self.baseline_traces,
            "post_sft_traces": self.post_sft_traces,
            "post_grpo_traces": self.post_grpo_traces,
            "config": asdict(self.config),
            "artifact_dir": str(run_dir(self.id)),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RunState":
        fields = {key: value for key, value in payload.items() if key in cls.__dataclass_fields__ and key != "config"}
        return cls(config=RunConfig(**payload["config"]), **fields)


def _load_saved_runs() -> dict[str, RunState]:
    saved: dict[str, RunState] = {}
    root = runs_dir()
    if not root.exists():
        return saved
    for path in root.glob("*/run.json"):
        try:
            state = RunState.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError, KeyError):
            continue
        if state.status in {"pending", "running"}:
            state.status = "failed"
            state.events.append(
                LogEvent(
                    timestamp=time.time(),
                    stage=state.stage,
                    level="error",
                    message="Run was interrupted when IL Optimus stopped",
                ).to_dict()
            )
        saved[state.id] = state
    return saved


_runs: dict[str, RunState] = _load_saved_runs()
_event_queues: dict[str, asyncio.Queue] = {}


def _persist_state(state: RunState) -> None:
    atomic_write_json(run_dir(state.id) / "run.json", state.to_dict())


def create_run(config: RunConfig) -> RunState:
    run_id = uuid.uuid4().hex[:12]
    state = RunState(id=run_id, config=config, started_at=time.time())
    _runs[run_id] = state
    _event_queues[run_id] = asyncio.Queue()
    folder = run_dir(run_id)
    folder.mkdir(parents=True, exist_ok=False)
    atomic_write_json(folder / "config.json", asdict(config))
    _persist_state(state)
    return state


def get_run(run_id: str) -> Optional[RunState]:
    return _runs.get(run_id)


def get_all_runs() -> list[RunState]:
    return sorted(_runs.values(), key=lambda state: state.started_at, reverse=True)


def _emit(run_id: str, stage: str, level: str, message: str, **data) -> LogEvent:
    event = LogEvent(timestamp=time.time(), stage=stage, level=level, message=message, data=data)
    state = _runs.get(run_id)
    if state:
        state.events.append(event.to_dict())
        events_path = run_dir(run_id) / "events.jsonl"
        events_path.parent.mkdir(parents=True, exist_ok=True)
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        _persist_state(state)
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
        _persist_state(state)


# Single-thread executor for MLX operations — MLX arrays are thread-local
# and cannot be shared across threads, so all MLX work must run on the same thread
_mlx_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
_pipeline_worker_lock = asyncio.Lock()


async def _run_in_executor(func, *args, **kwargs):
    """Run a sync function in the single-thread MLX executor to avoid blocking the event loop."""
    loop = asyncio.get_running_loop()
    if kwargs:
        func = functools.partial(func, **kwargs)
    return await loop.run_in_executor(_mlx_executor, func, *args)


def _sync_state_from_disk(run_id: str) -> RunState | None:
    """Merge a worker's durable state into this server process.

    MLX may terminate a process at the native Metal layer under memory pressure.
    Training therefore runs in a child process; this function keeps the API's
    in-memory object (and any open SSE stream referring to it) synchronized with
    the atomic run.json written by that child.
    """
    state = _runs.get(run_id)
    path = run_dir(run_id) / "run.json"
    try:
        fresh = RunState.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TypeError, KeyError):
        return state
    if state is None:
        _runs[run_id] = fresh
        return fresh
    for name in RunState.__dataclass_fields__:
        setattr(state, name, getattr(fresh, name))
    return state


async def run_pipeline_subprocess(run_id: str) -> RunState | None:
    """Run a pipeline in an isolated worker and mirror progress into the API.

    A Python exception is already handled inside :func:`run_pipeline`. Native
    MLX/Metal aborts are different: they can kill the interpreter outright.
    Isolating the full model lifecycle guarantees that a bad training run cannot
    take down chat, saved workspaces, or localhost itself.
    """
    state = _runs.get(run_id)
    if state is None:
        return None

    worker_log = run_dir(run_id) / "worker.log"
    observed_events = len(state.events)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    async with _pipeline_worker_lock:
        with worker_log.open("ab") as output:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "iloptimus.pipeline_worker",
                run_id,
                stdout=output,
                stderr=output,
                env=env,
            )
            try:
                while process.returncode is None:
                    try:
                        await asyncio.wait_for(process.wait(), timeout=0.5)
                    except asyncio.TimeoutError:
                        pass
                    state = _sync_state_from_disk(run_id) or state
                    new_events = state.events[observed_events:]
                    observed_events = len(state.events)
                    queue = _event_queues.get(run_id)
                    if queue:
                        for event in new_events:
                            queue.put_nowait(event)
            except asyncio.CancelledError:
                if process.returncode is None:
                    process.terminate()
                    await process.wait()
                raise

    state = _sync_state_from_disk(run_id) or state
    if process.returncode and state.status not in {
        RunStatus.COMPLETED.value,
        RunStatus.FAILED.value,
        RunStatus.CANCELLED.value,
    }:
        state.status = RunStatus.FAILED.value
        state.elapsed_seconds = time.time() - state.started_at
        _emit(
            run_id,
            state.stage,
            "error",
            f"Training worker exited unexpectedly (code {process.returncode}). "
            f"The app stayed online; diagnostics are in {worker_log}",
            worker_log=str(worker_log),
            exit_code=process.returncode,
        )
        _persist_state(state)
    return state


# ---------------------------------------------------------------------------
# Real pipeline stages
# ---------------------------------------------------------------------------


async def _load_model_stage(run_id: str, config: RunConfig, model: ModelInfo) -> object:
    """Stage 2: Load the model via mlx_lm.

    Uses QLoRA — trains directly on int4 quantized models without dequantization.
    MLX's QuantizedLinear supports gradients natively, saving ~15s dequant time
    and ~2.3GB memory (int4 is 1.2GB vs fp16 is 3.5GB).
    """
    from .inference import load_model
    from .model_store import resolve_model_source

    _emit(run_id, "loading-model", "info", f"Loading {model.huggingface_id} ({config.precision})...")
    await _update_progress(run_id, 0.05, "loading-model")

    source = resolve_model_source(model.id, config.precision, config.backend)
    if not source:
        raise RuntimeError("Model is not downloaded. Open Model Library and download it first.")

    handle = await _run_in_executor(
        load_model,
        huggingface_id=model.huggingface_id,
        precision=config.precision,
        dequantize=False,  # QLoRA — no dequantization needed
        source_override=source,
    )

    _emit(
        run_id,
        "loading-model",
        "success",
        f"Model loaded: {model.name} ({config.precision} QLoRA, ~{model.int4_gb if config.precision == 'int4' else model.fp16_gb:.1f}GB)",
    )
    await _update_progress(run_id, 0.1, "loading-model")
    return handle


async def _benchmark_stage(
    run_id: str,
    config: RunConfig,
    handle,
    domain: str,
    phase: str,
    progress_start: float,
    progress_end: float,
) -> tuple[float, list[dict]]:
    """Run a real benchmark: inference + grading on each task.

    Returns (accuracy, reasoning_traces) where reasoning_traces is a list of
    dicts with task_idx, reasoning, answer, score, correctness for each task.
    """
    from .benchmark import run_benchmark
    from .grader import get_num_tasks

    n = min(config.benchmark_tasks, get_num_tasks(domain))
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

    # Capture reasoning traces for before/after comparison
    traces = [
        {
            "task_idx": r.task_idx,
            "reasoning": r.reasoning,
            "answer": r.answer,
            "score": r.score,
            "correctness": r.correctness,
            "reasoning_quality": r.reasoning_quality,
            "forced_answer": r.forced_answer,
            "tokens_generated": r.tokens_generated,
        }
        for r in result.task_results
    ]

    _emit(
        run_id,
        f"benchmarking-{phase}",
        "success",
        f"[{phase}] Accuracy: {result.accuracy:.1%} | Mean score: {result.mean_score:.3f} | Tokens/s: {result.mean_tokens_per_sec:.1f}",
        accuracy=result.accuracy,
        mean_score=result.mean_score,
        tokens_per_sec=result.mean_tokens_per_sec,
        peak_memory_gb=result.peak_memory_gb,
    )
    await _update_progress(run_id, progress_end, f"benchmarking-{phase}")
    return result.accuracy, traces


async def _emit_and_progress(run_id, stage, idx, total, result, acc, p_start, p_end):
    """Emit a benchmark task completion event and update progress."""
    _emit(
        run_id,
        stage,
        "metric",
        f"Task {idx + 1}/{total}: score={result.score:.3f} correctness={result.correctness:.1%} ({result.tokens_per_sec:.0f} tok/s)",
        task=idx + 1,
        total=total,
        score=result.score,
        correctness=result.correctness,
        accuracy=acc,
        tokens_per_sec=result.tokens_per_sec,
    )
    progress = p_start + (p_end - p_start) * (idx + 1) / total
    await _update_progress(run_id, progress, stage)


async def _sft_stage(run_id: str, config: RunConfig, handle, domain: str) -> tuple[list[float], str]:
    """Stage 4: Run real SFT training."""
    from .sft import SFTConfig, generate_sft_data, run_sft

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
        num_tasks=config.sft_tasks or config.benchmark_tasks,
        task_offset=config.sft_task_offset,
        max_reasoning_tokens=config.max_reasoning_tokens,
        max_answer_tokens=config.max_answer_tokens,
        on_progress=on_data_progress,
    )

    _emit(run_id, "sft-training", "info", f"Generated {len(examples)} SFT examples")

    if not examples:
        _emit(run_id, "sft-training", "warn", "No SFT examples could be generated — skipping SFT stage")
        return [0.0], None

    # Run SFT training
    sft_config = SFTConfig(
        learning_rate=config.sft_lr,
        num_iters=config.sft_iters,
        batch_size=config.sft_batch_size,
        grad_accumulation_steps=config.sft_grad_accumulation_steps,
        memory_limit_gb=config.sft_memory_limit_gb,
        lora_rank=config.sft_lora_rank,
        lora_layers=config.sft_lora_layers,
        lora_targets=tuple(config.sft_lora_targets),
        lora_scale=config.sft_lora_scale,
        grad_clip=1.0,
        max_seq_length=config.max_seq_length,
        optimizer=config.sft_optimizer,
        mask_prompt=config.sft_mask_prompt,
        grad_checkpoint=config.sft_grad_checkpoint,
        compile_bucket_size=config.sft_compile_bucket_size,
        clear_cache_threshold_gb=config.sft_clear_cache_threshold_gb,
        prefix_cache=config.sft_prefix_cache,
        seed=config.sft_seed,
        steps_per_eval=max(1, min(10, config.sft_iters)),
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
        adapter_path=str(run_dir(run_id) / "adapters" / "sft"),
        on_metrics=on_sft_metrics,
    )

    _emit(run_id, "sft-training", "success", f"SFT complete. Final loss: {losses[-1]:.4f}", final_loss=losses[-1])
    await _update_progress(run_id, 0.45, "sft-training")
    return losses, adapter_path


async def _emit_sft_metrics(run_id, metrics, total):
    _emit(
        run_id,
        "sft-training",
        "metric",
        f"SFT iter {metrics.iteration + 1}/{total}: loss={metrics.loss:.4f} | "
        f"{metrics.iterations_per_second:.3f} step/s | {metrics.tokens_per_second:.1f} tok/s | "
        f"mem={metrics.peak_memory_gb:.1f}GB",
        iter=metrics.iteration + 1,
        total=total,
        loss=metrics.loss,
        peak_memory_gb=metrics.peak_memory_gb,
        iterations_per_second=metrics.iterations_per_second,
        tokens_per_second=metrics.tokens_per_second,
        trained_tokens=metrics.trained_tokens,
    )
    progress = 0.2 + 0.25 * (metrics.iteration + 1) / total
    await _update_progress(run_id, progress, "sft-training")


async def _grpo_stage(run_id: str, config: RunConfig, handle, domain: str, adapter_path: str | None) -> list[float]:
    """Stage 6: Run real GRPO RL training."""
    from .grader import build_prompt, get_num_tasks, grade_response
    from .grpo import GRPOConfig, GRPOTrainer

    _emit(run_id, "grpo-training", "info", f"Starting GRPO RL training ({config.grpo_iters} iterations)...")
    _emit(
        run_id,
        "grpo-training",
        "info",
        f"Group size: {config.grpo_group_size} | Temperature: {config.grpo_temperature}",
    )
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
        str(run_dir(run_id) / "adapters" / "grpo"),
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
        run_id,
        "grpo-training",
        "success",
        f"GRPO complete. Final reward: {rewards[-1]:.4f} | Peak mem: {rewards and 'N/A'}",
        final_reward=rewards[-1],
    )
    await _update_progress(run_id, 0.85, "grpo-training")
    return rewards


async def _emit_grpo_metrics(run_id, metrics, total):
    _emit(
        run_id,
        "grpo-training",
        "metric",
        f"GRPO iter {metrics.iteration + 1}/{total}: reward={metrics.mean_reward:.4f} ± {metrics.std_reward:.4f} | loss={metrics.loss:.4f} | mem={metrics.peak_memory_gb:.1f}GB | {metrics.total_time:.1f}s",
        iter=metrics.iteration + 1,
        total=total,
        reward=metrics.mean_reward,
        std_reward=metrics.std_reward,
        loss=metrics.loss,
        peak_memory_gb=metrics.peak_memory_gb,
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
    pipeline_mode = "IL + RL"
    if domain.startswith("custom:"):
        from .environments import get_environment

        environment = get_environment(domain.split(":", 1)[1])
        if environment:
            pipeline_mode = environment["mode"]

    try:
        # ---- Stage 1: Initializing ----
        _emit(run_id, "initializing", "info", f"Starting {pipeline_mode} pipeline: {model.name} on {taskset.name}")
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
        baseline_acc, baseline_traces = await _benchmark_stage(
            run_id,
            config,
            handle,
            domain,
            "baseline",
            0.1,
            0.2,
        )
        state.baseline_accuracy = baseline_acc
        state.baseline_traces = baseline_traces
        state.metrics["baseline_accuracy"] = baseline_acc
        _emit(
            run_id,
            "benchmarking-baseline",
            "info",
            f"Captured {len(baseline_traces)} baseline reasoning traces for comparison",
        )

        # ---- Stage 4: SFT training ----
        sft_losses, sft_adapter_path = await _sft_stage(run_id, config, handle, domain)
        state.sft_loss_history = sft_losses

        # ---- Stage 5: Post-SFT benchmark ----
        # run_sft updates the live LoRA layers in-place. Re-loading that same
        # adapter would try to wrap LoRALinear a second time and is invalid.
        if sft_adapter_path:
            _emit(run_id, "benchmarking-post-sft", "info", "Evaluating the trained in-memory SFT adapter...")

        post_sft_acc, post_sft_traces = await _benchmark_stage(
            run_id,
            config,
            handle,
            domain,
            "post-sft",
            0.45,
            0.55,
        )
        state.post_sft_accuracy = post_sft_acc
        state.post_sft_traces = post_sft_traces
        improvement = post_sft_acc - baseline_acc
        _emit(
            run_id,
            "benchmarking-post-sft",
            "success",
            f"Post-SFT accuracy: {post_sft_acc:.1%} ({improvement:+.1%} vs baseline)",
            accuracy=post_sft_acc,
            improvement=improvement,
        )
        state.metrics["post_sft_accuracy"] = post_sft_acc
        state.metrics["sft_improvement"] = improvement

        if pipeline_mode == "IL":
            # A demonstration-only environment has no executable reward objective.
            # Calling this RL would be scientifically incorrect, so stop at QLoRA SFT.
            _emit(run_id, "grpo-training", "info", "IL environment selected: RL/GRPO is intentionally skipped")
            post_grpo_acc = post_sft_acc
            post_grpo_traces = post_sft_traces
            total_improvement = improvement
            state.post_grpo_accuracy = post_grpo_acc
            state.post_grpo_traces = post_grpo_traces
            state.metrics["post_grpo_accuracy"] = post_grpo_acc
            state.metrics["total_improvement"] = total_improvement
        else:
            # ---- Stage 6: GRPO RL training ----
            grpo_rewards = await _grpo_stage(run_id, config, handle, domain, sft_adapter_path)
            state.grpo_reward_history = grpo_rewards

            # ---- Stage 7: Post-GRPO benchmark ----
            # GRPO updates the same live LoRA modules in-place. Loading its
            # checkpoint here would incorrectly wrap LoRALinear a second time.
            _emit(run_id, "benchmarking-post-grpo", "info", "Evaluating the trained in-memory GRPO adapter...")

            post_grpo_acc, post_grpo_traces = await _benchmark_stage(
                run_id,
                config,
                handle,
                domain,
                "post-grpo",
                0.85,
                0.95,
            )
            state.post_grpo_accuracy = post_grpo_acc
            state.post_grpo_traces = post_grpo_traces
            total_improvement = post_grpo_acc - baseline_acc
            _emit(
                run_id,
                "benchmarking-post-grpo",
                "success",
                f"Post-GRPO accuracy: {post_grpo_acc:.1%} ({total_improvement:+.1%} vs baseline)",
                accuracy=post_grpo_acc,
                total_improvement=total_improvement,
            )
            state.metrics["post_grpo_accuracy"] = post_grpo_acc
            state.metrics["total_improvement"] = total_improvement

        # ---- Done ----
        _emit(
            run_id,
            "done",
            "success",
            f"{pipeline_mode} pipeline complete! {baseline_acc:.1%} -> {post_grpo_acc:.1%} ({total_improvement:+.1%})",
            baseline=baseline_acc,
            final=post_grpo_acc,
            improvement=total_improvement,
        )

        # Save reasoning traces to a file for before/after comparison
        traces_path = run_dir(run_id) / "reasoning_traces.json"
        traces_data = {
            "run_id": run_id,
            "model": model.name,
            "taskset": taskset.name,
            "domain": domain,
            "baseline_accuracy": baseline_acc,
            "post_sft_accuracy": post_sft_acc,
            "post_grpo_accuracy": post_grpo_acc,
            "baseline_traces": baseline_traces,
            "post_sft_traces": post_sft_traces,
            "post_grpo_traces": post_grpo_traces,
        }
        atomic_write_json(traces_path, traces_data)
        _emit(run_id, "done", "info", f"Reasoning traces saved to {traces_path}", traces_file=str(traces_path))

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
    finally:
        _persist_state(state)
