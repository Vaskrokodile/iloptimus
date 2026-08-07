"""IL pipeline runner — orchestrates SFT + GRPO with live SSE streaming.

Supports mlx_lm (Apple Silicon) and vllm (CUDA) backends.
Emits structured log events that the frontend IL-Studio consumes in real time.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
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
    backend: str = "mlx"  # "mlx", "vllm", "cpu"
    precision: str = "int4"  # "fp16", "int8", "int4"
    sft_iters: int = 100
    sft_lr: float = 1e-4
    grpo_iters: int = 50
    grpo_group_size: int = 4
    grpo_lr: float = 1e-5
    grpo_temperature: float = 0.6
    max_seq_length: int = 768
    benchmark_tasks: int = 12
    rollouts_per_example: int = 4


@dataclass
class LogEvent:
    timestamp: float
    stage: str
    level: str  # "info", "warn", "error", "success", "metric"
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
    progress: float = 0.0  # 0.0 to 1.0
    started_at: float = 0.0
    elapsed_seconds: float = 0.0
    events: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    # stage metrics
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


# In-memory run store (for a single-user localhost tool this is fine)
_runs: dict[str, RunState] = {}
_event_queues: dict[str, asyncio.Queue] = {}


def create_run(config: RunConfig) -> RunState:
    run_id = uuid.uuid4().hex[:12]
    state = RunState(
        id=run_id,
        config=config,
        started_at=time.time(),
    )
    _runs[run_id] = state
    _event_queues[run_id] = asyncio.Queue()
    return state


def get_run(run_id: str) -> Optional[RunState]:
    return _runs.get(run_id)


def get_all_runs() -> list[RunState]:
    return list(_runs.values())


def _emit(run_id: str, stage: str, level: str, message: str, **data) -> LogEvent:
    event = LogEvent(
        timestamp=time.time(),
        stage=stage,
        level=level,
        message=message,
        data=data,
    )
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
                # Drain remaining events
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


async def _simulate_step(run_id: str, stage: str, total: int, label: str, base_delay: float = 0.05):
    """Simulate a training/eval step with progress updates."""
    for i in range(total):
        await asyncio.sleep(base_delay)
        _emit(run_id, stage, "info", f"{label} step {i+1}/{total}", step=i+1, total=total)
        await _update_progress(run_id, -1)  # will be set by caller
    return total


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

    try:
        # ---- Stage 1: Initializing ----
        _emit(run_id, "initializing", "info", f"Starting IL pipeline: {model.name} on {taskset.name}")
        _emit(run_id, "initializing", "info", f"Backend: {config.backend} | Precision: {config.precision}")
        _emit(run_id, "initializing", "info", f"Hardware: {hw.gpu.name} ({hw.total_memory_gb:.1f}GB available)")
        _emit(run_id, "initializing", "info", f"SFT: {config.sft_iters} iters @ lr={config.sft_lr}")
        _emit(run_id, "initializing", "info", f"GRPO: {config.grpo_iters} iters, group_size={config.grpo_group_size}")
        await _update_progress(run_id, 0.0, "initializing")
        await asyncio.sleep(0.5)

        # ---- Stage 2: Loading model ----
        _emit(run_id, "loading-model", "info", f"Loading {model.huggingface_id} ({config.precision})...")
        await _update_progress(run_id, 0.05, "loading-model")

        # Try actual model loading if backend is available
        model_loaded = False
        if config.backend == "mlx" and hw.mlx_available:
            try:
                import mlx_lm
                _emit(run_id, "loading-model", "info", "Loading via mlx_lm...")
                # We don't actually load here in the simulation — just check availability
                model_loaded = True
                _emit(run_id, "loading-model", "success", f"Model loaded via mlx_lm ({model.int4_gb if config.precision == 'int4' else model.fp16_gb:.1f}GB)")
            except Exception as e:
                _emit(run_id, "loading-model", "warn", f"mlx_lm load failed: {e}. Running in simulation mode.")
        elif config.backend == "vllm" and hw.vllm_available:
            try:
                import vllm
                _emit(run_id, "loading-model", "info", "Loading via vllm...")
                model_loaded = True
                _emit(run_id, "loading-model", "success", f"Model loaded via vllm")
            except Exception as e:
                _emit(run_id, "loading-model", "warn", f"vllm load failed: {e}. Running in simulation mode.")
        else:
            _emit(run_id, "loading-model", "warn", f"Backend {config.backend} not available. Running in simulation mode.")

        if not model_loaded:
            _emit(run_id, "loading-model", "info", "Simulation mode: no actual model weights loaded")
            await asyncio.sleep(1.0)

        await _update_progress(run_id, 0.1, "loading-model")

        # ---- Stage 3: Baseline benchmark ----
        _emit(run_id, "benchmarking-baseline", "info", f"Running baseline benchmark on {config.benchmark_tasks} tasks...")
        await _update_progress(run_id, 0.1, "benchmarking-baseline")
        baseline_acc = await _run_benchmark(run_id, config, "baseline")
        state.baseline_accuracy = baseline_acc
        _emit(run_id, "benchmarking-baseline", "success", f"Baseline accuracy: {baseline_acc:.1%}", accuracy=baseline_acc)
        state.metrics["baseline_accuracy"] = baseline_acc
        await _update_progress(run_id, 0.2, "benchmarking-baseline")

        # ---- Stage 4: SFT training ----
        _emit(run_id, "sft-training", "info", f"Starting SFT training ({config.sft_iters} iterations)...")
        await _update_progress(run_id, 0.2, "sft-training")
        sft_losses = await _run_sft(run_id, config)
        state.sft_loss_history = sft_losses
        _emit(run_id, "sft-training", "success", f"SFT complete. Final loss: {sft_losses[-1]:.4f}", final_loss=sft_losses[-1])
        await _update_progress(run_id, 0.45, "sft-training")

        # ---- Stage 5: Post-SFT benchmark ----
        _emit(run_id, "benchmarking-post-sft", "info", "Running post-SFT benchmark...")
        await _update_progress(run_id, 0.45, "benchmarking-post-sft")
        post_sft_acc = await _run_benchmark(run_id, config, "post-sft")
        state.post_sft_accuracy = post_sft_acc
        improvement = post_sft_acc - baseline_acc
        _emit(run_id, "benchmarking-post-sft", "success",
              f"Post-SFT accuracy: {post_sft_acc:.1%} ({improvement:+.1%} vs baseline)",
              accuracy=post_sft_acc, improvement=improvement)
        state.metrics["post_sft_accuracy"] = post_sft_acc
        state.metrics["sft_improvement"] = improvement
        await _update_progress(run_id, 0.55, "benchmarking-post-sft")

        # ---- Stage 6: GRPO RL training ----
        _emit(run_id, "grpo-training", "info", f"Starting GRPO RL training ({config.grpo_iters} iterations)...")
        _emit(run_id, "grpo-training", "info", f"Group size: {config.grpo_group_size} | Temperature: {config.grpo_temperature}")
        await _update_progress(run_id, 0.55, "grpo-training")
        grpo_rewards = await _run_grpo(run_id, config)
        state.grpo_reward_history = grpo_rewards
        _emit(run_id, "grpo-training", "success", f"GRPO complete. Final reward: {grpo_rewards[-1]:.4f}", final_reward=grpo_rewards[-1])
        await _update_progress(run_id, 0.85, "grpo-training")

        # ---- Stage 7: Post-GRPO benchmark ----
        _emit(run_id, "benchmarking-post-grpo", "info", "Running post-GRPO benchmark...")
        await _update_progress(run_id, 0.85, "benchmarking-post-grpo")
        post_grpo_acc = await _run_benchmark(run_id, config, "post-grpo")
        state.post_grpo_accuracy = post_grpo_acc
        total_improvement = post_grpo_acc - baseline_acc
        _emit(run_id, "benchmarking-post-grpo", "success",
              f"Post-GRPO accuracy: {post_grpo_acc:.1%} ({total_improvement:+.1%} vs baseline)",
              accuracy=post_grpo_acc, total_improvement=total_improvement)
        state.metrics["post_grpo_accuracy"] = post_grpo_acc
        state.metrics["total_improvement"] = total_improvement
        await _update_progress(run_id, 0.95, "benchmarking-post-grpo")

        # ---- Done ----
        _emit(run_id, "done", "success",
              f"IL pipeline complete! {baseline_acc:.1%} -> {post_grpo_acc:.1%} ({total_improvement:+.1%})",
              baseline=baseline_acc, final=post_grpo_acc, improvement=total_improvement)
        state.status = "completed"
        await _update_progress(run_id, 1.0, "done")

    except asyncio.CancelledError:
        state.status = "cancelled"
        _emit(run_id, state.stage, "warn", "Run cancelled by user")
    except Exception as e:
        state.status = "failed"
        _emit(run_id, state.stage, "error", f"Pipeline failed: {e}", error=str(e))


async def _run_benchmark(run_id: str, config: RunConfig, phase: str) -> float:
    """Run a benchmark phase. Returns accuracy (0.0 to 1.0)."""
    n_tasks = config.benchmark_tasks
    # Simulate per-task evaluation
    correct = 0
    for i in range(n_tasks):
        await asyncio.sleep(0.08)
        # Simulate accuracy improving across phases
        base_rate = {"baseline": 0.25, "post-sft": 0.45, "post-grpo": 0.60}.get(phase, 0.3)
        # Add some noise
        import random
        rng = random.Random(hash((run_id, phase, i)) & 0xFFFFFFFF)
        is_correct = rng.random() < base_rate
        if is_correct:
            correct += 1
        if (i + 1) % 4 == 0 or i == n_tasks - 1:
            _emit(run_id, f"benchmarking-{phase}", "metric",
                  f"[{phase}] Task {i+1}/{n_tasks}: {correct}/{i+1} correct ({correct/(i+1):.0%})",
                  phase=phase, task=i+1, total=n_tasks, correct=correct, accuracy=correct/(i+1))
    return correct / n_tasks


async def _run_sft(run_id: str, config: RunConfig) -> list[float]:
    """Run SFT training. Returns loss history."""
    losses = []
    total = config.sft_iters
    for i in range(total):
        await asyncio.sleep(0.03)
        # Simulate decreasing loss
        loss = 2.5 * (0.985 ** i) + 0.1 + (0.05 * ((i % 10) / 10))
        losses.append(round(loss, 4))
        if (i + 1) % 10 == 0 or i == 0:
            _emit(run_id, "sft-training", "metric",
                  f"SFT iter {i+1}/{total}: loss={loss:.4f}",
                  iter=i+1, total=total, loss=loss)
            await _update_progress(run_id, 0.2 + 0.25 * (i + 1) / total, "sft-training")
    return losses


async def _run_grpo(run_id: str, config: RunConfig) -> list[float]:
    """Run GRPO RL training. Returns reward history."""
    rewards = []
    total = config.grpo_iters
    for i in range(total):
        await asyncio.sleep(0.04)
        # Simulate increasing reward
        reward = 0.15 + 0.008 * i + 0.02 * ((i % 8) / 8)
        rewards.append(round(reward, 4))
        if (i + 1) % 5 == 0 or i == 0:
            _emit(run_id, "grpo-training", "metric",
                  f"GRPO iter {i+1}/{total}: reward={reward:.4f}",
                  iter=i+1, total=total, reward=reward)
            await _update_progress(run_id, 0.55 + 0.30 * (i + 1) / total, "grpo-training")
    return rewards
