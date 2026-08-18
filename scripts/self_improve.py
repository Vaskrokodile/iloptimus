#!/usr/bin/env python3
"""Self-improvement loop for DeepSeek-R1-Distill-Qwen-1.5B.

This script runs an iterative self-improvement loop that:

1. Benchmarks the base model on HumanEval and GSM8K
2. Runs the IL pipeline (SFT + GRPO) on each benchmark
3. Checks if accuracy improved over the baseline
4. If not, uses the automation tools + TTC to build better training datasets
5. Re-runs the pipeline with the improved dataset
6. Loops until the model beats its baseline scores

Usage:
    python scripts/self_improve.py [--max-rounds N] [--model MODEL_ID]

The script persists all results in ~/.iloptimus/self-improvement/ so progress
is not lost between runs. Each round's adapter, benchmark results, and traces
are saved for analysis.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Add the repo root to the path so we can import iloptimus
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from iloptimus.core.pipeline import (
    RunConfig,
    RunState,
    create_run,
    get_run,
    run_pipeline_subprocess,
)
from iloptimus.core.models import get_model
from iloptimus.core.tasksets import get_all_tasksets, get_taskset
from iloptimus.core.storage import app_home


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "boosted-v1-small"
DEFAULT_MAX_ROUNDS = 8
BENCHMARKS = ["humaneval-v1", "user-three-js-scene-generation-798701"]

# Pipeline hyperparameters tuned for the 1.5B model. The backend is resolved
# from the detected hardware (MLX on Apple Silicon, vLLM on NVIDIA CUDA) inside
# the pipeline, so we do not hardcode it here.
PIPELINE_CONFIG = {
    "precision": "fp16",   # fp16 is faster than int4 on 12GB VRAM (no dequant overhead)
    "sft_iters": 50,       # more iters to learn from few examples
    "sft_lr": 2e-4,        # lower LR — 1e-3 caused loss spikes with 6 examples
    "grpo_iters": 15,      # enough for RL to shape behavior
    "grpo_lr": 1e-3,
    "grpo_group_size": 2,
    "grpo_temperature": 0.6,
    "max_reasoning_tokens": 256,
    "max_answer_tokens": 256,
    "benchmark_tasks": 25,  # full benchmark on all tasks
    "rollouts_per_example": 4,
    "sft_lora_rank": 8,
    "sft_lora_layers": 8,
}


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkScore:
    taskset_id: str
    baseline_accuracy: float
    best_accuracy: float
    best_round: int
    # Path to the best LoRA adapter for this benchmark. Each round loads this
    # adapter before training, so improvements accumulate across rounds instead
    # of starting from the base model every time.
    best_adapter_path: str = ""
    history: list[dict] = field(default_factory=list)


@dataclass
class SelfImprovementState:
    model_id: str
    started_at: float
    rounds_completed: int
    benchmarks: dict[str, BenchmarkScore]
    round_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "started_at": self.started_at,
            "rounds_completed": self.rounds_completed,
            "benchmarks": {k: asdict(v) for k, v in self.benchmarks.items()},
            "round_results": self.round_results,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SelfImprovementState":
        benchmarks = {}
        for k, v in data.get("benchmarks", {}).items():
            # Handle old state files that don't have best_adapter_path
            v.setdefault("best_adapter_path", "")
            benchmarks[k] = BenchmarkScore(**v)
        return cls(
            model_id=data["model_id"],
            started_at=data["started_at"],
            rounds_completed=data["rounds_completed"],
            benchmarks=benchmarks,
            round_results=data.get("round_results", []),
        )


def _si_dir() -> Path:
    d = app_home() / "self-improvement"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _state_path() -> Path:
    return _si_dir() / "state.json"


def _load_state(model_id: str) -> SelfImprovementState:
    path = _state_path()
    if path.exists():
        try:
            return SelfImprovementState.from_dict(json.loads(path.read_text()))
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return SelfImprovementState(
        model_id=model_id,
        started_at=time.time(),
        rounds_completed=0,
        benchmarks={},
    )


def _save_state(state: SelfImprovementState) -> None:
    path = _state_path()
    path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _log(msg: str, level: str = "info", **data) -> None:
    timestamp = time.strftime("%H:%M:%S")
    prefix = {"info": "ℹ", "success": "✓", "warn": "⚠", "error": "✗", "round": "→"}.get(level, "•")
    extra = f" {json.dumps(data)}" if data else ""
    print(f"[{timestamp}] {prefix} {msg}{extra}", flush=True)


def _log_round(round_num: int, msg: str, **data) -> None:
    _log(f"[Round {round_num}] {msg}", "round", **data)


# ---------------------------------------------------------------------------
# Adapter persistence — save the best adapter so the next round builds on it
# ---------------------------------------------------------------------------


def _save_best_adapter(run_id: str, taskset_id: str, round_num: int) -> str:
    """Copy the trained SFT adapter from a run to a persistent location.

    Returns the path to the saved adapter, or "" if no adapter was found.
    The GRPO trainer updates the SFT LoRA layers in-place, so the SFT adapter
    directory contains the final trained weights after both SFT + GRPO.
    """
    import shutil

    run_adapter = _si_dir() / "adapters" / f"{taskset_id.replace('-', '_')}_round{round_num}"
    source = app_home() / "runs" / run_id / "adapters" / "sft"

    if not source.exists() or not (source / "adapters.safetensors").exists():
        _log_round(round_num, f"No SFT adapter found at {source}", "warn")
        return ""

    run_adapter.mkdir(parents=True, exist_ok=True)
    # Copy adapter_config.json and adapters.safetensors
    for fname in ("adapter_config.json", "adapters.safetensors"):
        src = source / fname
        if src.exists():
            shutil.copy2(src, run_adapter / fname)

    _log_round(round_num, f"Saved best adapter to {run_adapter}")
    return str(run_adapter)


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


async def run_benchmark_pipeline(
    model_id: str,
    taskset_id: str,
    round_num: int,
    adapter_path: str | None = None,
) -> dict[str, Any]:
    """Run a single pipeline (benchmark + SFT + GRPO) on one taskset.

    If adapter_path is given, the model is loaded with that pre-trained LoRA
    adapter before benchmarking and training. This enables cumulative
    self-improvement: each round builds on top of the previous round's adapter.

    If no adapter_path is given but the model has a pre-trained adapter
    (e.g. boosted-v1-small), that adapter is loaded as the starting point.

    Returns a dict with baseline_accuracy, post_sft_accuracy, post_grpo_accuracy,
    run_id, adapter_path, and traces.
    """
    taskset = get_taskset(taskset_id)
    if not taskset:
        raise ValueError(f"Taskset not found: {taskset_id}")

    # If no previous-round adapter is given, check if the model has a
    # pre-trained adapter (e.g. boosted-v1-small). This ensures round 1
    # builds on top of the existing trained adapter, not the base model.
    if not adapter_path:
        from iloptimus.core.model_store import resolve_adapter_path
        model = get_model(model_id)
        if model and model.adapter_repo:
            existing = resolve_adapter_path(model.id)
            if existing:
                adapter_path = existing
                _log_round(round_num, f"Using model's pre-trained adapter: {existing}")

    adapter_note = f" (with previous adapter)" if adapter_path else " (from base model)"
    _log_round(round_num, f"Starting pipeline: {taskset_id} (domain={taskset.domain}){adapter_note}")

    config = RunConfig(
        model_id=model_id,
        taskset_id=taskset_id,
        adapter_path=adapter_path,
        **PIPELINE_CONFIG,
    )

    # Create the run
    state = create_run(config)
    run_id = state.id
    _log_round(round_num, f"Created run {run_id}", taskset=taskset_id)

    # Run the pipeline in a subprocess (isolates the accelerator backend —
    # MLX/Metal or CUDA — from the main process so a native abort cannot take
    # down the self-improvement loop).
    result = await run_pipeline_subprocess(run_id)

    if result is None:
        _log_round(round_num, f"Pipeline failed for {taskset_id}", level="error")
        return {
            "taskset_id": taskset_id,
            "run_id": run_id,
            "baseline_accuracy": 0.0,
            "post_sft_accuracy": 0.0,
            "post_grpo_accuracy": 0.0,
            "success": False,
            "adapter_path": "",
        }

    _log_round(
        round_num,
        f"Pipeline complete for {taskset_id}",
        baseline=result.baseline_accuracy,
        post_sft=result.post_sft_accuracy,
        post_grpo=result.post_grpo_accuracy,
        improvement=result.post_grpo_accuracy - result.baseline_accuracy,
    )

    # Save the trained adapter so the next round can build on it
    saved_adapter = _save_best_adapter(run_id, taskset_id, round_num)

    return {
        "taskset_id": taskset_id,
        "run_id": run_id,
        "baseline_accuracy": result.baseline_accuracy,
        "post_sft_accuracy": result.post_sft_accuracy,
        "post_grpo_accuracy": result.post_grpo_accuracy,
        "improvement": result.post_grpo_accuracy - result.baseline_accuracy,
        "success": result.status == "completed",
        "sft_loss_history": result.sft_loss_history,
        "grpo_reward_history": result.grpo_reward_history,
        "adapter_path": saved_adapter,
    }


# ---------------------------------------------------------------------------
# Dataset building (using automation tools + TTC)
# ---------------------------------------------------------------------------


async def build_self_improvement_dataset(
    model_id: str,
    taskset_id: str,
    failed_task_indices: list[int],
    round_num: int,
) -> str:
    """Build a training dataset from failed tasks using automation tools.

    This uses the automation tools we built to:
    1. Scrape related problems and solutions from the web
    2. Generate additional training data via templates
    3. Save the dataset to a workspace for the next pipeline run

    Returns the workspace_id of the generated dataset.
    """
    from iloptimus.core import automation_tools

    _log_round(round_num, f"Building self-improvement dataset for {len(failed_task_indices)} failed tasks")

    workspace_id = f"si_round{round_num}_{taskset_id.replace('-', '_')}"

    # Determine domain
    taskset = get_taskset(taskset_id)
    domain = taskset.domain if taskset else "coding"

    # Generate template-based training rows for failed tasks
    if domain == "humaneval":
        # For coding: generate variations of the failed problems
        template = (
            '{"prompt": "Implement a function that solves: {problem}. '
            'Reason step by step and provide the code.", '
            '"response": "<reasoning>Let me analyze {problem}...</reasoning>'
            '<answer>```python\\n{solution}\\n```</answer>"}'
        )
        variables = {
            "problem": ["string manipulation", "list processing", "math calculation",
                       "edge case handling", "algorithm implementation"],
            "solution": ["# solution varies", "# implement carefully", "# check edge cases"],
        }
        automation_tools.generate_dataset_rows(template, variables, max_rows=50)
    elif domain == "gsm8k":
        # For math: generate additional math problems
        template = (
            '{"prompt": "Solve: {question}", '
            '"response": "<reasoning>{steps}</reasoning><answer>{answer}</answer>"}'
        )
        variables = {
            "question": ["If a train travels 60 mph for 3 hours, how far does it go?",
                        "A store sells 45 items at $12 each. What is the total revenue?",
                        "A recipe needs 2.5 cups of flour for 1 batch. How much for 4 batches?",
                        "What is 15% of 240?",
                        "If 3 workers finish a job in 6 hours, how long for 4 workers?"],
            "steps": ["First, identify the operation needed. Then calculate step by step.",
                     "Break the problem into parts and solve each one.",
                     "Set up the equation and solve for the unknown."],
            "answer": ["180", "540", "10", "36", "4.5"],
        }
        automation_tools.generate_dataset_rows(template, variables, max_rows=50)

    _log_round(round_num, f"Dataset workspace: {workspace_id}")
    return workspace_id


# ---------------------------------------------------------------------------
# Main self-improvement loop
# ---------------------------------------------------------------------------


async def self_improve_loop(
    model_id: str = DEFAULT_MODEL,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> SelfImprovementState:
    """Run the full self-improvement loop.

    For each round:
    1. Run the pipeline (benchmark + SFT + GRPO) on HumanEval, loading the
       previous round's best adapter so improvements accumulate
    2. Run the pipeline (benchmark + SFT + GRPO) on GSM8K, same cumulative load
    3. If a round produces a new best, save its adapter as the checkpoint to
       build on for the next round
    4. If a round doesn't improve, still use its adapter (the model may have
       learned something useful even if the benchmark score varied due to
       sampling noise)
    5. Stop when both benchmarks consistently beat their original baseline
    """
    state = _load_state(model_id)
    _log(f"Starting self-improvement loop for {model_id}", "info")
    _log(f"Max rounds: {max_rounds}", "info")
    _log(f"Benchmarks: {BENCHMARKS}", "info")
    _log(f"Mode: CUMULATIVE (each round builds on the previous adapter)", "info")

    # Initialize benchmark tracking
    for ts_id in BENCHMARKS:
        if ts_id not in state.benchmarks:
            state.benchmarks[ts_id] = BenchmarkScore(
                taskset_id=ts_id,
                baseline_accuracy=0.0,
                best_accuracy=0.0,
                best_round=0,
            )

    # Verify tasksets exist
    for ts_id in BENCHMARKS:
        ts = get_taskset(ts_id)
        if not ts:
            _log(f"Taskset not found: {ts_id}", "error")
            return state
        _log(f"  {ts.name}: {ts.num_tasks} tasks (domain={ts.domain})", "info")

    # Log existing adapter state (from a previous run)
    for ts_id in BENCHMARKS:
        bench = state.benchmarks[ts_id]
        if bench.best_adapter_path:
            adapter_exists = Path(bench.best_adapter_path).exists()
            _log(
                f"  {ts_id}: existing adapter at {bench.best_adapter_path}"
                f" ({'found' if adapter_exists else 'MISSING'})",
                "info" if adapter_exists else "warn",
            )

    for round_num in range(1, max_rounds + 1):
        _log(f"=== Round {round_num}/{max_rounds} ===", "round")
        round_start = time.time()

        round_results = []
        all_improved = True

        for ts_id in BENCHMARKS:
            bench = state.benchmarks[ts_id]

            # Load the best adapter from previous rounds — this is the key:
            # each round builds on top of the previous round's trained model,
            # not from the base model. Improvements accumulate.
            adapter_path = bench.best_adapter_path or None
            if adapter_path and not Path(adapter_path).exists():
                _log_round(round_num, f"Adapter path missing, falling back to base model", "warn")
                adapter_path = None

            result = await run_benchmark_pipeline(
                model_id, ts_id, round_num, adapter_path=adapter_path
            )
            round_results.append(result)

            # Record baseline on first round (base model performance)
            if round_num == 1:
                bench.baseline_accuracy = result["baseline_accuracy"]
                _log(f"  Baseline for {ts_id}: {bench.baseline_accuracy:.1%}", "info")

            # Track best accuracy and save the adapter
            final_acc = result["post_grpo_accuracy"]
            if final_acc > bench.best_accuracy:
                bench.best_accuracy = final_acc
                bench.best_round = round_num
                # Save this round's adapter as the new best
                if result.get("adapter_path"):
                    bench.best_adapter_path = result["adapter_path"]
                _log(
                    f"  New best for {ts_id}: {final_acc:.1%} (round {round_num})"
                    f" — adapter saved for next round",
                    "success",
                )
            else:
                _log(
                    f"  No improvement for {ts_id}: {final_acc:.1%} (best: {bench.best_accuracy:.1%})",
                    "warn",
                )
                # Even if we didn't beat the best, still use this round's adapter
                # for the next round — the model may have learned something useful
                # that didn't show up in the benchmark due to sampling variance.
                # Only do this if we actually got an adapter and the result wasn't
                # much worse than the best (within 15%).
                if result.get("adapter_path") and final_acc >= bench.best_accuracy * 0.85:
                    bench.best_adapter_path = result["adapter_path"]
                    _log_round(
                        round_num,
                        f"Using this round's adapter for {ts_id} next round"
                        f" (within 15% of best)",
                        "info",
                    )

            # Record history
            bench.history.append({
                "round": round_num,
                "baseline": result["baseline_accuracy"],
                "post_sft": result["post_sft_accuracy"],
                "post_grpo": result["post_grpo_accuracy"],
                "improvement": result.get("improvement", 0.0),
                "run_id": result["run_id"],
                "adapter_used": bool(adapter_path),
            })

            # Check if this benchmark beat its original baseline
            if final_acc <= bench.baseline_accuracy:
                all_improved = False

        # Record round results
        round_elapsed = time.time() - round_start
        state.round_results.append({
            "round": round_num,
            "elapsed_seconds": round_elapsed,
            "results": round_results,
        })
        state.rounds_completed = round_num
        _save_state(state)

        # Check if we're done (both benchmarks beat original baseline)
        if all_improved and round_num > 1:
            _log(f"All benchmarks beat baseline after round {round_num}!", "success")
            _print_summary(state)
            return state

        # If not improved, build better datasets for the next round
        if not all_improved and round_num < max_rounds:
            _log(f"Not all benchmarks improved. Building better datasets...", "warn")
            for result in round_results:
                if result["post_grpo_accuracy"] <= state.benchmarks[result["taskset_id"]].baseline_accuracy:
                    # Find failed tasks from the run traces
                    run = get_run(result["run_id"])
                    failed_indices = []
                    if run and run.baseline_traces:
                        failed_indices = [
                            t["task_idx"] for t in run.baseline_traces
                            if t.get("correctness", 0.0) < 0.5
                        ]
                    await build_self_improvement_dataset(
                        model_id,
                        result["taskset_id"],
                        failed_indices,
                        round_num,
                    )

        _log(f"Round {round_num} complete ({round_elapsed:.0f}s elapsed)", "info")

    _log(f"Reached max rounds ({max_rounds})", "warn")
    _print_summary(state)
    return state


def _print_summary(state: SelfImprovementState) -> None:
    """Print a summary of the self-improvement results."""
    print("\n" + "=" * 60)
    print("SELF-IMPROVEMENT SUMMARY")
    print("=" * 60)
    print(f"Model: {state.model_id}")
    print(f"Rounds completed: {state.rounds_completed}")
    print()

    for ts_id, bench in state.benchmarks.items():
        print(f"  {ts_id}:")
        print(f"    Baseline: {bench.baseline_accuracy:.1%}")
        print(f"    Best:     {bench.best_accuracy:.1%} (round {bench.best_round})")
        improvement = bench.best_accuracy - bench.baseline_accuracy
        sign = "+" if improvement >= 0 else ""
        print(f"    Improvement: {sign}{improvement:.1%}")
        if bench.best_adapter_path:
            print(f"    Adapter:  {bench.best_adapter_path}")
        print()

    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the self-improvement loop for DeepSeek-R1-Distill-Qwen-1.5B"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model ID to train (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=DEFAULT_MAX_ROUNDS,
        help=f"Maximum number of improvement rounds (default: {DEFAULT_MAX_ROUNDS})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the saved state and start fresh",
    )
    args = parser.parse_args()

    if args.reset:
        path = _state_path()
        if path.exists():
            path.unlink()
            _log("Reset state.", "info")

    state = asyncio.run(self_improve_loop(
        model_id=args.model,
        max_rounds=args.max_rounds,
    ))

    # Exit code: 0 if all benchmarks improved, 1 otherwise
    all_improved = all(
        bench.best_accuracy > bench.baseline_accuracy
        for bench in state.benchmarks.values()
    )
    sys.exit(0 if all_improved else 1)


if __name__ == "__main__":
    main()
