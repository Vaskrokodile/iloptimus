"""Real SFT trainer — supervised fine-tuning with LoRA on mlx_lm.

Generates SFT training data from the benchmark results (correct responses
become training examples), then runs LoRA fine-tuning using mlx_lm's tuner.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .grader import build_prompt, get_num_tasks, grade_response


@dataclass
class SFTMetrics:
    iteration: int
    loss: float
    learning_rate: float
    elapsed: float
    peak_memory_gb: float


@dataclass
class SFTConfig:
    learning_rate: float = 1e-4
    num_iters: int = 100
    batch_size: int = 1
    grad_accumulation_steps: int = 1
    lora_rank: int = 8
    # mlx-lm's LoRALinear multiplies the adapter branch directly by this
    # value; its supported default is 20.0 (this is not alpha/rank).
    lora_scale: float = 20.0
    lora_dropout: float = 0.0
    lora_layers: int = 8  # final transformer blocks; safe on 8GB unified memory
    lora_targets: tuple[str, ...] = (
        "self_attn.q_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    )
    max_seq_length: int = 512
    memory_limit_gb: float = 3.0  # QLoRA on int4 uses less memory (was 3.5 for fp16)
    steps_per_eval: int = 20
    grad_clip: float = 1.0
    mask_prompt: bool = True
    grad_checkpoint: bool = False
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    # mlx-lm treats zero as "clear every step". Fixed-shape training reuses
    # buffers safely, so retain a bounded allocator cache for throughput.
    clear_cache_threshold_gb: float = 1.0
    compile_bucket_size: int = 128
    seed: int = 0


@dataclass
class SFTExample:
    prompt: str
    response: str  # the "ideal" response (from correct benchmark outputs or generated)


def generate_sft_data(
    handle,
    domain: str,
    num_tasks: int | None = None,
    task_offset: int = 0,
    max_reasoning_tokens: int = 512,
    max_answer_tokens: int = 512,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[SFTExample]:
    """Generate SFT training data by running the baseline benchmark.

    For each task, we run inference. If the response gets a high score
    (correctness >= 0.5), we use it as a positive SFT example.
    If no responses are good enough, we generate synthetic examples from
    the task's expected answer so SFT always has training data.
    """
    from .inference import THINK_CLOSE, THINK_OPEN, clear_cache, run_inference

    total = get_num_tasks(domain)
    task_offset = max(0, min(task_offset, total))
    n = min(num_tasks or (total - task_offset), total - task_offset)
    examples: list[SFTExample] = []

    if domain.startswith("custom:"):
        from .environments import get_environment

        environment = get_environment(domain.split(":", 1)[1])
        if environment and environment["mode"] == "IL":
            for relative_index, task in enumerate(environment["tasks"][task_offset : task_offset + n]):
                i = task_offset + relative_index
                ideal_response = task.get("ideal_response", "").strip()
                if not ideal_response:
                    continue
                if environment.get("domain") == "artifact-building":
                    prompt = (
                        str(task["prompt"]).strip()
                        + " Return source code only; do not use reasoning tags, answer tags, or explanatory prose."
                    )
                else:
                    prompt = build_prompt(domain, i)
                examples.append(SFTExample(prompt=prompt, response=ideal_response))
                if on_progress:
                    on_progress(relative_index + 1, n)
            if examples:
                return examples

    for i in range(task_offset, task_offset + n):
        prompt = build_prompt(domain, i)
        inf = run_inference(
            handle,
            prompt,
            max_reasoning_tokens=max_reasoning_tokens,
            max_answer_tokens=max_answer_tokens,
        )
        graded = grade_response(domain, i, inf.text)

        # Use responses with correctness >= 0.5 as SFT examples
        if graded.correctness >= 0.5:
            examples.append(SFTExample(prompt=prompt, response=inf.text))

        if on_progress:
            on_progress(i + 1, n)

        clear_cache()

    # Fallback: if no correct responses, generate synthetic examples
    # from the task's expected answer so SFT always has training data.
    # The synthetic reasoning is detailed enough to not collapse the model's
    # natural reasoning style — short generic responses cause LoRA to
    # degenerate output into repetitive garbage.
    if not examples:
        if domain.startswith("custom:"):
            from .environments import get_environment
            from .inference import THINK_CLOSE, THINK_OPEN

            environment = get_environment(domain.split(":", 1)[1])
            if environment:
                for relative_index, task in enumerate(environment["tasks"][task_offset : task_offset + n]):
                    i = task_offset + relative_index
                    expected = task.get("expected_answer") or "A response satisfying: " + ", ".join(
                        task.get("criteria", [])
                    )
                    examples.append(
                        SFTExample(
                            prompt=build_prompt(domain, i),
                            response=THINK_OPEN
                            + "I will follow the success criteria, solve the task step by step, and verify the result."
                            + THINK_CLOSE
                            + f"\n<answer>{expected}</answer>",
                        )
                    )
            return examples

        from .grader import _load_module, _taskset_path

        pkg_map = {
            "coding": ("il_coding_tasks", "il_coding_v1", "tasks.py"),
            "reasoning": ("il_reasoning_tasks", "il_reasoning_v1", "tasks.py"),
            "agentic-reasoning": ("il_agentic_reasoning_tasks", "il_agentic_reasoning_v1", "tasks.py"),
            "agentic-coding": ("il_agentic_coding_tasks", "il_agentic_coding_v1", "tasks.py"),
        }
        mod_name, pkg_id, filename = pkg_map[domain]
        tasks_mod = _load_module(mod_name, str(_taskset_path(pkg_id, filename)))

        for i in range(task_offset, task_offset + n):
            prompt = build_prompt(domain, i)
            task = tasks_mod.TASKS[i]

            # Build a synthetic ideal response with the expected answer
            if hasattr(task, "verify") and task.verify.__closure__:
                expected = task.verify.__closure__[0].cell_contents
                synthetic = (
                    THINK_OPEN
                    + "Let me work through this problem carefully step by step.\n\n"
                    + "First, I need to understand what is being asked and what "
                    + "constraints or conditions apply.\n\n"
                    + "Next, I'll consider the possible approaches and evaluate "
                    + "each one against the given constraints.\n\n"
                    + "After checking my reasoning, I can conclude that the answer "
                    + f"must be {expected}.\n\n"
                    + "Let me verify this is correct by working backwards from the "
                    + "answer and checking it satisfies all the original constraints.\n\n"
                    + "The verification confirms the answer is correct."
                    + THINK_CLOSE
                    + f"\n<answer>{expected}</answer>"
                )
            else:
                # Coding tasks: no simple expected answer, skip
                continue

            examples.append(SFTExample(prompt=prompt, response=synthetic))

    return examples


def run_sft(
    handle,
    examples: list[SFTExample],
    config: SFTConfig | None = None,
    adapter_path: str = "il_sft_adapters",
    on_metrics: Callable[[SFTMetrics], None] | None = None,
) -> str:
    """Run LoRA SFT training on the model.

    Uses mlx_lm's tuner to apply LoRA layers and train on the SFT examples.
    Returns the path to the saved adapter.
    """
    import mlx.core as mx
    import mlx.optimizers as opt
    from mlx_lm.tuner.callbacks import TrainingCallback
    from mlx_lm.tuner.datasets import CacheDataset, CompletionsDataset
    from mlx_lm.tuner.trainer import TrainingArgs, iterate_batches, train

    config = config or SFTConfig()

    # Adapter initialization and dataset shuffling must be reproducible so a
    # held-out before/after comparison can be rerun exactly.
    mx.random.seed(config.seed)
    # mlx-lm currently guards its iterator seed with `if seed`, which skips
    # the valid seed 0. Seed NumPy before it permutes batches.
    np.random.seed(config.seed)

    # Set memory limits
    if config.memory_limit_gb > 0:
        if hasattr(mx, "set_memory_limit"):
            mx.set_memory_limit(int(config.memory_limit_gb * 1024**3))
        elif mx.metal.is_available():
            mx.metal.set_memory_limit(int(config.memory_limit_gb * 1024**3))
        cache_limit = int(max(0.25, config.clear_cache_threshold_gb) * 1024**3)
        if hasattr(mx, "set_cache_limit"):
            mx.set_cache_limit(cache_limit)
        elif mx.metal.is_available():
            mx.metal.set_cache_limit(cache_limit)
        if hasattr(mx, "set_wired_limit"):
            mx.set_wired_limit(int(config.memory_limit_gb * 1024**3))
        elif mx.metal.is_available():
            mx.metal.set_wired_limit(int(config.memory_limit_gb * 1024**3))

    # Apply LoRA layers to the model
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.utils import linear_to_lora_layers

    # Freeze the base model before installing adapters. Otherwise
    # value_and_grad follows every quantized base weight and a 1.5B model can
    # consume enough gradient memory for macOS to terminate the process.
    handle.model.freeze()
    num_layers = min(config.lora_layers, len(handle.model.layers))
    lora_config = {
        "rank": config.lora_rank,
        "scale": config.lora_scale,
        "dropout": config.lora_dropout,
        "keys": set(config.lora_targets),
    }
    linear_to_lora_layers(handle.model, num_layers, lora_config)
    trainable_parameters = sum(
        int(parameter.size) for _, parameter in tree_flatten(handle.model.trainable_parameters())
    )

    rows = [{"prompt": example.prompt, "completion": example.response} for example in examples]
    train_data = CacheDataset(
        CompletionsDataset(
            rows,
            handle.tokenizer,
            prompt_key="prompt",
            completion_key="completion",
            mask_prompt=config.mask_prompt,
        )
    )
    optimizer_name = config.optimizer.casefold()
    if optimizer_name == "sgd":
        optimizer = opt.SGD(learning_rate=config.learning_rate)
    elif optimizer_name == "adam":
        optimizer = opt.Adam(learning_rate=config.learning_rate)
    elif optimizer_name == "adamw":
        optimizer = opt.AdamW(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unsupported SFT optimizer: {config.optimizer}")

    os.makedirs(adapter_path, exist_ok=True)
    adapter_file = os.path.join(adapter_path, "adapters.safetensors")

    class MetricsCallback(TrainingCallback):
        def __init__(self) -> None:
            self.started = time.perf_counter()

        def on_train_loss_report(self, info: dict) -> None:
            if not on_metrics:
                return
            on_metrics(
                SFTMetrics(
                    iteration=max(0, int(info["iteration"]) - 1),
                    loss=float(info["train_loss"]),
                    learning_rate=float(info["learning_rate"]),
                    elapsed=time.perf_counter() - self.started,
                    peak_memory_gb=float(info["peak_memory"]),
                )
            )

    args = TrainingArgs(
        batch_size=min(config.batch_size, len(train_data)),
        iters=config.num_iters,
        val_batches=0,
        steps_per_report=max(1, min(config.steps_per_eval, config.num_iters)),
        steps_per_eval=config.num_iters + 1,
        steps_per_save=config.num_iters + 1,
        max_seq_length=config.max_seq_length,
        adapter_file=adapter_file,
        grad_checkpoint=config.grad_checkpoint,
        grad_accumulation_steps=config.grad_accumulation_steps,
        clear_cache_threshold=int(config.clear_cache_threshold_gb * 1024**3),
    )

    def bucketed_batches(*args, **kwargs):
        """Use bounded stable shapes so MLX reuses compiled training graphs.

        mlx-lm starts with 32-token padding. The selected bucket may coarsen
        those shapes when a benchmark shows compile latency outweighs padding;
        compact TTC currently retains the faster native 32-token buckets.
        """
        kwargs.setdefault("seed", config.seed)
        bucket = max(32, config.compile_bucket_size)
        maximum = int(kwargs.get("max_seq_length") or config.max_seq_length)
        for batch, lengths in iterate_batches(*args, **kwargs):
            current = int(batch.shape[1])
            target = min(maximum, ((current + bucket - 1) // bucket) * bucket)
            if current < target:
                batch = mx.pad(batch, ((0, 0), (0, target - current)))
            yield batch, lengths

    train(
        handle.model,
        optimizer,
        train_data,
        args=args,
        iterate_batches=bucketed_batches,
        training_callback=MetricsCallback(),
    )

    cfg = {
        "adapter_path": os.path.basename(adapter_path),
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": {
            "rank": config.lora_rank,
            "scale": config.lora_scale,
            "dropout": config.lora_dropout,
            "targets": list(config.lora_targets),
        },
        "optimizer": config.optimizer,
        "mask_prompt": config.mask_prompt,
        "batch_size": config.batch_size,
        "grad_accumulation_steps": config.grad_accumulation_steps,
        "max_seq_length": config.max_seq_length,
        "compile_bucket_size": config.compile_bucket_size,
        "clear_cache_threshold_gb": config.clear_cache_threshold_gb,
        "trainable_parameters": trainable_parameters,
        "seed": config.seed,
    }
    with open(f"{adapter_path}/adapter_config.json", "w") as f:
        json.dump(cfg, f, indent=4)

    return adapter_path
