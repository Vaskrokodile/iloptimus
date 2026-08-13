"""Real SFT trainer — supervised fine-tuning with LoRA on mlx_lm.

Generates SFT training data from the benchmark results (correct responses
become training examples), then runs LoRA fine-tuning using mlx_lm's tuner.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .grader import build_prompt, get_num_tasks, grade_response


@dataclass
class SFTMetrics:
    iteration: int
    loss: float
    learning_rate: float
    elapsed: float
    peak_memory_gb: float
    iterations_per_second: float = 0.0
    tokens_per_second: float = 0.0
    trained_tokens: int = 0


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
    preserve_native_bucket_shape: bool = True
    # Cache the frozen transformer prefix once when adapters touch only final
    # layers. This trades a small amount of RAM for eliminating repeated base
    # forward compute across epochs.
    prefix_cache: bool = False
    prefix_cache_batch_size: int = 8
    seed: int = 0


@dataclass
class SFTExample:
    prompt: str
    response: str  # the "ideal" response (from correct benchmark outputs or generated)


class EagerCompletionDataset:
    """Pre-tokenized rows with real lengths available to MLX's batch sorter."""

    def __init__(self, rows: list[tuple[list[int], int]]) -> None:
        self.rows = rows

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.rows[index]

    def __len__(self) -> int:
        return len(self.rows)


def tokenize_sft_rows(
    rows: list[dict[str, str]], tokenizer: Any, *, max_seq_length: int, mask_prompt: bool = True
) -> tuple[EagerCompletionDataset, dict[str, Any]]:
    """Tokenize once and prove how much supervised completion survives truncation."""
    from mlx_lm.tuner.datasets import CompletionsDataset

    source = CompletionsDataset(
        rows,
        tokenizer,
        prompt_key="prompt",
        completion_key="completion",
        mask_prompt=mask_prompt,
    )
    tokenized: list[tuple[list[int], int]] = []
    total_completion_tokens = 0
    retained_completion_tokens = 0
    fully_retained = 0
    sequence_lengths: list[int] = []
    for row in rows:
        tokens, offset = source.process(row)
        tokens = list(tokens)
        offset = int(offset)
        completion_tokens = max(0, len(tokens) - offset)
        retained_tokens = max(0, min(len(tokens), max_seq_length) - min(offset, max_seq_length))
        total_completion_tokens += completion_tokens
        retained_completion_tokens += retained_tokens
        fully_retained += int(len(tokens) <= max_seq_length)
        sequence_lengths.append(len(tokens))
        tokenized.append((tokens, offset))
    ordered = sorted(sequence_lengths)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered)))) if ordered else 0
    stats = {
        "rows": len(rows),
        "fully_retained_rows": fully_retained,
        "fully_retained_fraction": round(fully_retained / max(1, len(rows)), 4),
        "completion_retention": round(retained_completion_tokens / max(1, total_completion_tokens), 4),
        "completion_tokens": total_completion_tokens,
        "retained_completion_tokens": retained_completion_tokens,
        "mean_sequence_tokens": round(sum(sequence_lengths) / max(1, len(sequence_lengths)), 2),
        "p95_sequence_tokens": ordered[p95_index] if ordered else 0,
        "maximum_sequence_tokens": max(sequence_lengths, default=0),
    }
    return EagerCompletionDataset(tokenized), stats


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
                    prompt = str(task["prompt"]).strip()
                    if "code only" not in prompt.casefold():
                        prompt += " Return source code only; do not use reasoning tags or explanatory prose."
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
    import mlx.nn as nn
    import mlx.optimizers as opt
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.tuner.callbacks import TrainingCallback
    from mlx_lm.tuner.trainer import TrainingArgs, default_loss, iterate_batches, train

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
    train_data, data_stats = tokenize_sft_rows(
        rows,
        handle.tokenizer,
        max_seq_length=config.max_seq_length,
        mask_prompt=config.mask_prompt,
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
    throughput_reports: list[tuple[float, float, int]] = []

    class MetricsCallback(TrainingCallback):
        def __init__(self) -> None:
            self.started = time.perf_counter()

        def on_train_loss_report(self, info: dict) -> None:
            throughput_reports.append(
                (
                    float(info.get("iterations_per_second") or 0.0),
                    float(info.get("tokens_per_second") or 0.0),
                    int(info.get("trained_tokens") or 0),
                )
            )
            if not on_metrics:
                return
            on_metrics(
                SFTMetrics(
                    iteration=max(0, int(info["iteration"]) - 1),
                    loss=float(info["train_loss"]),
                    learning_rate=float(info["learning_rate"]),
                    elapsed=time.perf_counter() - self.started,
                    peak_memory_gb=float(info["peak_memory"]),
                    iterations_per_second=float(info.get("iterations_per_second") or 0.0),
                    tokens_per_second=float(info.get("tokens_per_second") or 0.0),
                    trained_tokens=int(info.get("trained_tokens") or 0),
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

    prefix_cache_stats: dict[str, Any] = {"enabled": False}
    training_model = handle.model
    training_loss = default_loss

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
            # mlx-lm emits ``1 + N * 32`` shapes. Preserve that sentinel
            # shape instead of padding it by another 31 tokens at bucket 32.
            target = (
                min(maximum, 1 + ((max(0, current - 1) + bucket - 1) // bucket) * bucket)
                if config.preserve_native_bucket_shape
                else min(maximum, ((current + bucket - 1) // bucket) * bucket)
            )
            if current < target:
                batch = mx.pad(batch, ((0, 0), (0, target - current)))
            yield batch, lengths

    training_batches = bucketed_batches
    if config.prefix_cache and config.batch_size == 1 and num_layers < len(handle.model.model.layers):
        prefix_started = time.perf_counter()
        split = len(handle.model.model.layers) - num_layers
        cached_rows: list[tuple[Any, list[int], int]] = []
        prepared_rows = []
        for index in range(len(train_data)):
            tokens, offset = train_data[index]
            tokens = list(tokens[: config.max_seq_length])
            if len(tokens) >= 2:
                prepared_rows.append((tokens, min(int(offset), len(tokens))))
        prepared_rows.sort(key=lambda item: len(item[0]))
        cache_batch_size = max(1, config.prefix_cache_batch_size)
        for start in range(0, len(prepared_rows), cache_batch_size):
            group = prepared_rows[start : start + cache_batch_size]
            maximum = max(len(tokens) - 1 for tokens, _ in group)
            token_batch = np.zeros((len(group), maximum), dtype=np.int32)
            for row_index, (tokens, _) in enumerate(group):
                token_batch[row_index, : len(tokens) - 1] = tokens[:-1]
            hidden = handle.model.model.embed_tokens(mx.array(token_batch))
            mask = create_attention_mask(hidden)
            for layer in handle.model.model.layers[:split]:
                hidden = layer(hidden, mask, None)
            mx.eval(hidden)
            for row_index, (tokens, offset) in enumerate(group):
                row_hidden = mx.array(hidden[row_index, : len(tokens) - 1, :])
                mx.eval(row_hidden)
                cached_rows.append((row_hidden, tokens, offset))

        class CachedSuffixModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = handle.model.model.layers[split:]
                self.norm = handle.model.model.norm
                self.lm_head = handle.model.lm_head

            def __call__(self, hidden):
                mask = create_attention_mask(hidden)
                for layer in self.layers:
                    hidden = layer(hidden, mask, None)
                return self.lm_head(self.norm(hidden))

        training_model = CachedSuffixModel()

        def cached_loss(model, hidden, tokens, lengths):
            targets = tokens[:, 1:]
            logits = model(hidden)
            steps = mx.arange(1, targets.shape[1] + 1)
            mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
            cross_entropy = nn.losses.cross_entropy(logits, targets) * mask
            token_count = mask.sum()
            loss = cross_entropy.astype(mx.float32).sum() / token_count
            return loss, token_count

        def cached_batches(*, loop=False, seed=None, **_kwargs):
            if seed is not None:
                np.random.seed(seed)
            order = list(range(len(cached_rows)))
            while True:
                for index in np.random.permutation(order):
                    hidden, tokens, offset = cached_rows[int(index)]
                    yield (
                        hidden[None, :, :],
                        mx.array([tokens]),
                        mx.array([[offset, len(tokens)]]),
                    )
                if not loop:
                    break

        training_batches = cached_batches
        training_loss = cached_loss
        cache_bytes = sum(int(hidden.nbytes) for hidden, _, _ in cached_rows)
        prefix_cache_stats = {
            "enabled": True,
            "prefix_layers": split,
            "trainable_suffix_layers": num_layers,
            "rows": len(cached_rows),
            "batch_size": cache_batch_size,
            "bytes": cache_bytes,
            "build_seconds": round(time.perf_counter() - prefix_started, 3),
        }

    train(
        training_model,
        optimizer,
        train_data,
        args=args,
        loss=training_loss,
        iterate_batches=training_batches,
        training_callback=MetricsCallback(),
    )
    # Prefix-cached training uses a suffix wrapper, but adapters must retain
    # their original full-model parameter paths for normal inference loading.
    mx.save_safetensors(adapter_file, dict(tree_flatten(handle.model.trainable_parameters())))

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
        "preserve_native_bucket_shape": config.preserve_native_bucket_shape,
        "prefix_cache": prefix_cache_stats,
        "trainable_parameters": trainable_parameters,
        "seed": config.seed,
        "training_data": data_stats,
        "mean_iterations_per_second": round(
            sum(item[0] for item in throughput_reports) / max(1, len(throughput_reports)), 4
        ),
        "mean_tokens_per_second": round(
            sum(item[1] for item in throughput_reports) / max(1, len(throughput_reports)), 4
        ),
        "trained_tokens": max((item[2] for item in throughput_reports), default=0),
    }
    with open(f"{adapter_path}/adapter_config.json", "w") as f:
        json.dump(cfg, f, indent=4)

    return adapter_path
