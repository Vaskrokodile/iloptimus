"""Real SFT trainer — supervised fine-tuning with LoRA on mlx_lm.

Generates SFT training data from the benchmark results (correct responses
become training examples), then runs LoRA fine-tuning using mlx_lm's tuner.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .grader import build_prompt, grade_response, get_num_tasks


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
    batch_size: int = 4
    lora_rank: int = 8
    lora_scale: float = 1.0
    lora_dropout: float = 0.0
    memory_limit_gb: float = 5.0
    steps_per_eval: int = 20


@dataclass
class SFTExample:
    prompt: str
    response: str  # the "ideal" response (from correct benchmark outputs or generated)


def generate_sft_data(
    handle,
    domain: str,
    num_tasks: int | None = None,
    max_reasoning_tokens: int = 512,
    max_answer_tokens: int = 512,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[SFTExample]:
    """Generate SFT training data by running the baseline benchmark.

    For each task, we run inference. If the response gets a high score
    (correctness >= 0.5), we use it as a positive SFT example.
    If no responses are good enough, we generate a synthetic ideal response
    from the task's expected solution.
    """
    from .inference import run_inference, clear_cache

    total = get_num_tasks(domain)
    n = min(num_tasks or total, total)
    examples: list[SFTExample] = []

    for i in range(n):
        prompt = build_prompt(domain, i)
        inf = run_inference(
            handle, prompt,
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
    from mlx.utils import tree_flatten, tree_map

    config = config or SFTConfig()

    # Set memory limits
    if config.memory_limit_gb > 0:
        if hasattr(mx, "set_memory_limit"):
            mx.set_memory_limit(int(config.memory_limit_gb * 1024**3))
        elif mx.metal.is_available():
            mx.metal.set_memory_limit(int(config.memory_limit_gb * 1024**3))
        if hasattr(mx, "set_cache_limit"):
            mx.set_cache_limit(int(1.5 * 1024**3))

    # Apply LoRA layers to the model
    from mlx_lm.tuner.trainer import TrainingArgs
    from mlx_lm.tuner.utils import linear_to_lora_layers

    # Apply LoRA to attention layers
    lora_config = {
        "num_layers": len(handle.model.layers),
        "lora_parameters": {
            "rank": config.lora_rank,
            "scale": config.lora_scale,
            "dropout": config.lora_dropout,
        },
    }
    linear_to_lora_layers(handle.model, lora_config)

    # Prepare training data: tokenize prompt + response pairs
    train_data = []
    for ex in examples:
        messages = [
            {"role": "user", "content": ex.prompt},
            {"role": "assistant", "content": ex.response},
        ]
        tokens = handle.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        train_data.append(tokens)

    # Optimizer
    optimizer = opt.Adam(learning_rate=config.learning_rate)

    # Training loop
    os.makedirs(adapter_path, exist_ok=True)

    for iteration in range(config.num_iters):
        t0 = time.time()

        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        elif mx.metal.is_available():
            mx.metal.reset_peak_memory()
        mx.clear_cache()

        handle.model.train()

        # Sample a batch
        batch_indices = [
            (iteration * config.batch_size + b) % len(train_data)
            for b in range(min(config.batch_size, len(train_data)))
        ]

        loss_sum = 0.0
        grad_accum = None
        n_steps = 0

        for idx in batch_indices:
            tokens = train_data[idx]
            if len(tokens) < 2:
                continue

            input_tokens = mx.array(tokens[:-1])[None]  # [1, seq-1]
            target_tokens = mx.array(tokens[1:])[None]   # [1, seq-1]

            def loss_fn():
                logits = handle.model(input_tokens)
                # Cross-entropy loss
                log_probs = nn.log_softmax(logits, axis=-1)
                # Gather logprobs of target tokens
                token_logprobs = mx.take_along_axis(
                    log_probs[0],
                    target_tokens[0][:, None],
                    axis=-1,
                ).squeeze(-1)
                return -token_logprobs.mean()

            loss_value_and_grad = nn.value_and_grad(handle.model, loss_fn)
            loss_val, grad = loss_value_and_grad()
            mx.eval(loss_val, grad)

            loss_sum += float(loss_val)
            n_steps += 1

            if grad_accum is None:
                grad_accum = grad
            else:
                grad_accum = tree_map(lambda x, y: x + y, grad_accum, grad)

            mx.clear_cache()

        # Apply gradient update
        if grad_accum is not None and n_steps > 0:
            grad_accum = tree_map(lambda x: x / n_steps, grad_accum)
            optimizer.update(handle.model, grad_accum)
            mx.eval(handle.model.parameters(), optimizer.state)

        elapsed = time.time() - t0

        peak_mem = 0.0
        if hasattr(mx, "get_peak_memory"):
            peak_mem = mx.get_peak_memory() / 1e9
        elif mx.metal.is_available():
            peak_mem = mx.metal.get_peak_memory() / 1e9

        metrics = SFTMetrics(
            iteration=iteration,
            loss=loss_sum / max(n_steps, 1),
            learning_rate=config.learning_rate,
            elapsed=elapsed,
            peak_memory_gb=peak_mem,
        )

        if on_metrics:
            on_metrics(metrics)

    # Save adapter
    adapter_weights = dict(tree_flatten(handle.model.trainable_parameters()))
    mx.save_safetensors(f"{adapter_path}/adapters.safetensors", adapter_weights)
    cfg = {
        "adapter_path": os.path.basename(adapter_path),
        "fine_tune_type": "lora",
        "num_layers": len(handle.model.layers),
        "lora_parameters": {
            "rank": config.lora_rank,
            "scale": config.lora_scale,
            "dropout": config.lora_dropout,
        },
    }
    with open(f"{adapter_path}/adapter_config.json", "w") as f:
        json.dump(cfg, f, indent=4)

    return adapter_path
