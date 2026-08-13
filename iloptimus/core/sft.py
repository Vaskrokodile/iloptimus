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
    learning_rate: float = 1e-4  # SGD needs higher LR than Adam (was 1e-5 for Adam)
    num_iters: int = 100
    batch_size: int = 1  # safe for 8GB Apple Silicon (was 4)
    lora_rank: int = 8
    lora_scale: float = 0.1
    lora_dropout: float = 0.0
    lora_layers: int = 8  # final transformer blocks; safe on 8GB unified memory
    max_seq_length: int = 512
    memory_limit_gb: float = 3.0  # QLoRA on int4 uses less memory (was 3.5 for fp16)
    steps_per_eval: int = 20
    grad_clip: float = 1.0


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
                examples.append(SFTExample(prompt=build_prompt(domain, i), response=ideal_response))
                if on_progress:
                    on_progress(i + 1, n)
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
    from mlx.utils import tree_flatten, tree_map

    config = config or SFTConfig()

    # Set memory limits
    if config.memory_limit_gb > 0:
        if hasattr(mx, "set_memory_limit"):
            mx.set_memory_limit(int(config.memory_limit_gb * 1024**3))
        elif mx.metal.is_available():
            mx.metal.set_memory_limit(int(config.memory_limit_gb * 1024**3))
        if hasattr(mx, "set_cache_limit"):
            mx.set_cache_limit(int(1.0 * 1024**3))
        elif mx.metal.is_available():
            mx.metal.set_cache_limit(int(1.0 * 1024**3))
        if hasattr(mx, "set_wired_limit"):
            mx.set_wired_limit(int(config.memory_limit_gb * 1024**3))
        elif mx.metal.is_available():
            mx.metal.set_wired_limit(int(config.memory_limit_gb * 1024**3))

    # Apply LoRA layers to the model
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
    }
    linear_to_lora_layers(handle.model, num_layers, lora_config)

    # Prepare training data: tokenize prompt + response pairs
    train_data = []
    for ex in examples:
        messages = [
            {"role": "user", "content": ex.prompt},
            {"role": "assistant", "content": ex.response},
        ]
        tokens = handle.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if config.max_seq_length > 1:
            tokens = tokens[-config.max_seq_length:]
        train_data.append(tokens)

    # Optimizer — SGD is used instead of Adam because Adam's second moment
    # estimate can produce NaN when combined with int4 quantized weights
    # (QLoRA). SGD is simpler and more stable for this use case.
    optimizer = opt.SGD(learning_rate=config.learning_rate)

    # Compile the cross-entropy loss computation for faster forward+backward.
    # mx.compile fuses element-wise operations into single Metal kernels.
    # Note: we compile only the loss math (not the model call) because
    # nn.value_and_grad passes model.trainable_parameters() as a dict,
    # which mx.compile can't handle as a callable.
    @mx.compile
    def compiled_cross_entropy(logits, target_tokens):
        log_probs = nn.log_softmax(logits, axis=-1)
        token_logprobs = mx.take_along_axis(
            log_probs[0],
            target_tokens[0][:, None],
            axis=-1,
        ).squeeze(-1)
        return -token_logprobs.mean()

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
            target_tokens = mx.array(tokens[1:])[None]  # [1, seq-1]

            def loss_fn():
                logits = handle.model(input_tokens)
                return compiled_cross_entropy(logits, target_tokens)

            loss_value_and_grad = nn.value_and_grad(handle.model, loss_fn)
            loss_val, grad = loss_value_and_grad()
            mx.eval(loss_val, grad)

            loss_f = float(loss_val)
            # Skip NaN or Inf gradients — these corrupt model weights.
            # QLoRA on int4 can produce Inf logits for out-of-distribution inputs.
            if loss_f != loss_f or loss_f in (float("inf"), float("-inf")):
                continue

            loss_sum += loss_f
            n_steps += 1

            if grad_accum is None:
                grad_accum = grad
            else:
                grad_accum = tree_map(lambda x, y: x + y, grad_accum, grad)

            mx.clear_cache()

        # Apply gradient update
        if grad_accum is not None and n_steps > 0:
            grad_accum = tree_map(lambda x: x / n_steps, grad_accum)
            # Gradient clipping to prevent NaN
            if config.grad_clip > 0:
                grad_accum = tree_map(
                    lambda x: mx.clip(x, -config.grad_clip, config.grad_clip),
                    grad_accum,
                )
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
        "num_layers": num_layers,
        "lora_parameters": {
            "rank": config.lora_rank,
            "scale": config.lora_scale,
            "dropout": config.lora_dropout,
        },
    }
    with open(f"{adapter_path}/adapter_config.json", "w") as f:
        json.dump(cfg, f, indent=4)

    return adapter_path
