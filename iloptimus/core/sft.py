"""SFT (supervised fine-tuning) — backend-agnostic dispatcher.

Generates SFT training data from benchmark results (correct responses become
training examples) and runs LoRA/QLoRA fine-tuning via the active backend:

- MLX backend: ``mlx_lm`` tuner with cached tokenization, stable length
  buckets, prompt masking, frozen-prefix caching, and selected attention
  targets.
- vLLM backend: HuggingFace Transformers + PEFT LoRA/QLoRA with a manual
  training loop that mirrors the MLX prompt-masking and streaming-metrics
  behavior.

The shared, backend-agnostic pieces (config, example/metrics dataclasses,
benchmark-driven data generation, and the MLX tokenization helper kept for
backward compatibility) live here.
"""

from __future__ import annotations

from typing import Any, Callable

from .backends import SFTConfig, SFTExample, SFTMetrics, get_backend
from .backends.mlx_backend import EagerCompletionDataset, _tokenize_sft_rows
from .grader import build_prompt, get_num_tasks, grade_response

__all__ = [
    "EagerCompletionDataset",
    "SFTConfig",
    "SFTExample",
    "SFTMetrics",
    "generate_sft_data",
    "run_sft",
    "tokenize_sft_rows",
]


# Backward-compat re-export of the MLX tokenization helper (used by tests and
# the MLX SFT path). It remains MLX-specific because it relies on
# ``mlx_lm.tuner.datasets.CompletionsDataset`` for offset computation.
def tokenize_sft_rows(
    rows: list[dict[str, str]], tokenizer: Any, *, max_seq_length: int, mask_prompt: bool = True
) -> tuple[EagerCompletionDataset, dict[str, Any]]:
    return _tokenize_sft_rows(rows, tokenizer, max_seq_length=max_seq_length, mask_prompt=mask_prompt)


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
    """Run LoRA SFT training on the model via the active backend.

    Returns the path to the saved adapter.
    """
    config = config or SFTConfig()
    backend = get_backend(handle.backend)
    return backend.run_sft(handle, examples, config, adapter_path, on_metrics)
