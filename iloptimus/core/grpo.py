"""GRPO (group-relative policy optimization) — backend-agnostic dispatcher.

The GRPO algorithm is identical across backends:

1. Collect G rollouts for the same task (same prompt, different sampling seeds)
2. Grade each response with the real grader (correctness * reasoning_quality)
3. Compute group-relative advantage: A_i = (R_i - mean(R)) / (std(R) + eps)
4. Forward each rollout through the model, compute new logprobs at action positions
5. Compute clipped policy gradient loss + KL penalty
6. Backprop and update LoRA params

The backend-specific pieces (rollout collection with logprob recording, the
policy update via autograd, checkpointing) live in the backend modules. The
shared config/metrics dataclasses and advantage computation live here.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .backends import GRPOConfig, GRPOMetrics, get_backend

__all__ = ["GRPOConfig", "GRPOMetrics", "GRPOTrainer"]


# DeepSeek-R1-Distill think tokens
THINK_CLOSE_TOKEN = 151649


def _compute_advantages(rollouts, eps=1e-8):
    """Group-relative advantages: A_i = (R_i - mean(R)) / (std(R) + eps)."""
    rewards = np.array([r["reward"] for r in rollouts])
    mean_r = rewards.mean()
    std_r = rewards.std()

    if std_r < eps:
        return [0.0] * len(rollouts), mean_r, std_r

    advantages = (rewards - mean_r) / (std_r + eps)
    return advantages.tolist(), mean_r, std_r


class GRPOTrainer:
    """Backend-agnostic GRPO trainer.

    Delegates to the active backend's trainer (``MLXGRPOTrainer`` on Apple
    Silicon, ``VLLMGRPOTrainer`` on NVIDIA CUDA). The pipeline constructs this
    wrapper and calls ``train_step`` / ``save`` exactly as before.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: GRPOConfig | None = None,
        adapter_path: str = "il_grpo_adapters",
    ):
        from .inference import ModelHandle

        self.config = config or GRPOConfig()
        self.adapter_path = adapter_path
        # Accept either a ModelHandle (preferred) or a raw model+tokenizer pair
        # (legacy callers that pass handle.model / handle.tokenizer).
        if isinstance(model, ModelHandle):
            self.handle = model
        else:
            # Reconstruct a minimal handle so the backend can find its state.
            self.handle = ModelHandle(
                model=model,
                tokenizer=tokenizer,
                model_id="",
                huggingface_id="",
                precision="int4",
                quantized=True,
                backend="mlx",
            )
        self._trainer = get_backend(self.handle.backend).make_grpo_trainer(
            self.handle, self.config, adapter_path
        )

    def train_step(
        self,
        prompt: str,
        grade_fn: Callable[[str], float],
        on_metrics: Callable[[GRPOMetrics], None] | None = None,
    ) -> GRPOMetrics:
        return self._trainer.train_step(prompt, grade_fn, on_metrics)

    def save(self, path: str | None = None) -> None:
        self._trainer.save(path)
