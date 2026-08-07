"""Real GRPO trainer — ported from ilresearch/il_rl/grpo.py.

Adapted for the IL Optimus taskset interface (build_prompt + grade_response)
instead of the grid-puzzle RLEnvironment.

Algorithm:
1. Collect G rollouts for the same task (same prompt, different sampling temperatures)
2. Grade each response with the real grader (correctness * reasoning_quality)
3. Compute group-relative advantage: A_i = (R_i - mean(R)) / (std(R) + eps)
4. Forward each rollout through the model, compute new logprobs at action positions
5. Compute clipped policy gradient loss + KL penalty
6. Backprop and update LoRA params
7. mx.eval to materialize updates (MLX is lazy)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class GRPOMetrics:
    iteration: int
    mean_reward: float
    std_reward: float
    max_reward: float
    min_reward: float
    mean_correctness: float
    mean_reasoning_quality: float
    loss: float
    rollout_time: float
    update_time: float
    total_time: float
    peak_memory_gb: float
    avg_episode_tokens: float


@dataclass
class GRPOConfig:
    learning_rate: float = 1e-5
    clip_eps: float = 0.2
    group_size: int = 4
    thinking_tokens: int = 256
    prediction_tokens: int = 256
    temperature: float = 0.6
    top_p: float = 0.9
    kl_beta: float = 0.04
    memory_limit_gb: float = 5.0


# DeepSeek-R1-Distill think tokens
THINK_CLOSE_TOKEN = 151649


def _compute_action_logprobs(model, tokens, action_positions):
    """Forward backbone only, apply LM head ONLY at action positions.

    This is the critical memory optimization from ilresearch — avoids
    materializing full-vocab logits over the entire episode.
    """
    import mlx.core as mx

    tokens_arr = mx.array(tokens)
    input_tokens = tokens_arr[:-1][None]  # [1, seq_len-1]
    hidden = model.model(input_tokens)    # [1, seq_len-1, hidden]
    hidden = hidden[0]                     # [seq_len-1, hidden]

    action_logprob_segments = []
    for (start, end) in action_positions:
        lp_start = start - 1
        lp_end = end - 1
        seg_hidden = hidden[lp_start:lp_end]
        segment_tokens = tokens_arr[start:end]
        seg_logits = model.lm_head(seg_hidden)
        seg_logprobs = seg_logits - mx.logsumexp(seg_logits, axis=-1, keepdims=True)
        segment_lp = mx.take_along_axis(
            seg_logprobs,
            segment_tokens[:, None],
            axis=-1,
        ).squeeze(-1)
        action_logprob_segments.append(segment_lp)

    return action_logprob_segments


def _grpo_loss(model, rollout, advantage, clip_eps=0.2, kl_beta=0.04):
    """Compute GRPO loss for a single rollout."""
    import mlx.core as mx

    tokens = rollout['tokens']
    action_positions = rollout['action_positions']
    old_logprobs = rollout['old_logprobs']

    new_logprob_segments = _compute_action_logprobs(model, tokens, action_positions)
    new_lp_flat = mx.concatenate(new_logprob_segments)
    old_lp_flat = mx.array(old_logprobs)

    ratio = mx.exp(new_lp_flat - old_lp_flat)
    clipped_ratio = mx.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    pg_loss = -mx.minimum(ratio * advantage, clipped_ratio * advantage)
    kl = (new_lp_flat - old_lp_flat).mean()
    loss = pg_loss + kl_beta * kl

    n_tokens = len(old_logprobs)
    return loss.mean(), n_tokens


def _compute_advantages(rollouts, eps=1e-8):
    """Group-relative advantages: A_i = (R_i - mean(R)) / (std(R) + eps)."""
    rewards = np.array([r['reward'] for r in rollouts])
    mean_r = rewards.mean()
    std_r = rewards.std()

    if std_r < eps:
        return [0.0] * len(rollouts), mean_r, std_r

    advantages = (rewards - mean_r) / (std_r + eps)
    return advantages.tolist(), mean_r, std_r


def _collect_rollout(
    model, tokenizer, prompt,
    thinking_tokens=256, prediction_tokens=256,
    temperature=0.8, top_p=0.9, seed=None,
):
    """Collect a single rollout: two-stage generation with logprob recording.

    Returns dict with tokens, action_positions, old_logprobs, reward, response_text.
    The reward is computed externally (by the caller via grade_response) because
    the grader needs the taskset domain + task index.
    """
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    messages = [{"role": "user", "content": prompt}]
    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )

    all_tokens = list(full_prompt)
    action_positions = []
    old_logprobs = []
    gen_ids = []

    gen_start = len(all_tokens)

    # Stage 1: Thinking
    if seed is not None:
        mx.random.seed(seed)
    sampler = make_sampler(temp=temperature, top_p=top_p)

    prompt_arr = mx.array(all_tokens)
    think_done = False
    for token_id, logprobs in generate_step(
        prompt_arr, model, max_tokens=thinking_tokens, sampler=sampler
    ):
        gen_ids.append(token_id)
        old_logprobs.append(float(logprobs[token_id]))
        all_tokens.append(token_id)
        if token_id == THINK_CLOSE_TOKEN or token_id == tokenizer.eos_token_id:
            think_done = True
            break

    # Force think close if model didn't emit it
    if not think_done:
        gen_ids.append(THINK_CLOSE_TOKEN)
        old_logprobs.append(0.0)
        all_tokens.append(THINK_CLOSE_TOKEN)
        gen_ids.append(198)  # newline
        old_logprobs.append(0.0)
        all_tokens.append(198)

    # Stage 2: Answer
    if seed is not None:
        mx.random.seed(seed + 1)
    sampler = make_sampler(temp=temperature, top_p=top_p)

    prompt_arr = mx.array(all_tokens)
    for token_id, logprobs in generate_step(
        prompt_arr, model, max_tokens=prediction_tokens, sampler=sampler
    ):
        gen_ids.append(token_id)
        old_logprobs.append(float(logprobs[token_id]))
        all_tokens.append(token_id)
        if token_id == tokenizer.eos_token_id:
            break

    gen_end = len(all_tokens)
    if gen_end > gen_start:
        action_positions.append((gen_start, gen_end))

    response_text = tokenizer.decode(gen_ids)

    return {
        'tokens': all_tokens,
        'action_positions': action_positions,
        'old_logprobs': old_logprobs,
        'response_text': response_text,
        'gen_ids': gen_ids,
    }


class GRPOTrainer:
    """Real GRPO trainer for IL Optimus.

    Manages LoRA model, rollout collection, advantage computation,
    policy gradient updates, and checkpointing.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: GRPOConfig | None = None,
        adapter_path: str = "il_grpo_adapters",
    ):
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as opt

        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GRPOConfig()
        self.adapter_path = adapter_path

        # Memory limits
        if self.config.memory_limit_gb > 0:
            if hasattr(mx, "set_memory_limit"):
                mx.set_memory_limit(int(self.config.memory_limit_gb * 1024**3))
            elif mx.metal.is_available():
                mx.metal.set_memory_limit(int(self.config.memory_limit_gb * 1024**3))
            if hasattr(mx, "set_cache_limit"):
                mx.set_cache_limit(int(1.5 * 1024**3))
            elif mx.metal.is_available():
                mx.metal.set_cache_limit(int(1.5 * 1024**3))
            if hasattr(mx, "set_wired_limit"):
                mx.set_wired_limit(int(self.config.memory_limit_gb * 1024**3))
            elif mx.metal.is_available():
                mx.metal.set_wired_limit(int(self.config.memory_limit_gb * 1024**3))

        self.optimizer = opt.Adam(learning_rate=self.config.learning_rate)
        self.iteration = 0

    def train_step(
        self,
        prompt: str,
        grade_fn: Callable[[str], float],
        on_metrics: Callable[[GRPOMetrics], None] | None = None,
    ) -> GRPOMetrics:
        """One GRPO training step.

        Args:
            prompt: the task prompt
            grade_fn: function that takes a response string and returns a reward (0.0 to 1.0)
            on_metrics: optional callback for streaming metrics

        Returns:
            GRPOMetrics for this step
        """
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten, tree_map

        t0 = time.time()

        # Reset peak memory
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        elif mx.metal.is_available():
            mx.metal.reset_peak_memory()
        mx.clear_cache()

        # Phase 1: Collect G rollouts (no gradients)
        self.model.eval()
        rollouts = []
        rewards = []

        for g in range(self.config.group_size):
            seed = 42 + self.iteration * 1000 + g * 10000
            rollout = _collect_rollout(
                self.model, self.tokenizer, prompt,
                thinking_tokens=self.config.thinking_tokens,
                prediction_tokens=self.config.prediction_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                seed=seed,
            )
            # Grade the response to get the reward
            reward = grade_fn(rollout['response_text'])
            rollout['reward'] = reward
            rollouts.append(rollout)
            rewards.append(reward)
            mx.clear_cache()

        rollout_time = time.time() - t0

        # Phase 2: Compute advantages
        advantages, mean_reward, std_reward = _compute_advantages(rollouts)

        # Phase 3: GRPO update (with gradients)
        self.model.train()
        t1 = time.time()

        loss_sum = 0.0
        n_updated = 0
        grad_accum = None

        for rollout, advantage in zip(rollouts, advantages):
            if abs(advantage) < 1e-8:
                continue

            def loss_fn():
                loss, _ = _grpo_loss(
                    self.model, rollout, advantage,
                    self.config.clip_eps, self.config.kl_beta,
                )
                return loss

            loss_value_and_grad = nn.value_and_grad(self.model, loss_fn)
            loss_val, grad = loss_value_and_grad()

            mx.eval(loss_val, grad)
            loss_sum += float(loss_val)
            n_updated += 1

            if grad_accum is None:
                grad_accum = grad
            else:
                grad_accum = tree_map(lambda x, y: x + y, grad_accum, grad)
            mx.eval(grad_accum)
            mx.clear_cache()

        # Apply gradient update
        if grad_accum is not None and n_updated > 0:
            grad_accum = tree_map(lambda x: x / n_updated, grad_accum)
            self.optimizer.update(self.model, grad_accum)
            mx.eval(self.model.parameters(), self.optimizer.state)

        mx.clear_cache()
        update_time = time.time() - t1
        total_time = time.time() - t0

        # Compute metrics
        peak_mem = 0.0
        if hasattr(mx, "get_peak_memory"):
            peak_mem = mx.get_peak_memory() / 1e9
        elif mx.metal.is_available():
            peak_mem = mx.metal.get_peak_memory() / 1e9

        avg_tokens = np.mean([len(r['tokens']) for r in rollouts])

        metrics = GRPOMetrics(
            iteration=self.iteration,
            mean_reward=mean_reward,
            std_reward=std_reward,
            max_reward=max(rewards),
            min_reward=min(rewards),
            mean_correctness=np.mean([r['reward'] for r in rollouts]),
            mean_reasoning_quality=0.0,  # filled by caller if available
            loss=loss_sum / max(n_updated, 1),
            rollout_time=rollout_time,
            update_time=update_time,
            total_time=total_time,
            peak_memory_gb=peak_mem,
            avg_episode_tokens=float(avg_tokens),
        )

        if on_metrics:
            on_metrics(metrics)

        self.iteration += 1
        return metrics

    def save(self, path: str | None = None):
        """Save LoRA adapter weights."""
        import json
        import os
        import mlx.core as mx
        from mlx.utils import tree_flatten

        path = path or self.adapter_path
        os.makedirs(path, exist_ok=True)
        adapter_weights = dict(tree_flatten(self.model.trainable_parameters()))
        mx.save_safetensors(f"{path}/adapters.safetensors", adapter_weights)
        ckpt = f"{path}/{self.iteration:07d}_adapters.safetensors"
        mx.save_safetensors(ckpt, adapter_weights)
        cfg = {
            "adapter_path": os.path.basename(path),
            "fine_tune_type": "lora",
            "num_layers": 16,
            "lora_parameters": {"rank": 8, "scale": 1.0, "dropout": 0.0},
        }
        with open(f"{path}/adapter_config.json", "w") as f:
            json.dump(cfg, f, indent=4)
