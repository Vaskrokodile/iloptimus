"""MLX backend — Apple Silicon inference + LoRA/QLoRA training.

This is the original, heavily tuned path for M-series Macs. The code was moved
here verbatim from ``inference.py`` / ``sft.py`` / ``grpo.py`` so those modules
could become backend-agnostic dispatchers. The MLX-specific optimizations
preserved here include:

- QLoRA training directly on int4 quantized weights (no dequantization)
- Persistent KV cache shared between reasoning and answer stages
- Prompt-lookup speculative decoding via ``mlx-dspark`` (lossless)
- Frozen-prefix caching for SFT (2.2x throughput on 8GB M1)
- Bucketed batch shapes for compiled-graph reuse
- Action-position-only lm_head application in GRPO (memory optimization)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Iterator, Optional

from .base import (
    Backend,
    EOS,
    GenerateChunk,
    GenerateResult,
    GRPOConfig,
    GRPOMetrics,
    GRPOTrainerLike,
    InferenceResult,
    ModelHandle,
    SFTConfig,
    SFTExample,
    SFTMetrics,
    THINK_CLOSE,
    THINK_CLOSE_TOKEN_ID,
    THINK_OPEN,
    is_reasoning_model,
)


def _local_model_path(hf_id: str, precision: str, cache_dir: str):
    """Get the local path where a converted model should be stored."""
    from pathlib import Path

    safe_name = hf_id.replace("/", "_")
    suffix = f"_{precision}" if precision != "fp16" else ""
    return Path(cache_dir) / f"{safe_name}{suffix}"


# In-memory cache of mlx-community repo lookups to avoid repeated network calls
_mlx_repo_cache: dict[str, Optional[str]] = {}


def _mlx_community_repo(huggingface_id: str, precision: str) -> Optional[str]:
    """Derive the mlx-community pre-quantized repo name for a model.

    mlx-community publishes pre-quantized MLX models following the naming
    convention: mlx-community/{ModelName}-{bits}bit (e.g.
    mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit). Loading these directly
    avoids the slow download-full-model + local-convert step.

    Returns the mlx-community repo id, or None if precision is fp16 or the
    model name can't be mapped.
    """
    if precision not in ("int4", "int8", "int3", "int6"):
        return None
    bits = {"int3": "3", "int4": "4", "int6": "6", "int8": "8"}[precision]
    model_name = huggingface_id.split("/")[-1]
    return f"mlx-community/{model_name}-{bits}bit"


def _set_mlx_memory_limits():
    """Set conservative MLX memory limits for 8GB Apple Silicon."""
    import mlx.core as mx

    if mx.metal.is_available():
        try:
            mx.set_memory_limit(int(3.5 * 1024**3))
            mx.set_cache_limit(int(1.0 * 1024**3))
            mx.set_wired_limit(int(3.5 * 1024**3))
        except Exception:
            try:
                mx.metal.set_memory_limit(int(3.5 * 1024**3))
                mx.metal.set_cache_limit(int(1.0 * 1024**3))
                mx.metal.set_wired_limit(int(3.5 * 1024**3))
            except Exception:
                pass


class MLXBackend(Backend):
    name = "mlx"

    # ------------------------------------------------------------------ load

    def load(
        self,
        *,
        huggingface_id: str,
        precision: str = "int4",
        adapter_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        source_override: Optional[str] = None,
    ) -> ModelHandle:
        """Load an MLX model from HuggingFace.

        Strategy (fastest first):
        1. If a pre-quantized mlx-community version exists, load it directly — no
           conversion needed, download is ~3x smaller.
        2. Otherwise, download the full HF model and convert to MLX quantized
           format locally (cached so subsequent loads are fast).

        If adapter_path is given, loads LoRA adapters on top of the base model.
        """
        import gc

        import mlx.core as mx
        import mlx_lm

        gc.collect()
        mx.clear_cache()
        _set_mlx_memory_limits()

        cache_dir = cache_dir or os.path.expanduser("~/.cache/iloptimus/models")
        os.makedirs(cache_dir, exist_ok=True)

        local_path = _local_model_path(huggingface_id, precision, cache_dir)
        quantized = precision in ("int4", "int8")
        q_bits = 4 if precision == "int4" else 8

        # ---- Try pre-quantized mlx-community model first (fast path) ----
        mlx_repo = _mlx_community_repo(huggingface_id, precision)
        load_source: str | None = source_override

        if load_source is None and mlx_repo and not local_path.exists():
            cache_key = f"{huggingface_id}_{precision}"
            if cache_key in _mlx_repo_cache:
                load_source = _mlx_repo_cache[cache_key]
            elif os.environ.get("HF_HUB_OFFLINE") == "1":
                try:
                    from huggingface_hub import try_to_load_from_cache

                    cached = try_to_load_from_cache(mlx_repo, "config.json")
                    if cached is not None and os.path.exists(cached):
                        load_source = mlx_repo
                        print(f"Using cached pre-quantized model: {mlx_repo}")
                except Exception:
                    pass
                _mlx_repo_cache[cache_key] = load_source
            else:
                from huggingface_hub import HfApi

                api = HfApi()
                try:
                    api.model_info(mlx_repo)
                    print(f"Found pre-quantized model: {mlx_repo} (loading directly, no conversion)")
                    load_source = mlx_repo
                except Exception:
                    load_source = None
                _mlx_repo_cache[cache_key] = load_source

        # ---- Fall back to download + local convert ----
        if load_source is None and not local_path.exists():
            from mlx_lm import convert

            print(f"Downloading and converting {huggingface_id} to MLX ({precision})...")
            t0 = time.time()
            convert(
                hf_path=huggingface_id,
                mlx_path=str(local_path),
                quantize=quantized,
                q_bits=q_bits,
            )
            print(f"Converted in {time.time() - t0:.1f}s -> {local_path}")
            load_source = str(local_path)
        elif load_source is None and local_path.exists():
            load_source = str(local_path)

        # Load the model
        print(f"Loading model from {load_source}...")
        t0 = time.time()
        model, tokenizer = mlx_lm.load(load_source)
        print(f"Model loaded in {time.time() - t0:.1f}s")

        # Load LoRA adapter if provided
        if adapter_path and os.path.exists(adapter_path):
            from mlx_lm.tuner.utils import load_adapters

            print(f"Loading LoRA adapters from {adapter_path}...")
            load_adapters(model, adapter_path)
            print("Adapters loaded")

        return ModelHandle(
            model=model,
            tokenizer=tokenizer,
            model_id=huggingface_id.split("/")[-1],
            huggingface_id=huggingface_id,
            precision=precision,
            quantized=quantized,
            adapter_path=adapter_path,
            cache_dir=cache_dir,
            backend="mlx",
        )

    # ------------------------------------------------------------- generate

    def generate(
        self,
        handle: ModelHandle,
        prompt_text: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        repetition_context_size: int = 128,
        stop_strings: Optional[list[str]] = None,
        stop_token_ids: Optional[list[int]] = None,
        return_logprobs: bool = False,
    ) -> GenerateResult:
        import mlx.core as mx
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        sampler = make_sampler(temp=temperature, top_p=top_p) if temperature > 0 else make_sampler(temp=0)
        logits_processors = make_logits_processors(
            repetition_penalty=repetition_penalty, repetition_context_size=repetition_context_size
        )
        text = generate(
            handle.model,
            handle.tokenizer,
            prompt=prompt_text,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            verbose=False,
        )
        mx.clear_cache()
        # mlx_lm.generate does not expose finish_reason. Infer it: if the
        # decoded text tokenizes to >= max_tokens, the budget was exhausted
        # (length); otherwise the model stopped naturally (stop / EOS).
        try:
            gen_token_count = len(handle.tokenizer.encode(text))
            finish_reason = "length" if gen_token_count >= max_tokens else "stop"
        except Exception:
            finish_reason = "stop"
        return GenerateResult(text=text, finish_reason=finish_reason)

    def stream_generate(
        self,
        handle: ModelHandle,
        prompt_text: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        repetition_context_size: int = 128,
    ) -> Iterator[GenerateChunk]:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        sampler = make_sampler(temp=temperature, top_p=top_p) if temperature > 0 else make_sampler(temp=0)
        processors = make_logits_processors(
            repetition_penalty=repetition_penalty, repetition_context_size=repetition_context_size
        )
        stream = stream_generate(
            handle.model,
            handle.tokenizer,
            prompt=prompt_text,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=processors,
        )
        try:
            for response in stream:
                yield GenerateChunk(
                    text=response.text,
                    token_id=-1,
                    generation_tokens=int(response.generation_tokens),
                )
        finally:
            stream.close()

    # --------------------------------------------------- two-stage inference

    def _run_inference_speculative(
        self,
        handle: ModelHandle,
        prompt: str,
        max_reasoning_tokens: int = 512,
        max_answer_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_draft_tokens: int = 6,
        long_draft_tokens: int = 32,
        ngram_min: int = 3,
        ngram_max: int = 4,
    ) -> InferenceResult:
        from mlx_dspark import lookup_generate
        from mlx_dspark.target import Target

        messages = [{"role": "user", "content": prompt}]
        chat_text = handle.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        t0 = time.time()
        target = Target(handle.model, handle.tokenizer)
        total_max_tokens = max_reasoning_tokens + max_answer_tokens
        result = lookup_generate(
            target,
            handle.tokenizer,
            prompt=chat_text,
            max_new_tokens=total_max_tokens,
            max_draft_tokens=max_draft_tokens,
            long_draft_tokens=long_draft_tokens,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p,
            apply_chat_template=False,
            stop=[THINK_CLOSE, EOS],
        )

        raw_text = result.text
        if EOS in raw_text:
            raw_text = raw_text.replace(EOS, "")

        elapsed = time.time() - t0
        forced = False
        if THINK_CLOSE in raw_text:
            reasoning, answer = raw_text.split(THINK_CLOSE, 1)
            reasoning = reasoning.strip()
            answer = answer.strip()
        else:
            reasoning = raw_text.strip()
            forced = True
            from mlx_lm import generate

            forced_prompt = chat_text + raw_text + THINK_CLOSE + "\n<answer>The answer is "
            out2 = generate(
                handle.model,
                handle.tokenizer,
                prompt=forced_prompt,
                max_tokens=max_answer_tokens,
                verbose=False,
            )
            out2 = out2.strip()
            if EOS in out2:
                out2 = out2.replace(EOS, "").strip()
            answer = out2

        full_text = reasoning + THINK_CLOSE + "\n" + answer
        total_tokens = len(result.token_ids)
        tps = total_tokens / elapsed if elapsed > 0 else 0.0
        return InferenceResult(
            text=full_text,
            reasoning=reasoning,
            answer=answer,
            elapsed=elapsed,
            tokens_generated=total_tokens,
            tokens_per_sec=tps,
            forced_answer=forced,
        )

    def run_two_stage_inference(
        self,
        handle: ModelHandle,
        prompt: str,
        *,
        max_reasoning_tokens: int = 512,
        max_answer_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.9,
        speculative: bool = False,
        speculative_config: dict | None = None,
    ) -> InferenceResult:
        """Run two-stage inference: reasoning then answer.

        Uses generate_step with a persistent KV cache so the answer stage
        continues from where reasoning left off — no reprocessing of the prompt
        or reasoning tokens. Early-stops when the think-close token is emitted.
        """
        if speculative:
            config = speculative_config or {}
            return self._run_inference_speculative(
                handle,
                prompt,
                max_reasoning_tokens=max_reasoning_tokens,
                max_answer_tokens=max_answer_tokens,
                temperature=temperature,
                top_p=top_p,
                **config,
            )

        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        messages = [{"role": "user", "content": prompt}]
        chat_tokens = handle.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

        t0 = time.time()
        prompt_cache = make_prompt_cache(handle.model)
        prompt_arr = mx.array(chat_tokens)
        eos_token_id = handle.tokenizer.eos_token_id

        sampler = make_sampler(temp=temperature, top_p=top_p) if temperature > 0 else make_sampler(temp=0)
        all_tokens = []
        reasoning_tokens = []
        answer_tokens = []
        think_done = False
        forced = False

        total_max_tokens = max_reasoning_tokens + max_answer_tokens

        for token_id, logprobs in generate_step(
            prompt_arr,
            handle.model,
            max_tokens=total_max_tokens,
            sampler=sampler,
            prompt_cache=prompt_cache,
        ):
            all_tokens.append(token_id)

            if not think_done:
                reasoning_tokens.append(token_id)
                if token_id == THINK_CLOSE_TOKEN_ID:
                    think_done = True
                    if len(all_tokens) >= max_reasoning_tokens:
                        forced = True
                elif token_id == eos_token_id:
                    # Model stopped at EOS during reasoning without closing
                    # the think tag. The text so far is reasoning, not an
                    # answer — force a second pass to get the actual answer.
                    think_done = True
                    forced = True
                elif len(reasoning_tokens) >= max_reasoning_tokens:
                    think_done = True
                    forced = True
                    break
            else:
                answer_tokens.append(token_id)
                if token_id == eos_token_id:
                    break
                if len(answer_tokens) >= max_answer_tokens:
                    break

        reasoning_text = handle.tokenizer.decode(reasoning_tokens)
        if EOS in reasoning_text:
            reasoning_text = reasoning_text.replace(EOS, "").strip()

        if forced:
            from mlx_lm import generate

            chat_text = handle.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            forced_prompt = chat_text + reasoning_text + THINK_CLOSE + "\n"
            out2 = generate(
                handle.model,
                handle.tokenizer,
                prompt=forced_prompt,
                max_tokens=max_answer_tokens,
                verbose=False,
            )
            out2 = out2.strip()
            if EOS in out2:
                out2 = out2.replace(EOS, "").strip()
            full_text = reasoning_text + THINK_CLOSE + "\n" + out2
        else:
            answer_text = handle.tokenizer.decode(answer_tokens)
            if EOS in answer_text:
                answer_text = answer_text.replace(EOS, "").strip()
            full_text = reasoning_text + answer_text

        elapsed = time.time() - t0
        mx.clear_cache()

        if THINK_CLOSE in full_text:
            reasoning, answer = full_text.split(THINK_CLOSE, 1)
            reasoning = reasoning.strip()
            answer = answer.strip()
        else:
            reasoning = ""
            answer = full_text

        tokens_generated = len(full_text) // 4
        tps = tokens_generated / elapsed if elapsed > 0 else 0.0
        return InferenceResult(
            text=full_text,
            reasoning=reasoning,
            answer=answer,
            elapsed=elapsed,
            tokens_generated=tokens_generated,
            tokens_per_sec=tps,
            forced_answer=forced,
        )

    # ------------------------------------------------------- memory/adapters

    def clear_cache(self, handle: Optional[ModelHandle] = None) -> None:
        try:
            import mlx.core as mx

            mx.clear_cache()
        except ImportError:
            pass

    def release_memory(self) -> None:
        import gc

        import mlx.core as mx

        gc.collect()
        mx.clear_cache()

    def get_memory_info(self) -> dict:
        try:
            import mlx.core as mx

            info = {}
            if hasattr(mx, "get_peak_memory"):
                info["peak_memory_gb"] = mx.get_peak_memory() / 1e9
            elif hasattr(mx, "metal") and mx.metal.is_available():
                info["peak_memory_gb"] = mx.metal.get_peak_memory() / 1e9
            if hasattr(mx, "get_active_memory"):
                info["active_memory_gb"] = mx.get_active_memory() / 1e9
            elif hasattr(mx, "metal") and mx.metal.is_available():
                info["active_memory_gb"] = mx.metal.get_active_memory() / 1e9
            return info
        except ImportError:
            return {}

    def swap_adapters(self, handle: ModelHandle, adapter_path: Optional[str]) -> ModelHandle:
        """Swap LoRA adapters without reloading the base model."""
        import mlx.core as mx
        from mlx_lm.tuner.utils import load_adapters

        model = handle.model
        has_lora = any("lora" in k.lower() for k, _ in model.named_modules())
        if has_lora:
            from mlx_lm.tuner.utils import remove_lora_layers

            model = remove_lora_layers(model)
            mx.eval(model.parameters())
            mx.clear_cache()

        if adapter_path and os.path.exists(adapter_path):
            load_adapters(model, adapter_path)
            mx.eval(model.parameters())
            print(f"Swapped to adapters: {adapter_path}")

        handle.model = model
        handle.adapter_path = adapter_path
        return handle

    def _dequantize_model(self, model):
        """Replace all QuantizedLinear/QuantizedEmbedding layers with regular ones.

        Legacy path — QLoRA (training directly on int4) is preferred and does not
        require this. Kept for backward compatibility.
        """
        import mlx.core as mx
        import mlx.nn as nn

        def _replace(module):
            for k, v in list(module.items()):
                if isinstance(v, nn.QuantizedLinear):
                    w, scales = v.weight, v.scales
                    biases = v.biases if hasattr(v, "biases") else None
                    out_dims = w.shape[0]
                    in_dims = w.shape[1] * 8
                    group_size = in_dims // scales.shape[1]
                    deq_w = mx.dequantize(w, scales=scales, biases=biases, group_size=group_size, bits=4)
                    mx.eval(deq_w)
                    new = nn.Linear(in_dims, out_dims, bias=True)
                    new.weight = deq_w
                    if hasattr(v, "bias") and v.bias is not None:
                        new.bias = v.bias
                    module[k] = new
                    del w, scales, deq_w, v
                    mx.clear_cache()
                elif isinstance(v, nn.QuantizedEmbedding):
                    w, scales = v.weight, v.scales
                    biases = v.biases if hasattr(v, "biases") else None
                    num_emb = w.shape[0]
                    emb_dim = w.shape[1] * 8
                    group_size = emb_dim // scales.shape[1]
                    deq_w = mx.dequantize(w, scales=scales, biases=biases, group_size=group_size, bits=4)
                    mx.eval(deq_w)
                    new = nn.Embedding(num_emb, emb_dim)
                    new.weight = deq_w
                    module[k] = new
                    del w, scales, deq_w, v
                    mx.clear_cache()
                elif hasattr(v, "items") and not isinstance(v, (nn.Linear, nn.Embedding)):
                    _replace(v)

        _replace(model)
        _replace(model.model)
        for layer in model.model.layers:
            _replace(layer)
        mx.clear_cache()

    # ------------------------------------------------------------------ SFT

    def run_sft(
        self,
        handle: ModelHandle,
        examples: list[SFTExample],
        config: SFTConfig,
        adapter_path: str,
        on_metrics: Callable[[SFTMetrics], None] | None = None,
    ) -> str:
        """Run LoRA SFT training on the model (mlx_lm tuner)."""
        import numpy as np

        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as opt
        from mlx_lm.models.base import create_attention_mask
        from mlx_lm.tuner.callbacks import TrainingCallback
        from mlx_lm.tuner.trainer import TrainingArgs, default_loss, iterate_batches, train

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

        from mlx.utils import tree_flatten
        from mlx_lm.tuner.lora import LoRALinear
        from mlx_lm.tuner.utils import linear_to_lora_layers

        has_lora = any(isinstance(m, LoRALinear) for _, m in handle.model.named_modules())
        num_layers = min(config.lora_layers, len(handle.model.layers))
        if has_lora:
            print("SFT: Model already has LoRA layers (from loaded adapter) — training on top of them")
            lora_children = set()
            for name, mod in handle.model.named_modules():
                if isinstance(mod, LoRALinear):
                    for child_name, _ in mod.named_modules():
                        if child_name:
                            lora_children.add(f"{name}.{child_name}" if name else child_name)
            for name, m in handle.model.named_modules():
                if not name or name in lora_children:
                    continue
                if isinstance(m, LoRALinear):
                    if hasattr(m, "linear"):
                        m.linear.freeze()
                elif isinstance(m, nn.Module):
                    if not list(m.children()):
                        m.freeze()
        else:
            handle.model.freeze()
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
        train_data, data_stats = _tokenize_sft_rows(
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
            """Use bounded stable shapes so MLX reuses compiled training graphs."""
            kwargs.setdefault("seed", config.seed)
            bucket = max(32, config.compile_bucket_size)
            maximum = int(kwargs.get("max_seq_length") or config.max_seq_length)
            for batch, lengths in iterate_batches(*args, **kwargs):
                current = int(batch.shape[1])
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

    # ----------------------------------------------------------------- GRPO

    def make_grpo_trainer(
        self,
        handle: ModelHandle,
        config: GRPOConfig,
        adapter_path: str,
    ) -> "MLXGRPOTrainer":
        return MLXGRPOTrainer(handle, config, adapter_path)


# ---------------------------------------------------------------------------
# SFT tokenization helper (MLX-specific — uses mlx_lm.tuner.datasets)
# ---------------------------------------------------------------------------


class EagerCompletionDataset:
    """Pre-tokenized rows with real lengths available to MLX's batch sorter."""

    def __init__(self, rows: list[tuple[list[int], int]]) -> None:
        self.rows = rows

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.rows[index]

    def __len__(self) -> int:
        return len(self.rows)


def _completion_row_tokens(row: dict[str, str], tokenizer: Any) -> tuple[list[int], int]:
    """Tokenize one prompt/completion row and return ``(tokens, offset)``.

    Backend-independent re-implementation of the offset computation that
    ``mlx_lm.tuner.datasets.CompletionsDataset.process`` performs: render the
    prompt alone for the offset, then prompt+completion for the full sequence.
    Keeping this local means SFT data preparation works on every backend
    (including the torch/CUDA path on platforms without ``mlx_lm``).
    """
    prompt = str(row.get("prompt") or "")
    completion = str(row.get("completion") or "")
    prompt_tokens = list(
        tokenizer.apply_chat_template([{"role": "user", "content": prompt}])
    )
    full_tokens = list(
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]
        )
    )
    return full_tokens, len(prompt_tokens)


def _tokenize_sft_rows(
    rows: list[dict[str, str]], tokenizer: Any, *, max_seq_length: int, mask_prompt: bool = True
) -> tuple[EagerCompletionDataset, dict[str, Any]]:
    """Tokenize once and prove how much supervised completion survives truncation."""
    tokenized: list[tuple[list[int], int]] = []
    total_completion_tokens = 0
    retained_completion_tokens = 0
    fully_retained = 0
    sequence_lengths: list[int] = []
    for row in rows:
        tokens, offset = _completion_row_tokens(row, tokenizer)
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


# ---------------------------------------------------------------------------
# MLX GRPO trainer — preserves the original rollout + update logic
# ---------------------------------------------------------------------------


def _compute_action_logprobs(model, tokens, action_positions):
    """Forward backbone only, apply LM head ONLY at action positions."""
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


def _collect_rollout(
    model, tokenizer, prompt,
    thinking_tokens=256, prediction_tokens=256,
    temperature=0.8, top_p=0.9, seed=None,
):
    """Collect a single rollout: two-stage generation with logprob recording."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
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

    prompt_cache = make_prompt_cache(model)

    if seed is not None:
        mx.random.seed(seed)
    sampler = make_sampler(temp=temperature, top_p=top_p)

    prompt_arr = mx.array(all_tokens)
    think_done = False
    for token_id, logprobs in generate_step(
        prompt_arr, model, max_tokens=thinking_tokens, sampler=sampler,
        prompt_cache=prompt_cache,
    ):
        gen_ids.append(token_id)
        old_logprobs.append(float(logprobs[token_id]))
        all_tokens.append(token_id)
        if token_id == THINK_CLOSE_TOKEN_ID or token_id == tokenizer.eos_token_id:
            think_done = True
            break

    if not think_done:
        gen_ids.append(THINK_CLOSE_TOKEN_ID)
        old_logprobs.append(0.0)
        all_tokens.append(THINK_CLOSE_TOKEN_ID)
        gen_ids.append(198)  # newline
        old_logprobs.append(0.0)
        all_tokens.append(198)

    if seed is not None:
        mx.random.seed(seed + 1)
    sampler = make_sampler(temp=temperature, top_p=top_p)

    prompt_arr = mx.array(all_tokens)
    for token_id, logprobs in generate_step(
        prompt_arr, model, max_tokens=prediction_tokens, sampler=sampler,
        prompt_cache=prompt_cache,
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


class MLXGRPOTrainer(GRPOTrainerLike):
    """Real GRPO trainer for Optimus Studio (MLX)."""

    def __init__(
        self,
        handle: ModelHandle,
        config: GRPOConfig,
        adapter_path: str = "il_grpo_adapters",
    ):
        import mlx.core as mx
        import mlx.optimizers as opt

        self.handle = handle
        self.model = handle.model
        self.tokenizer = handle.tokenizer
        self.config = config
        self.adapter_path = adapter_path

        from mlx_lm.tuner.lora import LoRALinear

        has_lora = any(isinstance(m, LoRALinear) for _, m in self.model.named_modules())
        if not has_lora:
            from mlx_lm.tuner.utils import linear_to_lora_layers

            self.model.freeze()
            num_layers = min(8, len(self.model.layers))
            lora_config = {"rank": 8, "scale": 1.0, "dropout": 0.0}
            linear_to_lora_layers(self.model, num_layers, lora_config)
            print(f"GRPO: Applied LoRA layers (rank 8) to {num_layers} layers")
        else:
            print("GRPO: Model already has LoRA layers (from SFT/adapter) — reusing them for RL")
            import mlx.nn as nn

            lora_children = set()
            for name, mod in self.model.named_modules():
                if isinstance(mod, LoRALinear):
                    for child_name, _ in mod.named_modules():
                        if child_name:
                            lora_children.add(f"{name}.{child_name}" if name else child_name)
            for name, m in self.model.named_modules():
                if not name or name in lora_children:
                    continue
                if isinstance(m, LoRALinear):
                    if hasattr(m, "linear"):
                        m.linear.freeze()
                elif isinstance(m, nn.Module):
                    if not list(m.children()):
                        m.freeze()

        if self.config.memory_limit_gb > 0:
            if hasattr(mx, "set_memory_limit"):
                mx.set_memory_limit(int(self.config.memory_limit_gb * 1024**3))
            elif mx.metal.is_available():
                mx.metal.set_memory_limit(int(self.config.memory_limit_gb * 1024**3))
            if hasattr(mx, "set_cache_limit"):
                mx.set_cache_limit(int(1.0 * 1024**3))
            elif mx.metal.is_available():
                mx.metal.set_cache_limit(int(1.0 * 1024**3))
            if hasattr(mx, "set_wired_limit"):
                mx.set_wired_limit(int(self.config.memory_limit_gb * 1024**3))
            elif mx.metal.is_available():
                mx.metal.set_wired_limit(int(self.config.memory_limit_gb * 1024**3))

        # SGD — Adam's second moment estimate produces NaN with int4 QLoRA
        self.optimizer = opt.SGD(learning_rate=self.config.learning_rate)
        self.iteration = 0

    def train_step(
        self,
        prompt: str,
        grade_fn: Callable[[str], float],
        on_metrics: Callable[[GRPOMetrics], None] | None = None,
    ) -> GRPOMetrics:
        import numpy as np

        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_map

        t0 = time.time()

        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        elif mx.metal.is_available():
            mx.metal.reset_peak_memory()
        mx.clear_cache()

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
            reward = grade_fn(rollout['response_text'])
            rollout['reward'] = reward
            rollouts.append(rollout)
            rewards.append(reward)

        rollout_time = time.time() - t0

        advantages, mean_reward, std_reward = _compute_advantages(rollouts)

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
            loss_f = float(loss_val)
            if loss_f != loss_f or loss_f in (float("inf"), float("-inf")):
                continue

            loss_sum += loss_f
            n_updated += 1

            if grad_accum is None:
                grad_accum = grad
            else:
                grad_accum = tree_map(lambda x, y: x + y, grad_accum, grad)
            mx.eval(grad_accum)
            mx.clear_cache()

        if grad_accum is not None and n_updated > 0:
            grad_accum = tree_map(lambda x: x / n_updated, grad_accum)
            if self.config.grad_clip > 0:
                grad_accum = tree_map(
                    lambda x: mx.clip(x, -self.config.grad_clip, self.config.grad_clip),
                    grad_accum,
                )
            self.optimizer.update(self.model, grad_accum)
            mx.eval(self.model.parameters(), self.optimizer.state)

        mx.clear_cache()
        update_time = time.time() - t1
        total_time = time.time() - t0

        peak_mem = 0.0
        if hasattr(mx, "get_peak_memory"):
            peak_mem = mx.get_peak_memory() / 1e9
        elif mx.metal.is_available():
            peak_mem = mx.metal.get_peak_memory() / 1e9

        avg_tokens = np.mean([len(r['tokens']) for r in rollouts])

        metrics = GRPOMetrics(
            iteration=self.iteration,
            mean_reward=float(mean_reward) if mean_reward == mean_reward else 0.0,
            std_reward=float(std_reward) if std_reward == std_reward else 0.0,
            max_reward=max(rewards) if rewards else 0.0,
            min_reward=min(rewards) if rewards else 0.0,
            mean_correctness=float(np.mean([r['reward'] for r in rollouts])) if rollouts else 0.0,
            mean_reasoning_quality=0.0,
            loss=loss_sum / max(n_updated, 1) if n_updated > 0 else 0.0,
            rollout_time=rollout_time,
            update_time=update_time,
            total_time=total_time,
            peak_memory_gb=peak_mem,
            avg_episode_tokens=float(avg_tokens),
        )

        if on_metrics:
            on_metrics(metrics)

        self.iteration += 1
        mx.clear_cache()
        return metrics

    def save(self, path: str | None = None):
        import mlx.core as mx
        from mlx.utils import tree_flatten

        path = path or self.adapter_path
        os.makedirs(path, exist_ok=True)
        adapter_weights = dict(tree_flatten(self.model.trainable_parameters()))
        mx.save_safetensors(f"{path}/adapters.safetensors", adapter_weights)
        ckpt = f"{path}/{self.iteration:07d}_adapters.safetensors"
        mx.save_safetensors(ckpt, adapter_weights)
        num_layers = min(8, len(self.model.layers))
        cfg = {
            "adapter_path": os.path.basename(path),
            "fine_tune_type": "lora",
            "num_layers": num_layers,
            "lora_parameters": {"rank": 8, "scale": 1.0, "dropout": 0.0},
        }
        with open(f"{path}/adapter_config.json", "w") as f:
            json.dump(cfg, f, indent=4)


# Shared advantage computation (kept here so the grpo module can re-use it)
def _compute_advantages(rollouts, eps=1e-8):
    import numpy as np

    rewards = np.array([r['reward'] for r in rollouts])
    mean_r = rewards.mean()
    std_r = rewards.std()

    if std_r < eps:
        return [0.0] * len(rollouts), mean_r, std_r

    advantages = (rewards - mean_r) / (std_r + eps)
    return advantages.tolist(), mean_r, std_r
