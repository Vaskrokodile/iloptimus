"""Real model loader and inference engine using mlx_lm.

Loads models from HuggingFace (with optional quantization), runs inference
via two-stage generation (reasoning + answer), and manages MLX memory.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# DeepSeek-R1-Distill think tokens
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
EOS = "<｜end▁of▁sentence｜>"


@dataclass
class InferenceResult:
    text: str  # full generated text (reasoning + answer)
    reasoning: str
    answer: str
    elapsed: float
    tokens_generated: int
    tokens_per_sec: float
    forced_answer: bool = False


@dataclass
class ModelHandle:
    """A loaded MLX model + tokenizer, ready for inference."""
    model: object
    tokenizer: object
    model_id: str
    huggingface_id: str
    precision: str
    quantized: bool
    adapter_path: Optional[str] = None
    # cache dir for downloaded models
    cache_dir: str = os.path.expanduser("~/.cache/iloptimus/models")


def _local_model_path(hf_id: str, precision: str, cache_dir: str) -> Path:
    """Get the local path where a converted model should be stored."""
    safe_name = hf_id.replace("/", "_")
    suffix = f"_{precision}" if precision != "fp16" else ""
    return Path(cache_dir) / f"{safe_name}{suffix}"


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


def load_model(
    huggingface_id: str,
    precision: str = "int4",
    adapter_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> ModelHandle:
    """Load an MLX model from HuggingFace.

    Strategy (fastest first):
    1. If a pre-quantized mlx-community version exists, load it directly — no
       conversion needed, download is ~3x smaller.
    2. Otherwise, download the full HF model and convert to MLX quantized
       format locally (cached so subsequent loads are fast).

    If adapter_path is given, loads LoRA adapters on top of the base model.
    """
    import mlx_lm

    cache_dir = cache_dir or os.path.expanduser("~/.cache/iloptimus/models")
    os.makedirs(cache_dir, exist_ok=True)

    local_path = _local_model_path(huggingface_id, precision, cache_dir)
    quantized = precision in ("int4", "int8")
    q_bits = 4 if precision == "int4" else 8

    # ---- Try pre-quantized mlx-community model first (fast path) ----
    mlx_repo = _mlx_community_repo(huggingface_id, precision)
    load_source: str | None = None  # what to pass to mlx_lm.load()

    if mlx_repo and not local_path.exists():
        from huggingface_hub import HfApi
        api = HfApi()
        try:
            api.model_info(mlx_repo)
            print(f"Found pre-quantized model: {mlx_repo} (loading directly, no conversion)")
            load_source = mlx_repo
        except Exception:
            # No pre-quantized version available — fall back to convert
            pass

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
        print(f"Converted in {time.time()-t0:.1f}s -> {local_path}")
        load_source = str(local_path)
    elif load_source is None and local_path.exists():
        load_source = str(local_path)

    # Load the model
    print(f"Loading model from {load_source}...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(load_source)
    print(f"Model loaded in {time.time()-t0:.1f}s")

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
    )


def run_inference(
    handle: ModelHandle,
    prompt: str,
    max_reasoning_tokens: int = 512,
    max_answer_tokens: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> InferenceResult:
    """Run two-stage inference: reasoning then answer.

    Stage 1: Let the model reason freely up to max_reasoning_tokens.
             If it emits </think> naturally, proceed to answer.
    Stage 2: If reasoning budget exhausted without </think>, force it
             and generate the answer.
    """
    import mlx_lm
    from mlx_lm import generate

    messages = [{"role": "user", "content": prompt}]
    chat_text = handle.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    t0 = time.time()
    forced = False

    # Stage 1: reasoning
    out1 = generate(
        handle.model,
        handle.tokenizer,
        prompt=chat_text,
        max_tokens=max_reasoning_tokens,
        verbose=False,
    )
    out1 = out1.strip()
    if EOS in out1:
        out1 = out1.replace(EOS, "").strip()

    if THINK_CLOSE in out1:
        full_text = out1
    else:
        # Force the closing think tag and generate answer
        forced = True
        forced_prompt = chat_text + out1 + THINK_CLOSE + "\n"
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
        full_text = out1 + THINK_CLOSE + "\n" + out2

    elapsed = time.time() - t0

    # Split into reasoning and answer
    if THINK_CLOSE in full_text:
        reasoning, answer = full_text.split(THINK_CLOSE, 1)
        reasoning = reasoning.strip()
        answer = answer.strip()
    else:
        reasoning = ""
        answer = full_text

    # Estimate token count (rough: 1 token ~ 4 chars)
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


def get_memory_info() -> dict:
    """Get current MLX memory usage info."""
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


def clear_cache():
    """Clear MLX cache to free memory."""
    try:
        import mlx.core as mx
        mx.clear_cache()
    except ImportError:
        pass
