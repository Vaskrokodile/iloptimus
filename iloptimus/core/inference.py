"""Real model loader and inference engine using mlx_lm.

Loads models from HuggingFace (with optional quantization), runs inference
via two-stage generation (reasoning + answer), and manages MLX memory.

Optimizations:
- QLoRA: trains directly on int4 quantized models (no dequantization needed)
- KV cache reuse: reasoning and answer stages share a persistent KV cache
- Early stopping: stops generation when think-close token is emitted
- Adapter hot-swapping: swap LoRA adapters without reloading the base model
- explicit local snapshot loading after a model is downloaded
- Speculative decoding: optional prompt-lookup n-gram speculation via mlx-dspark
  (lossless, helps when output copies from context — code reproduction, RAG,
  summarization; neutral or slightly slower for novel reasoning text)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# DeepSeek-R1-Distill think tokens — actual native tags (token 151648 / 151649)
# Using chr() to avoid the tags being interpreted as HTML by tooling
THINK_OPEN = chr(60) + "think" + chr(62)           # <think>
THINK_CLOSE = chr(60) + "/think" + chr(62)          # </think>
EOS = chr(60) + "\uff5cend\u2581of\u2581sentence\uff5c" + chr(62)  # <｜end▁of▁sentence｜>

# Token IDs for DeepSeek-R1-Distill (Qwen2 tokenizer)
THINK_CLOSE_TOKEN_ID = 151649


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


def run_completion(
    handle: ModelHandle,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> InferenceResult:
    """Single-pass completion for structured agent protocols.

    Unlike ``run_inference``, this never injects a forced natural-language
    answer prefix when a reasoning budget expires. That distinction is
    essential for tool JSON: a controller must receive the model's actual
    completion, not a fabricated ``The answer is`` continuation.
    """
    import mlx.core as mx
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    messages = [{"role": "user", "content": prompt}]
    chat_text = handle.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    started = time.time()
    sampler = make_sampler(temp=temperature, top_p=0.9) if temperature > 0 else make_sampler(temp=0)
    is_reasoning_model = "deepseek-r1" in handle.huggingface_id.lower()
    reasoning = ""
    if is_reasoning_model:
        # Agent prompts typically place the native tool JSON just after a
        # substantial reasoning trace. Preserve enough of that first pass to
        # reach the model's own closing tag; short conversational completions
        # still split evenly so they retain answer room.
        reasoning_budget = max(32, int(max_tokens * (0.75 if max_tokens >= 256 else 0.5)))
        answer_budget = max(32, max_tokens - reasoning_budget)
        first = generate(
            handle.model,
            handle.tokenizer,
            prompt=chat_text,
            max_tokens=reasoning_budget,
            sampler=sampler,
            verbose=False,
        ).strip()
        if THINK_CLOSE in first:
            reasoning, text = first.split(THINK_CLOSE, 1)
            text = text.strip()
        else:
            reasoning = first
            text = ""
        if not text:
            # DeepSeek-R1 often consumes a short completion entirely in thought.
            # Close that phase and let it emit the real answer or tool object. We
            # intentionally add no answer prefix or fabricated content.
            answer_prompt = chat_text + first + ("" if THINK_CLOSE in first else THINK_CLOSE) + "\n"
            text = generate(
                handle.model,
                handle.tokenizer,
                prompt=answer_prompt,
                max_tokens=answer_budget,
                sampler=sampler,
                verbose=False,
            ).strip()
    else:
        text = generate(
            handle.model,
            handle.tokenizer,
            prompt=chat_text,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        ).strip()
    if THINK_CLOSE in text:
        extra_reasoning, text = text.rsplit(THINK_CLOSE, 1)
        reasoning = (reasoning + "\n" + extra_reasoning).strip()
        text = text.strip()
    if text.startswith("<answer>") and text.endswith("</answer>"):
        text = text[len("<answer>") : -len("</answer>")].strip()
    elapsed = time.time() - started
    try:
        tokens = len(handle.tokenizer.encode(reasoning + text))
    except Exception:
        tokens = max(1, len(text) // 4)
    mx.clear_cache()
    return InferenceResult(
        text=(reasoning + THINK_CLOSE + "\n" + text) if reasoning else text,
        reasoning=reasoning,
        answer=text,
        elapsed=elapsed,
        tokens_generated=tokens,
        tokens_per_sec=tokens / max(elapsed, 1e-6),
    )


def run_tool_completion(
    handle: ModelHandle,
    prompt: str,
    tool_name: str,
    max_tokens: int = 384,
    temperature: float = 0.1,
    fixed_arguments: dict[str, str] | None = None,
    next_argument: str | None = None,
    next_argument_prefix: str = "",
) -> InferenceResult:
    """Generate arguments inside a fixed single-tool JSON envelope.

    Small reasoning models often explain a required tool call instead of
    emitting it. Supplying the protocol-only prefix is equivalent to grammar
    constrained decoding: the model still authors every argument and file
    byte, while the harness guarantees the outer call shape.
    """
    import mlx.core as mx
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    messages = [{"role": "user", "content": prompt}]
    chat_text = handle.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix = json.dumps({"tool_name": tool_name}, ensure_ascii=False)[:-1] + ', "arguments": {'
    fixed_arguments = fixed_arguments or {}
    fields = [f"{json.dumps(key)}: {json.dumps(value)}" for key, value in fixed_arguments.items()]
    if fields:
        prefix += ", ".join(fields) + ", "
    if next_argument:
        encoded_prefix = json.dumps(next_argument_prefix, ensure_ascii=False)[1:-1]
        prefix += f'{json.dumps(next_argument)}: "{encoded_prefix}'
    sampler = make_sampler(temp=temperature, top_p=0.9) if temperature > 0 else make_sampler(temp=0)
    reasoning = ""
    if "deepseek-r1" in handle.huggingface_id.lower():
        reasoning = generate(
            handle.model,
            handle.tokenizer,
            prompt=chat_text,
            max_tokens=min(192, max(96, max_tokens // 2)),
            sampler=sampler,
            verbose=False,
        ).strip()
        if THINK_CLOSE in reasoning:
            reasoning = reasoning.split(THINK_CLOSE, 1)[0].strip()
        generation_prompt = chat_text + reasoning + THINK_CLOSE + "\n" + prefix
    else:
        generation_prompt = chat_text + prefix
    started = time.time()
    continuation = generate(
        handle.model,
        handle.tokenizer,
        prompt=generation_prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    ).strip()
    text = prefix + continuation
    elapsed = time.time() - started
    try:
        tokens = len(handle.tokenizer.encode(reasoning + continuation))
    except Exception:
        tokens = max(1, len(continuation) // 4)
    mx.clear_cache()
    return InferenceResult(
        text=text,
        reasoning=reasoning,
        answer=text,
        elapsed=elapsed,
        tokens_generated=tokens,
        tokens_per_sec=tokens / max(elapsed, 1e-6),
    )


def run_source_completion(
    handle: ModelHandle,
    prompt: str,
    path: str,
    max_tokens: int = 384,
    temperature: float = 0.1,
) -> InferenceResult:
    """Generate plain source text for a trusted write-file wrapper."""
    import mlx.core as mx
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    language = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".sh": "bash",
    }.get(Path(path).suffix.lower(), "text")
    signatures = re.findall(r"\bimplement(?:ing)?\s+([A-Za-z_]\w*\([^)]*\))", prompt, flags=re.IGNORECASE)
    contract = ""
    if signatures:
        contract = " The file MUST define exactly this requested callable: " + ", ".join(
            f"def {signature}:" for signature in signatures
        )
    expected_output = re.search(r"output\s+(?:is\s+)?exactly\s+([^\s.,;]+)", prompt, flags=re.IGNORECASE)
    if expected_output:
        contract += (
            f" The file MUST include an executable entry point that prints {expected_output.group(1)} "
            "as its own output line when the requested command runs."
        )
    focused_prompt = (
        prompt
        + f"\n\nGenerate the complete runnable contents of {path}.{contract} Preserve every requested function name and signature exactly. "
        "Output source code only; no explanation."
    )
    chat_text = handle.tokenizer.apply_chat_template(
        [{"role": "user", "content": focused_prompt}], tokenize=False, add_generation_prompt=True
    )
    sampler = make_sampler(temp=temperature, top_p=0.9) if temperature > 0 else make_sampler(temp=0)
    reasoning = ""
    if "deepseek-r1" in handle.huggingface_id.lower():
        reasoning = generate(
            handle.model,
            handle.tokenizer,
            prompt=chat_text,
            max_tokens=min(192, max(96, max_tokens // 2)),
            sampler=sampler,
            verbose=False,
        ).strip()
        if THINK_CLOSE in reasoning:
            reasoning = reasoning.split(THINK_CLOSE, 1)[0].strip()
        generation_prompt = chat_text + reasoning + THINK_CLOSE + f"\n```{language}\n"
    else:
        generation_prompt = chat_text + f"```{language}\n"
    started = time.time()
    source = generate(
        handle.model,
        handle.tokenizer,
        prompt=generation_prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    ).strip()
    if "```" in source:
        source = source.split("```", 1)[0].rstrip()
    elapsed = time.time() - started
    try:
        tokens = len(handle.tokenizer.encode(reasoning + source))
    except Exception:
        tokens = max(1, len(source) // 4)
    mx.clear_cache()
    return InferenceResult(
        text=source,
        reasoning=reasoning,
        answer=source,
        elapsed=elapsed,
        tokens_generated=tokens,
        tokens_per_sec=tokens / max(elapsed, 1e-6),
    )


def _local_model_path(hf_id: str, precision: str, cache_dir: str) -> Path:
    """Get the local path where a converted model should be stored."""
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


def release_memory() -> None:
    """Release unreferenced MLX allocations before switching workloads."""
    import gc

    import mlx.core as mx

    gc.collect()
    mx.clear_cache()


def load_model(
    huggingface_id: str,
    precision: str = "int4",
    adapter_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    dequantize: bool = False,
    source_override: Optional[str] = None,
) -> ModelHandle:
    """Load an MLX model from HuggingFace.

    Strategy (fastest first):
    1. If a pre-quantized mlx-community version exists, load it directly — no
       conversion needed, download is ~3x smaller.
    2. Otherwise, download the full HF model and convert to MLX quantized
       format locally (cached so subsequent loads are fast).

    If adapter_path is given, loads LoRA adapters on top of the base model.

    dequantize is kept for backward compatibility but is no longer needed —
    MLX's QuantizedLinear supports gradients natively (QLoRA), so LoRA
    training works directly on int4 models without dequantization.
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
        # Check in-memory cache first
        cache_key = f"{huggingface_id}_{precision}"
        if cache_key in _mlx_repo_cache:
            load_source = _mlx_repo_cache[cache_key]
        elif os.environ.get("HF_HUB_OFFLINE") == "1":
            # Offline mode: try loading directly from cache — if the mlx-community
            # model is cached, mlx_lm.load will find it; if not, it'll error
            # and we fall through to the local convert path
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
        print(f"Converted in {time.time()-t0:.1f}s -> {local_path}")
        load_source = str(local_path)
    elif load_source is None and local_path.exists():
        load_source = str(local_path)

    # Load the model
    print(f"Loading model from {load_source}...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(load_source)
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # Dequantize only if explicitly requested (legacy path — QLoRA makes this unnecessary)
    if dequantize and quantized:
        _dequantize_model(model)
        quantized = False
        print("Model dequantized to fp16 in memory (legacy path — QLoRA is preferred)")

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
        precision="fp16" if dequantize else precision,
        quantized=quantized,
        adapter_path=adapter_path,
        cache_dir=cache_dir,
    )


def swap_adapters(model: object, adapter_path: str | None) -> object:
    """Swap LoRA adapters without reloading the base model.

    Removes existing LoRA layers and loads new ones in-place. This avoids
    the expensive model reload cycle (~15s per reload) when switching between
    baseline, SFT, and GRPO adapters.
    """
    import mlx.core as mx
    from mlx_lm.tuner.utils import load_adapters

    # Remove existing LoRA adapters if present
    has_lora = any("lora" in k.lower() for k, _ in model.named_modules())
    if has_lora:
        from mlx_lm.tuner.utils import remove_lora_layers
        model = remove_lora_layers(model)
        mx.eval(model.parameters())
        mx.clear_cache()

    # Load new adapters if provided
    if adapter_path and os.path.exists(adapter_path):
        load_adapters(model, adapter_path)
        mx.eval(model.parameters())
        print(f"Swapped to adapters: {adapter_path}")

    return model


def _dequantize_model(model):
    """Replace all QuantizedLinear/QuantizedEmbedding layers with regular ones.

    Legacy path — QLoRA (training directly on int4) is preferred and does not
    require this. Kept for backward compatibility.

    Memory-safe: materializes each dequantized weight with mx.eval() and clears
    the MLX cache after every layer replacement so the old int4 weights are
    freed before the next layer is processed.
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


def run_inference_speculative(
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
    """Run two-stage inference with prompt-lookup speculative decoding.

    Uses mlx-dspark's lookup_generate — a drafter-free speculative decoder
    that finds n-gram matches in the current context and proposes the tokens
    that followed them. The target model verifies every draft, so output is
    identical to plain greedy/sampled decoding (lossless).

    Best for: code reproduction, RAG, summarization, any task where the
    output copies heavily from the input context. For novel reasoning text
    (e.g. DeepSeek-R1 thinking traces), n-gram matches are rare and the
    wider verify passes cost more than they save — use run_inference instead.
    """
    from mlx_dspark import lookup_generate
    from mlx_dspark.target import Target

    messages = [{"role": "user", "content": prompt}]
    chat_text = handle.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    t0 = time.time()

    # Wrap our loaded model (may have LoRA adapters) in a dspark Target
    target = Target(handle.model, handle.tokenizer)

    # Single-pass generation with stop at think-close + EOS.
    # The chat template already includes  in the prompt, so the model's
    # output starts directly with reasoning text.
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

    # Split into reasoning and answer
    forced = False
    if THINK_CLOSE in raw_text:
        reasoning, answer = raw_text.split(THINK_CLOSE, 1)
        reasoning = reasoning.strip()
        answer = answer.strip()
    else:
        # No think-close — reasoning consumed the budget. Force an answer.
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


def run_inference(
    handle: ModelHandle,
    prompt: str,
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

    If speculative=True, uses prompt-lookup speculative decoding (mlx-dspark)
    for lossless speedup on copy-heavy text. See run_inference_speculative
    for details and trade-offs.
    """
    if speculative:
        config = speculative_config or {}
        return run_inference_speculative(
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
    chat_tokens = handle.tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )

    t0 = time.time()

    # Create a persistent KV cache shared between reasoning and answer stages
    prompt_cache = make_prompt_cache(handle.model)
    prompt_arr = mx.array(chat_tokens)

    # Determine stop tokens
    eos_token_id = handle.tokenizer.eos_token_id

    # Single generate_step call for reasoning + answer — the KV cache
    # flows naturally from reasoning to answer without reprocessing.
    # We track the think-close token to split the output.
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
            if token_id == THINK_CLOSE_TOKEN_ID or token_id == eos_token_id:
                think_done = True
                if len(all_tokens) >= max_reasoning_tokens:
                    forced = True
            # Reasoning budget exhausted without think-close — stop and force answer
            elif len(reasoning_tokens) >= max_reasoning_tokens:
                think_done = True
                forced = True
                break  # CRITICAL: stop generating, use forced prompt for answer
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
        # Reasoning budget exhausted without think-close — generate answer
        # from scratch with a forced think-close tag
        from mlx_lm import generate
        chat_text = handle.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        forced_prompt = chat_text + reasoning_text + THINK_CLOSE + "\n<answer>The answer is "
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
        full_text = reasoning_text + THINK_CLOSE + "\n<answer>The answer is " + out2 + "</answer>"
    else:
        answer_text = handle.tokenizer.decode(answer_tokens)
        if EOS in answer_text:
            answer_text = answer_text.replace(EOS, "").strip()
        full_text = reasoning_text + answer_text

    elapsed = time.time() - t0
    mx.clear_cache()

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
