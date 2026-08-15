"""Real model loader and inference engine (backend-agnostic dispatcher).

Loads models via the active backend (MLX on Apple Silicon, vLLM + HF Transformers
on NVIDIA CUDA) and runs the shared two-stage generation, tool-JSON, source, and
function-completion orchestration on top of backend primitives.

The backend-specific work (loading, generation, KV-cache management, training)
lives in :mod:`iloptimus.core.backends`. This module holds the orchestration
that is identical across backends:

- two-stage reasoning + answer generation with think-close splitting
- tool-call JSON envelope completion (grammar-constrained decoding equivalent)
- plain source / single-function / single-JSON completion with early-stop parsing
- adapter hot-swapping and memory release

DeepSeek-R1-Distill think tokens are shared constants re-exported from
:mod:`iloptimus.core.backends.base`.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

from .backends import (
    EOS,
    GenerateChunk,
    GenerateResult,
    InferenceResult,
    ModelHandle,
    THINK_CLOSE,
    THINK_CLOSE_TOKEN_ID,
    THINK_OPEN,
    get_backend,
    is_reasoning_model,
)

__all__ = [
    "EOS",
    "GenerateResult",
    "InferenceResult",
    "ModelHandle",
    "THINK_CLOSE",
    "THINK_CLOSE_TOKEN_ID",
    "THINK_OPEN",
    "clear_cache",
    "get_memory_info",
    "is_reasoning_model",
    "load_model",
    "release_memory",
    "run_completion",
    "run_function_completion",
    "run_inference",
    "run_inference_speculative",
    "run_json_completion",
    "run_source_completion",
    "run_tool_completion",
    "swap_adapters",
]


def _backend_for(handle: ModelHandle):
    return get_backend(handle.backend)


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
    backend = _backend_for(handle)
    messages = [{"role": "user", "content": prompt}]
    chat_text = handle.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    started = time.time()
    is_reasoning = is_reasoning_model(handle)
    reasoning = ""
    if is_reasoning:
        # Agent prompts typically place the native tool JSON just after a
        # substantial reasoning trace. Preserve enough of that first pass to
        # reach the model's own closing tag; short conversational completions
        # still split evenly so they retain answer room.
        reasoning_budget = max(32, int(max_tokens * (0.75 if max_tokens >= 256 else 0.5)))
        answer_budget = max(32, max_tokens - reasoning_budget)
        first = backend.generate(
            handle,
            chat_text,
            max_tokens=reasoning_budget,
            temperature=temperature,
            repetition_penalty=1.05,
            repetition_context_size=128,
        ).text.strip()
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
            text = backend.generate(
                handle,
                answer_prompt,
                max_tokens=answer_budget,
                temperature=temperature,
                repetition_penalty=1.05,
                repetition_context_size=128,
            ).text.strip()
    else:
        text = backend.generate(
            handle,
            chat_text,
            max_tokens=max_tokens,
            temperature=temperature,
            repetition_penalty=1.05,
            repetition_context_size=128,
        ).text.strip()
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
    backend.clear_cache(handle)
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
    backend = _backend_for(handle)
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
    reasoning = ""
    if is_reasoning_model(handle):
        reasoning = backend.generate(
            handle,
            chat_text,
            max_tokens=min(192, max(96, max_tokens // 2)),
            temperature=temperature,
            repetition_penalty=1.05,
            repetition_context_size=128,
        ).text.strip()
        if THINK_CLOSE in reasoning:
            reasoning = reasoning.split(THINK_CLOSE, 1)[0].strip()
        generation_prompt = chat_text + reasoning + THINK_CLOSE + "\n" + prefix
    else:
        generation_prompt = chat_text + prefix
    started = time.time()
    continuation = backend.generate(
        handle,
        generation_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        repetition_penalty=1.05,
        repetition_context_size=128,
    ).text.strip()
    text = prefix + continuation
    elapsed = time.time() - started
    try:
        tokens = len(handle.tokenizer.encode(reasoning + continuation))
    except Exception:
        tokens = max(1, len(continuation) // 4)
    backend.clear_cache(handle)
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
    backend = _backend_for(handle)
    language = {
        ".py": "python",
        ".html": "html",
        ".css": "css",
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
    reasoning = ""
    if is_reasoning_model(handle):
        reasoning = backend.generate(
            handle,
            chat_text,
            max_tokens=min(192, max(96, max_tokens // 2)),
            temperature=temperature,
            repetition_penalty=1.05,
            repetition_context_size=128,
        ).text.strip()
        if THINK_CLOSE in reasoning:
            reasoning = reasoning.split(THINK_CLOSE, 1)[0].strip()
        generation_prompt = chat_text + reasoning + THINK_CLOSE + f"\n```{language}\n"
    else:
        generation_prompt = chat_text + f"```{language}\n"
    started = time.time()
    source = backend.generate(
        handle,
        generation_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        repetition_penalty=1.05,
        repetition_context_size=128,
    ).text.strip()
    if "```" in source:
        source = source.split("```", 1)[0].rstrip()
    elapsed = time.time() - started
    try:
        tokens = len(handle.tokenizer.encode(reasoning + source))
    except Exception:
        tokens = max(1, len(source) // 4)
    backend.clear_cache(handle)
    return InferenceResult(
        text=source,
        reasoning=reasoning,
        answer=source,
        elapsed=elapsed,
        tokens_generated=tokens,
        tokens_per_sec=tokens / max(elapsed, 1e-6),
    )


def run_function_completion(
    handle: ModelHandle,
    prompt: str,
    function_name: str,
    max_tokens: int = 768,
    temperature: float = 0.0,
) -> InferenceResult:
    """Stream one JavaScript function and stop at its balanced closing brace.

    Component contracts make a separate reasoning rollout wasteful. DeepSeek's
    reasoning section is closed immediately, and deterministic generation ends
    as soon as the requested function is structurally complete.
    """
    backend = _backend_for(handle)
    from .artifact_composer import balanced_function_end

    focused_prompt = (
        prompt
        + f"\n\nOutput JavaScript only. Begin with: function {function_name}(world) {{"
    )
    chat_text = handle.tokenizer.apply_chat_template(
        [{"role": "user", "content": focused_prompt}], tokenize=False, add_generation_prompt=True
    )
    prefix = f"function {function_name}(world) {{"
    if is_reasoning_model(handle):
        generation_prompt = chat_text + THINK_CLOSE + "\n```javascript\n" + prefix
    else:
        generation_prompt = chat_text + "```javascript\n" + prefix
    started = time.time()
    source = prefix
    generated_tokens = 0
    for chunk in backend.stream_generate(
        handle,
        generation_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        repetition_penalty=1.08,
        repetition_context_size=192,
    ):
        source += chunk.text
        generated_tokens = max(generated_tokens, chunk.generation_tokens)
        bounds = balanced_function_end(source, function_name)
        if bounds:
            source = source[bounds[0] : bounds[1]]
            break
    elapsed = time.time() - started
    backend.clear_cache(handle)
    return InferenceResult(
        text=source.strip(),
        reasoning="",
        answer=source.strip(),
        elapsed=elapsed,
        tokens_generated=generated_tokens,
        tokens_per_sec=generated_tokens / max(elapsed, 1e-6),
    )


def run_json_completion(
    handle: ModelHandle,
    prompt: str,
    max_tokens: int = 768,
    temperature: float = 0.0,
) -> InferenceResult:
    """Generate one JSON object without spending tokens on a reasoning pass."""
    backend = _backend_for(handle)
    chat_text = handle.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt + "\nReturn one JSON object only."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    # Reasoning-distilled models (DeepSeek-R1 and its derivatives like BoostedV1)
    # generate  IMD... blocks before the answer. Prepend the think-close
    # token so the model skips reasoning and outputs JSON directly.
    generation_prompt = chat_text + (THINK_CLOSE + "\n" if is_reasoning_model(handle) else "") + "{"
    started = time.time()
    text = "{"
    generated_tokens = 0
    depth = 1
    quote = False
    escaped = False
    first_token = True
    for chunk in backend.stream_generate(
        handle,
        generation_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        repetition_penalty=1.05,
        repetition_context_size=192,
    ):
        piece = chunk.text
        # Small models often echo the opening brace that was already
        # prepended to the prompt, producing "{{". Skip a leading "{".
        if first_token:
            first_token = False
            piece = piece.lstrip()
            if piece.startswith("{"):
                piece = piece[1:]
        text += piece
        generated_tokens = max(generated_tokens, chunk.generation_tokens)
        for char in piece:
            if escaped:
                escaped = False
            elif char == "\\" and quote:
                escaped = True
            elif char == '"':
                quote = not quote
            elif not quote and char == "{":
                depth += 1
            elif not quote and char == "}":
                depth -= 1
        if depth <= 0:
            text = text[: text.rfind("}") + 1]
            break
    elapsed = time.time() - started
    backend.clear_cache(handle)
    return InferenceResult(
        text=text.strip(),
        reasoning="",
        answer=text.strip(),
        elapsed=elapsed,
        tokens_generated=generated_tokens,
        tokens_per_sec=generated_tokens / max(elapsed, 1e-6),
    )


def load_model(
    huggingface_id: str,
    precision: str = "int4",
    adapter_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    dequantize: bool = False,
    source_override: Optional[str] = None,
    backend: Optional[str] = None,
) -> ModelHandle:
    """Load a model via the active backend.

    ``backend`` selects the implementation (``"mlx"`` or ``"vllm"``); when
    omitted it is resolved from the detected hardware. The MLX backend supports
    a ``dequantize`` legacy path; the vLLM backend ignores it (QLoRA on CUDA
    uses bitsandbytes NF4 directly).
    """
    from .backends import resolve_backend

    name = resolve_backend(preferred=backend) if backend else resolve_backend()
    impl = get_backend(name)
    handle = impl.load(
        huggingface_id=huggingface_id,
        precision=precision,
        adapter_path=adapter_path,
        cache_dir=cache_dir,
        source_override=source_override,
    )
    # Legacy dequantize hook (MLX-only). QLoRA makes this unnecessary on both
    # backends, but it is kept for backward compatibility.
    if dequantize and name == "mlx" and handle.quantized:
        impl._dequantize_model(handle.model)  # type: ignore[attr-defined]
        handle.quantized = False
        handle.precision = "fp16"
        print("Model dequantized to fp16 in memory (legacy path — QLoRA is preferred)")
    return handle


def swap_adapters(model: object, adapter_path: str | None) -> object:
    """Swap LoRA adapters without reloading the base model.

    Accepts either a :class:`ModelHandle` (preferred) or a raw MLX model
    (legacy callers may pass ``handle.model``). When given a raw model, this
    falls back to the MLX backend's adapter swap.
    """
    if isinstance(model, ModelHandle):
        return _backend_for(model).swap_adapters(model, adapter_path).model
    # Legacy raw-model path: assume MLX.
    handle = ModelHandle(
        model=model,
        tokenizer=None,  # type: ignore[arg-type]
        model_id="",
        huggingface_id="",
        precision="int4",
        quantized=True,
        backend="mlx",
    )
    return _backend_for(handle).swap_adapters(handle, adapter_path).model


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

    Only the MLX backend implements speculative decoding; the vLLM backend
    ignores the speculative parameters and runs normal two-stage inference
    (vLLM's own speculative decoding is configured at engine init time).
    """
    backend = _backend_for(handle)
    if handle.backend == "mlx":
        return backend.run_two_stage_inference(  # type: ignore[attr-defined]
            handle,
            prompt,
            max_reasoning_tokens=max_reasoning_tokens,
            max_answer_tokens=max_answer_tokens,
            temperature=temperature,
            top_p=top_p,
            speculative=True,
            speculative_config={
                "max_draft_tokens": max_draft_tokens,
                "long_draft_tokens": long_draft_tokens,
                "ngram_min": ngram_min,
                "ngram_max": ngram_max,
            },
        )
    return backend.run_two_stage_inference(
        handle,
        prompt,
        max_reasoning_tokens=max_reasoning_tokens,
        max_answer_tokens=max_answer_tokens,
        temperature=temperature,
        top_p=top_p,
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

    Delegates to the active backend's ``run_two_stage_inference``. The MLX
    backend shares a persistent KV cache between stages; the vLLM backend
    generates in a single pass and splits on the think-close token. If
    ``speculative=True`` and the MLX backend is active, prompt-lookup
    speculative decoding (mlx-dspark) is used for lossless speedup on
    copy-heavy text.
    """
    backend = _backend_for(handle)
    return backend.run_two_stage_inference(
        handle,
        prompt,
        max_reasoning_tokens=max_reasoning_tokens,
        max_answer_tokens=max_answer_tokens,
        temperature=temperature,
        top_p=top_p,
        speculative=speculative,
        speculative_config=speculative_config,
    )


def get_memory_info() -> dict:
    """Get current backend memory usage info (best-effort across backends)."""
    # No handle available — report MLX info if present (legacy callers).
    try:
        return get_backend("mlx").get_memory_info()
    except Exception:
        return {}


def clear_cache() -> None:
    """Clear backend caches to free memory (best-effort across backends)."""
    try:
        get_backend("mlx").clear_cache(None)
    except Exception:
        pass


def release_memory() -> None:
    """Release unreferenced allocations before switching workloads."""
    try:
        get_backend("mlx").release_memory()
    except Exception:
        pass
