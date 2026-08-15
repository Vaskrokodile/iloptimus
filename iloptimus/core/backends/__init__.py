"""Backend factory and auto-detection.

Selects the MLX backend on Apple Silicon and the vLLM backend on NVIDIA CUDA.
The public inference/sft/grpo modules call :func:`get_backend` to obtain the
implementation matching the active hardware or an explicit ``backend`` string.
"""

from __future__ import annotations

from typing import Optional

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

__all__ = [
    "Backend",
    "EOS",
    "GenerateChunk",
    "GenerateResult",
    "GRPOConfig",
    "GRPOMetrics",
    "GRPOTrainerLike",
    "InferenceResult",
    "ModelHandle",
    "SFTConfig",
    "SFTExample",
    "SFTMetrics",
    "THINK_CLOSE",
    "THINK_CLOSE_TOKEN_ID",
    "THINK_OPEN",
    "get_backend",
    "is_reasoning_model",
    "resolve_backend",
]

_mlx_backend: Optional["Backend"] = None
_vllm_backend: Optional["Backend"] = None


def get_backend(name: str) -> Backend:
    """Return the backend instance for ``name`` (``"mlx"`` or ``"vllm"``).

    Backend instances are cached (they hold no per-model state — all state lives
    on the :class:`ModelHandle`). Unknown names raise ``ValueError``.
    """
    global _mlx_backend, _vllm_backend
    if name == "mlx":
        if _mlx_backend is None:
            from .mlx_backend import MLXBackend

            _mlx_backend = MLXBackend()
        return _mlx_backend
    if name in ("vllm", "cuda", "torch"):
        if _vllm_backend is None:
            from .vllm_backend import VLLMBackend

            _vllm_backend = VLLMBackend()
        return _vllm_backend
    raise ValueError(
        f"Unknown backend: {name!r}. Supported backends: 'mlx' (Apple Silicon), 'vllm' (NVIDIA CUDA)."
    )


def resolve_backend(
    *,
    hardware=None,
    preferred: Optional[str] = None,
) -> str:
    """Pick the backend name to use.

    Priority:
    1. An explicit ``preferred`` that is actually available on this machine.
    2. The hardware's ``recommended_backend`` (already computed by hardware
       detection from the installed MLX/vLLM/torch packages and GPU type).
    3. ``"mlx"`` on Apple Silicon, ``"vllm"`` on CUDA, else ``"cpu"`` (no
       training, inference only).
    """
    if preferred:
        if preferred == "mlx" and (hardware is None or hardware.mlx_available):
            return "mlx"
        if preferred in ("vllm", "cuda", "torch"):
            return "vllm"
        # Fall through if the preferred backend isn't available.

    if hardware is not None:
        if hardware.recommended_backend in ("mlx", "vllm"):
            return hardware.recommended_backend
        if hardware.gpu.type == "apple-silicon" and hardware.mlx_available:
            return "mlx"
        if hardware.gpu.type == "cuda":
            return "vllm"

    # Last-resort defaults from platform.
    import sys

    if sys.platform == "darwin":
        return "mlx"
    return "vllm"
