"""Backend abstraction for Optimus Studio.

Optimus Studio runs the same Intuition Learning pipeline (SFT + GRPO) on two
local accelerator stacks:

- **MLX** (Apple Silicon) — ``mlx_lm`` for inference and compiled LoRA/QLoRA
  fine-tuning. This is the original, heavily tuned path for M-series Macs.
- **vLLM + HF Transformers + PEFT** (NVIDIA CUDA) — ``vllm`` for high-throughput
  batched inference and HuggingFace Transformers + PEFT for LoRA/QLoRA SFT and a
  custom GRPO loop. This is the CUDA path.

This module defines the shared, backend-agnostic types and the :class:`Backend`
interface that both implementations satisfy. The public modules
(:mod:`iloptimus.core.inference`, :mod:`iloptimus.core.sft`,
:mod:`iloptimus.core.grpo`) keep their existing signatures and delegate the
backend-specific work through a :class:`Backend` instance stored on the
:class:`ModelHandle`.

Keeping the shared orchestration (think-token splitting, JSON brace tracking,
source extraction, advantage computation) here and in the public modules means
the two backends only implement the small set of primitives they actually differ
on (loading, generation, logprob computation, training).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional


# ---------------------------------------------------------------------------
# Shared token constants — DeepSeek-R1-Distill think tags (token 151648/151649)
# Using chr() so tooling does not interpret the tags as HTML.
# ---------------------------------------------------------------------------

THINK_OPEN = chr(60) + "think" + chr(62)  # <think>
THINK_CLOSE = chr(60) + "/think" + chr(62)  # </think>
EOS = chr(60) + "\uff5cend\u2581of\u2581sentence\uff5c" + chr(62)  # <｜end▁of▁sentence｜>

# Token id for DeepSeek-R1-Distill (Qwen2 tokenizer). Used by the MLX backend's
# generate_step stop logic and by the vLLM backend's stop_token_ids.
THINK_CLOSE_TOKEN_ID = 151649


# ---------------------------------------------------------------------------
# Shared dataclasses
# ---------------------------------------------------------------------------


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
class GenerateResult:
    """Result of a single non-streaming generate call."""

    text: str
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    finish_reason: str = "stop"


@dataclass
class GenerateChunk:
    """One emitted chunk from a streaming generate call."""

    text: str
    token_id: int = -1
    generation_tokens: int = 0


@dataclass
class ModelHandle:
    """A loaded model + tokenizer, ready for inference and training.

    ``model`` and ``tokenizer`` are backend-native objects (an MLX module pair
    or a HF ``AutoModelForCausalLM`` + tokenizer). ``backend`` is the backend
    name (``"mlx"`` or ``"vllm"``) and ``backend_obj`` holds any extra
    backend-specific state (e.g. a vLLM ``LLM`` engine).
    """

    model: object
    tokenizer: object
    model_id: str
    huggingface_id: str
    precision: str
    quantized: bool
    adapter_path: Optional[str] = None
    cache_dir: str = ""
    backend: str = "mlx"
    backend_obj: Any = None  # backend-specific engine/state (e.g. vllm.LLM)


def is_reasoning_model(handle: ModelHandle) -> bool:
    """Check if a model is a reasoning-distilled model that uses <think> tags."""
    return any(
        marker in handle.huggingface_id.lower()
        for marker in ("deepseek-r1", "boosted", "r1-distill", "r1-distilled")
    )


# ---------------------------------------------------------------------------
# Training config / metrics — shared across backends so the pipeline and
# self-improvement loop do not need to know which backend is active.
# ---------------------------------------------------------------------------


@dataclass
class SFTExample:
    prompt: str
    response: str  # the "ideal" response (from correct benchmark outputs or generated)


@dataclass
class SFTMetrics:
    iteration: int
    loss: float
    learning_rate: float
    elapsed: float
    peak_memory_gb: float
    iterations_per_second: float = 0.0
    tokens_per_second: float = 0.0
    trained_tokens: int = 0


@dataclass
class SFTConfig:
    learning_rate: float = 1e-4
    num_iters: int = 100
    batch_size: int = 1
    grad_accumulation_steps: int = 1
    lora_rank: int = 8
    # mlx-lm's LoRALinear multiplies the adapter branch directly by this
    # value; its supported default is 20.0 (this is not alpha/rank).
    lora_scale: float = 20.0
    lora_dropout: float = 0.0
    lora_layers: int = 8  # final transformer blocks; safe on 8GB unified memory
    lora_targets: tuple[str, ...] = (
        "self_attn.q_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    )
    max_seq_length: int = 512
    memory_limit_gb: float = 3.0  # QLoRA on int4 uses less memory (was 3.5 for fp16)
    steps_per_eval: int = 20
    grad_clip: float = 1.0
    mask_prompt: bool = True
    grad_checkpoint: bool = False
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    # mlx-lm treats zero as "clear every step". Fixed-shape training reuses
    # buffers safely, so retain a bounded allocator cache for throughput.
    clear_cache_threshold_gb: float = 1.0
    compile_bucket_size: int = 128
    preserve_native_bucket_shape: bool = True
    # Cache the frozen transformer prefix once when adapters touch only final
    # layers. This trades a small amount of RAM for eliminating repeated base
    # forward compute across epochs. (MLX only; ignored by the vLLM backend.)
    prefix_cache: bool = False
    prefix_cache_batch_size: int = 8
    seed: int = 0


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
    learning_rate: float = 1e-4  # SGD needs higher LR than Adam (was 1e-5 for Adam)
    clip_eps: float = 0.2
    group_size: int = 4
    thinking_tokens: int = 1024  # enough for DeepSeek-R1-Distill to finish reasoning
    prediction_tokens: int = 256  # enough for a detailed answer
    temperature: float = 0.6
    top_p: float = 0.9
    kl_beta: float = 0.04
    memory_limit_gb: float = 3.0  # QLoRA on int4 uses less memory (was 3.5 for fp16)
    grad_clip: float = 1.0


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------


class Backend(ABC):
    """Interface implemented by the MLX and vLLM backends.

    The public inference/sft/grpo modules call these primitives. Shared
    orchestration (think-token splitting, JSON parsing, advantage computation)
    lives in those modules and in :mod:`backends.base`, so each backend only
    implements what is genuinely backend-specific.
    """

    name: str

    # --- loading -----------------------------------------------------------

    @abstractmethod
    def load(
        self,
        *,
        huggingface_id: str,
        precision: str = "int4",
        adapter_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        source_override: Optional[str] = None,
    ) -> ModelHandle:
        """Load a model and return a :class:`ModelHandle` with ``backend`` set."""

    # --- inference primitives ---------------------------------------------

    @abstractmethod
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
        """Single non-streaming generation from an already-rendered prompt text."""

    @abstractmethod
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
        """Token-by-token generation for early-stop parsing (JSON braces, etc.)."""

    @abstractmethod
    def run_two_stage_inference(
        self,
        handle: ModelHandle,
        prompt: str,
        *,
        max_reasoning_tokens: int = 512,
        max_answer_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> InferenceResult:
        """Two-stage reasoning-then-answer inference.

        The MLX backend shares a persistent KV cache between stages; the vLLM
        backend generates in a single pass and splits on the think-close token,
        falling back to a forced second pass when reasoning exhausts its budget.
        """

    def run_batch_inference(
        self,
        handle: ModelHandle,
        prompts: list[str],
        *,
        max_reasoning_tokens: int = 512,
        max_answer_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.9,
        speculative: bool = False,
        speculative_config: dict | None = None,
    ) -> list[InferenceResult]:
        """Run independent prompts while preserving input order.

        Backends with native batching can override this method. The default is
        intentionally sequential so every backend remains correct before it
        grows a batch-specific implementation.
        """
        results: list[InferenceResult] = []
        for prompt in prompts:
            results.append(
                self.run_two_stage_inference(
                    handle,
                    prompt,
                    max_reasoning_tokens=max_reasoning_tokens,
                    max_answer_tokens=max_answer_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    speculative=speculative,
                    speculative_config=speculative_config,
                )
            )
            self.clear_cache(handle)
        return results

    # --- memory / adapters -------------------------------------------------

    @abstractmethod
    def clear_cache(self, handle: Optional[ModelHandle] = None) -> None:
        """Free unreferenced allocator buffers."""

    @abstractmethod
    def release_memory(self) -> None:
        """Release unreferenced allocations before switching workloads."""

    @abstractmethod
    def get_memory_info(self) -> dict:
        """Return current/peak memory usage info (backend-specific keys)."""

    @abstractmethod
    def swap_adapters(self, handle: ModelHandle, adapter_path: Optional[str]) -> ModelHandle:
        """Swap LoRA adapters without reloading the base model."""

    # --- training ----------------------------------------------------------

    @abstractmethod
    def run_sft(
        self,
        handle: ModelHandle,
        examples: list[SFTExample],
        config: SFTConfig,
        adapter_path: str,
        on_metrics: Any = None,
    ) -> str:
        """Run LoRA/QLoRA SFT. Returns the path to the saved adapter."""

    @abstractmethod
    def make_grpo_trainer(
        self,
        handle: ModelHandle,
        config: GRPOConfig,
        adapter_path: str,
    ) -> "GRPOTrainerLike":
        """Construct a backend-specific GRPO trainer."""


class GRPOTrainerLike(ABC):
    """Minimal interface a backend GRPO trainer must satisfy.

    The public :class:`iloptimus.core.grpo.GRPOTrainer` delegates to an instance
    of this so the pipeline code does not change.
    """

    @abstractmethod
    def train_step(
        self,
        prompt: str,
        grade_fn: Any,
        on_metrics: Any = None,
    ) -> GRPOMetrics: ...

    @abstractmethod
    def save(self, path: Optional[str] = None) -> None: ...
