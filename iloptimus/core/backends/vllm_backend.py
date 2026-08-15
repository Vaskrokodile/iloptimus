"""vLLM + HF Transformers + PEFT backend — NVIDIA CUDA inference + training.

This backend runs the same IL pipeline (SFT + GRPO) on NVIDIA GPUs:

- **Inference / chat / benchmarks**: ``vllm`` for high-throughput batched
  generation. When vLLM is not installed, generation falls back to HuggingFace
  ``model.generate`` so the backend still works on a torch-only box.
- **SFT**: HuggingFace Transformers + PEFT LoRA/QLoRA (4-bit NF4 via
  bitsandbytes when ``precision="int4"``) with a manual training loop that
  mirrors the MLX backend's prompt-masking and streaming-metrics behavior.
- **GRPO**: a custom group-relative policy optimization loop using the HF model
  for rollouts (``model.generate`` with ``output_scores=True`` for logprobs) and
  PEFT autograd for clipped policy-gradient + KL updates. This preserves the
  MLX GRPO semantics (action-position logprobs, group-relative advantages, KL
  penalty) so behavior matches across backends.

For the pipeline, vLLM serves benchmarks/chat and loads trained LoRA adapters
on the fly via ``LoRARequest``; SFT/GRPO train a separate HF+PEFT model and
save adapters that vLLM then picks up. GRPO rollouts use the HF model directly
so per-step weight updates stay in sync without reloading the vLLM engine.

All torch/transformers/peft/vllm imports are lazy (inside methods) so this
module imports cleanly on machines without those packages — only actually
*using* the backend requires them.
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
    is_reasoning_model,
)


def _vllm_available() -> bool:
    try:
        import vllm  # noqa: F401

        return True
    except ImportError:
        return False


def _torch_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


class VLLMBackend(Backend):
    name = "vllm"

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
        """Load a model for the CUDA backend.

        Loads a HuggingFace ``AutoModelForCausalLM`` (+ tokenizer) as the
        canonical model used for training and (when vLLM is absent) inference.
        When vLLM is available, a separate ``vllm.LLM`` engine is created from
        the same source for fast batched inference; trained LoRA adapters are
        served on the fly via ``LoRARequest``.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        source = source_override or huggingface_id
        cache_dir = cache_dir or os.path.expanduser("~/.cache/iloptimus/models")
        os.makedirs(cache_dir, exist_ok=True)

        quantized = precision in ("int4", "int8")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

        quantization_config = None
        if precision == "int4":
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                )
            except Exception:
                # bitsandbytes not installed — fall back to fp16 load
                quantized = False
        elif precision == "int8":
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            except Exception:
                quantized = False

        print(f"Loading HF model from {source} (precision={precision})...")
        t0 = time.time()
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": "auto" if torch.cuda.is_available() else None,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        model = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(source)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"HF model loaded in {time.time() - t0:.1f}s")

        # Apply a pre-trained LoRA adapter if provided (cumulative training).
        peft_model = None
        if adapter_path and os.path.exists(adapter_path):
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
            peft_model = model
            print(f"Loaded LoRA adapter from {adapter_path}")

        # Optional vLLM engine for fast inference. Created from the base source;
        # trained adapters are attached on the fly via LoRARequest.
        vllm_llm = None
        if _vllm_available() and _torch_cuda_available():
            try:
                from vllm import LLM

                vllm_kwargs: dict[str, Any] = {
                    "model": source,
                    "dtype": "bfloat16" if dtype == torch.bfloat16 else "float16",
                    "gpu_memory_utilization": 0.90,
                    "enable_lora": True,
                    "max_loras": 4,
                    "max_lora_rank": 64,
                    "trust_remote_code": True,
                }
                # Honour a quantized vLLM load when the source is an AWQ/GPTQ
                # checkpoint. For bitsandbytes NF4 adapters we keep vLLM on the
                # base fp16 source and attach LoRA via LoRARequest.
                vllm_llm = LLM(**vllm_kwargs)
                print("vLLM inference engine ready")
            except Exception as error:  # pragma: no cover - environment-specific
                print(f"vLLM engine unavailable, falling back to HF generate: {error}")
                vllm_llm = None

        backend_obj = {
            "vllm_llm": vllm_llm,
            "hf_model": model,
            "hf_tokenizer": tokenizer,
            "peft_model": peft_model,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "active_adapter_path": adapter_path,
        }

        return ModelHandle(
            model=model,
            tokenizer=tokenizer,
            model_id=huggingface_id.split("/")[-1],
            huggingface_id=huggingface_id,
            precision=precision,
            quantized=quantized,
            adapter_path=adapter_path,
            cache_dir=cache_dir,
            backend="vllm",
            backend_obj=backend_obj,
        )

    # ----------------------------------------------------------- vLLM helpers

    def _vllm_generate(
        self,
        handle: ModelHandle,
        prompt_text: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        stop_strings: Optional[list[str]] = None,
        stop_token_ids: Optional[list[int]] = None,
        return_logprobs: bool = False,
        adapter_path: Optional[str] = None,
    ) -> GenerateResult:
        from vllm import SamplingParams

        llm = handle.backend_obj["vllm_llm"]
        params_kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature if temperature > 0 else 0.0,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }
        if temperature <= 0:
            params_kwargs["temperature"] = 0.0
        if stop_strings:
            params_kwargs["stop"] = stop_strings
        if stop_token_ids:
            params_kwargs["stop_token_ids"] = stop_token_ids
        if return_logprobs:
            params_kwargs["logprobs"] = 0  # logprob of the sampled token

        lora_request = None
        if adapter_path and os.path.exists(adapter_path):
            from vllm import LoRARequest

            lora_request = LoRARequest(
                lora_name=os.path.basename(adapter_path),
                lora_int_id=abs(hash(adapter_path)) % (2**31) + 1,
                lora_path=adapter_path,
            )

        params = SamplingParams(**params_kwargs)
        outputs = llm.generate([prompt_text], params, lora_request=lora_request)
        out = outputs[0].outputs[0]
        token_ids = list(out.token_ids)
        logprobs: list[float] = []
        if return_logprobs and out.logprobs:
            for i, tid in enumerate(token_ids):
                lp_map = out.logprobs[i] or {}
                lp = lp_map.get(tid)
                logprobs.append(float(lp.logprob) if lp is not None else 0.0)
        return GenerateResult(
            text=out.text,
            token_ids=token_ids,
            logprobs=logprobs,
            finish_reason=out.finish_reason or "stop",
        )

    def _hf_generate(
        self,
        handle: ModelHandle,
        prompt_text: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        stop_strings: Optional[list[str]] = None,
        stop_token_ids: Optional[list[int]] = None,
        return_logprobs: bool = False,
    ) -> GenerateResult:
        import torch

        model = handle.backend_obj["hf_model"]
        tokenizer = handle.backend_obj["hf_tokenizer"]
        device = handle.backend_obj["device"]

        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "use_cache": True,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        if repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = repetition_penalty
        if stop_token_ids:
            gen_kwargs["eos_token_id"] = stop_token_ids

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs, return_dict_in_generate=True, output_scores=return_logprobs)
        new_tokens = out.sequences[0][inputs["input_ids"].shape[1]:]
        token_ids = new_tokens.tolist()
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        logprobs: list[float] = []
        if return_logprobs and getattr(out, "scores", None):
            import torch.nn.functional as F

            for i, scores in enumerate(out.scores):
                if i >= len(token_ids):
                    break
                logp = F.log_softmax(scores[0], dim=-1)
                logprobs.append(float(logp[token_ids[i]].item()))
        # Stop-string trimming
        if stop_strings:
            for s in stop_strings:
                if s in text:
                    text = text.split(s)[0]
        return GenerateResult(text=text, token_ids=token_ids, logprobs=logprobs)

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
        adapter = handle.backend_obj.get("active_adapter_path") if handle.backend_obj else None
        if handle.backend_obj and handle.backend_obj.get("vllm_llm") is not None:
            return self._vllm_generate(
                handle,
                prompt_text,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop_strings=stop_strings,
                stop_token_ids=stop_token_ids,
                return_logprobs=return_logprobs,
                adapter_path=adapter,
            )
        return self._hf_generate(
            handle,
            prompt_text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_strings=stop_strings,
            stop_token_ids=stop_token_ids,
            return_logprobs=return_logprobs,
        )

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
        """Token-by-token generation for early-stop parsing.

        vLLM offline generation is not token-streamed, so we generate the full
        sequence then yield per-token decode chunks. The consumer's early-break
        stops processing; the full ``max_tokens`` is only spent when no early
        stop triggers (e.g. malformed JSON), which matches correctness.
        """
        result = self.generate(
            handle,
            prompt_text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            return_logprobs=False,
        )
        tokenizer = handle.backend_obj["hf_tokenizer"] if handle.backend_obj else handle.tokenizer
        for i, tid in enumerate(result.token_ids):
            yield GenerateChunk(
                text=tokenizer.decode([tid], skip_special_tokens=False),
                token_id=tid,
                generation_tokens=i + 1,
            )

    # --------------------------------------------------- two-stage inference

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
        """Two-stage reasoning-then-answer inference.

        Single-pass generation up to ``reasoning+answer`` tokens, stopping on
        the think-close token / EOS, then split into reasoning + answer. If
        reasoning exhausts its budget without emitting think-close, a forced
        second pass produces the answer (mirroring the MLX fallback).
        ``speculative`` is accepted for API parity and ignored — vLLM's own
        speculative decoding is configured at engine init, not per-call.
        """
        messages = [{"role": "user", "content": prompt}]
        chat_text = handle.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        eos_token_id = handle.tokenizer.eos_token_id

        t0 = time.time()
        total_max = max_reasoning_tokens + max_answer_tokens
        result = self.generate(
            handle,
            chat_text,
            max_tokens=total_max,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.05,
            stop_strings=[THINK_CLOSE, EOS],
            stop_token_ids=[THINK_CLOSE_TOKEN_ID, eos_token_id] if eos_token_id is not None else [THINK_CLOSE_TOKEN_ID],
        )
        raw = result.text
        if EOS in raw:
            raw = raw.replace(EOS, "")

        forced = False
        if THINK_CLOSE in raw:
            reasoning, answer = raw.split(THINK_CLOSE, 1)
            reasoning = reasoning.strip()
            answer = answer.strip()
        else:
            # Reasoning consumed the budget without think-close — force an answer.
            reasoning = raw.strip()
            forced = True
            forced_prompt = chat_text + raw + THINK_CLOSE + "\n<answer>The answer is "
            ans = self.generate(
                handle,
                forced_prompt,
                max_tokens=max_answer_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.05,
                stop_strings=[EOS],
                stop_token_ids=[eos_token_id] if eos_token_id is not None else None,
            )
            answer = ans.text.replace(EOS, "").strip()

        full_text = reasoning + THINK_CLOSE + "\n" + answer if reasoning else answer
        if forced:
            full_text = reasoning + THINK_CLOSE + "\n<answer>The answer is " + answer + "</answer>"
        elapsed = time.time() - t0
        tokens_generated = max(1, len(result.token_ids)) if result.token_ids else len(full_text) // 4
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
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def release_memory(self) -> None:
        import gc

        self.clear_cache(None)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def get_memory_info(self) -> dict:
        info: dict[str, float] = {}
        try:
            import torch

            if torch.cuda.is_available():
                info["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
                info["active_memory_gb"] = torch.cuda.memory_allocated() / 1e9
        except ImportError:
            pass
        return info

    def swap_adapters(self, handle: ModelHandle, adapter_path: Optional[str]) -> ModelHandle:
        """Swap the active LoRA adapter.

        For vLLM inference this just records the new adapter path (applied via
        ``LoRARequest`` on the next generate). For the HF training model, the
        PEFT adapter is reloaded.
        """
        if handle.backend_obj is None:
            return handle
        handle.backend_obj["active_adapter_path"] = adapter_path
        if adapter_path and os.path.exists(adapter_path):
            try:
                from peft import PeftModel

                base = handle.backend_obj["hf_model"]
                # If already a PeftModel, swap adapter; otherwise wrap.
                if isinstance(base, PeftModel):
                    base.set_adapter(adapter_path)
                else:
                    handle.backend_obj["hf_model"] = PeftModel.from_pretrained(base, adapter_path)
                    handle.backend_obj["peft_model"] = handle.backend_obj["hf_model"]
                handle.model = handle.backend_obj["hf_model"]
            except Exception as error:  # pragma: no cover
                print(f"swap_adapters: could not reload HF adapter: {error}")
        handle.adapter_path = adapter_path
        return handle

    # ------------------------------------------------------------------ SFT

    def run_sft(
        self,
        handle: ModelHandle,
        examples: list[SFTExample],
        config: SFTConfig,
        adapter_path: str,
        on_metrics: Callable[[SFTMetrics], None] | None = None,
    ) -> str:
        """LoRA/QLoRA SFT via HF Transformers + PEFT.

        Applies LoRA to the configured projection layers (or reuses an already
        loaded PEFT model), then runs a manual training loop with prompt
        masking, gradient accumulation, gradient clipping, and streaming
        metrics — mirroring the MLX backend's behavior.
        """
        import torch
        from torch.utils.data import DataLoader

        os.makedirs(adapter_path, exist_ok=True)
        model = handle.backend_obj["hf_model"]
        tokenizer = handle.backend_obj["hf_tokenizer"]
        device = handle.backend_obj["device"]

        # Prepare for k-bit training and apply LoRA if not already a PEFT model.
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if config.grad_checkpoint:
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass

        is_peft = hasattr(model, "peft_type") or model.__class__.__name__.startswith("Peft")
        if not is_peft:
            try:
                model = prepare_model_for_kbit_training(model)
            except Exception:
                pass
            target_modules = list(config.lora_targets)
            # PEFT matches by suffix; the MLX targets use the "self_attn.q_proj"
            # form which PEFT resolves against module names.
            lora_cfg = LoraConfig(
                r=config.lora_rank,
                lora_alpha=int(config.lora_scale),
                lora_dropout=config.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)
            handle.backend_obj["hf_model"] = model
            handle.backend_obj["peft_model"] = model
            handle.model = model

        model.train()
        trainable_params, all_params = model.get_nb_trainable_parameters()
        print(f"SFT trainable params: {trainable_params}/{all_params}")

        # Build a tokenized, prompt-masked dataset.
        dataset = _SFTDataset(
            examples,
            tokenizer,
            max_length=config.max_seq_length,
            mask_prompt=config.mask_prompt,
        )
        loader = DataLoader(
            dataset,
            batch_size=min(config.batch_size, len(dataset)),
            shuffle=True,
            collate_fn=_sft_collate,
        )

        optimizer_name = config.optimizer.casefold()
        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
            )

        model.to(device)
        started = time.perf_counter()
        step = 0
        report_every = max(1, min(config.steps_per_eval, config.num_iters))
        optimizer.zero_grad()

        while step < config.num_iters:
            for batch in loader:
                if step >= config.num_iters:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / max(1, config.grad_accumulation_steps)
                loss.backward()
                if (step + 1) % config.grad_accumulation_steps == 0 or (step + 1) >= config.num_iters:
                    if config.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()
                self.clear_cache(handle)

                if (step + 1) % report_every == 0 or (step + 1) >= config.num_iters:
                    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                    if on_metrics:
                        on_metrics(
                            SFTMetrics(
                                iteration=step,
                                loss=float(loss.item()) * config.grad_accumulation_steps,
                                learning_rate=config.learning_rate,
                                elapsed=time.perf_counter() - started,
                                peak_memory_gb=peak,
                                iterations_per_second=(step + 1) / max(1e-6, time.perf_counter() - started),
                                tokens_per_second=0.0,
                                trained_tokens=int((step + 1) * config.batch_size * config.max_seq_length),
                            )
                        )
                step += 1

        # Save the LoRA adapter.
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        # Record an adapter_config.json compatible with the MLX loader's schema
        # so downstream tooling can inspect either backend's adapters.
        cfg = {
            "adapter_path": os.path.basename(adapter_path),
            "fine_tune_type": "lora",
            "num_layers": config.lora_layers,
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
            "backend": "vllm",
            "trainable_parameters": int(trainable_params),
            "seed": config.seed,
        }
        with open(os.path.join(adapter_path, "adapter_config.json"), "w") as f:
            json.dump(cfg, f, indent=4)

        # Mark the freshly trained adapter as active so subsequent vLLM inference
        # (post-SFT benchmark) serves it via LoRARequest.
        handle.backend_obj["active_adapter_path"] = adapter_path
        handle.adapter_path = adapter_path
        return adapter_path

    # ----------------------------------------------------------------- GRPO

    def make_grpo_trainer(
        self,
        handle: ModelHandle,
        config: GRPOConfig,
        adapter_path: str,
    ) -> "VLLMGRPOTrainer":
        return VLLMGRPOTrainer(handle, config, adapter_path)


# ---------------------------------------------------------------------------
# SFT dataset + collator (prompt-masked completion loss)
# ---------------------------------------------------------------------------


class _SFTDataset:
    """Tokenized SFT dataset with prompt-token masking.

    ``labels`` copies ``input_ids`` but sets prompt positions to -100 so the
    HF causal-LM loss only counts completion tokens (matching the MLX
    ``mask_prompt`` behavior). ``torch.utils.data.DataLoader`` accepts any
    object with ``__len__``/``__getitem__``, so no ``Dataset`` subclass is
    needed — this keeps the module importable without torch installed.
    """

    def __init__(self, examples, tokenizer, *, max_length=512, mask_prompt=True):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prompt = mask_prompt

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt_ids = self.tokenizer(ex.prompt, add_special_tokens=False).input_ids
        completion_ids = self.tokenizer(ex.response, add_special_tokens=False).input_ids
        ids = prompt_ids + completion_ids
        ids = ids[: self.max_length]
        # Append EOS if room and not already present.
        if len(ids) < self.max_length and (not ids or ids[-1] != self.tokenizer.eos_token_id):
            ids = ids + [self.tokenizer.eos_token_id]
        labels = list(ids)
        if self.mask_prompt:
            prompt_len = min(len(prompt_ids), len(ids))
            for i in range(prompt_len):
                labels[i] = -100
        attention = [1] * len(ids)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        return {"input_ids": ids, "attention_mask": attention, "labels": labels, "pad_id": pad_id}


def _sft_collate(features):
    import torch

    pad_id = features[0]["pad_id"]
    max_len = max(len(f["input_ids"]) for f in features)
    input_ids, attn, labels = [], [], []
    for f in features:
        n = len(f["input_ids"])
        pad = max_len - n
        input_ids.append(f["input_ids"] + [pad_id] * pad)
        attn.append(f["attention_mask"] + [0] * pad)
        labels.append(f["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# GRPO trainer (HF + PEFT) — mirrors the MLX GRPO semantics
# ---------------------------------------------------------------------------


class VLLMGRPOTrainer(GRPOTrainerLike):
    """Custom GRPO trainer for the CUDA/HF backend.

    Rollouts use the HF model's ``generate`` with ``output_scores=True`` to
    record per-token logprobs. Updates use PEFT autograd with a clipped
    policy-gradient loss + KL penalty, matching the MLX GRPO trainer.
    """

    def __init__(self, handle: ModelHandle, config: GRPOConfig, adapter_path: str = "il_grpo_adapters"):
        import torch

        self.handle = handle
        self.config = config
        self.adapter_path = adapter_path
        self.model = handle.backend_obj["hf_model"]
        self.tokenizer = handle.backend_obj["hf_tokenizer"]
        self.device = handle.backend_obj["device"]
        self.iteration = 0

        # Ensure LoRA is applied for RL (reuse SFT's PEFT model if present).
        from peft import LoraConfig, get_peft_model

        is_peft = hasattr(self.model, "peft_type") or self.model.__class__.__name__.startswith("Peft")
        if not is_peft:
            for p in self.model.parameters():
                p.requires_grad = False
            lora_cfg = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                target_modules=["self_attn.q_proj", "self_attn.v_proj", "self_attn.o_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_cfg)
            handle.backend_obj["hf_model"] = self.model
            handle.model = self.model
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.learning_rate)
        print("GRPO (vllm backend): ready with HF + PEFT model")

    def _collect_rollout(self, prompt, seed=None):
        import torch

        messages = [{"role": "user", "content": prompt}]
        chat_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if seed is not None:
            torch.manual_seed(seed)
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(self.device)
        total = self.config.thinking_tokens + self.config.prediction_tokens
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=total,
                do_sample=self.config.temperature > 0,
                temperature=self.config.temperature if self.config.temperature > 0 else 1.0,
                top_p=self.config.top_p,
                return_dict_in_generate=True,
                output_scores=True,
            )
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out.sequences[0][prompt_len:].tolist()
        # Per-token logprobs from scores.
        import torch.nn.functional as F

        old_logprobs: list[float] = []
        for i, scores in enumerate(out.scores):
            if i >= len(gen_ids):
                break
            logp = F.log_softmax(scores[0], dim=-1)
            old_logprobs.append(float(logp[gen_ids[i]].item()))
        # If generate produced fewer scores than tokens (e.g. pad), pad logprobs.
        while len(old_logprobs) < len(gen_ids):
            old_logprobs.append(0.0)

        all_tokens = inputs["input_ids"][0].tolist() + gen_ids
        gen_start = prompt_len
        gen_end = len(all_tokens)
        action_positions = [(gen_start, gen_end)] if gen_end > gen_start else []
        response_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return {
            "tokens": all_tokens,
            "action_positions": action_positions,
            "old_logprobs": old_logprobs,
            "response_text": response_text,
            "gen_ids": gen_ids,
        }

    def _compute_action_logprobs(self, tokens, action_positions):
        """Forward the full sequence and gather logprobs at action positions."""
        import torch
        import torch.nn.functional as F

        device = self.device
        input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
        logits = self.model(input_ids).logits  # [1, seq-1, vocab]
        logprobs = F.log_softmax(logits, dim=-1)[0]  # [seq-1, vocab]
        segments = []
        for (start, end) in action_positions:
            lp_start = start - 1
            lp_end = end - 1
            seg_tokens = torch.tensor(tokens[start:end], dtype=torch.long, device=device)
            seg_lp = logprobs[lp_start:lp_end].gather(-1, seg_tokens[:, None]).squeeze(-1)
            segments.append(seg_lp)
        return segments

    def train_step(
        self,
        prompt: str,
        grade_fn: Callable[[str], float],
        on_metrics: Callable[[GRPOMetrics], None] | None = None,
    ) -> GRPOMetrics:
        import numpy as np
        import torch
        import torch.nn.functional as F

        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.clear_cache()

        # Phase 1: rollouts (no grad).
        self.model.eval()
        rollouts = []
        rewards = []
        for g in range(self.config.group_size):
            seed = 42 + self.iteration * 1000 + g * 10000
            rollout = self._collect_rollout(prompt, seed=seed)
            reward = grade_fn(rollout["response_text"])
            rollout["reward"] = reward
            rollouts.append(rollout)
            rewards.append(reward)
        rollout_time = time.time() - t0

        # Phase 2: advantages.
        advantages, mean_reward, std_reward = _compute_advantages(rollouts)

        # Phase 3: GRPO update (with grad).
        self.model.train()
        t1 = time.time()
        loss_sum = 0.0
        n_updated = 0
        self.optimizer.zero_grad()
        grad_accum = None

        for rollout, advantage in zip(rollouts, advantages):
            if abs(advantage) < 1e-8:
                continue
            tokens = rollout["tokens"]
            action_positions = rollout["action_positions"]
            old_lp = torch.tensor(rollout["old_logprobs"], dtype=torch.float32, device=self.device)
            new_segments = self._compute_action_logprobs(tokens, action_positions)
            new_lp = torch.cat(new_segments)
            min_len = min(new_lp.shape[0], old_lp.shape[0])
            new_lp = new_lp[:min_len]
            old_lp_t = old_lp[:min_len]
            ratio = torch.exp(new_lp - old_lp_t)
            clipped = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps)
            pg_loss = -torch.min(ratio * advantage, clipped * advantage)
            kl = (new_lp - old_lp_t).mean()
            loss = (pg_loss + self.config.kl_beta * kl).mean()
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            loss_sum += float(loss.item())
            n_updated += 1
            if grad_accum is None:
                grad_accum = {k: v.grad.clone() if v.grad is not None else None for k, v in self.model.named_parameters()}
            else:
                for k, v in self.model.named_parameters():
                    if v.grad is not None:
                        if grad_accum[k] is None:
                            grad_accum[k] = v.grad.clone()
                        else:
                            grad_accum[k] += v.grad
            self.optimizer.zero_grad()
            self.clear_cache()

        if grad_accum is not None and n_updated > 0:
            for k, v in self.model.named_parameters():
                if grad_accum.get(k) is not None:
                    v.grad = grad_accum[k] / n_updated
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
        self.optimizer.zero_grad()
        self.clear_cache()

        update_time = time.time() - t1
        total_time = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        avg_tokens = float(np.mean([len(r["tokens"]) for r in rollouts])) if rollouts else 0.0

        metrics = GRPOMetrics(
            iteration=self.iteration,
            mean_reward=float(mean_reward) if mean_reward == mean_reward else 0.0,
            std_reward=float(std_reward) if std_reward == std_reward else 0.0,
            max_reward=max(rewards) if rewards else 0.0,
            min_reward=min(rewards) if rewards else 0.0,
            mean_correctness=float(np.mean([r["reward"] for r in rollouts])) if rollouts else 0.0,
            mean_reasoning_quality=0.0,
            loss=loss_sum / max(n_updated, 1) if n_updated > 0 else 0.0,
            rollout_time=rollout_time,
            update_time=update_time,
            total_time=total_time,
            peak_memory_gb=peak,
            avg_episode_tokens=avg_tokens,
        )
        if on_metrics:
            on_metrics(metrics)
        self.iteration += 1
        return metrics

    def save(self, path: str | None = None):
        path = path or self.adapter_path
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        cfg = {
            "adapter_path": os.path.basename(path),
            "fine_tune_type": "lora",
            "num_layers": 8,
            "lora_parameters": {"rank": 8, "scale": 1.0, "dropout": 0.0},
            "backend": "vllm",
        }
        with open(os.path.join(path, "adapter_config.json"), "w") as f:
            json.dump(cfg, f, indent=4)
        # Make the freshly trained adapter active for subsequent vLLM inference.
        if self.handle.backend_obj is not None:
            self.handle.backend_obj["active_adapter_path"] = path
        self.handle.adapter_path = path

    def clear_cache(self):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _compute_advantages(rollouts, eps=1e-8):
    import numpy as np

    rewards = np.array([r["reward"] for r in rollouts])
    mean_r = rewards.mean()
    std_r = rewards.std()
    if std_r < eps:
        return [0.0] * len(rollouts), mean_r, std_r
    advantages = (rewards - mean_r) / (std_r + eps)
    return advantages.tolist(), mean_r, std_r
