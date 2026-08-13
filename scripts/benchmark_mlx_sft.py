#!/usr/bin/env python3
"""Benchmark the compiled MLX-LM QLoRA path on a downloaded local model."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from iloptimus.core.dataset_tools import load_filtered_dataset
from iloptimus.core.hardware import detect_hardware
from iloptimus.core.inference import load_model, release_memory
from iloptimus.core.model_store import compatible_precision, resolve_model_source
from iloptimus.core.models import get_model
from iloptimus.core.sft import SFTConfig, SFTExample, run_sft


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--workspace", help="Use filtered rows from this dataset workspace")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--compile-bucket", type=int, default=128)
    parser.add_argument("--cache-threshold-gb", type=float, default=1.0)
    parser.add_argument("--legacy-bucket-rounding", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-scale", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--targets",
        default="self_attn.q_proj,self_attn.v_proj,self_attn.o_proj",
        help="Comma-separated LoRA module keys",
    )
    arguments = parser.parse_args()

    hardware = detect_hardware()
    model = get_model(arguments.model)
    if model is None:
        raise SystemExit(f"Unknown model: {arguments.model}")
    precision = compatible_precision(model, hardware)
    source = resolve_model_source(model.id, precision, hardware.recommended_backend)
    if source is None:
        raise SystemExit("Download the model before benchmarking training")
    if arguments.workspace:
        rows = load_filtered_dataset(arguments.workspace)
        if not rows:
            raise SystemExit(f"Workspace has no filtered rows: {arguments.workspace}")
        examples = [
            SFTExample(prompt=str(row["prompt"]), response=str(row["ideal_response"]))
            for row in rows
        ]
    else:
        examples = [
            SFTExample(
                prompt=f"Implement reusable rendering pattern {index}",
                response="const scene = new THREE.Scene(); function animate(){ requestAnimationFrame(animate); }",
            )
            for index in range(16)
        ]
    started = time.perf_counter()
    handle = load_model(model.huggingface_id, precision, source_override=source)
    adapter_config: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="iloptimus-sft-benchmark-") as temporary:
        losses: list[float] = []
        run_sft(
            handle,
            examples,
            config=SFTConfig(
                num_iters=arguments.iterations,
                learning_rate=arguments.learning_rate,
                max_seq_length=arguments.sequence_length,
                steps_per_eval=1,
                lora_rank=arguments.rank,
                lora_layers=arguments.layers,
                lora_scale=arguments.lora_scale,
                lora_targets=tuple(item.strip() for item in arguments.targets.split(",") if item.strip()),
                grad_accumulation_steps=arguments.grad_accumulation,
                batch_size=arguments.batch_size,
                compile_bucket_size=arguments.compile_bucket,
                clear_cache_threshold_gb=arguments.cache_threshold_gb,
                preserve_native_bucket_shape=not arguments.legacy_bucket_rounding,
                optimizer="adamw",
                mask_prompt=True,
                seed=arguments.seed,
            ),
            adapter_path=str(Path(temporary) / "adapter"),
            on_metrics=lambda metrics: losses.append(metrics.loss),
        )
        adapter_config = json.loads(
            (Path(temporary) / "adapter" / "adapter_config.json").read_text(encoding="utf-8")
        )
    elapsed = time.perf_counter() - started
    release_memory()
    print(
        json.dumps(
            {
                "model": model.id,
                "precision": precision,
                "iterations": arguments.iterations,
                "sequence_length": arguments.sequence_length,
                "examples": len(examples),
                "rank": arguments.rank,
                "layers": arguments.layers,
                "targets": [item.strip() for item in arguments.targets.split(",") if item.strip()],
                "grad_accumulation": arguments.grad_accumulation,
                "batch_size": arguments.batch_size,
                "compile_bucket": arguments.compile_bucket,
                "cache_threshold_gb": arguments.cache_threshold_gb,
                "preserve_native_bucket_shape": not arguments.legacy_bucket_rounding,
                "learning_rate": arguments.learning_rate,
                "lora_scale": arguments.lora_scale,
                "seed": arguments.seed,
                "elapsed_seconds_including_load": round(elapsed, 3),
                "reported_losses": losses,
                "trainable_parameters": adapter_config.get("trainable_parameters", 0),
                "mean_iterations_per_second": adapter_config.get("mean_iterations_per_second", 0),
                "mean_tokens_per_second": adapter_config.get("mean_tokens_per_second", 0),
                "trained_tokens": adapter_config.get("trained_tokens", 0),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
