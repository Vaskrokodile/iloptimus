"""Factory dataset quality run — measures a real local model before and after
SFT on a factory-built dataset.

This script drives only shipped ILOptimus functions end to end:

    DatasetJobRunner  -> factory-built, audited dataset (bulk corpus in one job)
    save_environment  -> dataset rows become an IL taskset (custom:<id>)
    model_store       -> download / resolve the smallest registered model
    inference         -> load the model on the torch/CUDA backend
    benchmark         -> run_benchmark grades responses deterministically
    sft               -> generate_sft_data + run_sft (LoRA on the dataset)

The comparison printed at the end (baseline vs post-SFT accuracy and mean
score) is the measured effect of training on the factory dataset.

Environment note: HF cache and the ILOptimus app home are placed on E:
because the C: drive has almost no free space on this machine.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

# --- environment must be set before any HF / iloptimus import -----------
os.environ.setdefault("HF_HOME", r"E:\hf-cache")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("ILOPTIMUS_HOME", r"E:\iloptimus-home")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _code_source(name: str, feature: str, origin_index: int = 0) -> dict:
    """Unique, permissively licensed, code-like source text."""
    lines: list[str] = []
    for p in range(6):
        lines.append(f"const {feature}_unit{p}_{name} = function (scene{p}) {{")
        for i in range(8):
            lines.append(
                f"  const value{p}_{i} = new {feature.title()}Widget({p}, {i}, '{name}-{p}-{i}');"
            )
            lines.append(
                f"  scene{p}.add(value{p}_{i}); // {feature} line {p}.{i} token-{name}-{p}-{i}"
            )
        lines.append("};")
    return {
        "title": name,
        "url": f"https://origin{origin_index % 40}.example/{name}",
        "text": "\n".join(lines),
        "license": "MIT",
        "kind": "repository-code",
    }


FEATURES = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]


def build_factory_dataset(corpus_size: int = 300) -> dict:
    from iloptimus.core.dataset_factory import DatasetJobRunner, DatasetJobSpec

    runner = DatasetJobRunner()
    spec = DatasetJobSpec(
        task="factory quality run holdout probe",
        artifact_kind="code",
        requested_features=FEATURES,
        sources=[
            _code_source(f"src{index}", FEATURES[index % len(FEATURES)], origin_index=index)
            for index in range(corpus_size)
        ],
        maximum_rows=50_000,
        assembled_examples=20_000,
        expanded_examples=30_000,
    )
    state = runner.create(spec)
    finished = runner.run(state.job_id)
    if finished.status != "done":
        raise RuntimeError(f"Dataset job failed: {finished.error}")
    return {
        "job_id": state.job_id,
        "workspace_id": spec.workspace_id,
        "result": finished.result,
        "audit": finished.result["filtering"],
    }


def make_environment(workspace_id: str, rows: list[dict]) -> str:
    """Convert filtered dataset rows into an IL environment taskset —
    the same schema the server's test-time-compute session uses."""
    from iloptimus.core.environments import save_environment

    tasks = []
    for index, row in enumerate(rows):
        expected = str(row.get("expected_answer") or row.get("ideal_response") or "")
        terms = re.findall(r"[A-Za-z_$][A-Za-z0-9_$]{4,}", expected)[:2] or ["source"]
        tasks.append(
            {
                "name": f"Factory pattern {index}",
                "prompt": str(row.get("prompt") or ""),
                "expected_answer": expected,
                "ideal_response": str(row.get("ideal_response") or ""),
                "criteria": ["preserves verified APIs", "returns runnable source"],
                "grader": {"type": "contains_all", "terms": terms},
                "difficulty": "hard",
            }
        )
    environment = save_environment(
        {
            "name": f"Factory dataset quality run {workspace_id}",
            "mode": "IL",
            "goal": "Learn reusable implementation patterns from the factory corpus",
            "description": "Dataset factory quality measurement environment",
            "domain": "artifact-building",
            "reward": {"correctness": 0.8, "reasoning": 0.1, "efficiency": 0.1},
            "tasks": tasks,
            "builder": {"model_id": "factory-quality-run", "used_model_output": False},
        }
    )
    return environment["id"]


def main() -> int:
    from iloptimus.core.models import get_all_models
    from iloptimus.core import model_store
    from iloptimus.core.dataset_tools import load_filtered_dataset

    started = time.time()
    print("=" * 76)
    print("FACTORY DATASET QUALITY RUN")
    print("=" * 76)

    # ------------------------------------------------------------------ model
    model = min(get_all_models(), key=lambda item: item.params_b)
    print(f"\n[1/6] Smallest registered model: {model.id} ({model.params_b}B, {model.huggingface_id})")

    import torch

    print(f"      torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}"
          + (f" | GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    if not torch.cuda.is_available():
        print("      WARNING: CUDA not available — falling back to CPU (slow)")

    print("      Downloading model (no-op when already on disk)...")
    state = model_store.download_model(model.id, "fp16", "vllm")
    print(f"      download status: {state.status} ({state.size_gb} GB) {state.error}")
    if state.status != "downloaded":
        print("MODEL DOWNLOAD FAILED — cannot run the quality comparison.")
        return 2
    source = model_store.resolve_model_source(model.id, "fp16", "vllm")
    print(f"      resolved source: {source}")

    # ---------------------------------------------------------------- dataset
    print("\n[2/6] Building factory dataset via DatasetJobRunner (bulk corpus, one job)...")
    dataset = build_factory_dataset(corpus_size=300)
    audit = dataset["audit"]
    print(f"      workspace: {dataset['workspace_id']}")
    print(f"      accepted rows: {audit['accepted_rows']} | sources: {audit['source_count']}"
          f" | origins: {audit['origin_count']} | mean quality: {audit['mean_quality_score']}")
    print(f"      gates: exact_dup={audit['exact_duplicates']} near_dup={audit['near_duplicates']}"
          f" short={audit['short_rows']} repetitive={audit['repetitive_rows']}"
          f" low_quality={audit['low_quality_rows']} dominated={audit['source_dominated_rows']}")

    rows = load_filtered_dataset(dataset["workspace_id"])
    train_rows = rows[:400]  # bounded for a tractable run on a single GPU
    if not train_rows:
        print("No rows survived curation — cannot continue.")
        return 2

    # ------------------------------------------------------------ environment
    print(f"\n[3/6] Converting {len(train_rows)} dataset rows into an IL environment taskset...")
    environment_id = make_environment(dataset["workspace_id"], train_rows)
    domain = f"custom:{environment_id}"
    print(f"      environment: {environment_id} (domain {domain})")

    # ------------------------------------------------------------------ load
    print("\n[4/6] Loading model on the torch/CUDA backend...")
    from iloptimus.core.inference import load_model

    handle = load_model(
        model.huggingface_id,
        precision="fp16",
        source_override=source,
        backend="vllm",
    )
    print(f"      loaded: backend={handle.backend}")

    # -------------------------------------------------------------- baseline
    print("\n[5/6] Baseline benchmark (real inference + deterministic grading)...")
    from iloptimus.core.benchmark import run_benchmark

    n_bench = min(24, len(train_rows))
    baseline = run_benchmark(
        handle, domain, num_tasks=n_bench, max_reasoning_tokens=192, max_answer_tokens=256
    )
    print(f"      BASELINE accuracy={baseline.accuracy:.3f} mean_score={baseline.mean_score:.3f}"
          f" tokens/s={baseline.mean_tokens_per_sec:.1f}")

    # ------------------------------------------------------------------- sft
    print(f"\n[6/6] SFT on the factory dataset ({len(train_rows)} rows)...")
    from iloptimus.core.backends import SFTConfig
    from iloptimus.core.sft import generate_sft_data, run_sft

    examples = generate_sft_data(handle, domain, num_tasks=len(train_rows))
    print(f"      SFT examples prepared: {len(examples)}")
    sft_config = SFTConfig(
        learning_rate=1e-4,
        num_iters=200,
        batch_size=1,
        lora_rank=8,
        lora_layers=8,
        max_seq_length=1024,
        mask_prompt=True,
    )
    adapter_dir = os.path.join(os.environ["ILOPTIMUS_HOME"], "adapters", "factory-quality-run")
    losses: list[float] = []

    def on_metrics(metrics):
        losses.append(metrics.loss)
        if metrics.iteration % 40 == 0 or metrics.iteration == sft_config.num_iters - 1:
            print(f"      sft iter {metrics.iteration}: loss={metrics.loss:.4f}")

    run_sft(handle, examples, sft_config, adapter_path=adapter_dir, on_metrics=on_metrics)
    print(f"      adapter saved to {adapter_dir}")
    if hasattr(handle.model, "eval"):
        handle.model.eval()

    # ------------------------------------------------------------- post-sft
    print("\n      Post-SFT benchmark (same tasks, adapter applied)...")
    post = run_benchmark(
        handle, domain, num_tasks=n_bench, max_reasoning_tokens=192, max_answer_tokens=256
    )
    print(f"      POST-SFT accuracy={post.accuracy:.3f} mean_score={post.mean_score:.3f}"
          f" tokens/s={post.mean_tokens_per_sec:.1f}")

    # ---------------------------------------------------------------- report
    report = {
        "model_id": model.id,
        "huggingface_id": model.huggingface_id,
        "backend": "vllm (torch/CUDA)" if torch.cuda.is_available() else "vllm (torch/CPU)",
        "dataset": {
            "workspace_id": dataset["workspace_id"],
            "accepted_rows": audit["accepted_rows"],
            "trained_rows": len(train_rows),
            "audit": audit,
        },
        "sft": {
            "examples": len(examples),
            "iters": sft_config.num_iters,
            "first_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
            "adapter_path": adapter_dir,
        },
        "benchmark_tasks": n_bench,
        "baseline": {
            "accuracy": baseline.accuracy,
            "mean_score": baseline.mean_score,
            "tokens_per_sec": baseline.mean_tokens_per_sec,
        },
        "post_sft": {
            "accuracy": post.accuracy,
            "mean_score": post.mean_score,
            "tokens_per_sec": post.mean_tokens_per_sec,
        },
        "delta": {
            "accuracy": round(post.accuracy - baseline.accuracy, 4),
            "mean_score": round(post.mean_score - baseline.mean_score, 4),
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    print("\n" + "=" * 76)
    print("QUALITY COMPARISON — factory dataset effect on a real local model")
    print("=" * 76)
    print(f"  baseline : accuracy={baseline.accuracy:.3f}  mean_score={baseline.mean_score:.3f}")
    print(f"  post-SFT : accuracy={post.accuracy:.3f}  mean_score={post.mean_score:.3f}")
    print(f"  delta    : accuracy={report['delta']['accuracy']:+.3f}"
          f"  mean_score={report['delta']['mean_score']:+.3f}")
    print(f"  total elapsed: {report['elapsed_seconds']}s")

    report_path = os.path.join(os.path.dirname(__file__), "..", "factory-quality-report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nReport written to {os.path.abspath(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
