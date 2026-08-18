"""
Boosted-v1-Small SFT Round 1 — Kaggle T4 x2
=============================================
Trains DeepSeek-R1-Distill-Qwen-1.5B with LoRA on:
  1. OpenThoughts-114k (code subset) — HumanEval improvement
  2. Threen-PT-ShareGPT — Three.js scene generation
  3. CodeForces-CoTs — competitive programming reasoning

Uses Flash Attention 2, gradient checkpointing, and TRL SFTTrainer.
Saves the trained adapter as a Kaggle output for download.
"""

import os
import sys
import json
import time
import subprocess

# =====================================================================
# 1. Install dependencies (Kaggle T4 has CUDA 12.x + Python 3.10/3.11)
# =====================================================================
def install_deps():
    print("=" * 60)
    print("Installing dependencies...")
    print("=" * 60)
    # Flash Attention 2 — prebuilt wheels for T4 (compute 7.5)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
        "flash-attn", "--no-build-isolation"], check=False)
    # Core training stack
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
        "transformers>=4.46.0", "datasets>=3.0.0", "peft>=0.13.0",
        "trl>=0.12.0", "accelerate>=1.0.0", "bitsandbytes>=0.44.0"], check=True)
    print("Dependencies installed.")

install_deps()

# =====================================================================
# 2. Imports & setup
# =====================================================================
import torch
import datasets
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)")

# Check Flash Attention
try:
    import flash_attn
    print(f"Flash Attention: {flash_attn.__version__}")
    FLASH_ATTN = True
except ImportError:
    print("Flash Attention not available, using SDPA")
    FLASH_ATTN = False

# =====================================================================
# 3. Load model with Flash Attention
# =====================================================================
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ADAPTER_REPO = "Akahsizrr/boosted-v1-small"  # Pre-trained adapter to build on

print("\n" + "=" * 60)
print(f"Loading {MODEL_ID} with Flash Attention...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load in bf16 (T4 supports bf16) with Flash Attention 2
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2" if FLASH_ATTN else "sdpa",
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
print(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Try to load the pre-trained adapter
print(f"\nLoading pre-trained adapter: {ADAPTER_REPO}")
try:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER_REPO)
    print("Pre-trained adapter loaded successfully!")
    # Keep as PeftModel for cumulative training (don't merge)
except Exception as e:
    print(f"Could not load pre-trained adapter: {e}")
    print("Training from base model instead.")

# =====================================================================
# 4. Load and prepare datasets
# =====================================================================
print("\n" + "=" * 60)
print("Loading datasets...")
print("=" * 60)

# --- Dataset 1: OpenThoughts-114k (code subset) ---
print("Loading OpenThoughts-114k (code subset)...")
try:
    ot_ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train")
    # Filter for code domain
    if "domain" in ot_ds.column_names:
        ot_code = ot_ds.filter(lambda x: x.get("domain") == "code")
    else:
        ot_code = ot_ds
    print(f"  OpenThoughts code: {len(ot_code)} examples")
    # Format: use deepseek_reasoning + deepseek_solution
    def format_openthoughts(ex):
        reasoning = ex.get("deepseek_reasoning", "")
        solution = ex.get("deepseek_solution", "")
        problem = ex.get("problem", "")
        if not problem or not solution:
            return None
        # DeepSeek R1 format: <think>reasoning</think> answer
        text = f"User: {problem}\n\nAssistant: <think>\n{reasoning}\n</think>\n\n{solution}{tokenizer.eos_token}"
        return {"text": text}
    ot_formatted = ot_code.map(format_openthoughts, remove_columns=ot_code.column_names)
    ot_formatted = ot_formatted.filter(lambda x: x["text"] is not None and len(x["text"]) > 50)
    print(f"  Formatted: {len(ot_formatted)} examples")
except Exception as e:
    print(f"  OpenThoughts failed: {e}")
    ot_formatted = Dataset.from_dict({"text": []})

# --- Dataset 2: Three.js scene generation ---
print("Loading Threen-PT-ShareGPT (Three.js)...")
try:
    threejs_ds = load_dataset("Akicou/Threen-PT-ShareGPT", split="train")
    print(f"  Three.js: {len(threejs_ds)} examples")
    def format_threejs(ex):
        convs = ex.get("conversations", [])
        if len(convs) < 2:
            return None
        user_msg = convs[0].get("value", "") if convs[0].get("from") == "human" else ""
        asst_msg = convs[1].get("value", "") if convs[1].get("from") == "gpt" else ""
        if not user_msg or not asst_msg:
            return None
        text = f"User: {user_msg}\n\nAssistant: {asst_msg}{tokenizer.eos_token}"
        return {"text": text}
    threejs_formatted = threejs_ds.map(format_threejs, remove_columns=threejs_ds.column_names)
    threejs_formatted = threejs_formatted.filter(lambda x: x["text"] is not None and len(x["text"]) > 50)
    # Oversample Three.js 5x to balance with the larger code dataset
    threejs_formatted = concatenate_datasets([threejs_formatted] * 5)
    print(f"  Formatted (5x oversampled): {len(threejs_formatted)} examples")
except Exception as e:
    print(f"  Three.js failed: {e}")
    threejs_formatted = Dataset.from_dict({"text": []})

# --- Dataset 3: CodeForces reasoning traces ---
print("Loading CodeForces-CoTs...")
try:
    cf_ds = load_dataset("open-r1/codeforces-cots", "solutions", split="train")
    print(f"  CodeForces: {len(cf_ds)} examples")
    def format_codeforces(ex):
        problem = ex.get("problem", {})
        if isinstance(problem, dict):
            problem_text = problem.get("statement", "") or problem.get("prompt", "")
        else:
            problem_text = str(problem)
        reasoning = ex.get("reasoning", "") or ex.get("deepseek_reasoning", "")
        solution = ex.get("solution", "") or ex.get("code", "")
        if not problem_text or not solution:
            return None
        text = f"User: Solve this competitive programming problem:\n{problem_text}\n\nAssistant: <think>\n{reasoning}\n</think>\n\n{solution}{tokenizer.eos_token}"
        return {"text": text}
    cf_formatted = cf_ds.map(format_codeforces, remove_columns=cf_ds.column_names)
    cf_formatted = cf_formatted.filter(lambda x: x["text"] is not None and len(x["text"]) > 50)
    # Limit to 5000 for time budget
    if len(cf_formatted) > 5000:
        cf_formatted = cf_formatted.shuffle(seed=42).select(range(5000))
    print(f"  Formatted (capped at 5000): {len(cf_formatted)} examples")
except Exception as e:
    print(f"  CodeForces failed: {e}")
    cf_formatted = Dataset.from_dict({"text": []})

# --- Combine all datasets ---
all_datasets = [d for d in [ot_formatted, threejs_formatted, cf_formatted] if len(d) > 0]
if not all_datasets:
    print("ERROR: No datasets loaded!")
    sys.exit(1)

train_dataset = concatenate_datasets(all_datasets)
train_dataset = train_dataset.shuffle(seed=42)
print(f"\nTotal training examples: {len(train_dataset)}")
print(f"  Sample (first 200 chars): {train_dataset[0]['text'][:200]}")

# =====================================================================
# 5. Configure LoRA + SFT training
# =====================================================================
print("\n" + "=" * 60)
print("Configuring LoRA + SFT training...")
print("=" * 60)

# LoRA config — same targets as the existing adapter for cumulative training
lora_config = LoraConfig(
    r=16,  # Higher rank for more capacity
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "self_attn.q_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "self_attn.k_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

# Training config
training_args = SFTConfig(
    output_dir="/kaggle/working/sft_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # effective batch size = 4 * 4 * 2 GPUs = 32
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    max_grad_norm=1.0,
    logging_steps=10,
    save_steps=200,
    save_total_limit=3,
    bf16=True,  # T4 supports bf16
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_seq_length=2048,
    dataset_text_field="text",
    report_to="none",
    optim="adamw_torch",
    seed=42,
    dataloader_num_workers=2,
    remove_unused_columns=True,
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=lora_config if not hasattr(model, 'peft_type') else None,
    processing_class=tokenizer,
)

# Print trainable params
if hasattr(model, 'get_nb_trainable_parameters'):
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

# =====================================================================
# 6. Train!
# =====================================================================
print("\n" + "=" * 60)
print("Starting SFT training...")
print("=" * 60)

t0 = time.time()
trainer.train()
elapsed = time.time() - t0
print(f"\nTraining complete in {elapsed / 60:.1f} minutes")

# =====================================================================
# 7. Save the trained adapter
# =====================================================================
print("\n" + "=" * 60)
print("Saving trained adapter...")
print("=" * 60)

output_dir = "/kaggle/working/sft_output"
os.makedirs(output_dir, exist_ok=True)

# Save adapter
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# Save training metrics
metrics = {
    "training_time_seconds": elapsed,
    "training_time_minutes": elapsed / 60,
    "num_examples": len(train_dataset),
    "epochs": training_args.num_train_epochs,
    "learning_rate": training_args.learning_rate,
    "batch_size": training_args.per_device_train_batch_size,
    "gradient_accumulation": training_args.gradient_accumulation_steps,
    "max_seq_length": training_args.max_seq_length,
    "lora_rank": lora_config.r,
    "lora_alpha": lora_config.lora_alpha,
    "flash_attention": FLASH_ATTN,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    "datasets": {
        "openthoughts_code": len(ot_formatted),
        "threejs": len(threejs_formatted),
        "codeforces": len(cf_formatted),
        "total": len(train_dataset),
    },
}
with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nAdapter saved to {output_dir}")
print(f"Training metrics: {json.dumps(metrics, indent=2)}")

# List output files
print("\nOutput files:")
for root, dirs, files in os.walk(output_dir):
    for fname in files:
        fpath = os.path.join(root, fname)
        size = os.path.getsize(fpath) / 1e6
        print(f"  {fpath} ({size:.1f} MB)")

print("\n" + "=" * 60)
print("DONE! Download the adapter from the Output tab.")
print("=" * 60)
