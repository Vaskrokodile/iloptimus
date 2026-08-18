"""
Boosted-v1-Small SFT Round 1 — Kaggle GPU
==========================================
Manual training loop (no TRL dependency) for maximum compatibility.
Trains DeepSeek-R1-Distill-Qwen-1.5B with LoRA on:
  1. OpenThoughts-114k (code subset) — HumanEval improvement
  2. Threen-PT-ShareGPT — Three.js scene generation
  3. CodeForces-CoTs — competitive programming reasoning
"""

import os
import sys
import json
import time
import subprocess

# =====================================================================
# 1. Install dependencies
# =====================================================================
print("=" * 60)
print("Installing dependencies...")
print("=" * 60)

# Check GPU type WITHOUT importing torch (to avoid triton namespace conflict)
# Use a subprocess to check GPU capability
gpu_cap = (99, 0)
try:
    result = subprocess.run([sys.executable, "-c",
        "import torch; print(torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (99,0))"],
        capture_output=True, text=True, timeout=30)
    gpu_cap = eval(result.stdout.strip())
    print(f"GPU compute capability: {gpu_cap[0]}.{gpu_cap[1]}")
except Exception as e:
    print(f"GPU detection failed: {e}")

if gpu_cap[0] < 7:
    # P100 (sm_60) — Kaggle's PyTorch 2.10 doesn't support it
    # Install PyTorch 2.4.1+cu121 which supports sm_60+
    print("P100 detected — installing PyTorch 2.4.1+cu121 for sm_60 support...")
    # Force-remove triton packages to avoid namespace conflict
    import shutil
    for triton_path in [
        "/usr/local/lib/python3.12/dist-packages/triton",
        "/usr/local/lib/python3.12/dist-packages/triton-nightly",
        "/usr/local/lib/python3.12/dist-packages/torch/triton",
    ]:
        if os.path.exists(triton_path):
            print(f"  Removing {triton_path}")
            shutil.rmtree(triton_path, ignore_errors=True)
    # Uninstall via pip too
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y",
        "triton", "triton-nightly", "torch", "torchvision",
        "--no-deps"], check=False, capture_output=True)
    # Now install PyTorch 2.4.1
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
        "torch==2.4.1", "torchvision==0.19.1",
        "--index-url", "https://download.pytorch.org/whl/cu121",
        "--no-deps"], check=True)
    # Install triton 3.0.0 (compatible with torch 2.4.1)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
        "triton==3.0.0"], check=False)
    print(f"PyTorch 2.4.1 installed")

# Install compatible versions of transformers/peft
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.46.3", "datasets==3.1.0",
    "peft==0.13.2", "accelerate==1.1.1"], check=True)
# Remove incompatible torchao
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
    check=False, capture_output=True)
print("Dependencies installed.")

# =====================================================================
# 2. Imports — torch is imported fresh here (no prior import in this process)
# =====================================================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  VRAM: {props.total_memory / 1e9:.1f} GB")
    gpu_cap = torch.cuda.get_device_capability(0)
    print(f"  Compute capability: {gpu_cap[0]}.{gpu_cap[1]}")

# =====================================================================
# 3. Load model
# =====================================================================
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

print(f"\nLoading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

gpu_cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
use_bf16 = gpu_cap[0] >= 7
dtype = torch.bfloat16 if use_bf16 else torch.float16

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    attn_implementation="sdpa",
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
print(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# =====================================================================
# 4. Apply LoRA
# =====================================================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "self_attn.q_proj", "self_attn.v_proj", "self_attn.o_proj",
        "self_attn.k_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.train()
trainable, total = model.get_nb_trainable_parameters()
print(f"LoRA applied. Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
if hasattr(model, 'enable_input_require_grads'):
    model.enable_input_require_grads()

# =====================================================================
# 5. Load and prepare datasets
# =====================================================================
print("\n" + "=" * 60)
print("Loading datasets...")
print("=" * 60)

# --- Dataset 1: OpenThoughts-114k (code subset) ---
print("Loading OpenThoughts-114k (code subset)...")
try:
    ot_ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train")
    if "domain" in ot_ds.column_names:
        ot_code = ot_ds.filter(lambda x: x.get("domain") == "code")
    else:
        ot_code = ot_ds
    print(f"  OpenThoughts code: {len(ot_code)} examples")

    def format_openthoughts(ex):
        reasoning = ex.get("deepseek_reasoning", "")
        solution = ex.get("deepseek_solution", "")
        problem = ex.get("problem", "")
        if not problem or not solution:
            return None
        text = f"User: {problem}\n\nAssistant: \n{reasoning}\n\n**Solution:**\n{solution}"
        return {"text": text}
    ot_formatted = ot_code.map(format_openthoughts, remove_columns=ot_code.column_names)
    ot_formatted = ot_formatted.filter(lambda x: x["text"] is not None and len(x["text"]) > 50)
    if len(ot_formatted) > 8000:
        ot_formatted = ot_formatted.shuffle(seed=42).select(range(8000))
    print(f"  Formatted (capped 8000): {len(ot_formatted)}")
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
        text = f"User: {user_msg}\n\nAssistant: {asst_msg}"
        return {"text": text}
    threejs_formatted = threejs_ds.map(format_threejs, remove_columns=threejs_ds.column_names)
    threejs_formatted = threejs_formatted.filter(lambda x: x["text"] is not None and len(x["text"]) > 50)
    threejs_formatted = concatenate_datasets([threejs_formatted] * 8)
    print(f"  Formatted (8x): {len(threejs_formatted)}")
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
        text = f"User: Solve this competitive programming problem:\n{problem_text}\n\nAssistant: \n{reasoning}\n\n**Solution:**\n{solution}"
        return {"text": text}
    cf_formatted = cf_ds.map(format_codeforces, remove_columns=cf_ds.column_names)
    cf_formatted = cf_formatted.filter(lambda x: x["text"] is not None and len(x["text"]) > 50)
    if len(cf_formatted) > 3000:
        cf_formatted = cf_formatted.shuffle(seed=42).select(range(3000))
    print(f"  Formatted (capped 3000): {len(cf_formatted)}")
except Exception as e:
    print(f"  CodeForces failed: {e}")
    cf_formatted = Dataset.from_dict({"text": []})

# --- Combine ---
all_datasets = [d for d in [ot_formatted, threejs_formatted, cf_formatted] if len(d) > 0]
if not all_datasets:
    print("ERROR: No datasets loaded!")
    sys.exit(1)

train_dataset = concatenate_datasets(all_datasets).shuffle(seed=42)
print(f"\nTotal training examples: {len(train_dataset)}")

# =====================================================================
# 6. Tokenize dataset
# =====================================================================
print("\nTokenizing dataset...")
MAX_LENGTH = 2048

def tokenize_fn(examples):
    texts = examples["text"]
    # Tokenize with truncation and padding
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    # Labels = input_ids (causal LM)
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

tokenized_dataset = train_dataset.map(
    tokenize_fn,
    batched=True,
    batch_size=100,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing",
)
print(f"Tokenized: {len(tokenized_dataset)} examples")

# =====================================================================
# 7. Training loop
# =====================================================================
print("\n" + "=" * 60)
print("Starting training...")
print("=" * 60)

BATCH_SIZE = 2
GRAD_ACCUM = 8
EPOCHS = 3
LR = 2e-4
WARMUP_STEPS = 50

# Simple data loader
def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(b["input_ids"]) for b in batch])
    attention_mask = torch.stack([torch.tensor(b["attention_mask"]) for b in batch])
    labels = torch.stack([torch.tensor(b["labels"]) for b in batch])
    # Mask padding in labels
    labels[attention_mask == 0] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

dataloader = DataLoader(
    tokenized_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True,
)

# Optimizer + scheduler
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR,
    weight_decay=0.01,
)

total_steps = len(dataloader) * EPOCHS // GRAD_ACCUM
def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Training
device = "cuda" if torch.cuda.is_available() else "cpu"
scaler = torch.amp.GradScaler('cuda', enabled=not use_bf16)

t0 = time.time()
global_step = 0
log_loss = 0.0

for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast('cuda', dtype=dtype, enabled=use_bf16 or not use_bf16):
            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACCUM

        if use_bf16:
            loss.backward()
        else:
            scaler.scale(loss).backward()

        log_loss += loss.item() * GRAD_ACCUM

        if (step + 1) % GRAD_ACCUM == 0:
            if use_bf16:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
            else:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 10 == 0:
                avg_loss = log_loss / (GRAD_ACCUM * 10)
                elapsed = time.time() - t0
                remaining = elapsed / global_step * (total_steps - global_step)
                vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                print(f"  step {global_step}/{total_steps} | loss={avg_loss:.4f} | "
                      f"lr={scheduler.get_last_lr()[0]:.2e} | "
                      f"{elapsed/60:.1f}min elapsed, {remaining/60:.1f}min remaining | "
                      f"VRAM={vram:.1f}GB")
                log_loss = 0.0

    # Save checkpoint after each epoch
    ckpt_dir = f"/kaggle/working/sft_output/epoch_{epoch+1}"
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"  Saved checkpoint: {ckpt_dir}")

elapsed = time.time() - t0
print(f"\nTraining complete in {elapsed/60:.1f} minutes ({global_step} steps)")

# =====================================================================
# 8. Save final adapter
# =====================================================================
output_dir = "/kaggle/working/sft_output"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

metrics = {
    "training_time_minutes": elapsed / 60,
    "total_steps": global_step,
    "num_examples": len(train_dataset),
    "epochs": EPOCHS,
    "learning_rate": LR,
    "batch_size": BATCH_SIZE,
    "grad_accumulation": GRAD_ACCUM,
    "max_length": MAX_LENGTH,
    "lora_rank": 16,
    "lora_alpha": 32,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    "dtype": "bf16" if use_bf16 else "fp16",
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
print(f"Metrics: {json.dumps(metrics, indent=2)}")
print("\nOutput files:")
for root, dirs, files in os.walk(output_dir):
    for fname in files:
        fpath = os.path.join(root, fname)
        size = os.path.getsize(fpath) / 1e6
        print(f"  {fpath} ({size:.1f} MB)")

print("\n" + "=" * 60)
print("DONE! Download the adapter from the Output tab.")
print("=" * 60)
