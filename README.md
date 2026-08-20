# IL Optimus — Intuition Learning Pipeline Studio

Run Intuition Learning (SFT + GRPO RL) pipelines locally with a web frontend.
Detects your hardware, recommends compatible models, lets you select tasksets,
and tracks training runs in real time.

## Install and Start

> **Requires Apple Silicon (M1/M2/M3/M4) or an NVIDIA CUDA GPU.** The pipeline
> runs on two local accelerator backends: **MLX** on Apple Silicon and
> **HuggingFace Transformers + PEFT** on NVIDIA CUDA (with optional vLLM for
> high-throughput inference on Linux). The active backend is auto-detected
> from your hardware.

### macOS (Apple Silicon)

```bash
curl -LsSf https://raw.githubusercontent.com/Vaskrokodile/iloptimus/main/scripts/install.sh | sh
```

The installer installs `uv` when needed, installs IL Optimus as an isolated
command-line app, builds a small native macOS app in `~/Applications`, starts
the local service, and opens the desktop window. If native installation is not
available, it prints the localhost URL and opens the default browser. Later,
start either surface with:

```bash
iloptimus desktop
iloptimus serve
```

### Windows (NVIDIA CUDA)

```powershell
# Download and run the PowerShell installer
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/Vaskrokodile/iloptimus/main/scripts/install.ps1" -OutFile "install.ps1"
powershell -ExecutionPolicy Bypass -File install.ps1
```

Or install manually:

```powershell
# Prerequisites: Python 3.11+, git, Node.js (for the web UI), NVIDIA CUDA GPU
git clone https://github.com/Vaskrokodile/iloptimus.git
cd iloptimus
pip install uv
uv pip install -e ".[cuda]"        # torch, transformers, peft, accelerate, bitsandbytes
npm install && npm run build       # build the web frontend
iloptimus serve                    # start the server at http://127.0.0.1:7860
```

> **Note on vLLM:** vLLM is Linux-only and does not build on Windows. On
> Windows/CUDA, IL Optimus uses HuggingFace Transformers `model.generate` for
> inference instead. This is fully functional — chat, IL training, GRPO, and
> the TTC pipeline all work. Inference is slower than vLLM but everything runs
> end-to-end. On Linux/CUDA, install with `uv pip install -e ".[cuda]"` to get
> vLLM for high-throughput batched inference.

### Linux (NVIDIA CUDA)

```bash
curl -LsSf https://raw.githubusercontent.com/Vaskrokodile/iloptimus/main/scripts/install.sh | sh
# Or manually:
git clone https://github.com/Vaskrokodile/iloptimus.git
cd iloptimus
pip install uv
uv pip install -e ".[cuda]"        # includes vLLM
npm install && npm run build
iloptimus serve
```

Then open `http://127.0.0.1:7860` in your browser.

## Autonomous Sakura Research Report

Open `http://127.0.0.1:7860/research/sakura-island` for the evidence-backed
research report covering the autonomous Sakura Island run. It includes the
stored Chromium capture, retry trajectory, model-versus-framework authorship
boundary, capability scores, measured pipeline improvements, limitations, and
links to the exact local experiment and provenance manifests.

Open `http://127.0.0.1:7860/research/optimus-map` for the interactive system
mind map. It covers every product surface, local runtime, no-code IL/RL path,
training stage, persistence layer, verification gate, and the current
self-improvement loop. Click any node for the exact behavior, or switch to
**Self-improvement loop** to isolate test-time compute, failure-skill memory,
automated curation, adapter training, exact retry, and the Sakura proof run.

## What It Does

1. **Detects your hardware** — CPU, RAM, GPU (Apple Silicon / CUDA / None),
   available backends (MLX, vLLM, PyTorch)
2. **Downloads local models** — compatible MLX checkpoints are saved locally and
   their real installation state is shown in Model Library
   (recommended / feasible / tight / not-recommended) based on memory requirements
3. **Browse tasksets** — 4 handcrafted IL tasksets (44 tasks total) spanning coding,
   reasoning, agentic reasoning, and agentic coding
4. **Run IL pipelines** — SFT + GRPO RL training with live SSE streaming of logs,
   training curves (loss/reward), and accuracy progression
5. **Build no-code environments** — IL and RL specifications become persistent,
   gradable tasksets that can be selected directly in Optimus Lab
6. **Track runs** — real-time progress bar, stage pipeline visualization, live log
   stream, and accuracy comparison (baseline → post-SFT → post-GRPO)
7. **Use prompt skills and tools** — trusted Markdown guidance is selected from
   the prompt, while public-web and configured MCP tools run through an audited
   execution boundary
8. **Run measured test-time compute** — `/ttc <artifact task>` generates and
   executes an unadapted baseline, gathers audited evidence through a bounded
   automated search frontier, builds a licensed provenance-tracked corpus, selects a
   defensible adaptation method, retries, and rejects adapters that do not pass
   every objective gate. Rejected small-model artifact runs can expose a
   separately labeled, browser-verified framework fallback without pretending
   the adapter succeeded. See [the TTC design and current experiment](docs/test-time-compute.md).
9. **Remember verified failures** — objective artifact failures become compact,
   validated repair skills. Matching future tasks retrieve them, while only a
   later passing model artifact can promote their success count.
10. **Build framework-backed model scenes** — small local models design a
    constrained, hash-audited scene specification while a trusted voxel-island
    Three.js runtime supplies rendering primitives. Invalid specifications receive
    mechanical feedback and retry; no Sakura fallback can count as a model
    success.

## Local Agent Skills and Tools

IL Optimus packages four prompt-only skills sourced from the official
`anthropics/skills` and `openai/skills` repositories: frontend design,
Playwright/browser testing, security best practices, and Jupyter notebooks.
The router activates at most two relevant skills from explicit prompt keywords.
Only `SKILL.md` is inserted into context; neighboring scripts are never exposed
to or executed by the local model.

Chat models can call the built-in `web_search`, `web_fetch`, `calculator`, and
`current_time` tools using the tool-call contract supplied in their prompt.
Public web requests reject loopback, private, link-local, multicast, reserved,
credential-bearing, and non-standard-port URLs, and every redirect is checked
again. Tool activity is appended to `~/.iloptimus/tool_calls.jsonl`.

The official MCP Python SDK provides real stdio MCP connectivity. Reference
`time` and `fetch` server configurations are created in
`~/.iloptimus/mcp.json`; both are disabled by default because enabling an MCP
server launches a local command. Set `enabled` to `true` only for a server you
trust. The built-in safe fetch implementation is preferred for ordinary web
access.

The context control in chat is capped by both the model's declared limit and a
KV-cache memory estimate for the detected hardware. Its TPS range combines
model weight size, architecture/KV cost, detected memory bandwidth and backend
efficiency. After a real response, the estimator calibrates itself against the
measured decode rate saved in `~/.iloptimus/performance.json`.

Artifact training data is compiled into complete syntax-bounded source units.
The curator records supervised-token retention before training. On the current
Sakura corpus this increased retained answer tokens from 61.6% to 99.64% while
expanding the accepted set from 79 to 111 independently sourced units.

For compact MLX adapters, IL Optimus can cache the frozen transformer prefix
once and train only the final LoRA-enabled suffix. The one-time cache build and
sustained suffix step rate are budgeted separately and persisted; the planner
does not hide cache construction behind the reported updates/second.

## How No-Code Environments Work

Type `/il <goal>` or `/rl <goal>` after selecting a downloaded model. IL Optimus
asks the local model for small task proposals, validates them, and compiles the
result into a versioned environment. Small models never write or execute Python:
they fill a constrained contract described in
`iloptimus/resources/environment-builder/SKILL.md`.

The trusted runtime supplies the parts that must be reliable:

- three to six self-contained tasks and prompts
- curated `ideal_response` demonstrations for IL/SFT
- deterministic `exact`, `numeric`, or `contains_all` graders for RL/GRPO
- benchmark prompts, reward weights, and an executable taskset adapter
- a validated fallback template when the local model's proposal is malformed

Every generated environment is saved under `~/.iloptimus/environments/`, appears
in **My environments**, and is registered immediately as a taskset in Optimus
Lab. Training uses the same grader for baseline evaluation, GRPO rollouts, and
post-training evaluation; IL demonstrations feed directly into LoRA SFT.

### Stateful simulator environments

Agent-style `/rl` goals can compile to framework-v3 state machines rather than
single responses. A simulator defines primitive state values, named actions,
action preconditions, safe effects, observations, terminal conditions, shaped
step rewards, invalid-action penalties, timeouts, and multiple initial-state
scenarios. The local API exposes real reset/step execution, and **My
environments → Test** provides an interactive episode console.

During training, the model emits an action trajectory. IL Optimus replays every
action through the same simulator, stops at terminal states, and feeds the
executed trajectory reward into benchmarking and GRPO. Built-in navigation,
tool-workflow, and resource-control templates give small local models a trusted
starting point without allowing generated code or external side effects.

## Local Data

All user-created data is kept outside the installed package under
`~/.iloptimus/`:

```text
~/.iloptimus/
├── models/          downloaded Hugging Face / MLX snapshots
├── environments/    no-code IL and RL specifications and generated tasksets
├── runs/<run-id>/   config, event log, metrics, traces, and LoRA adapters
├── skill-memory/    verifier-derived repair skills and cumulative evidence
├── mcp.json         explicit MCP server configuration (off by default)
├── performance.json local TPS calibration samples
├── training-performance.json measured optimizer-step profiles
└── tool_calls.jsonl append-only tool audit log
```

Run `iloptimus data-dir` to print the active folder. Set `ILOPTIMUS_HOME` to
use another location.

## Architecture

```
iloptimus/
├── cli.py                      CLI entry point (iloptimus serve)
├── server.py                   FastAPI app (API + static frontend serving)
├── core/
│   ├── hardware.py             Hardware detection (CPU/RAM/GPU/backends)
│   ├── models.py               14-model registry + compatibility scoring
│   ├── tasksets.py             4 IL taskset definitions
│   └── pipeline.py             IL pipeline runner (SFT + GRPO, SSE streaming)
├── web/
│   ├── dist/                   Built frontend (served by FastAPI)
│   └── src/                    React + TypeScript + Tailwind source
│       ├── App.tsx             Router (Dashboard / Models / IL-Studio / Tasksets)
│       ├── components/Navbar.tsx
│       ├── pages/
│       │   ├── DashboardPage.tsx    Hardware summary + quick stats
│       │   ├── ModelsPage.tsx       Model grid with compatibility badges
│       │   ├── ILStudioPage.tsx     Live run tracking with charts + logs
│       │   └── TasksetsPage.tsx     Taskset browser
│       └── api/client.ts       API client + types
├── il_coding_v1/               12 handcrafted coding tasks (sandboxed)
├── il_reasoning_v1/            12 handcrafted reasoning tasks
├── il_agentic_reasoning_v1/    10 multi-step reasoning tasks
└── il_agentic_coding_v1/       10 multi-file codebase tasks (sandboxed)
```

## Frontend Development

```bash
# Install Node deps
npm install

# Dev mode (hot reload, proxies API to localhost:7860)
npm run dev

# Build to iloptimus/web/dist/ (served by FastAPI in production)
npm run build
```

## IL Tasksets

All tasksets use efficiency-aware reward shaping:

```
final = correctness × (0.6 + 0.4 × reasoning_quality)
```

- Wrong answers always get 0
- Right answers with lazy reasoning get 0.6
- Right answers with thorough, verified reasoning get up to 1.0
- The 0.4 spread is the RL signal that shapes reasoning behavior

| Taskset | Domain | Tasks | Sandbox |
|---------|--------|-------|---------|
| `il-coding-v1` | coding | 12 | yes |
| `il-reasoning-v1` | reasoning | 12 | no |
| `il-agentic-reasoning-v1` | agentic reasoning | 10 | no |
| `il-agentic-coding-v1` | agentic coding | 10 | yes |

## CLI Commands

```bash
uv run iloptimus serve              # Start server + open browser
uv run iloptimus install-desktop    # Build the native macOS app
uv run iloptimus desktop            # Open the native macOS app
uv run iloptimus hardware           # Print detected hardware info
uv run iloptimus version            # Print version
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/hardware` | Detected hardware info |
| GET | `/api/models` | All models with compatibility scores |
| GET | `/api/models/{id}` | Single model detail |
| GET | `/api/tasksets` | All available tasksets |
| GET | `/api/tasksets/{id}` | Single taskset detail |
| POST | `/api/models/{id}/download` | Download a compatible local checkpoint |
| GET | `/api/models/{id}/status` | Read real local/download status |
| GET | `/api/models/{id}/context-estimate` | Hardware/model context and TPS estimate |
| GET | `/api/skills` | Packaged read-only prompt skills |
| GET | `/api/tools` | Built-in tools and configured MCP servers |
| POST | `/api/chat` | Chat with a downloaded local model |
| GET | `/api/learning-skills` | Inspect verifier-derived failure-skill memory |
| GET | `/api/learning/{id}` | Inspect persistent research/training/TTC state |
| GET | `/api/learning/{id}/events` | Stream subtask audits and adaptation progress |
| GET | `/api/learning/{id}/artifact/{variant}` | Open baseline, adapted, or verified framework artifacts/screenshots |
| GET/POST | `/api/environments` | List or create no-code IL/RL environments |
| POST | `/api/environments/from-chat` | Generate an environment with `/il` or `/rl` |
| POST | `/api/runs` | Start a persisted local training run |
| GET | `/api/runs` | List all runs |
| GET | `/api/runs/{id}` | Get run state |
| GET | `/api/runs/{id}/events` | SSE stream of run events |

## Backends

IL Optimus auto-detects your accelerator and selects one of two local
backends. Both implement the same `Backend` interface
(`iloptimus/core/backends/base.py`) so the pipeline, inference orchestration,
SFT data generation, and GRPO advantage computation are shared across
backends — only the primitives that genuinely differ (loading, generation,
logprob computation, training) are backend-specific.

- **MLX** (Apple Silicon) — uses `mlx_lm` for inference and compiled
  LoRA/QLoRA fine-tuning with cached tokenization, stable length buckets,
  prompt masking, frozen-prefix caching, selected attention targets, and
  hardware-budgeted steps. QLoRA trains directly on int4 quantized weights.
  Recommended for M-series Macs with unified memory.
- **vLLM + HuggingFace Transformers + PEFT** (NVIDIA CUDA) — uses `vllm` for
  high-throughput batched inference (with on-the-fly LoRA serving via
  `LoRARequest`) and HuggingFace Transformers + PEFT for LoRA/QLoRA SFT
  (4-bit NF4 via bitsandbytes) and a custom GRPO loop. When `vllm` is not
  installed (e.g. on Windows), inference falls back to `model.generate` so
  the backend still works on a torch-only CUDA box. Recommended for NVIDIA
  GPUs with >= 8GB VRAM. Fully tested on Windows 11 with CUDA.

Install the CUDA extras on Linux:

```bash
uv pip install -e ".[cuda]"      # torch, transformers, peft, accelerate, bitsandbytes, vllm
```

On Windows, the `[cuda]` extra installs everything except vLLM (which is
Linux-only). bitsandbytes ships Windows wheels for 4-bit NF4 quantization.

The MLX extras are installed by default on macOS and are darwin-only.

## Running TTC Experiments

The test-time-compute (TTC) pipeline runs an autonomous loop: generate a
baseline artifact, research the problem, build a dataset, train a temporary
LoRA adapter, and retry. To run one:

1. Open `http://127.0.0.1:7860` after starting the server
2. Download a model from the Models page (DeepSeek-R1-Distill-Qwen-1.5B or
   Boosted-v1-small are recommended — both fit in 4GB VRAM at int4)
3. In the chat, type: `/ttc Create a beautiful sakura cherry blossom island scene with three.js, with voxel terrain, shader water, sakura trees, falling petals, and a torii gate`
4. Or try: `/ttc Build a New York City cityscape in three.js with skyscrapers, cars, street lamps, and a dusk sky`
5. The pipeline runs end-to-end and stores all artifacts under `~/.iloptimus/learning/<session-id>/`

### Boosted-v1-small adapter

The `boosted-v1-small` model in the registry is a LoRA adapter for
DeepSeek-R1-Distill-Qwen-1.5B trained via the self-improvement pipeline. It
improved HumanEval from 24% to 70.88%. The adapter is downloaded
automatically from `Akahsizrr/boosted-v1-small` on Hugging Face when you
select the model — no manual setup required.

## License

MIT
