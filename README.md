# IL Optimus — Intuition Learning Pipeline Studio

Run Intuition Learning (SFT + GRPO RL) pipelines locally with a web frontend.
Detects your hardware, recommends compatible models, lets you select tasksets,
and tracks training runs in real time.

## Quick Start

```bash
# Install
pip install -e .

# Or with mlx support on Apple Silicon
pip install -e ".[mlx]"

# Start the server (opens browser automatically)
iloptimus serve

# Or manually
iloptimus serve --host 127.0.0.1 --port 7860 --no-browser
```

Then open `http://127.0.0.1:7860` in your browser.

## What It Does

1. **Detects your hardware** — CPU, RAM, GPU (Apple Silicon / CUDA / None),
   available backends (MLX, vLLM, PyTorch)
2. **Recommends models** — 14 popular models with hardware compatibility scoring
   (recommended / feasible / tight / not-recommended) based on memory requirements
3. **Browse tasksets** — 4 handcrafted IL tasksets (44 tasks total) spanning coding,
   reasoning, agentic reasoning, and agentic coding
4. **Run IL pipelines** — SFT + GRPO RL training with live SSE streaming of logs,
   training curves (loss/reward), and accuracy progression
5. **Track runs** — real-time progress bar, stage pipeline visualization, live log
   stream, and accuracy comparison (baseline → post-SFT → post-GRPO)

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
iloptimus serve              # Start server + open browser
iloptimus hardware           # Print detected hardware info
iloptimus version            # Print version
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
| POST | `/api/runs` | Start a new IL pipeline run |
| GET | `/api/runs` | List all runs |
| GET | `/api/runs/{id}` | Get run state |
| GET | `/api/runs/{id}/events` | SSE stream of run events |

## Backends

- **MLX** (Apple Silicon) — uses `mlx_lm` for inference and LoRA fine-tuning.
  Recommended for M-series Macs with unified memory.
- **vLLM** (CUDA) — uses `vllm` for high-throughput inference on NVIDIA GPUs.
- **CPU** — fallback for systems without GPU acceleration (slow, small models only).

## License

MIT
