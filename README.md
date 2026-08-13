# IL Optimus — Intuition Learning Pipeline Studio

Run Intuition Learning (SFT + GRPO RL) pipelines locally with a web frontend.
Detects your hardware, recommends compatible models, lets you select tasksets,
and tracks training runs in real time.

## Install and Start

> **Requires Apple Silicon (M1/M2/M3/M4)** — the pipeline uses MLX for
> inference and LoRA fine-tuning. CUDA/vLLM support is detected but not yet
> implemented in the training path.

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

Then open `http://127.0.0.1:7860` in your browser.

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
   executes an unadapted baseline, lets the failing local model research through
   audited web tools, builds a licensed provenance-tracked corpus, selects a
   defensible adaptation method, retries, and rejects adapters that do not pass
   every objective gate. See [the TTC design and current experiment](docs/test-time-compute.md).

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
├── mcp.json         explicit MCP server configuration (off by default)
├── performance.json local TPS calibration samples
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
| GET/POST | `/api/environments` | List or create no-code IL/RL environments |
| POST | `/api/environments/from-chat` | Generate an environment with `/il` or `/rl` |
| POST | `/api/runs` | Start a persisted local training run |
| GET | `/api/runs` | List all runs |
| GET | `/api/runs/{id}` | Get run state |
| GET | `/api/runs/{id}/events` | SSE stream of run events |

## Backends

- **MLX** (Apple Silicon) — uses `mlx_lm` for inference and LoRA
  fine-tuning. **This is the only backend currently implemented.**
  Recommended for M-series Macs with unified memory.
- **CUDA / CPU** — hardware is detected and reported, but release 0.2 refuses
  to claim model download or training support on these backends. They remain a
  future backend instead of silently falling back to simulated behavior.

## License

MIT
