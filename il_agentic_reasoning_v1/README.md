# il-agentic-reasoning-v1

Handcrafted **multi-step agentic reasoning** tasks for the IL pipeline — 10 long-horizon
scenarios requiring sustained deduction where each step depends on the previous.

## What makes this IL

These target the core IL failure mode: **small models solve the first step and miss the
cascade.** The 0.4 reasoning-quality spread shapes sustained, verified multi-step reasoning.

- **Handcrafted scenarios** — cascading pipeline traces, cross-module data flow (5 functions),
  invariant preservation, race-condition interleaving counts, API contract compliance (10
  constraints), recursive repair traces, state machine simulation, differential analysis,
  error propagation chains, coverage gap analysis
- **Deterministic verification** — answers precomputed and checked in Python
- **Efficiency-aware reward shaping**:

      final = correctness * (0.6 + 0.4 * reasoning_quality)

- **Reasoning quality** (4 dimensions): coverage (40%) + token efficiency (30%) +
  verification (20%) + no-filler (10%)
- **`<reasoning>...</reasoning><answer>...</answer>` format**
- **No sandbox** — runs under any harness including `null`

## Taskset

- **Source:** Handcrafted — 10 tasks in `tasks.py`
- **Size:** 10 tasks
- **Prompting:** zero-shot with IL sustained-reasoning instructions
- **Reward:** `correctness * (0.6 + 0.4 * reasoning_quality)` (binary correctness)

## Run

```bash
uv pip install -e primeILtasks/il_agentic_reasoning_v1
uv run eval il-agentic-reasoning-v1 -n 3 -r 1 --rich false -v --no-push
```

Full eval:

```bash
uv run eval il-agentic-reasoning-v1 --harness.id null -r 4 --rich false
```
