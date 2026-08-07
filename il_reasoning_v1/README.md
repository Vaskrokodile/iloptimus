# il-reasoning-v1

Handcrafted pure-reasoning tasks for the **IL (Intuition Learning)** pipeline — 12
self-contained logic/reasoning puzzles scored by deterministic verification with
efficiency-aware reasoning-quality shaping. No sandbox needed.

## What makes this IL

- **Handcrafted puzzles** — knights & knaves, constraint scheduling, loop invariants,
  type inference, grid path counting, zebra logic, recursive traces, set operations,
  probability, off-by-one reasoning, graph cycles, combinatorial counting
- **Deterministic verification** — answers checked in Python (int/set/str/list matchers)
- **Efficiency-aware reward shaping**:

      final = correctness * (0.6 + 0.4 * reasoning_quality)

- **Reasoning quality** (4 dimensions): coverage (40%) + token efficiency (30%) +
  verification (20%) + no-filler (10%)
- **`<reasoning>...</reasoning><answer>...</answer>` format**

## Taskset

- **Source:** Handcrafted — 12 tasks in `tasks.py`
- **Size:** 12 tasks
- **Prompting:** zero-shot with IL reasoning instructions
- **Reward:** `correctness * (0.6 + 0.4 * reasoning_quality)` (binary correctness)
- **No sandbox** — runs under any harness including `null`

## Run

```bash
uv pip install -e primeILtasks/il_reasoning_v1
uv run eval il-reasoning-v1 -n 3 -r 1 --rich false -v --no-push
```

Full eval:

```bash
uv run eval il-reasoning-v1 --harness.id null -r 4 --rich false
```
