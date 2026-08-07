# il-coding-v1

Handcrafted coding tasks for the **IL (Intuition Learning)** pipeline — 12 single-turn
coding scenarios scored by sandboxed test execution with anti-laziness detection and
efficiency-aware reasoning-quality shaping.

## What makes this IL

This taskset encodes the [ilresearch](https://github.com/PrimeIntellect-ai) Intuition
Learning philosophy:

- **Handcrafted, not procedural** — 12 unique scenarios (merge sorted lists, LRU cache,
  off-by-one fix, binary search, paren validation, word frequency, matrix rotation,
  dead code elimination, sliding window max, edge-case division, graph BFS, API migration)
- **Anti-laziness** — degenerate solutions (empty bodies, constant returns, ignored
  params, missing required constructs) get penalized up to 80% on correctness
- **Efficiency-aware reward shaping**:

      final = correctness * (0.6 + 0.4 * reasoning_quality)

  Wrong answers always get 0. Right answers with lazy reasoning get 0.6. Right answers
  with thorough, verified reasoning get up to 1.0. The 0.4 spread is the RL signal.
- **Reasoning quality** (4 dimensions): coverage (40%) + token efficiency (30%) +
  verification (20%) + no-filler (10%)
- **`<reasoning>...</reasoning><answer>```python\n...\n```</answer>` format** — the
  model must show its work, then produce code

## Taskset

- **Source:** Handcrafted — 12 tasks in `tasks.py`
- **Size:** 12 tasks (run with `-r 4` or higher for RL-grade variance)
- **Prompting:** zero-shot with IL reasoning instructions
- **Reward:** `correctness * (0.6 + 0.4 * reasoning_quality)` where correctness is
  sandboxed test pass-rate with anti-laziness penalty

Scoring runs hidden tests in a spawned worker that never sees the test source.

## Run

```bash
uv pip install -e primeILtasks/il_coding_v1
uv run eval il-coding-v1 -n 3 -r 1 --rich false -v --no-push
```

Full eval:

```bash
uv run eval il-coding-v1 -r 4 --rich false
```

## Security

Model-generated Python executes inside the selected Verifiers runtime. Use a Docker or
Prime runtime when evaluating models you do not trust.
