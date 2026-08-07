# primeILtasks — IL (Intuition Learning) Tasksets for prime-rl

Handcrafted tasksets for the **Intuition Learning** pipeline, built in the
`verifiers.v1` format so they run with the `eval` CLI and produce `traces.jsonl`
that `prime-rl` / GRPO consumes.

## What is IL?

IL (Intuition Learning) is an RL-like pipeline (tested on DeepSeek-R1-Distill-Qwen-1.5B
via mlx_lm and on Kaggle T4 via PyTorch) that uses **verifiable rewards** to shape both
WHAT a model computes and HOW it reasons. The core innovation is **efficiency-aware
reward shaping**:

```
final = correctness * (0.6 + 0.4 * reasoning_quality)
```

- Wrong answers always get 0 (no credit for efficient wrong answers)
- Right answers with lazy reasoning get 0.6 * correctness
- Right answers with thorough, verified reasoning get up to 1.0 * correctness
- The 0.4 spread is the RL signal that shapes reasoning behavior

**Reasoning quality** (4 dimensions):
- **Coverage** (40%) — did the model reason about the right concepts?
- **Token efficiency** (30%) — within a token budget?
- **Verification** (20%) — did the model check its work?
- **No-filler** (10%) — avoided generic boilerplate?

**Anti-laziness**: degenerate solutions (empty bodies, constant returns, ignored
params, no-op fixes) get penalized up to 80% on correctness.

All tasks use the `<reasoning>...</reasoning><answer>...</answer>` format so the
model must show its work before producing an answer.

## Tasksets

| Taskset | Domain | Tasks | Sandbox | Description |
|---------|--------|-------|---------|-------------|
| `il-coding-v1` | coding | 12 | yes | Algorithm implementation, debugging, refactoring, edge-case handling |
| `il-reasoning-v1` | reasoning | 12 | no | Logic puzzles, constraint satisfaction, invariants, type inference |
| `il-agentic-reasoning-v1` | agentic reasoning | 10 | no | Multi-step cascading deduction, cross-module traces, state machines |
| `il-agentic-coding-v1` | agentic coding | 10 | yes | Multi-file codebase navigation, cascading bugs, refactoring, test-writing |

## Quick start

Install from the repo root (editable, local):

```bash
# install all four
uv pip install -e primeILtasks/il_coding_v1
uv pip install -e primeILtasks/il_reasoning_v1
uv pip install -e primeILtasks/il_agentic_reasoning_v1
uv pip install -e primeILtasks/il_agentic_coding_v1
```

Smoke test (3 tasks, 1 rollout, plain logs):

```bash
uv run eval il-coding-v1 -n 3 -r 1 --rich false -v --no-push
uv run eval il-reasoning-v1 -n 3 -r 1 --rich false -v --no-push
uv run eval il-agentic-reasoning-v1 -n 3 -r 1 --rich false -v --no-push
uv run eval il-agentic-coding-v1 -n 3 -r 1 --rich false -v --no-push
```

Full eval (for RL-grade trace generation):

```bash
uv run eval il-coding-v1 -r 4 --rich false
uv run eval il-reasoning-v1 --harness.id null -r 4 --rich false
uv run eval il-agentic-reasoning-v1 --harness.id null -r 4 --rich false
uv run eval il-agentic-coding-v1 -r 4 --rich false
```

Each run saves to `outputs/<taskset>--<model>--<harness>/<uuid>/traces.jsonl` —
the data `prime-rl` consumes for GRPO training.

## Design principles (mechanize.work-style)

1. **Handcrafted over procedural** — each task is a unique scenario, not a
   procedurally generated trivial variant
2. **Quality over quantity** — rich, informative reward signals (not just pass/fail)
3. **Real codebases** — multi-file Python projects with realistic structure
4. **Distractor code** — irrelevant code the model must skip (punishes laziness)
5. **Partial credit** — rewards careful partial work via reasoning quality
6. **Edge cases** — tests that punish superficial pattern-matching
7. **Teacher traces** — the expected-concepts list shapes what thorough reasoning covers
8. **Difficulty scaling** — easy/medium/hard with token budgets tuned per task

## Layout

```
primeILtasks/
├── README.md                          (this file)
├── il_coding_v1/                      12 handcrafted coding tasks (sandboxed)
│   ├── pyproject.toml
│   ├── README.md
│   └── il_coding_v1/
│       ├── __init__.py
│       ├── taskset.py                 verifiers.v1 Taskset
│       ├── tasks.py                   12 handcrafted CodingTask definitions
│       ├── scoring.py                 anti-laziness + reasoning-quality shaping
│       └── verify.py                  sandboxed test runner
├── il_reasoning_v1/                   12 handcrafted reasoning tasks (no sandbox)
│   ├── pyproject.toml
│   ├── README.md
│   └── il_reasoning_v1/
│       ├── __init__.py
│       ├── taskset.py
│       ├── tasks.py                   12 handcrafted ReasoningTask definitions
│       └── scoring.py                 reasoning-quality shaping
├── il_agentic_reasoning_v1/           10 multi-step reasoning tasks (no sandbox)
│   ├── pyproject.toml
│   ├── README.md
│   └── il_agentic_reasoning_v1/
│       ├── __init__.py
│       ├── taskset.py
│       ├── tasks.py                   10 handcrafted AgenticReasoningTask defs
│       └── scoring.py
└── il_agentic_coding_v1/              10 multi-file codebase tasks (sandboxed)
    ├── pyproject.toml
    ├── README.md
    └── il_agentic_coding_v1/
        ├── __init__.py
        ├── taskset.py
        ├── tasks.py                   10 handcrafted AgenticCodingTask defs
        ├── scoring.py                 anti-laziness + reasoning-quality shaping
        └── verify.py                  multi-file sandboxed test runner
```
