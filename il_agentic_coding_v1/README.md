# il-agentic-coding-v1

Handcrafted **multi-file agentic coding** tasks for the IL pipeline — 10 mechanize.work-style
codebase scenarios where the model navigates a mini-codebase, reasons through the call chain,
and produces fixes as ```python:filename``` code blocks.

## What makes this IL

- **Handcrafted multi-file codebases** — real project structure with distractor code, not
  single-function puzzles
- **Cascading bugs** — bugs stacked in sequence where each is only visible after the previous
  is fixed (the core IL failure mode)
- **Anti-laziness** — no-op fixes, empty fixes, and degenerate solutions are penalized
- **Efficiency-aware reward shaping**:

      final = correctness * (0.6 + 0.4 * reasoning_quality)

- **Reasoning quality** (4 dimensions): coverage (40%) + token efficiency (30%) +
  verification (20%) + no-filler (10%)
- **`<reasoning>...</reasoning>` + ```python:filename``` format** — the model must trace the
  codebase, then produce targeted fixes

## Tasks

| # | Name | Skill |
|---|------|-------|
| 1 | cascading_bug_chain | Multi-step cascading bug fixing across 3 modules |
| 2 | codebase_nav_bug | Tracing a call chain across 3 files to find a data-layer bug |
| 3 | refactor_preserve_behavior | Refactoring if/elif to dispatch dict, preserving behavior |
| 4 | missing_error_handling | Adding None/empty-list handling to a list processor |
| 5 | api_client_impl | Implementing a missing method per spec |
| 6 | dead_code_and_logic_error | Removing dead code + fixing a boundary logic error |
| 7 | type_annotation_add | Adding complete type annotations without changing behavior |
| 8 | perf_optimize_list | Rewriting O(n^2) duplicate finder to O(n) with a set |
| 9 | config_fix_multi | Fixing 3 config bugs (type, value, filtering) |
| 10 | test_writing_catch_mutant | Writing tests that catch a specific mutant |

## Taskset

- **Source:** Handcrafted — 10 tasks in `tasks.py`
- **Size:** 10 tasks
- **Prompting:** zero-shot with IL reasoning + code-block instructions
- **Reward:** `correctness * (0.6 + 0.4 * reasoning_quality)` where correctness is
  sandboxed test-harness pass with anti-laziness penalty

Scoring merges the model's ```python:filename``` blocks into the codebase, writes all
files to a temp dir, and runs a hidden test harness that prints `ALL_PASS` on success.

## Run

```bash
uv pip install -e primeILtasks/il_agentic_coding_v1
uv run eval il-agentic-coding-v1 -n 3 -r 1 --rich false -v --no-push
```

Full eval:

```bash
uv run eval il-agentic-coding-v1 -r 4 --rich false
```

## Security

Model-generated Python executes inside the selected Verifiers runtime. Use a Docker or
Prime runtime when evaluating models you do not trust.
