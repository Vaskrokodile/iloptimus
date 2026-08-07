"""Real grader — bypasses verifiers.v1 framework, calls scoring functions directly.

For each taskset type, this module:
1. Loads the task definitions (TASKS list) from the taskset package
2. Builds the prompt for each task (same format as the taskset.py INSTRUCTION)
3. Grades a model response using the same scoring logic (correctness + reasoning quality)
4. Runs sandbox verify.py scripts directly via subprocess (no runtime abstraction)

This is the bridge between the web pipeline runner and the IL taskset scoring logic.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Module loading (bypasses __init__.py which imports verifiers)
# ---------------------------------------------------------------------------

_LOADED: dict[str, object] = {}


def _load_module(name: str, path: str):
    """Load a Python module from a file path, bypassing package __init__.py."""
    cache_key = f"{name}:{path}"
    if cache_key in _LOADED:
        return _LOADED[cache_key]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _LOADED[cache_key] = mod
    return mod


# Base paths
_TASKSETS_DIR = Path(__file__).parent.parent.parent / "il_coding_v1"
# Actually, the tasksets are siblings of the iloptimus package
_REPO_ROOT = Path(__file__).parent.parent.parent  # primeILtasks/


def _taskset_path(taskset_id: str, filename: str) -> Path:
    """Get path to a file within a taskset package."""
    # taskset_id like "il-coding-v1" -> package dir "il_coding_v1/il_coding_v1"
    pkg_dir = _REPO_ROOT / taskset_id.replace("-", "_") / taskset_id.replace("-", "_")
    return pkg_dir / filename


# ---------------------------------------------------------------------------
# Prompt builders (mirror the INSTRUCTION strings from each taskset.py)
# ---------------------------------------------------------------------------

_CODING_INSTRUCTION = (
    "Solve the coding task below. First, reason through the problem inside "
    "<reasoning>...</reasoning> tags — explain your approach, trace edge cases, "
    "and verify your solution mentally. Then provide your code inside "
    "<answer>```python\n...\n```</answer> tags.\n\n"
    "Your reasoning quality affects your score: be thorough but concise, cover "
    "the key concepts, and verify your work. Generic filler lowers your score.\n\n"
)

_REASONING_INSTRUCTION = (
    "Solve the reasoning puzzle below. First, work through it inside "
    "<reasoning>...</reasoning> tags — show your deduction step by step, "
    "check your answer, and avoid generic filler. Then give your final answer "
    "inside <answer>...</answer> tags.\n\n"
    "Your reasoning quality affects your score: be thorough but concise, cover "
    "the key concepts, and verify your work.\n\n"
)

_AGENTIC_REASONING_INSTRUCTION = (
    "Solve the multi-step reasoning task below. This requires SUSTAINED "
    "reasoning — trace through each stage carefully, as later steps depend on "
    "earlier ones. Work inside <reasoning>...</reasoning> tags, showing each "
    "step and verifying your answer. Then give your final answer inside "
    "<answer>...</answer> tags.\n\n"
    "Your reasoning quality affects your score: cover the key concepts, stay "
    "within budget, verify your work, and avoid generic filler.\n\n"
)

_AGENTIC_CODING_INSTRUCTION = (
    "You are given a multi-file codebase with a bug, missing feature, or "
    "refactoring task. First, reason through the code inside "
    "<reasoning>...</reasoning> tags — trace the call chain, identify the "
    "root cause, and verify your fix mentally. Then provide your fixed file(s) "
    "as code blocks tagged with the filename, like:\n"
    "```python:filename.py\n...your fixed code...\n```\n\n"
    "Your reasoning quality affects your score: be thorough but concise, cover "
    "the key concepts, and verify your work. Generic filler lowers your score.\n\n"
)

CODE_BLOCK_RE = re.compile(r"```python:([^\n]+)\n(.*?)```", re.DOTALL)


# ---------------------------------------------------------------------------
# Graded result
# ---------------------------------------------------------------------------

@dataclass
class GradedResult:
    score: float  # final IL score (0.0 to 1.0)
    correctness: float  # raw correctness (0.0 to 1.0)
    reasoning_quality: float  # reasoning quality (0.0 to 1.0)
    coverage: float = 0.0
    verification: float = 0.0
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sandbox runner (replaces verifiers Runtime)
# ---------------------------------------------------------------------------

def _run_sandbox(script_path: str, payload: dict, timeout: float) -> dict:
    """Run a verify.py script in a subprocess with a JSON payload.

    Writes payload to a temp file, runs the script, reads JSON output.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir="/tmp"
    ) as f:
        json.dump(payload, f)
        payload_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path, payload_path, str(timeout)],
            capture_output=True,
            text=True,
            timeout=timeout + 5,  # extra margin for script overhead
        )
        out = (result.stdout or "").strip()
        lines = out.splitlines()
        if lines:
            try:
                return json.loads(lines[-1])
            except json.JSONDecodeError:
                pass
        return {"error": (result.stderr or "")[-500:]}
    except subprocess.TimeoutExpired:
        return {"error": "TIMEOUT"}
    finally:
        # verify.py deletes the payload itself, but clean up just in case
        if os.path.exists(payload_path):
            try:
                os.unlink(payload_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Taskset graders
# ---------------------------------------------------------------------------

def grade_coding(response: str, task_idx: int) -> GradedResult:
    """Grade a coding task response."""
    tasks_mod = _load_module(
        "il_coding_tasks",
        str(_taskset_path("il_coding_v1", "tasks.py")),
    )
    scoring_mod = _load_module(
        "il_coding_scoring",
        str(_taskset_path("il_coding_v1", "scoring.py")),
    )
    verify_script = str(_taskset_path("il_coding_v1", "verify.py"))

    task = tasks_mod.TASKS[task_idx]
    code = scoring_mod.extract_code(response)
    if not code.strip():
        return GradedResult(score=0.0, correctness=0.0, reasoning_quality=0.0)

    # Run hidden tests in sandbox
    test_result = _run_sandbox(
        verify_script,
        {"code": code, "tests": task.tests},
        timeout=5.0,
    )
    correctness = test_result.get("pass_rate", 0.0)

    # Anti-laziness penalty
    laziness = scoring_mod.detect_laziness(
        code, task.required_params, task.required_constructs
    )
    if laziness.score > 0:
        correctness *= max(0.2, 1.0 - laziness.score * 0.8)

    # Reasoning quality shaping
    rq, breakdown = scoring_mod.score_reasoning_quality(
        response, task.expected_concepts, task.token_budget, correctness
    )
    final = scoring_mod.compute_final_score(correctness, rq)

    return GradedResult(
        score=final,
        correctness=correctness,
        reasoning_quality=rq,
        coverage=breakdown.coverage,
        verification=breakdown.verification,
        info={
            "laziness_reasons": laziness.reasons,
            "test_result": test_result,
        },
    )


def grade_reasoning(response: str, task_idx: int) -> GradedResult:
    """Grade a reasoning task response (no sandbox needed)."""
    tasks_mod = _load_module(
        "il_reasoning_tasks",
        str(_taskset_path("il_reasoning_v1", "tasks.py")),
    )
    scoring_mod = _load_module(
        "il_reasoning_scoring",
        str(_taskset_path("il_reasoning_v1", "scoring.py")),
    )

    task = tasks_mod.TASKS[task_idx]
    answer = tasks_mod._extract_answer_text(response)
    correct, info = task.verify(answer)
    correctness = 1.0 if correct else 0.0

    rq, breakdown = scoring_mod.score_reasoning_quality(
        response, task.expected_concepts, task.token_budget, correctness
    )
    final = scoring_mod.compute_final_score(correctness, rq)

    return GradedResult(
        score=final,
        correctness=correctness,
        reasoning_quality=rq,
        coverage=breakdown.coverage,
        verification=breakdown.verification,
        info={"verify_info": info},
    )


def grade_agentic_reasoning(response: str, task_idx: int) -> GradedResult:
    """Grade an agentic reasoning task response (no sandbox needed)."""
    tasks_mod = _load_module(
        "il_agentic_reasoning_tasks",
        str(_taskset_path("il_agentic_reasoning_v1", "tasks.py")),
    )
    scoring_mod = _load_module(
        "il_agentic_reasoning_scoring",
        str(_taskset_path("il_agentic_reasoning_v1", "scoring.py")),
    )

    task = tasks_mod.TASKS[task_idx]
    answer = tasks_mod._extract_answer_text(response)
    correct, info = task.verify(answer)
    correctness = 1.0 if correct else 0.0

    rq, breakdown = scoring_mod.score_reasoning_quality(
        response, task.expected_concepts, task.token_budget, correctness
    )
    final = scoring_mod.compute_final_score(correctness, rq)

    return GradedResult(
        score=final,
        correctness=correctness,
        reasoning_quality=rq,
        coverage=breakdown.coverage,
        verification=breakdown.verification,
        info={"verify_info": info},
    )


def grade_agentic_coding(response: str, task_idx: int) -> GradedResult:
    """Grade an agentic coding task response."""
    tasks_mod = _load_module(
        "il_agentic_coding_tasks",
        str(_taskset_path("il_agentic_coding_v1", "tasks.py")),
    )
    scoring_mod = _load_module(
        "il_agentic_coding_scoring",
        str(_taskset_path("il_agentic_coding_v1", "scoring.py")),
    )
    verify_script = str(_taskset_path("il_agentic_coding_v1", "verify.py"))

    task = tasks_mod.TASKS[task_idx]

    # Extract code blocks with filenames
    fixes: dict[str, str] = {}
    for match in CODE_BLOCK_RE.finditer(response):
        fname = match.group(1).strip()
        code = match.group(2)
        fixes[fname] = code

    if not fixes:
        return GradedResult(score=0.0, correctness=0.0, reasoning_quality=0.0)

    # Merge fixes into codebase
    merged = dict(task.codebase)
    merged.update(fixes)

    # Anti-laziness: check target files were actually changed
    laziness_reasons: list[str] = []
    for fname, code in fixes.items():
        if fname in task.target_files:
            if code.strip() == task.codebase.get(fname, "").strip():
                laziness_reasons.append(f"{fname}: unchanged (no-op fix)")
            elif not code.strip():
                laziness_reasons.append(f"{fname}: empty fix")

    # Run test harness in sandbox
    test_result = _run_sandbox(
        verify_script,
        {"files": merged, "harness": task.test_harness},
        timeout=10.0,
    )
    correctness = 1.0 if test_result.get("passed") else 0.0

    # Anti-laziness penalty
    if laziness_reasons:
        correctness *= 0.5

    # Structural laziness on changed files
    for fname, code in fixes.items():
        laz = scoring_mod.detect_laziness(code, [], [])
        if laz.score > 0:
            correctness *= max(0.2, 1.0 - laz.score * 0.5)
            laziness_reasons.extend(laz.reasons)

    # Reasoning quality shaping
    rq, breakdown = scoring_mod.score_reasoning_quality(
        response, task.expected_concepts, task.token_budget, correctness
    )
    final = scoring_mod.compute_final_score(correctness, rq)

    return GradedResult(
        score=final,
        correctness=correctness,
        reasoning_quality=rq,
        coverage=breakdown.coverage,
        verification=breakdown.verification,
        info={
            "laziness_reasons": laziness_reasons,
            "test_output": test_result.get("output", "")[-500:],
        },
    )


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

# Map taskset domain -> (grader function, prompt builder, instruction string)
_GRADERS = {
    "coding": grade_coding,
    "reasoning": grade_reasoning,
    "agentic-reasoning": grade_agentic_reasoning,
    "agentic-coding": grade_agentic_coding,
}

_INSTRUCTIONS = {
    "coding": _CODING_INSTRUCTION,
    "reasoning": _REASONING_INSTRUCTION,
    "agentic-reasoning": _AGENTIC_REASONING_INSTRUCTION,
    "agentic-coding": _AGENTIC_CODING_INSTRUCTION,
}


def grade_response(domain: str, task_idx: int, response: str) -> GradedResult:
    """Grade a model response for a given taskset domain and task index."""
    grader = _GRADERS.get(domain)
    if not grader:
        raise ValueError(f"Unknown domain: {domain}")
    return grader(response, task_idx)


def build_prompt(domain: str, task_idx: int) -> str:
    """Build the prompt for a given taskset domain and task index."""
    instruction = _INSTRUCTIONS.get(domain, "")
    pkg_name = domain.replace("-", "_")

    if domain == "coding":
        tasks_mod = _load_module(
            "il_coding_tasks",
            str(_taskset_path("il_coding_v1", "tasks.py")),
        )
        task = tasks_mod.TASKS[task_idx]
        return instruction + f"## Task: {task.name}\n\n{task.spec}\n\nSignature: `{task.signature}`"

    elif domain == "reasoning":
        tasks_mod = _load_module(
            "il_reasoning_tasks",
            str(_taskset_path("il_reasoning_v1", "tasks.py")),
        )
        task = tasks_mod.TASKS[task_idx]
        return instruction + f"## Task: {task.name}\n\n{task.spec}\n\nAnswer format: {task.answer_format}."

    elif domain == "agentic-reasoning":
        tasks_mod = _load_module(
            "il_agentic_reasoning_tasks",
            str(_taskset_path("il_agentic_reasoning_v1", "tasks.py")),
        )
        task = tasks_mod.TASKS[task_idx]
        return instruction + f"## Task: {task.name}\n\n{task.spec}\n\nAnswer format: {task.answer_format}."

    elif domain == "agentic-coding":
        tasks_mod = _load_module(
            "il_agentic_coding_tasks",
            str(_taskset_path("il_agentic_coding_v1", "tasks.py")),
        )
        task = tasks_mod.TASKS[task_idx]
        codebase_str = "\n\n".join(
            f"### {fname}\n```python\n{content}```"
            for fname, content in task.codebase.items()
        )
        return instruction + f"## Task: {task.name}\n\n{task.spec}\n\n## Codebase\n\n{codebase_str}"

    raise ValueError(f"Unknown domain: {domain}")


def get_num_tasks(domain: str) -> int:
    """Get the number of tasks for a given domain."""
    pkg_map = {
        "coding": ("il_coding_tasks", "il_coding_v1", "tasks.py"),
        "reasoning": ("il_reasoning_tasks", "il_reasoning_v1", "tasks.py"),
        "agentic-reasoning": ("il_agentic_reasoning_tasks", "il_agentic_reasoning_v1", "tasks.py"),
        "agentic-coding": ("il_agentic_coding_tasks", "il_agentic_coding_v1", "tasks.py"),
    }
    mod_name, pkg_id, filename = pkg_map[domain]
    tasks_mod = _load_module(mod_name, str(_taskset_path(pkg_id, filename)))
    return len(tasks_mod.TASKS)
