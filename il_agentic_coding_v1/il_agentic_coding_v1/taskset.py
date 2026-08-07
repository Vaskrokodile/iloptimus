"""il-agentic-coding-v1 — handcrafted multi-file codebase tasks with IL reward shaping.

A `verifiers.v1` taskset of 10 handcrafted multi-file codebase scenarios
(mechanize.work-style): cascading bug chains, codebase navigation, refactoring,
error handling, API client implementation, dead-code + logic-error removal,
type annotation, performance optimization, config fixes, and test-writing to
catch mutants.

The model reads a mini-codebase, reasons through it in <reasoning>...</reasoning>
tags, and produces fixes as ```python:filename``` code blocks. The grader merges
the fixes into the codebase, runs a hidden test harness in a sandbox, and scores
with anti-laziness + efficiency-aware shaping:

    final = correctness * (0.6 + 0.4 * reasoning_quality)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import verifiers.v1 as vf

from .scoring import (
    compute_final_score,
    detect_laziness,
    score_reasoning_quality,
)
from .tasks import TASKS, AgenticCodingTask

TIMEOUT = 10.0
VERIFY = (Path(__file__).parent / "verify.py").read_bytes()

# Matches ```python:filename\n...code...\n```
CODE_BLOCK_RE = re.compile(r"```python:([^\n]+)\n(.*?)```", re.DOTALL)

INSTRUCTION = (
    "You are given a multi-file codebase with a bug, missing feature, or "
    "refactoring task. First, reason through the code inside "
    "<reasoning>...</reasoning> tags — trace the call chain, identify the "
    "root cause, and verify your fix mentally. Then provide your fixed file(s) "
    "as code blocks tagged with the filename, like:\n"
    "```python:filename.py\\n...your fixed code...\\n```\n\n"
    "Your reasoning quality affects your score: be thorough but concise, cover "
    "the key concepts, and verify your work. Generic filler lowers your score.\n\n"
)


def parse_code_blocks(response: str) -> dict[str, str]:
    """Extract ```python:filename``` blocks from the response."""
    blocks: dict[str, str] = {}
    for match in CODE_BLOCK_RE.finditer(response):
        fname = match.group(1).strip()
        code = match.group(2)
        blocks[fname] = code
    return blocks


class ILAgenticCodingData(vf.TaskData):
    spec: str
    codebase: dict[str, str]
    test_harness: str
    expected_concepts: list[str]
    token_budget: int
    target_files: list[str]


class ILAgenticCodingTaskConfig(vf.TaskConfig):
    timeout: float = TIMEOUT


class ILAgenticCodingTask(vf.Task[ILAgenticCodingData, vf.State, ILAgenticCodingTaskConfig]):
    @vf.reward(weight=1.0)
    async def scored(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        response = trace.last_reply or ""
        fixes = parse_code_blocks(response)

        if not fixes:
            trace.info["correctness"] = 0.0
            trace.info["reasoning_quality"] = 0.0
            trace.info["no_code_blocks"] = True
            return 0.0

        # Merge fixes into the codebase
        merged = dict(self.data.codebase)
        merged.update(fixes)

        # Anti-laziness: check that target files were actually changed
        changed_files = [f for f in fixes if f in self.data.codebase and fixes[f] != self.data.codebase[f]]
        laziness_reasons: list[str] = []
        for fname, code in fixes.items():
            if fname in self.data.target_files:
                # Check the fix isn't a no-op or degenerate
                if code.strip() == self.data.codebase.get(fname, "").strip():
                    laziness_reasons.append(f"{fname}: unchanged (no-op fix)")
                elif not code.strip():
                    laziness_reasons.append(f"{fname}: empty fix")

        # Run the test harness in sandbox
        payload_path = f"/tmp/il_agentic_coding/{trace.id}.json"
        await runtime.write(
            payload_path,
            json.dumps({"files": merged, "harness": self.data.test_harness}).encode(),
        )
        result = await runtime.run_uv_script(
            VERIFY, args=[payload_path, str(self.config.timeout)]
        )
        if result.exit_code != 0:
            trace.info["verify_error"] = result.stderr.strip()[-1000:]
            trace.info["correctness"] = 0.0
            return 0.0

        out_line = result.stdout.strip().splitlines()[-1]
        test_result = json.loads(out_line)
        correctness = 1.0 if test_result.get("passed") else 0.0
        trace.info["test_output"] = test_result.get("output", "")[-1000:]

        # Anti-laziness penalty
        if laziness_reasons:
            correctness *= 0.5
            trace.info["laziness_reasons"] = laziness_reasons

        # Also run structural laziness on the changed files
        for fname, code in fixes.items():
            laz = detect_laziness(code, [], [])
            if laz.score > 0:
                correctness *= max(0.2, 1.0 - laz.score * 0.5)
                trace.info.setdefault("laziness_reasons", []).extend(laz.reasons)

        # Reasoning quality shaping
        reasoning_quality, breakdown = score_reasoning_quality(
            response, self.data.expected_concepts, self.data.token_budget, correctness
        )
        final = compute_final_score(correctness, reasoning_quality)

        trace.info["correctness"] = correctness
        trace.info["reasoning_quality"] = reasoning_quality
        trace.info["coverage"] = breakdown.coverage
        trace.info["verification"] = breakdown.verification
        trace.info["changed_files"] = changed_files
        trace.info["final_score"] = final
        return final


class ILAgenticCodingConfig(vf.TasksetConfig):
    task: ILAgenticCodingTaskConfig = ILAgenticCodingTaskConfig()


class ILAgenticCodingTaskset(vf.Taskset[ILAgenticCodingTask, ILAgenticCodingConfig]):
    def load(self) -> list[ILAgenticCodingTask]:
        return [
            ILAgenticCodingTask(
                ILAgenticCodingData(
                    idx=t.idx,
                    name=t.name,
                    prompt=INSTRUCTION + f"## Task: {t.name}\n\n{t.spec}\n\n"
                    "## Codebase\n\n"
                    + "\n\n".join(
                        f"### {fname}\n```python\n{content}```"
                        for fname, content in t.codebase.items()
                    ),
                    spec=t.spec,
                    codebase=t.codebase,
                    test_harness=t.test_harness,
                    expected_concepts=t.expected_concepts,
                    token_budget=t.token_budget,
                    target_files=t.target_files,
                ),
                self.config.task,
            )
            for t in TASKS
        ]


__all__ = [
    "ILAgenticCodingData",
    "ILAgenticCodingTask",
    "ILAgenticCodingTaskset",
    "ILAgenticCodingConfig",
]
