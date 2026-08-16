"""humaneval-v1 — HumanEval coding benchmark with IL efficiency-aware reward shaping.

A `verifiers.v1` taskset of 25 curated HumanEval coding problems scored by
sandboxed test execution with anti-laziness detection, then shaped by reasoning
quality:

    final = correctness * (0.6 + 0.4 * reasoning_quality)

The model responds in <reasoning>...</reasoning><answer>```python\n...\n```</answer>
format. The reward extracts the code, runs hidden tests in a spawned sandbox
worker (tests never exposed to the candidate), applies an anti-laziness penalty,
then multiplies by reasoning quality.

This taskset is used by the self-improvement loop to benchmark and train the
DeepSeek-R1-Distill-Qwen-1.5B model on real coding tasks.
"""

from __future__ import annotations

import json
from pathlib import Path

import verifiers.v1 as vf

from .scoring import (
    compute_final_score,
    detect_laziness,
    extract_code,
    score_reasoning_quality,
)
from .tasks import TASKS, HumanEvalTask

TIMEOUT = 5.0
VERIFY = (Path(__file__).parent / "verify.py").read_bytes()

INSTRUCTION = (
    "Solve the coding task below. First, reason through the problem inside "
    "<reasoning>...</reasoning> tags — explain your approach, trace edge cases, "
    "and verify your solution mentally. Then provide your code inside "
    "<answer>```python\\n...\\n```</answer> tags.\n\n"
    "Your reasoning quality affects your score: be thorough but concise, cover "
    "the key concepts, and verify your work. Generic filler lowers your score.\n\n"
)


class HumanEvalTaskData(vf.TaskData):
    spec: str
    signature: str
    tests: list[str]
    entry_point: str
    expected_concepts: list[str]
    token_budget: int
    required_params: list[str]
    required_constructs: list[str]


class HumanEvalTaskConfig(vf.TaskConfig):
    timeout: float = TIMEOUT


class HumanEvalCodingTask(vf.Task[HumanEvalTaskData, vf.State, HumanEvalTaskConfig]):
    @vf.reward(weight=1.0)
    async def scored(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        response = trace.last_reply or ""
        code = extract_code(response)
        if not code.strip():
            trace.info["correctness"] = 0.0
            trace.info["reasoning_quality"] = 0.0
            return 0.0

        # Run hidden tests in sandbox
        payload_path = f"/tmp/humaneval/{trace.id}.json"
        await runtime.write(
            payload_path,
            json.dumps({
                "code": code,
                "tests": self.data.tests,
                "entry_point": self.data.entry_point,
            }).encode(),
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
        correctness = test_result.get("pass_rate", 0.0)

        # Anti-laziness penalty on correctness
        laziness = detect_laziness(
            code, self.data.required_params, self.data.required_constructs
        )
        if laziness.score > 0:
            correctness *= max(0.2, 1.0 - laziness.score * 0.8)
            trace.info["laziness_reasons"] = laziness.reasons

        # Reasoning quality shaping
        reasoning_quality, breakdown = score_reasoning_quality(
            response, self.data.expected_concepts, self.data.token_budget, correctness
        )
        final = compute_final_score(correctness, reasoning_quality)

        trace.info["correctness"] = correctness
        trace.info["reasoning_quality"] = reasoning_quality
        trace.info["coverage"] = breakdown.coverage
        trace.info["verification"] = breakdown.verification
        trace.info["token_efficiency"] = breakdown.token_efficiency
        trace.info["final_score"] = final
        return final


class HumanEvalConfig(vf.TasksetConfig):
    task: HumanEvalTaskConfig = HumanEvalTaskConfig()


class HumanEvalTaskset(vf.Taskset[HumanEvalCodingTask, HumanEvalConfig]):
    def load(self) -> list[HumanEvalCodingTask]:
        return [
            HumanEvalCodingTask(
                HumanEvalTaskData(
                    idx=t.idx,
                    name=t.name,
                    prompt=INSTRUCTION + f"## Task: {t.name}\n\n{t.spec}\n\n"
                    f"Signature: `{t.signature}`",
                    spec=t.spec,
                    signature=t.signature,
                    tests=t.tests,
                    entry_point=t.entry_point,
                    expected_concepts=t.expected_concepts,
                    token_budget=t.token_budget,
                    required_params=t.required_params,
                    required_constructs=t.required_constructs,
                ),
                self.config.task,
            )
            for t in TASKS
        ]


__all__ = ["HumanEvalTaskData", "HumanEvalCodingTask", "HumanEvalTaskset", "HumanEvalConfig"]
