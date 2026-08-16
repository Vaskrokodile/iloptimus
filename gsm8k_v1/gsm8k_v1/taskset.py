"""gsm8k-v1 — GSM8K grade-school math benchmark with IL reward shaping.

A `verifiers.v1` taskset of 25 curated GSM8K math word problems covering
arithmetic, multi-step reasoning, unit conversion, percentages, fractions,
rates, geometry, and logic.

No sandbox needed — answers are verified deterministically by extracting the
final number and comparing to the expected value. The reward applies the IL
efficiency-aware shaping:

    final = correctness * (0.6 + 0.4 * reasoning_quality)

The model responds in <reasoning>...</reasoning><answer>...</answer> format.
"""

from __future__ import annotations

import verifiers.v1 as vf

from .scoring import compute_final_score, score_reasoning_quality
from .tasks import TASKS, GSM8KTask, _extract_answer_text

INSTRUCTION = (
    "Solve the math word problem below. First, work through it inside "
    "<reasoning>...</reasoning> tags — show your calculation step by step, "
    "check your answer, and avoid generic filler. Then give your final answer "
    "inside <answer>...</answer> tags.\n\n"
    "Your reasoning quality affects your score: be thorough but concise, show "
    "the key calculations, and verify your work.\n\n"
)


class GSM8KTaskData(vf.TaskData):
    spec: str
    answer_format: str
    verify_fn_id: int
    expected_concepts: list[str]
    token_budget: int


class GSM8KTaskConfig(vf.TaskConfig):
    pass


class GSM8KReasoningTask(vf.Task[GSM8KTaskData, vf.State, GSM8KTaskConfig]):
    @vf.reward(weight=1.0)
    async def scored(self, trace: vf.Trace) -> float:
        response = trace.last_reply or ""
        answer = _extract_answer_text(response)
        task = TASKS[self.data.verify_fn_id]
        correct, info = task.verify(answer)
        correctness = 1.0 if correct else 0.0

        reasoning_quality, breakdown = score_reasoning_quality(
            response, self.data.expected_concepts, self.data.token_budget, correctness
        )
        final = compute_final_score(correctness, reasoning_quality)

        trace.info["correctness"] = correctness
        trace.info["verify_info"] = info
        trace.info["reasoning_quality"] = reasoning_quality
        trace.info["coverage"] = breakdown.coverage
        trace.info["verification"] = breakdown.verification
        trace.info["final_score"] = final
        return final


class GSM8KConfig(vf.TasksetConfig):
    task: GSM8KTaskConfig = GSM8KTaskConfig()


class GSM8KTaskset(vf.Taskset[GSM8KReasoningTask, GSM8KConfig]):
    def load(self) -> list[GSM8KReasoningTask]:
        return [
            GSM8KReasoningTask(
                GSM8KTaskData(
                    idx=t.idx,
                    name=t.name,
                    prompt=INSTRUCTION + f"## Task: {t.name}\n\n{t.spec}\n\n"
                    f"Answer format: {t.answer_format}.",
                    spec=t.spec,
                    answer_format=t.answer_format,
                    verify_fn_id=t.idx,
                    expected_concepts=t.expected_concepts,
                    token_budget=t.token_budget,
                ),
                self.config.task,
            )
            for t in TASKS
        ]


__all__ = ["GSM8KTaskData", "GSM8KReasoningTask", "GSM8KTaskset", "GSM8KConfig"]
