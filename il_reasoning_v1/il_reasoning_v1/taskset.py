"""il-reasoning-v1 — handcrafted pure-reasoning tasks with IL reward shaping.

A `verifiers.v1` taskset of 12 handcrafted logic/reasoning puzzles (knights &
knaves, constraint scheduling, loop invariants, type inference, path counting,
zebra logic, recursive traces, set operations, probability, off-by-one
reasoning, graph cycles, combinatorial counting).

No sandbox needed — answers are verified deterministically in Python. The
reward applies the IL efficiency-aware shaping:

    final = correctness * (0.6 + 0.4 * reasoning_quality)

The model responds in <reasoning>...</reasoning><answer>...</answer> format.
"""

from __future__ import annotations

import re

import verifiers.v1 as vf

from .scoring import compute_final_score, score_reasoning_quality
from .tasks import TASKS, ReasoningTask, _extract_answer_text

INSTRUCTION = (
    "Solve the reasoning puzzle below. First, work through it inside "
    "<reasoning>...</reasoning> tags — show your deduction step by step, "
    "check your answer, and avoid generic filler. Then give your final answer "
    "inside <answer>...</answer> tags.\n\n"
    "Your reasoning quality affects your score: be thorough but concise, cover "
    "the key concepts, and verify your work.\n\n"
)


class ILReasoningData(vf.TaskData):
    spec: str
    answer_format: str
    verify_fn_id: int  # we store the task idx to look up the verifier
    expected_concepts: list[str]
    token_budget: int


class ILReasoningTaskConfig(vf.TaskConfig):
    pass


class ILReasoningTask(vf.Task[ILReasoningData, vf.State, ILReasoningTaskConfig]):
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


class ILReasoningConfig(vf.TasksetConfig):
    task: ILReasoningTaskConfig = ILReasoningTaskConfig()


class ILReasoningTaskset(vf.Taskset[ILReasoningTask, ILReasoningConfig]):
    def load(self) -> list[ILReasoningTask]:
        return [
            ILReasoningTask(
                ILReasoningData(
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


__all__ = ["ILReasoningData", "ILReasoningTask", "ILReasoningTaskset", "ILReasoningConfig"]
