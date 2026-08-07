"""il-agentic-reasoning-v1 — handcrafted multi-step agentic reasoning tasks.

A `verifiers.v1` taskset of 10 handcrafted long-horizon reasoning scenarios
requiring sustained multi-step deduction: cascading pipeline traces, cross-module
data flow, invariant preservation, race-condition interleavings, API contract
compliance, recursive repair traces, state machine simulation, differential
analysis, error propagation chains, and coverage gap analysis.

Answers verified deterministically (precomputed). IL efficiency-aware shaping:

    final = correctness * (0.6 + 0.4 * reasoning_quality)

These target the failure mode where small models solve the first step and miss
the cascade — the 0.4 reasoning-quality spread shapes sustained, verified
multi-step reasoning.
"""

from __future__ import annotations

import verifiers.v1 as vf

from .scoring import compute_final_score, score_reasoning_quality
from .tasks import TASKS, AgenticReasoningTask, _extract_answer_text

INSTRUCTION = (
    "Solve the multi-step reasoning task below. This requires SUSTAINED "
    "reasoning — trace through each stage carefully, as later steps depend on "
    "earlier ones. Work inside <reasoning>...</reasoning> tags, showing each "
    "step and verifying your answer. Then give your final answer inside "
    "<answer>...</answer> tags.\n\n"
    "Your reasoning quality affects your score: cover the key concepts, stay "
    "within budget, verify your work, and avoid generic filler.\n\n"
)


class ILAgenticReasoningData(vf.TaskData):
    spec: str
    answer_format: str
    task_idx: int
    expected_concepts: list[str]
    token_budget: int


class ILAgenticReasoningTaskConfig(vf.TaskConfig):
    pass


class ILAgenticReasoningTask(vf.Task[ILAgenticReasoningData, vf.State, ILAgenticReasoningTaskConfig]):
    @vf.reward(weight=1.0)
    async def scored(self, trace: vf.Trace) -> float:
        response = trace.last_reply or ""
        answer = _extract_answer_text(response)
        task = TASKS[self.data.task_idx]
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


class ILAgenticReasoningConfig(vf.TasksetConfig):
    task: ILAgenticReasoningTaskConfig = ILAgenticReasoningTaskConfig()


class ILAgenticReasoningTaskset(vf.Taskset[ILAgenticReasoningTask, ILAgenticReasoningConfig]):
    def load(self) -> list[ILAgenticReasoningTask]:
        return [
            ILAgenticReasoningTask(
                ILAgenticReasoningData(
                    idx=t.idx,
                    name=t.name,
                    prompt=INSTRUCTION + f"## Task: {t.name}\n\n{t.spec}\n\n"
                    f"Answer format: {t.answer_format}.",
                    spec=t.spec,
                    answer_format=t.answer_format,
                    task_idx=t.idx,
                    expected_concepts=t.expected_concepts,
                    token_budget=t.token_budget,
                ),
                self.config.task,
            )
            for t in TASKS
        ]


__all__ = [
    "ILAgenticReasoningData",
    "ILAgenticReasoningTask",
    "ILAgenticReasoningTaskset",
    "ILAgenticReasoningConfig",
]
