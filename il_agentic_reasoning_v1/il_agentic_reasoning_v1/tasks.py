"""Handcrafted agentic-reasoning tasks for the IL pipeline.

Each task requires SUSTAINED multi-step reasoning — tracing through multiple
modules, cascading constraints, or interleaved state — where each step depends
on the previous. These are the "long-horizon reasoning" tasks from the IL
philosophy: small models fix the first step and miss the cascade.

Answers are verified deterministically (precomputed expected outputs). The
reward applies IL efficiency-aware shaping:

    final = correctness * (0.6 + 0.4 * reasoning_quality)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class AgenticReasoningTask:
    idx: int
    name: str
    spec: str
    verify: Callable[[str], tuple[bool, str]]
    expected_concepts: list[str]
    token_budget: int = 800
    difficulty: str = "hard"
    answer_format: str = "a single value"
    reasoning_skill: str = ""


def _extract_answer_text(response: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"</reasoning>\s*(.*)", response, re.DOTALL)
    return m.group(1).strip() if m else response.strip()


def _int_verify(expected: int):
    def v(ans: str) -> tuple[bool, str]:
        nums = re.findall(r"-?\d+", ans)
        if nums and int(nums[-1]) == expected:
            return True, f"got {nums[-1]}"
        return False, f"expected {expected}"
    return v


def _str_verify(expected: str):
    def v(ans: str) -> tuple[bool, str]:
        norm = re.sub(r"[\s\W]+", "", ans.lower())
        exp = re.sub(r"[\s\W]+", "", expected.lower())
        return (norm == exp, f"expected '{expected}'")
    return v


def _list_verify(expected: list[int]):
    def v(ans: str) -> tuple[bool, str]:
        items = re.findall(r"-?\d+", ans)
        got = [int(x) for x in items]
        return (got == expected, f"expected {expected}, got {got}")
    return v


# ---------------------------------------------------------------------------
# 10 handcrafted agentic-reasoning tasks
# ---------------------------------------------------------------------------

TASKS: list[AgenticReasoningTask] = [
    AgenticReasoningTask(
        idx=0,
        name="cascading_pipeline_trace",
        difficulty="hard",
        token_budget=900,
        reasoning_skill="Multi-step cascading reasoning where each step depends on the previous",
        spec=(
            "Trace this 3-module data pipeline by hand (do NOT run it):\n\n"
            "Module 1 (parser):  parse('3,5,7') -> [3, 5, 7]\n"
            "    BUT it has a bug: drops the last element, so -> [3, 5]\n\n"
            "Module 2 (doubler):  doubles each value -> [6, 10]\n\n"
            "Module 3 (summer):  sums all values -> 16\n\n"
            "Now trace the FULL pipeline with input '10,20,30,40':\n"
            "  parser (buggy) -> doubler -> summer\n\n"
            "What is the final output? Show each stage."
        ),
        answer_format="a single integer",
        verify=_int_verify(60),
        expected_concepts=["parse", "drop", "last", "double", "sum", "stage", "cascade", "trace", "bug"],
    ),
    AgenticReasoningTask(
        idx=1,
        name="cross_module_data_flow",
        difficulty="hard",
        token_budget=900,
        reasoning_skill="Sustained reasoning across 5 modules",
        spec=(
            "Trace data through 5 functions (by hand):\n\n"
            "  f1(x) = x + 1\n"
            "  f2(x) = x * 2\n"
            "  f3(x) = x - 3\n"
            "  f4(x) = x if x > 0 else 0\n"
            "  f5(x) = x ** 2\n\n"
            "Pipeline: f5(f4(f3(f2(f1(2)))))\n\n"
            "What is the final result? Show each function application."
        ),
        answer_format="a single integer",
        verify=_int_verify(1),
        expected_concepts=["f1", "f2", "f3", "f4", "f5", "add", "multiply", "subtract", "positive", "square", "trace", "pipeline"],
    ),
    AgenticReasoningTask(
        idx=2,
        name="invariant_preservation",
        difficulty="hard",
        token_budget=700,
        reasoning_skill="Mathematical reasoning about invariants",
        spec=(
            "A loop maintains the invariant: `sum == k * (k + 1) // 2` where k "
            "is the number of iterations completed.\n\n"
            "    sum = 0\n"
            "    for k in range(1, 11):\n"
            "        sum += k\n\n"
            "After the loop, the invariant says sum == 10 * 11 // 2 = 55.\n\n"
            "Now consider a MODIFIED loop that adds k*2 instead of k:\n"
            "    sum = 0\n"
            "    for k in range(1, 11):\n"
            "        sum += k * 2\n\n"
            "What is the new invariant formula (in terms of k), and what is "
            "the final sum? Answer with just the final sum as an integer."
        ),
        answer_format="a single integer",
        verify=_int_verify(110),
        expected_concepts=["invariant", "loop", "sum", "formula", "k", "iteration", "double", "modify", "preserve"],
    ),
    AgenticReasoningTask(
        idx=3,
        name="race_interleaving_count",
        difficulty="hard",
        token_budget=800,
        reasoning_skill="Concurrency reasoning about interleavings",
        spec=(
            "Two threads each increment a shared counter `c` (initially 0) "
            "exactly 2 times each. Each increment is: `temp = c; c = temp + 1` "
            "(non-atomic — a read-then-write that can interleave).\n\n"
            "How many DISTINCT final values of `c` are possible across all "
            "interleavings? (Not the number of interleavings — the number of "
            "distinct outcomes.)"
        ),
        answer_format="a single integer",
        verify=_int_verify(3),
        expected_concepts=["thread", "interleave", "race", "counter", "atomic", "read", "write", "distinct", "outcome", "possible"],
    ),
    AgenticReasoningTask(
        idx=4,
        name="api_contract_compliance",
        difficulty="hard",
        token_budget=800,
        reasoning_skill="Constraint satisfaction over 10+ constraints",
        spec=(
            "An API endpoint must satisfy these constraints:\n"
            "1. Accepts only GET or POST methods.\n"
            "2. Requires an 'auth' header.\n"
            "3. POST must include a 'body' field.\n"
            "4. GET must NOT include a 'body' field.\n"
            "5. The 'auth' header must start with 'Bearer '.\n"
            "6. Rate limit: max 5 requests per minute.\n"
            "7. Response is JSON.\n"
            "8. POST response status is 201.\n"
            "9. GET response status is 200.\n"
            "10. Errors return status 400.\n\n"
            "A client sends: method=POST, headers={'auth': 'Bearer xyz'}, "
            "body={'data': 1}. This is the 3rd request this minute.\n\n"
            "How many of the 10 constraints are SATISFIED by this request "
            "(considering both request validity and expected response)?"
        ),
        answer_format="a single integer (0-10)",
        verify=_int_verify(8),
        expected_concepts=["constraint", "method", "post", "get", "auth", "bearer", "body", "rate", "status", "json", "satisfy"],
    ),
    AgenticReasoningTask(
        idx=5,
        name="recursive_repair_trace",
        difficulty="hard",
        token_budget=800,
        reasoning_skill="Recursion reasoning — tracing call paths",
        spec=(
            "Trace this recursive function by hand (do NOT run it):\n\n"
            "    def h(n):\n"
            "        if n <= 0:\n"
            "            return 1\n"
            "        if n % 2 == 0:\n"
            "            return h(n // 2) + h(n // 2 - 1)\n"
            "        return h(n - 1) + h(n - 2) + h(n - 3)\n\n"
            "What is h(5)? Show the key recursive calls."
        ),
        answer_format="a single integer",
        verify=_int_verify(9),
        expected_concepts=["recursive", "base", "even", "odd", "call", "trace", "tree", "return", "divide", "subtract"],
    ),
    AgenticReasoningTask(
        idx=6,
        name="state_machine_trace",
        difficulty="hard",
        token_budget=800,
        reasoning_skill="State machine simulation over many transitions",
        spec=(
            "A finite state machine:\n"
            "  States: S0, S1, S2, S3 (S0 is start, S3 is accept)\n"
            "  Transitions:\n"
            "    S0 --a--> S1    S0 --b--> S0\n"
            "    S1 --a--> S2    S1 --b--> S0\n"
            "    S2 --a--> S3    S2 --b--> S1\n"
            "    S3 --a--> S3    S3 --b--> S2\n\n"
            "Starting from S0, process the input string 'aabbaa'.\n"
            "What is the final state? Answer with the state name (S0, S1, S2, or S3)."
        ),
        answer_format="a state name: S0, S1, S2, or S3",
        verify=_str_verify("S3"),
        expected_concepts=["state", "transition", "machine", "start", "accept", "input", "process", "trace", "final"],
    ),
    AgenticReasoningTask(
        idx=7,
        name="differential_analysis",
        difficulty="hard",
        token_budget=700,
        reasoning_skill="Comparing two code paths to find the behavioral difference",
        spec=(
            "Two functions:\n\n"
            "    def f_a(lst):\n"
            "        return [x for x in lst if x > 0]\n\n"
            "    def f_b(lst):\n"
            "        result = []\n"
            "        for x in lst:\n"
            "            if x > 0:\n"
            "                result.append(x)\n"
            "        return result\n\n"
            "For input [3, -1, 0, 2, -5], what is f_a(lst) == f_b(lst)?\n"
            "And what is the LENGTH of the output list? Answer with just the length."
        ),
        answer_format="a single integer",
        verify=_int_verify(2),
        expected_concepts=["compare", "filter", "list", "comprehension", "loop", "append", "positive", "difference", "same", "length"],
    ),
    AgenticReasoningTask(
        idx=8,
        name="error_propagation_chain",
        difficulty="hard",
        token_budget=700,
        reasoning_skill="Tracing how an error propagates through a call chain",
        spec=(
            "A call chain:\n"
            "  layer3() calls layer2()\n"
            "  layer2() calls layer1()\n"
            "  layer1() calls base()\n\n"
            "base() raises ValueError('bad index') if its argument is negative.\n"
            "layer1() catches ValueError and converts it to RuntimeError.\n"
            "layer2() does NOT catch any exception.\n"
            "layer3() catches RuntimeError and returns -1.\n\n"
            "If base() is called with argument -5, what does layer3() return?"
        ),
        answer_format="a single integer",
        verify=_int_verify(-1),
        expected_concepts=["error", "propagate", "catch", "raise", "valueerror", "runtimeerror", "layer", "chain", "convert", "return"],
    ),
    AgenticReasoningTask(
        idx=9,
        name="coverage_gap_analysis",
        difficulty="hard",
        token_budget=800,
        reasoning_skill="Identifying which test cases are missing from a suite",
        spec=(
            "A function `classify_triangle(a, b, c)` takes 3 side lengths and "
            "returns 'equilateral', 'isosceles', 'scalene', or 'invalid'.\n\n"
            "Existing tests:\n"
            "  T1: (3,3,3) -> equilateral\n"
            "  T2: (3,3,4) -> isosceles\n"
            "  T3: (3,4,5) -> scalene\n"
            "  T4: (0,1,2) -> invalid\n"
            "  T5: (1,1,2) -> invalid (violates triangle inequality)\n\n"
            "How many of these EDGE CASE categories are NOT covered by T1-T5?\n"
            "Categories: {equilateral, isosceles, scalene, zero-side, "
            "negative-side, degenerate-inequality, very-large-input, "
            "non-integer-input}. Count only the categories with NO test."
        ),
        answer_format="a single integer",
        verify=_int_verify(3),
        expected_concepts=["test", "coverage", "edge", "case", "category", "missing", "gap", "triangle", "invalid", "negative", "large"],
    ),
]


__all__ = ["AgenticReasoningTask", "TASKS"]
