"""Handcrafted pure-reasoning tasks for the IL pipeline.

Each task is a self-contained logic/reasoning puzzle with a deterministic
verifier (no sandbox needed — answers are checked in Python). The model
responds in <reasoning>...</reasoning><answer>...</answer> format and is
scored:

    final = correctness * (0.6 + 0.4 * reasoning_quality)

These tasks target reasoning skills small models struggle with:
- multi-step deduction
- constraint satisfaction
- invariant reasoning
- type/flow inference
- counting with combinatorial structure
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ReasoningTask:
    """A single handcrafted reasoning task."""
    idx: int
    name: str
    spec: str
    # verifier: takes the extracted answer string, returns (correct: bool, info: str)
    verify: Callable[[str], tuple[bool, str]]
    expected_concepts: list[str]
    token_budget: int = 500
    difficulty: str = "medium"
    # the answer format hint shown in the prompt
    answer_format: str = "a single value on its own line"


def _extract_answer_text(response: str) -> str:
    """Extract <answer>...</answer> or fallback to text after </reasoning>."""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"</reasoning>\s*(.*)", response, re.DOTALL)
    return m.group(1).strip() if m else response.strip()


# ---------------------------------------------------------------------------
# Verifiers
# ---------------------------------------------------------------------------

def _int_answer(expected: int):
    def verify(ans: str) -> tuple[bool, str]:
        nums = re.findall(r"-?\d+", ans)
        if nums and int(nums[-1]) == expected:
            return True, f"got {nums[-1]}"
        return False, f"expected {expected}, got {ans[:80]}"
    return verify


def _set_answer(expected: set):
    def verify(ans: str) -> tuple[bool, str]:
        items = re.findall(r"-?\d+", ans)
        got = set(int(x) for x in items)
        if got == expected:
            return True, f"got {sorted(got)}"
        return False, f"expected {sorted(expected)}, got {sorted(got)}"
    return verify


def _str_answer(expected: str):
    def verify(ans: str) -> tuple[bool, str]:
        # normalize: lowercase, strip whitespace/punctuation
        norm = re.sub(r"[\s\W]+", "", ans.lower())
        exp = re.sub(r"[\s\W]+", "", expected.lower())
        if norm == exp:
            return True, f"got {ans[:80]}"
        return False, f"expected '{expected}', got '{ans[:80]}'"
    return verify


def _list_answer(expected: list):
    def verify(ans: str) -> tuple[bool, str]:
        items = re.findall(r"-?\d+", ans)
        got = [int(x) for x in items]
        if got == expected:
            return True, f"got {got}"
        return False, f"expected {expected}, got {got}"
    return verify


# ---------------------------------------------------------------------------
# 12 handcrafted reasoning tasks
# ---------------------------------------------------------------------------

TASKS: list[ReasoningTask] = [
    ReasoningTask(
        idx=0,
        name="knights_and_knaves_3",
        difficulty="medium",
        token_budget=500,
        spec=(
            "On an island, knights always tell the truth and knaves always lie.\n"
            "You meet three people: A, B, and C.\n\n"
            "A says: 'B is a knave.'\n"
            "B says: 'C is a knave.'\n"
            "C says: 'A and B are both knaves.'\n\n"
            "Determine who is a knight and who is a knave. "
            "Answer with the letter(s) of the knight(s), comma-separated."
        ),
        answer_format="the knight letter(s), e.g. 'A' or 'A,B'",
        verify=_str_answer("A,B"),
        expected_concepts=["knight", "knave", "truth", "lie", "assume", "contradict", "case"],
    ),
    ReasoningTask(
        idx=1,
        name="constraint_scheduling",
        difficulty="hard",
        token_budget=700,
        spec=(
            "Schedule 4 meetings (M1, M2, M3, M4) into 4 time slots (1,2,3,4).\n"
            "Constraints:\n"
            "- M1 must be before M2.\n"
            "- M3 must be after M4.\n"
            "- M2 and M3 cannot be in adjacent slots.\n"
            "- M1 cannot be in slot 1.\n\n"
            "How many valid schedules exist? (A schedule is an assignment of "
            "each meeting to a distinct slot.)"
        ),
        answer_format="a single integer",
        verify=_int_answer(2),
        expected_concepts=["constraint", "slot", "before", "after", "adjacent", "count", "permutation", "valid"],
    ),
    ReasoningTask(
        idx=2,
        name="invariant_loop",
        difficulty="hard",
        token_budget=600,
        spec=(
            "Consider this loop:\n\n"
            "    x = 1\n"
            "    for i in range(1, 6):\n"
            "        x = x * 2 + i\n\n"
            "What is the final value of x? Trace each iteration carefully."
        ),
        answer_format="a single integer",
        verify=_int_answer(94),
        expected_concepts=["trace", "iteration", "multiply", "add", "loop", "step", "value"],
    ),
    ReasoningTask(
        idx=3,
        name="type_flow_inference",
        difficulty="medium",
        token_budget=500,
        spec=(
            "Given this Python function (do NOT run it — infer the types):\n\n"
            "    def f(a, b):\n"
            "        c = a + b\n"
            "        d = c * 2\n"
            "        e = [d] * 3\n"
            "        return e[0]\n\n"
            "If a is an int and b is a float, what is the type of the return value? "
            "Answer with the Python type name."
        ),
        answer_format="a Python type name, e.g. 'int' or 'float' or 'list'",
        verify=_str_answer("float"),
        expected_concepts=["type", "int", "float", "add", "multiply", "coerce", "list", "index"],
    ),
    ReasoningTask(
        idx=4,
        name="counting_paths_grid",
        difficulty="medium",
        token_budget=500,
        spec=(
            "In a 4x4 grid, how many distinct shortest paths are there from the "
            "top-left corner to the bottom-right corner, moving only right or "
            "down? (A shortest path uses exactly 6 moves: 3 right + 3 down.)"
        ),
        answer_format="a single integer",
        verify=_int_answer(20),
        expected_concepts=["grid", "path", "right", "down", "binomial", "combination", "count", "shortest"],
    ),
    ReasoningTask(
        idx=5,
        name="logic_puzzle_zebra_mini",
        difficulty="hard",
        token_budget=800,
        spec=(
            "Three houses in a row: left, middle, right.\n"
            "Three people: Alice, Bob, Carol.\n"
            "Three colors: red, blue, green.\n\n"
            "Clues:\n"
            "1. Alice lives in the red house.\n"
            "2. Bob lives to the immediate left of Carol.\n"
            "3. The green house is the rightmost.\n"
            "4. The middle house is blue.\n\n"
            "Who lives in the middle house? Answer with the name."
        ),
        answer_format="a single name: Alice, Bob, or Carol",
        verify=_str_answer("Bob"),
        expected_concepts=["house", "left", "right", "middle", "color", "clue", "deduce", "assign"],
    ),
    ReasoningTask(
        idx=6,
        name="recursive_trace",
        difficulty="hard",
        token_budget=700,
        spec=(
            "Trace this recursive function by hand (do NOT run it):\n\n"
            "    def g(n):\n"
            "        if n <= 1:\n"
            "            return n\n"
            "        return g(n-1) + g(n-2) + 1\n\n"
            "What is g(6)? Show your trace of the call tree."
        ),
        answer_format="a single integer",
        verify=_int_answer(25),
        expected_concepts=["recursive", "base", "call", "tree", "trace", "fibonacci", "add", "return"],
    ),
    ReasoningTask(
        idx=7,
        name="set_operations_deduction",
        difficulty="medium",
        token_budget=500,
        spec=(
            "Set A = {1, 2, 3, 4, 5}.\n"
            "Set B = {3, 4, 5, 6, 7}.\n"
            "Compute (A - B) | (B - A) (the symmetric difference). "
            "List the elements in ascending order."
        ),
        answer_format="comma-separated integers in ascending order",
        verify=_set_answer({1, 2, 6, 7}),
        expected_concepts=["set", "difference", "symmetric", "union", "element", "subtract", "combine"],
    ),
    ReasoningTask(
        idx=8,
        name="probability_reasoning",
        difficulty="hard",
        token_budget=600,
        spec=(
            "A bag has 3 red and 2 blue balls. You draw 2 balls without "
            "replacement. What is the probability that both are the same color? "
            "Answer as a simplified fraction 'p/q'."
        ),
        answer_format="a fraction like '2/5'",
        verify=_str_answer("2/5"),
        expected_concepts=["probability", "draw", "red", "blue", "combination", "fraction", "same", "without"],
    ),
    ReasoningTask(
        idx=9,
        name="off_by_one_reasoning",
        difficulty="medium",
        token_budget=400,
        spec=(
            "A function is supposed to sum integers from 1 to n inclusive. "
            "The implementation uses `range(1, n)` instead of `range(1, n+1)`. "
            "For n=10, what is the difference between the correct sum and the "
            "buggy sum?"
        ),
        answer_format="a single integer",
        verify=_int_answer(10),
        expected_concepts=["range", "off-by-one", "sum", "inclusive", "exclusive", "difference", "bug"],
    ),
    ReasoningTask(
        idx=10,
        name="graph_deduction",
        difficulty="medium",
        token_budget=500,
        spec=(
            "A directed graph has edges: A->B, B->C, C->D, A->D, D->A.\n"
            "How many distinct cycles of length 3 exist? "
            "(A cycle visits 3 distinct nodes and returns to the start.)"
        ),
        answer_format="a single integer",
        verify=_int_answer(1),
        expected_concepts=["graph", "cycle", "directed", "edge", "node", "length", "distinct", "path"],
    ),
    ReasoningTask(
        idx=11,
        name="combinatorial_counting",
        difficulty="hard",
        token_budget=600,
        spec=(
            "How many ways can you arrange the letters in the word 'BANANA' "
            "such that no two A's are adjacent? (The three A's are identical, "
            "the two N's are identical, and B is unique.)"
        ),
        answer_format="a single integer",
        verify=_int_answer(1),
        expected_concepts=["arrange", "letter", "adjacent", "identical", "permutation", "slot", "gap", "count"],
    ),
]


__all__ = ["ReasoningTask", "TASKS"]
