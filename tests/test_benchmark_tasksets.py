"""Tests for HumanEval and GSM8K benchmark tasksets and graders."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from iloptimus.core.grader import (
    build_prompt,
    get_num_tasks,
    grade_gsm8k,
    grade_humaneval,
    grade_response,
)
from iloptimus.core.tasksets import get_all_tasksets, get_taskset

# ---------------------------------------------------------------------------
# Taskset registration
# ---------------------------------------------------------------------------


def test_humaneval_taskset_is_registered():
    ts = get_taskset("humaneval-v1")
    assert ts is not None
    assert ts.domain == "humaneval"
    assert ts.num_tasks == 25
    assert ts.needs_sandbox is True


def test_gsm8k_taskset_is_registered():
    ts = get_taskset("gsm8k-v1")
    assert ts is not None
    assert ts.domain == "gsm8k"
    assert ts.num_tasks == 25
    assert ts.needs_sandbox is False


def test_all_tasksets_include_benchmarks():
    all_ts = get_all_tasksets()
    ids = {t.id for t in all_ts}
    assert "humaneval-v1" in ids
    assert "gsm8k-v1" in ids


# ---------------------------------------------------------------------------
# Task count
# ---------------------------------------------------------------------------


def test_humaneval_task_count():
    assert get_num_tasks("humaneval") == 25


def test_gsm8k_task_count():
    assert get_num_tasks("gsm8k") == 25


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def test_humaneval_build_prompt():
    prompt = build_prompt("humaneval", 0)
    assert "has_close_elements" in prompt
    assert "def has_close_elements" in prompt
    assert "<reasoning>" in prompt
    assert "<answer>" in prompt


def test_gsm8k_build_prompt():
    prompt = build_prompt("gsm8k", 0)
    assert "Natalia" in prompt
    assert "<reasoning>" in prompt
    assert "<answer>" in prompt


def test_humaneval_build_prompt_all_tasks():
    for i in range(25):
        prompt = build_prompt("humaneval", i)
        assert len(prompt) > 50
        assert "## Task:" in prompt


def test_gsm8k_build_prompt_all_tasks():
    for i in range(25):
        prompt = build_prompt("gsm8k", i)
        assert len(prompt) > 50
        assert "## Task:" in prompt


# ---------------------------------------------------------------------------
# HumanEval grading
# ---------------------------------------------------------------------------


def test_grade_humaneval_correct_solution():
    """A correct solution for has_close_elements should get full correctness."""
    response = (
        "<reasoning>I need to check all pairs of numbers and see if any pair "
        "has a distance less than the threshold. I'll use nested loops and "
        "the abs function to compute distances.</reasoning>"
        "<answer>```python\ndef has_close_elements(numbers, threshold):\n"
        "    for i in range(len(numbers)):\n"
        "        for j in range(i + 1, len(numbers)):\n"
        "            if abs(numbers[i] - numbers[j]) < threshold:\n"
        "                return True\n"
        "    return False\n```</answer>"
    )
    result = grade_humaneval(response, 0)
    assert result.correctness == 1.0
    assert result.score > 0.6  # at least 0.6 for correct + some reasoning


def test_grade_humaneval_wrong_solution():
    """A wrong solution should get 0 correctness."""
    response = (
        "<reasoning>I'll just return True always.</reasoning>"
        "<answer>```python\ndef has_close_elements(numbers, threshold):\n"
        "    return True\n```</answer>"
    )
    result = grade_humaneval(response, 0)
    assert result.correctness < 1.0  # won't pass all tests


def test_grade_humaneval_empty_response():
    result = grade_humaneval("", 0)
    assert result.correctness == 0.0
    assert result.score == 0.0


def test_grade_humaneval_no_code():
    response = "<reasoning>I think the answer is to use a loop.</reasoning><answer>No code here</answer>"
    result = grade_humaneval(response, 0)
    assert result.correctness == 0.0


def test_grade_humaneval_fibonacci_correct():
    """Test a correct Fibonacci implementation."""
    response = (
        "<reasoning>I'll use iteration to compute Fibonacci. "
        "F(0)=0, F(1)=1, then F(n)=F(n-1)+F(n-2). "
        "I'll verify: F(10) should be 55.</reasoning>"
        "<answer>```python\ndef fibonacci(n):\n"
        "    if n == 0:\n"
        "        return 0\n"
        "    if n == 1:\n"
        "        return 1\n"
        "    a, b = 0, 1\n"
        "    for _ in range(2, n + 1):\n"
        "        a, b = b, a + b\n"
        "    return b\n```</answer>"
    )
    result = grade_humaneval(response, 17)  # fibonacci is idx 17
    assert result.correctness == 1.0


def test_grade_humaneval_palindrome_correct():
    """Test a correct palindrome implementation."""
    response = (
        "<reasoning>I'll compare the string with its reverse. "
        "If they're equal, it's a palindrome. "
        "I'll verify with 'racecar' which reversed is 'racecar'.</reasoning>"
        "<answer>```python\ndef is_palindrome(text):\n"
        "    return text == text[::-1]\n```</answer>"
    )
    result = grade_humaneval(response, 16)  # is_palindrome is idx 16
    assert result.correctness == 1.0


# ---------------------------------------------------------------------------
# GSM8K grading
# ---------------------------------------------------------------------------


def test_grade_gsm8k_correct_answer():
    """A correct answer to the first GSM8K problem should get full correctness."""
    response = (
        "<reasoning>Natalia sold 48 clips in April. In May she sold half as many, "
        "so 48/2 = 24 clips. Altogether: 48 + 24 = 72 clips. "
        "Let me verify: 48 + 24 = 72. Yes.</reasoning>"
        "<answer>72</answer>"
    )
    result = grade_gsm8k(response, 0)
    assert result.correctness == 1.0
    assert result.score > 0.6


def test_grade_gsm8k_wrong_answer():
    """A wrong answer should get 0 correctness."""
    response = (
        "<reasoning>Natalia sold 48 clips in April and 24 in May. "
        "Total is 48 + 24 = 100.</reasoning>"
        "<answer>100</answer>"
    )
    result = grade_gsm8k(response, 0)
    assert result.correctness == 0.0


def test_grade_gsm8k_empty_response():
    result = grade_gsm8k("", 0)
    assert result.correctness == 0.0


def test_grade_gsm8k_answer_with_text():
    """The verifier should extract the number even if there's extra text."""
    response = (
        "<reasoning>The answer is 72.</reasoning>"
        "<answer>The total number of clips is 72.</answer>"
    )
    result = grade_gsm8k(response, 0)
    assert result.correctness == 1.0


def test_grade_gsm8k_division_problem():
    """Test the division problem (idx 5: 36 pieces / 4 friends = 9)."""
    response = (
        "<reasoning>Mark has 36 pieces of candy and 4 friends. "
        "To share equally, divide 36 by 4. 36 / 4 = 9. "
        "Each friend gets 9 pieces.</reasoning>"
        "<answer>9</answer>"
    )
    result = grade_gsm8k(response, 5)
    assert result.correctness == 1.0


def test_grade_gsm8k_percentage_problem():
    """Test the percentage problem (idx 10: 20% off $25 = $20)."""
    response = (
        "<reasoning>The shirt costs $25 and the discount is 20%. "
        "20% of 25 = 0.20 * 25 = 5. "
        "So the discounted price is 25 - 5 = 20 dollars. "
        "Let me verify: 25 * 0.8 = 20. Correct.</reasoning>"
        "<answer>20</answer>"
    )
    result = grade_gsm8k(response, 10)
    assert result.correctness == 1.0


def test_grade_gsm8k_decimal_answer():
    """Test that decimal answers are handled correctly (idx 17: average = 86.6)."""
    response = (
        "<reasoning>The scores are 85, 90, 78, 92, 88. "
        "Sum = 85 + 90 + 78 + 92 + 88 = 433. "
        "Average = 433 / 5 = 86.6.</reasoning>"
        "<answer>86.6</answer>"
    )
    result = grade_gsm8k(response, 17)
    assert result.correctness == 1.0


# ---------------------------------------------------------------------------
# Unified grader interface
# ---------------------------------------------------------------------------


def test_grade_response_humaneval():
    """Test that grade_response routes to grade_humaneval."""
    response = (
        "<reasoning>Return the decimal part by subtracting the integer part.</reasoning>"
        "<answer>```python\ndef truncate_number(number):\n"
        "    return number - int(number)\n```</answer>"
    )
    result = grade_response("humaneval", 2, response)  # truncate_number is idx 2
    assert result.correctness == 1.0


def test_grade_response_gsm8k():
    """Test that grade_response routes to grade_gsm8k."""
    response = (
        "<reasoning>48 in April, 24 in May. Total = 72.</reasoning>"
        "<answer>72</answer>"
    )
    result = grade_response("gsm8k", 0, response)
    assert result.correctness == 1.0


# ---------------------------------------------------------------------------
# Taskset task data integrity
# ---------------------------------------------------------------------------


def _load_tasks_module(package_name: str, filename: str = "tasks.py"):
    """Load a taskset's tasks.py module directly."""
    repo_root = Path(__file__).parent.parent
    path = repo_root / package_name / package_name / filename
    if not path.exists():
        pytest.skip(f"Taskset file not found: {path}")
    spec = importlib.util.spec_from_file_location(f"{package_name}_tasks_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_humaneval_tasks_have_required_fields():
    tasks_mod = _load_tasks_module("humaneval_v1")
    for task in tasks_mod.TASKS:
        assert task.idx >= 0
        assert len(task.name) > 0
        assert len(task.spec) > 10
        assert len(task.signature) > 0
        assert len(task.tests) > 0
        assert task.entry_point
        assert len(task.expected_concepts) > 0
        assert task.token_budget > 0


def test_gsm8k_tasks_have_required_fields():
    tasks_mod = _load_tasks_module("gsm8k_v1")
    for task in tasks_mod.TASKS:
        assert task.idx >= 0
        assert len(task.name) > 0
        assert len(task.spec) > 10
        assert callable(task.verify)
        assert len(task.expected_concepts) > 0
        assert task.token_budget > 0


def test_humaneval_task_indices_are_sequential():
    tasks_mod = _load_tasks_module("humaneval_v1")
    indices = [t.idx for t in tasks_mod.TASKS]
    assert indices == list(range(25))


def test_gsm8k_task_indices_are_sequential():
    tasks_mod = _load_tasks_module("gsm8k_v1")
    indices = [t.idx for t in tasks_mod.TASKS]
    assert indices == list(range(25))


def test_humaneval_all_tests_are_valid_asserts():
    tasks_mod = _load_tasks_module("humaneval_v1")
    for task in tasks_mod.TASKS:
        for test in task.tests:
            assert test.startswith("assert "), f"Test in {task.name} doesn't start with 'assert'"


def test_gsm8k_all_verifiers_return_tuples():
    tasks_mod = _load_tasks_module("gsm8k_v1")
    for task in tasks_mod.TASKS:
        correct, info = task.verify("0")  # test with dummy answer
        assert isinstance(correct, bool)
        assert isinstance(info, str)


# ---------------------------------------------------------------------------
# Self-improvement state persistence
# ---------------------------------------------------------------------------


def test_self_improvement_state_roundtrip(tmp_path, monkeypatch):
    """Test that SelfImprovementState can be saved and loaded."""
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    # Force re-import to pick up the new env
    si_path = Path(__file__).parent.parent / "scripts" / "self_improve.py"
    spec = importlib.util.spec_from_file_location("self_improve", si_path)
    si_mod = importlib.util.module_from_spec(spec)
    sys.modules["self_improve"] = si_mod
    spec.loader.exec_module(si_mod)

    state = si_mod.SelfImprovementState(
        model_id="test-model",
        started_at=12345.0,
        rounds_completed=2,
        benchmarks={
            "humaneval-v1": si_mod.BenchmarkScore(
                taskset_id="humaneval-v1",
                baseline_accuracy=0.12,
                best_accuracy=0.20,
                best_round=2,
            ),
        },
    )
    si_mod._save_state(state)

    loaded = si_mod._load_state("test-model")
    assert loaded.model_id == "test-model"
    assert loaded.rounds_completed == 2
    assert loaded.benchmarks["humaneval-v1"].baseline_accuracy == 0.12
    assert loaded.benchmarks["humaneval-v1"].best_accuracy == 0.20
