"""Fixed runtime and model prompt for executable no-code environments."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

FRAMEWORK_VERSION = 2
SUPPORTED_GRADERS = {"exact", "numeric", "contains_all"}
PLACEHOLDER_TEXT = {
    "concrete task",
    "canonical answer",
    "observable reasoning feature",
    "complete problem statement",
    "model's response",
}


def builder_skill() -> str:
    path = Path(__file__).parents[1] / "resources" / "environment-builder" / "SKILL.md"
    return path.read_text(encoding="utf-8")


def build_design_prompt(mode: str, description: str) -> str:
    return (
        f"{builder_skill()}\n\n"
        "## Current build request\n\n"
        f"Mode: {mode.upper()}\n"
        f"User goal: {description.strip()}\n\n"
        "Use the framework shape but replace every example and placeholder. Create at least 3 complete, distinct tasks. "
        "Return only the completed JSON object."
    )


def build_repair_prompt(mode: str, description: str, generated: dict[str, Any], issues: list[str]) -> str:
    issue_list = "\n".join(f"- {issue}" for issue in issues)
    return (
        f"{builder_skill()}\n\n"
        "## Repair this environment\n\n"
        f"Mode: {mode.upper()}\nUser goal: {description.strip()}\n"
        f"Validation failures:\n{issue_list}\n\n"
        f"Rejected draft:\n{json.dumps(generated, ensure_ascii=False)}\n\n"
        "Return a corrected JSON object only. It must contain 3 to 6 distinct, self-contained tasks, no placeholder "
        "text, and each ideal_response must receive full correctness from its grader."
    )


def build_task_prompt(
    mode: str,
    description: str,
    difficulty: str,
    previous_prompts: list[str],
    issues: list[str] | None = None,
) -> str:
    avoid = "\n".join(f"- {prompt}" for prompt in previous_prompts) or "- none"
    repair = ""
    if issues:
        repair = "\nFix these validation failures:\n" + "\n".join(f"- {issue}" for issue in issues)
    return f"""You are filling one task in the IL Optimus executable environment framework.
Training mode: {mode.upper()}
Environment goal: {description.strip()}
Difficulty: {difficulty}

Return one JSON object with exactly these keys: name, prompt, expected_answer, ideal_response, criteria, grader, difficulty.
The prompt must be a concrete self-contained problem, not a description of a problem. Solve it yourself.
ideal_response must contain <reasoning>...</reasoning><answer>...</answer> and must be correct.
grader must be one of:
- exact: use keys type and target; target must equal the text inside ideal_response's answer tag
- numeric: use keys type, target, and tolerance; target must be the number inside the answer tag
- contains_all: use keys type and terms; every term must literally occur in ideal_response
Do not use nulls, placeholders, angle-bracket descriptions, or Markdown. Return JSON only.
Do not duplicate these earlier prompts:
{avoid}{repair}"""


def extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _answer_text(response: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    return (matches[-1] if matches else response).strip()


def _normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold().rstrip(". ")


def normalise_grader(task: dict[str, Any]) -> dict[str, Any]:
    raw = task.get("grader") if isinstance(task.get("grader"), dict) else {}
    grader_type = str(raw.get("type") or "").strip().lower()
    expected = str(task.get("expected_answer") or "").strip()
    criteria = [str(item).strip() for item in task.get("criteria", []) if str(item).strip()]

    if grader_type not in SUPPORTED_GRADERS:
        grader_type = "exact" if expected else "contains_all"

    if grader_type == "exact":
        target = str(raw.get("target") or expected).strip()
        answer_match = re.search(r"<answer>(.*?)</answer>", target, re.DOTALL | re.IGNORECASE)
        if answer_match:
            target = answer_match.group(1).strip()
        if not target:
            raise ValueError("Exact graders need a target answer")
        return {"type": "exact", "target": target}

    if grader_type == "numeric":
        target = raw.get("target", expected)
        try:
            numeric_target = float(target)
            tolerance = max(0.0, float(raw.get("tolerance", 0.0)))
        except (TypeError, ValueError) as error:
            raise ValueError("Numeric graders need a numeric target and tolerance") from error
        return {"type": "numeric", "target": numeric_target, "tolerance": tolerance}

    terms = raw.get("terms") or criteria
    clean_terms = [str(item).strip() for item in terms if str(item).strip()]
    if not clean_terms:
        raise ValueError("Contains-all graders need at least one observable term")
    return {"type": "contains_all", "terms": clean_terms[:12]}


def task_issues(task: dict[str, Any], index: int = 0) -> list[str]:
    issues: list[str] = []
    prompt = str(task.get("prompt") or "").strip()
    combined = " ".join(
        [
            str(task.get("name") or ""),
            prompt,
            str(task.get("expected_answer") or ""),
            str(task.get("ideal_response") or ""),
            " ".join(str(item) for item in task.get("criteria", [])),
        ]
    ).casefold()
    for placeholder in PLACEHOLDER_TEXT:
        if placeholder in combined:
            issues.append(f"task {index + 1} still contains placeholder text: {placeholder}")
    if len(prompt) < 20:
        issues.append(f"task {index + 1} prompt is too short to be self-contained")
    try:
        grader = normalise_grader(task)
    except ValueError as error:
        issues.append(f"task {index + 1}: {error}")
        return issues
    ideal_response = str(task.get("ideal_response") or "")
    if not ideal_response:
        issues.append(f"task {index + 1} needs an ideal_response")
    elif score_task({**task, "grader": grader}, ideal_response)["correctness"] < 1.0:
        issues.append(f"task {index + 1} ideal_response does not pass its grader")
    return issues


def design_issues(generated: dict[str, Any] | None) -> list[str]:
    if not generated:
        return ["The response did not contain a JSON object"]
    tasks = generated.get("tasks")
    if not isinstance(tasks, list):
        return ["tasks must be a JSON list"]

    issues: list[str] = []
    if not 3 <= len(tasks) <= 6:
        issues.append("tasks must contain between 3 and 6 items")
    prompts: set[str] = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            issues.append(f"task {index + 1} must be an object")
            continue
        issues.extend(task_issues(task, index))
        prompt_key = _normalise_text(str(task.get("prompt") or ""))
        if prompt_key in prompts:
            issues.append(f"task {index + 1} duplicates another prompt")
        prompts.add(prompt_key)
    return issues


def score_task(task: dict[str, Any], response: str) -> dict[str, float]:
    grader = normalise_grader(task)
    answer = _answer_text(response)
    grader_type = grader["type"]

    if grader_type == "exact":
        correctness = float(_normalise_text(answer) == _normalise_text(grader["target"]))
        coverage = correctness
    elif grader_type == "numeric":
        numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", answer.replace(",", ""))
        candidate = float(numbers[-1]) if numbers else math.inf
        correctness = float(abs(candidate - grader["target"]) <= grader["tolerance"])
        coverage = correctness
    else:
        response_text = _normalise_text(response)
        matches = [float(_normalise_text(term) in response_text) for term in grader["terms"]]
        coverage = sum(matches) / len(matches)
        correctness = float(coverage == 1.0)

    criteria = [str(item) for item in task.get("criteria", []) if str(item).strip()]
    response_text = _normalise_text(response)
    reasoning_coverage = (
        sum(float(_normalise_text(term) in response_text) for term in criteria) / len(criteria)
        if criteria
        else correctness
    )
    has_reasoning = bool(re.search(r"<reasoning>.+?</reasoning>", response, re.DOTALL | re.IGNORECASE))
    verification = float(bool(re.search(r"check|verif|confirm|test", response, re.IGNORECASE)))
    reasoning_quality = min(1.0, reasoning_coverage * 0.6 + float(has_reasoning) * 0.25 + verification * 0.15)
    return {
        "correctness": correctness,
        "coverage": coverage,
        "reasoning_quality": reasoning_quality,
        "verification": verification,
    }


def scaffold_tasks(description: str) -> list[dict[str, Any]]:
    goal = description.strip()
    lowered = goal.casefold()
    operations = {
        "add": (lambda left, right: left + right, "+", "subtraction"),
        "sum": (lambda left, right: left + right, "+", "subtraction"),
        "subtract": (lambda left, right: left - right, "−", "addition"),
        "multiply": (lambda left, right: left * right, "×", "division"),
    }
    operation = next((value for keyword, value in operations.items() if keyword in lowered), None)
    if operation:
        calculate, symbol, verification = operation
        pairs = [(2, 5), (8, 3), (7, 6)]
        tasks = []
        for index, (left, right) in enumerate(pairs):
            result = calculate(left, right)
            tasks.append(
                {
                    "name": f"Worked {symbol} example {index + 1}",
                    "prompt": (
                        f"Compute {left} {symbol} {right}. Explain the calculation, verify it using {verification}, "
                        "and put only the numeric result inside the answer tag."
                    ),
                    "expected_answer": str(result),
                    "ideal_response": (
                        f"<reasoning>{left} {symbol} {right} = {result}. I verify the result using {verification}."
                        f"</reasoning><answer>{result}</answer>"
                    ),
                    "criteria": ["verify"],
                    "grader": {"type": "numeric", "target": result, "tolerance": 0},
                    "difficulty": ("easy", "medium", "hard")[index],
                }
            )
        return tasks

    words = [
        word.casefold()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", goal)
        if word.casefold() not in {"teach", "model", "with", "that", "this", "from", "their", "should"}
    ]
    terms = list(dict.fromkeys(words))[:3] or ["explain", "verify"]
    focuses = ("a straightforward case", "an edge case", "a transfer case")
    tasks = []
    for index, focus in enumerate(focuses):
        answer = f"Demonstration for {focus}: {goal}. " + ", ".join(terms)
        tasks.append(
            {
                "name": focus.title(),
                "prompt": f"Demonstrate the following capability for {focus}: {goal}. Explain and verify the response.",
                "expected_answer": "",
                "ideal_response": f"<reasoning>I address {focus} and verify every requirement.</reasoning><answer>{answer}</answer>",
                "criteria": terms,
                "grader": {"type": "contains_all", "terms": terms},
                "difficulty": ("easy", "medium", "hard")[index],
            }
        )
    return tasks
