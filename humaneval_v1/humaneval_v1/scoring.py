"""IL scoring for HumanEval: anti-laziness + efficiency-aware reasoning-quality shaping.

Same formula as il-coding-v1:

    final = correctness * (0.6 + 0.4 * reasoning_quality)

Reuses the same scoring dimensions (coverage, efficiency, verification, no-filler)
and anti-laziness detection adapted for HumanEval's function-implementation format.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

VERIFICATION_KEYWORDS = [
    "verify", "check", "let me check", "confirm", "trace through",
    "let me trace", "test this", "if i run", "let me verify",
    "to confirm", "double-check", "let me walk through", "mentally",
    "let me verify by", "tracing the execution", "walk through",
    "let me simulate", "step through", "let me step through",
    "edge case", "boundary", "empty list", "empty string",
]

FILLER_PHRASES = [
    "let me think about this",
    "this is an interesting problem",
    "i need to analyze",
    "let me consider",
    "this is a complex",
    "i should first understand",
    "let me start by understanding",
    "this requires careful",
    "i'll need to think",
    "let me approach this",
    "this seems like",
    "i need to figure out",
    "let me break this down",
    "first, let me understand",
    "i should approach",
]

REASONING_RE = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
CODE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


@dataclass
class ReasoningBreakdown:
    coverage: float = 0.0
    concepts_found: list[str] = field(default_factory=list)
    concepts_missing: list[str] = field(default_factory=list)
    token_efficiency: float = 0.0
    tokens_used: int = 0
    token_budget: int = 0
    verification: float = 0.0
    verification_evidence: list[str] = field(default_factory=list)
    filler_penalty: float = 0.0
    filler_found: list[str] = field(default_factory=list)
    reasoning_quality: float = 0.0
    reasoning_length: int = 0


def extract_reasoning(response: str) -> str:
    match = REASONING_RE.search(response)
    return match.group(1).strip() if match else ""


def extract_answer(response: str) -> str:
    match = ANSWER_RE.search(response)
    if match:
        return match.group(1).strip()
    match = re.search(r"</reasoning>\s*(.*)", response, re.DOTALL)
    return match.group(1).strip() if match else response.strip()


def extract_code(response: str) -> str:
    """First ```python block from the answer section, or the answer verbatim."""
    answer = extract_answer(response)
    matches = CODE_RE.findall(answer)
    if matches:
        return matches[0]
    # Fallback: look for code block anywhere in the response
    matches = CODE_RE.findall(response)
    return matches[0] if matches else answer


def score_reasoning_quality(
    response: str,
    expected_concepts: list[str],
    token_budget: int = 600,
    correctness: float = 0.0,
) -> tuple[float, ReasoningBreakdown]:
    reasoning = extract_reasoning(response)
    reasoning_lower = reasoning.lower()
    breakdown = ReasoningBreakdown(token_budget=token_budget, reasoning_length=len(reasoning))

    # 1. Coverage
    breakdown.concepts_found = [c for c in expected_concepts if c.lower() in reasoning_lower]
    breakdown.concepts_missing = [c for c in expected_concepts if c.lower() not in reasoning_lower]
    breakdown.coverage = (
        len(breakdown.concepts_found) / len(expected_concepts) if expected_concepts else 1.0
    )

    # 2. Token efficiency (only rewarded for mostly-correct answers)
    breakdown.tokens_used = len(reasoning) // 4
    if correctness >= 0.5:
        if breakdown.tokens_used <= token_budget:
            ratio = breakdown.tokens_used / max(1, token_budget)
            if 0.5 <= ratio <= 0.9:
                breakdown.token_efficiency = 1.0
            elif ratio < 0.5:
                breakdown.token_efficiency = 0.5 + ratio
            else:
                breakdown.token_efficiency = max(0.7, 1.0 - (ratio - 0.9) * 0.5)
        else:
            overage = (breakdown.tokens_used - token_budget) / token_budget
            breakdown.token_efficiency = max(0.2, 1.0 - overage * 0.6)
    else:
        breakdown.token_efficiency = 0.0

    # 3. Verification
    breakdown.verification_evidence = [kw for kw in VERIFICATION_KEYWORDS if kw in reasoning_lower]
    n_checks = len(breakdown.verification_evidence)
    breakdown.verification = min(1.0, 0.3 + n_checks * 0.15) if n_checks > 0 else 0.0

    # 4. No-filler
    breakdown.filler_found = [p for p in FILLER_PHRASES if p in reasoning_lower]
    n_filler = len(breakdown.filler_found)
    breakdown.filler_penalty = min(1.0, n_filler * 0.15)

    breakdown.reasoning_quality = (
        breakdown.coverage * 0.40
        + breakdown.token_efficiency * 0.30
        + breakdown.verification * 0.20
        + (1.0 - breakdown.filler_penalty) * 0.10
    )
    breakdown.reasoning_quality = max(0.0, min(1.0, breakdown.reasoning_quality))
    return breakdown.reasoning_quality, breakdown


def compute_final_score(correctness: float, reasoning_quality: float) -> float:
    """final = correctness * (0.6 + 0.4 * reasoning_quality)"""
    return correctness * (0.6 + 0.4 * reasoning_quality)


# ---------------------------------------------------------------------------
# Anti-laziness detection
# ---------------------------------------------------------------------------

@dataclass
class LazinessReport:
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)


def detect_laziness(
    code: str,
    required_params: list[str],
    required_constructs: list[str],
) -> LazinessReport:
    """Static + structural laziness detection. Returns a 0..1 score."""
    report = LazinessReport()
    reasons: list[str] = []

    if not code.strip():
        report.score = 1.0
        report.reasons = ["empty code"]
        return report

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return report

    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

    all_body_src = "\n".join(ast.get_source_segment(code, n) or "" for n in (*funcs, *classes))
    for p in required_params:
        if p not in all_body_src:
            reasons.append(f"param '{p}' never referenced")

    src_lower = code.lower()
    for c in required_constructs:
        if c.lower() not in src_lower:
            reasons.append(f"construct '{c}' missing")

    for fn in funcs:
        body = fn.body
        if len(body) == 1:
            stmt = body[0]
            if isinstance(stmt, ast.Pass):
                reasons.append(f"{fn.name}: body is just 'pass'")
            elif isinstance(stmt, ast.Return) and (
                stmt.value is None
                or (isinstance(stmt.value, ast.Constant) and stmt.value.value in (None, 0, "", False))
            ):
                reasons.append(f"{fn.name}: returns constant")

    for fn in funcs:
        body = fn.body
        if len(body) == 1 and isinstance(body[0], ast.Return):
            if isinstance(body[0].value, ast.Constant):
                reasons.append(f"{fn.name}: constant-only return (ignores input)")

    report.reasons = reasons
    report.score = min(1.0, len(reasons) * 0.25)
    return report


__all__ = [
    "ReasoningBreakdown",
    "extract_reasoning",
    "extract_answer",
    "extract_code",
    "score_reasoning_quality",
    "compute_final_score",
    "LazinessReport",
    "detect_laziness",
]
