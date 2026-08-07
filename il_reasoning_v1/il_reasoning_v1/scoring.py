"""IL reasoning-quality scoring (shared shape across IL tasksets).

    final = correctness * (0.6 + 0.4 * reasoning_quality)

Reasoning quality (4 dimensions):
    coverage     (40%) — did the model reason about the right concepts?
    efficiency   (30%) — within token budget?
    verification (20%) — did the model check its work?
    no_filler    (10%) — avoided generic boilerplate?
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

VERIFICATION_KEYWORDS = [
    "verify", "check", "let me check", "confirm", "trace through",
    "let me trace", "test this", "if i run", "let me verify",
    "to confirm", "double-check", "let me walk through", "mentally",
    "let me verify by", "tracing the execution", "walk through",
    "let me simulate", "step through", "let me step through",
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


def score_reasoning_quality(
    response: str,
    expected_concepts: list[str],
    token_budget: int = 500,
    correctness: float = 0.0,
) -> tuple[float, ReasoningBreakdown]:
    reasoning = extract_reasoning(response)
    reasoning_lower = reasoning.lower()
    breakdown = ReasoningBreakdown(token_budget=token_budget, reasoning_length=len(reasoning))

    breakdown.concepts_found = [c for c in expected_concepts if c.lower() in reasoning_lower]
    breakdown.concepts_missing = [c for c in expected_concepts if c.lower() not in reasoning_lower]
    breakdown.coverage = (
        len(breakdown.concepts_found) / len(expected_concepts) if expected_concepts else 1.0
    )

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

    breakdown.verification_evidence = [kw for kw in VERIFICATION_KEYWORDS if kw in reasoning_lower]
    n_checks = len(breakdown.verification_evidence)
    breakdown.verification = min(1.0, 0.3 + n_checks * 0.15) if n_checks > 0 else 0.0

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
    return correctness * (0.6 + 0.4 * reasoning_quality)


__all__ = [
    "ReasoningBreakdown",
    "extract_reasoning",
    "score_reasoning_quality",
    "compute_final_score",
]
