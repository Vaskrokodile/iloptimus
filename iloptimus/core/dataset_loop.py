"""Recursive dataset quality optimizer.

Treats the dataset as the optimization target, not the model weights.
The model adapter is the measurement instrument: train on a dataset,
evaluate per-capability impact, and feed the failure analysis back into
targeted re-curation. Each iteration records what changed and what
effect it had, so the system compounds improvements across runs.

This is not self-play or unsupervised bootstrapping. Every training row
has provenance, every capability has a coverage audit, and every
iteration has a measured before/after score. An iteration that does not
improve is rolled back, not silently kept.

Loop structure::

    iteration 0:  curate(dataset_v0)  → train → evaluate → record
    iteration 1:  analyze_failures(v0) → targeted_search → re_curate(v1) → train → evaluate → compare
    ...
    stop when: improvement < threshold for N consecutive iterations, or budget exhausted

The impact log records per-capability deltas so the system learns which
curation strategies help and which produce noise.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .dataset_tools import (
    curate_dataset,
    load_source_bundle,
    save_source_bundle,
)
from .storage import app_home, atomic_write_json


@dataclass
class LoopConfig:
    """Controls the recursive dataset improvement cycle."""

    max_iterations: int = 12
    # Below this measured improvement, the iteration is not accepted.
    min_improvement: float = 0.01
    # Stop after this many consecutive non-improving iterations.
    convergence_window: int = 4
    # Total wall-clock budget across all iterations (10 hours by default —
    # the loop is designed to run for a long time seeking a leap).
    budget_seconds: float = 36000.0
    # Target dataset size — small and dense, not large and noisy.
    target_examples: int = 80
    # How much extra weight to give weak capabilities during re-curation.
    priority_boost: float = 0.3
    # Revert to the previous dataset when an iteration regresses.
    rollback_on_regression: bool = True
    # Drop rows for capabilities that are over-represented but not helping.
    prune_stale_capabilities: bool = True
    # Minimum rows per capability after re-curation.
    min_capability_rows: int = 4


@dataclass
class CapabilityImpact:
    """Per-capability measured change between two iterations."""

    capability: str
    baseline_score: float
    current_score: float
    delta: float
    row_count: int
    source_count: int
    verdict: str  # "improved", "regressed", "flat", "saturated"

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoopIteration:
    """One complete cycle of curate → train → evaluate → analyze."""

    iteration: int
    dataset_hash: str
    dataset_rows: int
    capability_scores: dict[str, float]
    overall_score: float
    improvement: float
    changes: dict[str, Any]
    capability_impacts: list[dict[str, Any]]
    accepted: bool
    adapter_path: str
    elapsed_seconds: float
    curation_manifest: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoopResult:
    """Final outcome of the recursive loop."""

    iterations: list[dict[str, Any]]
    best_iteration: int
    best_score: float
    best_dataset_hash: str
    best_adapter_path: str
    converged: bool
    stop_reason: str
    total_elapsed: float
    impact_log: list[dict[str, Any]]

    def public(self) -> dict[str, Any]:
        return asdict(self)


def loop_dir(session_id: str) -> Path:
    return app_home() / "learning" / session_id / "dataset-loop"


def _dataset_hash(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        sorted(
            (
                {
                    "prompt": str(row.get("prompt") or ""),
                    "ideal_response": str(row.get("ideal_response") or "")[:200],
                    "features": sorted(str(f) for f in row.get("features", [])),
                    "quality_score": round(float(row.get("quality_score") or 0.0), 4),
                }
                for row in rows
            ),
            key=lambda item: item["prompt"],
        ),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _capability_breakdown(
    rows: list[dict[str, Any]], requested_features: list[str]
) -> dict[str, dict[str, Any]]:
    """Count rows, sources, and mean quality per capability."""
    breakdown: dict[str, dict[str, Any]] = {}
    for feature in requested_features:
        matching = [row for row in rows if feature in row.get("features", [])]
        sources = {
            str(row.get("source_url") or "")
            for row in matching
            if row.get("source_url")
        }
        qualities = [float(row.get("quality_score") or 0.0) for row in matching]
        breakdown[feature] = {
            "rows": len(matching),
            "sources": len(sources),
            "mean_quality": round(sum(qualities) / max(1, len(qualities)), 4),
        }
    return breakdown


def analyze_capability_impacts(
    prev_scores: dict[str, float],
    curr_scores: dict[str, float],
    dataset_rows: list[dict[str, Any]],
    requested_features: list[str],
    *,
    saturation_threshold: float = 0.92,
) -> list[CapabilityImpact]:
    """Compute per-capability deltas and classify each as improved/regressed/flat/saturated.

    The saturation verdict fires when a capability is already near-perfect
    (above saturation_threshold) — adding more data for it is unlikely to help
    and the re-curation step should redirect effort to weaker capabilities.
    """
    breakdown = _capability_breakdown(dataset_rows, requested_features)
    impacts: list[CapabilityImpact] = []
    for capability in requested_features:
        baseline = float(prev_scores.get(capability, 0.0))
        current = float(curr_scores.get(capability, 0.0))
        delta = round(current - baseline, 4)
        info = breakdown.get(capability, {})
        if current >= saturation_threshold:
            verdict = "saturated"
        elif delta > 0.01:
            verdict = "improved"
        elif delta < -0.01:
            verdict = "regressed"
        else:
            verdict = "flat"
        impacts.append(
            CapabilityImpact(
                capability=capability,
                baseline_score=round(baseline, 4),
                current_score=round(current, 4),
                delta=delta,
                row_count=int(info.get("rows", 0)),
                source_count=int(info.get("sources", 0)),
                verdict=verdict,
            )
        )
    return impacts


def plan_re_curation(
    impacts: list[CapabilityImpact],
    config: LoopConfig,
) -> dict[str, Any]:
    """Decide what to search for and how to rebalance the next dataset.

    Returns a plan with:
    - weak_capabilities: capabilities that regressed or stayed flat (search for more data)
    - saturated_capabilities: capabilities already near-perfect (reduce their row quota)
    - priority_features: ordered list for the next curate_dataset call
    - search_hints: capability-specific search query suggestions
    """
    weak = [
        impact.capability
        for impact in impacts
        if impact.verdict in ("regressed", "flat")
        and impact.row_count < config.min_capability_rows * 3
    ]
    saturated = [
        impact.capability for impact in impacts if impact.verdict == "saturated"
    ]
    # Priority: weak capabilities first, then flat ones, then everything else.
    # Saturated capabilities go last so curate_dataset's rare-first reservation
    # doesn't waste slots on them.
    priority = weak + [
        impact.capability
        for impact in impacts
        if impact.verdict == "flat" and impact.capability not in weak
    ]
    remaining = [
        impact.capability
        for impact in impacts
        if impact.capability not in priority and impact.capability not in saturated
    ]
    priority.extend(remaining)
    search_hints = {
        capability: [
            f"{capability} complete implementation example GitHub MIT",
            f"{capability} official API documentation tutorial",
            f"{capability} minimal working source code",
        ]
        for capability in weak
    }
    return {
        "weak_capabilities": weak,
        "saturated_capabilities": saturated,
        "priority_features": priority,
        "search_hints": search_hints,
        "prune_saturated": config.prune_stale_capabilities and bool(saturated),
    }


def compute_token_density(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure how information-dense the dataset is per row.

    Dense rows have high unique-token ratio, low repetition, and
    substantial length. This is a mechanical proxy for "each row
    teaches something specific" without spending model tokens.
    """
    if not rows:
        return {"mean_density": 0.0, "mean_chars": 0, "dense_fraction": 0.0}
    densities: list[float] = []
    char_counts: list[int] = []
    for row in rows:
        response = str(row.get("ideal_response") or "")
        tokens = response.split()
        unique = set(tokens)
        density = len(unique) / max(1, len(tokens))
        densities.append(density)
        char_counts.append(len(response))
    mean_density = sum(densities) / len(densities)
    mean_chars = sum(char_counts) / len(char_counts)
    dense_fraction = sum(1 for d in densities if d > 0.6) / len(densities)
    return {
        "mean_density": round(mean_density, 4),
        "mean_chars": round(mean_chars, 1),
        "dense_fraction": round(dense_fraction, 4),
    }


def check_convergence(
    history: list[LoopIteration], config: LoopConfig
) -> tuple[bool, str]:
    """Decide whether the loop should stop.

    Returns (should_stop, reason). The loop stops when:
    - max_iterations reached
    - budget exhausted (cumulative elapsed time across all iterations)
    - N consecutive non-improving iterations (convergence_window)
    """
    if len(history) >= config.max_iterations:
        return True, f"Reached max_iterations ({config.max_iterations})"
    total_elapsed = sum(item.elapsed_seconds for item in history)
    if total_elapsed >= config.budget_seconds:
        return True, f"Budget exhausted ({config.budget_seconds:.0f}s)"
    if len(history) >= config.convergence_window + 1:
        recent = history[-(config.convergence_window + 1) :]
        non_improving = sum(
            1 for item in recent if item.improvement < config.min_improvement
        )
        if non_improving >= config.convergence_window:
            return True, (
                f"No improvement above {config.min_improvement} for "
                f"{config.convergence_window} consecutive iterations"
            )
    return False, ""


def select_best_iteration(history: list[LoopIteration]) -> tuple[int, LoopIteration]:
    """Return (index, iteration) for the highest-scoring accepted iteration."""
    accepted = [item for item in history if item.accepted]
    if not accepted:
        accepted = history
    best = max(accepted, key=lambda item: item.overall_score)
    return history.index(best), best


def run_curate_for_loop(
    workspace_id: str,
    *,
    task: str,
    artifact_kind: str,
    requested_features: list[str],
    priority_features: list[str] | None = None,
    config: LoopConfig,
    chunk_chars: int = 520,
    minimum_response_chars: int = 280,
) -> dict[str, Any]:
    """Wrap curate_dataset with loop-aware defaults (small, dense datasets)."""
    return curate_dataset(
        workspace_id,
        task=task,
        artifact_kind=artifact_kind,
        requested_features=requested_features,
        priority_features=priority_features,
        assembled_examples=max(64, config.target_examples * 2),
        expanded_examples=max(96, config.target_examples * 3),
        maximum_rows=config.target_examples,
        chunk_chars=chunk_chars,
        minimum_response_chars=minimum_response_chars,
        minimum_quality_score=0.5,
    )


def save_iteration_record(session_id: str, iteration: LoopIteration) -> Path:
    """Persist one iteration's full record for audit and replay."""
    directory = loop_dir(session_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"iteration-{iteration.iteration:03d}.json"
    atomic_write_json(path, iteration.public())
    return path


def load_loop_history(session_id: str) -> list[dict[str, Any]]:
    """Load all iteration records for a session, ordered by iteration."""
    directory = loop_dir(session_id)
    if not directory.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("iteration-*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return records


def save_loop_result(session_id: str, result: LoopResult) -> Path:
    directory = loop_dir(session_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "loop-result.json"
    atomic_write_json(path, result.public())
    return path


def build_iteration_record(
    iteration: int,
    dataset_rows: list[dict[str, Any]],
    capability_scores: dict[str, float],
    prev_overall: float,
    changes: dict[str, Any],
    capability_impacts: list[CapabilityImpact],
    accepted: bool,
    adapter_path: str,
    elapsed: float,
    curation_manifest: dict[str, Any],
) -> LoopIteration:
    overall = sum(capability_scores.values()) / max(1, len(capability_scores))
    return LoopIteration(
        iteration=iteration,
        dataset_hash=_dataset_hash(dataset_rows),
        dataset_rows=len(dataset_rows),
        capability_scores={k: round(v, 4) for k, v in capability_scores.items()},
        overall_score=round(overall, 4),
        improvement=round(overall - prev_overall, 4),
        changes=changes,
        capability_impacts=[impact.public() for impact in capability_impacts],
        accepted=accepted,
        adapter_path=adapter_path,
        elapsed_seconds=round(elapsed, 3),
        curation_manifest=curation_manifest,
    )


def merge_new_sources(
    workspace_id: str, new_sources: list[dict[str, str]]
) -> dict[str, Any]:
    """Add newly researched sources to the workspace's provenance pool."""
    if not new_sources:
        return {"added": 0, "total": len(load_source_bundle(workspace_id))}
    existing = load_source_bundle(workspace_id)
    existing_hashes = {item.get("sha256") for item in existing}
    novel = [
        s
        for s in new_sources
        if hashlib.sha256(str(s.get("text", "")).encode()).hexdigest()
        not in existing_hashes
    ]
    save_source_bundle(workspace_id, novel)
    return {"added": len(novel), "total": len(load_source_bundle(workspace_id))}


def summarize_impact_log(history: list[LoopIteration]) -> list[dict[str, Any]]:
    """Build a compact per-iteration ablation log for analysis.

    Each entry records what changed, what the per-capability delta was,
    and whether the iteration was accepted. This is the dataset-quality
    gradient: it shows which curation strategies compound and which
    produce noise.
    """
    log: list[dict[str, Any]] = []
    for item in history:
        helped = [i["capability"] for i in item.capability_impacts if i["delta"] > 0.01]
        hurt = [i["capability"] for i in item.capability_impacts if i["delta"] < -0.01]
        log.append(
            {
                "iteration": item.iteration,
                "improvement": item.improvement,
                "accepted": item.accepted,
                "dataset_rows": item.dataset_rows,
                "changes": item.changes,
                "helped_capabilities": helped,
                "hurt_capabilities": hurt,
                "overall_score": item.overall_score,
            }
        )
    return log
