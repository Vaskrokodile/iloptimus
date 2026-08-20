"""Verified failure-pattern skills and retrieval memory for test-time compute."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .storage import app_home, atomic_write_json


@dataclass(frozen=True)
class FailureSkill:
    id: str
    name: str
    description: str
    artifact_kind: str
    features: tuple[str, ...]
    failed_gates: tuple[str, ...]
    diagnostics: tuple[str, ...]
    anti_patterns: tuple[str, ...]
    checklist: tuple[str, ...]
    evidence_status: str
    source_session_id: str
    baseline_score: float
    adapted_score: float
    created_at: float
    uses: int = 0
    successful_uses: int = 0

    def public(self) -> dict[str, Any]:
        return asdict(self)


_GATE_RULES: dict[str, tuple[str, str]] = {
    "substantial": (
        "Do not pad or repeat source to meet a size gate.",
        "Budget complete scene, styling, interaction, lifecycle, and responsive sections before writing code.",
    ),
    "source_not_rendered_as_text": (
        "Do not emit escaped source or JavaScript as visible body text.",
        "Keep executable JavaScript inside script modules and verify the page body contains the rendered UI only.",
    ),
    "javascript_syntax": (
        "Do not return prose, markdown fences, or malformed partial modules.",
        "Return one complete entrypoint and run a syntax check before considering it finished.",
    ),
    "runtime_render": (
        "Do not treat static keyword presence as proof that the artifact runs.",
        "Serve the entrypoint, execute it in a browser, reject console exceptions, and require non-blank pixels.",
    ),
    "no_placeholders": (
        "Do not leave TODOs, placeholders, fake handlers, or implementation comments.",
        "Replace every stub with observable behavior before verification.",
    ),
}

_FEATURE_RULES: dict[str, str] = {
    "three.js": "Instantiate a renderer, scene, camera, render loop, and actual scene objects.",
    "voxel": "Create observable voxel geometry, preferably instanced boxes or equivalent repeated cells.",
    "shader": "Provide executable vertex and fragment shader source and bind changing uniforms where animation is requested.",
    "animation": "Create a live animation loop that updates scene state before rendering each frame.",
    "interaction": "Wire real pointer, keyboard, or camera controls to observable state changes.",
    "responsive": "Resize renderer and camera projection from the current viewport and device pixel ratio.",
    "island": "Build a readable land-water silhouette with terrain elevation and a shoreline.",
    "sakura": "Render recognizable blossom/tree forms and animated petals rather than naming them in comments.",
    "accessibility": "Add semantic roles, labels, keyboard behavior, and reduced-motion handling where applicable.",
}


def _root() -> Path:
    root = app_home() / "skill-memory"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")[:48] or "artifact-repair"


def _skill_markdown(skill: FailureSkill) -> str:
    anti_patterns = "\n".join(f"- {item}" for item in skill.anti_patterns)
    checklist = "\n".join(f"- [ ] {item}" for item in skill.checklist)
    evidence = "\n".join(f"- {item}" for item in skill.diagnostics)
    return (
        f"---\nname: {skill.name}\ndescription: {skill.description}\n---\n\n"
        "# Repair workflow\n\n"
        "Apply this compact, verifier-derived guardrail before generating a matching artifact. "
        "Treat it as a failure pattern, not as proof that a model has mastered the task.\n\n"
        f"## Avoid\n\n{anti_patterns}\n\n"
        f"## Completion gates\n\n{checklist}\n\n"
        f"## Evidence from the failed attempt\n\n{evidence}\n"
    )


def validate_failure_skill(skill: FailureSkill, markdown: str) -> list[str]:
    """Mechanically reject vague, malformed, or unevidenced generated skills."""
    errors: list[str] = []
    if not re.match(r"^---\nname: [a-z0-9-]+\ndescription: .+\n---\n", markdown):
        errors.append("invalid skill frontmatter")
    if not skill.diagnostics or not skill.failed_gates:
        errors.append("missing objective failure evidence")
    if not skill.anti_patterns or not skill.checklist:
        errors.append("missing repair rules")
    if len(markdown) > 12_000:
        errors.append("skill exceeds the retrieval budget")
    if any(not item.strip().endswith((".", "!")) for item in (*skill.anti_patterns, *skill.checklist)):
        errors.append("repair rules must be complete instructions")
    return errors


def build_failure_skill(
    *,
    session_id: str,
    contract: dict[str, Any],
    baseline: dict[str, Any],
    adapted: dict[str, Any] | None = None,
) -> FailureSkill:
    """Compile an on-the-fly skill from objective verifier failures without another model call."""
    adapted = adapted or {}
    latest = adapted if adapted else baseline
    hard_gates = dict(latest.get("hard_gates") or baseline.get("hard_gates") or {})
    failed_gates = tuple(sorted(key for key, passed in hard_gates.items() if not passed))
    feature_scores = dict(latest.get("feature_scores") or baseline.get("feature_scores") or {})
    # A partial keyword/static match is not mastery. Keep every feature below
    # the verifier's full score as a future completion gate.
    failed_features = tuple(sorted(key for key, score in feature_scores.items() if float(score) < 1.0))
    features = tuple(dict.fromkeys(str(item) for item in contract.get("requested_features", [])))
    anti_patterns: list[str] = []
    checklist: list[str] = []
    for gate in failed_gates:
        anti, check = _GATE_RULES.get(
            gate,
            (f"Do not ignore the {gate.replace('_', ' ')} verification failure.",
             f"Pass the {gate.replace('_', ' ')} verifier before completion."),
        )
        anti_patterns.append(anti)
        checklist.append(check)
    for feature in failed_features:
        checklist.append(_FEATURE_RULES.get(feature, f"Implement {feature} as observable runtime behavior."))
    diagnostics = tuple(
        dict.fromkeys(str(item).strip() for item in latest.get("diagnostics", []) if str(item).strip())
    )
    if not diagnostics:
        diagnostics = tuple(f"Failed verifier gate: {gate}" for gate in failed_gates)
    signature = json.dumps(
        {"artifact_kind": contract.get("artifact_kind"), "features": features, "gates": failed_gates},
        sort_keys=True,
    )
    skill_id = hashlib.sha256(signature.encode()).hexdigest()[:16]
    name = f"repair-{_slug(str(contract.get('artifact_kind') or 'artifact'))}-{skill_id[:8]}"
    return FailureSkill(
        id=skill_id,
        name=name,
        description=(
            "Apply verifier-derived repair gates when generating "
            f"{contract.get('artifact_kind', 'artifact')} work involving {', '.join(features) or 'runtime behavior'}."
        ),
        artifact_kind=str(contract.get("artifact_kind") or "artifact"),
        features=features,
        failed_gates=failed_gates,
        diagnostics=diagnostics,
        anti_patterns=tuple(dict.fromkeys(anti_patterns)),
        checklist=tuple(dict.fromkeys(checklist)),
        evidence_status="verified-failure-pattern",
        source_session_id=session_id,
        baseline_score=float(baseline.get("score") or 0.0),
        adapted_score=float(adapted.get("score") or 0.0),
        created_at=time.time(),
    )


def save_failure_skill(skill: FailureSkill) -> dict[str, Any]:
    folder = _root() / skill.id
    folder.mkdir(parents=True, exist_ok=True)
    existing_path = folder / "lesson.json"
    if existing_path.exists():
        try:
            existing = FailureSkill(**json.loads(existing_path.read_text(encoding="utf-8")))
            skill = FailureSkill(
                **{
                    **skill.public(),
                    "created_at": existing.created_at,
                    "uses": existing.uses,
                    "successful_uses": existing.successful_uses,
                }
            )
        except (OSError, TypeError, json.JSONDecodeError):
            pass
    markdown = _skill_markdown(skill)
    errors = validate_failure_skill(skill, markdown)
    if errors:
        raise ValueError("Failure skill validation failed: " + "; ".join(errors))
    atomic_write_json(existing_path, skill.public())
    (folder / "SKILL.md").write_text(markdown, encoding="utf-8")
    evidence_path = folder / "evidence.json"
    try:
        observations = json.loads(evidence_path.read_text(encoding="utf-8"))
        if not isinstance(observations, list):
            observations = []
    except (OSError, json.JSONDecodeError):
        observations = []
    observation = {
        "session_id": skill.source_session_id,
        "baseline_score": skill.baseline_score,
        "adapted_score": skill.adapted_score,
        "failed_gates": list(skill.failed_gates),
        "diagnostics": list(skill.diagnostics),
        "recorded_at": time.time(),
    }
    if not any(item.get("session_id") == skill.source_session_id for item in observations):
        observations.append(observation)
        atomic_write_json(evidence_path, observations)
    return {
        **skill.public(),
        "path": str(folder / "SKILL.md"),
        "evidence_observations": len(observations),
    }


def list_failure_skills() -> list[FailureSkill]:
    skills: list[FailureSkill] = []
    for path in _root().glob("*/lesson.json"):
        try:
            skills.append(FailureSkill(**json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, TypeError, json.JSONDecodeError):
            continue
    return sorted(skills, key=lambda item: (-item.successful_uses, -item.uses, -item.created_at))


def retrieve_failure_skills(
    *, artifact_kind: str, features: Iterable[str], limit: int = 4
) -> list[dict[str, Any]]:
    wanted = set(str(item) for item in features)
    ranked: list[tuple[float, FailureSkill]] = []
    for skill in list_failure_skills():
        overlap = wanted.intersection(skill.features)
        if skill.artifact_kind != artifact_kind or not overlap:
            continue
        coverage = len(overlap) / max(1, len(wanted))
        precision = len(overlap) / max(1, len(skill.features))
        proven_bonus = min(0.25, skill.successful_uses * 0.05)
        ranked.append((0.7 * coverage + 0.3 * precision + proven_bonus, skill))
    ranked.sort(key=lambda item: (-item[0], -item[1].successful_uses, -item[1].created_at))
    return [{**skill.public(), "retrieval_score": round(score, 4)} for score, skill in ranked[:limit]]


def mark_skill_use(skill_ids: Iterable[str], *, successful: bool) -> None:
    for skill_id in dict.fromkeys(str(item) for item in skill_ids):
        path = _root() / skill_id / "lesson.json"
        try:
            skill = FailureSkill(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, TypeError, json.JSONDecodeError):
            continue
        payload = skill.public()
        payload["uses"] = skill.uses + 1
        payload["successful_uses"] = skill.successful_uses + int(successful)
        atomic_write_json(path, payload)


def delete_failure_skill(skill_id: str) -> bool:
    """Delete a failure skill and all its artifacts. Returns True if deleted."""
    import shutil
    folder = _root() / str(skill_id)
    if not folder.exists() or not folder.is_dir():
        return False
    # Safety: only delete inside the skill-memory root
    if _root() not in folder.parents and folder != _root():
        return False
    shutil.rmtree(folder)
    return True


def skill_guardrails(skills: Iterable[dict[str, Any]], *, maximum_chars: int = 3_000) -> str:
    """Render only compact actionable gates into a local model prompt."""
    sections: list[str] = []
    for skill in skills:
        checklist = [str(item) for item in skill.get("checklist", [])]
        avoid = [str(item) for item in skill.get("anti_patterns", [])]
        block = "Avoid:\n" + "\n".join(f"- {item}" for item in avoid)
        block += "\nVerify:\n" + "\n".join(f"- {item}" for item in checklist)
        if sum(len(item) for item in sections) + len(block) > maximum_chars:
            break
        sections.append(block)
    return "\n\n".join(sections)
