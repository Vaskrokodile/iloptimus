"""Read-only prompt skills and deterministic automatic routing.

Only Markdown is loaded from the packaged skill directories.  Scripts and other
assets shipped beside a skill are deliberately outside this module's execution
boundary: a local model can learn from a skill, but cannot use it to run code.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

SKILLS_ROOT = Path(__file__).resolve().parents[1] / "resources" / "skills"

SKILL_KEYWORDS: dict[str, set[str]] = {
    "frontend-design": {
        "frontend",
        "website",
        "webapp",
        "react",
        "css",
        "component",
        "dashboard",
        "landing",
        "layout",
        "interface",
        "ui",
        "ux",
    },
    "playwright": {
        "playwright",
        "browser test",
        "e2e",
        "end to end",
        "screenshot",
        "navigate",
        "click",
        "form",
        "web test",
    },
    "security-best-practices": {
        "security review",
        "secure code",
        "vulnerability",
        "threat model",
        "owasp",
        "xss",
        "csrf",
        "injection",
        "authentication",
        "authorization",
    },
    "jupyter-notebook": {
        "jupyter",
        "notebook",
        "ipynb",
        "data analysis",
        "experiment",
        "tutorial notebook",
        "research notebook",
    },
    "knowledge-dataset": {
        "dataset",
        "fine tune",
        "finetune",
        "qlora",
        "lora",
        "train on",
        "learn about",
        "research this",
        "grounded data",
        "evaluation set",
    },
    "test-time-artifact": {
        "test time compute",
        "test-time compute",
        "/ttc",
        "artifact",
        "runnable code",
        "verify artifact",
        "failed generation",
    },
}

SKILL_MIN_SCORES = {
    "frontend-design": 2,
    "playwright": 2,
    "security-best-practices": 1,
    "jupyter-notebook": 1,
    "knowledge-dataset": 1,
    "test-time-artifact": 1,
}


@dataclass(frozen=True)
class PromptSkill:
    id: str
    name: str
    description: str
    source: str
    instructions: str

    def public(self) -> dict[str, str]:
        payload = asdict(self)
        payload.pop("instructions")
        return payload


def _frontmatter(markdown: str) -> tuple[dict[str, str], str]:
    if not markdown.startswith("---\n"):
        return {}, markdown
    end = markdown.find("\n---\n", 4)
    if end < 0:
        return {}, markdown
    metadata: dict[str, str] = {}
    for line in markdown[4:end].splitlines():
        key, separator, value = line.partition(":")
        if separator and re.fullmatch(r"[A-Za-z0-9_-]+", key.strip()):
            metadata[key.strip()] = value.strip().strip("'\"")
    return metadata, markdown[end + 5 :].strip()


def list_prompt_skills(root: Path = SKILLS_ROOT) -> list[PromptSkill]:
    skills: list[PromptSkill] = []
    if not root.exists():
        return skills
    for path in sorted(root.glob("*/SKILL.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        metadata, instructions = _frontmatter(text)
        skill_id = path.parent.name
        skills.append(
            PromptSkill(
                id=skill_id,
                name=metadata.get("name", skill_id.replace("-", " ").title()),
                description=metadata.get("description", "Packaged prompt guidance"),
                source=("anthropics/skills" if skill_id == "frontend-design" else "openai/skills"),
                instructions=instructions,
            )
        )
    return skills


def route_prompt_skills(prompt: str, *, limit: int = 2) -> list[PromptSkill]:
    """Select relevant skills without asking a small model to route itself."""
    normalized = " ".join(prompt.lower().split())
    scored: list[tuple[int, PromptSkill]] = []
    for skill in list_prompt_skills():
        score = sum(
            3 if " " in keyword else 1 for keyword in SKILL_KEYWORDS.get(skill.id, set()) if keyword in normalized
        )
        if score >= SKILL_MIN_SCORES.get(skill.id, 1):
            scored.append((score, skill))
    scored.sort(key=lambda pair: (-pair[0], pair[1].id))
    return [skill for _, skill in scored[:limit]]


def skill_prompt(skills: list[PromptSkill], *, max_chars: int = 18_000) -> str:
    if not skills:
        return ""
    sections = [
        "The following trusted, read-only skills are active. Use their guidance "
        "when it applies, but never claim that their scripts or external actions ran."
    ]
    remaining = max_chars - len(sections[0])
    for skill in skills:
        heading = f"\n\n## Active skill: {skill.name}\n"
        content = skill.instructions[: max(0, remaining - len(heading))]
        sections.append(heading + content)
        remaining -= len(heading) + len(content)
        if remaining <= 0:
            break
    return "".join(sections)
