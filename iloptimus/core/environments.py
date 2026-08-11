"""Persistent no-code IL/RL environment specifications and package export."""

from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from .environment_framework import FRAMEWORK_VERSION, builder_skill, normalise_grader, scaffold_tasks
from .storage import environments_dir


def _environments_dir():
    return environments_dir()


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned[:48] or "custom-environment"


def _normalise_task(task: dict[str, Any], index: int) -> dict[str, Any]:
    prompt = str(task.get("prompt", "")).strip()
    if not prompt:
        raise ValueError(f"Task {index + 1} needs a prompt")
    expected = str(task.get("expected_answer", "")).strip()
    answer_match = re.search(r"<answer>(.*?)</answer>", expected, re.DOTALL | re.IGNORECASE)
    if answer_match:
        expected = answer_match.group(1).strip()
    criteria = [str(item).strip() for item in task.get("criteria", []) if str(item).strip()]
    normalised = {
        "id": str(task.get("id") or f"task-{index + 1}"),
        "name": str(task.get("name") or f"Task {index + 1}").strip(),
        "prompt": prompt,
        "expected_answer": expected,
        "ideal_response": str(task.get("ideal_response") or "").strip(),
        "criteria": criteria,
        "difficulty": str(task.get("difficulty") or "medium"),
    }
    normalised["grader"] = normalise_grader({**task, **normalised})
    if not normalised["expected_answer"]:
        grader = normalised["grader"]
        if grader["type"] in {"exact", "numeric"}:
            normalised["expected_answer"] = str(grader["target"])
    if not normalised["ideal_response"]:
        answer = normalised["expected_answer"] or "A response containing: " + ", ".join(criteria)
        normalised["ideal_response"] = (
            "<reasoning>I will solve the task against each observable requirement and verify the result."
            f"</reasoning><answer>{answer}</answer>"
        )
    return normalised


def validate_environment(payload: dict[str, Any]) -> dict[str, Any]:
    name = str(payload.get("name", "")).strip()
    goal = str(payload.get("goal", "")).strip()
    if len(name) < 3:
        raise ValueError("Environment name must be at least 3 characters")
    if len(goal) < 12:
        raise ValueError("Describe the learning goal in at least 12 characters")

    mode = str(payload.get("mode", "IL")).upper()
    if mode not in {"IL", "RL"}:
        raise ValueError("Mode must be IL or RL")

    tasks = [_normalise_task(task, index) for index, task in enumerate(payload.get("tasks", []))]
    if not tasks:
        tasks = [
            {
                "id": "task-1",
                "name": "Core behavior check",
                "prompt": goal,
                "expected_answer": "",
                "criteria": ["addresses the goal", "explains the approach", "checks the result"],
                "grader": {
                    "type": "contains_all",
                    "terms": ["addresses the goal", "explains the approach"],
                },
                "difficulty": "medium",
            }
        ]

    reward = payload.get("reward", {})
    correctness = float(reward.get("correctness", 0.6))
    reasoning = float(reward.get("reasoning", 0.3))
    efficiency = float(reward.get("efficiency", 0.1))
    total = correctness + reasoning + efficiency
    if total <= 0:
        raise ValueError("Reward weights must add up to more than zero")

    env_id = str(payload.get("id") or f"{_slug(name)}-{uuid.uuid4().hex[:6]}")
    now = int(time.time())
    return {
        "id": env_id,
        "taskset_id": f"user-{env_id}",
        "name": name,
        "mode": mode,
        "goal": goal,
        "description": str(payload.get("description") or goal).strip(),
        "domain": str(payload.get("domain") or "reasoning").strip(),
        "interaction": {
            "observation": str(payload.get("interaction", {}).get("observation") or "A natural-language task prompt"),
            "action": str(payload.get("interaction", {}).get("action") or "A reasoned response with a final answer"),
            "max_steps": max(1, min(64, int(payload.get("interaction", {}).get("max_steps", 1)))),
        },
        "reward": {
            "correctness": round(correctness / total, 4),
            "reasoning": round(reasoning / total, 4),
            "efficiency": round(efficiency / total, 4),
            "method": str(reward.get("method") or "criteria"),
        },
        "tasks": tasks,
        "status": "ready",
        "executable": True,
        "framework_version": FRAMEWORK_VERSION,
        "template_id": f"{mode.lower()}-verifiable-v{FRAMEWORK_VERSION}",
        "builder": {
            "model_id": str(payload.get("builder", {}).get("model_id") or "manual"),
            "used_model_output": bool(payload.get("builder", {}).get("used_model_output", False)),
        },
        "created_at": int(payload.get("created_at") or now),
        "updated_at": now,
        "version": FRAMEWORK_VERSION,
    }


def _environment_path(env_id: str) -> Path:
    return _environments_dir() / env_id / "environment.json"


def get_environment(env_id: str) -> dict[str, Any] | None:
    path = _environment_path(env_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_environments() -> list[dict[str, Any]]:
    root = _environments_dir()
    if not root.exists():
        return []
    environments = []
    for path in root.glob("*/environment.json"):
        try:
            environments.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return sorted(environments, key=lambda item: item.get("updated_at", 0), reverse=True)


def _package_source(spec: dict[str, Any]) -> str:
    data = repr(spec)
    return f'''"""Generated by the IL Optimus no-code environment builder."""
from iloptimus.core.environment_framework import score_task

SPEC = {data}

class GeneratedTask:
    def __init__(self, data: dict):
        self.data = data

    @property
    def prompt(self) -> str:
        return self.data["prompt"]

    def score(self, response: str) -> float:
        metrics = score_task(self.data, response)
        weights = SPEC["reward"]
        score = metrics["correctness"] * weights["correctness"]
        score += metrics["reasoning_quality"] * weights["reasoning"]
        if metrics["correctness"]:
            score += weights["efficiency"]
        return min(1.0, score)

class GeneratedTaskset:
    def load(self):
        return [GeneratedTask(task) for task in SPEC["tasks"]]
'''


def save_environment(payload: dict[str, Any]) -> dict[str, Any]:
    spec = validate_environment(payload)
    folder = _environments_dir() / spec["id"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "environment.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    (folder / "taskset.py").write_text(_package_source(spec), encoding="utf-8")
    (folder / "README.md").write_text(
        f"# {spec['name']}\n\n{spec['description']}\n\n"
        f"Generated as an executable {spec['mode']} environment by IL Optimus framework v{FRAMEWORK_VERSION}.\n\n"
        "This folder contains the declarative environment specification, deterministic graders, and an executable taskset adapter.\n",
        encoding="utf-8",
    )
    (folder / "SKILL.md").write_text(builder_skill(), encoding="utf-8")
    return spec


def delete_environment(env_id: str) -> bool:
    folder = _environments_dir() / env_id
    if not folder.exists():
        return False
    shutil.rmtree(folder)
    return True


def draft_environment(mode: str, description: str, generated: dict[str, Any] | None = None) -> dict[str, Any]:
    mode = mode.upper()
    description = description.strip()
    base = generated or {}
    words = [word for word in re.findall(r"[A-Za-z0-9]+", description) if len(word) > 3]
    title = str(base.get("name") or " ".join(words[:5]) or f"Custom {mode} environment").title()
    tasks = base.get("tasks") or scaffold_tasks(description)
    return {
        **base,
        "name": title[:64],
        "mode": mode,
        "goal": str(base.get("goal") or description),
        "description": str(base.get("description") or description),
        "domain": str(base.get("domain") or "reasoning"),
        "tasks": tasks,
    }
