"""Uncertainty-gated, persisted test-time research and adaptation sessions."""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .storage import app_home, atomic_write_json

UNCERTAINTY_PHRASES = (
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "cannot verify",
    "can't verify",
    "uncertain",
    "might be",
    "may be",
    "as far as i know",
    "knowledge cutoff",
    "i don't have access",
)

TIME_SENSITIVE_TERMS = (
    "latest", "today", "currently", "current ", "recent", "this week", "this month",
    "this year", "price", "weather", "news", "version", "release", "president", "ceo",
)


@dataclass(frozen=True)
class UncertaintyAssessment:
    score: float
    needs_research: bool
    explicit: bool
    time_sensitive: bool
    reasons: list[str]

    def public(self) -> dict[str, Any]:
        return asdict(self)


def assess_uncertainty(query: str, answer: str, *, tool_failed: bool = False) -> UncertaintyAssessment:
    """Conservative observable-signal detector; it never claims to prove correctness."""
    normalized_query = " ".join(query.lower().split())
    normalized_answer = " ".join(answer.lower().split())
    explicit = normalized_query.startswith("/learn ") or any(
        phrase in normalized_query
        for phrase in ("research this", "verify this", "learn about", "investigate this")
    )
    time_sensitive = any(term in normalized_query for term in TIME_SENSITIVE_TERMS)
    reasons: list[str] = []
    score = 0.0
    if explicit:
        score += 1.0
        reasons.append("The user explicitly requested research or learning")
    phrase_hits = [phrase for phrase in UNCERTAINTY_PHRASES if phrase in normalized_answer]
    if phrase_hits:
        score += min(0.72, 0.36 + 0.12 * len(phrase_hits))
        reasons.append("The model expressed uncertainty")
    if tool_failed:
        score += 0.42
        reasons.append("A required grounding tool failed")
    if time_sensitive and not re.search(r"https?://", answer):
        score += 0.52
        reasons.append("The question is time-sensitive but the answer has no source")
    if query.rstrip().endswith("?") and len(answer.strip()) < 70:
        score += 0.22
        reasons.append("The answer is unusually short for a knowledge question")
    if any(marker in normalized_answer for marker in ("tool_name", "tool_call", "```json")):
        score += 0.7
        reasons.append("The answer leaked an unresolved tool protocol")
    score = min(1.0, score)
    return UncertaintyAssessment(
        score=score,
        needs_research=explicit or score >= 0.58,
        explicit=explicit,
        time_sensitive=time_sensitive,
        reasons=reasons,
    )


def select_learning_method(assessment: UncertaintyAssessment, *, training_available: bool) -> str:
    """Choose the smallest scientifically appropriate intervention."""
    if assessment.time_sensitive or not training_available:
        return "retrieval"
    return "qlora-il"


@dataclass
class LearningSession:
    id: str
    model_id: str
    query: str
    initial_answer: str
    method: str
    reason: str
    status: str = "running"
    stage: str = "uncertainty-detected"
    progress: float = 0.03
    sources: list[dict[str, str]] = field(default_factory=list)
    dataset_path: str = ""
    environment_id: str = ""
    run_id: str = ""
    final_answer: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def public(self) -> dict[str, Any]:
        return asdict(self)


class LearningManager:
    def __init__(self, root: Path | None = None):
        self.root = root or app_home() / "learning"
        self.root.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, LearningSession] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._conditions: dict[str, asyncio.Condition] = {}
        self._load()

    def _load(self) -> None:
        for path in self.root.glob("*/session.json"):
            try:
                session = LearningSession(**json.loads(path.read_text(encoding="utf-8")))
            except (OSError, TypeError, json.JSONDecodeError):
                continue
            if session.status == "running":
                session.status = "failed"
                session.stage = "interrupted"
                session.error = "The app stopped before this learning session completed"
            self._sessions[session.id] = session
            event_path = path.parent / "events.jsonl"
            events: list[dict[str, Any]] = []
            if event_path.exists():
                for line in event_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            self._events[session.id] = events
            self._persist(session)

    def _folder(self, session_id: str) -> Path:
        return self.root / session_id

    def _persist(self, session: LearningSession) -> None:
        session.updated_at = time.time()
        atomic_write_json(self._folder(session.id) / "session.json", session.public())

    def create(self, model_id: str, query: str, answer: str, method: str, reason: str) -> LearningSession:
        session = LearningSession(uuid.uuid4().hex[:12], model_id, query, answer, method, reason)
        self._sessions[session.id] = session
        self._events[session.id] = []
        self._conditions[session.id] = asyncio.Condition()
        self._persist(session)
        self.emit(session.id, "uncertainty-detected", "The answer needs stronger evidence", 0.03)
        return session

    def get(self, session_id: str) -> LearningSession | None:
        return self._sessions.get(session_id)

    def events(self, session_id: str, after: int = 0) -> list[dict[str, Any]]:
        return [event for event in self._events.get(session_id, []) if int(event["sequence"]) > after]

    def emit(self, session_id: str, stage: str, message: str, progress: float, **data: Any) -> None:
        session = self._sessions[session_id]
        session.stage = stage
        session.progress = max(session.progress, min(1.0, progress))
        event = {
            "sequence": len(self._events[session_id]) + 1,
            "timestamp": time.time(),
            "stage": stage,
            "message": message,
            "progress": session.progress,
            "data": data,
        }
        self._events[session_id].append(event)
        path = self._folder(session_id) / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._persist(session)
        condition = self._conditions.setdefault(session_id, asyncio.Condition())
        try:
            asyncio.get_running_loop().create_task(self._notify(condition))
        except RuntimeError:
            pass

    @staticmethod
    async def _notify(condition: asyncio.Condition) -> None:
        async with condition:
            condition.notify_all()

    def complete(self, session_id: str, answer: str) -> None:
        session = self._sessions[session_id]
        session.final_answer = answer
        session.status = "completed"
        self.emit(session_id, "completed", "The learned answer passed the final response step", 1.0)

    def fail(self, session_id: str, error: str) -> None:
        session = self._sessions[session_id]
        session.status = "failed"
        session.error = error
        self.emit(session_id, "failed", error, 1.0)

    async def stream(self, session_id: str, after: int = 0) -> AsyncIterator[dict[str, Any]]:
        cursor = after
        while True:
            for event in self.events(session_id, cursor):
                cursor = max(cursor, int(event["sequence"]))
                yield event
            session = self.get(session_id)
            if not session or session.status in {"completed", "failed"}:
                return
            condition = self._conditions.setdefault(session_id, asyncio.Condition())
            async with condition:
                try:
                    await asyncio.wait_for(condition.wait(), timeout=15)
                except TimeoutError:
                    yield {"sequence": cursor, "stage": "heartbeat", "message": "heartbeat", "progress": session.progress}


def build_research_dataset(query: str, sources: list[dict[str, str]]) -> list[dict[str, str]]:
    """Build grounded IL demonstrations without inventing unsupported facts."""
    examples: list[dict[str, str]] = []
    combined: list[str] = []
    for index, source in enumerate(sources[:6]):
        text = re.sub(r"\s+", " ", source.get("text", "")).strip()[:1800]
        if not text:
            continue
        title = source.get("title") or f"Source {index + 1}"
        url = source.get("url", "")
        combined.append(f"{title}: {text}")
        examples.append({
            "prompt": f"For the research question ‘{query}’, summarize only the useful evidence in {title} and cite its URL.",
            "ideal_response": f"<reasoning>I will use only the supplied source and avoid unsupported claims.</reasoning><answer>{text}\n\nSource: {url}</answer>",
            "expected_answer": text[:500],
            "source_url": url,
        })
    if combined:
        synthesis = "\n\n".join(combined)[:6000]
        examples.insert(0, {
            "prompt": query,
            "ideal_response": f"<reasoning>I will synthesize the retrieved evidence and preserve its citations.</reasoning><answer>{synthesis}</answer>",
            "expected_answer": synthesis[:700],
            "source_url": "",
        })
    return examples
