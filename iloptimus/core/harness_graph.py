"""Algorithmic harness self-improvement graph.

Tracks every meaningful action the model takes inside the harness — tool calls,
skills created/deleted/reused, mistakes, good actions — and links each action to
the success or failure of the task it belonged to.  Each action node carries an
**adaptive weight** that is not fixed: it is updated with a decaying-learning-rate
exponential moving average every time a linked task resolves, so the weights
converge toward the true success contribution of each action while still
adapting to drift.

The resulting graph is a directed attribution network:

    action nodes ──> task outcome nodes

Co-occurring actions inside the same task are also connected by weighted edges,
which lets a larger model (later) read the graph and reason about which action
combinations correlate with success.

Storage (file-based, matching the rest of the codebase):

    ~/.iloptimus/harness-graph/
    ├── graph.json          # current node/edge/weight state
    ├── events.jsonl        # append-only raw action + outcome stream
    └── efficiency.jsonl    # append-only efficiency snapshots for time-series
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .storage import app_home, atomic_write_json

# ---------------------------------------------------------------------------
# Action taxonomy
# ---------------------------------------------------------------------------

# Every recorded action maps to one of these categories.  The category drives
# default semantics (e.g. ``mistake`` actions start with a low weight).
ACTION_CATEGORIES: dict[str, str] = {
    "tool_call": "action",
    "tool_success": "action",
    "tool_failure": "mistake",
    "skill_created": "action",
    "skill_deleted": "action",
    "skill_used": "action",
    "skill_retrieved": "action",
    "duplicate_call": "mistake",
    "good_action": "action",
    "mistake": "mistake",
    "uncertainty_detected": "mistake",
    "learning_triggered": "action",
    "learning_succeeded": "action",
    "learning_failed": "mistake",
    "artifact_generated": "action",
    "artifact_verified": "action",
    "artifact_rejected": "mistake",
}

DEFAULT_WEIGHTS: dict[str, float] = {
    "action": 0.5,
    "mistake": 0.2,
}


def _category(action_type: str) -> str:
    return ACTION_CATEGORIES.get(action_type, "action")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ActionNode:
    """A single tracked action type (e.g. ``tool_call:web_search``)."""

    key: str  # unique identifier, e.g. "tool_call:web_search"
    action_type: str  # e.g. "tool_call"
    category: str  # "action" | "mistake"
    label: str  # human-readable label
    weight: float  # adaptive weight 0..1 (predicted success contribution)
    observations: int = 0  # total times linked to a resolved task
    successes: int = 0
    failures: int = 0
    ema_success: float = 0.5  # exponential moving average of outcomes
    last_seen: float = 0.0

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskNode:
    """A task outcome the model was assigned."""

    id: str
    kind: str  # "chat" | "learning" | "artifact" | "training"
    success: bool
    score: float  # 0..1 normalised outcome
    action_keys: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActionEdge:
    """Co-occurrence edge between two action nodes inside the same task."""

    source: str
    target: str
    co_occurrences: int = 0
    joint_successes: int = 0
    weight: float = 0.5  # P(success | both actions present)

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EfficiencySnapshot:
    """A point-in-time efficiency measurement for the time-series chart."""

    timestamp: float
    efficiency: float  # weighted success rate across all actions
    total_actions: int
    total_tasks: int
    success_rate: float  # raw task success rate

    def public(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Adaptive weight algorithm
# ---------------------------------------------------------------------------

# The learning rate for each node decays with its observation count so that:
#   - early observations shift the weight quickly (fast adaptation)
#   - later observations nudge it gently (stability without freezing)
#
#   alpha = 1 / (1 + observations)
#
# This is a classic diminishing-step-size stochastic approximation: it
# converges while remaining responsive to distribution drift because alpha
# never reaches zero.

_MIN_ALPHA = 0.02  # floor so weights never fully stop adapting
_MAX_ALPHA = 0.8  # cap so a single observation can't dominate a mature node


def _learning_rate(observations: int) -> float:
    return max(_MIN_ALPHA, min(_MAX_ALPHA, 1.0 / (1.0 + observations)))


def _update_node_weight(node: ActionNode, outcome: float) -> None:
    """Push a single task outcome into a node's adaptive weight.

    ``outcome`` is 1.0 for full success, 0.0 for full failure, or a fractional
    score in between.
    """
    alpha = _learning_rate(node.observations)
    node.ema_success = alpha * outcome + (1.0 - alpha) * node.ema_success
    node.weight = max(0.0, min(1.0, node.ema_success))
    node.observations += 1
    if outcome >= 0.5:
        node.successes += 1
    else:
        node.failures += 1
    node.last_seen = time.time()


def _update_edge_weight(edge: ActionEdge, outcome: float) -> None:
    alpha = _learning_rate(edge.co_occurrences)
    edge.weight = alpha * outcome + (1.0 - alpha) * edge.weight
    edge.co_occurrences += 1
    if outcome >= 0.5:
        edge.joint_successes += 1


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class HarnessGraphManager:
    """Owns the harness self-improvement graph and its persistence."""

    def __init__(self, root: Path | None = None):
        self.root = root or app_home() / "harness-graph"
        self.root.mkdir(parents=True, exist_ok=True)
        self._nodes: dict[str, ActionNode] = {}
        self._edges: dict[str, ActionEdge] = {}  # key = "src|tgt"
        self._tasks: dict[str, TaskNode] = {}
        self._pending: dict[str, list[str]] = {}  # task_id -> action keys not yet resolved
        self._efficiency: list[EfficiencySnapshot] = []
        self._load()

    # ---- persistence -------------------------------------------------------

    def _graph_path(self) -> Path:
        return self.root / "graph.json"

    def _events_path(self) -> Path:
        return self.root / "events.jsonl"

    def _efficiency_path(self) -> Path:
        return self.root / "efficiency.jsonl"

    def _load(self) -> None:
        path = self._graph_path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                for raw in data.get("nodes", []):
                    node = ActionNode(**raw)
                    self._nodes[node.key] = node
                for raw in data.get("edges", []):
                    edge = ActionEdge(**raw)
                    self._edges[f"{edge.source}|{edge.target}"] = edge
                for raw in data.get("tasks", []):
                    task = TaskNode(**raw)
                    self._tasks[task.id] = task
                    if task.success is None:  # unresolved task loaded from disk
                        self._pending.setdefault(task.id, list(task.action_keys))
            except (OSError, TypeError, json.JSONDecodeError):
                pass
        eff_path = self._efficiency_path()
        if eff_path.exists():
            for line in eff_path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    raw = json.loads(line)
                    self._efficiency.append(EfficiencySnapshot(**raw))
                except (json.JSONDecodeError, TypeError):
                    continue

    def _persist(self) -> None:
        atomic_write_json(
            self._graph_path(),
            {
                "nodes": [node.public() for node in self._nodes.values()],
                "edges": [edge.public() for edge in self._edges.values()],
                "tasks": [task.public() for task in self._tasks.values()],
                "updated_at": time.time(),
            },
        )

    def _append_event(self, event: dict[str, Any]) -> None:
        event.setdefault("timestamp", time.time())
        with self._events_path().open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _append_efficiency(self, snapshot: EfficiencySnapshot) -> None:
        self._efficiency.append(snapshot)
        # Keep the in-memory series bounded.
        if len(self._efficiency) > 2000:
            self._efficiency = self._efficiency[-2000:]
        with self._efficiency_path().open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot.public(), ensure_ascii=False) + "\n")

    # ---- public API --------------------------------------------------------

    def begin_task(self, task_id: str, kind: str, **metadata: Any) -> str:
        """Register a task the model is about to work on.

        Returns the task_id.  Actions recorded with ``record_action`` will be
        linked to this task until ``resolve_task`` is called.
        """
        if not task_id:
            task_id = uuid.uuid4().hex[:12]
        task = TaskNode(id=task_id, kind=kind, success=False, score=0.0)
        # Preserve any metadata we might want later without bloating the model.
        task.action_keys = []
        self._tasks[task_id] = task
        self._pending[task_id] = []
        self._append_event({"type": "task_begin", "task_id": task_id, "kind": kind, "metadata": metadata})
        return task_id

    def record_action(
        self,
        action_type: str,
        *,
        key: str | None = None,
        label: str | None = None,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ActionNode:
        """Record a single action and link it to the current task (if any).

        ``key`` should be a stable identifier for the *specific* action variant
        (e.g. ``"tool_call:web_search"``).  If omitted, the ``action_type`` is
        used as the key.
        """
        node_key = key or action_type
        node = self._nodes.get(node_key)
        if node is None:
            category = _category(action_type)
            node = ActionNode(
                key=node_key,
                action_type=action_type,
                category=category,
                label=label or node_key,
                weight=DEFAULT_WEIGHTS.get(category, 0.5),
                ema_success=DEFAULT_WEIGHTS.get(category, 0.5),
            )
            self._nodes[node_key] = node

        if label and node.label == node.key:
            node.label = label

        if task_id and task_id in self._pending:
            if node_key not in self._pending[task_id]:
                self._pending[task_id].append(node_key)
            task = self._tasks.get(task_id)
            if task and node_key not in task.action_keys:
                task.action_keys.append(node_key)
            # Add co-occurrence edges to all other actions in this task.
            for other_key in self._pending[task_id]:
                if other_key == node_key:
                    continue
                self._ensure_edge(other_key, node_key)

        self._append_event(
            {
                "type": "action",
                "action_type": action_type,
                "key": node_key,
                "task_id": task_id,
                "metadata": metadata or {},
            }
        )
        return node

    def _ensure_edge(self, source: str, target: str) -> ActionEdge:
        edge_key = f"{source}|{target}"
        edge = self._edges.get(edge_key)
        if edge is None:
            edge = ActionEdge(source=source, target=target)
            self._edges[edge_key] = edge
        return edge

    def resolve_task(
        self,
        task_id: str,
        *,
        success: bool,
        score: float | None = None,
        **metadata: Any,
    ) -> None:
        """Resolve a task and back-propagate the outcome into all linked nodes.

        This is where the adaptive weights update.  Every action node linked to
        the task gets its EMA nudged by the outcome, and every co-occurrence
        edge is updated too.  A new efficiency snapshot is appended.
        """
        task = self._tasks.get(task_id)
        if task is None:
            # Allow resolving a task that was never explicitly begun.
            task = TaskNode(id=task_id, kind=metadata.get("kind", "unknown"), success=success, score=score or 0.0)
            self._tasks[task_id] = task
            self._pending[task_id] = []

        outcome = float(score) if score is not None else (1.0 if success else 0.0)
        task.success = success
        task.score = outcome

        action_keys = self._pending.pop(task_id, task.action_keys)
        for node_key in action_keys:
            node = self._nodes.get(node_key)
            if node:
                _update_node_weight(node, outcome)

        # Update co-occurrence edges among the actions in this task.
        for i, src in enumerate(action_keys):
            for tgt in action_keys[i + 1:]:
                edge = self._ensure_edge(src, tgt)
                _update_edge_weight(edge, outcome)

        self._append_event(
            {
                "type": "task_resolve",
                "task_id": task_id,
                "success": success,
                "score": outcome,
                "action_keys": action_keys,
                "metadata": metadata,
            }
        )

        self._record_efficiency()
        self._persist()

    def _record_efficiency(self) -> None:
        """Compute and persist a graph-wide efficiency snapshot."""
        total_obs = sum(node.observations for node in self._nodes.values())
        if total_obs == 0:
            efficiency = 0.0
        else:
            efficiency = sum(node.weight * node.observations for node in self._nodes.values()) / total_obs
        resolved = [t for t in self._tasks.values() if t.success is not None]
        total_tasks = len(resolved)
        success_rate = (sum(1 for t in resolved if t.success) / total_tasks) if total_tasks else 0.0
        snapshot = EfficiencySnapshot(
            timestamp=time.time(),
            efficiency=round(efficiency, 6),
            total_actions=len(self._nodes),
            total_tasks=total_tasks,
            success_rate=round(success_rate, 6),
        )
        self._append_efficiency(snapshot)

    # ---- read paths --------------------------------------------------------

    def graph(self) -> dict[str, Any]:
        """Return the full graph payload for the frontend."""
        nodes = sorted(self._nodes.values(), key=lambda n: (-n.observations, n.key))
        edges = sorted(self._edges.values(), key=lambda e: (-e.co_occurrences, e.source, e.target))
        tasks = sorted(self._tasks.values(), key=lambda t: -t.created_at)
        return {
            "nodes": [n.public() for n in nodes],
            "edges": [e.public() for e in edges],
            "tasks": [t.public() for t in tasks[:200]],
            "total_tasks": len(self._tasks),
            "total_actions": len(self._nodes),
            "total_edges": len(self._edges),
            "pending_tasks": len(self._pending),
        }

    def efficiency_series(self, limit: int = 500) -> list[dict[str, Any]]:
        series = self._efficiency[-limit:]
        return [s.public() for s in series]

    def top_actions(self, limit: int = 20) -> list[dict[str, Any]]:
        ranked = sorted(self._nodes.values(), key=lambda n: (-n.observations, -n.weight))
        return [n.public() for n in ranked[:limit]]

    def reset(self) -> None:
        """Clear all graph state (used by tests and a manual reset endpoint)."""
        self._nodes.clear()
        self._edges.clear()
        self._tasks.clear()
        self._pending.clear()
        self._efficiency.clear()
        for path in (self._graph_path(), self._events_path(), self._efficiency_path()):
            try:
                path.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Convenience: ingest existing audit logs to bootstrap the graph
# ---------------------------------------------------------------------------


def ingest_tool_call_log(manager: HarnessGraphManager, max_lines: int = 10_000) -> int:
    """Read the existing ``tool_calls.jsonl`` audit log and seed the graph.

    Each tool call becomes an action.  Since the legacy log has no task
    linkage, we treat each line as its own micro-task so the weights still
    converge to the tool's raw success rate.

    Returns the number of entries ingested.
    """
    path = app_home() / "tool_calls.jsonl"
    if not path.exists():
        return 0
    ingested = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        tool = str(entry.get("tool", "unknown"))
        status = str(entry.get("status", "ok"))
        success = status == "ok"
        task_id = f"legacy-tool-{ingested}"
        manager.begin_task(task_id, kind="tool_call")
        manager.record_action(
            "tool_call" if success else "tool_failure",
            key=f"tool_call:{tool}",
            label=tool,
            task_id=task_id,
            metadata={"elapsed_ms": entry.get("elapsed_ms")},
        )
        manager.resolve_task(task_id, success=success, score=1.0 if success else 0.0, kind="tool_call")
        ingested += 1
    return ingested
