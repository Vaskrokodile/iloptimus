"""Tests for the algorithmic harness self-improvement graph."""

from iloptimus.core.harness_graph import (
    ActionNode,
    HarnessGraphManager,
    _learning_rate,
    _update_node_weight,
    ingest_tool_call_log,
)
from iloptimus.core.storage import app_home


def test_learning_rate_decays_but_never_freezes():
    assert _learning_rate(0) == 0.8  # capped at max
    assert _learning_rate(1) == 0.5
    assert _learning_rate(9) == 0.1
    assert _learning_rate(99) == _learning_rate(100)  # floor reached
    assert _learning_rate(1000) >= 0.02  # never freezes


def test_adaptive_weight_converges_to_success_rate(monkeypatch, tmp_path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    mgr = HarnessGraphManager()

    for i in range(50):
        tid = f"t1-{i}"
        mgr.begin_task(tid, kind="chat")
        mgr.record_action("tool_call", key="tool_call:web_search", label="web_search", task_id=tid)
        mgr.resolve_task(tid, success=True, score=1.0, kind="chat")

    node = mgr._nodes["tool_call:web_search"]
    assert node.observations >= 50
    assert node.weight > 0.85  # should converge near 1.0


def test_adaptive_weight_converges_to_failure_rate(monkeypatch, tmp_path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    mgr = HarnessGraphManager()

    for i in range(50):
        tid = f"fail-{i}"
        mgr.begin_task(tid, kind="chat")
        mgr.record_action("tool_failure", key="tool_call:broken_tool", label="broken_tool", task_id=tid)
        mgr.resolve_task(tid, success=False, score=0.0, kind="chat")

    node = mgr._nodes["tool_call:broken_tool"]
    assert node.observations >= 50
    assert node.weight < 0.15  # should converge near 0.0


def test_weight_adapts_to_distribution_drift(monkeypatch, tmp_path):
    """Weights must not freeze — if outcomes change, the weight must follow."""
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    mgr = HarnessGraphManager()

    # Phase 1: action succeeds 20 times.
    for i in range(20):
        tid = f"drift-s-{i}"
        mgr.begin_task(tid, kind="chat")
        mgr.record_action("tool_call:web_search", label="web_search", task_id=tid)
        mgr.resolve_task(tid, success=True, score=1.0, kind="chat")

    node = mgr._nodes["tool_call:web_search"]
    assert node.weight > 0.8

    # Phase 2: action fails 60 times — weight must drift downward.
    for i in range(60):
        tid = f"drift-f-{i}"
        mgr.begin_task(tid, kind="chat")
        mgr.record_action("tool_call:web_search", label="web_search", task_id=tid)
        mgr.resolve_task(tid, success=False, score=0.0, kind="chat")

    node = mgr._nodes["tool_call:web_search"]
    assert node.weight < 0.4  # adapted to the new failure regime


def test_co_occurrence_edges_created(monkeypatch, tmp_path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    mgr = HarnessGraphManager()

    tid = mgr.begin_task("co-occur", kind="chat")
    mgr.record_action("tool_call:web_search", label="web_search", task_id=tid)
    mgr.record_action("skill_used", key="skill_used:frontend-design", label="frontend-design", task_id=tid)
    mgr.resolve_task(tid, success=True, score=1.0, kind="chat")

    edges = mgr._edges
    assert len(edges) >= 1
    edge = next(iter(edges.values()))
    assert edge.co_occurrences >= 1
    assert edge.joint_successes >= 1
    assert edge.weight > 0.5


def test_graph_persistence_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    mgr1 = HarnessGraphManager()

    tid = mgr1.begin_task("persist-test", kind="chat")
    mgr1.record_action("tool_call:web_search", label="web_search", task_id=tid)
    mgr1.resolve_task(tid, success=True, score=1.0, kind="chat")

    # New manager instance loads from disk.
    mgr2 = HarnessGraphManager()
    assert "tool_call:web_search" in mgr2._nodes
    assert mgr2._nodes["tool_call:web_search"].observations >= 1
    assert len(mgr2._tasks) >= 1


def test_efficiency_series_recorded(monkeypatch, tmp_path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    mgr = HarnessGraphManager()

    for i in range(5):
        tid = f"eff-{i}"
        mgr.begin_task(tid, kind="chat")
        mgr.record_action("tool_call:web_search", label="web_search", task_id=tid)
        mgr.resolve_task(tid, success=i % 2 == 0, score=1.0 if i % 2 == 0 else 0.0, kind="chat")

    series = mgr.efficiency_series()
    assert len(series) >= 5
    assert all("efficiency" in s for s in series)
    assert all("success_rate" in s for s in series)


def test_top_actions_ranked_by_observations(monkeypatch, tmp_path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    mgr = HarnessGraphManager()

    # tool_a appears 10 times, tool_b appears 3 times.
    for i in range(10):
        tid = f"a-{i}"
        mgr.begin_task(tid, kind="chat")
        mgr.record_action("tool_call:tool_a", label="tool_a", task_id=tid)
        mgr.resolve_task(tid, success=True, score=1.0, kind="chat")

    for i in range(3):
        tid = f"b-{i}"
        mgr.begin_task(tid, kind="chat")
        mgr.record_action("tool_call:tool_b", label="tool_b", task_id=tid)
        mgr.resolve_task(tid, success=True, score=1.0, kind="chat")

    top = mgr.top_actions(limit=10)
    assert top[0]["key"] == "tool_call:tool_a"
    assert top[0]["observations"] >= 10
    assert top[1]["key"] == "tool_call:tool_b"


def test_mistake_actions_start_with_low_weight(monkeypatch, tmp_path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    mgr = HarnessGraphManager()

    tid = mgr.begin_task("mistake-test", kind="chat")
    node = mgr.record_action("duplicate_call", key="tool_call:web_search", label="web_search", task_id=tid)
    assert node.category == "mistake"
    assert node.weight <= 0.3  # mistakes start low


def test_reset_clears_all_state(monkeypatch, tmp_path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    mgr = HarnessGraphManager()

    tid = mgr.begin_task("reset-test", kind="chat")
    mgr.record_action("tool_call:web_search", label="web_search", task_id=tid)
    mgr.resolve_task(tid, success=True, score=1.0, kind="chat")

    assert len(mgr._nodes) > 0
    mgr.reset()
    assert len(mgr._nodes) == 0
    assert len(mgr._tasks) == 0
    assert len(mgr._edges) == 0
    assert len(mgr.efficiency_series()) == 0


def test_ingest_tool_call_log_bootstraps_graph(monkeypatch, tmp_path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))

    # Write a fake tool_calls.jsonl audit log.
    import json

    log_path = app_home() / "tool_calls.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        for i in range(10):
            f.write(json.dumps({
                "timestamp": f"2025-01-0{i+1}T00:00:00+00:00",
                "tool": "web_search" if i % 2 == 0 else "web_fetch",
                "status": "ok" if i % 3 != 0 else "error",
                "elapsed_ms": 100 + i,
            }) + "\n")

    mgr = HarnessGraphManager()
    ingested = ingest_tool_call_log(mgr)
    assert ingested == 10
    assert "tool_call:web_search" in mgr._nodes
    assert "tool_call:web_fetch" in mgr._nodes
    # web_search succeeded 5 times (i=0,2,4,6,8), failed 0 times in even indices
    # Actually: i=0 ok, i=2 ok, i=4 ok, i=6 ok, i=8 ok -> 5 successes
    # i=1 ok, i=3 error, i=5 ok, i=7 ok, i=9 error -> web_fetch: 3 ok, 2 error
    search_node = mgr._nodes["tool_call:web_search"]
    assert search_node.observations >= 5
    fetch_node = mgr._nodes["tool_call:web_fetch"]
    assert fetch_node.observations >= 5


def test_update_node_weight_directly():
    node = ActionNode(
        key="test",
        action_type="tool_call",
        category="action",
        label="test",
        weight=0.5,
        ema_success=0.5,
    )
    # First success should pull weight up.
    _update_node_weight(node, 1.0)
    assert node.weight > 0.5
    assert node.observations == 1
    assert node.successes == 1

    # A failure should pull it back down.
    _update_node_weight(node, 0.0)
    assert node.weight < 1.0
    assert node.failures == 1


def test_graph_payload_structure(monkeypatch, tmp_path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    mgr = HarnessGraphManager()

    tid = mgr.begin_task("payload-test", kind="chat")
    mgr.record_action("tool_call:web_search", label="web_search", task_id=tid)
    mgr.record_action("skill_used", key="skill_used:test", label="test", task_id=tid)
    mgr.resolve_task(tid, success=True, score=0.9, kind="chat")

    payload = mgr.graph()
    assert "nodes" in payload
    assert "edges" in payload
    assert "tasks" in payload
    assert "total_actions" in payload
    assert "total_tasks" in payload
    assert "total_edges" in payload
    assert payload["total_actions"] >= 2
    assert payload["total_tasks"] >= 1
    assert payload["total_edges"] >= 1
