"""Tests for the recursive dataset improvement loop."""

from iloptimus.core.dataset_loop import (
    LoopConfig,
    LoopIteration,
    _dataset_hash,
    analyze_capability_impacts,
    build_iteration_record,
    check_convergence,
    compute_token_density,
    plan_re_curation,
    save_iteration_record,
    select_best_iteration,
    summarize_impact_log,
)


def _make_rows(features_per_row: list[list[str]], quality: float = 0.8) -> list[dict]:
    return [
        {
            "prompt": f"Write source unit {i} for {', '.join(features)}",
            "ideal_response": f"const example{i} = new THREE.Mesh(geometry, material);",
            "expected_answer": f"example{i}",
            "source_url": f"https://github.com/repo{i}/file.js",
            "source_hash": f"hash{i}",
            "source_origin": f"github:repo{i}",
            "features": features,
            "quality_score": quality,
        }
        for i, features in enumerate(features_per_row)
    ]


def test_analyze_capability_impacts_classifies_each_capability():
    prev_scores = {"voxel": 0.3, "shader": 0.5, "animation": 0.95}
    curr_scores = {"voxel": 0.6, "shader": 0.5, "animation": 0.96}
    rows = _make_rows(
        [
            ["voxel"],
            ["voxel"],
            ["voxel"],
            ["voxel"],
            ["shader"],
            ["shader"],
            ["shader"],
            ["shader"],
            ["animation"],
            ["animation"],
            ["animation"],
            ["animation"],
        ]
    )
    impacts = analyze_capability_impacts(
        prev_scores, curr_scores, rows, ["voxel", "shader", "animation"]
    )
    by_cap = {i.capability: i for i in impacts}
    assert by_cap["voxel"].verdict == "improved"
    assert by_cap["voxel"].delta == 0.3
    assert by_cap["shader"].verdict == "flat"
    assert by_cap["shader"].delta == 0.0
    assert by_cap["animation"].verdict == "saturated"
    assert by_cap["voxel"].row_count == 4
    assert by_cap["voxel"].source_count == 4


def test_analyze_capability_impacts_detects_regression():
    prev_scores = {"voxel": 0.7}
    curr_scores = {"voxel": 0.4}
    rows = _make_rows([["voxel"]] * 4)
    impacts = analyze_capability_impacts(prev_scores, curr_scores, rows, ["voxel"])
    assert impacts[0].verdict == "regressed"
    assert impacts[0].delta == -0.3


def test_plan_re_curation_prioritizes_weak_capabilities():
    impacts = analyze_capability_impacts(
        {"voxel": 0.3, "shader": 0.5, "animation": 0.97},
        {"voxel": 0.3, "shader": 0.5, "animation": 0.97},
        _make_rows([["voxel"]] * 2 + [["shader"]] * 4 + [["animation"]] * 10),
        ["voxel", "shader", "animation"],
    )
    plan = plan_re_curation(impacts, LoopConfig())
    assert "voxel" in plan["weak_capabilities"]
    assert "animation" in plan["saturated_capabilities"]
    # Saturated capabilities are excluded from the priority list entirely
    # so curate_dataset's rare-first reservation doesn't waste slots on them.
    assert "animation" not in plan["priority_features"]
    # Weak capabilities should come first in priority
    assert plan["priority_features"][0] == "voxel"
    assert "voxel" in plan["search_hints"]
    assert len(plan["search_hints"]["voxel"]) == 3
    assert plan["prune_saturated"] is True


def test_plan_re_curation_no_weak_capabilities():
    impacts = analyze_capability_impacts(
        {"voxel": 0.95, "shader": 0.97},
        {"voxel": 0.96, "shader": 0.98},
        _make_rows([["voxel"]] * 10 + [["shader"]] * 10),
        ["voxel", "shader"],
    )
    plan = plan_re_curation(impacts, LoopConfig())
    assert plan["weak_capabilities"] == []
    assert "voxel" in plan["saturated_capabilities"]
    assert "shader" in plan["saturated_capabilities"]


def test_check_convergence_stops_at_max_iterations():
    config = LoopConfig(max_iterations=3)
    history = [
        LoopIteration(
            iteration=i,
            dataset_hash="h",
            dataset_rows=80,
            capability_scores={},
            overall_score=0.5 + i * 0.01,
            improvement=0.01,
            changes={},
            capability_impacts=[],
            accepted=True,
            adapter_path="p",
            elapsed_seconds=10.0,
        )
        for i in range(3)
    ]
    stopped, reason = check_convergence(history, config)
    assert stopped
    assert "max_iterations" in reason


def test_check_convergence_stops_on_plateau():
    config = LoopConfig(max_iterations=10, min_improvement=0.05, convergence_window=2)
    history = [
        LoopIteration(
            iteration=0,
            dataset_hash="h0",
            dataset_rows=80,
            capability_scores={},
            overall_score=0.5,
            improvement=0.0,
            changes={},
            capability_impacts=[],
            accepted=False,
            adapter_path="p0",
            elapsed_seconds=10.0,
        ),
        LoopIteration(
            iteration=1,
            dataset_hash="h1",
            dataset_rows=80,
            capability_scores={},
            overall_score=0.51,
            improvement=0.01,
            changes={},
            capability_impacts=[],
            accepted=False,
            adapter_path="p1",
            elapsed_seconds=10.0,
        ),
        LoopIteration(
            iteration=2,
            dataset_hash="h2",
            dataset_rows=80,
            capability_scores={},
            overall_score=0.515,
            improvement=0.005,
            changes={},
            capability_impacts=[],
            accepted=False,
            adapter_path="p2",
            elapsed_seconds=10.0,
        ),
    ]
    stopped, reason = check_convergence(history, config)
    assert stopped
    assert "consecutive" in reason


def test_check_convergence_continues_when_improving():
    config = LoopConfig(max_iterations=10, min_improvement=0.02, convergence_window=2)
    history = [
        LoopIteration(
            iteration=0,
            dataset_hash="h0",
            dataset_rows=80,
            capability_scores={},
            overall_score=0.5,
            improvement=0.0,
            changes={},
            capability_impacts=[],
            accepted=False,
            adapter_path="p0",
            elapsed_seconds=10.0,
        ),
        LoopIteration(
            iteration=1,
            dataset_hash="h1",
            dataset_rows=80,
            capability_scores={},
            overall_score=0.55,
            improvement=0.05,
            changes={},
            capability_impacts=[],
            accepted=True,
            adapter_path="p1",
            elapsed_seconds=10.0,
        ),
    ]
    stopped, _reason = check_convergence(history, config)
    assert not stopped


def test_select_best_iteration_picks_highest_score():
    history = [
        LoopIteration(
            iteration=0,
            dataset_hash="h0",
            dataset_rows=80,
            capability_scores={},
            overall_score=0.5,
            improvement=0.0,
            changes={},
            capability_impacts=[],
            accepted=False,
            adapter_path="p0",
            elapsed_seconds=10.0,
        ),
        LoopIteration(
            iteration=1,
            dataset_hash="h1",
            dataset_rows=80,
            capability_scores={},
            overall_score=0.65,
            improvement=0.15,
            changes={},
            capability_impacts=[],
            accepted=True,
            adapter_path="p1",
            elapsed_seconds=20.0,
        ),
        LoopIteration(
            iteration=2,
            dataset_hash="h2",
            dataset_rows=80,
            capability_scores={},
            overall_score=0.60,
            improvement=-0.05,
            changes={},
            capability_impacts=[],
            accepted=False,
            adapter_path="p2",
            elapsed_seconds=15.0,
        ),
    ]
    idx, best = select_best_iteration(history)
    assert idx == 1
    assert best.overall_score == 0.65


def test_compute_token_density_measures_information_density():
    rows = [
        {
            "ideal_response": "const a = new THREE.Mesh(geometry, material); renderer.add(a);"
        },
        {
            "ideal_response": "const a = new THREE.Mesh(geometry, material); renderer.add(a);"
        },
        {
            "ideal_response": "function animate() { requestAnimationFrame(animate); renderer.render(scene, camera); }"
        },
    ]
    density = compute_token_density(rows)
    assert 0 < density["mean_density"] <= 1.0
    assert density["mean_chars"] > 0
    assert 0 < density["dense_fraction"] <= 1.0


def test_compute_token_density_empty_dataset():
    density = compute_token_density([])
    assert density["mean_density"] == 0.0
    assert density["mean_chars"] == 0
    assert density["dense_fraction"] == 0.0


def test_dataset_hash_is_deterministic():
    rows = _make_rows([["voxel"], ["shader"]])
    assert _dataset_hash(rows) == _dataset_hash(rows)


def test_dataset_hash_changes_with_content():
    rows_a = _make_rows([["voxel"], ["shader"]])
    rows_b = _make_rows([["voxel"], ["animation"]])
    assert _dataset_hash(rows_a) != _dataset_hash(rows_b)


def test_build_iteration_record_computes_overall_and_improvement():
    record = build_iteration_record(
        iteration=1,
        dataset_rows=_make_rows([["voxel"]] * 4),
        capability_scores={"voxel": 0.6, "shader": 0.4},
        prev_overall=0.4,
        changes={"re_curated": True},
        capability_impacts=[],
        accepted=True,
        adapter_path="/tmp/adapter",
        elapsed=120.0,
        curation_manifest={"version": 1},
    )
    assert record.iteration == 1
    assert record.overall_score == 0.5
    assert record.improvement == 0.1
    assert record.accepted is True
    assert record.dataset_rows == 4


def test_summarize_impact_log_records_helped_and_hurt():
    history = [
        LoopIteration(
            iteration=0,
            dataset_hash="h0",
            dataset_rows=80,
            capability_scores={"voxel": 0.3, "shader": 0.5},
            overall_score=0.4,
            improvement=0.0,
            changes={"initial": True},
            capability_impacts=[
                {"capability": "voxel", "delta": 0.3, "verdict": "improved"},
                {"capability": "shader", "delta": -0.1, "verdict": "regressed"},
            ],
            accepted=False,
            adapter_path="p0",
            elapsed_seconds=10.0,
        ),
    ]
    log = summarize_impact_log(history)
    assert len(log) == 1
    assert "voxel" in log[0]["helped_capabilities"]
    assert "shader" in log[0]["hurt_capabilities"]
    assert log[0]["improvement"] == 0.0


def test_save_and_load_iteration_record(monkeypatch, tmp_path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    record = build_iteration_record(
        iteration=0,
        dataset_rows=_make_rows([["voxel"]] * 4),
        capability_scores={"voxel": 0.5},
        prev_overall=0.0,
        changes={"initial": True},
        capability_impacts=[],
        accepted=False,
        adapter_path="/tmp/adapter",
        elapsed=60.0,
        curation_manifest={},
    )
    path = save_iteration_record("test-session", record)
    assert path.exists()
    import json

    saved = json.loads(path.read_text())
    assert saved["iteration"] == 0
    assert saved["dataset_rows"] == 4


def test_loop_config_defaults_are_conservative():
    config = LoopConfig()
    assert config.max_iterations == 12
    assert config.min_improvement == 0.01
    assert config.target_examples == 80
    assert config.rollback_on_regression is True
