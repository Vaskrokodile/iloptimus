from iloptimus.core.failure_memory import (
    build_failure_skill,
    mark_skill_use,
    retrieve_failure_skills,
    save_failure_skill,
    skill_guardrails,
    validate_failure_skill,
)


def test_verifier_failures_become_valid_retrievable_skills(monkeypatch, tmp_path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    contract = {
        "artifact_kind": "web",
        "requested_features": ["three.js", "voxel", "shader", "animation"],
    }
    evaluation = {
        "score": 0.42,
        "hard_gates": {"substantial": False, "javascript_syntax": False, "exists": True},
        "feature_scores": {"three.js": 1.0, "voxel": 0.0, "shader": 0.0, "animation": 0.0},
        "diagnostics": ["Artifact was too small", "Requested feature is not implemented observably: shader"],
    }
    skill = build_failure_skill(
        session_id="failed-session",
        contract=contract,
        baseline=evaluation,
    )
    saved = save_failure_skill(skill)
    markdown = (tmp_path / "skill-memory" / skill.id / "SKILL.md").read_text(encoding="utf-8")
    assert validate_failure_skill(skill, markdown) == []
    assert saved["evidence_status"] == "verified-failure-pattern"
    assert saved["evidence_observations"] == 1
    assert "Do not return prose" in markdown
    assert "observable voxel" in markdown

    matches = retrieve_failure_skills(artifact_kind="web", features=["three.js", "shader"])
    assert [item["id"] for item in matches] == [skill.id]
    assert "Verify:" in skill_guardrails(matches)
    mark_skill_use([skill.id], successful=True)
    promoted = retrieve_failure_skills(artifact_kind="web", features=["shader"])[0]
    assert promoted["uses"] == 1
    assert promoted["successful_uses"] == 1

    second = build_failure_skill(
        session_id="failed-session-2",
        contract=contract,
        baseline=evaluation,
    )
    saved_again = save_failure_skill(second)
    assert saved_again["evidence_observations"] == 2


def test_partial_feature_score_becomes_a_completion_rule():
    skill = build_failure_skill(
        session_id="partial",
        contract={"artifact_kind": "web", "requested_features": ["sakura"]},
        baseline={
            "score": 0.4,
            "hard_gates": {"substantial": False},
            "feature_scores": {"sakura": 0.5},
            "diagnostics": ["Sakura was named but not fully rendered"],
        },
    )
    assert any("blossom/tree forms" in item for item in skill.checklist)


def test_failure_skills_do_not_cross_unrelated_artifact_kinds(monkeypatch, tmp_path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    skill = build_failure_skill(
        session_id="code-session",
        contract={"artifact_kind": "code", "requested_features": ["shader"]},
        baseline={
            "score": 0.0,
            "hard_gates": {"exists": False},
            "feature_scores": {"shader": 0.0},
            "diagnostics": ["Missing solution.py"],
        },
    )
    save_failure_skill(skill)
    assert retrieve_failure_skills(artifact_kind="web", features=["shader"]) == []
