import json
from pathlib import Path

from iloptimus.core.scene_spec import (
    SCENE_SCHEMA_EXAMPLE,
    audit_scene_authorship,
    audit_scene_spec,
    compile_scene_spec,
    complete_scene_spec,
    parse_scene_spec,
    scene_spec_prompt,
)


def test_scene_spec_parses_validates_and_compiles(tmp_path: Path):
    authored = {
        **SCENE_SCHEMA_EXAMPLE,
        "title": "Rose Sakura Island",
        "sky": "#101827",
        "fog": "#263449",
        "blossom": "#f9a8d4",
        "terrainRadius": 13,
        "waterSize": 110,
        "details": ["sakura shrine", "island dock", "lantern path"],
    }
    spec = parse_scene_spec("prose " + json.dumps(authored) + " trailing")
    assert audit_scene_spec(spec, "Build a Sakura island").passed
    output = tmp_path / "index.html"
    manifest = compile_scene_spec(spec, output, "Build a Sakura island")
    source = output.read_text()
    assert "const sceneSpec" in source
    assert "sceneSpec.trees.forEach" in source
    assert "createDesignedDetails" in source
    assert manifest["authorship"] == "local-model-scene-spec"
    assert audit_scene_authorship(manifest) == []


def test_scene_spec_rejects_invalid_motion_shape():
    broken = {**SCENE_SCHEMA_EXAMPLE, "motion": ["waterSpeed", 1.0]}
    audit = audit_scene_spec(broken)
    assert not audit.passed
    assert "motion must be an object" in audit.diagnostics


def test_scene_spec_rejects_copying_unrelated_example():
    audit = audit_scene_spec(SCENE_SCHEMA_EXAMPLE, "Build a Sakura island")
    assert not audit.passed
    assert any("copied" in diagnostic for diagnostic in audit.diagnostics)
    assert any("palette" in diagnostic for diagnostic in audit.diagnostics)
    assert any("Sakura" in diagnostic for diagnostic in audit.diagnostics)


def test_scene_spec_repairs_only_unquoted_identifier_keys():
    raw = json.dumps(
        {
            **SCENE_SCHEMA_EXAMPLE,
            "title": "Sakura Island",
            "sky": "#101827",
            "fog": "#263449",
            "blossom": "#f9a8d4",
            "terrainRadius": 12,
            "waterSize": 100,
            "details": ["sakura grove", "island dock", "torii gate"],
        }
    ).replace('{"x":', "{x:").replace(', "z":', ", z:").replace(', "scale":', ", scale:")
    spec = parse_scene_spec(raw)
    assert spec is not None
    assert spec["trees"][0]["x"] == -2
    assert audit_scene_spec(spec, "Build a Sakura Island").passed


def test_scene_compiler_can_supply_only_invalid_motion_default():
    spec = {
        **SCENE_SCHEMA_EXAMPLE,
        "title": "Sakura Island",
        "sky": "#221133",
        "fog": "#553366",
        "blossom": "#f9a8d4",
        "terrainRadius": 12,
        "waterSize": 100,
        "details": ["sakura grove", "island dock", "torii gate"],
        "motion": ["water", "petals"],
    }
    completed = complete_scene_spec(spec, "Build a Sakura Island")
    assert completed is not None
    repaired, defaults = completed
    assert defaults == ("motion",)
    assert audit_scene_spec(repaired, "Build a Sakura Island").passed


def test_scene_authorship_rejects_semantic_framework_defaults(tmp_path: Path):
    spec = {
        **SCENE_SCHEMA_EXAMPLE,
        "title": "Sakura Island",
        "sky": "#221133",
        "fog": "#553366",
        "blossom": "#f9a8d4",
        "terrainRadius": 12,
        "waterSize": 100,
        "details": ["sakura grove", "island dock", "torii gate"],
    }
    manifest = compile_scene_spec(spec, tmp_path / "index.html", "Build a Sakura Island")
    manifest["framework_default_fields"] = ["palette"]
    assert any("beyond" in error for error in audit_scene_authorship(manifest))


def test_scene_prompt_includes_schema_and_feedback():
    prompt = scene_spec_prompt("Build an island", ("motion must be an object",))
    assert "terrainRadius" in prompt
    assert "motion must be an object" in prompt
    repair = scene_spec_prompt("Build an island", ("Change the palette",), '{"title":"Desert"}')
    assert "Previous candidate" not in repair
    assert "Amber Desert Outpost" not in repair
