import json
from pathlib import Path

from iloptimus.core.dataset_tools import (
    assemble_dataset,
    audit_feature_coverage,
    create_dataset_workspace,
    curate_dataset,
    expand_dataset,
    filter_dataset,
    load_filtered_dataset,
    save_source_bundle,
    score_source_unit,
)


def _source(name: str, body: str) -> dict[str, str]:
    return {
        "title": name,
        "url": f"https://example.test/{name}",
        "text": body,
        "license": "MIT",
        "kind": "repository-code",
    }


def test_dataset_workspace_assembles_expands_and_filters(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    create_dataset_workspace("experiment")
    sources = [
        _source(
            "voxel",
            "\n".join(
                f"const geometry{index} = new THREE.BoxGeometry({index + 1}, 1, 1); "
                f"const mesh{index} = new THREE.InstancedMesh(geometry{index}, material, {index + 10});"
                for index in range(30)
            ),
        ),
        _source(
            "shader",
            "\n".join(
                f"const water{index} = new THREE.ShaderMaterial({{vertexShader, fragmentShader}}); "
                f"function animate{index}(){{ requestAnimationFrame(animate{index}); renderer.render(scene, camera); }}"
                for index in range(30)
            ),
        ),
        _source(
            "controls",
            "\n".join(
                f"const controls{index} = new OrbitControls(camera, renderer.domElement); "
                f"window.addEventListener('resize', resize{index}); "
                f"function resize{index}(){{ renderer.setSize(innerWidth, innerHeight); }}"
                for index in range(30)
            ),
        ),
    ]
    saved = save_source_bundle("experiment", sources + [sources[0]])
    assert saved["source_count"] == 3
    assembled = assemble_dataset(
        "experiment",
        task="Build a private heldout artifact phrase",
        artifact_kind="web",
        requested_features=["voxel", "shader", "animation", "interaction", "responsive"],
        target_examples=32,
        chunk_chars=500,
    )
    assert assembled["rows"] >= 3
    expanded = expand_dataset("experiment", target_examples=48)
    assert expanded["rows"] > assembled["rows"]
    audit = filter_dataset("experiment", holdout_task="Build a private heldout artifact phrase")
    assert audit["accepted_rows"] >= 3
    assert audit["exact_duplicates"] + audit["near_duplicates"] + audit["source_dominated_rows"] > 0
    assert audit["contaminated_rows"] == 0
    assert Path(audit["path"]).exists()
    assert all("row_sha256" in row for row in load_filtered_dataset("experiment"))
    json.dumps(audit)


def test_filter_reports_exact_and_near_duplicates_separately(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    root = Path(create_dataset_workspace("duplicates")["root"])
    base = "\n".join(f"const value{index} = compute({index});" for index in range(40))
    near = base.replace("compute(20)", "calculate(20)")
    rows = [
        {"prompt": "one", "ideal_response": base, "source_url": "https://one.test/a.js"},
        {"prompt": "two", "ideal_response": base, "source_url": "https://two.test/b.js"},
        {"prompt": "three", "ideal_response": near, "source_url": "https://three.test/c.js"},
    ]
    (root / "dataset-raw.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    audit = filter_dataset("duplicates", holdout_task="unrelated holdout")

    assert audit["accepted_rows"] == 1
    assert audit["exact_duplicates"] == 1
    assert audit["near_duplicates"] == 1


def test_filter_removes_holdout_contamination(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    root = Path(create_dataset_workspace("contamination")["root"])
    row = {
        "prompt": "pattern",
        "ideal_response": "The exact secret holdout request appears here. " + "implementation " * 80,
        "source_url": "https://example.test/source",
    }
    (root / "dataset-raw.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    audit = filter_dataset("contamination", holdout_task="exact secret holdout request")
    assert audit["accepted_rows"] == 0
    assert audit["contaminated_rows"] == 1


def test_jsonl_reader_preserves_unicode_line_separators(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    create_dataset_workspace("unicode-lines")
    saved = save_source_bundle(
        "unicode-lines",
        [_source("unicode", "const label = 'before\u2028after';\n" + "implementation();\n" * 30)],
    )
    assert saved["source_count"] == 1
    assembled = assemble_dataset(
        "unicode-lines",
        task="held out",
        artifact_kind="code",
        requested_features=[],
        target_examples=4,
    )
    assert assembled["rows"] >= 1
    rows = [json.loads(line) for line in Path(assembled["path"]).read_text(encoding="utf-8").split("\n") if line]
    assert any("\u2028" in row["ideal_response"] for row in rows)


def test_filter_caps_rows_without_losing_rare_feature_coverage(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    root = Path(create_dataset_workspace("balanced-cap")["root"])
    rows = []
    for index in range(40):
        feature = "sakura" if index == 39 else "three.js"
        rows.append(
            {
                "prompt": f"pattern {index}",
                "ideal_response": f"const unique{index} = new THREE.Scene(); " + f"implementation{index}(); " * 30,
                "source_url": f"https://source-{index % 8}.test/file-{index}.js",
                "source_hash": str(index),
                "features": [feature],
            }
        )
    (root / "dataset-raw.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    audit = filter_dataset("balanced-cap", holdout_task="secret holdout", maximum_rows=24)
    filtered = load_filtered_dataset("balanced-cap")
    assert audit["quality_rows"] > audit["accepted_rows"] == 24
    assert audit["capped_rows"] == audit["quality_rows"] - 24
    assert any("sakura" in row["features"] for row in filtered)


def test_feature_coverage_requires_rows_files_and_independent_origins():
    rows = []
    for index in range(4):
        rows.append(
            {
                "features": ["shader", "voxel" if index < 3 else "shader"],
                "source_url": f"https://github.com/org-{index % 2}/repo/blob/main/example-{index}.js",
                "source_origin": f"github:org-{index % 2}/repo",
            }
        )
    audit = audit_feature_coverage(rows, ["shader", "voxel"])
    assert audit["features"]["shader"]["passed"] is True
    assert audit["features"]["voxel"]["passed"] is False
    assert audit["missing_features"] == ["voxel"]


def test_artifact_units_can_filter_short_eos_targets(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    body = "\n".join(
        f"const mesh{index} = new THREE.InstancedMesh(new THREE.BoxGeometry(), material, {index + 1});"
        for index in range(100)
    )
    save_source_bundle("long-units", [_source("long-source", body)])
    assembly = assemble_dataset(
        "long-units",
        task="held-out full artifact",
        artifact_kind="web",
        requested_features=["three.js", "voxel"],
        target_examples=24,
    )
    assert assembly["chunk_chars"] == 2_400
    expand_dataset("long-units", target_examples=32)
    audit = filter_dataset(
        "long-units",
        holdout_task="held-out full artifact",
        minimum_response_chars=1_000,
        maximum_rows=32,
    )
    assert audit["accepted_rows"] > 0
    assert all(len(str(row["ideal_response"])) >= 1_000 for row in load_filtered_dataset("long-units"))


def test_automated_curator_prioritizes_failures_and_records_quality(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    sources = []
    for feature in ("voxel", "shader", "animation"):
        for origin in range(2):
            if feature == "voxel":
                statement = "new THREE.InstancedMesh(new THREE.BoxGeometry(), material, 32)"
            elif feature == "shader":
                statement = "new THREE.ShaderMaterial({vertexShader, fragmentShader})"
            else:
                statement = "requestAnimationFrame(animate); renderer.render(scene, camera)"
            body = "\n".join(f"const {feature}{origin}_{index} = {statement};" for index in range(80))
            sources.append(
                {
                    "title": f"{feature}-{origin}",
                    "url": f"https://github.com/org-{origin}/{feature}/blob/main/source.js",
                    "text": body,
                    "license": "MIT",
                    "kind": "repository-code",
                }
            )
    save_source_bundle("auto-curate", sources)
    result = curate_dataset(
        "auto-curate",
        task="held-out private task",
        artifact_kind="web",
        requested_features=["three.js", "voxel", "shader", "animation"],
        priority_features=["voxel", "shader", "animation"],
        assembled_examples=48,
        expanded_examples=64,
        maximum_rows=48,
        chunk_chars=1_400,
        minimum_response_chars=800,
    )
    assert result["elapsed_ms"] >= 0
    assert result["filtering"]["low_quality_rows"] >= 0
    assert result["filtering"]["mean_quality_score"] >= 0.5
    assert result["feature_coverage"]["passed"] is True
    rows = load_filtered_dataset("auto-curate")
    assert any(row["curriculum_role"] == "remediation" for row in rows)
    assert all(float(row["quality_score"]) >= 0.5 for row in rows)


def test_source_quality_penalizes_minified_or_legacy_units():
    good = "\n".join(f"const mesh{index} = new THREE.BoxGeometry();" for index in range(40))
    bad = "const legacy=" + "x" * 1800 + "; // deprecated minified vendor"
    assert score_source_unit(good, ["three.js", "voxel"]) > score_source_unit(bad, [])
