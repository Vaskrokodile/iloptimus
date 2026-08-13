import json
from pathlib import Path

from iloptimus.core.test_time_compute import (
    _png_has_visual_content,
    acceptance_decision,
    artifact_generation_prompt,
    audit_research_subtask,
    build_artifact_dataset,
    derive_artifact_contract,
    evaluate_artifact,
    fast_research_queries,
    framework_artifact_source,
    github_repository_search_terms,
    github_repository_url,
    parse_model_queries,
    rank_repository_paths,
    research_subtasks,
    select_method,
    task_requires_artifact,
)


def _png(width: int, height: int, pixels: bytes) -> bytes:
    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    rows = b"".join(b"\0" + pixels[y * width * 3 : (y + 1) * width * 3] for y in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def test_global_artifact_contract_derives_observable_requirements():
    query = "/learn Create a polished interactive Three.js voxel scene with shaders and animation"
    assert task_requires_artifact(query)
    contract = derive_artifact_contract(query)
    assert contract.task_type == "artifact"
    assert contract.entrypoint == "index.html"
    assert contract.minimum_bytes == 8_000
    assert {"three.js", "voxel", "shader", "animation", "interaction", "responsive"}.issubset(
        contract.requested_features
    )
    prompt = artifact_generation_prompt(query, contract, verifier_feedback=["missing runtime"])
    assert "index.html" in prompt
    assert "prior attempt failed" in prompt


def test_model_queries_are_bounded_and_fallback_is_available():
    parsed = parse_model_queries(
        'thinking\n["official API docs", "licensed examples github", "performance guide", "debugging guide"]',
        ["fallback official docs"],
    )
    assert parsed[0] == "official API docs"
    assert "licensed examples github" in parsed
    assert "fallback official docs" in parsed
    assert len(parsed) <= 6
    assert parse_model_queries("not json", ["fallback official docs"])
    rejected_comments = parse_model_queries(
        "// Now, proceed with implementation\n// documentation is in the source code",
        ["fallback official docs"],
    )
    assert rejected_comments == ["fallback official docs"]


def test_repository_urls_are_normalized_without_accepting_lookalikes():
    assert github_repository_url("https://github.com/mrdoob/three.js/tree/dev/examples") == (
        "https://github.com/mrdoob/three.js.git"
    )
    assert github_repository_url("https://github.example/mrdoob/three.js") is None
    assert github_repository_url("https://github.com/mrdoob") is None


def test_repository_search_preserves_niche_topic_and_path_ranking_avoids_vendor_code():
    assert github_repository_search_terms(
        "sakura three.js voxel shader complete implementation GitHub MIT Apache"
    ) == ["sakura", "threejs", "voxel"]
    ranked = rank_repository_paths(
        [
            "js/vendor/OrbitControls.js",
            "js/vendor/three.min.js",
            "js/main.js",
            "index.html",
            "src/tree/petals.js",
        ],
        "Three.js falling cherry blossom petals",
        preferred_features=("three.js", "sakura", "petal"),
    )
    assert ranked[:3] == ["src/tree/petals.js", "index.html", "js/main.js"]
    assert "js/vendor/three.min.js" not in ranked
    assert ranked.index("js/vendor/OrbitControls.js") > ranked.index("js/main.js")


def test_dataset_holds_out_exact_task_and_round_robins_sources():
    contract = derive_artifact_contract("Build a Three.js animated shader scene")
    sources = [
        {"title": "One", "url": "https://one.test", "text": "A" * 1500, "license": "MIT"},
        {"title": "Two", "url": "https://two.test", "text": "B" * 1500, "license": "MIT"},
    ]
    rows, manifest = build_artifact_dataset("Build a Three.js animated shader scene", sources, contract, max_examples=4)
    assert rows[0]["split"] == "holdout"
    assert [row["source_url"] for row in rows[1:3]] == ["https://one.test", "https://two.test"]
    assert manifest["holdout_rows"] == [0]
    assert manifest["contamination_check"]["exact_task_absent_from_train"] is True
    assert manifest["contamination_check"]["duplicate_chunks"] == 0
    json.dumps(manifest)


def test_static_verifier_rejects_keyword_stub_and_accepts_real_shape(monkeypatch, tmp_path: Path):
    contract = derive_artifact_contract("Build an animated Three.js shader scene")
    stub = tmp_path / "stub.html"
    stub.write_text("<html><script>// THREE. ShaderMaterial requestAnimationFrame resize</script></html>")
    rejected = evaluate_artifact(stub, contract)
    assert not rejected.passed
    assert not rejected.hard_gates["substantial"]

    source = """<!doctype html><html><body><canvas></canvas><script type="module">
const THREE = { ShaderMaterial: class {}, WebGLRenderer: class {} };
const shader = new THREE.ShaderMaterial({vertexShader:'void main(){gl_Position=vec4(0.0);}',fragmentShader:'void main(){}'});
function resize(){ window.devicePixelRatio; window.innerWidth; }
function animate(){ requestAnimationFrame(animate); }
resize(); animate();
</script></body></html>""" + "".join(f"\n<!-- purposeful visual configuration {index} -->" for index in range(200))
    artifact = tmp_path / "index.html"
    artifact.write_text(source)
    monkeypatch.setattr(
        "iloptimus.core.test_time_compute._runtime_render",
        lambda _path: (True, str(tmp_path / "runtime.png"), ""),
    )
    accepted = evaluate_artifact(artifact, contract)
    assert accepted.hard_gates["javascript_syntax"]
    assert accepted.hard_gates["runtime_render"]
    assert accepted.passed


def test_runtime_image_gate_rejects_blank_png(tmp_path: Path):
    blank = tmp_path / "blank.png"
    blank.write_bytes(_png(32, 32, bytes([255, 255, 255]) * 32 * 32))
    assert _png_has_visual_content(blank) is False

    varied = tmp_path / "varied.png"
    varied.write_bytes(
        _png(32, 32, bytes(value for y in range(32) for x in range(32) for value in (x * 8, y * 8, (x + y) * 4)))
    )
    assert _png_has_visual_content(varied) is True


def test_verifier_rejects_duplicate_padding_and_stub_comments(tmp_path: Path):
    contract = derive_artifact_contract("Build a polished Three.js voxel shader scene")
    artifact = tmp_path / "index.html"
    artifact.write_text(
        "<html><script>const x = THREE; function animate(){requestAnimationFrame(animate)}; "
        "// Implement water animation logic\n</script>" + "<script src='three.min.js'></script>" * 500
    )
    result = evaluate_artifact(artifact, contract)
    assert result.hard_gates["substantial"] is False
    assert result.hard_gates["no_placeholders"] is False
    assert result.feature_scores["voxel"] == 0.0


def test_verifier_rejects_source_rendered_as_visible_page_text(tmp_path: Path):
    contract = derive_artifact_contract("Build a responsive Three.js scene")
    artifact = tmp_path / "index.html"
    artifact.write_text(
        "<html><body>" + "function animate(){ const scene = new THREE.Scene(); document.body.textContent = scene; }\n" * 80
        + "<script>function valid(){ return true; }</script></body></html>",
        encoding="utf-8",
    )
    result = evaluate_artifact(artifact, contract)
    assert result.hard_gates["source_not_rendered_as_text"] is False
    assert "renders source code as page text" in " ".join(result.diagnostics)


def test_threejs_framework_is_substantial_and_covers_requested_capabilities(tmp_path: Path):
    query = "Build a polished voxel Sakura Island in Three.js with shader water and animation"
    contract = derive_artifact_contract(query)
    source = framework_artifact_source(query, contract)
    assert source is not None
    assert len(source.encode()) >= contract.minimum_bytes
    assert "new THREE.InstancedMesh" in source
    assert "new THREE.ShaderMaterial" in source
    assert "requestAnimationFrame" in source
    assert "OrbitControls" in source
    assert "window.addEventListener('resize'" in source


def test_method_and_adapter_acceptance_require_evidence_and_improvement():
    contract = derive_artifact_contract("Build an animated Three.js scene")
    no_data = select_method(contract=contract, training_available=True, source_count=1, train_examples=2)
    assert no_data.method == "retrieval"
    qlora = select_method(
        contract=contract,
        training_available=True,
        source_count=8,
        train_examples=96,
        model_params_b=1.5,
        memory_gb=8,
    )
    assert qlora.method == "qlora-il"
    assert qlora.training["iterations"] >= 48
    assert qlora.training["lora_rank"] == 16
    assert qlora.training["lora_layers"] == 8
    assert qlora.training["lora_scale"] == 20.0
    assert qlora.training["max_seq_length"] == 256
    assert qlora.training["compile_bucket_size"] == 32
    assert qlora.training["clear_cache_threshold_gb"] == 1.0
    assert qlora.training["estimated_training_seconds"] <= 600
    assert qlora.training["mask_prompt"] is True
    assert qlora.training["seed"] == 0
    assert qlora.training["optimizer_memory_strategy"] == "unified-memory"
    assert qlora.training["target_epochs"] == 4
    assert qlora.training["iterations"] > 234

    pqlora = select_method(
        contract=contract,
        training_available=True,
        source_count=8,
        train_examples=96,
        model_params_b=7,
        memory_gb=16,
        backend="cuda",
        paged_optimizer_available=True,
    )
    assert pqlora.method == "pqlora-il"
    assert pqlora.training["optimizer_memory_strategy"] == "paged"

    cuda_without_paging = select_method(
        contract=contract,
        training_available=True,
        source_count=8,
        train_examples=96,
        backend="cuda",
        paged_optimizer_available=False,
    )
    assert cuda_without_paging.method == "qlora-il"

    measured = select_method(
        contract=contract,
        training_available=True,
        source_count=8,
        train_examples=80,
        model_params_b=1.5,
        memory_gb=8,
        measured_seconds_per_iteration=3.1,
    )
    assert measured.training["throughput_source"] == "measured-local-profile"
    assert measured.training["seconds_per_iteration"] == 3.1
    assert measured.training["estimated_training_seconds"] <= 600
    assert measured.training["iterations"] < qlora.training["iterations"]

    rl = select_method(
        contract=contract,
        training_available=True,
        source_count=8,
        train_examples=48,
        multi_step_rollout=True,
        deterministic_reward=True,
    )
    assert rl.method == "qlora-il+grpo"

    baseline = evaluate_artifact(Path("/definitely/missing"), contract)
    retry = baseline.__class__(0.9, True, {"exists": True}, {"animation": 1.0}, [], 9000, 200)
    assert acceptance_decision(baseline, retry)["accepted"] is True
    too_close = retry.__class__(0.92, True, {"exists": True}, {"animation": 1.0}, [], 9000, 200)
    assert acceptance_decision(retry, too_close)["accepted"] is False


def test_research_subtasks_require_quantity_kind_and_capability_coverage():
    contract = derive_artifact_contract("Build a Three.js voxel shader scene")
    subtask = next(item for item in research_subtasks("task", contract) if item.capability == "voxel")
    sources = [
        {
            "url": f"https://source-{index}.test/example",
            "title": "Voxel implementation",
            "text": "new THREE.BoxGeometry(); new THREE.InstancedMesh();",
            "kind": "repository-code" if index else "web-documentation",
        }
        for index in range(3)
    ]
    audit = audit_research_subtask(subtask, sources)
    assert audit["passed"] is True
    assert subtask.status == "completed"
    incomplete = sources[:1]
    audit = audit_research_subtask(subtask, incomplete)
    assert audit["passed"] is False
    assert "repository-code" in audit["missing_kinds"]


def test_niche_research_requires_topical_sources_from_independent_origins():
    contract = derive_artifact_contract("Build a Three.js Sakura scene")
    subtask = next(item for item in research_subtasks("task", contract) if item.capability == "sakura")
    generic_particle_sources = [
        {
            "url": f"https://github.com/example/particles/blob/HEAD/demo-{index}.js",
            "title": "Three.js particle demo",
            "text": "const particles = new THREE.Points(geometry, material); animateFallingParticles();",
            "kind": "repository-code",
        }
        for index in range(3)
    ]
    audit = audit_research_subtask(subtask, generic_particle_sources)
    assert audit["passed"] is False
    assert audit["topic_origins"] == 0

    topical_sources = generic_particle_sources + [
        {
            "url": "https://github.com/one/sakura/blob/HEAD/petals.js",
            "title": "Sakura petals",
            "text": "const blossom = new THREE.Points(geometry, material);",
            "kind": "repository-code",
        },
        {
            "url": "https://github.com/two/cherry-tree/blob/HEAD/tree.js",
            "title": "Cherry blossom tree",
            "text": "const cherry = new THREE.Sprite(material);",
            "kind": "repository-code",
        },
    ]
    audit = audit_research_subtask(subtask, topical_sources)
    assert audit["passed"] is True
    assert audit["topic_sources"] == 2
    assert audit["topic_origins"] == 2


def test_fast_research_planner_is_bounded_deterministic_and_failure_focused():
    contract = derive_artifact_contract(
        "Build a polished Three.js voxel Sakura island with custom shaders and animation"
    )
    diagnostics = [
        "Requested feature is not implemented observably: shader",
        "Requested feature is not implemented observably: voxel",
    ]
    first = fast_research_queries(contract, diagnostics)
    second = fast_research_queries(contract, diagnostics)
    assert first == second
    assert len(first) <= 14
    assert len(first) == len(set(first))
    assert any("shader" in query.casefold() and "github" in query.casefold() for query in first)
    assert any("voxel" in query.casefold() and "github" in query.casefold() for query in first)
