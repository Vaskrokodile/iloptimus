import json
from pathlib import Path

from iloptimus.core.test_time_compute import (
    _png_has_visual_content,
    acceptance_decision,
    artifact_generation_prompt,
    build_artifact_dataset,
    derive_artifact_contract,
    evaluate_artifact,
    github_repository_url,
    parse_model_queries,
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


def test_repository_urls_are_normalized_without_accepting_lookalikes():
    assert github_repository_url("https://github.com/mrdoob/three.js/tree/dev/examples") == (
        "https://github.com/mrdoob/three.js.git"
    )
    assert github_repository_url("https://github.example/mrdoob/three.js") is None
    assert github_repository_url("https://github.com/mrdoob") is None


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


def test_method_and_adapter_acceptance_require_evidence_and_improvement():
    contract = derive_artifact_contract("Build an animated Three.js scene")
    no_data = select_method(contract=contract, training_available=True, source_count=1, train_examples=2)
    assert no_data.method == "retrieval"
    qlora = select_method(contract=contract, training_available=True, source_count=4, train_examples=12)
    assert qlora.method == "qlora-il"

    baseline = evaluate_artifact(Path("/definitely/missing"), contract)
    retry = baseline.__class__(0.9, True, {"exists": True}, {"animation": 1.0}, [], 9000, 200)
    assert acceptance_decision(baseline, retry)["accepted"] is True
    too_close = retry.__class__(0.92, True, {"exists": True}, {"animation": 1.0}, [], 9000, 200)
    assert acceptance_decision(retry, too_close)["accepted"] is False
