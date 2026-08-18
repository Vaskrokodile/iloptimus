from __future__ import annotations

import json

from iloptimus.core.models import get_model
from iloptimus.core.pipeline import RunConfig, create_run
from iloptimus.core.run_manifest import build_run_manifest
from iloptimus.core.storage import run_dir
from iloptimus.core.tasksets import get_taskset


def test_run_manifest_captures_reproducibility_context():
    config = RunConfig(model_id="qwen2.5-0.5b", taskset_id="gsm8k-v1")
    manifest = build_run_manifest(
        config,
        model=get_model(config.model_id),
        taskset=get_taskset(config.taskset_id),
    )

    assert manifest["schema_version"] == 1
    assert manifest["config"]["model_id"] == config.model_id
    assert manifest["model"]["huggingface_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert manifest["taskset"]["id"] == config.taskset_id
    assert "python" in manifest["runtime"]
    assert "packages" in manifest["runtime"]
    assert "revision" in manifest["git"]


def test_create_run_persists_manifest_alongside_run_state(monkeypatch, tmp_path):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    config = RunConfig(model_id="qwen2.5-0.5b", taskset_id="gsm8k-v1")
    manifest = {"schema_version": 1, "source": "test", "config": {"model_id": config.model_id}}

    state = create_run(config, manifest=manifest)
    folder = run_dir(state.id)

    assert json.loads((folder / "manifest.json").read_text()) == manifest
    saved_state = json.loads((folder / "run.json").read_text())
    assert saved_state["manifest"] == manifest
