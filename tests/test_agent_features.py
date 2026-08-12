import asyncio

import pytest
from iloptimus.core.hardware import GPUInfo, HardwareInfo
from iloptimus.core.models import get_model
from iloptimus.core.performance import estimate_context_performance
from iloptimus.core.skills import list_prompt_skills, route_prompt_skills
from iloptimus.core.tools import calculate, parse_tool_call, validate_public_url
from iloptimus.server import _trim_history, create_app


def test_packaged_skills_are_discoverable_and_frontend_routes_automatically():
    skills = list_prompt_skills()
    assert {skill.id for skill in skills} == {
        "frontend-design",
        "jupyter-notebook",
        "playwright",
        "security-best-practices",
    }
    selected = route_prompt_skills("Build a polished React frontend dashboard with responsive CSS")
    assert selected[0].id == "frontend-design"


def test_security_skill_requires_security_language():
    assert "security-best-practices" not in {
        skill.id for skill in route_prompt_skills("Create a normal login form component")
    }
    assert "security-best-practices" in {
        skill.id for skill in route_prompt_skills("Perform an OWASP security review for injection")
    }


def test_tool_call_parser_and_calculator_are_constrained():
    call = parse_tool_call('<tool_call>{"name":"calculator","arguments":{"expression":"(12 + 4) / 2"}}</tool_call>')
    assert call == ("calculator", {"expression": "(12 + 4) / 2"})
    assert calculate(call[1]["expression"]) == 8
    with pytest.raises(ValueError):
        calculate("__import__('os').getcwd()")


def test_public_url_validation_blocks_local_networks():
    with pytest.raises(ValueError, match="blocked"):
        asyncio.run(validate_public_url("http://127.0.0.1/private"))
    with pytest.raises(ValueError, match="ports"):
        asyncio.run(validate_public_url("https://example.com:8443"))


def test_context_estimate_uses_model_and_hardware_capacity(tmp_path, monkeypatch):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    model = get_model("qwen2.5-1.5b")
    hardware = HardwareInfo(
        cpu_name="Apple M2 Pro",
        cpu_cores=10,
        ram_gb=16,
        os="macOS",
        arch="arm64",
        gpu=GPUInfo(name="Apple M2 Pro", vram_gb=16, type="apple-silicon"),
        python_version="3.12",
        mlx_available=True,
        recommended_backend="mlx",
    )
    estimate = estimate_context_performance(model, hardware, 16_384)
    assert estimate.context_window == 16_384
    assert 2_048 <= estimate.max_safe_context <= model.context_length
    assert estimate.estimated_tps > 0
    assert estimate.low_tps < estimate.high_tps
    assert estimate.kv_cache_gb > 0


def test_history_trimming_keeps_recent_messages():
    history = [
        {"role": "user", "text": "old " * 200},
        {"role": "assistant", "text": "middle " * 20},
        {"role": "user", "text": "newest"},
    ]
    trimmed = _trim_history(history, 80)
    assert trimmed[-1]["text"] == "newest"
    assert all(item["text"] != history[0]["text"] for item in trimmed)


def test_agent_metadata_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    assert len(client.get("/api/skills").json()) == 4
    tools = client.get("/api/tools").json()
    assert {tool["name"] for tool in tools["built_in"]} >= {"web_search", "web_fetch"}
    assert {server["id"] for server in tools["mcp_servers"]} == {"fetch", "time"}
    estimate = client.get("/api/models/qwen2.5-1.5b/context-estimate?context_window=8192")
    assert estimate.status_code == 200
    assert estimate.json()["context_window"] == 8192
