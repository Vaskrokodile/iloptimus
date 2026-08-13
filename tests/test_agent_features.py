import asyncio
import json

import pytest
from iloptimus.core.dataset_tools import save_source_bundle
from iloptimus.core.hardware import GPUInfo, HardwareInfo
from iloptimus.core.models import get_model
from iloptimus.core.performance import estimate_context_performance, record_chat_performance
from iloptimus.core.skills import list_prompt_skills, route_prompt_skills
from iloptimus.core.tools import (
    calculate,
    execute_tool,
    ground_tool_answer,
    looks_like_tool_call,
    normalize_tool_call,
    parse_tool_call,
    parse_tool_calls,
    suggested_tool_call,
    tool_answer_needs_fallback,
    tool_result_fallback,
    validate_public_url,
)
from iloptimus.server import (
    OpenAIChatRequest,
    _openai_prompt,
    _openai_response_payload,
    _responses_tool_subset,
    _trim_history,
    create_app,
)


def test_packaged_skills_are_discoverable_and_frontend_routes_automatically():
    skills = list_prompt_skills()
    assert {skill.id for skill in skills} == {
        "frontend-design",
        "jupyter-notebook",
        "knowledge-dataset",
        "playwright",
        "security-best-practices",
        "test-time-artifact",
    }
    selected = route_prompt_skills("Build a polished React frontend dashboard with responsive CSS")
    assert selected[0].id == "frontend-design"
    assert route_prompt_skills("Find the official Model Context Protocol website") == []


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


def test_tool_parser_accepts_nested_fenced_small_model_format_and_repairs_query():
    response = """```json
{
  "tool_name": "web_search",
  "arguments": {},
  "source": "built-in"
}
```"""
    call = parse_tool_call(response)
    assert call == ("web_search", {})
    assert normalize_tool_call(call, "search the web for current MLX releases", {"web_search"}) == (
        "web_search",
        {"query": "search the web for current MLX releases"},
    )


def test_tool_parser_recovers_multiple_small_model_shorthand_calls():
    response = """```json
{"create_directory":{"path":"demo"}},
{"write_file":{"path":"demo/add.py","content":"print(42)"}},
{"run_command":{"command":"python add.py","cwd":"demo"}}
```"""
    assert parse_tool_calls(response) == [
        ("create_directory", {"path": "demo"}),
        ("write_file", {"path": "demo/add.py", "content": "print(42)"}),
        ("run_command", {"command": "python add.py", "cwd": "demo"}),
    ]


def test_tool_parser_recovers_parameters_misplaced_beside_scalar_arguments():
    response = '{"tool_name":"write_file","arguments":1060,"path":"proof/a.py","content":"print(1060)"}'
    assert parse_tool_call(response) == (
        "write_file",
        {"path": "proof/a.py", "content": "print(1060)"},
    )


def test_tool_parser_repairs_truncated_outer_object_brace():
    response = '{"tool_name":"write_file","arguments":{"path":"proof/a.py","content":"print(1)"}'
    assert parse_tool_call(response) == (
        "write_file",
        {"path": "proof/a.py", "content": "print(1)"},
    )


def test_explicit_web_requests_are_planned_without_model_formatting():
    assert suggested_tool_call("Please search online for MCP news", {"web_search", "web_fetch"}) == (
        "web_search",
        {"query": "Please search online for MCP news"},
    )
    assert suggested_tool_call("Read https://example.com/docs", {"web_search", "web_fetch"}) == (
        "web_fetch",
        {"url": "https://example.com/docs"},
    )


def test_raw_tool_requests_get_a_readable_fallback():
    raw = '{"tool_name":"web_search","arguments":{}}'
    assert looks_like_tool_call(raw, {"web_search"})
    answer = tool_result_fallback(
        "web_search",
        {"ok": True, "result": {"results": [{"title": "Official docs", "url": "https://example.com"}]}},
    )
    assert "Official docs" in answer
    assert "tool_name" not in answer
    assert tool_answer_needs_fallback("<answer>Wrong</answer>\n</think>", {"web_search"})
    assert tool_answer_needs_fallback("<answer>Ungrounded tool synthesis</answer>", {"web_search"})
    grounded = ground_tool_answer(
        "The source is https://html.duckduckgo.com/html/?q=test",
        "web_search",
        {"ok": True, "result": {"results": [{"title": "Official docs", "url": "https://example.com"}]}},
        {"web_search"},
    )
    assert "Official docs" in grounded
    assert "duckduckgo.com" not in grounded


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


def test_optional_performance_telemetry_cannot_break_chat(monkeypatch):
    def disk_full(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("iloptimus.core.performance.atomic_write_json", disk_full)
    record_chat_performance("qwen2.5-1.5b", 2048, 22.0)


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
    assert len(client.get("/api/skills").json()) == 6
    tool_names = {tool["name"] for tool in client.get("/api/tools").json()["built_in"]}
    assert {"scrape_source", "assemble_dataset", "expand_dataset", "filter_dataset"}.issubset(tool_names)
    tools = client.get("/api/tools").json()
    assert {tool["name"] for tool in tools["built_in"]} >= {"web_search", "web_fetch"}
    assert {server["id"] for server in tools["mcp_servers"]} == {"fetch", "time"}
    estimate = client.get("/api/models/qwen2.5-1.5b/context-estimate?context_window=8192")
    assert estimate.status_code == 200
    assert estimate.json()["context_window"] == 8192


def test_dataset_tools_execute_through_the_chat_tool_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv("ILOPTIMUS_HOME", str(tmp_path))
    source = "\n".join(
        f"const mesh{index} = new THREE.InstancedMesh(new THREE.BoxGeometry(), material, {index + 1});"
        for index in range(36)
    )
    save_source_bundle(
        "tool-workspace",
        [
            {
                "title": "Licensed voxel implementation",
                "url": "https://github.com/example/voxel/blob/HEAD/src/main.js",
                "text": source,
                "license": "MIT",
                "kind": "repository-code",
            }
        ],
    )
    assembled = asyncio.run(
        execute_tool(
            "assemble_dataset",
            {
                "workspace_id": "tool-workspace",
                "task": "held-out request",
                "artifact_kind": "web",
                "requested_features": ["three.js", "voxel"],
                "target_examples": 24,
            },
            {},
        )
    )
    assert assembled["ok"] is True
    expanded = asyncio.run(
        execute_tool("expand_dataset", {"workspace_id": "tool-workspace", "target_examples": 32}, {})
    )
    assert expanded["ok"] is True
    filtered = asyncio.run(
        execute_tool(
            "filter_dataset",
            {"workspace_id": "tool-workspace", "holdout_task": "held-out request", "maximum_rows": 32},
            {},
        )
    )
    assert filtered["ok"] is True
    assert filtered["result"]["accepted_rows"] > 0


def test_openai_compatibility_prompt_and_native_tool_response():
    request = OpenAIChatRequest(
        model="qwen2.5-1.5b",
        messages=[{"role": "user", "content": "Create demo/app.py"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["path", "content"],
                    },
                },
            }
        ],
    )
    prompt = _openai_prompt(request)
    assert "write_file" in prompt
    assert "Create demo/app.py" in prompt

    payload = _openai_response_payload(
        request,
        '```json\n{"tool_name":"write_file","arguments":{"path":"demo/app.py","content":"print(5)"}}\n```',
        20,
    )
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "write_file"
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"])["path"] == "demo/app.py"


def test_responses_adapter_hides_irrelevant_codex_tools_from_small_models():
    tools = [
        {"type": "function", "name": "apply_patch", "description": "Patch files", "parameters": {}},
        {"type": "function", "name": "exec_command", "description": "Run commands", "parameters": {}},
        {"type": "function", "name": "request_user_input", "description": "Ask", "parameters": {}},
    ]
    assert _responses_tool_subset(tools, ["user: Reply only with READY."]) == []
    selected = _responses_tool_subset(tools, ["user: Create hello.py and run it."])
    assert [tool["function"]["name"] for tool in selected] == ["apply_patch", "exec_command"]
