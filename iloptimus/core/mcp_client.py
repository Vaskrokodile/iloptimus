"""Small, explicit MCP client boundary for locally configured servers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .storage import app_home, atomic_write_json

DEFAULT_SERVERS = {
    "time": {
        "enabled": False,
        "command": "uvx",
        "args": ["mcp-server-time"],
        "description": "Official MCP reference server for timezone-aware time conversion.",
    },
    "fetch": {
        "enabled": False,
        "command": "uvx",
        "args": ["mcp-server-fetch"],
        "description": "Official MCP reference fetch server. Disabled by default; the built-in safe fetch tool is preferred.",
    },
}


def config_path() -> Path:
    return app_home() / "mcp.json"


def load_mcp_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        payload = {"servers": DEFAULT_SERVERS}
        atomic_write_json(path, payload)
        return payload
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"servers": DEFAULT_SERVERS}
    return payload if isinstance(payload.get("servers"), dict) else {"servers": DEFAULT_SERVERS}


def public_mcp_servers() -> list[dict[str, Any]]:
    return [
        {
            "id": server_id,
            "enabled": bool(config.get("enabled")),
            "description": str(config.get("description", "Configured MCP server")),
            "transport": "stdio",
        }
        for server_id, config in load_mcp_config()["servers"].items()
    ]


@dataclass(frozen=True)
class MCPTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    server_id: str
    remote_name: str


def _server_params(server_id: str) -> StdioServerParameters:
    config = load_mcp_config()["servers"].get(server_id)
    if not isinstance(config, dict) or not config.get("enabled"):
        raise ValueError(f"MCP server '{server_id}' is not enabled")
    command = config.get("command")
    args = config.get("args", [])
    env = config.get("env")
    if not isinstance(command, str) or not command or not isinstance(args, list):
        raise ValueError(f"MCP server '{server_id}' has an invalid configuration")
    return StdioServerParameters(
        command=command,
        args=[str(item) for item in args],
        env={str(key): str(value) for key, value in env.items()} if isinstance(env, dict) else None,
    )


async def list_mcp_tools() -> list[MCPTool]:
    tools: list[MCPTool] = []
    for server in public_mcp_servers():
        if not server["enabled"]:
            continue
        server_id = server["id"]
        try:
            async with stdio_client(_server_params(server_id)) as (reader, writer):
                async with ClientSession(reader, writer, read_timeout_seconds=timedelta(seconds=12)) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    tools.extend(
                        MCPTool(
                            name=f"mcp.{server_id}.{tool.name}",
                            description=tool.description or f"MCP tool {tool.name}",
                            input_schema=tool.inputSchema,
                            server_id=server_id,
                            remote_name=tool.name,
                        )
                        for tool in result.tools
                    )
        except Exception:
            continue
    return tools


async def call_mcp_tool(tool: MCPTool, arguments: dict[str, Any]) -> dict[str, Any]:
    async with stdio_client(_server_params(tool.server_id)) as (reader, writer):
        async with ClientSession(reader, writer, read_timeout_seconds=timedelta(seconds=20)) as session:
            await session.initialize()
            result = await session.call_tool(tool.remote_name, arguments)
            content: list[dict[str, Any]] = []
            for item in result.content:
                if hasattr(item, "model_dump"):
                    content.append(item.model_dump(mode="json"))
                else:
                    content.append({"type": "text", "text": str(item)})
            return {"is_error": bool(result.isError), "content": content[:20]}
