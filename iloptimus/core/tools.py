"""Registered low-risk tools available to local chat models."""

from __future__ import annotations

import ast
import asyncio
import html
import ipaddress
import json
import math
import operator
import re
import socket
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse

import httpx

from .mcp_client import MCPTool, call_mcp_tool, list_mcp_tools, public_mcp_servers
from .storage import app_home

MAX_WEB_BYTES = 128_000
USER_AGENT = "ILOptimus/0.2 (+local AI research workspace)"


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    source: str = "built-in"


BUILTIN_TOOLS = [
    ToolDefinition(
        "web_search",
        "Search the public web. Use for current information, unfamiliar topics, and finding sources.",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    ),
    ToolDefinition(
        "web_fetch",
        "Read a public HTTP or HTTPS page. Private, loopback, and link-local destinations are blocked.",
        {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    ),
    ToolDefinition(
        "calculator",
        "Evaluate a basic arithmetic expression without executing code.",
        {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    ),
    ToolDefinition(
        "current_time",
        "Return the current UTC timestamp.",
        {"type": "object", "properties": {}},
    ),
]


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored += 1
        elif tag in {"p", "br", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._ignored:
            self._ignored -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored:
            self.parts.append(data)

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", " ".join(self.parts)).strip()


class _SearchParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._url: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "a" and "result__a" in values.get("class", ""):
            self._url = values.get("href")
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._url is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._url is not None:
            parsed = urlparse(html.unescape(self._url))
            query = parse_qs(parsed.query)
            url = unquote(query.get("uddg", [self._url])[0])
            self.results.append({"title": " ".join(self._text).strip(), "url": url})
            self._url = None


async def _public_host(hostname: str) -> bool:
    try:
        records = await asyncio.get_running_loop().run_in_executor(
            None, lambda: socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
        )
    except socket.gaierror:
        return False
    addresses = {record[4][0].split("%", 1)[0] for record in records}
    if not addresses:
        return False
    return all(
        not (
            (ip := ipaddress.ip_address(address)).is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
        for address in addresses
    )


async def validate_public_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Only public HTTP(S) URLs without credentials are allowed")
    if parsed.port not in {None, 80, 443}:
        raise ValueError("Only standard HTTP(S) ports are allowed")
    if not await _public_host(parsed.hostname):
        raise ValueError("Private, local, reserved, or unresolved destinations are blocked")
    return url


async def web_fetch(url: str) -> dict[str, Any]:
    current = await validate_public_url(url)
    async with httpx.AsyncClient(timeout=12, headers={"User-Agent": USER_AGENT}) as client:
        for _ in range(4):
            response = await client.get(current, follow_redirects=False)
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise ValueError("Redirect had no destination")
                current = await validate_public_url(urljoin(current, location))
                continue
            response.raise_for_status()
            raw = response.content[:MAX_WEB_BYTES]
            content_type = response.headers.get("content-type", "")
            decoded = raw.decode(response.encoding or "utf-8", errors="replace")
            if "html" in content_type.lower() or "<html" in decoded[:500].lower():
                parser = _TextExtractor()
                parser.feed(decoded)
                decoded = parser.text()
            return {
                "url": str(response.url),
                "status": response.status_code,
                "content_type": content_type,
                "text": decoded[:40_000],
                "truncated": len(response.content) > len(raw) or len(decoded) > 40_000,
            }
    raise ValueError("Too many redirects")


async def web_search(query: str) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("Search query cannot be empty")
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query[:400])}"
    page = await web_fetch(url)
    # web_fetch strips HTML, so retrieve the fixed, already-validated search host
    async with httpx.AsyncClient(timeout=12, headers={"User-Agent": USER_AGENT}) as client:
        response = await client.get(url)
        response.raise_for_status()
    parser = _SearchParser()
    parser.feed(response.text[:MAX_WEB_BYTES])
    return {"query": query, "results": parser.results[:8], "search_page": page["url"]}


_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def calculate(expression: str) -> float | int:
    if len(expression) > 200:
        raise ValueError("Expression is too long")

    def evaluate(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError("Exponent is too large")
            value = _BINARY[type(node.op)](left, right)
            if not math.isfinite(float(value)) or abs(float(value)) > 1e100:
                raise ValueError("Result is outside the safe numeric range")
            return value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](evaluate(node.operand))
        raise ValueError("Only basic arithmetic is supported")

    return evaluate(ast.parse(expression, mode="eval"))


def parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    candidates = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.DOTALL)
    candidates += re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        name = payload.get("name") or payload.get("tool")
        arguments = payload.get("arguments", payload.get("input", {}))
        if isinstance(name, str) and isinstance(arguments, dict):
            return name, arguments
    return None


async def tool_definitions() -> tuple[list[ToolDefinition], dict[str, MCPTool]]:
    mcp_tools = await list_mcp_tools()
    mapping = {tool.name: tool for tool in mcp_tools}
    definitions = BUILTIN_TOOLS + [
        ToolDefinition(tool.name, tool.description, tool.input_schema, source=f"mcp:{tool.server_id}")
        for tool in mcp_tools
    ]
    return definitions, mapping


def tool_prompt(definitions: list[ToolDefinition]) -> str:
    schemas = [asdict(definition) for definition in definitions]
    return (
        "You may call one tool when needed. Reply with ONLY "
        '<tool_call>{"name":"tool_name","arguments":{...}}</tool_call>. '
        "After receiving a TOOL_RESULT, answer the user and cite returned URLs.\n"
        f"Available tools:\n{json.dumps(schemas, ensure_ascii=False)}"
    )


async def execute_tool(name: str, arguments: dict[str, Any], mcp_tools: dict[str, MCPTool]) -> dict[str, Any]:
    started = time.monotonic()
    try:
        if name == "web_search":
            result = await web_search(str(arguments.get("query", "")))
        elif name == "web_fetch":
            result = await web_fetch(str(arguments.get("url", "")))
        elif name == "calculator":
            result = {"result": calculate(str(arguments.get("expression", "")))}
        elif name == "current_time":
            result = {"utc": datetime.now(UTC).isoformat()}
        elif name in mcp_tools:
            result = await call_mcp_tool(mcp_tools[name], arguments)
        else:
            raise ValueError(f"Unknown or disabled tool: {name}")
        status = "ok"
        return {"ok": True, "result": result}
    except Exception as error:
        status = "error"
        return {"ok": False, "error": str(error)}
    finally:
        log = {
            "timestamp": datetime.now(UTC).isoformat(),
            "tool": name,
            "status": status,
            "elapsed_ms": round((time.monotonic() - started) * 1000),
        }
        path = app_home() / "tool_calls.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log) + "\n")


def tools_public() -> dict[str, Any]:
    return {
        "built_in": [asdict(tool) for tool in BUILTIN_TOOLS],
        "mcp_servers": public_mcp_servers(),
        "audit_log": str(app_home() / "tool_calls.jsonl"),
    }
