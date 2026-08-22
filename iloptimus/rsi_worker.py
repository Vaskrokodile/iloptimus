"""Packaged, workspace-confined RSI coding worker used by web panels.

The standalone TypeScript terminal remains the richer interactive client. This
worker deliberately implements the same small verified control loop using only
Optimus Studio runtime dependencies, so a ``uv tool install`` desktop build works
without Bun or a repository checkout.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

SYSTEM_PROMPT = (
    "You are RSI, a terminal coding agent restricted to one admitted workspace. "
    "Use tools to inspect, create, edit, and execute files. Never claim an action happened unless its tool succeeded. "
    "Call exactly one tool at a time, wait for its result, and finish with a short factual summary."
)


def _function(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict:
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return {"type": "function", "function": {"name": name, "description": description, "parameters": schema}}


TOOLS = {
    "read_file": _function("read_file", "Read a UTF-8 text file.", {"path": {"type": "string"}}, ["path"]),
    "write_file": _function(
        "write_file", "Write complete text content, creating parents.",
        {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"],
    ),
    "edit_file": _function(
        "edit_file", "Replace one exact unique string in a text file.",
        {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}},
        ["path", "old_string", "new_string"],
    ),
    "list_directory": _function("list_directory", "List directory entries.", {"path": {"type": "string"}}),
    "create_directory": _function("create_directory", "Create a directory and parents.", {"path": {"type": "string"}}, ["path"]),
    "run_command": _function(
        "run_command", "Run a shell command in the admitted workspace and return real output and exit code.",
        {"command": {"type": "string"}, "cwd": {"type": "string"}, "timeout": {"type": "number"}}, ["command"],
    ),
    "glob_search": _function(
        "glob_search", "Find files matching a glob under the workspace.",
        {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"],
    ),
    "grep_search": _function(
        "grep_search", "Search text files with a regular expression.",
        {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"],
    ),
}


@dataclass(frozen=True)
class Requirements:
    mutate: bool
    execute: bool
    requested_paths: tuple[str, ...]
    expected_output: str | None
    required_symbols: tuple[str, ...]


def requirements(prompt: str) -> Requirements:
    paths = tuple(re.findall(r"\b[\w.-]+(?:/[\w.-]+)*\.(?:py|js|ts|tsx|jsx|json|md|txt|html|css|sh)\b", prompt, re.I))
    expected = re.search(r"output\s+(?:is\s+)?exactly\s+([^\s.,;]+)", prompt, re.I)
    symbols = tuple(re.findall(r"\bimplement(?:ing)?\s+([A-Za-z_]\w*)\s*\(", prompt, re.I))
    lowered = prompt.lower()
    return Requirements(
        mutate=bool(re.search(r"\b(create|write|edit|modify|fix|build|implement|folder|file|code)\b", lowered)),
        execute=bool(re.search(r"\b(run|execute|test|verify|check|compile|build)\b", lowered)),
        requested_paths=paths,
        expected_output=expected.group(1) if expected else None,
        required_symbols=symbols,
    )


class Worker:
    def __init__(self) -> None:
        self.panel_id = os.environ.get("RSI_PANEL_ID", uuid.uuid4().hex[:12])
        self.session_id = uuid.uuid4().hex
        self.workspace = Path(os.environ.get("RSI_WORKSPACE", os.getcwd())).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.model = os.environ.get("RSI_MODEL_ID", "deepseek-r1-distill-qwen-1.5b")
        self.base_url = os.environ.get("RSI_PROVIDER_BASE_URL", "http://127.0.0.1:7860/v1").rstrip("/")
        self.max_tokens = int(os.environ.get("RSI_MAX_TOKENS", "384"))
        self.max_steps = int(os.environ.get("RSI_MAX_STEPS", "20"))
        self.session_path = Path(os.environ.get("RSI_SESSION_PATH", str(self.workspace / ".rsi-session.jsonl")))
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def emit(self, event_type: str, **data: Any) -> None:
        print(json.dumps({"type": event_type, "panelId": self.panel_id, "timestamp": time.time(), "data": data}), flush=True)

    def persist(self, payload: dict[str, Any]) -> None:
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        with self.session_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def confined(self, value: str | None) -> Path:
        candidate = (self.workspace / (value or ".")).resolve()
        if candidate != self.workspace and self.workspace not in candidate.parents:
            raise ValueError("Path is outside the admitted workspace")
        return candidate

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        try:
            if name == "read_file":
                return True, self.confined(str(arguments.get("path", ""))).read_text(encoding="utf-8")[:20_000]
            if name == "write_file":
                path = self.confined(str(arguments.get("path", "")))
                path.parent.mkdir(parents=True, exist_ok=True)
                content = str(arguments.get("content", ""))
                path.write_text(content, encoding="utf-8")
                return True, f"Wrote {len(content.encode())} bytes to {path.relative_to(self.workspace)}"
            if name == "edit_file":
                path = self.confined(str(arguments.get("path", "")))
                old, new = str(arguments.get("old_string", "")), str(arguments.get("new_string", ""))
                content = path.read_text(encoding="utf-8")
                if not old or content.count(old) != 1:
                    raise ValueError("old_string must match exactly once")
                path.write_text(content.replace(old, new), encoding="utf-8")
                return True, f"Edited {path.relative_to(self.workspace)}"
            if name == "create_directory":
                path = self.confined(str(arguments.get("path", "")))
                path.mkdir(parents=True, exist_ok=True)
                return True, f"Created {path.relative_to(self.workspace)}"
            if name == "list_directory":
                path = self.confined(str(arguments.get("path") or "."))
                return True, "\n".join(f"{'dir' if item.is_dir() else 'file'}\t{item.name}" for item in sorted(path.iterdir()))[:20_000]
            if name == "glob_search":
                root = self.confined(str(arguments.get("path") or "."))
                pattern = str(arguments.get("pattern") or "*")
                matches = [
                    str(item.relative_to(self.workspace))
                    for item in root.glob(pattern)
                    if self.confined(str(item))
                ]
                return True, "\n".join(matches[:500])
            if name == "grep_search":
                root = self.confined(str(arguments.get("path") or "."))
                pattern = re.compile(str(arguments.get("pattern") or ""))
                rows: list[str] = []
                for path in root.rglob("*"):
                    if not path.is_file() or len(rows) >= 500:
                        continue
                    try:
                        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                            if pattern.search(line):
                                rows.append(f"{path.relative_to(self.workspace)}:{number}:{line[:500]}")
                    except (OSError, UnicodeDecodeError):
                        continue
                return True, "\n".join(rows)
            if name == "run_command":
                cwd = self.confined(str(arguments.get("cwd") or "."))
                command = str(arguments.get("command") or "")
                if not command:
                    raise ValueError("command is required")
                timeout = min(300, max(1, float(arguments.get("timeout", 60_000)) / 1000))
                completed = subprocess.run(
                    ["/bin/zsh", "-lc", command], cwd=cwd, capture_output=True, text=True, timeout=timeout,
                    env={**os.environ, "PWD": str(cwd)},
                )
                output = (completed.stdout + completed.stderr)[-20_000:]
                return completed.returncode == 0, f"{output}\n[exit code {completed.returncode}]"
            raise ValueError(f"Unknown tool: {name}")
        except (OSError, ValueError, subprocess.SubprocessError, re.error) as error:
            return False, str(error)

    def completion(self, available: list[str]) -> tuple[str, list[dict[str, Any]]]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "tools": [TOOLS[name] for name in available],
            "stream": False,
            "max_tokens": self.max_tokens,
            "temperature": 0.1,
        }
        if len(available) == 1:
            body["tool_choice"] = "required"
        with httpx.Client(timeout=180) as client:
            response = client.post(f"{self.base_url}/chat/completions", json=body, headers={"Authorization": "Bearer local"})
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
        calls = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function") or {}
            try:
                args = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": raw.get("id") or f"call_{uuid.uuid4().hex[:12]}", "name": function.get("name", ""), "args": args})
        return str(message.get("content") or ""), calls

    def run(self, prompt: str) -> None:
        req = requirements(prompt)
        self.messages.append({"role": "user", "content": prompt})
        self.persist(self.messages[-1])
        self.emit("started", prompt=prompt, sessionId=self.session_id, workspace=str(self.workspace))
        successful: set[str] = set()
        last_run_failed = False
        controller_retries = 0
        final_text = ""
        for step in range(1, self.max_steps + 1):
            mutated = bool(successful & {"write_file", "edit_file", "create_directory"})
            if req.mutate and not mutated or last_run_failed:
                available = ["write_file"]
            elif req.execute and "run_command" not in successful:
                available = ["run_command"]
            else:
                available = list(TOOLS)
            try:
                final_text, calls = self.completion(available)
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
                self.emit("failed", error=str(error), steps=step)
                return
            self.emit("assistant_message", text=final_text, toolCalls=calls)
            assistant: dict[str, Any] = {"role": "assistant", "content": final_text or None}
            if calls:
                assistant["tool_calls"] = [
                    {"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": json.dumps(call["args"])}}
                    for call in calls
                ]
            self.messages.append(assistant)
            self.persist(assistant)
            if not calls:
                missing = []
                if req.mutate and not mutated:
                    missing.append("a successful file mutation")
                if req.execute and "run_command" not in successful:
                    missing.append("a successful command verification")
                if missing and controller_retries < 3:
                    controller_retries += 1
                    correction = "CONTROLLER: Not complete; perform " + " and ".join(missing) + ". Call the one available tool now."
                    self.messages.append({"role": "user", "content": correction})
                    self.emit("controller_retry", retry=controller_retries, missing=missing)
                    continue
                if missing:
                    self.emit("failed", error="Model stopped without " + " and ".join(missing), steps=step)
                else:
                    verified = []
                    if req.mutate:
                        verified.append("the requested file mutation succeeded")
                    if req.execute:
                        verified.append("the command completed with exit code 0")
                    if req.expected_output:
                        verified.append(f"output contained the exact line {req.expected_output}")
                    summary = "Verified: " + "; ".join(verified) + "." if verified else final_text
                    self.emit("completed", text=summary, steps=step, sessionId=self.session_id)
                return
            for call in calls:
                self.emit("tool_call", id=call["id"], name=call["name"], arguments=call["args"])
                if call["name"] not in available:
                    ok, output = False, f"{call['name']} was not admitted for this step"
                else:
                    ok, output = self.execute_tool(call["name"], call["args"])
                actual = str(call["args"].get("path", "")).replace("\\", "/")
                if ok and call["name"] in {"write_file", "edit_file"} and req.requested_paths:
                    if not any(actual.endswith(path) for path in req.requested_paths):
                        ok, output = False, "Verification failed: write the requested path " + " or ".join(req.requested_paths)
                if ok and call["name"] == "write_file" and req.required_symbols:
                    content = str(call["args"].get("content", ""))
                    missing = [symbol for symbol in req.required_symbols if not re.search(rf"\bdef\s+{re.escape(symbol)}\s*\(", content)]
                    if missing:
                        ok, output = False, "Verification failed: missing exact function " + ", ".join(missing)
                if ok and call["name"] == "write_file" and req.execute and str(call["args"].get("path", "")).endswith(".py"):
                    if "print(" not in str(call["args"].get("content", "")):
                        ok, output = False, "Verification failed: the Python file must print the requested result when executed"
                if ok and call["name"] == "run_command" and req.expected_output:
                    if req.expected_output not in [line.strip() for line in output.splitlines()]:
                        ok, output = False, f"Verification failed: expected output exactly {req.expected_output}. Actual:\n{output}"
                self.emit("tool_result", id=call["id"], name=call["name"], content=output, isError=not ok)
                if ok:
                    successful.add(call["name"])
                    if call["name"] in {"write_file", "edit_file"}:
                        last_run_failed = False
                if call["name"] == "run_command":
                    last_run_failed = not ok
                tool_message = {"role": "tool", "tool_call_id": call["id"], "content": output}
                self.messages.append(tool_message)
                self.persist(tool_message)
        self.emit("failed", error="Maximum agent steps reached", steps=self.max_steps)


def main() -> int:
    worker = Worker()
    print(json.dumps({"type": "ready", "panelId": worker.panel_id, "sessionId": worker.session_id, "workspace": str(worker.workspace), "model": worker.model}), flush=True)
    for line in sys.stdin:
        try:
            command = json.loads(line)
        except json.JSONDecodeError:
            worker.emit("protocol_error", error="Input must be one JSON object per line")
            continue
        if command.get("type") == "shutdown":
            break
        if command.get("type") != "prompt" or not isinstance(command.get("prompt"), str):
            worker.emit("protocol_error", error="Expected a prompt command")
            continue
        worker.run(command["prompt"])
    print(json.dumps({"type": "stopped", "panelId": worker.panel_id, "sessionId": worker.session_id}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
