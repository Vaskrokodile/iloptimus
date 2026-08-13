"""Supervised headless RSI panel processes and durable event logs."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from .storage import app_home, atomic_write_json


@dataclass
class RsiPanel:
    id: str
    title: str
    model_id: str
    workspace: str
    status: str = "starting"
    session_id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_error: str = ""
    pid: int | None = None

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Runtime:
    record: RsiPanel
    process: asyncio.subprocess.Process
    reader_task: asyncio.Task
    stderr_task: asyncio.Task
    condition: asyncio.Condition


class RsiPanelManager:
    def __init__(self, root: Path | None = None, worker_command: list[str] | None = None):
        self.root = root or app_home() / "rsi-panels"
        self.root.mkdir(parents=True, exist_ok=True)
        self.worker_command = worker_command
        self._records: dict[str, RsiPanel] = {}
        self._runtimes: dict[str, _Runtime] = {}
        self._load_records()

    def _panel_dir(self, panel_id: str) -> Path:
        return self.root / panel_id

    def _metadata_path(self, panel_id: str) -> Path:
        return self._panel_dir(panel_id) / "panel.json"

    def _events_path(self, panel_id: str) -> Path:
        return self._panel_dir(panel_id) / "events.jsonl"

    def _load_records(self) -> None:
        for path in self.root.glob("*/panel.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                record = RsiPanel(**payload)
            except (OSError, TypeError, json.JSONDecodeError):
                continue
            if record.status in {"starting", "ready", "running"}:
                record.status = "stopped"
                record.pid = None
                record.last_error = "The application restarted; reopen the panel to start a new worker."
                self._persist(record)
            self._records[record.id] = record

    def _persist(self, record: RsiPanel) -> None:
        record.updated_at = time.time()
        atomic_write_json(self._metadata_path(record.id), record.public())

    def _append_event(self, panel_id: str, event: dict[str, Any]) -> None:
        path = self._events_path(panel_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _resolve_worker_command(self) -> list[str]:
        if self.worker_command:
            return self.worker_command
        # This worker ships in the Python wheel. The TypeScript RSI terminal is
        # still developed and tested in rsiagent/, but installed desktop users
        # must not need Bun or a source checkout for web panels to work.
        return [sys.executable, "-m", "iloptimus.rsi_worker"]

    def list(self) -> list[dict[str, Any]]:
        return [record.public() for record in sorted(self._records.values(), key=lambda item: item.created_at)]

    def get(self, panel_id: str) -> RsiPanel | None:
        return self._records.get(panel_id)

    def events(self, panel_id: str, after: int = 0) -> list[dict[str, Any]]:
        path = self._events_path(panel_id)
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(event.get("sequence", 0)) > after:
                events.append(event)
        return events

    async def launch(
        self,
        *,
        model_id: str,
        workspace: Path,
        base_url: str,
        title: str | None = None,
        initial_prompt: str | None = None,
    ) -> dict[str, Any]:
        workspace = workspace.expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        panel_id = uuid.uuid4().hex[:12]
        record = RsiPanel(
            id=panel_id,
            title=title or f"RSI Agent {len(self._records) + 1}",
            model_id=model_id,
            workspace=str(workspace),
        )
        self._records[panel_id] = record
        self._persist(record)

        env = {
            **os.environ,
            "RSI_PANEL_ID": panel_id,
            "RSI_WORKSPACE": str(workspace),
            "RSI_MODEL_ID": model_id,
            "RSI_PROVIDER_BASE_URL": base_url.rstrip("/") + "/v1",
            "RSI_MAX_TOKENS": "384",
            "RSI_SESSION_PATH": str(self._panel_dir(panel_id) / "session.jsonl"),
        }
        command = self._resolve_worker_command()
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=workspace,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        record.pid = process.pid
        self._persist(record)
        condition = asyncio.Condition()
        runtime = _Runtime(
            record=record,
            process=process,
            reader_task=None,  # type: ignore[arg-type]
            stderr_task=None,  # type: ignore[arg-type]
            condition=condition,
        )
        runtime.reader_task = asyncio.create_task(self._read_stdout(runtime))
        runtime.stderr_task = asyncio.create_task(self._read_stderr(runtime))
        self._runtimes[panel_id] = runtime
        if initial_prompt:
            await self.prompt(panel_id, initial_prompt)
        return record.public()

    async def _read_stdout(self, runtime: _Runtime) -> None:
        assert runtime.process.stdout
        sequence = len(self.events(runtime.record.id))
        while line := await runtime.process.stdout.readline():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                payload = {"type": "worker_output", "data": {"text": line.decode(errors="replace").rstrip()}}
            sequence += 1
            payload["sequence"] = sequence
            payload.setdefault("timestamp", time.time())
            self._append_event(runtime.record.id, payload)
            event_type = payload.get("type")
            if event_type == "ready":
                runtime.record.status = "ready"
                runtime.record.session_id = str(payload.get("sessionId", ""))
            elif event_type == "started":
                runtime.record.status = "running"
            elif event_type == "completed":
                runtime.record.status = "ready"
            elif event_type == "failed":
                runtime.record.status = "failed"
                runtime.record.last_error = str(payload.get("error") or payload.get("data", {}).get("error", ""))
            self._persist(runtime.record)
            async with runtime.condition:
                runtime.condition.notify_all()

        return_code = await runtime.process.wait()
        if runtime.record.status not in {"failed", "stopped"}:
            runtime.record.status = "stopped" if return_code == 0 else "failed"
        runtime.record.pid = None
        if return_code and not runtime.record.last_error:
            runtime.record.last_error = f"RSI worker exited with code {return_code}"
        self._persist(runtime.record)
        async with runtime.condition:
            runtime.condition.notify_all()

    async def _read_stderr(self, runtime: _Runtime) -> None:
        assert runtime.process.stderr
        chunks: list[str] = []
        while line := await runtime.process.stderr.readline():
            chunks.append(line.decode(errors="replace").rstrip())
            chunks = chunks[-30:]
        if chunks and runtime.process.returncode:
            runtime.record.last_error = "\n".join(chunks)[-4000:]
            self._persist(runtime.record)

    async def prompt(self, panel_id: str, prompt: str) -> dict[str, Any]:
        runtime = self._runtimes.get(panel_id)
        if not runtime or runtime.process.returncode is not None or not runtime.process.stdin:
            raise ValueError("RSI panel is not running")
        if runtime.record.status == "running":
            raise ValueError("RSI panel is already working")
        runtime.process.stdin.write((json.dumps({"type": "prompt", "prompt": prompt}) + "\n").encode())
        await runtime.process.stdin.drain()
        runtime.record.status = "running"
        self._persist(runtime.record)
        return runtime.record.public()

    async def stream_events(self, panel_id: str, after: int = 0) -> AsyncIterator[dict[str, Any]]:
        record = self.get(panel_id)
        if not record:
            raise ValueError("Panel not found")
        cursor = after
        while True:
            events = self.events(panel_id, cursor)
            for event in events:
                cursor = max(cursor, int(event.get("sequence", cursor)))
                yield event
            runtime = self._runtimes.get(panel_id)
            if not runtime or runtime.process.returncode is not None:
                return
            async with runtime.condition:
                try:
                    await asyncio.wait_for(runtime.condition.wait(), timeout=15)
                except TimeoutError:
                    yield {"type": "heartbeat", "sequence": cursor, "timestamp": time.time()}

    async def stop(self, panel_id: str) -> dict[str, Any]:
        record = self.get(panel_id)
        if not record:
            raise ValueError("Panel not found")
        runtime = self._runtimes.get(panel_id)
        if runtime and runtime.process.returncode is None:
            if runtime.process.stdin:
                runtime.process.stdin.write(b'{"type":"shutdown"}\n')
                await runtime.process.stdin.drain()
            try:
                await asyncio.wait_for(runtime.process.wait(), timeout=3)
            except TimeoutError:
                runtime.process.terminate()
                await runtime.process.wait()
        record.status = "stopped"
        record.pid = None
        record.last_error = ""
        self._persist(record)
        return record.public()

    async def shutdown(self) -> None:
        await asyncio.gather(*(self.stop(panel_id) for panel_id in list(self._runtimes)), return_exceptions=True)
