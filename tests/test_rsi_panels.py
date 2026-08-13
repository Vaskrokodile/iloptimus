import asyncio
import sys

from iloptimus.core.rsi_panels import RsiPanelManager

FAKE_WORKER = r"""
import json
import os
import sys

panel = os.environ["RSI_PANEL_ID"]
print(json.dumps({"type": "ready", "panelId": panel, "sessionId": "session-test"}), flush=True)
for line in sys.stdin:
    command = json.loads(line)
    if command["type"] == "shutdown":
        break
    print(json.dumps({"type": "started", "panelId": panel, "data": {"prompt": command["prompt"]}}), flush=True)
    print(json.dumps({"type": "tool_call", "panelId": panel, "data": {"name": "read_file"}}), flush=True)
    print(json.dumps({"type": "completed", "panelId": panel, "data": {"text": "done"}}), flush=True)
"""


def test_panel_manager_launches_prompts_streams_and_stops_a_worker(tmp_path):
    async def scenario():
        manager = RsiPanelManager(
            root=tmp_path / "panels",
            worker_command=[sys.executable, "-u", "-c", FAKE_WORKER],
        )
        panel = await manager.launch(
            model_id="test-model",
            workspace=tmp_path / "workspace",
            base_url="http://127.0.0.1:7860",
        )
        panel_id = panel["id"]

        for _ in range(50):
            if manager.get(panel_id).status == "ready":
                break
            await asyncio.sleep(0.01)
        assert manager.get(panel_id).status == "ready"
        assert manager.get(panel_id).session_id == "session-test"

        await manager.prompt(panel_id, "inspect the workspace")
        for _ in range(50):
            events = manager.events(panel_id)
            if any(event["type"] == "completed" for event in events):
                break
            await asyncio.sleep(0.01)

        events = manager.events(panel_id)
        assert [event["type"] for event in events] == ["ready", "started", "tool_call", "completed"]
        assert manager.get(panel_id).status == "ready"
        assert (tmp_path / "panels" / panel_id / "events.jsonl").exists()

        stopped = await manager.stop(panel_id)
        assert stopped["status"] == "stopped"
        assert stopped["pid"] is None

    asyncio.run(scenario())
