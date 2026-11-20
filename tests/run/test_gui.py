import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from minisweagent.run.gui import GuiConfig, GuiSessionManager, create_server


class _FakeAgent:
    def set_event_callback(self, callback):
        self.callback = callback

    def run(self, task):
        self.callback({"type": "run_start", "task": task})
        self.callback({"type": "step_start", "step": 1})
        self.callback({"type": "assistant_start", "step": 1})
        for part in ["Hel", "lo ", "world"]:
            self.callback({"type": "assistant_delta", "text": part, "step": 1})
            time.sleep(0.005)
        self.callback({"type": "assistant_end", "step": 1})
        self.callback({"type": "assistant_message", "content": "Hello world", "actions": [], "cost": 0.0, "step": 1})
        self.callback({"type": "done", "exit_status": "Submitted", "submission": "ok", "cost": 0.0, "n_calls": 1})
        return {}


def test_gui_serves_html_and_streams_session_events():
    manager = GuiSessionManager(GuiConfig(agent_config={}), agent_factory=_FakeAgent)
    server = create_server("127.0.0.1", 0, manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"

    try:
        with urllib.request.urlopen(base_url + "/") as response:
            html = response.read().decode()
        assert "mini-SWE-agent" in html
        assert "EventSource" in html

        with urllib.request.urlopen(base_url + "/api/health") as response:
            assert json.load(response) == {"status": "ok"}

        request = urllib.request.Request(
            base_url + "/api/run",
            data=json.dumps({"task": "Say hello"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            assert response.status == 202
            session_id = json.load(response)["session_id"]

        events = []
        with urllib.request.urlopen(base_url + f"/api/events/{session_id}", timeout=10) as response:
            for raw_line in response:
                line = raw_line.decode().strip()
                if line.startswith("data: "):
                    event = json.loads(line.removeprefix("data: "))
                    events.append(event)
                    if event["type"] == "done":
                        break

        assert "".join(event["text"] for event in events if event["type"] == "assistant_delta") == "Hello world"
        assert events[-1]["type"] == "done"
        assert events[-1]["submission"] == "ok"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_gui_rejects_empty_task():
    manager = GuiSessionManager(GuiConfig(agent_config={}), agent_factory=_FakeAgent)
    server = create_server("127.0.0.1", 0, manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address

    try:
        request = urllib.request.Request(
            f"http://{host}:{port}/api/run",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(request)
        assert exc_info.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
