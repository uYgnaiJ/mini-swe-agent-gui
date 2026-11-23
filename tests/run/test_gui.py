import copy
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import UserInterruption
from minisweagent.models.test_models import DeterministicToolcallModel, make_toolcall_output
from minisweagent.run.gui import (
    ConversationState,
    GuiConfig,
    GuiModelSettings,
    GuiSessionManager,
    _qualified_model_name,
    build_gui_config,
    conversation_state,
    create_server,
)
from minisweagent.run.utilities.conversation_store import ConversationStore


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
        self.callback(
            {
                "type": "assistant_message",
                "content": "Hello world",
                "actions": [],
                "cost": 0.0,
                "step": 1,
                "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
            }
        )
        self.callback(
            {
                "type": "done",
                "exit_status": "Submitted",
                "submission": "ok",
                "cost": 0.0,
                "n_calls": 1,
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
                "tokens": 150,
                "token_usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
            }
        )
        return {}


def test_conversation_states_are_normalized():
    assert [state.value for state in ConversationState] == [
        "running",
        "submitted",
        "stopped",
        "failed",
        "interrupted",
    ]
    assert conversation_state("Running", done=False) is ConversationState.RUNNING
    assert conversation_state("Submitted", done=True) is ConversationState.SUBMITTED
    assert conversation_state("Stopped", done=True) is ConversationState.STOPPED
    assert conversation_state("LimitsExceeded", done=True) is ConversationState.FAILED
    assert conversation_state("Interrupted", done=True) is ConversationState.INTERRUPTED


def test_gui_exposes_the_conversation_database_path(tmp_path):
    db_path = tmp_path / "conversations.db"
    manager = GuiSessionManager(GuiConfig(agent_config={}), agent_factory=_FakeAgent, db_path=db_path)
    assert manager.database_path == str(db_path.resolve())
    assert manager.database_info()["path"] == str(db_path.resolve())
    assert manager.database_info()["url"] == f"sqlite:///{db_path.resolve()}"
    assert manager.database_info()["exists"] is True

    server = create_server("127.0.0.1", 0, manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    try:
        with urllib.request.urlopen(base_url + "/api/config") as response:
            config = json.load(response)
        assert config["database_path"] == str(db_path.resolve())

        with urllib.request.urlopen(base_url + "/api/database") as response:
            database = json.load(response)
        assert database["database_path"] == str(db_path.resolve())
        assert database["database"]["path"] == str(db_path.resolve())

        with urllib.request.urlopen(base_url + "/") as response:
            html = response.read().decode()
        assert 'id="database-path"' in html
        assert "SQLite: " in html
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        manager.close()


def test_gui_serves_html_and_streams_session_events(tmp_path):
    manager = GuiSessionManager(
        GuiConfig(agent_config={}), agent_factory=_FakeAgent, db_path=tmp_path / "conversations.db"
    )
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
        assert "Conversations" in html
        assert 'id="conversation-list"' in html
        assert "/api/conversations" in html
        assert 'id="conversation-state-summary"' not in html
        assert "renderConversationStateSummary" not in html
        header = html.split("<header>", 1)[1].split("</header>", 1)[0]
        conversation_section = html.split("<section>", 1)[1].split("</section>", 1)[0]
        assert 'id="status"' not in header
        assert 'id="status"' in conversation_section
        assert 'id="conversation-label"' in conversation_section
        assert 'id="conversation-workspace"' in conversation_section
        composer = conversation_section.split('<div class="panel composer">', 1)[1]
        conversation_bar = conversation_section.split('class="panel conversation-bar"', 1)[1].split(
            '<div class="feed-wrap">', 1
        )[0]
        assert 'id="conversation-id"' not in conversation_bar
        assert 'id="export"' not in conversation_bar
        assert "Export" not in conversation_section
        assert 'id="conversation-workspace"' not in conversation_bar
        assert 'id="conversation-workspace"' in composer
        assert 'class="composer-actions"' in composer
        assert '<label for="task">' not in composer
        assert "Commands run in the configured local environment." not in composer
        assert "min-height: 84px;" not in html
        assert "min-height: 42px;" in html
        assert 'id="task" rows="1"' in composer
        assert 'id="attach-image"' in composer
        assert composer.index('id="attach-image"') < composer.index('id="task"')
        assert 'id="send"' in composer
        assert 'id="stop"' in composer
        assert ">Send<" in composer
        assert ">Stop<" in composer
        assert ".conversation-bar .conversation-workspace" not in html
        assert "function setConversationId" not in html
        assert "resetRunView(conversation || { session_id: sessionId })" in html
        assert "exportButton" not in html
        assert "conversationWorkspacePath" in html
        assert 'id="stop"' in html
        assert "/api/stop/" in html
        assert "const sessionId = activeSessionId || selectedSessionId;" in html
        assert 'row.className = "tool-call"' in html
        assert "function makeToolCall" in html
        assert "function updateToolCall" in html
        assert "tool-call-command" in html
        assert "tool-call-status" in html
        assert "Open command details in Activity" in html
        assert "function stripCommandFromAssistantContent" in html
        assert "function mergeMatchingCommandIntoPendingAssistant" in html
        assert "pendingAssistantMessage" in html
        assert "function isRepeatedAssistantSubmission" in html
        assert "!isRepeatedAssistantSubmission(event.submission)" in html
        assert "keepsAssistantMergeWindow" in html
        assert "addRunMarker" in html
        assert "Run started" in html
        assert "Run finished" in html
        assert 'id="stat-tokens"' in html
        assert 'id="stat-token-detail"' in html
        assert "updateTokenStatsFromAssistant" in html
        assert "total_tokens_used" in html
        assert 'id="activity-summary"' in html
        assert 'id="activity-overview"' in html
        assert 'id="activity-scroll"' in html
        assert 'id="activity-expand"' in html
        assert "Step-by-step trajectory of model calls, tool runs, and outcomes." in html
        assert "function addActivityStep" in html
        assert "function addToolActivity" in html
        assert "function updateToolActivity" in html
        assert "function finalizeActivity" in html
        assert "function eventAffectsVisibleFeed" in html
        assert 'event.type !== "reasoning_delta" || reasoningIsExpanded()' in html
        assert "shouldFollowOutput && eventAffectsVisibleFeed(event)" in html
        assert "MAX_RECENT_ACTIVITIES" in html
        assert "function layoutActivityEntries" in html
        assert "function setActivityExpanded" in html
        assert "body.activity-expanded main > section { display: none; }" in html
        assert "--activity-offset" in html
        assert "--activity-width" in html
        activity_scroll_css = html.split(".activity-scroll {", 1)[1].split("}", 1)[0]
        assert "overflow: hidden" in activity_scroll_css
        assert "overflow-y: auto" not in activity_scroll_css
        # Compact activity entries are top-aligned so clipped history cannot
        # leave an unexplained blank gap above the newest entries.
        assert "justify-content: flex-start" in activity_scroll_css
        assert ".activity-scroll.has-activity::after" in html
        assert ".activity-scroll.has-clipped-history::before" in html
        assert "function setActivitySpineState" in html
        assert "clippedHistory" in html
        start_run = html.split("async function startRun()", 1)[1].split("async function stopRun()", 1)[0]
        assert 'taskEl.value = "";' in start_run
        assert "payload.session_id = selectedSessionId;" in start_run
        assert "connectEvents(sessionId, Number(data.event_offset || 0));" in start_run
        stop_run = html.split("async function stopRun()", 1)[1].split("function startNewConversation()", 1)[0]
        assert "setComposerRunning(false);" in stop_run
        assert "updateConversationStatus(sessionId, {" in stop_run

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
        assert events[-1]["total_tokens"] == 150
        summary = manager.get_session(session_id).summary()
        assert summary["total_tokens"] == 150
        assert summary["prompt_tokens"] == 120
        assert summary["completion_tokens"] == 30
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_gui_lists_all_conversations_and_replays_their_events(tmp_path):
    manager = GuiSessionManager(
        GuiConfig(agent_config={}), agent_factory=_FakeAgent, db_path=tmp_path / "conversations.db"
    )
    first = manager.start_session("First conversation")
    assert first.wait(timeout=5)
    second = manager.start_session("Second conversation")
    assert second.wait(timeout=5)

    server = create_server("127.0.0.1", 0, manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"

    try:
        with urllib.request.urlopen(base_url + "/api/conversations") as response:
            payload = json.load(response)
        conversations = payload["conversations"]
        assert {conversation["session_id"] for conversation in conversations} == {first.session_id, second.session_id}
        assert conversations[0]["task"] == "Second conversation"
        assert all(conversation["done"] for conversation in conversations)
        assert all(conversation["workspace"] for conversation in conversations)
        assert payload["state_count"] == 5
        assert payload["states"] == ["running", "submitted", "stopped", "failed", "interrupted"]
        assert payload["state_counts"] == {
            "running": 0,
            "submitted": 2,
            "stopped": 0,
            "failed": 0,
            "interrupted": 0,
        }
        assert all(conversation["state"] == "submitted" for conversation in conversations)

        with urllib.request.urlopen(base_url + "/api/conversations/statuses") as response:
            statuses = json.load(response)
        assert statuses["count"] == 5
        assert statuses["status_counts"] == payload["state_counts"]

        with urllib.request.urlopen(base_url + f"/api/conversations/{first.session_id}") as response:
            detail = json.load(response)
        assert detail["conversation"]["task"] == "First conversation"
        assert detail["events"][0] == {"type": "run_start", "task": "First conversation"}
        assert detail["events"][-1]["type"] == "done"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_gui_activity_expanded_view_is_readable_and_scrollable(tmp_path):
    manager = GuiSessionManager(
        GuiConfig(agent_config={}), agent_factory=_FakeAgent, db_path=tmp_path / "conversations.db"
    )
    server = create_server("127.0.0.1", 0, manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address

    try:
        with urllib.request.urlopen(f"http://{host}:{port}/") as response:
            html = response.read().decode()
        expanded_activity_css = html.split("body.activity-expanded .activity-scroll {", 1)[1].split("}", 1)[0]
        assert "overflow-y: auto" in expanded_activity_css
        assert "body.activity-expanded .activity-copy," not in html
        assert "body.activity-expanded .activity-meta-row { display: none; }" not in html
        assert "activityFollowOutput" in html
        assert 'activityScroller.addEventListener("scroll"' in html
        assert 'id="activity-open"' in html
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        manager.close()


def test_gui_rejects_empty_task(tmp_path):
    manager = GuiSessionManager(
        GuiConfig(agent_config={}), agent_factory=_FakeAgent, db_path=tmp_path / "conversations.db"
    )
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


class _StoppableAgent(_FakeAgent):
    def __init__(self):
        self.stop_requested = threading.Event()

    def request_stop(self):
        self.stop_requested.set()

    def run(self, task):
        self.callback({"type": "run_start", "task": task})
        self.stop_requested.wait(timeout=5)
        result = {"exit_status": "Stopped", "submission": "", "cost": 0.0, "n_calls": 0}
        self.callback({"type": "done", **result})
        return result


def test_gui_stop_button_stops_running_session(tmp_path):
    manager = GuiSessionManager(
        GuiConfig(agent_config={}), agent_factory=_StoppableAgent, db_path=tmp_path / "conversations.db"
    )
    server = create_server("127.0.0.1", 0, manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"

    try:
        request = urllib.request.Request(
            base_url + "/api/run",
            data=json.dumps({"task": "Keep running"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            session_id = json.load(response)["session_id"]

        with urllib.request.urlopen(base_url + "/api/conversations") as response:
            conversations = json.load(response)["conversations"]
        assert conversations[0]["session_id"] == session_id
        assert conversations[0]["done"] is False
        for _ in range(100):
            with urllib.request.urlopen(base_url + f"/api/conversations/{session_id}") as response:
                detail = json.load(response)
            if detail["events"]:
                break
            time.sleep(0.01)
        assert detail["events"] == [{"type": "run_start", "task": "Keep running"}]

        stop_request = urllib.request.Request(base_url + f"/api/stop/{session_id}", method="POST")
        with urllib.request.urlopen(stop_request) as response:
            assert response.status == 202
            assert json.load(response)["status"] == "stopping"

        events = []
        with urllib.request.urlopen(base_url + f"/api/events/{session_id}", timeout=10) as response:
            for raw_line in response:
                line = raw_line.decode().strip()
                if line.startswith("data: "):
                    event = json.loads(line.removeprefix("data: "))
                    events.append(event)
                    if event["type"] == "done":
                        break

        assert events[-1]["type"] == "done"
        assert events[-1]["exit_status"] == "Stopped"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class _InterruptingAgent(_FakeAgent):
    def __init__(self):
        self.stop_requested = threading.Event()

    def request_stop(self):
        self.stop_requested.set()

    def run(self, task):
        self.callback({"type": "run_start", "task": task})
        self.stop_requested.wait(timeout=5)
        raise UserInterruption({"role": "exit", "content": "Stopped by user"})


def test_gui_stop_handles_agents_that_propagate_user_interruption(tmp_path):
    manager = GuiSessionManager(
        GuiConfig(agent_config={}), agent_factory=_InterruptingAgent, db_path=tmp_path / "conversations.db"
    )
    session = manager.start_session("Keep running")
    for _ in range(100):
        if session.events:
            break
        time.sleep(0.01)

    assert session.request_stop() is True
    assert session.wait(timeout=5)
    assert session.exit_status == "Stopped"
    assert session.summary()["state"] == "stopped"
    assert session.events[-1]["type"] == "done"
    assert session.events[-1]["exit_status"] == "Stopped"


def test_gui_stop_unknown_session_returns_404(tmp_path):
    manager = GuiSessionManager(
        GuiConfig(agent_config={}), agent_factory=_FakeAgent, db_path=tmp_path / "conversations.db"
    )
    server = create_server("127.0.0.1", 0, manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        request = urllib.request.Request(f"http://{host}:{port}/api/stop/unknown", method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(request)
        assert exc_info.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_gui_conversations_persist_and_export_across_restarts(tmp_path):
    db_path = tmp_path / "conversations.db"
    manager = GuiSessionManager(GuiConfig(agent_config={}), agent_factory=_FakeAgent, db_path=db_path)
    session = manager.start_session("Persist me")
    assert session.wait(timeout=5)
    manager.close()

    restored = GuiSessionManager(GuiConfig(agent_config={}), agent_factory=_FakeAgent, db_path=db_path)
    loaded = restored.get_session(session.session_id)
    assert loaded is not None
    assert loaded.summary()["task"] == "Persist me"
    assert loaded.summary()["done"] is True
    assert loaded.summary()["total_tokens"] == 150
    assert loaded.events[0] == {"type": "run_start", "task": "Persist me"}
    assert loaded.events[-1]["type"] == "done"

    exported = restored.export_session(session.session_id)
    assert exported is not None
    assert exported["session_id"] == session.session_id
    assert exported["task"] == "Persist me"
    assert exported["events"][-1]["type"] == "done"
    assert exported["conversation"]["status"] == "Submitted"

    server = create_server("127.0.0.1", 0, restored)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/conversations/{session.session_id}/export") as response:
            payload = json.load(response)
            disposition = response.headers["Content-Disposition"]
        assert payload["task"] == "Persist me"
        assert f"mini-swe-agent-{session.session_id}.traj.json" in disposition
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_gui_restored_interrupted_conversation_has_terminal_event(tmp_path):
    db_path = tmp_path / "conversations.db"
    store = ConversationStore(db_path)
    summary = {
        "session_id": "interrupted-session",
        "task": "Do work",
        "workspace": str(tmp_path),
        "status": "running",
        "created_at": 1000.0,
        "done": False,
        "event_count": 2,
    }
    store.save_session(summary)
    store.record_event(summary, 0, {"type": "run_start", "task": "Do work"})
    store.record_event(summary, 1, {"type": "step_start", "step": 1})
    store.close()

    manager = GuiSessionManager(GuiConfig(agent_config={}), db_path=db_path)
    try:
        loaded = manager.get_session("interrupted-session")
        assert loaded is not None
        loaded_summary = loaded.summary()
        assert loaded_summary["status"] == "Interrupted"
        assert loaded_summary["state"] == "interrupted"
        assert loaded_summary["done"] is True
        # Event replay needs a terminal event so the header cannot remain on the
        # last historical "running" event after the stream closes.
        assert loaded.events[-1]["type"] == "done"
        assert loaded.events[-1]["exit_status"] == "Interrupted"
        assert loaded_summary["event_count"] == 3
    finally:
        manager.close()


def test_gui_greeting_runs_the_agent(tmp_path):
    calls: list[bool] = []

    def agent_factory():
        calls.append(True)
        return _FakeAgent()

    manager = GuiSessionManager(
        GuiConfig(agent_config={}), agent_factory=agent_factory, db_path=tmp_path / "conversations.db"
    )
    try:
        session = manager.start_session("hi")

        assert session.wait(timeout=5)
        assert calls == [True]
        assert session.events[0]["type"] == "run_start"
        assert session.events[-1]["exit_status"] == "Submitted"
        assert session.summary()["state"] == "submitted"
    finally:
        manager.close()


def test_gui_investigation_ends_with_visible_final_answer(tmp_path):
    agent_config = yaml.safe_load(Path("src/minisweagent/config/mini.yaml").read_text())

    def tool_call(command, call_id):
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"command": command})},
        }

    def agent_factory():
        outputs = [
            make_toolcall_output(
                "I will inspect the repository.",
                [tool_call("echo investigating", "call_1")],
                [{"command": "echo investigating", "tool_call_id": "call_1"}],
            ),
            make_toolcall_output(
                None,
                [tool_call("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "call_2")],
                [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "tool_call_id": "call_2"}],
            ),
            make_toolcall_output("The answer is in the global configuration directory.", [], []),
        ]
        return DefaultAgent(
            model=DeterministicToolcallModel(outputs=outputs),
            env=LocalEnvironment(cwd=str(tmp_path)),
            **agent_config["agent"],
        )

    manager = GuiSessionManager(
        GuiConfig(agent_config=agent_config, workspace=str(tmp_path)),
        agent_factory=agent_factory,
        db_path=tmp_path / "conversations.db",
    )
    session = manager.start_session("Where is the database?")
    assert session.wait(timeout=5)

    assistant_messages = [event for event in session.events if event["type"] == "assistant_message"]
    assert assistant_messages[-1]["content"] == "The answer is in the global configuration directory."
    assert session.events[-1]["submission"] == "The answer is in the global configuration directory."
    assert session.summary()["state"] == "submitted"
    manager.close()


class _ImageFakeAgent:
    def set_event_callback(self, callback):
        self.callback = callback

    def run(self, task, images=None):
        self.callback({"type": "run_start", "task": task, "images": list(images or [])})
        self.callback(
            {
                "type": "done",
                "exit_status": "Submitted",
                "submission": "ok",
                "cost": 0.0,
                "n_calls": 1,
                "token_usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )
        return {}


def test_gui_session_passes_images_to_agent(tmp_path):
    data_url = "data:image/png;base64,iVBORw0KGgo="
    manager = GuiSessionManager(
        GuiConfig(agent_config={}),
        agent_factory=_ImageFakeAgent,
        db_path=tmp_path / "conversations.db",
    )
    session = manager.start_session("inspect this screenshot", images=[data_url])

    assert session.wait(timeout=5)
    assert session.events[0]["images"] == [data_url]
    assert session.summary()["image_count"] == 1
    manager.close()


def test_gui_run_api_accepts_and_validates_images(tmp_path):
    manager = GuiSessionManager(
        GuiConfig(agent_config={}),
        agent_factory=_ImageFakeAgent,
        db_path=tmp_path / "conversations.db",
    )
    server = create_server("127.0.0.1", 0, manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    data_url = "data:image/png;base64,iVBORw0KGgo="
    try:
        request = urllib.request.Request(
            base_url + "/api/run",
            data=json.dumps({"task": "look", "images": [data_url]}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            assert response.status == 202
            payload = json.load(response)
        assert payload["conversation"]["image_count"] == 1

        bad_request = urllib.request.Request(
            base_url + "/api/run",
            data=json.dumps({"task": "look", "images": ["/etc/passwd"]}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(bad_request)
        assert exc_info.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        manager.close()


def test_gui_restores_image_attachments_from_events(tmp_path):
    data_url = "data:image/png;base64,iVBORw0KGgo="
    db_path = tmp_path / "conversations.db"
    manager = GuiSessionManager(GuiConfig(agent_config={}), agent_factory=_ImageFakeAgent, db_path=db_path)
    session = manager.start_session("keep this image", images=[data_url])
    assert session.wait(timeout=5)
    session_id = session.session_id
    manager.close()

    restored_manager = GuiSessionManager(GuiConfig(agent_config={}), db_path=db_path)
    restored = restored_manager.get_session(session_id)
    assert restored is not None
    assert restored.images == [data_url]
    assert restored.summary()["image_count"] == 1
    restored_manager.close()


def test_gui_run_api_continues_conversation_after_stop(tmp_path):
    manager = GuiSessionManager(
        GuiConfig(agent_config={}), agent_factory=_StoppableAgent, db_path=tmp_path / "conversations.db"
    )
    server = create_server("127.0.0.1", 0, manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"

    def get_detail(session_id):
        with urllib.request.urlopen(base_url + f"/api/conversations/{session_id}") as response:
            return json.load(response)

    def wait_for(predicate, *, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.01)
        raise AssertionError("timed out waiting for GUI state")

    try:
        request = urllib.request.Request(
            base_url + "/api/run",
            data=json.dumps({"task": "first turn"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            session_id = json.load(response)["session_id"]

        def first_event():
            d = get_detail(session_id)
            return d if d["events"] else None

        wait_for(first_event)
        stop_request = urllib.request.Request(base_url + f"/api/stop/{session_id}", method="POST")
        with urllib.request.urlopen(stop_request) as response:
            assert response.status == 202

        def stopped_detail():
            d = get_detail(session_id)
            return d if d["conversation"]["done"] else None

        detail = wait_for(stopped_detail)
        first_event_count = len(detail["events"])
        assert detail["conversation"]["state"] == "stopped"

        follow_up_request = urllib.request.Request(
            base_url + "/api/run",
            data=json.dumps({"task": "second turn", "session_id": session_id}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(follow_up_request) as response:
            payload = json.load(response)
        assert payload["session_id"] == session_id
        assert payload["continued"] is True
        assert payload["event_offset"] == first_event_count

        def follow_up_started():
            d = get_detail(session_id)
            return d if len(d["events"]) > first_event_count else None

        detail = wait_for(follow_up_started)
        assert [event["type"] for event in detail["events"]].count("run_start") == 2
        assert detail["events"][-1] == {"type": "run_start", "task": "second turn"}

        stop_request = urllib.request.Request(base_url + f"/api/stop/{session_id}", method="POST")
        with urllib.request.urlopen(stop_request) as response:
            assert response.status == 202

        def follow_up_finished():
            d = get_detail(session_id)
            return d if d["conversation"]["done"] and d["events"][-1]["type"] == "done" else None

        detail = wait_for(follow_up_finished)
        assert [event["type"] for event in detail["events"]].count("run_start") == 2
        assert [event["type"] for event in detail["events"]].count("done") == 2
        assert detail["conversation"]["task"] == "first turn"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class _RecordingToolcallModel(DeterministicToolcallModel):
    def __init__(self, *args, queries: list, **kwargs):
        super().__init__(*args, **kwargs)
        self.queries = queries

    def query(self, messages, **kwargs):
        self.queries.append(copy.deepcopy(messages))
        return super().query(messages, **kwargs)


def test_gui_follow_ups_keep_all_previous_instructions_as_context(tmp_path):
    config = yaml.safe_load(Path("src/minisweagent/config/mini.yaml").read_text())
    queries: list[list[dict]] = []
    answer_index = 0

    def agent_factory():
        nonlocal answer_index
        answer_index += 1
        return DefaultAgent(
            model=_RecordingToolcallModel(
                outputs=[make_toolcall_output(f"answer {answer_index}", [], [])], queries=queries
            ),
            env=LocalEnvironment(cwd=str(tmp_path)),
            **config["agent"],
        )

    manager = GuiSessionManager(
        GuiConfig(agent_config=config, workspace=str(tmp_path)),
        agent_factory=agent_factory,
        db_path=tmp_path / "conversations.db",
    )
    session = manager.start_session("first instruction")
    assert session.wait(timeout=5)

    session.continue_run("second instruction")
    assert session.wait(timeout=5)

    session.continue_run("third instruction")
    assert session.wait(timeout=5)

    assert len(queries) == 3
    third_turn_messages = queries[2]
    third_turn_text = " ".join(str(message.get("content") or "") for message in third_turn_messages)
    assert "first instruction" in third_turn_text
    assert "second instruction" in third_turn_text
    assert "third instruction" in third_turn_text
    assert [message.get("role") for message in session.messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "exit",
    ]

    # The history also survives a GUI process restart.
    session_id = session.session_id
    manager.close()
    restored_manager = GuiSessionManager(
        GuiConfig(agent_config=config, workspace=str(tmp_path)),
        agent_factory=agent_factory,
        db_path=tmp_path / "conversations.db",
    )
    restored = restored_manager.get_session(session_id)
    assert restored is not None
    restored.continue_run("fourth instruction")
    assert restored.wait(timeout=5)
    fourth_turn_text = " ".join(str(message.get("content") or "") for message in queries[3])
    for instruction in ("first instruction", "second instruction", "third instruction", "fourth instruction"):
        assert instruction in fourth_turn_text
    restored_manager.close()


@pytest.mark.parametrize(
    ("provider_id", "model_name", "expected"),
    [
        ("openai", "gpt-4o-mini", "openai/gpt-4o-mini"),
        ("openai", "openai/gpt-4o-mini", "openai/gpt-4o-mini"),
        ("openrouter", "openai/gpt-4o", "openrouter/openai/gpt-4o"),
        ("custom-gateway", "meta-llama/Llama-3", "openai/meta-llama/Llama-3"),
        ("vllm", "Qwen/Qwen3", "openai/Qwen/Qwen3"),
    ],
)
def test_provider_model_names_are_qualified_for_litellm(provider_id, model_name, expected):
    assert _qualified_model_name(provider_id, model_name) == expected


@pytest.mark.parametrize(
    ("provider_id", "model_name", "base_url", "expected"),
    [
        (
            "arkengine",
            "openai/kimi-k3-1-pro-261024",
            "https://ark.cn-beijing.volces.com/api/v3",
            "volcengine/kimi-k3-1-pro-261024",
        ),
        (
            "arkengine",
            "doubao-seed-2-1-pro-260915",
            "https://ark.cn-beijing.volces.com/api/v3",
            "volcengine/doubao-seed-2-1-pro-260915",
        ),
        (
            "openai",
            "openai/kimi-k3-1-pro-261024",
            "https://ark.cn-beijing.volces.com/api/v3",
            "openai/kimi-k3-1-pro-261024",
        ),
        ("custom-gateway", "meta-llama/Llama-3", "https://api.example.com/v1", "openai/meta-llama/Llama-3"),
    ],
)
def test_base_url_provider_prefix_is_used_for_litellm_model_names(provider_id, model_name, base_url, expected):
    assert _qualified_model_name(provider_id, model_name, base_url) == expected


def test_ark_provider_models_use_volcengine_prefix_and_cost_fallback(tmp_path):
    manager = GuiSessionManager(
        GuiConfig(agent_config={"model": {}}),
        agent_factory=_FakeAgent,
        db_path=tmp_path / "conversations.db",
    )
    base_url = "https://ark.cn-beijing.volces.com/api/v3"
    manager.model_settings.add_provider({"id": "arkengine", "name": "arkengine", "base_url": base_url})
    manager.model_settings.add_model({"model_name": "openai/kimi-k3-1-pro-261024", "provider": "arkengine"})

    config = manager._config_for_model("openai/kimi-k3-1-pro-261024")

    assert config.model_name == "volcengine/kimi-k3-1-pro-261024"
    assert config.agent_config["model"]["cost_tracking"] == "ignore_errors"
    assert config.agent_config["model"]["model_kwargs"]["api_base"] == base_url
    manager.close()


def test_loading_provider_models_qualifies_names_and_upgrades_existing_models(tmp_path, monkeypatch):
    settings = GuiModelSettings(tmp_path / "gui_settings.json")
    settings.add_provider({"id": "openai", "name": "OpenAI", "base_url": "https://api.openai.com/v1"})
    settings.add_model({"model_name": "gpt-4o-mini", "provider": "openai"})

    monkeypatch.setattr(
        "minisweagent.run.gui._query_provider_models",
        lambda provider, *, timeout=15.0: [
            {"model_name": "gpt-4o-mini", "name": "GPT-4o mini"},
            {"model_name": "gpt-4.1", "name": "GPT-4.1"},
        ],
    )
    snapshot = settings.load_provider_models("openai")

    assert [model["model_name"] for model in snapshot["enabled_models"]] == [
        "openai/gpt-4o-mini",
        "openai/gpt-4.1",
    ]
    assert snapshot["active_model"] == "openai/gpt-4o-mini"
    assert snapshot["fetched_models"] == ["gpt-4o-mini", "gpt-4.1"]
    assert [model["model_name"] for model in snapshot["loaded_models"]] == ["openai/gpt-4.1"]


def test_gui_config_enables_auto_budget_by_default(tmp_path):
    config = build_gui_config(config_spec=[str(Path("src/minisweagent/config/mini.yaml"))])
    assert config.agent_config["agent"]["auto_budget"] is True
    assert config.agent_config["agent"]["mode"] == "yolo"


def test_gui_replays_every_turn_before_closing_the_event_stream():
    """A finished conversation has one `done` per turn; replay must not stop at the first."""
    from minisweagent.run.gui import _HTML

    assert "replayingHistory = true" in _HTML
    assert "if (!replayingHistory) closeEventStream();" in _HTML


def test_mini_config_does_not_frame_requests_as_bug_reports():
    agent_config = yaml.safe_load(Path("src/minisweagent/config/mini.yaml").read_text())["agent"]
    instance_template = agent_config["instance_template"]
    assert "Please solve this issue" not in instance_template
    assert "Submit your changes" not in instance_template
    assert "How to proceed" not in instance_template
    assert instance_template.strip() == "User request:\n<request>\n{{task}}\n</request>"
    assert "How to proceed" in agent_config["system_template"]


def test_gui_config_respects_explicit_auto_budget_override(tmp_path):
    config = build_gui_config(config_spec=["agent.auto_budget=false"])
    assert config.agent_config["agent"]["auto_budget"] is False


def test_reloading_ark_models_upgrades_existing_openai_prefixed_names(tmp_path, monkeypatch):
    settings = GuiModelSettings(tmp_path / "gui_settings.json")
    settings.add_provider(
        {"id": "arkengine", "name": "arkengine", "base_url": "https://ark.cn-beijing.volces.com/api/v3"}
    )
    settings.add_model({"model_name": "openai/kimi-k3-1-pro-261024", "provider": "arkengine"})
    monkeypatch.setattr(
        "minisweagent.run.gui._query_provider_models",
        lambda provider, *, timeout=15.0: [{"model_name": "kimi-k3-1-pro-261024", "name": "Kimi K3"}],
    )

    snapshot = settings.load_provider_models("arkengine")

    assert [model["model_name"] for model in snapshot["enabled_models"]] == ["volcengine/kimi-k3-1-pro-261024"]
    assert snapshot["loaded_models"] == []


def test_model_config_adds_provider_prefix_when_calling_existing_raw_model_name(tmp_path):
    manager = GuiSessionManager(
        GuiConfig(agent_config={"model": {}}),
        agent_factory=_FakeAgent,
        db_path=tmp_path / "conversations.db",
    )
    manager.model_settings.add_provider({"id": "openai", "name": "OpenAI"})
    manager.model_settings.add_model({"model_name": "gpt-4o-mini", "provider": "openai"})

    config = manager._config_for_model("gpt-4o-mini")

    assert config.model_name == "openai/gpt-4o-mini"
    assert config.agent_config["model"]["model_name"] == "openai/gpt-4o-mini"
    manager.close()


def test_provider_model_list_is_grouped_and_delete_cascades_to_models(tmp_path):
    settings = GuiModelSettings(tmp_path / "gui_settings.json")
    settings.add_provider({"id": "openai", "name": "OpenAI"})
    settings.add_provider({"id": "anthropic", "name": "Anthropic"})
    settings.add_model({"model_name": "openai/gpt-4o", "provider": "openai"})
    settings.add_model({"model_name": "openai/gpt-4.1", "provider": "openai"})
    settings.add_model({"model_name": "anthropic/claude-sonnet", "provider": "anthropic"})

    openai = next(provider for provider in settings.list_providers() if provider["id"] == "openai")
    assert openai["model_count"] == 2
    assert openai["models"] == ["openai/gpt-4o", "openai/gpt-4.1"]

    settings.delete_provider("openai")
    assert [provider["id"] for provider in settings.list_providers()] == ["anthropic"]
    assert [model["model_name"] for model in settings.list_models()] == ["anthropic/claude-sonnet"]


def test_gui_html_exposes_expandable_provider_model_groups():
    from minisweagent.run.gui import _HTML

    assert "Click a provider to show its models" in _HTML
    assert 'className = "provider-models"' in _HTML
    assert "function toggleProviderModels" in _HTML
    assert '"Add model"' in _HTML
