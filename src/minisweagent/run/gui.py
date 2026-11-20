#!/usr/bin/env python3
"""A basic web GUI for mini-SWE-agent with live streaming responses.

Run ``mini-gui`` (or ``python -m minisweagent.run.gui``), open the printed URL,
enter a task, and watch the agent think and execute commands in real time.

The implementation intentionally uses only the Python standard library for the
web server, plus Server-Sent Events (SSE) for streaming. This keeps the GUI
lightweight and avoids adding another web-framework dependency to the package.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
import uuid
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import typer
from rich.console import Console

from minisweagent import global_config_dir
from minisweagent.agents import get_agent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.utilities.config import configure_if_first_time
from minisweagent.utils.serialize import UNSET, recursive_merge

logger = logging.getLogger("minisweagent.gui")

DEFAULT_CONFIG_FILE = Path(builtin_config_dir / "mini.yaml")
DEFAULT_OUTPUT_FILE = global_config_dir / "last_gui_run.traj.json"

_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>mini-SWE-agent</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f5f6f8;
      --panel: #ffffff;
      --text: #1d2433;
      --muted: #667085;
      --border: #d9dee8;
      --accent: #315efb;
      --accent-soft: #e8eeff;
      --ok: #147a4b;
      --warn: #9a6700;
      --err: #b42318;
      --code-bg: #111827;
      --code-text: #e5e7eb;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #0f141d;
        --panel: #171e29;
        --text: #e7ebf2;
        --muted: #9aa6b8;
        --border: #2b3647;
        --accent: #7c9cff;
        --accent-soft: #1d2b4d;
        --ok: #43c88a;
        --warn: #e0b44a;
        --err: #ff8b80;
        --code-bg: #080b10;
        --code-text: #e5e7eb;
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 5;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 20px;
      border-bottom: 1px solid var(--border);
      background: color-mix(in srgb, var(--panel) 92%, transparent);
      backdrop-filter: blur(10px);
    }
    .brand { display: flex; align-items: center; gap: 10px; font-weight: 700; letter-spacing: -0.01em; }
    .brand-mark {
      display: grid; place-items: center; width: 28px; height: 28px; border-radius: 8px;
      background: var(--accent); color: white; font-size: 16px;
    }
    .header-meta { color: var(--muted); font-size: 12px; display: flex; gap: 14px; align-items: center; }
    .status {
      display: inline-flex; align-items: center; gap: 6px; padding: 4px 9px; border-radius: 999px;
      background: var(--accent-soft); color: var(--accent); font-weight: 600;
    }
    .status::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
    .status.running::before { animation: pulse 1s ease-in-out infinite; }
    .status.error { background: color-mix(in srgb, var(--err) 13%, transparent); color: var(--err); }
    @keyframes pulse { 50% { opacity: 0.25; } }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 18px;
      width: min(1280px, 100%);
      margin: 0 auto;
      padding: 18px 20px 40px;
    }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: 0 1px 2px rgb(0 0 0 / 0.04);
    }
    .composer { padding: 16px; }
    .composer label { display: block; margin-bottom: 8px; color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; }
    textarea {
      width: 100%;
      min-height: 130px;
      resize: vertical;
      padding: 12px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: transparent;
      color: var(--text);
      font: inherit;
      outline: none;
    }
    textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
    .composer-actions { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-top: 12px; }
    .hint { color: var(--muted); font-size: 12px; }
    button {
      appearance: none; border: 0; border-radius: 9px; padding: 9px 15px; font: inherit;
      font-weight: 700; cursor: pointer; background: var(--accent); color: white;
    }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    button.secondary { background: transparent; color: var(--text); border: 1px solid var(--border); }
    .feed { margin-top: 18px; display: flex; flex-direction: column; gap: 12px; }
    .empty {
      padding: 42px 20px; text-align: center; color: var(--muted);
    }
    .card { padding: 14px 16px; border-radius: 12px; border: 1px solid var(--border); background: var(--panel); }
    .card-head { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; margin-bottom: 8px; }
    .role { font-size: 11px; font-weight: 800; letter-spacing: 0.07em; text-transform: uppercase; color: var(--muted); }
    .card.user { border-left: 3px solid var(--accent); }
    .card.assistant { border-left: 3px solid var(--ok); }
    .card.command { border-left: 3px solid var(--warn); }
    .card.error { border-left: 3px solid var(--err); }
    .content { white-space: pre-wrap; overflow-wrap: anywhere; }
    .reasoning { margin-top: 8px; color: var(--muted); font-size: 12px; }
    .reasoning summary { cursor: pointer; }
    pre {
      margin: 8px 0 0;
      padding: 12px;
      overflow: auto;
      border-radius: 9px;
      background: var(--code-bg);
      color: var(--code-text);
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .meta { color: var(--muted); font-size: 11px; }
    aside { display: flex; flex-direction: column; gap: 12px; }
    .stats { padding: 16px; }
    .stats h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin: 0 0 12px; }
    .stat { display: flex; justify-content: space-between; gap: 10px; padding: 7px 0; border-bottom: 1px solid var(--border); }
    .stat:last-child { border-bottom: 0; }
    .stat span:last-child { font-variant-numeric: tabular-nums; font-weight: 600; }
    .activity { padding: 16px; color: var(--muted); font-size: 12px; }
    .activity h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; margin: 0 0 10px; }
    #activity-log { margin: 0; padding: 0; list-style: none; display: flex; flex-direction: column; gap: 7px; }
    #activity-log li { padding-left: 12px; position: relative; overflow-wrap: anywhere; }
    #activity-log li::before { content: ""; position: absolute; left: 0; top: 7px; width: 5px; height: 5px; border-radius: 50%; background: var(--accent); }
  </style>
</head>
<body>
  <header>
    <div class="brand"><span class="brand-mark">m</span><span>mini-SWE-agent</span></div>
    <div class="header-meta">
      <span id="model-label"></span>
      <span class="status" id="status">idle</span>
    </div>
  </header>
  <main>
    <section>
      <div class="panel composer">
        <label for="task">Task</label>
        <textarea id="task" placeholder="Describe the software engineering task. For example: find and fix the failing tests in this repository."></textarea>
        <div class="composer-actions">
          <span class="hint">Commands run in the configured local environment.</span>
          <button id="run">Run agent</button>
        </div>
      </div>
      <div class="feed" id="feed">
        <div class="panel empty" id="empty">Responses stream here as the agent works.</div>
      </div>
    </section>
    <aside>
      <div class="panel stats">
        <h2>Run</h2>
        <div class="stat"><span>Step</span><span id="stat-step">-</span></div>
        <div class="stat"><span>Cost</span><span id="stat-cost">$0.00</span></div>
        <div class="stat"><span>Status</span><span id="stat-status">idle</span></div>
        <div class="stat"><span>Exit</span><span id="stat-exit">-</span></div>
      </div>
      <div class="panel activity">
        <h2>Activity</h2>
        <ul id="activity-log"><li>Waiting for a task.</li></ul>
      </div>
    </aside>
  </main>
  <script>
    const feed = document.getElementById("feed");
    const empty = document.getElementById("empty");
    const taskEl = document.getElementById("task");
    const runButton = document.getElementById("run");
    const statusEl = document.getElementById("status");
    const statStep = document.getElementById("stat-step");
    const statCost = document.getElementById("stat-cost");
    const statStatus = document.getElementById("stat-status");
    const statExit = document.getElementById("stat-exit");
    const activityLog = document.getElementById("activity-log");
    let eventSource = null;
    let currentAssistant = null;
    let currentReasoning = null;
    let submitted = false;

    function setStatus(text, state = "") {
      statusEl.textContent = text;
      statusEl.className = "status " + state;
      statStatus.textContent = text;
    }

    function addActivity(text) {
      if (submitted) {
        activityLog.innerHTML = "";
        submitted = false;
      }
      const li = document.createElement("li");
      li.textContent = text;
      activityLog.prepend(li);
      while (activityLog.children.length > 12) activityLog.lastElementChild.remove();
    }

    function clearFeed() {
      feed.innerHTML = "";
      currentAssistant = null;
      currentReasoning = null;
    }

    function makeCard(kind, role, meta = "") {
      const card = document.createElement("div");
      card.className = "card " + kind;
      const head = document.createElement("div");
      head.className = "card-head";
      const label = document.createElement("span");
      label.className = "role";
      label.textContent = role;
      head.appendChild(label);
      if (meta) {
        const m = document.createElement("span");
        m.className = "meta";
        m.textContent = meta;
        head.appendChild(m);
      }
      card.appendChild(head);
      feed.appendChild(card);
      return card;
    }

    function appendText(card, text) {
      let content = card.querySelector(".content");
      if (!content) {
        content = document.createElement("div");
        content.className = "content";
        card.appendChild(content);
      }
      content.textContent += text;
      return content;
    }

    function addUserCard(task) {
      const card = makeCard("user", "Task");
      appendText(card, task);
    }

    function ensureAssistantCard() {
      if (!currentAssistant) {
        currentAssistant = makeCard("assistant", "mini-SWE-agent");
      }
      return currentAssistant;
    }

    function handleEvent(event) {
      switch (event.type) {
        case "run_start":
          addUserCard(event.task || "");
          setStatus("running", "running");
          addActivity("Run started");
          break;
        case "step_start":
          statStep.textContent = String(event.step || "-");
          setStatus("running", "running");
          addActivity("Step " + event.step + " started");
          break;
        case "assistant_start":
          ensureAssistantCard();
          break;
        case "assistant_delta": {
          const card = ensureAssistantCard();
          appendText(card, event.text || "");
          break;
        }
        case "reasoning_delta": {
          const card = ensureAssistantCard();
          if (!currentReasoning) {
            const details = document.createElement("details");
            details.className = "reasoning";
            const summary = document.createElement("summary");
            summary.textContent = "Reasoning";
            details.appendChild(summary);
            const body = document.createElement("div");
            body.className = "reasoning-body";
            details.appendChild(body);
            card.appendChild(details);
            currentReasoning = body;
          }
          currentReasoning.textContent += event.text || "";
          break;
        }
        case "tool_call_delta":
          if (!currentAssistant) ensureAssistantCard();
          break;
        case "assistant_message": {
          const card = ensureAssistantCard();
          if (event.content) {
            let content = card.querySelector(".content");
            if (!content) {
              content = document.createElement("div");
              content.className = "content";
              card.insertBefore(content, card.querySelector(".reasoning"));
            }
            content.textContent = event.content;
          }
          const meta = document.createElement("span");
          meta.className = "meta";
          meta.textContent = "step " + (event.step || "?") + " · $" + Number(event.cost || 0).toFixed(4);
          card.querySelector(".card-head").appendChild(meta);
          const currentCost = Number(statCost.textContent.replace("$", "")) || 0;
          statCost.textContent = "$" + (currentCost + Number(event.cost || 0)).toFixed(4);
          currentAssistant = null;
          currentReasoning = null;
          break;
        }
        case "action_start": {
          const card = makeCard("command", "Command", "running");
          const pre = document.createElement("pre");
          pre.textContent = "$ " + (event.command || "");
          card.appendChild(pre);
          card.dataset.command = event.command || "";
          addActivity("Running: " + (event.command || "").split("\n")[0]);
          break;
        }
        case "action_output": {
          const cards = feed.querySelectorAll(".card.command");
          let card = null;
          for (let i = cards.length - 1; i >= 0; i--) {
            if (cards[i].dataset.command === (event.command || "")) { card = cards[i]; break; }
          }
          if (!card) card = makeCard("command", "Command");
          const headMeta = card.querySelector(".meta");
          if (headMeta) headMeta.textContent = "exit " + (event.returncode ?? "?");
          const pre = document.createElement("pre");
          const output = event.output || "";
          const max = 20000;
          pre.textContent = output.length > max ? output.slice(0, max) + "\n\n... output truncated in GUI ..." : output;
          if (output.length === 0) pre.textContent = "(no output)";
          card.appendChild(pre);
          addActivity("Finished with exit " + (event.returncode ?? "?"));
          break;
        }
        case "observation":
          break;
        case "format_error": {
          currentAssistant = null;
          currentReasoning = null;
          const card = makeCard("error", "Format error", "retrying");
          appendText(card, event.message || "");
          addActivity("Model response format error");
          break;
        }
        case "error": {
          currentAssistant = null;
          currentReasoning = null;
          const card = makeCard("error", "Error");
          appendText(card, event.message || "Unknown error");
          setStatus("error", "error");
          addActivity("Agent error");
          break;
        }
        case "done":
          statExit.textContent = event.exit_status || "-";
          statCost.textContent = "$" + Number(event.cost || 0).toFixed(4);
          statStep.textContent = String(event.n_calls ?? statStep.textContent);
          const successful = event.exit_status === "Submitted";
          setStatus(event.exit_status || "done", successful ? "" : "error");
          if (event.submission) {
            const card = makeCard("assistant", "Submission");
            appendText(card, event.submission);
          }
          addActivity("Run finished: " + (event.exit_status || "unknown"));
          closeEvents();
          runButton.disabled = false;
          break;
      }
      feed.scrollIntoView({ block: "end", behavior: "smooth" });
    }

    function closeEvents() {
      if (eventSource) {
        eventSource.close();
        eventSource = null;
      }
      runButton.disabled = false;
    }

    function connectEvents(sessionId) {
      eventSource = new EventSource("/api/events/" + encodeURIComponent(sessionId));
      eventSource.onmessage = (message) => {
        try { handleEvent(JSON.parse(message.data)); }
        catch (err) { console.error("Bad event", err, message.data); }
      };
      eventSource.onerror = () => {
        if (eventSource) {
          closeEvents();
          setStatus("disconnected", "error");
          addActivity("Stream disconnected");
        }
      };
    }

    async function startRun() {
      const task = taskEl.value.trim();
      if (!task) {
        taskEl.focus();
        return;
      }
      runButton.disabled = true;
      submitted = true;
      clearFeed();
      activityLog.innerHTML = "";
      statStep.textContent = "-";
      statCost.textContent = "$0.0000";
      statExit.textContent = "-";
      setStatus("starting", "running");
      try {
        const response = await fetch("/api/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ task }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || ("HTTP " + response.status));
        connectEvents(data.session_id);
      } catch (err) {
        runButton.disabled = false;
        setStatus("error", "error");
        const card = makeCard("error", "Error");
        appendText(card, String(err));
      }
    }

    runButton.addEventListener("click", startRun);
    taskEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) startRun();
    });

    fetch("/api/config").then((r) => r.json()).then((config) => {
      if (config.initial_task) taskEl.value = config.initial_task;
      document.getElementById("model-label").textContent = config.model_name || "default model";
    }).catch(() => {});
  </script>
</body>
</html>
"""


@dataclass
class GuiConfig:
    """Configuration shared by every GUI-triggered agent run."""

    agent_config: dict[str, Any]
    initial_task: str = ""
    model_name: str = ""


def build_gui_config(
    *,
    config_spec: list[str],
    model_name: str | None = None,
    model_class: str | None = None,
    agent_class: str | None = None,
    environment_class: str | None = None,
    output: Path | None = None,
    initial_task: str = "",
) -> GuiConfig:
    """Merge CLI config specs into the configuration used for each GUI run."""
    configs: list[dict[str, Any]] = [get_config_from_spec(spec) for spec in config_spec]
    configs.append(
        {
            "agent": {
                # A browser has no terminal, so confirmation prompts would block.
                # The GUI runs non-interactively; use the default agent unless the
                # caller explicitly selected another one.
                "agent_class": agent_class or "default",
                "mode": "yolo",
                "confirm_exit": False,
                "output_path": output or UNSET,
            },
            "model": {
                "model_class": model_class or UNSET,
                "model_name": model_name or UNSET,
            },
            "environment": {
                "environment_class": environment_class or UNSET,
            },
        }
    )
    config = recursive_merge(*configs)
    return GuiConfig(
        agent_config=config,
        initial_task=initial_task,
        model_name=str(config.get("model", {}).get("model_name") or ""),
    )


def create_agent_from_config(config: GuiConfig) -> Any:
    """Create a fresh model, environment, and agent for one GUI session."""
    model = get_model(config=config.agent_config.get("model", {}))
    env = get_environment(config.agent_config.get("environment", {}), default_type="local")
    return get_agent(model, env, config.agent_config.get("agent", {}), default_type="default")


def _json_safe(value: Any) -> Any:
    """Convert arbitrary event payloads into JSON-compatible values."""
    return json.loads(json.dumps(value, default=str))


class GuiSession:
    """One browser-triggered agent run and its replayable event stream."""

    def __init__(self, task: str, *, config: GuiConfig, agent_factory: Callable[[], Any] | None = None):
        self.session_id = uuid.uuid4().hex
        self.task = task
        self.config = config
        self.agent_factory = agent_factory or (lambda: create_agent_from_config(config))
        self.events: list[dict[str, Any]] = []
        self.condition = threading.Condition()
        self.done = False
        self._done_emitted = False
        self.thread = threading.Thread(target=self._run, name=f"mini-gui-{self.session_id[:8]}", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def emit(self, event: dict[str, Any]) -> None:
        safe_event = _json_safe(event)
        with self.condition:
            self.events.append(safe_event)
            if safe_event.get("type") == "done":
                self._done_emitted = True
            self.condition.notify_all()

    def _run(self) -> None:
        result: dict[str, Any] = {"exit_status": "Error", "submission": "", "cost": 0.0, "n_calls": 0}
        try:
            agent = self.agent_factory()
            if hasattr(agent, "set_event_callback"):
                agent.set_event_callback(self.emit)
            returned = agent.run(self.task)
            if isinstance(returned, dict):
                result = {**result, **returned}
        except BaseException as exc:  # noqa: BLE001 - report all failures to the browser
            self.emit(
                {
                    "type": "error",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            logger.exception("GUI agent run failed")
        finally:
            if not self._done_emitted:
                self.emit({"type": "done", **result})
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait until the run is complete. Returns ``True`` if it finished."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.condition:
            while not self.done:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if remaining == 0.0:
                    return False
                self.condition.wait(remaining)
            return True


class GuiSessionManager:
    """Owns GUI sessions and their shared agent configuration."""

    def __init__(self, config: GuiConfig, *, agent_factory: Callable[[], Any] | None = None):
        self.config = config
        self.agent_factory = agent_factory
        self.sessions: dict[str, GuiSession] = {}
        self._lock = threading.Lock()

    def start_session(self, task: str) -> GuiSession:
        task = task.strip()
        if not task:
            raise ValueError("Task must not be empty")
        session = GuiSession(task, config=self.config, agent_factory=self.agent_factory)
        with self._lock:
            self.sessions[session.session_id] = session
        session.start()
        return session

    def get_session(self, session_id: str) -> GuiSession | None:
        with self._lock:
            return self.sessions.get(session_id)


class GuiHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server with a reference to the session manager."""

    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], manager: GuiSessionManager):
        self.manager = manager
        super().__init__(server_address, GuiRequestHandler)


class GuiRequestHandler(BaseHTTPRequestHandler):
    server: GuiHTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        logger.debug("%s - %s", self.address_string(), format % args)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib method name
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send_html(_HTML)
        elif parsed.path == "/api/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
        elif parsed.path == "/api/config":
            self._send_json(
                HTTPStatus.OK,
                {
                    "initial_task": self.server.manager.config.initial_task,
                    "model_name": self.server.manager.config.model_name,
                },
            )
        elif parsed.path.startswith("/api/events/"):
            session_id = parsed.path.removeprefix("/api/events/").strip("/")
            session = self.server.manager.get_session(session_id)
            if session is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown session"})
                return
            self._stream_events(session)
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib method name
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length"})
            return
        if length <= 0 or length > 1_000_000:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid request body"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            task = payload.get("task", "")
            if not isinstance(task, str):
                raise ValueError("task must be a string")
            session = self.server.manager.start_session(task)
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001 - surface setup errors to UI
            logger.exception("Could not start GUI session")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return
        self._send_json(
            HTTPStatus.ACCEPTED,
            {"session_id": session.session_id, "events_url": f"/api/events/{session.session_id}"},
        )

    def _stream_events(self, session: GuiSession) -> None:
        try:
            cursor = int(self.headers.get("Last-Event-ID", "0") or "0")
        except ValueError:
            cursor = 0
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            while True:
                with session.condition:
                    while cursor >= len(session.events) and not session.done:
                        session.condition.wait(timeout=15.0)
                    batch = session.events[cursor:]
                    finished = session.done
                    cursor = len(session.events)
                if batch:
                    first_event_id = cursor - len(batch) + 1
                    for offset, event in enumerate(batch):
                        payload = json.dumps(event)
                        self.wfile.write(f"id: {first_event_id + offset}\ndata: {payload}\n\n".encode())
                    self.wfile.flush()
                elif not finished:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                if finished and cursor >= len(session.events):
                    break
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("GUI event stream disconnected")
        finally:
            self.close_connection = True


def create_server(host: str, port: int, manager: GuiSessionManager) -> GuiHTTPServer:
    """Create (but do not start) the GUI HTTP server."""
    return GuiHTTPServer((host, port), manager)


def serve(
    manager: GuiSessionManager,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
) -> None:
    """Serve the GUI until interrupted."""
    server = create_server(host, port, manager)
    actual_host, actual_port = server.server_address[:2]
    url_host = "127.0.0.1" if actual_host in ("0.0.0.0", "::") else actual_host
    url = f"http://{url_host}:{actual_port}"
    Console().print(f"[bold green]mini-SWE-agent GUI:[/bold green] {url}")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        Console().print("\n[dim]Shutting down GUI...[/dim]")
    finally:
        server.server_close()


app = typer.Typer(rich_markup_mode="rich", add_completion=False)


@app.command(help="Launch a basic web GUI for mini-SWE-agent with streaming responses.")
def main(
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model to use"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use"),
    agent_class: str | None = typer.Option(
        None, "--agent-class", help="Agent class to use (the default agent is recommended for the GUI)"
    ),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment class to use"),
    task: str | None = typer.Option(None, "-t", "--task", help="Initial task to prefill in the GUI"),
    config_spec: list[str] = typer.Option(
        [str(DEFAULT_CONFIG_FILE)], "-c", "--config", help="Path to config files or key=value config overrides"
    ),
    output: Path | None = typer.Option(DEFAULT_OUTPUT_FILE, "-o", "--output", help="Trajectory output path"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host interface to bind"),
    port: int = typer.Option(8000, "--port", help="Port to bind (0 chooses a free port)"),
    open_browser: bool = typer.Option(True, "--browser/--no-browser", help="Open the GUI in a browser"),
) -> None:
    configure_if_first_time()
    gui_config = build_gui_config(
        config_spec=config_spec,
        model_name=model_name,
        model_class=model_class,
        agent_class=agent_class,
        environment_class=environment_class,
        output=output,
        initial_task=task or "",
    )
    manager = GuiSessionManager(gui_config)
    serve(manager, host=host, port=port, open_browser=open_browser)


if __name__ == "__main__":
    app()
