#!/usr/bin/env python3
"""A basic web GUI for mini-SWE-agent with live streaming responses.

Run ``mini-gui`` (or ``python -m minisweagent.run.gui``), open the printed URL,
enter a task, and watch the agent think and execute commands in real time.

The implementation intentionally uses only the Python standard library for the
web server, plus Server-Sent Events (SSE) for streaming. This keeps the GUI
lightweight and avoids adding another web-framework dependency to the package.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import threading
import time
import traceback
import uuid
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import typer
from rich.console import Console

from minisweagent import global_config_dir
from minisweagent.agents import get_agent
from minisweagent.agents.utils.conversation import conversational_reply
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.exceptions import UserInterruption
from minisweagent.models import get_model
from minisweagent.run.utilities.config import configure_if_first_time
from minisweagent.run.utilities.conversation_store import ConversationStore
from minisweagent.utils.serialize import UNSET, recursive_merge

logger = logging.getLogger("minisweagent.gui")

DEFAULT_CONFIG_FILE = Path(builtin_config_dir / "mini.yaml")
DEFAULT_OUTPUT_FILE = global_config_dir / "last_gui_run.traj.json"
DEFAULT_DATABASE_FILE = Path(os.getenv("MSWEA_GUI_DB_PATH") or global_config_dir / "gui_conversations.db")


class ConversationState(str, Enum):
    """Canonical states shown for a GUI conversation."""

    RUNNING = "running"
    SUBMITTED = "submitted"
    STOPPED = "stopped"
    FAILED = "failed"
    INTERRUPTED = "interrupted"

    # Semantic aliases for callers that use completed/error terminology.
    COMPLETED = "submitted"
    ERROR = "failed"


def conversation_state(status: str, *, done: bool) -> ConversationState:
    """Normalize an agent exit status into one of the GUI conversation states."""
    normalized = str(status or "").strip().lower()
    if not done:
        return ConversationState.RUNNING
    if normalized in {"submitted", "complete", "completed", "success", "succeeded"}:
        return ConversationState.SUBMITTED
    if normalized in {"stopped", "stop", "cancelled", "canceled"}:
        return ConversationState.STOPPED
    if normalized in {"interrupted", "interrupt"}:
        return ConversationState.INTERRUPTED
    return ConversationState.FAILED


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
    html, body { height: 100%; }
    body {
      margin: 0;
      height: 100vh;
      height: 100dvh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      position: relative;
      z-index: 5;
      flex: 0 0 auto;
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
    .header-meta { color: var(--muted); font-size: 12px; display: flex; gap: 14px; align-items: center; min-width: 0; }
    #database-path {
      max-width: min(52vw, 720px);
      overflow: hidden;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 10px;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .status {
      display: inline-flex; align-items: center; gap: 6px; padding: 4px 9px; border-radius: 999px;
      background: var(--accent-soft); color: var(--accent); font-weight: 600;
    }
    .status::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
    .status.running::before { animation: pulse 1s ease-in-out infinite; }
    .status.error, .status.failed { background: color-mix(in srgb, var(--err) 13%, transparent); color: var(--err); }
    .status.submitted { background: color-mix(in srgb, var(--ok) 13%, transparent); color: var(--ok); }
    .status.stopped { background: color-mix(in srgb, var(--muted) 13%, transparent); color: var(--muted); }
    .status.interrupted { background: color-mix(in srgb, var(--warn) 13%, transparent); color: var(--warn); }
    .conversation-bar {
      flex: 0 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 14px;
    }
    .conversation-context {
      min-width: 0;
      display: flex;
      align-items: baseline;
      gap: 10px;
    }
    .conversation-label {
      flex: 0 1 auto;
      min-width: 0;
      overflow: hidden;
      color: var(--text);
      font-size: 13px;
      font-weight: 650;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .conversation-workspace {
      min-width: 0;
      overflow: hidden;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 11px;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .conversation-id {
      min-width: 0;
      overflow: hidden;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 11px;
      text-overflow: ellipsis;
      white-space: nowrap;
      user-select: all;
      cursor: pointer;
    }
    .conversation-id:hover { color: var(--accent); }
    .conversation-id-row { display: flex; min-width: 0; margin-top: 2px; }
    .conversation-id-row .conversation-id { max-width: 100%; }
    .conversation-actions { display: flex; align-items: center; gap: 8px; }
    .conversation-actions button { padding: 5px 9px; border-radius: 7px; font-size: 12px; white-space: nowrap; }
    .conversation-bar .status { flex: 0 0 auto; }
    @keyframes pulse { 50% { opacity: 0.25; } }
    main {
      flex: 1;
      min-height: 0;
      display: grid;
      grid-template-columns: 250px minmax(0, 1fr) 300px;
      grid-template-rows: minmax(0, 1fr);
      gap: 18px;
      width: min(1440px, 100%);
      margin: 0 auto;
      padding: 18px 20px;
    }
    section { display: flex; flex-direction: column; min-height: 0; gap: 12px; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: 0 1px 2px rgb(0 0 0 / 0.04);
    }
    .conversations {
      padding: 12px;
      gap: 8px;
      overflow: hidden;
    }
    .conversations-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 2px 2px 8px;
    }
    .conversations-head h2 {
      margin: 0;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }
    #new-conversation {
      padding: 5px 9px;
      border-radius: 7px;
      font-size: 12px;
      white-space: nowrap;
    }
    .conversation-list {
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 5px;
      padding-right: 2px;
    }
    .conversation-item {
      width: 100%;
      padding: 9px 10px;
      border: 1px solid transparent;
      border-radius: 9px;
      background: transparent;
      color: var(--text);
      text-align: left;
      font-weight: 500;
    }
    .conversation-item:hover {
      background: color-mix(in srgb, var(--accent-soft) 60%, transparent);
    }
    .conversation-item.active {
      background: var(--accent-soft);
      border-color: color-mix(in srgb, var(--accent) 35%, var(--border));
    }
    .conversation-title-row {
      display: flex;
      align-items: baseline;
      gap: 8px;
      min-width: 0;
    }
    .conversation-title {
      flex: 1 1 auto;
      min-width: 0;
      display: block;
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
      font-size: 13px;
      font-weight: 650;
    }
    .conversation-title-row .conversation-workspace { flex: 0 1 auto; max-width: 48%; }
    .conversation-meta {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      margin-top: 3px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 400;
    }
    .conversation-status.running { color: var(--accent); }
    .conversation-status.submitted { color: var(--ok); }
    .conversation-status.stopped { color: var(--muted); }
    .conversation-status.failed { color: var(--err); }
    .conversation-status.interrupted { color: var(--warn); }
    .conversation-empty {
      padding: 24px 10px;
      color: var(--muted);
      font-size: 12px;
      text-align: center;
    }
    .composer { flex: 0 0 auto; padding: 12px; }
    .composer-attachments { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; }
    .composer-attachment {
      position: relative; width: 72px; height: 72px; border-radius: 8px;
      overflow: hidden; border: 1px solid var(--border); background: var(--panel);
    }
    .composer-attachment img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .composer-attachment button {
      position: absolute; top: 3px; right: 3px; width: 20px; height: 20px;
      padding: 0; border-radius: 50%; font-size: 13px; line-height: 1;
      background: rgba(0, 0, 0, 0.68); color: white;
    }
    .composer-input-row { display: flex; align-items: flex-end; gap: 8px; }
    .composer-icon-button {
      flex: 0 0 auto;
      width: 42px;
      height: 42px;
      padding: 0;
      border-radius: 10px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 24px;
      font-weight: 400;
      line-height: 1;
    }
    .composer.dragging textarea { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
    /* Compact single-line task input; it grows only while the user types. */
    textarea {
      flex: 1 1 auto;
      width: 100%;
      min-width: 0;
      min-height: 42px;
      height: 42px;
      max-height: 168px;
      resize: none;
      overflow-y: auto;
      padding: 10px 12px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: transparent;
      color: var(--text);
      font: inherit;
      line-height: 1.4;
      outline: none;
    }
    textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
    .composer-action { flex: 0 0 auto; align-self: flex-end; }
    .composer-action button { min-width: 76px; height: 42px; padding: 0 16px; border-radius: 10px; }
    .composer-action button[hidden] { display: none; }
    .composer-action .danger { background: var(--err); color: white; }
    .composer-actions { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-top: 8px; min-height: 16px; }
    .composer-actions .conversation-workspace { flex: 1 1 auto; min-width: 0; }
    .button-group { display: flex; align-items: center; gap: 8px; }
    button {
      appearance: none; border: 0; border-radius: 9px; padding: 9px 15px; font: inherit;
      font-weight: 700; cursor: pointer; background: var(--accent); color: white;
    }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    button.secondary { background: transparent; color: var(--text); border: 1px solid var(--border); }
    .feed-wrap { position: relative; flex: 1; min-height: 0; display: flex; }
    .feed {
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 9px;
      padding: 6px 4px 28px;
      scroll-behavior: smooth;
    }
    .scroll-actions {
      position: absolute;
      right: 14px;
      bottom: 14px;
      z-index: 4;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .scroll-actions button {
      padding: 7px 11px;
      font-size: 12px;
      background: var(--panel);
      color: var(--text);
      border: 1px solid var(--border);
      box-shadow: 0 2px 8px rgb(0 0 0 / 0.12);
    }
    .scroll-actions button[hidden] { display: none; }
    .empty {
      padding: 42px 20px; text-align: center; color: var(--muted);
    }
    .run-marker {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 3px 4px;
      color: var(--muted);
      font-size: 10px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .run-marker::before {
      content: "";
      flex: 0 0 auto;
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--muted);
    }
    .run-marker::after {
      content: "";
      flex: 1 1 auto;
      height: 1px;
      background: linear-gradient(90deg, var(--border), transparent);
    }
    .run-marker.start::before { background: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
    .run-marker.end::before { background: var(--ok); box-shadow: 0 0 0 3px color-mix(in srgb, var(--ok) 16%, transparent); }
    .run-marker-meta {
      margin-left: auto;
      color: var(--muted);
      font-weight: 600;
      letter-spacing: 0;
      text-transform: none;
    }
    .card {
      position: relative;
      flex: 0 0 auto;
      max-width: min(94%, 980px);
      padding: 11px 14px;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: var(--panel);
      overflow-wrap: anywhere;
    }
    .card-head { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; margin-bottom: 7px; }
    .role { font-size: 11px; font-weight: 800; letter-spacing: 0.07em; text-transform: uppercase; color: var(--muted); }
    .card.user {
      align-self: flex-end;
      border-color: color-mix(in srgb, var(--accent) 28%, var(--border));
      border-bottom-right-radius: 4px;
      background: color-mix(in srgb, var(--accent-soft) 52%, var(--panel));
    }
    /* Assistant messages read like a transcript: no bubble or per-step chrome. */
    .card.assistant {
      align-self: stretch;
      width: 100%;
      max-width: 100%;
      padding: 2px 6px;
      border: 0;
      border-radius: 0;
      background: transparent;
      box-shadow: none;
    }
    .card.assistant .card-head { display: none; }
    .card.assistant .content { line-height: 1.65; }
    .card.error { align-self: stretch; border-left: 3px solid var(--err); }
    .card[hidden] { display: none !important; }
    .content { white-space: pre-wrap; overflow-wrap: anywhere; }
    .card-images { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 9px; }
    .card-images a { display: block; line-height: 0; }
    .card-images img {
      max-width: 220px; max-height: 220px; border-radius: 8px;
      border: 1px solid var(--border); object-fit: contain; background: var(--bg);
    }
    .reasoning { margin-top: 4px; color: var(--muted); font-size: 11px; }
    .reasoning summary {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      color: var(--muted);
      cursor: pointer;
      list-style: none;
      user-select: none;
    }
    .reasoning summary::-webkit-details-marker { display: none; }
    .reasoning summary::before {
      content: "\25B8";
      font-size: 9px;
      transition: transform 120ms ease;
    }
    .reasoning[open] summary::before { transform: rotate(90deg); }
    .reasoning-body { margin-top: 6px; padding-left: 13px; white-space: pre-wrap; overflow-wrap: anywhere; }

    /* A tool call is one quiet line in the transcript.  Its command, exit
       status, duration and output belong in the expandable Activity view. */
    .tool-call {
      display: flex;
      align-items: baseline;
      gap: 7px;
      flex: 0 0 auto;
      width: 100%;
      max-width: 100%;
      margin: -2px 0 2px;
      padding: 5px 7px;
      border: 1px solid transparent;
      border-radius: 7px;
      background: transparent;
      color: var(--text);
      font: 500 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      text-align: left;
      cursor: pointer;
    }
    .tool-call:hover {
      border-color: var(--border);
      background: color-mix(in srgb, var(--panel) 84%, var(--bg));
    }
    .tool-call:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
    .tool-call-icon { flex: 0 0 auto; color: var(--muted); font-size: 9px; }
    .tool-call-prompt { flex: 0 0 auto; color: var(--ok); font-weight: 800; }
    .tool-call-command {
      flex: 1 1 auto;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .tool-call-status {
      flex: 0 0 auto;
      color: var(--muted);
      font-size: 10px;
      font-variant-numeric: tabular-nums;
      font-weight: 700;
      white-space: nowrap;
    }
    .tool-call-status.running { color: var(--accent); }
    .tool-call-status.ok { color: var(--ok); }
    .tool-call-status.error { color: var(--err); }
    pre {
      margin: 8px 0 0;
      padding: 10px 12px;
      overflow: auto;
      border-radius: 8px;
      background: var(--code-bg);
      color: var(--code-text);
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .meta { color: var(--muted); font-size: 11px; }
    aside { min-height: 0; overflow: hidden; display: flex; flex-direction: column; gap: 12px; }
    aside.details { overflow: hidden; }
    .stats { padding: 16px; flex: 0 0 auto; }
    .stats h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin: 0 0 12px; }
    .stat { display: flex; justify-content: space-between; gap: 10px; padding: 7px 0; border-bottom: 1px solid var(--border); }
    .stat:last-child { border-bottom: 0; }
    .stat span:last-child { font-variant-numeric: tabular-nums; font-weight: 600; }

    /* Activity: a compact, turn-aware ledger modelled after the DSH trajectory view. */
    .activity {
      flex: 1 1 auto;
      min-height: 0;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      color: var(--text);
      font-size: 12px;
    }
    .activity-header {
      flex: 0 0 auto;
      display: flex;
      flex-direction: column;
      gap: 9px;
      padding: 13px 14px 10px;
      border-bottom: 1px solid var(--border);
    }
    .activity-heading { display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; }
    .activity-heading h2 {
      margin: 0;
      color: var(--text);
      font-size: 12px;
      font-weight: 750;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .activity-subtitle {
      margin: 3px 0 0;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
    }
    .activity-heading-actions { flex: 0 0 auto; display: flex; align-items: center; gap: 6px; }
    #activity-expand {
      flex: 0 0 auto;
      padding: 4px 8px;
      border-radius: 7px;
      font-size: 10px;
      line-height: 1.3;
    }
    #activity-expand[aria-expanded="true"] {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }
    .activity-summary {
      display: flex;
      align-items: center;
      gap: 6px;
      min-height: 19px;
      color: var(--muted);
      font-size: 10px;
      font-variant-numeric: tabular-nums;
      font-weight: 650;
      letter-spacing: 0.01em;
    }
    .activity-summary::before {
      content: "";
      flex: 0 0 auto;
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--muted);
    }
    .activity-summary.running { color: var(--accent); }
    .activity-summary.running::before { background: var(--accent); animation: pulse 1s ease-in-out infinite; }
    .activity-summary.done { color: var(--ok); }
    .activity-summary.done::before { background: var(--ok); }
    .activity-summary.error { color: var(--err); }
    .activity-summary.error::before { background: var(--err); }
    .activity-overview {
      display: flex;
      align-items: stretch;
      gap: 2px;
      height: 9px;
      padding: 2px;
      overflow: hidden;
      border-radius: 999px;
      background: var(--bg);
    }
    .activity-segment {
      flex: 1 1 0;
      min-width: 2px;
      max-width: 22px;
      border-radius: 2px;
      background: color-mix(in srgb, var(--accent) 75%, var(--panel));
    }
    .activity-segment.kind-tool { background: color-mix(in srgb, var(--warn) 78%, var(--panel)); }
    .activity-segment.kind-done { background: var(--ok); }
    .activity-segment.kind-error { background: var(--err); }
    .activity-segment.kind-run { background: var(--accent); }
    .activity-segment.kind-info { background: var(--muted); }
    .activity-overview-empty {
      flex: 1;
      color: var(--muted);
      font-size: 9px;
      line-height: 5px;
    }
    .activity-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 3px 9px;
      color: var(--muted);
      font-size: 9px;
      line-height: 1.25;
    }
    .activity-legend-item { display: inline-flex; align-items: center; gap: 4px; }
    .activity-legend-dot { width: 6px; height: 6px; border-radius: 2px; background: var(--accent); }
    .activity-legend-dot.tool { background: var(--warn); }
    .activity-legend-dot.outcome { background: var(--ok); }
    .activity-legend-dot.error { background: var(--err); }
    .activity-scroll {
      position: relative;
      flex: 1 1 auto;
      min-height: 0;
      display: flex;
      flex-direction: column;
      /* Keep the newest entries at the top.  When an older entry does not
         fit, the unused space belongs below the ledger rather than above it. */
      justify-content: flex-start;
      overflow: hidden;
      padding: 10px 12px 12px;
    }
    /* A continuous timeline spine keeps the guide line visible through the
       empty space below the last visible activity. */
    .activity-scroll.has-activity::after {
      content: "";
      position: absolute;
      left: 22px;
      top: 32px;
      bottom: 12px;
      width: 1px;
      background: var(--border);
      pointer-events: none;
      z-index: 0;
    }
    /* When older activities are clipped, lead into the first visible marker
       so the guide line still communicates that the timeline continues. */
    .activity-scroll.has-clipped-history::before {
      content: "";
      position: absolute;
      left: 22px;
      top: 10px;
      width: 1px;
      height: 22px;
      background: var(--border);
      pointer-events: none;
      z-index: 0;
    }
    #activity-log {
      position: relative;
      z-index: 1;
      margin: 0;
      padding: 0;
      list-style: none;
      display: flex;
      flex-direction: column;
      flex: 0 0 auto;
      min-height: 0;
    }
    .activity-entry[hidden] { display: none; }
    .activity-empty {
      padding: 18px 8px 12px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.45;
      text-align: center;
    }
    .activity-empty[hidden] { display: none; }
    .activity-entry {
      position: relative;
      display: grid;
      grid-template-columns: 22px minmax(0, 1fr);
      gap: 8px;
      padding-bottom: 12px;
      min-width: 0;
    }
    .activity-entry:not(:last-child)::before {
      content: "";
      position: absolute;
      left: 10px;
      top: 22px;
      bottom: 0;
      width: 1px;
      background: var(--border);
    }
    .activity-marker {
      position: relative;
      z-index: 1;
      display: grid;
      place-items: center;
      align-self: start;
      width: 22px;
      height: 22px;
      border-radius: 7px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 10px;
      font-weight: 800;
      line-height: 1;
      font-variant-numeric: tabular-nums;
    }
    .activity-entry.kind-tool .activity-marker { background: color-mix(in srgb, var(--warn) 13%, transparent); color: var(--warn); }
    .activity-entry.kind-done .activity-marker { background: color-mix(in srgb, var(--ok) 13%, transparent); color: var(--ok); }
    .activity-entry.kind-error .activity-marker { background: color-mix(in srgb, var(--err) 13%, transparent); color: var(--err); }
    .activity-entry.kind-info .activity-marker { background: color-mix(in srgb, var(--muted) 13%, transparent); color: var(--muted); }
    .activity-card { min-width: 0; padding-top: 1px; }
    .activity-row-head { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; min-width: 0; }
    .activity-label {
      min-width: 0;
      overflow: hidden;
      color: var(--text);
      font-size: 11px;
      font-weight: 700;
      line-height: 1.35;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .activity-time {
      flex: 0 0 auto;
      color: var(--muted);
      font-size: 9px;
      font-variant-numeric: tabular-nums;
      line-height: 1.35;
    }
    .activity-duration {
      display: none;
      flex: 0 0 auto;
      padding: 1px 5px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 9px;
      font-variant-numeric: tabular-nums;
      font-weight: 650;
      line-height: 1.35;
      white-space: nowrap;
    }
    .activity-copy {
      margin-top: 2px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.4;
      overflow-wrap: anywhere;
    }
    .activity-copy code {
      display: block;
      overflow: hidden;
      color: var(--text);
      font: 11px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .activity-command-full { display: none; }
    /* Keep tool output out of the compact rail; Activity Expand is the detail view. */
    .activity-output {
      display: none;
      margin-top: 7px;
      padding: 7px 9px;
      overflow: hidden;
      border-radius: 6px;
      background: color-mix(in srgb, var(--code-bg) 6%, var(--bg));
      color: var(--muted);
      font: 10px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
    }
    .activity-output-label {
      display: block;
      margin-bottom: 4px;
      color: var(--muted);
      font: 800 9px/1.2 ui-sans-serif, system-ui, sans-serif;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .activity-output-text { white-space: pre-wrap; overflow-wrap: anywhere; }
    .activity-entry.activity-highlight {
      border-color: color-mix(in srgb, var(--accent) 40%, var(--border));
      background: color-mix(in srgb, var(--accent-soft) 52%, transparent);
    }
    .activity-meta-row { display: flex; flex-wrap: wrap; align-items: center; gap: 5px; margin-top: 5px; }
    .activity-status {
      display: inline-flex;
      align-items: center;
      padding: 1px 6px;
      border-radius: 999px;
      background: color-mix(in srgb, var(--muted) 10%, transparent);
      color: var(--muted);
      font-size: 9px;
      font-weight: 750;
      line-height: 1.45;
      text-transform: lowercase;
    }
    .activity-status[hidden], .activity-meta[hidden] { display: none; }
    .activity-status.running { background: var(--accent-soft); color: var(--accent); }
    .activity-status.ok { background: color-mix(in srgb, var(--ok) 12%, transparent); color: var(--ok); }
    .activity-status.error { background: color-mix(in srgb, var(--err) 12%, transparent); color: var(--err); }
    .activity-meta { color: var(--muted); font-size: 9px; font-variant-numeric: tabular-nums; line-height: 1.45; }
    .activity-entry.kind-error .activity-label { color: var(--err); }
    .activity-entry.kind-done .activity-label { color: var(--ok); }
    /* Expanded activity page: the right rail takes over the conversation area. */
    body.activity-expanded main {
      grid-template-columns: 250px minmax(0, 1fr);
    }
    body.activity-expanded main > section { display: none; }
    body.activity-expanded aside.details {
      display: flex;
      min-width: 0;
      overflow: hidden;
    }
    body.activity-expanded aside.details .stats { display: none; }
    body.activity-expanded .activity {
      flex: 1 1 auto;
      min-height: 0;
      overflow: hidden;
    }
    body.activity-expanded .activity-header {
      padding: 16px 20px 12px;
    }
    body.activity-expanded .activity-subtitle {
      max-width: 720px;
    }
    body.activity-expanded .activity-scroll {
      justify-content: flex-start;
      overflow-x: hidden;
      overflow-y: auto;
      padding: 16px 20px 20px;
      scrollbar-gutter: stable;
    }
    body.activity-expanded #activity-log {
      /* Let the log grow with its content so the scroll container can reach it. */
      flex: 0 0 auto;
      justify-content: flex-start;
      gap: 8px;
      min-height: auto;
      width: 100%;
    }
    body.activity-expanded .activity-entry {
      display: grid;
      grid-template-columns: 26px minmax(0, 1fr);
      flex: 0 0 auto;
      align-items: start;
      gap: 10px;
      width: var(--activity-width, 100%);
      max-width: 100%;
      min-width: 0;
      min-height: 0;
      margin-left: var(--activity-offset, 0px);
      padding: 10px 12px;
      overflow: visible;
      border: 1px solid color-mix(in srgb, var(--accent) 24%, var(--border));
      border-radius: 10px;
      background: color-mix(in srgb, var(--accent-soft) 58%, var(--panel));
      box-shadow: 0 1px 2px rgb(0 0 0 / 0.035);
    }
    body.activity-expanded .activity-entry::before { display: none; }
    body.activity-expanded .activity-scroll::before,
    body.activity-expanded .activity-scroll::after { display: none; }
    body.activity-expanded .activity-entry.kind-tool {
      border-color: color-mix(in srgb, var(--warn) 28%, var(--border));
      background: color-mix(in srgb, var(--warn) 8%, var(--panel));
    }
    body.activity-expanded .activity-entry.kind-done {
      border-color: color-mix(in srgb, var(--ok) 30%, var(--border));
      background: color-mix(in srgb, var(--ok) 8%, var(--panel));
    }
    body.activity-expanded .activity-entry.kind-error {
      border-color: color-mix(in srgb, var(--err) 30%, var(--border));
      background: color-mix(in srgb, var(--err) 8%, var(--panel));
    }
    body.activity-expanded .activity-marker {
      width: 24px;
      height: 24px;
      border-radius: 8px;
      font-size: 10px;
    }
    body.activity-expanded .activity-card {
      min-width: 0;
      padding-top: 1px;
    }
    body.activity-expanded .activity-row-head {
      align-items: baseline;
      gap: 8px;
      flex-wrap: wrap;
    }
    body.activity-expanded .activity-label {
      flex: 1 1 auto;
      font-size: 12px;
      white-space: normal;
    }
    body.activity-expanded .activity-time { font-size: 10px; }
    body.activity-expanded .activity-duration {
      display: inline-flex;
      align-items: center;
      font-size: 10px;
    }
    body.activity-expanded .activity-copy {
      margin-top: 5px;
      font-size: 12px;
    }
    body.activity-expanded .activity-copy code {
      overflow: visible;
      white-space: pre-wrap;
    }
    body.activity-expanded .activity-command-preview { display: none; }
    body.activity-expanded .activity-command-full { display: block; }
    body.activity-expanded .activity-output {
      display: block;
      max-height: 18rem;
      overflow: auto;
      font-size: 11px;
    }
    body.activity-expanded .activity-meta-row { margin-top: 7px; }
    body.activity-expanded .activity-overview { display: none; }
    body.activity-expanded .activity-legend { display: none; }
    #activity-open { display: none; }

    @media (max-width: 1100px) {
      main { grid-template-columns: 220px minmax(0, 1fr); }
      aside.details { display: none; }
      #activity-open { display: inline-flex; }
      body.activity-expanded main { grid-template-columns: 220px minmax(0, 1fr); }
      body.activity-expanded aside.details { display: flex; }
      body.activity-expanded #activity-open { display: none; }
    }
    @media (max-width: 700px) {
      main { grid-template-columns: 1fr; grid-template-rows: auto minmax(0, 1fr); }
      aside.conversations { max-height: 180px; }
      body.activity-expanded main { grid-template-columns: 1fr; grid-template-rows: minmax(0, 1fr); }
      body.activity-expanded aside.conversations { display: none; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand"><span class="brand-mark">m</span><span>mini-SWE-agent</span></div>
    <div class="header-meta">
      <span id="database-path" title="SQLite conversation database path"></span>
      <span id="model-label"></span>
    </div>
  </header>
  <main>
    <aside class="panel conversations" aria-label="Conversations">
      <div class="conversations-head">
        <h2>Conversation List</h2>
        <button id="new-conversation" class="secondary" type="button">New</button>
      </div>
      <div class="conversation-list" id="conversation-list">
        <div class="conversation-empty" id="conversation-empty">No conversations yet.</div>
      </div>
    </aside>
    <section>
      <div class="panel conversation-bar" aria-label="Conversation status">
        <div class="conversation-context">
          <span class="conversation-label" id="conversation-label">New conversation</span>
        </div>
        <div class="conversation-actions">
          <button id="activity-open" class="secondary" type="button" aria-controls="activity-panel" title="Open activity timeline">Activity</button>
          <span class="status" id="status" role="status" aria-live="polite">idle</span>
        </div>
      </div>
      <div class="feed-wrap">
        <div class="feed" id="feed">
          <div class="panel empty" id="empty">Responses stream here as the agent works.</div>
        </div>
        <div class="scroll-actions">
          <button id="scroll-top" class="secondary" type="button" title="Go to top" aria-label="Go to top">↑ Go to top</button>
          <button id="scroll-bottom" class="secondary" type="button" title="Go to end" aria-label="Go to end">Go to end ↓</button>
        </div>
      </div>
      <div class="panel composer">
        <div class="composer-attachments" id="composer-attachments" hidden></div>
        <div class="composer-input-row">
          <button id="attach-image" class="secondary composer-icon-button" type="button" title="Attach an image" aria-label="Attach an image">+</button>
          <textarea id="task" rows="1" placeholder="Describe the software engineering task. For example: find and fix the failing tests in this repository."></textarea>
          <div class="composer-action">
            <button id="send" type="button" title="Send message" aria-label="Send message">Send</button>
            <button id="stop" class="danger" type="button" title="Stop run" aria-label="Stop run" disabled hidden>Stop</button>
          </div>
          <input id="image-input" type="file" accept="image/*" multiple hidden>
        </div>
        <div class="composer-actions">
          <span class="conversation-workspace" id="conversation-workspace" title="Workspace directory"></span>
        </div>
      </div>
    </section>
    <aside class="details">
      <div class="panel stats">
        <h2>Run</h2>
        <div class="stat"><span>Step</span><span id="stat-step">-</span></div>
        <div class="stat"><span>Tool calls</span><span id="stat-tools">-</span></div>
        <div class="stat"><span>Cost</span><span id="stat-cost">$0.00</span></div>
        <div class="stat"><span>Tokens</span><span id="stat-tokens">-</span></div>
        <div class="stat"><span>Input / output</span><span id="stat-token-detail">-</span></div>
        <div class="stat"><span>Status</span><span id="stat-status">idle</span></div>
        <div class="stat"><span>Exit</span><span id="stat-exit">-</span></div>
      </div>
      <section class="panel activity" id="activity-panel" aria-label="Activity timeline">
        <div class="activity-header">
          <div class="activity-heading">
            <div>
              <h2>Activity</h2>
              <p class="activity-subtitle">Step-by-step trajectory of model calls, tool runs, and outcomes.</p>
            </div>
            <div class="activity-heading-actions">
              <button id="activity-expand" class="secondary" type="button" aria-controls="activity-panel" aria-expanded="false" title="Expand activity view">Expand</button>
            </div>
          </div>
          <div class="activity-summary" id="activity-summary" role="status" aria-live="polite">No activity yet</div>
          <div class="activity-overview" id="activity-overview" aria-label="Activity overview"></div>
          <div class="activity-legend" aria-hidden="true">
            <span class="activity-legend-item"><i class="activity-legend-dot"></i>Model step</span>
            <span class="activity-legend-item"><i class="activity-legend-dot tool"></i>Tool run</span>
            <span class="activity-legend-item"><i class="activity-legend-dot outcome"></i>Outcome</span>
            <span class="activity-legend-item"><i class="activity-legend-dot error"></i>Error</span>
          </div>
        </div>
        <div class="activity-scroll" id="activity-scroll">
          <ol id="activity-log">
            <li class="activity-empty" id="activity-empty">Start a run to see each model step, command, and outcome here in order.</li>
          </ol>
        </div>
      </section>
    </aside>
  </main>
  <script>
    const feed = document.getElementById("feed");
    const empty = document.getElementById("empty");
    const taskEl = document.getElementById("task");
    const composerEl = document.querySelector(".composer");
    const attachImageButton = document.getElementById("attach-image");
    const imageInput = document.getElementById("image-input");
    const composerAttachments = document.getElementById("composer-attachments");
    const sendButton = document.getElementById("send");
    const stopButton = document.getElementById("stop");
    const statusEl = document.getElementById("status");
    const conversationLabel = document.getElementById("conversation-label");
    const conversationWorkspace = document.getElementById("conversation-workspace");
    const statStep = document.getElementById("stat-step");
    const statCost = document.getElementById("stat-cost");
    const statTokens = document.getElementById("stat-tokens");
    const statTokenDetail = document.getElementById("stat-token-detail");
    const statStatus = document.getElementById("stat-status");
    const statExit = document.getElementById("stat-exit");
    const statTools = document.getElementById("stat-tools");
    const activityLog = document.getElementById("activity-log");
    const activityEmpty = document.getElementById("activity-empty");
    const activityScroller = document.getElementById("activity-scroll");
    const activitySummary = document.getElementById("activity-summary");
    const activityOverview = document.getElementById("activity-overview");
    const activityExpandButton = document.getElementById("activity-expand");
    const activityOpenButton = document.getElementById("activity-open");
    const conversationList = document.getElementById("conversation-list");
    const conversationEmpty = document.getElementById("conversation-empty");
    const CONVERSATION_STATES = ["running", "submitted", "stopped", "failed", "interrupted"];
    const newConversationButton = document.getElementById("new-conversation");
    const scrollTopButton = document.getElementById("scroll-top");
    const scrollBottomButton = document.getElementById("scroll-bottom");
    const SCROLL_THRESHOLD = 32;
    const MAX_RECENT_ACTIVITIES = 6;
    const SESSION_STORAGE_KEY = "mini-swe-agent-session";
    let eventSource = null;
    let activeSessionId = null;
    let selectedSessionId = null;
    let conversations = [];
    let defaultWorkspace = "";
    let currentAssistant = null;
    let currentReasoning = null;
    let pendingImages = [];
    const MAX_IMAGES = 8;
    const MAX_IMAGE_BYTES = 20 * 1024 * 1024;
    let followOutput = true;
    let currentTokenUsage = { prompt: 0, completion: 0, total: 0 };
    // Completed assistant messages are kept here until the immediately following
    // events have been inspected.  If one of those events is a tool call for the
    // same command, the command is removed from the assistant text so that the
    // tool-call row is the single place where the command is shown.
    let pendingAssistantMessage = null;
    let activityTimer = null;
    let activityLayoutFrame = null;
    let activityFollowOutput = true;
    let currentActivityStep = null;
    let currentActivityStepNumber = null;
    let activityLiveText = "";
    let activityReasoningText = "";
    let activityState = {
      steps: new Set(),
      tools: 0,
      errors: 0,
      segments: [],
      startedAt: 0,
      finishedAt: 0,
      finishedWithError: false,
    };
    let activityStepEntries = new Map();
    let activityToolEntries = new Map();

    function numeric(value) {
      const number = Number(value);
      return Number.isFinite(number) ? number : 0;
    }

    function formatTokens(value) {
      return numeric(value).toLocaleString();
    }

    function setTokenStats(prompt, completion, total) {
      const promptTokens = numeric(prompt);
      const completionTokens = numeric(completion);
      const totalTokens = numeric(total) || promptTokens + completionTokens;
      statTokens.textContent = totalTokens ? formatTokens(totalTokens) : "-";
      statTokenDetail.textContent = totalTokens
        ? formatTokens(promptTokens) + " in / " + formatTokens(completionTokens) + " out"
        : "-";
    }

    function setTokenStatsFromConversation(conversation) {
      const usage = (conversation && (conversation.token_usage || conversation.usage)) || {};
      currentTokenUsage = {
        prompt: numeric(conversation && (conversation.total_prompt_tokens ?? conversation.prompt_tokens ?? usage.prompt_tokens)),
        completion: numeric(conversation && (conversation.total_completion_tokens ?? conversation.completion_tokens ?? usage.completion_tokens)),
        total: numeric(conversation && (conversation.total_tokens_used ?? conversation.total_tokens ?? conversation.tokens ?? usage.total_tokens)),
      };
      setTokenStats(currentTokenUsage.prompt, currentTokenUsage.completion, currentTokenUsage.total);
    }

    function updateTokenStatsFromAssistant(event) {
      const usage = event.usage || {};
      const callPrompt = numeric(event.prompt_tokens ?? usage.prompt_tokens);
      const callCompletion = numeric(event.completion_tokens ?? usage.completion_tokens);
      const callTotal = numeric(event.total_tokens ?? event.tokens ?? usage.total_tokens) || callPrompt + callCompletion;
      if (event.cumulative_usage) {
        const prompt = numeric(event.cumulative_usage.prompt_tokens);
        const completion = numeric(event.cumulative_usage.completion_tokens);
        currentTokenUsage = {
          prompt,
          completion,
          total: numeric(event.cumulative_usage.total_tokens) || prompt + completion,
        };
      } else if (event.total_tokens_used !== undefined) {
        const prompt = numeric(event.total_prompt_tokens);
        const completion = numeric(event.total_completion_tokens);
        currentTokenUsage = {
          prompt,
          completion,
          total: numeric(event.total_tokens_used) || prompt + completion,
        };
      } else {
        currentTokenUsage.prompt += callPrompt;
        currentTokenUsage.completion += callCompletion;
        currentTokenUsage.total += callTotal;
      }
      setTokenStats(currentTokenUsage.prompt, currentTokenUsage.completion, currentTokenUsage.total);
      return callTotal;
    }

    function updateTokenStatsFromDone(event) {
      const usage = event.token_usage || event.cumulative_usage || event.usage || {};
      const prompt = numeric(event.total_prompt_tokens ?? event.prompt_tokens ?? usage.prompt_tokens ?? currentTokenUsage.prompt);
      const completion = numeric(event.total_completion_tokens ?? event.completion_tokens ?? usage.completion_tokens ?? currentTokenUsage.completion);
      currentTokenUsage = {
        prompt,
        completion,
        total: numeric(event.total_tokens_used ?? event.total_tokens ?? event.tokens ?? usage.total_tokens) || prompt + completion,
      };
      setTokenStats(currentTokenUsage.prompt, currentTokenUsage.completion, currentTokenUsage.total);
    }

    function setStatus(text, state = "") {
      statusEl.textContent = text;
      statusEl.className = "status " + state;
      statStatus.textContent = text;
    }

    function setComposerRunning(running) {
      sendButton.hidden = running;
      stopButton.hidden = !running;
      sendButton.disabled = running;
      stopButton.disabled = !running;
    }

    function resizeTaskInput() {
      if (!taskEl.value) {
        taskEl.style.height = "42px";
        taskEl.style.overflowY = "hidden";
        return;
      }
      taskEl.style.height = "auto";
      const height = Math.max(42, Math.min(taskEl.scrollHeight, 168));
      taskEl.style.height = height + "px";
      taskEl.style.overflowY = taskEl.scrollHeight > 168 ? "auto" : "hidden";
    }

    function isFeedAtBottom() {
      return feed.scrollHeight - feed.scrollTop - feed.clientHeight <= SCROLL_THRESHOLD;
    }

    function updateScrollButtons() {
      scrollTopButton.hidden = feed.scrollTop <= 4;
      scrollBottomButton.hidden = isFeedAtBottom();
    }

    function scrollFeedToBottom(behavior = "auto") {
      feed.scrollTo({ top: feed.scrollHeight, behavior });
    }

    function reasoningIsExpanded() {
      if (!currentReasoning) return false;
      const details = currentReasoning.closest("details");
      return Boolean(details && details.open);
    }

    function eventAffectsVisibleFeed(event) {
      // A collapsed reasoning block hides its streamed text, so appending more
      // deltas must not keep forcing the transcript to the bottom.
      return event.type !== "reasoning_delta" || reasoningIsExpanded();
    }

    feed.addEventListener("scroll", () => {
      followOutput = isFeedAtBottom();
      updateScrollButtons();
    }, { passive: true });

    scrollTopButton.addEventListener("click", () => {
      followOutput = false;
      feed.scrollTo({ top: 0, behavior: "smooth" });
    });

    scrollBottomButton.addEventListener("click", () => {
      followOutput = true;
      scrollFeedToBottom();
    });

    function formatClock(timestamp = Date.now()) {
      return new Date(timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    }

    function formatElapsed(milliseconds) {
      const totalSeconds = Math.max(0, Math.floor(numeric(milliseconds) / 1000));
      if (totalSeconds < 60) return totalSeconds + "s";
      const minutes = Math.floor(totalSeconds / 60);
      const seconds = totalSeconds % 60;
      if (minutes < 60) return minutes + "m " + String(seconds).padStart(2, "0") + "s";
      const hours = Math.floor(minutes / 60);
      return hours + "h " + String(minutes % 60).padStart(2, "0") + "m";
    }

    function previewText(value, maxLength = 180) {
      const text = String(value ?? "").replace(/\r/g, "").trim();
      if (!text) return "";
      const lines = text.split("\n").map((line) => line.trim()).filter(Boolean);
      const first = (lines[0] || "").replace(/\s+/g, " ");
      const suffix = lines.length > 1 ? "  +" + (lines.length - 1) + " more line" + (lines.length === 2 ? "" : "s") : "";
      const shortened = first.length > maxLength ? first.slice(0, maxLength - 1) + "…" : first;
      return shortened + suffix;
    }

    function activityStatusTone(status) {
      const normalized = String(status || "").toLowerCase();
      if (normalized === "running" || normalized === "thinking" || normalized === "waiting") return "running";
      if (normalized === "responded" || normalized.includes("done") || normalized.includes("complete") || normalized === "submitted" || normalized === "exit 0") return "ok";
      if (normalized.includes("error") || normalized.includes("fail") || (normalized.startsWith("exit ") && normalized !== "exit 0")) return "error";
      return "";
    }

    function activityMarker(kind) {
      if (kind === "tool") return "→";
      if (kind === "done") return "✓";
      if (kind === "error") return "!";
      if (kind === "run") return "▶";
      return "•";
    }

    function activityStatusIsFinished(status) {
      const normalized = String(status || "").toLowerCase();
      return normalized === "responded"
        || normalized === "submitted"
        || normalized === "stopped"
        || normalized === "failed"
        || normalized === "finished"
        || normalized.includes("complete")
        || normalized.includes("failed")
        || normalized.includes("done")
        || normalized.startsWith("exit ");
    }

    function activityIsExpanded() {
      return document.body.classList.contains("activity-expanded");
    }

    function formatActivityDuration(milliseconds) {
      const duration = Math.max(0, numeric(milliseconds));
      if (duration < 1000) return Math.round(duration) + "ms";
      return formatElapsed(duration);
    }

    function activityDuration(entry) {
      const startedAt = Number(entry.dataset.startedAt || 0);
      if (!startedAt) return 0;
      const endedAt = Number(entry.dataset.endedAt || 0) || Date.now();
      return Math.max(0, endedAt - startedAt);
    }

    function markActivityEnded(entry, timestamp = Date.now()) {
      if (entry && !entry.dataset.endedAt) entry.dataset.endedAt = String(timestamp);
    }

    function finalizeOpenActivities(timestamp = Date.now()) {
      activityLog.querySelectorAll(".activity-entry").forEach((entry) => {
        if (!entry.dataset.endedAt) markActivityEnded(entry, timestamp);
      });
    }

    function scheduleActivityLayout() {
      if (activityLayoutFrame != null) return;
      activityLayoutFrame = requestAnimationFrame(() => {
        activityLayoutFrame = null;
        layoutActivityEntries();
        if (activityIsExpanded() && activityFollowOutput) scrollActivityToBottom();
      });
    }

    function activityIsNearBottom() {
      const remaining = activityScroller.scrollHeight - activityScroller.scrollTop - activityScroller.clientHeight;
      return remaining <= 24;
    }

    function scrollActivityToBottom() {
      activityScroller.scrollTop = activityScroller.scrollHeight;
      activityFollowOutput = true;
    }

    function setActivitySpineState(hasEntries, clippedHistory) {
      activityScroller.classList.toggle("has-activity", hasEntries);
      activityScroller.classList.toggle("has-clipped-history", clippedHistory);
    }

    function layoutActivityEntries() {
      const entries = Array.from(activityLog.querySelectorAll(".activity-entry"));
      if (!entries.length) {
        setActivitySpineState(false, false);
        return;
      }
      if (!activityIsExpanded()) {
        const firstVisible = Math.max(0, entries.length - MAX_RECENT_ACTIVITIES);
        entries.forEach((entry, index) => { entry.hidden = index < firstVisible; });
        const candidates = entries.slice(firstVisible);
        candidates.forEach((entry) => { entry.hidden = false; });
        // clientHeight includes the scroller padding; don't count that as
        // space that can hold entries.
        const scrollerStyles = getComputedStyle(activityScroller);
        const verticalPadding = (parseFloat(scrollerStyles.paddingTop) || 0)
          + (parseFloat(scrollerStyles.paddingBottom) || 0);
        const available = Math.max(0, activityScroller.clientHeight - verticalPadding);
        let used = 0;
        let clippedHistory = firstVisible > 0;
        for (let index = candidates.length - 1; index >= 0; index -= 1) {
          const entry = candidates[index];
          const height = entry.getBoundingClientRect().height;
          if (used > 0 && used + height > available + 0.5) {
            entry.hidden = true;
            clippedHistory = true;
          } else {
            // Each entry already includes its own bottom padding; there is no
            // gap between entries to account for here.
            used += height;
          }
        }
        setActivitySpineState(candidates.some((entry) => !entry.hidden), clippedHistory);
        entries.forEach((entry) => {
          entry.style.removeProperty("--activity-offset");
          entry.style.removeProperty("--activity-width");
          const duration = entry.querySelector(".activity-duration");
          if (duration) duration.hidden = true;
        });
        return;
      }

      // Expanded activity is a readable, scrollable ledger. Keep the legacy
      // custom properties as layout hooks, but give every row the full width.
      setActivitySpineState(true, false);
      entries.forEach((entry) => {
        entry.hidden = false;
        const duration = activityDuration(entry);
        entry.style.setProperty("--activity-offset", "0px");
        entry.style.setProperty("--activity-width", "100%");
        const durationEl = entry.querySelector(".activity-duration");
        if (durationEl) {
          durationEl.hidden = false;
          durationEl.textContent = formatActivityDuration(duration);
        }
        const label = entry.querySelector(".activity-label");
        entry.title = (label ? label.textContent : "Activity") + " · " + formatActivityDuration(duration);
      });
    }

    function setActivityExpanded(expanded) {
      document.body.classList.toggle("activity-expanded", expanded);
      activityExpandButton.textContent = expanded ? "Collapse" : "Expand";
      activityExpandButton.setAttribute("aria-expanded", String(expanded));
      activityExpandButton.title = expanded ? "Collapse activity view" : "Expand activity view";
      if (expanded) activityFollowOutput = true;
      scheduleActivityLayout();
    }

    activityExpandButton.addEventListener("click", () => setActivityExpanded(!activityIsExpanded()));
    activityOpenButton.addEventListener("click", () => setActivityExpanded(true));
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && activityIsExpanded()) setActivityExpanded(false);
    });
    activityScroller.addEventListener("scroll", () => {
      if (activityIsExpanded()) activityFollowOutput = activityIsNearBottom();
    }, { passive: true });
    window.addEventListener("resize", scheduleActivityLayout, { passive: true });

    function drawActivityOverview() {
      activityOverview.replaceChildren();
      const segments = activityState.segments.slice(-64);
      if (!segments.length) {
        const empty = document.createElement("span");
        empty.className = "activity-overview-empty";
        empty.textContent = "Timeline appears here";
        activityOverview.appendChild(empty);
        return;
      }
      segments.forEach((segment) => {
        const bar = document.createElement("span");
        bar.className = "activity-segment kind-" + segment.kind;
        bar.title = segment.label + " · " + segment.time;
        activityOverview.appendChild(bar);
      });
    }

    function updateActivitySummary() {
      const parts = [];
      const stepCount = activityState.steps.size;
      if (stepCount) parts.push(stepCount + (stepCount === 1 ? " model step" : " model steps"));
      if (activityState.tools) parts.push(activityState.tools + (activityState.tools === 1 ? " tool run" : " tool runs"));
      if (activityState.errors) parts.push(activityState.errors + (activityState.errors === 1 ? " error" : " errors"));
      if (activityState.startedAt) {
        const elapsed = (activityState.finishedAt || Date.now()) - activityState.startedAt;
        if (elapsed >= 1000) parts.push(formatElapsed(elapsed));
      }
      activitySummary.textContent = parts.length ? parts.join(" · ") : "No activity yet";
      let className = "activity-summary";
      if (activityState.finishedWithError) className += " error";
      else if (activityState.finishedAt) className += " done";
      else if (activityState.startedAt) className += " running";
      activitySummary.className = className;
      statTools.textContent = activityState.tools ? String(activityState.tools) : "-";
    }

    function addActivitySegment(kind, label, time = formatClock()) {
      activityState.segments.push({ kind, label, time });
      if (activityState.segments.length > 96) activityState.segments.shift();
      drawActivityOverview();
    }

    function addActivityEntry(kind, title, options = {}) {
      activityEmpty.hidden = true;
      finalizeOpenActivities();
      const entry = document.createElement("li");
      entry.className = "activity-entry kind-" + kind;
      entry.dataset.startedAt = String(Date.now());
      if (options.status) entry.dataset.status = String(options.status).toLowerCase();

      const marker = document.createElement("span");
      marker.className = "activity-marker";
      marker.textContent = options.marker || activityMarker(kind);
      entry.appendChild(marker);

      const card = document.createElement("div");
      card.className = "activity-card";
      const head = document.createElement("div");
      head.className = "activity-row-head";
      const label = document.createElement("span");
      label.className = "activity-label";
      label.textContent = title;
      const time = document.createElement("span");
      time.className = "activity-time";
      time.textContent = options.time || formatClock();
      const duration = document.createElement("span");
      duration.className = "activity-duration";
      duration.hidden = true;
      head.appendChild(label);
      head.appendChild(time);
      head.appendChild(duration);
      card.appendChild(head);

      const copy = document.createElement("div");
      copy.className = "activity-copy";
      card.appendChild(copy);

      const metaRow = document.createElement("div");
      metaRow.className = "activity-meta-row";
      const status = document.createElement("span");
      status.className = "activity-status";
      status.hidden = !options.status;
      const meta = document.createElement("span");
      meta.className = "activity-meta";
      metaRow.appendChild(status);
      metaRow.appendChild(meta);
      card.appendChild(metaRow);
      entry.appendChild(card);

      activityLog.appendChild(entry);
      if (options.detail !== undefined || options.command !== undefined) {
        updateActivityEntry(entry, { detail: options.detail, command: options.command });
      }
      if (options.status) updateActivityEntry(entry, { status: options.status, meta: options.meta });
      else if (options.meta) updateActivityEntry(entry, { meta: options.meta });
      if (options.output) updateActivityEntry(entry, { output: options.output });
      if (options.segment !== false) addActivitySegment(kind, title, time.textContent);
      updateActivitySummary();
      scheduleActivityLayout();
      return entry;
    }

    function updateActivityEntry(entry, options = {}) {
      if (!entry) return;
      const label = entry.querySelector(".activity-label");
      const copy = entry.querySelector(".activity-copy");
      const status = entry.querySelector(".activity-status");
      const meta = entry.querySelector(".activity-meta");
      if (options.title !== undefined && label) label.textContent = options.title;
      if (options.detail !== undefined && copy) copy.textContent = options.detail;
      if (options.command !== undefined && copy) {
        const fullCommand = "$ " + String(options.command || "");
        copy.replaceChildren();
        const preview = document.createElement("code");
        preview.className = "activity-command-preview";
        preview.textContent = "$ " + commandPreview(options.command);
        preview.title = fullCommand;
        const full = document.createElement("code");
        full.className = "activity-command-full";
        full.textContent = fullCommand;
        copy.append(preview, full);
        entry.dataset.command = String(options.command || "");
      }
      if (options.status !== undefined && status) {
        status.hidden = !options.status;
        status.textContent = options.status;
        status.className = "activity-status " + activityStatusTone(options.status);
        entry.dataset.status = String(options.status || "").toLowerCase();
        if (activityStatusIsFinished(options.status)) markActivityEnded(entry);
      }
      if (options.meta !== undefined && meta) {
        meta.textContent = options.meta;
        meta.hidden = !options.meta;
      }
      if (options.output !== undefined) {
        let output = entry.querySelector(".activity-output");
        if (!output) {
          output = document.createElement("div");
          output.className = "activity-output";
          const label = document.createElement("span");
          label.className = "activity-output-label";
          label.textContent = "Output";
          const body = document.createElement("div");
          body.className = "activity-output-text";
          output.append(label, body);
          entry.querySelector(".activity-card").appendChild(output);
        }
        const body = output.querySelector(".activity-output-text");
        const text = String(options.output ?? "");
        const max = 20000;
        if (body) {
          body.textContent = text.length > max ? text.slice(0, max) + "\n\n... output truncated in GUI ..." : text || "(no output)";
        }
      }
      scheduleActivityLayout();
    }

    function addActivity(text, options = {}) {
      if (typeof options === "string") options = { detail: options };
      return addActivityEntry(options.kind || "info", text, options);
    }

    function markActivityStarted() {
      activityState.startedAt = Date.now();
      activityState.finishedAt = 0;
      activityState.finishedWithError = false;
      if (activityTimer) clearInterval(activityTimer);
      activityTimer = setInterval(() => {
        if (!activityState.finishedAt) {
          updateActivitySummary();
          scheduleActivityLayout();
        }
      }, 1000);
      updateActivitySummary();
    }

    function markActivityFinished(withError = false) {
      activityState.finishedAt = Date.now();
      activityState.finishedWithError = withError;
      if (activityTimer) {
        clearInterval(activityTimer);
        activityTimer = null;
      }
      updateActivitySummary();
    }

    function resetActivity() {
      activityLog.querySelectorAll(".activity-entry").forEach((entry) => entry.remove());
      activityEmpty.hidden = false;
      activityStepEntries.clear();
      activityToolEntries.clear();
      currentActivityStep = null;
      currentActivityStepNumber = null;
      activityLiveText = "";
      activityReasoningText = "";
      activityState = {
        steps: new Set(),
        tools: 0,
        errors: 0,
        segments: [],
        startedAt: 0,
        finishedAt: 0,
        finishedWithError: false,
      };
      if (activityTimer) {
        clearInterval(activityTimer);
        activityTimer = null;
      }
      activityScroller.scrollTop = 0;
      activityFollowOutput = true;
      setActivitySpineState(false, false);
      drawActivityOverview();
      updateActivitySummary();
      scheduleActivityLayout();
    }

    function addActivityStep(step) {
      let stepNumber = Number(step);
      if (!Number.isFinite(stepNumber) || stepNumber <= 0) stepNumber = activityState.steps.size + 1;
      if (activityStepEntries.has(stepNumber)) {
        currentActivityStep = activityStepEntries.get(stepNumber);
        currentActivityStepNumber = stepNumber;
        return currentActivityStep;
      }
      activityState.steps.add(stepNumber);
      const entry = addActivityEntry("step", "Step " + stepNumber + " · Planning", {
        marker: String(stepNumber),
        detail: "Model is deciding what to do next…",
        status: "thinking",
        meta: "step " + stepNumber,
      });
      entry.dataset.step = String(stepNumber);
      activityStepEntries.set(stepNumber, entry);
      currentActivityStep = entry;
      currentActivityStepNumber = stepNumber;
      updateActivitySummary();
      return entry;
    }

    function updateActivityStep(step, options = {}) {
      const stepNumber = Number(step);
      let entry = Number.isFinite(stepNumber) ? activityStepEntries.get(stepNumber) : currentActivityStep;
      if (!entry) entry = addActivityStep(step);
      updateActivityEntry(entry, options);
      return entry;
    }

    function addToolActivity(event) {
      const toolCallId = event.tool_call_id || "";
      if (toolCallId && activityToolEntries.has(toolCallId)) return activityToolEntries.get(toolCallId);
      const entry = addActivityEntry("tool", "Bash command", {
        marker: "→",
        command: event.command || "",
        status: "running",
        meta: currentActivityStepNumber ? "step " + currentActivityStepNumber : "",
      });
      entry.dataset.startedAt = String(Date.now());
      if (toolCallId) {
        entry.dataset.toolCallId = toolCallId;
        activityToolEntries.set(toolCallId, entry);
      }
      activityState.tools += 1;
      if (currentActivityStepNumber) {
        updateActivityStep(currentActivityStepNumber, { status: "running" });
      }
      updateActivitySummary();
      return entry;
    }

    function findToolActivity(event) {
      const toolCallId = event.tool_call_id || "";
      if (toolCallId && activityToolEntries.has(toolCallId)) return activityToolEntries.get(toolCallId);
      const running = activityLog.querySelectorAll(".activity-entry.kind-tool[data-status='running']");
      return running.length ? running[running.length - 1] : null;
    }

    function updateToolActivity(event) {
      let entry = findToolActivity(event);
      if (!entry) entry = addToolActivity(event);
      const returncode = event.returncode;
      const failed = returncode != null && Number(returncode) !== 0;
      const status = returncode == null ? "finished" : "exit " + returncode;
      const output = String(event.output || "");
      const startedAt = Number(entry.dataset.startedAt || 0);
      const elapsed = startedAt ? Date.now() - startedAt : 0;
      const timing = elapsed >= 100 ? " · " + formatElapsed(elapsed) : "";
      const meta = "exit " + (returncode ?? "?") + timing + " · " + formatTokens(output.length) + " chars";
      // Store the complete result on the activity entry.  CSS hides it in the
      // compact rail and reveals it when Activity Expand takes over the page.
      updateActivityEntry(entry, { status, meta, output });
      if (currentActivityStepNumber) {
        updateActivityStep(currentActivityStepNumber, { status: failed ? "tool failed" : "tool done" });
      }
      return entry;
    }

    function finalizeActivity(event) {
      const exitStatus = event.exit_status || "done";
      const successful = exitStatus === "Submitted";
      const stopped = exitStatus === "Stopped";
      const withError = !successful && !stopped;
      if (activityState.steps.size) {
        const lastStep = Math.max(...activityState.steps);
        updateActivityStep(lastStep, {
          title: "Step " + lastStep + " · " + (successful ? "Complete" : stopped ? "Stopped" : "Failed"),
          status: successful ? "complete" : stopped ? "stopped" : "failed",
        });
      }
      const meta = [
        event.n_calls != null ? event.n_calls + " calls" : "",
        event.cost != null ? "$" + Number(event.cost).toFixed(4) : "",
        (event.total_tokens ?? event.tokens) != null ? formatTokens(event.total_tokens ?? event.tokens) + " tokens" : "",
      ].filter(Boolean).join(" · ");
      const detail = event.submission
        ? "Submission: " + previewText(event.submission, 180)
        : "Exit status: " + exitStatus;
      addActivityEntry(withError ? "error" : "done", "Run finished · " + exitStatus, {
        marker: successful ? "✓" : stopped ? "■" : "!",
        detail,
        status: successful ? "submitted" : stopped ? "stopped" : "failed",
        meta,
      });
      finalizeOpenActivities();
      markActivityFinished(withError);
      scheduleActivityLayout();
    }

    function clearFeed() {
      feed.innerHTML = "";
      currentAssistant = null;
      currentReasoning = null;
      pendingAssistantMessage = null;
      followOutput = true;
      updateScrollButtons();
    }

    function addRunMarker(kind, label, meta = "") {
      const marker = document.createElement("div");
      marker.className = "run-marker " + kind;
      const text = document.createElement("span");
      text.className = "run-marker-label";
      text.textContent = label;
      marker.appendChild(text);
      if (meta) {
        const detail = document.createElement("span");
        detail.className = "run-marker-meta";
        detail.textContent = meta;
        marker.appendChild(detail);
      }
      feed.appendChild(marker);
      return marker;
    }

    function commandPreview(command) {
      const lines = String(command || "").split(/\r?\n/);
      const nonEmpty = lines.filter((line) => line.trim().length > 0);
      const first = (nonEmpty[0] || "").trim().replace(/\s+/g, " ");
      const remaining = Math.max(0, nonEmpty.length - 1);
      return (first || "(empty command)") + (remaining ? "  +" + remaining + " more line" + (remaining === 1 ? "" : "s") : "");
    }

    function openToolActivity(toolCallId) {
      setActivityExpanded(true);
      const entry = toolCallId ? activityToolEntries.get(String(toolCallId)) : null;
      if (!entry) return;
      // Wait for the expanded layout pass, then reveal and focus the matching
      // activity entry.  This is the only place command details are shown.
      requestAnimationFrame(() => requestAnimationFrame(() => {
        entry.hidden = false;
        entry.scrollIntoView({ block: "center", behavior: "smooth" });
        entry.classList.add("activity-highlight");
        setTimeout(() => entry.classList.remove("activity-highlight"), 1800);
      }));
    }

    function makeToolCall(command, toolCallId = "") {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "tool-call";
      row.dataset.command = command || "";
      if (toolCallId) row.dataset.toolCallId = String(toolCallId);
      row.title = "Open command details in Activity";
      row.setAttribute("aria-label", "Open command details in Activity: " + commandPreview(command));

      const icon = document.createElement("span");
      icon.className = "tool-call-icon";
      icon.textContent = "\u25b8";
      const prompt = document.createElement("span");
      prompt.className = "tool-call-prompt";
      prompt.textContent = "$";
      const text = document.createElement("span");
      text.className = "tool-call-command";
      text.textContent = commandPreview(command);
      const status = document.createElement("span");
      status.className = "tool-call-status running";
      status.textContent = "running";

      row.append(icon, prompt, text, status);
      row.addEventListener("click", () => openToolActivity(toolCallId));
      feed.appendChild(row);
      return row;
    }

    function findToolCall(event) {
      const rows = Array.from(feed.querySelectorAll(".tool-call"));
      const toolCallId = event.tool_call_id == null ? "" : String(event.tool_call_id);
      if (toolCallId) {
        for (let i = rows.length - 1; i >= 0; i--) {
          if (rows[i].dataset.toolCallId === toolCallId) return rows[i];
        }
      }
      const command = event.command || "";
      for (let i = rows.length - 1; i >= 0; i--) {
        if (rows[i].dataset.command === command) return rows[i];
      }
      return null;
    }

    function updateToolCall(event) {
      let row = findToolCall(event);
      if (!row) row = makeToolCall(event.command || "", event.tool_call_id);
      const status = row.querySelector(".tool-call-status");
      if (!status) return row;
      if (event.returncode == null) {
        status.textContent = "done";
        status.className = "tool-call-status";
      } else {
        status.textContent = "exit " + event.returncode;
        status.className = "tool-call-status " + (Number(event.returncode) === 0 ? "ok" : "error");
      }
      return row;
    }

    function makeCard(kind, role, meta = "") {
      const card = document.createElement("div");
      card.className = "card " + kind;
      if (kind === "assistant") {
        // Assistant text is part of the transcript, not a widget.  Model
        // metadata (step, cost, tokens) is kept in the Activity ledger.
        card.setAttribute("aria-label", role || "Agent");
      } else {
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
      }
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

    function addUserCard(task, images) {
      const card = makeCard("user", "Task");
      appendText(card, task);
      const imageUrls = Array.isArray(images) ? images.filter(Boolean) : [];
      if (imageUrls.length) {
        const gallery = document.createElement("div");
        gallery.className = "card-images";
        imageUrls.forEach((url) => {
          const link = document.createElement("a");
          link.href = String(url);
          link.target = "_blank";
          link.rel = "noopener noreferrer";
          const img = document.createElement("img");
          img.src = String(url);
          img.alt = "Attached image";
          link.appendChild(img);
          gallery.appendChild(link);
        });
        card.appendChild(gallery);
      }
      return card;
    }

    function renderPendingImages() {
      composerAttachments.replaceChildren();
      composerAttachments.hidden = pendingImages.length === 0;
      pendingImages.forEach((url, index) => {
        const item = document.createElement("div");
        item.className = "composer-attachment";
        const img = document.createElement("img");
        img.src = url;
        img.alt = "Attached image " + (index + 1);
        const remove = document.createElement("button");
        remove.type = "button";
        remove.textContent = "\u00d7";
        remove.title = "Remove image";
        remove.setAttribute("aria-label", "Remove image " + (index + 1));
        remove.addEventListener("click", () => {
          pendingImages.splice(index, 1);
          renderPendingImages();
        });
        item.appendChild(img);
        item.appendChild(remove);
        composerAttachments.appendChild(item);
      });
    }

    function readImageFile(file) {
      return new Promise((resolve, reject) => {
        if (!file.type || !file.type.startsWith("image/")) {
          reject(new Error("Only image files can be attached."));
          return;
        }
        if (file.size > MAX_IMAGE_BYTES) {
          reject(new Error("Image " + file.name + " is too large; the limit is 20 MB."));
          return;
        }
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = () => reject(new Error("Could not read image " + file.name + "."));
        reader.readAsDataURL(file);
      });
    }

    async function addImageFiles(files) {
      const selected = Array.from(files || []).filter((file) => file && file.type && file.type.startsWith("image/"));
      if (!selected.length) return;
      const room = MAX_IMAGES - pendingImages.length;
      if (room <= 0) {
        setStatus("image limit reached", "error");
        return;
      }
      for (const file of selected.slice(0, room)) {
        try {
          pendingImages.push(await readImageFile(file));
        } catch (err) {
          setStatus(String(err.message || err), "error");
        }
      }
      renderPendingImages();
      taskEl.focus();
    }

    function ensureAssistantCard() {
      if (!currentAssistant) {
        currentAssistant = makeCard("assistant", "mini-SWE-agent");
      }
      return currentAssistant;
    }

    function normalizeCommandForComparison(value) {
      return String(value == null ? "" : value)
        .replace(/\r\n?/g, "\n")
        .split("\n")
        .map((line) => line.replace(/[ \t]+$/g, ""))
        .join("\n")
        .trim()
        .replace(/^\$\s+/, "");
    }

    function commandTextMatches(candidate, target) {
      return Boolean(target) && normalizeCommandForComparison(candidate) === target;
    }

    function tidyAssistantContent(content) {
      return String(content == null ? "" : content)
        .replace(/\r\n?/g, "\n")
        .split("\n")
        .map((line) => line.replace(/[ \t]+$/g, ""))
        .join("\n")
        .replace(/\n{3,}/g, "\n\n")
        .replace(/[ \t]{2,}/g, " ")
        .replace(/[ \t]+([.,!?;:])/g, "$1")
        .trim();
    }

    /**
     * Remove a command from an assistant message while preserving its prose.
     *
     * Models frequently repeat the command they are about to execute in their
     * textual response (often inside a fenced code block).  The tool-call row
     * is the authoritative representation of that command, so we remove matching
     * command fragments from the assistant text and leave only the explanation.
     */
    function stripCommandFromAssistantContent(content, command) {
      const original = String(content == null ? "" : content);
      const target = normalizeCommandForComparison(command);
      const result = { content: original, removed: false };
      if (!target) return result;

      let text = original;

      // Fenced code blocks can contain the command as their whole body or as one
      // line among surrounding comments/examples.
      text = text.replace(/```([^\n`]*)\r?\n?([\s\S]*?)```/g, (match, language, body) => {
        if (commandTextMatches(body, target)) {
          result.removed = true;
          return "";
        }
        const lines = body.split(/\r?\n/);
        const kept = lines.filter((line) => {
          if (commandTextMatches(line, target)) {
            result.removed = true;
            return false;
          }
          return true;
        });
        if (kept.length === lines.length) return match;
        const keptBody = kept.join("\n").trim();
        return keptBody ? "```" + language + "\n" + keptBody + "\n```" : "";
      });

      // Inline code spans are common in explanations such as
      // "I will run `sed -n ...` to inspect the file".
      text = text.replace(/`([^`\n]+)`/g, (match, body) => {
        if (commandTextMatches(body, target)) {
          result.removed = true;
          return "";
        }
        return match;
      });

      // Also handle plain command-only lines outside fenced blocks.
      const lines = text.split("\n");
      const keptLines = lines.filter((line) => {
        if (commandTextMatches(line, target)) {
          result.removed = true;
          return false;
        }
        return true;
      });
      text = keptLines.join("\n");

      // Finally remove raw occurrences embedded in prose.  Keep replacing until
      // every duplicate has been removed.
      let occurrence = text.indexOf(target);
      while (occurrence !== -1) {
        result.removed = true;
        text = text.slice(0, occurrence) + text.slice(occurrence + target.length);
        occurrence = text.indexOf(target);
      }

      result.content = result.removed ? tidyAssistantContent(text) : original;
      return result;
    }

    function hasVisibleReasoning(card) {
      const reasoning = card && card.querySelector(".reasoning .reasoning-body");
      return Boolean(reasoning && reasoning.textContent.trim());
    }

    function mergeMatchingCommandIntoPendingAssistant(command) {
      if (!pendingAssistantMessage || !pendingAssistantMessage.card) return false;
      const card = pendingAssistantMessage.card;
      if (!card.isConnected) {
        pendingAssistantMessage = null;
        return false;
      }
      const contentEl = card.querySelector(".content");
      if (!contentEl) return false;
      const merged = stripCommandFromAssistantContent(contentEl.textContent, command);
      if (!merged.removed) return false;

      contentEl.textContent = merged.content;
      pendingAssistantMessage.content = merged.content;
      card.classList.add("command-merged");
      if (!merged.content && !hasVisibleReasoning(card)) {
        // The assistant message contained nothing but the command.  Show only
        // the tool-call row, which is the preferred representation.
        card.hidden = true;
        card.classList.add("command-only");
      }
      return true;
    }

    function flushPendingAssistantMessage() {
      // Associate a completed assistant message with the next tool calls.  Once
      // a non-adjacent event arrives, no further command de-duplication applies.
      pendingAssistantMessage = null;
    }

    function lastVisibleAssistantMessageContent() {
      const cards = feed.querySelectorAll(".card.assistant:not([hidden])");
      for (let i = cards.length - 1; i >= 0; i--) {
        const content = cards[i].querySelector(".content");
        const text = content ? content.textContent.trim() : "";
        if (text) return text;
      }
      return "";
    }

    function isRepeatedAssistantSubmission(submission) {
      // The agent emits the final assistant message and then repeats it as the
      // submission in the done event.  Render that answer only once.
      const text = String(submission == null ? "" : submission).trim();
      return Boolean(text) && lastVisibleAssistantMessageContent() === text;
    }

    function queueAssistantMessageForMerge(card, content, step) {
      const contentEl = card && card.querySelector(".content");
      pendingAssistantMessage = {
        card,
        content: contentEl ? contentEl.textContent : String(content == null ? "" : content),
        step,
      };
    }

    function conversationId(conversation) {
      return conversation.session_id || conversation.id || "";
    }

    async function copyConversationId(id) {
      if (!id) return;
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(id);
          return;
        }
        throw new Error("Clipboard API unavailable");
      } catch (error) {
        const textarea = document.createElement("textarea");
        textarea.value = id;
        textarea.setAttribute("readonly", "");
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.select();
        const copied = document.execCommand("copy");
        textarea.remove();
        if (!copied) console.error("Failed to copy conversation ID", error);
      }
    }

    function conversationWorkspacePath(conversation) {
      return (conversation && (conversation.workspace || conversation.cwd || conversation.working_directory)) || defaultWorkspace || "";
    }

    function setConversationWorkspace(conversation) {
      const workspace = conversationWorkspacePath(conversation);
      conversationWorkspace.textContent = workspace;
      conversationWorkspace.title = workspace;
    }

    function rememberSession(sessionId) {
      if (sessionId) {
        sessionStorage.setItem(SESSION_STORAGE_KEY, sessionId);
        history.replaceState(null, "", "#" + encodeURIComponent(sessionId));
      } else {
        sessionStorage.removeItem(SESSION_STORAGE_KEY);
        history.replaceState(null, "", location.pathname + location.search);
      }
    }

    function rememberedSessionId() {
      return decodeURIComponent(location.hash.replace(/^#/, "")) || sessionStorage.getItem(SESSION_STORAGE_KEY) || "";
    }

    function conversationStatus(conversation) {
      if (!conversation) return "idle";
      return conversation.status || (conversation.done ? (conversation.exit_status || "done") : "running");
    }

    function conversationStateFromStatus(status, done) {
      if (!done) return "running";
      const normalized = String(status || "").trim().toLowerCase();
      if (["submitted", "complete", "completed", "success", "succeeded"].includes(normalized)) return "submitted";
      if (["stopped", "stop", "cancelled", "canceled"].includes(normalized)) return "stopped";
      if (["interrupted", "interrupt"].includes(normalized)) return "interrupted";
      return "failed";
    }

    function conversationState(conversation) {
      if (!conversation) return "";
      if (CONVERSATION_STATES.includes(conversation.state)) return conversation.state;
      return conversationStateFromStatus(conversationStatus(conversation), Boolean(conversation.done));
    }

    function applyConversationStatus(conversation) {
      if (!conversation) return;
      const status = conversationStatus(conversation);
      setStatus(status, statusClass(conversationState(conversation)));
      const running = Boolean(!conversation.done && String(status).toLowerCase() === "running");
      setComposerRunning(running);
    }

    function statusClass(state) {
      return CONVERSATION_STATES.includes(state) ? state : "";
    }

    function formatConversationTime(timestamp) {
      if (!timestamp) return "";
      return new Date(Number(timestamp) * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }

    function renderConversationList() {
      conversationList.querySelectorAll(".conversation-item").forEach((item) => item.remove());
      conversationEmpty.hidden = conversations.length > 0;
      for (const conversation of conversations) {
        const id = conversationId(conversation);
        if (!id) continue;
        const button = document.createElement("button");
        button.type = "button";
        button.className = "conversation-item" + (id === selectedSessionId ? " active" : "");
        button.dataset.sessionId = id;
        const workspacePath = conversationWorkspacePath(conversation);
        button.title = [
          conversation.task || "",
          "Click to copy full ID: " + id,
          workspacePath,
        ].filter(Boolean).join("\n");
        button.addEventListener("click", () => selectConversation(id));

        const titleRow = document.createElement("span");
        titleRow.className = "conversation-title-row";
        const title = document.createElement("span");
        title.className = "conversation-title";
        title.textContent = conversation.task || "Untitled conversation";
        titleRow.appendChild(title);
        if (workspacePath) {
          const workspace = document.createElement("span");
          workspace.className = "conversation-workspace";
          workspace.textContent = workspacePath;
          workspace.title = workspacePath;
          titleRow.appendChild(workspace);
        }
        button.appendChild(titleRow);

        const idRow = document.createElement("span");
        idRow.className = "conversation-id-row";
        const idValue = document.createElement("span");
        idValue.className = "conversation-id";
        idValue.textContent = id.slice(0, 8);
        idValue.dataset.fullId = id;
        idValue.title = "Click to copy full conversation ID: " + id;
        idValue.addEventListener("click", (event) => {
          event.preventDefault();
          event.stopPropagation();
          copyConversationId(id);
        });
        idRow.appendChild(idValue);
        button.appendChild(idRow);

        const meta = document.createElement("span");
        meta.className = "conversation-meta";
        const time = document.createElement("span");
        time.textContent = formatConversationTime(conversation.created_at);
        const status = document.createElement("span");
        status.className = "conversation-status " + statusClass(conversationState(conversation));
        status.textContent = conversationStatus(conversation);
        meta.appendChild(time);
        meta.appendChild(status);
        button.appendChild(meta);
        conversationList.appendChild(button);
      }
    }

    function resetRunView(conversation) {
      clearFeed();
      resetActivity();
      currentAssistant = null;
      currentReasoning = null;
      conversationLabel.textContent = (conversation && conversation.task) || (conversation && conversationId(conversation) ? "Untitled conversation" : "New conversation");
      setConversationWorkspace(conversation);
      statStep.textContent = conversation && conversation.n_calls ? String(conversation.n_calls) : "-";
      statCost.textContent = "$" + Number((conversation && conversation.cost) || 0).toFixed(4);
      setTokenStatsFromConversation(conversation);
      statExit.textContent = (conversation && conversation.exit_status) || "-";
      applyConversationStatus(conversation);
    }

    function updateConversationStatus(sessionId, changes) {
      const conversation = conversations.find((item) => conversationId(item) === sessionId);
      if (!conversation) return;
      Object.assign(conversation, changes);
      renderConversationList();
    }

    async function loadConversations({ selectNewest = false } = {}) {
      try {
        const response = await fetch("/api/conversations", { cache: "no-store" });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || ("HTTP " + response.status));
        conversations = Array.isArray(data) ? data : (data.conversations || data.sessions || data.items || []);
        renderConversationList();
        if (selectedSessionId) {
          const selectedConversation = conversations.find((item) => conversationId(item) === selectedSessionId);
          if (selectedConversation && selectedConversation.done) applyConversationStatus(selectedConversation);
        }
        if (selectNewest && !selectedSessionId && conversations.length) {
          await selectConversation(conversationId(conversations[0]));
        }
      } catch (err) {
        console.error("Could not load conversations", err);
      }
    }

    async function selectConversation(sessionId) {
      if (!sessionId) return;
      if (sessionId === selectedSessionId && eventSource) return;
      closeEventStream();
      selectedSessionId = sessionId;
      rememberSession(sessionId);
      const conversation = conversations.find((item) => conversationId(item) === sessionId);
      renderConversationList();
      resetRunView(conversation || { session_id: sessionId });
      if (!conversation) setStatus("loading", "running");
      connectEvents(sessionId);
    }

    function handleEvent(event) {
      const shouldFollowOutput = followOutput;
      // Assistant messages are buffered for one adjacent tool-call window.  A
      // matching action removes the repeated command from the assistant text;
      // any other event simply ends the merge window and keeps the message.
      const keepsAssistantMergeWindow = ["tool_call_delta", "action_start", "action_output", "observation"];
      if (pendingAssistantMessage && !keepsAssistantMergeWindow.includes(event.type)) {
        flushPendingAssistantMessage();
      }
      switch (event.type) {
        case "run_start":
          addRunMarker("start", "Run started", event.step_limit ? "limit " + event.step_limit + " steps" : "");
          addUserCard(event.task || "", event.images || []);
          setStatus("running", "running");
          setComposerRunning(true);
          markActivityStarted();
          addActivityEntry("run", "Run started", {
            marker: "▶",
            detail: event.task ? previewText(event.task, 180) : "Agent run accepted.",
            status: "running",
            segment: true,
          });
          break;
        case "step_start":
          statStep.textContent = String(event.step || "-");
          setStatus("running", "running");
          addActivityStep(event.step);
          break;
        case "assistant_start":
          activityLiveText = "";
          activityReasoningText = "";
          ensureAssistantCard();
          if (currentActivityStepNumber) {
            updateActivityStep(currentActivityStepNumber, {
              title: "Step " + currentActivityStepNumber + " · Thinking",
              detail: "Receiving the model response…",
              status: "thinking",
            });
          }
          break;
        case "assistant_delta": {
          const card = ensureAssistantCard();
          activityLiveText += event.text || "";
          appendText(card, event.text || "");
          if (activityLiveText.trim()) {
            updateActivityStep(event.step || currentActivityStepNumber, {
              title: "Step " + (event.step || currentActivityStepNumber || "?") + " · Writing response",
              detail: previewText(activityLiveText, 220),
              status: "thinking",
            });
          }
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
          activityReasoningText += event.text || "";
          if (!activityLiveText.trim() && activityReasoningText.trim()) {
            updateActivityStep(event.step || currentActivityStepNumber, {
              title: "Step " + (event.step || currentActivityStepNumber || "?") + " · Reasoning",
              detail: previewText(activityReasoningText, 220),
              status: "thinking",
            });
          }
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
          const callTokens = updateTokenStatsFromAssistant(event);
          const currentCost = Number(statCost.textContent.replace("$", "")) || 0;
          statCost.textContent = "$" + (currentCost + Number(event.cost || 0)).toFixed(4);
          const actions = Array.isArray(event.actions) ? event.actions : [];
          const activityTitle = "Step " + (event.step || currentActivityStepNumber || "?") + " · "
            + (actions.length ? "Command" + (actions.length === 1 ? "" : "s") + " requested" : "Model response");
          const activityDetail = previewText(event.content, 220)
            || (actions.length ? actions.length + " command" + (actions.length === 1 ? "" : "s") + " queued." : "(empty response)");
          const activityMeta = [
            callTokens ? formatTokens(callTokens) + " tokens" : "",
            event.total_tokens_used != null ? formatTokens(event.total_tokens_used) + " cumulative" : "",
            "$" + Number(event.cost || 0).toFixed(4),
          ].filter(Boolean).join(" · ");
          updateActivityStep(event.step || currentActivityStepNumber, {
            title: activityTitle,
            detail: activityDetail,
            status: "responded",
            meta: activityMeta,
          });
          updateConversationStatus(activeSessionId, {
            prompt_tokens: currentTokenUsage.prompt,
            completion_tokens: currentTokenUsage.completion,
            total_tokens: currentTokenUsage.total,
            token_usage: {
              prompt_tokens: currentTokenUsage.prompt,
              completion_tokens: currentTokenUsage.completion,
              total_tokens: currentTokenUsage.total,
            },
          });
          queueAssistantMessageForMerge(card, event.content || "", event.step);
          currentAssistant = null;
          currentReasoning = null;
          break;
        }
        case "action_start": {
          // A tool call is the source of truth for an executed command.  Remove
          // the same command from the adjacent assistant message, then show one
          // quiet line in the transcript.  Full details stay in Activity.
          mergeMatchingCommandIntoPendingAssistant(event.command || "");
          makeToolCall(event.command || "", event.tool_call_id);
          addToolActivity(event);
          break;
        }
        case "action_output": {
          updateToolCall(event);
          updateToolActivity(event);
          break;
        }
        case "observation":
          break;
        case "format_error": {
          currentAssistant = null;
          currentReasoning = null;
          activityLiveText = "";
          activityReasoningText = "";
          updateTokenStatsFromAssistant(event);
          activityState.errors += 1;
          const card = makeCard("error", "Format error", "retrying");
          appendText(card, event.message || "");
          if (event.step || currentActivityStepNumber) {
            updateActivityStep(event.step || currentActivityStepNumber, {
              title: "Step " + (event.step || currentActivityStepNumber) + " · Format error",
              detail: previewText(event.message, 220) || "The model response could not be parsed.",
              status: "retrying",
            });
          }
          addActivityEntry("error", "Model response format error", {
            detail: previewText(event.message, 220) || "The response could not be parsed; retrying.",
            status: "retrying",
            meta: event.consecutive_errors ? event.consecutive_errors + " consecutive error(s)" : "",
          });
          updateActivitySummary();
          break;
        }
        case "error": {
          currentAssistant = null;
          currentReasoning = null;
          activityLiveText = "";
          activityReasoningText = "";
          activityState.errors += 1;
          const card = makeCard("error", "Error");
          appendText(card, event.message || "Unknown error");
          setStatus("error", "error");
          if (currentActivityStepNumber) {
            updateActivityStep(currentActivityStepNumber, {
              title: "Step " + currentActivityStepNumber + " · Error",
              detail: previewText(event.message, 220) || "The agent reported an error.",
              status: "failed",
            });
          }
          addActivityEntry("error", "Agent error", {
            detail: previewText(event.message, 220) || "The agent reported an error.",
            status: "failed",
          });
          updateActivitySummary();
          break;
        }
        case "done": {
          statExit.textContent = event.exit_status || "-";
          statCost.textContent = "$" + Number(event.cost || 0).toFixed(4);
          statStep.textContent = String(event.n_calls ?? statStep.textContent);
          updateTokenStatsFromDone(event);
          const exitStatus = event.exit_status || "done";
          const doneState = conversationStateFromStatus(exitStatus, true);
          const headerState = doneState === "failed" ? "error" : doneState;
          setStatus(exitStatus, headerState);
          if (event.submission && !isRepeatedAssistantSubmission(event.submission)) {
            const card = makeCard("assistant", "Submission");
            appendText(card, event.submission);
          }
          addRunMarker("end", "Run finished", event.exit_status || "done");
          finalizeActivity(event);
          updateConversationStatus(activeSessionId, {
            done: true,
            status: exitStatus,
            state: doneState,
            exit_status: event.exit_status || "",
            cost: Number(event.cost || 0),
            n_calls: Number(event.n_calls || 0),
            prompt_tokens: currentTokenUsage.prompt,
            completion_tokens: currentTokenUsage.completion,
            total_tokens: currentTokenUsage.total,
            token_usage: {
              prompt_tokens: currentTokenUsage.prompt,
              completion_tokens: currentTokenUsage.completion,
              total_tokens: currentTokenUsage.total,
            },
          });
          closeEventStream();
          setComposerRunning(false);
          setTimeout(() => loadConversations(), 250);
          break;
        }
      }
      if (shouldFollowOutput && eventAffectsVisibleFeed(event)) scrollFeedToBottom();
      updateScrollButtons();
    }

    function closeEventStream() {
      if (eventSource) {
        eventSource.close();
        eventSource = null;
      }
      activeSessionId = null;
    }

    function connectEvents(sessionId, eventOffset = 0) {
      closeEventStream();
      activeSessionId = sessionId;
      const suffix = eventOffset > 0 ? "?since=" + encodeURIComponent(eventOffset) : "";
      const source = new EventSource("/api/events/" + encodeURIComponent(sessionId) + suffix);
      eventSource = source;
      source.onmessage = (message) => {
        if (eventSource !== source) return;
        try { handleEvent(JSON.parse(message.data)); }
        catch (err) { console.error("Bad event", err, message.data); }
      };
      source.onerror = () => {
        if (eventSource !== source) return;
        closeEventStream();
        const conversation = conversations.find((item) => conversationId(item) === sessionId);
        if (conversation && !conversation.done) {
          setStatus("disconnected", "error");
          addActivityEntry("error", "Connection lost", {
            detail: "The event stream disconnected before the run finished. Reconnecting reloads the conversation.",
            status: "disconnected",
          });
          activityState.errors += 1;
          markActivityFinished(true);
        } else if (conversation) {
          // Replaying historical events can leave the header at "running" when
          // the stored run ended without a terminal event (for example after a
          // server restart). The summary is authoritative for finished runs.
          applyConversationStatus(conversation);
        }
        loadConversations();
      };
    }

    async function startRun() {
      const images = pendingImages.slice();
      let task = taskEl.value.trim();
      if (!task && !images.length) {
        taskEl.focus();
        return;
      }
      if (!task) task = "Please analyze the attached image.";
      const conversation = selectedSessionId
        ? conversations.find((item) => conversationId(item) === selectedSessionId)
        : null;
      // A finished or stopped conversation accepts follow-up messages.  Keep
      // the transcript on screen and let the server append the next turn to the
      // same session instead of creating a new conversation.
      const continuing = Boolean(conversation && conversation.done);
      setComposerRunning(true);
      closeEventStream();
      if (!continuing) {
        conversationLabel.textContent = task;
        clearFeed();
        resetActivity();
        statStep.textContent = "-";
        statCost.textContent = "$0.0000";
        currentTokenUsage = { prompt: 0, completion: 0, total: 0 };
        setTokenStats(0, 0, 0);
        statExit.textContent = "-";
      }
      setStatus("starting", "running");
      stopButton.disabled = true;
      try {
        const payload = { task, images };
        if (continuing) payload.session_id = selectedSessionId;
        const response = await fetch("/api/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || ("HTTP " + response.status));
        taskEl.value = "";
        resizeTaskInput();
        pendingImages = [];
        renderPendingImages();
        imageInput.value = "";
        taskEl.focus();
        const sessionId = data.session_id;
        selectedSessionId = sessionId;
        rememberSession(sessionId);
        if (data.conversation) {
          conversations = conversations.filter((item) => conversationId(item) !== sessionId);
          conversations.unshift(data.conversation);
          setConversationWorkspace(data.conversation);
        } else {
          const newConversation = { session_id: sessionId, task, status: "running", state: "running", created_at: Date.now() / 1000, image_count: images.length };
          conversations.unshift(newConversation);
        }
        renderConversationList();
        setComposerRunning(true);
        connectEvents(sessionId, Number(data.event_offset || 0));
        setTimeout(() => loadConversations(), 250);
      } catch (err) {
        setComposerRunning(false);
        setStatus("error", "error");
        const card = makeCard("error", "Error");
        appendText(card, String(err));
        addActivityEntry("error", "Could not start run", {
          detail: String(err),
          status: "failed",
        });
      }
    }

    async function stopRun() {
      // Prefer the currently connected session, but fall back to the selected
      // conversation if the event stream has disconnected.  Otherwise a
      // transient SSE error makes the Stop button silently do nothing.
      const sessionId = activeSessionId || selectedSessionId;
      if (!sessionId) {
        setComposerRunning(false);
        return;
      }
      stopButton.disabled = true;
      setStatus("stopping", "running");
      addActivity("Stopping run…", { kind: "info", status: "stopping", segment: false });
      try {
        const response = await fetch("/api/stop/" + encodeURIComponent(sessionId), { method: "POST" });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || ("HTTP " + response.status));
        // Switch back to Send immediately, even if the SSE connection has
        // already dropped.  The server waits for the stopped run to finish
        // before starting a follow-up sent to this same conversation.
        updateConversationStatus(sessionId, {
          done: true,
          status: "Stopped",
          state: "stopped",
          exit_status: "Stopped",
        });
        setStatus("Stopped", "stopped");
        setComposerRunning(false);
        taskEl.focus();
      } catch (err) {
        stopButton.disabled = false;
        setStatus("running", "running");
        addActivity("Could not stop run", { kind: "error", detail: String(err), status: "failed" });
      }
    }

    function startNewConversation() {
      closeEventStream();
      selectedSessionId = null;
      rememberSession("");
      renderConversationList();
      clearFeed();
      resetActivity();
      conversationLabel.textContent = "New conversation";
      setConversationWorkspace(null);
      statStep.textContent = "-";
      statCost.textContent = "$0.0000";
      currentTokenUsage = { prompt: 0, completion: 0, total: 0 };
      setTokenStats(0, 0, 0);
      statExit.textContent = "-";
      setStatus("idle", "");
      setComposerRunning(false);
      taskEl.value = "";
      resizeTaskInput();
      taskEl.focus();
    }

    attachImageButton.addEventListener("click", () => imageInput.click());
    imageInput.addEventListener("change", async () => {
      await addImageFiles(imageInput.files);
      imageInput.value = "";
    });
    taskEl.addEventListener("paste", async (event) => {
      const files = Array.from((event.clipboardData && event.clipboardData.items) || [])
        .filter((item) => item.kind === "file" && item.type.startsWith("image/"))
        .map((item) => item.getAsFile())
        .filter(Boolean);
      if (files.length) await addImageFiles(files);
    });
    ["dragenter", "dragover"].forEach((name) => {
      composerEl.addEventListener(name, (event) => {
        event.preventDefault();
        composerEl.classList.add("dragging");
      });
    });
    ["dragleave", "drop"].forEach((name) => {
      composerEl.addEventListener(name, (event) => {
        event.preventDefault();
        composerEl.classList.remove("dragging");
      });
    });
    composerEl.addEventListener("drop", async (event) => {
      const files = event.dataTransfer ? event.dataTransfer.files : [];
      await addImageFiles(files);
    });
    sendButton.addEventListener("click", startRun);
    stopButton.addEventListener("click", stopRun);
    newConversationButton.addEventListener("click", startNewConversation);
    taskEl.addEventListener("input", resizeTaskInput);
    taskEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) startRun();
    });
    updateScrollButtons();

    fetch("/api/config").then((r) => r.json()).then((config) => {
      defaultWorkspace = config.workspace || "";
      if (!selectedSessionId) setConversationWorkspace(null);
      if (config.initial_task) {
        taskEl.value = config.initial_task;
        resizeTaskInput();
      }
      document.getElementById("model-label").textContent = config.model_name || "default model";
      const databasePathEl = document.getElementById("database-path");
      if (config.database_path) {
        databasePathEl.textContent = "SQLite: " + config.database_path;
        databasePathEl.title = "SQLite conversation database: " + config.database_path;
      }
    }).catch(() => {});

    async function restoreConversation() {
      const remembered = rememberedSessionId();
      await loadConversations();
      if (remembered && conversations.some((conversation) => conversationId(conversation) === remembered)) {
        await selectConversation(remembered);
      } else if (conversations.length) {
        await selectConversation(conversationId(conversations[0]));
      }
    }

    restoreConversation();
    setInterval(() => loadConversations(), 5000);
    window.addEventListener("hashchange", () => {
      const sessionId = rememberedSessionId();
      if (sessionId && sessionId !== selectedSessionId) {
        selectConversation(sessionId);
      }
    });
  </script>
</body>
</html>
"""


def _workspace_from_config(agent_config: dict[str, Any]) -> str:
    """Return the directory in which GUI-triggered commands run."""
    environment = agent_config.get("environment") or {}
    return str(environment.get("cwd") or agent_config.get("workspace") or Path.cwd())


@dataclass
class GuiConfig:
    """Configuration shared by every GUI-triggered agent run."""

    agent_config: dict[str, Any]
    initial_task: str = ""
    model_name: str = ""
    workspace: str = ""

    def __post_init__(self) -> None:
        self.workspace = self.workspace or _workspace_from_config(self.agent_config)


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

    _TRAJECTORY_EVENT_TYPES = {"assistant_message", "observation", "done", "error", "format_error"}

    def __init__(
        self,
        task: str,
        *,
        config: GuiConfig,
        images: list[str] | None = None,
        agent_factory: Callable[[], Any] | None = None,
        store: ConversationStore | None = None,
    ):
        self.session_id = uuid.uuid4().hex
        self.task = task
        self.images = list(images or [])
        # ``task`` is the conversation title (its first user message), while
        # ``current_*`` describes the turn that is currently executing.  Keeping
        # them separate lets a conversation contain several follow-up runs.
        self.current_task = task
        self.current_images = list(images or [])
        self.config = config
        self.workspace = config.workspace
        self.agent_factory = agent_factory or (lambda: create_agent_from_config(config))
        self.events: list[dict[str, Any]] = []
        self.condition = threading.Condition()
        self.stop_requested = threading.Event()
        self.agent: Any | None = None
        self.done = False
        self._done_emitted = False
        self.created_at = time.time()
        self.exit_status = ""
        self.cost = 0.0
        self.n_calls = 0
        self.tokens = 0
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.error = ""
        self.trajectory: dict[str, Any] | None = None
        self._store = store
        self.thread = threading.Thread(target=self._run, name=f"mini-gui-{self.session_id[:8]}", daemon=True)

    @classmethod
    def restore(
        cls,
        summary: dict[str, Any],
        events: list[dict[str, Any]],
        *,
        config: GuiConfig,
        store: ConversationStore | None = None,
    ) -> GuiSession:
        """Recreate a completed or interrupted session loaded from SQLite."""
        session = cls.__new__(cls)
        session.session_id = str(summary["session_id"])
        session.task = str(summary.get("task") or "")
        session.images = list(
            next((event.get("images") for event in events if event.get("type") == "run_start"), []) or []
        )
        session.config = config
        session.workspace = str(summary.get("workspace") or config.workspace)
        session.agent_factory = None
        session.events = list(events)
        session.condition = threading.Condition()
        session.stop_requested = threading.Event()
        session.agent = None
        session.done = bool(summary.get("done", True))
        session._done_emitted = True
        session.created_at = float(summary.get("created_at") or time.time())
        session.exit_status = str(summary.get("exit_status") or "")
        session.cost = float(summary.get("cost") or 0.0)
        session.n_calls = int(summary.get("n_calls") or 0)
        usage = summary.get("token_usage") or {}
        session.token_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens") or summary.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or summary.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or summary.get("total_tokens") or 0),
        }
        session.tokens = session.token_usage["total_tokens"]
        session.error = str(summary.get("error") or "")
        session.trajectory = None
        session._store = store
        session.thread = None
        if session.done and (not session.events or session.events[-1].get("type") != "done"):
            # A process can stop after persisting intermediate events but before
            # the agent emits its terminal event.  The summary is still marked
            # done for such restored runs, so synthesize the missing terminal
            # event to keep event replay consistent with the conversation state.
            session.events.append(
                {
                    "type": "done",
                    "exit_status": session.exit_status or str(summary.get("status") or "Interrupted"),
                    "submission": "",
                    "cost": session.cost,
                    "n_calls": session.n_calls,
                    "prompt_tokens": session.token_usage["prompt_tokens"],
                    "completion_tokens": session.token_usage["completion_tokens"],
                    "total_tokens": session.token_usage["total_tokens"],
                    "tokens": session.tokens,
                    "token_usage": dict(session.token_usage),
                }
            )
        return session

    def start(self) -> None:
        self.thread.start()

    def request_stop(self) -> bool:
        """Ask the running agent to stop and return whether it was still running."""
        with self.condition:
            if self.done:
                return False
            self.stop_requested.set()
            agent = self.agent
        if agent is not None:
            if hasattr(agent, "request_stop"):
                agent.request_stop()
            elif hasattr(agent, "stop"):
                agent.stop()
        return True

    def stop(self) -> bool:
        """Alias for :meth:`request_stop`."""
        return self.request_stop()

    def continue_run(self, task: str, images: list[str] | None = None) -> int:
        """Append a follow-up user turn and start another run in this conversation.

        Returns the event index at which the new run starts.  The browser uses
        that offset to skip already-rendered events when it reconnects to the
        session stream.
        """
        task = task.strip()
        images = list(images or [])
        if not task and not images:
            raise ValueError("Task must not be empty")
        if not task:
            task = "Please analyze the attached image."
        with self.condition:
            if not self.done:
                # A user may click Stop and immediately send the next message.
                # Give the previous agent a moment to observe the stop request
                # before rejecting the follow-up.
                if not self.stop_requested.is_set():
                    raise RuntimeError("Conversation is still running")
                deadline = time.monotonic() + 5.0
                while not self.done:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError("Conversation is still stopping")
                    self.condition.wait(remaining)
            self.current_task = task
            self.current_images = images
            self.done = False
            self._done_emitted = False
            self.stop_requested.clear()
            self.exit_status = ""
            self.error = ""
            self.agent = None
            event_offset = len(self.events)
            self.thread = threading.Thread(target=self._run, name=f"mini-gui-{self.session_id[:8]}", daemon=True)
        self._persist_summary()
        self.thread.start()
        return event_offset

    def _set_token_usage(self, prompt: Any = 0, completion: Any = 0, total: Any = 0) -> None:
        prompt_tokens = int(prompt or 0)
        completion_tokens = int(completion or 0)
        self.token_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(total or prompt_tokens + completion_tokens),
        }
        self.tokens = self.token_usage["total_tokens"]

    def _summary_locked(self) -> dict[str, Any]:
        status = self.exit_status or ("running" if not self.done else "done")
        state = conversation_state(status, done=self.done)
        return {
            "id": self.session_id,
            "session_id": self.session_id,
            "task": self.task,
            "image_count": len(self.images),
            "workspace": self.workspace,
            "status": status,
            "state": state.value,
            "created_at": self.created_at,
            "done": self.done,
            "exit_status": self.exit_status,
            "cost": self.cost,
            "n_calls": self.n_calls,
            "prompt_tokens": self.token_usage["prompt_tokens"],
            "completion_tokens": self.token_usage["completion_tokens"],
            "total_tokens": self.tokens,
            "tokens": self.tokens,
            "token_usage": dict(self.token_usage),
            "event_count": len(self.events),
            "error": self.error,
        }

    def summary(self) -> dict[str, Any]:
        """Return the metadata needed to render this conversation in the sidebar."""
        with self.condition:
            return self._summary_locked()

    def _persist_summary(self) -> None:
        if self._store is None:
            return
        try:
            self._store.save_session(self.summary())
        except Exception:  # noqa: BLE001 - persistence must not crash an agent run
            logger.exception("Could not persist GUI conversation summary")

    def _persist_trajectory(self) -> None:
        agent = self.agent
        if self._store is None or agent is None or not hasattr(agent, "serialize"):
            return
        try:
            trajectory = agent.serialize()
            if isinstance(trajectory, dict):
                self.trajectory = trajectory
                self._store.save_trajectory(self.session_id, trajectory)
        except Exception:  # noqa: BLE001 - persistence must not crash an agent run
            logger.exception("Could not persist GUI conversation trajectory")

    def _persist_event(
        self, summary: dict[str, Any], event_index: int, event: dict[str, Any], event_type: str | None
    ) -> None:
        if self._store is None:
            return
        try:
            self._store.record_event(summary, event_index, event)
            if event_type in self._TRAJECTORY_EVENT_TYPES:
                self._persist_trajectory()
        except Exception:  # noqa: BLE001 - persistence must not crash an agent run
            logger.exception("Could not persist GUI conversation event")

    def emit(self, event: dict[str, Any]) -> None:
        safe_event = _json_safe(event)
        with self.condition:
            self.events.append(safe_event)
            event_type = safe_event.get("type")
            if event_type in ("assistant_message", "format_error"):
                cumulative = safe_event.get("cumulative_usage")
                if isinstance(cumulative, dict):
                    self._set_token_usage(
                        cumulative.get("prompt_tokens"),
                        cumulative.get("completion_tokens"),
                        cumulative.get("total_tokens"),
                    )
                else:
                    usage = safe_event.get("usage") or {}
                    self._set_token_usage(
                        self.token_usage["prompt_tokens"]
                        + int(usage.get("prompt_tokens") or safe_event.get("prompt_tokens") or 0),
                        self.token_usage["completion_tokens"]
                        + int(usage.get("completion_tokens") or safe_event.get("completion_tokens") or 0),
                        self.token_usage["total_tokens"]
                        + int(
                            usage.get("total_tokens") or safe_event.get("tokens") or safe_event.get("total_tokens") or 0
                        ),
                    )
            if event_type == "done":
                self._done_emitted = True
                self.exit_status = str(safe_event.get("exit_status") or "done")
                self.cost = float(safe_event.get("cost") or 0.0)
                self.n_calls = int(safe_event.get("n_calls") or 0)
                usage = safe_event.get("token_usage") or safe_event.get("cumulative_usage") or safe_event.get("usage")
                if not isinstance(usage, dict):
                    usage = {}
                self._set_token_usage(
                    safe_event.get(
                        "total_prompt_tokens", usage.get("prompt_tokens", safe_event.get("prompt_tokens", 0))
                    ),
                    safe_event.get(
                        "total_completion_tokens",
                        usage.get("completion_tokens", safe_event.get("completion_tokens", 0)),
                    ),
                    safe_event.get(
                        "total_tokens_used",
                        usage.get("total_tokens", safe_event.get("total_tokens", safe_event.get("tokens", 0))),
                    ),
                )
            elif event_type == "error":
                self.error = str(safe_event.get("message") or "Unknown error")
            event_index = len(self.events) - 1
            summary = self._summary_locked()
            self.condition.notify_all()
        self._persist_event(summary, event_index, safe_event, event_type)

    def _run(self) -> None:
        result: dict[str, Any] = {
            "exit_status": "Error",
            "submission": "",
            "cost": 0.0,
            "n_calls": 0,
            "tokens": 0,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        try:
            reply = conversational_reply(self.current_task)
            if reply is not None and not self.current_images:
                agent_config = self.config.agent_config.get("agent", {})
                custom_reply = str(agent_config.get("conversational_response") or "").strip()
                reply = custom_reply or reply
                zero_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                self.emit(
                    {
                        "type": "run_start",
                        "task": self.current_task,
                        "images": list(self.current_images),
                        "step_limit": agent_config.get("step_limit", 0),
                        "cost_limit": agent_config.get("cost_limit", 0.0),
                    }
                )
                self.emit(
                    {
                        "type": "assistant_message",
                        "content": reply,
                        "actions": [],
                        "cost": 0.0,
                        "step": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "tokens": 0,
                        "usage": zero_usage,
                        "total_prompt_tokens": 0,
                        "total_completion_tokens": 0,
                        "total_tokens_used": 0,
                        "cumulative_usage": zero_usage,
                        "conversational": True,
                    }
                )
                self.emit(
                    {
                        "type": "done",
                        "exit_status": "Submitted",
                        "submission": reply,
                        "cost": 0.0,
                        "n_calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "tokens": 0,
                        "token_usage": zero_usage,
                    }
                )
                return
            agent = self.agent_factory()
            self.agent = agent
            if hasattr(agent, "set_event_callback"):
                agent.set_event_callback(self.emit)
            if self.stop_requested.is_set():
                if hasattr(agent, "request_stop"):
                    agent.request_stop()
                elif hasattr(agent, "stop"):
                    agent.stop()
            run_kwargs: dict[str, Any] = {}
            if self.current_images:
                try:
                    parameters = inspect.signature(agent.run).parameters.values()
                    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters) or any(
                        p.name == "images" for p in parameters
                    ):
                        run_kwargs["images"] = self.current_images
                except (TypeError, ValueError):
                    # Builtins and mocks may not expose a signature; try the
                    # richer call and let the exception path report failures.
                    run_kwargs["images"] = self.current_images
            returned = agent.run(self.current_task, **run_kwargs)
            if isinstance(returned, dict):
                result = {**result, **returned}
        except UserInterruption:
            # Some custom agents may propagate the stop signal rather than
            # converting it into an exit message. Treat it as a clean user stop
            # instead of surfacing a scary error and leaving the conversation
            # in the failed state.
            result = {**result, "exit_status": "Stopped", "submission": ""}
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
            self._persist_trajectory()
            self._persist_summary()

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
    """Owns GUI sessions, restoring and persisting them through SQLite."""

    def __init__(
        self,
        config: GuiConfig,
        *,
        agent_factory: Callable[[], Any] | None = None,
        db_path: str | Path | None = None,
        database_path: str | Path | None = None,
        store: ConversationStore | None = None,
    ):
        self.config = config
        self.agent_factory = agent_factory
        self.store = store or ConversationStore(
            db_path or database_path or os.getenv("MSWEA_GUI_DB_PATH") or DEFAULT_DATABASE_FILE
        )
        self.sessions: dict[str, GuiSession] = {}
        self._lock = threading.Lock()
        self._restore_sessions()

    @property
    def database_path(self) -> str:
        """Return the absolute path to the SQLite conversation database."""
        if self.store.path == ":memory:":
            return self.store.path
        return str(Path(self.store.path).expanduser().resolve())

    def database_info(self) -> dict[str, Any]:
        """Return information that helps users locate the conversation database."""
        path = self.database_path
        exists = path != ":memory:" and Path(path).exists()
        return {
            "path": path,
            "url": f"sqlite:///{path}" if path != ":memory:" else "sqlite:///:memory:",
            "exists": exists,
            "size_bytes": Path(path).stat().st_size if exists else 0,
        }

    def _restore_sessions(self) -> None:
        for summary in self.store.load_sessions():
            events = self.store.load_events(summary["session_id"])
            last_event_type = events[-1].get("type") if events else ""
            if last_event_type == "done":
                if not summary.get("done"):
                    summary["done"] = True
                    summary["exit_status"] = str(events[-1].get("exit_status") or summary.get("exit_status") or "")
                    summary["status"] = summary["exit_status"] or "done"
                    self.store.save_session(summary)
            elif not summary.get("done"):
                # A process restart cannot resume a previous agent thread.
                summary["done"] = True
                summary["status"] = "Interrupted"
                summary["exit_status"] = "Interrupted"
                self.store.save_session(summary)
            session = GuiSession.restore(summary, events, config=self.config, store=self.store)
            self.sessions[session.session_id] = session

    def start_session(self, task: str, images: list[str] | None = None) -> GuiSession:
        task = task.strip()
        images = list(images or [])
        if not task and not images:
            raise ValueError("Task must not be empty")
        if not task:
            task = "Please analyze the attached image."
        session = GuiSession(
            task,
            config=self.config,
            images=images,
            agent_factory=self.agent_factory,
            store=self.store,
        )
        with self._lock:
            self.sessions[session.session_id] = session
        self.store.save_session(session.summary())
        session.start()
        return session

    def continue_session(
        self, session_id: str, task: str, images: list[str] | None = None
    ) -> tuple[GuiSession, int] | None:
        """Append a follow-up message to an existing conversation."""
        session = self.get_session(session_id)
        if session is None:
            return None
        return session, session.continue_run(task, images=images)

    def get_session(self, session_id: str) -> GuiSession | None:
        with self._lock:
            return self.sessions.get(session_id)

    def list_sessions(self) -> list[GuiSession]:
        """Return every conversation, newest first."""
        with self._lock:
            return sorted(self.sessions.values(), key=lambda session: session.created_at, reverse=True)

    def list_conversations(self) -> list[GuiSession]:
        """Alias for :meth:`list_sessions`."""
        return self.list_sessions()

    def state_counts(self) -> dict[str, int]:
        """Return the number of conversations in each canonical state."""
        counts = {state.value: 0 for state in ConversationState}
        for session in self.list_sessions():
            counts[session.summary()["state"]] += 1
        return counts

    def status_counts(self) -> dict[str, int]:
        """Alias for :meth:`state_counts`."""
        return self.state_counts()

    def export_session(self, session_id: str) -> dict[str, Any] | None:
        """Return a self-contained JSON export for a conversation."""
        return self.store.export_conversation(session_id)

    def export_conversation(self, session_id: str) -> dict[str, Any] | None:
        """Alias for :meth:`export_session`."""
        return self.export_session(session_id)

    def stop_session(self, session_id: str) -> GuiSession | None:
        session = self.get_session(session_id)
        if session is not None:
            session.request_stop()
        return session

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self.store.close()


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

    def _send_conversation_export(self, session_id: str) -> None:
        payload = self.server.manager.export_session(session_id)
        if payload is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown session"})
            return
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition", f'attachment; filename="mini-swe-agent-{session_id}.traj.json"')
        self.send_header("Cache-Control", "no-store")
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
                    "workspace": self.server.manager.config.workspace,
                    "database_path": self.server.manager.database_path,
                    "db_path": self.server.manager.database_path,
                    "database": self.server.manager.database_info(),
                },
            )
        elif parsed.path in ("/api/database", "/api/database/path", "/api/info"):
            self._send_json(
                HTTPStatus.OK,
                {
                    "database_path": self.server.manager.database_path,
                    "db_path": self.server.manager.database_path,
                    "database": self.server.manager.database_info(),
                },
            )
        elif parsed.path.startswith("/api/export/"):
            self._send_conversation_export(parsed.path.removeprefix("/api/export/").strip("/"))
        elif parsed.path in ("/api/conversations/statuses", "/api/statuses", "/api/states"):
            counts = self.server.manager.state_counts()
            states = [state.value for state in ConversationState]
            self._send_json(
                HTTPStatus.OK,
                {
                    "count": len(states),
                    "states": states,
                    "state_counts": counts,
                    "status_counts": counts,
                },
            )
        elif parsed.path in (
            "/api/conversations",
            "/api/conversation",
            "/api/sessions",
            "/api/session",
            "/api/history",
        ):
            conversations = [session.summary() for session in self.server.manager.list_sessions()]
            counts = self.server.manager.state_counts()
            states = [state.value for state in ConversationState]
            self._send_json(
                HTTPStatus.OK,
                {
                    "conversations": conversations,
                    "sessions": conversations,
                    "items": conversations,
                    "states": states,
                    "state_count": len(states),
                    "state_counts": counts,
                    "status_counts": counts,
                },
            )
        elif parsed.path.startswith(("/api/conversations/", "/api/conversation/", "/api/sessions/", "/api/session/")):
            prefix = next(
                prefix
                for prefix in ("/api/conversations/", "/api/conversation/", "/api/sessions/", "/api/session/")
                if parsed.path.startswith(prefix)
            )
            session_id = parsed.path.removeprefix(prefix).strip("/")
            if session_id.endswith("/export"):
                self._send_conversation_export(session_id.removesuffix("/export").strip("/"))
                return
            session = self.server.manager.get_session(session_id)
            if session is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown session"})
                return
            with session.condition:
                events = list(session.events)
            self._send_json(
                HTTPStatus.OK,
                {"conversation": session.summary(), "session": session.summary(), "events": events},
            )
        elif parsed.path.startswith("/api/events/"):
            session_id = parsed.path.removeprefix("/api/events/").strip("/")
            session = self.server.manager.get_session(session_id)
            if session is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown session"})
                return
            try:
                since = max(0, int(parse_qs(parsed.query).get("since", ["0"])[0]))
            except ValueError:
                since = 0
            self._stream_events(session, since=since)
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length <= 0 or length > 64_000_000:
            raise ValueError("Invalid request body")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _stop_session(self, session_id: str) -> None:
        session = self.server.manager.get_session(session_id)
        if session is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown session"})
            return
        session.request_stop()
        self._send_json(HTTPStatus.ACCEPTED, {"status": "stopping", "session_id": session.session_id})

    def do_POST(self) -> None:  # noqa: N802 - stdlib method name
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/stop/"):
            session_id = parsed.path.removeprefix("/api/stop/").strip("/")
            self._stop_session(session_id)
            return
        if parsed.path == "/api/stop":
            try:
                session_id = self._read_json_body().get("session_id", "")
                if not isinstance(session_id, str):
                    raise ValueError("session_id must be a string")
            except (json.JSONDecodeError, ValueError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._stop_session(session_id)
            return
        if parsed.path != "/api/run":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        try:
            payload = self._read_json_body()
            task = payload.get("task", "")
            images = payload.get("images", [])
            session_id = payload.get("session_id") or ""
            if not isinstance(task, str):
                raise ValueError("task must be a string")
            if not isinstance(session_id, str):
                raise ValueError("session_id must be a string")
            if images is None:
                images = []
            if not isinstance(images, list) or not all(isinstance(image, str) for image in images):
                raise ValueError("images must be a list of data URLs or image URLs")
            if len(images) > 8:
                raise ValueError("At most 8 images can be attached")
            for image in images:
                if not image.startswith(("data:image/", "http://", "https://")):
                    raise ValueError("images must be data URLs or http(s) URLs")
            if session_id:
                continued = self.server.manager.continue_session(session_id, task, images=images)
                if continued is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown session"})
                    return
                session, event_offset = continued
            else:
                session = self.server.manager.start_session(task, images=images)
                event_offset = 0
        except RuntimeError as exc:
            self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
            return
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001 - surface setup errors to UI
            logger.exception("Could not start GUI session")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return
        self._send_json(
            HTTPStatus.ACCEPTED,
            {
                "session_id": session.session_id,
                "events_url": f"/api/events/{session.session_id}",
                "conversation": session.summary(),
                "event_offset": event_offset,
                "continued": bool(session_id),
            },
        )

    def _stream_events(self, session: GuiSession, *, since: int = 0) -> None:
        try:
            cursor = int(self.headers.get("Last-Event-ID", "0") or "0")
        except ValueError:
            cursor = 0
        # A stream opened for a follow-up run can skip the events that the
        # browser has already rendered.  Automatic reconnects send
        # Last-Event-ID, which takes precedence over the initial offset.
        if cursor <= 0:
            cursor = since
        with session.condition:
            # Event ids are 1-based; keep reconnects from replaying events or
            # waiting forever when they send an out-of-range id.
            cursor = max(0, min(cursor, len(session.events)))
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
    Console().print(f"[bold green]mini-SWE-agent GUI:[/bold green] {url}", soft_wrap=True)
    Console().print(f"[bold green]Conversation database:[/bold green] {manager.database_path}", soft_wrap=True)
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
    database: Path = typer.Option(
        DEFAULT_DATABASE_FILE,
        "--database",
        "--db",
        help="SQLite conversation database path (default: gui_conversations.db in the global config directory)",
    ),
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
    manager = GuiSessionManager(gui_config, db_path=database)
    serve(manager, host=host, port=port, open_browser=open_browser)


if __name__ == "__main__":
    app()
