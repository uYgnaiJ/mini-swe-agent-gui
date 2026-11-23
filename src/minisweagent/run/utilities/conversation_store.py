"""SQLite persistence for conversations shown by the web GUI."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class ConversationStore:
    """Persist GUI conversations and their streamed events in SQLite.

    A single connection is shared by the GUI's agent threads and HTTP handler
    threads. Access is serialized with a re-entrant lock so events are written
    atomically together with the conversation summary they belong to.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    session_id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    pinned INTEGER NOT NULL DEFAULT 0,
                    parent_id TEXT NOT NULL DEFAULT '',
                    branch_turn_index INTEGER NOT NULL DEFAULT -1,
                    deleted_at REAL NOT NULL DEFAULT 0,
                    workspace TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'running',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    done INTEGER NOT NULL DEFAULT 0,
                    exit_status TEXT NOT NULL DEFAULT '',
                    cost REAL NOT NULL DEFAULT 0,
                    n_calls INTEGER NOT NULL DEFAULT 0,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    token_usage TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    event_count INTEGER NOT NULL DEFAULT 0,
                    trajectory TEXT
                );
                CREATE TABLE IF NOT EXISTS events (
                    session_id TEXT NOT NULL,
                    event_index INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    data TEXT NOT NULL,
                    PRIMARY KEY (session_id, event_index)
                );
                CREATE INDEX IF NOT EXISTS events_session_index
                    ON events(session_id, event_index);
                """
            )
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(conversations)")}
            for name, definition in (
                ("title", "TEXT NOT NULL DEFAULT ''"),
                ("pinned", "INTEGER NOT NULL DEFAULT 0"),
                ("parent_id", "TEXT NOT NULL DEFAULT ''"),
                ("branch_turn_index", "INTEGER NOT NULL DEFAULT -1"),
                ("deleted_at", "REAL NOT NULL DEFAULT 0"),
            ):
                if name not in columns:
                    self._conn.execute(f"ALTER TABLE conversations ADD COLUMN {name} {definition}")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _summary_row(summary: dict[str, Any]) -> tuple[Any, ...]:
        session_id = str(summary.get("session_id") or summary.get("id") or "")
        if not session_id:
            raise ValueError("Conversation summary is missing a session_id")
        token_usage = summary.get("token_usage") or {}
        return (
            session_id,
            str(summary.get("task") or ""),
            str(summary.get("workspace") or ""),
            str(summary.get("status") or "running"),
            float(summary.get("created_at") or time.time()),
            time.time(),
            int(bool(summary.get("done"))),
            str(summary.get("exit_status") or ""),
            float(summary.get("cost") or 0.0),
            int(summary.get("n_calls") or 0),
            int(summary.get("prompt_tokens") or 0),
            int(summary.get("completion_tokens") or 0),
            int(summary.get("total_tokens") or summary.get("tokens") or 0),
            json.dumps(token_usage, default=str),
            str(summary.get("error") or ""),
            int(summary.get("event_count") or 0),
            json.dumps(summary["trajectory"], default=str) if summary.get("trajectory") is not None else None,
            str(summary.get('title') or ''),
            int(bool(summary.get('pinned'))),
            str(summary.get('parent_id') or ''),
            int(summary.get('branch_turn_index', -1)),
        )

    def _save_summary_locked(self, summary: dict[str, Any], *, commit: bool = True) -> None:
        row = self._summary_row(summary)
        self._conn.execute(
            """
            INSERT INTO conversations (
                session_id, task, workspace, status, created_at, updated_at,
                done, exit_status, cost, n_calls, prompt_tokens,
                completion_tokens, total_tokens, token_usage, error,
                event_count, trajectory, title, pinned, parent_id, branch_turn_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                task=excluded.task,
                workspace=excluded.workspace,
                status=excluded.status,
                updated_at=excluded.updated_at,
                done=excluded.done,
                exit_status=excluded.exit_status,
                cost=excluded.cost,
                n_calls=excluded.n_calls,
                prompt_tokens=excluded.prompt_tokens,
                completion_tokens=excluded.completion_tokens,
                total_tokens=excluded.total_tokens,
                token_usage=excluded.token_usage,
                error=excluded.error,
                event_count=excluded.event_count,
                trajectory=COALESCE(excluded.trajectory, conversations.trajectory),
                title=excluded.title,
                pinned=excluded.pinned,
                parent_id=excluded.parent_id,
                branch_turn_index=excluded.branch_turn_index
            """,
            row,
        )
        if commit:
            self._conn.commit()

    def save_session(self, summary: dict[str, Any]) -> None:
        """Insert or update a conversation summary."""
        with self._lock:
            self._save_summary_locked(summary)

    def record_event(self, summary: dict[str, Any], event_index: int, event: dict[str, Any]) -> None:
        """Atomically persist a conversation update and its event."""
        with self._lock:
            self._save_summary_locked(summary, commit=False)
            self._conn.execute(
                """
                INSERT INTO events (session_id, event_index, created_at, data)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id, event_index) DO UPDATE SET data=excluded.data
                """,
                (
                    str(summary.get("session_id") or summary.get("id") or ""),
                    int(event_index),
                    time.time(),
                    json.dumps(event, default=str),
                ),
            )
            self._conn.commit()

    def save_trajectory(self, session_id: str, trajectory: dict[str, Any]) -> None:
        """Store the latest agent trajectory snapshot for a conversation."""
        with self._lock:
            self._conn.execute(
                "UPDATE conversations SET trajectory=?, updated_at=? WHERE session_id=?",
                (json.dumps(trajectory, default=str), time.time(), session_id),
            )
            self._conn.commit()

    @staticmethod
    def _row_to_summary(row: sqlite3.Row) -> dict[str, Any]:
        try:
            token_usage = json.loads(row["token_usage"])
        except (TypeError, json.JSONDecodeError):
            token_usage = {}
        return {
            "id": row["session_id"],
            "session_id": row["session_id"],
            'task': row['task'],
            'title': row['title'] or row['task'],
            'pinned': bool(row['pinned']),
            'parent_id': row['parent_id'],
            'branch_turn_index': row['branch_turn_index'],
            'deleted_at': row['deleted_at'],
            "workspace": row["workspace"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "done": bool(row["done"]),
            "exit_status": row["exit_status"],
            "cost": row["cost"],
            "n_calls": row["n_calls"],
            "prompt_tokens": row["prompt_tokens"],
            "completion_tokens": row["completion_tokens"],
            "total_tokens": row["total_tokens"],
            "tokens": row["total_tokens"],
            "token_usage": token_usage,
            "event_count": row["event_count"],
            "error": row["error"],
        }

    def load_sessions(self, *, include_deleted: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            query = 'SELECT * FROM conversations'
            if not include_deleted:
                query += ' WHERE deleted_at = 0'
            rows = self._conn.execute(query + ' ORDER BY pinned DESC, created_at DESC').fetchall()
            return [self._row_to_summary(row) for row in rows]

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute(
                'UPDATE conversations SET deleted_at=?, updated_at=? WHERE session_id=?',
                (time.time(), time.time(), session_id),
            )
            self._conn.commit()

    def load_event_entries(self, session_id: str) -> list[tuple[int, dict[str, Any]]]:
        """Return (event_index, event) pairs so restores can resume numbering."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT event_index, data FROM events WHERE session_id=? ORDER BY event_index",
                (session_id,),
            ).fetchall()
        return [(int(row["event_index"]), json.loads(row["data"])) for row in rows]

    def load_events(self, session_id: str) -> list[dict[str, Any]]:
        """Return the complete event stream for a conversation."""
        return [event for _, event in self.load_event_entries(session_id)]

    def save_events(self, session_id: str, entries: list[tuple[int, dict[str, Any]]]) -> None:
        """Bulk-insert events, used when a branch copies its prefix history."""
        if not entries:
            return
        with self._lock:
            now = time.time()
            self._conn.executemany(
                """
                INSERT INTO events (session_id, event_index, created_at, data)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id, event_index) DO UPDATE SET data=excluded.data
                """,
                [(session_id, int(index), now, json.dumps(event, default=str)) for index, event in entries],
            )
            self._conn.commit()

    def load_trajectory(self, session_id: str) -> dict[str, Any] | None:
        """Return the latest persisted agent trajectory for a conversation."""
        with self._lock:
            row = self._conn.execute(
                "SELECT trajectory FROM conversations WHERE session_id=?",
                (session_id,),
            ).fetchone()
        if row is None or row["trajectory"] is None:
            return None
        try:
            trajectory = json.loads(row["trajectory"])
        except (TypeError, json.JSONDecodeError):
            return None
        return trajectory if isinstance(trajectory, dict) else None

    def export_conversation(self, session_id: str) -> dict[str, Any] | None:
        """Build a self-contained, downloadable JSON representation of a conversation."""
        with self._lock:
            row = self._conn.execute("SELECT * FROM conversations WHERE session_id=?", (session_id,)).fetchone()
            if row is None:
                return None
            summary = self._row_to_summary(row)
            raw_trajectory = row["trajectory"]
        events = self.load_events(session_id)
        try:
            trajectory = json.loads(raw_trajectory) if raw_trajectory else None
        except json.JSONDecodeError:
            trajectory = None
        export_data = dict(trajectory or {})
        export_data.update(
            {
                "trajectory_format": export_data.get("trajectory_format", "mini-swe-agent-gui-1.0"),
                "session_id": summary["session_id"],
                "task": summary["task"],
                "workspace": summary["workspace"],
                "status": summary["status"],
                "created_at": summary["created_at"],
                "updated_at": summary["updated_at"],
                "done": summary["done"],
                "exit_status": summary["exit_status"],
                "cost": summary["cost"],
                "n_calls": summary["n_calls"],
                "token_usage": summary["token_usage"],
                "events": events,
                "trajectory": trajectory,
                "conversation": summary,
                "session": summary,
            }
        )
        export_data.setdefault("messages", trajectory.get("messages", []) if trajectory else [])
        return export_data
