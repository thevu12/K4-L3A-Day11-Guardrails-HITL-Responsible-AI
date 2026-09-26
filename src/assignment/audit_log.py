"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import time


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input and a monotonic start time for the matching output."""
        key = request_id or user_id
        self._open[key] = {
            "request_id": request_id or key,
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
            "started_monotonic": time.perf_counter(),
        }

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Complete an audit entry with output, decision layer and latency."""
        key = request_id or user_id
        pending = self._open.pop(key, None)
        now = time.perf_counter()
        if pending is None:
            pending = {
                "request_id": request_id or key,
                "user_id": user_id,
                "input": "",
                "started_at": utc_now_iso(),
                "started_monotonic": now,
            }

        self.logs.append({
            "request_id": pending["request_id"],
            "user_id": pending["user_id"],
            "input": pending["input"],
            "output": text,
            "blocked": bool(blocked),
            "layer": layer,
            "started_at": pending["started_at"],
            "completed_at": utc_now_iso(),
            "latency_ms": round((now - pending["started_monotonic"]) * 1000, 3),
        })

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
