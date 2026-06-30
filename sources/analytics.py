#!/usr/bin/env python3
"""
Unified analytics event logger — JSONL format for easy SQL / pandas ingestion.

Usage::

    from sources.analytics import track_event

    track_event("auth:login", user_id="u_123", email="a@b.com")
    track_event("query_stream", user_id="u_123", session_id="s_456",
                query_text="分析专利 CN123456", query_id="q_789")
    track_event("long_task:start", user_id="u_123", task_id="lt_abc",
                patent_count=15, query_text="...")

Each call appends one JSON line to ``.logs/analytics.log``.

Computing DAU / MAU from JSONL (example with duckdb / pandas)::

    import pandas as pd
    df = pd.read_json(".logs/analytics.log", lines=True)
    df["date"] = df["timestamp"].str[:10]
    dau = df.groupby("date")["user_id"].nunique()
    mau = df["user_id"].drop_duplicates().resample("M", on="date").nunique()
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from sources.logger import Logger

_logger = Logger("analytics.log")

# ── Lock so concurrent coroutines don't interleave JSON lines ──
_write_lock = threading.Lock()


# ── Public API ──────────────────────────────────────────────────────────────────


def track_event(
    event: str,
    *,
    user_id: Optional[str] = None,
    email: Optional[str] = None,
    session_id: Optional[str] = None,
    query_id: Optional[str] = None,
    task_id: Optional[str] = None,
    query_text: Optional[str] = None,
    patent_count: Optional[int] = None,
    patent_source: Optional[str] = None,
    knowledge_id: Optional[int] = None,
    scene_id: Optional[int] = None,
    extra: Optional[dict] = None,
    dedup_window_s: float = 1.0,
) -> None:
    """Emit a structured analytics event as one JSON line.

    Parameters
    ----------
    event:
        Event name following ``domain:action`` convention, e.g.
        ``auth:login``, ``query_stream``, ``long_task:start``,
        ``knowledge:create``, ``session:new``.
    user_id:
        Backend user id (int-as-str from MySQL ``users.user_id``).
    email:
        User email (only included in auth events; never logged for
        authenticated query/knowledge events to keep log size down).
    session_id:
        Conversation session id.
    query_id:
        Per-query correlation id.
    task_id:
        Long task / Celery task id.
    query_text:
        First 80 chars of the user query (truncated for privacy + size).
    patent_count:
        Number of patents in a long-task batch.
    patent_source:
        ``uspto`` or ``cnipa``.
    knowledge_id:
        Knowledge record id.
    scene_id:
        Scene id.
    extra:
        Arbitrary extra fields to include in the JSON line.
    dedup_window_s:
        Ignore duplicate identical events within this window (default 1s).
    """
    ts = datetime.now(timezone.utc).isoformat()

    payload: dict = {
        "event": event,
        "timestamp": ts,
        "ts_epoch": time.time(),
    }

    if user_id is not None:
        payload["user_id"] = user_id
    if email is not None:
        payload["email"] = email
    if session_id is not None:
        payload["session_id"] = session_id
    if query_id is not None:
        payload["query_id"] = query_id
    if task_id is not None:
        payload["task_id"] = task_id
    if query_text is not None:
        payload["query_text"] = _truncate(query_text, 80)
    if patent_count is not None:
        payload["patent_count"] = patent_count
    if patent_source is not None:
        payload["patent_source"] = patent_source
    if knowledge_id is not None:
        payload["knowledge_id"] = knowledge_id
    if scene_id is not None:
        payload["scene_id"] = scene_id
    if extra:
        payload.update(extra)

    line = json.dumps(payload, ensure_ascii=False)

    # Dedup: suppress identical lines within dedup_window_s
    # (useful for retry-storms on auth endpoints)
    cache_key = _dedup_key(event, user_id, line)
    now = time.monotonic()
    with _dedup_lock:
        last_ts = _dedup_cache.get(cache_key)
        if last_ts is not None and (now - last_ts) < dedup_window_s:
            return
        _dedup_cache[cache_key] = now
        # Periodic cleanup of stale cache entries
        if len(_dedup_cache) > 500:
            _dedup_cache.clear()

    with _write_lock:
        _logger.info(line)


# ── Dedup internals ─────────────────────────────────────────────────────────────

_dedup_cache: dict[str, float] = {}
_dedup_lock = threading.Lock()


def _dedup_key(event: str, user_id: Optional[str], line: str) -> str:
    """Build a short dedup key from event + user + hash of the line."""
    # Use hash so large query_text doesn't bloat the cache dict keys
    line_hash = hash(line)
    return f"{event}|{user_id or '?'}|{line_hash}"


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len characters, appending '…' when cut."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 1] + "…"


# ── Example queries (for documentation) ─────────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke test
    track_event("auth:login", user_id="u_1", email="test@example.com")
    track_event("query_stream", user_id="u_1", session_id="s_1",
                query_text="帮我分析这个专利", query_id="q_1")
    track_event("long_task:start", user_id="u_1", task_id="lt_abc123",
                patent_count=15, patent_source="uspto",
                query_text="分析这些专利的技术方案")
    track_event("knowledge:create", user_id="u_1", knowledge_id=42, scene_id=7)
    print("✅ Wrote 4 events to .logs/analytics.log")
