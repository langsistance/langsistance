import json
import re

TASK_STATUS_PREFIX = "lt"
TASK_CHECKPOINT_PREFIX = "lt"
TASK_STATUS_TTL = 86400  # 24 hours

# ── Structured failure reason codes (spec §5.4) ─────────────────────────────
# Shared by the resolvability pre-check gates (chat main chain / submit /
# retry / detail routes) and the worker terminal-failure writer.  Task 4
# (submit/retry/detail gates) and Task 5 (worker single-point exit) reference
# these same constants + failure_guidance(); do not rename here without
# updating those callers.
ERR_UNRESOLVABLE_ID = "ERR_UNRESOLVABLE_ID"
ERR_EPO_REMOTE = "ERR_EPO_REMOTE"
ERR_OTHER = "ERR_OTHER"

# ── Generic guidance segments (no user-query vocabulary allowed) ────────────
# These are format-level suggestions only — never embed a concrete query word.
# Spec §5.4 template; zh + en.
_GUIDE_SEG = {
    ERR_UNRESOLVABLE_ID: {
        "zh": (
            "该分析需要公开号格式的号码。请提供 WO 公开号（如 WO2021/xxxxx）、"
            "美国授权/公开号或国家阶段申请号后重发；也可改为按申请人/优先权检索。"
        ),
        "en": (
            "This analysis needs a publication-format number. Please provide a "
            "WO publication number (e.g. WO2021/xxxxx), a US grant/publication "
            "number or a national-phase application number, and retry; or search "
            "by applicant / priority instead."
        ),
    },
    ERR_EPO_REMOTE: {
        "zh": "外部服务暂时不可用，请稍后在任务面板点击重试。",
        "en": "The external service is temporarily unavailable. Please retry from "
              "the task panel shortly.",
    },
    ERR_OTHER: {
        "zh": "可在任务面板点击重试，或重新描述需求后再试。",
        "en": "Please retry from the task panel, or restate your request and try "
              "again.",
    },
}
# Fall back to ERR_OTHER for any unrecognised reason_code (fail-closed to a
# generic next-step that never crashes).
_GUIDE_DEFAULT_CODE = ERR_OTHER

# Fields that persist across update_task_status calls.
# Once set by any call, they survive subsequent calls that don't include them.
# This prevents Redis-key-overwrite from losing metadata set by earlier phases.
_STICKY_FIELDS = frozenset({
    'analysis_type',
    'family_overview',
    'table_columns',
    'result_summary',
    'documents',
})


def _get_redis():
    from sources.knowledge.knowledge import get_redis_connection
    return get_redis_connection()


def _status_key(task_id: str) -> str:
    return f"{TASK_STATUS_PREFIX}:{task_id}:status"


def _checkpoint_key(task_id: str) -> str:
    return f"{TASK_CHECKPOINT_PREFIX}:{task_id}:checkpoint"


def update_task_status(task_id: str, phase: str, progress: int,
                       step_msg: str, status: str = 'running', **extra) -> None:
    """Write current task status to Redis.

    Sticky fields (analysis_type, family_overview, table_columns,
    result_summary) are preserved from the previous status when not
    explicitly provided in the current call.  This prevents metadata
    set early in a task lifecycle from being wiped by later updates
    that only change progress / step label.
    """
    import time
    r = _get_redis()

    # Preserve sticky fields from any existing status record
    raw = r.get(_status_key(task_id))
    if raw is not None:
        try:
            existing = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            existing = {}
        for field in _STICKY_FIELDS:
            if field not in extra and field in existing:
                extra[field] = existing[field]

    payload = {
        'task_id': task_id,
        'status': status,
        'current_phase': phase,
        'progress': progress,
        'current_step': step_msg,
        'last_update': time.time(),
        **extra,
    }
    r.set(_status_key(task_id), json.dumps(payload, ensure_ascii=False),
          ex=TASK_STATUS_TTL)


def get_task_status(task_id: str) -> dict:
    """Read current task status from Redis."""
    r = _get_redis()
    raw = r.get(_status_key(task_id))
    if raw is None:
        return {'task_id': task_id, 'status': 'unknown'}
    return json.loads(raw)


def save_checkpoint(task_id: str, checkpoint: dict) -> None:
    """Save per-patent processing checkpoint to Redis."""
    r = _get_redis()
    r.set(_checkpoint_key(task_id), json.dumps(checkpoint, ensure_ascii=False),
          ex=TASK_STATUS_TTL)


def load_checkpoint(task_id: str) -> dict | None:
    """Load checkpoint; returns None if not found."""
    r = _get_redis()
    raw = r.get(_checkpoint_key(task_id))
    if raw is None:
        return None
    return json.loads(raw)


def set_task_completed(task_id: str, report_files: list,
                       patent_ids: list | None = None,
                       anchor_payload: dict | None = None) -> None:
    """Mark task as completed with report file metadata and optional patent IDs.

    ``anchor_payload`` (dict|None) carries the session-anchor data (Task 1
    shape); when present the anchor is written and the completion receipt
    carries a ``patent_data`` list (≤50) derived from it.

    Completion is always terminal, so the outcome is also written back
    into the task's conversation (M1: results must be discussable in the
    next chat turn).  The write-back never raises — it degrades silently.
    """
    r = _get_redis()
    raw = r.get(_status_key(task_id))
    status = json.loads(raw) if raw else {}
    status['status'] = 'completed'
    status['progress'] = 100
    status['report_files'] = report_files
    if patent_ids:
        status['patent_ids'] = patent_ids
    r.set(_status_key(task_id), json.dumps(status, ensure_ascii=False),
          ex=TASK_STATUS_TTL)

    try:
        from sources.long_task.task_messages import (
            append_task_message, build_result_digest)
        from sources.long_task.session_anchor import write_session_anchor
        if isinstance(anchor_payload, dict):
            session_id = _lookup_task_session_id(task_id)
            if session_id:
                write_session_anchor(
                    session_id,
                    anchor_type=str(anchor_payload.get('anchor_type') or 'topic'),
                    target=str(anchor_payload.get('target') or ''),
                    target_summary=str(anchor_payload.get('target_summary') or ''),
                    source=str(anchor_payload.get('source') or ''),
                    result_ids=list(anchor_payload.get('result_ids') or []),
                    result_titles=anchor_payload.get('result_titles') or None,
                    task_id=str(anchor_payload.get('task_id') or task_id),
                )
            patent_data = None
            rids = list(anchor_payload.get('result_ids') or [])[:50]
            if rids:
                titles = anchor_payload.get('result_titles') or {}
                patent_data = [
                    {'patent_id': str(pid),
                     'title': str(titles.get(pid, ''))[:200],
                     'source': str(anchor_payload.get('source') or '')}
                    for pid in rids]
            append_task_message(
                task_id,
                event='completed',
                content=build_result_digest(task_id),
                patent_ids=patent_ids,
                patent_data=patent_data,
                report_files=report_files,
            )
        else:
            append_task_message(
                task_id,
                event='completed',
                content=build_result_digest(task_id),
                patent_ids=patent_ids,
                report_files=report_files,
            )
    except Exception:
        pass  # conversation write-back must never break task state


def _lookup_task_session_id(task_id: str) -> str | None:
    """Session_id of *task_id* (MySQL long_tasks), or '' on any failure.

    Mirrors the existing ``_lookup_task_user_id`` / task_messages lookup
    query shape; only used when an anchor payload forces the session write.
    Failure degrades silently — anchor plumbing must never break task state.
    """
    try:
        from sources.knowledge.knowledge import get_db_connection
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT session_id FROM long_tasks WHERE task_id = %s",
                    (task_id,))
                row = cur.fetchone()
            return (row.get('session_id') or '') if row else ''
        finally:
            conn.close()
    except Exception:
        return ''


def classify_failure_reason_code(error: str) -> str:
    """Map a worker terminal error to a structured ``reason_code`` (spec §5.4).

    Fallback classifier used by :func:`notify_terminal_failure` when the
    caller did not supply an explicit structured code.  It keys off stable
    framework / server markers only — HTTP status ranges, EPO semantic codes,
    transport-timeout terms, auth tokens — never arbitrary user-query text,
    so it stays deterministic for the classes the templates document:

    - InvalidCountryCode / a docdb 404 (publication-format id unresolvable)
      → :data:`ERR_UNRESOLVABLE_ID`;
    - 5xx / timeouts / connectivity / credentials-token → :data:`ERR_EPO_REMOTE`;
    - anything else → :data:`ERR_OTHER`.
    """
    import re as _re
    text = str(error or "")
    if not text:
        return ERR_OTHER
    # remote / transient first (5xx, transport timeouts, auth token issues)
    if (_re.search(r"\bHTTP\s*5\d\d\b", text, _re.IGNORECASE)
            or _re.search(r"(?i)time.?out", text)
            or _re.search(r"(?i)could not connect|connection (reset|refused)|"
                          r"network is unreachable", text)
            or _re.search(r"(?i)\b401\b|\b403\b|OAuth|access_token|invalid_grant|"
                          r"credential", text)):
        return ERR_EPO_REMOTE
    # unresolvable publication-format id / country-code (EPO semantic + 404).
    if (_re.search(r"(?i)invalidcountrycode|invalid country(code)?|"
                   r"could not resolve", text)
            or _re.search(r"\b404\b", text)):
        return ERR_UNRESOLVABLE_ID
    return ERR_OTHER


# Transient markers recognised beyond the shared classifier's buckets —
# plain-text transport / proxy / rate-limit wording that classify_failure_
# reason_code() falls through to ERR_OTHER but that a retry CAN fix.
_EXTRA_TRANSIENT_RE = re.compile(
    r"(?i)bad gateway|gateway timeout|service unavailable|temporar[iy]ly?|"
    r"too many requests|\b429\b|reset by peer|broken pipe|"
    r"econnreset|econnrefused|i/o error|read error|"
    r"tim(?:e|ed)\s?out")


def is_retryable_failure(error: str, task_type: str = "") -> bool:
    """Whether a worker failure is *transient* — worth a Celery retry.

    需求#17: the batch executor used to ``raise self.retry`` on ANY
    exception, so a deterministic failure (unresolvable id, parameter /
    parse error) burned the whole retry budget before terminal failure.
    Transient causes (remote 5xx, transport timeouts, credential errors —
    retry can help) return True; everything else is treated as
    deterministic and must terminate immediately.  Deterministic-first is
    deliberate: an unclassifiable error retried once may succeed, but the
    reported incidents (repeated same-cause failures) are exactly the
    deterministic class this guard exists to stop.
    """
    del task_type  # policy is marker-based for now; type reserved for future.
    text = str(error or "")
    if not text:
        return False
    code = classify_failure_reason_code(text)
    if code == ERR_EPO_REMOTE:
        return True
    if code == ERR_UNRESOLVABLE_ID:
        return False
    return bool(_EXTRA_TRANSIENT_RE.search(text))


def notify_terminal_failure(
    task_id: str,
    error: str,
    *,
    reason_code: str | None = None,
    task_type: str = "",
    lang: str = "zh",
) -> None:
    """Mark *task_id* failed AND surface a guidance message in its conversation.

    Only call at TERMINAL failure points (retries exhausted, hard stop,
    an explicit failed pipeline result) — ``set_task_failed`` alone is
    also used on retryable attempts and must NOT spam the conversation.

    The conversation message is composed from :func:`failure_guidance` keyed
    by a structured ``reason_code``.  ``reason_code`` should be supplied by the
    call site when it knows the failure class; otherwise
    :func:`classify_failure_reason_code` classifies the worker's *error* text
    (never matching user-query vocabulary).  The message leads with the
    actionable next-step rather than mechanically re-printing the raw error
    (spec §1.3 / §5.4).  ``set_task_failed`` fires the analytics failure event
    exactly once here for each failed task.
    """
    set_task_failed(task_id, error)
    code = reason_code or classify_failure_reason_code(error)
    lang = lang if lang in ("zh", "en") else "zh"
    content = failure_guidance(task_type, code, error=error, lang=lang)
    try:
        from sources.long_task.task_messages import append_task_message
        append_task_message(task_id, event='failed', content=content)
    except Exception:
        pass  # conversation write-back must never break task state


def failure_guidance(
    task_type: str, reason_code: str, error: str = "", lang: str = "zh",
) -> str:
    """Compose the generic next-step guidance for a blocked / failed flow.

    Shared (spec §5.4, §6.3) by the resolvability pre-check replies (chat
    main chain / submit / retry / detail routes — produced as a guidance
    reply or error, never a task) and by the worker terminal-failure message
    (Task 5 references this same function).

    ``reason_code`` selects the guidance segment; any unrecognised code falls
    back to :data:`ERR_OTHER` (never raises).  ``error`` is only echoed for the
    generic ERR_OTHER tail and is bounded to 500 chars so an oversized worker
    error cannot blow up the message nor misbehave.  The returned text carries
    no user-query vocabulary — it only states next-step options.
    """
    del task_type  # template is task-type-agnostic for now.
    code = reason_code if reason_code in _GUIDE_SEG else _GUIDE_DEFAULT_CODE
    seg = _GUIDE_SEG[code].get(lang if lang in ("zh", "en") else "zh")
    head = "批量分析失败。\n\n" if code is ERR_OTHER else ""
    tail = f"失败原因：{str(error)[:500]}" if (code is ERR_OTHER and error) else ""
    return head + seg + (f"\n\n{tail}" if tail else "")


def _lookup_task_user_id(task_id: str) -> str | None:
    """Fallback: resolve the task's user_id from MySQL long_tasks.

    Only reached when the status record carries no user_id (older
    submissions).  Failure degrades silently — analytics reporting must
    never block task state transitions.
    """
    try:
        from sources.knowledge.knowledge import get_db_connection
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT user_id FROM long_tasks WHERE task_id = %s",
                    (task_id,))
                row = cur.fetchone()
            return str(row['user_id']) if row and row.get('user_id') else None
        finally:
            conn.close()
    except Exception:
        return None


def set_task_failed(task_id: str, error: str) -> None:
    """Mark task as failed with error message.

    Every failure is reported to analytics (需求 4: 失败事件统一覆盖) —
    previously only a few call sites tracked ``long_task:fail``, so silent
    failures (e.g. no_patents_found paths) were invisible in analytics.
    """
    r = _get_redis()
    raw = r.get(_status_key(task_id))
    status = json.loads(raw) if raw else {}
    status['status'] = 'failed'
    status['error_message'] = error
    r.set(_status_key(task_id), json.dumps(status, ensure_ascii=False),
          ex=TASK_STATUS_TTL)

    user_id = status.get('user_id') or _lookup_task_user_id(task_id)
    if not user_id:
        return
    try:
        from sources.analytics import track_event
        track_event("long_task:fail", user_id=str(user_id),
                    task_id=task_id, extra={"error": str(error)[:200]})
    except Exception:
        pass  # Analytics failure must never break task state


# ── Pause / Resume ──────────────────────────────────────────────────────────

PAUSE_FLAG_TTL = 86400  # 24 h


def _pause_key(task_id: str) -> str:
    return f"{TASK_STATUS_PREFIX}:{task_id}:paused"


def _stop_key(task_id: str) -> str:
    return f"{TASK_STATUS_PREFIX}:{task_id}:stopped"


def is_task_paused(task_id: str) -> bool:
    """Check whether a pause has been requested for *task_id*."""
    r = _get_redis()
    return r.exists(_pause_key(task_id)) > 0


def request_task_pause(task_id: str) -> None:
    """Signal the running task to pause at its next checkpoint."""
    import time
    r = _get_redis()
    r.set(_pause_key(task_id), '1', ex=PAUSE_FLAG_TTL)
    # Update status so the frontend sees the transition immediately
    raw = r.get(_status_key(task_id))
    status = json.loads(raw) if raw else {}
    status['status'] = 'paused'
    status['last_update'] = time.time()
    r.set(_status_key(task_id), json.dumps(status, ensure_ascii=False),
          ex=TASK_STATUS_TTL)


def clear_task_pause(task_id: str) -> None:
    """Clear the pause flag (used on resume)."""
    r = _get_redis()
    r.delete(_pause_key(task_id))


def is_task_stopped(task_id: str) -> bool:
    """Check whether a stop has been requested for *task_id*."""
    r = _get_redis()
    return r.exists(_stop_key(task_id)) > 0


def request_task_stop(task_id: str) -> None:
    """Signal the running task to stop at its next checkpoint."""
    import time
    r = _get_redis()
    r.set(_stop_key(task_id), '1', ex=PAUSE_FLAG_TTL)
    # Update status so the frontend sees the transition immediately
    raw = r.get(_status_key(task_id))
    status = json.loads(raw) if raw else {}
    status['status'] = 'cancelling'
    status['last_update'] = time.time()
    r.set(_status_key(task_id), json.dumps(status, ensure_ascii=False),
          ex=TASK_STATUS_TTL)


class ThrottledSummaryUpdater:
    """Push partial result_summary to Redis without flooding on every LLM token."""

    __slots__ = ('task_id', 'phase', 'progress', 'step_msg', '_last_ts', '_interval')

    def __init__(
        self,
        task_id: str,
        phase: str = 'generating_report',
        progress: int = 76,
        step_msg: str = '',
        interval: float = 0.8,
    ):
        self.task_id = task_id
        self.phase = phase
        self.progress = progress
        self.step_msg = step_msg
        self._last_ts = 0.0
        self._interval = interval

    def push(
        self,
        summary: str,
        *,
        progress: int | None = None,
        step_msg: str | None = None,
        force: bool = False,
    ) -> None:
        import time
        now = time.time()
        if not force and now - self._last_ts < self._interval:
            return
        self._last_ts = now
        if progress is not None:
            self.progress = progress
        if step_msg is not None:
            self.step_msg = step_msg
        update_task_status(
            self.task_id,
            self.phase,
            self.progress,
            self.step_msg,
            result_summary=summary,
        )


# ── Query → task recovery (SSE disconnect) ───────────────────────────────────

QUERY_TASK_TTL = 3600  # 1 hour


def _query_task_key(user_id: str, query_id: str) -> str:
    return f"{TASK_STATUS_PREFIX}:query:{user_id}:{query_id}"


def register_query_task(
    user_id: str,
    query_id: str,
    task_id: str,
    session_id: str,
    queue_status: str = 'running',
) -> None:
    """Map a client query_id to a long task for post-disconnect recovery."""
    import time
    r = _get_redis()
    payload = {
        'task_id': task_id,
        'session_id': session_id,
        'status': queue_status,
        'registered_at': time.time(),
    }
    r.set(
        _query_task_key(str(user_id), query_id),
        json.dumps(payload, ensure_ascii=False),
        ex=QUERY_TASK_TTL,
    )


def lookup_query_task(user_id: str, query_id: str) -> dict | None:
    """Return task metadata registered for *query_id*, if any."""
    r = _get_redis()
    raw = r.get(_query_task_key(str(user_id), query_id))
    if not raw:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode()
    return json.loads(raw)


def set_task_cancelled(task_id: str) -> None:
    """Mark task as cancelled and clean up its Redis keys."""
    r = _get_redis()
    raw = r.get(_status_key(task_id))
    status = json.loads(raw) if raw else {}
    status['status'] = 'cancelled'
    status['progress'] = 0
    r.set(_status_key(task_id), json.dumps(status, ensure_ascii=False),
          ex=TASK_STATUS_TTL)
    # Clean up pause/stop flags and checkpoint
    r.delete(_pause_key(task_id))
    r.delete(_stop_key(task_id))
    r.delete(_checkpoint_key(task_id))
