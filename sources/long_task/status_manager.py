import json

TASK_STATUS_PREFIX = "lt"
TASK_CHECKPOINT_PREFIX = "lt"
TASK_STATUS_TTL = 86400  # 24 hours


def _get_redis():
    from sources.knowledge.knowledge import get_redis_connection
    return get_redis_connection()


def _status_key(task_id: str) -> str:
    return f"{TASK_STATUS_PREFIX}:{task_id}:status"


def _checkpoint_key(task_id: str) -> str:
    return f"{TASK_CHECKPOINT_PREFIX}:{task_id}:checkpoint"


def update_task_status(task_id: str, phase: str, progress: int,
                       step_msg: str, status: str = 'running', **extra) -> None:
    """Write current task status to Redis."""
    import time
    r = _get_redis()
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
                      patent_ids: list | None = None) -> None:
    """Mark task as completed with report file metadata and optional patent IDs."""
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


def set_task_failed(task_id: str, error: str) -> None:
    """Mark task as failed with error message."""
    r = _get_redis()
    raw = r.get(_status_key(task_id))
    status = json.loads(raw) if raw else {}
    status['status'] = 'failed'
    status['error_message'] = error
    r.set(_status_key(task_id), json.dumps(status, ensure_ascii=False),
          ex=TASK_STATUS_TTL)


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
