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
                       step_msg: str, **extra) -> None:
    """Write current task status to Redis."""
    import time
    r = _get_redis()
    status = {
        'task_id': task_id,
        'status': 'running',
        'current_phase': phase,
        'progress': progress,
        'current_step': step_msg,
        'last_update': time.time(),
        **extra,
    }
    r.set(_status_key(task_id), json.dumps(status, ensure_ascii=False),
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


def set_task_completed(task_id: str, report_files: list) -> None:
    """Mark task as completed with report file metadata."""
    r = _get_redis()
    raw = r.get(_status_key(task_id))
    status = json.loads(raw) if raw else {}
    status['status'] = 'completed'
    status['progress'] = 100
    status['report_files'] = report_files
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
