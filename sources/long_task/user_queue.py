"""Per-user long task queue using Redis.

Only one long task runs per user at a time.  Additional tasks are enqueued
and dispatched automatically when the running task completes.
"""

import json as _json
from sources.logger import Logger

_logger = Logger("long_task_pipeline.log")

# ── Redis key templates ──────────────────────────────────────────────────
_USER_RUNNING_KEY = "lt:user:{user_id}:running"
_USER_QUEUE_KEY  = "lt:user:{user_id}:queue"


def _r():
    """Lazily get a Redis connection."""
    from sources.knowledge.knowledge import get_redis_connection
    return get_redis_connection()


def try_start_user_task(user_id: str, task_id: str) -> str:
    """Attempt to start a long task for *user_id*.

    Returns ``"running"`` if the task is dispatched immediately, or
    ``"queued"`` if the user already has a running task and this one was
    placed at the back of their queue.
    """
    r = _r()
    running_key = _USER_RUNNING_KEY.format(user_id=user_id)
    queue_key = _USER_QUEUE_KEY.format(user_id=user_id)

    current = r.get(running_key)
    if current and (current.decode() if isinstance(current, bytes) else current) != task_id:
        r.rpush(queue_key, task_id)
        # Set an initial Redis status so the frontend polling can see the
        # task exists (otherwise it would show "unknown" until the worker
        # picks it up).
        from sources.long_task.status_manager import update_task_status as _uts
        _uts(task_id, "queued", 0, "排队中，等待当前任务完成...")
        _logger.info(
            f"[queue] task_queued — user_id={user_id}, task_id={task_id}"
        )
        return "queued"

    r.set(running_key, task_id, ex=86400)  # 24 h TTL
    _logger.info(
        f"[queue] task_running — user_id={user_id}, task_id={task_id}"
    )
    return "running"


def complete_user_task(user_id: str, task_id: str) -> str | None:
    """Mark *task_id* as complete and dispatch the next queued task.

    Returns the *task_id* of the next task to run, or ``None`` if the
    user's queue is empty.
    """
    r = _r()
    running_key = _USER_RUNNING_KEY.format(user_id=user_id)
    queue_key = _USER_QUEUE_KEY.format(user_id=user_id)

    current = r.get(running_key)
    current_str = current.decode() if isinstance(current, bytes) else current
    if current_str and current_str == task_id:
        r.delete(running_key)

    next_raw = r.lpop(queue_key)
    if next_raw:
        next_id = next_raw.decode() if isinstance(next_raw, bytes) else next_raw
        r.set(running_key, next_id, ex=86400)
        _logger.info(
            f"[queue] task_dequeued — user_id={user_id}, "
            f"next_task_id={next_id}"
        )
        return next_id

    _logger.info(
        f"[queue] task_completed_and_queue_empty — "
        f"user_id={user_id}, task_id={task_id}"
    )
    return None


def get_user_queue_status(user_id: str) -> dict:
    """Return the user's current queue state (for frontend polling)."""
    r = _r()
    running_key = _USER_RUNNING_KEY.format(user_id=user_id)
    queue_key  = _USER_QUEUE_KEY.format(user_id=user_id)

    current = r.get(running_key)
    current_str = current.decode() if isinstance(current, bytes) else current

    queue_len = r.llen(queue_key)

    return {
        "running_task_id": current_str or None,
        "queue_length": queue_len,
    }
