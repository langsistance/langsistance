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


def _is_task_terminal(task_id: str) -> bool:
    """Check if a task has reached a terminal state (completed, failed, stuck, or timed out)."""
    try:
        import time
        from sources.long_task.status_manager import get_task_status
        status = get_task_status(task_id)
        if not status:
            # Status key doesn't exist — task never started or was cleaned up
            return True
        task_status = status.get('status', '')
        if task_status in ('completed', 'failed'):
            return True
        if task_status in ('queued', 'pending'):
            # Task is marked as 'running' in the user queue but its actual status
            # is still queued/pending — it was dequeued but never dispatched.
            return True
        # Heartbeat check: if the task hasn't updated its status in 5+ minutes,
        # it's likely dead or stuck (e.g. Celery worker crashed, infinite retry).
        last_update = status.get('last_update', 0)
        if time.time() - last_update > 300:
            return True
        return False
    except Exception:
        return True  # If we can't check, assume stale


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
    if current:
        current_str = current.decode() if isinstance(current, bytes) else current
        if current_str != task_id:
            # Check if the "running" task is actually dead (crashed without cleanup)
            if _is_task_terminal(current_str):
                _logger.info(
                    f"[queue] stale_lock_detected — running_task={current_str} "
                    f"is terminal, taking over"
                )
                r.delete(running_key)
            else:
                r.rpush(queue_key, task_id)
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


def requeue_paused_task(user_id: str, task_id: str) -> str:
    """Re-queue a paused task at the back of the user's queue.

    If no task is currently running, it starts immediately.
    Otherwise it waits behind the running task and any already-queued tasks.
    """
    import time as _time
    r = _r()
    running_key = _USER_RUNNING_KEY.format(user_id=user_id)
    queue_key = _USER_QUEUE_KEY.format(user_id=user_id)

    current = r.get(running_key)
    current_str = current.decode() if isinstance(current, bytes) else current

    from sources.long_task.status_manager import update_task_status

    if current_str and current_str != task_id and not _is_task_terminal(current_str):
        # Someone else is running — queue behind them
        r.rpush(queue_key, task_id)
        update_task_status(task_id, "queued", 0, "排队中，等待当前任务完成...")
        _logger.info(
            f"[queue] task_requeued — user_id={user_id}, task_id={task_id}, "
            f"behind={current_str}"
        )
        return "queued"
    else:
        # No one running (or stale lock) — start immediately
        if current_str:
            r.delete(running_key)
        r.set(running_key, task_id, ex=86400)
        update_task_status(task_id, "pending", 0, "即将开始执行...")
        _logger.info(
            f"[queue] task_resumed_immediate — user_id={user_id}, task_id={task_id}"
        )
        return "running"
