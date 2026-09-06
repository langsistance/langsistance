"""Long-task lifecycle messages inside the conversation (M1, 2026-09-03).

A long task runs out-of-band in Celery; until now nothing ever wrote its
outcome back into ``conversations.messages``, so a follow-up like "分析
这些被驳回的专利失败原因" had no task results in the agent's memory.
This module bridges that gap with three pure-ish helpers:

- ``append_task_message``     — persist one task event (created/completed/
  failed) as an assistant-role message on the task's session.
- ``build_result_digest``     — derive a bounded, chat-usable digest from
  the full report text stored in the Redis status (sticky result_summary).
- ``hydrate_session_task_messages`` — inject task messages the frontend has
  not seen yet into the conversation history of the next chat turn
  (dedupe key = task_id#seq), so results are discussable even without a
  page reload.

Every helper degrades silently on any DB/Redis failure — task state
transitions and chat must never break because a write-back failed.
"""

import json

from sources.logger import Logger

_logger = Logger("task_messages.log")

# Bound for the digest pasted into conversation history (context budget).
DIGEST_MAX_CHARS = 9000
# Failure reason bound inside the failed message.
FAILURE_MAX_CHARS = 500

_TASK_MESSAGE_KIND = "long_task"


def _lookup_task_session(task_id: str) -> tuple | None:
    """Return (session_id, user_id) for *task_id*, or None on any failure."""
    try:
        from sources.knowledge.knowledge import get_db_connection
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT session_id, user_id FROM long_tasks "
                    "WHERE task_id = %s",
                    (task_id,))
                row = cur.fetchone()
            if not row:
                return None
            return (row.get("session_id") or "",
                    row.get("user_id") or "")
        finally:
            conn.close()
    except Exception as exc:
        _logger.warning(
            f"task_messages lookup failed for {task_id}: {exc}")
        return None


def _existing_seq(messages: list, task_id: str) -> int:
    """Next seq for *task_id* inside an existing message list."""
    seq = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        meta = m.get("meta")
        if (isinstance(meta, dict)
                and meta.get("kind") == _TASK_MESSAGE_KIND
                and meta.get("task_id") == task_id):
            try:
                seq = max(seq, int(meta.get("seq") or 0))
            except (TypeError, ValueError):
                continue
    return seq + 1


def append_task_message(
    task_id: str,
    *,
    event: str,
    content: str,
    patent_ids: list | None = None,
    report_files: list | None = None,
    patent_data: list | None = None,
) -> None:
    """Append one task lifecycle message to the task's session.

    ``event`` is one of ``created`` / ``completed`` / ``failed``.  The
    stored entry mirrors assistant messages the frontend already renders
    (role + content) and carries a ``meta`` marker plus top-level
    ``patent_ids`` so the conversation_refs machinery (which reads
    ``msg.get('patent_ids')`` on assistant messages) keeps working.

    Never raises: any failure is logged and skipped.
    """
    if not task_id or not content:
        return
    try:
        found = _lookup_task_session(task_id)
        if not found:
            _logger.warning(
                f"task_messages skip {task_id}: no long_tasks row")
            return
        session_id, user_id = found
        if not session_id or not user_id:
            return

        from sources.knowledge.knowledge import get_db_connection
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, messages FROM conversations "
                    "WHERE session_id = %s AND user_id = %s AND status != 2",
                    (session_id, user_id))
                row = cur.fetchone()
                if not row:
                    return
                conv_id = row["id"]
                raw_messages = row.get("messages")
                if isinstance(raw_messages, str) and raw_messages:
                    try:
                        messages = json.loads(raw_messages)
                    except (json.JSONDecodeError, TypeError):
                        messages = []
                elif isinstance(raw_messages, list):
                    messages = raw_messages
                else:
                    messages = []
                if not isinstance(messages, list):
                    messages = []

                entry = {
                    "role": "assistant",
                    "content": content,
                    "meta": {
                        "kind": _TASK_MESSAGE_KIND,
                        "event": event,
                        "task_id": task_id,
                        "seq": _existing_seq(messages, task_id),
                    },
                }
                if patent_ids:
                    entry["patent_ids"] = list(patent_ids)
                if report_files:
                    entry["report_files"] = list(report_files)
                if patent_data:
                    entry["patent_data"] = list(patent_data)[:50]

                messages = messages + [entry]
                cur.execute(
                    "UPDATE conversations SET messages = %s WHERE id = %s",
                    (json.dumps(messages, ensure_ascii=False), conv_id))
                conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        _logger.warning(
            f"task_messages append failed for {task_id}: {exc}")


def _truncate_markdown(text: str, cap: int = DIGEST_MAX_CHARS) -> str:
    """Cut *text* at the next markdown heading after ``cap``*0.5 chars.

    Truncating mid-table produces garbage for the LLM; cutting at a
    heading boundary keeps the digest readable.  Falls back to the last
    newline inside the cap when no heading follows.
    """
    if len(text) <= cap:
        return text
    head = text[:cap]
    head_start = max(1, cap // 2)
    next_heading = head.find("\n#", head_start)
    if next_heading != -1:
        return head[:next_heading]
    last_nl = head.rfind("\n")
    return head[:last_nl] if last_nl > 0 else head


def _clamp_target(text: str, cap: int = 120) -> str:
    """Trim *text* to ``cap`` chars, marking truncation with an ellipsis."""
    text = (text or "").strip()
    return text[:cap] + ("..." if len(text) > cap else "")


def build_result_digest(task_id: str) -> str:
    """Build a bounded, chat-usable completion digest for *task_id*.

    Source is the sticky ``result_summary`` (the full report text) on the
    task's Redis status; the digest keeps its head (executive summary /
    lead tables) and marks the truncation so the user knows the full
    report is still downloadable.
    """
    try:
        from sources.long_task.status_manager import get_task_status
        status = get_task_status(task_id)
    except Exception as exc:
        _logger.warning(
            f"task_messages digest status read failed for {task_id}: {exc}")
        status = {}
    if not isinstance(status, dict):
        status = {}

    patent_ids = status.get("patent_ids") or []
    report_files = status.get("report_files") or []
    summary = status.get("result_summary") or ""
    if not isinstance(summary, str):
        summary = str(summary)

    count_note = f"共 {len(patent_ids)} 件" if patent_ids else "批量分析"
    target = _clamp_target(str(status.get("target_name") or ""))
    if not summary:
        # No report text (e.g. tasks that only export files) — degrade to
        # whatever the status carries.  A target row is prepended only when
        # the executor recorded a target (Task 3); otherwise output matches
        # the pre-target wording exactly (backward compatible).
        parts = []
        if target:
            parts.append(f"任务已完成 —— 目标：{target}。")
        parts.append(f"批量分析任务已完成（{count_note}）。")
        if report_files:
            parts.append("报告文件：")
            parts.extend(f"- {f}" for f in report_files)
        return "\n".join(parts)

    body = _truncate_markdown(summary)
    truncated = len(summary) > len(body)
    head = f"任务已完成 —— 目标：{target}。\n\n" if target else ""
    digest = head + (
        f"批量分析任务已完成（{count_note}）。\n\n" + body
        if not target else
        f"共 {len(patent_ids)} 件结果摘要如下：\n\n" + body
    )
    if truncated:
        digest += (
            "\n\n…（以上为摘要截断，完整报告请在任务面板查看或下载）"
        )
    return digest


def _task_message_key(message: dict) -> str | None:
    """Dedupe key (task_id#seq) for a stored task message, or None."""
    if not isinstance(message, dict):
        return None
    meta = message.get("meta")
    if not (isinstance(meta, dict)
            and meta.get("kind") == _TASK_MESSAGE_KIND):
        return None
    task_id = meta.get("task_id")
    seq = meta.get("seq")
    if not task_id or seq is None:
        return None
    return f"{task_id}#{seq}"


def hydrate_session_task_messages(
    session_id: str,
    user_id,
    provided_history: list | None,
) -> list:
    """Merge stored task messages the frontend has not seen into history.

    Frontends send their in-memory message list as ``conversation_history``
    on every chat turn; task events written by the backend (created /
    completed / failed) are unknown to that list until a page reload.  This
    helper reads the session's stored messages and appends only the task
    messages whose ``task_id#seq`` key is missing, so a just-finished
    analysis is discussable on the very next turn.

    Returns a NEW list; ``provided_history`` is never mutated.
    """
    history = list(provided_history or [])
    if not session_id or not user_id:
        return history

    seen = set()
    for m in history:
        if not isinstance(m, dict):
            continue
        key = _task_message_key(m)
        if key:
            seen.add(key)

    try:
        from sources.knowledge.knowledge import get_db_connection
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT messages FROM conversations "
                    "WHERE session_id = %s AND user_id = %s AND status != 2",
                    (session_id, str(user_id)))
                row = cur.fetchone()
        finally:
            conn.close()
    except Exception as exc:
        _logger.warning(
            f"task_messages hydrate read failed for {session_id}: {exc}")
        return history

    if not row:
        return history
    raw_messages = row.get("messages")
    if isinstance(raw_messages, str) and raw_messages:
        try:
            stored = json.loads(raw_messages)
        except (json.JSONDecodeError, TypeError):
            return history
    elif isinstance(raw_messages, list):
        stored = raw_messages
    else:
        return history
    if not isinstance(stored, list):
        return history

    added = 0
    for m in stored:
        if not isinstance(m, dict):
            continue
        key = _task_message_key(m)
        if not key or key in seen:
            continue
        history.append(m)
        seen.add(key)
        added += 1
    if added:
        _logger.info(
            f"task_messages hydrated {added} message(s) into "
            f"session {session_id}"
        )
    return history
