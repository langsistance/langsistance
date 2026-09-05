"""Session anchor (需求#7, P2).

A session-scoped "current object" pointer (Redis) so follow-up turns can
resolve deixis ("给出相似文件的申请号") against the task that produced the
session's results.  The anchor is derived state — receipts are permanently
persisted in ``conversations.messages``; the Redis key only accelerates
lookup and is rebuilt from the latest completed receipt when missing.

Every helper degrades silently — anchor plumbing must never break chat or
task state.
"""

import json
import re
import time

from sources.logger import Logger

_logger = Logger("session_anchor.log")

ANCHOR_KEY_PREFIX = "sess"
ANCHOR_TTL = 86400  # 24h, sliding (refreshed on every successful load)
REUSE_WINDOW_SEC = 600  # R1: same-file re-upload within 10 min reuses

MAX_RESULT_IDS = 50
MAX_TARGET = 200
MAX_SUMMARY = 600
MAX_TITLE = 80


def _key(session_id: str) -> str:
    return f"{ANCHOR_KEY_PREFIX}:{session_id}:anchor"


def _get_redis():
    from sources.knowledge.knowledge import get_redis_connection
    return get_redis_connection()


def _clamp(text: str, cap: int) -> str:
    text = (text or "").strip()
    return text[:cap] + ("..." if len(text) > cap else "")


def _norm_filename(name: str) -> str:
    """Normalize a filename for R1 equality checks (basename, case, ext)."""
    base = re.sub(r"\.(pdf|docx?|xml)$", "", (name or ""), flags=re.I)
    return base.strip().lower()


def write_session_anchor(
    session_id: str,
    *,
    anchor_type: str,
    target: str,
    target_summary: str,
    source: str,
    result_ids: list,
    result_titles: dict | None,
    task_id: str,
) -> bool:
    """Write (or overwrite) the session anchor.  Returns False on failure."""
    if not session_id or not task_id:
        return False
    try:
        titles = result_titles or {}
        ids = [str(pid).strip() for pid in (result_ids or [])
               if str(pid).strip()][:MAX_RESULT_IDS]
        payload = {
            "version": 1,
            "anchor_type": anchor_type,
            "session_id": session_id,
            "task_id": task_id,
            "target": _clamp(target, MAX_TARGET),
            "target_summary": _clamp(target_summary, MAX_SUMMARY),
            "source": source or "",
            "result_ids": ids,
            "result_titles": {
                str(k): _clamp(str(v), MAX_TITLE) for k, v in titles.items()
                if str(k) in ids
            },
            "updated_at": time.time(),
        }
        r = _get_redis()
        r.set(_key(session_id), json.dumps(payload, ensure_ascii=False),
              ex=ANCHOR_TTL)
        return True
    except Exception as exc:
        _logger.warning(f"session_anchor write failed for {session_id}: {exc}")
        return False


def load_session_anchor(session_id: str) -> dict | None:
    """Read the session anchor, sliding-refreshing its TTL on success."""
    if not session_id:
        return None
    try:
        r = _get_redis()
        raw = r.get(_key(session_id))
        if not raw:
            # Key miss → degradation step ②: rebuild from the latest completed
            # receipt, then persist the rebuilt anchor so later turns hit the
            # cache.  rebuild_anchor_from_messages swallows its own errors and
            # returns None (never recursing into load), so this stays inside
            # the silent-degradation envelope.
            anchor = rebuild_anchor_from_messages(session_id)
            if anchor is None:
                return None
            payload = json.dumps(anchor, ensure_ascii=False)
            r.set(_key(session_id), payload, ex=ANCHOR_TTL)
            return anchor
        if isinstance(raw, bytes):
            raw = raw.decode()
        anchor = json.loads(raw)
        if not isinstance(anchor, dict):
            return None
        # Sliding TTL: keep the anchor alive while the session is active.
        r.set(_key(session_id), raw, ex=ANCHOR_TTL)
        return anchor
    except Exception as exc:
        _logger.warning(f"session_anchor load failed for {session_id}: {exc}")
        return None


def rebuild_anchor_from_messages(session_id: str) -> dict | None:
    """Rebuild an anchor from the latest completed task receipt message.

    Receipt messages are assistant-role entries with
    ``meta.kind == "long_task"`` / ``meta.event == "completed"`` (written by
    task_messages.append_task_message).  Returns None when no receipt exists
    (caller then behaves exactly as before anchors existed).
    """
    if not session_id:
        return None
    try:
        from sources.knowledge.knowledge import get_db_connection
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT messages FROM conversations "
                    "WHERE session_id = %s AND status != 2", (session_id,))
                row = cur.fetchone()
        finally:
            conn.close()
        if not row:
            return None
        raw = row.get("messages")
        if isinstance(raw, str) and raw:
            messages = json.loads(raw)
        elif isinstance(raw, list):
            messages = raw
        else:
            return None
        if not isinstance(messages, list):
            return None
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            meta = msg.get("meta")
            if not (isinstance(meta, dict)
                    and meta.get("kind") == "long_task"
                    and meta.get("event") == "completed"):
                continue
            task_id = str(meta.get("task_id") or "")
            patent_ids = [str(p).strip() for p in (msg.get("patent_ids") or [])
                          if str(p).strip()][:MAX_RESULT_IDS]
            if not task_id:
                continue
            return {
                "version": 1,
                "anchor_type": "topic",
                "session_id": session_id,
                "task_id": task_id,
                "target": _clamp(str(msg.get("content") or "")[:200], MAX_TARGET),
                "target_summary": "",
                "source": "",
                "result_ids": patent_ids,
                "result_titles": {},
                "updated_at": time.time(),
                "_rebuilt": True,
            }
        return None
    except Exception as exc:
        _logger.warning(
            f"session_anchor rebuild failed for {session_id}: {exc}")
        return None


def build_anchor_block(anchor: dict | None) -> str:
    """Render the anchor as a reference-only system-prompt section.

    Empty string when no anchor — callers splice the block verbatim.
    """
    if not isinstance(anchor, dict):
        return ""
    target = _clamp(anchor.get("target") or "", MAX_TARGET)
    summary = _clamp(anchor.get("target_summary") or "", 200)
    ids = (anchor.get("result_ids") or [])[:20]
    lines = [
        "\n\n## 当前会话任务锚点（本会话最近完成的上传/分析目标；"
        "仅当用户指代它时使用）",
        f"- 目标：{target}（{anchor.get('anchor_type') or 'topic'}，"
        f"来源 {anchor.get('source') or '-'}）",
        f"- 结果：{', '.join(ids) if ids else '无'}"
        f"（共 {len(anchor.get('result_ids') or [])} 件）",
    ]
    if summary:
        lines.append(f"- 摘要：{summary}")
    return "\n".join(lines)


def suggest_reuse(
    session_id: str, filename: str, *, window_sec: int = REUSE_WINDOW_SEC,
) -> dict | None:
    """R1: same-session same-file re-upload hint within the time window."""
    anchor = load_session_anchor(session_id)
    if not anchor or anchor.get("anchor_type") != "file":
        return None
    if _norm_filename(anchor.get("target")) != _norm_filename(filename):
        return None
    try:
        updated = float(anchor.get("updated_at") or 0)
    except (TypeError, ValueError):
        updated = 0
    if time.time() - updated > window_sec:
        return None
    return anchor
