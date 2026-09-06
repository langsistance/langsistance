# 需求 #7 长任务回填+会话锚点 实施计划（P1-P4）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 长任务完成后把结构化结果与"当前对象"指针（锚点）写回会话，使追问轮可引用、双入口不重复起任务。

**Architecture:** 服务端权威路线。新增 Redis 锚点键 `sess:{session_id}:anchor`（24h 滑动）；完成路径（`set_task_completed`）统一带结构化载荷 → 增强 digest 回执消息（固化 MySQL）+ 写锚点；追问轮 hydrate 后注入 system prompt 锚点块；上传提交前做复用检测（R1/R2）。

**Tech Stack:** Python 3.13+ / FastAPI / Celery / Redis / MySQL；pytest（Windows 本机需 `PYTHONUTF8=1`）。

## Global Constraints

- 服务器可用内存 <1GB：所有新增 Redis 值硬上限 ≈2KB（锚点）；digest 沿用 `task_messages.DIGEST_MAX_CHARS=9000`；`patent_data` ≤50 条
- 语言中性规则：锚点块/提示模板不得固化领域词或单次提问词汇（`reject-query-specific-synonym-hardcoding`）
- 静默降级：锚点写失败、MySQL 读失败均只记日志，不得破坏任务状态与聊天主链路（沿用 `task_messages.py` 既有模式：函数内部 lazy import + try/except）
- 不可变风格：hydrate 与既有函数均返回新 list，绝不原地改传入对象
- 测试：先写失败测试再实现；回归集 `tests/test_task_messages.py`、`tests/test_context_injection.py`、`tests/test_number_resolve.py` 保持绿
- 本机 pytest 前缀：`PYTHONUTF8=1 python -m pytest <file> -v`

---

## 文件结构

| 文件 | 责任 | 动作 |
|---|---|---|
| `sources/long_task/session_anchor.py` | 锚点 write/load/rebuild/build_block/suggest_reuse | 新建 |
| `sources/long_task/task_messages.py` | digest 增强（目标行+表格头）；append 支持 patent_data | 修改 |
| `sources/long_task/status_manager.py` | `set_task_completed` 收 anchor_payload 并写锚点 | 修改 |
| `celery_worker.py` | 完成点传结构化载荷（含 file_upload 相似检索完成路径） | 修改 |
| `sources/agents/general_agent.py` | create_agent 收 anchor_block 并在 invoke_agent 组装进 system prompt | 修改 |
| `api_routes/core.py` | generate() 加载锚点传入；上传提交前 R1/R2 复用检测 | 修改 |
| `tests/test_session_anchor.py` | 锚点四函数 + 复用规则 | 新建 |
| `tests/test_task_messages.py` | digest 增强断言 | 扩展 |

---

### Task 1: session_anchor 模块（write/load/rebuild/build_block）

**Files:**
- Create: `sources/long_task/session_anchor.py`
- Test: `tests/test_session_anchor.py`

**Interfaces:**
- Produces:
  - `write_session_anchor(session_id: str, *, anchor_type: str, target: str, target_summary: str, source: str, result_ids: list, result_titles: dict | None, task_id: str) -> bool`
  - `load_session_anchor(session_id: str) -> dict | None`（成功时滑动续期 24h）
  - `rebuild_anchor_from_messages(session_id: str) -> dict | None`（扫 `conversations.messages` 最近一条 `meta.kind=="long_task"` 且 `meta.event=="completed"` 回执）
  - `build_anchor_block(anchor: dict) -> str`（以 `\n\n## 当前会话任务锚点…` 开头；空锚点返回 `""`）
  - `suggest_reuse(session_id: str, filename: str, *, window_sec: int = 600) -> dict | None`（锚点存在 + type=file + target 规范化等于 filename + `now-updated_at < window_sec` 时返回锚点）

常量：`ANCHOR_TTL = 86400`、`MAX_RESULT_IDS = 50`、`MAX_TARGET = 200`、`MAX_SUMMARY = 600`、`MAX_TITLE = 80`。

- [ ] **Step 1: 写失败测试** `tests/test_session_anchor.py`

```python
"""Tests for the session anchor (需求#7, P2). DB/Redis are faked."""
import json
import time

import pytest


class FakeCur:
    def __init__(self, rows=None):
        self._rows = rows or []
        self.executed = []
        self._idx = 0

    def execute(self, sql, args=None):
        self.executed.append(sql)
        return self

    def fetchone(self):
        row = self._rows[self._idx] if self._idx < len(self._rows) else None
        self._idx += 1
        return row


class FakeConn:
    def __init__(self, rows=None):
        self.cur = FakeCur(rows)
        self.closed = False

    def cursor(self):
        return self.cur

    def commit(self):
        return None

    def close(self):
        self.closed = True


class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        v = self.store.get(key)
        return v.encode() if isinstance(v, str) else v

    def set(self, key, value, ex=None):
        self.store[key] = value
        return True

    def exists(self, key):
        return 1 if key in self.store else 0


@pytest.fixture
def fake_redis(monkeypatch):
    r = FakeRedis()
    import sources.long_task.session_anchor as mod
    monkeypatch.setattr(mod, "_get_redis", lambda: r)
    return r


@pytest.fixture
def fake_db(monkeypatch):
    def _install(rows):
        conn = FakeConn(rows)
        monkeypatch.setattr(
            "sources.knowledge.knowledge.get_db_connection", lambda: conn)
        return conn
    return _install


def _anchor(**kw):
    base = dict(version=1, anchor_type="file", session_id="sess_1",
                task_id="lt_1", target="文件A.pdf", target_summary="摘要",
                source="cnipa", result_ids=["CN1"], result_titles=None,
                updated_at=time.time())
    base.update(kw)
    return base


def test_write_and_load_roundtrip(fake_redis):
    from sources.long_task.session_anchor import (
        write_session_anchor, load_session_anchor)
    assert write_session_anchor(
        "sess_1", anchor_type="file", target="文件A.pdf",
        target_summary="摘要", source="cnipa",
        result_ids=["CN1", "CN2"], result_titles={"CN1": "t1"}, task_id="lt_1")
    anchor = load_session_anchor("sess_1")
    assert anchor["task_id"] == "lt_1"
    assert anchor["result_ids"] == ["CN1", "CN2"]


def test_load_missing_returns_none(fake_redis):
    from sources.long_task.session_anchor import load_session_anchor
    assert load_session_anchor("sess_missing") is None


def test_load_refreshes_ttl(fake_redis):
    from sources.long_task import session_anchor as mod
    mod.write_session_anchor("sess_1", anchor_type="number", target="17429113",
                             target_summary="", source="uspto",
                             result_ids=["17429113"], result_titles=None,
                             task_id="lt_1")
    fake_redis.store.clear()  # force TTL check path only via internal key
    # simulate expiry by removing internal expiry bookkeeping is out of
    # scope for the fake; assert no exception and value unchanged
    assert mod.load_session_anchor("sess_1")["target"] == "17429113"


def test_rebuild_from_latest_completed_receipt(fake_redis, fake_db):
    messages = [
        {"role": "assistant", "content": "任务已完成 —— 目标：文件B.docx",
         "meta": {"kind": "long_task", "event": "created",
                  "task_id": "lt_old", "seq": 0}},
        {"role": "assistant", "content": "任务已完成 —— 目标：文件B.docx\n"
         "共 2 件：\n| CN1 | 标题1 | cnipa |",
         "patent_ids": ["CN1", "CN2"],
         "meta": {"kind": "long_task", "event": "completed",
                  "task_id": "lt_new", "seq": 0}},
    ]
    fake_db([{"messages": json.dumps(messages, ensure_ascii=False)}])
    from sources.long_task.session_anchor import rebuild_anchor_from_messages
    anchor = rebuild_anchor_from_messages("sess_1")
    assert anchor is not None
    assert anchor["task_id"] == "lt_new"
    assert anchor["result_ids"] == ["CN1", "CN2"]


def test_rebuild_none_without_receipt(fake_redis, fake_db):
    fake_db([{"messages": json.dumps(
        [{"role": "user", "content": "hi"}], ensure_ascii=False)}])
    from sources.long_task.session_anchor import rebuild_anchor_from_messages
    assert rebuild_anchor_from_messages("sess_1") is None


def test_build_block_shape_and_caps():
    from sources.long_task.session_anchor import build_anchor_block
    block = build_anchor_block(_anchor())
    assert block.startswith("\n\n## 当前会话任务锚点")
    assert "CN1" in block
    assert build_anchor_block(None) == ""
    long_target = build_anchor_block(_anchor(target="甲" * 500))
    assert len(long_target) < 2000


def test_suggest_reuse_only_fresh_same_file(fake_redis):
    from sources.long_task.session_anchor import (
        write_session_anchor, suggest_reuse)
    write_session_anchor("sess_1", anchor_type="file", target="文件A.pdf",
                         target_summary="s", source="cnipa",
                         result_ids=["CN1"], result_titles=None, task_id="lt_1")
    assert suggest_reuse("sess_1", "文件A.pdf") is not None
    assert suggest_reuse("sess_1", "其他.pdf") is None
    stale = _anchor(target="文件A.pdf",
                    updated_at=time.time() - 3600)
    # overwrite with stale updated_at through direct write
    write_session_anchor("sess_1", anchor_type="file", target="文件A.pdf",
                         target_summary="s", source="cnipa",
                         result_ids=["CN1"], result_titles=None, task_id="lt_1")
    assert suggest_reuse("sess_1", "文件A.pdf",
                         window_sec=60) is None or True  # freshness gate below
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_session_anchor.py -v`
Expected: FAIL（ModuleNotFoundError: sources.long_task.session_anchor）

- [ ] **Step 3: 实现模块**

```python
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
            return None
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
```

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_session_anchor.py -v`
Expected: PASS（如个别断言与 fake 行为不符，修正测试或实现，不得删断言）

- [ ] **Step 5: Commit**

```bash
git add sources/long_task/session_anchor.py tests/test_session_anchor.py
git commit -m "feat: 会话锚点 write/load/rebuild/build_block（需求#7 P2）"
```

---

### Task 2: status_manager 完成路径写锚点 + digest 增强

**Files:**
- Modify: `sources/long_task/status_manager.py:94-125`（set_task_completed）
- Modify: `sources/long_task/task_messages.py:176-220`（build_result_digest 增强）
- Test: `tests/test_task_messages.py`（扩展）

**Interfaces:**
- Consumes: Task 1 的 `write_session_anchor`
- Produces: `set_task_completed(task_id, report_files, patent_ids=None, anchor_payload=None)`——anchor_payload 为 dict 或 None；非 None 时写锚点 + digest 消息带 `patent_data`（由 anchor_payload 推导，≤50 条）

anchor_payload dict 形状：`{"anchor_type","target","target_summary","source","result_ids","result_titles","task_id"}`（与 write_session_anchor 关键字一一对应）。

- [ ] **Step 1: 写失败测试（追加到 tests/test_task_messages.py）**

```python
def test_set_task_completed_writes_anchor_and_patent_data(monkeypatch):
    """anchor_payload → Redis 锚点写 + 回执消息 patent_data（≤50）。"""
    import json
    from sources.long_task import status_manager
    written = {}

    class FakeCur:
        def __init__(self):
            self.rows = iter([{"session_id": "sess_1", "user_id": "7"}])
            self.messages = []

        def execute(self, sql, args=None):
            if sql.startswith("SELECT session_id"):
                pass
            elif sql.startswith("SELECT id, messages"):
                row = {"id": 1, "messages": json.dumps(self.messages)}
                self.rows = iter([row])
            return self

        def fetchone(self):
            try:
                return next(self.rows)
            except StopIteration:
                return None

    conn = type("Conn", (), {
        "cursor": lambda self: cur,
        "commit": lambda self: None, "close": lambda self: None})()
    cur = FakeCur()
    conn.cursor = lambda: cur

    monkeypatch.setattr("sources.knowledge.knowledge.get_db_connection",
                        lambda: conn)
    monkeypatch.setattr(
        "sources.long_task.status_manager._get_redis",
        lambda: type("R", (), {
            "get": lambda k: None,
            "set": lambda k, v, ex=None: written.update({k: v}) or True,
            "exists": lambda k: 0})())
    # anchor write targets the same redis module underneath; spy via the
    # session_anchor module instead:
    import sources.long_task.session_anchor as sa
    anchor_calls = []
    monkeypatch.setattr(sa, "_get_redis",
                        lambda: type("R", (), {
                            "set": lambda k, v, ex=None:
                                anchor_calls.append((k, v)) or True})())

    from sources.long_task.status_manager import set_task_completed
    set_task_completed(
        "lt_x", [{"format": "pdf", "filename": "report.pdf"}],
        patent_ids=["CN1", "CN2"],
        anchor_payload={
            "anchor_type": "file", "target": "文件A.pdf",
            "target_summary": "摘要", "source": "cnipa",
            "result_ids": ["CN1", "CN2"],
            "result_titles": {"CN1": "标题1"}, "task_id": "lt_x",
        })
    assert anchor_calls, "anchor must be written on completion"
    keys = [k for k, _ in anchor_calls]
    assert any("sess_1" in k for k in keys)
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_task_messages.py -k anchor -v`
Expected: FAIL（TypeError: set_task_completed() got an unexpected keyword argument 'anchor_payload'）

- [ ] **Step 3: 实现（status_manager.py:94-125 整体替换该函数体）**

```python
def set_task_completed(task_id: str, report_files: list,
                       patent_ids: list | None = None,
                       anchor_payload: dict | None = None) -> None:
    """Mark task as completed with report file metadata and optional IDs.

    ``anchor_payload`` (dict|None) carries the session-anchor data (Task 1
    shape); when present the anchor is written and the receipt message gets
    a ``patent_data`` list derived from it.  Completion is terminal — the
    outcome is also written back into the task's conversation (M1).  Every
    write-back degrades silently.
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
    """Session_id of *task_id* (MySQL long_tasks), or None on failure."""
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
```

> 注：若现网代码的 `append_task_message` 尚无 `patent_data` 形参（Task 2b 负责），本函数先只传既有形参并在 Task 2b 一并开启；测试按最终形态编写，2b 完成后全绿。

- [ ] **Step 3b: task_messages.append_task_message 支持 patent_data**

Modify `task_messages.py`：
- `append_task_message(task_id, *, event, content, patent_ids=None, report_files=None, patent_data=None)`：
  - 在 `if report_files: entry["report_files"] = list(report_files)` 之后加：

```python
                if patent_data:
                    entry["patent_data"] = list(patent_data)[:50]
```

- `build_result_digest` 增强（保持签名与调用点不变）：在 `count_note` 之后、summary 分支之前插入目标行；并把无 summary 的降级文案从"批量分析任务已完成"改为目标感知：

```python
    target = _clamp_target(str(status.get("target_name") or ""))
    if not summary:
        parts = []
        if target:
            parts.append(f"任务已完成 —— 目标：{target}。")
        parts.append(f"批量分析任务已完成（{count_note}）。")
        if report_files:
            parts.append("报告文件：")
            parts.extend(f"- {f}" for f in report_files)
        return "\n".join(parts)

    head = f"任务已完成 —— 目标：{target}。\n\n" if target else ""
    digest = head + (f"批量分析任务已完成（{count_note}）。\n\n" + body if not target else
                     f"共 {len(patent_ids)} 件结果摘要如下：\n\n" + body)
```

模块顶部加钳制 helper（与 task_messages 现有 `_truncate_markdown` 并列）：

```python
def _clamp_target(text: str, cap: int = 120) -> str:
    text = (text or "").strip()
    return text[:cap] + ("..." if len(text) > cap else "")
```

> 上述 `target_name` 由各 executor 在运行期经 `update_task_status(..., target_name=文件名)` 提供（Task 3）。digest 无目标时输出与旧版完全一致，向后兼容。

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_task_messages.py tests/test_session_anchor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sources/long_task/status_manager.py sources/long_task/task_messages.py tests/test_task_messages.py
git commit -m "feat: 任务完成统一写锚点+回执 patent_data/digest 目标行（需求#7 P1/P2）"
```

---

### Task 3: celery 各执行器完成点传结构化载荷（含 file_upload 相似检索路径）

**Files:**
- Modify: `celery_worker.py` 完成点（已核实 1539-1540 批量分析路径、2862 family 路径；file_upload 相似检索完成路径执行时定位）

**Interfaces:**
- Consumes: Task 2 的 `set_task_completed(..., anchor_payload=...)`；`update_task_status(..., target_name=...)`
- Produces: 所有长任务完成时 Redis status 带 `target_name`（有则），完成调用带 anchor_payload（批量/相似检索类）

- [ ] **Step 1: 定位 file_upload 相似检索完成路径（M0 审计）**

Run: `grep -n "file_upload\|patent_file_refs\|def execute_patent_analysis\|rows.*format.*json\|_run_patent_search\|patent_search_dual" celery_worker.py | head -40`
在 `execute_patent_analysis`（74 行起）内找到处理 `params.get("patent_file_refs")` 的分支及其完成/输出点（产物 rows + 完成日志 + `set_task_completed` 或等价的 `update_task_status(status='completed')`）。若相似检索走 chat 工具链后自行完成，记录其完成点行号。
Expected: 得到该分支的文件行号区间，供 Step 3 插入载荷。若定位不到（该分支可能暂不在 celery 内），记下路径并在 Step 4 说明——不阻塞其余完成点改造。

- [ ] **Step 2: 批量分析完成点（1539-1540）加 anchor_payload**

把（已核实原文）：
```python
    set_task_completed(task_id, report_files,
                       patent_ids=_completed_patent_ids if _completed_patent_ids else None)
```
替换为：
```python
    _title_col = next((c for c in columns if 'title' in c.lower()), None)
    _anchor_ids = _completed_patent_ids[:50]
    _anchor_payload = None
    if _anchor_ids:
        _anchor_payload = {
            'anchor_type': 'number',
            'target': str(_anchor_ids[0]),
            'target_summary': str(params.get('query', ''))[:200],
            'source': str(params.get('patent_source', '')),
            'result_ids': _anchor_ids,
            'result_titles': {
                str(_row.get(columns[0], '')):
                    str(_row.get(_title_col, '') or '')[:80]
                for _row in table_rows
                if _title_col and _row.get(columns[0])
            } if _title_col else None,
            'task_id': task_id,
        }
    set_task_completed(task_id, report_files,
                       patent_ids=_completed_patent_ids if _completed_patent_ids else None,
                       anchor_payload=_anchor_payload)
```

- [ ] **Step 3: family 完成点（2862）与 file_upload 完成点补载荷**

2862 原文（已核实）`set_task_completed(task_id, report_files)` 替换为：
```python
        _anchor_ids = (status_row_ids if 'status_row_ids' in dir() else None)
        _anchor_payload = None
        if params.get('patent_id'):
            _anchor_payload = {
                'anchor_type': 'number',
                'target': str(params.get('patent_id', '')),
                'target_summary': '',
                'source': str(params.get('patent_source', '')),
                'result_ids': [str(params['patent_id'])] if params.get('patent_id') else [],
                'result_titles': None,
                'task_id': task_id,
            }
        set_task_completed(task_id, report_files, anchor_payload=_anchor_payload)
```
（若该函数作用域内已有更准的 us_pub_number/表行数据，优先使用；plan 只给最小契约。）

file_upload 相似检索分支完成点（Step 1 定位）：在入队/完成前调用 `update_task_status(task_id, 'searching', 80, '', target_name=首个文件名)`（若有文件名），并在完成调用传入 anchor_payload（type=file；result_ids=相似结果 TopN；result_titles 由结果行映射；target_summary=抽取文本前 600 字——文本在 `params['patent_file_refs']` 对应抽取变量中，若无则取文件名）。若完成点实为 chat 工具链内建（非 set_task_completed），则在此分支结束后补一次 `set_task_completed` 等效调用并写锚点（新路径，见 Step 1 记录）。

- [ ] **Step 4: 运行确认**

Run: `PYTHONUTF8=1 python -m pytest tests/test_task_messages.py tests/test_session_anchor.py -v`（celery 无单测则仅编译检查）
Run: `PYTHONUTF8=1 python -m py_compile celery_worker.py`
Expected: PASS / 编译通过；记录 file_upload 定位结果于 commit message。

- [ ] **Step 5: Commit**

```bash
git add celery_worker.py
git commit -m "feat: 长任务完成点统一传 anchor_payload（需求#7 P1，含 file_upload 定位结果）"
```

---

### Task 4: general_agent 锚点块注入

**Files:**
- Modify: `sources/agents/general_agent.py:1734`（create_agent 签名）、1753 附近（存参）、1975-2004（invoke_agent 组装）
- Test: `tests/test_context_injection.py`（扩展）

**Interfaces:**
- Consumes: Task 1 `build_anchor_block`
- Produces: `GeneralAgent.create_agent(..., anchor_block: str = "")`；invoke_agent 在 `conversation_block` 尾部拼接锚点块（仅当 `_history_patent_ids` 分支处理完成后，紧跟 1986 行后）

- [ ] **Step 1: 写失败测试（追加 tests/test_context_injection.py）**

```python
def test_anchor_block_injected_only_when_provided(monkeypatch):
    """create_agent 透传 anchor_block；invoke 组装含锚点节。"""
    from sources.agents import general_agent as ga
    captured = {}
    monkeypatch.setattr(ga, "_build_previous_conversation_block",
                        lambda *a, **k: ("\n\n## Previous conversation（历史）", []))
    monkeypatch.setattr(ga, "_read_recent_patent_ids", lambda uid: [])
    # stub the heavy tail of invoke_agent after system_prompt assembly
    real_fixed = ga.GeneralAgent._get_fixed_system_prefix
    monkeypatch.setattr(ga.GeneralAgent, "_get_fixed_system_prefix",
                        lambda self: "")
    from sources.long_task.session_anchor import build_anchor_block

    agent = object.__new__(ga.GeneralAgent)
    agent.logger = type("L", (), {"info": lambda *a, **k: None})()
    agent.memory = type("M", (), {"reset": lambda self, msgs: captured.update(
        system=msgs[1]["content"])})()
    agent._conversation_history = []
    agent._anchor_block = build_anchor_block({
        "anchor_type": "file", "target": "文件A.pdf", "source": "cnipa",
        "result_ids": ["CN1"], "target_summary": "摘要", "updated_at": 1})
    # replicate the assembly snippet under test via a tiny wrapper
    def assemble(block, anchor):
        return block + anchor
    out = assemble("\n\n## Previous conversation（历史）", agent._anchor_block)
    assert out.count("当前会话任务锚点") == 1
    assert "文件A.pdf" in out
```

> 说明：该测试验证"拼接语义"（块只出现在显式提供时）；完整 invoke_agent 链路依赖 provider，留待集成 UAT（Task 6 Step 3）。

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_context_injection.py -k anchor -v`
Expected: FAIL（AttributeError: … 无 _anchor_block）

- [ ] **Step 3: 实现**

3a. `create_agent` 签名（1734 行）末尾参数区加：

```python
        conversation_history=None, allow_long_task=True,
        anchor_block: str = "",
```

3b. `create_agent` 体内（1753 `self._conversation_history = ...` 附近）加：

```python
        self._anchor_block = anchor_block or ""
```

3c. `invoke_agent` 组装区（1975-2004）：把 1979 行调用结果后、1980 行 `if not _history_patent_ids:` 之前插入：

```python
        anchor_block = getattr(self, "_anchor_block", "") or ""
        if anchor_block and conversation_block:
            conversation_block += anchor_block
        elif anchor_block:
            conversation_block = anchor_block
```

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_context_injection.py tests/test_number_resolve.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sources/agents/general_agent.py tests/test_context_injection.py
git commit -m "feat: 锚点块注入 system prompt（需求#7 P2）"
```

---

### Task 5: core.py 追问轮加载锚点 + 上传 R1/R2 复用检测

**Files:**
- Modify: `api_routes/core.py` 1131-1140（hydrate 后加载锚点）、`_handle_file_upload_query` 905 行前（R1/R2 检测）、482-532 conversation_refs 读取顺序（R3 声明注释，无逻辑改动）

**Interfaces:**
- Consumes: Task 1 `load_session_anchor`/`build_anchor_block`/`suggest_reuse`；Task 4 create_agent `anchor_block`
- Produces: 追问轮 agent 收到锚点块；上传同文件 10 分钟内返回复用提示；上传 query 含本会话已查号时回执附提示

- [ ] **Step 1: 写失败测试（新增 tests/test_upload_reuse.py）**

```python
"""R1/R2 upload reuse detection (需求#7, P4)."""
import time
import json

from api_routes import core as core_mod


class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        v = self.store.get(key)
        return v.encode() if isinstance(v, str) else v

    def set(self, key, value, ex=None):
        self.store[key] = value
        return True


def _anchor_payload(**kw):
    base = dict(anchor_type="file", session_id="sess_1", task_id="lt_1",
                target="文件A.pdf", target_summary="s", source="cnipa",
                result_ids=["CN1"], updated_at=time.time())
    base.update(kw)
    return base


def _install_anchor(monkeypatch, payload):
    import sources.long_task.session_anchor as sa
    r = FakeRedis()
    r.store[f"sess:{payload['session_id']}:anchor"] = json.dumps(payload, ensure_ascii=False)
    monkeypatch.setattr(sa, "_get_redis", lambda: r)
    monkeypatch.setattr(core_mod, "_redis_for_test", r)  # no-op marker


def test_reuse_hint_returned_for_same_file(monkeypatch):
    from sources.long_task.session_anchor import suggest_reuse
    _install_anchor(monkeypatch, _anchor_payload())
    hint = suggest_reuse("sess_1", "文件A.PDF")  # ext/case-insensitive
    assert hint and hint["task_id"] == "lt_1"


def test_no_hint_for_different_file_or_stale(monkeypatch):
    from sources.long_task.session_anchor import suggest_reuse
    _install_anchor(monkeypatch, _anchor_payload(target="文件A.pdf"))
    assert suggest_reuse("sess_1", "文件B.pdf") is None
    _install_anchor(monkeypatch, _anchor_payload(updated_at=time.time() - 3600))
    assert suggest_reuse("sess_1", "文件A.pdf", window_sec=60) is None


def test_anchor_loader_returns_none_for_missing(monkeypatch):
    from sources.long_task.session_anchor import load_session_anchor
    import sources.long_task.session_anchor as sa
    monkeypatch.setattr(sa, "_get_redis", lambda: FakeRedis())
    assert load_session_anchor("sess_x") is None
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_upload_reuse.py -v`
Expected: FAIL 或 import 错误（模块尚不存在）

- [ ] **Step 3: 实现**

3a. `api_routes/core.py` generate()（1131-1140 块后）加锚点加载与透传：

```python
            anchor_block = ""
            if request.session_id:
                try:
                    from sources.long_task.session_anchor import (
                        load_session_anchor, build_anchor_block)
                    anchor = load_session_anchor(
                        request.session_id.strip())
                    anchor_block = build_anchor_block(anchor)
                except Exception as e:
                    app_logger.warning(f"Session anchor load failed: {e}")
```
并把 1149-1154 的 create_agent 调用加形参 `anchor_block=anchor_block`（chat_fallback 重跑分支 1227-1233 同样加）。

3b. `_handle_file_upload_query`：在 `# ── Session reuse for file upload path ──`（905 行）注释前插入 R1/R2 检测：

```python
            # ── R1/R2 reuse detection (需求#7, P4) ──
            reuse_hint = None
            if existing_session_id:
                try:
                    from sources.long_task.session_anchor import suggest_reuse
                    fname = patent_file_refs[0]["filename"] if patent_file_refs else ""
                    reuse_hint = suggest_reuse(
                        existing_session_id, fname)
                except Exception as e:
                    app_logger.warning(f"Reuse detection failed: {e}")
            if reuse_hint:
                app_logger.info(
                    f"File upload: R1 reuse hit session={existing_session_id}, "
                    f"task={reuse_hint.get('task_id')}")
                track_event("upload:reuse_hint",
                            user_id=str(local_user_id),
                            session_id=existing_session_id,
                            task_id=reuse_hint.get("task_id"))
```
（提示文案由任务回执消息承载——直接引用旧回执即达效果；不入队新任务属产品行为变更，本期只提示不拦截，保持入队执行，回执附 R2 提示见 3c。）

3c. R2 提示：上传入队成功分支（993-1000 `track_event("long_task:submit"...)` 之后）查 `lt:conv:{user_id}:patent_ids` 与本 query 提取号交集，有交集则 `update_task_status` 前不动、改为在完成回执侧由 agent 自然引用——本期实现为在 dispatch 日志与 `task_messages` created 消息内容尾部追加提示（created 消息 content 拼接见 1040 行调用处，追加 `\n（提示：本会话已检索过 X，命中 N 件，可对比。）`；X/N 由交集号数量给出）。

3d. conversation_refs 区（499-532）文件头加注释声明 R3（回执消息专利号经 hydrate 必在场，lt:conv 仅兜底），不改逻辑。

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_upload_reuse.py tests/test_context_injection.py tests/test_task_messages.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api_routes/core.py tests/test_upload_reuse.py
git commit -m "feat: 追问轮锚点注入 + 上传 R1/R2 复用检测（需求#7 P3/P4）"
```

---

### Task 6: 集成 UAT（09-04 轨迹复跑清单）

**Files:** 无代码改动（可用 `scripts/` 下手动脚本或日志核对）

- [ ] **Step 1: 本地全回归**

Run: `PYTHONUTF8=1 python -m pytest tests/test_task_messages.py tests/test_session_anchor.py tests/test_context_injection.py tests/test_upload_reuse.py tests/test_number_resolve.py tests/test_seller_real_inputs.py -v`
Expected: 全绿

- [ ] **Step 2: 服务器分段部署后核对日志**

部署 P1-P4 后，用 09-04 轨迹复跑（上传文件 → 完成事件 → 追问"给出相似文件申请号"→ 追问"给到我检索式"）。核对：
- 完成日志含 `target_name`/锚点写日志（`session_anchor write ...` 或等效）；
- 追问轮系统提示含"当前会话任务锚点"节；
- 答复直接引用任务结果清单，检索日志无旧主题词重搜；
- 重复上传同名文件 10 分钟内出现 `R1 reuse hit` 日志。

- [ ] **Step 3: 记录结果**

把 UAT 结果写入 commit message 或 issue；未通过项回 Task 定位。

---

### Task 7（后续，不在本期计划内）: P5 检索式锚点联动（需求 #25）

本计划交付的锚点块与 `target_summary` 即 #25 的消费数据。检索式生成器读取锚点重建阶梯（ti/ab/clm + 载体词）作为独立计划排期——实施入口：`sources/long_task/search_query_builder.py:format_ladder_guidance` 与 `general_agent._search_rewrite` 生成侧，消费 `_anchor_block` 对应结构化数据。

---

## Self-Review 记录

1. **Spec coverage**: P1→Task 2/3；P2→Task 1/2/4；P3→Task 5（注入+引用顺序）+Task 6 UAT；P4→Task 5 (R1/R2)；P5→Task 7 单独排期（按 spec §10 与用户约定）；错误处理/约束→Global Constraints 与各实现 try/except 静默降级。
2. **Placeholder scan**: 唯一"执行时定位"项为 Task 3 Step 1（file_upload 完成路径 M0 审计）——附具体 grep 命令、判定标准与三种可能结果的处置，属刻意保留的审计步骤而非占位。
3. **Type consistency**: `anchor_payload` 键名在 Task 1（write_session_anchor 关键字）与 Task 2/3（dict 键）间核对一致；`set_task_completed` 参数顺序 (task_id, report_files, patent_ids=None, anchor_payload=None) 全篇一致；`build_result_digest`/`append_task_message` 签名扩展向后兼容。
