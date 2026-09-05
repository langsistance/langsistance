"""Tests for the session anchor (需求#7, P2). DB/Redis are faked."""
import json
import time

import pytest


class FakeCur:
    def __init__(self, rows=None):
        self._rows = rows or []
        self.executed = []
        self._idx = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

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
        self.set_calls = []  # (key, ex) tuples so tests can observe renewal

    def get(self, key):
        v = self.store.get(key)
        return v.encode() if isinstance(v, str) else v

    def set(self, key, value, ex=None):
        self.store[key] = value
        self.set_calls.append((key, ex))
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
    fake_redis.set_calls = []  # count only the load's renewal set
    # load must return the preserved value unchanged AND sliding-renew the
    # TTL by re-setting the key with ANCHOR_TTL (the write above set it once;
    # a successful load issues a second set carrying ex=ANCHOR_TTL).
    assert mod.load_session_anchor("sess_1")["target"] == "17429113"
    assert any(
        key == mod._key("sess_1") and ex == mod.ANCHOR_TTL
        for (key, ex) in fake_redis.set_calls)


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
    # Seed the store directly with a STALE anchor (write always stamps a
    # fresh updated_at, so bypass write to exercise the freshness gate).
    stale = _anchor(target="文件A.pdf", updated_at=time.time() - 3600)
    del stale["result_titles"]  # optional; keep shape valid
    fake_redis.store["sess:sess_1:anchor"] = json.dumps(
        stale, ensure_ascii=False)
    assert suggest_reuse("sess_1", "文件A.pdf",
                         window_sec=60) is None
