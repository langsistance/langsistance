# -*- coding: utf-8 -*-
"""Long-task lifecycle messages inside conversations (M1, 2026-09-03).

Covers:
- append_task_message: entry shape (role/content/meta/patent_ids), seq
  monotonicity, silent skip when the task row is missing.
- build_result_digest: bounded head-cut of the sticky result_summary,
  truncation marker, degradation when no report text exists.
- hydrate_session_task_messages: injects unseen task messages keyed by
  task_id#seq; never duplicates; never mutates the provided history.
"""
import json
import unittest
from unittest.mock import patch

from sources.long_task import task_messages  # noqa: E402
from sources.long_task.task_messages import (  # noqa: E402
    append_task_message,
    build_result_digest,
    hydrate_session_task_messages,
)

# No sys.modules stub here: task_messages imports knowledge lazily inside
# functions only, and tests patch ``sources.knowledge.knowledge.get_db_connection``
# on whatever module instance is current (real one, or the fake installed by
# test_long_task_fallback when it ran first).  Keeping both surfaces aligned
# makes the suites order-independent.

DIGEST_MAX = task_messages.DIGEST_MAX_CHARS


class _FakeCursor:
    def __init__(self, rows):
        self._rows = list(rows)
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self._rows.pop(0) if self._rows else None


class _FakeConn:
    """Returns one cursor per cursor() call with the rows given for it."""

    def __init__(self, *cursor_rows):
        self._cursor_rows = cursor_rows
        self.cursors = []
        self.commits = 0
        self.closed = False

    def cursor(self):
        idx = len(self.cursors)
        rows = self._cursor_rows[idx] if idx < len(self._cursor_rows) else []
        c = _FakeCursor(rows)
        self.cursors.append(c)
        return c

    def commit(self):
        self.commits += 1

    def close(self):
        self.closed = True


class TestAppendTaskMessage(unittest.TestCase):
    def test_appends_assistant_message_with_meta_and_patent_ids(self):
        # _lookup_task_session is patched → only ONE cursor() is used by
        # append_task_message itself (the conversations SELECT+UPDATE).
        conn = _FakeConn(
            [{"id": 1, "messages": json.dumps([
                {"role": "user", "content": "hi"},
            ], ensure_ascii=False)}],
        )
        with patch.object(task_messages, "_lookup_task_session",
                          return_value=("s1", 123)), \
             patch("sources.knowledge.knowledge.get_db_connection",
                   return_value=conn):
            append_task_message(
                "lt_abc", event="completed",
                content="已完成", patent_ids=["CN1"], report_files=["r.pdf"])
        update = conn.cursors[0].executed[-1]
        self.assertIn("UPDATE conversations", update[0])
        messages = json.loads(update[1][0])
        entry = messages[-1]
        self.assertEqual(entry["role"], "assistant")
        self.assertEqual(entry["content"], "已完成")
        self.assertEqual(entry["patent_ids"], ["CN1"])
        self.assertEqual(entry["report_files"], ["r.pdf"])
        self.assertEqual(entry["meta"]["kind"], "long_task")
        self.assertEqual(entry["meta"]["event"], "completed")
        self.assertEqual(entry["meta"]["task_id"], "lt_abc")
        self.assertEqual(entry["meta"]["seq"], 1)
        self.assertTrue(conn.commits >= 1)

    def test_seq_increments_within_task(self):
        conn = _FakeConn(
            [{"id": 1, "messages": json.dumps([
                {"role": "assistant", "content": "已提交",
                 "meta": {"kind": "long_task", "event": "created",
                          "task_id": "lt_abc", "seq": 1}},
            ], ensure_ascii=False)}],
        )
        with patch.object(task_messages, "_lookup_task_session",
                          return_value=("s1", 123)), \
             patch("sources.knowledge.knowledge.get_db_connection",
                   return_value=conn):
            append_task_message("lt_abc", event="failed", content="失败")
        messages = json.loads(conn.cursors[0].executed[-1][1][0])
        self.assertEqual(messages[-1]["meta"]["seq"], 2)

    def test_missing_task_row_skips_silently(self):
        with patch.object(task_messages, "_lookup_task_session",
                          return_value=None):
            append_task_message("lt_missing", event="failed", content="x")
        # No exception raised — pure no-op.

    def test_exception_never_raises(self):
        with patch.object(task_messages, "_lookup_task_session",
                          side_effect=RuntimeError("db down")):
            append_task_message("lt_abc", event="failed", content="x")
        # No exception raised — write-back degrades silently.


def _long_report():
    head = "\n".join(f"## 第 {i} 节标题" + "\n" + "内容行" * 30
                     for i in range(1, 8))
    return ("# 批量分析报告\n\n" + head + "\n## 尾部结论\n收尾")


class TestBuildResultDigest(unittest.TestCase):
    def test_truncates_long_report_at_heading_boundary(self):
        report = _long_report() * 20  # far beyond DIGEST_MAX
        self.assertGreater(len(report), DIGEST_MAX)
        with patch("sources.long_task.status_manager.get_task_status",
                   return_value={
                       "patent_ids": ["CN1", "CN2", "CN3"],
                       "result_summary": report,
                   }):
            digest = build_result_digest("lt_abc")
        self.assertIn("共 3 件", digest)
        self.assertIn("摘要截断", digest)
        self.assertLess(len(digest), DIGEST_MAX + 400)
        # Cut on a markdown heading, never mid-table/mid-line.
        cut = digest.find("…（以上为摘要截断")
        self.assertGreater(cut, 0)
        self.assertTrue(digest[:cut].endswith("\n")
                        or digest[cut - 20:cut].count("\n") > 0)

    def test_short_report_kept_whole(self):
        report = "# 小报告\n\n就这些。"
        with patch("sources.long_task.status_manager.get_task_status",
                   return_value={"patent_ids": ["CN1"],
                                 "result_summary": report}):
            digest = build_result_digest("lt_abc")
        self.assertIn("小报告", digest)
        self.assertNotIn("摘要截断", digest)

    def test_no_report_text_degrades_to_file_list(self):
        with patch("sources.long_task.status_manager.get_task_status",
                   return_value={
                       "patent_ids": ["CN1"],
                       "report_files": ["a.pdf", "b.docx"],
                   }):
            digest = build_result_digest("lt_abc")
        self.assertIn("已完成", digest)
        self.assertIn("a.pdf", digest)

    def test_unknown_status_degrades_to_generic(self):
        with patch("sources.long_task.status_manager.get_task_status",
                   return_value={"status": "unknown"}):
            digest = build_result_digest("lt_abc")
        self.assertIn("已完成", digest)


class TestHydrateSessionTaskMessages(unittest.TestCase):
    def _stored(self):
        return json.dumps([
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "content": "任务已提交",
             "meta": {"kind": "long_task", "event": "created",
                      "task_id": "lt_abc", "seq": 1}},
            {"role": "assistant", "content": "已完成 3 件摘要……",
             "meta": {"kind": "long_task", "event": "completed",
                      "task_id": "lt_abc", "seq": 2},
             "patent_ids": ["CN1", "CN2", "CN3"]},
        ], ensure_ascii=False)

    def test_injects_unseen_task_messages_only(self):
        conn = _FakeConn(
            [{"messages": self._stored()}],  # conversations row
        )
        # Frontend has seen the created message (seq 1) but not the
        # completion digest (seq 2).
        provided = [{"role": "user", "content": "旧问题"},
                    {"role": "assistant", "content": "任务已提交",
                     "meta": {"kind": "long_task", "event": "created",
                              "task_id": "lt_abc", "seq": 1}}]
        with patch("sources.knowledge.knowledge.get_db_connection",
                   return_value=conn):
            out = hydrate_session_task_messages("s1", 123, provided)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[-1]["meta"]["seq"], 2)
        self.assertEqual(out[-1]["patent_ids"], ["CN1", "CN2", "CN3"])
        # Provided history is never mutated.
        self.assertEqual(len(provided), 2)

    def test_no_duplicates_when_history_complete(self):
        conn = _FakeConn([{"messages": self._stored()}])
        provided = json.loads(self._stored())
        with patch("sources.knowledge.knowledge.get_db_connection",
                   return_value=conn):
            out = hydrate_session_task_messages("s1", 123, provided)
        self.assertEqual(len(out), len(provided))

    def test_returns_provided_history_without_session(self):
        provided = [{"role": "user", "content": "hi"}]
        out = hydrate_session_task_messages("", 123, provided)
        self.assertEqual(out, provided)


if __name__ == "__main__":
    unittest.main()
