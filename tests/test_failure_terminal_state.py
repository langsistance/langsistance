# -*- coding: utf-8 -*-
"""Family-task failure single-point terminal exit (plan Task 5 / spec §5.4).

The family executor's in-_run fail branches no longer call
``set_task_failed`` / ``_update_mysql_progress`` themselves — they only
``return {'status': 'failed', ...}``.  The single terminal exit for a
failed result dict is the outer returned-failed handler
(``_family_failed_terminal``), which pairs a terminal notification
(Redis failed + analytics ``long_task:fail`` once, via ``set_task_failed``)
with a MySQL terminal write — mirroring every other executor's exception
branch.  Hence ``set_task_failed`` / analytics fire exactly once per failed
*task*, never twice (spec §7.5).

Also covers:
- ``notify_terminal_failure`` content is now built from ``failure_guidance``
  with a structured reason-code classification of the worker error
  (InvalidCountryCode / docdb 404 → ERR_UNRESOLVABLE_ID; 5xx / timeout /
  credentials → ERR_EPO_REMOTE); the failed message never mechanically
  re-prints the raw EPO error as its only content.
- families Phase 0 EPO DOCDB candidate list is now delegated to the shared
  translator (``candidates.epo_docdb``); the attempt order equals the
  translator order and falls back to ``[patent_id]`` when the translator
  is empty or raises.
"""
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import celery_worker  # noqa: E402  (imports cleanly off-server, broker not used)
from sources.long_task import status_manager  # noqa: E402


def _run(coro):
    try:
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── 1. family returned-failed → single terminal exit (notify + mysql once) ──

class TestFamilyFailedTerminal(unittest.TestCase):
    def _helpers(self):
        calls = {"notify": [], "mysql": []}

        def fake_notify(tid, err):
            calls["notify"].append((tid, err))

        def fake_mysql(tid, phase, progress, **kw):
            calls["mysql"].append((tid, phase, progress))
        return calls, fake_notify, fake_mysql

    def test_failed_terminal_pairs_notify_and_mysql_once(self):
        calls, fake_notify, fake_mysql = self._helpers()
        with patch.object(celery_worker, "_notify_terminal_failure",
                          side_effect=fake_notify), \
             patch.object(celery_worker, "_update_mysql_progress",
                          side_effect=fake_mysql):
            celery_worker._family_failed_terminal(
                "lt_f1", "EPO family lookup failed for all formats: CLIENT.InvalidCountryCode")
        self.assertEqual(len(calls["notify"]), 1)
        self.assertEqual(len(calls["mysql"]), 1)
        self.assertEqual(calls["mysql"][0], ("lt_f1", "failed", 0))
        self.assertEqual(calls["notify"][0][0], "lt_f1")

    def test_mysql_signature_marks_terminal_failed_phase(self):
        # The MySQL terminal write must mirror the other executors' paired
        # convention (phase='failed', progress=0) so resume/panel reads agree.
        calls, fake_notify, fake_mysql = self._helpers()
        with patch.object(celery_worker, "_notify_terminal_failure",
                          side_effect=fake_notify), \
             patch.object(celery_worker, "_update_mysql_progress",
                          side_effect=fake_mysql):
            celery_worker._family_failed_terminal("lt_f2", "boom")
        self.assertEqual(calls["mysql"][0], ("lt_f2", "failed", 0))


# ── 2. notify content derives from failure_guidance + reason classification ──

class _NoopRedis:
    def get(self, key, *a, **k):
        return None

    def set(self, *a, **k):
        return True

    def exists(self, key, *a, **k):
        return 0

    def delete(self, *a, **k):
        pass


class TestNotifyTerminalGuidance(unittest.TestCase):
    def _notify(self, error):
        # Patch the Redis + conversation write-back so we can inspect content.
        seen = {}
        with patch.object(status_manager, "_get_redis", lambda: _NoopRedis()), \
             patch.object(status_manager, "set_task_failed",
                          wraps=status_manager.set_task_failed) as sp:
            def fake_append(task_id, **kw):
                seen["content"] = kw.get("content", "")
            with patch("sources.long_task.status_manager._lookup_task_user_id",
                       return_value=None), \
                 patch("sources.long_task.task_messages.append_task_message",
                       side_effect=fake_append):
                status_manager.notify_terminal_failure("lt_x", error)
            return seen, sp

    def test_invalid_country_code_classifies_unresolvable_guidance(self):
        seen, sp = self._notify(
            "EPO family lookup failed for all formats: CLIENT.InvalidCountryCode")
        self.assertEqual(sp.call_count, 1)  # Redis failed + analytics exactly once
        content = seen.get("content", "")
        # ERR_UNRESOLVABLE_ID guidance segment present (WO / national-phase advice).
        self.assertIn("WO 公开号", content)
        self.assertIn("该分析需要公开号格式", content)

    def test_docdb_404_classifies_unresolvable(self):
        seen, _ = self._notify(
            "EPO family lookup failed: HTTP 404 for US61500000")
        self.assertIn("该分析需要公开号格式", seen.get("content", ""))

    def test_http_500_classifies_remote_guidance(self):
        seen, _ = self._notify(
            "EPO family lookup failed: HTTP 503 Service Unavailable")
        self.assertIn("外部服务暂时不可用", seen.get("content", ""))
        self.assertIn("点击重试", seen.get("content", ""))

    def test_timeout_classifies_remote(self):
        seen, _ = self._notify("httpx.ConnectTimeout.connect timed out to ops.epo.org")
        self.assertIn("外部服务暂时不可用", seen.get("content", ""))

    def test_credentials_classifies_remote(self):
        seen, _ = self._notify(
            "EPO OAuth2 token request failed: HTTP 403 invalid_grant")
        self.assertIn("外部服务暂时不可用", seen.get("content", ""))

    def test_other_error_falls_back_to_generic(self):
        seen, sp = self._notify("some unrelated worker exception")
        self.assertEqual(sp.call_count, 1)
        content = seen.get("content", "")
        self.assertIn("可 在任务面板点击重试，或重新描述需求后再试".replace(" ", ""),
                      content.replace("\n", "").replace(" ", ""))


# ── 3. families Phase 0 candidates follow the translator ──────────────────

class _FakeResult:
    def __init__(self, epo_docdb):
        self.candidates = {"epo_docdb": epo_docdb}


class TestFamilyEpoDocdbCandidates(unittest.TestCase):
    def test_uses_translator_candidate_order(self):
        tr = AsyncMock(return_value=_FakeResult(
            ["US12506212", "US12506212A1", "US.61500000.44"]))
        with patch("sources.patent_id_translator.translate", side_effect=tr):
            out = _run(celery_worker._family_epo_docdb_candidates("US12506212"))
        self.assertEqual(out, ["US12506212", "US12506212A1", "US.61500000.44"])

    def test_empty_candidates_fall_back_to_original(self):
        tr = AsyncMock(return_value=_FakeResult([]))
        with patch("sources.patent_id_translator.translate", side_effect=tr):
            out = _run(celery_worker._family_epo_docdb_candidates("17429113"))
        self.assertEqual(out, ["17429113"])

    def test_translator_exception_falls_back_to_original(self):
        async def boom(*a, **k):
            raise RuntimeError("translator down")
        with patch("sources.patent_id_translator.translate", side_effect=boom):
            out = _run(celery_worker._family_epo_docdb_candidates("17429113"))
        self.assertEqual(out, ["17429113"])


if __name__ == "__main__":
    unittest.main()
