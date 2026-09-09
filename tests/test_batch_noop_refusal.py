# -*- coding: utf-8 -*-
"""需求#1: batch intent with zero patent references must not reach Celery.

_prepare_long_task_inputs documents (direct_ids, empty list) as "should
fall through to search mode", but nothing downstream implemented that
fall-through: the intent branch would insert a DB row and dispatch an
empty Celery batch (the historical minutes-long empty pipeline).  The
dispatch-level guard `_is_noop_batch_intent` refuses such intents so the
request reruns through the normal chat path.
"""
import asyncio
import sys
import types
import unittest
from types import SimpleNamespace

_fake_passport = types.ModuleType("sources.user.passport")
_fake_passport.verify_firebase_token = lambda *a, **k: {"uid": "1"}
_fake_passport.check_and_increase_usage = lambda *a, **k: True
_fake_passport.ensure_local_user_record = lambda *a, **k: None
sys.modules.setdefault("sources.user.passport", _fake_passport)

from api_routes.core import (  # noqa: E402  (stubs must precede import)
    _is_noop_batch_intent,
    _log_sse_task_crash,
    _prepare_long_task_inputs,
)


async def _cancelled_task():
    task = asyncio.create_task(asyncio.sleep(10))
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return task


class TestLogSseTaskCrash(unittest.TestCase):
    """done-callback: .exception() on a *cancelled* task raises
    CancelledError inside the callback — check cancelled() first."""

    def test_cancelled_task_is_silent(self):
        logger = SimpleNamespace(errors=[],
                                 error=lambda msg: logger.errors.append(msg))
        task = asyncio.run(_cancelled_task())
        _log_sse_task_crash(task, logger)
        self.assertEqual(logger.errors, [])

    def test_failed_task_logs_once(self):
        logger = SimpleNamespace(errors=[],
                                 error=lambda msg: logger.errors.append(msg))

        async def _run_boom():
            async def _boom():
                raise RuntimeError("boom")
            task = asyncio.create_task(_boom())
            try:
                await task
            except RuntimeError:
                pass
            return task
        task = asyncio.run(_run_boom())
        self.assertIsNotNone(task.exception())  # done-failed, not cancelled
        _log_sse_task_crash(task, logger)
        self.assertEqual(len(logger.errors), 1)
        self.assertIn("boom", logger.errors[0])

    def test_success_task_is_silent(self):
        logger = SimpleNamespace(errors=[],
                                 error=lambda msg: logger.errors.append(msg))

        async def _run_ok():
            async def _ok():
                return 1
            task = asyncio.create_task(_ok())
            await task
            return task
        task = asyncio.run(_run_ok())
        _log_sse_task_crash(task, logger)
        self.assertEqual(logger.errors, [])


class TestIsNoopBatchIntent(unittest.TestCase):
    def test_direct_ids_empty_is_noop(self):
        # LLM classified direct_ids but returned no ids — the exact case
        # whose comment promises "fall through to search mode".
        self.assertTrue(_is_noop_batch_intent("direct_ids", [], None))

    def test_conversation_refs_empty_is_noop(self):
        self.assertTrue(_is_noop_batch_intent("conversation_refs", [], None))

    def test_nonempty_ids_not_noop(self):
        self.assertFalse(
            _is_noop_batch_intent("direct_ids", ["17429113"], None))

    def test_pasted_patent_text_not_noop(self):
        self.assertFalse(
            _is_noop_batch_intent(
                "conversation_refs", [], {"CN1": "spec" * 30}))

    def test_single_patent_scenarios_protected(self):
        # prosecution/families/china/epo/japan route on a single patent_id
        # (possibly empty pre-resolvability-check); never refused as a
        # *batch* noop.
        for scenario in ("prosecution", "families", "china_prosecution",
                         "epo_prosecution", "japan_prosecution"):
            self.assertFalse(
                _is_noop_batch_intent(scenario, [], None),
                f"{scenario} must not be treated as a noop batch")

    def test_file_upload_with_ids_not_noop(self):
        self.assertFalse(
            _is_noop_batch_intent("file_upload", ["CN1.spec"], None))

    def test_unknown_empty_scenario_refused_as_noop(self):
        # Anything else that is a batch-shaped intent with no references
        # has nothing to analyze — refuse rather than dispatch empty.
        self.assertTrue(_is_noop_batch_intent("direct_ids", [], {}))


class TestPrepareEmptyDirectIds(unittest.TestCase):
    def test_direct_ids_empty_list_stays_empty(self):
        # Regression contract for the guard: when the LLM returns
        # direct_ids with no ids, patent_ids must stay empty (no history
        # backfill) so the dispatch gate can refuse it.
        inputs = _prepare_long_task_inputs(
            query="帮我看看能实现什么新功能",
            conv_history=[{"role": "assistant", "content": "旧结果",
                           "patent_ids": ["17429113"]}],
            app_logger=None,
            llm_result={
                "scenario": "direct_ids",
                "patent_ids": [],
                "patent_source": "auto",
                "reasoning": "new topic, no ids",
            },
        )
        self.assertEqual(inputs["scenario"], "direct_ids")
        self.assertEqual(inputs["patent_ids"], [])
        self.assertTrue(
            _is_noop_batch_intent(
                inputs["scenario"], inputs["patent_ids"],
                inputs.get("patent_texts")))

    def test_direct_ids_with_ids_not_refused(self):
        inputs = _prepare_long_task_inputs(
            query="分析 17429113 的审查历史",
            conv_history=[],
            app_logger=None,
            llm_result={
                "scenario": "direct_ids",
                "patent_ids": ["17429113"],
                "patent_source": "uspto",
                "reasoning": "single id",
            },
        )
        self.assertFalse(
            _is_noop_batch_intent(
                inputs["scenario"], inputs["patent_ids"],
                inputs.get("patent_texts")))


if __name__ == "__main__":
    unittest.main()
