# -*- coding: utf-8 -*-
"""Chat main-chain resolvability pre-check gate (spec §5.3 入口1 / plan Task 3).

A ``families`` query whose id is deterministically unresolvable must end the
turn with a template guidance *reply* on the SSE channel — it must NOT create
a session/task row, must NOT dispatch ``execute_family_analysis``, and must
NOT emit any ``long_task:submit`` / ``long_task:fail`` event (ruling ①: guidance
reply, no task).

The pre-check lives as one module-level helper in ``api_routes.core`` that the
big ``generate()`` SSE closure calls right after scenario determination and
*before* any DB insert.  When it returns ``True`` the closure returns, so the
DB insert / Celery dispatch / long-task tracking lines are never reached
(verified by construction — those clauses sit strictly below the early return).

Here we drive the helper against a real ``asyncio.Queue`` SSE pipe and patch
``verdict_of`` on the ``api_routes.core`` namespace (module import needs the
passport stub — same trick as test_long_task_fallback.py).
"""
import asyncio
import sys
import types
import unittest
from unittest.mock import patch, AsyncMock

# api_routes.core imports sources.user.passport at module level (server-only
# deps).  Stub ONLY passport; real sources.knowledge.knowledge imports cleanly
# off-server.
_fake_passport = types.ModuleType("sources.user.passport")
_fake_passport.verify_firebase_token = lambda *a, **k: {"uid": "1"}
_fake_passport.check_and_increase_usage = lambda *a, **k: True
_fake_passport.ensure_local_user_record = lambda *a, **k: None
sys.modules.setdefault("sources.user.passport", _fake_passport)

from api_routes.core import _families_precheck_guidance  # noqa: E402


class _Drain:
    """Collect SSE events pushed onto a queue during one helper call."""

    def __init__(self):
        self.queue = asyncio.Queue()

    async def run(self, coro):
        ret = await coro
        events = []
        while not self.queue.empty():
            try:
                events.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return ret, events


# Use a module-level get_event_loop per asyncio policy; drive each test in an
# event loop so the unittests stay simple.
def _run(coro):
    try:
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestFamiliesPrecheckGuidance(unittest.TestCase):
    def _call(self, scenario, patent_ids, verdict="unresolvable",
              verdict_side_effect=None, query="分析同族"):  # zh query triggers zh guide
        async def inner():
            drain = _Drain()
            from unittest.mock import MagicMock
            with patch(
                "api_routes.core.verdict_of",
                side_effect=verdict_side_effect
                if verdict_side_effect is not None
                else (lambda pid, sc: verdict),
            ):
                handled = await _families_precheck_guidance(
                    scenario=scenario,
                    patent_ids=patent_ids,
                    query=query,
                    queue=drain.queue,
                    app_logger=MagicMock(),
                )
                events = []
                while not drain.queue.empty():
                    try:
                        events.append(drain.queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
            return handled, events

        return _run(inner())

    def _event_types(self, events):
        return [e.get("type") for e in events]

    # ── unresolvable families → terminal guide reply, handled ──
    def test_unresolvable_families_returns_handled_with_guide(self):
        handled, events = self._call(
            "families", ["PCTUS2021059064"], verdict="unresolvable")
        self.assertTrue(handled)
        types_ = self._event_types(events)
        self.assertEqual(types_, ["token", "end"])
        self.assertEqual(events[-1]["type"], "end")
        guide = events[0].get("content", "")
        self.assertIn("未能启动该分析任务", guide)
        self.assertIn("WO 公开号", guide)

    def test_unresolvable_never_emits_long_task_events(self):
        _, events = self._call(
            "families", ["PCTUS2021059064"], verdict="unresolvable")
        types_ = self._event_types(events)
        for banned in ("long_task:submit", "long_task:fail", "long_task:queued",
                       "created", "long_task_created"):
            self.assertNotIn(banned, types_)

    # resolvable → not handled (caller proceeds to INSERT + delay).
    def test_resolvable_families_not_handled(self):
        handled, events = self._call(
            "families", ["US12506212"], verdict="resolvable")
        self.assertFalse(handled)
        self.assertEqual(events, [])

    # verdict raises → gate falls through (old behavior, worker backstop).
    def test_verdict_exception_falls_through(self):
        handled, events = self._call(
            "families", ["17429113"],
            verdict_side_effect=RuntimeError("boom"))
        self.assertFalse(handled)
        self.assertEqual(events, [])

    # non-families scenario → gate inactive.
    def test_non_families_scenario_not_handled(self):
        handled, events = self._call(
            "prosecution", ["17429113"], verdict="unresolvable")
        self.assertFalse(handled)
        self.assertEqual(events, [])

    # empty patent_ids → gate inactive.
    def test_no_patent_ids_not_handled(self):
        handled, events = self._call("families", [])
        self.assertFalse(handled)
        self.assertEqual(events, [])

    # verdict None (no recognisable number — review T3 MEDIUM-1) → gate falls
    # through to the legacy path; a bare token must NOT be blocked as
    # "unresolvable".
    def test_verdict_none_not_handled(self):
        handled, events = self._call(
            "families", ["some-bare-token"], verdict=None)
        self.assertFalse(handled)
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
