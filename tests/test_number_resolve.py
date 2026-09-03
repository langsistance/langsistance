"""Tests for the deterministic number-resolution path (react_tools).

Feature (2026-09-03, sample #16): a bare-number question was closed
after a single USPTO 404.  The built-in ``patent_number_resolve`` tool
and the zero-hit cross round must (a) run the primary source first,
(b) verify the OTHER source when the primary returns nothing, and
(c) respect the shared per-request gateway budget.  Transports are
mocked at module level — no network, no LLM.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sources.agents import react_tools


def _run(coro):
    return asyncio.run(coro)


def _agent(candidates=None):
    return SimpleNamespace(
        logger=None,
        _number_candidates=candidates or [],
        _number_cross_done=False,
        _number_cross_used=0,
        _react_loop_ran=True,
        _pending_raw_items=None,
        _search_pool=None,
        _last_user_prompt="117941643",
        llm=None,
        _lang="zh",
        _last_query_id="q1",
        knowledgeTool=(None, None),
        _conversation_turns=[],
    )


_USPTO_ITEM = {
    "applicationNumberText": "117941643",
    "applicationMetaData": {
        "inventionTitle": "A method",
        "applicationStatusDescriptionText": "Patented Case",
    },
}

_BAITEN_ITEM = {
    "source": "baiten",
    "patent_id": "CN117941643A",
    "title": "某方法",
    "applicant": "某公司",
    "pub_date": "2024-02-23",
    "pn": "CN117941643A",
    "an": "CN202311794164.0",
}


class TestLookupPrimarySourceFirst(unittest.TestCase):
    def test_cn_candidate_hits_baiten_first(self):
        candidates = [{"country": "CN", "display": "CN117941643",
                       "lookups": ["CN117941643A", "117941643"]}]
        agent = _agent(candidates)
        with patch.object(react_tools, "_baiten_search_by_query",
                          new=AsyncMock(return_value=([_BAITEN_ITEM],
                                                       "Baiten 1 hits"))) as bm, \
             patch.object(react_tools, "_uspto_search_by_number",
                          new=AsyncMock(return_value=([], "USPTO 0 hits"))) as um:
            merged, notes = _run(
                react_tools._lookup_number_candidates(agent, candidates))
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["source"], "baiten")
        bm.assert_awaited_once()          # 主源命中 → 不打对侧
        um.assert_not_awaited()
        self.assertEqual(agent._number_cross_used, 1)

    def test_cn_zero_then_uspto_cross_check(self):
        """样本16 主场景: 佰腾也 0 命中时, USPTO 数字复核仍会执行。"""
        candidates = [{"country": "CN", "display": "CN117941643",
                       "lookups": ["CN117941643A", "CN117941643",
                                   "117941643"]}]
        agent = _agent(candidates)
        with patch.object(react_tools, "_baiten_search_by_query",
                          new=AsyncMock(return_value=([], "Baiten 0 hits"))), \
             patch.object(react_tools, "_uspto_search_by_number",
                          new=AsyncMock(return_value=([_USPTO_ITEM],
                                                       "USPTO 1 hits"))) as um:
            merged, notes = _run(
                react_tools._lookup_number_candidates(agent, candidates))
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["applicationNumberText"], "117941643")
        um.assert_awaited_once()
        self.assertTrue(any("USPTO" in n for n in notes))
        self.assertTrue(any("Baiten" in n for n in notes))

    def test_us_candidate_zero_then_baiten(self):
        candidates = [{"country": "US", "display": "US19511555",
                       "lookups": ["19511555"]}]
        agent = _agent(candidates)
        with patch.object(react_tools, "_uspto_search_by_number",
                          new=AsyncMock(return_value=([], "USPTO 0 hits"))), \
             patch.object(react_tools, "_baiten_search_by_query",
                          new=AsyncMock(return_value=([_BAITEN_ITEM],
                                                       "Baiten 1 hits"))) as bm:
            merged, _notes = _run(
                react_tools._lookup_number_candidates(agent, candidates))
        self.assertEqual(len(merged), 1)
        bm.assert_awaited_once()

    def test_budget_caps_legs(self):
        candidates = [{"country": "CN", "display": "CN117941643",
                       "lookups": ["CN117941643A", "117941643"]}]
        agent = _agent(candidates)
        with patch.object(react_tools, "_baiten_search_by_query",
                          new=AsyncMock(return_value=([], "Baiten 0 hits"))), \
             patch.object(react_tools, "_uspto_search_by_number",
                          new=AsyncMock(return_value=([_USPTO_ITEM], "USPTO 1"))) as um, \
             patch.object(react_tools, "NUMBER_CROSS_MAX_QUERIES", 1):
            _run(react_tools._lookup_number_candidates(agent, candidates))
        um.assert_not_awaited()           # 预算耗尽 → 对侧复核被截断
        self.assertEqual(agent._number_cross_used, 1)

    def test_empty_candidates_no_calls(self):
        with patch.object(react_tools, "_baiten_search_by_query",
                          new=AsyncMock()) as bm, \
             patch.object(react_tools, "_uspto_search_by_number",
                          new=AsyncMock()) as um:
            merged, notes = _run(
                react_tools._lookup_number_candidates(_agent(), []))
        self.assertEqual(merged, [])
        self.assertEqual(notes, [])
        bm.assert_not_awaited()
        um.assert_not_awaited()


class TestResolveToolObservation(unittest.TestCase):
    def _resolve(self, candidates, merged, notes):
        agent = _agent(candidates)
        args = {"number": "117941643"}
        with patch.object(react_tools, "_lookup_number_candidates",
                          new=AsyncMock(return_value=(merged, notes))) as lk, \
             patch.object(react_tools, "_merge_pending_items",
                          side_effect=lambda ex, new: list(ex or []) + list(new)), \
             patch.object(react_tools, "_rank_builtin_patent_pool",
                          new=AsyncMock(side_effect=lambda a, items, lang: items)), \
             patch.object(react_tools, "_order_pending_for_lang",
                          side_effect=lambda items, lang: items):
            obs = _run(react_tools._run_patent_number_resolve(agent, args, "zh"))
        lk.assert_awaited_once()
        self.assertEqual(agent._number_cross_done, True)
        return obs, agent

    def test_hit_observation_and_pending(self):
        obs, agent = self._resolve(
            [{"country": "CN", "display": "CN117941643",
              "lookups": ["CN117941643A"]}],
            [_BAITEN_ITEM], ["Baiten 1 hits"])
        self.assertEqual(obs["kind"], "observation")
        self.assertIn("CN117941643A", obs["text"])
        self.assertEqual(len(agent._pending_raw_items), 1)

    def test_zero_observation_lists_sources_checked(self):
        obs, _agent = self._resolve(
            [{"country": "CN", "display": "CN117941643",
              "lookups": ["CN117941643A"]}],
            [], ["Baiten 0 hits", "USPTO 0 hits"])
        self.assertIn("未按该号码查到专利记录", obs["text"])
        self.assertIn("USPTO 0 hits", obs["text"])

    def test_unrecognized_number(self):
        agent = _agent([])
        obs = _run(react_tools._run_patent_number_resolve(
            agent, {"number": "量子纠缠装置"}, "zh"))
        self.assertIn("未能识别出专利号格式", obs["text"])


class TestAutoNumberCrossRound(unittest.TestCase):
    def test_fires_once_with_candidates(self):
        agent = _agent([{"country": "CN", "display": "CN117941643",
                         "lookups": ["CN117941643A"]}])
        with patch.object(react_tools, "_lookup_number_candidates",
                          new=AsyncMock(return_value=([_BAITEN_ITEM],
                                                       ["Baiten 1 hits"]))), \
             patch.object(react_tools, "_pool_candidates_for_items",
                          return_value=[{"patent_id": "CN117941643A",
                                         "_raw": _BAITEN_ITEM}]), \
             patch.object(react_tools, "_rank_pending_pool",
                          new=AsyncMock(return_value=(
                              [{"patent_id": "CN117941643A",
                                "_raw": _BAITEN_ITEM}], ""))):
            ranked, _rn, note = _run(
                react_tools._auto_number_cross_round(agent, "zh"))
        self.assertTrue(ranked)
        self.assertIn("号码跨源复核命中", note)
        # once-per-request: 第二次直接 None
        self.assertIsNone(_run(
            react_tools._auto_number_cross_round(agent, "zh")))

    def test_skipped_without_candidates(self):
        agent = _agent([])
        self.assertIsNone(_run(
            react_tools._auto_number_cross_round(agent, "zh")))

    def test_zero_note_when_nothing_found(self):
        agent = _agent([{"country": "CN", "display": "CN117941643",
                         "lookups": ["CN117941643A"]}])
        with patch.object(react_tools, "_lookup_number_candidates",
                          new=AsyncMock(return_value=([], ["Baiten 0 hits"]))), \
             patch.object(react_tools, "_rank_pending_pool",
                          new=AsyncMock(return_value=([], ""))):
            ranked, _rn, note = _run(
                react_tools._auto_number_cross_round(agent, "zh"))
        self.assertEqual(ranked, [])
        self.assertIn("号码跨源复核无命中", note)


if __name__ == "__main__":
    unittest.main()
