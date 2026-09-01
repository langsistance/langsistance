# -*- coding: utf-8 -*-
"""2026-09-01: 数据来源/状态噪声 (USPTO HTTP 404 / Baiten N hits /
自动补跑阶梯) 只进日志, 不得拼进用户可见的流式 observation 文本。"""
import asyncio
import sys
import unittest
from unittest.mock import patch


class _FakeAgent:
    """Minimal agent (mirrors test_dual_patent_search._FakeAgent)."""

    def __init__(self):
        self._pending_raw_items = None
        self._last_user_prompt = "test"
        self._last_user_id = "u1"
        self._last_query_id = "q1"
        self._lang = "zh"
        self._search_rewrite = {"queries": ["us-tight", "us-loose"]}
        self._search_rewrite_cn = {"queries": ["ti:(散热)", "ti:(载体)"]}
        self._tried_queries = []
        self._patent_auto_used = {"us": 0, "cn": 0}
        self.logger = None


class TestRunPatentSearchStreamNotes(unittest.TestCase):

    def _run(self, agent, us_items, cn_items, us_note="USPTO HTTP 404",
             cn_note="Baiten 2 hits"):
        from sources.agents.react_tools import _run_patent_search

        async def _us(q, page=1, page_size=20):
            return list(us_items), us_note

        async def _cn(q, page=1, page_size=20, agent=None):
            return list(cn_items), cn_note

        with patch("sources.agents.react_tools._uspto_search_by_query", _us), \
             patch("sources.agents.react_tools._baiten_search_by_query", _cn):
            return asyncio.run(_run_patent_search(
                agent, {"query_string_us": "us-tight",
                        "query_string_cn": "ti:(散热)"}, "zh"))

    def test_notes_not_streamed_to_user(self):
        agent = _FakeAgent()
        result = self._run(
            agent,
            us_items=[{"applicationNumberText": "18317505",
                       "title": "Dry air apparatus"}],
            cn_items=[{"patent_id": "CN220271258U", "source": "baiten",
                       "title": "干燥气体发生装置"}],
        )
        text = result["text"]
        # 用户可见文本不得出现数据来源/状态噪声
        self.assertNotIn("USPTO", text)
        self.assertNotIn("Baiten", text)
        self.assertNotIn("404", text)
        self.assertNotIn("HTTP", text)
        # 候选内容仍在
        self.assertIn("18317505", text)
        self.assertIn("CN220271258U", text)

    def test_nested_application_number_us_candidates_enter_pool(self):
        # 2026-09-01: USPTO API 的 applicationNumberText 在顶层/嵌套间漂移,
        # 嵌套结构曾被 _rank_builtin_patent_pool 的顶层检查跳过, 导致 20 条
        # US 候选全丢、池里只剩 1 条 CN。build_candidates 兼容嵌套读取。
        agent = _FakeAgent()
        result = self._run(
            agent,
            us_items=[{"applicationMetaData": {
                "applicationNumberText": "16544963",
                "inventionTitle": "RGB LED driver with independent channels"}}],
            cn_items=[],
        )
        self.assertIn("16544963", result["text"])
        # _pending_raw_items 存的是 _raw (原始 USPTO item)
        self.assertEqual(len(agent._pending_raw_items or []), 1)
        raw = agent._pending_raw_items[0]
        self.assertEqual(
            str(raw.get("applicationMetaData", {}).get("applicationNumberText")),
            "16544963")

    def test_notes_still_logged(self):
        agent = _FakeAgent()
        calls = []

        class _Logger:
            def info(self, *a, **k):
                calls.append(a[0])

            def warning(self, *a, **k):
                calls.append(a[0])

        agent.logger = _Logger()
        self._run(
            agent,
            us_items=[{"applicationNumberText": "18317505",
                       "title": "Dry air apparatus"}],
            cn_items=[],
        )
        # 日志仍记录来源/状态 (可观测性不丢)
        self.assertTrue(any("patent_search_notes" in c for c in calls))
        self.assertTrue(any("USPTO HTTP 404" in c for c in calls))


if __name__ == "__main__":
    unittest.main()


class TestWordLevelQueryFallback(unittest.TestCase):
    """2026-09-01: applications/search 短语匹配词序敏感不稳定 —
    "RGB LED driver" 404 而 "RGB LED" 200。404 时降级为词级 AND 重试。"""

    def test_word_level_rewrite(self):
        from sources.agents.react_tools import _word_level_query
        self.assertEqual(
            _word_level_query('("RGB LED driver" OR "three-channel LED driver")'),
            "(RGB AND LED AND driver OR three-channel AND LED AND driver)")
        self.assertIsNone(_word_level_query("RGB AND LED"))
        self.assertIsNone(_word_level_query(""))

    async def _run(self, q, first_status, second_status, second_items):
        from sources.agents.react_tools import _uspto_search_by_query
        from unittest.mock import MagicMock
        calls = []

        async def _fake_arequest(method, url, purpose=None, headers=None,
                                json=None, timeout=None):
            calls.append(json["q"])
            resp = MagicMock()
            resp.status_code = first_status if len(calls) == 1 else second_status
            resp.json = lambda: {"patentFileWrapperDataBag": second_items}
            return resp

        with patch("sources.http_outbound.outbound_http") as mock_http:
            mock_http.arequest = _fake_arequest
            return await _uspto_search_by_query(q), calls

    def test_phrase_404_falls_back_to_word_level(self):
        import asyncio
        items = [{"applicationNumberText": "19511555",
                  "applicationMetaData": {"inventionTitle": "RGB LED driver"}}]
        (result, note), calls = asyncio.run(self._run(
            '("RGB LED driver")', 404, 200, items))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1], "(RGB AND LED AND driver)")
        self.assertEqual(len(result), 1)

    def test_200_no_fallback(self):
        import asyncio
        items = [{"applicationNumberText": "19511555"}]
        (result, _note), calls = asyncio.run(self._run(
            '"RGB LED"', 200, 200, items))
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(result), 1)
