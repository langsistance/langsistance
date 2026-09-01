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
