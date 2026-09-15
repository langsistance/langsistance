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

        async def _cn(q, page=1, page_size=20, agent=None, enrich=True):
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

    def test_language_quota_caps_nonpreferred_source_ladder(self):
        """中文提问：非优选源（美国）阶梯补跑预算收窄，CN 取得更多。

        生产 2026-09-15：中文提问下美国侧跑了 7 条阶梯检索式取回 83 条候选，
        而 CN 侧受每查询 10 条硬限 —— 结果答成"基本都是美国专利"。
        """
        from sources.agents.react_tools import (
            REACT_NONPREFERRED_LADDER_MAX, _run_patent_search)
        agent = _FakeAgent()
        agent._search_rewrite = {"queries": ["us-q%d" % i for i in range(6)]}
        agent._search_rewrite_cn = {"queries": ["ti:(q%d)" % i for i in range(6)]}
        us_calls, cn_calls = [], []

        async def _us(q, page=1, page_size=20):
            us_calls.append(q)
            return [], "USPTO HTTP 404 (true zero, no retry)"

        async def _cn(q, page=1, page_size=20, agent=None, enrich=True):
            cn_calls.append(q)
            return [], "CN 0 hits (gateway 0 records)"

        async def _go(lang):
            us_calls.clear()
            cn_calls.clear()
            with patch("sources.agents.react_tools._uspto_search_by_query", _us), \
                 patch("sources.agents.react_tools._baiten_search_by_query", _cn):
                await _run_patent_search(
                    agent, {"query_string_us": "us-q0",
                            "query_string_cn": "ti:(q0)"}, lang)
            return list(us_calls), list(cn_calls)

        us, cn = asyncio.run(_go("zh"))
        self.assertLessEqual(len(us), 1 + REACT_NONPREFERRED_LADDER_MAX)
        self.assertGreater(len(cn), len(us), "中文提问 CN 应比 US 取更多")

        us_en, cn_en = asyncio.run(_go("en"))
        self.assertLessEqual(len(cn_en), 1 + REACT_NONPREFERRED_LADDER_MAX)
        self.assertGreater(len(us_en), len(cn_en), "英文提问 US 应比 CN 取更多")

    def test_executed_queries_ride_the_digest(self):
        # 需求#25：模型必须能看到本轮真正执行的检索式，才能逐字复述给用户
        #（否则"给我检索式"只能靠编）。来源标注保持中立（不用供应商名）。
        agent = _FakeAgent()
        result = self._run(
            agent,
            us_items=[{"applicationNumberText": "18317505",
                       "title": "Dry air apparatus"}],
            cn_items=[{"patent_id": "CN220271258U", "source": "baiten",
                       "title": "干燥气体发生装置"}],
        )
        text = result["text"]
        self.assertIn("[US] us-tight", text)
        self.assertIn("[CN] ti:(散热)", text)
        self.assertNotIn("Baiten", text)       # 中立标注，供应商名不上可见面

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


class TestAndBudgetTrimRetry(unittest.TestCase):
    """2026-09-12 生产实测：applications/search 404 掉的查询是 **2 个 AND 算子**
    （3 个连接组），例如
        ("cervical rehabilitation" OR "neck exercise")
          AND ("head support assembly" OR "head restraint") AND resistance

    生成侧的 MAX_USPTO_AND_OPS=2 对它们判定"合规、不裁" —— 若重发沿用同一预算，
    重发查询与首次**逐字相同**，等于没重发（本轮修复最初就是这么写的，被这条
    测试抓出来）。既然已经 404，就只能按更严的预算裁掉尾部合取项。
    """

    Q_2AND = ('("cervical rehabilitation" OR "neck exercise") '
              'AND ("head support assembly" OR "head restraint") '
              'AND resistance')
    Q_TRIMMED = ('("cervical rehabilitation" OR "neck exercise") '
                 'AND ("head support assembly" OR "head restraint")')

    def _run_then_200(self, q, fail_times):
        """前 *fail_times* 次请求返 404，其后返 200；返回 (结果, 备注, 送出的查询)。"""
        import asyncio
        from unittest.mock import MagicMock

        from sources.agents.react_tools import _uspto_search_by_query

        calls = []
        items = [{"applicationNumberText": "19511555",
                  "applicationMetaData": {"inventionTitle": "cervical rehab"}}]

        async def _fake_arequest(method, url, purpose=None, headers=None,
                                 json=None, timeout=None):
            calls.append(json["q"])
            resp = MagicMock()
            resp.status_code = 404 if len(calls) <= fail_times else 200
            resp.json = lambda: {"patentFileWrapperDataBag": items}
            return resp

        with patch("sources.http_outbound.outbound_http") as mock_http:
            mock_http.arequest = _fake_arequest
            (result, note) = asyncio.run(_uspto_search_by_query(q))
        return result, note, calls

    def test_404_retry_trims_the_real_production_query(self):
        # 2026-09-15 起：2 个 AND 属**合规形态**（≤MAX_USPTO_AND_OPS），404 = 标题域
        # 真零 —— 只保留语义等价的词级降级（引号短语词序敏感）；放宽式裁尾交给
        # 自动阶梯（同一轮里 ladder 的下一级就是同一条更松查询）。
        result, note, calls = self._run_then_200(self.Q_2AND, fail_times=2)
        self.assertEqual(len(calls), 2, "应为 原发 → 词级降级 两次")
        self.assertTrue(calls[1].startswith("(cervical AND rehabilitation"))
        self.assertNotIn(self.Q_TRIMMED, calls,
                         "合规形态不再就地放宽（阶梯负责降档）")
        self.assertIn("true zero", note)
        self.assertEqual(result, [])

    def test_dialect_overflow_still_trims(self):
        # 3+ 个 AND 算子 = 该端点方言超限，查询根本没被解析 —— 裁尾重发是唯一
        # 不丢约束的救法，必须保留。
        q = ('("cervical rehabilitation" OR "neck exercise") '
             'AND ("head support assembly" OR "head restraint") '
             'AND resistance AND (feedback OR sensor)')
        result, note, calls = self._run_then_200(q, fail_times=2)
        self.assertEqual(len(calls), 3, "超限形态：原发 → 词级降级 → 裁尾重发")
        self.assertLess(calls[2].count(" AND "), q.count(" AND "))
        self.assertIn("AND-budget trim", note)
        self.assertEqual(len(result), 1)

    def test_already_within_budget_is_not_needlessly_retried(self):
        # 1 个 AND 本就合规：裁尾不产生新查询，不应多发请求
        q = '("pressure transducer" OR "pressure sensor array") AND (cervical OR neck)'
        result, note, calls = self._run_then_200(q, fail_times=0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(result), 1)
        self.assertNotIn("trim", note)


class TestSortFallbackDefaultOrder(unittest.TestCase):
    """2026-09-15 (#31): 强制 sort=_score 在标题级语料上把短标题的临时申请/
    失效件顶满整页（09-14 探针：前 20 槽位存活 _score 41/120 vs API 默认序
    103/120，LED 类 0/20；09-12 生产 20 条美方候选 18 条失效）。存活占比低于
    阈值时用 API 默认序（省略 sort）重取一次，按存活数择优。"""

    @staticmethod
    def _items(status, prefix):
        return [{"applicationNumberText": "%s%04d" % (prefix, i),
                 "applicationMetaData": {
                     "inventionTitle": "wafer test",
                     "applicationStatusDescriptionText": status}}
                for i in range(20)]

    def _run(self, first_items, second_items=None):
        from unittest.mock import MagicMock

        from sources.agents.react_tools import _uspto_search_by_query

        calls = []

        async def _fake_arequest(method, url, purpose=None, headers=None,
                                 json=None, timeout=None):
            calls.append((json.get("q"), "sort" in json))
            resp = MagicMock()
            resp.status_code = 200
            payload = first_items if len(calls) == 1 else (second_items or [])
            resp.json = lambda payload=payload: {
                "patentFileWrapperDataBag": payload}
            return resp

        with patch("sources.http_outbound.outbound_http") as mock_http:
            mock_http.arequest = _fake_arequest
            (result, note) = asyncio.run(_uspto_search_by_query("wafer"))
        return result, note, calls

    def test_all_dead_page_refetches_with_default_order(self):
        dead = self._items("Provisional Application Expired", "6100")
        alive = self._items("Patented", "1900")
        result, note, calls = self._run(dead, alive)
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0][1], "首发仍是相关度序")
        self.assertFalse(calls[1][1], "回退必须省略 sort（API 默认序）")
        self.assertIn("sort fallback", note)
        self.assertEqual(len(result), 20)
        self.assertEqual(result[0]["applicationNumberText"], "19000000")

    def test_healthy_page_not_refetched(self):
        result, note, calls = self._run(self._items("Patented", "1900"))
        self.assertEqual(len(calls), 1)
        self.assertNotIn("sort fallback", note)
        self.assertEqual(len(result), 20)

    def test_keeps_primary_when_fallback_not_better(self):
        dead = self._items("Provisional Application Expired", "6100")
        also_dead = self._items("Abandoned  --  Failure to Respond", "6200")
        result, note, calls = self._run(dead, also_dead)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("sort fallback", note)
        self.assertEqual(result[0]["applicationNumberText"], "61000000")

    def test_disabled_when_sort_field_not_score(self):
        dead = self._items("Provisional Application Expired", "6100")
        with patch("sources.agents.react_tools.REACT_USPTO_SORT_FIELD",
                   "applicationMetaData.filingDate"):
            result, note, calls = self._run(dead, self._items("Patented", "1900"))
        self.assertEqual(len(calls), 1)
        self.assertNotIn("sort fallback", note)
