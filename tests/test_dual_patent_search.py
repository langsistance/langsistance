"""Tests for the built-in dual/single-source patent search tool."""
import asyncio
import unittest
from unittest.mock import patch

from sources.agents.react_tools import (
    _baiten_results_to_candidates,
    _items_digest,
    _run_patent_search,
    build_tool_set,
)
from sources.patent_source_detect import (
    detect_patent_source_text,
    map_source_for_tool_route,
)


class _FakeAgent:
    _pending_raw_items = None
    _last_user_prompt = "test"
    _lang = "zh"


class TestBaitenResultsToCandidates(unittest.TestCase):
    def test_maps_field_values(self):
        body = {"code": "200", "data": {"fieldValues": [
            {"id": "1", "an": "CN202310123456", "ad": "2023-02-01",
             "pn": "CN118000001A", "pd": "2024-01-01",
             "ti": "一种散热装置", "pa": "华为"},
        ]}}
        cands = _baiten_results_to_candidates(body)
        self.assertEqual(len(cands), 1)
        c = cands[0]
        self.assertEqual(c["patent_id"], "CN118000001A")
        self.assertEqual(c["source"], "baiten")
        self.assertEqual(c["title"], "一种散热装置")
        self.assertEqual(c["app_num"], "CN202310123456")
        self.assertEqual(c["pub_date"], "2024-01-01")

    def test_skips_rows_without_pn_and_junk(self):
        body = {"fieldValues": [
            {"ti": "no pn"}, {"junk": True},
        ]}
        self.assertEqual(_baiten_results_to_candidates(body), [])

    def test_handles_top_level_and_absent_field(self):
        self.assertEqual(_baiten_results_to_candidates({"code": "200"}), [])


class TestItemsDigestBaiten(unittest.TestCase):
    def test_renders_baiten_rows(self):
        items = [
            {"patent_id": "CN118000001A", "source": "baiten",
             "title": "散热装置", "applicant": "华为", "pub_date": "2024-01-01"},
        ]
        digest = _items_digest(items, lang="zh")
        self.assertIn("CN118000001A", digest)
        self.assertIn("散热装置", digest)
        self.assertIn("华为", digest)

    def test_limits_rows(self):
        items = [
            {"patent_id": f"CN11{i}0001A", "source": "baiten",
             "title": f"标题{i}"}
            for i in range(50)
        ]
        digest = _items_digest(items, lang="zh")
        self.assertIn("共 50 条", digest)


class TestRunPatentSearch(unittest.TestCase):
    async def _run(self, args, us_result=None, cn_result=None, lang="zh"):
        async def _us(q, page=1, page_size=20):
            return us_result if us_result is not None else ([], "USPTO n/a")

        async def _cn(q, page=1, page_size=20):
            return cn_result if cn_result is not None else ([], "Baiten n/a")

        with patch("sources.agents.react_tools._uspto_search_by_query", _us), \
             patch("sources.agents.react_tools._baiten_search_by_query", _cn):
            agent = _FakeAgent()
            result = await _run_patent_search(agent, args, lang)
            return agent, result

    def test_dual_parallel_and_mapping(self):
        us_items = [{"applicationNumberText": "19511555", "applicationMetaData": {
            "inventionTitle": "Cooling device", "firstApplicantName": "Intel",
            "filingDate": "2024-01-01"}}]
        cn_items = [{"patent_id": "CN118000001A", "source": "baiten",
                     "title": "散热装置"}]
        agent, result = asyncio.run(self._run(
            {"query_string_us": "ab:(cool)", "query_string_cn": "ti:(散热)"},
            us_result=(us_items, "USPTO 1 hits"),
            cn_result=(cn_items, "Baiten 1 hits")))
        self.assertEqual(result["kind"], "observation")
        self.assertIn("19511555", result["text"])
        self.assertIn("CN118000001A", result["text"])
        self.assertEqual(len(agent._pending_raw_items), 2)

    def test_cn_failure_degrades_without_blocking_us(self):
        us_items = [{"applicationNumberText": "19511555", "applicationMetaData": {
            "inventionTitle": "Cooling device", "firstApplicantName": "Intel",
            "filingDate": "2024-01-01"}}]
        agent, result = asyncio.run(self._run(
            {"query_string_us": "ab:(cool)", "query_string_cn": "ti:(散热)"},
            us_result=(us_items, "USPTO 1 hits"),
            cn_result=([], "Baiten failed: method path wrong (P0 unverified)")))
        self.assertIn("19511555", result["text"])
        self.assertIn("Baiten failed", result["text"])
        self.assertEqual(len(agent._pending_raw_items), 1)

    def test_both_empty_sources(self):
        agent, result = asyncio.run(self._run(
            {"query_string_cn": "ti:(散热)"},
            cn_result=([], "Baiten not configured")))
        self.assertIn("两个数据源均未返回结果", result["text"])
        self.assertIn("Baiten not configured", result["text"])
        self.assertEqual(agent._pending_raw_items, [])

    def test_requires_at_least_one_query(self):
        agent, result = asyncio.run(self._run({}))
        self.assertIn("Error", result["text"])


class TestSourceRouting(unittest.TestCase):
    def test_text_detection(self):
        self.assertEqual(detect_patent_source_text("查一下华为的专利"), "cnipa")
        self.assertEqual(detect_patent_source_text("show me Apple patents"), "uspto")
        self.assertEqual(detect_patent_source_text("散热装置有哪些专利"), "auto")

    def test_map_source(self):
        self.assertEqual(map_source_for_tool_route("uspto"), "uspto")
        self.assertEqual(map_source_for_tool_route("cnipa"), "cn")
        self.assertEqual(map_source_for_tool_route("auto"), "dual")


class TestBuildToolSetRegistration(unittest.TestCase):
    async def _build(self, patent_source):
        kwargs = {"patent_source": patent_source} if patent_source else {}
        with patch("sources.agents.react_tools.get_knowledge_tool_candidates",
                   return_value=[]):
            registry, tools = await build_tool_set(
                _FakeAgent(), "u1", "q", None, **kwargs)
            return registry, tools

    def test_dual_registers_combined_tool(self):
        registry, _ = asyncio.run(self._build("dual"))
        self.assertIn("patent_search_dual", registry)
        self.assertNotIn("patent_search_cn", registry)

    def test_cn_registers_single_source_tool(self):
        registry, _ = asyncio.run(self._build("cn"))
        self.assertIn("patent_search_cn", registry)
        self.assertNotIn("patent_search_dual", registry)

    def test_uspto_registers_neither(self):
        registry, _ = asyncio.run(self._build("uspto"))
        self.assertNotIn("patent_search_dual", registry)
        self.assertNotIn("patent_search_cn", registry)

    def test_default_registers_dual(self):
        # 未指定国别（默认）→ 双源工具注册（未传 patent_source）
        registry, _ = asyncio.run(self._build(None))
        self.assertIn("patent_search_dual", registry)


if __name__ == "__main__":
    unittest.main()
