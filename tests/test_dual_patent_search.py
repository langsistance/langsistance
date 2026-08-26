"""Tests for the built-in dual/single-source patent search tool."""
import asyncio
import unittest
from unittest.mock import patch

from sources.agents.react_tools import (
    _baiten_results_to_candidates,
    _baiten_search_by_query,
    _items_digest,
    _resolve_patent_queries,
    _run_patent_search,
    build_tool_set,
)
from sources.patent_source_detect import (
    detect_patent_source_text,
    map_source_for_tool_route,
)


class _FakeAgent:
    """Minimal agent: mutable per-instance state (no cross-test leakage)."""

    def __init__(self, us_ladder=("us-tight", "us-loose"),
                 cn_ladder=("ti:(散热)", "ti:(载体)")):
        self._pending_raw_items = None
        self._last_user_prompt = "test"
        self._last_user_id = "u1"
        self._last_query_id = "q1"
        self._lang = "zh"
        self._search_rewrite = {"queries": list(us_ladder)}
        self._search_rewrite_cn = {"queries": list(cn_ladder)}
        self._tried_queries = []
        self._patent_auto_used = 0
        self.logger = None


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

    def test_maps_documented_documents_shape(self):
        # 2023 API docs: search returns documents[] wrapping fieldValues.
        body = {"qTime": 1, "totalHits": 1, "documents": [
            {"fieldValues": {"pn": "CN118000001A", "ti": "散热装置",
                             "an": "CN202310123456"}},
        ]}
        cands = _baiten_results_to_candidates(body)
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0]["patent_id"], "CN118000001A")
        self.assertEqual(cands[0]["title"], "散热装置")


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


class TestResolvePatentQueries(unittest.TestCase):
    def test_explicit_queries_preserved(self):
        agent = _FakeAgent()
        us, cn = _resolve_patent_queries(
            {"query_string_us": "ab:(cool)", "query_string_cn": "ti:(散热)"},
            ["us-tight"], ["ti:(散热)"], agent, dual=True)
        self.assertEqual(us, "ab:(cool)")
        self.assertEqual(cn, "ti:(散热)")

    def test_missing_cn_filled_from_cn_ladder(self):
        agent = _FakeAgent()
        us, cn = _resolve_patent_queries(
            {"query_string_us": "ab:(cool)"},
            ["us-tight"], ["ti:(散热)", "ti:(载体)"], agent, dual=True)
        self.assertEqual(us, "ab:(cool)")
        self.assertEqual(cn, "ti:(散热)")

    def test_missing_us_filled_in_dual_mode(self):
        agent = _FakeAgent()
        us, cn = _resolve_patent_queries(
            {"query_string_cn": "ti:(散热)"},
            ["us-tight"], ["ti:(散热)"], agent, dual=True)
        self.assertEqual(us, "us-tight")
        self.assertEqual(cn, "ti:(散热)")

    def test_cn_only_tool_never_fills_us(self):
        agent = _FakeAgent()
        us, cn = _resolve_patent_queries(
            {"query_string_cn": "ti:(散热)"},
            ["us-tight"], ["ti:(散热)"], agent, dual=False)
        self.assertEqual(us, "")
        self.assertEqual(cn, "ti:(散热)")

    def test_empty_ladders_leave_slots_empty(self):
        agent = _FakeAgent(us_ladder=(), cn_ladder=())
        us, cn = _resolve_patent_queries({}, [], [], agent, dual=True)
        self.assertEqual((us, cn), ("", ""))

    def test_session_sentinel_treated_as_blank(self):
        agent = _FakeAgent()
        us, cn = _resolve_patent_queries(
            {"query_string_us": "u1", "query_string_cn": "q1"},
            ["us-tight"], ["ti:(散热)"], agent, dual=True)
        self.assertEqual(us, "us-tight")
        self.assertEqual(cn, "ti:(散热)")


class TestBaitenSearchByQueryNotes(unittest.TestCase):
    """The note must tell a real zero from a parse zero from a failure."""

    async def _run(self, body, cfg=None, raise_exc=None):
        class _FakeClient:
            async def search(self, q, page=1, page_size=20):
                if raise_exc is not None:
                    raise raise_exc
                return body
        effective_cfg = cfg if cfg is not None else {
            "app_key": "k", "app_secret": "s", "gateway_url": "http://x"}
        with patch("sources.baiten_client.BaitenClient",
                   return_value=_FakeClient()), \
             patch("sources.long_task.config.get_baiten_config",
                   return_value=effective_cfg):
            return await _baiten_search_by_query("ti:(散热)", agent=_FakeAgent())

    def test_gateway_zero_records_note(self):
        items, note = asyncio.run(self._run({"code": "200"}))
        self.assertEqual(items, [])
        self.assertEqual(note, "Baiten 0 hits (gateway 0 records)")

    def test_gateway_error_note(self):
        # _request_json raises (HTTP non-200 / gateway error code) → the
        # note must say "failed" — never a misleading "0 hits".
        from sources.baiten_client import BaitenAPIError
        items, note = asyncio.run(self._run(
            {}, raise_exc=BaitenAPIError("Baiten API error code=404: msg")))
        self.assertEqual(items, [])
        self.assertIn("Baiten failed", note)
        self.assertIn("error code=404", note)

    def test_records_but_parse_zero_note(self):
        # Rows present but keyed differently than the mapping expects.
        body = {"code": "200", "data": {"fieldValues": [
            {"id": "1", "title": "没有 pn 字段"},
        ]}}
        items, note = asyncio.run(self._run(body))
        self.assertEqual(items, [])
        self.assertEqual(note, "Baiten 0 candidates (parsed from 1 records)")

    def test_valid_rows_note(self):
        body = {"code": "200", "data": {"fieldValues": [
            {"pn": "CN118000001A", "ti": "散热装置"},
            {"pn": "CN118000002A", "ti": "冷却装置"},
        ]}}
        items, note = asyncio.run(self._run(body))
        self.assertEqual(len(items), 2)
        self.assertEqual(note, "Baiten 2 hits")

    def test_not_configured_note(self):
        items, note = asyncio.run(self._run(None, cfg={
            "app_key": "", "app_secret": "", "gateway_url": "http://x"}))
        self.assertEqual(items, [])
        self.assertEqual(note, "Baiten not configured (BAITEN_APP_KEY/APP_SECRET)")


class TestRunPatentSearch(unittest.TestCase):
    async def _run(self, args, us_result=None, cn_result=None, lang="zh",
                   agent=None):
        async def _us(q, page=1, page_size=20, agent=None):
            return us_result if us_result is not None else ([], "USPTO n/a")

        async def _cn(q, page=1, page_size=20, agent=None):
            return cn_result if cn_result is not None else ([], "Baiten n/a")

        with patch("sources.agents.react_tools._uspto_search_by_query", _us), \
             patch("sources.agents.react_tools._baiten_search_by_query", _cn):
            agent = agent or _FakeAgent()
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
        # Auto-fill only helps when a ladder exists; with no ladder the
        # error path still fires.
        agent = _FakeAgent(us_ladder=(), cn_ladder=())
        agent, result = asyncio.run(self._run({}, agent=agent))
        self.assertIn("Error", result["text"])

    def test_missing_cn_leg_auto_filled(self):
        # The production incident: the LLM passed only query_string_us and
        # the CN leg silently never ran.  Now the CN tightest is auto-filled.
        cn_calls = []

        async def _cn(q, page=1, page_size=20, agent=None):
            cn_calls.append(q)
            return [{"patent_id": "CN118000001A", "source": "baiten",
                     "title": "散热装置"}], "Baiten 1 hits"

        async def _us(q, page=1, page_size=20, agent=None):
            return [{"applicationNumberText": "19511555", "applicationMetaData": {
                "inventionTitle": "Cooling", "firstApplicantName": "Intel",
                "filingDate": "2024-01-01"}}], "USPTO 1 hits"

        with patch("sources.agents.react_tools._uspto_search_by_query", _us), \
             patch("sources.agents.react_tools._baiten_search_by_query", _cn):
            agent = _FakeAgent()
            result = asyncio.run(_run_patent_search(
                agent, {"query_string_us": "ab:(cool)"}, "zh"))
        self.assertEqual(cn_calls, ["ti:(散热)"])  # CN leg ran with tightest
        self.assertIn("CN118000001A", result["text"])
        self.assertIn("19511555", result["text"])
        self.assertEqual(len(agent._pending_raw_items), 2)

    def test_zh_cn_zero_triggers_auto_ladder(self):
        # 中文提问：CN 首轮 0 命中 → 系统自动补跑未尝试的 CN 阶梯式。
        cn_calls = []

        async def _cn(q, page=1, page_size=20, agent=None):
            cn_calls.append(q)
            if q == "ti:(载体)":
                return [{"patent_id": "CN118000002A", "source": "baiten",
                         "title": "载体词命中"}], "Baiten 1 hits"
            return [], "Baiten 0 hits (gateway 0 records)"

        async def _us(q, page=1, page_size=20, agent=None):
            return [], "USPTO 0 hits"

        with patch("sources.agents.react_tools._uspto_search_by_query", _us), \
             patch("sources.agents.react_tools._baiten_search_by_query", _cn):
            agent = _FakeAgent()
            result = asyncio.run(_run_patent_search(
                agent, {"query_string_us": "us-tight",
                        "query_string_cn": "ti:(散热)"}, "zh"))
        # 首轮 tightest + 自动补跑（tightest 已记入 tried，补跑只执行载体词式）
        self.assertEqual(cn_calls, ["ti:(散热)", "ti:(载体)"])
        self.assertIn("CN118000002A", result["text"])
        self.assertIn("已自动补跑中国专利阶梯式", result["text"])
        self.assertEqual(agent._patent_auto_used, 1)
        self.assertIn("ti:(载体)", agent._tried_queries)
        self.assertEqual(len(agent._pending_raw_items), 1)

    def test_en_us_zero_triggers_us_auto_ladder(self):
        # 策略一致：英文提问 US 0 命中 → 自动补跑 US 阶梯式。
        us_calls = []

        async def _us(q, page=1, page_size=20, agent=None):
            us_calls.append(q)
            if q == "us-loose":
                return [{"applicationNumberText": "19511555",
                         "applicationMetaData": {
                             "inventionTitle": "Cooling",
                             "firstApplicantName": "Intel",
                             "filingDate": "2024-01-01"}}], "USPTO 1 hits"
            return [], "USPTO 0 hits"

        async def _cn(q, page=1, page_size=20, agent=None):
            return [], "Baiten 0 hits (gateway 0 records)"

        with patch("sources.agents.react_tools._uspto_search_by_query", _us), \
             patch("sources.agents.react_tools._baiten_search_by_query", _cn):
            agent = _FakeAgent()
            result = asyncio.run(_run_patent_search(
                agent, {"query_string_us": "us-tight",
                        "query_string_cn": "ti:(散热)"}, "en"))
        self.assertEqual(us_calls, ["us-tight", "us-loose"])
        self.assertIn("Auto-ran 1 US ladder", result["text"])
        self.assertIn("19511555", result["text"])

    def test_zh_us_zero_does_not_auto_run_us(self):
        # 中文提问：US 0 命中不补跑 US 阶梯（预算留给 CN 侧重）。
        us_calls = []

        async def _us(q, page=1, page_size=20, agent=None):
            us_calls.append(q)
            return [], "USPTO 0 hits"

        async def _cn(q, page=1, page_size=20, agent=None):
            return [{"patent_id": "CN118000001A", "source": "baiten",
                     "title": "散热装置"}], "Baiten 1 hits"

        with patch("sources.agents.react_tools._uspto_search_by_query", _us), \
             patch("sources.agents.react_tools._baiten_search_by_query", _cn):
            agent = _FakeAgent()
            result = asyncio.run(_run_patent_search(
                agent, {"query_string_us": "us-tight",
                        "query_string_cn": "ti:(散热)"}, "zh"))
        self.assertEqual(us_calls, ["us-tight"])
        self.assertIn("CN118000001A", result["text"])

    def test_auto_ladder_respects_request_cap(self):
        cn_calls = []

        async def _cn(q, page=1, page_size=20, agent=None):
            cn_calls.append(q)
            return [], "Baiten 0 hits (gateway 0 records)"

        async def _us(q, page=1, page_size=20, agent=None):
            return [], "USPTO 0 hits"

        with patch("sources.agents.react_tools._uspto_search_by_query", _us), \
             patch("sources.agents.react_tools._baiten_search_by_query", _cn):
            agent = _FakeAgent()
            agent._patent_auto_used = 3  # 距 REACT_PATENT_AUTO_LADDER_MAX=4 只剩 1
            result = asyncio.run(_run_patent_search(
                agent, {"query_string_us": "us-tight",
                        "query_string_cn": "ti:(散热)"}, "zh"))
        self.assertEqual(cn_calls, ["ti:(散热)", "ti:(载体)"])
        self.assertEqual(agent._patent_auto_used, 4)


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
