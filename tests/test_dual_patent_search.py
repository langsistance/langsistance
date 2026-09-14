"""Tests for the built-in dual/single-source patent search tool."""
import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from types import SimpleNamespace

from sources.agents.react_tools import (
    BUILTIN_DEEP_ANALYSIS_TOOL_NAME,
    _us_citing_note,
    REACT_PATENT_AUTO_LADDER_MAX,
    _auto_run_patent_ladder,
    _baiten_results_to_candidates,
    _baiten_search_by_query,
    _cn_item_to_pool_candidate,
    _enrich_baiten_law_status,
    _items_digest,
    _normalize_uspto_items,
    _order_pending_for_lang,
    _rank_builtin_patent_pool,
    _resolve_patent_queries,
    _run_patent_search,
    PATENT_LEGAL_STATUS_TOOL_NAME,
    _builtin_deep_analysis_description,
    build_tool_set,
)
from sources.agents import react_tools
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
        self._patent_auto_used = {"us": 0, "cn": 0}
        self.logger = None


class TestCnItemToPoolCandidate(unittest.TestCase):
    def test_maps_flat_baiten_fields(self):
        item = {
            "patent_id": "CN118000001A", "source": "baiten",
            "title": "散热装置", "applicant": "华为",
            "status": "专利权维持", "pub_date": "2024-02-02",
            "apply_date": "2023-11-03", "patent_number": "CN118000001A",
            "type_code": "", "cpc_codes": [],
        }
        c = _cn_item_to_pool_candidate(item)
        self.assertEqual(c["patent_id"], "CN118000001A")
        self.assertEqual(c["title"], "散热装置")
        self.assertEqual(c["applicant"], "华为")
        self.assertEqual(c["status"], "专利权维持")
        self.assertEqual(c["filing_date"], "2023-11-03")  # apply_date wins
        self.assertIs(c["_raw"], item)

    def test_filing_date_falls_back_to_pub_date(self):
        c = _cn_item_to_pool_candidate(
            {"patent_id": "CN118000001A", "pub_date": "2024-02-02"})
        self.assertEqual(c["filing_date"], "2024-02-02")


class TestRankBuiltinPatentPool(unittest.TestCase):
    async def _run(self, agent, items, lang="zh"):
        return await _rank_builtin_patent_pool(agent, items, lang)

    def test_returns_ranked_raw_items(self):
        # Both sources converted; the pool ranking (mocked here) decides
        # the display order, and _raw round-trips the original items.
        from unittest.mock import patch
        cn = [{"patent_id": "CN118000001A", "source": "baiten",
               "title": "散热装置"}]
        us = [{"applicationNumberText": "19511555",
               "applicationMetaData": {"inventionTitle": "Cooling"}}]
        ranked = [
            {"patent_id": "CN118000001A", "title": "散热装置", "_raw": cn[0]},
            {"patent_id": "19511555", "title": "Cooling", "_raw": us[0]},
        ]
        with patch("sources.agents.react_tools._rank_pending_pool",
                   new=lambda a, c, l: (ranked, "note")):
            out = asyncio.run(self._run(_FakeAgent(), cn + us))
        self.assertEqual(out, [cn[0], us[0]])

    def test_failure_degrades_to_unranked_items(self):
        from unittest.mock import patch
        cn = [{"patent_id": "CN118000001A", "source": "baiten",
               "title": "散热装置"}]
        with patch("sources.agents.react_tools._rank_pending_pool",
                   side_effect=RuntimeError("boom")):
            out = asyncio.run(self._run(_FakeAgent(), cn))
        self.assertEqual(out, cn)


class TestOrderPendingForLang(unittest.TestCase):
    def test_zh_groups_cn_first(self):
        us = [{"applicationNumberText": "19511555"}, {"applicationNumberText": "19511556"}]
        cn = [{"patent_id": "CN118000001A", "source": "baiten"}]
        ordered = _order_pending_for_lang(us + cn, "zh")
        self.assertEqual(ordered[0]["patent_id"], "CN118000001A")
        self.assertEqual([c["applicationNumberText"] for c in ordered[1:]],
                         ["19511555", "19511556"])

    def test_non_zh_keeps_source_order(self):
        us = [{"applicationNumberText": "19511555"}]
        cn = [{"patent_id": "CN118000001A", "source": "baiten"}]
        ordered = _order_pending_for_lang(us + cn, "en")
        self.assertEqual(ordered, us + cn)


class TestEnrichBaitenLawStatus(unittest.TestCase):
    """One FLZT call per candidate → status + legal_timeline; FSWX (复审
    无效) decisions are attached only for small (number-lookup) lists,
    with fullText truncated (chat-path wiring, 2026-09-03)."""

    async def _run(self, candidates, timeline=None, decisions=None,
                   fail=False):
        class _Client:
            async def query_legal_state_timeline(self, app_num):
                if fail:
                    raise RuntimeError("gateway down")
                return timeline or []

            async def query_patent_review(self, app_num):
                if fail:
                    raise RuntimeError("gateway down")
                return decisions or []

        await _enrich_baiten_law_status(_Client(), candidates, None)

    def test_fills_status_and_timeline_from_flzt(self):
        candidates = [{"patent_id": "CN118000001A", "app_num": "CN2023XXX",
                       "status": ""}]
        asyncio.run(self._run(candidates, timeline=[
            {"date": "2024-06-01", "lawStatus": "驳回等"},
            {"date": "2024-03-01", "lawStatus": "实质审查的生效"},
        ]))
        c = candidates[0]
        self.assertEqual(c["status"], "驳回等")
        self.assertEqual(len(c["legal_timeline"]), 2)
        self.assertEqual(c["legal_timeline"][1]["lawStatus"], "实质审查的生效")

    def test_small_list_attaches_fswx_decisions_truncated(self):
        candidates = [{"patent_id": "CN118000001A", "app_num": "CN2023XXX",
                       "status": ""}]
        decisions = [{"declareDate": "2024-08-01", "declareNum": "FS12345",
                      "lawBase": "专利法第22条第3款",
                      "fullText": "决定全文内容。" * 300}]
        asyncio.run(self._run(
            candidates,
            timeline=[{"date": "2024-06-01", "lawStatus": "驳回等"}],
            decisions=decisions))
        c = candidates[0]
        self.assertEqual(len(c["review_decisions"]), 1)
        self.assertEqual(c["review_decisions"][0]["declareNum"], "FS12345")
        self.assertLessEqual(
            len(c["review_decisions"][0]["fullText"]),
            800)

    def test_large_hit_list_skips_fswx(self):
        # Topic sweep (many candidates) must not fire an FSWX call per row.
        candidates = [
            {"patent_id": f"CN11800000{i}A", "app_num": f"CN2023{i}",
             "status": ""} for i in range(5)
        ]
        asyncio.run(self._run(candidates, timeline=[
            {"date": "2024-06-01", "lawStatus": "驳回等"}]))
        for c in candidates:
            self.assertEqual(c["status"], "驳回等")
            self.assertNotIn("review_decisions", c)

    def test_failure_degrades_to_empty_fields(self):
        candidates = [{"patent_id": "CN118000001A", "app_num": "CN2023XXX",
                       "status": ""}]
        asyncio.run(self._run(candidates, fail=True))
        self.assertEqual(candidates[0]["status"], "")
        self.assertNotIn("legal_timeline", candidates[0])
        self.assertNotIn("review_decisions", candidates[0])

    def test_skips_candidates_without_app_num(self):
        candidates = [{"patent_id": "CN118000001A", "status": ""}]
        asyncio.run(self._run(
            candidates, timeline=[{"date": "2024-06-01",
                                   "lawStatus": "驳回等"}]))
        self.assertEqual(candidates[0]["status"], "")

    def test_digest_rows_carry_compact_law_tail(self):
        items = [{"patent_id": "CN118000001A", "source": "baiten",
                  "title": "散热装置", "applicant": "华为",
                  "pub_date": "2024-01-01", "status": "驳回等",
                  "legal_timeline": [
                      {"date": "2024-06-01", "lawStatus": "驳回等"},
                      {"date": "2024-03-01", "lawStatus": "实质审查的生效"},
                      {"date": "2023-11-01", "lawStatus": "公开"},
                  ],
                  "review_decisions": [
                      {"declareDate": "2024-08-01", "declareNum": "FS1"},
                  ]}]
        digest = _items_digest(items)
        self.assertIn("状态:驳回等", digest)
        self.assertIn("3次状态变更", digest)
        self.assertIn("复审/无效决定1条(最近2024-08-01)", digest)


class TestLawLookupMemoization(unittest.TestCase):
    """同一请求内、同一申请号的 FLZT 只查一次。

    2026-09-13 生产日志实测：一次提问的 ~100 次 lawInfos 里有 40 次是
    **完全重复**——自动补跑阶梯把同一条检索式又发了一遍，同一批 10 件
    专利被重复查了两遍法律状态。缓存按**线调用**分开（FLZT / FSWX），
    这样大列表没查过 FSWX 时，法律状态工具仍会按需补查，但不会重查
    昂贵的 FLZT。
    """

    class _CountingClient:
        def __init__(self, timeline=None, decisions=None):
            self.timeline_calls = []
            self.review_calls = []
            self._timeline = timeline or []
            self._decisions = decisions or []

        async def query_legal_state_timeline(self, app_num):
            self.timeline_calls.append(app_num)
            return self._timeline

        async def query_patent_review(self, app_num):
            self.review_calls.append(app_num)
            return self._decisions

    _TIMELINE = [{"date": "2025-09-09", "lawStatus": "授权"}]

    def _cands(self, app_num="CN202311111111.1"):
        return [{"patent_id": "CN111A", "app_num": app_num, "status": ""}]

    def test_repeat_search_of_same_app_num_hits_cache(self):
        agent = SimpleNamespace(logger=None)
        client = self._CountingClient(self._TIMELINE)
        asyncio.run(_enrich_baiten_law_status(
            client, self._cands(), None, agent=agent))
        asyncio.run(_enrich_baiten_law_status(
            client, self._cands(), None, agent=agent))
        self.assertEqual(client.timeline_calls, ["CN202311111111.1"])

    def test_cached_row_still_gets_the_status_filled(self):
        # 命中缓存的行也必须被填上状态 —— 缓存的是线调用，不是跳过赋值。
        agent = SimpleNamespace(logger=None)
        client = self._CountingClient(self._TIMELINE)
        asyncio.run(_enrich_baiten_law_status(
            client, self._cands(), None, agent=agent))
        fresh = self._cands()
        asyncio.run(_enrich_baiten_law_status(client, fresh, None, agent=agent))
        self.assertEqual(fresh[0]["status"], "授权")
        self.assertEqual(len(fresh[0]["legal_timeline"]), 1)

    def test_empty_result_is_cached_too(self):
        # "查过且为空" 也要缓存，否则空结果会被反复重查。
        agent = SimpleNamespace(logger=None)
        client = self._CountingClient([])
        asyncio.run(_enrich_baiten_law_status(
            client, self._cands(), None, agent=agent))
        asyncio.run(_enrich_baiten_law_status(
            client, self._cands(), None, agent=agent))
        self.assertEqual(client.timeline_calls, ["CN202311111111.1"])

    def test_cache_is_per_agent_not_global(self):
        # per-request：两个 agent（两次请求）不得互相污染。
        a1, a2 = SimpleNamespace(logger=None), SimpleNamespace(logger=None)
        client = self._CountingClient(self._TIMELINE)
        asyncio.run(_enrich_baiten_law_status(
            client, self._cands(), None, agent=a1))
        asyncio.run(_enrich_baiten_law_status(
            client, self._cands(), None, agent=a2))
        self.assertEqual(len(client.timeline_calls), 2)

    def test_no_agent_still_works_uncached(self):
        # 向后兼容：不传 agent 时保持旧行为（既有测试就这么调）。
        client = self._CountingClient(self._TIMELINE)
        cands = self._cands()
        asyncio.run(_enrich_baiten_law_status(client, cands, None))
        self.assertEqual(cands[0]["status"], "授权")

    def test_fswx_only_cached_when_actually_fetched(self):
        # 大列表不查 FSWX；此时缓存里不该留下"已查"的痕迹，
        # 否则法律状态工具会误以为查过而不补查。
        agent = SimpleNamespace(logger=None)
        client = self._CountingClient(self._TIMELINE)
        big = [{"patent_id": f"CN{i}A", "app_num": f"CN2023{i}", "status": ""}
               for i in range(5)]
        asyncio.run(_enrich_baiten_law_status(client, big, None, agent=agent))
        self.assertEqual(client.review_calls, [])
        # 缓存字典本身是惰性创建的：没查过 FSWX 就不该留下任何条目。
        self.assertNotIn("CN20230", getattr(agent, "_law_fswx_cache", {}))


class TestLawEnrichmentConcurrency(unittest.TestCase):
    """富化必须限并发 —— 佰腾网关对并发敏感。

    2026-09-13 生产日志：单轮 30 条候选一起去（30 路并发）时，约一半的
    FLZT 调用返回 500 ``no access for this api: DATA_PAT_PATAFFAIRSDATA_ONE``；
    同一个号码先失败、26 秒后重试成功，说明不是权限缺失而是**并发节流**。
    改动前是「每次检索 10 路」，改动后变成「单轮 N 路」——去重省了调用，
    却把瞬时并发放宽了，必须设上限。
    """

    class _TrackingClient:
        def __init__(self):
            self.in_flight = 0
            self.peak = 0

        async def query_legal_state_timeline(self, app_num):
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            # 让出控制权，使所有并发任务都有机会进入临界区
            await asyncio.sleep(0)
            self.in_flight -= 1
            return [{"date": "2025-09-09", "lawStatus": "授权"}]

        async def query_patent_review(self, app_num):
            return []

    _CANDS = [{"patent_id": "CN%dA" % i, "app_num": "CN2023%d" % i,
               "status": ""} for i in range(24)]

    def test_flzt_concurrency_is_capped(self):
        client = self._TrackingClient()
        with patch.object(react_tools, "LAW_ENRICH_CONCURRENCY", 4):
            asyncio.run(_enrich_baiten_law_status(
                client, self._CANDS, None))
        self.assertLessEqual(client.peak, 4)
        self.assertGreater(client.peak, 1)   # 仍然是并发，不是串行

    def test_cap_is_configurable(self):
        client = self._TrackingClient()
        with patch.object(react_tools, "LAW_ENRICH_CONCURRENCY", 2):
            asyncio.run(_enrich_baiten_law_status(
                client, self._CANDS, None))
        self.assertLessEqual(client.peak, 2)

    def test_default_cap_is_conservative(self):
        # 默认值必须明显低于改动前的「单轮不限并发」。
        self.assertLessEqual(react_tools.LAW_ENRICH_CONCURRENCY, 5)
        self.assertGreaterEqual(react_tools.LAW_ENRICH_CONCURRENCY, 1)

    def test_all_candidates_still_enriched_under_the_cap(self):
        # 限并发不得丢候选。用 12 条（< 每次调用配额上限）以便只考并发。
        client = self._TrackingClient()
        cands = [dict(c) for c in self._CANDS[:12]]
        with patch.object(react_tools, "LAW_ENRICH_CONCURRENCY", 3):
            asyncio.run(_enrich_baiten_law_status(client, cands, None))
        for c in cands:
            self.assertEqual(c["status"], "授权")


class TestLawEnrichmentPerCallBudget(unittest.TestCase):
    """每次工具调用的 lawInfos 条数上限。

    配额是硬约束（2026-09-13 账号额度耗尽 → 大片 500），而自动补跑阶梯
    一次可能收进 30+ 条候选，全查一遍是配额的主要消耗方式。

    取舍（明确记录）：**超出上限的行没有法律状态** —— 摘要的状态列与
    导出文件的状态列都会空着。上限默认等于摘要展示条数，可用 env
    ``REACT_LAW_ENRICH_MAX_PER_CALL`` 按配额松紧调整；设为 0 表示不限。
    """

    class _Client:
        def __init__(self):
            self.calls = []

        async def query_legal_state_timeline(self, app_num):
            self.calls.append(app_num)
            return [{"date": "2025-09-09", "lawStatus": "授权"}]

        async def query_patent_review(self, app_num):
            return []

    def _cands(self, n):
        return [{"patent_id": "CN%dA" % i, "app_num": "CN2023%d" % i,
                 "status": ""} for i in range(n)]

    def test_enriches_at_most_the_cap(self):
        client = self._Client()
        cands = self._cands(30)
        with patch.object(react_tools, "LAW_ENRICH_MAX_PER_CALL", 20):
            asyncio.run(_enrich_baiten_law_status(client, cands, None))
        self.assertEqual(len(client.calls), 20)

    def test_rows_beyond_the_cap_keep_empty_status(self):
        client = self._Client()
        cands = self._cands(30)
        with patch.object(react_tools, "LAW_ENRICH_MAX_PER_CALL", 20):
            asyncio.run(_enrich_baiten_law_status(client, cands, None))
        for c in cands[:20]:
            self.assertEqual(c["status"], "授权")
        for c in cands[20:]:
            self.assertEqual(c["status"], "")

    def test_cap_is_tunable(self):
        client = self._Client()
        with patch.object(react_tools, "LAW_ENRICH_MAX_PER_CALL", 5):
            asyncio.run(_enrich_baiten_law_status(
                client, self._cands(30), None))
        self.assertEqual(len(client.calls), 5)

    def test_zero_means_unlimited(self):
        client = self._Client()
        with patch.object(react_tools, "LAW_ENRICH_MAX_PER_CALL", 0):
            asyncio.run(_enrich_baiten_law_status(
                client, self._cands(30), None))
        self.assertEqual(len(client.calls), 30)

    def test_small_lists_are_unaffected(self):
        client = self._Client()
        asyncio.run(_enrich_baiten_law_status(client, self._cands(3), None))
        self.assertEqual(len(client.calls), 3)

    def test_default_cap_equals_the_digest_display_size(self):
        self.assertEqual(react_tools.LAW_ENRICH_MAX_PER_CALL,
                         react_tools.SEARCH_DIGEST_LIMIT)


class TestLawEnrichmentPerRequestBudget(unittest.TestCase):
    """每**请求**的 lawInfos 总预算 —— 按调用限流治不了总量。

    一次提问会跑 5–6 次工具调用（阶梯 + 自动补跑），每次调用各自限 20 条
    仍然合计 ~50 次。配额是每请求的硬约束，所以必须有总量闸门。

    默认与 ``SEARCH_DIGEST_LIMIT`` 一致，含义：**每个请求只够富化一份完整
    摘要**。0 = 不限（回到旧行为）。
    """

    class _Client:
        def __init__(self):
            self.calls = []

        async def query_legal_state_timeline(self, app_num):
            self.calls.append(app_num)
            return [{"date": "2025-09-09", "lawStatus": "授权"}]

        async def query_patent_review(self, app_num):
            self.calls.append("FSWX:" + app_num)
            return []

    def _cands(self, n, offset=0):
        return [{"patent_id": "CN%dA" % i, "app_num": "CN2023%d" % i,
                 "status": ""} for i in range(offset, offset + n)]

    def _agent(self):
        return SimpleNamespace(logger=None, _law_budget_used=0)

    def test_total_across_passes_is_capped(self):
        client = self._Client()
        agent = self._agent()
        with patch.object(react_tools, "LAW_ENRICH_MAX_PER_REQUEST", 20):
            for r in range(4):          # 模拟 4 轮工具调用
                asyncio.run(_enrich_baiten_law_status(
                    client, self._cands(10, offset=r * 10), None, agent=agent))
        self.assertEqual(len(client.calls), 20)
        self.assertEqual(agent._law_budget_used, 20)

    def test_budget_is_per_request_flag_not_module_state(self):
        # 两个 agent（两次请求）各自有预算，互不扣减。
        client = self._Client()
        with patch.object(react_tools, "LAW_ENRICH_MAX_PER_REQUEST", 5):
            for _ in range(2):
                asyncio.run(_enrich_baiten_law_status(
                    client, self._cands(10), None, agent=self._agent()))
        self.assertEqual(len(client.calls), 10)

    def test_cache_hits_do_not_consume_budget(self):
        client = self._Client()
        agent = self._agent()
        with patch.object(react_tools, "LAW_ENRICH_MAX_PER_REQUEST", 10):
            cands = self._cands(5)
            asyncio.run(_enrich_baiten_law_status(client, cands, None,
                                                  agent=agent))
            asyncio.run(_enrich_baiten_law_status(
                client, self._cands(5), None, agent=agent))
        self.assertEqual(len(client.calls), 5)      # 第二次全命中缓存
        self.assertEqual(agent._law_budget_used, 5)

    def test_exhausted_budget_leaves_the_rest_empty(self):
        client = self._Client()
        agent = self._agent()
        with patch.object(react_tools, "LAW_ENRICH_MAX_PER_REQUEST", 3):
            cands = self._cands(10)
            asyncio.run(_enrich_baiten_law_status(client, cands, None,
                                                  agent=agent))
        self.assertEqual(len(client.calls), 3)
        # 恰好 3 条拿到状态，其余 7 条留空（不指定是哪 3 条：并发顺序不定）
        enriched = [c for c in cands if c["status"]]
        self.assertEqual(len(enriched), 3)

    def test_zero_means_unlimited(self):
        client = self._Client()
        agent = self._agent()
        with patch.object(react_tools, "LAW_ENRICH_MAX_PER_REQUEST", 0):
            for r in range(3):
                asyncio.run(_enrich_baiten_law_status(
                    client, self._cands(10, offset=r * 10), None, agent=agent))
        self.assertEqual(len(client.calls), 30)

    def test_no_agent_is_unbudgeted(self):
        # 向后兼容：不传 agent 时保持旧行为（既有测试就这么调）。
        client = self._Client()
        with patch.object(react_tools, "LAW_ENRICH_MAX_PER_REQUEST", 2):
            cands = self._cands(10)
            asyncio.run(_enrich_baiten_law_status(client, cands, None))
        self.assertEqual(len(client.calls), 10)

    def test_default_covers_a_full_export(self):
        # 2026-09-13 实测：该查询导出 30 条 CN，预算 20 时 13 行状态列空白。
        # 默认必须够覆盖整份导出，同时明显低于「每轮都富化」的旧行为。
        self.assertGreaterEqual(react_tools.LAW_ENRICH_MAX_PER_REQUEST, 30)
        self.assertLess(react_tools.LAW_ENRICH_MAX_PER_REQUEST, 50)


class TestUsptoSpaceFlattenDisabled(unittest.TestCase):
    """空格兜底（OR 语义）默认关闭，且 2026-09-15 起**仅在方言超限形态**下才会
    被考虑（合规形态的 404 = 标题域真零，拍平只会拿 OR 噪声换 200）。

    2026-09-13 生产实证：它每轮返回 20 条噪声，**全部**被后续过滤丢弃（30 条
    导出里美国只剩 2 条），却进入 observation 摘要 —— 模型把其中"活着"的
    PCT/美国申请当成 Top 结果报给用户，实测 9/10 在结果面板里不存在。
    纯成本 + 污染判断，零收益。
    """

    # 3 个 AND 算子 = 该端点方言超限（唯一还会走满重试链的形态）。
    _Q = ('"humidity sensor" AND "desiccant dryer" AND "air supply" '
          'AND (feedback OR sensor)')

    def _attempts(self, enabled, q=None):
        calls = []

        async def _arequest(method, url, **kw):
            calls.append((kw.get("json") or {}).get("q", ""))
            return SimpleNamespace(status_code=404, json=lambda: {})

        with patch.object(react_tools, "USPTO_SPACE_FLATTEN_ENABLED", enabled), \
             patch("sources.http_outbound.outbound_http") as mock_http:
            mock_http.arequest = AsyncMock(side_effect=_arequest)
            items, note = asyncio.run(
                react_tools._uspto_search_by_query(q or self._Q))
        return calls, items, note

    def test_disabled_by_default(self):
        self.assertFalse(react_tools.USPTO_SPACE_FLATTEN_ENABLED)

    def test_no_flatten_attempt_when_disabled(self):
        # 超限形态：原发 → 词级 → AND 裁尾，三发为止。
        calls, items, _note = self._attempts(False)
        self.assertEqual(len(calls), 3)
        self.assertEqual(items, [])

    def test_flatten_attempt_returns_when_enabled(self):
        # 对照：打开后确实多一发（证明是这道闸门在拦，不是别的）。
        calls, _items, _note = self._attempts(True)
        self.assertEqual(len(calls), 4)

    def test_compliant_query_never_flattens(self):
        # 合规形态（≤2 AND）即使打开兜底也不拍平：真零只标注、不再换噪声。
        q = '"humidity sensor" AND "desiccant dryer" AND "dry air supply"'
        calls, items, note = self._attempts(True, q=q)
        self.assertEqual(len(calls), 2, "原发 → 词级降级，两发为止")
        self.assertEqual(items, [])
        self.assertIn("true zero", note)


class TestNormalizeUsptoItems(unittest.TestCase):
    def test_lifts_title_from_meta_invention_title(self):
        items = [{"applicationNumberText": "19511555", "applicationMetaData": {
            "inventionTitle": "Air dryer", "filingDate": "2024-01-01"}}]
        out = _normalize_uspto_items(items)
        self.assertEqual(out[0]["title"], "Air dryer")
        self.assertEqual(out[0]["applicationMetaData"]["inventionTitle"],
                         "Air dryer")

    def test_lifts_title_from_meta_title_of_invention(self):
        # Schema drift observed 2026-08-27: real responses carried the
        # title under titleOfInvention, artifact rows showed blank titles.
        items = [{"applicationNumberText": "19511555", "applicationMetaData": {
            "titleOfInvention": "Cooling device"}}]
        out = _normalize_uspto_items(items)
        self.assertEqual(out[0]["title"], "Cooling device")

    def test_keeps_existing_top_level_title(self):
        items = [{"applicationNumberText": "19511555", "title": "Already"}]
        out = _normalize_uspto_items(items)
        self.assertEqual(out[0]["title"], "Already")
        self.assertIs(out[0], items[0])

    def test_no_title_passes_through(self):
        items = [{"applicationNumberText": "19511555",
                  "applicationMetaData": {"filingDate": "2024-01-01"}}]
        out = _normalize_uspto_items(items)
        self.assertNotIn("title", out[0])
        self.assertEqual(len(out), 1)


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
        # 中立取值：用户可下载的导出文件里不得出现供应商名（见
        # test_cn_source_value.py）。读取侧仍接受历史值 "baiten"。
        self.assertEqual(c["source"], "cn")
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

    def test_maps_live_field_values_shape_with_pa_list(self):
        # Live-verified response (2026-08-26, real key): documents[] with
        # field_values (snake_case) and multi-valued pa as a list.
        body = {"qTime": 31, "total_hits": 864544, "grouped_hits": 0,
                "documents": [
                    {"field_values": {
                        "an": "CN201610553976.0", "ad": "20160714",
                        "pn": "CN107618459A", "pd": "20180123",
                        "ti": "汽车后备箱开启方法",
                        "pa": ["中山市澳多电子科技有限公司"],
                        "id": "CN201610553976.0"},
                     "hl_field_values": {"pa": ["中山市澳多电子科技有限公司"]}},
                ]}
        cands = _baiten_results_to_candidates(body)
        self.assertEqual(len(cands), 1)
        c = cands[0]
        self.assertEqual(c["patent_id"], "CN107618459A")
        self.assertEqual(c["title"], "汽车后备箱开启方法")
        self.assertEqual(c["app_num"], "CN201610553976.0")
        self.assertEqual(c["applicant"], "中山市澳多电子科技有限公司")
        self.assertEqual(c["pub_date"], "20180123")


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
            async def search(self, q, page=1, page_size=20,
                             api_level="ONE"):
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
        self.assertEqual(note, "CN 0 hits (gateway 0 records)")

    def test_gateway_error_note(self):
        # _request_json raises (HTTP non-200 / gateway error code) → the
        # note must say "failed" — never a misleading "0 hits".
        from sources.baiten_client import BaitenAPIError
        items, note = asyncio.run(self._run(
            {}, raise_exc=BaitenAPIError("Baiten API error code=404: msg")))
        self.assertEqual(items, [])
        self.assertIn("CN source failed", note)
        self.assertIn("error code=404", note)

    def test_records_but_parse_zero_note(self):
        # Rows present but keyed differently than the mapping expects.
        body = {"code": "200", "data": {"fieldValues": [
            {"id": "1", "title": "没有 pn 字段"},
        ]}}
        items, note = asyncio.run(self._run(body))
        self.assertEqual(items, [])
        self.assertEqual(note, "CN 0 candidates (parsed from 1 records)")

    def test_valid_rows_note(self):
        body = {"code": "200", "data": {"fieldValues": [
            {"pn": "CN118000001A", "ti": "散热装置"},
            {"pn": "CN118000002A", "ti": "冷却装置"},
        ]}}
        items, note = asyncio.run(self._run(body))
        self.assertEqual(len(items), 2)
        self.assertEqual(note, "CN 2 hits")

    def test_page_capped_note_reports_gateway_total(self):
        # 需求#9：单查询恒 rows=page_size（页上限）时，真实命中数只有网关的
        # total 能回答 —— 必须出现在备注里，不能继续靠"rows 恒 10"猜。
        body = {"code": "200", "total_hits": 347,
                "data": {"fieldValues": [
                    {"pn": "CN118%06dA" % i, "ti": "散热装置"}
                    for i in range(20)]}}
        items, note = asyncio.run(self._run(body))
        self.assertEqual(len(items), 20)
        self.assertIn("showing 20 of 347", note)

    def test_page_capped_without_total_marks_cap(self):
        body = {"code": "200", "data": {"fieldValues": [
            {"pn": "CN118%06dA" % i, "ti": "散热装置"} for i in range(20)]}}
        items, note = asyncio.run(self._run(body))
        self.assertIn("page-capped at 20", note)

    def test_not_configured_note(self):
        items, note = asyncio.run(self._run(None, cfg={
            "app_key": "", "app_secret": "", "gateway_url": "http://x"}))
        self.assertEqual(items, [])
        self.assertEqual(note, "CN source not configured (key missing)")

    def test_api_level_from_config_reaches_client(self):
        received = {}

        class _FakeClient:
            async def search(self, q, page=1, page_size=20,
                             api_level="ONE"):
                received["api_level"] = api_level
                return {"code": "200"}

        with patch("sources.baiten_client.BaitenClient",
                   return_value=_FakeClient()), \
             patch("sources.long_task.config.get_baiten_config",
                   return_value={"app_key": "k", "app_secret": "s",
                                 "gateway_url": "http://x",
                                 "api_level": "TWO"}):
            asyncio.run(_baiten_search_by_query("ti:(散热)", agent=_FakeAgent()))
        self.assertEqual(received["api_level"], "TWO")


class TestTurnLevelSearchCache(unittest.TestCase):
    """2026-09-15 需求#27：同 turn 相同的 CN 检索式复用上次结果。

    生产实证（2026-09-14 1309…）：不同 US 同义词组轮询时，CN 阶梯被逐字重跑
    （两次 `ab:(泌乳计划…) AND ab:(智能提醒…)` 各返回同样的 10 行）。缓存按请求
    重置（general_agent.create_agent 的重置块），不得跨请求泄漏。
    """

    _BODY = {"code": "200", "data": {"fieldValues": [
        {"pn": "CN118000001A", "ti": "散热装置"}]}}

    def _agent_and_client(self, calls):
        body = self._BODY

        class _FakeClient:
            async def search(self, q, page=1, page_size=20,
                             api_level="ONE"):
                calls.append(q)
                return body
        return _FakeAgent(), _FakeClient()

    def test_same_query_in_one_turn_hits_cache(self):
        calls = []
        agent = _FakeAgent()
        agent._search_result_cache = {}
        _, client = self._agent_and_client(calls)

        async def _go():
            with patch("sources.baiten_client.BaitenClient",
                       return_value=client), \
                 patch("sources.long_task.config.get_baiten_config",
                       return_value={"app_key": "k", "app_secret": "s",
                                     "gateway_url": "http://x"}):
                first = await _baiten_search_by_query("ti:(散热)", agent=agent)
                second = await _baiten_search_by_query("ti:(散热)", agent=agent)
            return first, second

        (items1, note1), (items2, note2) = asyncio.run(_go())
        self.assertEqual(calls, ["ti:(散热)"], "同 turn 同式只应外发一次")
        self.assertEqual(note2, note1)
        self.assertEqual(items2, items1)

    def test_no_cache_without_agent(self):
        # agent=None（KB 路径的旧调用形态）不缓存，保持旧行为。
        calls = []
        _, client = self._agent_and_client(calls)

        async def _go():
            with patch("sources.baiten_client.BaitenClient",
                       return_value=client), \
                 patch("sources.long_task.config.get_baiten_config",
                       return_value={"app_key": "k", "app_secret": "s",
                                     "gateway_url": "http://x"}):
                await _baiten_search_by_query("ti:(散热)")
                await _baiten_search_by_query("ti:(散热)")

        asyncio.run(_go())
        self.assertEqual(len(calls), 2)


class TestBaitenConfig(unittest.TestCase):
    """api_level maps to the purchased data product (DATA_PAT_BASE_<LEVEL>);
    level=ONE was live-verified for the production account 2026-08-26."""

    def test_env_api_level_override(self):
        from sources.long_task.config import get_baiten_config
        with patch.dict("os.environ",
                        {"BAITEN_APP_KEY": "k", "BAITEN_APP_SECRET": "s",
                         "BAITEN_API_LEVEL": "TWO"}):
            cfg = get_baiten_config("nonexistent.ini")
        self.assertEqual(cfg["api_level"], "TWO")

    def test_default_api_level(self):
        from sources.long_task.config import get_baiten_config
        with patch.dict("os.environ",
                        {"BAITEN_APP_KEY": "k", "BAITEN_APP_SECRET": "s"}):
            cfg = get_baiten_config("nonexistent.ini")
        self.assertEqual(cfg["api_level"], "ONE")


class TestRunPatentSearch(unittest.TestCase):
    async def _run(self, args, us_result=None, cn_result=None, lang="zh",
                   agent=None):
        async def _us(q, page=1, page_size=20, agent=None):
            return us_result if us_result is not None else ([], "USPTO n/a")

        async def _cn(q, page=1, page_size=20, agent=None, enrich=True):
            return cn_result if cn_result is not None else ([], "Baiten n/a")

        with patch("sources.agents.react_tools._uspto_search_by_query", _us), \
             patch("sources.agents.react_tools._baiten_search_by_query", _cn), \
             patch("sources.agents.react_tools._enrich_baiten_law_status",
                   new=AsyncMock()), \
             patch("sources.agents.react_tools._baiten_client_or_none",
                   return_value=None):
            agent = agent or _FakeAgent()
            result = await _run_patent_search(agent, args, lang)
            return agent, result

    def test_digest_excludes_items_that_will_be_filtered(self):
        # 需求#26：摘要只渲染**可能进入交付集**的结果。此前用全量 merged
        # 渲染，模型会引用随即被失效过滤丢掉的行 —— 2026-09-13 生产实证：
        # 回答里列出的美国专利，导出文件里根本没有。
        dead = {"applicationNumberText": "11111111", "status": "Abandoned",
                "applicationMetaData": {"inventionTitle": "DEADONE"}}
        live = {"applicationNumberText": "22222222",
                "status": "Patented Case",
                "applicationMetaData": {"inventionTitle": "LIVEONE"}}
        _agent, result = asyncio.run(self._run(
            {"query_string_us": "ab:(cool)"},
            us_result=([dead, live], "USPTO 2 hits")))
        self.assertIn("22222222", result["text"])
        self.assertNotIn("11111111", result["text"])

    def test_all_hits_dead_says_so_instead_of_no_results(self):
        # 有命中但全部失效时，不能笼统说"未返回结果"。
        dead = {"applicationNumberText": "11111111", "status": "Abandoned",
                "applicationMetaData": {"inventionTitle": "DEADONE"}}
        _agent, result = asyncio.run(self._run(
            {"query_string_us": "ab:(cool)"},
            us_result=([dead], "USPTO 1 hits")))
        self.assertIn("失效", result["text"])
        self.assertNotIn("未返回结果", result["text"])

    def test_second_call_merges_instead_of_overwriting(self):
        # Production incident (2026-08-27): the LLM called patent_search_dual
        # four times per the ladder prompt.  The first two calls returned
        # US + CN candidates; the later calls hit US 404 with the auto-ladder
        # budget exhausted, and each call unconditionally overwrote
        # _pending_raw_items — the final CN-only result silently dropped the
        # earlier US candidates from the result list.  A later, narrower
        # call must MERGE into the pending pool, never discard it.
        us_items = [{"applicationNumberText": "19511555", "applicationMetaData": {
            "inventionTitle": "Cooling device", "firstApplicantName": "Intel",
            "filingDate": "2024-01-01"}}]
        cn_items = [{"patent_id": "CN118000001A", "source": "baiten",
                     "title": "散热装置"}]
        agent = _FakeAgent()
        agent, result = asyncio.run(self._run(
            {"query_string_us": "ab:(cool)", "query_string_cn": "ti:(散热)"},
            us_result=(us_items, "USPTO 1 hits"),
            cn_result=(cn_items, "Baiten 1 hits"), agent=agent))
        self.assertEqual(len(agent._pending_raw_items), 2)
        # Second call: US 404 (auto-ladder budget already spent), CN returns
        # a new patent.  The US candidate from the first call must survive.
        cn_items2 = [{"patent_id": "CN118000002A", "source": "baiten",
                      "title": "除湿装置"}]
        agent, result = asyncio.run(self._run(
            {"query_string_us": "ab:(cool)", "query_string_cn": "ti:(除湿)"},
            us_result=([], "USPTO HTTP 404"),
            cn_result=(cn_items2, "Baiten 1 hits"), agent=agent))
        ids = [c.get("patent_id") or c.get("applicationNumberText")
               for c in agent._pending_raw_items]
        self.assertEqual(len(agent._pending_raw_items), 3)
        self.assertIn("19511555", ids)  # first call's US candidate kept
        self.assertIn("CN118000001A", ids)
        self.assertIn("CN118000002A", ids)

    def test_merge_dedupes_repeated_patents(self):
        # The same CN patent surfaced by two ladder queries appears once in
        # the pending pool (first occurrence wins).
        cn_items = [{"patent_id": "CN118000001A", "source": "baiten",
                     "title": "散热装置"}]
        agent = _FakeAgent()
        agent, _ = asyncio.run(self._run(
            {"query_string_us": "ab:(cool)", "query_string_cn": "ti:(散热)"},
            us_result=([], "USPTO HTTP 404"),
            cn_result=(cn_items, "Baiten 1 hits"), agent=agent))
        agent, _ = asyncio.run(self._run(
            {"query_string_us": "ab:(cool)", "query_string_cn": "ti:(干燥)"},
            us_result=([], "USPTO HTTP 404"),
            cn_result=(cn_items, "Baiten 1 hits"), agent=agent))
        self.assertEqual(len(agent._pending_raw_items), 1)

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
        # 数据来源/状态噪声不进用户可见文本 (2026-09-01)
        self.assertNotIn("Baiten failed", result["text"])
        self.assertEqual(len(agent._pending_raw_items), 1)

    def test_both_empty_sources(self):
        agent, result = asyncio.run(self._run(
            {"query_string_cn": "ti:(散热)"},
            cn_result=([], "Baiten not configured")))
        self.assertIn("两个数据源均未返回结果", result["text"])
        # 数据来源/状态噪声不进用户可见文本 (2026-09-01)
        self.assertNotIn("Baiten not configured", result["text"])
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

        async def _cn(q, page=1, page_size=20, agent=None, enrich=True):
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

        async def _cn(q, page=1, page_size=20, agent=None, enrich=True):
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
        # 首轮 tightest + 自动补跑（tightest 已记入 tried，补跑只执行载体词式；
        # US 首轮 0 命中同样补跑 1 条——共享预算）
        self.assertEqual(cn_calls, ["ti:(散热)", "ti:(载体)"])
        self.assertIn("CN118000002A", result["text"])
        # 数据来源/状态噪声不进用户可见文本 (2026-09-01), 只进日志
        self.assertNotIn("已自动补跑中国专利阶梯式", result["text"])
        self.assertEqual(agent._patent_auto_used, {"us": 1, "cn": 1})
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

        async def _cn(q, page=1, page_size=20, agent=None, enrich=True):
            return [], "Baiten 0 hits (gateway 0 records)"

        with patch("sources.agents.react_tools._uspto_search_by_query", _us), \
             patch("sources.agents.react_tools._baiten_search_by_query", _cn):
            agent = _FakeAgent()
            result = asyncio.run(_run_patent_search(
                agent, {"query_string_us": "us-tight",
                        "query_string_cn": "ti:(散热)"}, "en"))
        self.assertEqual(us_calls, ["us-tight", "us-loose"])
        # 数据来源/状态噪声不进用户可见文本 (2026-09-01)
        self.assertNotIn("Auto-ran 1 US ladder", result["text"])
        self.assertIn("19511555", result["text"])

    def test_non_preferred_source_gets_fallback(self):
        # zh 提问：CN 首轮 0 命中自动补跑后，US 首轮 0 命中也要补跑
        # （用户要求中美都有结果——单个 404 不能饿死另一源）。
        cn_calls = []
        us_calls = []

        async def _cn(q, page=1, page_size=20, agent=None, enrich=True):
            cn_calls.append(q)
            if q == "ti:(载体)":
                return [{"patent_id": "CN118000002A", "source": "baiten",
                         "title": "载体词命中"}], "Baiten 1 hits"
            return [], "Baiten 0 hits (gateway 0 records)"

        async def _us(q, page=1, page_size=20, agent=None):
            us_calls.append(q)
            if q == "us-loose":
                return [{"applicationNumberText": "19511555",
                         "applicationMetaData": {
                             "inventionTitle": "Cooling",
                             "firstApplicantName": "Intel",
                             "filingDate": "2024-01-01"}}], "USPTO 1 hits"
            return [], "USPTO 0 hits"

        with patch("sources.agents.react_tools._uspto_search_by_query", _us), \
             patch("sources.agents.react_tools._baiten_search_by_query", _cn):
            agent = _FakeAgent()
            result = asyncio.run(_run_patent_search(
                agent, {"query_string_us": "us-tight",
                        "query_string_cn": "ti:(散热)"}, "zh"))
        # CN 首轮 tightest → CN 补跑载体词式 → US 首轮 tightest → US 补跑 loose
        self.assertIn("CN118000002A", result["text"])
        self.assertIn("19511555", result["text"])
        self.assertEqual(len(agent._pending_raw_items), 2)
        self.assertIn("ti:(载体)", cn_calls)
        self.assertIn("us-loose", us_calls)

    def test_auto_ladder_respects_per_source_cap(self):
        # 预算按源独立:CN 已用 3/4,补跑只能再执行 1 条(US 侧另有自己的 4 条)。
        cn_calls = []

        async def _cn(q, page=1, page_size=20, agent=None, enrich=True):
            cn_calls.append(q)
            return [], "Baiten 0 hits (gateway 0 records)"

        async def _us(q, page=1, page_size=20, agent=None):
            return [], "USPTO 0 hits"

        with patch("sources.agents.react_tools._uspto_search_by_query", _us), \
             patch("sources.agents.react_tools._baiten_search_by_query", _cn):
            agent = _FakeAgent()
            agent._patent_auto_used = {"us": 0, "cn": 3}  # CN 距上限只剩 1
            result = asyncio.run(_run_patent_search(
                agent, {"query_string_us": "us-tight",
                        "query_string_cn": "ti:(散热)"}, "zh"))
        self.assertEqual(cn_calls, ["ti:(散热)", "ti:(载体)"])
        self.assertEqual(agent._patent_auto_used, {"us": 1, "cn": 4})

    def test_cn_ladder_not_starved_by_us_budget_exhaustion(self):
        # 生产事故 (2026-08-29): zh 提问首选源 CN 0 命中时,CN 自动阶梯因
        # US 侧先用光共享预算(每请求 4 条)而静默跳过,结果 us=0 cn=0 total=0。
        # 预算按源独立后,US 用尽不再影响 CN 兜底。
        cn_calls = []

        async def _cn(q, page=1, page_size=20, agent=None, enrich=True):
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
            agent._patent_auto_used = {"us": REACT_PATENT_AUTO_LADDER_MAX, "cn": 0}  # US 已用尽
            result = asyncio.run(_run_patent_search(
                agent, {"query_string_us": "us-tight",
                        "query_string_cn": "ti:(散热)"}, "zh"))
        # CN 兜底不受 US 预算耗尽影响;US 侧 0 命中但预算尽,静默降级
        self.assertEqual(cn_calls, ["ti:(散热)", "ti:(载体)"])
        self.assertIn("CN118000002A", result["text"])
        # 数据来源/状态噪声不进用户可见文本 (2026-09-01), 只进日志
        self.assertNotIn("已自动补跑中国专利阶梯式", result["text"])
        self.assertEqual(agent._patent_auto_used, {"us": REACT_PATENT_AUTO_LADDER_MAX, "cn": 1})
        self.assertEqual(len(agent._pending_raw_items), 1)

    def test_budget_exhausted_logs_warning(self):
        # 预算耗尽必须打 warning——之前静默 return 0,日志里没有任何痕迹,
        # 线上 total=0 无从排查(2026-08-29 事故)。
        calls = []

        class _Logger:
            def info(self, *a, **k):
                calls.append(("info", a[0]))

            def warning(self, *a, **k):
                calls.append(("warning", a[0]))

        async def _search(q, page=1, page_size=20):
            return [], "USPTO 0 hits"

        agent = _FakeAgent()
        agent.logger = _Logger()
        agent._patent_auto_used = {"us": REACT_PATENT_AUTO_LADDER_MAX, "cn": 0}
        result = asyncio.run(_auto_run_patent_ladder(
            agent, ["us-loose"], _search, [], [], "zh", "us", 1, 20))
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "warning")
        self.assertIn("budget exhausted", calls[0][1])
        self.assertIn("1 untried", calls[0][1])


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

    def test_legal_status_tool_registered_in_every_mode(self):
        # 需求#18: 法律状态问题可能在任何国别模式下到来，确定性路径
        # 不能依赖 LLM 挑到合适的 KB 工具 —— 无条件注册。
        for src in ("dual", "cn", "uspto"):
            registry, _ = asyncio.run(self._build(src))
            entry = registry.get(PATENT_LEGAL_STATUS_TOOL_NAME)
            self.assertIsNotNone(entry, src)
            self.assertEqual(entry.kind, "patent_legal_status", src)


class TestEnrichmentOncePerToolCall(unittest.TestCase):
    """富化从「每次佰腾检索一次」改为「每次工具调用一次」。

    生产日志（2026-09-13）：一次 ``patent_search_dual`` 会跑 2–4 次佰腾
    检索（首轮 + 自动补跑阶梯），每次都内联 ``await`` 富化（~1.5s），全部
    串在关键路径上。改为收尾统一跑一遍 —— 仍在**排名之前**，因为排名要读
    ``status``；配合每请求缓存，同号也不会重查。
    """

    _CN_ITEM = {"patent_id": "CN118000001A", "source": "baiten",
                "app_num": "CN202311111111.1", "title": "散热装置"}

    def _drive(self, cn_runs):
        """cn_runs: 连续几次佰腾检索的返回值（最后一次起循环复用）。"""
        calls = {"n": 0}

        async def _cn(q, page=1, page_size=20, agent=None, enrich=True):
            idx = min(calls["n"], len(cn_runs) - 1)
            calls["n"] += 1
            return cn_runs[idx]

        async def _us(q, page=1, page_size=20):
            return [], "USPTO 0 hits"

        with patch.object(react_tools, "_baiten_search_by_query", _cn), \
             patch.object(react_tools, "_uspto_search_by_query", _us), \
             patch.object(react_tools, "_enrich_baiten_law_status",
                          new=AsyncMock()) as enrich_mock, \
             patch.object(react_tools, "_baiten_client_or_none",
                          return_value=object()):
            agent = _FakeAgent()
            asyncio.run(_run_patent_search(
                agent, {"query_string_cn": "ti:(散热)"}, "zh"))
        return calls["n"], enrich_mock

    def test_one_enrichment_for_a_single_search(self):
        n, enrich_mock = self._drive([([self._CN_ITEM], "CN 1 hits")])
        self.assertEqual(n, 1)
        self.assertEqual(enrich_mock.await_count, 1)

    def test_one_enrichment_across_auto_ladder_rounds(self):
        # 首轮 0 命中 → 自动补跑阶梯多发几次检索；富化仍然只跑一次。
        n, enrich_mock = self._drive(
            [([], "CN 0 hits"), ([self._CN_ITEM], "CN 1 hits")])
        self.assertGreater(n, 1)
        self.assertEqual(enrich_mock.await_count, 1)

    def test_enrichment_receives_only_cn_candidates(self):
        _n, enrich_mock = self._drive([([self._CN_ITEM], "CN 1 hits")])
        passed = enrich_mock.await_args[0][1]
        self.assertTrue(passed)
        self.assertTrue(all(c.get("source") == "baiten" for c in passed))

    def test_no_baiten_client_skips_enrichment(self):
        with patch.object(react_tools, "_baiten_search_by_query",
                          new=AsyncMock(return_value=(
                              [self._CN_ITEM], "CN 1 hits"))), \
             patch.object(react_tools, "_uspto_search_by_query",
                          new=AsyncMock(return_value=([], "USPTO 0"))), \
             patch.object(react_tools, "_enrich_baiten_law_status",
                          new=AsyncMock()) as enrich_mock, \
             patch.object(react_tools, "_baiten_client_or_none",
                          return_value=None):
            asyncio.run(_run_patent_search(
                _FakeAgent(), {"query_string_cn": "ti:(散热)"}, "zh"))
        enrich_mock.assert_not_awaited()


class TestCnPoolCandidateKeepsNativeKey(unittest.TestCase):
    """需求#29: 池化曾丢弃 app_num，导致下游再也拿不到 CN 申请号 ——
    而它正是佰腾法律状态/取件 API 需要的键。"""

    def test_app_num_survives_pool_mapping(self):
        item = {"patent_id": "CN116570413A", "patent_number": "CN116570413A",
                "title": "某方法", "applicant": "某公司",
                "app_num": "CN202310123456.7", "source": "baiten"}
        pool = _cn_item_to_pool_candidate(item)
        self.assertEqual(pool["app_num"], "CN202310123456.7")

    def test_missing_app_num_is_empty_string(self):
        item = {"patent_id": "CN116570413A", "source": "baiten"}
        self.assertEqual(_cn_item_to_pool_candidate(item)["app_num"], "")


class TestDeepAnalysisDescriptionRedirect(unittest.TestCase):
    """需求#18: 法律状态问题此前被 deep_analysis 描述推回关键词检索
    （"请走检索"），而检索答不了状态级问题 —— 必须改指专用工具。"""

    def test_zh_redirects_to_legal_status_tool(self):
        d = _builtin_deep_analysis_description("zh")
        self.assertNotIn("请走检索", d)
        self.assertIn(PATENT_LEGAL_STATUS_TOOL_NAME, d)

    def test_en_redirects_to_legal_status_tool(self):
        d = _builtin_deep_analysis_description("en")
        self.assertIn(PATENT_LEGAL_STATUS_TOOL_NAME, d)



class TestBuiltinDeepAnalysisTool(unittest.TestCase):
    """#22: family/prosecution analysis entry must exist even for users whose
    knowledge base matched no tailored type-3 long-task tool."""

    async def _build(self, candidates):
        with patch("sources.agents.react_tools.get_knowledge_tool_candidates",
                   return_value=candidates):
            return await build_tool_set(
                _FakeAgent(), "u1", "分析 CN105414512A 的全球同族审查差异",
                patent_source="dual")

    def test_registered_when_no_tailored_long_task(self):
        registry, tools = asyncio.run(self._build([]))
        entry = registry.get(BUILTIN_DEEP_ANALYSIS_TOOL_NAME)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.kind, "long_task")
        self.assertIsNone(entry.knowledge)
        bound_names = {t["name"] for t in tools}
        self.assertIn(BUILTIN_DEEP_ANALYSIS_TOOL_NAME, bound_names)

    def test_not_registered_when_tailored_long_task_exists(self):
        knowledge = SimpleNamespace(
            id="k1", type=3, scene_id=1,
            question="zh:审查历史分析|en:prosecution history analysis",
            description="",
        )
        registry, _tools = asyncio.run(self._build([(knowledge, None)]))
        self.assertNotIn(BUILTIN_DEEP_ANALYSIS_TOOL_NAME, registry)
        self.assertTrue(any(e.kind == "long_task" and e.knowledge is not None
                            for e in registry.values()))


class TestUsCitingNote(unittest.TestCase):
    """#22c: US hits on a single-CN-publication search are citing docs."""

    def _us(self, n=20):
        return [{"applicationNumberText": f"19{i:06d}"} for i in range(n)]

    def test_family_intent_gets_deep_task_nudge(self):
        note = _us_citing_note(
            "分析 CN105414512A 及其全球同族申请的审查差异",
            '(CN105414512A OR "CN 105414512 A")',
            [{"patent_id": "CN105414512A", "source": "baiten"}],
            self._us(), lang="zh")
        self.assertIn("引用该中国专利", note)
        self.assertIn("同族", note)

    def test_generic_query_gets_light_annotation(self):
        note = _us_citing_note(
            "查询一下这个专利的相关技术",
            "CN105414512A",
            [{"patent_id": "CN105414512A", "source": "baiten"}],
            self._us(), lang="zh")
        self.assertIn("引用该中国专利", note)
        self.assertNotIn("深度分析任务", note)

    def test_en_family_intent(self):
        note = _us_citing_note(
            "Analyze CN105414512A worldwide family examination differences",
            "CN105414512A",
            [{"patent_id": "CN105414512A", "source": "baiten"}],
            self._us(), lang="en")
        self.assertTrue(note)

    def test_no_note_when_cn_not_single_publication(self):
        # Two CN publications in the query → the pattern does not apply.
        self.assertEqual(
            _us_citing_note(
                "对比 CN105414512A 与 CN112139463A",
                "CN105414512A OR CN112139463A",
                [{"patent_id": "CN105414512A", "source": "baiten"}],
                self._us(), lang="zh"),
            "")

    def test_no_note_when_no_us_hits(self):
        self.assertEqual(
            _us_citing_note(
                "分析 CN105414512A 的同族",
                "CN105414512A",
                [{"patent_id": "CN105414512A", "source": "baiten"}],
                [], lang="zh"),
            "")

    def test_no_note_when_cn_candidates_not_single(self):
        self.assertEqual(
            _us_citing_note(
                "分析 CN105414512A 的同族",
                "CN105414512A",
                [{"patent_id": "CN105414512A", "source": "baiten"},
                 {"patent_id": "CN112139463A", "source": "baiten"}],
                self._us(), lang="zh"),
            "")


if __name__ == "__main__":
    unittest.main()
