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
from sources.long_task.legal_status import official_portal


def _run(coro):
    return asyncio.run(coro)


def _agent(candidates=None):
    return SimpleNamespace(
        logger=None,
        _number_candidates=candidates or [],
        _number_cross_done=False,
        _number_cross_used=0,
        _legal_status_used=0,
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


# 中靶形态: 记录自带被查的申请号 (需求#36 后 "主源命中即收手" 要求中靶)
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
    "patent_number": "CN117941643A",   # 中靶槽: 公开号含被查号码
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
                                                       "CN 1 hits"))) as bm, \
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
                          new=AsyncMock(return_value=([], "CN 0 hits"))), \
             patch.object(react_tools, "_uspto_search_by_number",
                          new=AsyncMock(return_value=([_USPTO_ITEM],
                                                       "USPTO 1 hits"))) as um:
            merged, notes = _run(
                react_tools._lookup_number_candidates(agent, candidates))
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["applicationNumberText"], "117941643")
        um.assert_awaited_once()
        self.assertTrue(any("USPTO" in n for n in notes))
        self.assertTrue(any("CN" in n for n in notes))

    def test_us_candidate_zero_then_baiten(self):
        candidates = [{"country": "US", "display": "US19511555",
                       "lookups": ["19511555"]}]
        agent = _agent(candidates)
        with patch.object(react_tools, "_uspto_search_by_number",
                          new=AsyncMock(return_value=([], "USPTO 0 hits"))), \
             patch.object(react_tools, "_baiten_search_by_query",
                          new=AsyncMock(return_value=([_BAITEN_ITEM],
                                                       "CN 1 hits"))) as bm:
            merged, _notes = _run(
                react_tools._lookup_number_candidates(agent, candidates))
        self.assertEqual(len(merged), 1)
        bm.assert_awaited_once()

    def test_budget_caps_legs(self):
        candidates = [{"country": "CN", "display": "CN117941643",
                       "lookups": ["CN117941643A", "117941643"]}]
        agent = _agent(candidates)
        with patch.object(react_tools, "_baiten_search_by_query",
                          new=AsyncMock(return_value=([], "CN 0 hits"))), \
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


class TestCandidateConfirmationHints(unittest.TestCase):
    """需求#24: a bare number must never close with plain "not found" —
    surface what was tried and what the user may have meant."""

    def test_zero_hit_observation_offers_candidates(self):
        agent = _agent([{"country": "CN", "display": "CN117941643",
                         "reason": "9 位纯数字以 1 开头，符合中国公开号核心号段",
                         "lookups": ["CN117941643A", "CN117941643",
                                     "117941643"]}])
        with patch.object(react_tools, "_lookup_number_candidates",
                          new=AsyncMock(return_value=([], ["CN 0 hits",
                                                           "USPTO 0 hits"]))), \
             patch.object(react_tools, "_merge_pending_items",
                          side_effect=lambda ex, new: list(ex or []) + list(new)), \
             patch.object(react_tools, "_rank_builtin_patent_pool",
                          new=AsyncMock(side_effect=lambda a, items, lang: items)), \
             patch.object(react_tools, "_order_pending_for_lang",
                          side_effect=lambda items, lang: items):
            obs = _run(react_tools._run_patent_number_resolve(
                agent, {"number": "117941643"}, "zh"))
        self.assertIn("未按该号码查到专利记录", obs["text"])
        self.assertIn("您可能查的是", obs["text"])
        self.assertIn("CN117941643", obs["text"])

    def test_hit_observation_has_no_hint_section(self):
        # Direct hits keep the existing digest — no candidate noise.
        agent = _agent([{"country": "CN", "display": "CN117941643",
                         "lookups": ["CN117941643A"]}])
        with patch.object(react_tools, "_lookup_number_candidates",
                          new=AsyncMock(return_value=([_BAITEN_ITEM],
                                                       ["CN 1 hits"]))), \
             patch.object(react_tools, "_merge_pending_items",
                          side_effect=lambda ex, new: list(ex or []) + list(new)), \
             patch.object(react_tools, "_rank_builtin_patent_pool",
                          new=AsyncMock(side_effect=lambda a, items, lang: items)), \
             patch.object(react_tools, "_order_pending_for_lang",
                          side_effect=lambda items, lang: items):
            obs = _run(react_tools._run_patent_number_resolve(
                agent, {"number": "CN117941643"}, "zh"))
        self.assertIn("CN117941643A", obs["text"])
        self.assertNotIn("您可能查的是", obs["text"])

    def test_hint_text_lists_candidates_bilingually(self):
        cands = [{"country": "CN", "display": "CN114948588",
                  "reason": "9 位纯数字以 1 开头，符合中国公开号核心号段",
                  "lookups": ["CN114948588A"]},
                 {"country": "US", "display": "US19511555",
                  "reason": "纯数字无法区分美国授权号与申请号",
                  "lookups": ["19511555"]}]
        zh = react_tools._candidate_confirmation_hints(cands, "zh")
        en = react_tools._candidate_confirmation_hints(cands, "en")
        self.assertIn("您可能查的是", zh)
        self.assertIn("CN114948588", zh)
        self.assertIn("US19511555", zh)
        self.assertIn("did you mean", en.lower())
        self.assertIn("US19511555", en)
        self.assertEqual(react_tools._candidate_confirmation_hints([], "zh"),
                         "")

    def test_unsupported_shape_hint_carries_parser_reason(self):
        # ≥9-digit junk parses to id_type=unsupported with an actionable
        # reason ("补全如 WO…") — the zero-hit card must surface it.
        cands = [{"country": "", "display": "202399999999",
                  "reason": "疑似残缺国际申请号，请补全后再查",
                  "lookups": []}]
        zh = react_tools._candidate_confirmation_hints(cands, "zh")
        self.assertIn("您可能查的是", zh)
        self.assertIn("202399999999", zh)
        self.assertIn("疑似残缺", zh)


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
            [_BAITEN_ITEM], ["CN 1 hits"])
        self.assertEqual(obs["kind"], "observation")
        self.assertIn("CN117941643A", obs["text"])
        self.assertEqual(len(agent._pending_raw_items), 1)

    def test_zero_observation_lists_sources_checked(self):
        obs, _agent = self._resolve(
            [{"country": "CN", "display": "CN117941643",
              "lookups": ["CN117941643A"]}],
            [], ["CN 0 hits", "USPTO 0 hits"])
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
                                                       ["CN 1 hits"]))), \
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
                          new=AsyncMock(return_value=([], ["CN 0 hits"]))), \
             patch.object(react_tools, "_rank_pending_pool",
                          new=AsyncMock(return_value=([], ""))):
            ranked, _rn, note = _run(
                react_tools._auto_number_cross_round(agent, "zh"))
        self.assertEqual(ranked, [])
        self.assertIn("号码跨源复核无命中", note)


class TestOfficialPortal(unittest.TestCase):
    """需求#18: 数据源未覆盖时必须给出官方查询入口。"""

    def test_known_countries(self):
        self.assertIn("cnipa", official_portal("CN"))
        self.assertIn("uspto", official_portal("US"))

    def test_case_insensitive(self):
        self.assertEqual(official_portal("cn"), official_portal("CN"))

    def test_unknown_country_is_empty(self):
        self.assertEqual(official_portal("ZZ"), "")
        self.assertEqual(official_portal(""), "")


_ENTRY_CN = {
    "display": "CN116570413A",
    "app_num": "CN202310123456.7",
    "country": "CN",
    "status": "专利权终止",
    "status_date": "2024-08-15",
    "timeline": [
        {"date": "2024-08-15", "lawStatus": "专利权终止"},
        {"date": "2023-11-20", "lawStatus": "授权"},
    ],
    "reviews": [],
    "reviews_checked": True,
    "checked": ["cn_legal_status"],
    "covered": True,
}


class TestLegalStatusDigest(unittest.TestCase):
    """需求#18 observation 渲染：状态 + 时间线 + 复审事实 + 诚实缺口。"""

    def test_zh_reports_status_timeline_and_disclaimer(self):
        out = react_tools._legal_status_digest([_ENTRY_CN], "zh")
        self.assertIn("CN116570413A", out)
        self.assertIn("专利权终止", out)
        self.assertIn("2023-11-20", out)
        self.assertIn("不构成法律意见", out)

    def test_en_reports_status_timeline_and_disclaimer(self):
        out = react_tools._legal_status_digest([_ENTRY_CN], "en")
        self.assertIn("CN116570413A", out)
        self.assertIn("Patent Terminated", out)
        self.assertIn("not legal advice", out.lower())

    def test_reviews_rendered_when_present(self):
        entry = dict(_ENTRY_CN, reviews=[{
            "declareNum": "5W123456", "declareDate": "2025-01-10",
            "lawBase": "", "fullText": "维持专利权有效。"}])
        out = react_tools._legal_status_digest([entry], "zh")
        self.assertIn("5W123456", out)

    def test_no_reviews_states_none_recorded(self):
        out = react_tools._legal_status_digest([_ENTRY_CN], "zh")
        self.assertIn("未检索到", out)

    def test_reviews_never_queried_is_stated_honestly(self):
        # 检索期富化只在命中 ≤3 条时才拉 FSWX —— 更宽的命中列表里复审
        # 数据从未被查询，此时宣称"未检索到"就是无据的否定。
        entry = dict(_ENTRY_CN, reviews=[], reviews_checked=False)
        out = react_tools._legal_status_digest([entry], "zh")
        self.assertNotIn("未检索到", out)
        self.assertIn("未查询", out)

    def test_reviews_queried_and_empty_states_none_recorded(self):
        entry = dict(_ENTRY_CN, reviews=[], reviews_checked=True)
        out = react_tools._legal_status_digest([entry], "zh")
        self.assertIn("未检索到", out)

    def test_uncovered_entry_marks_gap_with_official_portal(self):
        entry = {"display": "US19511555", "app_num": "", "country": "US",
                 "status": "", "status_date": "", "timeline": [],
                 "reviews": [], "checked": ["USPTO"], "covered": False}
        out = react_tools._legal_status_digest([entry], "zh")
        self.assertIn("未覆盖", out)
        self.assertIn("ppubs.uspto.gov", out)

    def test_uncovered_does_not_claim_no_reviews(self):
        # 未查过就不能说"没有"——不可验证的不下断言。
        entry = {"display": "US19511555", "app_num": "", "country": "US",
                 "status": "", "status_date": "", "timeline": [],
                 "reviews": [], "checked": ["USPTO"], "covered": False}
        out = react_tools._legal_status_digest([entry], "en")
        self.assertNotIn("No re-examination", out)

    def test_empty_entries_is_empty_string(self):
        self.assertEqual(react_tools._legal_status_digest([], "zh"), "")

    def test_multiple_entries_all_reported(self):
        second = dict(_ENTRY_CN, display="CN118453362A")
        out = react_tools._legal_status_digest([_ENTRY_CN, second], "zh")
        self.assertIn("CN116570413A", out)
        self.assertIn("CN118453362A", out)


class TestBaitenLawLookup(unittest.TestCase):
    """需求#18: 按申请号键直查 FLZT/FSWX —— 独立预算，不挤占跨源检索。"""

    def _client(self, timeline=None, reviews=None, tl_exc=None):
        return SimpleNamespace(
            query_legal_state_timeline=AsyncMock(
                side_effect=tl_exc, return_value=timeline or []),
            query_patent_review=AsyncMock(return_value=reviews or []),
        )

    def test_returns_status_timeline_and_reviews(self):
        client = self._client(
            timeline=[{"date": "2024-08-15", "lawStatus": "专利权终止"},
                      {"date": "2023-11-20", "lawStatus": "授权"}],
            reviews=[{"declareNum": "5W1", "declareDate": "2025-01-10",
                      "fullText": "维持专利权有效。"}])
        agent = _agent()
        with patch.object(react_tools, "_baiten_client_or_none",
                          return_value=client):
            res = _run(react_tools._baiten_law_lookup(
                agent, "CN202310123456.7"))
        self.assertEqual(res["status"], "专利权终止")
        self.assertEqual(res["status_date"], "2024-08-15")
        self.assertEqual(len(res["timeline"]), 2)
        self.assertEqual(len(res["reviews"]), 1)

    def test_budget_is_isolated_from_number_cross(self):
        client = self._client(timeline=[{"date": "d", "lawStatus": "授权"}])
        agent = _agent()
        with patch.object(react_tools, "_baiten_client_or_none",
                          return_value=client):
            _run(react_tools._baiten_law_lookup(agent, "CN1"))
        self.assertEqual(agent._legal_status_used, 1)
        self.assertEqual(agent._number_cross_used, 0)

    def test_reuses_flzt_cache_from_search_enrichment(self):
        # 检索期已查过的号，法律状态工具不该再打一次 FLZT —— 同一请求内
        # 同一申请号的状态不会变。缓存按**线调用**分开，所以 FSWX 仍会
        # 按需补查（大列表富化时没查过它）。
        agent = _agent()
        agent._law_flzt_cache = {
            "CN202310123456.7": [{"date": "2025-09-09", "lawStatus": "授权"}]}
        client = SimpleNamespace(
            query_legal_state_timeline=AsyncMock(),
            query_patent_review=AsyncMock(return_value=[]))
        with patch.object(react_tools, "_baiten_client_or_none",
                          return_value=client):
            res = _run(react_tools._baiten_law_lookup(
                agent, "CN202310123456.7"))
        self.assertEqual(res["status"], "授权")
        client.query_legal_state_timeline.assert_not_awaited()
        client.query_patent_review.assert_awaited()   # FSWX 未查过 → 补查

    def test_unconfigured_client_returns_empty(self):
        agent = _agent()
        with patch.object(react_tools, "_baiten_client_or_none",
                          return_value=None):
            res = _run(react_tools._baiten_law_lookup(agent, "CN1"))
        self.assertEqual(res, {})

    def test_budget_cap_stops_lookup(self):
        agent = _agent()
        agent._legal_status_used = react_tools.LEGAL_STATUS_MAX_LOOKUPS
        with patch.object(react_tools, "_baiten_client_or_none") as f:
            res = _run(react_tools._baiten_law_lookup(agent, "CN1"))
        self.assertEqual(res, {})
        f.assert_not_called()

    def test_gateway_failure_degrades_to_empty(self):
        client = self._client(tl_exc=RuntimeError("gateway down"))
        agent = _agent()
        with patch.object(react_tools, "_baiten_client_or_none",
                          return_value=client):
            res = _run(react_tools._baiten_law_lookup(agent, "CN1"))
        self.assertEqual(res, {})

    def test_empty_app_num_returns_empty(self):
        agent = _agent()
        with patch.object(react_tools, "_baiten_client_or_none") as f:
            res = _run(react_tools._baiten_law_lookup(agent, ""))
        self.assertEqual(res, {})
        f.assert_not_called()


class TestLegalStatusToolExecutor(unittest.TestCase):
    def test_no_recognizable_number(self):
        agent = _agent([])
        obs = _run(react_tools._run_patent_legal_status(
            agent, {"number": "量子纠缠装置"}, "zh"))
        self.assertIn("未能识别出专利号格式", obs["text"])

    def test_cn_resolved_number_reports_status(self):
        agent = _agent()
        item = dict(_BAITEN_ITEM)
        item["status"] = "专利权终止"
        item["legal_timeline"] = [
            {"date": "2024-08-15", "lawStatus": "专利权终止"},
            {"date": "2023-11-20", "lawStatus": "授权"}]
        with patch.object(react_tools, "_lookup_number_candidates",
                          new=AsyncMock(return_value=([item], ["CN 1"]))):
            obs = _run(react_tools._run_patent_legal_status(
                agent, {"number": "CN117941643A"}, "zh"))
        self.assertIn("专利权终止", obs["text"])

    def test_empty_status_falls_back_to_direct_law_lookup(self):
        # 检索附带的富化可能失败（status 为空）——此时不能直接对用户说
        # "未覆盖"，手里有申请号就应当按号直查。
        agent = _agent()
        item = dict(_BAITEN_ITEM)
        item["app_num"] = "CN202311794164.0"
        with patch.object(react_tools, "_lookup_number_candidates",
                          new=AsyncMock(return_value=([item], ["CN 1"]))), \
             patch.object(react_tools, "_baiten_law_lookup",
                          new=AsyncMock(return_value={
                              "status": "专利权终止",
                              "timeline": [{"date": "2024-08-15",
                                            "lawStatus": "专利权终止"}]})) as lm:
            obs = _run(react_tools._run_patent_legal_status(
                agent, {"number": "CN117941643A"}, "zh"))
        self.assertIn("专利权终止", obs["text"])
        lm.assert_awaited()

    def test_no_app_num_does_not_attempt_direct_lookup(self):
        agent = _agent()
        item = dict(_BAITEN_ITEM)          # 无 app_num
        with patch.object(react_tools, "_lookup_number_candidates",
                          new=AsyncMock(return_value=([item], ["CN 1"]))), \
             patch.object(react_tools, "_baiten_law_lookup",
                          new=AsyncMock()) as lm:
            obs = _run(react_tools._run_patent_legal_status(
                agent, {"number": "CN117941643A"}, "zh"))
        lm.assert_not_awaited()
        self.assertIn("未覆盖", obs["text"])

    def test_unresolved_number_carries_confirmation_hints(self):
        # 需求#24 行为必须延续：禁止以"未找到"直接结案。
        agent = _agent()
        with patch.object(react_tools, "_lookup_number_candidates",
                          new=AsyncMock(return_value=([], ["CN 0 hits"]))):
            obs = _run(react_tools._run_patent_legal_status(
                agent, {"number": "CN117941643A"}, "zh"))
        self.assertIn("您可能查的是", obs["text"])


class TestNativeKeyReadback(unittest.TestCase):
    """需求#29: 有原生键（CN 申请号）时按号直查——不再把公开号当作
    佰腾的自由文本检索词，这是「系统读不回自己刚产出的号码」的真因。"""

    _CAND = {"country": "CN", "display": "CN116570413A",
             "lookups": ["CN116570413A", "CN116570413", "116570413"],
             "native_key": "CN202310123456.7",
             "native_key_kind": "app_num"}

    def test_prefers_native_key_lookup(self):
        agent = _agent([self._CAND])
        with patch.object(react_tools, "_baiten_law_lookup",
                          new=AsyncMock(return_value={
                              "status": "专利权终止",
                              "timeline": [{"date": "2024-08-15",
                                            "lawStatus": "专利权终止"}]})), \
             patch.object(react_tools, "_baiten_search_by_query",
                          new=AsyncMock(return_value=([], "CN 0"))) as bs, \
             patch.object(react_tools, "_uspto_search_by_number",
                          new=AsyncMock(return_value=([], "USPTO 0"))):
            merged, _notes = _run(
                react_tools._lookup_number_candidates(agent, [self._CAND]))
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["status"], "专利权终止")
        self.assertEqual(merged[0]["app_num"], "CN202310123456.7")
        bs.assert_not_awaited()          # 原生键命中 → 不再自由文本检索

    def test_falls_back_to_free_text_when_key_lookup_empty(self):
        agent = _agent([self._CAND])
        with patch.object(react_tools, "_baiten_law_lookup",
                          new=AsyncMock(return_value={})), \
             patch.object(react_tools, "_baiten_search_by_query",
                          new=AsyncMock(return_value=([_BAITEN_ITEM],
                                                      "CN 1"))) as bs:
            merged, _notes = _run(
                react_tools._lookup_number_candidates(agent, [self._CAND]))
        self.assertEqual(len(merged), 1)
        bs.assert_awaited()

    def test_without_native_key_uses_free_text_only(self):
        cand = {"country": "CN", "display": "CN116570413A",
                "lookups": ["CN116570413A"]}
        agent = _agent([cand])
        with patch.object(react_tools, "_baiten_law_lookup",
                          new=AsyncMock()) as lm, \
             patch.object(react_tools, "_baiten_search_by_query",
                          new=AsyncMock(return_value=([_BAITEN_ITEM],
                                                      "CN 1"))):
            _run(react_tools._lookup_number_candidates(agent, [cand]))
        lm.assert_not_awaited()

    def test_unreadable_records_still_consult_the_cross_source(self):
        # 原生键查不到时不能就此收手 —— 对侧数据源仍要打。
        agent = _agent([self._CAND])
        with patch.object(react_tools, "_baiten_law_lookup",
                          new=AsyncMock(return_value={})), \
             patch.object(react_tools, "_baiten_search_by_query",
                          new=AsyncMock(return_value=([], "CN 0"))), \
             patch.object(react_tools, "_uspto_search_by_number",
                          new=AsyncMock(return_value=([_USPTO_ITEM],
                                                      "USPTO 1"))) as um:
            merged, _notes = _run(
                react_tools._lookup_number_candidates(agent, [self._CAND]))
        self.assertEqual(len(merged), 1)
        um.assert_awaited()


if __name__ == "__main__":
    unittest.main()


# ── 需求#36: 按号解析的中靶校验 ──────────────────────────────────────────────
# 2026-09-19 生产: 用户三问「US12253745B2符合吗」, number_resolve 的命中集是
# 两条**非中靶**记录(题名 LEAK DETECTOR 排在首位), 模型据此断言"不符合" —
# 事实基础错位, 用户连问三遍未获澄清。 以下测试钉住修法:
# (a) 号码自身命中(applicationNumberText 或 patentNumber 对得上)在命中集中
#     必须排首位, 且带 _number_hit 标记;
# (b) 全非中靶时仍返回记录, 但观察文本必须显式声明"未取得该号码本身的记录";
# (c) 非中靶不得像命中那样提前终止其他数据源的复核。

_LEAK_DETECTOR_ITEM = {
    "applicationNumberText": "17000001",
    "applicationMetaData": {
        "inventionTitle": "LEAK DETECTOR",
        "applicationStatusDescriptionText": "Patented Case",
    },
}

_TARGET_ITEM = {
    "applicationNumberText": "18363489",
    "applicationMetaData": {
        "inventionTitle": "Silicon Photonic Device With Backup Light Paths",
        "applicationStatusDescriptionText": "Patented Case",
        "patentNumber": "12253745",
    },
}

_US12253745_CAND = {
    "country": "US", "display": "US12253745B2",
    "id_type": "ambiguous",
    "lookups": ["12253745", "122537452"],
}


class TestNumberHitVerification(unittest.TestCase):
    """需求#36 — 中靶记录与"相关但非该号"必须可区分。"""

    def test_target_record_is_flagged_and_ordered_first(self):
        """号码自身命中: 标注 _number_hit, 并与非中靶记录一起按中靶优先排序。"""
        agent = _agent([_US12253745_CAND])
        with patch.object(react_tools, "_uspto_search_by_number",
                          new=AsyncMock(return_value=(
                              [_LEAK_DETECTOR_ITEM, _TARGET_ITEM],
                              "USPTO 2 hits"))):
            merged, _notes = _run(react_tools._lookup_number_candidates(
                agent, [_US12253745_CAND]))
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["applicationNumberText"], "18363489")
        self.assertEqual(merged[0]["_number_hit"], "12253745")
        self.assertNotIn("_number_hit", merged[1])

    def test_target_matched_by_application_number(self):
        """中靶判据覆盖申请号一侧: 号码是申请号时同样命中。"""
        cand = {"country": "US", "display": "US18363489",
                "lookups": ["18363489"]}
        agent = _agent([cand])
        with patch.object(react_tools, "_uspto_search_by_number",
                          new=AsyncMock(return_value=(
                              [_LEAK_DETECTOR_ITEM, _TARGET_ITEM],
                              "USPTO 2 hits"))):
            merged, _notes = _run(react_tools._lookup_number_candidates(
                agent, [cand]))
        self.assertEqual(merged[0]["applicationNumberText"], "18363489")
        self.assertEqual(merged[0]["_number_hit"], "18363489")

    def test_target_is_matched_by_either_id(self):
        """判据对两个号槽都成立: 申请号在 patent_id、授权号在 patentNumber。"""
        self.assertEqual(
            react_tools.number_hit_kind(_TARGET_ITEM, ["12253745"]), "12253745")
        self.assertEqual(
            react_tools.number_hit_kind(_TARGET_ITEM, ["US18363489"]),
            "18363489")
        self.assertEqual(
            react_tools.number_hit_kind(_LEAK_DETECTOR_ITEM, ["12253745"]), "")

    def test_non_hit_does_not_stop_cross_check(self):
        """非中靶命中不得当作"主源已命中"提前收手 —— 对侧仍须复核。"""
        agent = _agent([_US12253745_CAND])
        with patch.object(react_tools, "_uspto_search_by_number",
                          new=AsyncMock(return_value=(
                              [_LEAK_DETECTOR_ITEM], "USPTO 1 hits"))), \
             patch.object(react_tools, "_baiten_search_by_query",
                          new=AsyncMock(return_value=(
                              [], "CN 0 hits"))) as bm:
            merged, _notes = _run(react_tools._lookup_number_candidates(
                agent, [_US12253745_CAND]))
        self.assertEqual(len(merged), 1)
        bm.assert_awaited()          # 非中靶 → 继续打对侧

    def test_hit_stops_cross_check(self):
        """中靶命中仍保持原有的"主源命中即收手"行为, 不额外耗预算。"""
        agent = _agent([_US12253745_CAND])
        with patch.object(react_tools, "_uspto_search_by_number",
                          new=AsyncMock(return_value=(
                              [_TARGET_ITEM], "USPTO 1 hits"))), \
             patch.object(react_tools, "_baiten_search_by_query",
                          new=AsyncMock()) as bm:
            merged, _notes = _run(react_tools._lookup_number_candidates(
                agent, [_US12253745_CAND]))
        self.assertEqual(len(merged), 1)
        bm.assert_not_awaited()
        self.assertEqual(agent._number_cross_used, 1)


class TestNumberHitDigestWording(unittest.TestCase):
    """需求#36 — 全非中靶时, 回答必须说清"这些不是你要的那个号"。

    2026-09-19 生产: 模型拿到两条非中靶记录(题名 LEAK DETECTOR)后直接断言
    "US12253745B2 不符合...记录为 LEAK DETECTOR" —— 记录本身没错, 错在把
    相关件当成了号码自身。观察文本必须把这条边界说出来。
    """

    def _run_resolve(self, items, notes_tail=""):
        agent = _agent([_US12253745_CAND])
        agent._number_cross_done = False
        with patch.object(react_tools, "_lookup_number_candidates",
                          new=AsyncMock(return_value=(
                              items, ["USPTO(nums=12253745) — USPTO 2 hits"
                                      + notes_tail]))), \
             patch.object(react_tools, "_rank_builtin_patent_pool",
                          new=AsyncMock(side_effect=lambda a, p, l: p)):
            return _run(react_tools._run_patent_number_resolve(
                agent, {"number": "US12253745B2"}, "zh"))

    def test_non_hit_records_declare_the_number_itself_was_not_found(self):
        obs = self._run_resolve([dict(_LEAK_DETECTOR_ITEM)])
        self.assertIn("未取得该号码本身的记录", obs["text"])

    def test_non_hit_declaration_keeps_the_records_retrieved(self):
        """声明边界不等于藏起记录 —— 用户仍须看到这次查到了什么。"""
        obs = self._run_resolve([dict(_LEAK_DETECTOR_ITEM)])
        self.assertIn("LEAK DETECTOR", obs["text"])

    def test_hit_record_carries_no_missing_declaration(self):
        item = dict(_TARGET_ITEM)
        item["_number_hit"] = "12253745"
        obs = self._run_resolve([item])
        self.assertNotIn("未取得该号码本身的记录", obs["text"])

    def test_hit_line_is_marked_for_the_model(self):
        """中靶行在观察文本里必须带显式标记 —— 模型要能一眼看出哪条是号码本身。"""
        item = dict(_TARGET_ITEM)
        item["_number_hit"] = "12253745"
        digest = react_tools._items_digest([item], lang="zh")
        self.assertIn("★该号码本身", digest)
        self.assertIn("Silicon Photonic Device With Backup Light Paths", digest)

    def test_non_hit_line_carries_no_hit_marker(self):
        digest = react_tools._items_digest(
            [dict(_LEAK_DETECTOR_ITEM)], lang="zh")
        self.assertNotIn("★该号码本身", digest)

    def test_cn_publication_number_is_matched(self):
        """CN 侧判据: 公开号 in patent_id 亦须中靶, 否则 CN 查询永不中靶。"""
        cand = {"country": "CN", "display": "CN117941643",
                "lookups": ["CN117941643A", "117941643"]}
        agent = _agent([cand])
        with patch.object(react_tools, "_baiten_search_by_query",
                          new=AsyncMock(return_value=(
                              [_BAITEN_ITEM], "CN 1 hits"))):
            merged, _notes = _run(react_tools._lookup_number_candidates(
                agent, [cand]))
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["_number_hit"], "117941643")


class TestNativeKeyHitTagging(unittest.TestCase):
    """需求#36 × 需求#29: 原生键直查取回的就是该号本身, 必须打中靶标记。

    否则 `_baiten_native_leg` 走完后 merged 无一条带 `_number_hit`,
    `_run_patent_number_resolve` 会**误报**"未取得该号码本身的记录" ——
    把修复变成反向错误。
    """

    _CAND = {"country": "CN", "display": "CN116570413A",
             "lookups": ["CN116570413A", "116570413"],
             "native_key": "CN202310123456.7",
             "native_key_kind": "app_num"}

    def test_native_key_record_is_tagged_as_hit(self):
        agent = _agent([self._CAND])
        with patch.object(react_tools, "_baiten_law_lookup",
                          new=AsyncMock(return_value={
                              "status": "专利权终止", "timeline": []})):
            merged, _notes = _run(react_tools._lookup_number_candidates(
                agent, [self._CAND]))
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["_number_hit"], "CN202310123456.7")


class TestInternalKeysNeverLeak(unittest.TestCase):
    """内部标记键(下划线前缀)不得进入用户可见的导出列。"""

    def test_number_hit_not_exported(self):
        from sources.result_export import normalize_result_rows
        item = dict(_TARGET_ITEM)
        item["_number_hit"] = "12253745"
        columns, rows = normalize_result_rows([item])
        self.assertFalse([c for c in columns if c.startswith("_")],
                         f"内部键泄漏进导出列: {columns}")
        self.assertFalse(any(k.startswith("_") for r in rows for k in r))

    def test_cn_hit_line_is_marked_for_the_model(self):
        """CN 侧中靶行同样要带标记 —— 否则中文检索的中靶信号是空的。"""
        item = dict(_BAITEN_ITEM)
        item["_number_hit"] = "117941643"
        digest = react_tools._items_digest([item], lang="zh")
        self.assertIn("★该号码本身", digest)
        self.assertIn("某方法", digest)

    def test_marker_is_language_aware(self):
        item = dict(_TARGET_ITEM)
        item["_number_hit"] = "12253745"
        self.assertIn("★该号码本身",
                      react_tools._items_digest([item], lang="zh"))
        self.assertNotIn("★该号码本身",
                         react_tools._items_digest([item], lang="en"))


class TestAutoNumberCrossRoundHitWording(unittest.TestCase):
    """需求#36 第 2 条入口: KB 号码工具零命中时走的跨源复核。

    这条路径同样会把"相关但非该号"的记录当成"复核命中"交给模型 ——
    与 patch tool 路径同源, 声明必须一并给到。
    """

    def test_non_hit_records_declare_the_number_itself_was_not_found(self):
        agent = _agent([_US12253745_CAND])
        with patch.object(react_tools, "_lookup_number_candidates",
                          new=AsyncMock(return_value=(
                              [dict(_LEAK_DETECTOR_ITEM)],
                              ["USPTO(nums=12253745) — USPTO 1 hits"]))), \
             patch.object(react_tools, "_rank_pending_pool",
                          new=AsyncMock(return_value=(
                              [{"patent_id": "17000001"}], ""))):
            _ranked, _rn, note = _run(
                react_tools._auto_number_cross_round(agent, "zh"))
        self.assertIn("未取得该号码本身的记录", note)

    def test_hit_records_carry_no_missing_declaration(self):
        item = dict(_TARGET_ITEM)
        item["_number_hit"] = "12253745"
        agent = _agent([_US12253745_CAND])
        with patch.object(react_tools, "_lookup_number_candidates",
                          new=AsyncMock(return_value=(
                              [item], ["USPTO 1 hits"]))), \
             patch.object(react_tools, "_rank_pending_pool",
                          new=AsyncMock(return_value=(
                              [{"patent_id": "18363489"}], ""))):
            _ranked, _rn, note = _run(
                react_tools._auto_number_cross_round(agent, "zh"))
        self.assertNotIn("未取得该号码本身的记录", note)

    def test_both_id_slots_are_consulted(self):
        """顶层 patent_number 不匹配时, 嵌套槽仍须被查到(勿用 or 短路)。"""
        item = {
            "applicationNumberText": "18363489",
            "patent_number": "99999999",       # 顶层槽不匹配
            "applicationMetaData": {"patentNumber": "12253745"},  # 嵌套槽匹配
        }
        self.assertEqual(
            react_tools.number_hit_kind(item, ["12253745"]), "12253745")
