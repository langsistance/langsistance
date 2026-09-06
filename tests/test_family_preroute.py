# -*- coding: utf-8 -*-
"""Family-analysis intent pre-route narrow gate + resume executor dispatch by
type (plan Task 6 / spec §5.5b + A9).

§5.5b: a user whose knowledge base carries NO tailored type-3 family item
still loses ~16s spinning in the ReAct loop on a "analyze {pct|wo|unsupported}
issue + worldwide family examination" ask.  When the parser produces such a
candidate AND the query genuinely ANALYSES proceedings (not merely searches
for family members) AND no tailored type-3 route would fire anyway, route it
straight to a families intent.

Two behaviours protected by the RED-side agreement:

1. The predicate must NOT mis-route a pure *search* ask ("检索…同族…已授权"
   is the search ladder's job), even though it carries family + exam words.
2. When a tailored type-3 KB family item WOULD be matched by the existing
   deterministic router, that router keeps priority — the narrow gate must
   not pre-empt it.

A9: ``_dispatch_from_mysql`` (resume path) reads the stored ``task_type`` and
dispatches to the matching executor instead of always
``execute_patent_analysis`` — so a paused ``family_analysis`` resumes on the
right worker.

Counter-example sentences here use generic patent ids (never user query
vocabulary baked into production code).
"""
import logging
import sys
import types
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

# api_routes.long_task imports sources.user.passport + celery_worker lazily.
_fake_passport = types.ModuleType("sources.user.passport")
_fake_passport.verify_firebase_token = lambda hdr=None: {"uid": "1"}
sys.modules.setdefault("sources.user.passport", _fake_passport)

_fake_celery = types.ModuleType("celery_worker")
sys.modules.setdefault("celery_worker", _fake_celery)


def _reset_celery():
    from unittest.mock import MagicMock as _M
    for name in ("execute_china_examination_analysis",
                 "execute_family_analysis",
                 "execute_patent_analysis",
                 "execute_prosecution_analysis"):
        setattr(_fake_celery, name, _M())
        getattr(_fake_celery, name).delay = _M()


_fake_celery.reset = _reset_celery


def _cands(*types):
    """Synthetic parser output subset — only the keys the narrow gate reads."""
    return [{"id_type": t, "display": t, "country": "WO" if t == "wo"
             else ("US" if t == "pct" else "")} for t in types]


from sources.agents.general_agent import _should_route_family_analysis  # noqa: E402


class TestNarrowGatePredicate(unittest.TestCase):
    """The (§5.5b) discrimination: pct/wo/unsupported candidate + ANALYSIS
    intent (search / retrieval phrasing rejected).  Pure function — no I/O."""

    def test_pct_candidate_family_examination_analysis_routes(self):
        query = "帮我分析 PCTUS2021059064 及其全球同族申请的审查差异"
        self.assertTrue(_should_route_family_analysis(_cands("pct"), query))

    def test_wo_candidate_family_analysis_routes(self):
        query = "请分析该号与各国同族的审查过程有什么差异"
        self.assertTrue(_should_route_family_analysis(_cands("wo"), query))

    def test_unsupported_candidate_family_analysis_routes(self):
        query = "帮我审查一下这个号码的世界各国同族授权情况差异"
        self.assertTrue(_should_route_family_analysis(_cands("unsupported"), query))

    def test_pct_candidate_search_phrasing_not_routed(self):
        # "检索/查找/搜索 …同族…授权" asks search, NOT proceedings analysis.
        query = "帮我检索与 PCTUS2021059064 同族且美国已授权的专利"
        self.assertFalse(_should_route_family_analysis(_cands("pct"), query))

    def test_family_word_without_examination_analysis_not_routed(self):
        # Family scope but no proceedings-analysis sense (pure member lookup).
        query = "把 PCTUS2021059064 的同族成员各国公开文本索引找一下"
        self.assertFalse(_should_route_family_analysis(_cands("pct"), query))

    def test_bare_us_candidate_not_narrow_routed(self):
        # A resolved US grant is NOT a pct/wo/unsupported candidate → the gate
        # must stay closed even with family-analysis wording (no KB change).
        query = "帮我分析 19519846 及其全球同族申请的审查差异"
        self.assertFalse(_should_route_family_analysis(_cands("grant"), query))

    def test_empty_candidates_never_routes(self):
        self.assertFalse(_should_route_family_analysis([], "分析全球同族审查差异"))

    def test_en_analysis_scope_still_routes(self):
        query = ("analyze worldwide family examination differences for "
                 "WO2021059064 across jurisdictions")
        self.assertTrue(_should_route_family_analysis(_cands("wo"), query))

    # F3 · T6 search-verb robustness + retrieval delegation (general word
    # classes, never user-query vocabulary).
    def test_en_bare_find_search_phrasing_not_routed(self):
        # Regression: the old table stored ``"find "`` (trailing space) so a
        # bare leading "find …" was missed and wrongly routed.  A \b word
        # boundary must catch "find US12506212 family members".
        query = "find US12506212 family members"
        self.assertFalse(_should_route_family_analysis(_cands("wo"), query))

    def test_en_look_up_search_phrasing_not_routed(self):
        query = "look up the family members of WO2021059064"
        self.assertFalse(_should_route_family_analysis(_cands("wo"), query))

    def test_zh_trailing_punctuation_search_not_routed(self):
        # "检索一下…" embeds the search verb 检索 — substring match must still
        # reject even with trailing punctuation / particles.
        query = "检索一下 WO2021059064 的同族成员"
        self.assertFalse(_should_route_family_analysis(_cands("wo"), query))

    def test_retrieval_document_wording_not_routed(self):
        # delegated to react_tools._is_retrieval_request (verb ∧ document
        # object) — "获取…" + "…文件" must never take the analysis route.
        query = "获取该号的 WO 公开号完整文件，列出同族各国公开文档"
        self.assertFalse(_should_route_family_analysis(_cands("pct"), query))

    # F5 · T6 candidate mixing + EN negatives.
    def test_mixed_us_grant_and_wo_candidates_still_routes(self):
        # any()-semantics anchor (review F5 · T6): a wo present among mixed
        # US-grant + WO candidates keeps the gate open for a genuine ANALYSIS
        # ask — one narrow id suffices, per the window's definition.
        query = ("分析 US12506212 及其同族、对照 WO2021/059064 的审查差异")
        self.assertTrue(
            _should_route_family_analysis(_cands("grant", "wo"), query))

    def test_bare_us_grant_en_family_analysis_not_routed(self):
        # A resolved US grant (id_type=grant) is NOT a pct/wo/unsupported
        # candidate → gate stays closed even under English family-analysis
        # wording.
        query = ("Analyze the prosecution differences of US12506212 "
                 "and its family")
        self.assertFalse(_should_route_family_analysis(_cands("grant"), query))

    def test_pct_candidate_en_search_phrasing_not_routed(self):
        # pct candidate + English search wording → member search, not ANALYSIS.
        query = "search for the granted US family members of PCTUS2021059064"
        self.assertFalse(_should_route_family_analysis(_cands("pct"), query))


class TestNarrowGateWiring(unittest.TestCase):
    """create_agent routes a pct/wo ask to the families intent WITHOUT running
    the ReAct loop when there is no tailored type-3 route; a search-phrased ask
    still falls through to the loop; a matched type-3 item keeps priority."""

    @classmethod
    def setUpClass(cls):
        self = cls
        from sources.agents.general_agent import GeneralAgent
        from sources.agents.react_loop import RoundResult
        cls._GeneralAgent = GeneralAgent
        cls._RoundResult = RoundResult

    def _mk_agent(self):
        with patch.object(self._GeneralAgent, "load_prompt",
                          return_value="sys prompt"):
            class _Prov:
                def get_model_name(self):
                    return "fake"
                def _get_langchain_llm(self, streaming=True):
                    return None
            agent = self._GeneralAgent("test", "prompts/base/general_agent.txt",
                                       _Prov(), verbose=False)
        agent.enabled = True
        agent.llm = MagicMock()
        agent.llm.get_model_name.return_value = "fake"
        return agent

    def _builtin_registry(self):
        from sources.agents.react_tools import (
            ToolEntry, BUILTIN_DEEP_ANALYSIS_TOOL_NAME)
        tool = MagicMock()
        return {BUILTIN_DEEP_ANALYSIS_TOOL_NAME: ToolEntry(
            name=BUILTIN_DEEP_ANALYSIS_TOOL_NAME, kind="long_task",
            knowledge=None, tool_info=None, tool=tool)}

    def _family_entry(self):
        from sources.agents.react_tools import ToolEntry
        knowledge = MagicMock()
        knowledge.id = 512
        knowledge.type = 3
        knowledge.question = (
            "zh:输入美国专利号，分析其全球同族在该国审查差异|"
            "en:Enter a US patent number to analyze worldwide family examination")
        return ToolEntry(name="family_analysis", kind="long_task",
                         knowledge=knowledge, tool_info=MagicMock(),
                         tool=MagicMock())

    def test_family_analysis_returns_intent_without_loop(self):
        agent = self._mk_agent()
        handler = _FakeHandler()
        registry = self._builtin_registry()
        with patch("sources.agents.general_agent.build_tool_set",
                   new=AsyncMock(return_value=(registry, []))), \
             patch("sources.agents.general_agent.ReActLoop") as MockLoop, \
             patch("sources.patent_number_parser.NUMBER_PARSE_ENABLED", True), \
             patch("sources.agents.general_agent._should_route_family_analysis",
                   side_effect=lambda cands, q: True) as nav:
            result = _Async.run(agent.create_agent(
                "u1",
                "帮我分析 WO2021059064 及其全球同族申请的审查差异",
                "q1", "", handler, push_filter=None))
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("intent"), "long_task")
        nav.assert_called()
        MockLoop.assert_not_called()
        self.assertFalse(getattr(agent, "_react_loop_ran", False))

    def test_search_phrased_family_ask_keeps_loop(self):
        # Even with a pct/wo candidate, a *search* ask must NOT take the
        # families narrow route — only the ReAct loop may run.
        agent = self._mk_agent()
        handler = _FakeHandler()
        registry = self._builtin_registry()
        with patch("sources.agents.general_agent.build_tool_set",
                   new=AsyncMock(return_value=(registry, []))), \
             patch("sources.agents.general_agent.ReActLoop") as MockLoop, \
             patch("sources.agents.general_agent._should_route_family_analysis",
                   side_effect=lambda cands, q: False):
            MockLoop.return_value.run = AsyncMock(
                return_value=self._RoundResult(kind="answer",
                                               answer_text="ok", steps=1))
            result = _Async.run(agent.create_agent(
                "u1", "帮我检索与 WO2021059064 同族且已授权的专利",
                "q1", "", handler, push_filter=None))
        self.assertIsNone(result)
        MockLoop.return_value.run.assert_awaited_once()
        self.assertTrue(getattr(agent, "_react_loop_ran", False))

    def test_type3_kb_match_keeps_priority_over_narrow(self):
        # A tailored type-3 family item routed by the deterministic matcher
        # pre-empts the narrow gap-fill gate entirely.
        agent = self._mk_agent()
        handler = _FakeHandler()
        entry = self._family_entry()
        registry = {"family_analysis": entry}
        with patch("sources.agents.general_agent.build_tool_set",
                   new=AsyncMock(return_value=(registry, []))), \
             patch("sources.agents.general_agent._match_long_task_intent",
                   new=AsyncMock(return_value=entry)) as match, \
             patch("sources.agents.general_agent.ReActLoop") as MockLoop, \
             patch("sources.agents.general_agent._should_route_family_analysis",
                   side_effect=lambda cands, q: True) as nav:
            MockLoop.return_value.run = AsyncMock(
                return_value=self._RoundResult(kind="answer",
                                               answer_text="x", steps=1))
            result = _Async.run(agent.create_agent(
                "u1", "帮我分析 WO2021059064 及其全球同族申请的审查差异",
                "q1", "", handler, push_filter=None))
        self.assertEqual(result.get("knowledge"), entry.knowledge)
        match.assert_awaited_once()
        # the narrow gate must never have been consulted (KB route won)
        nav.assert_not_called()
        MockLoop.assert_not_called()


class _Async:
    import asyncio
    run = staticmethod(asyncio.run)


class _FakeHandler:
    def __init__(self):
        self.queue = None
        self.statuses = []
        self.tokens = []

    async def on_status(self, message, **kwargs):
        self.statuses.append(message)

    async def on_llm_new_token(self, token, **kwargs):
        self.tokens.append(token)


from api_routes.long_task import _dispatch_from_mysql  # noqa: E402


class TestResumeDispatchByTaskType(unittest.TestCase):
    """A9: resume dispatches by the stored MySQL task_type, not hard-coded"""

    def _mk_conn(self, input_params, task_type="family_analysis",
                 session_id="sess_x", scene_id=None):
        conn = MagicMock()
        cur = MagicMock()
        cur.fetchone.return_value = {
            "input_params": input_params, "session_id": session_id,
            "scene_id": scene_id, "task_type": task_type,
        }
        conn.cursor.return_value.__enter__.return_value = cur
        return conn

    def _dispatch(self, conn):
        logger = logging.getLogger("test")
        with patch("sources.knowledge.knowledge.get_db_connection",
                   return_value=conn):
            _dispatch_from_mysql("7", "lt_abc", logger)

    def setUp(self):
        _reset_celery()

    def tearDown(self):
        _reset_celery()

    # F1: single-patent rows (family/prosecution/china/ep/jp) persist a
    # singular ``patent_id`` in MySQL input_params — resume must forward it.
    def test_family_single_patent_task_keeps_patent_id(self):
        # Real stored shape: singular patent_id, NO patent_ids key.
        import json
        conn = self._mk_conn(json.dumps({"query": "分析同族",
                                         "patent_id": "US12506212"}))
        self._dispatch(conn)
        delay = _fake_celery.execute_family_analysis.delay
        self.assertTrue(delay.called)
        self.assertFalse(_fake_celery.execute_patent_analysis.delay.called)
        _, kwargs = delay.call_args
        dispat_params = kwargs["params"]
        # resume must survive the singular field the executor reads.
        self.assertIsNotNone(dispat_params.get("patent_id"))
        self.assertNotEqual(str(dispat_params.get("patent_id", "")).strip(), "")
        self.assertEqual(dispat_params["patent_id"], "US12506212")

    def test_family_row_without_patent_ids_key_still_resumes(self):
        # Regression guard (F1): stored family rows must not rely on a
        # ``patent_ids`` key being present.
        import json
        conn = self._mk_conn(json.dumps({"query": "帮我分析审查差异",
                                         "patent_id": "17429113"}))
        self._dispatch(conn)
        delay = _fake_celery.execute_family_analysis.delay
        self.assertTrue(delay.called)
        _, kwargs = delay.call_args
        self.assertNotEqual(
            str(kwargs["params"].get("patent_id", "")).strip(), "")

    def test_prosecution_single_patent_task_forwards_patent_id(self):
        import json
        conn = self._mk_conn(json.dumps({"query": "审查历史",
                                         "patent_id": "17429113"}),
                             task_type="prosecution_analysis")
        self._dispatch(conn)
        delay = _fake_celery.execute_prosecution_analysis.delay
        self.assertTrue(delay.called)
        _, kwargs = delay.call_args
        self.assertEqual(kwargs["params"].get("patent_id"), "17429113")

    def test_batch_task_dispatches_batch_executor(self):
        # Batch rows carry the plural multi-value patent_ids (unchanged).
        import json
        conn = self._mk_conn(json.dumps({"query": "q",
                                         "patent_ids": ["17429113",
                                                        "18012525"]}),
                             task_type="patent_analysis")
        self._dispatch(conn)
        delay = _fake_celery.execute_patent_analysis.delay
        self.assertTrue(delay.called)
        self.assertFalse(_fake_celery.execute_family_analysis.delay.called)
        _, kwargs = delay.call_args
        self.assertIsNone(kwargs["params"].get("patent_id"))
        self.assertEqual(kwargs["params"].get("patent_ids"),
                         ["17429113", "18012525"])

    def test_missing_task_type_defaults_to_batch(self):
        import json
        conn = self._mk_conn(json.dumps({"query": "q",
                                         "patent_ids": ["17429113"]}),
                             task_type=None)
        self._dispatch(conn)
        self.assertTrue(_fake_celery.execute_patent_analysis.delay.called)


if __name__ == "__main__":
    unittest.main()
