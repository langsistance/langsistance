"""Wiring tests: GeneralAgent.create_agent → ReActLoop → intent/None."""
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from sources.agents.general_agent import GeneralAgent
from sources.agents.react_loop import RoundResult


class _FakeProvider:
    def get_model_name(self):
        return "fake"

    def _get_langchain_llm(self, streaming=True):
        return None


class _FakeHandler:
    """Minimal callback stand-in with the methods the agent touches."""

    def __init__(self):
        self.queue = None
        self.statuses = []
        self.tokens = []

    async def on_status(self, message, **kwargs):
        self.statuses.append(message)

    async def on_llm_new_token(self, token, **kwargs):
        self.tokens.append(token)


def _make_agent():
    with patch.object(GeneralAgent, "load_prompt", return_value="sys prompt"):
        agent = GeneralAgent("test", "prompts/base/general_agent.txt",
                             _FakeProvider(), verbose=False)
    agent.enabled = True
    agent.llm = MagicMock()
    agent.llm.get_model_name.return_value = "fake"
    return agent


# NOTE (reconciliation): the brief's `_run(coro)` helper used
# `asyncio.get_event_loop()` which is removed/no-ed on Python 3.14.
# Tests here call `asyncio.run(coro)` directly instead.
def _run(coro):
    return asyncio.run(coro)


class TestCreateAgentWiring(unittest.TestCase):
    def test_answer_kind_returns_none_and_sets_flag(self):
        agent = _make_agent()
        handler = _FakeHandler()
        with patch("sources.agents.general_agent.build_tool_set",
                   new=AsyncMock(return_value=({}, []))), \
             patch("sources.agents.general_agent.ReActLoop") as MockLoop:
            MockLoop.return_value.run = AsyncMock(
                return_value=RoundResult(kind="answer", answer_text="hi", steps=1))
            result = _run(agent.create_agent(
                "u1", "hello", "q1", "", handler, push_filter=None))
        self.assertIsNone(result)
        self.assertTrue(getattr(agent, "_react_loop_ran", False))
        # per-request state reset (pool reuse safety)
        self.assertIsNone(getattr(agent, "_pending_raw_items", "unset"))
        self.assertIsNone(getattr(agent, "_search_pool", "unset"))
        self.assertFalse(getattr(agent, "_search_ranked", "unset"))
        self.assertTrue(handler.statuses)  # "正在分析您的问题..."

    def test_pooled_agent_resets_turn_flags_between_requests(self):
        # Agents are reused from the pool; every per-turn flag must be
        # cleared in create_agent or one request leaks into the next.
        agent = _make_agent()
        handler = _FakeHandler()
        agent._missing_dir_done = True
        agent._missing_dir_queries = ["stale query"]
        agent._auto_round_done = True
        agent._auto_ladder_used = 5
        agent._patent_auto_used = 2
        agent._tried_queries = ["stale tried query"]
        agent._cpc_hints = [{"code": "STALE", "title": "stale hint"}]
        agent._feedback_done = True
        agent._feedback_queries = ["stale feedback query"]
        agent._auto_feedback_done = True
        agent._recall_done = True
        agent._ladder_capped = True
        agent._search_ranked = True
        with patch("sources.agents.general_agent.build_tool_set",
                   new=AsyncMock(return_value=({}, []))), \
             patch("sources.agents.general_agent.ReActLoop") as MockLoop:
            MockLoop.return_value.run = AsyncMock(
                return_value=RoundResult(kind="answer", answer_text="hi", steps=1))
            _run(agent.create_agent(
                "u1", "hello", "q1", "", handler, push_filter=None))
        self.assertFalse(getattr(agent, "_missing_dir_done", True))
        self.assertIsNone(getattr(agent, "_missing_dir_queries", "unset"))
        self.assertFalse(getattr(agent, "_auto_round_done", True))
        self.assertEqual(getattr(agent, "_auto_ladder_used", "unset"), 0)
        self.assertEqual(getattr(agent, "_patent_auto_used", "unset"), {"us": 0, "cn": 0})
        self.assertEqual(getattr(agent, "_tried_queries", "unset"), [])
        self.assertIsNone(getattr(agent, "_cpc_hints", "unset"))
        self.assertFalse(getattr(agent, "_feedback_done", True))
        self.assertIsNone(getattr(agent, "_feedback_queries", "unset"))
        self.assertFalse(getattr(agent, "_auto_feedback_done", True))
        self.assertFalse(getattr(agent, "_recall_done", True))
        self.assertFalse(getattr(agent, "_ladder_capped", True))
        self.assertFalse(getattr(agent, "_search_ranked", True))

    def test_create_agent_matches_cpc_once_per_request(self):
        agent = _make_agent()
        handler = _FakeHandler()
        hints = [{"code": "H05B45/00", "title": "LED circuits", "score": 0.9}]
        with patch("sources.agents.general_agent.CPC_EXPANSION_ENABLED", True), \
             patch("sources.long_task.cpc_semantic.match_query_to_cpc",
                   return_value=hints) as mock_cpc, \
             patch("sources.agents.general_agent.build_tool_set",
                   new=AsyncMock(return_value=({}, []))), \
             patch("sources.agents.general_agent.ReActLoop") as MockLoop:
            MockLoop.return_value.run = AsyncMock(
                return_value=RoundResult(kind="answer", answer_text="hi", steps=1))
            _run(agent.create_agent(
                "u1", "hello", "q1", "", handler, push_filter=None))
        mock_cpc.assert_called_once_with("hello", extra_terms=[])
        self.assertEqual(getattr(agent, "_cpc_hints", None), hints)

    def test_create_agent_structured_mode_skips_cpc_and_interpretation(self):
        # Identifier/document/prosecution queries classify as structured:
        # the ladder still gets the rewrite, but the CPC match and the
        # architecture interpretation are skipped entirely.
        agent = _make_agent()
        handler = _FakeHandler()
        with patch("sources.long_task.query_mode.classify_query_mode",
                   new=AsyncMock(return_value="structured")), \
             patch("sources.long_task.search_query_builder.build_search_queries",
                   new=AsyncMock(return_value={"concepts": [], "queries": ["q1"]})) as mock_rewrite, \
             patch("sources.long_task.cpc_semantic.match_query_to_cpc") as mock_cpc, \
             patch("sources.long_task.technical_interpretation.interpret_query",
                   new=AsyncMock(return_value=None)) as mock_interp, \
             patch("sources.agents.general_agent.build_tool_set",
                   new=AsyncMock(return_value=({}, []))), \
             patch("sources.agents.general_agent.ReActLoop") as MockLoop:
            MockLoop.return_value.run = AsyncMock(
                return_value=RoundResult(kind="answer", answer_text="hi", steps=1))
            _run(agent.create_agent(
                "u1", "I want all patent documents for application 18893954",
                "q1", "", handler, push_filter=None))
        self.assertEqual(agent._query_mode, "structured")
        self.assertEqual(agent._search_rewrite.get("queries"), ["q1"])
        self.assertIsNone(agent._cpc_hints)
        self.assertIsNone(agent._search_interpretation)
        mock_rewrite.assert_awaited_once()
        mock_cpc.assert_not_called()
        mock_interp.assert_not_called()

    def test_create_agent_semantic_mode_runs_all_three(self):
        # Semantic technology searches keep rewrite + CPC + interpretation
        # (now run concurrently).
        agent = _make_agent()
        handler = _FakeHandler()
        hints = [{"code": "H05B45/00", "title": "LED circuits"}]
        interp = {"scheme": "温控闭环", "queries": ["\"temperature\" AND \"feedback\""]}
        with patch("sources.agents.general_agent.CPC_EXPANSION_ENABLED", True), \
             patch("sources.long_task.query_mode.classify_query_mode",
                   new=AsyncMock(return_value="semantic")), \
             patch("sources.long_task.search_query_builder.build_search_queries",
                   new=AsyncMock(return_value={"concepts": [], "queries": ["q1"]})) as mock_rewrite, \
             patch("sources.long_task.cpc_semantic.match_query_to_cpc",
                   return_value=hints) as mock_cpc, \
             patch("sources.long_task.technical_interpretation.interpret_query",
                   new=AsyncMock(return_value=interp)) as mock_interp, \
             patch("sources.agents.general_agent.build_tool_set",
                   new=AsyncMock(return_value=({}, []))), \
             patch("sources.agents.general_agent.ReActLoop") as MockLoop:
            MockLoop.return_value.run = AsyncMock(
                return_value=RoundResult(kind="answer", answer_text="hi", steps=1))
            _run(agent.create_agent(
                "u1", "保持温度稳定的装置", "q1", "", handler, push_filter=None))
        self.assertEqual(agent._query_mode, "semantic")
        self.assertEqual(agent._cpc_hints, hints)
        self.assertEqual(agent._search_interpretation, interp)
        mock_rewrite.assert_awaited_once()
        mock_cpc.assert_called_once()
        mock_interp.assert_awaited_once()
        # interpretation queries merged into the ladder
        self.assertIn("\"temperature\" AND \"feedback\"",
                      agent._search_rewrite.get("queries"))

    def test_long_task_kind_returns_intent_dict(self):
        agent = _make_agent()
        handler = _FakeHandler()
        class _K:
            id = 7
            type = 3
        class _T:
            title = "lt"
        with patch("sources.agents.general_agent.build_tool_set",
                   new=AsyncMock(return_value=({}, []))), \
             patch("sources.agents.general_agent.ReActLoop") as MockLoop:
            MockLoop.return_value.run = AsyncMock(
                return_value=RoundResult(kind="long_task", steps=1,
                                         long_task_knowledge=_K(),
                                         long_task_tool_info=_T()))
            result = _run(agent.create_agent(
                "u1", "帮我分析这批专利", "q1", "", handler, push_filter=None))
        self.assertEqual(result["intent"], "long_task")
        self.assertEqual(result["knowledge"].id, 7)
        self.assertTrue(getattr(agent, "_react_loop_ran", False))

    def test_invoke_agent_streams_pending_raw_items_after_loop(self):
        agent = _make_agent()
        agent._react_loop_ran = True
        agent._pending_raw_items = [{"a": 1}]
        agent._active_collector = None
        handler = _FakeHandler()
        with patch.object(agent, "_stream_raw_items",
                          new=AsyncMock()) as mock_stream, \
             patch.object(agent, "_store_current_turn"):
            _run(agent.invoke_agent(None, handler))
        mock_stream.assert_awaited_once_with([{"a": 1}], handler)

    def test_invoke_agent_stores_turn_when_no_pending(self):
        agent = _make_agent()
        agent._react_loop_ran = True
        agent._pending_raw_items = None
        agent._active_collector = MagicMock(collected_text="answer text")
        with patch.object(agent, "_stream_raw_items",
                          new=AsyncMock()) as mock_stream, \
             patch.object(agent, "_store_current_turn") as mock_store:
            _run(agent.invoke_agent(None, MagicMock()))
        mock_stream.assert_not_called()
        mock_store.assert_called_once_with("answer text")


from sources.agents.general_agent import (
    RELEVANT_TOP_N,
    _summary_system_prompt,
)


class TestDynamicToolDescriptionGuide(unittest.TestCase):
    """The loop's dynamic tool description must carry the knowledge item's
    answer as a usage guide — the pre-loop flow injected it as "Context
    from knowledge base".  Without it the LLM passed the USPTO application
    number as a query param, sending the literal {applicationNumberText}
    path template to the gateway (403)."""

    def _make_agent(self):
        with patch.object(GeneralAgent, "load_prompt", return_value="sys prompt"):
            return GeneralAgent("test", "prompts/base/general_agent.txt",
                                _FakeProvider(), verbose=False)

    def test_description_includes_knowledge_answer_usage_guide(self):
        from unittest.mock import MagicMock
        agent = self._make_agent()

        knowledge = MagicMock()
        knowledge.id = 289
        knowledge.answer = (
            "调用方式：URL 已包含 /applications 时，path 只需 /{patent_id}/documents，"
            "将申请号替换进 path，不要放入 query。"
        )
        tool_info = MagicMock()
        tool_info.title = "USPTO 专利文档"
        tool_info.description = "获取美国专利文档列表"
        tool_info.push = 2
        tool_info.url = (
            "https://api.uspto.gov/api/v1/patent/applications/"
            "{applicationNumberText}/documents"
        )
        tool_info.params = '{"method":"GET","query":{}}'

        tool = agent.get_dynamic_tool_for(knowledge, tool_info)

        self.assertIn("获取美国专利文档列表", tool.description)
        self.assertIn("Usage guide:", tool.description)
        self.assertIn("path 只需 /{patent_id}/documents", tool.description)

    def test_description_without_answer_uses_bare_tool_description(self):
        from unittest.mock import MagicMock
        agent = self._make_agent()

        knowledge = MagicMock()
        knowledge.id = 289
        knowledge.answer = ""
        tool_info = MagicMock()
        tool_info.title = "USPTO 专利文档"
        tool_info.description = "获取美国专利文档列表"
        tool_info.push = 2
        tool_info.url = "https://api.uspto.gov/api/v1/patent/applications/search"
        tool_info.params = '{"method":"GET","query":{}}'

        tool = agent.get_dynamic_tool_for(knowledge, tool_info)

        self.assertEqual(tool.description, "获取美国专利文档列表")
        self.assertNotIn("Usage guide:", tool.description)


class TestLoopGuidanceTopN(unittest.TestCase):
    def test_guidance_requires_top_n_relevant_listing(self):
        agent = _make_agent()
        text = agent._loop_system_guidance()
        self.assertIn("relevance-ranked", text)
        self.assertIn(str(RELEVANT_TOP_N), text)
        self.assertIn("top items", text)
        self.assertNotIn("patents", text)
        self.assertNotIn("相关度", text)

    def test_top_n_default_is_10(self):
        self.assertEqual(RELEVANT_TOP_N, 10)


class TestSummarySystemPrompt(unittest.TestCase):
    def test_ranked_variant_mentions_relevance_order_zh(self):
        text = _summary_system_prompt(True, "zh")
        self.assertIn("相关度排序", text)
        self.assertIn("摘要", text)

    def test_ranked_variant_mentions_relevance_order_en(self):
        text = _summary_system_prompt(True, "en")
        self.assertIn("relevance-ranked", text)

    def test_default_variant_has_no_ranking_note(self):
        self.assertNotIn("相关度排序", _summary_system_prompt(False, "zh"))
        self.assertNotIn("relevance-ranked", _summary_system_prompt(False, "en"))


if __name__ == "__main__":
    unittest.main()


class TestCreateAgentLongTaskRouting(unittest.TestCase):
    """Deterministic long-task routing in create_agent: when the request
    matches a type-3 knowledge item, the intent is returned WITHOUT running
    the ReAct loop (the LLM picking the long-task tool from the bound list
    is unreliable — observed going down the USPTO keyword ladder instead)."""

    def _long_task_entry(self):
        from sources.agents.react_tools import ToolEntry
        from unittest.mock import MagicMock
        knowledge = MagicMock()
        knowledge.id = 289
        knowledge.type = 3
        knowledge.question = (
            "zh:输入美国专利申请号，分析其审查历史（USPTO）|"
            "en:Enter a US patent application number to analyze its "
            "prosecution history")
        tool = MagicMock()
        return ToolEntry(name="prosecution_history", kind="long_task",
                         knowledge=knowledge, tool_info=MagicMock(),
                         tool=tool)

    def test_matched_request_returns_intent_without_loop(self):
        agent = _make_agent()
        handler = _FakeHandler()
        entry = self._long_task_entry()
        registry = {"prosecution_history": entry}
        with patch("sources.agents.general_agent.build_tool_set",
                   new=AsyncMock(return_value=(registry, []))), \
             patch("sources.agents.general_agent._match_long_task_intent",
                   new=AsyncMock(return_value=entry)) as mock_match, \
             patch("sources.agents.general_agent.ReActLoop") as MockLoop:
            MockLoop.return_value.run = AsyncMock(return_value=RoundResult(
                kind="answer", answer_text="unused", steps=0))
            result = _run(agent.create_agent(
                "u1", "分析专利 11701773 的审查历史", "q1", "",
                handler, push_filter=None))
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("intent"), "long_task")
        self.assertEqual(result.get("knowledge"), entry.knowledge)
        mock_match.assert_awaited_once()
        # The loop must NOT run when routing matched
        MockLoop.return_value.run.assert_not_awaited()
        self.assertFalse(getattr(agent, "_react_loop_ran", False))

    def test_no_match_runs_loop(self):
        agent = _make_agent()
        handler = _FakeHandler()
        entry = self._long_task_entry()
        registry = {"prosecution_history": entry}
        with patch("sources.agents.general_agent.build_tool_set",
                   new=AsyncMock(return_value=(registry, []))), \
             patch("sources.agents.general_agent._match_long_task_intent",
                   new=AsyncMock(return_value=None)), \
             patch("sources.agents.general_agent.ReActLoop") as MockLoop:
            MockLoop.return_value.run = AsyncMock(return_value=RoundResult(
                kind="answer", answer_text="hi", steps=1))
            result = _run(agent.create_agent(
                "u1", "帮我找工业机器人专利", "q1", "",
                handler, push_filter=None))
        self.assertIsNone(result)
        MockLoop.return_value.run.assert_awaited_once()
        self.assertTrue(getattr(agent, "_react_loop_ran", False))
