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
        self.assertTrue(handler.statuses)  # "正在分析您的问题..."

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


if __name__ == "__main__":
    unittest.main()
