"""Tests for ReAct tool supply and action dispatch (react_tools)."""
import asyncio
import re
import unittest
from unittest.mock import patch

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from sources.agents.react_tools import (
    MAX_PATENT_LIST_ITEMS,
    SEARCH_KNOWLEDGE_TOOL_NAME,
    _cap_patent_list,
    _summarize_observation,
    build_tool_set,
    make_action_executor,
)


class _Knowledge:
    def __init__(self, kid, question="q", ktype=1):
        self.id = kid
        self.question = question
        self.description = ""
        self.answer = ""
        self.public = False
        self.model_name = ""
        self.tool_id = 1
        self.params = ""
        self.type = ktype
        self.scene_id = None
        self.update_time = ""


class _ToolInfo:
    def __init__(self, title, url="https://api.example.com/search", push=2):
        self.title = title
        self.description = f"{title} description"
        self.url = url
        self.push = push
        self.params = '{"method":"GET","query":{}}'


class _FakeAgent:
    def __init__(self):
        self._lang = "zh"
        self._last_user_id = "u1"
        self._pending_raw_items = None
        self.knowledgeTool = None
        self.tools_made = []

    async def get_dynamic_tool_for(self, knowledge_item, tool_info):
        class _Args(BaseModel):
            params: str = Field(description="params")

        async def _noop(**kwargs):
            return "tool result"

        # Mirror _clean_tool_name / the real agent: keep the underscore that
        # appears in the test assertions ("uspto_search"), instead of the
        # earlier drop-the-space rule that produced "usptosearch".
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_info.title).strip("_") or "tool"
        tool = StructuredTool.from_function(_noop, name=name,
                                            description=tool_info.description,
                                            args_schema=_Args)
        self.tools_made.append((knowledge_item.id, name))
        return tool


class TestCapPatentList(unittest.TestCase):
    def test_search_list_capped_at_100(self):
        items = list(range(MAX_PATENT_LIST_ITEMS + 40))
        capped, note = _cap_patent_list(_ToolInfo("s"), items, "zh")
        self.assertEqual(len(capped), MAX_PATENT_LIST_ITEMS)
        self.assertIn("已截断", note)

    def test_search_list_under_100_untouched(self):
        items = list(range(7))
        capped, note = _cap_patent_list(_ToolInfo("s"), items, "zh")
        self.assertEqual(capped, items)
        self.assertNotIn("截断", note)

    def test_document_list_never_capped(self):
        items = list(range(MAX_PATENT_LIST_ITEMS + 40))
        capped, note = _cap_patent_list(
            _ToolInfo("d", url="https://api.example.com/documents"), items, "zh")
        self.assertEqual(len(capped), MAX_PATENT_LIST_ITEMS + 40)
        self.assertIn("不截断", note)


class TestSummarizeObservation(unittest.TestCase):
    def test_long_text_truncated(self):
        text = _summarize_observation("x" * 1000, "zh")
        self.assertLessEqual(len(text), 303)
        self.assertTrue(text.endswith("..."))

    def test_dict_serialized(self):
        text = _summarize_observation({"a": 1, "b": [1, 2]}, "zh")
        self.assertIn('"a": 1', text)

    def test_none_becomes_empty(self):
        self.assertEqual(_summarize_observation(None, "zh"), "")


class TestBuildToolSet(unittest.TestCase):
    @patch("sources.agents.react_tools.get_knowledge_tool_candidates")
    def test_workflow_type_2_skipped_and_top_n_respected(self, mock_candidates):
        mock_candidates.return_value = [
            (_Knowledge(1, ktype=3), None),                        # long task
            (_Knowledge(2, ktype=2), _ToolInfo("wf")),             # workflow — retired
            (_Knowledge(3, ktype=1), _ToolInfo("uspto search")),   # normal
            (_Knowledge(4, ktype=1), _ToolInfo("cnipa search")),   # normal
        ]
        agent = _FakeAgent()
        registry, tools = asyncio.run(
            build_tool_set(agent, "u1", "专利检索", push_filter=None))

        self.assertIn(SEARCH_KNOWLEDGE_TOOL_NAME, registry)
        kinds = {e.name: e.kind for e in registry.values()}
        self.assertIn("uspto_search", kinds)
        self.assertEqual(kinds["uspto_search"], "knowledge")
        self.assertNotIn("wf", kinds)          # type-2 retired
        # type-3 became a long_task tool
        lt_names = [n for n, k in kinds.items() if k == "long_task"]
        self.assertEqual(len(lt_names), 1)
        self.assertEqual(len(tools), len(registry))


class TestExecuteAction(unittest.TestCase):
    @patch("sources.agents.react_tools.get_knowledge_tool_candidates")
    def test_search_knowledge_returns_matches_and_mount_tools(self, mock_candidates):
        # Build phase: no type-1, so build_tool_set registers no knowledge tool.
        build_candidates = [
            (_Knowledge(1, ktype=3), None),
            (_Knowledge(2, ktype=2), _ToolInfo("wf")),
        ]
        # Search phase: finds a type-1 tool that is NOT yet in the registry,
        # so the executor exercises the mount path the assertions target.
        search_candidates = [
            (_Knowledge(1, ktype=3), None),
            (_Knowledge(2, ktype=2), _ToolInfo("wf")),
            (_Knowledge(3, ktype=1), _ToolInfo("uspto search")),
        ]
        mock_candidates.side_effect = [build_candidates, search_candidates]
        agent = _FakeAgent()
        registry, tools = asyncio.run(
            build_tool_set(agent, "u1", "专利检索", push_filter=None))
        executor = asyncio.run(make_action_executor(agent, registry, None))

        result = asyncio.run(
            executor(SEARCH_KNOWLEDGE_TOOL_NAME, {"query": "美国专利"}, 1))
        self.assertEqual(result["kind"], "observation")
        self.assertIn("2 个匹配", result["text"])
        mounted = [t["name"] for t in result["mount_tools"]]
        self.assertIn("uspto_search", mounted)
        # type-2 not in matches
        self.assertNotIn("wf", result["text"])

    def test_knowledge_action_sets_pending_and_caps(self):
        agent = _FakeAgent()
        entry_k = _Knowledge(3, ktype=1)
        agent.get_dynamic_tool_for = _make_tool_with_pending(agent)
        registry, tools = asyncio.run(
            _registry_with_one_knowledge(agent, entry_k))
        executor = asyncio.run(make_action_executor(agent, registry, None))
        agent._pending_raw_items = list(range(MAX_PATENT_LIST_ITEMS + 10))
        result = asyncio.run(executor("uspto_search", {"params": "{}"}, 1))
        self.assertEqual(len(agent._pending_raw_items), MAX_PATENT_LIST_ITEMS)
        self.assertIn("已截断", result["text"])
        self.assertEqual(agent.knowledgeTool[0], entry_k)

    def test_long_task_action_returns_intent(self):
        agent = _FakeAgent()
        k = _Knowledge(9, ktype=3)
        registry = {
            "批量专利分析": type("E", (), {
                "name": "批量专利分析", "kind": "long_task",
                "knowledge": k, "tool_info": None, "tool": None,
            })(),
        }
        executor = asyncio.run(make_action_executor(agent, registry, None))
        result = asyncio.run(executor("批量专利分析", {"query": "分析"}, 1))
        self.assertEqual(result["kind"], "long_task")
        self.assertEqual(result["knowledge"], k)

    def test_unknown_tool_returns_error_observation(self):
        agent = _FakeAgent()
        executor = asyncio.run(make_action_executor(agent, {}, None))
        result = asyncio.run(executor("nope", {}, 1))
        self.assertEqual(result["kind"], "observation")
        self.assertTrue(result["text"].startswith("Error:"))


def _make_tool_with_pending(agent):
    async def get_dynamic_tool_for(knowledge_item, tool_info):
        class _Args(BaseModel):
            params: str = Field(description="params")

        def _noop(**kwargs):
            return "The query returned N items."

        return StructuredTool.from_function(
            _noop, name="uspto_search", description="d", args_schema=_Args)
    return get_dynamic_tool_for


async def _registry_with_one_knowledge(agent, knowledge):
    tool_info = _ToolInfo("uspto search")
    from sources.agents.react_tools import ToolEntry
    dynamic_tool = await agent.get_dynamic_tool_for(knowledge, tool_info)
    entry = ToolEntry(name=dynamic_tool.name, kind="knowledge",
                      knowledge=knowledge, tool_info=tool_info, tool=dynamic_tool)
    return {entry.name: entry}, []


if __name__ == "__main__":
    unittest.main()
