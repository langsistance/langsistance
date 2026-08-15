"""Tests for ReAct tool supply and action dispatch (react_tools)."""
import asyncio
import re
import unittest
from unittest.mock import AsyncMock, patch

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

    def get_dynamic_tool_for(self, knowledge_item, tool_info):
        # sync — mirrors the production GeneralAgent.get_dynamic_tool_for
        class _Args(BaseModel):
            params: str = Field(description="params")

        def _noop(**kwargs):
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

    @patch("sources.agents.react_tools.get_knowledge_tool_candidates")
    def test_build_tool_set_supports_sync_get_dynamic_tool_for(self, mock_candidates):
        """Production GeneralAgent.get_dynamic_tool_for is SYNC (returns a
        StructuredTool directly) — build_tool_set must not await its result."""
        mock_candidates.return_value = [
            (_Knowledge(3, ktype=1), _ToolInfo("uspto search")),
        ]

        class _SyncAgent(_FakeAgent):
            def get_dynamic_tool_for(self, knowledge_item, tool_info):
                # sync — mirrors sources/agents/general_agent.py:1245
                class _Args(BaseModel):
                    params: str = Field(description="params")

                def _noop(**kwargs):
                    return "tool result"

                name = (re.sub(r"[^a-zA-Z0-9_-]", "_", tool_info.title)
                        .strip("_") or "tool")
                return StructuredTool.from_function(
                    _noop, name=name, description=tool_info.description,
                    args_schema=_Args)

        agent = _SyncAgent()
        registry, tools = asyncio.run(
            build_tool_set(agent, "u1", "专利检索", push_filter=None))
        self.assertIn("uspto_search", registry)
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
    def get_dynamic_tool_for(knowledge_item, tool_info):
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
    dynamic_tool = agent.get_dynamic_tool_for(knowledge, tool_info)
    entry = ToolEntry(name=dynamic_tool.name, kind="knowledge",
                      knowledge=knowledge, tool_info=tool_info, tool=dynamic_tool)
    return {entry.name: entry}, []


# ── Search observation digest ────────────────────────────────────────────────

from sources.agents.react_tools import _items_digest


def _usp_raw_item(app_number, title, applicant="ACME Corp", filing="2024-01-15"):
    return {
        "applicationMetaData": {
            "applicationNumberText": app_number,
            "inventionTitle": title,
            "firstApplicantName": applicant,
            "filingDate": filing,
            "applicationStatusDescriptionText": "Patented Case",
        },
    }


class TestItemsDigest(unittest.TestCase):
    def test_formats_usp_items(self):
        items = [_usp_raw_item("19511555", "Air dryer humidity control",
                               applicant="New York Air Brake")]
        text = _items_digest(items)
        self.assertIn("19511555", text)
        self.assertIn("Air dryer humidity control", text)
        self.assertIn("New York Air Brake", text)

    def test_caps_at_20_with_total_note(self):
        items = [_usp_raw_item(str(19500000 + i), f"Title {i}") for i in range(30)]
        text = _items_digest(items)
        self.assertNotIn("Title 20", text)  # 21st item excluded
        self.assertIn("共 30 条", text)

    def test_non_usp_falls_back_to_truncated_json(self):
        text = _items_digest([{"patentNumber": "US10150077B2",
                               "inventionTitle": "Air dryer"}])
        self.assertIn("US10150077B2", text)

    def test_empty_returns_empty(self):
        self.assertEqual(_items_digest([]), "")
        self.assertEqual(_items_digest(None), "")


class TestSearchObservationContent(unittest.TestCase):
    """Search results observation carries real items, not just counts."""

    def test_executor_returns_digest_for_raw_items(self):
        agent = _FakeAgent()
        entry_k = _Knowledge(3, ktype=1)
        agent.get_dynamic_tool_for = _make_tool_with_pending(agent)
        registry, tools = asyncio.run(
            _registry_with_one_knowledge(agent, entry_k))
        executor = asyncio.run(make_action_executor(agent, registry, None))
        agent._pending_raw_items = [
            _usp_raw_item("19511555", "Air dryer humidity control"),
            _usp_raw_item("18184836", "Moisture control enclosure"),
        ]
        result = asyncio.run(executor("uspto_search", {"params": "{}"}, 1))
        self.assertEqual(result["kind"], "observation")
        self.assertIn("19511555", result["text"])
        self.assertIn("Air dryer humidity control", result["text"])
        self.assertIn("完整列表已展示", result["text"])


if __name__ == "__main__":
    unittest.main()


# ── Deterministic search query rewriting ─────────────────────────────────────

from sources.agents.react_tools import _maybe_rewrite_search_query


class _RewriteAgent(_FakeAgent):
    def __init__(self):
        super().__init__()
        self._last_user_prompt = "工业在线干燥空气源提供与湿度控制"
        self._search_rewrite = None

        class _LLM:
            def __init__(self):
                self.calls = 0

            async def complete_json(self, system, user):
                self.calls += 1
                return {"queries": [
                    '("compressed air dryer" OR "air dryer") AND ("humidity control" OR "dew point")',
                ]}
        self.llm = _LLM()


class TestMaybeRewriteSearchQuery(unittest.IsolatedAsyncioTestCase):
    async def test_rewrites_q_key(self):
        agent = _RewriteAgent()
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"q": "raw noise patent"})
        self.assertIn('"compressed air dryer"', out["q"])

    async def test_rewrites_query_key(self):
        agent = _RewriteAgent()
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"query": "raw"})
        self.assertIn('"air dryer"', out["query"])

    async def test_rewrites_params_json_string(self):
        agent = _RewriteAgent()
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"),
            {"params": '{"q": "raw", "pagination": {"offset": 0, "limit": 50}}'})
        import json
        parsed = json.loads(out["params"])
        self.assertIn('"compressed air dryer"', parsed["q"])
        self.assertEqual(parsed["pagination"]["limit"], 50)

    async def test_cached_across_calls(self):
        agent = _RewriteAgent()
        await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"q": "a"})
        await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"q": "b"})
        self.assertEqual(agent.llm.calls, 1)

    async def test_skips_non_keyword_tools(self):
        agent = _RewriteAgent()
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("get_patent_documents_application_number"), {"q": "raw"})
        self.assertEqual(out, {"q": "raw"})

    async def test_rewrite_failure_keeps_original_args(self):
        class _FailingLLM:
            async def complete_json(self, system, user):
                raise RuntimeError("down")
        agent = _RewriteAgent()
        agent.llm = _FailingLLM()
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"q": "raw"})
        self.assertEqual(out, {"q": "raw"})

    async def test_empty_queries_keep_original_args(self):
        class _EmptyLLM:
            async def complete_json(self, system, user):
                return {"queries": []}
        agent = _RewriteAgent()
        agent.llm = _EmptyLLM()
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"q": "raw"})
        self.assertEqual(out, {"q": "raw"})

    async def test_args_without_query_key_unchanged(self):
        agent = _RewriteAgent()
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"other": "x"})
        self.assertEqual(out, {"other": "x"})


# ── fetch_patent_spec built-in tool ──────────────────────────────────────────

from sources.agents.react_tools import (
    FETCH_PATENT_SPEC_TOOL_NAME,
    build_tool_set,
)


class _SpecAgent(_FakeAgent):
    def __init__(self):
        super().__init__()
        self._last_user_prompt = "分析这篇专利的技术方案"
        self.llm = _RewriteAgent().llm  # reuse complete_json stub


class TestFetchPatentSpecRegistered(unittest.TestCase):
    @patch("sources.agents.react_tools.get_knowledge_tool_candidates")
    def test_tool_always_registered(self, mock_candidates):
        mock_candidates.return_value = []
        agent = _FakeAgent()
        registry, tools = asyncio.run(
            build_tool_set(agent, "u1", "任意问题", push_filter=None))
        self.assertIn(FETCH_PATENT_SPEC_TOOL_NAME, registry)
        self.assertEqual(
            registry[FETCH_PATENT_SPEC_TOOL_NAME].kind, "patent_spec")
        self.assertEqual(len(tools), len(registry))


class TestFetchPatentSpecExecution(unittest.IsolatedAsyncioTestCase):
    def _registry(self, agent):
        entry = type("E", (), {
            "name": FETCH_PATENT_SPEC_TOOL_NAME, "kind": "patent_spec",
            "knowledge": None, "tool_info": None, "tool": None,
        })()
        return {FETCH_PATENT_SPEC_TOOL_NAME: entry}

    async def test_downloads_distills_and_returns_observation(self):
        agent = _SpecAgent()
        executor = await make_action_executor(agent, self._registry(agent), None)
        with patch("sources.long_task.uspto_download.download_uspto_patent_text",
                   new=AsyncMock(return_value=("FULL SPEC TEXT", None))):
            with patch("sources.long_task.patent_distill.distill_patent_spec",
                       new=AsyncMock(return_value={
                           "发明点": "a", "技术方案": "b",
                           "权利要求要点": "c",
                       })):
                result = await executor(
                    FETCH_PATENT_SPEC_TOOL_NAME,
                    {"patent_id": "19511555"}, 2)
        self.assertEqual(result["kind"], "observation")
        self.assertIn("发明点", result["text"])
        self.assertIn("c", result["text"])

    async def test_download_failure_returns_error_observation(self):
        agent = _SpecAgent()
        executor = await make_action_executor(agent, self._registry(agent), None)
        with patch("sources.long_task.uspto_download.download_uspto_patent_text",
                   new=AsyncMock(return_value=(None, None))):
            result = await executor(
                FETCH_PATENT_SPEC_TOOL_NAME,
                {"patent_id": "19511555"}, 2)
        self.assertTrue(result["text"].startswith("Error:"))
        self.assertIn("说明书", result["text"])

    async def test_distill_failure_falls_back_to_truncated_text(self):
        agent = _SpecAgent()
        executor = await make_action_executor(agent, self._registry(agent), None)
        with patch("sources.long_task.uspto_download.download_uspto_patent_text",
                   new=AsyncMock(return_value=("x" * 50000, None))):
            with patch("sources.long_task.patent_distill.distill_patent_spec",
                       new=AsyncMock(return_value={})):
                result = await executor(
                    FETCH_PATENT_SPEC_TOOL_NAME,
                    {"patent_id": "19511555"}, 2)
        self.assertEqual(len(result["text"]), 16000)

    async def test_binary_only_result_returns_error(self):
        agent = _SpecAgent()
        executor = await make_action_executor(agent, self._registry(agent), None)
        with patch("sources.long_task.uspto_download.download_uspto_patent_text",
                   new=AsyncMock(return_value=(None, b"PDF"))):
            result = await executor(
                FETCH_PATENT_SPEC_TOOL_NAME,
                {"patent_id": "19511555"}, 2)
        self.assertTrue(result["text"].startswith("Error:"))


# ── Total-hit count in search observations ───────────────────────────────────

class TestSearchObservationTotalCount(unittest.TestCase):
    def test_observation_includes_total_hits_when_present(self):
        agent = _FakeAgent()
        agent._last_search_total = 8750325
        entry_k = _Knowledge(3, ktype=1)
        agent.get_dynamic_tool_for = _make_tool_with_pending(agent)
        registry, tools = asyncio.run(
            _registry_with_one_knowledge(agent, entry_k))
        executor = asyncio.run(make_action_executor(agent, registry, None))
        agent._pending_raw_items = [
            _usp_raw_item("19511555", "Air dryer humidity control"),
        ]
        result = asyncio.run(executor("uspto_search", {"params": "{}"}, 1))
        self.assertIn("总命中 8750325", result["text"])

    def test_observation_omits_total_when_absent(self):
        agent = _FakeAgent()
        entry_k = _Knowledge(3, ktype=1)
        agent.get_dynamic_tool_for = _make_tool_with_pending(agent)
        registry, tools = asyncio.run(
            _registry_with_one_knowledge(agent, entry_k))
        executor = asyncio.run(make_action_executor(agent, registry, None))
        agent._pending_raw_items = [
            _usp_raw_item("19511555", "Air dryer humidity control"),
        ]
        result = asyncio.run(executor("uspto_search", {"params": "{}"}, 1))
        self.assertNotIn("总命中", result["text"])


if __name__ == "__main__":
    unittest.main()
