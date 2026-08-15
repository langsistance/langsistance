"""Tests for ReAct tool supply and action dispatch (react_tools)."""
import asyncio
import re
import sys
import types
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
        "applicationNumberText": app_number,
        "applicationMetaData": {
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
    """v4 semantics: the LLM's explicit q wins; the tightest ladder query
    is injected only when the q slot is absent or empty."""

    async def test_explicit_q_preserved(self):
        agent = _RewriteAgent()
        agent._search_rewrite = {"queries": ['("a" OR "b") AND ("c" OR "d")']}
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"),
            {"q": '("humidity control" OR "dew point") AND industrial'})
        self.assertEqual(
            out["q"], '("humidity control" OR "dew point") AND industrial')

    async def test_empty_q_injected_tightest(self):
        agent = _RewriteAgent()
        agent._search_rewrite = {"queries": ['("a" OR "b") AND ("c" OR "d")']}
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"q": ""})
        self.assertEqual(out["q"], '("a" OR "b") AND ("c" OR "d")')

    async def test_missing_q_injected_tightest(self):
        agent = _RewriteAgent()
        agent._search_rewrite = {"queries": ['("a" OR "b") AND ("c" OR "d")']}
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"page": 1})
        self.assertEqual(out["q"], '("a" OR "b") AND ("c" OR "d")')
        self.assertEqual(out["page"], 1)

    async def test_explicit_query_key_preserved(self):
        agent = _RewriteAgent()
        agent._search_rewrite = {"queries": ['("a" OR "b")']}
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"query": "my own"})
        self.assertEqual(out["query"], "my own")

    async def test_params_json_with_explicit_q_preserved(self):
        agent = _RewriteAgent()
        agent._search_rewrite = {"queries": ['("a" OR "b")']}
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"),
            {"params": '{"q": "mine", "pagination": {"offset": 0, "limit": 50}}'})
        import json
        parsed = json.loads(out["params"])
        self.assertEqual(parsed["q"], "mine")
        self.assertEqual(parsed["pagination"]["limit"], 50)

    async def test_params_json_without_q_injected(self):
        agent = _RewriteAgent()
        agent._search_rewrite = {"queries": ['("a" OR "b")']}
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"),
            {"params": '{"pagination": {"offset": 0, "limit": 50}}'})
        import json
        parsed = json.loads(out["params"])
        self.assertEqual(parsed["q"], '("a" OR "b")')
        self.assertEqual(parsed["pagination"]["limit"], 50)

    async def test_skips_non_keyword_tools(self):
        agent = _RewriteAgent()
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("get_patent_documents_application_number"), {"q": "raw"})
        self.assertEqual(out, {"q": "raw"})

    async def test_no_cache_no_queries_keeps_original(self):
        agent = _RewriteAgent()
        agent._search_rewrite = {"queries": []}
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"q": ""})
        self.assertEqual(out, {"q": ""})

    async def test_keyless_args_injected_tightest(self):
        agent = _RewriteAgent()
        agent._search_rewrite = {"queries": ['("a" OR "b")']}
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"other": "x"})
        self.assertEqual(out["q"], '("a" OR "b")')
        self.assertEqual(out["other"], "x")

    async def test_lazy_build_when_cache_missing(self):
        """Executor fallback: cache absent (non-create_agent path) → builds
        the rewrite once via the agent's provider."""
        agent = _RewriteAgent()
        agent._search_rewrite = None
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"q": ""})
        self.assertEqual(agent.llm.calls, 1)
        self.assertIn('"compressed air dryer"', out["q"])
        # second call reuses the cache
        out2 = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"q": ""})
        self.assertEqual(agent.llm.calls, 1)
        self.assertEqual(out2["q"], out["q"])

    async def test_lazy_build_failure_keeps_original_args(self):
        class _FailingLLM:
            async def complete_json(self, system, user):
                raise RuntimeError("down")
        agent = _RewriteAgent()
        agent._search_rewrite = None
        agent.llm = _FailingLLM()
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"q": ""})
        # build failure degrades to empty queries → original args untouched
        self.assertEqual(out, {"q": ""})

    async def test_malformed_params_json_keeps_original_args(self):
        agent = _RewriteAgent()
        agent._search_rewrite = {"queries": ['("a" OR "b")']}
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"),
            {"params": "not valid json{{{"})
        self.assertEqual(out, {"params": "not valid json{{{"})


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


# ── Chat-path relevance ranking pool ─────────────────────────────────────────

from sources.agents.react_tools import _get_flash_provider, _ranked_digest


class _PoolAgent(_FakeAgent):
    def __init__(self):
        super().__init__()
        self._last_user_prompt = "工业干燥空气供应"
        self._search_pool = None
        self._search_ranked = False
        self._last_search_total = None

        class _LLM:
            def __init__(self):
                self.calls = 0
                self._scores = {}
                self.fail = False

            def set_scores(self, scores):
                self._scores = scores

            async def complete_json(self, system, user):
                self.calls += 1
                if self.fail:
                    raise RuntimeError("down")
                ids = re.findall(r"id=(\d+)", user)
                return {"scores": [
                    {"id": i, "score": self._scores.get(i, 3)} for i in ids]}

        self.llm = _LLM()
        self._flash_llm = self.llm  # pre-seeded cache: fake scores via flash path


async def _pool_executor(agent, registry):
    return await make_action_executor(agent, registry, None)


class TestRankedDigest(unittest.TestCase):
    def test_lines_carry_scores(self):
        from sources.long_task.candidate_metadata import build_candidates
        items = [_usp_raw_item("19511555", "Air dryer humidity control"),
                 _usp_raw_item("18184836", "Moisture control enclosure")]
        candidates = build_candidates(items)
        candidates[0]["relevance_score"] = 5
        text = _ranked_digest(candidates)
        self.assertIn("相关度5/5", text)
        # second candidate is unscored — its line carries no score suffix
        unscored_line = text.split("\n")[1]
        self.assertNotIn("相关度", unscored_line)

    def test_caps_at_20_with_total_note(self):
        from sources.long_task.candidate_metadata import build_candidates
        items = [_usp_raw_item(str(19500000 + i), f"Title {i}") for i in range(30)]
        text = _ranked_digest(build_candidates(items))
        self.assertNotIn("Title 20", text)
        self.assertIn("共 30 条", text)
        self.assertIn("已按相关度排序", text)


_USPTO_URL = "https://api.uspto.gov/api/v1/patent/applications/search"


class TestExecuteActionPoolPath(unittest.TestCase):
    @patch("sources.agents.react_tools.RELEVANCE_RANK_ENABLED", True)
    def test_pool_path_ranks_observation_by_score(self):
        agent = _PoolAgent()
        agent.llm.set_scores({"19511555": 1, "18184836": 5})
        entry_k = _Knowledge(3, ktype=1)
        agent.get_dynamic_tool_for = _make_tool_with_pending(agent)
        tool_info = _ToolInfo("search_patent_by_key_word", url=_USPTO_URL)
        from sources.agents.react_tools import ToolEntry
        dynamic_tool = agent.get_dynamic_tool_for(entry_k, tool_info)
        entry = ToolEntry(name=dynamic_tool.name, kind="knowledge",
                          knowledge=entry_k, tool_info=tool_info,
                          tool=dynamic_tool)
        executor = asyncio.run(make_action_executor(agent, {entry.name: entry}, None))
        agent._pending_raw_items = [
            _usp_raw_item("19511555", "Air dryer humidity control"),
            _usp_raw_item("18184836", "Moisture control enclosure"),
        ]
        result = asyncio.run(executor("uspto_search", {"params": "{}"}, 1))
        self.assertEqual(result["kind"], "observation")
        self.assertIn("已按相关度排序", result["text"])
        self.assertIn("相关度5/5", result["text"])
        pos_high = result["text"].find("18184836")
        pos_low = result["text"].find("19511555")
        self.assertLess(pos_high, pos_low)
        self.assertTrue(agent._search_ranked)
        # display list reordered to ranked order
        ids = [str(i.get("applicationNumberText"))
               for i in agent._pending_raw_items]
        self.assertEqual(ids, ["18184836", "19511555"])

    @patch("sources.agents.react_tools.RELEVANCE_RANK_ENABLED", True)
    def test_pool_merges_across_calls(self):
        agent = _PoolAgent()
        agent.llm.set_scores({})
        entry_k = _Knowledge(3, ktype=1)
        agent.get_dynamic_tool_for = _make_tool_with_pending(agent)
        tool_info = _ToolInfo("search_patent_by_key_word", url=_USPTO_URL)
        from sources.agents.react_tools import ToolEntry
        dynamic_tool = agent.get_dynamic_tool_for(entry_k, tool_info)
        entry = ToolEntry(name=dynamic_tool.name, kind="knowledge",
                          knowledge=entry_k, tool_info=tool_info,
                          tool=dynamic_tool)
        executor = asyncio.run(make_action_executor(agent, {entry.name: entry}, None))
        agent._pending_raw_items = [_usp_raw_item("19511555", "First call")]
        asyncio.run(executor("uspto_search", {"params": "{}"}, 1))
        agent._pending_raw_items = [_usp_raw_item("18184836", "Second call")]
        result = asyncio.run(executor("uspto_search", {"params": "{}"}, 2))
        self.assertIn("19511555", result["text"])  # first call still present
        self.assertIn("18184836", result["text"])
        self.assertIn("池共 2 条", result["text"])

    @patch("sources.agents.react_tools.RELEVANCE_RANK_ENABLED", True)
    def test_pool_skipped_for_non_uspto_url(self):
        agent = _PoolAgent()
        entry_k = _Knowledge(3, ktype=1)
        agent.get_dynamic_tool_for = _make_tool_with_pending(agent)
        tool_info = _ToolInfo("uspto search", url="https://api.example.com/search")
        from sources.agents.react_tools import ToolEntry
        dynamic_tool = agent.get_dynamic_tool_for(entry_k, tool_info)
        entry = ToolEntry(name=dynamic_tool.name, kind="knowledge",
                          knowledge=entry_k, tool_info=tool_info,
                          tool=dynamic_tool)
        executor = asyncio.run(make_action_executor(agent, {entry.name: entry}, None))
        agent._pending_raw_items = [_usp_raw_item("19511555", "A")]
        result = asyncio.run(executor("uspto_search", {"params": "{}"}, 1))
        self.assertNotIn("已按相关度排序", result["text"])
        self.assertFalse(getattr(agent, "_search_ranked", False))

    @patch("sources.agents.react_tools.RELEVANCE_RANK_ENABLED", True)
    def test_pool_skipped_for_non_usp_shape(self):
        agent = _PoolAgent()
        entry_k = _Knowledge(3, ktype=1)
        agent.get_dynamic_tool_for = _make_tool_with_pending(agent)
        tool_info = _ToolInfo("search_patent_by_key_word", url=_USPTO_URL)
        from sources.agents.react_tools import ToolEntry
        dynamic_tool = agent.get_dynamic_tool_for(entry_k, tool_info)
        entry = ToolEntry(name=dynamic_tool.name, kind="knowledge",
                          knowledge=entry_k, tool_info=tool_info,
                          tool=dynamic_tool)
        executor = asyncio.run(make_action_executor(agent, {entry.name: entry}, None))
        agent._pending_raw_items = [{"patentNumber": "US10150077B2"}]
        result = asyncio.run(executor("uspto_search", {"params": "{}"}, 1))
        self.assertNotIn("已按相关度排序", result["text"])

    def test_flag_off_falls_back_to_legacy(self):
        agent = _PoolAgent()
        entry_k = _Knowledge(3, ktype=1)
        agent.get_dynamic_tool_for = _make_tool_with_pending(agent)
        tool_info = _ToolInfo("search_patent_by_key_word", url=_USPTO_URL)
        from sources.agents.react_tools import ToolEntry
        dynamic_tool = agent.get_dynamic_tool_for(entry_k, tool_info)
        entry = ToolEntry(name=dynamic_tool.name, kind="knowledge",
                          knowledge=entry_k, tool_info=tool_info,
                          tool=dynamic_tool)
        with patch("sources.agents.react_tools.RELEVANCE_RANK_ENABLED", False):
            executor = asyncio.run(make_action_executor(agent, {entry.name: entry}, None))
            agent._pending_raw_items = [_usp_raw_item("19511555", "A")]
            result = asyncio.run(executor("uspto_search", {"params": "{}"}, 1))
        self.assertNotIn("已按相关度排序", result["text"])
        self.assertIn("完整列表已展示", result["text"])

    @patch("sources.agents.react_tools.RELEVANCE_RANK_ENABLED", True)
    def test_scoring_failure_keeps_rank_stable(self):
        agent = _PoolAgent()
        agent.llm.fail = True
        entry_k = _Knowledge(3, ktype=1)
        agent.get_dynamic_tool_for = _make_tool_with_pending(agent)
        tool_info = _ToolInfo("search_patent_by_key_word", url=_USPTO_URL)
        from sources.agents.react_tools import ToolEntry
        dynamic_tool = agent.get_dynamic_tool_for(entry_k, tool_info)
        entry = ToolEntry(name=dynamic_tool.name, kind="knowledge",
                          knowledge=entry_k, tool_info=tool_info,
                          tool=dynamic_tool)
        executor = asyncio.run(make_action_executor(agent, {entry.name: entry}, None))
        agent._pending_raw_items = [
            _usp_raw_item("19511555", "A"), _usp_raw_item("18184836", "B")]
        result = asyncio.run(executor("uspto_search", {"params": "{}"}, 1))
        self.assertIn("已按相关度排序", result["text"])
        self.assertIn("本次新评分 0 条", result["text"])

    @patch("sources.agents.react_tools.RELEVANCE_RANK_ENABLED", True)
    def test_assignee_style_uspto_tool_pools(self):
        agent = _PoolAgent()
        agent.llm.set_scores({})
        entry_k = _Knowledge(3, ktype=1)
        agent.get_dynamic_tool_for = _make_tool_with_pending(agent)
        tool_info = _ToolInfo("search_patent_by_assignee", url=_USPTO_URL)
        from sources.agents.react_tools import ToolEntry
        dynamic_tool = agent.get_dynamic_tool_for(entry_k, tool_info)
        entry = ToolEntry(name=dynamic_tool.name, kind="knowledge",
                          knowledge=entry_k, tool_info=tool_info,
                          tool=dynamic_tool)
        executor = asyncio.run(make_action_executor(agent, {entry.name: entry}, None))
        agent._pending_raw_items = [_usp_raw_item("19511555", "A")]
        result = asyncio.run(executor(entry.name, {"params": "{}"}, 1))
        self.assertIn("已按相关度排序", result["text"])

    @patch("sources.agents.react_tools.RELEVANCE_RANK_ENABLED", True)
    def test_legacy_call_keeps_pooled_display(self):
        agent = _PoolAgent()
        agent.llm.set_scores({"19511555": 5})
        entry_k = _Knowledge(3, ktype=1)
        agent.get_dynamic_tool_for = _make_tool_with_pending(agent)
        tool_info = _ToolInfo("search_patent_by_key_word", url=_USPTO_URL)
        from sources.agents.react_tools import ToolEntry
        dynamic_tool = agent.get_dynamic_tool_for(entry_k, tool_info)
        entry = ToolEntry(name=dynamic_tool.name, kind="knowledge",
                          knowledge=entry_k, tool_info=tool_info,
                          tool=dynamic_tool)
        executor = asyncio.run(make_action_executor(agent, {entry.name: entry}, None))
        # First: pooled call populates the ranked display
        agent._pending_raw_items = [_usp_raw_item("19511555", "A")]
        asyncio.run(executor(entry.name, {"params": "{}"}, 1))
        pooled_display = agent._pending_raw_items
        self.assertEqual(len(pooled_display), 1)
        self.assertIs(agent._search_ranked, True)
        # Then: a legacy (non-USPTO-url) tool call must NOT overwrite the
        # ranked display. The ranked pool exists, so the legacy else-branch
        # skips its own `_pending_raw_items = shown` overwrite and the
        # ranked display list is preserved across the dispatch.
        legacy_info = _ToolInfo("uspto search", url="https://api.example.com/search")
        legacy_tool = agent.get_dynamic_tool_for(entry_k, legacy_info)
        legacy_entry = ToolEntry(name=legacy_tool.name, kind="knowledge",
                                 knowledge=entry_k, tool_info=legacy_info,
                                 tool=legacy_tool)
        executor2 = asyncio.run(
            make_action_executor(agent, {legacy_entry.name: legacy_entry}, None))
        agent._pending_raw_items = [_usp_raw_item("18184836", "Legacy noise")]
        result = asyncio.run(executor2(legacy_entry.name, {"params": "{}"}, 2))
        # production clobber happened above; the legacy branch must RESTORE
        # the ranked pool display (content equality — ranked() rebuilds the list)
        restored_ids = [str(i.get("applicationNumberText"))
                        for i in agent._pending_raw_items]
        self.assertEqual(restored_ids, ["19511555"])
        self.assertNotIn("已按相关度排序", result["text"])


# sources/llm_provider imports `ollama` (optional, not installed here); that
# would break patch("sources.llm_provider.Provider", ...) resolution.  Pre-
# register a stub module so the patch target resolves without the real import.
_LLM_MODULE_STUB = types.ModuleType("sources.llm_provider")
_LLM_MODULE_STUB.Provider = object  # patch target must pre-exist (no create=True)
sys.modules.setdefault("sources.llm_provider", _LLM_MODULE_STUB)


class TestFlashProviderSelection(unittest.TestCase):
    def test_returns_cached_provider(self):
        agent = _PoolAgent()
        self.assertIs(_get_flash_provider(agent), agent.llm)

    def test_constructs_deepseek_flash_when_cache_empty(self):
        agent = _FakeAgent()
        fake_provider = object()
        with patch("sources.llm_provider.Provider",
                   return_value=fake_provider) as mock_provider, \
             patch("sources.long_task.config.get_long_task_config",
                   return_value={"provider_family": "deepseek"}):
            result = _get_flash_provider(agent)
        self.assertIs(result, fake_provider)
        mock_provider.assert_called_once_with(
            provider_name="deepseek", model="deepseek-v4-flash",
            server_address="", is_local=False)
        self.assertIs(agent._flash_llm, fake_provider)

    def test_constructs_minimax_highspeed_for_minimax_family(self):
        agent = _FakeAgent()
        fake_provider = object()
        with patch("sources.llm_provider.Provider",
                   return_value=fake_provider) as mock_provider, \
             patch("sources.long_task.config.get_long_task_config",
                   return_value={"provider_family": "minimax"}):
            result = _get_flash_provider(agent)
        mock_provider.assert_called_once_with(
            provider_name="minimax", model="MiniMax-M2.7-highspeed",
            server_address="", is_local=False)

    def test_construction_failure_returns_none(self):
        agent = _FakeAgent()
        with patch("sources.llm_provider.Provider",
                   side_effect=RuntimeError("no keys")):
            self.assertIsNone(_get_flash_provider(agent))


class _ListLogger:
    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(msg)


class TestRelevancePoolGateLogging(unittest.TestCase):
    def test_gate_decision_logged_for_legacy_tool(self):
        agent = _PoolAgent()
        agent.logger = _ListLogger()
        entry_k = _Knowledge(3, ktype=1)
        agent.get_dynamic_tool_for = _make_tool_with_pending(agent)
        registry, tools = asyncio.run(_registry_with_one_knowledge(agent, entry_k))
        executor = asyncio.run(make_action_executor(agent, registry, None))
        agent._pending_raw_items = [_usp_raw_item("19511555", "A")]
        asyncio.run(executor("uspto_search", {"params": "{}"}, 1))
        gate_lines = [ln for ln in agent.logger.lines
                      if ln.startswith("relevance_pool gate")]
        self.assertEqual(len(gate_lines), 1)
        self.assertIn("applies=False", gate_lines[0])
        self.assertIn("uspto search", gate_lines[0])

    def test_no_logger_does_not_crash(self):
        agent = _PoolAgent()  # _PoolAgent has no logger attribute
        entry_k = _Knowledge(3, ktype=1)
        agent.get_dynamic_tool_for = _make_tool_with_pending(agent)
        registry, tools = asyncio.run(_registry_with_one_knowledge(agent, entry_k))
        executor = asyncio.run(make_action_executor(agent, registry, None))
        agent._pending_raw_items = [_usp_raw_item("19511555", "A")]
        result = asyncio.run(executor("uspto_search", {"params": "{}"}, 1))
        self.assertEqual(result["kind"], "observation")


# ── USPTO envelope + pagination ─────────────────────────────────────────────

from sources.agents.react_tools import (
    REACT_POOL_MAX_PAGES,
    _build_uspto_envelope,
    _collect_search_pages,
    _effective_query,
)


def _keyword_tool_template():
    """A realistic push=2 USPTO keyword tool template (tool_info.params)."""
    import json
    return json.dumps({
        "method": "POST",
        "query": {},
        "path": "/api/v1/patent/applications/search",
        "header": {},
        "body": {
            "q": "",
            "pagination": {"offset": 0, "limit": 50},
            "fields": ["applicationNumberText", "applicationMetaData.inventionTitle"],
            "sort": [{"field": "assignmentBag.assignmentRecordedDate", "order": "desc"}],
        },
    })


class _PagedToolInfo(_ToolInfo):
    def __init__(self):
        super().__init__("search_patent_by_key_word",
                         url="https://api.uspto.gov/api/v1/patent/applications/search")
        self.params = _keyword_tool_template()


class TestEffectiveQuery(unittest.TestCase):
    def test_flat_first_string_wins(self):
        self.assertEqual(
            _effective_query({"q": '("dry air" OR drying)', "page": 1, "pageSize": 10}),
            '("dry air" OR drying)')

    def test_body_q_extracted(self):
        self.assertEqual(
            _effective_query({"method": "POST", "body": {"q": "dryer AND humidit*"}}),
            "dryer AND humidit*")

    def test_params_json_q_extracted(self):
        self.assertEqual(
            _effective_query({"params": '{"q": "dry* AND humid*", "page": 1}'}),
            "dry* AND humid*")

    def test_missing_returns_empty(self):
        self.assertEqual(_effective_query({"page": 1}), "")
        self.assertEqual(_effective_query(None), "")


class TestBuildUsptoEnvelope(unittest.TestCase):
    def test_envelope_carries_template_and_ensures_fields(self):
        tool_info = _PagedToolInfo()
        envelope = _build_uspto_envelope(tool_info, 'dry* AND humid*')
        self.assertEqual(envelope["method"], "POST")
        self.assertEqual(envelope["body"]["q"], "dry* AND humid*")
        self.assertIn("applicationMetaData.cpcClassificationBag",
                      envelope["body"]["fields"])
        self.assertEqual(envelope["body"]["pagination"]["limit"], 50)
        self.assertEqual(envelope["query"], {})
        self.assertEqual(envelope["path"], "/api/v1/patent/applications/search")

    def test_envelope_sort_overridden_to_relevance(self):
        tool_info = _PagedToolInfo()
        envelope = _build_uspto_envelope(tool_info, 'dry* AND humid*')
        self.assertEqual(envelope["body"]["sort"],
                         [{"field": "_score", "order": "desc"}])

    def test_sort_field_env_override(self):
        tool_info = _PagedToolInfo()
        with patch("sources.agents.react_tools.REACT_USPTO_SORT_FIELD",
                   "applicationMetaData.filingDate"):
            envelope = _build_uspto_envelope(tool_info, 'dry* AND humid*')
        self.assertEqual(envelope["body"]["sort"],
                         [{"field": "applicationMetaData.filingDate",
                           "order": "desc"}])


class TestCollectSearchPages(unittest.IsolatedAsyncioTestCase):
    async def _agent_with_pages(self, pages_by_offset):
        agent = _PoolAgent()
        agent.invoked_offsets = []
        entry_k = _Knowledge(3, ktype=1)

        def get_dynamic_tool_for(knowledge_item, tool_info):
            from sources.agents.react_tools import StructuredTool
            class _Args(BaseModel):
                params: object = Field(description="params")
            def _noop(**kwargs):
                # production side effect: write page items + total hits
                import json as _json
                raw = kwargs.get("params")
                if isinstance(raw, str):
                    envelope = _json.loads(raw or "{}")
                else:
                    envelope = raw or {}
                body = envelope.get("body") or {}
                offset = (body.get("pagination") or {}).get("offset", 0)
                agent.invoked_offsets.append(offset)
                page_items, total = pages_by_offset.get(offset, ([], 0))
                agent._pending_raw_items = page_items
                agent._last_search_total = total
                return f"The query returned {len(page_items)} items."
            return StructuredTool.from_function(_noop, name="search_patent_by_key_word",
                                                description="d", args_schema=_Args)
        agent.get_dynamic_tool_for = get_dynamic_tool_for
        tool_info = _PagedToolInfo()
        from sources.agents.react_tools import ToolEntry
        dynamic_tool = agent.get_dynamic_tool_for(entry_k, tool_info)
        entry = ToolEntry(name=dynamic_tool.name, kind="knowledge",
                          knowledge=entry_k, tool_info=tool_info,
                          tool=dynamic_tool)
        return agent, entry

    async def test_pages_merged_until_total_or_cap(self):
        p0 = [_usp_raw_item("19500001", "P0-1"), _usp_raw_item("19500002", "P0-2")]
        p50 = [_usp_raw_item("19500003", "P50-1")]
        p100 = [_usp_raw_item("19500004", "P100-1")]
        agent, entry = await self._agent_with_pages(
            {0: (p0, 120), 50: (p50, 120), 100: (p100, 120)})
        collected = await _collect_search_pages(agent, entry, {"q": "dry*"}, p0)
        self.assertEqual([c["patent_id"] for c in collected],
                         ["19500001", "19500002", "19500003"])
        # page 3 (offset 100) NOT fetched: REACT_POOL_MAX_PAGES=2 pages after first

    async def test_total_exhausted_stops_early(self):
        p0 = [_usp_raw_item("19500001", "P0-1")]
        agent, entry = await self._agent_with_pages({0: (p0, 50)})
        agent._last_search_total = 50  # universe exhausted at page 1
        collected = await _collect_search_pages(agent, entry, {"q": "dry*"}, p0)
        self.assertEqual(len(collected), 1)
        self.assertEqual(agent.invoked_offsets, [])  # total-stop: zero extra invokes

    async def test_duplicate_page_stops(self):
        p0 = [_usp_raw_item("19500001", "P0-1")]
        # template ignores offset → same items come back
        agent, entry = await self._agent_with_pages(
            {0: (p0, 200), 50: (p0, 200)})
        collected = await _collect_search_pages(agent, entry, {"q": "dry*"}, p0)
        self.assertEqual(len(collected), 1)

    async def test_no_q_returns_first_page_only(self):
        agent, entry = await self._agent_with_pages({})
        collected = await _collect_search_pages(agent, entry, {"page": 1}, [])
        self.assertEqual(collected, [])


from sources.agents.react_tools import (
    REACT_TIGHTEN_SUGGEST_THRESHOLD,
    _tighten_hint,
)


class TestTightenHint(unittest.TestCase):
    def test_hint_above_threshold_zh(self):
        self.assertIn("收紧", _tighten_hint(REACT_TIGHTEN_SUGGEST_THRESHOLD, "zh"))

    def test_hint_above_threshold_en(self):
        self.assertIn("tighten", _tighten_hint(REACT_TIGHTEN_SUGGEST_THRESHOLD, "en"))

    def test_no_hint_below_threshold(self):
        self.assertEqual(_tighten_hint(REACT_TIGHTEN_SUGGEST_THRESHOLD - 1, "zh"), "")

    def test_no_hint_for_non_int(self):
        self.assertEqual(_tighten_hint(None, "zh"), "")
        self.assertEqual(_tighten_hint("99999", "zh"), "")


class TestEnvelopeInvokeSchema(unittest.TestCase):
    def test_payload_matches_real_schema(self):
        from sources.agents.general_agent import DynamicBackendToolFunction
        from sources.agents.react_tools import _tool_invoke_payload
        from langchain_core.tools import StructuredTool

        def _noop(**kwargs):
            return "ok"

        tool = StructuredTool.from_function(_noop, name="schema_probe",
                                            description="d",
                                            args_schema=DynamicBackendToolFunction)
        envelope = {"method": "POST", "body": {"q": "x"}, "query": {},
                    "path": None, "header": {}}
        # bare envelope must FAIL the real schema (the production bug)
        with self.assertRaises(Exception):
            tool.invoke(envelope)
        # the payload helper must PASS it
        agent = _FakeAgent()
        agent._last_user_id = "u1"
        agent._last_query_id = "q1"
        result = tool.invoke(_tool_invoke_payload(agent, envelope))
        self.assertEqual(result, "ok")
