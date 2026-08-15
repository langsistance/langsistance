"""Tests for relevance_gate — candidate scoring and gated search loop."""
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from sources.long_task.relevance_gate import (
    apply_scores,
    filter_by_relevance,
    phase0_gated_search,
    run_gated_search,
)
from sources.long_task.search_query_builder import build_search_queries


class _FakeProvider:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def complete_json(self, system, user):
        self.calls.append(user)
        if not self._responses:
            return {}
        r = self._responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _usp_item(app_number, title):
    return {
        "applicationNumberText": app_number,
        "applicationMetaData": {
            "inventionTitle": title,
            "firstApplicantName": "ACME",
            "filingDate": "2024-01-15",
            "applicationStatusDescriptionText": "Patented Case",
        },
        "parentContinuityBag": [],
    }


class TestApplyScores(unittest.TestCase):
    def test_attaches_scores_by_id(self):
        candidates = [
            {"patent_id": "11111111", "title": "A"},
            {"patent_id": "22222222", "title": "B"},
        ]
        result = {"scores": [
            {"id": "11111111", "score": 5},
            {"id": "22222222", "score": 1},
            {"id": "99999999", "score": 5},
        ]}
        out = apply_scores(candidates, result)
        self.assertEqual(out[0]["relevance_score"], 5)
        self.assertEqual(out[1]["relevance_score"], 1)

    def test_ignores_invalid_scores(self):
        candidates = [{"patent_id": "11111111", "title": "A"}]
        result = {"scores": [
            {"id": "11111111", "score": 99},
            {"id": "11111111", "score": "five"},
        ]}
        out = apply_scores(candidates, result)
        self.assertNotIn("relevance_score", out[0])

    def test_garbage_result_keeps_candidates(self):
        candidates = [{"patent_id": "11111111", "title": "A"}]
        out = apply_scores(candidates, None)
        self.assertEqual(out, candidates)


class TestFilterByRelevance(unittest.TestCase):
    def test_keeps_above_threshold_sorted_desc(self):
        candidates = [
            {"patent_id": "1", "relevance_score": 2},
            {"patent_id": "2", "relevance_score": 5},
            {"patent_id": "3", "relevance_score": 3},
        ]
        kept = filter_by_relevance(candidates)
        self.assertEqual([c["patent_id"] for c in kept], ["2", "3"])


class TestRunGatedSearch(unittest.IsolatedAsyncioTestCase):
    async def test_overrides_keyword_tool_query_and_scores(self):
        tool = MagicMock()
        tool.title = "search_patent_by_key_word"
        tool.url = "https://api.uspto.gov/api/v1/patent/applications/search"
        selected = {
            "tool": tool,
            "params": {
                "body": {
                    "q": "bad literal translation",
                    "fields": ["applicationMetaData.inventionTitle"],
                    "pagination": {"offset": 0, "limit": 50},
                }
            },
        }
        page = {
            "data": {"count": 2},
            "raw_items": [_usp_item("11111111", "Air dryer humidity control"),
                          _usp_item("22222222", "EV charging station")],
        }
        with patch("sources.long_task.relevance_gate.execute_tool",
                   new=AsyncMock(return_value=page)) as mock_exec:
            provider = _FakeProvider([
                {"scores": [
                    {"id": "11111111", "score": 5},
                    {"id": "22222222", "score": 1},
                ]},
            ])
            out = await run_gated_search(
                selected=selected,
                user_query="工业在线干燥空气源",
                provider=provider,
                rewrite={"queries": ['("air dryer" OR desiccant)']},
                target_count=10,
            )
        # q was replaced with the rewritten query
        sent_params = mock_exec.call_args[0][1]
        self.assertEqual(
            sent_params["body"]["q"], '("air dryer" OR desiccant)')
        # fields were completed
        self.assertIn("applicationMetaData.cpcClassificationBag",
                      sent_params["body"]["fields"])
        # only the relevant candidate survives
        self.assertEqual([c["patent_id"] for c in out["candidates"]],
                         ["11111111"])
        self.assertEqual(out["search_meta"]["gated_dropped"], 1)
        self.assertEqual(out["search_meta"]["final_count"], 1)

    async def test_keeps_original_query_when_rewrite_empty(self):
        tool = MagicMock()
        tool.title = "search_patent_by_key_word"
        tool.url = "https://api.uspto.gov/api/v1/patent/applications/search"
        selected = {
            "tool": tool,
            "params": {"body": {"q": "original llm query",
                                "pagination": {"offset": 0, "limit": 50}}},
        }
        page = {
            "data": {"count": 1},
            "raw_items": [_usp_item("11111111", "Air dryer control")],
        }
        with patch("sources.long_task.relevance_gate.execute_tool",
                   new=AsyncMock(return_value=page)) as mock_exec:
            provider = _FakeProvider([
                {"scores": [{"id": "11111111", "score": 5}]},
            ])
            out = await run_gated_search(
                selected=selected, user_query="干燥空气",
                provider=provider, rewrite={"queries": []},
                target_count=10,
            )
        sent_params = mock_exec.call_args[0][1]
        self.assertEqual(sent_params["body"]["q"], "original llm query")
        self.assertEqual(len(out["candidates"]), 1)

    async def test_pages_through_results_until_target(self):
        tool = MagicMock()
        tool.title = "search_patent_by_key_word"
        tool.url = "https://api.uspto.gov/api/v1/patent/applications/search"
        selected = {
            "tool": tool,
            "params": {"body": {"q": "old", "pagination": {"offset": 0, "limit": 2}}},
        }
        page1 = {
            "data": {"count": 3},
            "raw_items": [_usp_item("11111111", "Relevant dryer one"),
                          _usp_item("22222222", "Noise patent")],
        }
        # Short page (1 item < limit 2) signals the last page → loop stops
        page2 = {
            "data": {"count": 3},
            "raw_items": [_usp_item("33333333", "Relevant dryer two")],
        }
        with patch("sources.long_task.relevance_gate.execute_tool",
                   new=AsyncMock(side_effect=[page1, page2])) as mock_exec:
            provider = _FakeProvider([
                {"scores": [
                    {"id": "11111111", "score": 5},
                    {"id": "22222222", "score": 1},
                ]},
                {"scores": [
                    {"id": "33333333", "score": 4},
                ]},
            ])
            out = await run_gated_search(
                selected=selected, user_query="干燥空气",
                provider=provider,
                rewrite={"queries": ['"air dryer"']},
                target_count=2,
            )
        self.assertEqual(mock_exec.call_count, 2)
        # second page offset advanced by page size
        self.assertEqual(mock_exec.call_args_list[1][0][1]["body"]["pagination"]["offset"], 2)
        self.assertEqual([c["patent_id"] for c in out["candidates"]],
                         ["11111111", "33333333"])

    async def test_scoring_failure_keeps_candidates_unscored(self):
        tool = MagicMock()
        tool.title = "search_patent_by_key_word"
        tool.url = "https://api.uspto.gov/api/v1/patent/applications/search"
        selected = {
            "tool": tool,
            "params": {"body": {"q": "old", "pagination": {"offset": 0, "limit": 50}}},
        }
        page = {
            "data": {"count": 1},
            "raw_items": [_usp_item("11111111", "Air dryer control")],
        }
        with patch("sources.long_task.relevance_gate.execute_tool",
                   new=AsyncMock(return_value=page)):
            provider = _FakeProvider([RuntimeError("provider down")])
            out = await run_gated_search(
                selected=selected, user_query="干燥空气",
                provider=provider, rewrite={"queries": []},
                target_count=10,
            )
        # Gate LLM down → candidates kept unscored (degrade to legacy
        # behavior), pipeline must not raise
        self.assertEqual([c["patent_id"] for c in out["candidates"]],
                         ["11111111"])
        self.assertEqual(out["search_meta"]["candidates_scored"], 1)


class TestPhase0GatedSearch(unittest.IsolatedAsyncioTestCase):
    async def test_rewrites_then_runs_gated_search(self):
        tool = MagicMock()
        tool.title = "search_patent_by_key_word"
        tool.url = "https://api.uspto.gov/api/v1/patent/applications/search"
        selected = {
            "tool": tool,
            "params": {"body": {"q": "old", "pagination": {"offset": 0, "limit": 50}}},
        }
        page = {
            "data": {"count": 1},
            "raw_items": [_usp_item("11111111", "Air dryer control")],
        }
        with patch("sources.long_task.relevance_gate.execute_tool",
                   new=AsyncMock(return_value=page)) as mock_exec:
            # first provider response = rewrite, second = gate scores
            provider = _FakeProvider([
                {"queries": ['("air dryer" OR desiccant)']},
                {"scores": [{"id": "11111111", "score": 5}]},
            ])
            out = await phase0_gated_search(
                selected=selected, user_query="工业干燥空气源",
                provider=provider, target_count=10,
            )
        sent_params = mock_exec.call_args[0][1]
        self.assertEqual(sent_params["body"]["q"], '("air dryer" OR desiccant)')
        self.assertEqual(out["search_meta"]["final_count"], 1)

    async def test_rewrite_failure_keeps_llm_query(self):
        tool = MagicMock()
        tool.title = "search_patent_by_key_word"
        tool.url = "https://api.uspto.gov/api/v1/patent/applications/search"
        selected = {
            "tool": tool,
            "params": {"body": {"q": "llm built query",
                                "pagination": {"offset": 0, "limit": 50}}},
        }
        page = {
            "data": {"count": 1},
            "raw_items": [_usp_item("11111111", "Air dryer control")],
        }
        with patch("sources.long_task.relevance_gate.execute_tool",
                   new=AsyncMock(return_value=page)) as mock_exec:
            provider = _FakeProvider([
                RuntimeError("rewrite down"),
                {"scores": [{"id": "11111111", "score": 5}]},
            ])
            out = await phase0_gated_search(
                selected=selected, user_query="干燥空气",
                provider=provider, target_count=10,
            )
        sent_params = mock_exec.call_args[0][1]
        self.assertEqual(sent_params["body"]["q"], "llm built query")
        self.assertEqual(len(out["candidates"]), 1)


class TestBatchTextIncludesCpc(unittest.TestCase):
    def test_cpc_codes_rendered_in_batch_lines(self):
        from sources.long_task.relevance_gate import _batch_text
        candidates = [{
            "patent_id": "19511555",
            "title": "Air dryer humidity control",
            "applicant": "ACME",
            "filing_date": "2024-01-15",
            "cpc_codes": ["F26B 21/08", "B01D 53/26"],
        }]
        text = _batch_text(candidates, "干燥空气")
        self.assertIn("F26B 21/08", text)
        self.assertIn("B01D 53/26", text)

    def test_missing_cpc_renders_empty(self):
        from sources.long_task.relevance_gate import _batch_text
        candidates = [{
            "patent_id": "19511555", "title": "T", "applicant": "A",
            "filing_date": "2024-01-15", "cpc_codes": [],
        }]
        text = _batch_text(candidates, "q")
        self.assertIn("cpc=[]", text)


class TestGatePromptAllowsCpc(unittest.TestCase):
    def test_prompt_mentions_cpc(self):
        from sources.long_task.relevance_gate import GATE_SYSTEM_PROMPT
        self.assertIn("CPC", GATE_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
