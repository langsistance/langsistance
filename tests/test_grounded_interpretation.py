"""Tests for the post-retrieval grounded interpretation module."""
import asyncio
import os
import unittest
from unittest import mock

from sources.long_task import grounded_interpretation as gi
from sources.long_task import technical_interpretation as ti


def _cand(pid, title, applicant="ACME", cpc=("H05B45/20",), score=4,
          filing="2023-01-01"):
    return {"patent_id": pid, "title": title, "applicant": applicant,
            "cpc_codes": list(cpc), "relevance_score": score,
            "filing_date": filing}


class TestCandidateStats(unittest.TestCase):
    def test_applicant_and_cpc_frequency_desc(self):
        stats = gi.candidate_stats([
            _cand("1", "A", "ERP Power", ("H05B45/20", "H05B45/10")),
            _cand("2", "B", "ERP Power", ("H05B45/20",)),
            _cand("3", "C", "Samsung", ()),
        ])
        self.assertEqual(stats["applicants"][0],
                         {"name": "ERP Power", "count": 2})
        self.assertEqual(stats["cpc"][0],
                         {"name": "H05B45/20", "count": 2})
        self.assertEqual(len(stats["applicants"]), 2)

    def test_empty_input(self):
        self.assertEqual(gi.candidate_stats([]),
                         {"applicants": [], "cpc": []})


class TestParseGrounded(unittest.TestCase):
    def test_valid_output_parsed(self):
        parsed = gi.parse_grounded({
            "dimensions": [
                {"name": "D1", "role": "核心层", "line": "L1",
                 "representatives": ["ERP"], "players": ["ERP", "TI"],
                 "cpc": ["h05b45/20"]},
            ],
            "supplementary_queries": ["中文垃圾 AND stuff", '("a")'],
            "supplementary_cpc": ["h05b45/20", "H05B45/20"],
        })
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["dimensions"][0]["representatives"], ["ERP"])
        self.assertEqual(parsed["dimensions"][0]["cpc"], ["H05B45/20"])
        self.assertNotIn("中文", parsed["supplementary_queries"][0])
        self.assertEqual(parsed["supplementary_cpc"], ["H05B45/20"])  # dedup

    def test_unusable_returns_none(self):
        self.assertIsNone(gi.parse_grounded({}))
        self.assertIsNone(gi.parse_grounded("nope"))
        self.assertIsNone(gi.parse_grounded(
            {"dimensions": [], "players": [], "supplementary_queries": []}))


class TestMergeGrounded(unittest.TestCase):
    def test_llm_output_wins_over_stats(self):
        stats = {"applicants": [{"name": "X", "count": 5}], "cpc": []}
        llm = gi.parse_grounded({
            "dimensions": [{"name": "D", "line": "L"}],
            "players": ["ERP"], "supplementary_queries": ['("q")'],
            "supplementary_cpc": ["H05B45/20"],
        })
        out = gi.merge_grounded(stats, llm, {"scheme": "S",
                                             "structure_terms": ["t"]})
        self.assertEqual(out["players"], ["ERP"])
        self.assertEqual(out["scheme"], "S")
        self.assertEqual(out["structure_terms"], ["t"])

    def test_stats_only_fallback_when_llm_fails(self):
        stats = {"applicants": [{"name": "ERP Power", "count": 3},
                                {"name": "Samsung", "count": 2}],
                 "cpc": [{"name": "H05B45/20", "count": 3}]}
        out = gi.merge_grounded(stats, None, None)
        self.assertEqual(out["players"], ["ERP Power", "Samsung"])
        self.assertEqual(out["dimensions"], [])
        self.assertEqual(out["supplementary_queries"], [])


class TestSynthesizeGrounded(unittest.IsolatedAsyncioTestCase):
    class _FakeProvider:
        def __init__(self, result=None, fail=False):
            self.result = result
            self.fail = fail

        async def complete_json(self, system, user, max_retries=0):
            if self.fail:
                raise RuntimeError("down")
            return self.result

    async def test_success_returns_merged(self):
        llm = {"dimensions": [{"name": "D", "line": "L",
                               "representatives": ["ERP"]}],
               "players": ["ERP"],
               "supplementary_queries": ['("q")'],
               "supplementary_cpc": ["H05B45/20"]}
        provider = self._FakeProvider(result=llm)
        with mock.patch.object(gi, "_grounded_provider",
                               return_value=provider):
            out = await gi.synthesize_grounded(
                "问题", [_cand("1", "T")],
                pre_interp={"scheme": "S"})
        self.assertIsNotNone(out)
        self.assertEqual(out["dimensions"][0]["name"], "D")
        self.assertEqual(out["scheme"], "S")

    async def test_failure_falls_back_to_stats_only(self):
        provider = self._FakeProvider(fail=True)
        with mock.patch.object(gi, "_grounded_provider",
                               return_value=provider):
            out = await gi.synthesize_grounded(
                "问题", [_cand("1", "T", applicant="ERP")],
                pre_interp=None)
        self.assertIsNotNone(out)
        self.assertEqual(out["players"], ["ERP"])

    async def test_disabled_returns_none_without_calling(self):
        provider = self._FakeProvider(result={"players": ["X"]})
        with mock.patch.object(gi, "GROUNDED_ENABLED", False):
            with mock.patch.object(gi, "_grounded_provider",
                                   return_value=provider):
                self.assertIsNone(await gi.synthesize_grounded("问题", []))
        self.assertEqual(provider.fail, False)  # no call happened

    async def test_empty_question_returns_none(self):
        with mock.patch.object(gi, "_grounded_provider") as p:
            self.assertIsNone(await gi.synthesize_grounded("", []))
        p.assert_not_called()

    async def test_hanging_provider_times_out_to_stats_fallback(self):
        class _Slow:
            async def complete_json(self, system, user, max_retries=0):
                await asyncio.sleep(30)
                return {}

        with mock.patch.object(gi, "GROUNDED_TIMEOUT", 0.1):
            with mock.patch.object(gi, "_grounded_provider",
                                   return_value=_Slow()):
                out = await gi.synthesize_grounded(
                    "问题", [_cand("1", "T", applicant="ERP")])
        self.assertIsNotNone(out)
        self.assertEqual(out["players"], ["ERP"])


class TestPromptGenericity(unittest.TestCase):
    """通用性铁律（第三次重申）：生产 prompt 零测试提问词汇。"""

    FORBIDDEN = ("RGB", "控制放大器", "独立控制")

    def test_interpret_prompt_free_of_test_vocabulary(self):
        for word in self.FORBIDDEN:
            self.assertNotIn(word, ti.INTERPRET_SYSTEM_PROMPT)

    def test_grounded_prompt_free_of_test_vocabulary(self):
        for word in self.FORBIDDEN:
            self.assertNotIn(word, gi.GROUNDED_SYSTEM_PROMPT)

    def test_grounded_prompt_contract_requires_top_level_players(self):
        self.assertIn('"players"', gi.GROUNDED_SYSTEM_PROMPT)
        # the top-level players rule must also exist (data-driven, no fabrication)
        self.assertIn("players", gi.GROUNDED_SYSTEM_PROMPT)


class TestGroundedTimeoutDefault(unittest.TestCase):
    def test_timeout_default_is_sixty_seconds(self):
        import importlib
        env_backup = dict(os.environ)
        os.environ.pop("REACT_GROUNDED_TIMEOUT", None)
        try:
            fresh = importlib.reload(gi)
        finally:
            os.environ.clear()
            os.environ.update(env_backup)
        self.assertEqual(fresh.GROUNDED_TIMEOUT, 60)


if __name__ == "__main__":
    unittest.main()
