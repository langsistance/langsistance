"""Tests for the architecture-level technical interpretation module."""
import unittest
from unittest import mock

from sources.long_task import technical_interpretation as ti
from sources.long_task.chat_relevance import score_candidates_concurrent
from sources.long_task.relevance_gate import GATE_SYSTEM_PROMPT, score_candidates


class _FakeProvider:
    def __init__(self, result=None, fail=False):
        self.result = result
        self.fail = fail
        self.calls = 0
        self.system_prompts = []

    async def complete_json(self, system, user, max_retries=0):
        self.calls += 1
        self.system_prompts.append(system)
        if self.fail:
            raise RuntimeError("down")
        return self.result


def _valid_raw():
    return {
        "scheme": "Per-channel constant-current control loops",
        "structure_terms": ["error amplifier", "constant current",
                            "reference signal"],
        "independence_terms": ["per-channel", "independently"],
        "scenarios": ["LED backlighting"],
        "main_lines": ["模拟恒流环路", "数字 PWM 驱动"],
        "key_players": ["ERP Power", "Infineon", "TI"],
        "queries": [
            '("error amplifier" AND "RGB")',
            '中文垃圾查询 AND stuff',
        ],
    }


class TestParseInterpretation(unittest.TestCase):
    def test_valid_raw_parses_to_canonical(self):
        parsed = ti.parse_interpretation(_valid_raw())
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["scheme"], "Per-channel constant-current "
                                           "control loops")
        self.assertEqual(parsed["structure_terms"][0], "error amplifier")
        self.assertEqual(parsed["queries"][0],
                         '("error amplifier" AND "RGB")')

    def test_cjk_and_junk_queries_sanitized(self):
        parsed = ti.parse_interpretation(_valid_raw())
        self.assertTrue(parsed)
        self.assertNotIn("中文", parsed["queries"][1])

    def test_non_dict_returns_none(self):
        self.assertIsNone(ti.parse_interpretation("nope"))
        self.assertIsNone(ti.parse_interpretation(None))
        self.assertIsNone(ti.parse_interpretation([]))

    def test_no_scheme_and_no_terms_returns_none(self):
        self.assertIsNone(ti.parse_interpretation(
            {"queries": ['("a" AND "b")']}))
        self.assertIsNone(ti.parse_interpretation({}))

    def test_queries_deduped_and_capped(self):
        raw = _valid_raw()
        raw["queries"] = ["q1", "q1", "q2"] * 10
        parsed = ti.parse_interpretation(raw)
        self.assertEqual(parsed["queries"], ["q1", "q2"])


class TestAndChainExpansion(unittest.TestCase):
    def test_split_and_groups_top_level(self):
        self.assertEqual(
            ti._split_and_groups('("a" OR "b") AND ("c") AND ("d")'),
            ['("a" OR "b")', '("c")', '("d")'])

    def test_split_ignores_and_inside_quotes(self):
        self.assertEqual(
            ti._split_and_groups('"RGB AND LED" AND ("c")'),
            ['"RGB AND LED"', '("c")'])

    def test_split_no_top_level_and(self):
        self.assertEqual(
            ti._split_and_groups('("a" OR "b")'), ['("a" OR "b")'])

    def test_expand_drops_groups_tight_to_loose(self):
        self.assertEqual(
            ti.expand_query_ladder('("a") AND ("b") AND ("c")'),
            ['("a") AND ("b") AND ("c")', '("a") AND ("b")', '("a")'])

    def test_expand_single_group_unchanged(self):
        self.assertEqual(
            ti.expand_query_ladder('("a" OR "b")'), ['("a" OR "b")'])

    def test_expand_empty(self):
        self.assertEqual(ti.expand_query_ladder(""), [])

    def test_merge_expands_interp_chains(self):
        merged = ti.merge_interpretation_queries(
            {"queries": []},
            {"queries": ['("a") AND ("b") AND ("c")']})
        self.assertEqual(
            merged["queries"],
            ['("a") AND ("b") AND ("c")', '("a") AND ("b")', '("a")'])

    def test_merge_chain_dedupes_against_existing(self):
        merged = ti.merge_interpretation_queries(
            {"queries": ['("a")']},
            {"queries": ['("a") AND ("b")']})
        self.assertEqual(merged["queries"],
                         ['("a") AND ("b")', '("a")'])

    def test_merge_chain_capped_to_slots_and_rewrite_kept(self):
        interp = {"queries": [f'("a{i}") AND ("b{i}") AND ("c{i}")'
                              for i in range(5)]}
        rewrite = {"queries": ["tail1", "tail2", "tail3", "tail4"]}
        merged = ti.merge_interpretation_queries(rewrite, interp)
        # chain capped at MAX_INTERP_LADDER_SLOTS=3, rewrite tail follows
        self.assertEqual(merged["queries"][:3],
                         ['("a0") AND ("b0") AND ("c0")',
                          '("a0") AND ("b0")',
                          '("a0")'])
        self.assertIn("tail1", merged["queries"])
        self.assertIn("tail3", merged["queries"])  # rewrite tail kept
        self.assertNotIn("tail4", merged["queries"])  # beyond cap=6
        self.assertEqual(len(merged["queries"]), ti.MAX_LADDER_QUERIES)


class TestMergeInterpretationQueries(unittest.TestCase):
    def test_prepends_and_keeps_original(self):
        rewrite = {"concepts": [], "queries": ["old1", "old2"]}
        interp = {"queries": ["new1", "new2"]}
        merged = ti.merge_interpretation_queries(rewrite, interp)
        self.assertEqual(merged["queries"],
                         ["new1", "new2", "old1", "old2"])
        # input untouched
        self.assertEqual(rewrite["queries"], ["old1", "old2"])

    def test_dedupes_against_existing(self):
        rewrite = {"queries": ["dup", "old"]}
        merged = ti.merge_interpretation_queries(rewrite, {"queries": ["dup"]})
        self.assertEqual(merged["queries"], ["dup", "old"])

    def test_caps_ladder_length(self):
        rewrite = {"queries": [f"old{i}" for i in range(6)]}
        interp = {"queries": [f"new{i}" for i in range(6)]}
        merged = ti.merge_interpretation_queries(rewrite, interp, cap=6)
        self.assertEqual(len(merged["queries"]), 6)
        self.assertEqual(merged["queries"][0], "new0")

    def test_none_interp_returns_copy(self):
        rewrite = {"queries": ["a", "b"]}
        merged = ti.merge_interpretation_queries(rewrite, None)
        self.assertEqual(merged["queries"], ["a", "b"])
        self.assertIsNot(merged, rewrite)


class TestFormatRubric(unittest.TestCase):
    def test_none_returns_empty(self):
        self.assertEqual(ti.format_interpretation_rubric(None), "")
        self.assertEqual(ti.format_interpretation_rubric({}), "")

    def test_contains_scheme_and_terms(self):
        rubric = ti.format_interpretation_rubric(_valid_raw())
        self.assertIn("Per-channel constant-current", rubric)
        self.assertIn("error amplifier", rubric)
        self.assertIn("per-channel", rubric)

    def test_contains_main_lines_and_players(self):
        rubric = ti.format_interpretation_rubric(_valid_raw())
        self.assertIn("模拟恒流环路", rubric)
        self.assertIn("ERP Power", rubric)

    def test_players_are_weak_signal_only(self):
        rubric = ti.format_interpretation_rubric(_valid_raw())
        self.assertIn("本身不构成相关性依据", rubric)

    def test_main_lines_and_players_parsed(self):
        parsed = ti.parse_interpretation(_valid_raw())
        self.assertEqual(parsed["main_lines"], ["模拟恒流环路", "数字 PWM 驱动"])
        self.assertEqual(parsed["key_players"], ["ERP Power", "Infineon", "TI"])

    def test_missing_main_lines_and_players_ok(self):
        raw = _valid_raw()
        raw.pop("main_lines")
        raw.pop("key_players")
        parsed = ti.parse_interpretation(raw)
        self.assertEqual(parsed["main_lines"], [])
        self.assertEqual(parsed["key_players"], [])


class TestInterpretQuery(unittest.IsolatedAsyncioTestCase):
    async def test_returns_parsed_interpretation(self):
        provider = _FakeProvider(result=_valid_raw())
        with mock.patch.object(ti, "_interpret_provider",
                               return_value=provider):
            parsed = await ti.interpret_query("问题")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["queries"][0],
                         '("error amplifier" AND "RGB")')

    async def test_failure_returns_none(self):
        provider = _FakeProvider(result=_valid_raw(), fail=True)
        with mock.patch.object(ti, "_interpret_provider",
                               return_value=provider):
            self.assertIsNone(await ti.interpret_query("问题"))

    async def test_disabled_returns_none_without_calling(self):
        provider = _FakeProvider(result=_valid_raw())
        with mock.patch.object(ti, "INTERPRET_ENABLED", False):
            with mock.patch.object(ti, "_interpret_provider",
                                   return_value=provider):
                self.assertIsNone(await ti.interpret_query("问题"))
        self.assertEqual(provider.calls, 0)

    async def test_cpc_hints_forwarded_in_payload(self):
        provider = _FakeProvider(result=_valid_raw())
        with mock.patch.object(ti, "_interpret_provider",
                               return_value=provider):
            await ti.interpret_query(
                "问题", cpc_hints=[{"code": "H05B45/20", "title": "Colour"}])
        self.assertEqual(provider.calls, 1)

    async def test_hanging_provider_times_out_to_none(self):
        import asyncio

        class _SlowProvider:
            async def complete_json(self, system, user, max_retries=0):
                await asyncio.sleep(30)  # far beyond the patched timeout
                return _valid_raw()

        with mock.patch.object(ti, "INTERPRET_TIMEOUT", 0.1):
            with mock.patch.object(ti, "_interpret_provider",
                                   return_value=_SlowProvider()):
                self.assertIsNone(await ti.interpret_query("问题"))

    def test_env_int_falls_back_on_garbage(self):
        with mock.patch.dict(
                "os.environ", {"REACT_INTERPRET_TIMEOUT": "junk"}):
            self.assertEqual(ti._env_int("REACT_INTERPRET_TIMEOUT", 45), 45)
        with mock.patch.dict(
                "os.environ", {"REACT_INTERPRET_TIMEOUT": "12"}):
            self.assertEqual(ti._env_int("REACT_INTERPRET_TIMEOUT", 45), 12)


class TestRubricInScoring(unittest.IsolatedAsyncioTestCase):
    def _candidate(self, pid, title):
        return {"patent_id": pid, "title": title, "applicant": "ACME",
                "status": "Patented Case", "patent_number": "12" + pid}

    async def test_score_candidates_appends_rubric(self):
        provider = _FakeProvider(result={"scores": []})
        await score_candidates([self._candidate("345", "LED driver")],
                               "问题", provider,
                               rubric=ti.format_interpretation_rubric(
                                   _valid_raw()))
        self.assertEqual(provider.calls, 1)
        self.assertIn("评分补充", provider.system_prompts[0])
        self.assertIn("error amplifier", provider.system_prompts[0])
        self.assertTrue(provider.system_prompts[0].startswith(
            GATE_SYSTEM_PROMPT))

    async def test_score_candidates_without_rubric_keeps_plain_prompt(self):
        provider = _FakeProvider(result={"scores": []})
        await score_candidates([self._candidate("345", "LED driver")],
                               "问题", provider)
        self.assertEqual(provider.system_prompts[0], GATE_SYSTEM_PROMPT)

    async def test_concurrent_passthrough_rubric(self):
        provider = _FakeProvider(result={"scores": []})
        await score_candidates_concurrent(
            [self._candidate("345", "LED driver")],
            "问题", provider, rubric="评分补充：测试")
        self.assertIn("评分补充：测试", provider.system_prompts[0])


class TestParseDimensions(unittest.TestCase):
    def test_dimensions_parsed_and_capped_at_three(self):
        raw = _valid_raw()
        raw["dimensions"] = [
            {"name": "d1", "role": "核心器件/电路层", "terms": ["a"],
             "queries": ['("a")']},
            {"name": "d2", "role": "控制算法/电路层", "terms": [], "queries": []},
            {"name": "d3", "role": "场景应用层", "terms": ["b"], "queries": []},
            {"name": "d4", "role": "多余层", "terms": ["c"], "queries": []},
        ]
        parsed = ti.parse_interpretation(raw)
        self.assertEqual(len(parsed["dimensions"]), ti.MAX_DIMENSIONS)
        self.assertEqual(parsed["dimensions"][0]["name"], "d1")

    def test_dimension_role_deduped_and_empties_dropped(self):
        raw = _valid_raw()
        raw["dimensions"] = [
            {"name": "d1", "role": "核心层", "terms": ["a"]},
            {"name": "d2", "role": "核心层", "terms": ["b"]},   # dup role
            {"name": "", "role": "", "terms": []},               # empty
            {"name": "d3", "role": "场景层", "terms": ["c"]},
        ]
        parsed = ti.parse_interpretation(raw)
        self.assertEqual([d["name"] for d in parsed["dimensions"]], ["d1", "d3"])

    def test_dimension_queries_sanitized(self):
        raw = _valid_raw()
        raw["dimensions"] = [
            {"name": "d1", "role": "核心层",
             "queries": ["中文垃圾 AND stuff", '("a")']},
        ]
        parsed = ti.parse_interpretation(raw)
        self.assertNotIn("中文", parsed["dimensions"][0]["queries"][0])
        self.assertEqual(parsed["dimensions"][0]["queries"], ['("a")'])

    def test_no_dimensions_key_returns_empty_list(self):
        parsed = ti.parse_interpretation(_valid_raw())
        self.assertEqual(parsed["dimensions"], [])


class TestGroundedRubric(unittest.TestCase):
    """format_interpretation_rubric 的接地分支：数据驱动玩家强信号。"""

    def _grounded(self):
        return {
            "scheme": "多通道恒流驱动",
            "structure_terms": ["error amplifier"],
            "dimensions": [
                {"name": "驱动核心", "role": "核心器件/电路层",
                 "line": "逐通道独立闭环恒流", "representatives": ["ERP Power"]},
            ],
            "players": ["ERP Power", "Samsung"],
        }

    def test_grounded_branch_renders_dimensions_and_strong_players(self):
        rubric = ti.format_interpretation_rubric(self._grounded())
        self.assertIn("驱动核心", rubric)
        self.assertIn("逐通道独立闭环恒流", rubric)
        self.assertIn("ERP Power", rubric)
        self.assertIn("真实玩家榜", rubric)
        self.assertIn("评分可上调 3-5 分", rubric)

    def test_grounded_branch_never_uses_weak_signal_wording(self):
        rubric = ti.format_interpretation_rubric(self._grounded())
        self.assertNotIn("本身不构成相关性依据", rubric)

    def test_pre_branch_keeps_weak_signal_wording(self):
        rubric = ti.format_interpretation_rubric(_valid_raw())
        self.assertIn("本身不构成相关性依据", rubric)

    def test_pre_dimensions_without_lines_stay_in_pre_branch(self):
        raw = _valid_raw()
        raw["dimensions"] = [{"name": "d1", "role": "核心层", "terms": ["a"]}]
        rubric = ti.format_interpretation_rubric(raw)
        self.assertIn("本身不构成相关性依据", rubric)

    def test_grounded_empty_returns_empty(self):
        self.assertEqual(ti.format_interpretation_rubric(
            {"players": [], "dimensions": []}), "")


if __name__ == "__main__":
    unittest.main()
