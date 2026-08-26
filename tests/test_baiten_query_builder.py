"""Tests for the Baiten (CN) search query assembler."""
import unittest

from sources.long_task.search_query_builder import (
    MAX_ASSEMBLED_QUERY_CHARS,
    REWRITE_SYSTEM_PROMPT_CN,
    _assemble_baiten_ladder,
    assemble_baiten_query,
    build_baiten_queries,
    format_ladder_guidance,
    sanitize_baiten_query,
)


class _FakeProvider:
    def __init__(self, response):
        self._response = response

    async def complete_json(self, system, user):
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class TestSanitizeBaitenQuery(unittest.TestCase):
    def test_keeps_cjk(self):
        self.assertEqual(sanitize_baiten_query("散热 装置"), "散热 装置")

    def test_keeps_latin_and_digits(self):
        self.assertEqual(sanitize_baiten_query("AI 芯片 5nm"), "AI 芯片 5nm")

    def test_strips_wildcards_and_control_chars(self):
        self.assertEqual(sanitize_baiten_query("散热* 装置\x00"), "散热 装置")

    def test_caps_length(self):
        long_q = "词" * 500
        self.assertLessEqual(len(sanitize_baiten_query(long_q)), 250)


class TestAssembleBaitenQuery(unittest.TestCase):
    def test_single_field_prefix(self):
        self.assertEqual(
            assemble_baiten_query([["散热", "冷却"], ["风扇"]], "ti"),
            "ti:(散热 OR 冷却) AND ti:(风扇)",
        )

    def test_multiword_phrases_quoted(self):
        self.assertEqual(
            assemble_baiten_query([["heat sink", "散热器"]], "ab"),
            'ab:("heat sink" OR 散热器)',
        )

    def test_skips_empty_groups(self):
        self.assertEqual(
            assemble_baiten_query([[], ["风扇"]], "clm"),
            "clm:(风扇)",
        )

    def test_ladder_strips_wildcards(self):
        # Sanitization happens in the ladder pass (assemble itself is a
        # pure joiner) — wildcard input must never reach the query text.
        ladder = _assemble_baiten_ladder([["cool*", "散热"]])
        self.assertTrue(ladder)
        for q in ladder:
            self.assertNotIn("*", q)
            self.assertIn("cool", q)


class TestAssembleBaitenLadder(unittest.TestCase):
    def test_fields_sweep_tightest_first(self):
        ladder = _assemble_baiten_ladder([["散热", "冷却"], ["风扇"], ["静音"]])
        self.assertTrue(ladder[0].startswith("ti:("))
        self.assertTrue(ladder[1].startswith("ab:("))
        self.assertTrue(ladder[2].startswith("clm:("))
        # Dropping the weakest concept cycles fields again.
        self.assertGreater(len(ladder), 3)

    def test_drops_weakest_concept(self):
        ladder = _assemble_baiten_ladder([["甲"], ["乙"]])
        # Level 4 (after ti/ab/clm full-concept) keeps only the strongest.
        self.assertTrue(any("甲" in q and "乙" not in q for q in ladder[3:]))

    def test_length_budget_respected(self):
        long_groups = [[f"关键词{i}超长扩展词" * 5 for i in range(8)], ["第二组"]]
        for q in _assemble_baiten_ladder(long_groups):
            self.assertLessEqual(len(q), MAX_ASSEMBLED_QUERY_CHARS)

    def test_empty_groups_yield_empty_ladder(self):
        self.assertEqual(_assemble_baiten_ladder([]), [])
        self.assertEqual(_assemble_baiten_ladder([[], []]), [])


class TestBuildBaitenQueries(unittest.TestCase):
    def test_parses_concepts_and_carriers(self):
        provider = _FakeProvider({
            "concepts": [
                {"concept": "散热", "keywords": ["散热", "冷却"],
                 "carriers": ["散热片", "热管"]},
                {"concept": "风扇", "keywords": ["风扇"], "carriers": []},
            ],
        })
        import asyncio
        result = asyncio.run(build_baiten_queries("测试", provider))
        queries = result["queries"]
        self.assertTrue(queries)
        self.assertTrue(any(q.startswith("ti:") for q in queries))
        self.assertTrue(any("散热片" in q for q in queries))  # carrier ladder
        self.assertEqual(len(result["concepts"]), 2)

    def test_failure_returns_empty(self):
        import asyncio
        result = asyncio.run(
            build_baiten_queries("测试", _FakeProvider(RuntimeError("boom"))))
        self.assertEqual(result, {"concepts": [], "queries": []})

    def test_non_dict_result_returns_empty(self):
        import asyncio
        result = asyncio.run(
            build_baiten_queries("测试", _FakeProvider("nope")))
        self.assertEqual(result, {"concepts": [], "queries": []})


class TestPromptGenerality(unittest.TestCase):
    """memory: reject-query-specific-synonym-hardcoding — the CN prompt
    must describe extraction rules only, never example questions or
    concrete technology words from any test query."""

    def test_no_example_technology_words(self):
        for banned in ("散热", "冷却", "风扇", "芯片", "华为", "手机"):
            self.assertNotIn(banned, REWRITE_SYSTEM_PROMPT_CN)

    def test_explains_chinese_first_rule(self):
        self.assertIn("中文为主", REWRITE_SYSTEM_PROMPT_CN)
        self.assertIn("通配符", REWRITE_SYSTEM_PROMPT_CN)

    def test_explains_short_word_rule(self):
        # 0-命中教训（2026-08-26 实测）：专利标题用短词，长表述必 0 命中。
        self.assertIn("2-4 字短词", REWRITE_SYSTEM_PROMPT_CN)
        self.assertIn("完整长表述", REWRITE_SYSTEM_PROMPT_CN)


class TestFormatLadderGuidanceDual(unittest.TestCase):
    def test_cn_section_appended_with_field_semantics(self):
        us = {"concepts": [], "queries": ["ab:(cool OR cooler)"]}
        cn = {"concepts": [], "queries": ["ti:(散热 OR 冷却)"]}
        text = format_ladder_guidance(us, "zh", cn_rewrite=cn)
        self.assertIn("ti=标题", text)
        self.assertIn("ab:(cool OR cooler)", text)
        self.assertIn("ti:(散热 OR 冷却)", text)

    def test_single_source_unchanged(self):
        us = {"concepts": [], "queries": ["ab:(cool)"]}
        text = format_ladder_guidance(us, "zh")
        self.assertNotIn("ti=标题", text)
        self.assertIn("ab:(cool)", text)

    def test_empty_cn_rewrite_returns_us_only(self):
        us = {"concepts": [], "queries": ["ab:(cool)"]}
        text = format_ladder_guidance(us, "zh", cn_rewrite=None)
        self.assertEqual(text.count("ab:(cool)"), 1)

    def test_zh_renders_cn_ladder_first(self):
        us = {"concepts": [], "queries": ["ab:(cool)"]}
        cn = {"concepts": [], "queries": ["ti:(散热)"]}
        text = format_ladder_guidance(us, "zh", cn_rewrite=cn)
        self.assertLess(text.index("ti:(散热)"), text.index("ab:(cool)"))

    def test_en_keeps_us_ladder_first(self):
        us = {"concepts": [], "queries": ["ab:(cool)"]}
        cn = {"concepts": [], "queries": ["ti:(散热)"]}
        text = format_ladder_guidance(us, "en", cn_rewrite=cn)
        self.assertLess(text.index("ab:(cool)"), text.index("ti:(散热)"))


if __name__ == "__main__":
    unittest.main()
