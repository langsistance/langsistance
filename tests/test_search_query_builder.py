"""Tests for search_query_builder — USPTO query rewriting helpers."""
import unittest

from sources.long_task.search_query_builder import (
    assemble_query,
    build_search_queries,
    sanitize_uspto_query,
)


class _FakeProvider:
    def __init__(self, response):
        self._response = response
        self.calls = []

    async def complete_json(self, system, user):
        self.calls.append((system, user))
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class TestAssembleQuery(unittest.TestCase):
    def test_single_concept_quotes_multiword_terms(self):
        self.assertEqual(
            assemble_query([["compressed air dryer", "air dryer", "desiccant"]]),
            '("compressed air dryer" OR "air dryer" OR desiccant)',
        )

    def test_multi_concept_joins_with_and(self):
        self.assertEqual(
            assemble_query([
                ["compressed air dryer", "desiccant dryer"],
                ["humidity control", "dew point", "dehumidif*"],
            ]),
            '("compressed air dryer" OR "desiccant dryer")'
            ' AND ("humidity control" OR "dew point" OR dehumidif*)',
        )

    def test_skips_empty_groups(self):
        self.assertEqual(
            assemble_query([[], ["dew point", "humidity"]]),
            '("dew point" OR humidity)',
        )


class TestSanitizeUsptoQuery(unittest.TestCase):
    def test_removes_cjk_characters(self):
        self.assertEqual(
            sanitize_uspto_query('"compressed air dryer" AND 湿度控制'),
            '"compressed air dryer" AND',
        )

    def test_strips_fullwidth_punctuation_and_kana(self):
        self.assertEqual(
            sanitize_uspto_query('("air dryer" OR desiccant) AND 湿度，。'),
            '("air dryer" OR desiccant) AND',
        )
        self.assertEqual(
            sanitize_uspto_query('ドライヤー air dryer'),
            'air dryer',
        )

    def test_caps_length(self):
        long_q = " AND ".join([f'term{i}' for i in range(50)])
        result = sanitize_uspto_query(long_q)
        self.assertLessEqual(len(result), 250)

    def test_empty_input(self):
        self.assertEqual(sanitize_uspto_query(""), "")


class TestBuildSearchQueries(unittest.IsolatedAsyncioTestCase):
    async def test_returns_validated_queries(self):
        provider = _FakeProvider({
            "concepts": [
                {"concept": "干燥空气源", "keywords": ["air dryer", "desiccant dryer"]},
            ],
            "queries": [
                '("compressed air dryer" OR "air dryer") AND 湿度',
                '"desiccant dryer"',
            ],
        })
        result = await build_search_queries("工业在线干燥空气源", provider)
        self.assertEqual(len(result["queries"]), 2)
        self.assertEqual(
            result["queries"][0],
            '("compressed air dryer" OR "air dryer") AND',
        )

    async def test_provider_failure_returns_empty(self):
        provider = _FakeProvider(RuntimeError("boom"))
        result = await build_search_queries("任意查询", provider)
        self.assertEqual(result, {"concepts": [], "queries": []})

    async def test_garbage_response_returns_empty_queries(self):
        provider = _FakeProvider({"queries": [123, None, "   "]})
        result = await build_search_queries("任意查询", provider)
        self.assertEqual(result["queries"], [])


# ── Ladder ordering + guidance formatting ────────────────────────────────────

from sources.long_task.search_query_builder import (
    REWRITE_SYSTEM_PROMPT,
    format_ladder_guidance,
)


class TestRewritePromptLadderRules(unittest.TestCase):
    def test_prompt_requires_tight_to_loose_ordering(self):
        self.assertIn("最紧", REWRITE_SYSTEM_PROMPT)
        self.assertIn("放宽", REWRITE_SYSTEM_PROMPT)

    def test_prompt_allows_justified_domain_constraints(self):
        self.assertIn("限定", REWRITE_SYSTEM_PROMPT)


class TestFormatLadderGuidance(unittest.TestCase):
    def test_renders_ordered_queries_with_lang_zh(self):
        rewrite = {"queries": [
            '("a" OR "b") AND ("c" OR "d") AND (x OR y)',
            '("a" OR "b") AND ("c" OR "d")',
            '("a" OR "b")',
        ]}
        text = format_ladder_guidance(rewrite, "zh")
        self.assertIn("由紧到松", text)
        # Each query renders on its own numbered line; anchor the find with
        # BOTH the line number and the full per-line content so a looser
        # query's substring inside a tighter line cannot mask ordering.
        pos_0 = text.find('1. ("a" OR "b") AND ("c" OR "d") AND (x OR y)')
        pos_1 = text.find('2. ("a" OR "b") AND ("c" OR "d")')
        pos_2 = text.find('3. ("a" OR "b")')
        self.assertLess(pos_0, pos_1)
        self.assertLess(pos_1, pos_2)

    def test_empty_rewrite_returns_empty(self):
        self.assertEqual(format_ladder_guidance({}, "zh"), "")
        self.assertEqual(format_ladder_guidance({"queries": []}, "zh"), "")

    def test_english_variant(self):
        text = format_ladder_guidance(
            {"queries": ['"a" AND "b"']}, "en")
        self.assertIn("tightest", text)


if __name__ == "__main__":
    unittest.main()


class TestRewritePromptWildcardRules(unittest.TestCase):
    def test_prompt_documents_wildcard_usage(self):
        self.assertIn("通配符", REWRITE_SYSTEM_PROMPT)
        self.assertIn("词尾", REWRITE_SYSTEM_PROMPT)

    def test_prompt_forbids_wildcard_inside_quoted_phrases(self):
        self.assertIn("引号", REWRITE_SYSTEM_PROMPT)


class TestRewritePromptLadderDepth(unittest.TestCase):
    def test_prompt_requires_loosest_level_single_concept(self):
        self.assertIn("只含一个核心概念", REWRITE_SYSTEM_PROMPT)
        self.assertIn("单组", REWRITE_SYSTEM_PROMPT)


class TestRewritePromptWildcardClarity(unittest.TestCase):
    def test_prompt_requires_clarity_judgment(self):
        self.assertIn("明确程度", REWRITE_SYSTEM_PROMPT)
        self.assertIn("一般性技术概念", REWRITE_SYSTEM_PROMPT)
        self.assertIn("判断不清", REWRITE_SYSTEM_PROMPT)

    def test_prompt_wildcard_examples_are_cross_domain(self):
        self.assertIn("filter*", REWRITE_SYSTEM_PROMPT)
        self.assertIn("cataly*", REWRITE_SYSTEM_PROMPT)

    def test_prompt_has_no_single_domain_anchor(self):
        self.assertNotIn("air dry", REWRITE_SYSTEM_PROMPT)
        self.assertNotIn("dehumidif", REWRITE_SYSTEM_PROMPT)


class TestLadderGuidanceWildcardRetry(unittest.TestCase):
    def test_guidance_suggests_wildcard_retry_on_false_zero_zh(self):
        text = format_ladder_guidance({"queries": ['("a" OR "b") AND ("c" OR "d")']}, "zh")
        self.assertIn("假性零命中", text)
        self.assertIn("通配符", text)

    def test_guidance_suggests_wildcard_retry_on_false_zero_en(self):
        text = format_ladder_guidance({"queries": ['("a" OR "b") AND ("c" OR "d")']}, "en")
        self.assertIn("false zero", text)
        self.assertIn("wildcard", text)
