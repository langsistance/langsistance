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
    async def test_assembles_ladder_from_concepts(self):
        # LLM-written queries are ignored: the ladder is assembled
        # deterministically from the concept keyword lists so no variant
        # can be dropped by the model.
        provider = _FakeProvider({
            "concepts": [
                {"concept": "干燥空气源",
                 "keywords": ["dry air supply", "air dryer", "dry* air"]},
                {"concept": "湿度控制",
                 "keywords": ["humidity control", "moisture control"]},
            ],
            "queries": ["ignored llm written query"],
        })
        result = await build_search_queries("工业在线干燥空气源", provider)
        self.assertEqual(result["queries"], [
            '("dry air supply" OR "air dryer" OR dry* air)'
            ' AND ("humidity control" OR "moisture control")',
            '("dry air supply" OR "air dryer" OR dry* air)',
        ])

    async def test_keyword_cap_keeps_queries_under_length_limit(self):
        provider = _FakeProvider({
            "concepts": [
                {"concept": f"c{i}",
                 "keywords": [f"very long keyword phrase number {i}-{j}"
                              for j in range(8)]}
                for i in range(3)
            ],
        })
        result = await build_search_queries("q", provider)
        self.assertEqual(len(result["queries"]), 3)
        for q in result["queries"]:
            self.assertLessEqual(len(q), 250)

    async def test_llm_queries_used_when_concepts_missing(self):
        provider = _FakeProvider({
            "concepts": [],
            "queries": ['"fallback" AND query'],
        })
        result = await build_search_queries("q", provider)
        self.assertEqual(result["queries"], ['"fallback" AND query'])

    async def test_single_concept_gives_single_query(self):
        provider = _FakeProvider({
            "concepts": [{"concept": "c", "keywords": ["k1", "k2"]}],
        })
        result = await build_search_queries("q", provider)
        self.assertEqual(result["queries"], ["(k1 OR k2)"])

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
    def test_prompt_orders_concepts_and_keywords_by_importance(self):
        self.assertIn("按重要性排序", REWRITE_SYSTEM_PROMPT)

    def test_prompt_says_code_assembles_the_ladder(self):
        self.assertIn("由紧到松", REWRITE_SYSTEM_PROMPT)
        self.assertIn("代码", REWRITE_SYSTEM_PROMPT)

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

    def test_prompt_warns_wildcards_lose_phrase_semantics(self):
        self.assertIn("单词", REWRITE_SYSTEM_PROMPT)


class TestLadderGuidanceConceptBank(unittest.TestCase):
    def test_guidance_lists_concept_keywords_for_substitution(self):
        rewrite = {
            "concepts": [
                {"concept": "干燥空气源",
                 "keywords": ["dry air supply", "air dryer", "dry* air"]},
                {"concept": "湿度控制",
                 "keywords": ["humidity control"]},
            ],
            "queries": ['("dry air supply" OR "air dryer" OR dry* air)'
                        ' AND ("humidity control")'],
        }
        text = format_ladder_guidance(rewrite, "zh")
        # the full keyword bank is rendered for synonym substitution
        self.assertIn("干燥空气源", text)
        self.assertIn("air dryer", text)
        self.assertIn("dry* air", text)
        self.assertIn("湿度控制", text)


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


class TestRewritePromptVariantCoverage(unittest.TestCase):
    def test_prompt_requires_at_least_five_variants_per_concept(self):
        self.assertIn("至少 5 个", REWRITE_SYSTEM_PROMPT)

    def test_prompt_requires_word_form_coverage(self):
        self.assertIn("词形", REWRITE_SYSTEM_PROMPT)
        self.assertIn("cool", REWRITE_SYSTEM_PROMPT)


class TestLadderGuidanceLowHitSubstitution(unittest.TestCase):
    def test_guidance_substitutes_before_loosening_zh(self):
        text = format_ladder_guidance({"queries": ['("a" OR "b")']}, "zh")
        self.assertIn("少于 10", text)
        self.assertIn("同义表述", text)
        self.assertIn("10-300", text)

    def test_guidance_substitutes_before_loosening_en(self):
        text = format_ladder_guidance({"queries": ['("a" OR "b")']}, "en")
        self.assertIn("fewer than 10", text)
        self.assertIn("synonym", text)
        self.assertIn("10-300", text)


# ── Title feedback ───────────────────────────────────────────────────────────

from sources.long_task.search_query_builder import (
    FEEDBACK_SYSTEM_PROMPT,
    build_feedback_queries,
)


class TestBuildFeedbackQueries(unittest.IsolatedAsyncioTestCase):
    async def test_returns_sanitized_queries(self):
        provider = _FakeProvider({
            "queries": ['"air dryer" AND humidity', 42, "   ", None],
        })
        out = await build_feedback_queries(
            "干燥空气", ["AIR DRYER CONTROL USING HUMIDITY"], provider)
        self.assertEqual(out, ['"air dryer" AND humidity'])

    async def test_empty_titles_returns_empty_without_calling(self):
        provider = _FakeProvider({"queries": ["q"]})
        out = await build_feedback_queries("干燥空气", [], provider)
        self.assertEqual(out, [])
        self.assertEqual(provider.calls, [])

    async def test_provider_failure_returns_empty(self):
        provider = _FakeProvider(RuntimeError("boom"))
        out = await build_feedback_queries("干燥空气", ["t1"], provider)
        self.assertEqual(out, [])


class TestFeedbackPromptDomainNeutral(unittest.TestCase):
    def test_feedback_prompt_has_no_domain_anchor(self):
        self.assertNotIn("air dry", FEEDBACK_SYSTEM_PROMPT)
        self.assertNotIn("dehumidif", FEEDBACK_SYSTEM_PROMPT)

    def test_feedback_prompt_mentions_word_forms(self):
        self.assertIn("词形", FEEDBACK_SYSTEM_PROMPT)
