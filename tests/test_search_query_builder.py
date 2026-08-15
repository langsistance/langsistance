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


if __name__ == "__main__":
    unittest.main()
