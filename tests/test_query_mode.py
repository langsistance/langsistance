"""Tests for the query-mode classifier (sources/long_task/query_mode.py).

Structured/analytical requests skip the CPC match and the architecture
interpretation; semantic technology searches keep them.  Any classifier
failure must fall back to "semantic" so tech searches keep the full
pipeline.
"""
import unittest
from unittest.mock import AsyncMock, patch


class TestQueryModeClassifier(unittest.TestCase):

    def _provider(self, result):
        provider = AsyncMock()
        provider.complete_json = AsyncMock(return_value=result)
        return provider

    def test_structured_mode_passthrough(self):
        from sources.long_task.query_mode import classify_query_mode
        result = _run(classify_query_mode(
            "I want all patent documents for application 18893954",
            self._provider({"mode": "structured"})))
        self.assertEqual(result, "structured")

    def test_semantic_mode_passthrough(self):
        from sources.long_task.query_mode import classify_query_mode
        result = _run(classify_query_mode(
            "保持温度稳定的装置", self._provider({"mode": "semantic"})))
        self.assertEqual(result, "semantic")

    def test_accepts_raw_json_string_output(self):
        from sources.long_task.query_mode import classify_query_mode
        result = _run(classify_query_mode(
            "prosecution history analysis of 18/893,954",
            self._provider('{"mode": "structured"}')))
        self.assertEqual(result, "structured")

    def test_unknown_mode_defaults_to_semantic(self):
        from sources.long_task.query_mode import classify_query_mode
        result = _run(classify_query_mode(
            "anything", self._provider({"mode": "other"})))
        self.assertEqual(result, "semantic")

    def test_provider_failure_defaults_to_semantic(self):
        from sources.long_task.query_mode import classify_query_mode
        provider = AsyncMock()
        provider.complete_json = AsyncMock(side_effect=RuntimeError("boom"))
        result = _run(classify_query_mode("anything", provider))
        self.assertEqual(result, "semantic")

    def test_disabled_returns_semantic_without_calling_provider(self):
        from sources.long_task import query_mode
        provider = self._provider({"mode": "structured"})
        with patch("sources.long_task.query_mode.QUERY_MODE_ENABLED", False):
            result = _run(query_mode.classify_query_mode("anything", provider))
        self.assertEqual(result, "semantic")
        provider.complete_json.assert_not_called()

    def test_empty_query_returns_semantic(self):
        from sources.long_task.query_mode import classify_query_mode
        provider = self._provider({"mode": "structured"})
        result = _run(classify_query_mode("  ", provider))
        self.assertEqual(result, "semantic")
        provider.complete_json.assert_not_called()


def _run(coro):
    import asyncio
    return asyncio.run(coro)


if __name__ == "__main__":
    unittest.main()
