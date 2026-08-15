"""Tests for patent_distill — spec distillation into loop observations."""
import unittest

from sources.long_task.patent_distill import (
    SPEC_FALLBACK_LIMIT,
    distill_patent_spec,
    format_distilled,
    truncated_fallback,
)


class _FakeProvider:
    def __init__(self, response):
        self._response = response
        self.calls = []

    async def complete_json(self, system, user):
        self.calls.append(user)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class TestTruncatedFallback(unittest.TestCase):
    def test_caps_at_16000(self):
        text = truncated_fallback("x" * 50000)
        self.assertEqual(len(text), SPEC_FALLBACK_LIMIT)

    def test_short_text_untouched(self):
        self.assertEqual(truncated_fallback("short"), "short")


class TestFormatDistilled(unittest.TestCase):
    def test_renders_all_sections(self):
        text = format_distilled({
            "发明点": "A", "解决的技术问题": "B", "技术方案": "C",
            "权利要求要点": "D", "与用户问题的相关性": "E",
        })
        for label in ("发明点", "解决的技术问题", "技术方案",
                      "权利要求要点", "与用户问题的相关性"):
            self.assertIn(label, text)

    def test_missing_keys_skipped(self):
        text = format_distilled({"发明点": "A"})
        self.assertIn("发明点", text)
        self.assertNotIn("技术方案", text)

    def test_empty_distilled_returns_empty(self):
        self.assertEqual(format_distilled({}), "")


class TestDistillPatentSpec(unittest.IsolatedAsyncioTestCase):
    async def test_returns_structured_result(self):
        provider = _FakeProvider({
            "发明点": "x", "解决的技术问题": "y", "技术方案": "z",
        })
        result = await distill_patent_spec("SPEC TEXT", "查询", provider)
        self.assertEqual(result["发明点"], "x")

    async def test_provider_failure_returns_empty(self):
        result = await distill_patent_spec(
            "SPEC TEXT", "查询", _FakeProvider(RuntimeError("down")))
        self.assertEqual(result, {})

    async def test_garbage_response_returns_empty(self):
        result = await distill_patent_spec("SPEC TEXT", "查询", _FakeProvider("junk"))
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
