"""Tests for prosecution_analyzer module — question-driven report generation."""

import unittest
from unittest.mock import AsyncMock, MagicMock

from sources.long_task.prosecution_analyzer import (
    generate_executive_summary,
    generate_report_outline,
    generate_report_section,
    build_failed_row,
)


# ── Helper to create a streaming mock provider ─────────────────────────────────

def _make_streaming_provider(chunks: list[str]):
    """Return a mock provider whose _get_langchain_llm().astream() yields chunks."""
    mock_provider = MagicMock()
    mock_llm = MagicMock()

    async def _astream(*args, **kwargs):
        class Chunk:
            def __init__(self, content):
                self.content = content
        for c in chunks:
            yield Chunk(c)

    mock_llm.astream = _astream
    mock_provider._get_langchain_llm.return_value = mock_llm
    return mock_provider


# ── generate_executive_summary ─────────────────────────────────────────────────


class TestGenerateExecutiveSummary(unittest.IsolatedAsyncioTestCase):
    """generate_executive_summary returns a question-driven summary."""

    async def test_returns_summary_text(self):
        provider = _make_streaming_provider([
            "基于用户关于无效性风险的问题，审查过程显示...",
            "因此，该专利存在较高的无效性风险。",
        ])
        result = await generate_executive_summary(
            table_rows=[
                {"文件类型": "Office Action", "拒绝理由": "§103 obviousness",
                 "_summary": "Examiner rejected claims under §103"}
            ],
            columns=["文件类型", "拒绝理由"],
            query="这个专利的无效性风险如何？",
            patent_id="US12345678",
            provider=provider,
            lang="zh",
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)
        self.assertIn("无效性", result)

    async def test_empty_rows_returns_fallback(self):
        result = await generate_executive_summary(
            table_rows=[], columns=[], query="test",
            patent_id="US123", provider=MagicMock(), lang="zh",
        )
        self.assertIn("无分析数据", result)

    async def test_english_output(self):
        provider = _make_streaming_provider([
            "Based on the invalidity risk question, ",
            "the prosecution history shows significant §103 vulnerabilities.",
        ])
        result = await generate_executive_summary(
            table_rows=[
                {"Document Type": "Office Action",
                 "Key Content": "§103 rejection over D1 and D2",
                 "_summary": "Claims rejected as obvious"}
            ],
            columns=["Document Type", "Key Content"],
            query="What is the invalidity risk?",
            patent_id="US123",
            provider=provider,
            lang="en",
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)

    async def test_prompt_contains_question_framework(self):
        """Verify the system prompt includes the 4-step question-driven framework."""
        captured_messages = []

        mock_provider = MagicMock()
        mock_llm = MagicMock()

        async def _astream(messages, **kwargs):
            captured_messages.append(messages)
            class Chunk:
                def __init__(self, content):
                    self.content = content
            yield Chunk("Test summary content.")
            yield Chunk(" More content.")

        mock_llm.astream = _astream
        mock_provider._get_langchain_llm.return_value = mock_llm

        await generate_executive_summary(
            table_rows=[
                {"文件类型": "Office Action", "拒绝理由": "§103",
                 "_summary": "Examiner rejected"}
            ],
            columns=["文件类型", "拒绝理由"],
            query="无效性风险分析",
            patent_id="US123",
            provider=mock_provider,
            lang="zh",
        )

        self.assertTrue(len(captured_messages) > 0, "Expected LLM to be called")
        system_msg = captured_messages[0][0][1]  # (system, content) tuple
        self.assertIn("专利律师", system_msg)
        self.assertIn("审查策略", system_msg)
        self.assertIn("驳回策略", system_msg)
        self.assertIn("授权原因", system_msg)
        self.assertIn("不是按时间顺序总结", system_msg)

    async def test_strips_think_block(self):
        """Output containing </think> should have the think block stripped."""
        provider = _make_streaming_provider([
            "Reasoning here.</think>",
            "Clean summary starts here.",
        ])
        result = await generate_executive_summary(
            table_rows=[
                {"文件类型": "Office Action", "拒绝理由": "§103",
                 "_summary": "Test"}
            ],
            columns=["文件类型", "拒绝理由"],
            query="分析",
            patent_id="US123",
            provider=provider,
            lang="zh",
        )
        self.assertNotIn("Reasoning here", result)
        self.assertIn("Clean summary starts here", result)


# ── generate_report_outline ────────────────────────────────────────────────────


class TestGenerateReportOutline(unittest.IsolatedAsyncioTestCase):
    """generate_report_outline returns a question-driven section plan."""

    async def test_returns_outline_with_sections(self):
        mock_provider = MagicMock()
        mock_provider.complete_json = AsyncMock(return_value={
            "title": "审查历史分析报告",
            "sections": [
                {"heading": "Claim 1 的 §103 驳回分析",
                 "description": "分析审查员基于显而易见性的驳回逻辑"},
                {"heading": "申请人答复策略评估",
                 "description": "评估申请人对 §103 驳回的争辩有效性"},
            ],
        })

        result = await generate_report_outline(
            table_rows=[
                {"文件类型": "Office Action", "拒绝理由": "§103"}
            ],
            columns=["文件类型", "拒绝理由"],
            query="分析 §103 驳回的有效性",
            provider=mock_provider,
            lang="zh",
        )
        self.assertIn("sections", result)
        self.assertGreaterEqual(len(result["sections"]), 1)

    async def test_llm_failure_returns_fallback(self):
        mock_provider = MagicMock()
        mock_provider.complete_json = AsyncMock(side_effect=Exception("API error"))

        result = await generate_report_outline(
            table_rows=[], columns=[], query="test",
            provider=mock_provider, lang="zh",
        )
        self.assertIn("sections", result)
        self.assertGreaterEqual(len(result["sections"]), 2)

    async def test_english_fallback(self):
        mock_provider = MagicMock()
        mock_provider.complete_json = AsyncMock(side_effect=Exception("API error"))

        result = await generate_report_outline(
            table_rows=[], columns=[], query="test",
            provider=mock_provider, lang="en",
        )
        self.assertIn("sections", result)
        self.assertTrue(
            any("Claim Amendment Analysis" in s["heading"] for s in result["sections"])
        )


# ── generate_report_section ────────────────────────────────────────────────────


class TestGenerateReportSection(unittest.IsolatedAsyncioTestCase):
    """generate_report_section writes a question-driven section."""

    async def test_returns_section_text(self):
        provider = _make_streaming_provider([
            "## 分析结果\n\n",
            "审查员基于 §103 驳回了 Claim 1 **[Office Action]**。",
            "该驳回的依据是 D1 与 D2 的组合，但 D2 并未公开关键特征。",
        ])
        result = await generate_report_section(
            section={"heading": "§103 驳回分析", "description": "分析驳回逻辑"},
            query="评估 §103 驳回的有效性",
            table_rows=[
                {"文件类型": "Office Action", "拒绝理由": "§103",
                 "_summary": "Examiner rejected under §103"}
            ],
            columns=["文件类型", "拒绝理由"],
            provider=provider,
            lang="zh",
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)
        self.assertIn("Office Action", result)

    async def test_english_section(self):
        provider = _make_streaming_provider([
            "## Analysis\n\n",
            "The examiner rejected Claim 1 under §103 **[Office Action]**.",
            " However, the combination of D1 and D2 is questionable.",
        ])
        result = await generate_report_section(
            section={"heading": "§103 Analysis", "description": "Analyze rejection"},
            query="Evaluate §103 rejection validity",
            table_rows=[
                {"Document Type": "Office Action",
                 "Key Content": "§103 rejection",
                 "_summary": "Rejection under §103"}
            ],
            columns=["Document Type", "Key Content"],
            provider=provider,
            lang="en",
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)

    async def test_empty_section_heading_handled(self):
        """Empty heading should not crash."""
        provider = _make_streaming_provider(["Content for untitled section."])
        result = await generate_report_section(
            section={},
            query="test query",
            table_rows=[
                {"文件类型": "Office Action", "拒绝理由": "§103",
                 "_summary": "Test"}
            ],
            columns=["文件类型", "拒绝理由"],
            provider=provider,
            lang="zh",
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    async def test_failed_rows_handled(self):
        """Rows marked as _failed should not crash section generation."""
        provider = _make_streaming_provider(["Summary for failed docs."])
        result = await generate_report_section(
            section={"heading": "Summary", "description": "Overview"},
            query="test",
            table_rows=[
                {"文件类型": "Office Action", "_failed": True,
                 "_summary": ""}
            ],
            columns=["文件类型", "拒绝理由"],
            provider=provider,
            lang="zh",
        )
        self.assertIsInstance(result, str)


# ── build_failed_row ───────────────────────────────────────────────────────────


class TestBuildFailedRow(unittest.TestCase):
    """build_failed_row returns a placeholder row for failed documents."""

    def test_returns_row_with_failure_info(self):
        row = build_failed_row("OA001", "text extraction failed",
                               ["文件类型", "文件描述", "核心内容"], "zh")
        self.assertEqual(row["文件类型"], "分析失败")
        for col in ["文件描述", "核心内容"]:
            self.assertIn(col, row)

    def test_english_labels(self):
        row = build_failed_row("OA001", "text extraction failed",
                               ["Document Type", "Description", "Key Content"], "en")
        self.assertEqual(row["Document Type"], "Analysis Failed")
