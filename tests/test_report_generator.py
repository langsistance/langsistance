"""Tests for report_generator module — question-driven report generation."""

import unittest
from unittest.mock import AsyncMock, MagicMock

from sources.long_task.report_generator import (
    generate_executive_summary,
    generate_report_outline,
    generate_report_section,
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
    """generate_executive_summary returns a question-driven executive summary."""

    async def test_returns_summary_text(self):
        provider = _make_streaming_provider([
            "基于用户关于技术趋势的问题，分析发现：\n\n",
            "### 核心发现\n\n",
            "该领域普遍采用深度学习方案 **[CN001]**。",
        ])
        result = await generate_executive_summary(
            table_rows=[
                {"专利号": "CN001", "发明点": "双流注意力机制",
                 "技术方案": "采用双流架构提高效率",
                 "_summary": "该专利提出了双流注意力方案"}
            ],
            columns=["专利号", "发明点", "技术方案"],
            query="分析该领域的技术趋势",
            provider=provider,
            lang="zh",
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)

    async def test_empty_rows_returns_fallback(self):
        result = await generate_executive_summary(
            table_rows=[], columns=[], query="test",
            provider=MagicMock(), lang="zh",
        )
        self.assertIn("无分析数据", result)

    async def test_english_output(self):
        provider = _make_streaming_provider([
            "Based on the technology trend question:\n\n",
            "### Key Findings\n\n",
            "Deep learning dominates **[US18331482]**.",
        ])
        result = await generate_executive_summary(
            table_rows=[
                {"patent_id": "US18331482", "发明点": "DL-based approach",
                 "_summary": "Deep learning innovation"}
            ],
            columns=["patent_id", "发明点"],
            query="technology trend analysis",
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
            yield Chunk("Executive summary content.")
            yield Chunk(" More summary.")

        mock_llm.astream = _astream
        mock_provider._get_langchain_llm.return_value = mock_llm

        await generate_executive_summary(
            table_rows=[
                {"专利号": "CN001", "发明点": "双流注意力",
                 "_summary": "创新方案"}
            ],
            columns=["专利号", "发明点"],
            query="分析技术趋势和创新方向",
            provider=mock_provider,
            lang="zh",
        )

        self.assertTrue(len(captured_messages) > 0, "Expected LLM to be called")
        system_msg = captured_messages[0][0][1]
        self.assertIn("理解问题", system_msg)
        self.assertIn("过滤信息", system_msg)
        self.assertIn("逻辑组织", system_msg)
        self.assertIn("给出建议", system_msg)

    async def test_strips_think_block(self):
        """Output containing </think> should have the think block stripped."""
        provider = _make_streaming_provider([
            "<think>Reasoning...</think>",
            "Clean executive summary here.",
        ])
        result = await generate_executive_summary(
            table_rows=[
                {"专利号": "CN001", "发明点": "Test", "_summary": "Test"}
            ],
            columns=["专利号", "发明点"],
            query="分析",
            provider=provider,
            lang="zh",
        )
        self.assertNotIn("<think>", result)
        self.assertNotIn("Reasoning", result)
        self.assertIn("Clean executive summary here", result)

    async def test_failed_rows_handled(self):
        """Rows with _failed=True should not crash summary generation."""
        provider = _make_streaming_provider(["Summary with failed patents."])
        result = await generate_executive_summary(
            table_rows=[
                {"专利号": "CN001", "_failed": True, "_failure_reason": "download error"}
            ],
            columns=["专利号"],
            query="test",
            provider=provider,
            lang="zh",
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    async def test_handles_exception_gracefully(self):
        """LLM failure should return fallback text, not raise."""
        mock_provider = MagicMock()
        mock_provider._get_langchain_llm.side_effect = Exception("LLM unavailable")

        result = await generate_executive_summary(
            table_rows=[{"专利号": "CN001", "发明点": "Test"}],
            columns=["专利号", "发明点"],
            query="test",
            provider=mock_provider,
            lang="zh",
        )
        self.assertIn("执行摘要生成失败", result)


# ── generate_report_outline ────────────────────────────────────────────────────


class TestGenerateReportOutline(unittest.IsolatedAsyncioTestCase):
    """generate_report_outline returns a question-driven section plan."""

    async def test_returns_outline_with_sections(self):
        mock_provider = MagicMock()
        mock_provider.complete_json = AsyncMock(return_value={
            "title": "技术趋势分析报告",
            "sections": [
                {"heading": "核心技术方向", "description": "该领域主流技术路线"},
                {"heading": "创新点分布", "description": "各专利创新点对比"},
            ],
        })

        result = await generate_report_outline(
            query="分析技术趋势",
            columns=["专利号", "发明点", "技术方案"],
            table_rows=[{"专利号": "CN001", "发明点": "双流注意力"}],
            provider=mock_provider,
            lang="zh",
        )
        self.assertIn("sections", result)
        self.assertGreaterEqual(len(result["sections"]), 1)

    async def test_llm_failure_returns_fallback(self):
        mock_provider = MagicMock()
        mock_provider.complete_json = AsyncMock(side_effect=Exception("API error"))

        result = await generate_report_outline(
            query="test", columns=["专利号"], table_rows=[],
            provider=mock_provider, lang="zh",
        )
        self.assertIn("sections", result)
        self.assertGreaterEqual(len(result["sections"]), 2)

    async def test_english_fallback(self):
        mock_provider = MagicMock()
        mock_provider.complete_json = AsyncMock(side_effect=Exception("API error"))

        result = await generate_report_outline(
            query="test", columns=["Patent ID"], table_rows=[],
            provider=mock_provider, lang="en",
        )
        self.assertIn("sections", result)
        self.assertTrue(
            any("Key Findings" in s["heading"] for s in result["sections"])
        )

    async def test_non_dict_result_returns_fallback(self):
        mock_provider = MagicMock()
        mock_provider.complete_json = AsyncMock(return_value="not a dict")

        result = await generate_report_outline(
            query="test", columns=["专利号"], table_rows=[],
            provider=mock_provider, lang="zh",
        )
        self.assertIn("sections", result)
        self.assertEqual(result["title"], "专利分析报告")


# ── generate_report_section ────────────────────────────────────────────────────


class TestGenerateReportSection(unittest.IsolatedAsyncioTestCase):
    """generate_report_section writes a question-driven section."""

    async def test_returns_section_text(self):
        provider = _make_streaming_provider([
            "## 核心技术方向\n\n",
            "该领域的主流方案是深度学习架构 **[CN001]**。",
            "多个专利采用了注意力机制 **[CN001]** **[CN002]**。",
        ])
        result = await generate_report_section(
            section={"heading": "核心技术方向",
                     "description": "主流技术路线分析"},
            query="分析该领域的技术趋势",
            columns=["专利号", "发明点", "技术方案"],
            table_rows=[
                {"专利号": "CN001", "发明点": "双流注意力",
                 "技术方案": "双流架构"},
                {"专利号": "CN002", "发明点": "Transformer优化",
                 "技术方案": "轻量化Transformer"},
            ],
            provider=provider,
            lang="zh",
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)
        self.assertIn("CN001", result)

    async def test_english_section(self):
        provider = _make_streaming_provider([
            "## Technology Direction\n\n",
            "Deep learning dominates **[US001]**.",
            " Multiple patents use attention **[US001]** **[US002]**.",
        ])
        result = await generate_report_section(
            section={"heading": "Technology Direction",
                     "description": "Mainstream approaches"},
            query="Analyze technology trends",
            columns=["Patent ID", "Innovation", "Technical Solution"],
            table_rows=[
                {"Patent ID": "US001", "Innovation": "Dual attention",
                 "Technical Solution": "Dual stream"},
                {"Patent ID": "US002", "Innovation": "Transformer opt",
                 "Technical Solution": "Lightweight transformer"},
            ],
            provider=provider,
            lang="en",
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)

    async def test_empty_section_heading_handled(self):
        """Empty heading should still produce content."""
        provider = _make_streaming_provider(["Content without heading."])
        result = await generate_report_section(
            section={},
            query="test",
            columns=["专利号"],
            table_rows=[{"专利号": "CN001"}],
            provider=provider,
            lang="zh",
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    async def test_failed_rows_handled(self):
        """Rows with _failed=True should not crash section generation."""
        provider = _make_streaming_provider(["Section covering failed patents."])
        result = await generate_report_section(
            section={"heading": "Overview", "description": "All patents"},
            query="test",
            columns=["专利号", "发明点"],
            table_rows=[
                {"专利号": "CN001", "_failed": True, "_summary": ""},
                {"专利号": "CN002", "发明点": "Valid patent",
                 "_summary": "Valid summary"},
            ],
            provider=provider,
            lang="zh",
        )
        self.assertIsInstance(result, str)

    async def test_strips_think_block(self):
        """Output containing </think> should have the think block stripped."""
        provider = _make_streaming_provider([
            "Pre-reasoning content</think>",
            "Clean section content.",
        ])
        result = await generate_report_section(
            section={"heading": "Test", "description": "Test section"},
            query="test",
            columns=["专利号"],
            table_rows=[{"专利号": "CN001"}],
            provider=provider,
            lang="zh",
        )
        self.assertNotIn("Pre-reasoning", result)
        self.assertIn("Clean section content", result)
