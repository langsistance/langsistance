"""Tests for patent_analyzer module (Task 13, Phase 1 & 2)."""

import unittest
from unittest.mock import AsyncMock, MagicMock

from sources.long_task.patent_analyzer import (
    analyze_single_patent,
    build_failed_row,
    generate_patent_summary,
    generate_table_columns,
)


class TestBuildFailedRow(unittest.TestCase):
    """build_failed_row returns placeholder data with failure marker."""

    def test_returns_failure_row(self):
        row = build_failed_row("CN001", "download failed")
        self.assertEqual(row["patent_id"], "CN001")
        self.assertTrue(row["_failed"])
        self.assertIn("download failed", row["_failure_reason"])


class TestGenerateTableColumns(unittest.IsolatedAsyncioTestCase):
    """generate_table_columns calls LLM and returns list of column names."""

    async def test_returns_columns_from_llm(self):
        mock_provider = MagicMock()
        mock_provider.complete_json = AsyncMock(return_value={
            "columns": ["专利号", "技术领域", "核心技术方案", "创新点"],
        })

        result = await generate_table_columns(
            query="分析技术分布和创新点",
            patent_count=10,
            provider=mock_provider,
        )
        self.assertIsInstance(result, list)
        self.assertIn("专利号", result)
        self.assertGreaterEqual(len(result), 3)


class TestAnalyzeSinglePatent(unittest.IsolatedAsyncioTestCase):
    """analyze_single_patent calls LLM with patent text and columns."""

    async def test_returns_row_with_columns(self):
        mock_provider = MagicMock()
        mock_provider.complete_json = AsyncMock(return_value={
            "patent_id": "CN001",
            "技术领域": "G06V 计算机视觉",
            "核心技术方案": "双流注意力机制",
            "创新点": "小样本学习",
        })

        row = await analyze_single_patent(
            patent_id="CN001",
            patent_text="专利说明书全文...",
            columns=["专利号", "技术领域", "核心技术方案", "创新点"],
            query="分析技术分布",
            provider=mock_provider,
        )
        self.assertEqual(row["patent_id"], "CN001")
        self.assertEqual(row["技术领域"], "G06V 计算机视觉")
        self.assertNotIn("_summary", row)  # summary is separate


class TestGeneratePatentSummary(unittest.IsolatedAsyncioTestCase):
    """generate_patent_summary returns a short text summary."""

    async def test_returns_summary_string(self):
        mock_provider = MagicMock()
        mock_provider.complete_json = AsyncMock(return_value={
            "summary": "该专利提出了一种基于双流注意力的图像识别方法，核心创新在于解决小样本过拟合问题。",
        })

        summary = await generate_patent_summary(
            patent_id="CN001",
            row={"patent_id": "CN001", "技术领域": "G06V"},
            query="分析技术分布",
            provider=mock_provider,
        )
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 10)
