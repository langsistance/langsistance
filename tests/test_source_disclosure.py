# -*- coding: utf-8 -*-
"""数据供应商名称泄露防护。

**背景（2026-09-13 生产实证）**：系统提示词里的中文检索式引导语写着
「针对用户问题可用的佰腾（中国专利）检索式（由紧到松排列）」，模型照抄
进了可见回答——「本次检索同时覆盖 美国专利（USPTO）与中国专利（佰腾）」。

供应商名称是商务信息，**绝不允许出现在 LLM 可见面**。本测试把「LLM
可见面」当契约来锁：系统提示词片段、工具描述与参数 schema、observation
渲染结果，一律不得出现供应商名。

注意：USPTO 是美国政府登记机构（等同公开信息），不在禁列；只有商业
数据供应商名才禁止。
"""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sources.agents import react_tools
from sources.long_task.search_query_builder import format_ladder_guidance
from sources.patent_id_translator import _decide
from sources.patent_number_parser import format_number_guidance

# 商业数据供应商名称 —— 一旦进入 LLM 可见面，模型会在回答里写出来。
BANNED_SOURCE_NAMES = ("佰腾", "Baiten", "baiten", "BAITEN")


class _FakeAgent:
    def __init__(self):
        self._pending_raw_items = None
        self._last_user_prompt = "test"
        self._last_user_id = "u1"
        self._last_query_id = "q1"
        self._lang = "zh"
        self._search_rewrite = None
        self._search_rewrite_cn = None
        self._tried_queries = []


def _assert_clean(case: unittest.TestCase, text: str, where: str) -> None:
    for term in BANNED_SOURCE_NAMES:
        case.assertNotIn(term, text, f"{where}: 出现供应商名 {term!r}")


class TestLadderGuidanceDisclosure(unittest.TestCase):
    _REWRITE = {"queries": ['"humidity sensor"', "humidity AND sensor"]}

    def test_zh_ladder_guidance(self):
        text = format_ladder_guidance(
            self._REWRITE, "zh", cn_rewrite=self._REWRITE)
        self.assertTrue(text)
        _assert_clean(self, text, "format_ladder_guidance/zh")

    def test_en_ladder_guidance(self):
        text = format_ladder_guidance(
            self._REWRITE, "en", cn_rewrite=self._REWRITE)
        self.assertTrue(text)
        _assert_clean(self, text, "format_ladder_guidance/en")

    def test_cn_only_ladder_guidance(self):
        text = format_ladder_guidance(self._REWRITE, "zh",
                                      cn_rewrite=self._REWRITE)
        _assert_clean(self, text, "cn-only/zh")


class TestNumberGuidanceDisclosure(unittest.TestCase):
    _CANDIDATES = [{"display": "CN116570413A", "country": "CN",
                    "id_type": "publication", "confidence": "high",
                    "reason": "中国公开/公告号格式（CN 前缀）"}]

    def test_zh_number_guidance(self):
        text = format_number_guidance(self._CANDIDATES, "zh")
        self.assertTrue(text)
        _assert_clean(self, text, "format_number_guidance/zh")

    def test_en_number_guidance(self):
        text = format_number_guidance(self._CANDIDATES, "en")
        self.assertTrue(text)
        _assert_clean(self, text, "format_number_guidance/en")


class TestToolSurfaceDisclosure(unittest.TestCase):
    """工具描述与参数 schema 都是直接进 LLM 上下文的。"""

    async def _build(self, patent_source):
        with patch("sources.agents.react_tools.get_knowledge_tool_candidates",
                   return_value=[]):
            return await react_tools.build_tool_set(
                _FakeAgent(), "u1", "q", None, patent_source=patent_source)

    def test_all_modes_tool_descriptions(self):
        for src in ("dual", "cn", "uspto"):
            _registry, tools = asyncio.run(self._build(src))
            for t in tools:
                blob = repr(t.get("function") or t)
                _assert_clean(self, blob, f"tool description ({src})")

    def test_args_schema_field_descriptions(self):
        for schema in (react_tools._DualPatentSearchArgs,
                       react_tools._CnPatentSearchArgs,
                       react_tools._NumberResolveArgs,
                       react_tools._LegalStatusArgs):
            _assert_clean(self, repr(schema.model_json_schema()),
                          f"args schema {schema.__name__}")


class TestLegalStatusObservationDisclosure(unittest.TestCase):
    """法律状态 observation 里会渲染「已核验的数据源」。

    夹具**必须走真实映射函数**（`_legal_status_entries`），否则测试里的
    中立假值会掩盖生产代码里的供应商名——第一版就这么漏过去了。
    """

    _ITEM = {"source": "baiten", "patent_id": "CN116570413A",
             "app_num": "CN202310123456.7", "status": ""}

    def _entries(self, **overrides):
        item = dict(self._ITEM, **overrides)
        return react_tools._legal_status_entries([item], [])

    def test_uncovered_block_zh(self):
        text = react_tools._legal_status_digest(self._entries(), "zh")
        self.assertIn("未获取到", text)      # 确保真的渲染到了缺口分支
        _assert_clean(self, text, "legal_status digest/zh")

    def test_uncovered_block_en(self):
        text = react_tools._legal_status_digest(self._entries(), "en")
        _assert_clean(self, text, "legal_status digest/en")

    def test_covered_block_zh(self):
        entries = self._entries(status="专利权终止")
        text = react_tools._legal_status_digest(entries, "zh")
        self.assertIn("专利权终止", text)
        _assert_clean(self, text, "legal_status covered/zh")

    def test_covered_with_reviews_zh(self):
        entries = self._entries(status="专利权终止", reviews_checked=True,
                                review_decisions=[{"declareNum": "5W1",
                                                   "declareDate": "2025-01-10"}])
        text = react_tools._legal_status_digest(entries, "zh")
        _assert_clean(self, text, "legal_status reviews/zh")


class TestTranslatorReasonDisclosure(unittest.TestCase):
    def test_cn_default_reason(self):
        out = _decide({"country": "CN", "display": "CN116570413A",
                       "id_type": "publication", "lookups": ["CN116570413A"]})
        _assert_clean(self, str(out.get("reason") or ""), "translator CN")

    def test_us_default_reason_keeps_uspto(self):
        # USPTO 是政府登记机构，不是商业供应商 —— 允许保留。
        out = _decide({"country": "US", "display": "19511555",
                       "id_type": "application", "lookups": ["19511555"]})
        self.assertIn("USPTO", str(out.get("reason") or ""))


class TestSourceNoteDisclosure(unittest.TestCase):
    """检索降级链的 notes 会被拼进 observation（`已核验的数据源：`）。

    这里走**真实返回路径**，而不是自造一个假的 note 字符串——否则测试只是
    在断言自己的夹具（第一版就这么写过，改完代码它反而仍失败）。
    """

    def test_not_configured_note_is_neutral(self):
        with patch("sources.long_task.config.get_baiten_config",
                   return_value={"app_key": "", "app_secret": "",
                                 "gateway_url": "http://gw"}):
            items, note = asyncio.run(
                react_tools._baiten_search_by_query("q", agent=None))
        self.assertEqual(items, [])
        _assert_clean(self, note, "not-configured note")

    def test_gateway_error_note_excludes_vendor_name(self):
        # baiten_client 的异常消息**自带**供应商名（"Baiten API error
        # code=..."/"Baiten API HTTP ..."），而 note 会被渲染进 observation
        # —— 必须在构造 note 时中立化。错误细节本身要保留。
        from sources.baiten_client import BaitenAPIError
        with patch("sources.long_task.config.get_baiten_config",
                   return_value={"app_key": "k", "app_secret": "s",
                                 "gateway_url": "http://gw"}), \
             patch("sources.baiten_client.BaitenClient") as MockClient:
            MockClient.return_value.search = AsyncMock(
                side_effect=BaitenAPIError(
                    "Baiten API error code=404: boom"))
            items, note = asyncio.run(
                react_tools._baiten_search_by_query("q", agent=None))
        self.assertEqual(items, [])
        _assert_clean(self, note, "gateway error note")
        self.assertIn("404", note)
        self.assertIn("boom", note)

    def test_lookup_number_notes_are_neutral(self):
        # 组合层（`_lookup_number_candidates` 给 note 加前缀）也不得引入
        # 供应商名；内层返回生产实际的中立字符串。
        agent = SimpleNamespace(logger=None, _number_cross_used=0,
                                _legal_status_used=0)
        with patch.object(react_tools, "_baiten_search_by_query",
                          new=AsyncMock(return_value=([], "CN 0 hits"))), \
             patch.object(react_tools, "_uspto_search_by_number",
                          new=AsyncMock(return_value=([], "USPTO 0 hits"))):
            _merged, notes = asyncio.run(
                react_tools._lookup_number_candidates(
                    agent, [{"country": "CN", "display": "CN1",
                             "lookups": ["CN1"]}]))
        self.assertTrue(notes)
        _assert_clean(self, " ".join(notes), "number-resolve notes")


class TestDownloadRouteBodyDisclosure(unittest.TestCase):
    """HTTP 响应体会直达前端 —— 错误文案同样不得带供应商名。

    同仓 `patent_detail.py` 已有更好的做法：上游原文只进日志，对外给通用
    文案。这里对齐它（顺带避免把上游原始错误文本吐给客户端）。
    """

    @classmethod
    def setUpClass(cls):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from api_routes.baiten import router
        app = FastAPI()
        app.include_router(router)
        cls.client = TestClient(app, raise_server_exceptions=False)

    def _get(self):
        return self.client.get(
            "/baiten/download?pub_num=CN118000001A&pub_date=20240101")

    def test_not_configured_body_is_neutral(self):
        with patch("sources.long_task.config.get_baiten_config",
                   return_value={"app_key": "", "app_secret": "",
                                 "gateway_url": "http://gw"}):
            resp = self._get()
        self.assertEqual(resp.status_code, 400)
        _assert_clean(self, resp.text, "download not-configured body")

    def test_upstream_error_body_is_neutral(self):
        from sources.baiten_client import BaitenAPIError
        with patch("sources.long_task.config.get_baiten_config",
                   return_value={"app_key": "k", "app_secret": "s",
                                 "gateway_url": "http://gw"}), \
             patch("sources.baiten_client.BaitenClient") as MockClient:
            MockClient.return_value.get_file = AsyncMock(
                side_effect=BaitenAPIError(
                    "Baiten API error code=404: boom"))
            resp = self._get()
        self.assertEqual(resp.status_code, 502)
        _assert_clean(self, resp.text, "download upstream-error body")

    def test_unexpected_error_body_is_neutral(self):
        with patch("sources.long_task.config.get_baiten_config",
                   return_value={"app_key": "k", "app_secret": "s",
                                 "gateway_url": "http://gw"}), \
             patch("sources.baiten_client.BaitenClient") as MockClient:
            MockClient.return_value.get_file = AsyncMock(
                side_effect=RuntimeError("boom"))
            resp = self._get()
        self.assertEqual(resp.status_code, 502)
        _assert_clean(self, resp.text, "download unexpected-error body")


if __name__ == "__main__":
    unittest.main()
