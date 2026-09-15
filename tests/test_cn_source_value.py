# -*- coding: utf-8 -*-
"""CN 来源在结果行里的取值。

**背景**：导出文件（CSV/Excel/JSON）里 `source` 列的值曾是 `"baiten"` ——
用户可下载的文件里带着商业数据供应商名，属于泄露（见
`test_source_disclosure.py` 的同一条产品约束）。

**做法**：产出侧统一写中立值 `"cn"`，但**必须继续接受** `"baiten"` /
`"cnipa"` —— 已经持久化在前端的结果行、以及旧客户端仍在用这些写法。
丢掉它们会让 CN 行被路由到 US 详情端点：2026-08-29 事故现场是
``/patent/uspto/CN213905456U/spec``。
"""
import io
import json
import os
import re
import sys
import unittest
import zipfile
from unittest.mock import MagicMock

# 环境垫片（同 test_patent_detail_api.py）：passport 在 import 期就初始化
# Firebase + Redis，本地跑不起来。
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
sys.modules.setdefault("firebase_admin", MagicMock())
_passport_stub = MagicMock()
_passport_stub.verify_firebase_token = MagicMock(return_value={"uid": "1"})
sys.modules["sources.user.passport"] = _passport_stub

from api_routes.patent_detail import VALID_SOURCES  # noqa: E402
from sources.agents.react_tools import _baiten_results_to_candidates
from sources.patent_source_detect import is_cn_source
from sources.result_export import build_result_artifacts


def _sheet_names(xlsx_bytes):
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes)) as archive:
        workbook = archive.read("xl/workbook.xml").decode("utf-8")
    return re.findall(r'<sheet[^>]*name="([^"]+)"', workbook)


def _artifact(artifacts, fmt):
    return next(a for a in artifacts if a["format"] == fmt)


class TestIsCnSource(unittest.TestCase):
    """共享谓词：产出写新值，读取同时认旧值。"""

    def test_accepts_current_and_legacy_values(self):
        for value in ("cn", "CN", " cn ", "baiten", "BAITEN", "cnipa"):
            self.assertTrue(is_cn_source(value), repr(value))

    def test_rejects_other_sources(self):
        for value in ("uspto", "USPTO", "", None, "google_patents",
                      "uspto_documents"):
            self.assertFalse(is_cn_source(value), repr(value))


class TestProducerEmitsNeutralValue(unittest.TestCase):
    def test_cn_candidates_carry_cn_source(self):
        body = {"code": "200", "data": {"fieldValues": [
            {"pn": "CN118000001A", "ti": "散热装置", "an": "CN202311111111.1"},
        ]}}
        cands = _baiten_results_to_candidates(body)
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0]["source"], "cn")


class TestExportCarriesNoVendorName(unittest.TestCase):
    _CN_ROW = {"patent_id": "CN118000001A", "source": "cn",
               "title": "散热装置", "applicant": "华为"}
    _US_ROW = {"applicationNumberText": "19511555", "source": "uspto",
               "title": "Cooling device"}

    def test_csv_and_json_contain_no_vendor_name(self):
        artifacts = build_result_artifacts(
            [self._CN_ROW, self._US_ROW], source="cn", lang="zh")
        self.assertTrue(artifacts)
        for fmt in ("csv", "json"):
            blob = _artifact(artifacts, fmt)["content"].decode("utf-8")
            self.assertNotIn("baiten", blob.lower(), fmt)

    def test_xlsx_sheet_names_are_jurisdiction_not_vendor(self):
        artifacts = build_result_artifacts(
            [self._CN_ROW, self._US_ROW], source="cn", lang="zh")
        names = _sheet_names(_artifact(artifacts, "xlsx")["content"])
        self.assertIn("中国专利", names)
        self.assertIn("美国专利", names)
        for name in names:
            self.assertNotIn("baiten", name.lower())

    def test_sheet_split_still_recognises_legacy_value(self):
        # 已持久化的旧行仍是 "baiten" —— 分表不能因此失效
        # （否则 CN 行会混进美国表）。
        legacy = dict(self._CN_ROW, source="baiten")
        artifacts = build_result_artifacts(
            [legacy, self._US_ROW], source="cn", lang="zh")
        names = _sheet_names(_artifact(artifacts, "xlsx")["content"])
        self.assertIn("中国专利", names)
        self.assertIn("美国专利", names)

    def test_json_payload_source_is_neutral(self):
        artifacts = build_result_artifacts([self._CN_ROW], source="cn")
        payload = json.loads(
            _artifact(artifacts, "json")["content"].decode("utf-8"))
        self.assertNotIn("baiten", json.dumps(payload).lower())


class TestDetailRouteAcceptsBothValues(unittest.TestCase):
    """`source` 会被拼进详情接口的 URL 路径，后端必须新旧都认。"""

    def test_valid_sources_accepts_new_and_legacy(self):
        self.assertIn("cn", VALID_SOURCES)
        self.assertIn("baiten", VALID_SOURCES)   # 旧客户端 / 已持久化行

    def test_cn_source_routes_to_the_cn_branch(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        import api_routes.patent_detail as pd
        with patch.object(pd, "_fetch_baiten_spec",
                          new=AsyncMock(return_value={"success": True})) as m:
            asyncio.run(pd._fetch_spec_pdf("cn", "CN213905456U", "20240101"))
        self.assertEqual(m.await_count, 1)

    def test_legacy_source_still_routes_to_the_cn_branch(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        import api_routes.patent_detail as pd
        with patch.object(pd, "_fetch_baiten_spec",
                          new=AsyncMock(return_value={"success": True})) as m:
            asyncio.run(pd._fetch_spec_pdf("baiten", "CN213905456U", "20240101"))
        self.assertEqual(m.await_count, 1)


if __name__ == "__main__":
    unittest.main()
