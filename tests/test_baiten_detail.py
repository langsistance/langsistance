"""Tests for the Baiten branch of the patent detail endpoints."""
import sys
import unittest
from unittest.mock import MagicMock

# Environment shim — same pattern as test_patent_detail_api.py: passport
# initializes Firebase at import time and cannot load in the local venv.
_passport_stub = MagicMock()
_passport_stub.verify_firebase_token = MagicMock(return_value={"uid": "1"})
sys.modules["sources.user.passport"] = _passport_stub

from api_routes.patent_detail import (
    PatentDetailError,
    _build_baiten_download_url,
    _fetch_baiten_claims,
    _fetch_baiten_spec,
    _flatten_baiten_claims,
    _normalize_baiten_date,
)


class TestNormalizeBaitenDate(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(_normalize_baiten_date("2024-01-05"), "20240105")
        self.assertEqual(_normalize_baiten_date("20240105"), "20240105")
        self.assertEqual(_normalize_baiten_date("2024.01.05"), "20240105")
        self.assertEqual(_normalize_baiten_date(""), "")


class TestBuildBaitenDownloadUrl(unittest.TestCase):
    def test_joins_pub_num_and_normalized_date(self):
        url = _build_baiten_download_url("CN118000001A", "2024-01-05")
        self.assertEqual(
            url, "/baiten/download?pub_num=CN118000001A&pub_date=20240105")


class TestFlattenBaitenClaims(unittest.TestCase):
    def test_flattens_hierarchy_and_dedupes(self):
        body = {"data": {"patentClaimses": [
            {"claim": "一种散热装置，其特征在于……",
             "claimsNum": 1, "claimsParentNum": None},
            {"claim": "一种散热装置，其特征在于……",
             "claimsNum": 2, "claimsParentNum": 1},
            {"junk": 1},
        ]}}
        texts = _flatten_baiten_claims(body)
        self.assertEqual(len(texts), 1)

    def test_top_level_and_empty(self):
        body = {"patentClaimses": [{"claim": "X", "claimsNum": 1}]}
        self.assertEqual(_flatten_baiten_claims(body), ["X"])
        self.assertEqual(_flatten_baiten_claims({}), [])

    def test_live_snake_case_container_strips_html(self):
        # Live-verified shape (2026-08-27, real key): patent_claims_list
        # with claims_num/claims_parentNum and <p>-wrapped claim text.
        body = {"patent_claims_list": [
            {"claims_num": "1", "claims_parentNum": "0",
             "claim": "<p>1.一种智能旋转式湿度控制装置，其特征在于，"
                      "包括外壳(1)、安装于所述外壳(1)内的固定隔板(52)</p>"},
            {"claims_num": "2", "claims_parentNum": "1",
             "claim": "<p>2.如权利要求1所述的装置，其特征在于，"
                      "还包括密封隔板(6)</p>"},
        ]}
        texts = _flatten_baiten_claims(body)
        self.assertEqual(len(texts), 2)
        self.assertTrue(texts[0].startswith("1.一种智能旋转式"))
        self.assertNotIn("<p>", texts[0])
        self.assertNotIn("</p>", texts[0])

    def test_collapses_markup_whitespace(self):
        # Line breaks inside <p> become stray spaces mid-sentence after
        # tag stripping — collapse them so claims read cleanly.
        body = {"patent_claims_list": [
            {"claim": "<p>该载气源\n    经分流阀后分为两路，\n"
                      "第一路依次经电磁阀流量控制器I</p>"},
        ]}
        texts = _flatten_baiten_claims(body)
        self.assertEqual(
            texts[0],
            "该载气源 经分流阀后分为两路， 第一路依次经电磁阀流量控制器I")
        self.assertNotIn("\n", texts[0])


class _FakeBaitenClient:
    """Records calls; scriptable get_doc / get_claims responses."""

    def __init__(self, doc=None, claims_map=None):
        self.doc = doc or {"data": {"an": "CN202310123456", "pd": "2024-01-05"}}
        self.claims_map = claims_map or {}
        self.calls = []

    async def get_doc(self, patent_id):
        self.calls.append(("get_doc", patent_id))
        return self.doc

    async def get_claims(self, app_num, pat_type):
        self.calls.append(("get_claims", app_num, pat_type))
        return self.claims_map.get(pat_type, {"data": {"patentClaimses": []}})

    async def get_file(self, pub_num, pub_date):
        raise AssertionError("get_file should not be called from detail branch")


class TestFetchBaitenClaims(unittest.TestCase):
    def test_auth_then_app_fallback(self):
        import asyncio
        client = _FakeBaitenClient(claims_map={
            "AUTH": {"data": {"patentClaimses": []}},
            "APP": {"data": {"patentClaimses": [
                {"claim": "一种冷却系统", "claimsNum": 1,
                 "claimsParentNum": None}]}},
        })
        from api_routes import patent_detail as pd
        pd._get_baiten_client = lambda: client
        payload = asyncio.run(pd._fetch_baiten_claims("CN118000001A"))
        self.assertTrue(payload["success"])
        self.assertEqual(payload["claims"][0]["text"], "一种冷却系统")
        self.assertTrue(payload["claims"][0]["independent"])
        self.assertEqual(
            client.calls,
            [("get_doc", "CN118000001A"),
             ("get_claims", "CN202310123456", "AUTH"),
             ("get_claims", "CN202310123456", "APP")])

    def test_app_num_passes_through_without_get_doc(self):
        # The frontend sends the CN application number; the broken
        # extService getDoc hop must be skipped entirely (2026-08-27:
        # signature gate passes, data service reports system error).
        import asyncio
        client = _FakeBaitenClient(claims_map={
            "APP": {"data": {"patentClaimses": [
                {"claim": "一种干燥装置", "claimsNum": 1,
                 "claimsParentNum": None}]}},
        })
        from api_routes import patent_detail as pd
        pd._get_baiten_client = lambda: client
        payload = asyncio.run(pd._fetch_baiten_claims("CN202311458694.9"))
        self.assertTrue(payload["success"])
        self.assertEqual(
            client.calls,
            [("get_claims", "CN202311458694.9", "AUTH"),
             ("get_claims", "CN202311458694.9", "APP")])

    def test_no_claims_falls_back_to_pdf_proxy(self):
        import asyncio
        client = _FakeBaitenClient(claims_map={})  # both AUTH/APP empty
        from api_routes import patent_detail as pd
        pd._get_baiten_client = lambda: client
        payload = asyncio.run(pd._fetch_baiten_claims("CN118000001A"))
        self.assertTrue(payload["success"])
        self.assertIn("/baiten/download?pub_num=CN118000001A", payload["pdf_url"])

    def test_missing_app_num_raises(self):
        import asyncio
        client = _FakeBaitenClient(doc={"data": {}})
        from api_routes import patent_detail as pd
        pd._get_baiten_client = lambda: client
        with self.assertRaises(PatentDetailError):
            asyncio.run(pd._fetch_baiten_claims("CN118000001A"))


class TestFetchBaitenSpec(unittest.TestCase):
    def test_returns_pdf_proxy_url(self):
        import asyncio
        client = _FakeBaitenClient()
        from api_routes import patent_detail as pd
        pd._get_baiten_client = lambda: client
        payload = asyncio.run(pd._fetch_baiten_spec("CN118000001A"))
        self.assertTrue(payload["success"])
        self.assertEqual(
            payload["pdf_url"],
            "/baiten/download?pub_num=CN118000001A&pub_date=20240105")

    def test_missing_pub_date_raises(self):
        import asyncio
        client = _FakeBaitenClient(doc={"data": {}})
        from api_routes import patent_detail as pd
        pd._get_baiten_client = lambda: client
        with self.assertRaises(PatentDetailError):
            asyncio.run(pd._fetch_baiten_spec("CN118000001A"))


if __name__ == "__main__":
    unittest.main()
