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
