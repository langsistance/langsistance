"""Tests for api_routes/seller."""
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# passport stub：必须在导入 api_routes.seller 之前（先例 test_patent_detail_api.py）
_passport_stub = MagicMock()
_passport_stub.verify_firebase_token = MagicMock(return_value={"uid": "1"})
sys.modules["sources.user.passport"] = _passport_stub

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from sources.seller.query_classifier import classify_seller_query
from api_routes.seller import register_seller_routes


class _FakeProvider:
    def __init__(self):
        self.complete_json = AsyncMock(return_value={
            "protection_summary": "保护这种带卡扣的折叠水杯结构",
            "risk_level": "high",
            "next_step": "改掉卡扣闭合特征，或向权利人询价授权",
        })


def _make_client(provider=None, claims_payload=None):
    app = FastAPI()
    app.include_router(
        register_seller_routes(MagicMock(), MagicMock(), provider or _FakeProvider()))
    if claims_payload is not None:
        # 实现经 api_routes.seller._fetch_claims_lazy → api_routes.patent_detail._fetch_claims
        # 拉取；patch 必须打在源头模块上。
        patcher = patch("api_routes.patent_detail._fetch_claims",
                        new=AsyncMock(return_value=claims_payload))
        patcher.start()
        _make_client._patcher = patcher
    return TestClient(app)


class TestSellerPatentCardEndpoint(unittest.TestCase):
    def tearDown(self):
        p = getattr(_make_client, "_patcher", None)
        if p:
            p.stop()
            del _make_client._patcher
        from sources.seller import patent_card
        patent_card._cache.clear()

    def test_patent_query_returns_card(self):
        client = _make_client(claims_payload={
            "success": True,
            "claims": [
                {"number": 1, "independent": True, "text": "1. A folding cup with a clasp."},
                {"number": 2, "independent": False, "text": "2. The cup of claim 1."},
            ],
        })
        resp = client.post("/seller/patent_card", json={"query": "US 11,675,432"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["success"])
        expected_id = classify_seller_query("US 11,675,432")["patent_id"]
        self.assertEqual(body["card"]["patent_id"], expected_id)
        self.assertEqual(body["card"]["risk_level"], "high")
        self.assertIn("不构成法律意见", body["card"]["disclaimer"])
        self.assertTrue(body["card"]["llm_available"])
        self.assertTrue(body["claims_available"])

    def test_claims_unavailable_degrades(self):
        client = _make_client(claims_payload={"success": False})
        resp = client.post("/seller/patent_card", json={"query": "CN 306,998,821"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["success"])
        self.assertIn("claims_available", body)
        self.assertFalse(body["claims_available"])

    def test_product_query_returns_product_hint(self):
        client = _make_client()
        resp = client.post("/seller/patent_card", json={"query": "折叠水杯"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["kind"], "product")

    def test_missing_auth_is_rejected(self):
        client = _make_client()
        with patch("api_routes.seller.verify_firebase_token",
                   side_effect=HTTPException(status_code=401, detail="Unauthorized")):
            resp = client.post("/seller/patent_card",
                               json={"query": "US D1,088,888"})
        self.assertEqual(resp.status_code, 401)


if __name__ == "__main__":
    unittest.main()
