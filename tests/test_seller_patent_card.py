"""Tests for sources/seller/patent_card."""
import asyncio
import unittest
from unittest.mock import AsyncMock

from sources.seller.patent_card import (
    build_patent_card,
    card_cache_put,
    card_cache_get,
    DISCLAIMER,
)


class FakeProvider:
    def __init__(self, payload):
        self.complete_json = AsyncMock(return_value=payload)


CARD_COPY_OK = {
    "protection_summary": "保护这种带卡扣的折叠水杯结构：杯身可分两段折叠，折叠处用卡扣固定",
    "risk_level": "high",
    "next_step": "高风险：建议规避或授权。改掉卡扣闭合特征，或向权利人询价授权。",
}


class TestBuildPatentCard(unittest.TestCase):
    def tearDown(self):
        # 进程内缓存跨用例共享——每个用例前清空。
        from sources.seller import patent_card
        patent_card._cache.clear()

    def test_success_card_has_four_blocks_and_disclaimer(self):
        provider = FakeProvider(dict(CARD_COPY_OK))
        result = asyncio.run(build_patent_card(
            provider, "1. A folding cup with a clasp.", "uspto", "US 11,675,432"))
        self.assertTrue(result["success"])
        card = result["card"]
        self.assertEqual(card["patent_id"], "US 11,675,432")
        self.assertEqual(card["risk_level"], "high")
        self.assertIsNone(card["legal_status"])
        self.assertIn("M1.5", card["status_note"])
        self.assertIn("不构成法律意见", card["disclaimer"])
        self.assertTrue(card["llm_available"])
        self.assertEqual(provider.complete_json.await_count, 1)

    def test_second_call_hits_cache_no_llm_call(self):
        provider = FakeProvider(dict(CARD_COPY_OK))
        asyncio.run(build_patent_card(provider, "1. A folding cup.", "uspto",
                                      "US 11,675,432"))
        result = asyncio.run(build_patent_card(provider, "1. A folding cup.",
                                               "uspto", "US 11,675,432"))
        self.assertTrue(result["cached"])
        self.assertEqual(provider.complete_json.await_count, 1)

    def test_llm_failure_degrades_without_raising(self):
        provider = FakeProvider(None)
        result = asyncio.run(build_patent_card(provider, "1. A folding cup.",
                                               "uspto", "US 11,675,432"))
        self.assertTrue(result["success"])
        card = result["card"]
        self.assertFalse(card["llm_available"])
        self.assertIsNone(card["protection_summary"])
        self.assertIn("disclaimer", card)

    def test_cache_ttl_expiry(self):
        card_cache_put("uspto:US 1:zh", {"patent_id": "US 1"}, ttl_seconds=-1)
        self.assertIsNone(card_cache_get("uspto:US 1:zh"))

    def test_cache_roundtrip(self):
        card_cache_put("uspto:US 2:zh", {"patent_id": "US 2"})
        self.assertEqual(card_cache_get("uspto:US 2:zh")["patent_id"], "US 2")


if __name__ == "__main__":
    unittest.main()
