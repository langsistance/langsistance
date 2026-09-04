"""Tests for sources/seller/scene_pack (A1 scene voice pack)."""
import unittest

from sources.seller.scene_pack import (
    SELLER_SCENE_ID,
    seller_voice_addendum,
    scene_id_for_request,
)

# 红线：卖家口径提示词不得出现专利法术语（通用禁词，非测试词）
_BANNED_TERMS = ("权利要求", "本领域技术人员", "实施例")


class TestSellerVoiceAddendum(unittest.TestCase):
    def test_zh_addendum_present_and_clean(self):
        text = seller_voice_addendum("zh")
        self.assertTrue(text)
        for term in _BANNED_TERMS:
            self.assertNotIn(term, text, f"禁用术语出现在卖家口径: {term}")
        self.assertIn("不构成法律意见", text)

    def test_en_addendum_present(self):
        self.assertTrue(seller_voice_addendum("en"))

    def test_unknown_lang_empty(self):
        self.assertEqual(seller_voice_addendum("fr"), "")


class TestSceneIdForRequest(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(scene_id_for_request("seller"), SELLER_SCENE_ID)
        self.assertEqual(scene_id_for_request("pro"), 1)
        self.assertIsNone(scene_id_for_request(None))
        self.assertIsNone(scene_id_for_request(""))
        self.assertIsNone(scene_id_for_request("unknown"))


if __name__ == "__main__":
    unittest.main()
