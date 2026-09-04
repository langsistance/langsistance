"""Tests for sources/seller/query_classifier."""
import sys
import unittest

# passport stub 需要在导入 seller 路由相关模块前就位；本文件只测纯逻辑，
# 无需 stub，但保持目录测试隔离一致。
from sources.seller.query_classifier import classify_seller_query


class TestClassifySellerQuery(unittest.TestCase):
    def test_us_design_number_is_patent(self):
        result = classify_seller_query("US D1,088,888")
        self.assertEqual(result["kind"], "patent")
        self.assertEqual(result["source"], "uspto")

    def test_cn_number_is_patent(self):
        result = classify_seller_query("CN 306,998,821")
        self.assertEqual(result["kind"], "patent")
        self.assertEqual(result["source"], "baiten")

    def test_bare_us_application_style_is_patent(self):
        result = classify_seller_query("30/076,484")
        self.assertEqual(result["kind"], "patent")

    def test_product_name_is_product(self):
        result = classify_seller_query("折叠水杯")
        self.assertEqual(result["kind"], "product")

    def test_mixed_text_with_patent_number_is_patent(self):
        result = classify_seller_query("帮我看看 US 11,675,432 这件的保护范围")
        self.assertEqual(result["kind"], "patent")
        self.assertIn("US 11,675,432", result["matched"])

    def test_empty_and_long_input_are_product(self):
        self.assertEqual(classify_seller_query("")["kind"], "product")
        self.assertEqual(classify_seller_query("水" * 300)["kind"], "product")


if __name__ == "__main__":
    unittest.main()
