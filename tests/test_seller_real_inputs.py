"""Real-shape seller inputs (generic forms only — no verbatim user asks).

遵守"测试提问词不固化"红线：只保留专利号/产品描述的形式模板，
不复刻任何单个用户的原始提问。
"""
import unittest

from sources.seller.query_classifier import classify_seller_query

# 结构合法但明显虚构的形态（数字为编造，仅验证解析路径；
# 已对照 sources/patent_number_parser.py 实际校验规则逐一验证可解析）
CASES = [
    # (输入形态, 期望 kind, 期望 source 或 None)
    ("US D9999999 美国外观 D 号形态", "patent", "uspto"),
    ("US 99,999,999 美国授权号形态", "patent", "uspto"),
    ("30/999,999 美国申请回执号形态", "patent", "uspto"),
    ("CN 309,999,999 中国外观号形态", "patent", "baiten"),
    ("一款可折叠的便携水杯", "product", None),
    ("ASIN 开头的一串字符", "product", None),
]


class TestRealShapes(unittest.TestCase):
    def test_generic_forms(self):
        for text, kind, source in CASES:
            result = classify_seller_query(text)
            self.assertEqual(result["kind"], kind, text)
            if source is not None:
                self.assertEqual(result.get("source"), source, text)


if __name__ == "__main__":
    unittest.main()
