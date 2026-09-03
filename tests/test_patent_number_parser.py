"""Tests for sources/patent_number_parser — identifier recognition.

Feature (2026-09-03, sample #16): a bare-number question such as
"117941643" was searched once against USPTO (404) and closed — no
country disambiguation, no Baiten CN cross-check.  The parser is the
deterministic front end of the fix: it turns raw user text into
candidate patent identifiers (country / type / confidence / reason)
that routing and the cross-source number lookup consume.
"""

import unittest

from sources.patent_number_parser import (
    decide_number_source,
    format_number_guidance,
    parse_patent_identifiers,
)


def _top(text):
    out = parse_patent_identifiers(text)
    return out[0] if out else None


class TestPrefixedIdentifiers(unittest.TestCase):
    def test_cn_publication_full(self):
        c = _top("CN117941643A")
        self.assertEqual(c["country"], "CN")
        self.assertEqual(c["id_type"], "publication")
        self.assertEqual(c["display"], "CN117941643A")
        self.assertEqual(c["confidence"], "high")

    def test_cn_publication_lower_kind_normalized(self):
        c = _top("cn117941643a")
        self.assertEqual(c["display"], "CN117941643A")

    def test_cn_utility_model(self):
        c = _top("CN221535608U")
        self.assertEqual(c["id_type"], "utility")
        self.assertEqual(c["confidence"], "high")

    def test_cn_grant(self):
        c = _top("CN102345678B")
        self.assertEqual(c["id_type"], "grant")
        self.assertEqual(c["confidence"], "high")

    def test_cn_application_with_dot_and_prefix(self):
        c = _top("CN202311794164.0")
        self.assertEqual(c["country"], "CN")
        self.assertEqual(c["id_type"], "application")
        self.assertEqual(c["confidence"], "high")

    def test_cn_application_13_digits_no_dot(self):
        c = _top("2023117941640")
        self.assertEqual(c["country"], "CN")
        self.assertEqual(c["id_type"], "application")
        self.assertEqual(c["confidence"], "high")

    def test_cn_application_bad_checksum_low_confidence(self):
        c = _top("CN202311794164.4")  # 校验位应为 3
        self.assertEqual(c["country"], "CN")
        self.assertIn(c["confidence"], ("medium", "low"))
        self.assertNotEqual(c["confidence"], "high")

    def test_cn_application_sample_1055(self):
        c = _top("CN201510526456.6")
        self.assertEqual(c["id_type"], "application")
        self.assertEqual(c["confidence"], "high")

    def test_us_grant_with_kind(self):
        c = _top("US9019058B2")
        self.assertEqual(c["country"], "US")
        self.assertEqual(c["id_type"], "grant")
        self.assertEqual(c["display"], "US9019058B2")
        self.assertEqual(c["confidence"], "high")

    def test_us_publication(self):
        c = _top("US20250103146A1")
        self.assertEqual(c["country"], "US")
        self.assertEqual(c["id_type"], "publication")
        self.assertEqual(c["confidence"], "high")

    def test_us_application_slash_form(self):
        c = _top("17/027,484")
        self.assertEqual(c["country"], "US")
        self.assertEqual(c["id_type"], "application")

    def test_us_design_receipt_series(self):
        c = _top("30/076,484")
        self.assertEqual(c["country"], "US")

    def test_us_design_d_number(self):
        c = _top("USD123456S")
        self.assertEqual(c["country"], "US")
        self.assertEqual(c["id_type"], "design")

    def test_us_reissue(self):
        c = _top("RE45678E1")
        self.assertEqual(c["country"], "US")

    def test_external_office_marked_unsupported(self):
        c = _top("EP3456789A1")
        self.assertEqual(c["country"], "EP")
        self.assertEqual(c["confidence"], "low")


class TestBareNumbers(unittest.TestCase):
    def test_sample_16_bare_nine_digit(self):
        """真实样本: 纯数字「117941643」→ CN 公开号段候选(中置信)。"""
        c = _top("117941643")
        self.assertEqual(c["country"], "CN")
        self.assertEqual(c["id_type"], "publication")
        self.assertEqual(c["confidence"], "medium")
        self.assertIn("CN117941643", c["lookups"][0])

    def test_sample_16_with_metadata_noise(self):
        """真实样本带系统拼接噪音行(用户 id/query id)仍能识别。"""
        text = ("117941643,\n user id is 15845258328802749617,\n "
                "query id is ksnhmxxpke,\n")
        out = parse_patent_identifiers(text)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["display"], "CN117941643")

    def test_bare_eight_digit_ambiguous_us(self):
        c = _top("19511555")
        self.assertEqual(c["country"], "US")
        self.assertEqual(c["id_type"], "ambiguous")
        self.assertEqual(c["confidence"], "medium")

    def test_bare_seven_digit(self):
        c = _top("9019058")
        self.assertEqual(c["country"], "US")

    def test_comma_grouped_us_grant(self):
        c = _top("11,794,164")
        self.assertEqual(c["country"], "US")

    def test_number_in_question_text(self):
        out = parse_patent_identifiers("帮我查一下117941643是什么专利")
        self.assertTrue(out)
        self.assertEqual(out[0]["country"], "CN")

    def test_english_cue(self):
        out = parse_patent_identifiers("patent number 9019058 please")
        self.assertTrue(out)


class TestNegativeCases(unittest.TestCase):
    def test_no_number(self):
        self.assertEqual(parse_patent_identifiers("太空光伏盖板材料"), [])

    def test_year_only_not_a_number(self):
        self.assertEqual(parse_patent_identifiers("2023年申请的专利"), [])

    def test_short_phrase(self):
        self.assertEqual(parse_patent_identifiers("RGB LED 驱动芯片"), [])

    def test_document_paste_ignored(self):
        """长文本粘贴(权利要求/说明书, >600 字符)不触发号码路由。"""
        body = ("1. 一种方法，包括：\n2. 根据权利要求1所述的方法，"
                "其中所述控制器配置为执行 117941643 次循环……\n"
                "3. 根据权利要求1所述的方法，还包括：\n" * 60)
        self.assertEqual(parse_patent_identifiers(body), [])

    def test_query_id_noise_alone(self):
        self.assertEqual(
            parse_patent_identifiers("query id is ksnhmxxpke"), [])


class TestOrderingAndGuards(unittest.TestCase):
    def test_dedupe_and_max_three(self):
        out = parse_patent_identifiers(
            "CN117941643A CN117941643A CN221535608U US9019058B2")
        displays = [c["display"] for c in out]
        self.assertEqual(displays.count("CN117941643A"), 1)
        self.assertLessEqual(len(out), 3)

    def test_prefixed_high_sorts_first(self):
        out = parse_patent_identifiers("117941643 CN221535608U")
        self.assertEqual(out[0]["country"], "CN")
        self.assertEqual(out[0]["confidence"], "high")

    def test_candidate_fields_complete(self):
        for c in parse_patent_identifiers("117941643"):
            for key in ("raw", "display", "country", "id_type",
                        "confidence", "reason", "lookups"):
                self.assertIn(key, c)


class TestSourceDecision(unittest.TestCase):
    def test_hard_cn_prefix_routes_cn(self):
        out = parse_patent_identifiers("CN221535608U 这个专利怎么样")
        self.assertEqual(decide_number_source(out), "cn")

    def test_hard_us_prefix_routes_uspto(self):
        out = parse_patent_identifiers("US9019058B2")
        self.assertEqual(decide_number_source(out), "uspto")

    def test_bare_number_keeps_dual(self):
        out = parse_patent_identifiers("117941643")
        self.assertIsNone(decide_number_source(out))

    def test_empty(self):
        self.assertIsNone(decide_number_source([]))


class TestGuidanceText(unittest.TestCase):
    def test_zh_guidance_lists_candidates_and_contract(self):
        out = parse_patent_identifiers("117941643")
        text = format_number_guidance(out, lang="zh")
        self.assertIn("CN117941643", text)
        self.assertIn("复核", text)

    def test_en_guidance(self):
        text = format_number_guidance(
            parse_patent_identifiers("117941643"), lang="en")
        self.assertIn("CN117941643", text)

    def test_empty_guidance(self):
        self.assertEqual(format_number_guidance([], lang="zh"), "")


if __name__ == "__main__":
    unittest.main()
