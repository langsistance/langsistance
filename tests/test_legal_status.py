# -*- coding: utf-8 -*-
"""需求#18 法律状态归一化 — 纯函数单元测试。

覆盖：CN 有序规则表（首匹配胜出）、US 委托既有 is_dead_status、
时间线归约、复审决定事实陈述、法务语气护栏。
"""
import unittest

from sources.long_task.legal_status import (
    FORBIDDEN_ADVICE_TERMS,
    LEGAL_STATUS_DISCLAIMER,
    STATUS_ALIVE,
    STATUS_DEAD,
    STATUS_PENDING,
    STATUS_UNKNOWN,
    classify_status,
    is_dead_status_i18n,
    summarize_review_decisions,
    summarize_timeline,
)


class TestClassifyStatusCN(unittest.TestCase):
    def test_terminated_is_dead(self):
        r = classify_status("专利权终止", country="CN")
        self.assertEqual(r["category"], STATUS_DEAD)
        self.assertEqual(r["code"], "CN_TERMINATED")

    def test_terminated_with_de_particle_is_dead(self):
        # 佰腾真实取值带「的」：law_state = "专利权的终止 专利权有效期届满"
        r = classify_status("专利权的终止", country="CN")
        self.assertEqual(r["category"], STATUS_DEAD)

    def test_expiry_is_dead(self):
        r = classify_status("专利权有效期届满", country="CN")
        self.assertEqual(r["category"], STATUS_DEAD)

    def test_combined_real_status_is_dead(self):
        r = classify_status("专利权的终止 专利权有效期届满", country="CN")
        self.assertEqual(r["category"], STATUS_DEAD)

    def test_fee_termination_is_dead(self):
        r = classify_status("未缴年费专利权终止", country="CN")
        self.assertEqual(r["category"], STATUS_DEAD)

    def test_invalidated_is_dead(self):
        r = classify_status("宣告专利权全部无效", country="CN")
        self.assertEqual(r["category"], STATUS_DEAD)

    def test_withdrawn_rejected_abandoned_are_dead(self):
        for s in ["撤回", "公布后撤回", "视为撤回", "驳回",
                  "视为放弃", "放弃"]:
            self.assertEqual(classify_status(s, country="CN")["category"],
                             STATUS_DEAD, s)

    def test_office_action_notice_is_not_dead(self):
        # 「驳回理由通知」是审查意见通知书，申请仍然在审 —— 必须先于
        # 宽规则「驳回」匹配，否则在审申请会被误判为死案。
        r = classify_status("驳回理由通知", country="CN")
        self.assertNotEqual(r["category"], STATUS_DEAD)

    def test_granted_and_pending_are_alive(self):
        for s in ["授权", "实质审查", "公开", "受理"]:
            self.assertIn(classify_status(s, country="CN")["category"],
                          (STATUS_ALIVE, STATUS_PENDING), s)

    def test_unknown_chinese_stays_unknown_and_not_dead(self):
        r = classify_status("某种未收录的状态", country="CN")
        self.assertEqual(r["category"], STATUS_UNKNOWN)
        self.assertFalse(is_dead_status_i18n("某种未收录的状态", country="CN"))

    def test_empty_and_none_unknown(self):
        for s in ["", None, "   "]:
            r = classify_status(s, country="CN")
            self.assertEqual(r["category"], STATUS_UNKNOWN, repr(s))
            self.assertEqual(r["code"], "UNKNOWN", repr(s))

    def test_never_raises_on_non_string(self):
        for s in [123, [], {}, object()]:
            self.assertEqual(classify_status(s, country="CN")["category"],
                             STATUS_UNKNOWN)


class TestClassifyStatusUS(unittest.TestCase):
    def test_dead_boolean_delegates_to_candidate_metadata(self):
        from sources.long_task.candidate_metadata import is_dead_status
        samples = [
            "Patent Expired Due to NonPayment of Maintenance Fees",
            "Abandoned  --  Failure to Respond to an Office Action",
            "Express Abandonment",
            "RO PROCESSING COMPLETED-PLACED IN STORAGE",
            "Patented Case",
            "Non Final Action Mailed",
            "",
            None,
        ]
        for s in samples:
            self.assertEqual(is_dead_status_i18n(s, country="US"),
                             is_dead_status(s), repr(s))

    def test_us_dead_and_live_categories(self):
        self.assertEqual(classify_status("Abandoned", country="US")["category"],
                         STATUS_DEAD)
        self.assertEqual(classify_status("Patented Case",
                                         country="US")["category"],
                         STATUS_ALIVE)

    def test_us_never_raises_on_unknown(self):
        self.assertEqual(classify_status("Some New Status", country="US")["code"],
                         "US_UNMAPPED")


class TestSummarizeTimeline(unittest.TestCase):
    def test_latest_event_drives_category(self):
        timeline = [
            {"date": "2024-08-15", "lawStatus": "专利权终止",
             "lawStatusCode": "", "lawStatusDetail": ""},
            {"date": "2023-11-20", "lawStatus": "授权",
             "lawStatusCode": "", "lawStatusDetail": ""},
        ]
        s = summarize_timeline(timeline, country="CN")
        self.assertEqual(s["event_count"], 2)
        self.assertEqual(s["latest"], "专利权终止")
        self.assertEqual(s["category"], STATUS_DEAD)
        self.assertTrue(s["dead"])
        self.assertEqual(s["latest_date"], "2024-08-15")

    def test_empty_timeline_is_unknown(self):
        s = summarize_timeline([], country="CN")
        self.assertEqual(s["event_count"], 0)
        self.assertEqual(s["category"], STATUS_UNKNOWN)
        self.assertFalse(s["dead"])
        self.assertEqual(s["latest"], "")

    def test_single_alive_event(self):
        s = summarize_timeline(
            [{"date": "2023-11-20", "lawStatus": "授权"}], country="CN")
        self.assertEqual(s["event_count"], 1)
        self.assertFalse(s["dead"])

    def test_malformed_entries_are_tolerated(self):
        s = summarize_timeline([None, "x", {"date": "", "lawStatus": ""}],
                               country="CN")
        self.assertEqual(s["category"], STATUS_UNKNOWN)


class TestSummarizeReviewDecisions(unittest.TestCase):
    def test_no_decisions_is_empty_string(self):
        self.assertEqual(summarize_review_decisions([], lang="zh"), "")
        self.assertEqual(summarize_review_decisions(None, lang="zh"), "")

    def test_states_decision_facts(self):
        decisions = [{
            "declareNum": "5W123456",
            "declareDate": "2025-01-10",
            "lawBase": "专利法第22条第3款",
            "fullText": "宣告专利权全部无效。",
        }]
        out = summarize_review_decisions(decisions, lang="zh")
        self.assertIn("5W123456", out)
        self.assertIn("2025-01-10", out)
        self.assertIn("全部无效", out)

    def test_english_labels(self):
        decisions = [{"declareNum": "5W1", "declareDate": "2025-01-10",
                      "lawBase": "", "fullText": "维持专利权有效。"}]
        out = summarize_review_decisions(decisions, lang="en")
        self.assertTrue(out.isascii() or "5W1" in out)
        self.assertNotIn("复审/无效", out)

    def test_malformed_decisions_tolerated(self):
        out = summarize_review_decisions([None, "x", {}], lang="zh")
        self.assertIsInstance(out, str)


class TestGuardrailVocabulary(unittest.TestCase):
    def test_disclaimer_present_in_both_languages(self):
        self.assertTrue(LEGAL_STATUS_DISCLAIMER.get("zh"))
        self.assertTrue(LEGAL_STATUS_DISCLAIMER.get("en"))
        self.assertIn("不构成法律意见", LEGAL_STATUS_DISCLAIMER["zh"])
        self.assertIn("not legal advice",
                      LEGAL_STATUS_DISCLAIMER["en"].lower())

    def test_disclaimer_contains_no_advice_terms(self):
        blob = " ".join(LEGAL_STATUS_DISCLAIMER.values())
        for term in FORBIDDEN_ADVICE_TERMS:
            self.assertNotIn(term, blob, term)

    def test_generated_decision_text_contains_no_advice_terms(self):
        decisions = [{"declareNum": "5W1", "declareDate": "2025-01-10",
                      "lawBase": "", "fullText": "宣告专利权全部无效。"}]
        for lang in ("zh", "en"):
            out = summarize_review_decisions(decisions, lang=lang)
            for term in FORBIDDEN_ADVICE_TERMS:
                self.assertNotIn(term, out, f"{lang}:{term}")


if __name__ == "__main__":
    unittest.main()
