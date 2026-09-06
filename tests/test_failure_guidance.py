# -*- coding: utf-8 -*-
"""Shared failure-guidance template function (spec §5.4 / plan Task 3).

``failure_guidance(task_type, reason_code, error, lang)`` is the single
source the whole pipeline uses to give users actionable next steps when a
family / prosecution / examination flow cannot proceed:

- pre-check refusal (chat main chain / submit / retry / detail routes) —
  produced as a guidance reply / error rather than a task;
- worker terminal failure message (T5 wires it into the failure writer).

Guidance segments are generic sentences; they must NEVER embed a concrete
user query word (repo rule: no query-specific vocabulary in prompts/code).
"""
import unittest
from sources.long_task.status_manager import (
    failure_guidance,
    ERR_UNRESOLVABLE_ID,
    ERR_EPO_REMOTE,
    ERR_OTHER,
)


class TestFailureGuidance(unittest.TestCase):
    def test_unresolvable_id_zh_suggests_publication(self):
        msg = failure_guidance("families", ERR_UNRESOLVABLE_ID, lang="zh")
        self.assertIn("WO 公开号", msg)

    def test_unresolvable_id_en_suggests_publication(self):
        msg = failure_guidance("families", ERR_UNRESOLVABLE_ID, lang="en")
        self.assertIn("publication", msg)

    def test_epo_remote_zh_suggests_retry(self):
        msg = failure_guidance("families", ERR_EPO_REMOTE, lang="zh")
        self.assertIn("稍后", msg)

    def test_epo_remote_en_suggests_retry_later(self):
        msg = failure_guidance("families", ERR_EPO_REMOTE, lang="en")
        self.assertTrue("later" in msg.lower() or "retry" in msg.lower())

    def test_other_zh_generic(self):
        msg = failure_guidance("families", ERR_OTHER, error="boom", lang="zh")
        self.assertIn("重试", msg)

    def test_other_en_generic(self):
        msg = failure_guidance("families", ERR_OTHER, error="boom", lang="en")
        self.assertTrue("try again" in msg.lower() or "retry" in msg.lower())

    def test_error_is_bounded_no_raise(self):
        # Oversized error must never cause an exception nor blow the message up.
        long_err = "e" * 20000
        msg = failure_guidance("families", ERR_OTHER, error=long_err, lang="zh")
        self.assertLess(len(msg), 2000)

    def test_unknown_reason_code_falls_back_to_other(self):
        msg = failure_guidance("families", "ERR_BOGUS_CODE", error="x", lang="zh")
        self.assertIn("重试", msg)

    def test_no_query_word_embedded(self):
        # Guidance must stay generic — a probe input must never leak in.
        probe = "PCTUS2021059064 抽查 XX勘察 专用词 1234"
        failure_guidance("families", ERR_UNRESOLVABLE_ID, error=probe, lang="zh")
        for code in (ERR_UNRESOLVABLE_ID, ERR_EPO_REMOTE, ERR_OTHER):
            out = failure_guidance("families", code, lang="zh")
            self.assertNotIn("PCTUS2021059064", out, code)

    def test_default_lang_zh(self):
        self.assertIn("WO 公开号", failure_guidance("families", ERR_UNRESOLVABLE_ID))


if __name__ == "__main__":
    unittest.main()
