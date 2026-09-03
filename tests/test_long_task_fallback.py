# -*- coding: utf-8 -*-
"""Low-confidence long-task fallback guard (incident 2026-09-03).

The LLM scenario classifier failed with the 'thinking' TypeError and the
regex fallback guessed a CN-family question ("分析 CN114948588A 及其全球
同族申请的审查差异") as conversation_refs, sweeping 140 mixed history ids
into the USPTO pipeline.  Guards added:

1. CN publication numbers (CN114948588A) and CN-prefixed application
   numbers (CN202210498116.7) are recognised by the id token scanner
   (previously only bare 20xx application numbers matched).
2. Without a usable LLM result the fallback only launches a long task for
   query-local, exclusively-US, <=3 ids (direct_ids); everything else
   yields scenario "chat_fallback" (no history sweep, no Celery).
"""
import sys
import types
import unittest

# api_routes.core imports passport at module level; passport.py
# initializes firebase + redis at import time (server-only deps).  Stub
# ONLY passport — real sources.knowledge.knowledge imports cleanly
# off-server, so it must stay real to avoid poisoning other suites that
# import react_tools/general_agent surfaces from it.
_fake_passport = types.ModuleType("sources.user.passport")
_fake_passport.verify_firebase_token = lambda *a, **k: {"uid": "1"}
_fake_passport.check_and_increase_usage = lambda *a, **k: True
_fake_passport.ensure_local_user_record = lambda *a, **k: None
sys.modules.setdefault("sources.user.passport", _fake_passport)

from api_routes.core import (  # noqa: E402  (stubs must precede import)
    _extract_patent_id_tokens,
    _is_us_patent_token,
    _prepare_long_task_inputs,
)


def _conv_history_with_hidden_ids(count: int = 140) -> list:
    """Conversation whose assistant messages carry many hidden patent_ids —
    the 140-id sweep source in the incident."""
    ids = [f"CN11807932{i % 10}A" for i in range(count // 2)]
    ids += [f"17{str(100000 + i)}" for i in range(count // 2)]
    return [{"role": "assistant", "content": "检索完成", "patent_ids": ids}]


class TestPatentIdTokenScan(unittest.TestCase):
    def test_cn_publication_number(self):
        toks = _extract_patent_id_tokens("分析 CN114948588A 及其全球同族申请")
        self.assertEqual(toks, ["CN114948588A"])

    def test_cn_publication_with_spaces(self):
        toks = _extract_patent_id_tokens("查 CN 1149 48588 A")
        self.assertEqual(toks, ["CN114948588A"])

    def test_cn_application_number_bare_and_prefixed(self):
        self.assertEqual(
            _extract_patent_id_tokens("申请 CN202210498116.7 的情况"),
            ["CN202210498116.7"],
        )
        self.assertEqual(
            _extract_patent_id_tokens("申请 202210498116.7 的情况"),
            ["202210498116.7"],
        )

    def test_us_numbers(self):
        self.assertEqual(
            _extract_patent_id_tokens("分析 17429113 和 17/027,484"),
            ["17429113", "17027484"],
        )

    def test_mixed_cn_and_us(self):
        self.assertEqual(
            _extract_patent_id_tokens("CN114948588A 与 17429113"),
            ["17429113", "CN114948588A"],
        )

    def test_no_false_match_inside_longer_digits(self):
        # 9-digit CN publication body must not leak an 8-digit US token.
        self.assertEqual(
            _extract_patent_id_tokens("CN114948588A 再举例 12345678"),
            ["12345678", "CN114948588A"],
        )

    def test_us_token_classifier(self):
        self.assertTrue(_is_us_patent_token("17429113"))
        self.assertFalse(_is_us_patent_token("CN114948588A"))
        self.assertFalse(_is_us_patent_token("CN202210498116.7"))
        self.assertFalse(_is_us_patent_token("202210498116.7"))


class TestPrepareLongTaskFallback(unittest.TestCase):
    def test_cn_family_question_refuses_long_task(self):
        # Incident query: CN publication + analysis intent + rich history.
        inputs = _prepare_long_task_inputs(
            query="分析 CN114948588A 及其全球同族申请的审查差异",
            conv_history=_conv_history_with_hidden_ids(),
            app_logger=None,
            llm_result=None,
        )
        self.assertEqual(inputs["scenario"], "chat_fallback")
        self.assertEqual(inputs["patent_ids"], [])

    def test_us_single_id_in_query_direct_ids(self):
        inputs = _prepare_long_task_inputs(
            query="分析 17429113 的审查历史",
            conv_history=_conv_history_with_hidden_ids(),
            app_logger=None,
            llm_result=None,
        )
        self.assertEqual(inputs["scenario"], "direct_ids")
        self.assertEqual(inputs["patent_ids"], ["17429113"])

    def test_no_ids_new_topic_refuses_long_task(self):
        inputs = _prepare_long_task_inputs(
            query="帮我看看特斯拉近期的专利",
            conv_history=_conv_history_with_hidden_ids(),
            app_logger=None,
            llm_result=None,
        )
        self.assertEqual(inputs["scenario"], "chat_fallback")
        self.assertEqual(inputs["patent_ids"], [])

    def test_mixed_cn_us_ids_refuses_long_task(self):
        inputs = _prepare_long_task_inputs(
            query="CN114948588A 与 17429113 谁的审查历史更复杂",
            conv_history=_conv_history_with_hidden_ids(),
            app_logger=None,
            llm_result=None,
        )
        self.assertEqual(inputs["scenario"], "chat_fallback")
        self.assertEqual(inputs["patent_ids"], [])

    def test_more_than_three_ids_refuses_long_task(self):
        inputs = _prepare_long_task_inputs(
            query="分析 17429113, 18012525, 18331482, 19592637",
            conv_history=[],
            app_logger=None,
            llm_result=None,
        )
        self.assertEqual(inputs["scenario"], "chat_fallback")
        self.assertEqual(inputs["patent_ids"], [])

    def test_llm_result_still_authoritative(self):
        # A successful classification keeps full control (unchanged path).
        inputs = _prepare_long_task_inputs(
            query="分析 17429113 的审查历史",
            conv_history=_conv_history_with_hidden_ids(),
            app_logger=None,
            llm_result={
                "scenario": "prosecution",
                "patent_ids": ["17429113"],
                "patent_source": "uspto",
                "patent_id_type": "application_number",
                "reasoning": "single US application",
            },
        )
        self.assertEqual(inputs["scenario"], "prosecution")
        self.assertEqual(inputs["patent_ids"], ["17429113"])


if __name__ == "__main__":
    unittest.main()
