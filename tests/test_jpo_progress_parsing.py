"""Tests for JPO progress parsing (Japan examination history fix).

Root cause of the JP report bug: the JPO app_progress response has NO
``progressList`` key.  Real examination events live in
``bibliographyInformation[].documentList[]`` (documentCode /
documentDescription / legalDate / documentNumber).  The old parser looked
for ``progressList``, found nothing, and silently synthesized 3 fake
events from bibliographic dates — which masked the data gap downstream.

These tests pin the corrected behaviour:
  - real events are extracted from bibliographyInformation.documentList
  - biblio synthesis is an explicitly flagged fallback, counted separately
  - timelines clearly label synthesized data instead of presenting it
    as a real examination history
"""

import unittest
from unittest.mock import AsyncMock, MagicMock

from sources.jpo_client import (
    JpoAPIError,
    parse_jp_progress_events,
    translate_jp_event,
)
from sources.long_task.japan_examination import (
    build_examination_timeline,
    fetch_examination_data,
)


# ── Sample data (mirrors the official JPO OpenAPI response structure) ──────────

PROGRESS_WITH_DOCUMENTS = {
    "applicationNumber": "2019159764",
    "inventionTitle": "二次電池収容構造及び人型ロボット",
    "filingDate": "20190902",
    "publicationNumber": "2021037573",
    "publicationDate": "20210305",
    "registrationNumber": "7274385",
    "registrationDate": "20230508",
    "bibliographyInformation": [
        {
            "numberType": "01",
            "number": "2019159764",
            "documentList": [
                {
                    "legalDate": "20190902",
                    "documentCode": "A0100001",
                    "documentDescription": "出願",
                    "documentNumber": "201912345678",
                },
                {
                    "legalDate": "20210305",
                    "documentCode": "A6210002",
                    "documentDescription": "出願公開",
                    "documentNumber": "202112345678",
                },
                {
                    "legalDate": "20220810",
                    "documentCode": "A6210003",
                    "documentDescription": "拒絶理由通知書",
                    "documentNumber": "202212345678",
                },
                {
                    "legalDate": "20221014",
                    "documentCode": "A1120004",
                    "documentDescription": "手続補正書",
                    "documentNumber": "202212345679",
                },
                {
                    "legalDate": "20230301",
                    "documentCode": "A2520005",
                    "documentDescription": "特許査定",
                    "documentNumber": "202312345678",
                },
                {
                    "legalDate": "20230508",
                    "documentCode": "G0100006",
                    "documentDescription": "設定登録",
                    "documentNumber": "202312345679",
                },
            ],
        },
    ],
}

PROGRESS_BIBLIO_ONLY = {
    "applicationNumber": "2019159764",
    "inventionTitle": "二次電池収容構造及び人型ロボット",
    "filingDate": "20190902",
    "publicationNumber": "2021037573",
    "publicationDate": "20210305",
    "registrationNumber": "7274385",
    "registrationDate": "20230508",
    # NOTE: no bibliographyInformation / documentList — the old failure mode
}


# ── parse_jp_progress_events ───────────────────────────────────────────────────


class TestParseJpProgressEvents(unittest.TestCase):
    def test_extracts_real_events_from_bibliography_information(self):
        events, synthesized = parse_jp_progress_events(PROGRESS_WITH_DOCUMENTS)

        self.assertFalse(synthesized)
        self.assertEqual(len(events), 6)
        # Sorted by date ascending
        self.assertEqual(
            [e["event_date"] for e in events],
            ["20190902", "20210305", "20220810", "20221014", "20230301", "20230508"],
        )
        refusal = events[2]
        self.assertEqual(refusal["event"], "拒絶理由通知書")
        self.assertEqual(refusal["event_code"], "A6210003")
        self.assertEqual(refusal["event_number"], "202212345678")
        self.assertEqual(refusal["event_date"], "20220810")

    def test_synthesizes_flagged_events_when_document_list_absent(self):
        events, synthesized = parse_jp_progress_events(PROGRESS_BIBLIO_ONLY)

        self.assertTrue(synthesized)
        self.assertEqual(len(events), 3)
        names = [e["event"] for e in events]
        self.assertEqual(names, ["出願", "出願公開", "特許登録"])
        for event in events:
            self.assertTrue(event.get("synthesized"), "synthesized events must be marked")

    def test_skips_fully_empty_document_rows(self):
        data = {
            "bibliographyInformation": [
                {
                    "numberType": "01",
                    "number": "2019159764",
                    "documentList": [
                        {},
                        {"legalDate": "", "documentCode": "", "documentDescription": ""},
                        {
                            "legalDate": "20220810",
                            "documentCode": "A6210003",
                            "documentDescription": "拒絶理由通知書",
                            "documentNumber": "202212345678",
                        },
                    ],
                },
            ],
        }

        events, synthesized = parse_jp_progress_events(data)

        self.assertFalse(synthesized)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "拒絶理由通知書")

    def test_handles_non_list_bibliography_information(self):
        data = {"bibliographyInformation": None, "filingDate": "20190902"}

        events, synthesized = parse_jp_progress_events(data)

        self.assertTrue(synthesized)
        self.assertEqual(len(events), 1)  # only filing date present


class TestTranslateJpEvent(unittest.TestCase):
    def test_partial_match_for_refusal_notice_full_name(self):
        self.assertEqual(translate_jp_event("拒絶理由通知書", "zh"), "驳回理由通知")

    def test_partial_match_for_amendment(self):
        self.assertEqual(translate_jp_event("手続補正書", "zh"), "手续补正书")


# ── build_examination_timeline ─────────────────────────────────────────────────


class TestBuildExaminationTimeline(unittest.TestCase):
    def test_real_events_include_document_numbers_in_detail(self):
        jp_data = {
            "jp_app_number": "2019159764",
            "progress": parse_jp_progress_events(PROGRESS_WITH_DOCUMENTS)[0],
            "progress_synthesized": False,
        }

        timeline = build_examination_timeline(jp_data, lang="zh")

        self.assertIn("共 6 个审查事件", timeline)
        self.assertIn("驳回理由通知", timeline)  # translated refusal notice
        self.assertIn("202212345678", timeline)  # document number in detail column

    def test_synthesized_timeline_is_explicitly_labelled(self):
        events, synthesized = parse_jp_progress_events(PROGRESS_BIBLIO_ONLY)
        self.assertTrue(synthesized)
        jp_data = {
            "jp_app_number": "2019159764",
            "progress": [],
            "synthesized_events": events,
            "progress_synthesized": True,
        }

        timeline = build_examination_timeline(jp_data, lang="zh")

        self.assertIn("著录项", timeline)  # degraded-data label
        self.assertIn("提交申请", timeline)  # translated filing event

    def test_no_data_message_when_no_events_at_all(self):
        jp_data = {
            "jp_app_number": "2019159764",
            "progress": [],
            "progress_synthesized": False,
        }

        timeline = build_examination_timeline(jp_data, lang="zh")

        self.assertIn("未获取到日本审查经过数据", timeline)


# ── fetch_examination_data ─────────────────────────────────────────────────────


class TestFetchExaminationData(unittest.IsolatedAsyncioTestCase):
    def _mock_jpo_client(self, progress_response):
        client = MagicMock()
        client.get_patent_progress = AsyncMock(return_value=progress_response)
        client.get_registration_info = AsyncMock(return_value={
            "registrationNumber": "7274385",
            "registrationDate": "20230508",
        })
        client.get_citations = AsyncMock(return_value={"citeList": []})
        client.get_refusal_reasons = AsyncMock(side_effect=JpoAPIError("no entity"))
        client.get_amendments = AsyncMock(side_effect=JpoAPIError("no entity"))
        return client

    async def test_counts_only_real_events_when_document_list_present(self):
        client = self._mock_jpo_client(PROGRESS_WITH_DOCUMENTS)

        result = await fetch_examination_data("2019159764", client)

        self.assertEqual(result["progress_count"], 6)
        self.assertFalse(result["progress_synthesized"])
        self.assertEqual(result["synthesized_events"], [])
        self.assertEqual(len(result["progress"]), 6)

    async def test_synthesized_fallback_is_separate_from_real_progress(self):
        client = self._mock_jpo_client(PROGRESS_BIBLIO_ONLY)

        result = await fetch_examination_data("2019159764", client)

        # Real events: none.  Synthesized fallback must NOT mask this.
        self.assertEqual(result["progress_count"], 0)
        self.assertTrue(result["progress_synthesized"])
        self.assertEqual(len(result["synthesized_events"]), 3)
        self.assertEqual(result["progress"], [])


if __name__ == "__main__":
    unittest.main()
