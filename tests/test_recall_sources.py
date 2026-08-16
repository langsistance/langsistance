"""Tests for the recall-expansion sources (family fetch + CPC fetch).

No network — outbound_http is mocked.  Every transport degrades to
empty results on failure; recall expansion is an enhancement, never a
hard dependency.
"""
import os
import unittest
from unittest.mock import patch

from sources.long_task.recall_sources import (
    collect_family_refs,
    fetch_by_cpc,
    fetch_by_numbers,
    records_to_candidates,
)


def _candidate(raw, patent_number="9000001"):
    return {
        "patent_id": raw.get("applicationNumberText", "10000001"),
        "patent_number": patent_number,
        "_raw": raw,
    }


class TestCollectFamilyRefs(unittest.TestCase):
    def test_extracts_child_and_parent_numbers(self):
        raw = {
            "applicationNumberText": "10000001",
            "childContinuityBag": [
                {"childPatentNumber": "7061668",
                 "childApplicationNumberText": "10393563"},
                {"childApplicationNumberText": "10800200"},
            ],
            "parentContinuityBag": [
                {"parentPatentNumber": "6500000",
                 "parentApplicationNumberText": "60366413"},
            ],
        }
        refs = collect_family_refs([_candidate(raw)])
        self.assertEqual(refs["patents"], ["7061668", "6500000"])
        self.assertEqual(refs["applications"],
                         ["10393563", "10800200", "60366413"])

    def test_excludes_numbers_already_in_pool(self):
        raw = {
            "applicationNumberText": "10000001",
            "childContinuityBag": [
                {"childPatentNumber": "7061668",
                 "childApplicationNumberText": "10393563"},
            ],
        }
        refs = collect_family_refs([
            _candidate(raw),
            {"patent_id": "10393563", "patent_number": "7061668",
             "_raw": {}},
        ])
        self.assertEqual(refs["patents"], [])
        self.assertEqual(refs["applications"], [])

    def test_dedupes_and_caps(self):
        raw = {
            "applicationNumberText": "10000001",
            "childContinuityBag": [
                {"childPatentNumber": f"{7000000 + i}",
                 "childApplicationNumberText": f"{20000000 + i}"}
                for i in range(30)
            ],
        }
        refs = collect_family_refs([_candidate(raw)], limit=5)
        self.assertEqual(refs["patents"], ["7000000", "7000001",
                                           "7000002", "7000003", "7000004"])
        self.assertEqual(len(refs["applications"]), 5)

    def test_missing_bags_returns_empty(self):
        refs = collect_family_refs([_candidate({})])
        self.assertEqual(refs, {"patents": [], "applications": []})


class TestRecordsToCandidates(unittest.TestCase):
    def test_usp_shaped_records_flatten(self):
        records = [{
            "applicationNumberText": "19511555",
            "applicationMetaData": {
                "inventionTitle": "AIR DRYER CONTROL",
                "firstApplicantName": "ACME Corp",
                "applicationStatusDescriptionText": "Patented Case",
                "filingDate": "2024-01-15",
            },
        }]
        cands = records_to_candidates(records)
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0]["patent_id"], "19511555")
        self.assertEqual(cands[0]["title"], "AIR DRYER CONTROL")

    def test_minimal_records_use_patent_number_as_id(self):
        records = [{
            "patent_number": "11882632",
            "patent_title": "LED DRIVER",
            "assignee_organization": "ERP Power",
            "patent_date": "2024-03-05",
        }]
        cands = records_to_candidates(records)
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0]["patent_id"], "11882632")
        self.assertEqual(cands[0]["patent_number"], "11882632")
        self.assertEqual(cands[0]["title"], "LED DRIVER")
        self.assertEqual(cands[0]["applicant"], "ERP Power")
        self.assertEqual(cands[0]["filing_date"], "2024-03-05")

    def test_skips_records_without_any_number(self):
        self.assertEqual(records_to_candidates([{"title": "x"}]), [])
        self.assertEqual(records_to_candidates([None, "junk", 42]), [])


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class TestFetchByNumbers(unittest.TestCase):
    def test_posts_or_joined_numbers_query(self):
        payload = {"count": 1, "patentFileWrapperDataBag": [
            {"applicationNumberText": "10393563",
             "applicationMetaData": {"inventionTitle": "T"}}]}
        with patch("sources.long_task.recall_sources.outbound_http") as mock, \
             patch.dict(os.environ, {"USPTO_API_KEY": "k"}):
            mock.request.return_value = _FakeResponse(200, payload)
            items = fetch_by_numbers(["11882632", "7061668"])
        self.assertEqual(len(items), 1)
        call = mock.request.call_args
        self.assertEqual(call.args[0], "POST")
        self.assertTrue(call.args[1].startswith("https://api.uspto.gov"))
        body = call.kwargs["json"]
        self.assertIn('"11882632"', body["q"])
        self.assertIn('"7061668"', body["q"])
        self.assertIn("X-API-Key", call.kwargs["headers"])

    def test_non_200_returns_empty(self):
        with patch("sources.long_task.recall_sources.outbound_http") as mock:
            mock.request.return_value = _FakeResponse(404, {})
            self.assertEqual(fetch_by_numbers(["11882632"]), [])

    def test_http_failure_returns_empty(self):
        with patch("sources.long_task.recall_sources.outbound_http") as mock:
            mock.request.side_effect = RuntimeError("down")
            self.assertEqual(fetch_by_numbers(["11882632"]), [])

    def test_empty_numbers_skip_http(self):
        with patch("sources.long_task.recall_sources.outbound_http") as mock:
            self.assertEqual(fetch_by_numbers([]), [])
        mock.request.assert_not_called()


class TestFetchByCpc(unittest.TestCase):
    def test_without_key_returns_empty_without_http(self):
        with patch("sources.long_task.recall_sources.outbound_http") as mock, \
             patch.dict(os.environ, {}, clear=True):
            self.assertEqual(fetch_by_cpc(["H05B45/20"]), [])
        mock.request.assert_not_called()

    def test_with_key_posts_ppubs_query(self):
        payload = {"totalCount": 1, "results": [
            {"patentNumber": "11882632", "inventionTitle": "LED DRIVER"}]}
        with patch("sources.long_task.recall_sources.outbound_http") as mock, \
             patch.dict(os.environ, {"PPS_API_KEY": "pps-key"}):
            mock.request.return_value = _FakeResponse(200, payload)
            records = fetch_by_cpc(["H05B45/20"])
        self.assertEqual(len(records), 1)
        call = mock.request.call_args
        self.assertEqual(call.args[0], "POST")
        self.assertIn("ppubs.uspto.gov", call.args[1])
        self.assertEqual(call.kwargs["headers"]["X-API-KEY"], "pps-key")
        self.assertIn("H05B45/20", call.kwargs["json"]["searchText"])

    def test_failure_returns_empty(self):
        with patch("sources.long_task.recall_sources.outbound_http") as mock, \
             patch.dict(os.environ, {"PPS_API_KEY": "pps-key"}):
            mock.request.side_effect = RuntimeError("down")
            self.assertEqual(fetch_by_cpc(["H05B45/20"]), [])


if __name__ == "__main__":
    unittest.main()
