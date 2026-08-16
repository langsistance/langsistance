"""Tests for candidate_metadata — USPTO raw_items flattening + dedupe."""
import unittest

from sources.long_task.candidate_metadata import (
    build_candidates,
    dedupe_candidates,
    ensure_search_fields,
    is_dead_status,
    is_keyword_search_tool,
    is_uspto_tool,
)


def _usp_item(app_number, title, applicant="ACME Corp", filing="2024-01-15",
              status="Patented Case", cpc=None, grant=None, continuity=None):
    meta = {
        "inventionTitle": title,
        "firstApplicantName": applicant,
        "filingDate": filing,
        "applicationStatusDescriptionText": status,
    }
    if cpc:
        meta["cpcClassificationBag"] = [{"cpcClassCode": c} for c in cpc]
    if grant:
        meta["grantDate"] = grant
        meta["patentNumber"] = grant.replace(",", "")
    item = {
        "applicationNumberText": app_number,
        "applicationMetaData": meta,
    }
    if continuity:
        item["parentContinuityBag"] = continuity
    return item


class _FakeTool:
    def __init__(self, title, url):
        self.title = title
        self.url = url


class TestBuildCandidates(unittest.TestCase):
    def test_extracts_standard_metadata(self):
        items = [_usp_item("19511555", "Hydrogen Supply System",
                           applicant="Robert Bosch GmbH", cpc=["F02M 21/02"])]
        candidates = build_candidates(items)
        self.assertEqual(len(candidates), 1)
        c = candidates[0]
        self.assertEqual(c["patent_id"], "19511555")
        self.assertEqual(c["title"], "Hydrogen Supply System")
        self.assertEqual(c["applicant"], "Robert Bosch GmbH")
        self.assertEqual(c["filing_date"], "2024-01-15")
        self.assertEqual(c["status"], "Patented Case")
        self.assertEqual(c["cpc_codes"], ["F02M 21/02"])

    def test_skips_items_without_application_number(self):
        candidates = build_candidates([
            {"applicationMetaData": {"inventionTitle": "No number"}},
            _usp_item("18184836", "Valid one"),
        ])
        self.assertEqual([c["patent_id"] for c in candidates], ["18184836"])

    def test_handles_non_dict_items(self):
        candidates = build_candidates([None, "junk", 42])
        self.assertEqual(candidates, [])

    def test_nested_application_number_fallback_still_parses(self):
        item = {
            "applicationMetaData": {
                "applicationNumberText": "19511555",
                "inventionTitle": "Legacy nested shape",
            },
        }
        candidates = build_candidates([item])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["patent_id"], "19511555")


class TestEnsureSearchFields(unittest.TestCase):
    def test_adds_missing_fields_without_mutating_input(self):
        params = {"body": {"q": "air dryer", "fields": ["applicationMetaData.inventionTitle"]}}
        out = ensure_search_fields(params)
        self.assertIn("applicationMetaData.cpcClassificationBag", out["body"]["fields"])
        self.assertIn("parentContinuityBag", out["body"]["fields"])
        self.assertNotIn("applicationMetaData.cpcClassificationBag",
                         params["body"]["fields"])

    def test_keeps_fields_without_body(self):
        out = ensure_search_fields({"query": {}})
        self.assertEqual(out, {"query": {}})


class TestDedupeCandidates(unittest.TestCase):
    def test_same_title_deduped_keeps_higher_score(self):
        candidates = [
            {"patent_id": "17361306", "title": "Hybrid Level EVSE",
             "relevance_score": 3, "filing_date": "2021-06-28", "_raw": {}},
            {"patent_id": "16821726", "title": "Hybrid Level EVSE",
             "relevance_score": 4, "filing_date": "2020-03-17", "_raw": {}},
        ]
        kept, dropped = dedupe_candidates(candidates)
        self.assertEqual([c["patent_id"] for c in kept], ["16821726"])
        self.assertEqual(dropped, 1)

    def test_continuity_relationship_deduped(self):
        candidates = [
            {"patent_id": "19511555", "title": "Hydrogen supply system",
             "relevance_score": 5, "filing_date": "2024-08-05",
             "_raw": {"parentContinuityBag": [
                 {"parentApplicationNumberText": "PCTEP2024072117",
                  "childApplicationNumberText": "19511555"}]}},
            {"patent_id": "19504130", "title": "Unrelated fuel injection",
             "relevance_score": 4, "filing_date": "2024-06-01",
             "_raw": {"parentContinuityBag": [
                 {"parentApplicationNumberText": "19511555"}]}},
        ]
        kept, dropped = dedupe_candidates(candidates)
        # 19511555 scores higher and is referenced by 19504130's parent bag
        self.assertEqual([c["patent_id"] for c in kept], ["19511555"])
        self.assertEqual(dropped, 1)

    def test_distinct_titles_kept(self):
        candidates = [
            {"patent_id": "11111111", "title": "Air dryer control",
             "relevance_score": 3, "filing_date": "2020-01-01", "_raw": {}},
            {"patent_id": "22222222", "title": "Moisture control enclosure",
             "relevance_score": 3, "filing_date": "2020-01-01", "_raw": {}},
        ]
        kept, dropped = dedupe_candidates(candidates)
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, 0)


class TestIsDeadStatus(unittest.TestCase):
    def test_dead_statuses_detected(self):
        for s in [
            "Patent Expired Due to NonPayment of Maintenance Fees Under 37 CFR 1.362",
            "Provisional Application Expired",
            "Abandoned  --  Failure to Respond to an Office Action",
            "Expressly Abandoned  --  During Examination",
            "Abandonment - Failure to Respond to an Office action",
            "Express Abandonment",
            "RO PROCESSING COMPLETED-PLACED IN STORAGE",
        ]:
            self.assertTrue(is_dead_status(s), s)

    def test_live_or_unknown_statuses_kept(self):
        for s in ["Patented Case",
                  "Publications -- Issue Fee Payment Verified",
                  "Docketed New Case - Ready for Examination",
                  "Non Final Action Mailed",
                  "", None]:
            self.assertFalse(is_dead_status(s), repr(s))


class TestDeadStatusRanking(unittest.TestCase):
    def test_dead_sinks_below_live_even_when_granted_and_higher_score(self):
        candidates = [
            {"patent_id": "19999999", "title": "Expired granted dryer",
             "relevance_score": 5, "filing_date": "2015-01-01",
             "patent_number": "9123456",
             "status": "Patent Expired Due to NonPayment of Maintenance Fees Under 37 CFR 1.362",
             "_raw": {}},
            {"patent_id": "18123456", "title": "Pending dry air control",
             "relevance_score": 1, "filing_date": "2024-01-01",
             "status": "Docketed New Case - Ready for Examination",
             "_raw": {}},
        ]
        kept, dropped = dedupe_candidates(candidates)
        self.assertEqual([c["patent_id"] for c in kept],
                         ["18123456", "19999999"])
        self.assertEqual(dropped, 0)

    def test_family_keeps_live_member_over_dead_granted_one(self):
        # Intent: within one family, a live pending member is the
        # preferred representative over an expired granted patent even
        # when the dead one scored higher.
        candidates = [
            {"patent_id": "19999999", "title": "Dryer family",
             "relevance_score": 5, "filing_date": "2015-01-01",
             "patent_number": "9123456",
             "status": "Patent Expired Due to NonPayment of Maintenance Fees Under 37 CFR 1.362",
             "_raw": {"parentContinuityBag": [
                 {"childApplicationNumberText": "18123456"}]}},
            {"patent_id": "18123456", "title": "Dryer family continuation",
             "relevance_score": 1, "filing_date": "2024-01-01",
             "status": "Docketed New Case - Ready for Examination",
             "_raw": {"parentContinuityBag": [
                 {"parentApplicationNumberText": "19999999"}]}},
        ]
        kept, dropped = dedupe_candidates(candidates)
        self.assertEqual([c["patent_id"] for c in kept], ["18123456"])
        self.assertEqual(dropped, 1)


class TestToolPredicates(unittest.TestCase):
    def test_keyword_tool_detection(self):
        self.assertTrue(is_keyword_search_tool(
            _FakeTool("search_patent_by_key_word", "https://api.uspto.gov/api/v1/patent/applications/search")))
        self.assertFalse(is_keyword_search_tool(
            _FakeTool("get_patent_documents_application_number",
                      "https://api.uspto.gov/api/v1/patent/applications")))

    def test_uspto_tool_detection(self):
        self.assertTrue(is_uspto_tool(
            _FakeTool("search_patent_by_key_word", "https://api.uspto.gov/api/v1/patent/applications/search")))
        self.assertFalse(is_uspto_tool(
            _FakeTool("cnipa_search", "https://open.zldsj.com/api/search")))


if __name__ == "__main__":
    unittest.main()
