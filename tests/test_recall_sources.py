"""Tests for the recall-expansion sources (family fetch + CPC fetch).

No network — outbound_http is mocked.  Every transport degrades to
empty results on failure; recall expansion is an enhancement, never a
hard dependency.
"""
import os
import tempfile
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

    def test_chunks_long_number_lists(self):
        numbers = [str(7000000 + i) for i in range(25)]
        with patch("sources.long_task.recall_sources.outbound_http") as mock:
            mock.request.side_effect = [
                _FakeResponse(200, {"count": 1,
                                    "patentFileWrapperDataBag": [
                                        {"applicationNumberText": "a",
                                         "applicationMetaData":
                                             {"inventionTitle": "T"}}]}),
                _FakeResponse(200, {"count": 1,
                                    "patentFileWrapperDataBag": [
                                        {"applicationNumberText": "b",
                                         "applicationMetaData":
                                             {"inventionTitle": "T"}}]}),
            ]
            items = fetch_by_numbers(numbers)
        self.assertEqual(len(items), 2)
        # two requests: 20 numbers then 5
        self.assertEqual(mock.request.call_count, 2)
        first_q = mock.request.call_args_list[0].kwargs["json"]["q"]
        second_q = mock.request.call_args_list[1].kwargs["json"]["q"]
        self.assertNotIn("7000020", first_q)   # 21st number is not in chunk 1
        self.assertIn("7000020", second_q)     # but starts chunk 2
        self.assertIn("7000024", second_q)


class TestFetchByCpc(unittest.TestCase):
    """CPC recall reads the local MCF index (built by
    scripts/build_cpc_index.py) and fetches the matching patents'
    metadata through the number search."""

    def _make_index_db(self, tmp):
        import sqlite3
        db_path = os.path.join(tmp, "cpc_index.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE cpc_patents (cpc TEXT, patent TEXT)")
        conn.execute("CREATE INDEX idx_cpc ON cpc_patents(cpc)")
        conn.executemany(
            "INSERT INTO cpc_patents VALUES (?, ?)",
            [("H05B45/20", "11882632"),
             ("H05B45/20", "12289808"),
             ("H05B45/21", "11647569")])
        conn.commit()
        conn.close()
        return db_path

    def test_queries_index_and_fetches_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = self._make_index_db(tmp)
            items = [{"applicationNumberText": "17473914",
                      "applicationMetaData": {
                          "inventionTitle": "LED DRIVER",
                          "applicationStatusDescriptionText":
                              "Patented Case"}}]
            with patch("sources.long_task.recall_sources.CPC_INDEX_DB",
                       db_path), \
                 patch("sources.long_task.recall_sources.fetch_by_numbers",
                       return_value=items) as mock_fetch:
                records = fetch_by_cpc(["H05B45/20"])
        self.assertEqual(len(records), 1)
        numbers = mock_fetch.call_args[0][0]
        self.assertIn("11882632", numbers)
        self.assertIn("12289808", numbers)

    def test_main_group_hint_matches_subgroups(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = self._make_index_db(tmp)
            with patch("sources.long_task.recall_sources.CPC_INDEX_DB",
                       db_path), \
                 patch("sources.long_task.recall_sources.fetch_by_numbers",
                       return_value=[]) as mock_fetch:
                self.assertEqual(fetch_by_cpc(["H05B45/00"]), [])
        numbers = mock_fetch.call_args[0][0]
        # LIKE prefix pulls both subgroups, the exact one excluded
        self.assertIn("11882632", numbers)
        self.assertIn("11647569", numbers)

    def test_missing_index_returns_empty_without_http(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("sources.long_task.recall_sources.CPC_INDEX_DB",
                       os.path.join(tmp, "absent.db")), \
                 patch("sources.long_task.recall_sources.fetch_by_numbers") \
                    as mock_fetch:
                self.assertEqual(fetch_by_cpc(["H05B45/20"]), [])
        mock_fetch.assert_not_called()

    def test_prefix_query_uses_index_not_scan(self):
        # sqlite refuses the index for a case-insensitive LIKE pattern
        # that starts with a letter (full scan of the ~50M-row index per
        # query) — the prefix branch must stay on GLOB, which is
        # case-sensitive and always index-usable.
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            db_path = self._make_index_db(tmp)
            conn = sqlite3.connect(db_path)
            try:
                plan = conn.execute(
                    "EXPLAIN QUERY PLAN SELECT DISTINCT patent FROM "
                    "cpc_patents WHERE cpc GLOB 'H05B45/*'").fetchone()[3]
            finally:
                conn.close()
        self.assertIn("USING INDEX idx_cpc", plan)
        self.assertNotIn("SCAN", plan)

    def test_lowercase_codes_normalized(self):
        # fetch_by_cpc uppercases at the boundary — a lowercase hint
        # must still hit the uppercase canonical index (GLOB is
        # case-sensitive, unlike LIKE).
        with tempfile.TemporaryDirectory() as tmp:
            db_path = self._make_index_db(tmp)
            with patch("sources.long_task.recall_sources.CPC_INDEX_DB",
                       db_path), \
                 patch("sources.long_task.recall_sources.fetch_by_numbers",
                       return_value=[]) as mock_fetch:
                self.assertEqual(fetch_by_cpc(["h05b45/20"]), [])
        numbers = mock_fetch.call_args[0][0]
        self.assertIn("11882632", numbers)
        self.assertIn("12289808", numbers)

    def test_newest_patents_preferred_numerically(self):
        # patent numbers vary in length (7-digit grants from the 1900s,
        # 8/9-digit from 2000s+); TEXT ordering would put "9999999"
        # (1870s) ahead of "12289808" (2025) — only length-aware
        # ordering picks the truly newest patents, which are also the
        # only ones the applications/search API can fetch.
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "cpc_index.db")
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE cpc_patents (cpc TEXT, patent TEXT)")
            conn.execute("CREATE INDEX idx_cpc ON cpc_patents(cpc)")
            conn.executemany(
                "INSERT INTO cpc_patents VALUES (?, ?)",
                [("H05B45/20", "11882632"),
                 ("H05B45/20", "9999999"),    # 1870s grant
                 ("H05B45/20", "12289808")])
            conn.commit()
            conn.close()
            with patch("sources.long_task.recall_sources.CPC_INDEX_DB",
                       db_path), \
                 patch("sources.long_task.recall_sources.fetch_by_numbers",
                       return_value=[]) as mock_fetch:
                self.assertEqual(fetch_by_cpc(["H05B45/20"]), [])
        numbers = mock_fetch.call_args[0][0]
        self.assertEqual(numbers[0], "12289808")
        self.assertEqual(numbers[1], "11882632")
        self.assertEqual(numbers[-1], "9999999")

    def test_top_code_sampled_deeply(self):
        # recency-only sampling crowds the top code out with current-year
        # grants (observed: a 2025 anchor patent at rank 281 of its code
        # never reached a newest-50 window) — the first, semantically
        # closest code gets a much deeper window than the rest.
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "cpc_index.db")
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE cpc_patents (cpc TEXT, patent TEXT)")
            conn.execute("CREATE INDEX idx_cpc ON cpc_patents(cpc)")
            conn.executemany(
                "INSERT INTO cpc_patents VALUES (?, ?)",
                [("H05B45/20", str(12600000 - i)) for i in range(320)]
                + [("A01B1/00", str(7000000 - i)) for i in range(5)])
            conn.commit()
            conn.close()
            with patch("sources.long_task.recall_sources.CPC_INDEX_DB",
                       db_path), \
                 patch("sources.long_task.recall_sources.fetch_by_numbers",
                       return_value=[]) as mock_fetch:
                self.assertEqual(
                    fetch_by_cpc(["H05B45/20", "A01B1/00"]), [])
        numbers = mock_fetch.call_args[0][0]
        # 300 from the top code + 5 from the second
        self.assertEqual(len(numbers), 305)
        self.assertIn("12599701", numbers)   # 300th newest of the top code
        self.assertNotIn("12599700", numbers)  # 301st — beyond the window
        self.assertIn("7000000", numbers)

    def test_no_patents_in_index_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = self._make_index_db(tmp)
            with patch("sources.long_task.recall_sources.CPC_INDEX_DB",
                       db_path), \
                 patch("sources.long_task.recall_sources.fetch_by_numbers") \
                    as mock_fetch:
                self.assertEqual(fetch_by_cpc(["A01B1/00"]), [])
        mock_fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
