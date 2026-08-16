"""Tests for _prune_for_summary in general_agent.

Regression: PEDS search items carry huge nested bags (eventDataBag,
parentContinuityBag, ...) whose direct values are lists — the old
shallow string-only check kept them wholesale and the summary LLM call
blew past its 400k-token context limit.
"""
import unittest

from sources.agents.general_agent import _prune_for_summary


def _peds_item() -> dict:
    return {
        "applicationNumberText": "18405037",
        "applicationMetaData": {
            "inventionTitle": "Agentic AI workflow orchestration",
            "patentNumber": "US12000123B2",
        },
        "eventDataBag": [
            {"eventCode": "X" * 6000, "eventDate": "2024-01-01"}
            for _ in range(30)
        ],
        "parentContinuityBag": [
            {"continuityTypeCode": "Y" * 6000} for _ in range(30)
        ],
        "correspondenceAddressBag": ["Z" * 6000],
        "smallList": ["keep me", "and me"],
    }


class TestPruneForSummary(unittest.TestCase):
    def test_drops_oversized_nested_arrays(self):
        pruned = _prune_for_summary([_peds_item()])[0]
        self.assertNotIn("eventDataBag", pruned)
        self.assertNotIn("parentContinuityBag", pruned)
        self.assertNotIn("correspondenceAddressBag", pruned)

    def test_keeps_small_values_and_nested_dicts(self):
        pruned = _prune_for_summary([_peds_item()])[0]
        self.assertEqual(pruned["applicationNumberText"], "18405037")
        self.assertEqual(
            pruned["applicationMetaData"]["inventionTitle"],
            "Agentic AI workflow orchestration",
        )
        self.assertEqual(pruned["smallList"], ["keep me", "and me"])

    def test_non_dict_items_pass_through(self):
        self.assertEqual(_prune_for_summary(["plain", 42]), ["plain", 42])

    def test_total_payload_stays_bounded(self):
        # 50 PEDS-sized items must not exceed the summary LLM budget —
        # the old behaviour produced ~780k tokens for exactly this shape.
        import json

        items = [_peds_item() for _ in range(50)]
        pruned = _prune_for_summary(items)
        total_chars = sum(len(json.dumps(item, default=str)) for item in pruned)
        # ~190k-token worst case for 50 x 15k chars; require well under it.
        self.assertLess(total_chars, 50 * 15000)


class TestBoundedSummaryItems(unittest.TestCase):
    def test_total_payload_within_budget(self):
        from sources.agents.general_agent import _bounded_summary_items
        big = "x" * 20000  # oversized value gets truncated by the pruner
        items = [{"title": f"T{i}", "bag": big} for i in range(30)]
        kept = _bounded_summary_items(items)
        import json
        total = sum(len(json.dumps(i, ensure_ascii=False, default=str)) for i in kept)
        self.assertLessEqual(total, 121000)
        self.assertLess(len(kept), len(items))

    def test_keeps_leading_items_in_order(self):
        from sources.agents.general_agent import _bounded_summary_items
        items = [{"title": f"T{i}", "pad": "y" * 6000} for i in range(10)]
        kept = _bounded_summary_items(items)
        self.assertEqual([i["title"] for i in kept],
                         [f"T{i}" for i in range(len(kept))])
        self.assertGreaterEqual(len(kept), 1)

    def test_empty_returns_empty(self):
        from sources.agents.general_agent import _bounded_summary_items
        self.assertEqual(_bounded_summary_items([]), [])


if __name__ == "__main__":
    unittest.main()


class TestLargeListSummaryHeading(unittest.TestCase):
    def test_heading_reports_summarized_over_total_zh(self):
        from sources.agents.general_agent import _large_list_summary_heading
        self.assertEqual(
            _large_list_summary_heading("zh", 6, 100),
            "## 结果摘要 (6 / 100 项)")

    def test_heading_reports_summarized_over_total_en(self):
        from sources.agents.general_agent import _large_list_summary_heading
        self.assertEqual(
            _large_list_summary_heading("en", 6, 100),
            "## Results — Summary (6 / 100 items)")
