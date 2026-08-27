"""Tests for _extract_patent_ids_from_items — the conversation_refs channel.

Regression (2026-08-27): Baiten flat candidates carry patent_id /
patent_number / app_num, none of which the extractor recognized, so CN
patents never reached follow-up conversation_refs queries.
"""
import unittest

from sources.agents.general_agent import _extract_patent_ids_from_items


class TestExtractPatentIds(unittest.TestCase):
    def test_uspto_application_number_text(self):
        items = [{"applicationNumberText": "19511555"}]
        self.assertEqual(_extract_patent_ids_from_items(items), ["19511555"])

    def test_baiten_patent_id_publication_number(self):
        items = [{"patent_id": "CN117491049A", "source": "baiten",
                  "title": "压缩空气干燥器"}]
        self.assertEqual(
            _extract_patent_ids_from_items(items), ["CN117491049A"])

    def test_baiten_app_num_fallback(self):
        items = [{"patent_id": "CN117491049A", "app_num": "CN202311458694.9",
                  "source": "baiten"}]
        # patent_id wins (listed first), app_num is the fallback.
        self.assertEqual(
            _extract_patent_ids_from_items(items), ["CN117491049A"])

    def test_baiten_app_num_when_no_patent_id(self):
        items = [{"app_num": "CN202311458694.9", "source": "baiten"}]
        self.assertEqual(
            _extract_patent_ids_from_items(items), ["CN202311458694.9"])

    def test_mixed_sources_deduped_preserving_order(self):
        items = [
            {"applicationNumberText": "19511555"},
            {"patent_id": "CN117491049A", "source": "baiten"},
            {"applicationNumberText": "19511555"},  # duplicate
            {"patent_id": "CN117491049A", "source": "baiten"},  # duplicate
        ]
        self.assertEqual(
            _extract_patent_ids_from_items(items),
            ["19511555", "CN117491049A"])

    def test_ignores_items_without_identifiers(self):
        items = [{"title": "nothing"}, "junk", None, 42]
        self.assertEqual(_extract_patent_ids_from_items(items), [])


if __name__ == "__main__":
    unittest.main()
