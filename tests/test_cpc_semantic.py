"""Tests for the CPC scheme parser and semantic matcher (plan B).

The parser runs against a tiny synthetic scheme zip built in-memory;
the matcher is tested with hand-made vectors.  No network, no provider.
"""
import io
import json
import os
import tempfile
import unittest
import zipfile

from sources.long_task.cpc_semantic import (
    MAIN_GROUP_RE,
    load_cpc_titles,
    match_cpc_codes,
    match_query_to_cpc,
    parse_cpc_zip,
)


def _sample_xml():
    return """<?xml version="1.0" encoding="UTF-8"?>
<class-scheme publication-date="2026-08-01" scheme-type="cpc">
<classification-item breakdown-code="false" level="3" sort-key="H05B">
  <classification-symbol>H05B</classification-symbol>
  <class-title><title-part><text>ELECTRIC HEATING; ELECTRIC LIGHT SOURCES</text></title-part></class-title>
</classification-item>
<classification-item breakdown-code="false" level="4" sort-key="H05B45">
  <classification-symbol>H05B45</classification-symbol>
  <class-title><title-part><text>CIRCUIT ARRANGEMENTS FOR OPERATING LIGHT EMITTING DIODES</text></title-part>
    <title-part><text>LED <reference><text>details <class-ref scheme="cpc">H01L33/00</class-ref></text></reference></text></title-part></class-title>
</classification-item>
<classification-item breakdown-code="false" level="5" sort-key="H05B4500">
  <classification-symbol>H05B45/00</classification-symbol>
  <class-title><title-part><text>Circuit arrangements for operating light emitting diodes [LED]</text></title-part></class-title>
</classification-item>
<classification-item breakdown-code="false" level="6" sort-key="H05B4520">
  <classification-symbol>H05B45/20</classification-symbol>
  <class-title><title-part><text>Controlling the colour of the light</text></title-part></class-title>
</classification-item>
</class-scheme>
"""


def _sample_zip(tmpdir):
    path = os.path.join(tmpdir, "scheme.zip")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("cpc-scheme-H.xml", _sample_xml())
    return path


class TestMainGroupRegex(unittest.TestCase):
    def test_main_groups_only(self):
        self.assertTrue(MAIN_GROUP_RE.match("H05B45/00"))
        self.assertTrue(MAIN_GROUP_RE.match("G09G3/00"))
        self.assertFalse(MAIN_GROUP_RE.match("H05B45/20"))
        self.assertFalse(MAIN_GROUP_RE.match("H05B"))
        self.assertFalse(MAIN_GROUP_RE.match("A01B"))


class TestParseCpcZip(unittest.TestCase):
    def test_parses_all_titled_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = parse_cpc_zip(_sample_zip(tmp), main_groups_only=False)
        codes = {e["code"] for e in entries}
        self.assertIn("H05B45/00", codes)
        self.assertIn("H05B45/20", codes)
        self.assertIn("H05B", codes)

    def test_main_groups_only_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = parse_cpc_zip(_sample_zip(tmp), main_groups_only=True)
        codes = {e["code"] for e in entries}
        self.assertEqual(codes, {"H05B45/00"})

    def test_title_takes_first_title_part_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = parse_cpc_zip(_sample_zip(tmp), main_groups_only=False)
        by_code = {e["code"]: e["title"] for e in entries}
        self.assertEqual(
            by_code["H05B45"],
            "CIRCUIT ARRANGEMENTS FOR OPERATING LIGHT EMITTING DIODES")
        self.assertNotIn("H01L33/00", by_code["H05B45"])

    def test_missing_zip_returns_empty(self):
        self.assertEqual(parse_cpc_zip("/nonexistent/scheme.zip"), [])


class TestLoadCpcTitles(unittest.TestCase):
    def test_loads_json_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "titles.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump([
                    {"code": "H05B45/00", "title": "Led circuits"},
                    {"code": "G09G3/00", "title": "Control arrangements"},
                ], f)
            entries = load_cpc_titles(path)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["code"], "H05B45/00")

    def test_missing_json_returns_empty(self):
        self.assertEqual(load_cpc_titles("/nonexistent/titles.json"), [])


class TestMatchCpcCodes(unittest.TestCase):
    def _entries(self):
        return [
            {"code": "H05B45/00", "title": "LED circuits"},
            {"code": "G09G3/00", "title": "Display control"},
            {"code": "A01B1/00", "title": "Hand tools"},
        ]

    def test_ranks_by_cosine(self):
        query = [1.0, 0.0]
        vectors = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
        matches = match_cpc_codes(query, vectors, self._entries(), top_k=2)
        self.assertEqual([m["code"] for m in matches], ["H05B45/00", "G09G3/00"])
        self.assertAlmostEqual(matches[0]["score"], 1.0)
        self.assertGreater(matches[1]["score"], 0.9)
        self.assertLessEqual(matches[1]["score"], 1.0)

    def test_top_k_and_defensive(self):
        query = [1.0, 0.0]
        vectors = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
        self.assertEqual(
            len(match_cpc_codes(query, vectors, self._entries(), top_k=2)), 2)
        self.assertEqual(match_cpc_codes(None, vectors, self._entries()), [])
        self.assertEqual(match_cpc_codes(query, [], self._entries()), [])
        self.assertEqual(match_cpc_codes(query, vectors, []), [])
        self.assertEqual(
            match_cpc_codes(query, [[1.0, 0.0]], self._entries()), [])


class TestMatchQueryToCpc(unittest.TestCase):
    """End-to-end wrapper degrades to [] on every failure mode."""

    def test_empty_query_returns_empty(self):
        self.assertEqual(match_query_to_cpc("  "), [])

    def test_missing_data_files_returns_empty(self):
        from unittest.mock import patch
        with patch("sources.long_task.cpc_semantic.load_cpc_titles",
                   return_value=[]):
            self.assertEqual(match_query_to_cpc("某技术问题"), [])

    def test_embedding_failure_returns_empty(self):
        from unittest.mock import patch
        entries = [{"code": "H05B45/00", "title": "LED circuits"}]
        with patch("sources.long_task.cpc_semantic.load_cpc_titles",
                   return_value=entries), \
             patch("sources.long_task.cpc_semantic.load_cpc_vectors",
                   return_value=[[1.0, 0.0]]), \
             patch("sources.long_task.semantic_rerank.embed_texts",
                   side_effect=RuntimeError("down")):
            self.assertEqual(match_query_to_cpc("某技术问题"), [])

    def test_matches_when_everything_present(self):
        from unittest.mock import patch
        entries = [{"code": "H05B45/00", "title": "LED circuits"},
                   {"code": "A01B1/00", "title": "Hand tools"}]
        with patch("sources.long_task.cpc_semantic.load_cpc_titles",
                   return_value=entries), \
             patch("sources.long_task.cpc_semantic.load_cpc_vectors",
                   return_value=[[1.0, 0.0], [0.0, 1.0]]), \
             patch("sources.long_task.semantic_rerank.embed_texts",
                   return_value=[[1.0, 0.0]]):
            matches = match_query_to_cpc("某技术问题", top_k=1)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["code"], "H05B45/00")
        self.assertAlmostEqual(matches[0]["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
