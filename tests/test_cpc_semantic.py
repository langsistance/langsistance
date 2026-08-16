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
from unittest.mock import patch

from sources.long_task.cpc_semantic import (
    CPC_TITLES_JSON,
    CPC_VECTORS_NPY,
    MAIN_GROUP_RE,
    load_cpc_titles,
    load_cpc_vectors,
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

    def test_extra_terms_broaden_matching(self):
        from unittest.mock import patch
        entries = [{"code": "H05B45/00", "title": "LED circuits"},
                   {"code": "A01B1/00", "title": "Hand tools"}]
        # query vector matches nothing much (0.5/0.5 split), the extra
        # terms vector matches H05B45/00 strongly
        with patch("sources.long_task.cpc_semantic.load_cpc_titles",
                   return_value=entries), \
             patch("sources.long_task.cpc_semantic.load_cpc_vectors",
                   return_value=[[1.0, 0.0], [0.0, 1.0]]), \
             patch("sources.long_task.semantic_rerank.embed_texts",
                   return_value=[[0.707, 0.707], [1.0, 0.0]]):
            matches = match_query_to_cpc(
                "某技术问题", top_k=2, extra_terms="led driver color")
        # round-robin guarantee: query text contributes H05B45 first,
        # the extra-terms text contributes its best unseen (A01B1)
        self.assertEqual([m["code"] for m in matches],
                         ["H05B45/00", "A01B1/00"])
        self.assertAlmostEqual(matches[0]["score"], 0.707, places=3)

    def test_extra_terms_as_per_concept_groups(self):
        from unittest.mock import patch
        entries = [{"code": "H05B45/00", "title": "LED circuits"},
                   {"code": "A01B1/00", "title": "Hand tools"}]
        # query vector weak, group-1 vector weak, group-2 vector strong
        with patch("sources.long_task.cpc_semantic.load_cpc_titles",
                   return_value=entries), \
             patch("sources.long_task.cpc_semantic.load_cpc_vectors",
                   return_value=[[1.0, 0.0], [0.0, 1.0]]), \
             patch("sources.long_task.semantic_rerank.embed_texts",
                   return_value=[[0.707, 0.707], [0.707, 0.707],
                                 [1.0, 0.0]]) as mock_embed:
            matches = match_query_to_cpc(
                "q", top_k=2, extra_terms=["amp group", "led color group"])
        # three texts embedded: query + both groups
        self.assertEqual(len(mock_embed.call_args[0][0]), 3)
        self.assertEqual(matches[0]["code"], "H05B45/00")
        self.assertAlmostEqual(matches[0]["score"], 0.707, places=3)

    def test_empty_extra_terms_behave_like_absent(self):
        from unittest.mock import patch
        entries = [{"code": "H05B45/00", "title": "LED circuits"}]
        with patch("sources.long_task.cpc_semantic.load_cpc_titles",
                   return_value=entries), \
             patch("sources.long_task.cpc_semantic.load_cpc_vectors",
                   return_value=[[1.0, 0.0]]), \
             patch("sources.long_task.semantic_rerank.embed_texts",
                   return_value=[[1.0, 0.0]]) as mock_embed:
            matches = match_query_to_cpc("q", top_k=1, extra_terms="   ")
        self.assertEqual(len(matches), 1)
        # only the query text was embedded
        self.assertEqual(len(mock_embed.call_args[0][0]), 1)


class TestCpcPathsForLevel(unittest.TestCase):
    """Tier selection: main groups vs full subgroup data files."""

    def test_explicit_levels(self):
        from sources.long_task.cpc_semantic import (
            CPC_TITLES_SUB_JSON, CPC_VECTORS_SUB_NPY, cpc_paths_for_level)
        self.assertEqual(
            cpc_paths_for_level("main"), (CPC_TITLES_JSON, CPC_VECTORS_NPY))
        self.assertEqual(
            cpc_paths_for_level("sub"),
            (CPC_TITLES_SUB_JSON, CPC_VECTORS_SUB_NPY))

    def test_default_resolves_env(self):
        from sources.long_task.cpc_semantic import (
            CPC_TITLES_SUB_JSON, CPC_VECTORS_SUB_NPY, cpc_paths_for_level)
        with patch.dict(os.environ, {"CPC_VECTOR_LEVEL": ""}):
            self.assertEqual(
                cpc_paths_for_level(), (CPC_TITLES_JSON, CPC_VECTORS_NPY))
            os.environ["CPC_VECTOR_LEVEL"] = "sub"
            self.assertEqual(
                cpc_paths_for_level(),
                (CPC_TITLES_SUB_JSON, CPC_VECTORS_SUB_NPY))

    def test_unknown_level_falls_back_to_main(self):
        from sources.long_task.cpc_semantic import cpc_paths_for_level
        self.assertEqual(
            cpc_paths_for_level("bogus"), (CPC_TITLES_JSON, CPC_VECTORS_NPY))


class TestLoadCpcTitlesLevel(unittest.TestCase):
    def test_no_arg_respects_level_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            sub_json = os.path.join(tmp, "cpc_titles_subgroups.json")
            with open(sub_json, "w", encoding="utf-8") as f:
                json.dump([{"code": "H05B45/20",
                            "title": "Controlling the colour of the light"}], f)
            with patch("sources.long_task.cpc_semantic.CPC_TITLES_SUB_JSON",
                       sub_json), \
                 patch.dict(os.environ, {"CPC_VECTOR_LEVEL": "sub"}):
                entries = load_cpc_titles()
        self.assertEqual(entries[0]["code"], "H05B45/20")


class TestLoadCpcVectorsCache(unittest.TestCase):
    """The .npy cache is load-once per path: the sub tier is ~300MB and
    the matcher runs once per agent round — re-reading it every round
    would dominate request latency."""

    def test_caches_loaded_vectors_by_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            npy_path = os.path.join(tmp, "v.npy")
            with open(npy_path, "wb") as f:
                f.write(b"x")
            with patch("numpy.load", return_value="ARRAY") as mock_load:
                first = load_cpc_vectors(npy_path)
                second = load_cpc_vectors(npy_path)
        self.assertEqual(first, "ARRAY")
        self.assertIs(first, second)
        self.assertEqual(mock_load.call_count, 1)

    def test_no_arg_respects_level_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            npy_path = os.path.join(tmp, "v_sub.npy")
            with open(npy_path, "wb") as f:
                f.write(b"x")
            with patch("sources.long_task.cpc_semantic.CPC_VECTORS_SUB_NPY",
                       npy_path), \
                 patch.dict(os.environ, {"CPC_VECTOR_LEVEL": "sub"}), \
                 patch("numpy.load", return_value="ARRAY"):
                arr = load_cpc_vectors()
        self.assertEqual(arr, "ARRAY")


class TestMatchCpcCodesNumpyPath(unittest.TestCase):
    """The sub tier has ~150k entries — pure-Python cosine would take
    minutes per round, so the matcher takes a numpy fast path when
    available and keeps the pure loop as a no-numpy fallback."""

    def test_numpy_path_ranks_like_pure_python(self):
        import numpy as np
        entries = [{"code": "A", "title": "a"}, {"code": "B", "title": "b"},
                   {"code": "C", "title": "c"}]
        vectors = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        matches = match_cpc_codes(
            np.asarray([1.0, 0.0]), vectors, entries, top_k=2)
        self.assertEqual([m["code"] for m in matches], ["A", "B"])
        self.assertAlmostEqual(matches[0]["score"], 1.0)

    def test_zero_norm_vectors_score_zero(self):
        import numpy as np
        entries = [{"code": "A", "title": "a"}, {"code": "B", "title": "b"}]
        matches = match_cpc_codes(
            [1.0, 0.0], np.asarray([[0.0, 0.0], [1.0, 0.0]]), entries)
        self.assertEqual(matches[0]["code"], "B")
        self.assertEqual(matches[1]["code"], "A")
        self.assertEqual(matches[1]["score"], 0.0)

    def test_falls_back_to_pure_python_without_numpy(self):
        import sys
        entries = [{"code": "A", "title": "a"}, {"code": "B", "title": "b"}]
        with patch.dict(sys.modules, {"numpy": None}):
            matches = match_cpc_codes(
                [1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], entries, top_k=1)
        self.assertEqual(matches[0]["code"], "A")


if __name__ == "__main__":
    unittest.main()
