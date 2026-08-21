"""Tests for the CPC Master Classification File index builder.

The MCF text format has two record kinds per line:
  A — scheme-version records (skipped);
  B — patent records: B + patent number (5-9 digits) + 8-digit record
      index + CPC symbol + version/position token + trailing 0 0.
Parsing must be tolerant: fixed-width columns vary with number widths.
"""
import io
import os
import sqlite3
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from scripts.build_cpc_index import (
    _normalize_symbol,
    _parse_mcf_line,
    build_index,
)

B_LINE_9 = ("B21849611012650000E02F   3/844   20130101FI  0 0\r\n")
B_LINE_8 = ("B12345678011007234E02F   3/844   20130101FI  0 0\r\n")
B_LINE_SUB = ("B21849611012650000H05B45/20    20130101FI  0 0\r\n")
A_LINE = ("A           100000B68B   1/04    20130101FI  0 0\r\n")


class TestNormalizeSymbol(unittest.TestCase):
    def test_collapses_internal_spaces(self):
        self.assertEqual(_normalize_symbol("E02F   3/844"), "E02F3/844")

    def test_keeps_canonical_form(self):
        self.assertEqual(_normalize_symbol("H05B45/20"), "H05B45/20")

    def test_y_section_symbols(self):
        self.assertEqual(_normalize_symbol("Y10T 408/03"), "Y10T408/03")

    def test_invalid_returns_none(self):
        self.assertIsNone(_normalize_symbol("ZZ99X/1"))
        self.assertIsNone(_normalize_symbol(""))
        self.assertIsNone(_normalize_symbol(None))


class TestParseMcfLine(unittest.TestCase):
    def test_parses_zero_padded_patent_from_real_line(self):
        # real 2026-08 MCF line: 8-char prefix + 9-char zero-padded
        # patent number ("012650000" -> 12650000)
        result = _parse_mcf_line(B_LINE_9)
        self.assertEqual(result, ("12650000", "E02F3/844"))

    def test_parses_eight_digit_patent(self):
        result = _parse_mcf_line(B_LINE_8)
        self.assertEqual(result, ("11007234", "E02F3/844"))

    def test_parses_slash_symbol_without_group_digits(self):
        result = _parse_mcf_line(B_LINE_SUB)
        self.assertEqual(result, ("12650000", "H05B45/20"))

    def test_parses_patent_after_nine_char_index(self):
        # the real lines whose 9-char index ends in a non-zero digit —
        # an off-by-one parse produced "912627121" here; the patent is
        # the trailing 8 chars: 12,627,121 (a 2026 grant)
        line = ("B21849611912627121H05B45/20    20130101FI  0 0\r\n")
        self.assertEqual(_parse_mcf_line(line),
                         ("12627121", "H05B45/20"))

    def test_skips_scheme_a_lines(self):
        self.assertIsNone(_parse_mcf_line(A_LINE))

    def test_skips_junk(self):
        for line in ("", "garbage", "Babc", "C1234567801234567X/1"):
            self.assertIsNone(_parse_mcf_line(line))


def _synthetic_zip(tmp):
    path = os.path.join(tmp, "mcf.zip")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mcf/00000001.txt", "".join([
            A_LINE,
            B_LINE_9,
            B_LINE_SUB,
            "garbage line\n",
            B_LINE_9,  # duplicate patent+code — deduped
        ]))
        z.writestr("mcf/00050000.txt", "".join([
            B_LINE_8,
            "B21849611012650001E02F   3/764   20130101LI  0 0\r\n",
        ]))
    return path


class TestBuildIndex(unittest.TestCase):
    def test_builds_sqlite_index_from_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = _synthetic_zip(tmp)
            db_path = os.path.join(tmp, "cpc_index.db")
            stats = build_index(zip_path, db_path)
            conn = sqlite3.connect(db_path)
            rows = conn.execute(
                "SELECT cpc, patent FROM cpc_patents ORDER BY cpc").fetchall()
            conn.close()
        self.assertGreaterEqual(stats["patents"], 2)
        self.assertIn(("E02F3/764", "12650001"), rows)
        self.assertIn(("E02F3/844", "11007234"), rows)
        self.assertIn(("E02F3/844", "12650000"), rows)
        self.assertIn(("H05B45/20", "12650000"), rows)
        # duplicates deduped — one row per (cpc, patent)
        self.assertEqual(rows.count(("E02F3/844", "12650000")), 1)
        # A lines and junk never appear
        self.assertNotIn("B68B", [c for c, _ in rows])

    def test_success_renames_atomically_and_leaves_no_tmp(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = _synthetic_zip(tmp)
            db_path = os.path.join(tmp, "cpc_index.db")
            build_index(zip_path, db_path)
            self.assertTrue(os.path.exists(db_path))
            self.assertFalse(os.path.exists(db_path + ".tmp"))

    def test_cleans_up_leftover_tmp_from_killed_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = _synthetic_zip(tmp)
            db_path = os.path.join(tmp, "cpc_index.db")
            with open(db_path + ".tmp", "wb") as f:
                f.write(b"junk from a killed build")
            build_index(zip_path, db_path)
            conn = sqlite3.connect(db_path)
            count = conn.execute(
                "SELECT COUNT(*) FROM cpc_patents").fetchone()[0]
            conn.close()
            self.assertGreater(count, 0)
            self.assertFalse(os.path.exists(db_path + ".tmp"))

    def test_failed_build_never_leaves_final_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = os.path.join(tmp, "corrupt.zip")
            with open(zip_path, "wb") as f:
                f.write(b"not a zip archive")
            db_path = os.path.join(tmp, "cpc_index.db")
            with self.assertRaises(Exception):
                build_index(zip_path, db_path)
            # the final path must never hold a partial index
            self.assertFalse(os.path.exists(db_path))

    def test_missing_zip_raises(self):
        with self.assertRaises(Exception):
            build_index("/nonexistent/mcf.zip", "/tmp/never.db")


if __name__ == "__main__":
    unittest.main()
