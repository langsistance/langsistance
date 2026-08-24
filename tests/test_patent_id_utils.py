"""Tests for patent identifier normalization (kind-code stripping).

The naive "keep every digit" extraction mangles kind codes: US9019058B2
becomes "90190582" (the B2 digit leaks into the number), which then
404s/403s against the USPTO API.  extract_us_patent_digits() parses the
identifier shape instead so the bare number survives.
"""
import unittest

from sources.patent_id_utils import extract_us_patent_digits, kind_code_of


class TestExtractUsPatentDigits(unittest.TestCase):
    def test_patent_number_with_kind_code(self):
        # The production failure: kind code digit B2 leaks in.
        self.assertEqual(extract_us_patent_digits("US9019058B2"), "9019058")
        self.assertEqual(extract_us_patent_digits("US9019058B1"), "9019058")
        self.assertEqual(extract_us_patent_digits("US9019058A1"), "9019058")

    def test_patent_number_with_spaces_and_commas(self):
        self.assertEqual(extract_us_patent_digits("US 9,019,058 B2"), "9019058")
        self.assertEqual(extract_us_patent_digits("9,019,058"), "9019058")

    def test_bare_patent_number_unchanged(self):
        self.assertEqual(extract_us_patent_digits("9019058"), "9019058")
        self.assertEqual(extract_us_patent_digits("US9019058"), "9019058")

    def test_application_number_unchanged(self):
        self.assertEqual(extract_us_patent_digits("13850906"), "13850906")
        self.assertEqual(extract_us_patent_digits("17/027,484"), "17027484")

    def test_publication_number_with_kind_code(self):
        self.assertEqual(extract_us_patent_digits("US20250103146A1"), "20250103146")

    def test_eight_digit_grant_number_with_kind_code(self):
        # Post-2024 US grant numbers are 8 digits — kind code must not
        # extend them to 9.
        self.assertEqual(extract_us_patent_digits("US12345678B2"), "12345678")

    def test_reexamination_and_plant_patents(self):
        self.assertEqual(extract_us_patent_digits("USRE45678E1"), "45678")
        self.assertEqual(extract_us_patent_digits("USPP12345P3"), "12345")

    def test_lowercase_prefix(self):
        self.assertEqual(extract_us_patent_digits("us9019058b2"), "9019058")

    def test_empty_and_junk_input(self):
        self.assertEqual(extract_us_patent_digits(""), "")
        self.assertEqual(extract_us_patent_digits(None), "")
        # Unrecognizable shapes keep the legacy all-digits behavior —
        # never raises, never changes semantics for what did work.
        self.assertEqual(extract_us_patent_digits("abc"), "")


class TestKindCodeOf(unittest.TestCase):
    def test_returns_kind_code(self):
        self.assertEqual(kind_code_of("US9019058B2"), "B2")
        self.assertEqual(kind_code_of("US20250103146A1"), "A1")
        self.assertEqual(kind_code_of("US12345678B2"), "B2")
        self.assertEqual(kind_code_of("USRE45678E1"), "E1")

    def test_empty_when_no_kind_code(self):
        self.assertEqual(kind_code_of("9019058"), "")
        self.assertEqual(kind_code_of("13850906"), "")
        self.assertEqual(kind_code_of("US9019058"), "")
        self.assertEqual(kind_code_of("17/027,484"), "")

    def test_empty_for_empty_and_junk(self):
        self.assertEqual(kind_code_of(""), "")
        self.assertEqual(kind_code_of(None), "")
        self.assertEqual(kind_code_of("abc"), "")


if __name__ == "__main__":
    unittest.main()
