"""Tests for the long task submit endpoint helpers."""
import sys
from unittest.mock import MagicMock

# Test-environment shim: sources.user.passport initializes Firebase+Redis at
# import time and cannot load in the local venv. Mask it — the functions
# under test never call passport.
_passport_stub = MagicMock()
_passport_stub.verify_firebase_token = MagicMock(return_value={"uid": "1"})
sys.modules["sources.user.passport"] = _passport_stub

import unittest

from api_routes.long_task import _normalize_submit_patent_id


class TestNormalizeSubmitPatentId(unittest.TestCase):
    def test_strips_us_prefix(self):
        self.assertEqual(
            _normalize_submit_patent_id(" US17638216 ", "prosecution"),
            "17638216",
        )

    def test_prosecution_requires_exactly_8_digits(self):
        with self.assertRaises(ValueError):
            _normalize_submit_patent_id("US12000123B2", "prosecution")
        with self.assertRaises(ValueError):
            _normalize_submit_patent_id("12345", "prosecution")

    def test_family_accepts_publication_numbers(self):
        self.assertEqual(
            _normalize_submit_patent_id("US12000123B2", "family"),
            "US12000123B2",
        )

    def test_rejects_garbage(self):
        with self.assertRaises(ValueError):
            _normalize_submit_patent_id("", "family")
        with self.assertRaises(ValueError):
            _normalize_submit_patent_id("<>script", "family")

    def test_rejects_unknown_scenario(self):
        with self.assertRaises(ValueError):
            _normalize_submit_patent_id("17638216", "batch")
