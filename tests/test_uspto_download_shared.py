"""Tests for the shared USPTO spec download module (sources.long_task.uspto_download).

Kept separate from tests/test_uspto_download.py, which covers the unrelated
sources.uspto_download module (16 tests).
"""
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from sources.long_task.uspto_download import (
    download_uspto_patent_text,
    normalize_app_number,
)


class TestNormalizeAppNumber(unittest.TestCase):
    def test_strips_us_prefix_and_punctuation(self):
        self.assertEqual(normalize_app_number("US 19/511,555"), "19511555")
        self.assertEqual(normalize_app_number("19511555"), "19511555")

    def test_too_short_returns_empty(self):
        self.assertEqual(normalize_app_number("12345"), "")
        self.assertEqual(normalize_app_number(""), "")


class TestDownloadUsptoPatentText(unittest.IsolatedAsyncioTestCase):
    async def test_no_documents_returns_none_none(self):
        with patch("sources.long_task.uspto_download._uspto_get_with_retry",
                   new=AsyncMock(return_value=_resp(200, {"documentBag": []}))):
            text, binary = await download_uspto_patent_text("19511555")
        self.assertEqual((text, binary), (None, None))

    async def test_spec_text_extracted(self):
        doc_list = {"documentBag": [{
            "documentCode": "SPEC",
            "downloadOptionBag": [{
                "downloadUrl": "https://api.uspto.gov/api/v1/download/applications/19511555/x/doc.docx",
                "mimeTypeIdentifier": "MS_WORD",
            }],
        }]}
        # The verbatim source only accepts extracted text longer than 200 chars,
        # so the mocked body must exceed that threshold.
        spec_body = "SPEC TEXT BODY " * 30
        with patch("sources.long_task.uspto_download._uspto_get_with_retry",
                   new=AsyncMock(return_value=_resp(200, doc_list))):
            with patch("sources.long_task.uspto_download._download_uspto_spec_with_redirect",
                       new=AsyncMock(return_value=(spec_body, None))):
                text, binary = await download_uspto_patent_text("19511555")
        self.assertIn("SPEC TEXT BODY", text)
        self.assertIsNone(binary)

    async def test_binary_fallback_when_text_extraction_empty(self):
        doc_list = {"documentBag": [{
            "documentCode": "SPEC",
            "downloadOptionBag": [{
                "downloadUrl": "https://api.uspto.gov/api/v1/download/applications/19511555/x.pdf",
                "mimeTypeIdentifier": "PDF",
            }],
        }]}
        with patch("sources.long_task.uspto_download._uspto_get_with_retry",
                   new=AsyncMock(return_value=_resp(200, doc_list))):
            with patch("sources.long_task.uspto_download._download_uspto_spec_with_redirect",
                       new=AsyncMock(return_value=(None, b"PDFBYTES"))):
                text, binary = await download_uspto_patent_text("19511555")
        self.assertIsNone(text)
        self.assertEqual(binary, b"PDFBYTES")

    async def test_invalid_app_number_returns_none_none(self):
        text, binary = await download_uspto_patent_text("abc")
        self.assertEqual((text, binary), (None, None))

    async def test_internal_error_never_raises(self):
        with patch("sources.long_task.uspto_download._uspto_get_with_retry",
                   new=AsyncMock(side_effect=RuntimeError("boom"))):
            text, binary = await download_uspto_patent_text("19511555")
        self.assertEqual((text, binary), (None, None))


def _resp(status, data):
    resp = MagicMock()
    resp.status_code = status
    resp.text = str(data) if not isinstance(data, dict) else ""
    resp.content = b""
    if isinstance(data, dict):
        resp.json = lambda: data
    return resp


if __name__ == "__main__":
    unittest.main()
