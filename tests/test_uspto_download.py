"""Tests for the USPTO download helpers used by patent_detail (spec/claims).

HTTP is exercised through the module's internal seams (_request_json /
_download) so no network access happens.
"""
import sys
import unittest
from unittest.mock import AsyncMock, patch

from sources.uspto_download import (
    UsptoHttpResponse,
    download_document_text,
    fetch_document_bag,
    resolve_application_number,
)


class TestFetchDocumentBag(unittest.IsolatedAsyncioTestCase):
    async def test_returns_bag_list(self):
        with patch(
            "sources.uspto_download._request_json",
            new=AsyncMock(return_value={"documentBag": [{"documentCode": "SPEC"}]}),
        ):
            bag = await fetch_document_bag("18893954")
        self.assertEqual(bag, [{"documentCode": "SPEC"}])

    async def test_returns_empty_list_for_application_without_documents(self):
        with patch(
            "sources.uspto_download._request_json",
            new=AsyncMock(return_value={"documentBag": []}),
        ):
            self.assertEqual(await fetch_document_bag("18893954"), [])

    async def test_returns_none_when_api_fails(self):
        with patch(
            "sources.uspto_download._request_json",
            new=AsyncMock(return_value=None),
        ):
            self.assertIsNone(await fetch_document_bag("18893954"))


class TestResolveApplicationNumber(unittest.IsolatedAsyncioTestCase):
    async def test_application_number_with_documents_resolves_directly(self):
        with patch(
            "sources.uspto_download.fetch_document_bag",
            new=AsyncMock(return_value=[{"documentCode": "SPEC"}]),
        ):
            self.assertEqual(
                await resolve_application_number("18893954"), "18893954"
            )

    async def test_patent_number_resolves_via_search(self):
        with patch(
            "sources.uspto_download.fetch_document_bag",
            new=AsyncMock(return_value=[]),
        ), patch(
            "sources.uspto_download._search_application_number_by_patent_number",
            new=AsyncMock(return_value="18893954"),
        ):
            self.assertEqual(
                await resolve_application_number("12429341"), "18893954"
            )

    async def test_design_patent_keeps_d_prefix_in_search(self):
        # Design patents are numbered D754082 — the D prefix must survive
        # into the patentNumber search (the documents endpoint only takes
        # pure-digit application numbers).
        fetch_mock = AsyncMock(return_value=[])
        search_mock = AsyncMock(return_value="29754082")
        with patch(
            "sources.uspto_download.fetch_document_bag", new=fetch_mock
        ), patch(
            "sources.uspto_download._search_application_number_by_patent_number",
            new=search_mock,
        ):
            self.assertEqual(
                await resolve_application_number("D754082"), "29754082"
            )
        fetch_mock.assert_awaited_once_with("754082")
        search_mock.assert_awaited_once_with("D754082")

    async def test_design_patent_falls_back_to_bare_digits_search(self):
        # Some sources store design numbers without the D prefix.
        search_mock = AsyncMock(side_effect=[None, "29754082"])
        with patch(
            "sources.uspto_download.fetch_document_bag",
            new=AsyncMock(return_value=[]),
        ), patch(
            "sources.uspto_download._search_application_number_by_patent_number",
            new=search_mock,
        ):
            self.assertEqual(
                await resolve_application_number("D754082"), "29754082"
            )

    async def test_raises_when_unresolvable(self):
        with patch(
            "sources.uspto_download.fetch_document_bag",
            new=AsyncMock(return_value=None),
        ), patch(
            "sources.uspto_download._search_application_number_by_patent_number",
            new=AsyncMock(return_value=None),
        ):
            with self.assertRaises(ValueError):
                await resolve_application_number("99999999")

    async def test_rejects_short_ids(self):
        with self.assertRaises(ValueError):
            await resolve_application_number("123")


class TestDownloadDocumentText(unittest.IsolatedAsyncioTestCase):
    DOC = {
        "downloadOptionBag": [
            {
                "mimeTypeIdentifier": "PDF",
                "downloadUrl": "https://api.uspto.gov/api/v1/download/1.pdf",
            }
        ]
    }

    async def test_extracts_text_from_binary_document(self):
        with patch(
            "sources.uspto_download._download",
            new=AsyncMock(
                return_value=UsptoHttpResponse(
                    status_code=200,
                    content=b"%PDF-1.4 fake",
                    content_type="application/pdf",
                )
            ),
        ), patch(
            "sources.long_task.text_extractor.extract_text_from_binary",
            return_value="  extracted specification text  ",
        ):
            text = await download_document_text(self.DOC)
        self.assertEqual(text, "extracted specification text")

    async def test_follows_in_body_redirect(self):
        redirect_body = "Please use redirect URL: https://api.uspto.gov/x.pdf"
        with patch(
            "sources.uspto_download._download",
            new=AsyncMock(
                side_effect=[
                    UsptoHttpResponse(
                        status_code=200,
                        content=redirect_body.encode(),
                        content_type="text/plain",
                    ),
                    UsptoHttpResponse(
                        status_code=200,
                        content=b"%PDF-1.4 fake",
                        content_type="application/pdf",
                    ),
                ]
            ),
        ), patch(
            "sources.long_task.text_extractor.extract_text_from_binary",
            return_value="redirected text",
        ):
            text = await download_document_text(self.DOC)
        self.assertEqual(text, "redirected text")

    async def test_returns_empty_on_http_failure(self):
        with patch(
            "sources.uspto_download._download",
            new=AsyncMock(
                return_value=UsptoHttpResponse(
                    status_code=404, content=b"", content_type="text/html"
                )
            ),
        ):
            self.assertEqual(await download_document_text(self.DOC), "")

    async def test_scanned_pdf_degrades_to_empty(self):
        # Image-only scanned PDFs have no text layer — extraction returns
        # empty (the claims endpoint serves such documents as a viewer URL).
        with patch(
            "sources.uspto_download._download",
            new=AsyncMock(
                return_value=UsptoHttpResponse(
                    status_code=200,
                    content=b"%PDF-1.4 scanned pages",
                    content_type="application/octet-stream",
                )
            ),
        ), patch(
            "sources.long_task.text_extractor.extract_text_from_binary",
            return_value=None,
        ):
            self.assertEqual(await download_document_text(self.DOC), "")

    async def test_custom_mime_order_selects_xml_url(self):
        doc = {
            "downloadOptionBag": [
                {"mimeTypeIdentifier": "PDF", "downloadUrl": "https://api.uspto.gov/api/v1/download/x.pdf"},
                {"mimeTypeIdentifier": "XML", "downloadUrl": "https://api.uspto.gov/api/v1/download/x.xmlarchive"},
            ]
        }
        download_mock = AsyncMock(
            return_value=UsptoHttpResponse(status_code=200, content=b"", content_type="text/xml")
        )
        with patch("sources.uspto_download._download", new=download_mock):
            await download_document_text(doc, mime_order=("XML", "application/xml", "text/xml"))
        download_mock.assert_awaited_once()
        self.assertIn(".xmlarchive", download_mock.await_args.args[0])

    async def test_returns_empty_when_doc_has_no_url(self):
        self.assertEqual(await download_document_text({}), "")

    async def test_treats_nul_byte_binary_mislabelled_as_xml(self):
        # USPTO xmlarchive tars can be labelled application/xml; NUL bytes
        # reveal the binary and must route to the binary extractor.
        with patch(
            "sources.uspto_download._download",
            new=AsyncMock(
                return_value=UsptoHttpResponse(
                    status_code=200,
                    content=b"\x00\x01binary tar",
                    content_type="application/xml",
                )
            ),
        ), patch(
            "sources.long_task.text_extractor.extract_text_from_binary",
            return_value="xml archive text",
        ):
            text = await download_document_text(self.DOC)
        self.assertEqual(text, "xml archive text")


if __name__ == "__main__":
    unittest.main()
