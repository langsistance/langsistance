"""Tests for patent detail endpoints (spec / claims)."""
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Environment shim: sources/user/passport initializes Firebase + Redis at
# import time and cannot load in the local test venv.  The functions under
# test never call it; stub the module so the route module imports cleanly.
_passport_stub = MagicMock()
_passport_stub.verify_firebase_token = MagicMock(return_value={"uid": "1"})
sys.modules["sources.user.passport"] = _passport_stub

from api_routes.patent_detail import (
    PatentDetailError,
    build_claims_payload,
    register_patent_detail_routes,
    split_claims_text,
    split_description_sections,
    _find_claims_document,
    _find_spec_document,
    _split_paragraphs,
    _strip_xml_tags,
)


class TestSplitDescriptionSections(unittest.TestCase):
    def test_chunks_paragraphs_without_headings(self):
        paras = [f"paragraph {i}" for i in range(40)]
        sections = split_description_sections(paras)
        self.assertEqual(len(sections), 3)  # 15 + 15 + 10
        self.assertEqual(sections[0]["heading"], "段落 1-15")
        self.assertEqual(len(sections[0]["paragraphs"]), 15)

    def test_uses_natural_headings_when_present(self):
        paras = [
            "技术领域", "本申请涉及电池。",
            "背景技术", "现有技术存在不足。",
            "发明内容", "提供一种新的结构。",
        ]
        sections = split_description_sections(paras)
        headings = [s["heading"] for s in sections]
        self.assertIn("技术领域", headings)
        self.assertIn("背景技术", headings)
        self.assertIn("发明内容", headings)

    def test_empty_paragraphs(self):
        self.assertEqual(split_description_sections([]), [])


class TestBuildClaimsPayload(unittest.TestCase):
    def test_marks_first_claim_independent(self):
        payload = build_claims_payload(["1. 一种机器人。", "2. 如权利要求1所述。"])
        self.assertEqual(payload["success"], True)
        self.assertEqual(len(payload["claims"]), 2)
        self.assertTrue(payload["claims"][0]["independent"])
        self.assertFalse(payload["claims"][1]["independent"])
        self.assertEqual(payload["claims"][0]["number"], 1)

    def test_empty_claims(self):
        self.assertEqual(build_claims_payload([]), {"success": False, "claims": []})

    def test_detects_dependent_openers(self):
        claims = [
            "1. A method comprising steps.",
            "2. The method of claim 1, further comprising a widget.",
            "3. The method according to claim 2, wherein the widget spins.",
            "4. A method according to any one of claims 1 to 3, wherein x.",
            "5. The system as claimed in claim 1.",
            "6. An independent apparatus unrelated to prior claims.",
        ]
        payload = build_claims_payload(claims)
        independent = [c["number"] for c in payload["claims"] if c["independent"]]
        self.assertEqual(independent, [1, 6])


class TestSplitClaimsText(unittest.TestCase):
    def test_splits_numbered_claims(self):
        text = (
            "HEADER NOISE\n\n"
            "1. A first claim body.\n\n"
            "2. The claim of claim 1.\n"
            "   continued line.\n\n"
            "3. A third claim.\n"
        )
        claims = split_claims_text(text)
        self.assertEqual(len(claims), 3)
        self.assertTrue(claims[0].startswith("A first claim body"))
        self.assertIn("continued line", claims[1])

    def test_ignores_text_without_claim_numbers(self):
        self.assertEqual(split_claims_text("just some text"), [])
        self.assertEqual(split_claims_text(""), [])

    def test_canceled_claims_are_kept(self):
        text = "1. (canceled)\n2. A real claim."
        claims = split_claims_text(text)
        self.assertEqual(len(claims), 2)
        self.assertIn("canceled", claims[0])


class TestSplitParagraphs(unittest.TestCase):
    def test_splits_on_blank_lines(self):
        text = "para one line\nmore\n\npara two\n\n\npara three"
        self.assertEqual(
            _split_paragraphs(text),
            ["para one line\nmore", "para two", "para three"],
        )

    def test_normalizes_carriage_returns_and_form_feeds(self):
        self.assertEqual(_split_paragraphs("a\r\nb\r\n\x0cc"), ["a\nb", "c"])


class TestStripXmlTags(unittest.TestCase):
    def test_strips_tags_and_unescapes(self):
        self.assertEqual(
            _strip_xml_tags("<p>Hello <b>world</b> &amp; co.</p>"),
            "Hello world & co.",
        )

    def test_plain_text_passes_through(self):
        self.assertEqual(_strip_xml_tags("plain text"), "plain text")


class TestDocumentSelection(unittest.TestCase):
    def _bag(self):
        return [
            {"documentCode": "BIB", "documentCodeDescriptionText": "Bibliographic Data Sheet"},
            {"documentCode": "SPEC", "documentCodeDescriptionText": "Specification"},
            {"documentCode": "CLM", "documentCodeDescriptionText": "Claims"},
            {"documentCode": "DRW", "documentCodeDescriptionText": "Drawings"},
        ]

    def test_finds_spec_document(self):
        doc = _find_spec_document(self._bag())
        self.assertEqual(doc["documentCode"], "SPEC")

    def test_finds_claims_document_by_code(self):
        doc = _find_claims_document(self._bag())
        self.assertEqual(doc["documentCode"], "CLM")

    def test_finds_claims_document_by_description_fallback(self):
        bag = [
            {"documentCode": "X", "documentCodeDescriptionText": "something"},
            {"documentCode": "X2", "documentCodeDescriptionText": "Amended Claims"},
        ]
        doc = _find_claims_document(bag)
        self.assertEqual(doc["documentCode"], "X2")

    def test_returns_none_when_missing(self):
        self.assertIsNone(_find_spec_document([]))
        self.assertIsNone(_find_claims_document([]))


class TestSpecHandlerLogic(unittest.IsolatedAsyncioTestCase):
    async def test_spec_uses_uspto_download_and_splits(self):
        from api_routes.patent_detail import _fetch_spec_text

        with patch(
            "sources.uspto_download.resolve_application_number",
            new=AsyncMock(return_value="18893954"),
        ), patch(
            "sources.uspto_download.fetch_document_bag",
            new=AsyncMock(return_value=[
                {"documentCode": "SPEC", "documentCodeDescriptionText": "Specification"},
            ]),
        ), patch(
            "sources.uspto_download.download_document_text",
            new=AsyncMock(return_value="技术领域\n\n本申请涉及电池。"),
        ):
            result = await _fetch_spec_text("uspto", "US12000123B2")

        self.assertEqual(len(result["sections"]), 1)
        self.assertIn("patents.google.com", result["source_url"])

    async def test_spec_raises_when_document_list_unavailable(self):
        from api_routes.patent_detail import _fetch_spec_text

        with patch(
            "sources.uspto_download.resolve_application_number",
            new=AsyncMock(return_value="18893954"),
        ), patch(
            "sources.uspto_download.fetch_document_bag",
            new=AsyncMock(return_value=None),
        ):
            with self.assertRaises(PatentDetailError):
                await _fetch_spec_text("uspto", "US12000123B2")

    async def test_spec_raises_when_spec_document_missing(self):
        from api_routes.patent_detail import _fetch_spec_text

        with patch(
            "sources.uspto_download.resolve_application_number",
            new=AsyncMock(return_value="18893954"),
        ), patch(
            "sources.uspto_download.fetch_document_bag",
            new=AsyncMock(return_value=[
                {"documentCode": "DRW", "documentCodeDescriptionText": "Drawings"},
            ]),
        ):
            with self.assertRaises(PatentDetailError):
                await _fetch_spec_text("uspto", "US12000123B2")

    async def test_claims_uses_uspto_claims_document(self):
        from api_routes.patent_detail import _fetch_claims

        claims_text = (
            "1. A first independent claim.\n"
            "2. The method of claim 1, further limited.\n"
            "3. 如权利要求1所述的方法。\n"
        )
        with patch(
            "sources.uspto_download.resolve_application_number",
            new=AsyncMock(return_value="18893954"),
        ), patch(
            "sources.uspto_download.fetch_document_bag",
            new=AsyncMock(return_value=[
                {"documentCode": "CLM", "documentCodeDescriptionText": "Claims"},
            ]),
        ), patch(
            "sources.uspto_download.download_document_text",
            new=AsyncMock(return_value=claims_text),
        ):
            result = await _fetch_claims("uspto", "US12000123B2")

        self.assertTrue(result["success"])
        self.assertEqual(len(result["claims"]), 3)
        self.assertTrue(result["claims"][0]["independent"])
        self.assertFalse(result["claims"][1]["independent"])
        self.assertFalse(result["claims"][2]["independent"])


class TestPatentDetailRoutes(unittest.TestCase):
    """Route-level regression: upstream misses must not produce 5xx.

    Cloudflare swaps origin 5xx responses for its own error page, which
    carries no Access-Control-Allow-Origin header — the browser then
    reports the response as a CORS failure instead of a readable error.
    Expected upstream misses return 200 + success:false instead.
    """

    @classmethod
    def setUpClass(cls):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.include_router(register_patent_detail_routes(MagicMock(), MagicMock()))
        cls.client = TestClient(app, raise_server_exceptions=False)

    def test_spec_returns_200_success_false_on_upstream_miss(self):
        with patch(
            "api_routes.patent_detail._fetch_spec_text",
            new=AsyncMock(side_effect=PatentDetailError("Patent not found (404)")),
        ):
            response = self.client.get(
                "/patent/uspto/US12000123B2/spec",
                headers={"Authorization": "Bearer test"},
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["success"])
        self.assertIn("unavailable", body["message"])

    def test_claims_returns_200_success_false_on_upstream_miss(self):
        with patch(
            "api_routes.patent_detail._fetch_claims",
            new=AsyncMock(side_effect=PatentDetailError("boom")),
        ):
            response = self.client.get(
                "/patent/uspto/US12000123B2/claims",
                headers={"Authorization": "Bearer test"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["success"])

    def test_spec_returns_200_success_true_with_sections(self):
        with patch(
            "api_routes.patent_detail._fetch_spec_text",
            new=AsyncMock(
                return_value={
                    "sections": [{"heading": "技术领域", "paragraphs": ["内容"]}],
                    "source_url": "https://patents.google.com/patent/US12000123B2",
                }
            ),
        ):
            response = self.client.get(
                "/patent/uspto/US12000123B2/spec",
                headers={"Authorization": "Bearer test"},
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertEqual(len(body["sections"]), 1)

    def test_unsupported_source_still_400(self):
        response = self.client.get(
            "/patent/unknown/US12000123B2/spec",
            headers={"Authorization": "Bearer test"},
        )
        self.assertEqual(response.status_code, 400)
