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
    build_claims_payload,
    split_description_sections,
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


class TestSpecHandlerLogic(unittest.IsolatedAsyncioTestCase):
    async def test_spec_fetches_google_description_and_splits(self):
        fake_client = MagicMock()
        fake_client.query_description = AsyncMock(return_value=["para1", "para2"])
        fake_client.close = AsyncMock()

        from api_routes.patent_detail import _fetch_spec_text

        with patch(
            "sources.google_patents_client.GooglePatentsClient",
            return_value=fake_client,
        ):
            result = await _fetch_spec_text("uspto", "US12000123B2")

        self.assertEqual(len(result["sections"]), 1)
        self.assertIn("patents.google.com", result["source_url"])

    async def test_spec_raises_on_backend_failure(self):
        fake_client = MagicMock()
        fake_client.query_description = AsyncMock(
            side_effect=Exception("boom")
        )
        fake_client.close = AsyncMock()

        from api_routes.patent_detail import _fetch_spec_text, PatentDetailError

        with patch(
            "sources.google_patents_client.GooglePatentsClient",
            return_value=fake_client,
        ):
            with self.assertRaises(PatentDetailError):
                await _fetch_spec_text("uspto", "US12000123B2")
