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
    _find_claims_document,
    _find_spec_document,
    _is_cn_patent_id,
    _parse_claims_xml,
    _strip_claim_status_markers,
    _strip_document_noise,
    _strip_xml_tags,
)


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

    def test_dependent_detection_survives_amendment_markers(self):
        claims = [
            "1. A welding contact tip.",
            "2. (original) The contact tip of claim 1, comprising a brush.",
            "3. (previously presented) The method of claim 11, comprising forming.",
        ]
        payload = build_claims_payload(claims)
        self.assertTrue(payload["claims"][0]["independent"])
        self.assertFalse(payload["claims"][1]["independent"])
        self.assertFalse(payload["claims"][2]["independent"])
        # Markers stripped from the displayed text
        self.assertTrue(payload["claims"][1]["text"].startswith("The contact tip of claim 1"))

    def test_canceled_claims_are_marked_not_independent(self):
        claims = ["1. A widget.", "2. (canceled)", "3. Another widget."]
        payload = build_claims_payload(claims)
        self.assertEqual(payload["claims"][1]["status"], "canceled")
        self.assertFalse(payload["claims"][1]["independent"])
        self.assertEqual(payload["claims"][0]["status"], "active")
        self.assertEqual(payload["claims"][2]["status"], "active")


class TestSplitUnnumberedClaims(unittest.TestCase):
    def test_splits_paragraph_claims_by_starters(self):
        from api_routes.patent_detail import split_unnumbered_claims
        text = (
            "What is claimed is:\n\n"
            "A system for generating automated post-mission logs for an "
            "unmanned vehicle, the system comprising:\n\n"
            "a data collection module configured to receive flight telemetry "
            "data, operator inputs, and maintenance status information from an "
            "unmanned vehicle;\n\n"
            "a preprocessing module configured to normalize the telemetry data;\n\n"
            "The system of claim 1, wherein the preprocessing module further "
            "filters noise.\n\n"
            "An apparatus comprising a widget and a spring.\n"
        )
        claims = split_unnumbered_claims(text)
        self.assertEqual(len(claims), 3)
        self.assertTrue(claims[0].startswith("A system for"))
        self.assertIn("data collection module", claims[0])
        self.assertIn("preprocessing module", claims[0])
        self.assertTrue(claims[1].startswith("The system of claim 1"))
        self.assertTrue(claims[2].startswith("An apparatus comprising"))

    def test_continuation_paragraphs_join_previous_claim(self):
        from api_routes.patent_detail import split_unnumbered_claims
        text = (
            "A method comprising:\n\n"
            "providing a substrate;\n\n"
            "depositing a layer on the substrate;\n\n"
            "wherein the layer is heated.\n\n"
            "A second independent method.\n"
        )
        claims = split_unnumbered_claims(text)
        self.assertEqual(len(claims), 2)
        self.assertIn("depositing a layer", claims[0])
        self.assertIn("wherein the layer is heated", claims[0])

    def test_splits_docx_single_newline_paragraphs(self):
        # DOCX extraction emits one paragraph per line (no blank lines).
        from api_routes.patent_detail import split_unnumbered_claims
        text = (
            "What is claimed is:\n"
            "A system for generating automated post-mission logs.\n"
            "a data collection module;\n"
            "The system of claim 1, further comprising a lens.\n"
            "An apparatus with a spring.\n"
        )
        claims = split_unnumbered_claims(text)
        self.assertEqual(len(claims), 3)
        self.assertTrue(claims[0].startswith("A system for"))
        self.assertIn("data collection module", claims[0])
        self.assertTrue(claims[1].startswith("The system of claim 1"))
        self.assertTrue(claims[2].startswith("An apparatus"))

    def test_empty_text(self):
        from api_routes.patent_detail import split_unnumbered_claims
        self.assertEqual(split_unnumbered_claims(""), [])


class TestStripClaimStatusMarkers(unittest.TestCase):
    def test_strips_original_marker(self):
        text, status = _strip_claim_status_markers(
            "(original) The contact tip of claim 1, comprising a widget."
        )
        self.assertEqual(status, "active")
        self.assertTrue(text.startswith("The contact tip of claim 1"))

    def test_strips_previously_presented_marker(self):
        text, status = _strip_claim_status_markers(
            "(previously presented) A method for making a welding device."
        )
        self.assertEqual(status, "active")
        self.assertTrue(text.startswith("A method for"))

    def test_detects_canceled_claims(self):
        text, status = _strip_claim_status_markers("(canceled)")
        self.assertEqual(status, "canceled")
        self.assertEqual(text, "")

    def test_no_marker_passes_through(self):
        text, status = _strip_claim_status_markers("A method comprising steps.")
        self.assertEqual((text, status), ("A method comprising steps.", "active"))


class TestStripDocumentNoise(unittest.TestCase):
    def test_removes_page_headers_and_footers(self):
        raw = (
            "1. A first claim.\n\n"
            "Page 2 of 12\n\n"
            "Serial No. 12/098,926\n\n"
            "Response to Office Action\n\n"
            "Mailed on November 2, 2011\n\n"
            "2. A second claim.\n"
        )
        cleaned = _strip_document_noise(raw)
        self.assertNotIn("Page 2 of 12", cleaned)
        self.assertNotIn("Serial No.", cleaned)
        self.assertNotIn("Mailed on", cleaned)
        self.assertIn("1. A first claim.", cleaned)
        self.assertIn("2. A second claim.", cleaned)

    def test_removes_amendments_preamble(self):
        raw = (
            "AMENDMENTS TO THE CLAIMS\n"
            "The following is a complete listing of the claims.\n\n"
            "1. A real claim.\n"
        )
        cleaned = _strip_document_noise(raw)
        self.assertNotIn("AMENDMENTS", cleaned)
        self.assertNotIn("complete listing", cleaned)
        self.assertIn("1. A real claim.", cleaned)


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


class TestParseClaimsXml(unittest.TestCase):
    CLAIMS_XML = """<?xml version="1.0"?>
        <us-patent-grant>
          <claims id="claims">
            <claim id="CLM-00001" num="00001">
              <claim-text>First claim text.</claim-text>
            </claim>
            <claim id="CLM-00002" num="00002">
              <claim-text>Second claim <b>with markup</b> here.</claim-text>
            </claim>
          </claims>
        </us-patent-grant>
    """

    def test_parses_claims_with_num_attributes(self):
        claims = _parse_claims_xml(self.CLAIMS_XML)
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0]["number"], 1)
        self.assertEqual(claims[0]["text"], "First claim text.")
        self.assertEqual(claims[1]["number"], 2)
        self.assertEqual(claims[1]["text"], "Second claim with markup here.")

    def test_returns_none_for_plain_text(self):
        self.assertIsNone(_parse_claims_xml("1. A plain text claim."))
        self.assertIsNone(_parse_claims_xml(""))

    def test_returns_none_when_no_claim_elements(self):
        self.assertIsNone(
            _parse_claims_xml("<specification><p>no claims here</p></specification>")
        )

    def test_number_falls_back_to_sequence(self):
        xml = (
            "<claims>"
            "<claim><claim-text>one</claim-text></claim>"
            "<claim num='00003'><claim-text>three</claim-text></claim>"
            "</claims>"
        )
        claims = _parse_claims_xml(xml)
        self.assertEqual([c["number"] for c in claims], [1, 3])

    def test_parses_xml_with_doctype_declaration(self):
        # USPTO CLM.XML carries DOCTYPE/ENTITY declarations that make a
        # plain ElementTree parse fail — the parser must strip them.
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE us-patent-application PUBLIC "-//USPTO//DTD CLM 1.0//EN" "USPTO-CLM-1.0.dtd">\n'
            "<us-patent-application>"
            "<claims><claim num='00001'><claim-text>First claim text.</claim-text></claim>"
            "<claim num='00002'><claim-text>Second claim text.</claim-text></claim>"
            "</claims></us-patent-application>"
        )
        claims = _parse_claims_xml(xml)
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[1]["number"], 2)

    def test_parses_st96_vastec_claims_schema(self):
        # Modern USPTO CLM.XML (ST.96 VASTEC): namespaced <uspat:Claim>,
        # number in a <pat:ClaimNumber> child, text in <uspat:ClaimText>
        # segments (no hyphen!), status in <uspat:ClaimStatusCategory>.
        xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<uspat:ClaimsDocument xmlns:uspat="urn:us:gov:doc:uspto:patent" '
            'xmlns:pat="http://www.wipo.int/standards/XMLSchema/ST96/Patent" '
            'xmlns:uscom="urn:us:gov:doc:uspto:common" '
            'xmlns:com="http://www.wipo.int/standards/XMLSchema/ST96/Common">'
            '<uspat:Claims com:id="CLM-00000">'
            '<uspat:Claim com:id="CLM-00001">'
            '<pat:ClaimNumber>1</pat:ClaimNumber>'
            '<uspat:ClaimText>1. (Previously Presented) A display device comprising:</uspat:ClaimText>'
            '<uspat:ClaimText>a first light emission area;</uspat:ClaimText>'
            '<uspat:ClaimStatusCategory>Previously presented</uspat:ClaimStatusCategory>'
            '</uspat:Claim>'
            '<uspat:Claim com:id="CLM-00002">'
            '<pat:ClaimNumber>2</pat:ClaimNumber>'
            '<uspat:ClaimText>2. (Previously Presented) The display device of claim 1, further comprising:</uspat:ClaimText>'
            '<uspat:ClaimText>7C2B18MM\\261532 Amendment to 2025-09-10 FOA 4937-</uspat:ClaimText>'
            '<uspat:ClaimStatusCategory>Previously presented</uspat:ClaimStatusCategory>'
            '</uspat:Claim>'
            '</uspat:Claims>'
            '</uspat:ClaimsDocument>'
        )
        claims = _parse_claims_xml(xml)
        self.assertEqual(len(claims), 2)
        self.assertEqual([c["number"] for c in claims], [1, 2])
        self.assertIn("display device", claims[0]["text"])
        self.assertIn("light emission area", claims[0]["text"])
        # OCR page-footer garbage segments are dropped
        self.assertNotIn("FOA", claims[1]["text"])
        self.assertNotIn("7C2B18MM", claims[1]["text"])

    def test_st96_claims_flow_into_payload_cleanly(self):
        xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<uspat:ClaimsDocument xmlns:uspat="urn:us:gov:doc:uspto:patent" '
            'xmlns:pat="http://www.wipo.int/standards/XMLSchema/ST96/Patent">'
            '<uspat:Claims>'
            '<uspat:Claim>'
            '<pat:ClaimNumber>1</pat:ClaimNumber>'
            '<uspat:ClaimText>1. (Previously Presented) A display device comprising:</uspat:ClaimText>'
            '<uspat:ClaimText>a first light emission area.</uspat:ClaimText>'
            '</uspat:Claim>'
            '<uspat:Claim>'
            '<pat:ClaimNumber>2</pat:ClaimNumber>'
            '<uspat:ClaimText>2. (Previously Presented) The display device of claim 1, further comprising a lens.</uspat:ClaimText>'
            '</uspat:Claim>'
            '</uspat:Claims>'
            '</uspat:ClaimsDocument>'
        )
        payload = build_claims_payload([c["text"] for c in _parse_claims_xml(xml)])
        self.assertTrue(payload["claims"][0]["independent"])
        self.assertFalse(payload["claims"][1]["independent"])
        # Number + marker stripped from the displayed text
        self.assertTrue(payload["claims"][0]["text"].startswith("A display device"))
        self.assertTrue(payload["claims"][1]["text"].startswith("The display device"))

    def test_skips_claims_without_text(self):
        xml = (
            "<claims>"
            "<claim num='00001'><claim-text>real</claim-text></claim>"
            "<claim num='00002'><claim-text>   </claim-text></claim>"
            "</claims>"
        )
        claims = _parse_claims_xml(xml)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["text"], "real")

    def test_parses_design_patent_claim_without_claim_elements(self):
        # Design patents (D-numbers) carry the single claim as a plain
        # paragraph — no <Claim> elements exist in the CLM.XML.
        xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<uspat:ClaimsDocument xmlns:uscom="urn:us:gov:doc:uspto:common" '
            'xmlns:uspat="urn:us:gov:doc:uspto:patent" '
            'xmlns:com="http://www.wipo.int/standards/XMLSchema/ST96/Common">'
            '<uspat:DocumentMetadata><uscom:DocumentCode>CLM</uscom:DocumentCode></uspat:DocumentMetadata>'
            '<uscom:Heading>I <com:Del>CL</com:Del>laim:</uscom:Heading>'
            '<uscom:P com:pNumber="0">The ornamental design for the False Eyelashes, '
            'as shown and described.Page 3 of 6</uscom:P>'
            '<uspat:Claims com:id="CLM-00000">'
            '<uscom:P com:pNumber="1">Application No.: 29/967,887 '
            'Response to Office Action dated December 4, 2024. '
            'The dashed broken lines in the figures depict portions that '
            'form no part of the claimed design.</uscom:P>'
            '</uspat:Claims>'
            '</uspat:ClaimsDocument>'
        )
        claims = _parse_claims_xml(xml)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["number"], 1)
        self.assertEqual(
            claims[0]["text"],
            "The ornamental design for the False Eyelashes, as shown and described.",
        )
        # The amendment-status paragraph is NOT a claim
        self.assertNotIn("Application No.", claims[0]["text"])

    def test_design_claim_strips_page_number_remnants(self):
        # OCR sometimes glues the page number to the claim text:
        # "as shown and described.2" / "described.Page 2 of 6"
        xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<uspat:ClaimsDocument xmlns:uscom="urn:us:gov:doc:uspto:common" '
            'xmlns:uspat="urn:us:gov:doc:uspto:patent" '
            'xmlns:com="http://www.wipo.int/standards/XMLSchema/ST96/Common">'
            '<uscom:P com:pNumber="0">The ornamental design for a False Eyelash, '
            'as shown and described.2</uscom:P>'
            '</uspat:ClaimsDocument>'
        )
        claims = _parse_claims_xml(xml)
        self.assertEqual(len(claims), 1)
        self.assertEqual(
            claims[0]["text"],
            "The ornamental design for a False Eyelash, as shown and described.",
        )


class TestStripXmlTags(unittest.TestCase):
    def test_strips_tags_and_unescapes(self):
        self.assertEqual(
            _strip_xml_tags("<p>Hello <b>world</b> &amp; co.</p>"),
            "Hello world & co.",
        )

    def test_plain_text_passes_through(self):
        self.assertEqual(_strip_xml_tags("plain text"), "plain text")

    def test_preserves_newlines_for_block_elements(self):
        # Closing tags of claim/paragraph elements must become newlines so
        # numbered-claim splitting still works after tag stripping.
        xml = (
            "<claims><claim num='1'><claim-text>1. First claim text</claim-text></claim>"
            "<claim num='2'><claim-text>2. Second claim text</claim-text></claim></claims>"
        )
        stripped = _strip_xml_tags(xml)
        self.assertTrue(stripped.startswith("1. First claim text"))
        # Line-start numbered claims survive — exactly what
        # split_claims_text matches.
        from api_routes.patent_detail import split_claims_text
        self.assertEqual(
            split_claims_text(stripped),
            ["First claim text", "Second claim text"],
        )


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


class TestIsCnPatentId(unittest.TestCase):
    def test_publication_numbers(self):
        self.assertTrue(_is_cn_patent_id("CN213905456U"))
        self.assertTrue(_is_cn_patent_id("CN112345678A"))
        self.assertTrue(_is_cn_patent_id("CN109830778B"))

    def test_application_numbers(self):
        self.assertTrue(_is_cn_patent_id("CN202022899373.0"))

    def test_non_cn_ids_are_false(self):
        self.assertFalse(_is_cn_patent_id("18893954"))
        self.assertFalse(_is_cn_patent_id("US12000123B2"))
        self.assertFalse(_is_cn_patent_id(""))
        self.assertFalse(_is_cn_patent_id("CN"))


class TestSpecHandlerLogic(unittest.IsolatedAsyncioTestCase):
    async def test_cn_patent_id_routes_to_baiten_spec(self):
        # 事故 (2026-08-29): 前端持久化丢 source 列后,CN 专利说明书请求
        # 打到 /patent/uspto/CN213905456U/spec,CN 号被当成美国申请号解析
        # (403/404)而失败。CN 专利号必须无条件走佰腾下载,不依赖调用方
        # 声称的 source。
        from api_routes.patent_detail import _fetch_spec_pdf

        with patch(
            "api_routes.patent_detail._fetch_baiten_spec",
            new=AsyncMock(return_value={
                "pdf_url": ("https://api-test.copiioai.com/baiten/download"
                            "?pub_num=CN213905456U&pub_date=20210806"),
            }),
        ) as mock_baiten, patch(
            "sources.uspto_download.resolve_application_number",
            new=AsyncMock(return_value="213905456"),
        ) as mock_resolve:
            result = await _fetch_spec_pdf(
                "uspto", "CN213905456U", "20210806")
        self.assertEqual(
            result["pdf_url"],
            "https://api-test.copiioai.com/baiten/download"
            "?pub_num=CN213905456U&pub_date=20210806",
        )
        mock_baiten.assert_awaited_once()
        mock_resolve.assert_not_awaited()  # 绝不打 USPTO

    async def test_cn_patent_id_routes_to_baiten_claims(self):
        from api_routes.patent_detail import _fetch_claims

        with patch(
            "api_routes.patent_detail._fetch_baiten_claims",
            new=AsyncMock(return_value={
                "success": True,
                "claims": [{"number": 1, "text": "一种电池冷却板。",
                            "status": "active", "independent": True}],
            }),
        ) as mock_baiten, patch(
            "sources.uspto_download.resolve_application_number",
            new=AsyncMock(return_value="213905456"),
        ) as mock_resolve:
            result = await _fetch_claims("uspto", "CN213905456U")
        self.assertTrue(result["success"])
        mock_baiten.assert_awaited_once()
        mock_resolve.assert_not_awaited()

    async def test_us_patent_id_keeps_uspto_route(self):
        # 防呆不能误伤 US 专利:纯数字/美国号仍走 USPTO 解析。
        from api_routes.patent_detail import _fetch_spec_pdf

        with patch(
            "api_routes.patent_detail._fetch_baiten_spec",
            new=AsyncMock(return_value={"pdf_url": "baiten"}),
        ) as mock_baiten, patch(
            "sources.uspto_download.resolve_application_number",
            new=AsyncMock(return_value="18893954"),
        ) as mock_resolve, patch(
            "sources.uspto_download.fetch_document_bag",
            new=AsyncMock(return_value=[
                {"documentCode": "SPEC",
                 "documentCodeDescriptionText": "Specification"},
            ]),
        ), patch(
            "sources.long_task.text_extractor.get_download_url_from_doc",
            return_value="https://api.uspto.gov/api/v1/download/spec.pdf",
        ), patch(
            "sources.dynamic_tool_params._build_uspto_download_proxy_url",
            return_value="https://api-test.copiioai.com/uspto/download?url=encoded",
        ):
            result = await _fetch_spec_pdf("uspto", "18893954")
        self.assertIn("/uspto/download", result["pdf_url"])
        mock_baiten.assert_not_awaited()
        mock_resolve.assert_awaited_once()
    async def test_spec_returns_proxy_pdf_url(self):
        from api_routes.patent_detail import _fetch_spec_pdf

        with patch(
            "sources.uspto_download.resolve_application_number",
            new=AsyncMock(return_value="18893954"),
        ), patch(
            "sources.uspto_download.fetch_document_bag",
            new=AsyncMock(return_value=[
                {"documentCode": "SPEC", "documentCodeDescriptionText": "Specification"},
            ]),
        ), patch(
            "sources.long_task.text_extractor.get_download_url_from_doc",
            return_value="https://api.uspto.gov/api/v1/download/spec.pdf",
        ), patch(
            "sources.dynamic_tool_params._build_uspto_download_proxy_url",
            return_value="https://api-test.copiioai.com/uspto/download?url=encoded",
        ):
            result = await _fetch_spec_pdf("uspto", "US12000123B2")

        self.assertEqual(
            result["pdf_url"],
            "https://api-test.copiioai.com/uspto/download?url=encoded",
        )

    async def test_spec_raises_when_document_list_unavailable(self):
        from api_routes.patent_detail import _fetch_spec_pdf

        with patch(
            "sources.uspto_download.resolve_application_number",
            new=AsyncMock(return_value="18893954"),
        ), patch(
            "sources.uspto_download.fetch_document_bag",
            new=AsyncMock(return_value=None),
        ):
            with self.assertRaises(PatentDetailError):
                await _fetch_spec_pdf("uspto", "US12000123B2")

    async def test_spec_raises_when_spec_document_missing(self):
        from api_routes.patent_detail import _fetch_spec_pdf

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
                await _fetch_spec_pdf("uspto", "US12000123B2")

    async def test_spec_raises_when_no_pdf_url(self):
        from api_routes.patent_detail import _fetch_spec_pdf

        with patch(
            "sources.uspto_download.resolve_application_number",
            new=AsyncMock(return_value="18893954"),
        ), patch(
            "sources.uspto_download.fetch_document_bag",
            new=AsyncMock(return_value=[
                {"documentCode": "SPEC", "documentCodeDescriptionText": "Specification"},
            ]),
        ), patch(
            "sources.long_task.text_extractor.get_download_url_from_doc",
            return_value="",
        ):
            with self.assertRaises(PatentDetailError):
                await _fetch_spec_pdf("uspto", "US12000123B2")

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
            "sources.long_task.text_extractor.get_download_url_from_doc",
            return_value="https://api.uspto.gov/api/v1/download/clm.xmlarchive",
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

    async def test_claims_falls_back_to_pdf_viewer_url(self):
        # No XML / DOCX options (or both failed to parse) — the endpoint
        # must return the PDF proxy URL instead of extracting anything.
        from api_routes.patent_detail import _fetch_claims

        pdf_url = "https://api.uspto.gov/api/v1/download/clm.pdf"
        with patch(
            "sources.uspto_download.resolve_application_number",
            new=AsyncMock(return_value="18893954"),
        ), patch(
            "sources.uspto_download.fetch_document_bag",
            new=AsyncMock(return_value=[
                {"documentCode": "CLM", "documentCodeDescriptionText": "Claims"},
            ]),
        ), patch(
            "sources.long_task.text_extractor.get_download_url_from_doc",
            side_effect=lambda _doc, mime_order=None, fallback_to_any=True: (
                pdf_url if mime_order and "PDF" in mime_order else ""
            ),
        ), patch(
            "sources.dynamic_tool_params._build_uspto_download_proxy_url",
            return_value="https://api-test.copiioai.com/uspto/download?url=clm",
        ):
            result = await _fetch_claims("uspto", "US12000123B2")

        self.assertEqual(
            result["pdf_url"],
            "https://api-test.copiioai.com/uspto/download?url=clm",
        )
        self.assertNotIn("claims", result)


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
            "api_routes.patent_detail._fetch_spec_pdf",
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

    def test_spec_returns_200_success_true_with_pdf_url(self):
        with patch(
            "api_routes.patent_detail._fetch_spec_pdf",
            new=AsyncMock(
                return_value={
                    "pdf_url": "https://api-test.copiioai.com/uspto/download?url=encoded",
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
        self.assertIn("/uspto/download", body["pdf_url"])

    def test_unsupported_source_still_400(self):
        response = self.client.get(
            "/patent/unknown/US12000123B2/spec",
            headers={"Authorization": "Bearer test"},
        )
        self.assertEqual(response.status_code, 400)
