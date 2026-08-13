"""Tests for result_export column-role inference and JSON artifact."""
import json
import unittest

from sources.result_export import (
    build_result_artifacts,
    infer_column_role,
)

VALID_ROLES = {
    "title", "patent_id", "application_number", "publication_number",
    "assignee", "inventors", "filing_date", "publication_date", "ipc",
    "abstract", "document_title", "document_date", "url", "text",
}


class TestInferColumnRole(unittest.TestCase):
    def test_known_roles_across_sources(self):
        cases = {
            "patentTitle": "title",
            "applicationMetaData.patentTitle": "title",
            "inventionTitle": "title",
            "patentNumber": "patent_id",
            "publicationNumber": "patent_id",
            "applicationNumberText": "application_number",
            "application_number": "application_number",
            "assigneeEntityName": "assignee",
            "applicant": "assignee",
            "inventors": "inventors",
            "inventorName": "inventors",
            "filingDate": "filing_date",
            "publicationDate": "publication_date",
            "applicationMetaData.earliestPublicationNumber": "publication_number",
            "pctPublicationNumber": "publication_number",
            "grantDate": "publication_date",
            "ipcClass": "ipc",
            "cpcClass": "ipc",
            "abstract": "abstract",
            "abstractText": "abstract",
            "documentTitle": "document_title",
            "documentCodeDescriptionText": "document_title",
            "documentDate": "document_date",
            "mailRoomDate": "document_date",
            "pdfUrl": "url",
            "download_url": "url",
            "url": "url",
        }
        for key, expected in cases.items():
            with self.subTest(key=key):
                self.assertEqual(infer_column_role(key), expected)

    def test_unknown_keys_fall_back_to_text(self):
        self.assertEqual(infer_column_role("someCustomField"), "text")
        self.assertEqual(infer_column_role(""), "text")
        self.assertEqual(infer_column_role("value"), "text")

    def test_always_returns_role_from_closed_set(self):
        for key in ["patentTitle", "xyz", "applicationMetaData.claims",
                    "documentList", "numberOfPages", "title"]:
            self.assertIn(infer_column_role(key), VALID_ROLES)


class TestBuildResultArtifactsJson(unittest.TestCase):
    def _items(self):
        return [
            {
                "patentTitle": "一种图像处理方法",
                "patentNumber": "US12000123B2",
                "applicationNumberText": "17638216",
                "assigneeEntityName": "华为",
                "filingDate": "2022-02-25",
                "abstractText": "本申请公开了一种图像处理方法。",
            },
        ]

    def test_includes_json_artifact_with_roles_and_source(self):
        artifacts = build_result_artifacts(self._items() * 6, source="uspto")

        formats = {a["format"] for a in artifacts}
        self.assertIn("json", formats)
        self.assertIn("csv", formats)
        self.assertIn("xlsx", formats)

        json_artifact = next(a for a in artifacts if a["format"] == "json")
        payload = json.loads(json_artifact["content"].decode("utf-8"))
        self.assertEqual(payload["source"], "uspto")
        self.assertEqual(json_artifact["row_count"], 6)
        roles = {c["key"]: c["role"] for c in payload["columns"]}
        self.assertEqual(roles["patentTitle"], "title")
        self.assertEqual(roles["applicationNumberText"], "application_number")
        self.assertIn("label", payload["columns"][0])
        self.assertEqual(payload["rows"][0]["patentTitle"], "一种图像处理方法")

    def test_respects_min_rows_threshold(self):
        # Default threshold is 6 — 1 item produces no artifacts at all
        artifacts = build_result_artifacts(self._items(), source="uspto")
        self.assertEqual(artifacts, [])

    def test_source_defaults_to_uspto(self):
        items = self._items() * 6
        artifacts = build_result_artifacts(items)
        payload = json.loads(
            next(a for a in artifacts if a["format"] == "json")["content"]
            .decode("utf-8")
        )
        self.assertEqual(payload["source"], "uspto")

    def _document_item(self, download_url=None):
        item = {
            "documentCode": "CTNF",
            "documentCodeDescriptionText": "Non-Final Rejection",
            "mailRoomDate": "2023-04-04",
            "documentIdentifier": "MMU2X3JJX89X113",
            "pageTotalQuantity": 12,
        }
        if download_url is not None:
            item["downloadOptionBag"] = [
                {"mimeType": "application/pdf", "downloadUrl": download_url}
            ]
        return item

    def test_document_items_lift_download_url_into_url_column(self):
        # Document rows carry their download URL inside the nested
        # downloadOptionBag list — it must surface as a top-level
        # downloadUrl column so the frontend can render the download button.
        download_url = "https://api.copiioai.com/uspto/download?url=example"
        items = [self._document_item(download_url)] * 6
        artifacts = build_result_artifacts(items, source="uspto_documents")

        payload = json.loads(
            next(a for a in artifacts if a["format"] == "json")["content"]
            .decode("utf-8")
        )
        roles = {c["key"]: c["role"] for c in payload["columns"]}
        self.assertEqual(roles["downloadUrl"], "url")
        self.assertEqual(roles["downloadOptionBag"], "text")  # raw kept
        self.assertEqual(payload["rows"][0]["downloadUrl"], download_url)

    def test_document_items_without_download_option_bag_have_no_url_column(self):
        items = [self._document_item(None)] * 6
        artifacts = build_result_artifacts(items, source="uspto_documents")

        payload = json.loads(
            next(a for a in artifacts if a["format"] == "json")["content"]
            .decode("utf-8")
        )
        roles = {c["key"]: c["role"] for c in payload["columns"]}
        self.assertNotIn("downloadUrl", roles)

    def test_document_items_skip_empty_download_url_options(self):
        item = self._document_item(None)
        item["downloadOptionBag"] = [
            {"mimeType": "application/pdf", "downloadUrl": ""},
            {"mimeType": "application/pdf",
             "downloadUrl": "https://api.copiioai.com/uspto/download?url=second"},
        ]
        artifacts = build_result_artifacts([item] * 6, source="uspto_documents")

        payload = json.loads(
            next(a for a in artifacts if a["format"] == "json")["content"]
            .decode("utf-8")
        )
        self.assertEqual(
            payload["rows"][0]["downloadUrl"],
            "https://api.copiioai.com/uspto/download?url=second",
        )
