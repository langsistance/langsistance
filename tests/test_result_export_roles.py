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
            "documentDate": "document_date",
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
