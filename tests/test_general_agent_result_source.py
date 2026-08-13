"""Tests for _infer_result_source in general_agent."""
import unittest
from types import SimpleNamespace

from sources.agents.general_agent import _infer_result_source


def _tool(url: str) -> SimpleNamespace:
    return SimpleNamespace(url=url)


class TestInferResultSource(unittest.TestCase):
    def test_google_patents(self):
        self.assertEqual(
            _infer_result_source(_tool("https://patents.google.com/x")),
            "google_patents",
        )

    def test_uspto_documents(self):
        self.assertEqual(
            _infer_result_source(
                _tool("https://api.uspto.gov/api/v1/patent/applications/1/documents")
            ),
            "uspto_documents",
        )

    def test_uspto_default(self):
        self.assertEqual(
            _infer_result_source(
                _tool("https://api.uspto.gov/api/v1/patent/applications/search")
            ),
            "uspto",
        )
        self.assertEqual(_infer_result_source(_tool("")), "uspto")
        self.assertEqual(_infer_result_source(None), "uspto")
