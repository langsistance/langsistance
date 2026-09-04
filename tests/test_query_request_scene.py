"""Tests: QueryRequest carries the optional scene field (A1)."""
import unittest

from sources.schemas import QueryRequest


class TestQueryRequestScene(unittest.TestCase):
    def test_scene_defaults_none(self):
        req = QueryRequest(query="折叠水杯", query_id="q1")
        self.assertIsNone(req.scene)

    def test_scene_roundtrip(self):
        req = QueryRequest(query="折叠水杯", query_id="q1", scene="seller")
        self.assertEqual(req.scene, "seller")
        self.assertEqual(req.jsonify()["scene"], "seller")

    def test_jsonify_backward_compatible_without_scene(self):
        req = QueryRequest(query="折叠水杯", query_id="q1")
        self.assertIn("scene", req.jsonify())
        self.assertIsNone(req.jsonify()["scene"])


if __name__ == "__main__":
    unittest.main()
