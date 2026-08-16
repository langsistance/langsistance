import unittest

from sources.knowledge.embedding_text import build_knowledge_embedding_text


class TestKnowledgeEmbeddingText(unittest.TestCase):

    def test_embedding_text_includes_routing_hint_description(self):
        text = build_knowledge_embedding_text(
            question="Patent document workflow",
            description="Use when users ask for publication documents.",
            answer="Fetch bibliographic data, then documents.",
        )

        self.assertIn("Patent document workflow", text)
        self.assertIn("Routing hint", text)
        self.assertIn("Use when users ask for publication documents.", text)
        self.assertIn("Fetch bibliographic data, then documents.", text)

    def test_embedding_text_omits_empty_sections(self):
        text = build_knowledge_embedding_text(
            question="Patent document workflow",
            description="",
            answer="",
        )

        self.assertIn("Question", text)
        self.assertNotIn("Routing hint", text)
        self.assertNotIn("Knowledge content", text)


if __name__ == "__main__":
    unittest.main()


class TestGetEmbeddingsBatch(unittest.TestCase):
    """Batch embedding keeps input order and is a single provider call."""

    def test_returns_vectors_in_input_order(self):
        from unittest.mock import MagicMock, patch
        from sources.knowledge.knowledge import get_embeddings_batch

        client = MagicMock()
        resp = MagicMock()
        resp.data = [
            MagicMock(index=1, embedding=[0.0, 1.0]),
            MagicMock(index=0, embedding=[1.0, 0.0]),
        ]
        client.embeddings.create.return_value = resp
        with patch("sources.knowledge.knowledge._get_embedding_client",
                   return_value=(client, "fake-model")):
            vectors = get_embeddings_batch(["first", "second"])
        client.embeddings.create.assert_called_once()
        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])

    def test_empty_input_returns_empty_without_call(self):
        from unittest.mock import MagicMock, patch
        from sources.knowledge.knowledge import get_embeddings_batch

        client = MagicMock()
        with patch("sources.knowledge.knowledge._get_embedding_client",
                   return_value=(client, "fake-model")):
            self.assertEqual(get_embeddings_batch([]), [])
        client.embeddings.create.assert_not_called()
