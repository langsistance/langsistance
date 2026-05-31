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
