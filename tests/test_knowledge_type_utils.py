import unittest


class TestKnowledgeTypeUtils(unittest.TestCase):

    def test_infers_composed_type_from_workflow_params_when_type_is_missing(self):
        from sources.knowledge.type_utils import infer_knowledge_type

        params = '{"type":"workflow","version":1,"steps":[]}'

        self.assertEqual(infer_knowledge_type(None, params), 2)

    def test_workflow_params_take_precedence_over_default_normal_type(self):
        from sources.knowledge.type_utils import infer_knowledge_type

        params = {"type": "workflow", "version": 1, "steps": []}

        self.assertEqual(infer_knowledge_type(1, params), 2)

    def test_defaults_to_normal_for_missing_or_invalid_type(self):
        from sources.knowledge.type_utils import infer_knowledge_type

        self.assertEqual(infer_knowledge_type(None, ""), 1)
        self.assertEqual(infer_knowledge_type("bad", "{}"), 1)


if __name__ == "__main__":
    unittest.main()
