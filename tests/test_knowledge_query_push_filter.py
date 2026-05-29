import unittest


class TestKnowledgeQueryPushFilter(unittest.TestCase):

    def test_knowledge_push_filter_condition_requires_matching_tool_push(self):
        from sources.knowledge.query_filters import build_knowledge_push_filter_condition

        sql, params = build_knowledge_push_filter_condition(2)

        self.assertIn("tools.id = knowledge.tool_id", sql)
        self.assertIn("tools.status = 1", sql)
        self.assertIn("tools.push = %s", sql)
        self.assertEqual(params, [2])

    def test_share_push_filter_condition_requires_shared_knowledge_tool_push(self):
        from sources.knowledge.query_filters import build_share_push_filter_condition

        sql, params = build_share_push_filter_condition(2)

        self.assertIn("k.id = knowledge_share.knowledge_id", sql)
        self.assertIn("k.status = 1", sql)
        self.assertIn("tools.id = k.tool_id", sql)
        self.assertIn("tools.push = %s", sql)
        self.assertEqual(params, [2])

    def test_push_filter_condition_is_empty_when_not_requested(self):
        from sources.knowledge.query_filters import (
            build_knowledge_push_filter_condition,
            build_share_push_filter_condition,
        )

        self.assertEqual(build_knowledge_push_filter_condition(None), ("", []))
        self.assertEqual(build_share_push_filter_condition(None), ("", []))


if __name__ == "__main__":
    unittest.main()
