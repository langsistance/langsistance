import unittest


class TestKnowledgeQueryPushFilter(unittest.TestCase):

    def test_knowledge_push_filter_condition_includes_composed_or_matching_tool_push_without_exists(self):
        from sources.knowledge.query_filters import build_knowledge_push_filter_condition

        sql, params = build_knowledge_push_filter_condition(2, [11, 12])

        self.assertIn("knowledge.type = 2", sql)
        self.assertIn("knowledge.tool_id IN (%s,%s)", sql)
        self.assertNotIn("EXISTS", sql.upper())
        self.assertNotIn("tools.", sql)
        self.assertEqual(params, [11, 12])

    def test_knowledge_push_filter_condition_keeps_composed_when_no_tools_match(self):
        from sources.knowledge.query_filters import build_knowledge_push_filter_condition

        sql, params = build_knowledge_push_filter_condition(2, [])

        self.assertIn("knowledge.type = 2", sql)
        self.assertNotIn("tool_id IN", sql)
        self.assertNotIn("EXISTS", sql.upper())
        self.assertEqual(params, [])

    def test_share_push_filter_condition_uses_precomputed_knowledge_ids_without_exists(self):
        from sources.knowledge.query_filters import build_share_push_filter_condition

        sql, params = build_share_push_filter_condition(2, [21, 22])

        self.assertIn("knowledge_share.knowledge_id IN (%s,%s)", sql)
        self.assertNotIn("EXISTS", sql.upper())
        self.assertNotIn("tools.", sql)
        self.assertEqual(params, [21, 22])

    def test_share_push_filter_condition_matches_no_shares_when_no_knowledge_matches(self):
        from sources.knowledge.query_filters import build_share_push_filter_condition

        sql, params = build_share_push_filter_condition(2, [])

        self.assertIn("AND 1 = 0", sql)
        self.assertNotIn("EXISTS", sql.upper())
        self.assertEqual(params, [])

    def test_push_filter_condition_is_empty_when_not_requested(self):
        from sources.knowledge.query_filters import (
            build_knowledge_push_filter_condition,
            build_share_push_filter_condition,
        )

        self.assertEqual(build_knowledge_push_filter_condition(None, [1]), ("", []))
        self.assertEqual(build_share_push_filter_condition(None, [1]), ("", []))


if __name__ == "__main__":
    unittest.main()
