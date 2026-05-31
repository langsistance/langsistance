import json
import unittest


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def execute(self, sql, params=None):
        params = tuple(params or ())
        self.calls.append((sql, params))

    def fetchall(self):
        return list(self.rows)


class TestKnowledgePublicDependencies(unittest.TestCase):

    def test_extracts_unique_workflow_dependency_ids(self):
        from sources.knowledge.public_dependencies import extract_workflow_dependency_ids

        params = json.dumps({
            "type": "workflow",
            "steps": [
                {"knowledge_id": 101},
                {"knowledge_id": "102"},
                {"knowledge_id": 101},
                {"knowledge_id": "bad"},
            ],
        })

        self.assertEqual(extract_workflow_dependency_ids(params), [101, 102])

    def test_promotes_private_normal_dependencies_when_workflow_is_public(self):
        from sources.knowledge.public_dependencies import promote_public_workflow_dependencies

        params = json.dumps({
            "type": "workflow",
            "steps": [
                {"knowledge_id": 101},
                {"knowledge_id": 102},
                {"knowledge_id": 103},
            ],
        })
        cursor = FakeCursor([
            {"id": 101, "public": 1, "type": 1, "params": "{}"},
            {"id": 102, "public": 2, "type": 1, "params": "{}"},
            {"id": 103, "public": 1, "type": 2, "params": '{"type":"workflow","steps":[]}'},
        ])

        promoted = promote_public_workflow_dependencies(
            cursor,
            user_id="user-1",
            public=2,
            knowledge_type=2,
            params=params,
        )

        self.assertEqual(promoted, [101])
        self.assertEqual(len(cursor.calls), 2)
        update_sql, update_params = cursor.calls[1]
        self.assertIn("UPDATE knowledge", update_sql)
        self.assertEqual(update_params, (2, "user-1", 101))

    def test_skips_dependency_updates_when_workflow_is_private(self):
        from sources.knowledge.public_dependencies import promote_public_workflow_dependencies

        cursor = FakeCursor([])
        promoted = promote_public_workflow_dependencies(
            cursor,
            user_id="user-1",
            public=1,
            knowledge_type=2,
            params='{"type":"workflow","steps":[{"knowledge_id":101}]}',
        )

        self.assertEqual(promoted, [])
        self.assertEqual(cursor.calls, [])

    def test_skips_dependency_updates_for_normal_knowledge(self):
        from sources.knowledge.public_dependencies import promote_public_workflow_dependencies

        cursor = FakeCursor([])
        promoted = promote_public_workflow_dependencies(
            cursor,
            user_id="user-1",
            public=2,
            knowledge_type=1,
            params="{}",
        )

        self.assertEqual(promoted, [])
        self.assertEqual(cursor.calls, [])


if __name__ == "__main__":
    unittest.main()
