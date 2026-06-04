import unittest
import sys
import types
from unittest.mock import patch

from scripts.backfill_knowledge_embeddings import (
    _load_env,
    backfill_knowledge_embeddings,
    fetch_knowledge_rows,
    redis_key_for_knowledge,
)


class FakeRedis:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.set_calls = []

    def exists(self, key):
        return 1 if key in self.values else 0

    def set(self, key, value):
        self.set_calls.append((key, value))
        self.values[key] = value


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.sql = None
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, sql, params):
        self.sql = sql
        self.params = params

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, rows):
        self.cursor_instance = FakeCursor(rows)

    def cursor(self):
        return self.cursor_instance


class TestBackfillKnowledgeEmbeddings(unittest.TestCase):
    def test_backfill_only_rebuilds_missing_redis_embeddings(self):
        redis = FakeRedis({
            redis_key_for_knowledge(1): str([1.0, 0.0]),
        })
        rows = [
            {"id": 1, "question": "Q1", "description": "D1", "answer": "A1"},
            {"id": 2, "question": "Q2", "description": "D2", "answer": "A2"},
        ]
        embedding_inputs = []

        def embed(text):
            embedding_inputs.append(text)
            return [0.2, 0.8]

        summary = backfill_knowledge_embeddings(rows, redis, embed)

        self.assertEqual(summary.scanned, 2)
        self.assertEqual(summary.skipped_existing, 1)
        self.assertEqual(summary.rebuilt, 1)
        self.assertEqual(summary.failed, 0)
        self.assertEqual(redis.values[redis_key_for_knowledge(2)], str([0.2, 0.8]))
        self.assertEqual(redis.set_calls, [(redis_key_for_knowledge(2), str([0.2, 0.8]))])
        self.assertEqual(len(embedding_inputs), 1)
        self.assertIn("Question:\nQ2", embedding_inputs[0])
        self.assertIn("Routing hint:\nD2", embedding_inputs[0])
        self.assertIn("Knowledge content:\nA2", embedding_inputs[0])

    def test_force_rebuilds_existing_embeddings(self):
        redis = FakeRedis({
            redis_key_for_knowledge(1): str([1.0, 0.0]),
        })
        rows = [{"id": 1, "question": "Q1", "description": "", "answer": "A1"}]

        summary = backfill_knowledge_embeddings(
            rows,
            redis,
            lambda text: [0.4, 0.6],
            force=True,
        )

        self.assertEqual(summary.skipped_existing, 0)
        self.assertEqual(summary.rebuilt, 1)
        self.assertEqual(redis.values[redis_key_for_knowledge(1)], str([0.4, 0.6]))

    def test_dry_run_does_not_call_embedding_or_write_to_redis(self):
        redis = FakeRedis()
        rows = [{"id": 10, "question": "Q10", "description": "", "answer": "A10"}]

        def embed(_text):
            raise AssertionError("dry-run should not call embedding provider")

        summary = backfill_knowledge_embeddings(rows, redis, embed, dry_run=True)

        self.assertEqual(summary.scanned, 1)
        self.assertEqual(summary.would_rebuild, 1)
        self.assertEqual(summary.rebuilt, 0)
        self.assertEqual(redis.set_calls, [])

    def test_fetch_knowledge_rows_applies_optional_filters(self):
        connection = FakeConnection([{"id": 7}])

        rows = fetch_knowledge_rows(
            connection,
            user_id="user-1",
            knowledge_id=7,
            limit=5,
        )

        self.assertEqual(rows, [{"id": 7}])
        self.assertIn("FROM knowledge", connection.cursor_instance.sql)
        self.assertIn("status = 1", connection.cursor_instance.sql)
        self.assertIn("user_id = %s", connection.cursor_instance.sql)
        self.assertIn("id = %s", connection.cursor_instance.sql)
        self.assertIn("LIMIT %s", connection.cursor_instance.sql)
        self.assertEqual(connection.cursor_instance.params, ["user-1", 7, 5])

    def test_empty_env_file_skips_dotenv_loading(self):
        calls = []
        dotenv_module = types.ModuleType("dotenv")
        dotenv_module.load_dotenv = lambda env_file=None: calls.append(env_file)

        with patch.dict(sys.modules, {"dotenv": dotenv_module}):
            _load_env("")

        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
