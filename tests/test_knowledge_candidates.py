import unittest
import sys
import types
from unittest.mock import patch


openai_module = types.ModuleType("openai")
openai_module.OpenAI = lambda *args, **kwargs: types.SimpleNamespace()
sys.modules.setdefault("openai", openai_module)

numpy_module = types.ModuleType("numpy")
numpy_module.array = lambda value: value
sys.modules.setdefault("numpy", numpy_module)

pairwise_module = types.ModuleType("sklearn.metrics.pairwise")


def _cosine_similarity(left, right):
    query = left[0]
    scores = []
    for embedding in right:
        numerator = sum(a * b for a, b in zip(query, embedding))
        query_norm = sum(a * a for a in query) ** 0.5
        embedding_norm = sum(b * b for b in embedding) ** 0.5
        scores.append(numerator / (query_norm * embedding_norm))
    return [scores]


pairwise_module.cosine_similarity = _cosine_similarity
metrics_module = types.ModuleType("sklearn.metrics")
metrics_module.pairwise = pairwise_module
sklearn_module = types.ModuleType("sklearn")
sklearn_module.metrics = metrics_module
sys.modules.setdefault("sklearn", sklearn_module)
sys.modules.setdefault("sklearn.metrics", metrics_module)
sys.modules.setdefault("sklearn.metrics.pairwise", pairwise_module)

pydantic_module = types.ModuleType("pydantic")


class BaseModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


pydantic_module.BaseModel = BaseModel
sys.modules.setdefault("pydantic", pydantic_module)

bs4_module = types.ModuleType("bs4")
bs4_module.BeautifulSoup = object
sys.modules.setdefault("bs4", bs4_module)

pymysql_module = types.ModuleType("pymysql")
pymysql_cursors_module = types.ModuleType("pymysql.cursors")
pymysql_cursors_module.DictCursor = object
pymysql_module.cursors = pymysql_cursors_module
sys.modules.setdefault("pymysql", pymysql_module)
sys.modules.setdefault("pymysql.cursors", pymysql_cursors_module)

redis_module = types.ModuleType("redis")
redis_module.Redis = object
sys.modules.setdefault("redis", redis_module)

logger_module = types.ModuleType("sources.logger")


class FakeLogger:
    def info(self, message):
        pass

    def warning(self, message):
        pass

    def error(self, message):
        pass


logger_module.Logger = lambda *args, **kwargs: FakeLogger()
sys.modules.setdefault("sources.logger", logger_module)

utility_module = types.ModuleType("sources.utility")
utility_module.pretty_print = lambda *args, **kwargs: None
sys.modules.setdefault("sources.utility", utility_module)

from sources.knowledge import knowledge as knowledge_module
from sources.knowledge.knowledge import KnowledgeItem, ToolItem


class FakeRedis:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def get(self, key):
        return self.embeddings.get(key)


def _knowledge(knowledge_id, tool_id, question="question"):
    return KnowledgeItem(
        id=knowledge_id,
        user_id="user-1",
        question=question,
        description="routing hint",
        answer="answer",
        public=0,
        model_name="",
        tool_id=tool_id,
        params="",
        type=1,
    )


def _tool(tool_id, push):
    return ToolItem(
        id=tool_id,
        user_id="user-1",
        title=f"tool-{tool_id}",
        description="tool",
        url="https://example.com",
        push=push,
        status=True,
        timeout=30,
        params="{}",
    )


class TestKnowledgeCandidates(unittest.TestCase):

    def test_candidates_filter_tool_push_before_selection(self):
        knowledge_items = [
            _knowledge(1, 101, "high similarity but frontend tool"),
            _knowledge(2, 202, "lower similarity backend tool"),
        ]
        redis = FakeRedis({
            "knowledge_embedding_1": str([1.0, 0.0]),
            "knowledge_embedding_2": str([0.5, 0.5]),
        })

        def get_tool_by_id(tool_id, push_filter=None):
            tool = _tool(tool_id, push=1 if tool_id == 101 else 2)
            if push_filter is not None and tool.push != push_filter:
                return None
            return tool

        with patch.object(knowledge_module, "get_embedding", return_value=[1.0, 0.0]), \
                patch.object(knowledge_module, "get_user_knowledge", return_value=knowledge_items), \
                patch.object(knowledge_module, "get_redis_connection", return_value=redis), \
                patch.object(knowledge_module, "get_tool_by_id", side_effect=get_tool_by_id):
            candidates = knowledge_module.get_knowledge_tool_candidates(
                "user-1",
                "find backend data",
                top_k=2,
                push_filter=2,
            )

        self.assertEqual([knowledge.id for knowledge, _ in candidates], [2])
        self.assertEqual(candidates[0][0].extra_info["similarity"], 0.7071067811865475)


if __name__ == "__main__":
    unittest.main()
