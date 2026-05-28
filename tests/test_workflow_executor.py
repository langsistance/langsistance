import json
import sys
import types
import unittest

if "pydantic" not in sys.modules:
    pydantic_module = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def dict(self):
            return self.__dict__.copy()

    pydantic_module.BaseModel = BaseModel
    sys.modules["pydantic"] = pydantic_module

if "openai" not in sys.modules:
    openai_module = types.ModuleType("openai")
    openai_module.OpenAI = lambda *args, **kwargs: object()
    sys.modules["openai"] = openai_module

if "numpy" not in sys.modules:
    numpy_module = types.ModuleType("numpy")
    numpy_module.array = lambda value: value
    sys.modules["numpy"] = numpy_module

if "sklearn.metrics.pairwise" not in sys.modules:
    sklearn_module = types.ModuleType("sklearn")
    metrics_module = types.ModuleType("sklearn.metrics")
    pairwise_module = types.ModuleType("sklearn.metrics.pairwise")
    pairwise_module.cosine_similarity = lambda *args, **kwargs: [[0]]
    sys.modules["sklearn"] = sklearn_module
    sys.modules["sklearn.metrics"] = metrics_module
    sys.modules["sklearn.metrics.pairwise"] = pairwise_module

for module_name in ("bs4", "pymysql", "pymysql.cursors", "redis", "requests"):
    if module_name not in sys.modules:
        sys.modules[module_name] = types.ModuleType(module_name)

sys.modules["bs4"].BeautifulSoup = lambda *args, **kwargs: types.SimpleNamespace(get_text=lambda *a, **k: "")
sys.modules["redis"].Redis = object

logger_module = types.ModuleType("sources.logger")

class FakeLogger:
    def __init__(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

logger_module.Logger = FakeLogger
sys.modules["sources.logger"] = logger_module

utility_module = types.ModuleType("sources.utility")
utility_module.pretty_print = lambda *args, **kwargs: None
sys.modules["sources.utility"] = utility_module

from sources.knowledge.knowledge import KnowledgeItem, ToolItem


class FakeLlm:
    def __init__(self):
        self.complete_json_calls = []

    async def complete_json(self, system_prompt, user_content):
        self.complete_json_calls.append({
            "system_prompt": system_prompt,
            "user_content": user_content,
        })
        if len(self.complete_json_calls) == 1:
            return {"query": {"publicationId": "US123"}}
        return {"query": {"applicationId": "18244278"}}


class TestWorkflowExecutor(unittest.IsolatedAsyncioTestCase):

    def test_knowledge_item_exposes_type(self):
        item = KnowledgeItem(
            id=1,
            user_id="user-1",
            question="普通知识",
            description="",
            answer="answer",
            public=1,
            model_name="gpt-4o-mini",
            tool_id=10,
            params="{}",
            type=2,
        )

        self.assertEqual(item.type, 2)

    async def test_context_chain_passes_previous_full_result_to_next_step(self):
        import sources.workflow.workflow_executor as workflow_executor
        from sources.workflow.workflow_executor import WorkflowExecutor

        class CaptureLogger:
            def __init__(self):
                self.messages = []

            def info(self, message):
                self.messages.append(message)

        capture_logger = CaptureLogger()
        original_logger = workflow_executor.logger
        workflow_executor.logger = capture_logger
        self.addCleanup(lambda: setattr(workflow_executor, "logger", original_logger))

        knowledge_by_id = {
            101: KnowledgeItem(
                id=101,
                user_id="user-1",
                question="根据公开 ID 查询专利信息",
                description="",
                answer="用公开 ID 查询专利信息",
                public=1,
                model_name="gpt-4o-mini",
                tool_id=201,
                params="{}",
                type=1,
            ),
            102: KnowledgeItem(
                id=102,
                user_id="user-1",
                question="根据申请 ID 查询所有文档",
                description="",
                answer="从前一步结果中找到申请 ID，再查询文档",
                public=1,
                model_name="gpt-4o-mini",
                tool_id=202,
                params="{}",
                type=1,
            ),
        }
        tool_by_id = {
            201: ToolItem(
                id=201,
                user_id="user-1",
                title="lookup_patent",
                description="lookup patent",
                push=2,
                url="https://example.test/patent",
                status=True,
                timeout=30,
                params='{"method": "GET"}',
            ),
            202: ToolItem(
                id=202,
                user_id="user-1",
                title="list_documents",
                description="list documents",
                push=2,
                url="https://example.test/documents",
                status=True,
                timeout=30,
                params='{"method": "GET"}',
            ),
        }
        tool_calls = []

        def execute_tool(tool_info, params):
            tool_calls.append((tool_info.title, params))
            if tool_info.id == 201:
                return {
                    "data": {
                        "publicationId": "US123",
                        "applicationNumberText": "18244278",
                    },
                    "raw_items": None,
                }
            return {
                "data": {
                    "documentBag": [
                        {"documentCode": "SPEC"},
                        {"documentCode": "CLM"},
                    ]
                },
                "raw_items": [{"documentCode": "SPEC"}, {"documentCode": "CLM"}],
            }

        executor = WorkflowExecutor(
            llm=FakeLlm(),
            knowledge_resolver=lambda knowledge_id: knowledge_by_id[knowledge_id],
            tool_resolver=lambda tool_id: tool_by_id[tool_id],
            tool_executor=execute_tool,
        )

        result = await executor.execute(
            workflow_spec=json.dumps({
                "type": "workflow",
                "version": 1,
                "mode": "context_chain",
                "steps": [
                    {"id": "step_1", "knowledge_id": 101},
                    {"id": "step_2", "knowledge_id": 102},
                ],
            }),
            user_prompt="根据公开 ID US123 查询所有文档",
        )

        self.assertEqual(
            result.final_data,
            {"documentBag": [{"documentCode": "SPEC"}, {"documentCode": "CLM"}]},
        )
        self.assertEqual(result.raw_items, [{"documentCode": "SPEC"}, {"documentCode": "CLM"}])
        self.assertEqual(tool_calls[0], ("lookup_patent", {"query": {"publicationId": "US123"}}))
        self.assertEqual(tool_calls[1], ("list_documents", {"query": {"applicationId": "18244278"}}))
        self.assertIn("applicationNumberText", executor.llm.complete_json_calls[1]["user_content"])
        log_text = "\n".join(capture_logger.messages)
        self.assertIn("workflow step 1 knowledge:", log_text)
        self.assertIn("workflow step 1 tool:", log_text)
        self.assertIn("workflow step 1 params:", log_text)
        self.assertIn("workflow step 1 tool_result:", log_text)
        self.assertIn("workflow step 2 knowledge:", log_text)
        self.assertIn("workflow step 2 tool:", log_text)
        self.assertIn("workflow step 2 params:", log_text)
        self.assertIn("workflow step 2 tool_result:", log_text)

    async def test_generate_tool_params_prompt_requires_preserving_original_api_key_params(self):
        from sources.workflow.workflow_executor import WorkflowExecutor

        knowledge = KnowledgeItem(
            id=101,
            user_id="user-1",
            question="query patent",
            description="",
            answer="Use the patent API",
            public=1,
            model_name="gpt-4o-mini",
            tool_id=201,
            params="{}",
            type=1,
        )
        tool = ToolItem(
            id=201,
            user_id="user-1",
            title="lookup_patent",
            description="lookup patent",
            push=2,
            url="https://example.test/patent",
            status=True,
            timeout=30,
            params='{"method": "GET", "query": {"api-key": "secret", "publicationId": ""}}',
        )
        executor = WorkflowExecutor(
            llm=FakeLlm(),
            knowledge_resolver=lambda knowledge_id: knowledge,
            tool_resolver=lambda tool_id: tool,
            tool_executor=lambda tool_info, params: {"data": {}, "raw_items": None},
        )

        await executor._generate_tool_params(
            user_prompt="lookup US123",
            step_index=1,
            total_steps=1,
            knowledge=knowledge,
            tool=tool,
            previous_results=[],
        )

        call = executor.llm.complete_json_calls[0]
        combined_prompt = f"{call['system_prompt']}\n{call['user_content']}"
        self.assertIn("api-key", combined_prompt)
        self.assertIn("preserve", combined_prompt.lower())
        self.assertIn("exactly", combined_prompt.lower())
        self.assertIn("secret", combined_prompt)


if __name__ == "__main__":
    unittest.main()
