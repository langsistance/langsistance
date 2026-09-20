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

# 这些第三方桩只为让本模块 import 得过。**用完必须还原** —— 留着的话，同一
# 进程里后续任何 `import redis` 都会拿到这个空 module（`Redis` 被替换成
# `object`），于是 patent_token._get_redis() 在 `redis.Redis(...)` 处抛
# TypeError。此前 test_workflow_executor 与 test_dynamic_tool_params 同批运行时
# 就是这样把 TestHttpErrorSemantics 打红的（与本次改动无关，2026-09-20 定位）。
_stubbed_modules = {}
for module_name in ("bs4", "pymysql", "pymysql.cursors", "redis", "requests"):
    if module_name not in sys.modules:
        _stubbed_modules[module_name] = types.ModuleType(module_name)
        sys.modules[module_name] = _stubbed_modules[module_name]

sys.modules["bs4"].BeautifulSoup = lambda *args, **kwargs: types.SimpleNamespace(get_text=lambda *a, **k: "")


class _StubRedis:
    """可调用的 redis 替身。

    **不能是 `object`**：``patent_token._get_redis()`` 是在**调用时**查
    ``redis.Redis``，被污染的模块会让它 `TypeError: object() takes no arguments`
    —— 此前本文件与 test_dynamic_tool_params 同批运行时就是这样把它打红的。
    ``get`` 一律返回 None = 没有 token，与"拿不到凭据"的既有行为一致。
    """

    def __init__(self, *args, **kwargs):
        pass

    def get(self, *args, **kwargs):
        return None

    def set(self, *args, **kwargs):
        return True

    def delete(self, *args, **kwargs):
        return 0

    def close(self, *args, **kwargs):
        return None


sys.modules["redis"].Redis = _StubRedis

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
_original_sources_logger = sys.modules.get("sources.logger")
sys.modules["sources.logger"] = logger_module

utility_module = types.ModuleType("sources.utility")
utility_module.pretty_print = lambda *args, **kwargs: None
sys.modules["sources.utility"] = utility_module

from sources.knowledge.knowledge import KnowledgeItem, ToolItem

# 桩用完还原（本模块已持有自己的模块引用，不受影响）。
for _name in _stubbed_modules:
    sys.modules.pop(_name, None)

if _original_sources_logger is not None:
    sys.modules["sources.logger"] = _original_sources_logger
else:
    sys.modules.pop("sources.logger", None)


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
        self.assertIn("workflow step 2 knowledge:", log_text)
        self.assertIn("workflow step 2 tool:", log_text)
        self.assertIn("workflow step 2 params:", log_text)

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

    async def test_workflow_instructions_are_in_prompt_but_descriptions_are_not(self):
        from sources.workflow.workflow_executor import WorkflowExecutor

        workflow_knowledge = KnowledgeItem(
            id=10,
            user_id="user-1",
            question="workflow name",
            description="workflow admin notes should stay out",
            answer="Prefer publication ID over application ID when both are available.",
            public=1,
            model_name="gpt-4o-mini",
            tool_id=0,
            params="{}",
            type=2,
        )
        step_knowledge = KnowledgeItem(
            id=101,
            user_id="user-1",
            question="query patent",
            description="step admin notes should stay out",
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
            params='{"method": "GET"}',
        )
        executor = WorkflowExecutor(
            llm=FakeLlm(),
            knowledge_resolver=lambda knowledge_id: step_knowledge,
            tool_resolver=lambda tool_id: tool,
            tool_executor=lambda tool_info, params: {"data": {}, "raw_items": None},
        )

        await executor.execute(
            workflow_spec=json.dumps({
                "type": "workflow",
                "version": 1,
                "mode": "context_chain",
                "steps": [{"id": "step_1", "knowledge_id": 101}],
            }),
            user_prompt="lookup US123",
            workflow_knowledge=workflow_knowledge,
        )

        user_content = executor.llm.complete_json_calls[0]["user_content"]
        self.assertIn("workflow name", user_content)
        self.assertIn("Prefer publication ID over application ID", user_content)
        self.assertIn("Use the patent API", user_content)
        self.assertNotIn("workflow admin notes should stay out", user_content)
        self.assertNotIn("step admin notes should stay out", user_content)


    async def test_step_1_never_gets_terminate_instruction_step_2_does(self):
        """Step 1 derives params from user request — it must NOT be told to
        terminate on empty previous_results. Step 2+ gets the termination rule."""
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
            params='{"method": "GET"}',
        )
        executor = WorkflowExecutor(
            llm=FakeLlm(),
            knowledge_resolver=lambda knowledge_id: knowledge,
            tool_resolver=lambda tool_id: tool,
            tool_executor=lambda tool_info, params: {"data": {}, "raw_items": None},
        )

        # Step 1: _terminate must NOT appear in the prompt
        await executor._generate_tool_params(
            user_prompt="lookup US123",
            step_index=1,
            total_steps=2,
            knowledge=knowledge,
            tool=tool,
            previous_results=[],
        )
        step1_prompt = executor.llm.complete_json_calls[0]["system_prompt"]
        self.assertNotIn("_terminate", step1_prompt)

        # Step 2: _terminate MUST appear (previous_results is empty)
        await executor._generate_tool_params(
            user_prompt="lookup US123",
            step_index=2,
            total_steps=2,
            knowledge=knowledge,
            tool=tool,
            previous_results=[],  # empty → should trigger _terminate instruction
        )
        step2_prompt = executor.llm.complete_json_calls[1]["system_prompt"]
        self.assertIn("_terminate", step2_prompt)


if __name__ == "__main__":
    unittest.main()


class TestWorkflowToolFailureIsStepLevel(unittest.IsolatedAsyncioTestCase):
    """2026-09-20: execute_backend_tool_request 对未替换的模板占位符改为抛
    ValueError(需求#32/#33)。工作流若不接住, 一次占位符失配会掀翻整个工作流 ——
    而这本应是**步骤级**结果: 让后续步骤和 LLM 看到原因再决定。
    """

    async def test_failing_tool_records_error_without_killing_the_workflow(self):
        from sources.workflow.workflow_executor import WorkflowExecutor

        knowledge = KnowledgeItem(
            id=101, user_id="u1", question="取文档", description="",
            answer="取文档", public=1, model_name="m", tool_id=201,
            params="{}", type=1)
        tool = ToolItem(
            id=201, user_id="u1", title="docs", description="d", push=2,
            url="https://example.test/{applicationNumberText}", status=True,
            timeout=30, params='{"method": "GET"}')

        def _raise(tool_info, params):
            raise ValueError(
                "Request failed: missing value for URL parameter(s): "
                "applicationNumberText")

        executor = WorkflowExecutor(
            llm=FakeLlm(),
            knowledge_resolver=lambda kid: knowledge,
            tool_resolver=lambda tid: tool,
            tool_executor=_raise,
        )
        result = await executor.execute(
            workflow_spec=json.dumps({
                "type": "workflow", "version": 1, "mode": "context_chain",
                "steps": [{"id": "step_1", "knowledge_id": 101}],
            }),
            user_prompt="取该申请的文档",
        )
        self.assertEqual(len(result.steps), 1)
        self.assertIn("applicationNumberText", str(result.steps[0].data))
