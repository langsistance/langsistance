import unittest
import importlib.util
import os
import sys
import types
from unittest.mock import patch


class FakeCallbackHandler:
    def __init__(self):
        self.tokens = []

    async def on_llm_new_token(self, token):
        self.tokens.append(token)


class FakeMemory:
    def __init__(self):
        self.messages = []

    def get(self):
        return self.messages

    def reset(self, messages):
        self.messages = list(messages)

    def push(self, role, content):
        self.messages.append({"role": role, "content": content})


class FakeLogger:
    def info(self, message):
        pass


class FakeWorkflowResult:
    def __init__(self, final_data, raw_items=None):
        self.final_data = final_data
        self.raw_items = raw_items


class FakeWorkflowExecutor:
    result = FakeWorkflowResult({"ok": True})

    def __init__(self, llm):
        self.llm = llm

    async def execute(self, workflow_spec, user_prompt):
        return self.result


class FakeLlm:
    def __init__(self, complete_json_response=None):
        self.stream_calls = []
        self.complete_json_calls = []
        self.openai_create_calls = []
        self.openai_invoke_calls = []
        self.complete_json_response = complete_json_response or {
            "has_filter_requirement": False,
            "decisions": [],
        }

    async def openai_invoke(self, agent, memory, callback_handler):
        self.openai_invoke_calls.append({
            "agent": agent,
            "memory": memory,
        })
        return None

    def openai_create(self, tools, history, callback_handler=None, verbose=False):
        self.openai_create_calls.append({
            "tools": tools,
            "history": history,
        })
        return {"agent": "created"}

    async def stream_simple(self, system_prompt, user_content, callback_handler):
        self.stream_calls.append({
            "system_prompt": system_prompt,
            "user_content": user_content,
        })

    async def complete_json(self, system_prompt, user_content):
        self.complete_json_calls.append({
            "system_prompt": system_prompt,
            "user_content": user_content,
        })
        return self.complete_json_response


class TestGeneralAgentBatchPrompt(unittest.IsolatedAsyncioTestCase):

    def _load_general_agent_class(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        module_path = os.path.join(repo_root, "sources", "agents", "general_agent.py")

        agent_module = types.ModuleType("sources.agents.agent")

        class Agent:
            pass

        agent_module.Agent = Agent

        knowledge_module = types.ModuleType("sources.knowledge.knowledge")
        knowledge_module.get_redis_connection = lambda *args, **kwargs: None
        knowledge_module.get_knowledge_tool = lambda *args, **kwargs: {}
        knowledge_module.clean_html_text = lambda value: value

        utility_module = types.ModuleType("sources.utility")
        utility_module.pretty_print = lambda *args, **kwargs: None
        utility_module.animate_thinking = lambda *args, **kwargs: None

        mcp_module = types.ModuleType("sources.tools.mcpFinder")
        mcp_module.MCP_finder = object

        memory_module = types.ModuleType("sources.memory")
        memory_module.Memory = object

        logger_module = types.ModuleType("sources.logger")
        logger_module.Logger = lambda *args, **kwargs: FakeLogger()

        langchain_tools_module = types.ModuleType("langchain_core.tools")
        langchain_tools_module.StructuredTool = object

        pydantic_module = types.ModuleType("pydantic")

        class BaseModel:
            pass

        pydantic_module.BaseModel = BaseModel
        pydantic_module.Field = lambda *args, **kwargs: None

        bs4_module = types.ModuleType("bs4")
        bs4_module.BeautifulSoup = object

        requests_module = types.ModuleType("requests")

        workflow_module = types.ModuleType("sources.workflow.workflow_executor")
        workflow_module.WorkflowExecutor = object
        workflow_module.is_workflow_knowledge = lambda item: bool(getattr(item, "type", 1) == 2)

        stubs = {
            "pydantic": pydantic_module,
            "bs4": bs4_module,
            "requests": requests_module,
            "sources.agents.agent": agent_module,
            "sources.knowledge.knowledge": knowledge_module,
            "sources.utility": utility_module,
            "sources.tools.mcpFinder": mcp_module,
            "sources.memory": memory_module,
            "sources.logger": logger_module,
            "sources.workflow.workflow_executor": workflow_module,
            "langchain_core.tools": langchain_tools_module,
        }

        spec = importlib.util.spec_from_file_location(
            "test_loaded_general_agent",
            module_path,
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, stubs):
            spec.loader.exec_module(module)
        return module.GeneralAgent

    def _patch_workflow_globals(self, GeneralAgent, workflow_result):
        globals_dict = GeneralAgent.create_agent.__globals__
        originals = {
            "get_knowledge_tool": globals_dict["get_knowledge_tool"],
            "WorkflowExecutor": globals_dict["WorkflowExecutor"],
        }
        knowledge = types.SimpleNamespace(
            question="根据公开 ID 查询所有文档",
            params='{"type":"workflow","mode":"context_chain","steps":[]}',
            type=2,
            tool_id=0,
            answer="workflow",
            description="",
        )
        globals_dict["get_knowledge_tool"] = lambda *args, **kwargs: (knowledge, None)
        FakeWorkflowExecutor.result = workflow_result
        globals_dict["WorkflowExecutor"] = FakeWorkflowExecutor
        self.addCleanup(lambda: globals_dict.update(originals))

    async def test_workflow_scalar_result_streams_once_without_creating_or_invoking_agent(self):
        GeneralAgent = self._load_general_agent_class()
        self._patch_workflow_globals(
            GeneralAgent,
            FakeWorkflowResult({"applicationNumberText": "18244278"}),
        )

        agent = GeneralAgent.__new__(GeneralAgent)
        agent.llm = FakeLlm()
        agent.memory = FakeMemory()
        agent.logger = FakeLogger()
        callback_handler = FakeCallbackHandler()

        openai_agent = await agent.create_agent(
            user_id="user-1",
            prompt="根据公开 ID US123 查询专利",
            query_id="query-1",
            tool_data="",
            callback_handler=callback_handler,
            push_filter=2,
        )
        await agent.invoke_agent(openai_agent, callback_handler)

        self.assertIsNone(openai_agent)
        self.assertEqual(agent.llm.openai_create_calls, [])
        self.assertEqual(agent.llm.openai_invoke_calls, [])
        self.assertEqual(len(agent.llm.stream_calls), 1)
        stream_content = agent.llm.stream_calls[0]["user_content"]
        self.assertIn("根据公开 ID US123 查询专利", stream_content)
        self.assertIn("applicationNumberText", stream_content)

    async def test_workflow_list_result_uses_batch_output_without_creating_or_invoking_agent(self):
        GeneralAgent = self._load_general_agent_class()
        self._patch_workflow_globals(
            GeneralAgent,
            FakeWorkflowResult(
                {"documentBag": [{"documentCode": "SPEC"}, {"documentCode": "CLM"}]},
                raw_items=[{"documentCode": "SPEC"}, {"documentCode": "CLM"}],
            ),
        )

        agent = GeneralAgent.__new__(GeneralAgent)
        agent.llm = FakeLlm()
        agent.memory = FakeMemory()
        agent.logger = FakeLogger()
        callback_handler = FakeCallbackHandler()

        openai_agent = await agent.create_agent(
            user_id="user-1",
            prompt="根据公开 ID US123 查询所有文档",
            query_id="query-1",
            tool_data="",
            callback_handler=callback_handler,
            push_filter=2,
        )
        await agent.invoke_agent(openai_agent, callback_handler)

        self.assertIsNone(openai_agent)
        self.assertEqual(agent.llm.openai_create_calls, [])
        self.assertEqual(agent.llm.openai_invoke_calls, [])
        self.assertEqual(len(agent.llm.stream_calls), 1)
        self.assertIn("Full Results (2 items)", "".join(callback_handler.tokens))
        self.assertIn("SPEC", agent.llm.stream_calls[0]["user_content"])
        self.assertIn("CLM", agent.llm.stream_calls[0]["user_content"])

    async def test_batch_prompt_requires_all_urls_verbatim(self):
        GeneralAgent = self._load_general_agent_class()

        agent = GeneralAgent.__new__(GeneralAgent)
        agent.llm = FakeLlm()
        agent.memory = FakeMemory()
        agent.logger = FakeLogger()
        agent._pending_raw_items = [
            {
                "title": "Item 1",
                "url": "https://example.com/result-1",
                "documentBag": [
                    {
                        "downloadOptionBag": [
                            {
                                "downloadUrl": (
                                    "https://api.uspto.gov/api/v1/download/applications/"
                                    "18893954/MMU2X3JJX89X113.pdf"
                                )
                            }
                        ]
                    }
                ],
            },
            {
                "title": "Item 2",
                "url": "https://example.com/result-2?x=1",
                "thumbnailUrl": "https://example.com/image.jpg",
            },
        ]

        await agent.invoke_agent(agent=None, callback_handler=FakeCallbackHandler())

        self.assertEqual(len(agent.llm.stream_calls), 1)
        call = agent.llm.stream_calls[0]
        combined_prompt = f"{call['system_prompt']}\n{call['user_content']}"

        self.assertIn("Every URL is mandatory", combined_prompt)
        self.assertIn("verbatim", combined_prompt)
        self.assertIn("Do not omit", combined_prompt)
        self.assertIn("Mandatory URL checklist", combined_prompt)
        self.assertIn("https://example.com/result-1", combined_prompt)
        self.assertIn("https://example.com/result-2?x=1", combined_prompt)
        self.assertIn("https://example.com/image.jpg", combined_prompt)
        self.assertIn("image URL", combined_prompt)
        self.assertIn("![alt text](image_URL)", combined_prompt)
        self.assertIn("https://api.copiioai.com/uspto/download?url=", combined_prompt)

    async def test_filters_pending_raw_items_before_batch_display(self):
        GeneralAgent = self._load_general_agent_class()

        agent = GeneralAgent.__new__(GeneralAgent)
        agent.llm = FakeLlm({
            "has_filter_requirement": True,
            "decisions": [
                {"index": 0, "keep": True, "confidence": 0.98, "reason": "Tokyo"},
                {"index": 1, "keep": False, "confidence": 0.94, "reason": "Osaka"},
                {"index": 2, "keep": True, "confidence": 0.97, "reason": "Tokyo"},
            ],
        })
        agent.memory = FakeMemory()
        agent.logger = FakeLogger()
        agent._last_user_prompt = "只保留东京的公司"
        agent._pending_raw_items = [
            {"name": "Alpha", "city": "Tokyo", "url": "https://example.com/a"},
            {"name": "Beta", "city": "Osaka", "url": "https://example.com/b"},
            {"name": "Gamma", "city": "Tokyo", "url": "https://example.com/c"},
        ]
        callback_handler = FakeCallbackHandler()

        await agent.invoke_agent(agent=None, callback_handler=callback_handler)

        self.assertEqual(len(agent.llm.complete_json_calls), 1)
        self.assertEqual(len(agent.llm.stream_calls), 1)
        self.assertIn("Filtered Results (2 of 3 items)", "".join(callback_handler.tokens))
        batch_content = agent.llm.stream_calls[0]["user_content"]
        self.assertIn("Alpha", batch_content)
        self.assertIn("Gamma", batch_content)
        self.assertNotIn("Beta", batch_content)

    async def test_template_prompt_requires_preserving_original_api_key_params(self):
        GeneralAgent = self._load_general_agent_class()

        agent = GeneralAgent.__new__(GeneralAgent)
        agent.logger = FakeLogger()
        agent.knowledgeTool = (
            types.SimpleNamespace(answer=""),
            types.SimpleNamespace(
                title="patent_api",
                description="patent api",
                push=2,
                params='{"method":"GET","query":{"api-key":"secret","q":""}}',
            ),
        )
        GeneralAgent.generate_template_system_prompt.__globals__["ZoneInfo"] = lambda key: None

        prompt = agent.generate_template_system_prompt()

        self.assertIn("api-key", prompt)
        self.assertIn("preserve", prompt.lower())
        self.assertIn("exactly", prompt.lower())
        self.assertIn("secret", prompt)


if __name__ == "__main__":
    unittest.main()
