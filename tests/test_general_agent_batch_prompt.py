import unittest
import importlib.util
import os
import sys
import types
from unittest.mock import patch


class FakeCallbackHandler:
    def __init__(self):
        self.tokens = []
        self.statuses = []
        self.artifacts = []

    async def on_llm_new_token(self, token):
        self.tokens.append(token)

    async def on_status(self, message, **kwargs):
        self.statuses.append({"message": message, **kwargs})

    async def on_artifacts(self, artifacts):
        self.artifacts.extend(artifacts)


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

    def error(self, message):
        pass


class FakeWorkflowResult:
    def __init__(self, final_data, raw_items=None, workflow_question="", workflow_instructions=""):
        self.final_data = final_data
        self.raw_items = raw_items
        self.workflow_question = workflow_question
        self.workflow_instructions = workflow_instructions


class FakeWorkflowExecutor:
    result = FakeWorkflowResult({"ok": True})
    calls = []

    def __init__(self, llm):
        self.llm = llm

    async def execute(self, workflow_spec, user_prompt, workflow_knowledge=None):
        self.calls.append({
            "workflow_spec": workflow_spec,
            "user_prompt": user_prompt,
            "workflow_knowledge": workflow_knowledge,
        })
        if workflow_knowledge:
            self.result.workflow_question = workflow_knowledge.question
            self.result.workflow_instructions = workflow_knowledge.answer
        return self.result


class FakeLlm:
    def __init__(
        self,
        complete_json_response=None,
        complete_json_responses=None,
        stream_error=None,
        stream_errors=None,
        stream_outputs=None,
    ):
        self.stream_calls = []
        self.complete_json_calls = []
        self.openai_create_calls = []
        self.openai_invoke_calls = []
        self.stream_error = stream_error
        self.stream_errors = list(stream_errors or [])
        self.stream_outputs = list(stream_outputs or [])
        if complete_json_responses is None and isinstance(complete_json_response, list):
            complete_json_responses = complete_json_response
            complete_json_response = None
        self.complete_json_responses = list(complete_json_responses or [])
        self.complete_json_response = complete_json_response if complete_json_response is not None else {
            "has_filter_criteria": False,
            "filter_criteria": "",
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
        if self.stream_error:
            raise self.stream_error
        if self.stream_errors:
            stream_error = self.stream_errors.pop(0)
            if stream_error:
                raise stream_error
        if self.stream_outputs:
            stream_output = self.stream_outputs.pop(0)
            if stream_output and callback_handler:
                await callback_handler.on_llm_new_token(stream_output)

    async def complete_json(self, system_prompt, user_content):
        self.complete_json_calls.append({
            "system_prompt": system_prompt,
            "user_content": user_content,
        })
        if self.complete_json_responses:
            return self.complete_json_responses.pop(0)
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

        async def select_knowledge_tool_with_llm(*args, **kwargs):
            return None, None

        knowledge_module.select_knowledge_tool_with_llm = select_knowledge_tool_with_llm
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
            "select_knowledge_tool_with_llm": globals_dict["select_knowledge_tool_with_llm"],
            "WorkflowExecutor": globals_dict["WorkflowExecutor"],
        }
        knowledge = types.SimpleNamespace(
            question="根据公开 ID 查询所有文档",
            params='{"type":"workflow","mode":"context_chain","steps":[]}',
            type=2,
            tool_id=0,
            answer="Prefer PDF documents when available.",
            description="workflow admin notes should stay out",
        )
        async def fake_selector(*args, **kwargs):
            return knowledge, None

        globals_dict["get_knowledge_tool"] = lambda *args, **kwargs: (knowledge, None)
        globals_dict["select_knowledge_tool_with_llm"] = fake_selector
        FakeWorkflowExecutor.result = workflow_result
        FakeWorkflowExecutor.calls = []
        globals_dict["WorkflowExecutor"] = FakeWorkflowExecutor
        self.addCleanup(lambda: globals_dict.update(originals))

    async def test_create_agent_uses_llm_knowledge_selector(self):
        GeneralAgent = self._load_general_agent_class()
        globals_dict = GeneralAgent.create_agent.__globals__
        originals = {
            "select_knowledge_tool_with_llm": globals_dict["select_knowledge_tool_with_llm"],
            "WorkflowExecutor": globals_dict["WorkflowExecutor"],
        }
        selector_calls = []
        knowledge = types.SimpleNamespace(
            question="workflow",
            params='{"type":"workflow","mode":"context_chain","steps":[]}',
            type=2,
            tool_id=0,
            answer="workflow instructions",
            description="routing hint",
        )

        async def fake_selector(user_id, prompt, complete_json, **kwargs):
            selector_calls.append({
                "user_id": user_id,
                "prompt": prompt,
                "complete_json": complete_json,
                "kwargs": kwargs,
            })
            return knowledge, None

        globals_dict["select_knowledge_tool_with_llm"] = fake_selector
        FakeWorkflowExecutor.result = FakeWorkflowResult({"ok": True})
        FakeWorkflowExecutor.calls = []
        globals_dict["WorkflowExecutor"] = FakeWorkflowExecutor
        self.addCleanup(lambda: globals_dict.update(originals))

        agent = GeneralAgent.__new__(GeneralAgent)
        agent.llm = FakeLlm()
        agent.memory = FakeMemory()
        agent.logger = FakeLogger()

        openai_agent = await agent.create_agent(
            user_id="user-1",
            prompt="run workflow",
            query_id="query-1",
            tool_data="",
            callback_handler=FakeCallbackHandler(),
            push_filter=2,
        )

        self.assertIsNone(openai_agent)
        self.assertEqual(selector_calls[0]["user_id"], "user-1")
        self.assertEqual(selector_calls[0]["prompt"], "run workflow")
        self.assertIs(selector_calls[0]["complete_json"].__self__, agent.llm)
        self.assertEqual(selector_calls[0]["complete_json"].__func__, agent.llm.complete_json.__func__)
        self.assertEqual(selector_calls[0]["kwargs"]["push_filter"], 2)

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
        self.assertEqual(
            FakeWorkflowExecutor.calls[0]["workflow_knowledge"].answer,
            "Prefer PDF documents when available.",
        )
        self.assertIn("Prefer PDF documents when available.", stream_content)
        self.assertNotIn("workflow admin notes should stay out", stream_content)

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

    def test_deterministic_markdown_renderer_preserves_nested_result_data(self):
        GeneralAgent = self._load_general_agent_class()

        agent = GeneralAgent.__new__(GeneralAgent)
        long_url = "https://example.com/download/" + ("a" * 650) + ".pdf"

        markdown = agent._render_list_as_md(None, [
            {
                "applicationNumberText": "18893954",
                "documentBag": [
                    {
                        "documentCode": "SPEC",
                        "downloadOptionBag": [
                            {
                                "mimeType": "application/pdf",
                                "downloadUrl": long_url,
                            }
                        ],
                    },
                    {
                        "documentCode": "CLM",
                    },
                ],
            }
        ])

        self.assertIn("applicationNumberText", markdown)
        self.assertIn("18893954", markdown)
        self.assertIn("documentBag", markdown)
        self.assertIn("documentCode", markdown)
        self.assertIn("SPEC", markdown)
        self.assertIn("downloadOptionBag", markdown)
        self.assertIn("mimeType", markdown)
        self.assertIn("application/pdf", markdown)
        self.assertIn(long_url, markdown)
        self.assertNotIn("[2 items]", markdown)
        self.assertNotIn("[truncated", markdown)

    async def test_filters_pending_raw_items_before_batch_display(self):
        GeneralAgent = self._load_general_agent_class()

        agent = GeneralAgent.__new__(GeneralAgent)
        agent.llm = FakeLlm([
            {
                "has_filter_criteria": True,
                "filter_criteria": "Tokyo results",
            },
            {
                "has_filter_requirement": True,
                "decisions": [
                    {"index": 0, "keep": True, "confidence": 0.98, "reason": "Tokyo"},
                    {"index": 1, "keep": False, "confidence": 0.94, "reason": "Osaka"},
                    {"index": 2, "keep": True, "confidence": 0.97, "reason": "Tokyo"},
                ],
            },
        ])
        agent.memory = FakeMemory()
        agent.logger = FakeLogger()
        agent._last_user_prompt = "Please show these companies, but only keep Tokyo results."
        agent._pending_raw_items = [
            {"name": "Alpha", "city": "Tokyo", "url": "https://example.com/a"},
            {"name": "Beta", "city": "Osaka", "url": "https://example.com/b"},
            {"name": "Gamma", "city": "Tokyo", "url": "https://example.com/c"},
        ]
        callback_handler = FakeCallbackHandler()

        await agent.invoke_agent(agent=None, callback_handler=callback_handler)

        self.assertEqual(len(agent.llm.complete_json_calls), 2)
        self.assertEqual(len(agent.llm.stream_calls), 1)
        self.assertIn("Filtered Results (2 of 3 items)", "".join(callback_handler.tokens))
        batch_content = agent.llm.stream_calls[0]["user_content"]
        self.assertIn("Alpha", batch_content)
        self.assertIn("Gamma", batch_content)
        self.assertNotIn("Beta", batch_content)

    async def test_stream_raw_items_forwards_filter_status_to_callback(self):
        GeneralAgent = self._load_general_agent_class()

        agent = GeneralAgent.__new__(GeneralAgent)
        agent.llm = FakeLlm([
            {
                "has_filter_criteria": True,
                "filter_criteria": "Tokyo results",
            },
            {
                "has_filter_requirement": True,
                "decisions": [
                    {"index": 0, "keep": True, "confidence": 0.98, "reason": "Tokyo"},
                    {"index": 1, "keep": False, "confidence": 0.94, "reason": "Osaka"},
                    {"index": 2, "keep": True, "confidence": 0.97, "reason": "Tokyo"},
                ],
            },
        ])
        agent.memory = FakeMemory()
        agent.logger = FakeLogger()
        agent._last_user_prompt = "Please show these companies, but only keep Tokyo results."
        agent._pending_raw_items = [
            {"name": "Alpha", "city": "Tokyo"},
            {"name": "Beta", "city": "Osaka"},
            {"name": "Gamma", "city": "Tokyo"},
        ]
        callback_handler = FakeCallbackHandler()

        await agent.invoke_agent(agent=None, callback_handler=callback_handler)

        self.assertGreaterEqual(len(callback_handler.statuses), 3)
        self.assertEqual(callback_handler.statuses[0]["phase"], "criteria")
        batch_statuses = [
            status for status in callback_handler.statuses
            if status.get("phase") == "batch"
        ]
        self.assertEqual(len(batch_statuses), 1)
        self.assertIn("Filtering results 1-3 of 3", batch_statuses[0]["message"])
        self.assertTrue(batch_statuses[0]["transient"])
        self.assertEqual(callback_handler.statuses[-1]["phase"], "complete")

    async def test_skips_result_filtering_when_user_prompt_has_no_filter_criteria(self):
        GeneralAgent = self._load_general_agent_class()

        agent = GeneralAgent.__new__(GeneralAgent)
        agent.llm = FakeLlm(stream_outputs=["formatted full results"])
        agent.memory = FakeMemory()
        agent.logger = FakeLogger()
        agent._last_user_prompt = "List the patent documents."
        agent._pending_raw_items = [
            {"documentCode": "SPEC"},
            {"documentCode": "CLM"},
        ]
        callback_handler = FakeCallbackHandler()

        await agent.invoke_agent(agent=None, callback_handler=callback_handler)

        self.assertEqual(len(agent.llm.complete_json_calls), 1)
        self.assertEqual(len(agent.llm.stream_calls), 1)
        self.assertIn("Full Results (2 items)", "".join(callback_handler.tokens))

    async def test_stream_raw_items_emits_csv_and_xlsx_artifacts_for_long_lists(self):
        GeneralAgent = self._load_general_agent_class()

        agent = GeneralAgent.__new__(GeneralAgent)
        agent.llm = FakeLlm(stream_outputs=["formatted full results"])
        agent.memory = FakeMemory()
        agent.logger = FakeLogger()
        agent._last_user_prompt = "List these companies."
        agent._pending_raw_items = [
            {"name": f"Company {index}", "city": "Tokyo", "metrics": {"rank": index}}
            for index in range(1, 7)
        ]
        callback_handler = FakeCallbackHandler()

        await agent.invoke_agent(agent=None, callback_handler=callback_handler)

        streamed = "".join(callback_handler.tokens)
        self.assertIn("Full Results (6 items)", streamed)
        formats = {artifact["format"] for artifact in callback_handler.artifacts}
        self.assertEqual(formats, {"csv", "xlsx"})
        csv_artifact = next(
            artifact for artifact in callback_handler.artifacts
            if artifact["format"] == "csv"
        )
        self.assertEqual(csv_artifact["row_count"], 6)
        csv_text = csv_artifact["content"].decode("utf-8-sig")
        self.assertIn("name", csv_text)
        self.assertIn("metrics.rank", csv_text)
        self.assertIn("Company 1", csv_text)
        xlsx_artifact = next(
            artifact for artifact in callback_handler.artifacts
            if artifact["format"] == "xlsx"
        )
        self.assertEqual(xlsx_artifact["row_count"], 6)
        self.assertTrue(xlsx_artifact["content"].startswith(b"PK"))

    async def test_mid_sized_raw_item_uses_formatter_llm(self):
        GeneralAgent = self._load_general_agent_class()

        agent = GeneralAgent.__new__(GeneralAgent)
        agent.llm = FakeLlm(stream_outputs=["formatted patent output"])
        agent.memory = FakeMemory()
        agent.logger = FakeLogger()
        agent._last_user_prompt = "show patent details"
        agent._pending_raw_items = [
            {
                "applicationNumberText": "18893954",
                "applicationMetaData": {
                    "earliestPublicationNumber": "US20250014493A1",
                    "inventionTitle": "DISPLAY DEVICE",
                },
                "eventDataBag": [
                    {"eventCode": "EPG/", "eventDescriptionText": "Recordation of Patent eGrant"}
                ],
                "largePayload": "x" * 15000,
            }
        ]
        callback_handler = FakeCallbackHandler()

        await agent.invoke_agent(agent=None, callback_handler=callback_handler)

        streamed = "".join(callback_handler.tokens)
        self.assertEqual(len(agent.llm.stream_calls), 1)
        self.assertIn("formatted patent output", streamed)

    async def test_formatter_error_retries_single_item_contexts_before_fallback(self):
        GeneralAgent = self._load_general_agent_class()

        agent = GeneralAgent.__new__(GeneralAgent)
        agent.llm = FakeLlm(
            stream_errors=[RuntimeError("context length exceeded"), None, None],
            stream_outputs=["formatted item 1", "formatted item 2"],
        )
        agent.memory = FakeMemory()
        agent.logger = FakeLogger()
        agent._last_user_prompt = "show patent details"
        agent._pending_raw_items = [
            {
                "applicationNumberText": "18893954",
                "applicationMetaData": {
                    "earliestPublicationNumber": "US20250014493A1",
                    "inventionTitle": "DISPLAY DEVICE",
                },
            },
            {
                "applicationNumberText": "18893955",
                "applicationMetaData": {
                    "earliestPublicationNumber": "US20250014494A1",
                    "inventionTitle": "DISPLAY DEVICE 2",
                },
            }
        ]
        callback_handler = FakeCallbackHandler()

        with patch.dict(os.environ, {"GENERAL_AGENT_BATCH_FORMATTING_MODE": "existing"}):
            await agent.invoke_agent(agent=None, callback_handler=callback_handler)

        streamed = "".join(callback_handler.tokens)
        self.assertEqual(len(agent.llm.stream_calls), 3)
        self.assertIn("formatted item 1", streamed)
        self.assertIn("formatted item 2", streamed)

    async def test_oversized_single_item_uses_direct_llm_formatter_by_default(self):
        GeneralAgent = self._load_general_agent_class()

        agent = GeneralAgent.__new__(GeneralAgent)
        agent.llm = FakeLlm(stream_outputs=["formatted oversized item"])
        agent.memory = FakeMemory()
        agent.logger = FakeLogger()
        agent._last_user_prompt = "show patent details"
        agent._pending_raw_items = [
            {
                "applicationMetaData": {
                    "earliestPublicationNumber": "US20250014493A1",
                    "largeValue": "x" * 30000,
                },
                "eventDataBag": [
                    {
                        "eventDescriptionText": "Recordation of Patent eGrant",
                        "largeValue": "y" * 30000,
                    }
                ],
            }
        ]
        callback_handler = FakeCallbackHandler()

        await agent.invoke_agent(agent=None, callback_handler=callback_handler)

        streamed = "".join(callback_handler.tokens)
        self.assertEqual(len(agent.llm.stream_calls), 1)
        self.assertIn("formatted oversized item", streamed)
        self.assertIn('"applicationMetaData"', agent.llm.stream_calls[0]["user_content"])
        self.assertIn('"eventDataBag"', agent.llm.stream_calls[0]["user_content"])

    async def test_existing_batch_formatting_mode_prunes_oversized_values_instead_of_splitting(self):
        """Oversized values should be pruned proactively, so the item fits in one LLM call."""
        GeneralAgent = self._load_general_agent_class()

        agent = GeneralAgent.__new__(GeneralAgent)
        agent.llm = FakeLlm(stream_outputs=["formatted compact item"])
        agent.memory = FakeMemory()
        agent.logger = FakeLogger()
        agent._last_user_prompt = "show patent details"
        agent._pending_raw_items = [
            {
                "applicationMetaData": {
                    "earliestPublicationNumber": "US20250014493A1",
                    "largeValue": "x" * 30000,
                },
                "eventDataBag": [
                    {
                        "eventDescriptionText": "Recordation of Patent eGrant",
                        "largeValue": "y" * 30000,
                    }
                ],
            }
        ]
        callback_handler = FakeCallbackHandler()

        with patch.dict(os.environ, {"GENERAL_AGENT_BATCH_FORMATTING_MODE": "existing"}):
            await agent.invoke_agent(agent=None, callback_handler=callback_handler)

        streamed = "".join(callback_handler.tokens)
        # Long values are pruned before reaching the LLM — single call is enough.
        self.assertEqual(len(agent.llm.stream_calls), 1)
        self.assertIn("formatted compact item", streamed)
        # The oversized values must NOT appear in the LLM input.
        user_content = agent.llm.stream_calls[0]["user_content"]
        self.assertNotIn("x" * 100, user_content)
        self.assertNotIn("y" * 100, user_content)
        # Non-oversized metadata should still be present.
        self.assertIn("US20250014493A1", user_content)
        self.assertIn("Recordation of Patent eGrant", user_content)

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
