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
    def get(self):
        return []


class FakeLogger:
    def info(self, message):
        pass


class FakeLlm:
    def __init__(self):
        self.stream_calls = []

    async def openai_invoke(self, agent, memory, callback_handler):
        return None

    async def stream_simple(self, system_prompt, user_content, callback_handler):
        self.stream_calls.append({
            "system_prompt": system_prompt,
            "user_content": user_content,
        })


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
        self.assertIn("https://api.copiioai.com/uspto/download?url=", combined_prompt)


if __name__ == "__main__":
    unittest.main()
