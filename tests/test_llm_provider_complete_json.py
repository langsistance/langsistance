import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import patch


class FakeLogger:
    def info(self, message):
        pass

    def warning(self, message):
        pass


class FakeChatOpenAI:
    last_init_kwargs = None
    last_messages = None

    def __init__(self, **kwargs):
        FakeChatOpenAI.last_init_kwargs = kwargs

    async def ainvoke(self, messages):
        FakeChatOpenAI.last_messages = messages
        return types.SimpleNamespace(content='```json\n{"keep": true}\n```')


class TestLlmProviderCompleteJson(unittest.IsolatedAsyncioTestCase):

    def _load_provider_class(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        module_path = os.path.join(repo_root, "sources", "llm_provider.py")

        dotenv_module = types.ModuleType("dotenv")
        dotenv_module.load_dotenv = lambda *args, **kwargs: None

        httpx_module = types.ModuleType("httpx")

        requests_module = types.ModuleType("requests")

        ollama_module = types.ModuleType("ollama")
        ollama_module.Client = object

        openai_module = types.ModuleType("openai")
        openai_module.OpenAI = object

        langchain_openai_module = types.ModuleType("langchain_openai")
        langchain_openai_module.ChatOpenAI = FakeChatOpenAI

        prompts_module = types.ModuleType("langchain_core.prompts")
        prompts_module.ChatPromptTemplate = object
        prompts_module.MessagesPlaceholder = object

        agents_module = types.ModuleType("langchain.agents")
        agents_module.create_agent = lambda *args, **kwargs: None

        logger_module = types.ModuleType("sources.logger")
        logger_module.Logger = lambda *args, **kwargs: FakeLogger()

        utility_module = types.ModuleType("sources.utility")
        utility_module.pretty_print = lambda *args, **kwargs: None
        utility_module.animate_thinking = lambda *args, **kwargs: None

        stubs = {
            "httpx": httpx_module,
            "requests": requests_module,
            "dotenv": dotenv_module,
            "ollama": ollama_module,
            "openai": openai_module,
            "langchain_openai": langchain_openai_module,
            "langchain_core.prompts": prompts_module,
            "langchain.agents": agents_module,
            "sources.logger": logger_module,
            "sources.utility": utility_module,
        }

        spec = importlib.util.spec_from_file_location(
            "test_loaded_llm_provider",
            module_path,
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, stubs):
            spec.loader.exec_module(module)
        return module.Provider

    async def test_complete_json_returns_parsed_non_streaming_response(self):
        Provider = self._load_provider_class()
        provider = Provider("test", "gpt-test")

        result = await provider.complete_json("system prompt", "user content")

        self.assertEqual(result, {"keep": True})
        self.assertEqual(
            FakeChatOpenAI.last_messages,
            [("system", "system prompt"), ("human", "user content")],
        )
        self.assertFalse(FakeChatOpenAI.last_init_kwargs["streaming"])
        self.assertEqual(FakeChatOpenAI.last_init_kwargs["temperature"], 0)


if __name__ == "__main__":
    unittest.main()
