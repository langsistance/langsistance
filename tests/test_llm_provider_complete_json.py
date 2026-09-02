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

    async def astream(self, messages):
        # complete_json collects chunks via llm.astream (streaming path).
        FakeChatOpenAI.last_messages = messages
        yield types.SimpleNamespace(content='```json\n{"keep": true}\n```')


def _load_provider_class():
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


class TestLlmProviderCompleteJson(unittest.IsolatedAsyncioTestCase):

    def _load_provider_class(self):
        return _load_provider_class()

    async def test_complete_json_returns_parsed_non_streaming_response(self):
        Provider = self._load_provider_class()
        provider = Provider("test", "gpt-test")

        result = await provider.complete_json("system prompt", "user content")

        self.assertEqual(result, {"keep": True})
        self.assertEqual(
            FakeChatOpenAI.last_messages,
            [("system", "system prompt"), ("human", "user content")],
        )
        self.assertTrue(FakeChatOpenAI.last_init_kwargs["streaming"])
        self.assertEqual(FakeChatOpenAI.last_init_kwargs["temperature"], 0)


class TestLlmProviderFlashThinking(unittest.IsolatedAsyncioTestCase):
    """deepseek flash models disable thinking by default (scoring latency).

    The toggle lives in _get_langchain_llm: model_kwargs thinking disabled
    only for deepseek + "flash"-named models, unless DEEPSEEK_THINKING
    =enabled.  Non-flash deepseek and other providers are never touched.
    """

    def _llm_init_kwargs(self, provider_name: str, model: str) -> dict:
        Provider = _load_provider_class()
        provider = Provider(provider_name, model, is_local=True)
        provider._get_langchain_llm()
        return FakeChatOpenAI.last_init_kwargs

    def test_deepseek_flash_disables_thinking_by_default(self):
        kwargs = self._llm_init_kwargs("deepseek", "deepseek-v4-flash")
        self.assertEqual(
            kwargs.get("model_kwargs"),
            {"thinking": {"type": "disabled"}},
        )

    def test_non_flash_deepseek_untouched(self):
        kwargs = self._llm_init_kwargs("deepseek", "deepseek-chat")
        self.assertNotIn("model_kwargs", kwargs)

    def test_other_provider_untouched(self):
        kwargs = self._llm_init_kwargs("minimax", "MiniMax-M2.7-highspeed")
        self.assertNotIn("model_kwargs", kwargs)

    def test_env_override_reenables_thinking(self):
        with patch.dict(os.environ, {"DEEPSEEK_THINKING": "enabled"}):
            kwargs = self._llm_init_kwargs("deepseek", "deepseek-v4-flash")
        self.assertNotIn("model_kwargs", kwargs)


if __name__ == "__main__":
    unittest.main()
