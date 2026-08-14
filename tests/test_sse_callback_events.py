"""SSE events for the ReAct loop: step / observation / agent_elapsed."""
import asyncio
import json
import unittest

from sources.callback.sse_callback import SSECallbackHandler


class TestSSECallbackReActEvents(unittest.TestCase):
    def setUp(self):
        # Bind an explicit event loop so queue/coroutine share the same loop.
        # Python 3.14 no longer provides an implicit loop in the main thread.
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.queue = asyncio.Queue()
        self.handler = SSECallbackHandler(self.queue)

    def tearDown(self):
        self.loop.close()
        asyncio.set_event_loop(None)

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def test_on_step_pushes_step_event(self):
        self._run(self.handler.on_step(
            2, "第 2 步 · 正在调用「美国专利检索」", "uspto_search",
            params_brief='{"query":"apple"}',
            reasoning_text="need search",
        ))
        event = self.queue.get_nowait()
        self.assertEqual(event["type"], "step")
        self.assertEqual(event["round"], 2)
        self.assertEqual(event["thought"], "第 2 步 · 正在调用「美国专利检索」")
        self.assertEqual(event["action"], "uspto_search")
        self.assertEqual(event["params_brief"], '{"query":"apple"}')
        self.assertEqual(event["reasoning_text"], "need search")

    def test_on_observation_pushes_observation_event(self):
        self._run(self.handler.on_observation(2, "返回 3 条"))
        event = self.queue.get_nowait()
        self.assertEqual(event, {
            "type": "observation", "round": 2, "result_brief": "返回 3 条",
        })

    def test_on_agent_elapsed_pushes_elapsed_event(self):
        self._run(self.handler.on_agent_elapsed(3.2, 5))
        event = self.queue.get_nowait()
        self.assertEqual(event, {
            "type": "agent_elapsed", "elapsed_seconds": 3.2, "steps": 5,
        })

    def test_all_payloads_are_json_serializable(self):
        self._run(self.handler.on_step(1, "t", "a", reasoning_text="r"))
        self._run(self.handler.on_observation(1, "o"))
        self._run(self.handler.on_agent_elapsed(1.0, 1))
        for _ in range(3):
            event = self.queue.get_nowait()
            json.dumps(event, ensure_ascii=False)  # must not raise


if __name__ == "__main__":
    unittest.main()
