import asyncio
import sys
import types
import unittest


class TestSSECallbackHandler(unittest.IsolatedAsyncioTestCase):

    async def test_on_status_puts_transient_status_event_on_queue(self):
        callbacks_module = types.ModuleType("langchain_core.callbacks.base")

        class AsyncCallbackHandler:
            pass

        callbacks_module.AsyncCallbackHandler = AsyncCallbackHandler
        sys.modules.setdefault("langchain_core", types.ModuleType("langchain_core"))
        sys.modules.setdefault("langchain_core.callbacks", types.ModuleType("langchain_core.callbacks"))
        sys.modules["langchain_core.callbacks.base"] = callbacks_module

        from sources.callback.sse_callback import SSECallbackHandler

        queue = asyncio.Queue()
        handler = SSECallbackHandler(queue)

        await handler.on_status(
            "Filtering results 1-5 of 12...",
            phase="batch",
            current=1,
            end=5,
            total=12,
        )

        event = await queue.get()
        self.assertEqual(event["type"], "status")
        self.assertEqual(event["message"], "Filtering results 1-5 of 12...")
        self.assertEqual(event["phase"], "batch")
        self.assertEqual(event["current"], 1)
        self.assertEqual(event["end"], 5)
        self.assertEqual(event["total"], 12)
        self.assertTrue(event["transient"])


if __name__ == "__main__":
    unittest.main()
