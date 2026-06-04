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

    async def test_on_artifacts_puts_chunked_download_events_on_queue(self):
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

        await handler.on_artifacts([
            {
                "artifact_id": "artifact-1",
                "format": "csv",
                "filename": "results.csv",
                "mime_type": "text/csv;charset=utf-8",
                "row_count": 6,
                "column_count": 2,
                "content": b"name\nAlpha\n",
            }
        ])

        start = await queue.get()
        chunk = await queue.get()
        end = await queue.get()

        self.assertEqual(start["type"], "artifact_start")
        self.assertEqual(start["artifact_id"], "artifact-1")
        self.assertEqual(start["filename"], "results.csv")
        self.assertNotIn("content", start)
        self.assertEqual(chunk["type"], "artifact_chunk")
        self.assertEqual(chunk["artifact_id"], "artifact-1")
        self.assertEqual(chunk["data"], "bmFtZQpBbHBoYQo=")
        self.assertEqual(end, {"type": "artifact_end", "artifact_id": "artifact-1"})

    async def test_on_agent_finish_does_not_end_stream_before_post_agent_events(self):
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

        await handler.on_agent_finish(finish=None)

        self.assertTrue(queue.empty())


if __name__ == "__main__":
    unittest.main()
