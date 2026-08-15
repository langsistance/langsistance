"""Tests for the hand-rolled ReAct loop (sources/agents/react_loop.py)."""
import asyncio
import unittest

from sources.agents.react_loop import ReActLoop


class _FakeModel:
    """Scripted LLM: pops queued (text, tool_calls, reasoning) responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def __call__(self, messages, tools):
        self.calls.append((list(messages), list(tools)))
        return self.responses.pop(0)


class _FakeExecutor:
    def __init__(self, results):
        self.results = dict(results)
        self.seen = []

    async def __call__(self, name, args, round_no):
        self.seen.append((name, args, round_no))
        return self.results[name]


class _Events(list):
    async def __call__(self, event_type, payload):
        self.append((event_type, payload))


def _run(loop, model, executor, tools=None):
    messages = [{"role": "user", "content": "?"}]
    return asyncio.run(
        loop.run(messages, tools if tools is not None else [])
    ), messages


class TestReActLoop(unittest.TestCase):
    def test_direct_answer_without_tools(self):
        model = _FakeModel([("直接回答", [], "")])
        events = _Events()
        result, _ = _run(ReActLoop(model, _FakeExecutor({}), events), model, _FakeExecutor({}))
        self.assertEqual(result.kind, "answer")
        self.assertEqual(result.steps, 0)
        self.assertEqual(events[-1][0], "agent_elapsed")
        self.assertEqual(events[-1][1]["steps"], 0)

    def test_should_stop_terminates_before_next_round(self):
        model = _FakeModel([("", [{"id": "c1", "name": "t", "args": {}}], "")])
        executor = _FakeExecutor({"t": {"kind": "observation", "text": "ok"}})
        stop_flag = {"stop": False}
        loop = ReActLoop(model, executor, _Events(), max_rounds=5,
                         should_stop=lambda: stop_flag["stop"])
        stop_flag["stop"] = True
        result, _ = _run(loop, model, executor)
        self.assertEqual(result.kind, "fallback")
        self.assertEqual(result.steps, 0)

    def test_single_tool_round_then_answer(self):
        model = _FakeModel([
            ("", [{"id": "c1", "name": "search", "args": {"q": "x"}}], "需要检索"),
            ("答案", [], ""),
        ])
        executor = _FakeExecutor({"search": {"kind": "observation", "text": "返回 3 条"}})
        events = _Events()
        result, messages = _run(ReActLoop(model, executor, events), model, executor)
        self.assertEqual(result.kind, "answer")
        self.assertEqual(result.steps, 1)
        self.assertEqual(executor.seen, [("search", {"q": "x"}, 1)])
        # assistant(tool_calls) + tool messages appended
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertEqual(messages[-1]["tool_call_id"], "c1")
        self.assertEqual(messages[-1]["content"], "返回 3 条")
        self.assertEqual([e[0] for e in events],
                         ["step", "status", "observation", "agent_elapsed"])
        self.assertEqual(events[0][1]["action"], "search")
        self.assertEqual(events[0][1]["reasoning_text"], "需要检索")

    def test_tool_error_then_recovery_via_other_tool(self):
        model = _FakeModel([
            ("", [{"id": "c1", "name": "a", "args": {}}], ""),
            ("", [{"id": "c2", "name": "b", "args": {}}], ""),
            ("回答", [], ""),
        ])
        executor = _FakeExecutor({
            "a": {"kind": "observation", "text": "Error: boom"},
            "b": {"kind": "observation", "text": "ok"},
        })
        result, _ = _run(ReActLoop(model, executor, _Events()), model, executor)
        self.assertEqual(result.kind, "answer")

    def test_same_tool_fails_twice_triggers_fallback(self):
        model = _FakeModel([
            ("", [{"id": "c1", "name": "a", "args": {}}], ""),
            ("", [{"id": "c2", "name": "a", "args": {}}], ""),
            ("抱歉，无法完成。", [], ""),
        ])
        executor = _FakeExecutor({"a": {"kind": "observation", "text": "Error: boom"}})
        result, _ = _run(ReActLoop(model, executor, _Events()), model, executor)
        self.assertEqual(result.kind, "fallback")
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.answer_text, "抱歉，无法完成。")

    def test_long_task_kind_terminates(self):
        model = _FakeModel([("", [{"id": "c1", "name": "lt", "args": {}}], "")])
        executor = _FakeExecutor({"lt": {"kind": "long_task", "knowledge": "K", "tool_info": "T"}})
        result, _ = _run(ReActLoop(model, executor, _Events()), model, executor)
        self.assertEqual(result.kind, "long_task")
        self.assertEqual(result.long_task_knowledge, "K")
        self.assertEqual(result.long_task_tool_info, "T")

    def test_max_rounds_fallback(self):
        responses = [("", [{"id": f"c{i}", "name": "t", "args": {}}], "") for i in range(3)]
        responses.append(("达到上限的总结", [], ""))
        model = _FakeModel(responses)
        executor = _FakeExecutor({"t": {"kind": "observation", "text": "ok"}})
        result, _ = _run(ReActLoop(model, executor, _Events(), max_rounds=3), model, executor)
        self.assertEqual(result.kind, "fallback")
        self.assertEqual(result.steps, 3)
        self.assertEqual(result.answer_text, "达到上限的总结")

    def test_mount_tools_appended_and_bound_next_round(self):
        model = _FakeModel([
            ("", [{"id": "c1", "name": "search_my_knowledge", "args": {}}], ""),
            ("回答", [], ""),
        ])
        executor = _FakeExecutor({
            "search_my_knowledge": {
                "kind": "observation", "text": "找到 1 项",
                "mount_tools": [{"name": "uspto", "description": "d", "parameters": {}}],
            },
        })
        tools = []
        result, _ = _run(ReActLoop(model, executor, _Events()), model, executor, tools)
        self.assertEqual(result.kind, "answer")
        self.assertEqual(tools[-1]["name"], "uspto")
        self.assertEqual(model.calls[1][1][-1]["name"], "uspto")

    def test_executor_exception_becomes_error_observation(self):
        class _Boom:
            async def __call__(self, name, args, round_no):
                raise RuntimeError("boom")

        model = _FakeModel([
            ("", [{"id": "c1", "name": "a", "args": {}}], ""),
            ("回答", [], ""),
        ])
        result, messages = _run(ReActLoop(model, _Boom(), _Events()), model, _Boom())
        self.assertEqual(result.kind, "answer")
        self.assertIn("Error: boom", messages[-1]["content"])

    def test_fallback_completes_tool_outputs_for_all_calls(self):
        """Same-round multiple failing calls → fallback LLM call must carry
        a tool output for EVERY assistant tool_call (OpenAI-compatible APIs
        reject histories with dangling tool_calls: 400 'No tool output
        found for function call ...')."""
        model = _FakeModel([
            ("", [{"id": "c1", "name": "a", "args": {}},
                  {"id": "c2", "name": "a", "args": {}}], ""),
            ("抱歉，无法完成。", [], ""),
        ])
        executor = _FakeExecutor({"a": {"kind": "observation", "text": "Error: boom"}})
        result, _ = _run(ReActLoop(model, executor, _Events()), model, executor)
        self.assertEqual(result.kind, "fallback")
        fallback_messages = model.calls[-1][0]
        assistant = next(m for m in fallback_messages
                         if m.get("role") == "assistant")
        call_ids = [c["id"] for c in assistant["tool_calls"]]
        tool_ids = [m.get("tool_call_id") for m in fallback_messages
                    if m.get("role") == "tool"]
        self.assertEqual(set(call_ids), set(tool_ids))
        self.assertEqual(len(tool_ids), 2)

    def test_status_emitted_for_each_tool_call(self):
        """A transient status accompanies every tool call so old frontends
        see live progress during silent tool rounds (no token stream)."""
        model = _FakeModel([
            ("", [{"id": "c1", "name": "search", "args": {"q": "x"}}], ""),
            ("答案", [], ""),
        ])
        events = _Events()
        executor = _FakeExecutor({"search": {"kind": "observation", "text": "3 条"}})
        _run(ReActLoop(model, executor, events), model, executor)
        statuses = [p for t, p in events if t == "status"]
        self.assertEqual(len(statuses), 1)
        self.assertIn("search", statuses[0]["message"])
        self.assertIn("正在调用", statuses[0]["message"])


if __name__ == "__main__":
    unittest.main()
