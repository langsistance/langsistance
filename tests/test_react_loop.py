"""Tests for the hand-rolled ReAct loop (sources/agents/react_loop.py)."""
import asyncio
import unittest

from sources.agents.react_loop import ReActLoop, make_llm_call
from langchain_core.messages import AIMessageChunk


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

    def test_final_result_ends_loop_with_no_tools_answer(self):
        # A document-list tool marks its observation "final": the loop
        # gives the LLM one no-tools pass to phrase the answer and ends —
        # no further tool rounds (observed: after the 68-document result
        # the loop kept running search/spec rounds and streamed pool
        # patents instead of the documents).
        model = _FakeModel([
            ("", [{"id": "c1", "name": "documents", "args": {}}], ""),
            ("已获取 68 份文档。", [], ""),
        ])
        executor = _FakeExecutor({
            "documents": {"kind": "observation",
                          "text": "工具返回 68 条记录（文档列表不截断）",
                          "final": True},
        })
        events = _Events()
        result, messages = _run(ReActLoop(model, executor, events), model, executor)
        self.assertEqual(result.kind, "answer")
        self.assertEqual(result.answer_text, "已获取 68 份文档。")
        self.assertEqual(result.steps, 1)
        self.assertEqual(executor.seen, [("documents", {}, 1)])
        # user + assistant(tool_calls) + tool + assistant(final answer)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[-1]["role"], "assistant")
        # the final phrasing call had NO tools bound
        self.assertEqual(model.calls[-1][1], [])

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


class _TokenRecorder:
    """Collects every token the adapter forwards to the handler."""

    def __init__(self):
        self.tokens = []

    async def on_llm_new_token(self, content):
        self.tokens.append(content)


class _FakeStreamLLM:
    """langchain-shaped fake: bind_tools no-op, astream yields given chunks."""

    def __init__(self, chunks):
        self.chunks = list(chunks)

    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        for chunk in self.chunks:
            yield chunk


class _FakeStreamProvider:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def _get_langchain_llm(self, streaming=True):
        return _FakeStreamLLM(self._chunks)


def _tool_chunk(content, args="{}"):
    return AIMessageChunk(
        content=content,
        tool_call_chunks=[
            {"index": 0, "id": "call_0", "name": "search_patent", "args": args},
        ],
    )


class TestLLMCallStreamSemantics(unittest.TestCase):
    """Narration routing in make_llm_call (2026-09-06 stream fix).

    A tool round whose provider writes pre-tool narration into *content* must
    NOT stream it as answer tokens — it folds into the step reasoning.  A
    pure-text round under bound tools (the final answer) replays its buffer so
    progressive typing survives.  Without tools, forwarding stays live.
    """

    def _call(self, chunks, tools):
        rec = _TokenRecorder()
        llm_call = make_llm_call(_FakeStreamProvider(chunks), handler=rec)
        return asyncio.run(llm_call([{"role": "user", "content": "?"}], tools)), rec

    def test_tool_round_narration_not_streamed_and_folded(self):
        (text, calls, reasoning), rec = self._call(
            [_tool_chunk("I'll search the ladder first.")], tools=[{"name": "t"}])
        self.assertEqual(text, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "search_patent")
        self.assertEqual(rec.tokens, [], "tool-round narration must not stream")
        self.assertIn("I'll search the ladder first.", reasoning)

    def test_tool_round_folds_narration_after_provider_thinking(self):
        chunk = AIMessageChunk(
            content="narration text",
            additional_kwargs={"reasoning_content": "provider think"},
            tool_call_chunks=[
                {"index": 0, "id": "c1", "name": "lookup", "args": "{}"},
            ],
        )
        (text, calls, reasoning), rec = self._call([chunk], tools=[{"name": "t"}])
        self.assertEqual(text, "")
        self.assertIn("provider think", reasoning)
        self.assertIn("narration text", reasoning)
        self.assertLess(reasoning.index("provider think"),
                        reasoning.index("narration text"))
        self.assertEqual(rec.tokens, [])

    def test_final_answer_round_replays_buffered_text(self):
        (text, calls, reasoning), rec = self._call(
            [AIMessageChunk(content="Here is "),
             AIMessageChunk(content="the final answer.")],
            tools=[{"name": "t"}])
        self.assertEqual(text, "Here is the final answer.")
        self.assertEqual(calls, [])
        self.assertEqual("".join(rec.tokens), "Here is the final answer.")

    def test_no_tools_forwards_live_and_returns_text(self):
        (text, calls, reasoning), rec = self._call(
            [AIMessageChunk(content="plain answer")], tools=[])
        self.assertEqual(text, "plain answer")
        self.assertEqual(calls, [])
        self.assertEqual(rec.tokens, ["plain answer"])

    def test_tool_round_empty_content_no_reasoning_noop(self):
        chunk = AIMessageChunk(
            content="", tool_call_chunks=[
                {"index": 0, "id": "c1", "name": "lookup", "args": "{}"},
            ])
        (text, calls, reasoning), rec = self._call([chunk], tools=[{"name": "t"}])
        self.assertEqual(text, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(reasoning, "")
        self.assertEqual(rec.tokens, [])


class TestToolMessagePairing(unittest.TestCase):
    """2026-09-15 (需求#32)：assistant 的每个 tool_call 都必须有配对的 tool
    结果，否则 OpenAI 兼容端点整轮 400 "No tool output found for function
    call …"（2026-09-14 生产：整轮中断，用户只看到"连接中断"）。
    sanitize_tool_message_pairs 在 llm_call 出站前补齐/清理 —— 任何上游路径
    都不能把残缺历史送到 provider。
    """

    def test_placeholder_added_for_missing_output(self):
        from sources.agents.react_loop import sanitize_tool_message_pairs
        messages = [
            {"role": "user", "content": "?"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "call_X", "name": "fetch", "args": {}}]},
        ]
        out = sanitize_tool_message_pairs(messages)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[2]["role"], "tool")
        self.assertEqual(out[2]["tool_call_id"], "call_X")
        self.assertEqual(out[2]["name"], "fetch")
        self.assertIn("missing", out[2]["content"])

    def test_second_call_of_a_round_filled_too(self):
        from sources.agents.react_loop import sanitize_tool_message_pairs
        messages = [
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "call_1", "name": "a", "args": {}},
                            {"id": "call_2", "name": "b", "args": {}}]},
            {"role": "tool", "tool_call_id": "call_1", "name": "a",
             "content": "ok"},
        ]
        out = sanitize_tool_message_pairs(messages)
        self.assertEqual([m.get("tool_call_id") for m in out
                          if m["role"] == "tool"], ["call_1", "call_2"])

    def test_paired_history_untouched_and_orphan_dropped(self):
        from sources.agents.react_loop import sanitize_tool_message_pairs
        paired = [
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "name": "t", "args": {}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "t", "content": "ok"},
        ]
        self.assertEqual(sanitize_tool_message_pairs(paired), paired)
        orphan = [{"role": "tool", "tool_call_id": "ghost", "content": "x"}]
        self.assertEqual(sanitize_tool_message_pairs(orphan), [])

    def test_llm_call_sends_repaired_history_to_provider(self):
        # 落点验证：make_llm_call 出站前确实跑了 sanitizer。
        seen = {}

        class _RecordingLLM:
            def bind_tools(self, tools):
                return self

            async def astream(self, messages):
                seen["messages"] = list(messages)
                yield AIMessageChunk(content="ok")

        class _Provider:
            def _get_langchain_llm(self, streaming=True):
                return _RecordingLLM()

        llm_call = make_llm_call(_Provider())
        messages = [
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "call_X", "name": "t", "args": {}}]},
        ]
        text, calls, _reasoning = asyncio.run(llm_call(messages, []))
        roles = [type(m).__name__ for m in seen["messages"]]
        self.assertIn("ToolMessage", roles)
        self.assertEqual(
            seen["messages"][-1].tool_call_id, "call_X",
            "provider 必须看到与 tool_call 配对的 ToolMessage")
        self.assertEqual(text, "ok")


class TestUnpairedCallsOnEarlyReturn(unittest.TestCase):
    """需求#32: 同轮多个 tool_call、中途 return 时其余调用必须补齐结果。

    sanitize_tool_message_pairs 是出站前的最后防线，但循环自己提前 return
    时（long_task / final）应当就地补齐 —— 只有一层防线意味着上游任何一次
    改动都可能把残缺历史送到 provider（2026-09-14 生产：整轮 400，用户只
    看到"连接中断"）。
    """

    def test_long_task_return_pairs_remaining_calls(self):
        model = _FakeModel([("", [
            {"id": "c1", "name": "a", "args": {}},
            {"id": "c2", "name": "b", "args": {}},
        ], "")])
        executor = _FakeExecutor({"a": {"kind": "long_task",
                                        "knowledge": "k", "tool_info": None}})
        loop = ReActLoop(model, executor, _Events(), max_rounds=3)
        _result, messages = _run(loop, model, executor)
        tool_ids = [m.get("tool_call_id") for m in messages
                    if m.get("role") == "tool"]
        self.assertEqual(tool_ids, ["c1", "c2"])

    def test_final_return_pairs_remaining_calls(self):
        model = _FakeModel([
            ("", [{"id": "c1", "name": "a", "args": {}},
                  {"id": "c2", "name": "b", "args": {}}], ""),
            ("答案", [], ""),
        ])
        executor = _FakeExecutor({"a": {"kind": "observation", "text": "ok",
                                        "final": True}})
        loop = ReActLoop(model, executor, _Events(), max_rounds=3)
        _result, messages = _run(loop, model, executor)
        tool_ids = [m.get("tool_call_id") for m in messages
                    if m.get("role") == "tool"]
        self.assertEqual(tool_ids, ["c1", "c2"])


class TestPlaceholderFailureReachesTheGuard(unittest.TestCase):
    """需求#32 × #33: 占位符失败必须以"错误"的身份进入循环。

    这是 2026-09-14 事故的因果链：占位符闸门返回一个成功语义的字典 →
    `obs.startswith("Error:")` 为假 → 不计入 consecutive_failures → "连续
    失败就收手并补齐本轮调用"的保护不触发 → 模型盲目重试 → 整轮被 provider
    以 400 拒绝（悬空 tool_call）。闸门改为 raise 后，下面这条链才成立。
    """

    def test_placeholder_error_triggers_the_runaway_guard(self):
        # 执行器抛出占位符错误（真实路径里由 execute_action 的 try 转成
        # "Error: ..." 观察结果）。
        class _RaisingExecutor:
            async def __call__(self, name, args, round_no):
                raise ValueError(
                    "Request failed: missing value for URL parameter(s): "
                    "applicationNumberText")

        model = _FakeModel([
            ("", [{"id": "c1", "name": "docs", "args": {}}], ""),
            ("", [{"id": "c2", "name": "docs", "args": {}}], ""),
            ("无法取得该文件。", [], ""),
        ])
        executor = _RaisingExecutor()
        result, messages = _run(
            ReActLoop(model, executor, _Events()), model, executor)

        self.assertEqual(result.kind, "fallback")   # 保护生效, 没有继续重试
        self.assertEqual(result.steps, 2)
        # 每个 tool_call 都有配对结果 —— 送给 provider 的历史不残缺
        assistant = [m for m in messages if m.get("tool_calls")]
        tool_ids = [m.get("tool_call_id") for m in messages
                    if m.get("role") == "tool"]
        for msg in assistant:
            for call in msg["tool_calls"]:
                self.assertIn(call["id"], tool_ids)
