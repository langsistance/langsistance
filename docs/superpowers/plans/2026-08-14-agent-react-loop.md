# Agent ReAct 循环升级 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把聊天管线从「单选 1 个 knowledge → 单发工具调用」升级为手写轻量 ReAct 循环（思考→决策→行动→观察），对话窗口按 workbuddy 风格展示步骤并折叠为「已用时间 · N 步」。

**Architecture:** 新模块 `sources/agents/react_loop.py`（循环核心，依赖注入、无 LangChain 即可测试）+ `sources/agents/react_tools.py`（工具供给与行动分派）；`GeneralAgent.create_agent` 改为构建工具集并运行循环（长任务返回 intent 标记，其余返回 None），`invoke_agent` 只做收尾（raw_items 流式批处理 + 多轮存储）；SSE 新增 step/observation/agent_elapsed 事件；前端 `useChatStream` 累积 `agentSteps`/`elapsedSeconds`，`MarkdownMessage` 渲染步骤行 + 折叠时间线。

**Tech Stack:** Python/FastAPI/LangChain（bind_tools/astream，仅做通道）、pytest(unittest)、Next.js React 19 + TypeScript、node:test、i18n (en/zh)。

**Spec:** `docs/superpowers/specs/2026-08-14-agent-react-loop-design.md`（commit b2f1f02）

## Global Constraints

- 后端测试必须 `PYTHONUTF8=1 python -m pytest ...`（Windows GBK 问题）；前端测试 `node --test lib/*.test.mjs`，构建 `npm run build`
- 工作区有**无关的未提交改动**（分栏视图批次）：每个 commit 只 `git add` 本任务列出的文件，**禁止 `git add -A`**
- 每个任务先写失败测试（RED）→ 跑失败 → 实现（GREEN）→ 跑通过 → commit
- 新增后端代码注释用英文（与 general_agent.py 新代码一致）；SSE 事件负载必须 JSON 可序列化（core.py 会 `json.dumps(event)`）
- 新代码禁止 `console.log`/`print`（日志走 `Logger`）
- i18n 键必须同时加 en.ts 与 zh.ts；type=2 workflow 相关代码分支移除但 `WorkflowExecutor` 文件保留不删
- `ChatContext.tsx` 中现存的 `[DIAG3]` 探针 effect **不要动**（属上一批次的收尾项）

---

### Task 1: SSE 新增 step/observation/agent_elapsed 事件（后端）

**Files:**
- Modify: `sources/callback/sse_callback.py`
- Modify: `api_routes/core.py:1452-1454`（SSE 透传分支）
- Test: `tests/test_sse_callback_events.py`

**Interfaces:**
- Produces（后续任务依赖）:
  - `SSECallbackHandler.on_step(round: int, thought: str, action: str, params_brief: str = "", reasoning_text: str = "") -> None` → 队列事件 `{'type':'step','round','thought','action','params_brief','reasoning_text'}`
  - `SSECallbackHandler.on_observation(round: int, result_brief: str) -> None` → `{'type':'observation','round','result_brief'}`
  - `SSECallbackHandler.on_agent_elapsed(elapsed_seconds: float, steps: int) -> None` → `{'type':'agent_elapsed','elapsed_seconds','steps'}`
  - core.py 透传事件类型集合：`{'step', 'observation', 'agent_elapsed'}`（前端 consume 契约）

- [ ] **Step 1: 写失败测试**

`tests/test_sse_callback_events.py`:

```python
"""SSE events for the ReAct loop: step / observation / agent_elapsed."""
import asyncio
import json
import unittest

from sources.callback.sse_callback import SSECallbackHandler


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestSSECallbackReActEvents(unittest.TestCase):
    def setUp(self):
        self.queue = asyncio.Queue()
        self.handler = SSECallbackHandler(self.queue)

    def test_on_step_pushes_step_event(self):
        _run(self.handler.on_step(
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
        _run(self.handler.on_observation(2, "返回 3 条"))
        event = self.queue.get_nowait()
        self.assertEqual(event, {
            "type": "observation", "round": 2, "result_brief": "返回 3 条",
        })

    def test_on_agent_elapsed_pushes_elapsed_event(self):
        _run(self.handler.on_agent_elapsed(3.2, 5))
        event = self.queue.get_nowait()
        self.assertEqual(event, {
            "type": "agent_elapsed", "elapsed_seconds": 3.2, "steps": 5,
        })

    def test_all_payloads_are_json_serializable(self):
        _run(self.handler.on_step(1, "t", "a", reasoning_text="r"))
        _run(self.handler.on_observation(1, "o"))
        _run(self.handler.on_agent_elapsed(1.0, 1))
        for _ in range(3):
            event = self.queue.get_nowait()
            json.dumps(event, ensure_ascii=False)  # must not raise


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd E:\online\workspace\copiioai\langsistance && PYTHONUTF8=1 python -m pytest tests/test_sse_callback_events.py -v`
Expected: FAIL — `AttributeError: 'SSECallbackHandler' object has no attribute 'on_step'`

- [ ] **Step 3: 实现**

在 `sse_callback.py` 的 `on_artifacts` 之后（第 68 行前）插入：

```python
    async def on_step(self, round: int, thought: str, action: str,
                      params_brief: str = "", reasoning_text: str = "") -> None:
        """One ReAct action round started — emitted before the tool executes.

        Frontend renders *thought* as a live one-line status; *reasoning_text*
        (model reasoning, may be empty) is kept for the expandable timeline.
        """
        await self.queue.put({
            'type': 'step',
            'round': round,
            'thought': thought,
            'action': action,
            'params_brief': params_brief,
            'reasoning_text': reasoning_text,
        })

    async def on_observation(self, round: int, result_brief: str) -> None:
        """One ReAct action round finished — brief result summary."""
        await self.queue.put({
            'type': 'observation',
            'round': round,
            'result_brief': result_brief,
        })

    async def on_agent_elapsed(self, elapsed_seconds: float, steps: int) -> None:
        """ReAct loop finished — frontend collapses the steps into a summary."""
        await self.queue.put({
            'type': 'agent_elapsed',
            'elapsed_seconds': elapsed_seconds,
            'steps': steps,
        })
```

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_sse_callback_events.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: core.py 透传**

`api_routes/core.py` 中 `elif event['type'] == 'long_task_intent':` 分支（约 1452 行）之前插入：

```python
                        elif event['type'] in {'step', 'observation', 'agent_elapsed'}:
                            if token_buffer:
                                combined = ''.join(token_buffer)
                                token_json = json.dumps(combined)
                                yield f"data:{token_json}\n\n"
                                token_buffer.clear()
                            yield f"data:{json.dumps(event)}\n\n"
                            current_time = asyncio.get_event_loop().time()
                            last_flush_time = current_time
                            last_stream_time = current_time

```

- [ ] **Step 6: 回归 + commit**

Run: `PYTHONUTF8=1 python -m pytest tests/test_sse_callback_events.py tests/test_patent_detail_api.py -v`（后者确认 core.py 未破坏）
Expected: 全部 PASS

```bash
git add tests/test_sse_callback_events.py sources/callback/sse_callback.py api_routes/core.py
git commit -m "feat: add step/observation/agent_elapsed SSE events for ReAct loop"
```

---

### Task 2: ReActLoop 循环核心（后端，纯逻辑可注入）

**Files:**
- Create: `sources/agents/react_loop.py`
- Test: `tests/test_react_loop.py`

**Interfaces:**
- Consumes: `SSECallbackHandler.on_step/on_observation/on_agent_elapsed`（Task 1）
- Produces（Task 3/4 依赖）:
  - `ReActLoop(llm_call, execute_action, emit=None, max_rounds=MAX_ROUNDS, lang='zh', max_reasoning_chars=800)`
  - `async ReActLoop.run(messages: list[dict], tools: list[dict]) -> RoundResult`
  - `RoundResult(kind: 'answer'|'long_task'|'fallback', answer_text, long_task_knowledge, long_task_tool_info, steps)`
  - `llm_call(messages, tools) -> (text: str, tool_calls: list[{'id','name','args'}], reasoning: str)`
  - `execute_action(name, args, round_no) -> {'kind': 'observation'|'long_task', 'text', 'knowledge', 'tool_info', 'mount_tools': list[dict]}`
  - `emit(event_type: 'step'|'observation'|'agent_elapsed'|'status'|'token', payload: dict) -> Awaitable[None]`
  - `make_llm_call(provider, handler=None) -> llm_call`；`make_event_emitter(handler) -> emit`

- [ ] **Step 1: 写失败测试**

`tests/test_react_loop.py`:

```python
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
        loop.run(messages, list(tools or []))
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
                         ["step", "observation", "agent_elapsed"])
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
        result, _ = _run(ReActLoop(model, executor, _Events(), ), model, executor)
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
        self.assertIn("Error: boom", messages[-2]["content"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_loop.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sources.agents.react_loop'`

- [ ] **Step 3: 实现**

`sources/agents/react_loop.py`（完整文件）:

```python
"""Hand-rolled ReAct loop for the general agent chat pipeline.

One round = one LLM call with the current tool list bound.  A round either
produces tool calls (executed one by one, each observed and fed back) or a
final answer (streamed to the frontend, loop ends).

The loop itself is framework-free: llm_call / execute_action / emit are
injected, so unit tests drive it with scripted fakes.  The production
adapters (make_llm_call / make_event_emitter) live in this module too.
"""
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

MAX_ROUNDS = int(os.getenv("REACT_MAX_ROUNDS", "10"))

# messages: list of {"role", "content", ("name"), ("tool_calls"), ("tool_call_id")}
# tools: list of {"name", "description", "parameters"} (bind_tools dict form)
# returns (text, tool_calls, reasoning) — tool_calls: [{"id","name","args"}]
LLMCall = Callable[[List[dict], List[dict]], Awaitable[Tuple[str, List[dict], str]]]

# returns {"kind": "observation"|"long_task", "text", "knowledge", "tool_info",
#          "mount_tools": list[dict]} — mount_tools are appended to the loop's tool list
ActionExecutor = Callable[[str, dict, int], Awaitable[dict]]

# event_type in {"step", "observation", "agent_elapsed", "status", "token"}
EventEmitter = Callable[[str, dict], Awaitable[None]]


@dataclass
class RoundResult:
    kind: str                      # 'answer' | 'long_task' | 'fallback'
    answer_text: str = ""
    long_task_knowledge: Any = None
    long_task_tool_info: Any = None
    steps: int = 0


@dataclass
class ReActLoop:
    llm_call: LLMCall
    execute_action: ActionExecutor
    emit: Optional[EventEmitter] = None
    max_rounds: int = MAX_ROUNDS
    lang: str = "zh"
    max_reasoning_chars: int = 800
    should_stop: Optional[Callable[[], bool]] = None

    async def _emit(self, event_type: str, payload: dict) -> None:
        if self.emit is not None:
            await self.emit(event_type, payload)

    def _thought_line(self, step_no: int, action_name: str) -> str:
        if self.lang == "en":
            return f"Step {step_no} · Calling '{action_name}'"
        return f"第 {step_no} 步 · 正在调用「{action_name}」"

    @staticmethod
    def _params_brief(args: dict, limit: int = 120) -> str:
        try:
            text = json.dumps(args, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(args)
        if len(text) > limit:
            text = text[:limit] + "..."
        return text

    async def run(self, messages: List[dict], tools: List[dict]) -> RoundResult:
        """Run the loop. *messages* grows assistant/tool turns in place;
        tools mounted by search_my_knowledge are appended to *tools*."""
        start = time.monotonic()
        steps = 0
        last_failed_action: Optional[str] = None
        consecutive_failures = 0

        for _round in range(1, self.max_rounds + 1):
            if self.should_stop is not None and self.should_stop():
                return await self._finish("fallback", messages, steps, start)

            text, tool_calls, reasoning = await self.llm_call(messages, tools)

            if not tool_calls:
                return await self._finish("answer", messages, steps, start,
                                          answer_text=text)

            messages.append({
                "role": "assistant",
                "content": text or "",
                "tool_calls": tool_calls,
            })

            for call in tool_calls:
                steps += 1
                action_name = call.get("name", "")
                args = call.get("args") or {}
                await self._emit("step", {
                    "round": steps,
                    "thought": self._thought_line(steps, action_name),
                    "action": action_name,
                    "params_brief": self._params_brief(args),
                    "reasoning_text": (reasoning or "")[:self.max_reasoning_chars],
                })

                try:
                    result = await self.execute_action(action_name, args, steps)
                except Exception as exc:
                    # Executor failure becomes an observation — the LLM may
                    # retry with another tool instead of the loop crashing.
                    result = {"kind": "observation", "text": f"Error: {exc}"}

                if result.get("kind") == "long_task":
                    return await self._finish(
                        "long_task", messages, steps, start,
                        long_task_knowledge=result.get("knowledge"),
                        long_task_tool_info=result.get("tool_info"),
                    )

                for tool_dict in result.get("mount_tools") or []:
                    tools.append(tool_dict)

                obs = str(result.get("text") or "")
                await self._emit("observation", {"round": steps, "result_brief": obs})

                is_error = obs.startswith("Error:")
                if is_error and action_name == last_failed_action:
                    consecutive_failures += 1
                elif is_error:
                    consecutive_failures = 1
                else:
                    consecutive_failures = 0
                last_failed_action = action_name if is_error else None
                if consecutive_failures >= 2:
                    return await self._finish("fallback", messages, steps, start)

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id") or f"call_{steps}",
                    "name": action_name,
                    "content": obs,
                })

        return await self._finish("fallback", messages, steps, start)

    async def _finish(self, kind: str, messages: List[dict], steps: int,
                      start: float, **fields: Any) -> RoundResult:
        if kind == "fallback":
            messages.append({
                "role": "user",
                "content": (
                    "已达到最大执行步数。请基于以上观察结果，用简洁的语言"
                    "向用户说明当前进展和未能完成的部分。"
                    if self.lang == "zh" else
                    "You have reached the maximum number of steps. Based on the "
                    "observations above, briefly tell the user what was achieved "
                    "and what could not be completed."
                ),
            })
            text, _, _ = await self.llm_call(messages, [])
            elapsed = round(time.monotonic() - start, 1)
            await self._emit("agent_elapsed", {"elapsed_seconds": elapsed, "steps": steps})
            return RoundResult(kind="fallback", answer_text=text, steps=steps)
        elapsed = round(time.monotonic() - start, 1)
        await self._emit("agent_elapsed", {"elapsed_seconds": elapsed, "steps": steps})
        return RoundResult(kind=kind, steps=steps, **fields)


def make_llm_call(provider, handler=None) -> LLMCall:
    """Production llm_call adapter.

    Always streams (MiniMax returns empty content on non-streaming calls).
    Content tokens are forwarded to *handler* live — rounds that produce
    tool calls carry no content tokens on OpenAI-compatible APIs, so only
    the final answer round ever streams.  Reasoning text (DeepSeek/MiniMax
    think blocks) is captured for the step timeline.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    def _to_message(msg: dict):
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "system":
            return SystemMessage(content=content)
        if role == "assistant":
            tool_calls = [
                {"id": c.get("id"), "name": c.get("name"), "args": c.get("args") or {}}
                for c in (msg.get("tool_calls") or [])
            ] or None
            return AIMessage(content=content, tool_calls=tool_calls)
        if role == "tool":
            return ToolMessage(content=content, tool_call_id=msg.get("tool_call_id", ""))
        return HumanMessage(content=content)

    async def llm_call(messages: List[dict], tools: List[dict]):
        llm = provider._get_langchain_llm(streaming=True)
        if tools:
            llm = llm.bind_tools(tools)
        lc_messages = [_to_message(m) for m in messages]

        text_parts: List[str] = []
        reasoning_parts: List[str] = []
        calls: Dict[int, dict] = {}
        async for chunk in llm.astream(lc_messages):
            if getattr(chunk, "content", None):
                if handler is not None:
                    await handler.on_llm_new_token(chunk.content)
                text_parts.append(chunk.content)
            reasoning = None
            if hasattr(chunk, "reasoning_content"):
                reasoning = chunk.reasoning_content
            if not reasoning:
                reasoning = (chunk.additional_kwargs or {}).get("reasoning_content")
            if reasoning:
                reasoning_parts.append(reasoning)
            for part in getattr(chunk, "tool_call_chunks", None) or []:
                idx = part.get("index", 0)
                entry = calls.setdefault(idx, {"id": "", "name": "", "args": ""})
                if part.get("id"):
                    entry["id"] = part["id"]
                if part.get("name"):
                    entry["name"] += part["name"]
                entry["args"] += part.get("args") or ""

        tool_calls: List[dict] = []
        for idx in sorted(calls):
            entry = calls[idx]
            try:
                args = json.loads(entry["args"]) if entry["args"].strip() else {}
            except (json.JSONDecodeError, ValueError):
                args = {}
            tool_calls.append({
                "id": entry["id"] or f"call_{idx}",
                "name": entry["name"],
                "args": args,
            })
        return "".join(text_parts), tool_calls, "".join(reasoning_parts)

    return llm_call


def make_event_emitter(handler) -> EventEmitter:
    """Map ReActLoop event names onto the SSE callback handler methods."""

    async def emit(event_type: str, payload: dict) -> None:
        if handler is None:
            return
        try:
            if event_type == "step":
                await handler.on_step(**payload)
            elif event_type == "observation":
                await handler.on_observation(**payload)
            elif event_type == "agent_elapsed":
                await handler.on_agent_elapsed(**payload)
            elif event_type == "status":
                await handler.on_status(payload.get("message", ""))
            elif event_type == "token":
                await handler.on_llm_new_token(payload.get("content", ""))
        except Exception:
            pass  # event delivery must never break the loop

    return emit
```

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_loop.py -v`
Expected: PASS（9 passed）

- [ ] **Step 5: commit**

```bash
git add sources/agents/react_loop.py tests/test_react_loop.py
git commit -m "feat: add hand-rolled ReAct loop core with injectable llm/action/emit"
```

---

### Task 3: 工具供给与行动分派（后端）

**Files:**
- Create: `sources/agents/react_tools.py`
- Test: `tests/test_react_tools.py`

**Interfaces:**
- Consumes: `ReActLoop` 契约（Task 2）；`get_knowledge_tool_candidates`（knowledge.py:418）；`GeneralAgent.get_dynamic_tool_for(knowledge_item, tool_info)`（Task 4 提供，本任务测试用 fake agent）
- Produces（Task 4 依赖）:
  - `SEARCH_KNOWLEDGE_TOOL_NAME = "search_my_knowledge"`
  - `async build_tool_set(agent, user_id, question, push_filter=None) -> (registry: dict[str, ToolEntry], tools: list[dict])`
  - `async make_action_executor(agent, registry, push_filter=None) -> execute_action`
  - `ToolEntry(name, kind: 'search'|'knowledge'|'long_task', knowledge, tool_info, tool)`
  - `_cap_patent_list(tool_info, items, lang) -> (items, note)`；`_summarize_observation(result, lang, limit=300) -> str`
  - 规则：type=2 全部跳过；type=3 → long_task 工具；type=1 预注册 top-5；搜索类专利列表截断 100（`uspto_documents` 类不截断）

- [ ] **Step 1: 写失败测试**

`tests/test_react_tools.py`:

```python
"""Tests for ReAct tool supply and action dispatch (react_tools)."""
import asyncio
import unittest
from unittest.mock import patch

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from sources.agents.react_tools import (
    MAX_PATENT_LIST_ITEMS,
    SEARCH_KNOWLEDGE_TOOL_NAME,
    _cap_patent_list,
    _summarize_observation,
    build_tool_set,
    make_action_executor,
)


class _Knowledge:
    def __init__(self, kid, question="q", ktype=1):
        self.id = kid
        self.question = question
        self.description = ""
        self.answer = ""
        self.public = False
        self.model_name = ""
        self.tool_id = 1
        self.params = ""
        self.type = ktype
        self.scene_id = None
        self.update_time = ""


class _ToolInfo:
    def __init__(self, title, url="https://api.example.com/search", push=2):
        self.title = title
        self.description = f"{title} description"
        self.url = url
        self.push = push
        self.params = '{"method":"GET","query":{}}'


class _FakeAgent:
    def __init__(self):
        self._lang = "zh"
        self._last_user_id = "u1"
        self._pending_raw_items = None
        self.knowledgeTool = None
        self.tools_made = []

    async def get_dynamic_tool_for(self, knowledge_item, tool_info):
        class _Args(BaseModel):
            params: str = Field(description="params")

        async def _noop(**kwargs):
            return "tool result"

        name = "".join(c for c in tool_info.title if c.isalnum() or c in "_-") or "tool"
        tool = StructuredTool.from_function(_noop, name=name,
                                            description=tool_info.description,
                                            args_schema=_Args)
        self.tools_made.append((knowledge_item.id, name))
        return tool


class TestCapPatentList(unittest.TestCase):
    def test_search_list_capped_at_100(self):
        items = list(range(MAX_PATENT_LIST_ITEMS + 40))
        capped, note = _cap_patent_list(_ToolInfo("s"), items, "zh")
        self.assertEqual(len(capped), MAX_PATENT_LIST_ITEMS)
        self.assertIn("已截断", note)

    def test_search_list_under_100_untouched(self):
        items = list(range(7))
        capped, note = _cap_patent_list(_ToolInfo("s"), items, "zh")
        self.assertEqual(capped, items)
        self.assertNotIn("截断", note)

    def test_document_list_never_capped(self):
        items = list(range(MAX_PATENT_LIST_ITEMS + 40))
        capped, note = _cap_patent_list(
            _ToolInfo("d", url="https://api.example.com/documents"), items, "zh")
        self.assertEqual(len(capped), MAX_PATENT_LIST_ITEMS + 40)
        self.assertIn("不截断", note)


class TestSummarizeObservation(unittest.TestCase):
    def test_long_text_truncated(self):
        text = _summarize_observation("x" * 1000, "zh")
        self.assertLessEqual(len(text), 303)
        self.assertTrue(text.endswith("..."))

    def test_dict_serialized(self):
        text = _summarize_observation({"a": 1, "b": [1, 2]}, "zh")
        self.assertIn('"a": 1', text)

    def test_none_becomes_empty(self):
        self.assertEqual(_summarize_observation(None, "zh"), "")


class TestBuildToolSet(unittest.TestCase):
    @patch("sources.agents.react_tools.get_knowledge_tool_candidates")
    def test_workflow_type_2_skipped_and_top_n_respected(self, mock_candidates):
        mock_candidates.return_value = [
            (_Knowledge(1, ktype=3), None),                        # long task
            (_Knowledge(2, ktype=2), _ToolInfo("wf")),             # workflow — retired
            (_Knowledge(3, ktype=1), _ToolInfo("uspto search")),   # normal
            (_Knowledge(4, ktype=1), _ToolInfo("cnipa search")),   # normal
        ]
        agent = _FakeAgent()
        registry, tools = asyncio.run(
            build_tool_set(agent, "u1", "专利检索", push_filter=None))

        self.assertIn(SEARCH_KNOWLEDGE_TOOL_NAME, registry)
        kinds = {e.name: e.kind for e in registry.values()}
        self.assertIn("uspto_search", kinds)
        self.assertEqual(kinds["uspto_search"], "knowledge")
        self.assertNotIn("wf", kinds)          # type-2 retired
        # type-3 became a long_task tool
        lt_names = [n for n, k in kinds.items() if k == "long_task"]
        self.assertEqual(len(lt_names), 1)
        self.assertEqual(len(tools), len(registry))


class TestExecuteAction(unittest.TestCase):
    @patch("sources.agents.react_tools.get_knowledge_tool_candidates")
    def test_search_knowledge_returns_matches_and_mount_tools(self, mock_candidates):
        mock_candidates.return_value = [
            (_Knowledge(1, ktype=3), None),
            (_Knowledge(2, ktype=2), _ToolInfo("wf")),
            (_Knowledge(3, ktype=1), _ToolInfo("uspto search")),
        ]
        agent = _FakeAgent()
        registry, tools = asyncio.run(
            build_tool_set(agent, "u1", "专利检索", push_filter=None))
        executor = asyncio.run(make_action_executor(agent, registry, None))

        result = asyncio.run(
            executor(SEARCH_KNOWLEDGE_TOOL_NAME, {"query": "美国专利"}, 1))
        self.assertEqual(result["kind"], "observation")
        self.assertIn("2 个匹配", result["text"])
        mounted = [t["name"] for t in result["mount_tools"]]
        self.assertIn("uspto_search", mounted)
        # type-2 not in matches
        self.assertNotIn("wf", result["text"])

    def test_knowledge_action_sets_pending_and_caps(self):
        agent = _FakeAgent()
        entry_k = _Knowledge(3, ktype=1)
        agent.get_dynamic_tool_for = _make_tool_with_pending(agent)
        registry, tools = asyncio.run(
            _registry_with_one_knowledge(agent, entry_k))
        executor = asyncio.run(make_action_executor(agent, registry, None))
        agent._pending_raw_items = list(range(MAX_PATENT_LIST_ITEMS + 10))
        result = asyncio.run(executor("uspto_search", {"params": "{}"}, 1))
        self.assertEqual(len(agent._pending_raw_items), MAX_PATENT_LIST_ITEMS)
        self.assertIn("已截断", result["text"])
        self.assertEqual(agent.knowledgeTool[0], entry_k)

    def test_long_task_action_returns_intent(self):
        agent = _FakeAgent()
        k = _Knowledge(9, ktype=3)
        registry = {
            "批量专利分析": type("E", (), {
                "name": "批量专利分析", "kind": "long_task",
                "knowledge": k, "tool_info": None, "tool": None,
            })(),
        }
        executor = asyncio.run(make_action_executor(agent, registry, None))
        result = asyncio.run(executor("批量专利分析", {"query": "分析"}, 1))
        self.assertEqual(result["kind"], "long_task")
        self.assertEqual(result["knowledge"], k)

    def test_unknown_tool_returns_error_observation(self):
        agent = _FakeAgent()
        executor = asyncio.run(make_action_executor(agent, {}, None))
        result = asyncio.run(executor("nope", {}, 1))
        self.assertEqual(result["kind"], "observation")
        self.assertTrue(result["text"].startswith("Error:"))


def _make_tool_with_pending(agent):
    async def get_dynamic_tool_for(knowledge_item, tool_info):
        class _Args(BaseModel):
            params: str = Field(description="params")

        def _noop(**kwargs):
            return "The query returned N items."

        return StructuredTool.from_function(
            _noop, name="uspto_search", description="d", args_schema=_Args)
    return get_dynamic_tool_for


async def _registry_with_one_knowledge(agent, knowledge):
    tool_info = _ToolInfo("uspto search")
    from sources.agents.react_tools import ToolEntry
    dynamic_tool = await agent.get_dynamic_tool_for(knowledge, tool_info)
    entry = ToolEntry(name=dynamic_tool.name, kind="knowledge",
                      knowledge=knowledge, tool_info=tool_info, tool=dynamic_tool)
    return {entry.name: entry}, []


if __name__ == "__main__":
    unittest.main()
```

注意：`test_workflow_type_2_skipped...` 中 `tools` 与 `registry` 长度相等的断言依赖 `build_tool_set` 对每个 entry 只 add 一次。

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sources.agents.react_tools'`

- [ ] **Step 3: 实现**

`sources/agents/react_tools.py`（完整文件）:

```python
"""Tool supply and action dispatch for the ReAct loop.

build_tool_set() turns the user's knowledge base into the loop's initial
tool list: the search_my_knowledge meta-tool, the top-N vector-recalled
type-1 knowledge tools, and one long-task tool per type-3 knowledge item.
make_action_executor() dispatches a tool call to the right handler and
returns the observation the loop feeds back to the LLM.

Type-2 (workflow) knowledge is retired — the loop composes type-1 tools
itself, so workflow items are never offered.
"""
import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from sources.knowledge.knowledge import get_knowledge_tool_candidates

TOP_N = int(os.getenv("REACT_TOOL_TOP_N", "5"))
MAX_PATENT_LIST_ITEMS = int(os.getenv("REACT_MAX_PATENT_LIST_ITEMS", "100"))
SEARCH_KNOWLEDGE_TOOL_NAME = "search_my_knowledge"
MAX_SEARCH_RESULTS = 5
MAX_OBSERVATION_CHARS = 300


class _QueryArgs(BaseModel):
    query: str = Field(description="Natural-language description of what you need")


@dataclass
class ToolEntry:
    name: str
    kind: str                    # 'search' | 'knowledge' | 'long_task'
    knowledge: Any               # KnowledgeItem or None
    tool_info: Any               # ToolItem or None
    tool: Optional[StructuredTool]


def _clean_tool_name(knowledge) -> str:
    """Sanitise a knowledge title into a tool name (same rules as before)."""
    title = (getattr(knowledge, "question", "") or "").strip() or "dynamic_knowledge_tool"
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", title)
    return cleaned or "dynamic_knowledge_tool"


def _long_task_description(knowledge) -> str:
    question = (getattr(knowledge, "question", "") or "").strip()
    desc = (getattr(knowledge, "description", "") or "").strip()
    return (
        f"Start a background batch-analysis task (long task). {question}. "
        f"{desc} After calling, the task runs asynchronously and the user "
        f"is notified — do not wait for results."
    )[:800]


async def _search_knowledge_stub(query: str) -> str:
    raise NotImplementedError("executed via dispatch, not directly")


async def _long_task_stub(query: str) -> str:
    raise NotImplementedError("executed via dispatch, not directly")


def _tool_to_bind_dict(tool: StructuredTool) -> dict:
    """bind_tools-compatible dict for a StructuredTool."""
    return {
        "name": tool.name,
        "description": tool.description or "",
        "parameters": tool.args_schema.model_json_schema(),
    }


async def build_tool_set(
    agent,
    user_id: str,
    question: str,
    push_filter: Optional[int] = None,
) -> Tuple[Dict[str, ToolEntry], List[dict]]:
    """Build (registry, tools) for one query.

    search_my_knowledge is always first; type-3 items become long-task
    tools; vector-recalled type-1 items (top TOP_N) become knowledge tools.
    """
    registry: Dict[str, ToolEntry] = {}
    tools: List[dict] = []

    def add(entry: ToolEntry) -> None:
        if entry.name in registry:
            return  # duplicate title — first registration wins
        registry[entry.name] = entry
        tools.append(_tool_to_bind_dict(entry.tool))

    search_tool = StructuredTool.from_function(
        func=_search_knowledge_stub,
        name=SEARCH_KNOWLEDGE_TOOL_NAME,
        description=(
            "Search your available knowledge base for knowledge or tools "
            "matching a natural-language description. Use this when none of "
            "the tools you already have fits the user's request. Returns "
            "matching knowledge items; their tools become available to you "
            "immediately afterwards."
        ),
        args_schema=_QueryArgs,
    )
    add(ToolEntry(name=SEARCH_KNOWLEDGE_TOOL_NAME, kind="search",
                  knowledge=None, tool_info=None, tool=search_tool))

    candidates = await asyncio.to_thread(
        get_knowledge_tool_candidates, user_id, question, TOP_N, 0, push_filter,
    )
    seen_knowledge_ids = set()
    normal_count = 0
    for knowledge, tool_info in candidates:
        k_type = int(getattr(knowledge, "type", 1) or 1)
        if k_type == 2:
            continue  # workflow knowledge retired
        knowledge_id = getattr(knowledge, "id", None)
        if knowledge_id is not None:
            if knowledge_id in seen_knowledge_ids:
                continue
            seen_knowledge_ids.add(knowledge_id)

        title = _clean_tool_name(knowledge)
        if k_type == 3:
            tool = StructuredTool.from_function(
                func=_long_task_stub,
                name=title,
                description=_long_task_description(knowledge),
                args_schema=_QueryArgs,
            )
            add(ToolEntry(name=title, kind="long_task",
                          knowledge=knowledge, tool_info=tool_info, tool=tool))
            continue

        if tool_info is None:
            continue
        if normal_count >= TOP_N:
            continue
        normal_count += 1
        dynamic_tool = await agent.get_dynamic_tool_for(knowledge, tool_info)
        if dynamic_tool is None:
            continue
        add(ToolEntry(name=dynamic_tool.name, kind="knowledge",
                      knowledge=knowledge, tool_info=tool_info, tool=dynamic_tool))
    return registry, tools


def _cap_patent_list(tool_info, items: list, lang: str) -> Tuple[list, str]:
    """Cap search-style patent lists at MAX_PATENT_LIST_ITEMS.

    Document-list tools (uspto_documents, URL contains 'documents') are
    uncapped — the user wants every document of a single patent.
    """
    url = (getattr(tool_info, "url", "") or "").lower()
    if "documents" in url:
        note = "document list (uncapped)" if lang == "en" else "文档列表不截断"
        return items, note
    if len(items) > MAX_PATENT_LIST_ITEMS:
        if lang == "en":
            note = (f"truncated — {len(items)} total, showing first "
                    f"{MAX_PATENT_LIST_ITEMS}")
        else:
            note = (f"已截断，共 {len(items)} 条，展示前 "
                    f"{MAX_PATENT_LIST_ITEMS} 条")
        return items[:MAX_PATENT_LIST_ITEMS], note
    note = f"共 {len(items)} 条" if lang != "en" else f"{len(items)} items total"
    return items, note


def _summarize_observation(result, lang: str, limit: int = MAX_OBSERVATION_CHARS) -> str:
    """Turn a tool result into a bounded observation string for the LLM."""
    if result is None:
        return ""
    text = result
    if isinstance(result, (dict, list)):
        try:
            text = json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(result)
    text = str(text)
    if len(text) > limit:
        text = text[:limit] + "..."
    return text


async def _run_search_knowledge(agent, registry, user_id, args, push_filter) -> dict:
    """Execute search_my_knowledge: recall candidates and mount their tools."""
    lang = getattr(agent, "_lang", "zh")
    query = str((args or {}).get("query", "") or "").strip() or str(args or "")
    candidates = await asyncio.to_thread(
        get_knowledge_tool_candidates,
        user_id, query, MAX_SEARCH_RESULTS, 0, push_filter,
    )

    matches: List[str] = []
    mount_tools: List[dict] = []
    for knowledge, tool_info in candidates:
        k_type = int(getattr(knowledge, "type", 1) or 1)
        if k_type == 2:
            continue
        kind = "long_task" if k_type == 3 else "knowledge"
        if kind == "knowledge" and tool_info is None:
            continue
        if kind == "long_task":
            title = _clean_tool_name(knowledge)
        else:
            dynamic_tool = await agent.get_dynamic_tool_for(knowledge, tool_info)
            if dynamic_tool is None:
                continue
            title = dynamic_tool.name
        matches.append(
            f"- [{kind}] id={knowledge.id} {knowledge.question or ''}（tool: {title}）"
        )
        if title in registry:
            continue  # already available to the loop
        if kind == "long_task":
            entry_tool = StructuredTool.from_function(
                func=_long_task_stub, name=title,
                description=_long_task_description(knowledge),
                args_schema=_QueryArgs,
            )
            entry_tool_info = None
        else:
            entry_tool = dynamic_tool
            entry_tool_info = tool_info
        entry = ToolEntry(name=title, kind=kind, knowledge=knowledge,
                          tool_info=entry_tool_info, tool=entry_tool)
        registry[title] = entry
        mount_tools.append(_tool_to_bind_dict(entry_tool))

    if not matches:
        if lang == "en":
            text = ("No matching knowledge found. Answer directly and suggest "
                    "the user check the community for shared knowledge.")
        else:
            text = "没有找到匹配的知识。请直接回答用户，并建议用户到社区查找共享知识。"
        return {"kind": "observation", "text": text, "mount_tools": []}

    if lang == "en":
        text = f"Found {len(matches)} matching knowledge item(s):\n" + "\n".join(matches)
    else:
        text = f"找到 {len(matches)} 个匹配的知识：\n" + "\n".join(matches)
    return {"kind": "observation", "text": text, "mount_tools": mount_tools}


async def make_action_executor(agent, registry, push_filter=None):
    """Return the loop's execute_action closure."""
    user_id = getattr(agent, "_last_user_id", None)
    lang = getattr(agent, "_lang", "zh")

    async def execute_action(name: str, args: dict, round_no: int) -> dict:
        entry = registry.get(name)
        if entry is None:
            return {"kind": "observation", "text": f"Error: unknown tool '{name}'"}

        if entry.kind == "search":
            return await _run_search_knowledge(agent, registry, user_id, args, push_filter)

        if entry.kind == "long_task":
            # The loop terminates and core.py's existing long-task branch
            # handles classification + Celery submission.
            return {"kind": "long_task", "text": "",
                    "knowledge": entry.knowledge, "tool_info": entry.tool_info}

        try:
            result = await asyncio.to_thread(entry.tool.invoke, args)
        except Exception as exc:
            return {"kind": "observation", "text": f"Error: {exc}"}

        # Keep the exact pairing used later by _stream_raw_items for
        # source inference and artifact building.
        agent.knowledgeTool = (entry.knowledge, entry.tool_info)

        pending = getattr(agent, "_pending_raw_items", None)
        if pending:
            capped, note = _cap_patent_list(entry.tool_info, pending, lang)
            agent._pending_raw_items = capped
            if lang == "en":
                text = f"Tool returned {len(capped)} record(s) ({note}); the full list is displayed afterwards."
            else:
                text = f"工具返回 {len(capped)} 条记录（{note}），完整列表稍后展示。"
            return {"kind": "observation", "text": text}

        return {"kind": "observation", "text": _summarize_observation(result, lang)}

    return execute_action
```

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_tools.py -v`
Expected: PASS（全部；`build_tool_set` 测试通过 mock 不触数据库）

- [ ] **Step 5: commit**

```bash
git add sources/agents/react_tools.py tests/test_react_tools.py
git commit -m "feat: add ReAct tool supply (top-N recall + search meta-tool + long-task tools)"
```

---

### Task 4: GeneralAgent 集成（后端）

**Files:**
- Modify: `sources/agents/general_agent.py`
- Test: `tests/test_react_integration_general_agent.py`

**Interfaces:**
- Consumes: Task 2 `ReActLoop/make_llm_call/make_event_emitter`、Task 3 `build_tool_set/make_action_executor`、Task 1 SSE 方法
- Produces:
  - `GeneralAgent.get_dynamic_tool_for(knowledge_item, tool_info) -> Optional[StructuredTool]`（从 get_dynamic_tools 提取）
  - `create_agent(...)` 新契约：运行完整循环；长任务 → 返回 `{'intent':'long_task','knowledge','tool_info'}`；其余 → 返回 None 并置 `self._react_loop_ran = True`
  - `invoke_agent(...)`：仅收尾（`_stream_raw_items` + `_store_current_turn`）；workflow 分支移除
  - core.py **无需改动**（intent dict 与 None 的既有处理保持不变）

- [ ] **Step 1: 写失败测试**

`tests/test_react_integration_general_agent.py`:

```python
"""Wiring tests: GeneralAgent.create_agent → ReActLoop → intent/None."""
import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from sources.agents.general_agent import GeneralAgent
from sources.agents.react_loop import RoundResult


class _FakeProvider:
    def get_model_name(self):
        return "fake"

    def _get_langchain_llm(self, streaming=True):
        return None


class _FakeHandler:
    """Minimal callback stand-in with the methods the agent touches."""

    def __init__(self):
        self.queue = None
        self.statuses = []
        self.tokens = []

    async def on_status(self, message, **kwargs):
        self.statuses.append(message)

    async def on_llm_new_token(self, token, **kwargs):
        self.tokens.append(token)


def _make_agent():
    with patch.object(GeneralAgent, "load_prompt", return_value="sys prompt"):
        agent = GeneralAgent("test", "prompts/base/general_agent.txt",
                             _FakeProvider(), verbose=False)
    agent.enabled = True
    agent.llm = MagicMock()
    agent.llm.get_model_name.return_value = "fake"
    return agent


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestCreateAgentWiring(unittest.TestCase):
    def test_answer_kind_returns_none_and_sets_flag(self):
        agent = _make_agent()
        handler = _FakeHandler()
        with patch("sources.agents.general_agent.build_tool_set",
                   new=AsyncMock(return_value=({}, []))), \
             patch("sources.agents.general_agent.ReActLoop") as MockLoop:
            MockLoop.return_value.run = AsyncMock(
                return_value=RoundResult(kind="answer", answer_text="hi", steps=1))
            result = _run(agent.create_agent(
                "u1", "hello", "q1", "", handler, push_filter=None))
        self.assertIsNone(result)
        self.assertTrue(getattr(agent, "_react_loop_ran", False))
        # per-request state reset (pool reuse safety)
        self.assertIsNone(getattr(agent, "_pending_raw_items", "unset"))
        self.assertTrue(handler.statuses)  # "正在分析您的问题..."

    def test_long_task_kind_returns_intent_dict(self):
        agent = _make_agent()
        handler = _FakeHandler()
        class _K:
            id = 7
            type = 3
        class _T:
            title = "lt"
        with patch("sources.agents.general_agent.build_tool_set",
                   new=AsyncMock(return_value=({}, []))), \
             patch("sources.agents.general_agent.ReActLoop") as MockLoop:
            MockLoop.return_value.run = AsyncMock(
                return_value=RoundResult(kind="long_task", steps=1,
                                         long_task_knowledge=_K(),
                                         long_task_tool_info=_T()))
            result = _run(agent.create_agent(
                "u1", "帮我分析这批专利", "q1", "", handler, push_filter=None))
        self.assertEqual(result["intent"], "long_task")
        self.assertEqual(result["knowledge"].id, 7)
        self.assertTrue(getattr(agent, "_react_loop_ran", False))

    def test_invoke_agent_streams_pending_raw_items_after_loop(self):
        agent = _make_agent()
        agent._react_loop_ran = True
        agent._pending_raw_items = [{"a": 1}]
        agent._active_collector = None
        handler = _FakeHandler()
        with patch.object(agent, "_stream_raw_items",
                          new=AsyncMock()) as mock_stream, \
             patch.object(agent, "_store_current_turn"):
            _run(agent.invoke_agent(None, handler))
        mock_stream.assert_awaited_once_with([{"a": 1}], handler)

    def test_invoke_agent_stores_turn_when_no_pending(self):
        agent = _make_agent()
        agent._react_loop_ran = True
        agent._pending_raw_items = None
        agent._active_collector = MagicMock(collected_text="answer text")
        with patch.object(agent, "_stream_raw_items",
                          new=AsyncMock()) as mock_stream, \
             patch.object(agent, "_store_current_turn") as mock_store:
            _run(agent.invoke_agent(None, MagicMock()))
        mock_stream.assert_not_called()
        mock_store.assert_called_once_with("answer text")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_integration_general_agent.py -v`
Expected: FAIL（`build_tool_set`/`ReActLoop` 未在 general_agent 导入）

- [ ] **Step 3: 提取 `get_dynamic_tool_for`**

在 `general_agent.py` 的 `get_dynamic_tools` 方法前（约 1227 行）插入新方法，并把 `get_dynamic_tools` 内部的 dynamic tool 构建替换为调用它：

```python
    def get_dynamic_tool_for(self, knowledge_item, tool_info):
        """Build the LangChain StructuredTool for one knowledge/tool pair.

        Extracted from get_dynamic_tools so the ReAct loop can build tools
        for arbitrary knowledge items (top-N recall + search results),
        not just the single self.knowledgeTool pairing.
        """
        if tool_info is None:
            return None
        # ── closures moved verbatim from get_dynamic_tools ──
        def dynamic_frontend_tool_function(user_id: str, query_id: str, params: str):
            # Always use stored values — ignore LLM-provided IDs
            user_id = self._last_user_id
            query_id = self._last_query_id
            self.logger.info(f"dynamic_frontend_tool_function user id is {user_id} - query id is {query_id} - param is {params}")
            try:
                redis_conn = get_redis_connection()
                redis_key = f"tool_request_{query_id}_{user_id}"
                params_json = json.dumps(params)
                redis_conn.set(redis_key, params_json, ex=1200)
                response_key = f"tool_response_{query_id}_{user_id}"
                timeout = 300  # 5 minutes
                interval = 1
                elapsed = 0
                while elapsed < timeout:
                    response_value = redis_conn.get(response_key)
                    if response_value is not None:
                        return response_value
                    time.sleep(interval)
                    elapsed += interval
                return None
            except Exception as e:
                self.logger.error(f"Failed to write to Redis: {str(e)}")
                return None

        def dynamic_backend_tool_function(user_id: str, query_id: str, params: Dict[str, Any] | str):
            # Always use stored values — ignore LLM-provided IDs
            user_id = self._last_user_id
            query_id = self._last_query_id
            self.logger.info(
                f"dynamic_backend_tool_function user_id={user_id} query_id={query_id}"
            )
            tool_result = execute_backend_tool_request(tool_info, params)
            raw_items = tool_result.get("raw_items")
            if raw_items:
                list_count = len(raw_items)
                self._pending_raw_items = raw_items
                return (
                    f"The query returned {list_count} items. "
                    f"Please write a brief 2-3 sentence summary of what was found. "
                    f"The complete list will be analyzed and displayed item by item automatically - "
                    f"do NOT enumerate the items yourself."
                )
            data = tool_result.get("data")
            if isinstance(data, dict):
                pruned = _prune_item_for_llm(data)
                self._pending_raw_items = [pruned]
                return (
                    "The query returned structured data. "
                    "Please write a brief 1-2 sentence summary of what was found. "
                    "The complete data will be displayed automatically - "
                    "do NOT enumerate the fields yourself."
                )
            if isinstance(data, list):
                return json.dumps(data, ensure_ascii=False, indent=2)
            return data

        # ── name / schema / push selection (same as before) ──
        tool_name = tool_info.title if tool_info.title else "dynamic_knowledge_tool"
        cleaned_tool_name = re.sub(r'[^a-zA-Z0-9_-]', '_', tool_name)
        if not cleaned_tool_name or cleaned_tool_name.strip() == "":
            cleaned_tool_name = "dynamic_knowledge_tool"

        if tool_info.push == 1 or tool_info.push == 3:
            tool_func = dynamic_frontend_tool_function
        elif tool_info.push == 2:
            tool_func = dynamic_backend_tool_function
        else:
            tool_func = dynamic_frontend_tool_function

        args_schema = (
            DynamicBackendToolFunction
            if tool_info.push == 2
            else DynamicToolFunction
        )
        return StructuredTool.from_function(
            func=tool_func,
            name=cleaned_tool_name,
            description=tool_info.description if tool_info.description else "Dynamic knowledge tool",
            args_schema=args_schema,
        )
```

再把 `get_dynamic_tools`（约 1467 行）的主体改为：

```python
    async def get_dynamic_tools(self) -> list:
        """Backwards-compatible wrapper: dynamic tools for self.knowledgeTool."""
        try:
            if not hasattr(self, 'knowledgeTool') or not self.knowledgeTool:
                return None
            knowledge_item, tool_info = self.knowledgeTool
            if not tool_info:
                return None
            dynamic_tool = self.get_dynamic_tool_for(knowledge_item, tool_info)
            if dynamic_tool is None:
                return None
            self.logger.info(f"tools{[dynamic_tool]}")
            return [dynamic_tool]
        except Exception as e:
            raise Exception(f"get_tool failed: {str(e)}") from e
```

（原有 `get_dynamic_tools` 里被删掉的死代码段 `url = tool_info.url ...` 位于原 `dynamic_backend_tool_function` return 之后，从未执行，一并丢弃。）

- [ ] **Step 4: 改造 create_agent / invoke_agent + 移除 workflow 分支**

顶部 import 调整：

```python
# 移除:
from sources.workflow.workflow_executor import WorkflowExecutor, is_workflow_knowledge
# 新增:
from sources.agents.react_loop import ReActLoop, make_event_emitter, make_llm_call
from sources.agents.react_tools import build_tool_set, make_action_executor
```

`create_agent`（原 1578-1676 行）整体替换为：

```python
    async def create_agent(self, user_id, prompt, query_id, tool_data, callback_handler, push_filter=None):
        """Build the ReAct tool set and run the loop for one user query.

        Long-task tool calls surface as the same {'intent': 'long_task'}
        marker core.py already handles; every other outcome streams through
        the callback handler inside the loop and returns None.
        """
        # ── per-request state reset (agents are pooled and reused) ──
        self._last_user_prompt = prompt
        self._last_query_id = query_id
        self._last_user_id = user_id
        self._pending_raw_items = None
        self._workflow_result = None
        self._react_loop_ran = False
        self.knowledgeTool = {}
        self.tools = []
        lang = self._detect_lang(prompt)
        self._lang = lang
        if callback_handler:
            await _emit_status(callback_handler,
                "正在分析您的问题..." if lang == 'zh' else "Analyzing your question...")

        # Wrap the handler so the final answer text is collected for
        # multi-turn storage (_store_current_turn).
        wrapped = _ResponseCollector(callback_handler)
        self._active_collector = wrapped

        user_prompt = self.generate_user_prompt(prompt, user_id, query_id)

        # ── Multi-turn memory (same mechanics as before) ──
        conversation_block = ""
        for turn in getattr(self, '_conversation_turns', []):
            clean_user = _STRIP_IDS_RE.sub('', turn['user']).strip()
            conversation_block += f"\n\n## Previous conversation\n\nUser: {clean_user}\n\nAssistant: {turn['assistant']}"

        system_prompt = (
            self._get_fixed_system_prefix()
            + conversation_block
            + self._loop_system_guidance()
        )
        self.memory.reset([
            {'role': 'user', 'content': user_prompt},
            {'role': 'system', 'content': system_prompt},
        ])
        self.logger.info(f"memory.get():{self.memory.get()}")

        registry, bind_tools = await build_tool_set(self, user_id, prompt, push_filter)
        self._react_registry = registry

        loop = ReActLoop(
            llm_call=make_llm_call(self.llm, wrapped),
            execute_action=await make_action_executor(self, registry, push_filter),
            emit=make_event_emitter(wrapped),
            lang=lang,
            should_stop=lambda: bool(getattr(self, "stop", False)),
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        result = await loop.run(messages, bind_tools)
        self._react_loop_ran = True
        if result.kind == 'long_task':
            self.logger.info("Long task triggered inside ReAct loop — returning intent")
            return _build_long_task_intent(result.long_task_knowledge,
                                           result.long_task_tool_info)
        return None
```

新增 `_loop_system_guidance`（放在 `_get_fixed_system_prefix` 之后）：

```python
    def _loop_system_guidance(self) -> str:
        """Tool-usage guidance appended to the ReAct loop system prompt."""
        return """

## Tool Usage

- You have tools built from the user's knowledge base, plus `search_my_knowledge`
  to find more. Use them to complete the user's task.
- Work step by step: think about what is needed, call the right tool, observe
  the result, then decide the next step or write the final answer.
- You may call several tools in sequence and combine their results.
- If no tool matches the task, answer directly and suggest the user check the
  community for shared knowledge that may help.
- Never fabricate tool results. If a tool fails, try another approach or
  explain the failure honestly.
"""
```

`invoke_agent`（原 2227-2265 行）整体替换为：

```python
    async def invoke_agent(self, agent, callback_handler):
        """Post-loop wrap-up: stream pending raw items, store the turn."""
        if not getattr(self, "_react_loop_ran", False):
            self.logger.warning("invoke_agent called before the ReAct loop ran — no-op")
            return
        pending = getattr(self, "_pending_raw_items", None)
        if pending:
            await self._stream_raw_items(pending, callback_handler)
        collector = getattr(self, "_active_collector", None)
        self._store_current_turn(
            getattr(collector, "collected_text", "") if collector else ""
        )
```

（`_stream_workflow_final_result`、`_stream_raw_items` 等其余方法保留不动；`WorkflowExecutor` 文件保留不删。）

- [ ] **Step 5: 跑测试确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_integration_general_agent.py -v`
Expected: PASS（4 passed）

- [ ] **Step 6: 后端全量回归**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_loop.py tests/test_react_tools.py tests/test_react_integration_general_agent.py tests/test_sse_callback_events.py tests/test_general_agent_prune_summary.py tests/test_tool_result_filter.py tests/test_patent_detail_api.py tests/test_uspto_download.py tests/test_text_extractor.py tests/test_http_outbound.py -v`
Expected: 全部 PASS（基线环境性失败不在此列：firebase/ollama/redis 依赖的测试与本次无关）

- [ ] **Step 7: commit**

```bash
git add sources/agents/general_agent.py tests/test_react_integration_general_agent.py
git commit -m "feat: run ReAct loop in GeneralAgent pipeline; retire workflow branch"
```

---

### Task 5: 前端消息模型与流消费

**Files:**
- Modify: `frontend/nextjs/contexts/ChatContext.tsx`
- Modify: `frontend/nextjs/lib/chatSession.js`
- Modify: `frontend/nextjs/lib/useChatStream.ts`
- Modify: `frontend/nextjs/lib/chatStore.js`
- Test: `frontend/nextjs/lib/agentSteps.test.mjs`

**Interfaces:**
- Consumes: SSE 事件 `step/observation/agent_elapsed`（Task 1 契约）
- Produces（Task 6 依赖）:
  - `ChatMessage` 新字段：`agentSteps?: AgentStep[]`、`elapsedSeconds?: number`
  - `AgentStep = { round: number; thought: string; action: string; paramsBrief?: string; observationBrief?: string; reasoningText?: string; status: 'running'|'done'|'error' }`
  - `applyAgentStep(messages, messageId, event)`、`applyAgentObservation(messages, messageId, event)`、`applyAgentElapsed(messages, messageId, event)`（chatSession.js 纯函数）

- [ ] **Step 1: 写失败测试**

`frontend/nextjs/lib/agentSteps.test.mjs`:

```javascript
import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  applyAgentStep,
  applyAgentObservation,
  applyAgentElapsed,
} from './chatSession.js'
import { pruneMessagesForPersistence } from './chatStore.js'

const msg = { id: 'm1', role: 'assistant', content: '' }

test('applyAgentStep appends a running step', () => {
  const out = applyAgentStep([msg], 'm1', {
    round: 1, thought: '第 1 步', action: 'a',
    params_brief: '{}', reasoning_text: 'r',
  })
  assert.equal(out[0].agentSteps.length, 1)
  assert.equal(out[0].agentSteps[0].status, 'running')
  assert.equal(out[0].agentSteps[0].reasoningText, 'r')
  assert.equal(out[0].agentSteps[0].paramsBrief, '{}')
})

test('applyAgentStep merges into an existing round', () => {
  const once = applyAgentStep([msg], 'm1', { round: 1, thought: 'a', action: 'x' })
  const twice = applyAgentStep(once, 'm1', { round: 1, thought: 'b', action: 'y' })
  assert.equal(twice[0].agentSteps.length, 1)
  assert.equal(twice[0].agentSteps[0].thought, 'b')
  assert.equal(twice[0].agentSteps[0].action, 'y')
})

test('applyAgentObservation marks the step done', () => {
  const once = applyAgentStep([msg], 'm1', { round: 1, action: 'a' })
  const out = applyAgentObservation(once, 'm1', { round: 1, result_brief: 'ok' })
  assert.equal(out[0].agentSteps[0].status, 'done')
  assert.equal(out[0].agentSteps[0].observationBrief, 'ok')
})

test('applyAgentElapsed sets elapsedSeconds and closes running steps', () => {
  const once = applyAgentStep([msg], 'm1', { round: 1, action: 'a' })
  const out = applyAgentElapsed(once, 'm1', { elapsed_seconds: 3.2, steps: 1 })
  assert.equal(out[0].elapsedSeconds, 3.2)
  assert.equal(out[0].agentSteps[0].status, 'done')
})

test('events target only the matching message', () => {
  const other = { id: 'm2', role: 'assistant', content: '' }
  const out = applyAgentStep([msg, other], 'm9', { round: 1, action: 'a' })
  assert.equal(out[0].agentSteps, undefined)
  assert.equal(out[1].agentSteps, undefined)
})

test('persistence keeps agentSteps and elapsedSeconds', () => {
  const out = pruneMessagesForPersistence([{
    id: 'm1', role: 'assistant', content: 'x',
    agentSteps: [{ round: 1, action: 'a', status: 'done' }],
    elapsedSeconds: 2,
  }])
  assert.equal(out[0].agentSteps.length, 1)
  assert.equal(out[0].elapsedSeconds, 2)
})
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd E:\online\workspace\copiioai\langsistance\frontend\nextjs && node --test lib/agentSteps.test.mjs`
Expected: FAIL — `applyAgentStep` 未导出

- [ ] **Step 3: 实现 chatSession.js 纯函数**

`frontend/nextjs/lib/chatSession.js` 末尾追加（`applyAgentStep` 里 `reasoning_text` 为空时不写入字段）：

```javascript
/**
 * ReAct loop step events → message.agentSteps timeline.
 * step appends/merges a running step; observation closes it; agent_elapsed
 * stamps elapsedSeconds and closes any step still running.
 */
export function applyAgentStep(messages, messageId, event) {
  if (!event || !Number.isFinite(event.round)) return messages
  return messages.map((msg) => {
    if (msg.id !== messageId) return msg
    const steps = Array.isArray(msg.agentSteps) ? msg.agentSteps : []
    const idx = steps.findIndex((s) => s.round === event.round)
    const patch = {
      round: event.round,
      thought: event.thought || '',
      action: event.action || '',
      paramsBrief: event.params_brief || '',
      status: 'running',
    }
    if (event.reasoning_text) patch.reasoningText = event.reasoning_text
    if (idx >= 0) {
      const merged = { ...steps[idx], ...patch }
      return { ...msg, agentSteps: steps.map((s, i) => (i === idx ? merged : s)) }
    }
    return { ...msg, agentSteps: [...steps, patch] }
  })
}

export function applyAgentObservation(messages, messageId, event) {
  if (!event || !Number.isFinite(event.round)) return messages
  return messages.map((msg) => {
    if (msg.id !== messageId) return msg
    const steps = Array.isArray(msg.agentSteps) ? msg.agentSteps : []
    return {
      ...msg,
      agentSteps: steps.map((s) =>
        s.round === event.round
          ? { ...s, observationBrief: event.result_brief || '', status: 'done' }
          : s
      ),
    }
  })
}

export function applyAgentElapsed(messages, messageId, event) {
  const seconds = Number(event.elapsed_seconds)
  if (!Number.isFinite(seconds)) return messages
  return messages.map((msg) => {
    if (msg.id !== messageId) return msg
    const steps = Array.isArray(msg.agentSteps) ? msg.agentSteps : []
    return {
      ...msg,
      elapsedSeconds: seconds,
      agentSteps: steps.map((s) =>
        s.status === 'running' ? { ...s, status: 'done' } : s
      ),
    }
  })
}
```

- [ ] **Step 4: ChatContext.tsx 类型**

`frontend/nextjs/contexts/ChatContext.tsx`：

```typescript
export interface AgentStep {
  round: number
  thought: string
  action: string
  paramsBrief?: string
  observationBrief?: string
  reasoningText?: string
  status: 'running' | 'done' | 'error'
}

export interface ChatMessage {
  id: string
  role: string
  content: string
  artifacts?: ChatArtifact[]
  taskId?: string  // long task ID for progress tracking across save/load
  resultSummary?: string  // long task report markdown preview
  patent_ids?: string[]  // hidden — carried in conversation_history for follow-up queries
  results?: {
    setId: string
    source: string
    columns: Array<{ key: string; label: string; role: string }>
    rows: Array<Record<string, unknown>>
  }
  agentSteps?: AgentStep[]  // ReAct loop timeline — collapsed after completion
  elapsedSeconds?: number   // set by agent_elapsed; drives the collapse header
}
```

（同文件内 `[DIAG3]` 探针 effect **保持原样不动**。）

- [ ] **Step 5: useChatStream.ts 事件消费**

`frontend/nextjs/lib/useChatStream.ts` import 段追加：

```typescript
import {
  addAssistantArtifactChunk,
  addAssistantArtifactEnd,
  addAssistantArtifactStart,
  addAssistantPatentIds,
  applyAgentElapsed,
  applyAgentObservation,
  applyAgentStep,
  createChatId,
  createChatMessage,
  updateAssistantMessage,
  replaceAssistantMessage,
} from '@/lib/chatSession'
```

在 `if (event.type === 'long_task_created') { ... }` 分支之后、通用 token 回退之前插入：

```typescript
            if (event.type === 'step') {
              setMessages((m) => applyAgentStep(m, assistantId, event))
              continue
            }
            if (event.type === 'observation') {
              setMessages((m) => applyAgentObservation(m, assistantId, event))
              continue
            }
            if (event.type === 'agent_elapsed') {
              setMessages((m) => applyAgentElapsed(m, assistantId, event))
              continue
            }
```

- [ ] **Step 6: chatStore.js 持久化透传**

`frontend/nextjs/lib/chatStore.js` 的 `pruneMessagesForPersistence` 中 `if (msg.results)` 行之前追加：

```javascript
      // ReAct step timeline is small (bounded text per step) — keep it so
      // the collapsed "elapsed · N steps" header survives remounts.
      if (Array.isArray(msg.agentSteps) && msg.agentSteps.length > 0) pruned.agentSteps = msg.agentSteps
      if (Number.isFinite(msg.elapsedSeconds)) pruned.elapsedSeconds = msg.elapsedSeconds
```

- [ ] **Step 7: 跑测试 + 构建**

Run: `cd frontend/nextjs && node --test lib/*.test.mjs && npm run build`
Expected: 全部单测 PASS（含新增 6 条 + 现有 122 条）；build 无错误

- [ ] **Step 8: commit**

```bash
git add frontend/nextjs/contexts/ChatContext.tsx frontend/nextjs/lib/chatSession.js frontend/nextjs/lib/useChatStream.ts frontend/nextjs/lib/chatStore.js frontend/nextjs/lib/agentSteps.test.mjs
git commit -m "feat: consume ReAct step/observation/agent_elapsed events in chat stream"
```

---

### Task 6: 前端 workbuddy 步骤渲染（MarkdownMessage + i18n + 样式）

**Files:**
- Modify: `frontend/nextjs/components/app/MarkdownMessage.tsx`
- Modify: `frontend/nextjs/app/app/(auth)/chat/page.tsx:370-380`
- Modify: `frontend/nextjs/app/app/(auth)/results/page.tsx:226-236`
- Modify: `frontend/nextjs/lib/app-i18n/locales/en.ts`（chat 段）
- Modify: `frontend/nextjs/lib/app-i18n/locales/zh.ts`（chat 段）
- Modify: `frontend/nextjs/styles/app.css`

**Interfaces:**
- Consumes: Task 5 的 `AgentStep`/`agentSteps`/`elapsedSeconds`
- Produces: 用户可见交互——流式中显示已完成步骤行 + 进行中一行；完成后折叠为「⏱ 已用时间 · N 步」按钮，点击展开结构化时间线（思考 / 原始推理 / 工具+参数摘要 / 观察摘要）

- [ ] **Step 1: MarkdownMessage.tsx 改造**

Props 与类型（顶部，`interface ChatArtifact` 之后）：

```tsx
export interface AgentStep {
  round: number
  thought: string
  action: string
  paramsBrief?: string
  observationBrief?: string
  reasoningText?: string
  status: 'running' | 'done' | 'error'
}

interface Props {
  content: string
  artifacts?: ChatArtifact[]
  resultSummary?: string
  streaming: boolean
  transientStatus?: string
  analysisType?: string
  tableColumns?: string[]
  familyOverview?: Record<string, any>
  jurisdictions?: Array<{
    code: string
    label: string
    status: string
    progress: number
    detail: string
    file_count: number
    files_done: number
  }>
  agentSteps?: AgentStep[]
  elapsedSeconds?: number
}
```

函数签名与组件体：

```tsx
export default function MarkdownMessage({ content, artifacts = [], resultSummary, streaming, transientStatus = '', analysisType, tableColumns, familyOverview, jurisdictions, agentSteps = [], elapsedSeconds }: Props) {
  const { t } = useI18n()
  const [copied, setCopied] = useState(false)
  const [downloaded, setDownloaded] = useState(false)
  const [downloadedArtifactId, setDownloadedArtifactId] = useState<string | null>(null)
  const [html, setHtml] = useState('')
  const [stepsExpanded, setStepsExpanded] = useState(false)
  // ... 其余现有 state/effects/handlers 原样保留 ...

  const steps = agentSteps ?? []
  const hasSteps = steps.length > 0
  const doneSteps = steps.filter((s) => s.status === 'done')
  const runningStep = steps.find((s) => s.status === 'running')
```

return 内、`{showWaiting && (...)}` 块之前插入步骤 UI：

```tsx
      {streaming && hasSteps && (
        <div className="agent-steps" role="status" aria-live="polite">
          {doneSteps.map((step) => (
            <div key={step.round} className="agent-step-row agent-step-done">
              <span className="agent-step-check" aria-hidden="true">✓</span>
              <span className="agent-step-thought">{step.thought}</span>
            </div>
          ))}
          {runningStep && (
            <div key={`running-${runningStep.round}`} className="agent-step-row agent-step-running">
              <span className="agent-step-spinner" aria-hidden="true" />
              <span className="agent-step-thought">{runningStep.thought}</span>
            </div>
          )}
        </div>
      )}
      {!streaming && hasSteps && (
        <div className="agent-steps-collapsed">
          <button
            type="button"
            className="agent-steps-toggle"
            onClick={() => setStepsExpanded((v) => !v)}
            aria-expanded={stepsExpanded}
            aria-label={stepsExpanded ? t('chat.agentCollapseSteps') : t('chat.agentExpandSteps')}
          >
            <span aria-hidden="true">⏱</span>
            <span>
              {t('chat.agentElapsed')
                .replace('{seconds}', String(elapsedSeconds ?? 0))
                .replace('{steps}', String(steps.length))}
            </span>
            <span className={`agent-steps-chevron${stepsExpanded ? ' expanded' : ''}`} aria-hidden="true">▾</span>
          </button>
          {stepsExpanded && (
            <div className="agent-steps-expanded">
              {steps.map((step) => (
                <div key={step.round} className="agent-step-detail">
                  <div className="agent-step-detail-header">{step.thought}</div>
                  {step.reasoningText && (
                    <div className="agent-step-reasoning">{step.reasoningText}</div>
                  )}
                  <div className="agent-step-action">
                    <span className="agent-step-label">{t('chat.agentStepAction')}</span>
                    {step.action}{step.paramsBrief ? ` (${step.paramsBrief})` : ''}
                  </div>
                  {step.observationBrief && (
                    <div className="agent-step-observation">
                      <span className="agent-step-label">{t('chat.agentStepObservation')}</span>
                      {step.observationBrief}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
```

- [ ] **Step 2: i18n 键（en.ts + zh.ts）**

en.ts 的 `chat:` 段（`longTaskRunning` 附近）追加：

```typescript
    agentElapsed: 'Elapsed {seconds}s · {steps} steps',
    agentStepAction: 'Action',
    agentStepObservation: 'Observed',
    agentExpandSteps: 'Expand detailed steps',
    agentCollapseSteps: 'Collapse detailed steps',
```

zh.ts 对应位置追加：

```typescript
    agentElapsed: '已用时间 {seconds} 秒 · {steps} 步',
    agentStepAction: '调用',
    agentStepObservation: '观察',
    agentExpandSteps: '展开详细步骤',
    agentCollapseSteps: '收起详细步骤',
```

- [ ] **Step 3: 两个调用点传新 props**

chat/page.tsx 的 MarkdownMessage（370 行起）追加：

```tsx
                    agentSteps={msg.agentSteps}
                    elapsedSeconds={msg.elapsedSeconds}
```

results/page.tsx 的 MarkdownMessage（226 行起）同样追加这两行。

- [ ] **Step 4: styles/app.css 样式**

文件末尾追加（使用现有设计变量）：

```css
/* ── ReAct agent steps timeline (workbuddy style) ── */
.agent-steps {
  margin: 0 0 var(--spacing-sm);
  font-size: 13px;
  color: var(--text-secondary);
}
.agent-step-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 2px 0;
}
.agent-step-check { color: var(--accent); }
.agent-step-thought { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.agent-step-spinner {
  width: 12px;
  height: 12px;
  border: 2px solid var(--divider);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: agent-step-spin 0.8s linear infinite;
  flex-shrink: 0;
}
@keyframes agent-step-spin { to { transform: rotate(360deg); } }

.agent-steps-collapsed { margin: 0 0 var(--spacing-sm); }
.agent-steps-toggle {
  display: flex;
  align-items: center;
  gap: 8px;
  background: none;
  border: none;
  padding: 4px 0;
  cursor: pointer;
  font-size: 13px;
  color: var(--text-secondary);
}
.agent-steps-toggle:hover { color: var(--text-primary); }
.agent-steps-chevron { transition: transform 0.2s ease; display: inline-block; }
.agent-steps-chevron.expanded { transform: rotate(180deg); }

.agent-steps-expanded {
  margin: var(--spacing-sm) 0 0;
  padding-left: var(--spacing-md);
  border-left: 2px solid var(--divider);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  font-size: 13px;
  color: var(--text-secondary);
}
.agent-step-detail-header { font-weight: 600; color: var(--text-primary); }
.agent-step-label { font-weight: 600; margin-right: 6px; color: var(--text-primary); }
.agent-step-reasoning {
  margin: 2px 0;
  font-style: italic;
  white-space: pre-wrap;
  opacity: 0.8;
}
.agent-step-action, .agent-step-observation { margin: 2px 0; overflow-wrap: anywhere; }
```

- [ ] **Step 5: 构建验证**

Run: `cd frontend/nextjs && npm run build`
Expected: 构建通过（TypeScript 检查含新 props 类型）

- [ ] **Step 6: commit**

```bash
git add frontend/nextjs/components/app/MarkdownMessage.tsx "frontend/nextjs/app/app/(auth)/chat/page.tsx" "frontend/nextjs/app/app/(auth)/results/page.tsx" frontend/nextjs/lib/app-i18n/locales/en.ts frontend/nextjs/lib/app-i18n/locales/zh.ts frontend/nextjs/styles/app.css
git commit -m "feat: render ReAct step timeline with collapse/expand in chat messages"
```

---

### Task 7: 全量回归与手动验收清单

**Files:** 无（验证任务）

- [ ] **Step 1: 后端全量相关套件**

Run: `cd E:\online\workspace\copiioai\langsistance && PYTHONUTF8=1 python -m pytest tests/test_react_loop.py tests/test_react_tools.py tests/test_react_integration_general_agent.py tests/test_sse_callback_events.py tests/test_general_agent_prune_summary.py tests/test_tool_result_filter.py tests/test_patent_detail_api.py tests/test_uspto_download.py tests/test_text_extractor.py tests/test_http_outbound.py -v`
Expected: 全部 PASS（firebase/ollama/redis 依赖的基线环境性失败不计，与本次无关）

- [ ] **Step 2: 前端单测 + 构建**

Run: `cd frontend/nextjs && node --test lib/*.test.mjs && npm run build`
Expected: 全部 PASS + 构建成功

- [ ] **Step 3: 手动验收清单（test.copiioai.com，用户确认）**

1. 聊天提问（普通闲聊）→ 直接回答，无步骤条
2. 专利搜索提问 → 执行中看到「✓ 第1步」+「⏳ 进行中」行；完成后折叠为「⏱ 已用时间 · N 步」；点击展开看到思考/工具/参数/观察时间线；**自动跳转结果页照旧**、下载按钮照旧
3. 提问需要多工具串联（先检索再下载文档）→ 步骤时间线 ≥2 步
4. 专利列表 >100 条 → 展开时间线中观察注明「已截断…前 100 条」，结果页 100 行
5. 专利文档列表（documents 类）→ 不限条数
6. 长任务提问 → 步骤行出现后消息转为任务进度卡片（现状行为）
7. 多轮跟进提问 → 对话上下文保持
8. 中断按钮 → 循环收尾、无卡死
9. 刷新页面 → 历史消息的折叠头「⏱ 已用时间 · N 步」仍在（sessionStorage 持久化）
10. 无匹配知识的提问 → 直接回答并建议社区（现状文案）

- [ ] **Step 4: 全部验收通过后，按需将本轮 6 个 commit 推送到远程 / 合并**

```bash
git log --oneline main..HEAD   # 确认只含本计划的 6 个 commit 及 spec/plan 文档
```

---

## Self-Review 记录

1. **Spec coverage**：§3 循环控制流→Task 2；§4 工具供给（search 元工具/top-5/长任务工具/type-2 退役）→Task 3；§5 行动处理器（100 条截断/文档不限/长任务 intent）→Task 3+4；§6 SSE 事件→Task 1；§7 错误处理（工具异常继续、同工具 2 连败兜底、轮次上限、中断）→Task 2 循环内实现（中断经 core 现有 Abort 链路传播，SSE 断开后 task 被 cancel）；§8 前端交互→Task 5+6；§9 测试→各任务 + Task 7；§10 文件清单→一致（api_routes/core.py 有 Task 1 的透传小改，spec 说「基本不动」，属契约性最小改动）
2. **Placeholder scan**：无 TBD/TODO；所有步骤含完整代码与命令
3. **Type consistency**：`RoundResult`/`ToolEntry`/`on_step` 签名/Task 5 事件字段（`reasoning_text`→`reasoningText`）跨任务一致；`applyAgentStep` 合并语义与 `MarkdownMessage` 的 `AgentStep` 接口一致
