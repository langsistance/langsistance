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

            for _ci, call in enumerate(tool_calls):
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
                # A transient status accompanies every tool call so frontends
                # without the step timeline still show live progress during
                # the silent tool rounds (no token stream in between).
                await self._emit("status", {
                    "message": self._thought_line(steps, action_name),
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

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": call.get("id") or f"call_{steps}",
                    "name": action_name,
                    "content": obs,
                }

                is_error = obs.startswith("Error:")
                if is_error and action_name == last_failed_action:
                    consecutive_failures += 1
                elif is_error:
                    consecutive_failures = 1
                else:
                    consecutive_failures = 0
                last_failed_action = action_name if is_error else None
                if consecutive_failures >= 2:
                    # Every assistant tool_call must have a tool output
                    # before the fallback LLM call, or OpenAI-compatible
                    # APIs reject the history (400 "No tool output found
                    # for function call ...").  Complete the current and
                    # remaining calls of this round first.
                    messages.append(tool_msg)
                    for _c in tool_calls[_ci + 1:]:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": _c.get("id") or "",
                            "name": _c.get("name", ""),
                            "content": "Error: skipped due to repeated failures",
                        })
                    return await self._finish("fallback", messages, steps, start)

                messages.append(tool_msg)

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
