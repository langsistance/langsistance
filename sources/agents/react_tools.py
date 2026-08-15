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
        dynamic_tool = agent.get_dynamic_tool_for(knowledge, tool_info)
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
            dynamic_tool = agent.get_dynamic_tool_for(knowledge, tool_info)
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
