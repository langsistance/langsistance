"""Scene-aware tool discovery, selection, and execution for long tasks.

The long task does NOT hardcode patent search/download logic.
Instead it reads all knowledge+tools from the bound scene and lets
the Flash LLM select which tool to call for each step.
"""

import json
import os
from typing import Any, Dict, List, Optional

from sources.knowledge.knowledge import (
    KnowledgeItem,
    ToolItem,
    get_db_connection,
    get_tool_by_id,
)
from sources.dynamic_tool_params import execute_backend_tool_request


# ── Scene knowledge discovery ──────────────────────────────────────────────

def get_scene_knowledge_tools(scene_id: int) -> List[Dict[str, Any]]:
    """Read all knowledge + linked tools for a scene.

    Returns a list of dicts each with:
      knowledge_id, knowledge_question, knowledge_description, knowledge_type,
      knowledge_params,
      tool_id, tool_title, tool_description, tool_url, tool_push, tool_params
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT
                       k.id          AS knowledge_id,
                       k.question    AS knowledge_question,
                       k.description AS knowledge_description,
                       k.`type`      AS knowledge_type,
                       k.params      AS knowledge_params,
                       t.id          AS tool_id,
                       t.title       AS tool_title,
                       t.description AS tool_description,
                       t.url         AS tool_url,
                       t.push        AS tool_push,
                       t.params      AS tool_params,
                       t.timeout     AS tool_timeout
                   FROM knowledge k
                   LEFT JOIN tools t
                     ON k.tool_id = t.id AND t.status = 1
                   WHERE k.status = 1
                     AND k.scene_id = %s
                   ORDER BY k.`type` DESC, k.update_time DESC""",
                (scene_id,))
            rows = cur.fetchall()
    finally:
        conn.close()

    candidates = []
    for row in rows:
        entry: Dict[str, Any] = {
            "knowledge_id": row["knowledge_id"],
            "knowledge_question": row["knowledge_question"],
            "knowledge_description": row["knowledge_description"] or "",
            "knowledge_type": row["knowledge_type"],
            "knowledge_params": row["knowledge_params"] or "",
            "tool_id": row["tool_id"],
            "tool_title": row["tool_title"] or "",
            "tool_description": row["tool_description"] or "",
            "tool_url": row["tool_url"] or "",
            "tool_push": row["tool_push"],
            "tool_params": row["tool_params"] or "",
            "tool_timeout": row["tool_timeout"],
        }
        # Build a ToolItem for internal use
        if row["tool_id"]:
            entry["_tool_item"] = ToolItem(
                id=row["tool_id"],
                user_id="",
                title=row["tool_title"] or "",
                description=row["tool_description"] or "",
                push=row["tool_push"],
                url=row["tool_url"] or "",
                status=True,
                timeout=row["tool_timeout"] or 30,
                params=row["tool_params"] or "",
            )
        candidates.append(entry)

    return candidates


# ── LLM-driven tool selection ──────────────────────────────────────────────

async def select_tool(
    goal: str,
    context: str,
    candidates: List[Dict[str, Any]],
    flash_provider,
) -> Optional[Dict[str, Any]]:
    """Let the Flash LLM pick the best tool + generate call parameters.

    Args:
        goal: What we need — e.g. "search patents" or "download patent document"
        context: The user's query or patent_id being worked on
        candidates: List from ``get_scene_knowledge_tools()``
        flash_provider: Provider instance with ``complete_json()``

    Returns:
        None if no suitable tool, or dict with ``tool`` (ToolItem) and
        ``params`` (dict ready for ``execute_backend_tool_request``).
    """
    if not candidates:
        return None

    # Filter to only candidates that have a linked tool
    tool_candidates = [c for c in candidates if c.get("_tool_item")]
    if not tool_candidates:
        return None

    # Build a compact candidate list for the LLM
    candidate_lines = []
    for c in tool_candidates:
        candidate_lines.append(
            f"  - id={c['knowledge_id']}, name={c['knowledge_question']}, "
            f"desc={c['knowledge_description']}, "
            f"tool={c['tool_title']}, tool_desc={c['tool_description']}"
        )
    candidate_text = "\n".join(candidate_lines)

    system_prompt = (
        "你是一个工具选择器。根据给定的目标和上下文，从候选中选择最合适的工具，"
        "并生成调用该工具所需的参数。\n\n"
        "可用工具：\n"
        f"{candidate_text}\n\n"
        "返回 JSON 格式：\n"
        '{"knowledge_id": <选中的知识ID>, '
        '"reason": "<选择原因>", '
        '"params": {"query": {...}, "body": {...}}}'
    )

    result = await flash_provider.complete_json(
        system_prompt,
        f"目标：{goal}\n上下文：{context}",
    )

    if not result or not result.get("knowledge_id"):
        return None

    knowledge_id = result["knowledge_id"]
    selected = next(
        (c for c in tool_candidates if c["knowledge_id"] == knowledge_id),
        None,
    )
    if selected is None:
        return None

    return {
        "tool": selected["_tool_item"],
        "knowledge": selected,
        "params": result.get("params", {}),
        "reason": result.get("reason", ""),
    }


# ── Tool execution wrapper ─────────────────────────────────────────────────

async def execute_tool(
    tool_info: ToolItem,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute a backend tool and return parsed results.

    Wraps ``execute_backend_tool_request()`` and extracts ``raw_items``
    from the response.  For CNIPA (open.zldsj.com) responses the
    ``context.records`` unwrapping is already handled internally.
    """
    import asyncio

    result = await asyncio.to_thread(
        execute_backend_tool_request, tool_info, params
    )
    return result


# ── Result extraction ──────────────────────────────────────────────────────

def extract_patent_ids(items: List[Dict[str, Any]]) -> List[str]:
    """Extract patent IDs from tool results (raw_items), deduplicated.

    Tries these fields in order: patent_id, patentNumber,
    applicationNumberText, application_number, 申请号.
    """
    return list(_extract_patent_id_url_map(items).keys())


def extract_patent_id_url_map(
    items: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Extract patent_id → document_url mapping from search results.

    Probes common URL field names found in CNIPA and USPTO responses.
    """
    id_url_map: Dict[str, str] = {}
    for item in (items or []):
        if not isinstance(item, dict):
            continue
        pid = (
            item.get("patent_id")
            or item.get("patentNumber")
            or item.get("applicationNumberText")
            or item.get("application_number")
            or item.get("申请号")
        )
        if not pid:
            continue
        pid = str(pid)
        if pid in id_url_map:
            continue
        url = (
            item.get("document_url")
            or item.get("fulltext_url")
            or item.get("download_url")
            or item.get("url")
            or item.get("说明书URL")
        )
        id_url_map[pid] = url or ""
    return id_url_map
    """
    seen = set()
    ids = []
    for item in (items or []):
        if not isinstance(item, dict):
            continue
        pid = (
            item.get("patent_id")
            or item.get("patentNumber")
            or item.get("applicationNumberText")
            or item.get("application_number")
            or item.get("申请号")
        )
        if pid and pid not in seen:
            seen.add(pid)
            ids.append(str(pid))
    return ids


def extract_document_text(result: Dict[str, Any]) -> Optional[str]:
    """Extract patent document text from a download tool result.

    Tries raw_items first, then description/abstract fields.
    """
    raw_items = result.get("raw_items")
    if raw_items:
        for item in raw_items:
            if isinstance(item, dict):
                text = (
                    item.get("description")
                    or item.get("abstract")
                    or item.get("full_text")
                    or item.get("专利内容")
                )
                if text:
                    return text

    data = result.get("data", {})
    if isinstance(data, dict):
        return (
            data.get("description")
            or data.get("abstract")
            or str(data)
        )
    return str(data) if data else None
