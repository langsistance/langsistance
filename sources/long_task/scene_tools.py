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
      knowledge_params, knowledge_answer,
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
                       k.answer      AS knowledge_answer,
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
            "knowledge_answer": row["knowledge_answer"] or "",
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


# ── Params sanitisation ────────────────────────────────────────────────────

_SENSITIVE_HEADER_RE = __import__('re').compile(
    r'api[_-]?key|token|secret|password|auth',
    __import__('re').IGNORECASE,
)


def _sanitize_params_for_llm(params_str: str) -> str:
    """Return params JSON with sensitive header values replaced by '****'.

    The real header values stay in the tool definition for server-side use only.
    This sanitised copy is the only version the LLM ever sees.
    """
    if not params_str:
        return ""
    try:
        data = json.loads(params_str)
        if not isinstance(data, dict):
            return params_str
        sanitized = {}
        for k, v in data.items():
            if k == "header" and isinstance(v, dict):
                sanitized[k] = {
                    hk: "****" if _SENSITIVE_HEADER_RE.search(hk) else hv
                    for hk, hv in v.items()
                }
            else:
                sanitized[k] = v
        return json.dumps(sanitized, ensure_ascii=False, indent=2)
    except Exception:
        return params_str


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

    # Separate tool candidates (selectable) from workflow knowledge (reference only)
    tool_candidates = [c for c in candidates if c.get("_tool_item")]
    workflow_candidates = [
        c for c in candidates
        if not c.get("_tool_item") and c.get("knowledge_type") in (2, 3)
    ]
    if not tool_candidates:
        return None

    # Build candidate tool list — include usage guide (answer) and params template
    tool_blocks = []
    for c in tool_candidates:
        params_template = _sanitize_params_for_llm(c.get("tool_params", "") or c.get("knowledge_params", ""))
        params_block = ""
        if params_template:
            params_block = f"\n    Params 模板: {params_template}"
        # Include knowledge answer as usage guide (same role as general_agent's Context from knowledge base)
        usage_guide = c.get("knowledge_answer", "")
        usage_block = ""
        if usage_guide:
            usage_block = f"\n    使用指南: {usage_guide}"
        tool_blocks.append(
            f"  [id={c['knowledge_id']}] {c['tool_title']}\n"
            f"    URL: {c['tool_url']}\n"
            f"    描述: {c['knowledge_description'] or c['tool_description']}"
            f"{usage_block}"
            f"{params_block}"
        )
    tool_text = "\n".join(tool_blocks)

    # Build workflow reference (not selectable, but provides context)
    workflow_text = ""
    if workflow_candidates:
        wf_lines = []
        for c in workflow_candidates:
            wf_lines.append(
                f"  - {c['knowledge_question']}: {c['knowledge_description']}"
            )
        workflow_text = (
            "\n\n参考工作流（描述正确的调用方式，但不直接可选）：\n"
            + "\n".join(wf_lines)
        )

    system_prompt = (
        "你是一个工具选择器。根据给定的目标和上下文，从可用工具中选择最合适的工具，"
        "并基于该工具的 Params 模板生成调用参数。\n\n"
        "可用工具：\n"
        f"{tool_text}"
        f"{workflow_text}\n\n"
        "工具选择优先级（CRITICAL）：\n"
        "— 默认优先选择 assignee（专利受让人）搜索工具，按公司/机构名检索专利\n"
        "— 只有在用户明确要求用关键词检索时，才选择 keyword 搜索工具\n"
        "— 其他工具（如文档下载）根据目标选择合适的即可\n\n"
        "参数生成规则（与 general_agent 对齐）：\n"
        "1. 基于选中工具的 Params 模板来生成参数，保持完全相同的 JSON 结构\n"
        "2. 只修改模板中与用户目标明确相关的字段值，不要新增或删除字段\n"
        "   — Replace a value only if the user query clearly maps to the meaning of an existing field\n"
        "   — 模板中的示例值展示了该字段的预期格式和语法，参照示例的格式来替换值\n"
        "3. 不要修改 method、Content-Type、header 等固定字段\n"
        "4. 如果模板中某个字段的值与用户目标无关，保持原值不变\n"
        "5. 如果工具 URL 已包含 /applications，path 只需 /{patent_id}/documents\n"
        "6. CRITICAL — assignee 搜索规则：\n"
        "   — assignee 搜索的 body.q 只能用公司名检索，不要附加标题、技术关键词等额外条件\n"
        "   — 即使用户提到了某个技术领域（如'自动驾驶'），也只搜公司名\n"
        "   — 技术主题的筛选留给后续的专利分析阶段处理\n"
        "   — 反例：q: \"Tesla AND autonomous\"（错）\n"
        "   — 正例：q: \"Tesla Inc.\" 或 q: \"Tesla Motors\"（对）\n\n"
        "Return JSON:\n"
        '{"knowledge_id": <selected tool id>, '
        '"reason": "<why this tool>", '
        '"params": <完整的 params 对象，基于模板修改，保持原结构>}'
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
    return list(extract_patent_id_url_map(items).keys())


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
