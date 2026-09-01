from typing import Dict, Any
import json
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup

from sources.long_task.candidate_metadata import is_documents_tool
from sources.knowledge.knowledge import (
    get_redis_connection,
    get_knowledge_tool,
    select_knowledge_tool_with_llm,
    clean_html_text,
)
from sources.utility import pretty_print, animate_thinking
from sources.agents.agent import Agent
from sources.tools.mcpFinder import MCP_finder
from sources.memory import Memory
from sources.logger import Logger
from sources.dynamic_tool_params import (
    _append_path_to_url,
    _coerce_json_object,
    _is_path_query_body_empty,
    _replace_uspto_download_urls_for_batch,
    execute_backend_tool_request,
)
from sources.tool_result_filter import (
    filter_tool_result_items,
    tool_result_filter_enabled,
    unfiltered_result,
)
from sources.result_export import build_result_artifacts
from sources.agents.react_loop import ReActLoop, make_event_emitter, make_llm_call
from sources.agents.react_tools import (
    CPC_EXPANSION_ENABLED,
    build_tool_set,
    make_action_executor,
    _match_long_task_intent,
)
from sources.http_outbound import outbound_http

from langchain_core.tools import StructuredTool
from langchain_core.callbacks import BaseCallbackHandler

import os
import time
import re
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

# Headers whose values must never be exposed to the LLM
_SENSITIVE_HEADER_RE = re.compile(
    r'(auth|api.?key|token|secret|credential|password|bearer)',
    re.IGNORECASE
)
_URL_IN_TEXT_RE = re.compile(r'https?://[^\s"\'<>\])}]+')
MAX_BATCH_JSON_CHARS_FOR_LLM = 200000
MAX_MARKDOWN_VALUE_CHARS = 500
MAX_ITEM_CHARS_FOR_LLM = int(os.getenv("GENERAL_AGENT_MAX_ITEM_CHARS", "15000"))
MAX_VALUE_CHARS_THRESHOLD = int(os.getenv("GENERAL_AGENT_MAX_VALUE_CHARS", "10000"))
SMALL_LIST_THRESHOLD = int(os.getenv("GENERAL_AGENT_SMALL_LIST_THRESHOLD", "3"))
USE_LARGE_LIST_SUMMARY = True  # 设为 False 切回逐条批处理模式
RELEVANT_TOP_N = int(os.getenv("REACT_RELEVANT_TOP_N", "10"))
# ── Multi-turn context injection (需求 2: 追问必须带前文) ──
# The request's conversation_history is injected into the system prompt
# as a reference-only "Previous conversation" block.  Bounded so old
# rounds cannot crowd out the latest question (token cost + steering
# risk both grow with history size).
CONTEXT_TURNS_MAX = int(os.getenv("REACT_CONTEXT_TURNS_MAX", "6"))
CONTEXT_TURN_CHARS = int(os.getenv("REACT_CONTEXT_TURN_CHARS", "800"))
# Long-task progress cards (🔬 running, ✅ completed, ❌ failed, ⏸ paused,
# ⏹ stopping) are UI noise — never injected into the LLM context.
_HISTORY_NOISE_MARKERS = ("🔬", "✅", "❌", "⏸", "⏹", "Task ID", "任务ID")

# Redis key prefix and TTL for storing patent IDs from conversation artifacts
_CONV_PATENT_IDS_KEY_PREFIX = "lt:conv"
_CONV_PATENT_IDS_TTL = 3600  # 1 hour

# Regex to strip user_id / query_id from historical user messages before
# retaining them in multi-turn memory (current call still embeds them fresh).
_STRIP_IDS_RE = re.compile(
    r',?\s*user id is \d+,\s*query id is \w+,?\s*',
    re.IGNORECASE,
)


def _is_history_noise(content: str) -> bool:
    """Long-task progress cards and similar UI noise must never enter the
    LLM context (需求 2 防干扰: history must not steer the latest
    question)."""
    return any(marker in content for marker in _HISTORY_NOISE_MARKERS)


def _conversation_history_turns(conv_history, current_query: str = "") -> list:
    """Extract clean turns from the request's conversation_history (flat
    role/content list with optional hidden patent_ids on assistant
    messages).  Filters UI noise and non-user/assistant roles, strips id
    markers, truncates each turn, drops the current question (it is the
    loop's last user message, not history), and caps at CONTEXT_TURNS_MAX."""
    if not isinstance(conv_history, list):
        return []
    current_query = str(current_query or "").strip()
    cleaned = []
    for msg in conv_history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        if role not in ("user", "assistant"):
            continue
        content = str(msg.get("content") or "").strip()
        if _is_history_noise(content):
            continue
        content = _STRIP_IDS_RE.sub("", content).strip()
        if not content:
            continue
        if role == "user" and current_query and content == current_query:
            continue  # the latest question — already the loop's user message
        if len(content) > CONTEXT_TURN_CHARS:
            content = content[:CONTEXT_TURN_CHARS] + "..."
        cleaned.append({
            "role": role,
            "content": content,
            "patent_ids": msg.get("patent_ids") or [],
        })
    return cleaned[-CONTEXT_TURNS_MAX:]


def _build_previous_conversation_block(
    conv_history, pooled_turns, user_id, current_query: str = "",
) -> tuple:
    """Build the reference-only "Previous conversation" system-prompt block.

    Priority: the request's conversation_history (frontend-authoritative,
    carries hidden patent_ids).  Fallback: the pooled agent's
    _conversation_turns, filtered to this user (pooled agents are shared,
    so turns from other users must never leak in).

    Returns (block, patent_ids_found) — the caller uses the second value
    to decide whether a Redis recent-patent-note is needed.
    """
    history_turns = _conversation_history_turns(conv_history, current_query)
    if not history_turns and pooled_turns:
        history_turns = []
        for turn in pooled_turns:
            if not isinstance(turn, dict):
                continue
            # Legacy turns (pre-user-keying) carry no user_id — treat as
            # belonging to the current user; new turns must match exactly.
            if turn.get("user_id", user_id) != user_id:
                continue
            user_txt = str(turn.get("user", "") or "").strip()
            assistant_txt = str(turn.get("assistant", "") or "").strip()
            if _is_history_noise(user_txt) or _is_history_noise(assistant_txt):
                continue
            user_txt = _STRIP_IDS_RE.sub("", user_txt).strip()
            if len(user_txt) > CONTEXT_TURN_CHARS:
                user_txt = user_txt[:CONTEXT_TURN_CHARS] + "..."
            if len(assistant_txt) > CONTEXT_TURN_CHARS:
                assistant_txt = assistant_txt[:CONTEXT_TURN_CHARS] + "..."
            history_turns.append({"role": "user", "content": user_txt,
                                  "patent_ids": []})
            history_turns.append({"role": "assistant",
                                  "content": assistant_txt or "(tool executed)",
                                  "patent_ids": []})
        history_turns = history_turns[-CONTEXT_TURNS_MAX * 2:]

    if not history_turns:
        return "", []

    lines = []
    patent_ids = []
    for t in history_turns:
        label = "User" if t.get("role") == "user" else "Assistant"
        lines.append(f"{label}: {t.get('content', '')}")
        if t.get("role") == "assistant":
            for pid in (t.get("patent_ids") or []):
                pid = str(pid).strip()
                if pid and pid not in patent_ids:
                    patent_ids.append(pid)

    block = (
        "\n\n## Previous conversation (historical, reference only)\n"
        "以下为历史对话，仅作参考。必须严格以用户最新提问为准作答，"
        "不得被历史中的主题、专利号或要求带偏。\n"
        "Reference only — answer the user's LATEST question; never let "
        "history override it.\n\n"
        + "\n".join(lines)
    )
    if patent_ids:
        block += ("\n\n前序检索命中专利号（仅当用户引用时使用）："
                  + ", ".join(patent_ids[:20]))
    return block, patent_ids


def _read_recent_patent_ids(user_id) -> list:
    """Read the user's recent patent IDs from Redis (written after every
    artifact generation, 1h TTL — 需求 2 跨会话记忆).  Failure degrades
    silently to empty."""
    if not user_id:
        return []
    try:
        r = get_redis_connection()
        stored = r.get(f"{_CONV_PATENT_IDS_KEY_PREFIX}:{user_id}:patent_ids")
        if not stored:
            return []
        ids = json.loads(stored)
        if not isinstance(ids, list):
            return []
        return [str(pid).strip() for pid in ids if str(pid).strip()]
    except Exception:
        return []


def _summary_system_prompt(ranked: bool, lang: str) -> str:
    """Build the large-list summary prompt; ranked lists get a
    relevance-order note so the summary leads with the most relevant."""
    ranked_note_zh = "列表已按相关度排序，开头优先呈现最相关项。"
    ranked_note_en = ("The list is already relevance-ranked — lead with "
                      "the most relevant items.")
    if lang == "en":
        base = (
            "You are a professional data analyst. Create a concise, well-structured summary of the data items below. "
            "Group similar items, highlight key patterns or trends, and present information clearly for non-technical readers. "
            "Use Markdown formatting — including tables where appropriate. Keep it under 600 words. "
            "Do NOT list every item individually; synthesize and summarize. "
            "Focus on: what the data shows overall, key differences between items, any notable outliers. "
            "IMPORTANT: The user asked in English — respond entirely in English."
        )
        return base + (" " + ranked_note_en if ranked else "")
    base = (
        "你是一名专业的数据分析师。请对以下数据项创建一个简洁、结构清晰的摘要。"
        "将相似的项目分组，突出关键模式或趋势，并以非技术读者易于理解的方式呈现。"
        "使用 Markdown 格式——包括适当的表格。保持在 600 字以内。"
        "不要逐项列出；综合和总结。"
        "重点关注：数据整体显示的内容、项目之间的关键差异、任何值得注意的异常值。"
        "重要：用户使用中文提问——请用中文回答。"
    )
    return base + (" " + ranked_note_zh if ranked else "")


def _store_conversation_patent_ids(agent, items_for_export: list) -> list | None:
    """Store patent IDs from conversation artifacts into Redis.

    Called after every artifact generation so that follow-up long-task
    queries (scenario='conversation_refs') can retrieve patent IDs even
    when the conversation text is a summary (large-list mode).

    Keyed by user_id — always available on both the write side (general
    agent) and read side (long task), without requiring client-side
    session_id propagation.

    Returns the extracted patent_ids list (may be empty) so callers can
    also emit them to the frontend via SSE.
    """
    user_id = getattr(agent, '_last_user_id', None)
    patent_ids = _extract_patent_ids_from_items(items_for_export)
    if not patent_ids:
        return None
    if user_id:
        try:
            from sources.knowledge.knowledge import get_redis_connection
            from sources.logger import Logger
            _store_logger = Logger("general_agent.log")
            r = get_redis_connection()
            key = f"{_CONV_PATENT_IDS_KEY_PREFIX}:{user_id}:patent_ids"
            r.set(key, json.dumps(patent_ids, ensure_ascii=False),
                  ex=_CONV_PATENT_IDS_TTL)
            _store_logger.info(
                f"stored_conversation_patent_ids — "
                f"count={len(patent_ids)}, key={key}"
            )
        except Exception:
            pass  # Non-critical: long task will fall back to text extraction
    return patent_ids


async def _emit_patent_ids_to_frontend(agent, items_for_export, callback_handler) -> None:
    """Extract patent IDs and emit as a hidden SSE event to the frontend.

    The frontend stores these IDs in the assistant message (not displayed),
    so follow-up conversation_refs queries include them in conversation_history.
    """
    patent_ids = _extract_patent_ids_from_items(items_for_export)
    if not patent_ids:
        return
    cb_queue = getattr(callback_handler, 'queue', None)
    if cb_queue is None:
        agent.logger.warning("_emit_patent_ids_to_frontend: callback_handler has no queue")
        return
    await cb_queue.put({
        'type': 'patent_ids',
        'patent_ids': patent_ids,
    })
    agent.logger.info(
        f"_emit_patent_ids_to_frontend — count={len(patent_ids)}, "
        f"ids={patent_ids[:5]}{'...' if len(patent_ids) > 5 else ''}"
    )


def _extract_patent_ids_from_items(items: list) -> list:
    """Extract patent application numbers from raw search result items.

    Handles both USPTO (8-digit applicationNumberText) and CNIPA
    (YYYY + 8+ digits in various fields) formats.
    """
    patent_ids = []
    for item in items:
        if not isinstance(item, dict):
            continue
        # USPTO: applicationNumberText is the primary field
        app_num = str(item.get('applicationNumberText', '')).strip()
        if app_num and app_num.isdigit() and 7 <= len(app_num) <= 12:
            patent_ids.append(app_num)
            continue
        # CNIPA / other sources: check common patent ID field names —
        # Baiten flat candidates carry patent_id (publication number) /
        # patent_number / app_num, none of which matched before, so CN
        # patents never reached the conversation_refs channel.
        for key in ('patent_id', 'patent_number', 'app_num',
                    'applicationNumber', 'patentApplicationNumber',
                    'apc', 'patentNumber', '专利申请号'):
            val = str(item.get(key, '')).strip()
            if val and len(val) >= 8:
                patent_ids.append(val)
                break
        # Check nested applicationMetaData
        meta = item.get('applicationMetaData')
        if isinstance(meta, dict):
            pn = str(meta.get('patentNumber', '')).strip()
            if pn and pn.isdigit() and len(pn) >= 8 and pn not in patent_ids:
                # This is a granted patent number — also check for app number
                pass  # applicationNumberText already handled above
    # Deduplicate preserving order
    seen = set()
    result = []
    for pid in patent_ids:
        if pid not in seen:
            seen.add(pid)
            result.append(pid)
    return result


def _infer_result_source(tool_info) -> str:
    """Infer the search result source from the backend tool URL."""
    url = (getattr(tool_info, "url", "") or "").lower()
    if "patents.google.com" in url:
        return "google_patents"
    if is_documents_tool(tool_info):
        return "uspto_documents"
    return "uspto"


def _is_long_task_knowledge(knowledge_item) -> bool:
    """Check if knowledge item represents a long task (type=3)."""
    if knowledge_item is None:
        return False
    k_type = knowledge_item.get('type', 1) if isinstance(knowledge_item, dict) else getattr(knowledge_item, 'type', 1)
    return int(k_type) == 3


def _build_long_task_intent(knowledge_item, tool_info) -> dict:
    """Build a long task intent marker returned by create_agent."""
    return {
        'intent': 'long_task',
        'knowledge': knowledge_item,
        'tool_info': tool_info,
    }


def _json_len(obj) -> int:
    """Return the byte length of *obj* serialised as compact JSON."""
    return len(json.dumps(obj, ensure_ascii=False, default=str))


def _prune_item_for_llm(item, max_item_chars=MAX_ITEM_CHARS_FOR_LLM, max_value_chars=MAX_VALUE_CHARS_THRESHOLD):
    """Recursively prune *item* so it fits within *max_item_chars*.

    Rules (applied depth-first):
    1. If the whole item serialised is 鈮?max_item_chars, return it unchanged.
    2. For a dict: iterate over keys.  For each value:
       - If the value is a **list** and its JSON size exceeds max_value_chars,
         drop the key entirely (the array is too large).
       - If the value is a **dict** and its JSON size exceeds max_value_chars,
         recurse into it to drop oversized nested arrays.
       - If the value is a **string** that exceeds max_value_chars, truncate it.
       - Otherwise keep it as-is.
    3. For a list: apply the same pruning to every element.
    4. Scalars (str / int / float / bool / None) are truncated when over
       max_value_chars, otherwise kept.

    The function returns a **new** object 鈥?the input is never mutated.
    """
    # Fast path: the item already fits.
    if _json_len([item]) <= max_item_chars:
        return item

    if isinstance(item, dict):
        result: dict[str, Any] = {}
        for key, value in item.items():
            if isinstance(value, list):
                if _json_len(value) > max_value_chars:
                    continue  # drop oversized array
                result[key] = value
            elif isinstance(value, dict):
                if _json_len(value) > max_value_chars:
                    # Recurse to strip oversized arrays from the nested dict.
                    result[key] = _prune_item_for_llm(value, max_item_chars, max_value_chars)
                else:
                    result[key] = value
            elif isinstance(value, str):
                if len(value) > max_value_chars:
                    result[key] = (
                        value[:max_value_chars]
                        + f"... [truncated {len(value) - max_value_chars} chars]"
                    )
                else:
                    result[key] = value
            else:
                result[key] = value
        return result

    if isinstance(item, list):
        return [_prune_item_for_llm(i, max_item_chars, max_value_chars) for i in item]

    # Scalar (string / number / bool / None)
    if isinstance(item, str) and len(item) > max_value_chars:
        return item[:max_value_chars] + f"... [truncated {len(item) - max_value_chars} chars]"
    return item


def _collect_urls_from_value(value):
    urls = []
    if isinstance(value, dict):
        for nested_value in value.values():
            urls.extend(_collect_urls_from_value(nested_value))
    elif isinstance(value, list):
        for item in value:
            urls.extend(_collect_urls_from_value(item))
    elif isinstance(value, str):
        urls.extend(match.rstrip(".,;") for match in _URL_IN_TEXT_RE.findall(value))
    return urls


def _format_markdown_value(value) -> str:
    text = str(value)
    if len(text) > MAX_MARKDOWN_VALUE_CHARS:
        hidden_count = len(text) - MAX_MARKDOWN_VALUE_CHARS
        return f"{text[:MAX_MARKDOWN_VALUE_CHARS]}... [truncated {hidden_count} chars]"
    return text


LARGE_LIST_SUMMARY_MAX_VALUE_CHARS = 5000
SUMMARY_MAX_CHARS = int(os.getenv("GENERAL_AGENT_SUMMARY_MAX_CHARS", "120000"))


def _prune_for_summary(items: list, max_value_chars: int = None) -> list:
    """Return a copy of items pruned so the whole summary payload stays bounded.

    The shallow per-key check was not enough: PEDS items carry huge bags
    (eventDataBag, parentContinuityBag, ...) whose direct values are
    lists, which the old check kept wholesale — the summary LLM call blew
    past its 400k-token context limit.  Reuse the recursive pruner so
    oversized nested arrays are dropped entirely and long strings are
    truncated.
    """
    if max_value_chars is None:
        max_value_chars = LARGE_LIST_SUMMARY_MAX_VALUE_CHARS
    result = []
    for item in items:
        if not isinstance(item, dict):
            result.append(item)
            continue
        result.append(_prune_item_for_llm(item, MAX_ITEM_CHARS_FOR_LLM, max_value_chars))
    return result


def _bounded_summary_items(items: list) -> list:
    """Prune *items* and keep only the leading slice that fits the
    summary character budget — the pool ranks by relevance, so the head
    of the list is the most valuable part."""
    pruned = _prune_for_summary(items)
    kept: list = []
    budget = 0
    for item in pruned:
        try:
            size = len(json.dumps(item, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            size = LARGE_LIST_SUMMARY_MAX_VALUE_CHARS
        if kept and budget + size > SUMMARY_MAX_CHARS:
            break
        kept.append(item)
        budget += size
    return kept


def _large_list_summary_heading(lang: str, summarized: int, total: int) -> str:
    """Heading that reports how many items the summary LLM actually saw
    (the char budget may slice the pool head)."""
    if lang == "en":
        return f"## Results — Summary ({summarized} / {total} items)"
    return f"## 结果摘要 ({summarized} / {total} 项)"


# 瀹氫箟鍙傛暟妯″瀷
class DynamicToolFunction(BaseModel):
    user_id: str = Field(description="user id (provided in the user prompt)")
    query_id: str = Field(description="query id (provided in the user prompt)")
    params: str = Field(
        description=(
            "params. If the original tool params contain api-key, api_key, "
            "apikey, x-api-key, or another API key field, preserve that key "
            "and its value exactly in the generated params."
        )
    )


class DynamicBackendToolFunction(BaseModel):
    user_id: str = Field(description="user id (provided in the user prompt)")
    query_id: str = Field(description="query id (provided in the user prompt)")
    params: Dict[str, Any] | str = Field(
        description=(
            "API request parameters as a JSON object. For push=2 tools, "
            "params may contain path, query, and body; path is appended to "
            "the tool URL after replacing template placeholders from the user request. "
            "If the original tool params contain api-key, api_key, apikey, "
            "x-api-key, or another API key field, preserve that key and its value "
            "exactly in the generated params. "
            "Legacy JSON strings are also accepted."
        )
    )


class _ResponseCollector(BaseCallbackHandler):
    """Wrap a callback handler to collect the full response text as it streams.

    Delegates all attribute access to the real handler (via __getattr__)
    while accumulating on_llm_new_token payloads in `collected_text`.
    """

    def __init__(self, real_handler):
        self._real = real_handler
        self.collected_text = ""

    def __getattr__(self, name):
        # Proxy everything to the wrapped handler — including sync/async
        # methods, attributes like `queue`, `run_inline`, etc.
        if name in ('_real', 'collected_text'):
            raise AttributeError(name)
        return getattr(self._real, name)

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.collected_text += token
        if self._real:
            await self._real.on_llm_new_token(token, **kwargs)


async def _emit_status(callback_handler, message: str):
    """Send a transient status through the callback handler if supported."""
    if callback_handler is None:
        return
    on_status = getattr(callback_handler, "on_status", None)
    if on_status is None:
        return
    try:
        await on_status(message)
    except Exception:
        pass


def _status_callback_for(callback_handler):
    """Create an async status callback suitable for workflow executors."""
    if callback_handler is None:
        return None
    on_status = getattr(callback_handler, "on_status", None)
    if on_status is None:
        return None

    async def emit(message: str):
        try:
            await on_status(message)
        except Exception:
            pass

    return emit


class GeneralAgent(Agent):

    def __init__(self, name, prompt_path, provider, verbose=False):
        """
        The mcp agent is a special agent for using MCPs.
        MCP agent will be disabled if the user does not explicitly set the MCP_FINDER_API_KEY in environment variable.
        """
        super().__init__(name, prompt_path, provider, verbose, None)
        keys = self.get_api_keys()
        self.tools = {
            #"mcp_finder": MCP_finder(keys["mcp_finder"]),
            # add mcp tools here
        }
        self.role = "mcp"
        self.type = "mcp_agent"
        self.memory = Memory(self.load_prompt(prompt_path),
                                recover_last_session=False, # session recovery in handled by the interaction class
                                memory_compression=False,
                                model_provider=provider.get_model_name())
        self.enabled = True
        self.knowledgeTool = {}
        self.logger = Logger("general_agent.log")
        # Multi-turn conversation storage: each entry is
        # {'user': query_text, 'assistant': response_summary}
        self._conversation_turns = []
        # Detected language of the current user query ('zh' or 'en')
        self._lang = 'zh'

    def get_api_keys(self) -> dict:
        """
        Returns the API keys for the tools.
        """
        api_key_mcp_finder = os.getenv("MCP_FINDER_API_KEY")
        if not api_key_mcp_finder or api_key_mcp_finder == "":
            pretty_print("MCP Finder disabled.", color="warning")
            self.enabled = False
        return {
            "mcp_finder": api_key_mcp_finder
        }

    def _sanitize_params_for_llm(self, params_str: str) -> str:
        """Return params JSON with sensitive header values replaced by '****'.

        The real header values stay in tool_info.params for server-side use only.
        This sanitised copy is the only version the LLM ever sees.
        """
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
    
    def set_knowledge_tool(self, knowledge_tool: Dict[str, Any]) -> None:
        """Set the knowledge tool dictionary."""
        self.knowledgeTool = knowledge_tool

    def _flatten_dict(self, d: dict) -> str:
        """Flatten a dict into readable 'key: value' pairs, skipping empty values."""
        parts = []
        for k, v in d.items():
            if v is None or v == "" or v == {} or v == []:
                continue
            if isinstance(v, dict):
                sub = self._flatten_dict(v)
                if sub:
                    parts.append(f"{k}: {sub}")
            elif isinstance(v, list):
                if v and not isinstance(v[0], (dict, list)):
                    parts.append(f"{k}: {', '.join(_format_markdown_value(i) for i in v)}")
            else:
                parts.append(f"{k}: {_format_markdown_value(v)}")
        return " | ".join(parts)

    def _render_list_as_md(self, label: str | None, items: list) -> str:
        """Render every item in a list as readable markdown bullets without JSON dumps."""
        header = f"**{label}** ({len(items)} items total):\n\n" if label else f"({len(items)} items total):\n\n"
        blocks = [header]
        for i, item in enumerate(items, 1):
            if isinstance(item, (dict, list)):
                nested_lines = self._render_markdown_node(item, indent_level=1)
                blocks.append("\n".join([f"- **[{i}]**"] + nested_lines))
            else:
                blocks.append(f"- **[{i}]** {self._format_full_markdown_value(item)}")
        return "\n".join(blocks)

    def _format_full_markdown_value(self, value) -> str:
        return str(value)

    def _is_empty_markdown_value(self, value) -> bool:
        return value is None or value == "" or value == {} or value == []

    def _render_markdown_node(self, value, indent_level: int, label: str | None = None) -> list[str]:
        indent = "  " * indent_level

        if isinstance(value, dict):
            lines = []
            child_indent = indent_level
            if label is not None:
                lines.append(f"{indent}- **{label}**:")
                child_indent += 1

            for key, nested_value in value.items():
                if self._is_empty_markdown_value(nested_value):
                    continue
                lines.extend(self._render_markdown_node(nested_value, child_indent, str(key)))
            return lines

        if isinstance(value, list):
            lines = []
            child_indent = indent_level
            if label is not None:
                lines.append(f"{indent}- **{label}**:")
                child_indent += 1

            item_indent = "  " * child_indent
            for index, item in enumerate(value, 1):
                if self._is_empty_markdown_value(item):
                    continue
                if isinstance(item, (dict, list)):
                    lines.append(f"{item_indent}- **[{index}]**")
                    lines.extend(self._render_markdown_node(item, child_indent + 1))
                else:
                    lines.append(f"{item_indent}- **[{index}]** {self._format_full_markdown_value(item)}")
            return lines

        if label is not None:
            return [f"{indent}- **{label}**: {self._format_full_markdown_value(value)}"]
        return [f"{indent}- {self._format_full_markdown_value(value)}"]


    def _get_markdown_formatting_guide(self) -> str:
        """Return a Markdown formatting guide injected into direct-mode system prompts."""
        return """
## Markdown Formatting Guidelines

You MUST follow these formatting rules to ensure beautiful, readable output:

### 1. Structure & Organization
- Use clear heading hierarchy: # for main title, ## for sections, ### for subsections
- Add blank lines between different content blocks for better readability
- Group related information together

### 2. Links
- **Inline links**: Use `[descriptive text](URL)` format
- Make link text meaningful and descriptive (not just "click here")
- Example: `[OpenAI Documentation](https://platform.openai.com/docs)`

### 3. Images
- Display images using: `![alt text](image_URL)`
- Always provide meaningful alt text
- If multiple images, consider organizing them in a list or grid pattern
- Example: `![Product Screenshot](https://example.com/image.jpg)`

### 4. Lists
- Use `-` for unordered lists (more visually appealing than `*`)
- Use numbered lists `1.` for sequential steps
- Add space after list markers
- Indent sub-items with 2-4 spaces
- **Long list handling**: If the tool result is a JSON array with many items, provide a concise summary with count statistics (e.g., total count, counts per category) instead of enumerating every item.

### 5. Emphasis
- Use **bold** for important terms: `**text**`
- Use *italic* for emphasis: `*text*`
- Use `code` for technical terms: `` `code` ``

### 6. Tables (for structured data)
- Use tables for comparing information

### 7. CRITICAL Rules
- **Long list handling**: If the tool result is a JSON list with many items, summarize with count statistics (total, per-category counts) rather than listing each item.
- **No source sections**: Do NOT add "Sources:", "References:", or "Resources:" sections at the end
- **Inline links only**: Integrate links naturally within the content, not as a separate list at the bottom
- **No code block wrapper**: Output DIRECT Markdown content, do NOT wrap your entire response in a code block

**Remember**: Your goal is to make the content scannable, visually appealing, and easy to read.
"""
    def expand_prompt(self, prompt):
        """
        Expands the prompt with the tools available.
        """
        tools_str = self.get_tools_description()
        prompt += f"""
        You can use the following tools and MCPs:
        {tools_str}
        """
        return prompt

    def is_query_and_body_empty(self) -> bool:
        """Return True if path, query, and body in tool_info.params are empty."""
        _, tool_info = self.knowledgeTool
        if not tool_info:
            return True

        try:
            params_data = _coerce_json_object(tool_info.params, "tool_info.params")
            return _is_path_query_body_empty(params_data)
        except ValueError:
            return False

    def _get_fixed_system_prefix(self) -> str:
        """Return the stable system prompt prefix that is safe to retain across
        multi-turn conversations. Contains role description and formatting
        rules only — NO tool-specific API templates."""
        knowledge_item, _ = self.knowledgeTool
        context = ""
        if knowledge_item and hasattr(knowledge_item, 'answer') and knowledge_item.answer:
            context = f"""

        Context from knowledge base:
        {knowledge_item.answer}

        Use this context to better understand the task and provide more accurate responses.
        """

        return f"""
        You are an intelligent assistant capable of deciding when and how to use APIs to complete tasks.
        {context}

        Do not fabricate tool results. Do not assume tool behavior beyond the provided output.

        Do NOT reveal any API keys, tokens, header values, or authentication credentials in your response.

        ## Markdown Formatting Requirements

        When generating your response based on the tool's output, you MUST format it beautifully using Markdown:

        ### Essential Formatting Rules:

        1. **Structure**: Use clear heading hierarchy (## for main sections, ### for subsections)
        2. **Links**: Convert ALL URLs to descriptive links: `[meaningful text](URL)`
        3. **Images**: Display images using: `![description](image_URL)`
        4. **Lists**: Use `-` for bullet points, `1.` for numbered lists
        5. **Emphasis**: Use **bold** for key terms, *italic* for emphasis, `code` for technical terms
        6. **Tables**: Use tables for structured data comparison
        7. **Spacing**: Add blank lines between content blocks for readability
        8. **Code blocks**: Use fenced code blocks with language specification when showing code
        9. **Long list handling**: If the tool result is a JSON array with many items, provide a concise summary with count statistics (e.g., total count, counts per category) instead of enumerating every item.

        ### Response Structure Template:

        Your response should follow this structure (but output DIRECTLY, not in a code block):

        ## [Main Topic]

        [Brief summary of what the tool returned]

        ### Key Information
        - Important point 1 (with details)
        - Important point 2 (with details)
        - ... (if the tool result is a long JSON list, show a summary with count statistics instead)

        ### Details
        [Organized detailed content — if the tool result is a long JSON list, show a summary with count statistics]

        [Display images inline where relevant]
        ![Image Description](image_URL)

        **CRITICAL OUTPUT FORMAT**:
        - Output your response as DIRECT Markdown content
        - Do NOT wrap your entire response in a code block
        - Do NOT start with ```markdown or ```
        - Start directly with Markdown formatting (e.g., ## Title or plain text)
        - Only use code blocks for actual code snippets within your content, not for the entire response

        **CRITICAL CONTENT RULES**:
        - If the tool result is a long JSON list, provide a concise summary with count statistics instead of listing every item
        - Do NOT add a "Resources", "Sources", or "References" section at the end of your response
        - Do NOT create a separate list of links at the bottom
        - Integrate all links naturally within the content itself

        **IMPORTANT**: Make your response visually appealing, easy to scan, and professionally formatted. Transform raw data into a beautiful, user-friendly presentation while ensuring ALL content from the tool result is displayed.

        ## Language Rule (CRITICAL — NEVER VIOLATE)

        The user's question determines the response language. This is an ABSOLUTE rule:

        - If the user asks in **Chinese (中文)**, you MUST respond entirely in Chinese.
        - If the user asks in **English**, you MUST respond entirely in English.

        Do NOT mix languages. Do NOT answer an English question with Chinese text
        or vice versa. Match the user's language exactly throughout your entire response.
        """
    def _loop_system_guidance(self) -> str:
        """Tool-usage guidance appended to the ReAct loop system prompt."""
        return f"""

## Tool Usage

- You have tools built from the user's knowledge base, plus `search_my_knowledge`
  to find more. Use them to complete the user's task.
- Work step by step: think about what is needed, call the right tool, observe
  the result, then decide the next step or write the final answer.
- You may call several tools in sequence and combine their results.
- Never fabricate tool results. If a tool fails, try another approach or
  explain the failure honestly.

## Adaptive Search Discipline

- The same search tool may be called multiple times with adjusted parameters.
  When a search returns 0 results, do NOT immediately drop a constraint — a
  zero hit usually means the wording does not match how the domain's
  literature phrases the technology, not that the technology is absent.
  First retry at the SAME tightness level using carrier terms / synonyms
  from the concept keyword bank (the same technology is phrased very
  differently from literal translations), then word-ending wildcard
  variants; only when same-level substitutions still return 0 should you
  loosen the query (drop the weakest constraint or use a looser ladder
  variant). If it returns far too many noisy results, tighten it (add a
  justified constraint). Keep adjusting until the result count and
  relevance are reasonable (suggested: up to 6 attempts per tool).
- Before switching tools, exhaust the same tool's options first: try every
  ladder variant (including the carrier-term variants) down to the loosest
  single-concept query, and vary the keywords themselves (different
  synonyms, spellings, inflections, or wildcards) when zero results
  persist.
- Only after those attempts still give poor results, consider another tool —
  and judge whether that tool actually fits the user's problem.
- If no available tool fits the problem, honestly report the search failure and
  its reason to the user. Do not keep chaining unsuitable tools.
- If a search observation shows relevance-ranked results (each line ends
  with a relevance score), your final answer must first list the most
  relevant top items (at most {RELEVANT_TOP_N}) — identifier, title, and
  one sentence on why each fits the user's question — before the overall
  summary.

## Search Result Delivery Format (MANDATORY — applies unconditionally)

Whenever a search tool was called and returned candidates, structure the final answer as follows (the user wants a usable identifier, not just a list):

1. Open with the top matching result conclusion, formatted exactly as:
   `**Top matching result: {{number}}** — {{title}} ({{one-sentence reason}})`
   - When candidates are relevance-ranked, take the first one; otherwise pick the most relevant per the user's question.
   - The identifier must be complete and copyable — never omit or paraphrase it.
2. Then list the top {RELEVANT_TOP_N} candidates (identifier, title, one-sentence relevance reason).
3. Close by telling the user the full results are available in the results panel (JSON/CSV/Excel download).

This structure overrides other formatting preferences; only when the search returned no candidates should you honestly say no matching results were found.
"""


    def generate_fixed_system_prompt(self) -> str:
        """Generate system prompt for fixed (no-parameter) tools."""
        knowledge_item, tool_info = self.knowledgeTool
        self.logger.info(f"knowledge item:{knowledge_item} - tool:{tool_info}")

        # 鑾峰彇缇庡浗涓滈儴鏃跺尯鐨勫綋鍓嶆椂闂?
        eastern_tz = ZoneInfo("America/New_York")
        eastern_time = datetime.now(eastern_tz)

        # 鏍煎紡鍖栦负瀛楃涓诧紙鍖呭惈鏃跺尯淇℃伅锛?
        time_str = eastern_time.strftime("%Y-%m-%d %H:%M:%S %Z")

        # 鑾峰彇knowledge item鐨刟nswer浣滀负涓婁笅鏂?
        context = ""
        if knowledge_item and hasattr(knowledge_item, 'answer') and knowledge_item.answer:
            context = f"""

        Context from knowledge base:
        {knowledge_item.answer}

        Use this context to better understand the task and provide more accurate responses.
        """

        if not tool_info:
            system_prompt = f"""

            You are an intelligent API-enabled assistant. Current time is {time_str}.
            {context}

            If no relevant knowledge is available to complete the user's task, clearly inform the user that no matching knowledge was found and suggest checking the community for shared knowledge or tools that may solve the problem.

            If a tool response indicates that the user is not authenticated, or returns a login page, tell the user clearly that the tool requires login: e.g. "该工具需要登录后才能使用，请先完成登录再重试". Do NOT output internal markers or technical tags.

            """
            return system_prompt

        tool_title = tool_info.title
        tool_description = None
        if tool_info.description:
            tool_description = tool_info.description
        else:
            tool_description = tool_title

        # 瑙ｆ瀽宸ュ叿鍙傛暟淇℃伅
        tool_params_info = ""
        if tool_info.params:
            try:
                params_data = json.loads(tool_info.params)
                if isinstance(params_data, dict):
                    tool_params_info = "Tool parameters: user_id, query_id\n"
                    for param_name, param_type in params_data.items():
                        if param_name in ("method", "content-type", "header"):
                            continue
                        tool_params_info += f"  - {param_name} ({param_type})\n"
                else:
                    tool_params_info = f"Tool parameters: {tool_info.params}"
            except json.JSONDecodeError:
                tool_params_info = f"Tool parameters: {tool_info.params}"

        system_prompt = f"""
        You are an intelligent assistant capable of deciding when and how to use APIs to complete tasks.
        {context}

        Based on the user's request and the available context, decide whether invoking a tool is necessary.

        If a tool is required, use the following tool:

        Tool: {tool_title}
        Purpose: {tool_description}
        Input parameters: {tool_params_info}

        Execute the tool with the appropriate parameters and generate the final response strictly based on the tool's output.

        If the task can be completed without invoking the tool, respond directly to the user without calling any tool.

        Do not fabricate tool results. Do not assume tool behavior beyond the provided output.

        Do not return internal identifiers or tool parameters in your response.
        Do NOT reveal any API keys, tokens, header values, or authentication credentials in your response.
        """
        # return self.expand_prompt(system_prompt)
        return system_prompt

    def generate_template_system_prompt(self) -> str:
        """Generate system prompt for template-parameter tools (LLM fills params)."""
        knowledge_item, tool_info = self.knowledgeTool
        self.logger.info(f"knowledge item:{knowledge_item} - tool:{tool_info}")

        # 鑾峰彇缇庡浗涓滈儴鏃跺尯鐨勫綋鍓嶆椂闂?
        eastern_tz = ZoneInfo("America/New_York")
        eastern_time = datetime.now(eastern_tz)

        # 鏍煎紡鍖栦负瀛楃涓诧紙鍖呭惈鏃跺尯淇℃伅锛?
        time_str = eastern_time.strftime("%Y-%m-%d %H:%M:%S %Z")

        # 鑾峰彇knowledge item鐨刟nswer浣滀负涓婁笅鏂?
        context = ""
        if knowledge_item and hasattr(knowledge_item, 'answer') and knowledge_item.answer:
            context = f"""

        Context from knowledge base:
        {knowledge_item.answer}

        Use this context to better understand the task and provide more accurate responses.
        """

        if not tool_info:
            system_prompt = f"""

            You are an intelligent API-enabled assistant. Current time is {time_str}.
            {context}

            If no relevant knowledge is available to complete the user's task, clearly inform the user that no matching knowledge was found and suggest checking the community for shared knowledge or tools that may solve the problem.

            If a tool response indicates that the user is not authenticated, or returns a login page, tell the user clearly that the tool requires login: e.g. "该工具需要登录后才能使用，请先完成登录再重试". Do NOT output internal markers or technical tags.

            """
            return system_prompt

        tool_title = tool_info.title
        tool_description = None
        if tool_info.description:
            tool_description = tool_info.description
        else:
            tool_description = tool_title

        tool_params_info = "tool requires three parameters:user id - query id - params\n"
        if tool_info.push == 2:
            params_call_instruction = """
        3. params: A JSON object containing ONLY the API request parameters (template below).
           Pass params as a structured object in the tool call, not as a JSON-encoded string.
           Do NOT wrap the whole params object in quotes.
            """
        else:
            params_call_instruction = """
        3. params: A valid JSON string containing ONLY the API request parameters (template below)
            """

        path_field_semantics = ""
        path_replacement_rules = ""
        if tool_info.push == 2:
            path_field_semantics = """
           - path contains a relative URL path that will be appended to the configured tool URL
            """
            path_replacement_rules = """
           - For path placeholders like `{applicationNumberText}`, replace the placeholder with the exact value from the user's request or the knowledge context
           - Preserve the rest of the path string, including slashes and fixed suffixes such as `/document`
           - Do not leave unresolved `{...}` placeholders in path. If the required value is missing, ask the user for it instead of calling the tool
           - path must remain a relative URL path, never an absolute URL
            """

        system_prompt = f"""
        You are an intelligent assistant capable of deciding when and how to use APIs to complete tasks.
        {context}

        Based on the user's request and the available context, decide whether invoking a tool is necessary.

        If a tool is required, use the following tool:

        Tool: {tool_title}
        Purpose: {tool_description}
        Input parameters: {tool_params_info}


        IMPORTANT: The tool takes THREE separate parameters:
        1. user_id: The user identifier (provided in the user prompt, DO NOT include in params)
        2. query_id: The query identifier (provided in the user prompt, DO NOT include in params)
        {params_call_instruction}

        The third parameter "params" template: {self._sanitize_params_for_llm(tool_info.params)}

        Your task is to analyze the user's input and modify the third parameter "params" template according to the user's specific requirements. Generate the COMPLETE params JSON with ALL fields from the template — keep unchanged fields as-is, and modify only the values that need to match the user's request.

        You MUST follow all rules below without exception:

        1. Modify the JSON to match the user's intent exactly.
           - Replace values the user wants to change
           - Remove query conditions (AND / OR clauses) that are NOT relevant to the user's request
             Example: if the template query is `assignee:"Samsung" AND assignor:"Seiko Epson"`
             and the user only asks for "Apple Inc." as assignee, simplify it to:
             `assignee:"Apple Inc."` (remove the unrelated assignor condition entirely)
           - DO NOT add new fields or query conditions the user did not ask for
           - DO NOT change the JSON structure or top-level nesting
           - CRITICAL: DO NOT include user_id or query_id in the params JSON - these are separate parameters

        2. Field semantics:
           - method MUST remain unchanged
           {path_field_semantics}
           - query contains URL query parameters
           - header contains HTTP headers
           - body contains the HTTP request body

        3. Value replacement rules:
           - Replace a value only if the user query clearly maps to the meaning of an existing field
           - If the user query does not mention or imply a field, keep its original value unchanged (but REMOVE it if it appears as an AND/OR clause in a query string that would make the search too restrictive — see rule 1)
           - When the user asks to search by ONE criterion only (e.g., assignee), strip away unrelated AND clauses so the query returns results instead of nothing
           - If the original tool params contain api-key, api_key, apikey, x-api-key, or another API key field, preserve that key and its value exactly in the generated params
           - Do NOT infer or invent information not explicitly expressed by the user
           - DO NOT extract or infer user_id or query_id from the user's request into the params JSON
           {path_replacement_rules}

        4. Tool calling rules:
           - You MUST call the tool with the three required arguments: user_id, query_id, and params
           - For the params argument: provide the COMPLETE JSON from the template above, with values modified per the user's request. Include ALL top-level fields (method, Content-Type, query, header, body) even if unchanged.
           - DO NOT put user_id or query_id inside the params JSON — they are separate tool arguments

        5. JSON FORMAT RULES — params MUST be valid JSON (CRITICAL):
           - Every key-value pair MUST be separated by a comma (except the last pair in each object)
           - Trailing commas (after the last value in an object/array) are FORBIDDEN
           - All string values containing double quotes MUST escape them with backslash: \\"
           - All string values containing backslashes MUST escape them: \\\\
           - All keys and string values MUST use double quotes ("), NEVER single quotes (')
           - Every opening brace/bracket MUST have a matching closing brace/bracket
           - BEFORE outputting, mentally verify: "Is this JSON parseable by a strict parser?"

        Execute the tool with the appropriate parameters and generate the final response strictly based on the tool's output.

        If the task can be completed without invoking the tool, respond directly to the user without calling any tool.

        Do not fabricate tool results. Do not assume tool behavior beyond the provided output.

        Do not return tool parameters, such as the user id and query id.
        Do NOT reveal any API keys, tokens, header values, or authentication credentials in your response.

        **CRITICAL**: Follow the Markdown formatting requirements from the system instructions at the start of this conversation. Do NOT wrap your response in a code block.
        """

        return system_prompt

    def generate_system_prompt(self, tool_data: str = "") -> str:
        """Select and return the appropriate system prompt based on push mode."""
        _, tool_info = self.knowledgeTool

        # 鏍规嵁tool_info.push鐨勫€奸€夋嫨涓嶅悓绯荤粺鎻愮ず璇?
        if not tool_info:
            return self.generate_fixed_system_prompt()

        if tool_info.push == 1:
            if tool_data and tool_data.strip():  # 鍒ゆ柇tool_data鏄惁闈炵┖
                return self.generate_frontend_tool_direct_system_prompt(tool_data)
            else:
                return self.generate_template_system_prompt()
        elif tool_info.push == 2:
            if self.is_query_and_body_empty():
                return self.generate_backend_tool_direct_system_prompt()
            else:
                return self.generate_template_system_prompt()
        elif tool_info.push == 3:
            return self.generate_frontend_tool_direct_system_prompt(tool_data)
        else:
            # 榛樿鎯呭喌涓嬪浐瀹氱殑绯荤粺鎻愮ず璇?
            return self.generate_template_system_prompt()



    def generate_user_prompt(self, prompt, user_id=None, query_id=None) -> str:
        # user_id / query_id are included so the LLM schema sees them as
        # required parameters and generates the full API template. The tool
        # wrapper overrides whatever the LLM passes with self._last_*.
        uid = str(user_id) if user_id else ""
        qid = str(query_id) if query_id else ""
        user_prompt = f"""
        {prompt},
        user id is {uid},
        query id is {qid},
        """
        self.logger.info(f"user prompt:{user_prompt}")
        return user_prompt

    def generate_frontend_tool_direct_system_prompt(self, tool_data: str) -> str:
        """Generate system prompt from pre-fetched tool_data without making an HTTP request."""
        # self.logger.info(f"generate_frontend_tool_direct_system_prompt - tool_data: {tool_data}")

        try:
            # 鑾峰彇knowledge item鐨刟nswer浣滀负涓婁笅鏂?
            knowledge_item, _ = self.knowledgeTool
            context = ""
            if knowledge_item and hasattr(knowledge_item, 'answer') and knowledge_item.answer:
                context = f"""

            Context from knowledge base:
            {knowledge_item.answer}

            Use this context to better understand the task and provide more accurate responses.
            """

            # 瑙ｆ瀽 tool_data 鍐呭
            if "text/html" in tool_data:
                # 濡傛灉鏄?HTML 鍐呭锛屼娇鐢ㄥ叕鐢ㄦ柟娉曟竻鐞嗘枃鏈?
                result_str = clean_html_text(tool_data)
            else:
                # 灏濊瘯灏?tool_data 瑙ｆ瀽涓?JSON
                try:
                    result_data = json.loads(tool_data)
                    # 濡傛灉缁撴灉鏄竴涓?list锛屾彁鍙?raw_items 浠ヤ緵鍚庣画鎵瑰鐞?
                    if isinstance(result_data, list) and result_data:
                        raw_items = result_data
                    elif isinstance(result_data, dict):
                        raw_items = next(
                            (v for v in result_data.values() if isinstance(v, list) and v),
                            None
                        )
                    else:
                        raw_items = None

                    if raw_items:
                        self._pending_raw_items = raw_items
                        result_str = (
                            f"The query returned {len(raw_items)} items. "
                            f"Please write a brief 2鈥? sentence summary of what was found. "
                            f"The complete list will be analyzed and displayed item by item automatically 鈥?"
                            f"do NOT enumerate the items yourself."
                        )
                    elif isinstance(result_data, dict):
                        # dict 鍖呰涓哄崟鍏冪礌鍒楄〃锛岃蛋 Phase 2 蹇犲疄杈撳嚭
                        pruned = _prune_item_for_llm(result_data)
                        self._pending_raw_items = [pruned]
                        result_str = (
                            "The query returned structured data. "
                            "Please write a brief 1-2 sentence summary of what was found. "
                            "The complete data will be displayed automatically - "
                            "do NOT enumerate the fields yourself."
                        )
                    else:
                        result_str = json.dumps(result_data, ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    # 濡傛灉涓嶆槸鏈夋晥鐨?JSON锛屽垯鐩存帴浣跨敤鍘熷鍐呭
                    result_str = tool_data

            # 鑾峰彇鏍煎紡鍖栨寚鍗?
            formatting_guide = self._get_markdown_formatting_guide()

            # 鏋勯€犵郴缁熸彁绀鸿瘝
            system_prompt = f"""
Act as a self-contained intelligent assistant. Follow these instructions strictly:
{context}

## Core Instructions

1.  **Core Principle:** You must perform tasks and generate answers using **only** the data, text, or context that I provide to you within this chat.
2.  **No External Access:** Do not attempt to invoke or use any internal or external tools (such as search functions, code interpreters, calculators, or knowledge retrieval from your base training data) to complete the task.
3.  **Direct Processing:** Analyze, reason, and respond directly based on the provided input. If the necessary information is not contained in my messages, state that clearly instead of making assumptions.
4.  **Privacy Protection:** Do NOT include or output any `user_id`, `query_id`, or similar internal identifiers in your response. These are system metadata and should never appear in user-facing output.
5.  **Long list handling:** If the input data is a JSON array with many items, provide a concise summary with count statistics (e.g., total count, counts per category) instead of enumerating every item.
6.  **No Source Sections:** Do NOT add a "Sources", "References", or "Resources" section at the end of your response. Do NOT create a separate list of links at the bottom.

{formatting_guide}

## Input Data

{result_str}

## Your Task

Generate a beautiful, well-formatted Markdown response based on the above data. Follow ALL the formatting guidelines provided above. Make your response:
- Visually appealing with proper structure
- Easy to scan with clear headings
- Rich with properly formatted links and images (integrated naturally within content)
- Professional and polished
- If the data is a long JSON list, summarize with count statistics rather than listing every item

**CRITICAL OUTPUT FORMAT**:
- Output your response as DIRECT Markdown content
- Do NOT wrap your entire response in a code block
- Do NOT start with ```markdown or ```
- Start directly with Markdown formatting (e.g., ## Title or plain text)
- Only use code blocks for actual code snippets within your content, not for the entire response

**CRITICAL CONTENT RULES**:
- If the Input Data is a long JSON list, provide a summary with count statistics instead of enumerating every item
- Do NOT add a separate "Sources" or "References" section at the end
- Integrate all links naturally within the content

Begin your response now:
            """
            return system_prompt

        except Exception as e:
            self.logger.error(f"Failed to generate frontend tool direct system prompt: {str(e)}")
            return self.generate_template_system_prompt()

    def generate_backend_tool_direct_system_prompt(self) -> str:
        """Make the HTTP request from tool_info, embed the result in the system prompt, and return it."""
        # 鑾峰彇宸ュ叿淇℃伅
        knowledge_item, tool_info = self.knowledgeTool
        self.logger.info(f"generate_tool_direct_system_prompt - tool:{tool_info}")

        try:
            # 鑾峰彇knowledge item鐨刟nswer浣滀负涓婁笅鏂?
            context = ""
            if knowledge_item and hasattr(knowledge_item, 'answer') and knowledge_item.answer:
                context = f"""

            Context from knowledge base:
            {knowledge_item.answer}

            Use this context to better understand the task and provide more accurate responses.
            """

            # 浠?tool_info 涓彁鍙?URL 鍜屽弬鏁版ā鏉?
            url = tool_info.url
            params_data = _coerce_json_object(tool_info.params, "tool_info.params")
            url = _append_path_to_url(url, params_data.get("path", ""))

            # 鑾峰彇 HTTP 鏂规硶鍜?Content-Type
            method = params_data.get("method", "GET").upper()
            content_type = params_data.get("Content-Type", "application/json")

            # 鍑嗗璇锋眰澶?
            headers = {
                "Content-Type": content_type
            }

            # params_data
            user_headers = params_data.get("header", {})
            if isinstance(user_headers, dict):
                headers.update(user_headers)

            # 娣诲姞鏃堕棿鎴冲弬鏁扮粫杩?CDN 缂撳瓨
            cache_bust_params = {"_t": str(int(time.time() * 1000))}

            # 鍙戣捣 HTTP 璇锋眰
            if method not in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
                raise ValueError(f"Unsupported HTTP method: {method}")
            request_kwargs = {
                "params": cache_bust_params,
                "headers": headers,
            }
            if method in {"POST", "PUT", "PATCH"}:
                request_kwargs["json"] = {}
            response = outbound_http.request(method, url, purpose="backend_tool_direct", **request_kwargs)

            # 澶勭悊鍝嶅簲缁撴灉
            if response.status_code == 200:

                content_type = response.headers.get("Content-Type", "").lower()

                if "text/html" in content_type:
                    # 浣跨敤BeautifulSoup绉婚櫎HTML鏍囩
                    result_str = BeautifulSoup(response.content, "html.parser").get_text()
                elif "application/xml" in content_type or "text/xml" in content_type:
                    # 澶勭悊XML鏍煎紡鍝嶅簲
                    try:
                        # 浣跨敤BeautifulSoup瑙ｆ瀽XML骞舵彁鍙栨枃鏈唴瀹?
                        soup = BeautifulSoup(response.content, "xml")
                        # 绉婚櫎XML鏍囩锛屽彧淇濈暀鏂囨湰鍐呭
                        result_str = soup.get_text()
                        # 濡傛灉XML瑙ｆ瀽澶辫触鎴栧唴瀹逛负绌猴紝浣跨敤鍘熷鍐呭
                        if not result_str.strip():
                            result_str = response.text
                    except Exception as xml_e:
                        self.logger.warning(f"XML parsing failed: {str(xml_e)}, using raw content")
                        result_str = response.text
                else:
                    try:
                        result_data = response.json() if response.content else {}
                        if isinstance(result_data, list) and result_data:
                            raw_items = result_data
                        elif isinstance(result_data, dict):
                            raw_items = next(
                                (v for v in result_data.values() if isinstance(v, list) and v),
                                None
                            )
                        else:
                            raw_items = None

                        if raw_items:
                            self._pending_raw_items = raw_items
                            result_str = (
                                f"The query returned {len(raw_items)} items. "
                                f"Please write a brief 2鈥? sentence summary of what was found. "
                                f"The complete list will be analyzed and displayed item by item automatically 鈥?"
                                f"do NOT enumerate the items yourself."
                            )
                        elif isinstance(result_data, dict):
                            # dict 鍖呰涓哄崟鍏冪礌鍒楄〃锛岃蛋 Phase 2 蹇犲疄杈撳嚭
                            pruned = _prune_item_for_llm(result_data)
                            self._pending_raw_items = [pruned]
                            result_str = (
                                "The query returned structured data. "
                                "Please write a brief 1-2 sentence summary of what was found. "
                                "The complete data will be displayed automatically - "
                                "do NOT enumerate the fields yourself."
                            )
                        else:
                            result_str = json.dumps(result_data, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError:
                        # 濡傛灉JSON瑙ｆ瀽澶辫触锛屼娇鐢ㄥ師濮嬪搷搴斿唴瀹?
                        result_str = response.text if response.text else "Empty response"
            else:
                result_str = f"Request failed, status code: {response.status_code}"

            # 鑾峰彇鏍煎紡鍖栨寚鍗?
            formatting_guide = self._get_markdown_formatting_guide()

            # 鏋勯€犵郴缁熸彁绀鸿瘝
            system_prompt = f"""
Act as a self-contained intelligent assistant. Follow these instructions strictly:
{context}

## Core Instructions

1.  **Core Principle:** You must perform tasks and generate answers using **only** the data, text, or context that I provide to you within this chat.
2.  **No External Access:** Do not attempt to invoke or use any internal or external tools (such as search functions, code interpreters, calculators, or knowledge retrieval from your base training data) to complete the task.
3.  **Direct Processing:** Analyze, reason, and respond directly based on the provided input. If the necessary information is not contained in my messages, state that clearly instead of making assumptions.
4.  **Privacy Protection:** Do NOT include or output any `user_id`, `query_id`, or similar internal identifiers in your response. These are system metadata and should never appear in user-facing output.
5.  **Long list handling:** If the input data is a JSON array with many items, provide a concise summary with count statistics (e.g., total count, counts per category) instead of enumerating every item.
6.  **No Source Sections:** Do NOT add a "Sources", "References", or "Resources" section at the end of your response. Do NOT create a separate list of links at the bottom.

{formatting_guide}

## Input Data

{result_str}

## Your Task

Generate a beautiful, well-formatted Markdown response based on the above data. Follow ALL the formatting guidelines provided above. Make your response:
- Visually appealing with proper structure
- Easy to scan with clear headings
- Rich with properly formatted links and images (integrated naturally within content)
- Professional and polished
- If the data is a long JSON list, summarize with count statistics rather than listing every item

**CRITICAL OUTPUT FORMAT**:
- Output your response as DIRECT Markdown content
- Do NOT wrap your entire response in a code block
- Do NOT start with ```markdown or ```
- Start directly with Markdown formatting (e.g., ## Title or plain text)
- Only use code blocks for actual code snippets within your content, not for the entire response

**CRITICAL CONTENT RULES**:
- If the Input Data is a long JSON list, provide a summary with count statistics instead of enumerating every item
- Do NOT add a separate "Sources" or "References" section at the end
- Integrate all links naturally within the content

Begin your response now:
            """
            return system_prompt

        except Exception as e:
            self.logger.error(f"Failed to generate tool direct system prompt: {str(e)}")
            return self.generate_system_prompt(result_str)

    def get_dynamic_tool_for(self, knowledge_item, tool_info):
        """Build the LangChain StructuredTool for one knowledge/tool pair.

        Extracted from get_dynamic_tools so the ReAct loop can build tools
        for arbitrary knowledge items (top-N recall + search results),
        not just the single self.knowledgeTool pairing.
        """
        if tool_info is None:
            return None
        # -- closures moved verbatim from get_dynamic_tools --
        def dynamic_frontend_tool_function(user_id: str, query_id: str, params: str):
            # Always use stored values -- ignore LLM-provided IDs
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
            # Always use stored values -- ignore LLM-provided IDs
            user_id = self._last_user_id
            query_id = self._last_query_id
            self.logger.info(
                f"dynamic_backend_tool_function user_id={user_id} query_id={query_id}"
            )
            tool_result = execute_backend_tool_request(tool_info, params)
            # Capture the total hit count for the loop's adaptive search
            # (e.g. 0 hits → loosen the query, 8M hits → tighten it).
            _result_data = tool_result.get("data")
            if isinstance(_result_data, dict):
                _count = _result_data.get("count")
                if isinstance(_count, (int, float)):
                    self._last_search_total = int(_count)
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

        # -- name / schema / push selection (same as before) --
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
        description = (
            tool_info.description if tool_info.description else "Dynamic knowledge tool"
        )
        usage_guide = (getattr(knowledge_item, "answer", "") or "").strip()
        if usage_guide:
            # Restore the usage guide the pre-loop flow injected as
            # "Context from knowledge base".  It explains how the tool's
            # params map (e.g. which value goes into path vs query) — the
            # loop's bare tool_info.description let the LLM pass the USPTO
            # application number as a query param, sending the literal
            # {applicationNumberText} path template to the gateway (403).
            description = f"{description}\n\nUsage guide: {usage_guide[:1000]}"[:1600]
        return StructuredTool.from_function(
            func=tool_func,
            name=cleaned_tool_name,
            description=description,
            args_schema=args_schema,
        )

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


    async def get_tools(self, tool_data: str = "") -> list:
        """Select and return the appropriate LangChain tool list based on push mode."""
        _, tool_info = self.knowledgeTool
        if not tool_info:
            return []

        tools = []
        # 鏍规嵁tool_info.push鐨勫€奸€夋嫨涓嶅悓绯荤粺鎻愮ず璇?
        if tool_info.push == 1:
            # If tool_data is already provided, the system prompt contains the
            # pre-fetched result. Do NOT give the LLM a LangChain tool 鈥?
            # it would call the tool, receive a ToolMessage, and ignore the
            # pre-formatted list we placed in the system prompt.
            if tool_data and tool_data.strip():
                return tools
            if self.is_query_and_body_empty():
                return tools
            else:
                return await self.get_dynamic_tools()
        elif tool_info.push == 2:
            if self.is_query_and_body_empty():
                return tools
            else:
                return await self.get_dynamic_tools()
        elif tool_info.push == 3:
            return tools
        else:
            # 榛樿鎯呭喌涓嬪浐瀹氱殑绯荤粺鎻愮ず璇?
            return await self.get_dynamic_tools()

    async def process(self, user_id, prompt, query_id, speech_module, push_filter=None) -> str | tuple[str, str]:
        if not self.enabled:
            return "general Agent is disabled."
        self._last_user_prompt = prompt
        self._last_query_id = query_id
        self._lang = self._detect_lang(prompt)
        conv_history = self.memory.get()
        self.knowledgeTool = await select_knowledge_tool_with_llm(
            user_id,
            prompt,
            self.llm.complete_json,
            push_filter=push_filter,
            conversation_history=conv_history,
        )
        # user_prompt = self.expand_prompt(prompt)
        user_prompt = self.generate_user_prompt(prompt, user_id, query_id)
        system_prompt = self.generate_system_prompt()
        # Append language rule for CLI/terminal path (create_agent uses _get_fixed_system_prefix instead)
        lang_rule = self._get_language_rule()
        system_prompt += lang_rule
        prior = self.memory.get()
        if len(prior) > 6:
            prior = prior[:1] + prior[-1:]
        self.memory.reset(prior)
        self.memory.push('user', user_prompt)
        self.memory.push('system', system_prompt)

        self.logger.info(f"memory.get():{self.memory.get()}")
        self.tools = await self.get_tools()
        working = True
        while working == True:
            self.logger.info(f"tools:{self.tools}")
            animate_thinking("Thinking...", color="status")
            answer, reasoning = await self.llm_request()
            # exec_success, _ = self.execute_modules(answer)
            # answer = self.remove_blocks(answer)
            self.last_answer = answer
            self.status_message = "Ready"
            if len(self.blocks_result) == 0:
                working = False
        return answer, reasoning

    @staticmethod
    def _detect_lang(text: str) -> str:
        """Return 'zh' or 'en' based on CJK character ratio in *text*."""
        if not text:
            return 'zh'
        cjk = sum(1 for c in text if '一' <= c <= '鿿')
        alpha = sum(1 for c in text if c.isalpha())
        total = cjk + alpha
        if total == 0:
            return 'zh'
        return 'zh' if cjk / max(total, 1) > 0.15 else 'en'

    def _get_language_rule(self) -> str:
        """Return a language-enforcement rule based on the detected user language.

        The rule is appended to system prompts to ensure the LLM responds
        in the same language the user wrote their question in.
        """
        lang = getattr(self, '_lang', 'zh')
        if lang == 'en':
            return (
                "\n\n## Language Rule (CRITICAL — NEVER VIOLATE)\n\n"
                "The user asked their question in English. "
                "You MUST respond entirely in English. "
                "Do NOT use any Chinese characters, phrases, or mixed-language output. "
                "Every heading, paragraph, list item, and label must be in English.\n"
            )
        else:
            return (
                "\n\n## 语言规则（极其重要 — 绝对禁止违反）\n\n"
                "用户使用中文提问，你必须全程使用中文回答。"
                "禁止在回答中出现任何英文单词、短语或中英混杂的输出。"
                "所有标题、段落、列表项和标签都必须使用中文。\n"
            )

    async def create_agent(self, user_id, prompt, query_id, tool_data, callback_handler, push_filter=None, conversation_history=None):
        """Build the ReAct tool set and run the loop for one user query.

        Long-task tool calls surface as the same {'intent': 'long_task'}
        marker core.py already handles; every other outcome streams through
        the callback handler inside the loop and returns None.

        ``conversation_history`` (frontend-sent, optional) is injected into
        the system prompt as reference-only "Previous conversation" context
        so follow-up questions always carry prior turns (需求 2).
        """
        # -- per-request state reset (agents are pooled and reused) --
        self._last_user_prompt = prompt
        self._conversation_history = conversation_history or []
        self._last_query_id = query_id
        self._last_user_id = user_id
        self._callback_handler = callback_handler
        self._pending_raw_items = None
        self._workflow_result = None
        self._react_loop_ran = False
        self._search_rewrite = None   # deterministic q rewrite cache, per request
        self._search_rewrite_cn = None  # Baiten CN rewrite, per request
        self._patent_tool_source = "auto"  # uspto/cn/dual — built-in search tool
        self._search_interpretation = None  # architecture-level interpretation, per request
        self._request_started = time.monotonic()  # whole-request timer (agent_elapsed origin)
        self._grounded_done = False  # post-retrieval grounded synthesis, once per request
        self._grounded_interpretation = None  # data-grounded interpretation (players/lines)
        self._grounded_cpc = None  # supplementary CPC codes from the grounded synthesis
        self._last_search_total = None   # total-hit count captured per request
        self._search_pool = None   # relevance-ranked candidate pool, per request
        self._search_ranked = False  # True once a search list was relevance-ranked
        self._feedback_done = False  # title-feedback fired flag, per request
        self._feedback_queries = None  # refined queries from low-hit feedback, per request
        self._auto_feedback_done = False  # feedback queries auto-executed, per request
        self._ladder_capped = False  # hits exceeded LADDER_MAX_HITS, per request
        self._missing_dir_done = False  # missing-direction feedback fired flag, per request
        self._missing_dir_queries = None  # inferred supplementary queries, per request
        self._auto_round_done = False  # system-driven second round fired, per request
        self._auto_ladder_used = 0  # auto-executed ladder queries, per request
        self._patent_auto_used = {"us": 0, "cn": 0}  # built-in patent tool auto-ladder, per source per request
        self._recall_done = False  # recall expansion (family/CPC) fired, per request
        self._tried_queries = []  # queries already sent to the search tool, per request
        self._cpc_hints = None  # matched CPC codes for the question, per request
        self.knowledgeTool = (None, None)  # (knowledge_item, tool_info) — selected inside the loop
        self.tools = []
        lang = self._detect_lang(prompt)
        self._lang = lang
        # Query-mode classification: structured/analytical requests
        # (identifier, assignee, keyword, prosecution/family analysis,
        # document lists) skip the CPC match and the architecture
        # interpretation — those only serve semantic technology searches.
        # Never raises; defaults to "semantic" so tech searches keep the
        # full pipeline.
        self._query_mode = "semantic"
        try:
            from sources.long_task.query_mode import (
                QUERY_MODE_ENABLED, classify_query_mode)
            if QUERY_MODE_ENABLED:
                self._query_mode = await classify_query_mode(prompt, self.llm)
        except Exception:
            self._query_mode = "semantic"

        # ── Patent source routing (built-in search tool + CN ladder) ──
        from sources.patent_source_detect import (
            detect_patent_source_text, map_source_for_tool_route)
        _conv_turns = getattr(self, "_conversation_turns", []) or []
        self._patent_tool_source = map_source_for_tool_route(
            detect_patent_source_text(prompt, [
                {"content": t.get("user", "")} for t in _conv_turns
                if isinstance(t, dict)
            ])
        )
        _need_cn = self._patent_tool_source in ("dual", "cn")

        if callback_handler:
            await _emit_status(callback_handler,
                "正在构造检索式..." if lang == 'zh' else "Building search queries...")

        from sources.long_task.search_query_builder import (
            build_baiten_queries, build_search_queries)

        async def _rewrite() -> dict:
            try:
                return await build_search_queries(prompt, self.llm)
            except Exception:
                return {"concepts": [], "queries": []}

        async def _rewrite_cn() -> dict:
            try:
                return await build_baiten_queries(prompt, self.llm)
            except Exception:
                return {"concepts": [], "queries": []}

        async def _cpc_match():
            # The rewrite's carrier vocabulary joins the match text: a raw
            # question where one word dominates (e.g. "control") would
            # otherwise match control-themed classes instead of the
            # technical domain.  Waits for the concurrently running
            # rewrite; the embedding search itself is blocking, so it
            # runs in a thread and never stalls the other tasks.
            if not CPC_EXPANSION_ENABLED:
                return None
            try:
                from sources.long_task.cpc_semantic import match_query_to_cpc
                rewrite = await _rewrite_task
                # Each concept's carrier vocabulary forms its own match
                # text — one dominant concept must not dilute the others.
                extra_term_groups = [
                    " ".join(str(kw) for kw in (
                        concept.get("carriers") or [])[:4])
                    for concept in (rewrite.get("concepts") or [])
                    if isinstance(concept, dict) and concept.get("carriers")
                ]
                return await asyncio.to_thread(
                    match_query_to_cpc, prompt, extra_terms=extra_term_groups)
            except Exception:
                return None

        async def _interpret():
            # Architecture-level interpretation (strong model): maps the
            # question to the circuit/system patterns patent literature
            # actually uses.  Its queries seed the ladder top so the
            # auto-ladder and the blank-q injection try the architecture
            # wording first; its scheme/terms feed the scoring rubric.
            # Never raises — any failure keeps the flash rewrite untouched.
            try:
                from sources.long_task.technical_interpretation import (
                    INTERPRET_ENABLED, interpret_query,
                )
                if not INTERPRET_ENABLED:
                    return None
                return await interpret_query(prompt)
            except Exception:
                return None

        if self._query_mode == "structured":
            # Identifier/assignee/keyword/analysis requests: the ladder
            # still needs the rewrite; CPC and interpretation add nothing.
            self._search_rewrite = await _rewrite()
            if _need_cn:
                self._search_rewrite_cn = await _rewrite_cn()
            self._cpc_hints = None
            self._search_interpretation = None
        else:
            # Semantic technology search: rewrite + CPC match +
            # interpretation run concurrently — the interpretation no
            # longer waits for the CPC hints (they were a guidance
            # nicety, not a dependency).
            _rewrite_task = asyncio.create_task(_rewrite())
            _cpc_task = asyncio.create_task(_cpc_match())
            _interp_task = asyncio.create_task(_interpret())
            _cn_task = (asyncio.create_task(_rewrite_cn())
                        if _need_cn else None)
            (self._search_rewrite, self._cpc_hints,
             self._search_interpretation) = await asyncio.gather(
                _rewrite_task, _cpc_task, _interp_task)
            if _cn_task is not None:
                self._search_rewrite_cn = await _cn_task
            if self._search_interpretation:
                from sources.long_task.technical_interpretation import (
                    merge_interpretation_queries)
                self._search_rewrite = merge_interpretation_queries(
                    self._search_rewrite, self._search_interpretation)
                scheme = str(
                    self._search_interpretation.get("scheme") or ""
                )[:120]
                players = ", ".join(
                    (self._search_interpretation.get("key_players")
                     or [])[:5])
                self.logger.info(
                    f"search_interpretation — scheme={scheme}"
                    + (f" | players={players}" if players else "")
                )
        # Morph the rewrite ladder: USPTO's applications/search 404s most
        # 3-concept AND combinations, so every query is expanded into its
        # AND-drop chain (3-group → 2-group → single-concept OR) — the
        # agent/auto-ladder then always has a form that can return hits.
        try:
            from sources.long_task.technical_interpretation import (
                expand_rewrite_queries)
            _rewrite_queries = (self._search_rewrite or {}).get("queries") or []
            _expanded = expand_rewrite_queries(_rewrite_queries)
            if _expanded:
                self._search_rewrite = {
                    **(self._search_rewrite or {}), "queries": _expanded}
        except Exception:
            pass  # ladder morphing is an enhancement, never a hard dep
        self.logger.info(
            f"search_rewrite — queries={self._search_rewrite.get('queries')} "
            f"| mode={self._query_mode}"
        )
        if callback_handler:
            await _emit_status(callback_handler,
                "正在分析您的问题..." if lang == 'zh' else "Analyzing your question...")

        # Wrap the handler so the final answer text is collected for
        # multi-turn storage (_store_current_turn).
        wrapped = _ResponseCollector(callback_handler)
        self._active_collector = wrapped

        user_prompt = self.generate_user_prompt(prompt, user_id, query_id)

        # -- Multi-turn memory: request history first, pooled turns fallback --
        conversation_block, _history_patent_ids = _build_previous_conversation_block(
            getattr(self, "_conversation_history", None),
            getattr(self, "_conversation_turns", []),
            user_id, current_query=prompt,
        )
        if not _history_patent_ids:
            # Cross-session memory: recent patent numbers from Redis (需求 2).
            _recent_ids = _read_recent_patent_ids(user_id)
            if _recent_ids:
                conversation_block += (
                    "\n\n最近检索的专利号（仅当用户引用时使用）："
                    + ", ".join(_recent_ids[:20]))

        from sources.long_task.search_query_builder import format_ladder_guidance
        system_prompt = (
            self._get_fixed_system_prefix()
            + conversation_block
            + self._loop_system_guidance()
            + format_ladder_guidance(
                self._search_rewrite, lang,
                cn_rewrite=self._search_rewrite_cn)
        )
        self.memory.reset([
            {'role': 'user', 'content': user_prompt},
            {'role': 'system', 'content': system_prompt},
        ])
        self.logger.info(f"memory.get():{self.memory.get()}")

        registry, bind_tools = await build_tool_set(
            self, user_id, prompt, push_filter,
            patent_source=getattr(self, "_patent_tool_source", "auto"))
        self._react_registry = registry

        # Deterministic long-task routing: when the request clearly
        # matches a type-3 knowledge item's question, trigger the long
        # task directly instead of relying on the LLM to pick the
        # long-task tool from the bound list (observed: prosecution
        # requests went down the USPTO keyword ladder with the long-task
        # tool bound).  Failure-safe: any routing failure falls through
        # to the normal loop.
        long_task_entries = [
            entry for entry in registry.values()
            if entry.kind == "long_task"
        ]
        if long_task_entries:
            matched = await _match_long_task_intent(
                self, prompt, long_task_entries, lang)
            if matched is not None:
                self.logger.info(
                    "Long task intent routed by query match — returning intent")
                return _build_long_task_intent(
                    matched.knowledge, matched.tool_info)

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
        result = await loop.run(
            messages, bind_tools,
            start=getattr(self, "_request_started", None))
        self._react_loop_ran = True
        if result.kind == 'long_task':
            self.logger.info("Long task triggered inside ReAct loop — returning intent")
            return _build_long_task_intent(result.long_task_knowledge,
                                           result.long_task_tool_info)
        return None



    async def _stream_workflow_final_result(self, workflow_result, callback_handler):
        output_mode = getattr(workflow_result, "output_mode", "last")
        workflow_question = getattr(workflow_result, "workflow_question", "")
        workflow_instructions = getattr(workflow_result, "workflow_instructions", "")

        if output_mode == "all":
            system_prompt = (
                "You are a professional data presenter serving non-technical readers. "
                "Below are results from multiple data queries about the same topic. "
                "Synthesize ALL results into a single coherent, complete report.\n\n"
                "CRITICAL RULES:\n"
                "1. Do NOT structure your answer by data source or query step. "
                "Present the information as one unified document 鈥?the reader must not "
                "be able to tell which piece of data came from which query.\n"
                "2. Merge duplicate information across sources 鈥?if the same fact appears "
                "in multiple results, present it once.\n"
                "3. ALL meaningful data must be preserved 鈥?do NOT summarize, abbreviate, "
                "or skip any field with a real value. This is the most important rule.\n"
                "4. Group related information logically: titles and abstracts together, "
                "dates together, people and organizations together, legal/classification "
                "together, documents and references together.\n"
                "5. Filter noise: skip API wrapper fields (errorCode, errorDesc, page, "
                "page_row, total, sort_column), empty values, '0'/'鍚? placeholder values, "
                "and internal system IDs (pid).\n"
                "6. Present as a reader-friendly document with clear section headings. "
                "Use comparison tables when comparing multiple records with shared fields.\n"
                "7. Do NOT add any meta-commentary such as 'based on the query results' or "
                "'here is the synthesized report' 鈥?just output the content directly.\n"
                "8. MANDATORY 鈥?EVERY document URL, image URL, and external link found "
                "in the source data MUST appear verbatim in your output. Even if two URLs "
                "look similar, include BOTH if they point to different resources. "
                "Use descriptive link text: [Title](URL) for documents, "
                "![Description](URL) for images. This rule overrides rule #2 鈥?never "
                "merge or skip URLs.\n"
                "9. LANGUAGE: Match the user's question language. If they asked in Chinese, "
                "output in Chinese. If they asked in English, output in English."
            )
            # Include raw_items from each step so links/images are not lost
            payload = {
                "user_request": getattr(self, "_last_user_prompt", ""),
                "workflow_question": workflow_question,
                "workflow_instructions": workflow_instructions,
                "combined_results": [s.data for s in workflow_result.steps],
                "raw_items_from_all_steps": [
                    getattr(s, "data", None) for s in workflow_result.steps
                ],
            }
        else:
            system_prompt = (
                "You are a self-contained assistant answering from a completed "
                "composed-knowledge workflow result. "
                "Use only the user request and workflow result provided here. "
                "Do not call tools, search externally, or invent missing data. "
                "Return a concise, well-formatted Markdown answer.\n\n"
                "CRITICAL: Every document URL, image URL, and external link in the "
                "workflow result MUST appear in your output. Use descriptive link text: "
                "[Title](URL) for documents, ![Description](URL) for images. "
                "Never omit a link that exists in the source data.\n"
                "LANGUAGE: Match the user's question language."
            )
            payload = {
                "user_request": getattr(self, "_last_user_prompt", ""),
                "workflow": {
                    "question": workflow_question,
                    "instructions": workflow_instructions,
                },
                "workflow_result": workflow_result.final_data,
                "raw_items": getattr(workflow_result, "raw_items", None),
            }

        user_content = json.dumps(payload, ensure_ascii=False, indent=2)
        await self.llm.stream_simple(
            system_prompt=system_prompt,
            user_content=user_content,
            callback_handler=callback_handler,
        )

    def _build_formatter_user_content(self, batch, url_checklist: str = "") -> str:
        batch_json = json.dumps(batch, ensure_ascii=False, indent=2)
        return (
            f"Format and analyze these {len(batch)} search result items as readable Markdown:\n\n"
            f"{url_checklist}"
            f"{batch_json}"
        )

    def _build_url_checklist(self, batch) -> str:
        batch_urls = _collect_urls_from_value(batch)
        if not batch_urls:
            return ""
        return (
            "Mandatory URL checklist. Copy every line verbatim into the corresponding item. "
            "Do not omit any URL from this checklist:\n"
            + "\n".join(f"- {url}" for url in batch_urls)
            + "\n\n"
        )

    async def _stream_deterministic_batch(self, batch, callback_handler):
        await callback_handler.on_llm_new_token(self._render_list_as_md(None, batch))

    async def _stream_formatter_batch(self, system_prompt: str, batch, callback_handler) -> None:
        await self.llm.stream_simple(
            system_prompt=system_prompt,
            user_content=self._build_formatter_user_content(
                batch,
                self._build_url_checklist(batch),
            ),
            callback_handler=callback_handler,
        )

    async def _stream_formatter_or_markdown(
        self,
        system_prompt: str,
        batch,
        callback_handler,
        label: str = "",
    ) -> None:
        """Try the formatter LLM; fall back to deterministic markdown on failure."""
        try:
            await self._stream_formatter_batch(system_prompt, batch, callback_handler)
        except Exception as exc:
            self.logger.warning(
                f"formatter LLM failed{label}; "
                f"streaming deterministic markdown fallback. error={exc}"
            )
            await self._stream_deterministic_batch(batch, callback_handler)

    async def _stream_items_individually(
        self,
        batch,
        system_prompt: str,
        callback_handler,
    ) -> None:
        """Process items one by one: LLM 鈫?markdown on failure."""
        for item_index, item in enumerate(batch, start=1):
            item_batch = [item]
            await self._stream_formatter_or_markdown(
                system_prompt, item_batch, callback_handler,
                label=f" for item {item_index}",
            )

    async def _stream_batch_with_retries(
        self,
        batch,
        system_prompt: str,
        callback_handler,
    ) -> None:
        """Try the batch in one LLM call; fall back to individual items; then markdown."""
        batch_json = json.dumps(batch, ensure_ascii=False, indent=2, default=str)
        if len(batch_json) > MAX_BATCH_JSON_CHARS_FOR_LLM:
            self.logger.info(
                "batch JSON too large for formatter LLM; "
                f"retrying by item. chars={len(batch_json)}"
            )
            await self._stream_items_individually(batch, system_prompt, callback_handler)
            return

        try:
            await self._stream_formatter_batch(system_prompt, batch, callback_handler)
        except Exception as exc:
            self.logger.warning(
                "batch formatter LLM failed; "
                f"retrying by item. error={exc}"
            )
            await self._stream_items_individually(batch, system_prompt, callback_handler)

    async def _stream_raw_items(self, raw_items, callback_handler):
        self._pending_raw_items = None
        original_total = len(raw_items)
        user_prompt = getattr(self, "_last_user_prompt", "")
        batch_size = 1

        async def emit_filter_status(event):
            on_status = getattr(callback_handler, "on_status", None)
            if not on_status:
                return
            message = event.get("message", "")
            metadata = {
                key: value
                for key, value in event.items()
                if key != "message"
            }
            await on_status(message, **metadata)

        # 鈹€鈹€ 灏忓垪琛ㄥ揩閫熻矾寰勶細璺宠繃杩囨护锛屼笓鐢?prompt 涓€娆℃€у繝瀹炶緭鍑?鈹€鈹€
        if original_total <= SMALL_LIST_THRESHOLD:
            self.logger.info(
                f"[SMALL-LIST] ({original_total} items), "
                f"using faithful single-call reproduction path"
            )
            pruned = [_prune_item_for_llm(item) for item in raw_items]
            items_for_export = list(raw_items)
            lang = getattr(self, '_lang', 'zh')
            if lang == 'en':
                heading = f"## Results ({original_total} items)"
            else:
                heading = f"## 结果（{original_total} 项）"
            await callback_handler.on_llm_new_token(
                f"\n\n---\n\n{heading}\n\n"
            )
            _replace_uspto_download_urls_for_batch(pruned)

            url_checklist = self._build_url_checklist(pruned)
            batch_json = json.dumps(pruned, ensure_ascii=False, indent=2, default=str)

            if len(batch_json) <= MAX_BATCH_JSON_CHARS_FOR_LLM:
                # 浠庣涓€鏉?item 鎻愬彇鎵€鏈夊瓧娈靛悕锛岀敓鎴愬己鍒惰緭鍑烘ā鏉?
                first_keys = list(pruned[0].keys()) if pruned else []
                field_checklist = "\n".join(
                    f"- {k}" for k in first_keys
                )

                lang = getattr(self, '_lang', 'zh')
                if lang == 'en':
                    faithful_system_prompt = (
                        "You are a professional data presenter serving non-technical readers. "
                        "Your output must be readable, well-structured, and free of jargon.\n\n"
                        "CRITICAL RULES:\n\n"
                        "1. TRANSLATE field codes into clear English labels. "
                        "For example: 'apc' \u2192 'Applicant', 'ad' \u2192 'Application Date', "
                        "'pdt' \u2192 'Patent Type', 'pk' \u2192 'Document Kind', "
                        "'pns' \u2192 'Patent Number', 'lsscn' \u2192 'Legal Status'. "
                        "Use your knowledge to interpret every code.\n\n"
                        "2. FILTER noise: skip fields whose value is empty, '0', '\u5426', "
                        "or clearly an internal system ID (like 'pid'). "
                        "Also skip the top-level API wrapper fields (errorCode, errorDesc, "
                        "page_row, page, total, sort_column) \u2014 they are not part of the data.\n\n"
                        "3. GROUP related fields logically: "
                        "titles together, abstracts together, dates together, "
                        "people & organizations together, legal/classification together.\n\n"
                        "4. PRESENT as a reader-friendly document with clear section headings, "
                        "NOT as a flat key-value dump. Use comparison tables when comparing "
                        "multiple records with shared fields.\n\n"
                        "5. ALL meaningful data must be preserved \u2014 do not summarize, "
                        "abbreviate, or skip any non-noise field. If a field has a real value, "
                        "it belongs in the output.\n\n"
                        "6. Every URL must be copied exactly and verbatim. "
                        "Image URLs MUST use ![description](URL) syntax.\n\n"
                        "7. Do NOT add a concluding summary \u2014 let the data speak for itself.\n\n"
                        "8. LANGUAGE: The user asked in English \u2014 you MUST output "
                        "everything in English. All labels, headings, and descriptions "
                        "must be in English."
                    )
                    faithful_user_content = (
                        f"Here are {len(pruned)} data item(s) to present for non-technical readers. "
                        f"Translate all field codes into plain English labels. "
                        f"Group related information logically. "
                        f"Skip empty/noise fields. Keep ALL meaningful data.\n\n"
                        f"Reference \u2014 all fields present in the data:\n"
                        f"{field_checklist}\n\n"
                        f"{url_checklist}"
                        f"{batch_json}"
                    )
                else:
                    faithful_system_prompt = (
                        "You are a professional data presenter serving non-technical readers. "
                        "Your output must be readable, well-structured, and free of jargon.\n\n"
                        "CRITICAL RULES:\n\n"
                        "1. TRANSLATE field codes into clear Chinese labels. "
                        "For example: 'apc'\u2192\u7533\u8bf7\u4eba, 'ad'\u2192\u7533\u8bf7\u65e5, 'pdt'\u2192\u4e13\u5229\u7c7b\u578b, "
                        "'pk'\u2192\u6587\u732e\u79cd\u7c7b, 'pns'\u2192\u4e13\u5229\u53f7, 'lsscn'\u2192\u6cd5\u5f8b\u72b6\u6001. "
                        "Use your knowledge to interpret every code.\n\n"
                        "2. FILTER noise: skip fields whose value is empty, '0', '\u5426', "
                        "or clearly an internal system ID (like 'pid'). "
                        "Also skip the top-level API wrapper fields (errorCode, errorDesc, "
                        "page_row, page, total, sort_column) \u2014 they are not part of the data.\n\n"
                        "3. GROUP related fields logically: "
                        "titles together, abstracts together, dates together, "
                        "people & organizations together, legal/classification together.\n\n"
                        "4. PRESENT as a reader-friendly document with clear section headings, "
                        "NOT as a flat key-value dump. Use comparison tables when comparing "
                        "multiple records with shared fields.\n\n"
                        "5. ALL meaningful data must be preserved \u2014 do not summarize, "
                        "abbreviate, or skip any non-noise field. If a field has a real value, "
                        "it belongs in the output.\n\n"
                        "6. Every URL must be copied exactly and verbatim. "
                        "Image URLs MUST use ![description](URL) syntax.\n\n"
                        "7. Do NOT add a concluding summary \u2014 let the data speak for itself.\n\n"
                        "8. LANGUAGE: \u7528\u6237\u4f7f\u7528\u4e2d\u6587\u63d0\u95ee \u2014 \u4f60\u5fc5\u987b\u5168\u7a0b\u4f7f\u7528\u4e2d\u6587\u8f93\u51fa\u3002"
                        "\u6240\u6709\u6807\u7b7e\u3001\u6807\u9898\u548c\u63cf\u8ff0\u90fd\u5fc5\u987b\u4f7f\u7528\u4e2d\u6587\u3002"
                    )
                    faithful_user_content = (
                        f"Here are {len(pruned)} data item(s) to present for non-technical readers. "
                        f"Translate all field codes into plain Chinese labels. "
                        f"Group related information logically. "
                        f"Skip empty/noise fields. Keep ALL meaningful data.\n\n"
                        f"Reference \u2014 all fields present in the data:\n"
                        f"{field_checklist}\n\n"
                        f"{url_checklist}"
                        f"{batch_json}"
                    )
                try:
                    await self.llm.stream_simple(
                        system_prompt=faithful_system_prompt,
                        user_content=faithful_user_content,
                        callback_handler=callback_handler,
                    )
                except Exception as exc:
                    self.logger.warning(
                        "small-list faithful transcriber failed; "
                        f"falling back to deterministic markdown. error={exc}"
                    )
                    await self._stream_deterministic_batch(pruned, callback_handler)
            else:
                self.logger.info(
                    "small-list pruned JSON too large for single LLM call "
                    f"({len(batch_json)} chars), using deterministic markdown"
                )
                await self._stream_deterministic_batch(pruned, callback_handler)

            await callback_handler.on_llm_new_token("\n\n")

            on_artifacts = getattr(callback_handler, "on_artifacts", None)
            if on_artifacts:
                _replace_uspto_download_urls_for_batch(items_for_export)
                artifacts = build_result_artifacts(
                    items_for_export,
                    source=_infer_result_source(self.knowledgeTool[1]),
                    query_id=getattr(self, "_last_query_id", None),
                    original_count=original_total,
                    filter_applied=False,
                    lang=getattr(self, "_lang", "zh"),
                )
                if artifacts:
                    await on_artifacts(artifacts)

            # Store patent IDs for potential follow-up long-task conversation refs
            _store_conversation_patent_ids(self, items_for_export)
            # Also emit to frontend so follow-up queries carry patent IDs in
            # conversation_history (independent of Redis).
            await _emit_patent_ids_to_frontend(self, items_for_export, callback_handler)

            return
        # 鈹€鈹€ 澶у垪琛細璧板師鏈夌殑杩囨护 + 鎵归噺鏍煎紡鍖栬矾寰?鈹€鈹€

        if tool_result_filter_enabled():
            filter_result = await filter_tool_result_items(
                raw_items,
                user_prompt,
                self.llm.complete_json,
                batch_size=batch_size,
                status_callback=emit_filter_status,
            )
        else:
            # Feature off (default): results pass through unfiltered —
            # skips the per-query filter-criteria LLM call entirely.
            filter_result = unfiltered_result(raw_items)
        pending = filter_result.items

        # Save the original (filtered but un-pruned) items for Excel / CSV
        # export.  The pruning below only affects the LLM input path.
        items_for_export = list(pending)

        # ── 大列表摘要模式：剔除超长字段后做整体总结，跳过逐条批处理 ──
        if USE_LARGE_LIST_SUMMARY:
            summary_items = _bounded_summary_items(pending)
            lang = getattr(self, '_lang', 'zh')
            heading = _large_list_summary_heading(
                lang, len(summary_items), len(pending))
            await callback_handler.on_llm_new_token(
                f"\n\n---\n\n{heading}\n\n"
            )

            summary_system_prompt = _summary_system_prompt(
                bool(getattr(self, "_search_ranked", False)), lang)

            try:
                await self.llm.stream_simple(
                    system_prompt=summary_system_prompt,
                    user_content=json.dumps(summary_items, ensure_ascii=False, indent=2, default=str),
                    callback_handler=callback_handler,
                )
            except Exception as exc:
                self.logger.warning(
                    f"large-list summary LLM failed: {exc}"
                )
                await callback_handler.on_llm_new_token(
                    "\n\n*Summary generation failed. Please download the data file below.*"
                )

            lang = getattr(self, '_lang', 'zh')
            if lang == 'en':
                await callback_handler.on_llm_new_token(
                    "\n\n> \U0001f4e5 For complete data, please download the Excel or CSV file below.\n\n"
                )
            else:
                await callback_handler.on_llm_new_token(
                    "\n\n> \U0001f4e5 如需完整数据，请下载下方的 Excel 或 CSV 文件。\n\n"
                )

            on_artifacts = getattr(callback_handler, "on_artifacts", None)
            if on_artifacts:
                _replace_uspto_download_urls_for_batch(items_for_export)
                artifacts = build_result_artifacts(
                    items_for_export,
                    source=_infer_result_source(self.knowledgeTool[1]),
                    query_id=getattr(self, "_last_query_id", None),
                    original_count=original_total,
                    filter_applied=True,
                    lang=getattr(self, "_lang", "zh"),
                )
                if artifacts:
                    await on_artifacts(artifacts)

            # Store patent IDs for potential follow-up long-task conversation refs
            _store_conversation_patent_ids(self, items_for_export)
            # Also emit to frontend so follow-up queries carry patent IDs in
            # conversation_history (independent of Redis).
            await _emit_patent_ids_to_frontend(self, items_for_export, callback_handler)

            return


        # Prune each item so that no single element exceeds 15 000 chars.
        # Oversized arrays (> 10 000 chars) are dropped; oversized dicts are
        # recursed into; oversized strings are truncated.
        pending_before_prune = sum(_json_len(item) for item in pending)
        pending = [_prune_item_for_llm(item) for item in pending]
        pending_after_prune = sum(_json_len(item) for item in pending)
        if pending_before_prune != pending_after_prune:
            self.logger.info(
                "tool_result pruned long values before batch formatting; "
                f"chars_before={pending_before_prune}, chars_after={pending_after_prune}"
            )

        total = len(pending)
        lang = getattr(self, '_lang', 'zh')
        if lang == 'en':
            heading = (
                f"## Filtered Results ({filter_result.filtered_count} of {filter_result.original_count} items)"
                if filter_result.applied
                else f"## Full Results ({original_total} items)"
            )
        else:
            heading = (
                f"## 筛选结果（{filter_result.filtered_count}/{filter_result.original_count} 项）"
                if filter_result.applied
                else f"## 完整结果（{original_total} 项）"
            )
        await callback_handler.on_llm_new_token(
            f"\n\n---\n\n{heading}\n\n"
        )
        if lang == 'en':
            system_prompt = (
                "You are presenting search result items clearly and concisely. "
                "For each item, extract and present the most important information as clean Markdown. "
                "Use **bold** for field names. Number each item. "
                "Every URL is mandatory: copy every URL from the input exactly and verbatim. "
                "Do not omit, shorten, summarize, translate, decode, re-encode, or alter any URL. "
                "If an item has multiple URLs, include all of them under that item. "
                "If a URL is an image URL, display it using Markdown image syntax exactly as "
                "![alt text](image_URL) so the frontend can render the image inline. "
                "Do NOT add any preamble, summary, or conclusion - output only the formatted items. "
                "IMPORTANT: The user asked in English — all labels and descriptions must be in English."
            )
        else:
            system_prompt = (
                "你正在清晰简洁地展示搜索结果项目。"
                "对于每个项目，提取并呈现最重要的信息，使用干净的 Markdown 格式。"
                "使用 **粗体** 标记字段名。给每个项目编号。"
                "每个 URL 都是强制性的：从输入中准确且逐字复制每个 URL。"
                "不要省略、缩短、总结、翻译、解码、重新编码或更改任何 URL。"
                "如果一个项目有多个 URL，请在该项目下包含所有 URL。"
                "如果 URL 是图像 URL，请使用 Markdown 图像语法显示："
                "![替代文本](image_URL)，以便前端可以内联渲染图像。"
                "不要添加任何前言、总结或结论——只输出格式化的项目。"
                "重要：用户使用中文提问——所有标签和描述必须使用中文。"
            )
        for batch_start in range(0, total, batch_size):
            batch = pending[batch_start:batch_start + batch_size]
            self.logger.info(f"batch: {batch}")
            _replace_uspto_download_urls_for_batch(batch)
            batch_end = min(batch_start + batch_size, total)
            if lang == 'en':
                await callback_handler.on_llm_new_token(
                    f"### Items {batch_start + 1}-{batch_end}\n\n"
                )
            else:
                await callback_handler.on_llm_new_token(
                    f"### 项目 {batch_start + 1}-{batch_end}\n\n"
                )
            await self._stream_batch_with_retries(batch, system_prompt, callback_handler)
            await callback_handler.on_llm_new_token("\n\n")

        on_artifacts = getattr(callback_handler, "on_artifacts", None)
        if on_artifacts:
            _replace_uspto_download_urls_for_batch(items_for_export)
            artifacts = build_result_artifacts(
                items_for_export,
                source=_infer_result_source(self.knowledgeTool[1]),
                query_id=getattr(self, "_last_query_id", None),
                original_count=original_total,
                filter_applied=filter_result.applied,
                lang=getattr(self, "_lang", "zh"),
            )
            if artifacts:
                await on_artifacts(artifacts)

        # Store patent IDs for potential follow-up long-task conversation refs
        _store_conversation_patent_ids(self, items_for_export)
        # Also emit to frontend so follow-up queries carry patent IDs in
        # conversation_history (independent of Redis).
        await _emit_patent_ids_to_frontend(self, items_for_export, callback_handler)


    def _store_current_turn(self, assistant_text: str = "") -> None:
        """Store the current query + assistant response for multi-turn context.

        The stored text is truncated to keep the conversation context compact.
        Old system prompts and internal IDs are never stored — only the user's
        question and the assistant's visible Markdown response.
        """
        user_query = getattr(self, '_last_user_prompt', '') or ''
        if not user_query.strip():
            return
        # Keep only a short summary of the assistant response
        short = (assistant_text or '')[:800]
        if len(assistant_text or '') > 800:
            short += '...'
        self._conversation_turns.append({
            'user': user_query.strip(),
            'assistant': short or '(tool executed successfully)',
            # Pooled agents are shared across users — the user_id keeps
            # each user's turns isolated (需求 2: turns 按 user 隔离).
            'user_id': getattr(self, '_last_user_id', None),
        })
        # Keep at most 10 turns to bound memory growth
        if len(self._conversation_turns) > 10:
            self._conversation_turns = self._conversation_turns[-10:]
        self.logger.info(
            f"_store_current_turn — total turns={len(self._conversation_turns)}, "
            f"assistant_len={len(assistant_text or '')}"
        )

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


if __name__ == "__main__":
    pass
