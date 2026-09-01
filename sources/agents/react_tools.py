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
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from sources.knowledge.knowledge import get_knowledge_tool_candidates
from sources.long_task.candidate_metadata import (
    build_candidates,
    ensure_search_fields,
    is_dead_status,
    is_documents_tool,
    is_identifying_number_tool,
    is_keyword_search_tool,
    is_uspto_tool,
)
from sources.long_task.chat_relevance import (
    SCORE_PER_CALL,
    SearchPool,
    score_candidates_concurrent,
)
from sources.long_task.recall_sources import (
    collect_family_refs,
    fetch_by_cpc,
    fetch_by_numbers,
    records_to_candidates,
)
from sources.long_task.semantic_rerank import (
    PRESCORE_ENABLED,
    RERANK_ENABLED,
    semantic_scores_batch,
)

TOP_N = int(os.getenv("REACT_TOOL_TOP_N", "5"))
LOW_HIT_FEEDBACK_THRESHOLD = int(os.getenv(
    "REACT_LOW_HIT_FEEDBACK_THRESHOLD", "10"))
MAX_PATENT_LIST_ITEMS = int(os.getenv("REACT_MAX_PATENT_LIST_ITEMS", "100"))
# Family scoring: candidates whose direct family member scored high with
# the Flash LLM get scored too, even when their own title scored low in
# the bge-m3 prescore (same invention, different wording — observed with
# ERP Power "human centric black body dimming" titles).
FAMILY_SCORE_ENABLED = os.getenv("REACT_FAMILY_SCORE", "1") == "1"
FAMILY_SEED_MIN = int(os.getenv("REACT_FAMILY_SEED_MIN", "4"))
FAMILY_SCORE_BUDGET = int(os.getenv("REACT_FAMILY_SCORE_BUDGET", "30"))
# Post-retrieval grounded interpretation: fires once per request when
# the scored pool clears the minimum. Set just above the single-round
# scoring cap so a first-round noise pool never triggers, but a
# recall-enlarged pool does. Clusters the scored head.
GROUNDED_MIN = int(os.getenv("REACT_GROUNDED_MIN", "45"))
# Pool-size floor: a single-round noise pool (≤50 candidates) must
# never trigger synthesis even when fully scored; recall-scale pools
# (200+) always pass.  Two conditions — pool size AND scored count —
# are more robust than any single threshold.
GROUNDED_POOL_MIN = int(os.getenv("REACT_GROUNDED_POOL_MIN", "120"))
GROUNDED_HEAD = int(os.getenv("REACT_GROUNDED_HEAD", "30"))
RELEVANCE_RANK_ENABLED = os.getenv("REACT_RELEVANCE_RANK", "1") != "0"
REACT_POOL_MAX_PAGES = int(os.getenv("REACT_POOL_MAX_PAGES", "2"))
# Queries whose total hits exceed this threshold are never paged through:
# the first page already feeds scoring/feedback, and extra pages of a
# huge noisy pool waste API calls and scoring time (observed: a 153k-hit
# query whose page 2 added 44 noise candidates and 57s of scoring).
REACT_POOL_MAX_TOTAL_PAGES = int(os.getenv("REACT_POOL_MAX_TOTAL_PAGES", "1000"))
# Missing-direction feedback fires once per turn after a scoring round,
# but only when the pool demonstrably holds relevant hits — a noise pool
# (low best score) must not seed the suggested queries.
MISSING_DIR_MIN_CANDIDATES = int(os.getenv("REACT_MISSING_DIR_MIN_CANDIDATES", "3"))
MISSING_DIR_MIN_SCORE = float(os.getenv("REACT_MISSING_DIR_MIN_SCORE", "4"))
REACT_AUTO_ROUND_MAX_QUERIES = int(os.getenv("REACT_AUTO_ROUND_MAX_QUERIES", "2"))
# Ladder exhaustion is system-driven, not left to the agent's discretion:
# when a search leaves nothing displayable, the next untried ladder
# queries are executed by the system (observed: the agent concluded "no
# results" with half the ladder untried despite the zero-hit nudge).
AUTO_LADDER_BATCH = 2  # untried ladder queries auto-executed per observation
AUTO_LADDER_MAX = int(os.getenv("REACT_AUTO_LADDER_MAX_QUERIES", "6"))
# Refined-query feedback (low-hit title feedback) is executed by the
# system too — observed agents answering with the suggestions ignored.
AUTO_FEEDBACK_MAX = int(os.getenv("REACT_AUTO_FEEDBACK_MAX_QUERIES", "2"))
# Recall expansion (citation/family + CPC routes): once per request, the
# pool's family numbers and the matched CPC codes pull records beyond
# the keyword ladder.
RECALL_MAX_CPC = int(os.getenv("REACT_RECALL_MAX_CPC", "3"))
RECALL_POOL_HEAD = 20
# CPC semantic expansion (plan B, route C): matched CPC code/title pairs
# seed the missing-direction prompt with the domain's classification
# language.  Off by default — requires the data files and vector cache
# built by scripts/build_cpc_vectors.py on the server.
CPC_EXPANSION_ENABLED = os.getenv("REACT_CPC_EXPANSION", "0") == "1"
REACT_USPTO_SORT_FIELD = os.getenv("REACT_USPTO_SORT_FIELD", "_score")
LADDER_MAX_HITS = int(os.getenv("REACT_LADDER_MAX_HITS")
                      or os.getenv("REACT_TIGHTEN_SUGGEST_THRESHOLD", "300"))
# Per-number verification tools (search_patent_by_identifying_number_...)
# are capped per request: the LLM was observed looping through 8+ one-by-one
# fetches (each followed by a full ~2.5s semantic rerank) without
# converging.  At the cap the tool returns a stop-nudge instead of fetching
# yet another application.
VERIFY_CALL_MAX = int(os.getenv("REACT_VERIFY_CALL_MAX", "8"))


async def _agent_status(agent, message: str) -> None:
    """Fire a transient status event through the agent's callback handler.

    The handler is stored on the agent per request (create_agent); the
    long silent phases (scoring / recall / synthesis) use this so the
    streaming client always has a live "what is happening now" line.
    Never raises.  Probe log on every call so a silently-broken
    status channel is visible in general_agent.log.
    """
    _glog = getattr(agent, "logger", None)
    handler = getattr(agent, "_callback_handler", None)
    if handler is None:
        if _glog is not None:
            _glog.info(f"agent status — dropped (no handler): {message}")
        return
    on_status = getattr(handler, "on_status", None)
    if on_status is None:
        if _glog is not None:
            _glog.info(f"agent status — dropped (no on_status): {message}")
        return
    try:
        await on_status(message)
    except Exception:
        pass


def _ladder_cap_note(lang: str = "zh") -> str:
    """Deterministic constraint: once hits exceed the system's processing
    capacity, wider ladder queries are off the table — only tightening
    remains."""
    if lang == "en":
        return (f"\nHit counts have exceeded the system's processing "
                f"capacity ({LADDER_MAX_HITS}): the wider pre-built "
                f"ladder queries no longer apply. If you keep searching, "
                f"you must tighten the query with additional constraint "
                f"terms.")
    return (f"\n检索命中已超出系统处理容量（{LADDER_MAX_HITS} 条）："
            f"预置阶梯中更宽的检索式不再适用；如继续检索，必须添加"
            f"限定词显著收紧。")


def _apply_ladder_cap(agent, text: str, total, lang: str) -> str:
    """Persist the ladder cap once any search exceeds LADDER_MAX_HITS —
    the constraint then rides along on every later search observation
    so the agent cannot ignore it by switching queries."""
    if isinstance(total, int) and total > LADDER_MAX_HITS:
        agent._ladder_capped = True
    if getattr(agent, "_ladder_capped", False):
        return text + _ladder_cap_note(lang)
    return text
SEARCH_KNOWLEDGE_TOOL_NAME = "search_my_knowledge"
MAX_SEARCH_RESULTS = 5
MAX_OBSERVATION_CHARS = 300


class _QueryArgs(BaseModel):
    query: str = Field(description="Natural-language description of what you need")


FETCH_PATENT_SPEC_TOOL_NAME = "fetch_patent_spec"


class _PatentIdArgs(BaseModel):
    patent_id: str = Field(description="USPTO application number (8 digits, e.g. 19511555)")


async def _fetch_patent_spec_stub(patent_id: str) -> str:
    raise NotImplementedError("executed via dispatch, not directly")


# ── Built-in dual/single-source patent search (USPTO + Baiten CN) ────────────
# Registered per query in build_tool_set based on the detected patent
# source; executed via make_action_executor (kind="patent_search").

DUAL_PATENT_SEARCH_TOOL_NAME = "patent_search_dual"
CN_PATENT_SEARCH_TOOL_NAME = "patent_search_cn"


class _DualPatentSearchArgs(BaseModel):
    query_string_us: str | None = Field(
        default=None,
        description="USPTO free-form search query (English, from the guidance ladder)",
    )
    query_string_cn: str | None = Field(
        default=None,
        description="Baiten CN search query (Chinese, from the guidance ladder)",
    )
    page: int = Field(default=1, description="Page number")
    page_size: int = Field(default=20, description="Results per page")


class _CnPatentSearchArgs(BaseModel):
    query_string_cn: str = Field(
        description="Baiten CN search query (Chinese, from the guidance ladder)",
    )
    page: int = Field(default=1, description="Page number")
    page_size: int = Field(default=20, description="Results per page")


async def _patent_search_stub(query_string_cn: str = None) -> str:
    raise NotImplementedError("executed via dispatch, not directly")


@dataclass
class ToolEntry:
    name: str
    kind: str                    # 'search' | 'knowledge' | 'long_task'
    knowledge: Any               # KnowledgeItem or None
    tool_info: Any               # ToolItem or None
    tool: Optional[StructuredTool]


def _parse_bilingual_question(text: str, lang: str) -> str:
    """Pick the ``zh:`` / ``en:`` side of a bilingual knowledge question.

    Knowledge items store question/description as ``zh:...|en:...`` (see
    mysql/init/update_uspto_prosecution_knowledge.sql).  The raw payload
    must never leak into tool names/descriptions — the LLM then sees
    ``zh:|en:`` scaffolding instead of the actual scenario.  Falls back
    to the first non-empty side when only one language is present.
    """
    raw = (text or "").strip()
    if not raw:
        return raw
    parts: dict[str, str] = {}
    for seg in raw.split("|"):
        seg = seg.strip()
        m = re.match(r"^(zh|en):(.*)$", seg, re.IGNORECASE)
        if m:
            parts[m.group(1).lower()] = m.group(2).strip()
    if parts:
        return parts.get("zh" if lang == "zh" else "en") \
            or next(iter(parts.values()))
    return raw


def _clean_tool_name(knowledge) -> str:
    """Sanitise a knowledge title into a tool name (same rules as before)."""
    question = _parse_bilingual_question(
        getattr(knowledge, "question", "") or "", "zh")
    title = question.strip() or "dynamic_knowledge_tool"
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", title)
    return cleaned or "dynamic_knowledge_tool"


def _long_task_description(knowledge, lang: str = "zh") -> str:
    question = _parse_bilingual_question(
        getattr(knowledge, "question", "") or "", lang)
    desc = _parse_bilingual_question(
        getattr(knowledge, "description", "") or "", lang)
    return (
        f"Background analysis long task: {question}. "
        f"{desc} After calling, the task runs asynchronously and the user "
        f"is notified — do not wait for results."
    )[:800]


def _parse_match_index(raw) -> int | None:
    """Extract the matched entry index from a classifier output.

    Accepts the parsed dict ({'match': n}), a JSON string, or a bare
    number/string.  None for null/none/no-match/parse failures so the
    caller falls through to the normal loop.
    """
    val = None
    if isinstance(raw, dict):
        val = raw.get("match")
    elif isinstance(raw, str):
        try:
            val = json.loads(raw).get("match")
        except (ValueError, TypeError):
            stripped = raw.strip()
            val = None if stripped.lower() in ("null", "none", "") else stripped
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)) and float(val).is_integer():
        return int(val)
    if isinstance(val, str):
        try:
            return int(float(val.strip()))
        except (ValueError, TypeError):
            return None
    return None


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
    patent_source: str = "dual",
) -> Tuple[Dict[str, ToolEntry], List[dict]]:
    """Build (registry, tools) for one query.

    search_my_knowledge is always first; type-3 items become long-task
    tools; vector-recalled type-1 items (top TOP_N) become knowledge tools.

    *patent_source* (uspto/cn/dual — post map_source_for_tool_route
    semantics) decides whether the built-in patent search tool is
    registered: ``dual`` (unspecified country, the default) registers
    the combined USPTO+Baiten tool, ``cn`` registers the Baiten-only
    tool, ``uspto`` registers neither (the existing USPTO scene tools
    keep their current behavior).
    """
    registry: Dict[str, ToolEntry] = {}
    tools: List[dict] = []

    def add(entry: ToolEntry) -> None:
        if entry.name in registry:
            return  # duplicate title — first registration wins
        registry[entry.name] = entry
        tools.append(_tool_to_bind_dict(entry.tool))

    spec_tool = StructuredTool.from_function(
        func=_fetch_patent_spec_stub,
        name=FETCH_PATENT_SPEC_TOOL_NAME,
        description=(
            "Download and analyze the specification (说明书) of one USPTO "
            "patent application by its application number. Use this when the "
            "user asks for the technical solution, claims, or details of a "
            "specific patent. Returns a structured analysis of the full text."
        ),
        args_schema=_PatentIdArgs,
    )
    add(ToolEntry(name=FETCH_PATENT_SPEC_TOOL_NAME, kind="patent_spec",
                  knowledge=None, tool_info=None, tool=spec_tool))

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

    # ── Built-in patent search (Baiten CN + optional USPTO) ──
    if patent_source in ("dual", "cn"):
        is_dual = patent_source == "dual"
        if is_dual:
            search_name = DUAL_PATENT_SEARCH_TOOL_NAME
            search_schema = _DualPatentSearchArgs
            search_desc = (
                "Search patents when the user does NOT specify a country: "
                "returns BOTH US patents (USPTO) and Chinese patents "
                "(Baiten) in one call. Pass the English ladder query as "
                "query_string_us and the Chinese ladder query as "
                "query_string_cn. One source failing returns only the "
                "other source's results. When the user asks in Chinese "
                "without naming a country, Chinese patents are the "
                "primary target — query_string_cn is required (use the "
                "Chinese ladder query)."
            )
        else:
            search_name = CN_PATENT_SEARCH_TOOL_NAME
            search_schema = _CnPatentSearchArgs
            search_desc = (
                "Search Chinese patents (Baiten) for a user question "
                "about Chinese patents. Pass the Chinese ladder query "
                "as query_string_cn."
            )
        search_tool = StructuredTool.from_function(
            func=_patent_search_stub,
            name=search_name,
            description=search_desc,
            args_schema=search_schema,
        )
        add(ToolEntry(name=search_name, kind="patent_search",
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
                description=_long_task_description(
                    knowledge, getattr(agent, "_lang", "zh")),
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


SEARCH_DIGEST_LIMIT = 20
SEARCH_DIGEST_CHARS = 3000


def _items_digest(raw_items, limit: int = SEARCH_DIGEST_LIMIT,
                  lang: str = "zh") -> str:
    """Serialize search raw_items into a bounded digest for the LLM.

    USPTO-shaped items are flattened via build_candidates into
    ``申请号 | 标题 | 申请人 | 申请日 | 状态`` lines.  Non-USPTO shapes
    fall back to a truncated JSON dump.
    """
    items = raw_items or []
    if not items:
        return ""
    lines: list = []
    candidates = build_candidates(items)
    for c in candidates[:limit]:
        parts = [
            c.get("patent_id") or "?",
            c.get("title") or "(无标题)",
            c.get("applicant") or "?",
            c.get("filing_date") or "?",
            c.get("status") or "?",
        ]
        lines.append(" | ".join(str(p) for p in parts))
    # Baiten CN candidates (flat mapped shape with source="baiten") ride
    # alongside USPTO rows in a mixed dual-source pool.
    flat = [c for c in items if isinstance(c, dict)
            and c.get("source") == "baiten"
            and c.get("patent_id")]
    for c in flat[:limit]:
        parts = [
            c.get("patent_id") or "?",
            c.get("title") or "(无标题)",
            c.get("applicant") or "?",
            c.get("pub_date") or c.get("apply_date") or "?",
        ]
        lines.append(" | ".join(str(p) for p in parts))
    if lines:
        text = "\n".join(lines)
        total = len(candidates) + len(flat)
        if total > limit:
            note = (f"\n…共 {total} 条" if lang == "zh"
                    else f"\n...{total} items total")
            text += note
        return text[:SEARCH_DIGEST_CHARS]
    import json
    try:
        dumped = json.dumps(items, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        dumped = str(items)
    return dumped[:SEARCH_DIGEST_CHARS]


def _cap_patent_list(tool_info, items: list, lang: str) -> Tuple[list, str]:
    """Cap search-style patent lists at MAX_PATENT_LIST_ITEMS.

    Document-list tools (uspto_documents, URL contains 'documents') are
    uncapped — the user wants every document of a single patent.
    """
    if is_documents_tool(tool_info):
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


def _relevance_pool_applies_tool(agent, tool_info) -> bool:
    """Tool-level (pre-invoke) half of the pool gate: switch on, backend,
    USPTO URL.  The parse check happens post-invoke on the results."""
    if not RELEVANCE_RANK_ENABLED:
        return False
    if getattr(tool_info, "push", None) != 2:
        return False
    return is_uspto_tool(tool_info)


def _relevance_pool_applies(agent, tool_info, raw_items) -> bool:
    """Pool + ranking applies to backend USPTO search tools whose results
    flatten via build_candidates (any USPTO-shaped patent list — keyword,
    assignee, or otherwise — merges into the turn's ranked pool).

    Document-list tools (uspto_documents — URL contains 'documents') are
    the final answer: every document of ONE application, never a search
    pool.  Ranking/recall would replace them with unrelated pool patents
    (observed: 68 documents streamed to the frontend as 34 recall patents).
    """
    if not _relevance_pool_applies_tool(agent, tool_info):
        return False
    if is_documents_tool(tool_info):
        return False
    return bool(build_candidates(raw_items or []))


def _ranked_digest(candidates, limit: int = SEARCH_DIGEST_LIMIT,
                   lang: str = "zh") -> str:
    """Serialize ranked candidate dicts into a bounded digest with scores."""
    lines = []
    for c in candidates[:limit]:
        score = c.get("relevance_score")
        score_txt = ""
        if isinstance(score, (int, float)):
            score_txt = (f" 相关度{int(score)}/5" if lang == "zh"
                         else f" relevance {int(score)}/5")
        parts = [
            c.get("patent_id") or "?",
            c.get("title") or "(无标题)",
            c.get("applicant") or "?",
            c.get("filing_date") or "?",
            c.get("status") or "?",
        ]
        lines.append(" | ".join(str(p) for p in parts) + score_txt)
    text = "\n".join(lines)
    if len(candidates) > limit:
        note = (f"\n…共 {len(candidates)} 条，已按相关度排序" if lang == "zh"
                else f"\n...{len(candidates)} items total, relevance-ranked")
        text += note
    return text[:SEARCH_DIGEST_CHARS]


def _get_flash_provider(agent):
    """Return the agent's cached Flash scoring provider, constructing it
    lazily from the long-task config (deepseek-v4-flash / MiniMax
    M2.7-highspeed, following api_routes/core.py's pattern).

    Construction failure returns None — callers fall back to the main
    LLM so scoring degrades instead of breaking.
    """
    cached = getattr(agent, "_flash_llm", None)
    if cached is not None:
        return cached
    try:
        from sources.llm_provider import Provider
        from sources.long_task.config import get_long_task_config
        family = (os.getenv("REACT_SCORE_PROVIDER_FAMILY")
                  or ((get_long_task_config() or {}).get("provider_family")
                      or "deepseek"))
        model = (os.getenv("REACT_SCORE_MODEL")
                 or ("deepseek-v4-flash" if family == "deepseek"
                     else "MiniMax-M2.7-highspeed"))
        cached = Provider(provider_name=family, model=model,
                          server_address="", is_local=False)
    except Exception:
        cached = None
    agent._flash_llm = cached
    return cached


LONG_TASK_ROUTE_ENABLED = os.getenv("REACT_LONG_TASK_ROUTE", "1") == "1"
# Rule-fallback floor for the long-task router: the flash classifier has
# been observed missing an obvious prosecution request ({"match": null}
# on "分析专利 12096133 的审查历史" — the request then spent 5 minutes
# in the keyword ladder before the LLM picked the long-task tool).  When
# the classifier says null, a query that carries an 8-digit US patent
# number AND shares a ≥4-char contiguous overlap with the item question
# (the intent phrase, e.g. 审查历史) still routes.  No domain vocabulary
# is hardcoded — both signals are generic.
ROUTE_RULE_MIN_OVERLAP = 4
US_PATENT_NUMBER_RE = re.compile(r"\b\d{8}\b")
# Retrieval-intent pre-check: "获取/下载/查看...文档/档案" asks for the
# DOCUMENTS, not an analysis — the document-list tools in the ReAct loop
# answer it, so the analysis long task must never hijack it (observed:
# "我想要获取US9019058B2的审查档案" routed into the prosecution task and
# the user got a patent list instead of the document list).  Both the
# verb and the object must appear; "查看审查历史" keeps the 分析 verb
# missing so it still routes to the analysis task.
RETRIEVAL_VERBS = ("获取", "下载", "查看", "列出", "导出", "拿",
                   "get", "download", "view", "list", "fetch", "retrieve")
RETRIEVAL_OBJECTS = ("文档", "档案", "文件", "清单", "目录", "列表",
                     "document", "file", "docket")


def _is_retrieval_request(query: str) -> bool:
    """True when the query asks to retrieve/obtain documents — the
    document-list tools' job, never the analysis long task's."""
    text = str(query or "").lower()
    has_verb = any(v in text for v in RETRIEVAL_VERBS)
    has_object = any(o in text for o in RETRIEVAL_OBJECTS)
    return has_verb and has_object


def _common_substring_len(a: str, b: str) -> int:
    """Length of the longest common CONTIGUOUS substring of a and b."""
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    best = 0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > best:
                    best = dp[i][j]
    return best


async def _match_long_task_intent(agent, query: str, entries: list,
                                  lang: str) -> Optional[ToolEntry]:
    """Deterministic long-task routing.

    The LLM freely choosing the long-task tool from the bound list is
    unreliable — observed in production: a prosecution-history request
    ("分析专利 11701773 的审查历史") went down the USPTO keyword ladder
    for minutes with the long-task tool bound, because the search
    discipline prompt steered it into searching.  When the request
    clearly matches a type-3 knowledge item's question, trigger the long
    task directly instead.

    One small flash-LLM classification call (no embedding, no context
    dump — just the item questions).  Never raises: any failure returns
    None and the request falls through to the normal loop.  Returns the
    matched ToolEntry or None.
    """
    if not LONG_TASK_ROUTE_ENABLED or not entries:
        return None
    query_text = str(query or "").strip()
    # Retrieval requests ("获取...档案/文档") are the document-list tools'
    # job — the analysis long task must never answer them.  Determined
    # BEFORE the LLM call so the classifier's "宁可命中不可漏判" bias
    # cannot hijack a document download.
    if _is_retrieval_request(query_text):
        return None
    provider = _get_flash_provider(agent)
    if provider is None:
        return None
    lines = []
    for idx, entry in enumerate(entries):
        question = _parse_bilingual_question(
            getattr(entry.knowledge, "question", "") or "", lang)
        lines.append(f"{idx}. {question}")
    system = (
        "你是任务路由分类器。下面列出可用的后台分析长任务及其触发条件。"
        "判断用户请求是否明确命中其中一个任务：请求对该任务所指对象（如某专利）"
        "提出了该任务所描述的分析需求，且请求包含任务的核心意图（如审查历史、"
        "同族分析等任务 question 中的意图表述）。"
        "只要请求包含任务的核心意图并针对该任务的适用对象，就必须返回该任务序号"
        "（宁可命中不可漏判）；只有请求与所有任务都明显无关时才返回 null。"
        "注意：获取、下载、查看、导出文档/档案/文件类请求不是分析需求（用户要的是"
        "原始文件，由文档列表工具提供），一律返回 null。"
        "只输出 JSON：{\"match\": 序号或 null}\n\n任务列表：\n"
        + "\n".join(lines)
    )
    try:
        result = await provider.complete_json(system, query_text)
    except Exception:
        return None
    idx = _parse_match_index(result)
    if idx is not None and 0 <= idx < len(entries):
        return entries[idx]
    # ── Rule fallback ──
    # The classifier missed a clear match before ({"match": null} on a
    # prosecution request); the request then burned minutes in the search
    # ladder.  When the query carries an 8-digit US patent number AND the
    # item question overlaps it by ≥ ROUTE_RULE_MIN_OVERLAP contiguous
    # chars, route anyway.  Both signals are generic; either missing →
    # stay on the normal loop.
    if not US_PATENT_NUMBER_RE.search(query_text):
        return None
    best_entry = None
    best_overlap = 0
    for entry in entries:
        question = _parse_bilingual_question(
            getattr(entry.knowledge, "question", "") or "", lang)
        overlap = _common_substring_len(query_text, question)
        if overlap >= ROUTE_RULE_MIN_OVERLAP and overlap > best_overlap:
            best_overlap = overlap
            best_entry = entry
    return best_entry


_ENVELOPE_KEYS = frozenset({"method", "body", "query", "path", "header"})


def _effective_query(args: dict) -> str:
    """Extract the query string that actually reaches the search API.

    Understands body.q, params-JSON, and top-level q/query shapes.  The
    generic first-string scan is the last resort and runs AFTER the
    explicit query slots — a non-query string field (the LLM sometimes
    echoes the user_id into args) must not shadow a real query.  Returns
    "" when no query can be recovered — callers then skip envelope
    building.
    """
    if not isinstance(args, dict):
        return ""
    body = args.get("body")
    if isinstance(body, dict):
        q = body.get("q")
        if isinstance(q, str) and q.strip():
            return q.strip()
    params = args.get("params")
    if isinstance(params, str) and params.strip():
        try:
            parsed = json.loads(params)
            if isinstance(parsed, dict):
                q = parsed.get("q")
                if isinstance(q, str) and q.strip():
                    return q.strip()
                return ""  # params dict carries no q — nothing to recover
        except (ValueError, TypeError):
            pass  # malformed params string — fall through to flat scan
    for key in ("q", "query"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in args.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _build_uspto_envelope(tool_info, q: str) -> dict:
    """Build a template-faithful request envelope for a USPTO search tool.

    Carries the tool template's body (fields list included), injects *q*
    the way the flat-merge does, ensures the relevance fields
    (cpcClassificationBag etc.) are requested, and preserves method /
    query / path / header from the template.  Never raises — on any
    template problem returns a minimal envelope with just q.
    """
    try:
        from sources.dynamic_tool_params import _coerce_json_object
        template = _coerce_json_object(tool_info.params, "tool_info.params") or {}
        body = dict(template.get("body") or {})
    except Exception:
        template, body = {}, {}
    body["q"] = q
    try:
        body = ensure_search_fields({"body": body})["body"]
    except Exception:
        pass
    # The tool template sorts by assignment-recorded date (newest
    # transactions first — mostly noise); USPTO's Elasticsearch accepts
    # _score, which ranks by query relevance and surfaces matching
    # patents regardless of age.  Env-overridable for safety.
    body["sort"] = [{"field": REACT_USPTO_SORT_FIELD, "order": "desc"}]
    return {
        "method": template.get("method", "POST"),
        "body": body,
        "query": template.get("query", {}),
        "path": template.get("path"),
        "header": template.get("header", {}),
    }


def _tool_invoke_payload(agent, params) -> dict:
    """Invoke payload matching DynamicBackendToolFunction's required
    fields (user_id / query_id / params).  The backend tool function
    ignores provided IDs in favour of the agent's stored values — they
    only exist to satisfy schema validation.

    `params` may be a request envelope dict or a raw params value; it is
    passed through to the backend tool's `params` field unchanged.
    """
    return {
        "user_id": getattr(agent, "_last_user_id", "") or "",
        "query_id": getattr(agent, "_last_query_id", "") or "",
        "params": params,
    }


async def _collect_search_pages(agent, entry, args, first_raw: list) -> list:
    """Fetch extra result pages for a USPTO search call and merge them.

    Stops early when total hits are exhausted, when a page returns items
    already seen (template ignored the offset), or after
    REACT_POOL_MAX_PAGES extra pages.  Never raises — failures return
    whatever was collected so far.
    """
    items = [c for c in build_candidates(first_raw or [])]
    q = _effective_query(args or {})
    if not q:
        return items
    seen_ids = {c["patent_id"] for c in items}
    total = getattr(agent, "_last_search_total", None)
    page_size = 50
    try:
        from sources.dynamic_tool_params import _coerce_json_object
        template = _coerce_json_object(entry.tool_info.params,
                                       "tool_info.params") or {}
        body = template.get("body") or {}
        page_size = int((body.get("pagination") or {}).get("limit", 50))
    except Exception:
        pass
    if isinstance(total, int) and total > REACT_POOL_MAX_TOTAL_PAGES:
        return items  # huge noisy pool — first page suffices, do not page
    offset = page_size
    for _page in range(REACT_POOL_MAX_PAGES - 1):
        if isinstance(total, int) and offset >= total:
            break
        envelope = _build_uspto_envelope(entry.tool_info, q)
        envelope["body"]["pagination"] = {"offset": offset, "limit": page_size}
        try:
            await asyncio.to_thread(entry.tool.invoke,
                                    _tool_invoke_payload(agent, envelope))
        except Exception:
            break
        raw = getattr(agent, "_pending_raw_items", None) or []
        fresh = [c for c in build_candidates(raw) if c["patent_id"] not in seen_ids]
        if not fresh:
            break  # offset ignored or universe exhausted
        for c in fresh:
            seen_ids.add(c["patent_id"])
        items.extend(fresh)
        offset += page_size
        if isinstance(total, int) and len(items) >= total:
            break
    return items


def _family_ids(c: dict) -> set:
    """Continuity application numbers of a candidate (parent + child).

    Direct-family linkage only: sharing a parent/child application
    number means the two records describe the same invention chain —
    the strongest available evidence that they belong together, without
    any title/text comparison.  Records without continuity data (e.g.
    bare patent numbers from the CPC index) return an empty set and can
    never be linked.
    """
    ids: set = set()
    raw = c.get("_raw") if isinstance(c, dict) else None
    if not isinstance(raw, dict):
        return ids
    for key in ("parentContinuityBag", "childContinuityBag"):
        bag = raw.get(key)
        if not isinstance(bag, list):
            continue
        for entry in bag:
            if not isinstance(entry, dict):
                continue
            for k in ("parentApplicationNumberText",
                      "childApplicationNumberText"):
                v = entry.get(k)
                if isinstance(v, str) and v.strip():
                    ids.add(v.strip())
    return ids


def _unscored_family_members(pool, seeds: list, budget: int) -> list:
    """Unscored direct-family members of the high-scoring seeds.

    A member is linked when its continuity ids intersect a seed's ids.
    Members are ordered by semantic prescore desc (the semantically
    closest wording variant gets scored first) and capped by *budget*.
    Returns a list of candidate dicts; never mutates the pool.
    """
    if not seeds:
        return []
    seed_ids = set()
    for s in seeds:
        seed_ids |= _family_ids(s)
    if not seed_ids:
        return []
    members = []
    for c in pool._by_id.values():
        if "relevance_score" in c:
            continue
        if _family_ids(c) & seed_ids:
            members.append(c)
    members.sort(key=lambda c: -(c.get("semantic_score") or 0.0))
    return members[:budget]


async def _rank_pending_pool(agent, candidates, lang,
                             apply_rerank: bool = True) -> Tuple[list, str]:
    """Merge collected candidate dicts into the turn's SearchPool, score
    new arrivals against the user's question, and return (ranked
    candidates, note).

    The pool lives on the agent for the whole request (created lazily;
    create_agent resets it per request).  *apply_rerank* gates the
    semantic rerank pass — internal merge calls (auto second round)
    pass False so the final merged pool is reranked exactly once by the
    outermost call.
    """
    pool = getattr(agent, "_search_pool", None)
    if pool is None:
        pool = SearchPool(getattr(agent, "_last_user_prompt", "") or "")
        agent._search_pool = pool
    new_cands = pool.add_from_candidates(candidates)
    # Dead candidates are never scored and sink in ranking — filter them
    # out before slicing so they cannot crowd live candidates out of the
    # per-call scoring head.
    live = [c for c in new_cands if not is_dead_status(c.get("status"))]
    head = live[:SCORE_PER_CALL]
    if PRESCORE_ENABLED and len(live) > SCORE_PER_CALL:
        # Two-stage scoring: bge-m3 prescores the whole batch in one
        # embedding call, then flash scores only the semantic head —
        # the LLM budget lands on the semantically closest candidates
        # instead of the newest slice.
        sem_map = await semantic_scores_batch(pool.query, live)
        if sem_map:
            for c in live:
                if c["patent_id"] in sem_map:
                    c["semantic_score"] = sem_map[c["patent_id"]]
            head = sorted(
                live,
                key=lambda c: -(c.get("semantic_score") or 0.0)
            )[:SCORE_PER_CALL]
    _score_start = time.monotonic()
    await _agent_status(agent,
        f"正在评估 {len(head)} 条候选专利与您问题的相关度..."
        if lang == "zh" else
        f"Scoring {len(head)} candidate patents against your question...")
    try:
        from sources.long_task.technical_interpretation import (
            format_interpretation_rubric,
        )
        # Grounded interpretation (post-retrieval) wins over the
        # pre-retrieval one once it exists: its players/lines are
        # data-driven.
        _grounded = getattr(agent, "_grounded_interpretation", None)
        _rubric = format_interpretation_rubric(
            _grounded or getattr(agent, "_search_interpretation", None))
    except Exception:
        _rubric = ""
    scored = await score_candidates_concurrent(
        head, pool.query,
        _get_flash_provider(agent) or getattr(agent, "llm", None),
        rubric=_rubric)
    _glog = getattr(agent, "logger", None)
    if _glog is not None:
        _glog.info(
            f"relevance scoring — candidates={len(head)} scored={scored} "
            f"elapsed={round(time.monotonic() - _score_start, 1)}s"
        )
    # Family scoring: a high-scoring seed lifts its direct-family members
    # into the Flash scoring budget even when their own titles scored
    # low in the prescore — same invention, different wording.
    # The probe line logs every round (even seeds=0/members=0) so a
    # silently-disabled mechanism is visible in general_agent.log.
    _family_scored = 0
    if FAMILY_SCORE_ENABLED:
        try:
            seeds = [
                c for c in pool._by_id.values()
                if isinstance(c.get("relevance_score"), (int, float))
                and c["relevance_score"] >= FAMILY_SEED_MIN
            ]
            members = _unscored_family_members(pool, seeds,
                                               FAMILY_SCORE_BUDGET)
            if _glog is not None:
                _glog.info(
                    f"family scoring probe — seeds={len(seeds)} "
                    f"members={len(members)} enabled={FAMILY_SCORE_ENABLED}"
                )
            if members:
                await _agent_status(agent,
                    "正在评估同族专利的相关性..." if lang == "zh"
                    else "Scoring family-member patents...")
                _family_scored = await score_candidates_concurrent(
                    members, pool.query,
                    _get_flash_provider(agent) or getattr(agent, "llm", None),
                    rubric=_rubric)
                if _glog is not None:
                    _glog.info(
                        f"family scoring — seeds={len(seeds)} "
                        f"members={len(members)} scored={_family_scored}"
                    )
        except Exception as exc:
            if _glog is not None:
                _glog.info(
                    f"family scoring error — {type(exc).__name__}: {exc}"
                )
    pool.prune()
    ranked = pool.ranked(MAX_PATENT_LIST_ITEMS)
    rerank_note = ""
    if apply_rerank and RERANK_ENABLED and len(ranked) > 1:
        from sources.long_task.semantic_rerank import (
            RERANK_TOP_K, RERANK_ALPHA, rerank_candidates,
        )
        ranked = await rerank_candidates(
            pool.query, ranked, RERANK_TOP_K, RERANK_ALPHA)
        rerank_note = (", semantic rerank applied" if lang == "en"
                       else "，语义重排已应用")
        _glog = getattr(agent, "logger", None)
        if _glog is not None:
            _glog.info(
                f"semantic rerank applied — candidates={len(ranked)}")
    if lang == "en":
        note = f"relevance-ranked — pool {len(pool)}, scored {scored} new{rerank_note}"
    else:
        note = f"已按相关度排序（池共 {len(pool)} 条、本次新评分 {scored} 条{rerank_note}）"
    note = await _maybe_append_missing_directions(agent, ranked, note, lang)
    return ranked, note


async def _maybe_append_missing_directions(agent, ranked: list, note: str,
                                           lang: str) -> str:
    """Infer missing technical directions after a scoring round and store
    them on the agent for the auto second round.

    Fires at most once per turn and only when the ranked pool holds at
    least MISSING_DIR_MIN_CANDIDATES candidates with a best relevance
    score >= MISSING_DIR_MIN_SCORE — a noise pool must not seed the
    queries.  The note itself is never mutated here: the caller's
    auto-round decides whether to execute the queries or present them as
    suggestions.  Never raises.
    """
    if getattr(agent, "_missing_dir_done", False):
        return note
    if len(ranked) < MISSING_DIR_MIN_CANDIDATES:
        return note
    best = max((c.get("relevance_score") or -1) for c in ranked)
    if best < MISSING_DIR_MIN_SCORE:
        return note
    titles = [c.get("title") for c in ranked[:8] if c.get("title")]
    if not titles:
        return note
    agent._missing_dir_done = True
    from sources.long_task.search_query_builder import (
        build_missing_direction_queries,
    )
    provider = _get_flash_provider(agent) or getattr(agent, "llm", None)
    # CPC hints were matched once in create_agent (cpc_semantic.log
    # records every round); reuse them here.
    cpc_hints = getattr(agent, "_cpc_hints", None) or None
    if CPC_EXPANSION_ENABLED and not cpc_hints:
        _glog = getattr(agent, "logger", None)
        if _glog is not None:
            _glog.warning(
                "cpc expansion enabled but no CPC matches — "
                "check data/cpc titles json and vector cache "
                "(scripts/build_cpc_vectors.py)")
    queries = await build_missing_direction_queries(
        getattr(agent, "_last_user_prompt", "") or "", titles, provider,
        cpc_hints=cpc_hints)
    if queries:
        agent._missing_dir_queries = queries
    return note


async def _invoke_and_merge(agent, entry, q: str, lang) -> Optional[Tuple[list, str, int]]:
    """Invoke one query through the search tool and merge+score its
    results into the pool.

    Returns (ranked, ranking_note, live_gained) — with ranked=[] and
    live=0 when the response carried nothing parseable.  Returns None
    when the invoke itself failed.  Never raises.  No rerank here: the
    outermost merge call reranks the final pool exactly once.
    """
    try:
        envelope = _build_uspto_envelope(entry.tool_info, q)
        await asyncio.to_thread(
            entry.tool.invoke, _tool_invoke_payload(agent, envelope))
    except Exception:
        return None
    raw = getattr(agent, "_pending_raw_items", None) or []
    if not raw:
        return [], "", 0
    try:
        collected = await _collect_search_pages(agent, entry, {"q": q}, raw)
    except Exception:
        collected = []
    live = len([c for c in collected
                if not is_dead_status(c.get("status"))])
    ranked, ranking_note = await _rank_pending_pool(
        agent, collected, lang, apply_rerank=False)
    return ranked, ranking_note, live


async def _auto_second_round(agent, entry, args, ranked: list, note: str,
                             lang: str) -> Tuple[list, str]:
    """Execute the missing-direction queries as a system-driven second
    round instead of leaving them to the agent's discretion.

    At most once per turn, at most REACT_AUTO_ROUND_MAX_QUERIES queries,
    each capped at the first page (the huge-total guard still applies).
    New candidates merge into the pool and get scored.  Never raises —
    on any failure the queries are presented as suggestions instead.
    """
    queries = getattr(agent, "_missing_dir_queries", None) or []
    if not queries or getattr(agent, "_auto_round_done", False):
        return ranked, note
    agent._auto_round_done = True
    new_total = 0
    for q in queries[:REACT_AUTO_ROUND_MAX_QUERIES]:
        merged = await _invoke_and_merge(agent, entry, q, lang)
        if merged is None:
            break
        ranked, note, live = merged
        if not ranked and live <= 0:
            continue
        new_total += live
    if new_total > 0:
        if lang == "en":
            executed = (
                f"\n\nAuto-executed supplementary queries (merged "
                f"{new_total} new candidates into the pool):\n"
            )
        else:
            executed = (
                f"\n\n已自动执行补充检索式（并入 {new_total} 条新候选）：\n"
            )
        return ranked, note + executed + _query_lines(queries)
    # Nothing was gained by executing — fall back to suggestion mode so
    # the agent can decide whether the queries are worth another round.
    return ranked, note + _format_feedback_note(queries, lang, kind="missing")


async def _auto_ladder_round(agent, entry, lang) -> Optional[Tuple[list, str, str]]:
    """Execute the untried ladder queries when a search leaves nothing
    displayable, so the ladder is exhausted even if the agent concludes
    early (observed: "no results" with half the ladder untried despite
    the zero-hit nudge).

    Bounded: AUTO_LADDER_BATCH per observation, AUTO_LADDER_MAX per
    request.  Executed queries are recorded as tried so the nudge lists
    stay accurate.  Returns (ranked, ranking_note, ladder_note) when at
    least one query executed; None when there was nothing to run.
    ladder_note says how many LIVE candidates landed (dead-only gains do
    not count).  Never raises.
    """
    if not _relevance_pool_applies_tool(agent, entry.tool_info):
        return None
    used = getattr(agent, "_auto_ladder_used", 0) or 0
    if used >= AUTO_LADDER_MAX:
        return None
    queries = (getattr(agent, "_search_rewrite", None) or {}).get("queries") or []
    tried = getattr(agent, "_tried_queries", None) or []
    untried = [q for q in queries if q not in tried]
    if not untried:
        return None
    take = untried[:min(AUTO_LADDER_BATCH, AUTO_LADDER_MAX - used)]
    gained = 0
    executed: list = []
    ranked: list = []
    ranking_note = ""
    for q in take:
        agent._auto_ladder_used = used + 1
        used += 1
        merged = await _invoke_and_merge(agent, entry, q, lang)
        if merged is None:
            break
        executed.append(q)
        if q not in tried:
            tried.append(q)
        ranked, ranking_note, live = merged
        gained += live
    if not executed:
        return None
    # The internal ranking may have inferred missing-direction queries
    # (CPC language included) — execute them here too; execute_action's
    # applies branch never runs for the zero-hit observations that
    # trigger this path, so they would otherwise sit unused.
    ranked, ranking_note = await _auto_second_round(
        agent, entry, {"q": executed[-1]}, ranked, ranking_note, lang)
    if lang == "en":
        merged = (f"(merged {gained} new candidates into the pool)"
                  if gained > 0 else "(no live hits)")
        ladder_note = (f"\n\nAuto-executed untried ladder queries "
                       f"{merged}:\n" + _query_lines(executed))
    else:
        merged = (f"（并入 {gained} 条新候选）" if gained > 0
                  else "（均无有效命中）")
        ladder_note = (f"\n\n已自动执行未尝试的阶梯检索式{merged}：\n"
                       + _query_lines(executed))
    return ranked, ranking_note, ladder_note


async def _auto_feedback_round(agent, entry, lang) -> Optional[Tuple[list, str, str]]:
    """Execute the low-hit feedback's refined queries system-side.

    The feedback queries are distilled from the pool's hit titles (the
    domain's own vocabulary) — executing them is the only guarantee they
    run at all: the agent has been observed answering with the
    suggestions ignored.  Once per request, at most AUTO_FEEDBACK_MAX
    queries.  Returns (ranked, ranking_note, feedback_note) when at
    least one query executed; None otherwise.  Never raises.
    """
    if getattr(agent, "_auto_feedback_done", False):
        return None
    queries = getattr(agent, "_feedback_queries", None) or []
    if not queries:
        return None
    agent._auto_feedback_done = True
    gained = 0
    executed: list = []
    ranked: list = []
    ranking_note = ""
    for q in queries[:AUTO_FEEDBACK_MAX]:
        merged = await _invoke_and_merge(agent, entry, q, lang)
        if merged is None:
            break
        executed.append(q)
        ranked, ranking_note, live = merged
        gained += live
    if not executed:
        return None
    if lang == "en":
        merged = (f"(merged {gained} new candidates into the pool)"
                  if gained > 0 else "(no live hits)")
        fb_note = (f"\n\nAuto-executed refined-query feedback "
                   f"{merged}:\n" + _query_lines(executed))
    else:
        merged = (f"（并入 {gained} 条新候选）" if gained > 0
                  else "（均无有效命中）")
        fb_note = (f"\n\n已自动执行建议检索式{merged}：\n"
                   + _query_lines(executed))
    return ranked, ranking_note, fb_note


async def _grounded_synthesis_round(agent, entry, lang) -> Optional[Tuple[list, str, str]]:
    """Post-retrieval grounded synthesis with loop feedback.

    Once per request: when the scored pool clears GROUNDED_MIN, cluster
    the scored head into data-driven dimensions/players (Flash), store
    the grounded interpretation (rubric upgrades to real signals) and
    its supplementary CPC codes (recall expansion widens), then
    auto-execute its supplementary queries into the pool — mirroring
    the auto-feedback round.  The probe line logs every request (even
    skipped) so a silent path stays visible.  Fires at most once per
    request — the flag burns only when a synthesis actually runs, so an
    early empty pool never wastes the single shot.  Returns
    (ranked, ranking_note, grounded_note) when queries executed; None
    otherwise.  Never raises.
    """
    if getattr(agent, "_grounded_done", False):
        return None
    from sources.long_task.grounded_interpretation import (
        GROUNDED_ENABLED, synthesize_grounded,
    )
    pool = getattr(agent, "_search_pool", None)
    _glog = getattr(agent, "logger", None)
    if pool is None:
        if _glog is not None:
            _glog.info(
                "grounded_interpretation probe — pool=0 scored=0 "
                f"trigger={GROUNDED_ENABLED}")
        return None
    scored = [
        c for c in pool._by_id.values()
        if isinstance(c.get("relevance_score"), (int, float))
    ]
    if _glog is not None:
        _glog.info(
            f"grounded_interpretation probe — pool={len(pool)} "
            f"scored={len(scored)} trigger={GROUNDED_ENABLED}")
    if not GROUNDED_ENABLED or len(scored) < GROUNDED_MIN \
            or len(pool) < GROUNDED_POOL_MIN:
        return None
    # The single-shot flag burns only now — the pool cleared the
    # minimum and synthesis is actually about to run.  An early empty
    # or below-min pool (common on the first round) must not waste the
    # one shot before recall expansion grows the pool.
    agent._grounded_done = True
    top = sorted(
        scored, key=lambda c: -(c.get("relevance_score") or 0))[:GROUNDED_HEAD]
    await _agent_status(agent,
        "正在归纳检索结果的技术主线..." if lang == "zh"
        else "Summarizing technical themes from the results...")
    try:
        grounded = await synthesize_grounded(
            pool.query, top,
            pre_interp=getattr(agent, "_search_interpretation", None),
            cpc_hints=getattr(agent, "_cpc_hints", None))
    except Exception:
        grounded = None
    if not grounded:
        return None
    agent._grounded_interpretation = grounded
    agent._grounded_cpc = list(
        grounded.get("supplementary_cpc") or [])[:RECALL_MAX_CPC]
    lines = [str(d.get("name") or "") for d in
             (grounded.get("dimensions") or [])[:3]]
    players = ", ".join(str(p) for p in (grounded.get("players") or [])[:5])
    if _glog is not None:
        _glog.info(
            f"grounded_interpretation — lines={lines}"
            + (f" | players={players}" if players else ""))
    queries = [q for q in
               (grounded.get("supplementary_queries") or [])[:AUTO_FEEDBACK_MAX]
               if q]
    if not queries:
        return None
    executed: list = []
    gained = 0
    ranked: list = []
    ranking_note = ""
    for q in queries:
        merged = await _invoke_and_merge(agent, entry, q, lang)
        if merged is None:
            break
        executed.append(q)
        ranked, ranking_note, live = merged
        gained += live
    if not executed:
        return None
    if lang == "en":
        merged_note = (f"(merged {gained} new candidates)" if gained > 0
                       else "(no live hits)")
        grounded_note = (f"\n\nAuto-executed grounded queries {merged_note}:\n"
                         + _query_lines(executed))
    else:
        merged_note = (f"（并入 {gained} 条新候选）" if gained > 0
                       else "（均无有效命中）")
        grounded_note = (f"\n\n已自动执行接地解读补检索式{merged_note}：\n"
                         + _query_lines(executed))
    return ranked, ranking_note, grounded_note


async def _recall_expansion_round(agent, entry, lang) -> Optional[Tuple[list, str, str]]:
    """System-driven recall expansion (citation/family + CPC routes).

    Once per request: collect family numbers from the pool candidates'
    continuity bags and the matched CPC codes, fetch their records via
    the recall transports, merge the new candidates into the pool and
    score them.  Missing-direction queries inferred during the internal
    ranking are executed too (mirrors _auto_ladder_round).  Returns
    (ranked, ranking_note, recall_note) when new live candidates
    landed; None otherwise.  Never raises.
    """
    if getattr(agent, "_recall_done", False):
        return None
    pool = getattr(agent, "_search_pool", None)
    if pool is None:
        return None
    candidates = pool.ranked(RECALL_POOL_HEAD)
    if not candidates:
        return None
    refs = collect_family_refs(candidates)
    grounded_codes = [
        str(c).strip().upper()
        for c in (getattr(agent, "_grounded_cpc", None) or [])
        if str(c).strip()]
    codes = [str(h.get("code", "")).strip() for h in
             (getattr(agent, "_cpc_hints", None) or [])
             if isinstance(h, dict) and h.get("code")]
    codes = (codes + grounded_codes)[:RECALL_MAX_CPC]
    if not (refs["patents"] or refs["applications"] or codes):
        return None
    agent._recall_done = True
    await _agent_status(agent,
        "正在扩展相关专利族与分类..." if lang == "zh"
        else "Expanding related patent families and classes...")
    records: list = []
    if refs["patents"] or refs["applications"]:
        try:
            records = await asyncio.to_thread(
                fetch_by_numbers,
                refs["patents"] + refs["applications"])
        except Exception:
            records = []
    if codes:
        try:
            records = records + await asyncio.to_thread(fetch_by_cpc, codes)
        except Exception:
            pass
    known = {c["patent_id"] for c in candidates}
    fresh = [c for c in records_to_candidates(records)
             if c["patent_id"] not in known]
    live = [c for c in fresh if not is_dead_status(c.get("status"))]
    if not live:
        return None
    # The pool scores only the first SCORE_PER_CALL of each merge
    # and prunes the rest — pass a spread-ordered batch so the
    # scored head represents the whole recall window instead of just
    # the newest slice (the deep end of the CPC sampling is where
    # established multi-year-old grants sit).
    stride = max(1, len(fresh) // max(1, SCORE_PER_CALL))
    spread_head = fresh[::stride][:SCORE_PER_CALL]
    spread_ids = {c["patent_id"] for c in spread_head}
    ordered = spread_head + [c for c in fresh
                             if c["patent_id"] not in spread_ids]
    ranked, ranking_note = await _rank_pending_pool(
        agent, ordered, lang, apply_rerank=False)
    ranked, ranking_note = await _auto_second_round(
        agent, entry, {"q": ""}, ranked, ranking_note, lang)
    # Grounded synthesis gets its reliable trigger here: the recall
    # candidates are already merged and scored (pool at full scale),
    # while the main-path trigger may have missed its window (the LLM
    # often stops calling the search tool after recall lands).  When
    # its supplementary queries land new candidates, the fresher
    # ranking replaces this round's.
    grounded_result = await _grounded_synthesis_round(agent, entry, lang)
    grounded_note = ""
    if grounded_result is not None:
        g_ranked, g_ranking_note, g_note = grounded_result
        if g_ranked:
            ranked = g_ranked
            ranking_note = g_ranking_note
        grounded_note = g_note
    if lang == "en":
        recall_note = (f"\n\nRecall expansion (family/CPC) merged "
                       f"{len(live)} new candidates into the pool."
                       + grounded_note)
    else:
        recall_note = (f"\n\n已自动执行分类/引文扩展检索"
                       f"（并入 {len(live)} 条新候选）。"
                       + grounded_note)
    return ranked, ranking_note, recall_note


def _query_lines(queries: list) -> str:
    """Render a bare numbered query list (no guidance header)."""
    return "\n".join(f"{i}. {q}" for i, q in enumerate(queries, start=1))


def _append_untried_ladder_note(agent, text: str, lang: str) -> str:
    """On zero hits, list the ladder queries that have not been tried yet
    so the agent substitutes vocabulary instead of concluding the API is
    broken.  Pure: never mutates *agent*, only reads the rewrite cache
    and the tried-query log."""
    queries = (getattr(agent, "_search_rewrite", None) or {}).get("queries") or []
    if not queries:
        return text
    tried = getattr(agent, "_tried_queries", None) or []
    untried = [q for q in queries if q not in tried][:3]
    if not untried:
        return text
    if lang == "en":
        header = ("No displayable results — untried ladder queries "
                  "(substitute vocabulary before loosening; adjacent "
                  "carrier-term variants first):")
    else:
        header = ("本次检索无可展示的有效结果（0 命中或均为失效专利）——"
                  "以下阶梯检索式尚未尝试（请先替换用词再放宽；优先取用"
                  "相邻的载体词版）：")
    return text + f"\n\n{header}\n" + _query_lines(untried)


def _format_feedback_note(queries: list, lang: str, kind: str = "refined") -> str:
    """Render query suggestions for the observation text.

    kind="refined" — title-extracted refinements (low-hit feedback);
    kind="missing" — inferred missing technical directions (post-scoring).
    """
    if not queries:
        return ""
    if kind == "missing":
        if lang == "en":
            header = ("\n\nSupplementary queries (inferred missing "
                      "technical directions for the current pool — try "
                      "these first; use them only for your search calls "
                      "and do not reproduce raw query syntax in the "
                      "final answer):\n")
        else:
            header = ("\n\n补充检索式（基于当前池推断的缺失技术方向，"
                      "可优先尝试；这些仅供你调整检索用，"
                      "回答中不要原样复述检索式语法）：\n")
    else:
        if lang == "en":
            header = ("\n\nSuggested refined queries (extracted from hit "
                      "titles — try these before loosening; use them only "
                      "for your search calls and do not reproduce raw query "
                      "syntax in the final answer):\n")
        else:
            header = ("\n\n建议检索式（基于已命中专利标题提炼，"
                      "可优先尝试后再放宽；这些仅供你调整检索用，"
                      "回答中不要原样复述检索式语法）：\n")
    lines = [header]
    for i, q in enumerate(queries, start=1):
        lines.append(f"{i}. {q}")
    return "\n".join(lines)


async def _maybe_append_feedback(agent, text: str, total, lang: str) -> str:
    """Append title-based query suggestions to a low-hit observation.

    Fires at most once per turn: the first search with fewer than
    LOW_HIT_FEEDBACK_THRESHOLD hits triggers one Flash call that
    distills the pool's hit titles into refined queries.  Never raises
    and never mutates *text* on failure.
    """
    if not isinstance(total, int) or total >= LOW_HIT_FEEDBACK_THRESHOLD:
        return text
    if getattr(agent, "_ladder_capped", False):
        # The cap note demands tightening — refined-query suggestions
        # in the same observation would steer the agent two ways.
        return text
    if getattr(agent, "_feedback_done", False):
        return text
    pool = getattr(agent, "_search_pool", None)
    ranked = pool.ranked(20) if pool is not None else []
    titles = [c.get("title") for c in ranked if c.get("title")][:10]
    if not titles:
        return text
    # Mark attempted only once a real feedback call is about to happen —
    # an empty-titles skip must not burn the one shot for a later search.
    agent._feedback_done = True
    from sources.long_task.search_query_builder import build_feedback_queries
    provider = _get_flash_provider(agent) or getattr(agent, "llm", None)
    queries = await build_feedback_queries(
        getattr(agent, "_last_user_prompt", "") or "", titles, provider,
        cpc_hints=getattr(agent, "_cpc_hints", None) or None)
    if queries:
        # Store them so _auto_feedback_round can execute them — the
        # suggestion text alone has been observed to be ignored.
        agent._feedback_queries = queries
    return text + _format_feedback_note(queries, lang)


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


def _is_session_sentinel(value, agent) -> bool:
    """True when the LLM pasted a session ID (user_id / query_id) into
    the query slot instead of a search expression — observed in
    production logs (q filled with the user_id)."""
    text = str(value or "").strip()
    if not text:
        return False
    uid = str(getattr(agent, "_last_user_id", "") or "")
    qid = str(getattr(agent, "_last_query_id", "") or "")
    return text == uid or text == qid


async def _maybe_rewrite_search_query(agent, tool_info, args) -> dict:
    """Inject the tightest ladder query ONLY when the q slot is absent.

    v4 semantics: the LLM owns q.  An explicit non-empty q the LLM passed
    (its own adaptation — loosened, tightened, or a ladder variant) is
    always respected.  The deterministic ladder (built in create_agent)
    fills in only when the q slot is missing or empty.  Applies only to
    backend (push=2) keyword search tools; every failure keeps the
    original args untouched.
    """
    if getattr(tool_info, "push", None) != 2 or not is_keyword_search_tool(tool_info):
        return args
    cached = getattr(agent, "_search_rewrite", None)
    if cached is None:
        from sources.long_task.search_query_builder import build_search_queries
        try:
            cached = await build_search_queries(
                getattr(agent, "_last_user_prompt", "") or "", agent.llm,
            )
        except Exception:
            cached = {"queries": []}
        agent._search_rewrite = cached
    queries = (cached or {}).get("queries") or []
    if not queries:
        return args
    tightest = queries[0]
    out = dict(args or {})

    def _blank(value) -> bool:
        return not str(value or "").strip() or _is_session_sentinel(value, agent)

    if "q" in out:
        if _blank(out.get("q")):
            out["q"] = tightest
        return out
    if "query" in out:
        if _blank(out.get("query")):
            out["query"] = tightest
        return out
    if "params" in out:
        try:
            import json
            if isinstance(out["params"], str):
                p = json.loads(out["params"])
            elif isinstance(out["params"], dict):
                p = dict(out["params"])
            else:
                return args
        except (ValueError, TypeError):
            return args
        if "q" in p:
            if _blank(p.get("q")):
                p["q"] = tightest
        elif "query" in p:
            if _blank(p.get("query")):
                p["query"] = tightest
        else:
            p["q"] = tightest
        out["params"] = json.dumps(p, ensure_ascii=False)
        return out

    # Top-level args without any q/query/params slot: the LLM asked for a
    # search without specifying a query — inject the tightest ladder query.
    out["q"] = tightest
    return out


def _baiten_first_str(value) -> str:
    """First non-empty string from a scalar or list field value.

    The live gateway returns multi-valued fields (pa, in, ...) as lists
    (``["中山市澳多电子科技有限公司"]``) and scalars otherwise.
    """
    if isinstance(value, list):
        for item in value:
            text = str(item or "").strip()
            if text:
                return text
        return ""
    return str(value or "").strip()


def _baiten_results_to_candidates(body: dict) -> list:
    """Map Baiten search hit rows into candidate structures.

    Live-verified response shape (2026-08-26, real key):
    ``{total_hits, documents: [{field_values: {an, pn, pd, ti, pa[], ...},
    hl_field_values: {...}}]}``.  ``patent_id`` is the CN publication
    number (pn, e.g. CN112345678A), which the existing candidate
    consumers (_extract_patent_ids_from_items via patentNumber) handle
    natively.  The older SDK shapes (top-level fieldValues, camelCase
    fieldValues wrappers) are also tolerated.  Unknown shapes are
    skipped; never raises.
    """
    data = body.get("data")
    rows = None
    if isinstance(data, dict):
        rows = data.get("fieldValues")
    if rows is None:
        rows = body.get("fieldValues")
    if rows is None:
        rows = body.get("documents")
    candidates = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        for wrap_key in ("field_values", "fieldValues"):
            wrapped = row.get(wrap_key)
            if isinstance(wrapped, dict):
                row = wrapped
                break
        pn = _baiten_first_str(row.get("pn"))
        if not pn:
            continue
        candidates.append({
            "patent_id": pn,
            "source": "baiten",
            "title": _baiten_first_str(row.get("ti")),
            "pub_date": _baiten_first_str(row.get("pd")),
            "app_num": _baiten_first_str(row.get("an")),
            "apply_date": _baiten_first_str(row.get("ad")),
            "applicant": _baiten_first_str(row.get("pa")),
            "status": "",
            "grant_date": "",
            "patent_number": pn,
            "type_code": "",
            "cpc_codes": [],
            "_raw": row,
        })
    return candidates


def _normalize_uspto_items(items: list) -> list:
    """Lift the patent title to a top-level ``title`` key on USPTO items.

    The applications/search endpoint has returned the title under
    different names and locations across schema versions (inventionTitle
    vs titleOfInvention, top-level vs inside applicationMetaData —
    observed 2026-08-27: the artifact rows showed a blank title column
    while the data clearly carried titles).  The export/artifact pipeline
    maps a fixed ``title`` role, so normalize here instead of chasing the
    API's current shape.  Items already carrying a top-level title (or
    with no recognizable title field) pass through unchanged.
    """
    out = []
    for item in items or []:
        if not isinstance(item, dict):
            out.append(item)
            continue
        if any(isinstance(item.get(k), str) and item.get(k).strip()
               for k in ("title", "inventionTitle", "titleOfInvention")):
            out.append(item)
            continue
        normalized = dict(item)
        meta = item.get("applicationMetaData")
        if isinstance(meta, dict):
            for key in ("inventionTitle", "titleOfInvention"):
                value = meta.get(key)
                if isinstance(value, str) and value.strip():
                    normalized["title"] = value.strip()
                    break
        out.append(normalized)
    return out


async def _uspto_search_by_query(
    q: str, page: int = 1, page_size: int = 20,
) -> tuple[list, str]:
    """POST USPTO applications/search; returns (raw_items, note)."""
    try:
        from sources.http_outbound import outbound_http
        from sources.long_task.recall_sources import (
            USPTO_SEARCH_URL, RECALL_SEARCH_FIELDS,
        )
        import os as _os
        headers = {"Content-Type": "application/json"}
        uspto_key = _os.getenv("USPTO_API_KEY")
        if uspto_key:
            headers["X-API-Key"] = uspto_key
        body = {
            "q": q,
            "pagination": {
                "offset": max(page - 1, 0) * page_size,
                "limit": page_size,
            },
            "fields": RECALL_SEARCH_FIELDS,
            "sort": [{"field": "_score", "order": "desc"}],
        }
        response = await outbound_http.arequest(
            "POST", USPTO_SEARCH_URL, purpose="dual_patent_search",
            headers=headers, json=body, timeout=30,
        )
        if getattr(response, "status_code", 0) != 200:
            return [], f"USPTO HTTP {response.status_code}"
        data = response.json()
        items = _normalize_uspto_items(
            data.get("patentFileWrapperDataBag") or [])
        return items, f"USPTO {len(items)} hits"
    except Exception as exc:
        return [], f"USPTO failed: {exc}"


async def _enrich_baiten_law_status(client, candidates: list, glog) -> None:
    """Fill Baiten candidate ``status`` from the /openService/law gateway.

    One FLZT (法律状态) call per candidate, fired concurrently with a short
    per-call timeout so a slow gateway never blocks the result list; any
    failure degrades to the empty status the candidate already carries.
    Pure enrichment — never raises.
    """
    if not candidates:
        return

    async def _one(c: dict) -> None:
        app_num = str(c.get("app_num") or "").strip()
        if not app_num:
            return
        try:
            state = await asyncio.wait_for(
                client.query_law_state(app_num), timeout=5)
        except Exception as exc:
            if glog is not None:
                glog.warning(
                    f"baiten law status failed for {app_num}: {exc}")
            return
        law = str((state or {}).get("lawStatus") or "").strip()
        if law:
            c["status"] = law

    await asyncio.gather(*[_one(c) for c in candidates])


async def _baiten_search_by_query(
    q: str, page: int = 1, page_size: int = 20, agent=None,
) -> tuple[list, str]:
    """BaitenClient.search(source=15); returns (candidates, note).

    Any failure — missing key, wrong method path (unverified until the
    live gateway is smoke-tested), network error — degrades to an empty
    list so the parallel USPTO source is never blocked.  The note
    distinguishes a real zero (gateway returned no records) from a parse
    zero (records present but the candidate mapping dropped them), so a
    schema drift is never mistaken for an empty result set.
    """
    _glog = getattr(agent, "logger", None)
    try:
        from sources.baiten_client import (
            BaitenClient, summarize_search_response,
        )
        from sources.long_task.config import get_baiten_config

        cfg = get_baiten_config()
        if not cfg["app_key"] or not cfg["app_secret"]:
            if _glog is not None:
                _glog.warning(
                    "baiten_search — not configured "
                    "(BAITEN_APP_KEY/APP_SECRET)")
            return [], "Baiten not configured (BAITEN_APP_KEY/APP_SECRET)"
        client = BaitenClient(
            cfg["app_key"], cfg["app_secret"], cfg["gateway_url"])
        body = await client.search(
            q, page=page, page_size=page_size,
            api_level=cfg.get("api_level", "ONE"))
        summary = summarize_search_response(body)
        items = _baiten_results_to_candidates(body)
        if items:
            await _enrich_baiten_law_status(client, items, _glog)
        if _glog is not None:
            _glog.info(
                f"baiten_search_map — query={q[:60]!r} "
                f"rows={summary['rows']} candidates={len(items)}"
            )
        if summary["rows"] == 0:
            return items, "Baiten 0 hits (gateway 0 records)"
        if not items:
            return [], (
                f"Baiten 0 candidates (parsed from "
                f"{summary['rows']} records)"
            )
        return items, f"Baiten {len(items)} hits"
    except Exception as exc:
        if _glog is not None:
            _glog.warning(f"baiten_search — failed: {exc}")
        return [], f"Baiten failed: {exc}"


# ── Dual-source query resolution + preferred-source auto-ladder ─────────────

PATENT_AUTO_LADDER_BATCH = 2  # untried ladder queries auto-run per call
REACT_PATENT_AUTO_LADDER_MAX = int(os.getenv(
    "REACT_PATENT_AUTO_LADDER_MAX", "4"))


def _resolve_patent_queries(args, us_ladder, cn_ladder, agent,
                            dual: bool) -> tuple[str, str]:
    """Resolve the effective US/CN queries for one patent_search call.

    An explicit non-empty query the LLM passed is always respected; an
    absent, blank, or session-sentinel slot is auto-filled with the
    ladder's tightest query, so a dual-source call can never silently
    drop a source.  *dual* True fills both legs, False (CN-only tool)
    fills only the CN leg — US never runs unless the dual tool asked.
    Returns (us_q, cn_q); an empty ladder leaves its slot empty.
    """
    raw_us = str((args or {}).get("query_string_us") or "").strip()
    raw_cn = str((args or {}).get("query_string_cn") or "").strip()
    us_ladder = us_ladder or []
    cn_ladder = cn_ladder or []
    us_q = "" if not raw_us or _is_session_sentinel(raw_us, agent) else raw_us
    cn_q = "" if not raw_cn or _is_session_sentinel(raw_cn, agent) else raw_cn
    if not us_q and dual and us_ladder:
        us_q = us_ladder[0]
    if not cn_q and cn_ladder:
        cn_q = cn_ladder[0]
    return us_q, cn_q


def _item_patent_id(item) -> str:
    """Patent identifier used for cross-call dedup of pending raw items.

    Baiten candidates carry a flat ``patent_id`` (CN publication number);
    USPTO rows carry ``applicationNumberText`` top-level or nested under
    applicationMetaData.  Anything else returns "" and is kept as-is.
    """
    if not isinstance(item, dict):
        return ""
    for key in ("patent_id", "applicationNumberText"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    meta = item.get("applicationMetaData")
    if isinstance(meta, dict):
        for key in ("applicationNumberText", "patentNumber"):
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _merge_pending_items(existing, new_items) -> list:
    """Merge new patent candidates into the pending display list.

    The loop may call the built-in patent search several times per request
    (the ladder prompt walks the LLM down level by level).  A later call
    whose one source came back empty (404 with the auto-ladder budget
    exhausted) used to unconditionally overwrite ``_pending_raw_items`` and
    silently drop the earlier dual-source result.  Merging by patent id
    keeps every candidate the request has found — first occurrence wins,
    new items append.
    """
    merged = list(existing or [])
    seen = {_item_patent_id(item) for item in merged if _item_patent_id(item)}
    for item in new_items or []:
        pid = _item_patent_id(item)
        if pid:
            if pid in seen:
                continue
            seen.add(pid)
        merged.append(item)
    return merged


def _order_pending_for_lang(items: list, lang: str) -> list:
    """Group pending candidates CN-first for Chinese questions.

    The ladder guidance already puts the CN ladder first for zh users
    (strategy parity); the merged display list must match, so Chinese
    questions list Baiten patents before USPTO ones.  Stable within each
    group; other languages keep source order.
    """
    if lang != "zh":
        return items
    cn = [c for c in items
          if isinstance(c, dict) and c.get("source") == "baiten"]
    others = [c for c in items
              if not (isinstance(c, dict) and c.get("source") == "baiten")]
    return cn + others


def _cn_item_to_pool_candidate(item: dict) -> dict:
    """Map a flat Baiten candidate to the relevance-pool candidate shape.

    The pool consumes patent_id/title/applicant/status/filing_date/
    patent_number/type_code/cpc_codes/_raw; Baiten candidates carry most
    of these natively — filing_date derives from apply_date/pub_date.
    """
    return {
        "patent_id": str(item.get("patent_id") or ""),
        "title": str(item.get("title") or ""),
        "applicant": str(item.get("applicant") or ""),
        "status": str(item.get("status") or ""),
        "filing_date": str(item.get("apply_date")
                          or item.get("pub_date") or ""),
        "patent_number": str(item.get("patent_number") or ""),
        "type_code": str(item.get("type_code") or ""),
        "cpc_codes": item.get("cpc_codes") or [],
        "_raw": item,
    }


async def _rank_builtin_patent_pool(agent, items: list, lang: str) -> list:
    """Run built-in patent-search candidates through the relevance pool.

    The USPTO dynamic tools get Flash relevance scoring, semantic rerank,
    dead/design filtering and family dedupe via SearchPool; the built-in
    dual/single-source tool used to bypass the pipeline entirely, so CN
    patents surfaced in gateway order with no filtering (the observed
    precision gap vs USPTO, 2026-08-27).  Both sources are converted to
    pool candidates and ranked the same way.  Any failure degrades to the
    unranked list — ranking is an enhancement, never a hard dependency.
    """
    try:
        candidates: list = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("source") == "baiten":
                candidates.append(_cn_item_to_pool_candidate(item))
            else:
                # build_candidates reads applicationNumberText from BOTH the
                # top level and applicationMetaData — the USPTO API has
                # drifted between the two across schema versions (observed
                # 2026-09-01: 20 US hits for an RGB-LED question were all
                # nested, skipped by a top-level-only check, and the pool
                # ended up with a single CN candidate).  Items without any
                # pid yield [] and are skipped safely.
                candidates.extend(build_candidates([item]))
        if not candidates:
            return items
        ranked, _note = await _rank_pending_pool(agent, candidates, lang)
        raw = [c.get("_raw") for c in ranked]
        kept = [c for c in raw if c is not None]
        return kept or items
    except Exception:
        return items


async def _auto_run_patent_ladder(agent, ladder: list, search_fn, merged: list,
                                  notes: list, lang: str, source: str,
                                  page: int, page_size: int) -> int:
    """System-run untried ladder queries for one patent source.

    Triggered when the preferred source returned nothing displayable;
    bounded (PATENT_AUTO_LADDER_BATCH per call, REACT_PATENT_AUTO_LADDER_MAX
    per source).  The budget is tracked per source (``_patent_auto_used``
    is a {source: used} map) so the non-preferred source's auto-runs can
    never exhaust the budget the preferred source needs when IT hits zero
    (production incident 2026-08-29: US auto-runs in two earlier calls
    spent the shared per-request budget, so the CN preferred-source zero
    silently got no ladder and the call returned total=0).  Executed
    queries are recorded as tried so the zero-hit nudge stays accurate.
    Returns the number of candidates gained.
    """
    tried = getattr(agent, "_tried_queries", None)
    if tried is None:
        tried = agent._tried_queries = []
    untried = [q for q in ladder if q not in tried]
    if not untried:
        return 0
    used_map = getattr(agent, "_patent_auto_used", None)
    if not isinstance(used_map, dict):
        used_map = {}
    used = used_map.get(source, 0)
    take = untried[:min(PATENT_AUTO_LADDER_BATCH,
                        REACT_PATENT_AUTO_LADDER_MAX - used)]
    if not take:
        _glog = getattr(agent, "logger", None)
        if _glog is not None:
            _glog.warning(
                f"patent_search_auto_ladder — source={source} auto-ladder "
                f"budget exhausted (used={used}/"
                f"{REACT_PATENT_AUTO_LADDER_MAX}), {len(untried)} untried "
                f"ladder queries skipped"
            )
        return 0
    gained = 0
    executed: list = []
    for q in take:
        used_map[source] = used + 1
        used += 1
        agent._patent_auto_used = used_map
        try:
            items, note = await search_fn(q, page=page, page_size=page_size)
        except Exception as exc:
            notes.append(f"{source}: {exc}")
            continue
        executed.append(q)
        if q not in tried:
            tried.append(q)
        gained += len(items)
        merged.extend(items)
        if note:
            notes.append(note)
    if executed:
        _glog = getattr(agent, "logger", None)
        if _glog is not None:
            _glog.info(
                f"patent_search_auto_ladder — source={source} "
                f"queries={[q[:40] for q in executed]} gained={gained}"
            )
        label = "中国" if source == "cn" else "美国"
        notes.append(
            f"已自动补跑{label}专利阶梯式 {len(executed)} 条"
            f"（并入 {gained} 条候选）" if lang == "zh"
            else f"Auto-ran {len(executed)} {source.upper()} ladder "
                 f"queries ({gained} candidates merged)"
        )
    return gained


async def _run_patent_search(agent, args, lang: str, dual: bool = True) -> dict:
    """Built-in dual/single-source patent search for the loop.

    Runs the requested sources in parallel (``asyncio.gather`` with
    ``return_exceptions=True``): one source failing never blocks the
    other.  Missing query slots are auto-filled from the deterministic
    ladders (``_search_rewrite`` / ``_search_rewrite_cn``) so a
    dual-source call always runs BOTH legs.  When the preferred source
    (CN for Chinese questions, US for English — same strategy both
    sides) returns nothing, untried ladder queries are run system-side
    instead of leaving the outcome to the LLM's discretion.  Merged
    candidates land on ``agent._pending_raw_items`` (the same channel
    the dynamic search tools use) so the pool, patent-id extraction and
    result-page artifact paths all work unchanged.
    """
    us_ladder = ((getattr(agent, "_search_rewrite", None) or {})
                 .get("queries") or [])
    cn_ladder = ((getattr(agent, "_search_rewrite_cn", None) or {})
                 .get("queries") or [])
    us_q, cn_q = _resolve_patent_queries(
        args, us_ladder, cn_ladder, agent, dual=dual)
    if not us_q and not cn_q:
        return {"kind": "observation",
                "text": ("Error: provide at least query_string_us or "
                         "query_string_cn" if lang == "en"
                         else "Error: 至少需要 query_string_us 或 query_string_cn")}

    page = int((args or {}).get("page") or 1)
    page_size = int((args or {}).get("page_size") or 20)
    _glog = getattr(agent, "logger", None)
    if _glog is not None:
        _glog.info(
            f"patent_search_dual — lang={lang} dual={dual} "
            f"us_q={us_q[:60]!r} cn_q={cn_q[:60]!r}"
        )

    async def _uspto(q, page=1, page_size=20):
        return await _uspto_search_by_query(q, page=page, page_size=page_size)

    async def _baiten(q, page=1, page_size=20):
        return await _baiten_search_by_query(
            q, page=page, page_size=page_size, agent=agent)

    tasks = []
    if us_q:
        tasks.append(_uspto(us_q, page=page, page_size=page_size))
    if cn_q:
        tasks.append(_baiten(cn_q, page=page, page_size=page_size))
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Record the first-round queries as tried so the auto-ladder never
    # re-runs them and the zero-hit nudge lists only the untried ladder.
    tried = getattr(agent, "_tried_queries", None)
    if tried is None:
        tried = agent._tried_queries = []
    for q in (us_q, cn_q):
        if q and q not in tried:
            tried.append(q)

    merged: list = []
    notes: list = []
    for result in results:
        if isinstance(result, Exception):
            notes.append(f"{type(result).__name__}: {result}")
            continue
        items, note = result
        merged.extend(items)
        if note:
            notes.append(note)

    # Preferred source auto-ladder: when the source the language favours
    # (CN for zh, US for en — strategy parity, both sides share this
    # mechanism) returned nothing displayable, run untried ladder queries
    # system-side (prompt-level nudges do not work for weak models).
    # The NON-preferred source gets the same fallback afterwards (shared
    # per-request budget caps the total) — the user wants BOTH sides to
    # return results, and a single first-round 404 must not starve the
    # other source of its looser ladder forms.
    if not dual:
        preferred = "cn"
    elif lang == "zh":
        preferred = "cn"
    else:
        preferred = "us"

    def _source_cands(source: str) -> list:
        if source == "cn":
            return [c for c in merged
                    if isinstance(c, dict) and c.get("source") == "baiten"]
        return [c for c in merged
                if not (isinstance(c, dict) and c.get("source") == "baiten")]

    async def _run_for(source: str):
        q = cn_q if source == "cn" else us_q
        if not q or _source_cands(source):
            return
        ladder = cn_ladder if source == "cn" else us_ladder
        fn = _baiten if source == "cn" else _uspto
        await _auto_run_patent_ladder(
            agent, ladder, fn, merged, notes, lang, source,
            page, page_size)

    await _run_for(preferred)
    await _run_for("us" if preferred == "cn" else "cn")

    # Merge with anything already pending from earlier patent_search calls
    # in this request — a later narrower call (one source 404 with the
    # auto-ladder budget spent) must never discard the earlier complete
    # dual-source result (production incident 2026-08-27).  The merged
    # pool then rides the same relevance pipeline as the USPTO dynamic
    # tools (scoring / rerank / dead+design filter / family dedupe);
    # Chinese questions still list CN patents first, now relevance-ranked
    # within each group.
    pending = _merge_pending_items(
        getattr(agent, "_pending_raw_items", None), merged)
    ranked_pending = await _rank_builtin_patent_pool(agent, pending, lang)
    agent._pending_raw_items = _order_pending_for_lang(ranked_pending, lang)
    if _glog is not None:
        cn_hits = len([c for c in merged
                       if isinstance(c, dict) and c.get("source") == "baiten"])
        _glog.info(
            f"patent_search_result — us_hits={len(merged) - cn_hits} "
            f"cn_hits={cn_hits} total={len(merged)}"
        )

    digest = _items_digest(merged, lang=lang)
    if not digest:
        digest = ("No results from any source." if lang == "en"
                  else "两个数据源均未返回结果。")
    if notes:
        if _glog is not None:
            _glog.info("patent_search_notes — " + "; ".join(notes))
        # 数据来源/状态噪声 (USPTO HTTP 404 / Baiten N hits / 自动补跑阶梯)
        # 只进日志, 不拼进用户可见的流式 observation 文本 (2026-09-01)。
    return {"kind": "observation", "text": digest}


async def _run_patent_spec(agent, args, lang: str) -> dict:
    """Download one patent's specification and distill it into the loop."""
    patent_id = str((args or {}).get("patent_id") or "").strip()
    if not patent_id:
        return {"kind": "observation", "text": "Error: missing patent_id"}
    from sources.long_task.patent_distill import (
        distill_patent_spec, format_distilled, truncated_fallback,
    )
    from sources.long_task.uspto_download import download_uspto_patent_text

    text, binary = await download_uspto_patent_text(
        patent_id,
        spec_selector_provider=getattr(agent, "llm", None),
        logger=getattr(agent, "logger", None),
    )
    if not text:
        if binary is not None:
            return {"kind": "observation",
                    "text": "Error: 说明书为扫描件，暂无法自动提取文本分析"}
        return {"kind": "observation",
                "text": f"Error: 说明书下载失败（专利号 {patent_id}）"}
    query = getattr(agent, "_last_user_prompt", "") or ""
    distilled = await distill_patent_spec(text, query, agent.llm)
    if distilled:
        return {"kind": "observation", "text": format_distilled(distilled, lang)}
    return {"kind": "observation", "text": truncated_fallback(text)}


async def make_action_executor(agent, registry, push_filter=None):
    """Return the loop's execute_action closure."""
    user_id = getattr(agent, "_last_user_id", None)
    lang = getattr(agent, "_lang", "zh")

    async def execute_action(name: str, args: dict, round_no: int) -> dict:
        entry = registry.get(name)
        if entry is None:
            return {"kind": "observation", "text": f"Error: unknown tool '{name}'"}

        if entry.kind == "patent_spec":
            return await _run_patent_spec(agent, args, lang)

        if entry.kind == "patent_search":
            return await _run_patent_search(
                agent, args, lang,
                dual=(entry.name == DUAL_PATENT_SEARCH_TOOL_NAME))

        if entry.kind == "search":
            return await _run_search_knowledge(agent, registry, user_id, args, push_filter)

        if entry.kind == "long_task":
            # The loop terminates and core.py's existing long-task branch
            # handles classification + Celery submission.
            return {"kind": "long_task", "text": "",
                    "knowledge": entry.knowledge, "tool_info": entry.tool_info}

        # Per-number verification calls are capped per request — the LLM
        # was observed looping through 8+ one-by-one fetches (each
        # followed by a ~2.5s semantic rerank) without converging.  At
        # the cap, stop and nudge the LLM to answer from the pool.
        is_verify = is_identifying_number_tool(entry.tool_info)
        if is_verify:
            verify_count = (getattr(agent, "_verify_call_count", 0) or 0) + 1
            agent._verify_call_count = verify_count
            if verify_count > VERIFY_CALL_MAX:
                return {
                    "kind": "observation",
                    "text": ("Already verified enough candidate details; "
                             "stop fetching by number and answer from the "
                             "results on hand." if lang == "en"
                             else "已按编号核实了足够多的候选专利，"
                                  "请停止逐条查证，直接基于现有检索结果给出最终答案。")
                }

        try:
            args = await _maybe_rewrite_search_query(agent, entry.tool_info, args)
            pool_eligible = _relevance_pool_applies_tool(agent, entry.tool_info)
            invoke_args = args
            if pool_eligible and isinstance(args, dict) \
                    and not (_ENVELOPE_KEYS.intersection(args)):
                q = _effective_query(args)
                if q:
                    invoke_args = _tool_invoke_payload(
                        agent, _build_uspto_envelope(entry.tool_info, q))
            result = await asyncio.to_thread(entry.tool.invoke, invoke_args)
        except Exception as exc:
            return {"kind": "observation", "text": f"Error: {exc}"}

        # Record the query that actually reached the tool so zero-hit
        # observations can list the ladder variants still untried.
        q_used = _effective_query(args) if isinstance(args, dict) else ""
        if q_used:
            tried = getattr(agent, "_tried_queries", None)
            if tried is None:
                tried = agent._tried_queries = []
            if q_used not in tried:
                tried.append(q_used)

        # Keep the exact pairing used later by _stream_raw_items for
        # source inference and artifact building.
        agent.knowledgeTool = (entry.knowledge, entry.tool_info)

        pending = getattr(agent, "_pending_raw_items", None)
        if pending:
            _is_doc_list = is_documents_tool(entry.tool_info)
            applies = _relevance_pool_applies(agent, entry.tool_info, pending)
            _glog = getattr(agent, "logger", None)
            if _glog is not None:
                _glog.info(
                    "relevance_pool gate — "
                    f"tool_title={getattr(entry.tool_info, 'title', None)!r} "
                    f"push={getattr(entry.tool_info, 'push', None)!r} "
                    f"flag={RELEVANCE_RANK_ENABLED!r} "
                    f"parseable={len(build_candidates(pending))} "
                    f"applies={applies}"
                )
            if applies:
                collected = await _collect_search_pages(agent, entry, args, pending)
                # A one-by-one number verification adds a single candidate —
                # re-ranking the whole pool (~2.5s) for it is pure waste.
                # Skip the rerank on verification calls; the pool keeps its
                # last keyword-round ranking.
                ranked, note = await _rank_pending_pool(
                    agent, collected, lang,
                    apply_rerank=not is_identifying_number_tool(entry.tool_info))
                ranked, note = await _auto_second_round(
                    agent, entry, args, ranked, note, lang)
                shown = [c["_raw"] for c in ranked]
                agent._pending_raw_items = shown
                agent._search_ranked = True
                digest = _ranked_digest(ranked, lang=lang)
            else:
                shown, note = _cap_patent_list(entry.tool_info, pending, lang)
                pool = getattr(agent, "_search_pool", None)
                if pool is not None and not _is_doc_list:
                    # The tool function already wrote this legacy result
                    # into _pending_raw_items; restore the turn's ranked
                    # pool as the display list — the legacy result still
                    # feeds the observation digest below.  Document-list
                    # tools keep their own result: the documents of ONE
                    # application are the answer, not a search pool.
                    ranked = pool.ranked(MAX_PATENT_LIST_ITEMS)
                    agent._pending_raw_items = [c["_raw"] for c in ranked]
                else:
                    agent._pending_raw_items = shown
                digest = _items_digest(shown, lang=lang)
            total = getattr(agent, "_last_search_total", None)
            total_note = ""
            if isinstance(total, int):
                if not shown and total > 0:
                    # Every hit was filtered out as a dead patent — say
                    # so explicitly instead of showing a silent empty list.
                    total_note = (f", {total} total hits (all dead patents, filtered)"
                                  if lang == "en"
                                  else f"，总命中 {total} 条（均为失效专利，已过滤）")
                else:
                    total_note = (f", {total} total hits" if lang == "en"
                                  else f"，总命中 {total}")
            if lang == "en":
                text = (f"Search results ({len(shown)} records{total_note}, {note}):\n"
                        f"{digest}\n\n"
                        "The full list is displayed to the user.")
            else:
                text = (f"检索结果（{len(shown)} 条{total_note}，{note}）：\n"
                        f"{digest}\n\n"
                        "完整列表已展示给用户。")
            text = _apply_ladder_cap(agent, text, total, lang)
            if not shown or (isinstance(total, int) and total == 0):
                # No displayable results — zero hits, or every hit was
                # dead-filtered.  First execute the untried ladder
                # queries system-side (the agent cannot conclude early
                # while they remain untried), then point the agent at
                # whatever is still left.
                auto = await _auto_ladder_round(agent, entry, lang)
                if auto is not None:
                    ranked, ranking_note, ladder_note = auto
                    if ranked:
                        shown = [c["_raw"] for c in ranked]
                        agent._pending_raw_items = shown
                        agent._search_ranked = True
                        digest = _ranked_digest(ranked, lang=lang)
                        note = ranking_note + ladder_note
                        if lang == "en":
                            text = (f"Search results ({len(shown)} records, "
                                    f"{note}):\n{digest}\n\n"
                                    "The full list is displayed to the user.")
                        else:
                            text = (f"检索结果（{len(shown)} 条，{note}）：\n"
                                    f"{digest}\n\n"
                                    "完整列表已展示给用户。")
                    else:
                        # Nothing displayable either — keep the existing
                        # text (dead-filter note etc.) and append only the
                        # ladder outcome.
                        text = text.rstrip() + "\n" + ladder_note
                text = _append_untried_ladder_note(agent, text, lang)
            text = await _maybe_append_feedback(agent, text, total, lang)
            grounded = await _grounded_synthesis_round(agent, entry, lang)
            if grounded is not None:
                ranked, ranking_note, grounded_note = grounded
                if ranked:
                    shown = [c["_raw"] for c in ranked]
                    agent._pending_raw_items = shown
                    agent._search_ranked = True
                    digest = _ranked_digest(ranked, lang=lang)
                    note = ranking_note + grounded_note
                    if lang == "en":
                        text = (f"Search results ({len(shown)} records, "
                                f"{note}):\n{digest}\n\n"
                                "The full list is displayed to the user.")
                    else:
                        text = (f"检索结果（{len(shown)} 条，{note}）：\n"
                                f"{digest}\n\n"
                                "完整列表已展示给用户。")
            feedback = await _auto_feedback_round(agent, entry, lang)
            if feedback is not None:
                ranked, ranking_note, fb_note = feedback
                if ranked:
                    shown = [c["_raw"] for c in ranked]
                    agent._pending_raw_items = shown
                    agent._search_ranked = True
                    digest = _ranked_digest(ranked, lang=lang)
                    note = ranking_note + fb_note
                    if lang == "en":
                        text = (f"Search results ({len(shown)} records, "
                                f"{note}):\n{digest}\n\n"
                                "The full list is displayed to the user.")
                    else:
                        text = (f"检索结果（{len(shown)} 条，{note}）：\n"
                                f"{digest}\n\n"
                                "完整列表已展示给用户。")
                else:
                    # Nothing displayable either — keep the existing text
                    # and append only the feedback outcome.
                    text = text.rstrip() + "\n" + fb_note
            recall = await _recall_expansion_round(agent, entry, lang)
            if recall is not None:
                ranked, ranking_note, recall_note = recall
                if ranked:
                    shown = [c["_raw"] for c in ranked]
                    agent._pending_raw_items = shown
                    agent._search_ranked = True
                    digest = _ranked_digest(ranked, lang=lang)
                    note = ranking_note + recall_note
                    if lang == "en":
                        text = (f"Search results ({len(shown)} records, "
                                f"{note}):\n{digest}\n\n"
                                "The full list is displayed to the user.")
                    else:
                        text = (f"检索结果（{len(shown)} 条，{note}）：\n"
                                f"{digest}\n\n"
                                "完整列表已展示给用户。")
                else:
                    text = text.rstrip() + "\n" + recall_note
            if _is_doc_list:
                # Document-list tools are the final answer — the documents
                # of ONE application; one call suffices.  The loop ends
                # after this observation (the result list is displayed).
                return {"kind": "observation", "text": text, "final": True}
            return {"kind": "observation", "text": text}

        return {"kind": "observation", "text": _summarize_observation(result, lang)}

    return execute_action
