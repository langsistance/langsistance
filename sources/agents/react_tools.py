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
    is_provisional_application,
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
from sources.patent_source_detect import is_cn_source

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


def _env_float(name: str, default: float) -> float:
    """Float env knob that never raises at import (bad value → default)."""
    try:
        return float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


# 排序按需回退 (2026-09-15)：在 applications/search 的**标题级语料**上强制
# sort=_score 会让短标题的临时申请/失效件顶满整页 —— 2026-09-14 探针实测
# 前 20 槽位存活 _score 41/120 vs API 默认序 (filingDate desc) 103/120，最坏类
# (LED/RGB) 0/20；2026-09-12 生产同向（20 条美方候选 18 条失效、次轮 10/10 全灭）。
# 常态仍用相关度序；仅当本轮存活候选占比低于阈值时，用 API 默认序重取一次并按
# 存活数择优。默认 0.25 = 只在“近乎整页失效”时回退，避免把 USPTO 请求量翻倍
# （生产已观测 429 限流）。阈值是产品旋钮；REACT_USPTO_SORT_FALLBACK=0 关闭。
REACT_USPTO_SORT_FALLBACK_ENABLED = (
    os.getenv("REACT_USPTO_SORT_FALLBACK", "1") == "1")
REACT_USPTO_SORT_FALLBACK_ALIVE_RATIO = _env_float(
    "REACT_USPTO_SORT_FALLBACK_ALIVE_RATIO", 0.25)
LADDER_MAX_HITS = int(os.getenv("REACT_LADDER_MAX_HITS")
                      or os.getenv("REACT_TIGHTEN_SUGGEST_THRESHOLD", "300"))
# Per-number verification tools (search_patent_by_identifying_number_...)
# are capped per request: the LLM was observed looping through 8+ one-by-one
# fetches (each followed by a full ~2.5s semantic rerank) without
# converging.  At the cap the tool returns a stop-nudge instead of fetching
# yet another application.
VERIFY_CALL_MAX = int(os.getenv("REACT_VERIFY_CALL_MAX", "8"))
# Built-in number resolution (2026-09-03, sample #16): a bare-number
# question was closed after a single USPTO 404 — no CN cross-check, no
# format guidance.  The deterministic resolve tool + zero-hit cross
# round share one per-request gateway-call budget so a multi-candidate
# parse can never fan out without bound.
NUMBER_CROSS_MAX_QUERIES = int(os.getenv(
    "REACT_NUMBER_CROSS_MAX_QUERIES", "4"))


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


# ── Built-in number resolution (USPTO + Baiten CN) ───────────────────────────
# Registered on every request in build_tool_set; executed via
# make_action_executor (kind="patent_number").  Deterministic: the tool
# function parses the number itself and runs the sources — no LLM query
# construction, so a bare or malformed number can never send a keyword
# ladder down the wrong pipe.

PATENT_NUMBER_RESOLVE_TOOL_NAME = "patent_number_resolve"


class _NumberResolveArgs(BaseModel):
    number: str = Field(
        description=("专利号/公开号/申请号原文，无需格式化或补国别前缀"
                     "（纯数字或带 CN/US 前缀均可）；"
                     "系统会解析格式并自动做中美双库核验"),
    )


async def _patent_number_stub(number: str) -> str:
    raise NotImplementedError("executed via dispatch, not directly")


# ── Built-in legal-status lookup (需求#18, CN-first) ─────────────────────────
# Registered on every request, like the number tool: a legal-status question
# can arrive in any country mode and must not depend on the LLM picking a
# pushed KB tool.  Deterministic — the number is parsed system-side, then the
# owning source's legal-status API is queried by key.

PATENT_LEGAL_STATUS_TOOL_NAME = "patent_legal_status"

# FLZT/FSWX are a DIFFERENT gateway method from /openService/search: they are
# key-addressed and cost no search fan-out.  They therefore get their own
# per-request cap rather than sharing NUMBER_CROSS_MAX_QUERIES — conflating
# the two would let one status question starve a later number lookup.
LEGAL_STATUS_MAX_LOOKUPS = int(os.getenv("REACT_LEGAL_STATUS_MAX_LOOKUPS", "3"))


class _LegalStatusArgs(BaseModel):
    number: str = Field(
        description=("专利号/公开号/申请号原文，可含 CN/US 前缀；"
                     "系统会解析格式并查询法律状态事件与复审/无效决定"),
    )


async def _patent_legal_status_stub(number: str) -> str:
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
        description="China patent search query (Chinese, from the guidance ladder)",
    )
    page: int = Field(default=1, description="Page number")
    page_size: int = Field(default=20, description="Results per page")


class _CnPatentSearchArgs(BaseModel):
    query_string_cn: str = Field(
        description="China patent search query (Chinese, from the guidance ladder)",
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


# Built-in deep-analysis entry (#22, 2026-09-03): family/prosecution
# analysis tools came ONLY from per-user type-3 knowledge items, so a
# fresh user or a scene without them silently had no way to start one —
# the same "分析 X 的全球同族审查差异" ask worked or degraded depending on
# configuration (observed with users 3044…/1825…).  Register one generic
# entry whenever no tailored long-task entry exists.
BUILTIN_DEEP_ANALYSIS_TOOL_NAME = "patent_deep_analysis"
BUILTIN_DEEP_ANALYSIS_QUESTION = (
    "对指定专利进行深度后台分析（审查历史 / 全球同族审查差异 / 复审无效）"
)

# CN publication shape (CN114948588A / CN 1149 48588 A) for the citing-US
# annotation (#22c): a CN-number search whose USPTO leg returns hits usually
# matched US applications CITING that CN document, not its family members.
_RE_CN_PUB_NUMBER = re.compile(r"CN\s*\d{7,12}\s*[A-Za-z]{1,2}")
_FAMILY_INTENT_KEYWORDS_ZH = ("同族", "家族", "全球", "各国", "审查差异", "跨国")
_FAMILY_INTENT_KEYWORDS_EN = ("family", "worldwide", "jurisdiction",
                              "counterpart", "examination difference")


def _builtin_deep_analysis_description(lang: str = "zh") -> str:
    """Description for the built-in deep-analysis long task entry.

    Deliberately narrow: it must NOT hijack plain legal-status questions
    ("被驳回了吗" — answered by search + legal-status enrichment) or
    retrieval requests (document download).  It exists for examination /
    family ANALYSIS of a specific patent id.
    """
    if lang == "en":
        return (
            "Start a background deep-analysis task for ONE patent: "
            "prosecution/examination history (office actions, rejections, "
            "re-examination, invalidation) OR its worldwide family "
            "examination across countries (family members, examination "
            "differences). Requires a specific patent/application id in the "
            "question. The task runs asynchronously — after calling it, tell "
            "the user the analysis task was created and that results will "
            "appear here when done. Do NOT substitute a keyword search for "
            "this analysis."
            " A plain legal-status question (e.g. \"was it rejected?\") is "
            "NOT this task - call patent_legal_status instead."
        )[:800]
    return (
        "对指定专利发起后台深度分析任务：审查历史/审查意见/OA/驳回/复审无效，"
        "或全球同族申请在各国的审查过程与差异。问题中必须包含具体专利号/申请号。"
        "任务异步执行——调用后告知用户任务已创建，完成后结果会出现在本会话；"
        "不要用普通关键词检索代替该分析。仅法律状态查询（如“被驳回了吗”）"
        "不属于本任务，请改用 patent_legal_status 工具。"
    )[:800]


def _us_citing_note(query: str, cn_q: str, cn_cands: list,
                    us_cands: list, lang: str = "zh") -> str:
    """Annotation for a CN-publication-number search whose US leg hit docs.

    When the Chinese query slot is exactly ONE CN publication number, Baiten
    returned exactly that document, and the USPTO leg still returned hits,
    those US hits are almost always applications CITING the CN document
    (USPTO full-text search matched the cited number), not its family
    members.  Returning them unlabelled misleads family questions.
    Returns the note text, or "" when the pattern does not apply.
    """
    if not (cn_q and us_cands):
        return ""
    if len(cn_cands) != 1:
        return ""
    # Canonicalize matches by stripping ALL internal whitespace, then dedupe:
    # rewrite ladders repeat the SAME number across OR variants
    # (… OR "CN 105414512 A" OR CN-105414512-A), which must count as one.
    tokens = {re.sub(r"\s+", "", t)
              for t in _RE_CN_PUB_NUMBER.findall(cn_q or "")}
    if len(tokens) != 1:
        return ""
    lower_query = (query or "").lower()
    if lang == "zh":
        hits_intent = any(k in (query or "") for k in _FAMILY_INTENT_KEYWORDS_ZH)
    else:
        hits_intent = any(k in lower_query
                          for k in _FAMILY_INTENT_KEYWORDS_EN)
    if hits_intent:
        return (
            "注：本检索式命中单一中国专利文献，美国端返回的申请多为引用该"
            "中国专利的文献（非同族成员）。如需该专利的全球同族及各国审查"
            "情况，请发起深度分析任务（回复“分析 <专利号> 的全球同族审查差异”"
            "或使用分析任务入口）。"
        )
    return (
        "注：美国端命中多为引用该中国专利的申请（非同族成员），相关性低于"
        "中国端结果，请注意区分。"
    )


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
    conversation_history=None,
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
    # 资格门（需求#1）：无专利引用的文本诉求不绑定任何 long_task 工具，
    # 消灭「ReAct 循环里 LLM 自主调 long_task」这条误路由路径。
    long_task_eligible = _is_long_task_eligible(question, conversation_history)

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

    # Built-in exact-number lookup — always registered: identifiers can
    # arrive in any country mode, and the deterministic resolve must not
    # depend on the LLM picking the right pushed KB tool (sample #16).
    number_tool = StructuredTool.from_function(
        func=_patent_number_stub,
        name=PATENT_NUMBER_RESOLVE_TOOL_NAME,
        description=(
            "Look up a patent by its exact number (application / "
            "publication / grant / design number, US or CN). Use this "
            "when the user gives a number or patent identifier instead "
            "of a technical description. Runs a deterministic parse and "
            "checks BOTH the USPTO and China patent sources, attaching "
            "bibliographic data and legal status when found."
        ),
        args_schema=_NumberResolveArgs,
    )
    add(ToolEntry(name=PATENT_NUMBER_RESOLVE_TOOL_NAME, kind="patent_number",
                  knowledge=None, tool_info=None, tool=number_tool))

    # Built-in legal-status lookup (需求#18) — always registered, for the
    # same reason as the number tool above.
    legal_status_tool = StructuredTool.from_function(
        func=_patent_legal_status_stub,
        name=PATENT_LEGAL_STATUS_TOOL_NAME,
        description=(
            "Query the LEGAL STATUS of one patent by its exact number "
            "(grant / publication / application number, CN or US). Use this "
            "when the user asks whether a patent was granted, rejected, "
            "withdrawn, terminated or expired, or asks for its legal-status "
            "timeline, re-examination / invalidation decisions, or whether "
            "re-examination is available. Returns the recorded status "
            "events; where the sources do not cover a status, the answer "
            "states that explicitly and gives the official lookup entry. "
            "Answer ONLY from the returned record — never infer a reason "
            "the record does not state."
        ),
        args_schema=_LegalStatusArgs,
    )
    add(ToolEntry(name=PATENT_LEGAL_STATUS_TOOL_NAME,
                  kind="patent_legal_status",
                  knowledge=None, tool_info=None, tool=legal_status_tool))

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
                "returns BOTH US patents (USPTO) and Chinese patents in "
                "one call. Pass the English ladder query as "
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
                "Search Chinese patents for a user question "
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
            if not long_task_eligible:
                continue
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

    # ── Built-in deep-analysis entry (#22) ──
    # Tailored type-3 entries above depend on per-user/scene knowledge; when
    # none matched, a fresh user must still be able to start a
    # family/prosecution analysis — otherwise the identical question works
    # for some users and silently degrades for others (users 3044…/1825…).
    # knowledge=None keeps it out of the deterministic pre-route (see
    # _match_long_task_intent guard) — it stays reachable via ReAct tool
    # choice, which is the intended gap-fill.
    if long_task_eligible and not any(
            e.kind == "long_task" for e in registry.values()):
        builtin_tool = StructuredTool.from_function(
            func=_long_task_stub,
            name=BUILTIN_DEEP_ANALYSIS_TOOL_NAME,
            description=_builtin_deep_analysis_description(
                getattr(agent, "_lang", "zh")),
            args_schema=_QueryArgs,
        )
        add(ToolEntry(name=BUILTIN_DEEP_ANALYSIS_TOOL_NAME,
                      kind="long_task", knowledge=None, tool_info=None,
                      tool=builtin_tool))
    return registry, tools


SEARCH_DIGEST_LIMIT = 20
SEARCH_DIGEST_CHARS = 3000


def _items_digest(raw_items, limit: int = SEARCH_DIGEST_LIMIT,
                  lang: str = "zh", us_limit: int = None,
                  cn_limit: int = None) -> str:
    """Serialize search raw_items into a bounded digest for the LLM.

    USPTO-shaped items are flattened via build_candidates into
    ``申请号 | 标题 | 申请人 | 申请日 | 状态`` lines.  Non-USPTO shapes
    fall back to a truncated JSON dump.
    """
    items = raw_items or []
    if not items:
        return ""
    us_lines: list = []
    # 需求#36: 按号查询时中靶行的前缀标记（与"相关件"区分）。
    hit_marker = "★该号码本身" if lang == "zh" else "★ THE NUMBER ITSELF"
    candidates = build_candidates(items)
    _us_limit = limit if us_limit is None else max(0, int(us_limit))
    _cn_limit = limit if cn_limit is None else max(0, int(cn_limit))
    for c in candidates[:_us_limit]:
        parts = [
            c.get("patent_id") or "?",
            c.get("title") or "(无标题)",
            c.get("applicant") or "?",
            c.get("filing_date") or "?",
            c.get("status") or "?",
        ]
        # 需求#36: 按号查询时把"这一条就是该号码本身"标给模型, 与相关件区分。
        if isinstance(c.get("_raw"), dict) and c["_raw"].get("_number_hit"):
            parts.insert(0, hit_marker)
        us_lines.append(" | ".join(str(p) for p in parts))
    # Baiten CN candidates (flat mapped shape with source="baiten") ride
    # alongside USPTO rows in a mixed dual-source pool.
    cn_lines: list = []
    flat = [c for c in items if isinstance(c, dict)
            and is_cn_source(c.get("source"))
            and c.get("patent_id")]
    for c in flat[:_cn_limit]:
        parts = [
            c.get("patent_id") or "?",
            c.get("title") or "(无标题)",
            c.get("applicant") or "?",
            c.get("pub_date") or c.get("apply_date") or "?",
        ]
        # 需求#36: CN 侧中靶行同样要标 —— 中文检索的中靶信号不能是空的。
        if c.get("_number_hit"):
            parts.insert(0, hit_marker)
        tail_bits = []
        current_status = str(c.get("status") or "").strip()
        if current_status:
            tail_bits.append(f"状态:{current_status}")
        law_tail = _compact_baiten_law_summary(c)
        if law_tail:
            tail_bits.append(law_tail)
        if tail_bits:
            parts.append("; ".join(tail_bits))
        cn_lines.append(" | ".join(str(p) for p in parts))
    # 中文提问：CN 行在前。摘要的行序就是模型的行文序 —— 此前固定 US 在前，
    # 中文提问的回答被带成"基本都是美国专利"（生产 2026-09-15），与面板
    # _order_pending_for_lang 的 CN-first 也不一致。
    lines = cn_lines + us_lines if lang == "zh" else us_lines + cn_lines
    if lines:
        # 候选构成行：两侧供给量摆出来，模型才不会凭空断言某一侧"命中较少"。
        header = ""
        if flat or candidates:
            header = (f"[候选构成] CN {len(flat)} / US {len(candidates)}\n"
                      if lang == "zh"
                      else f"[composition] CN {len(flat)} / US {len(candidates)}\n")
        text = header + "\n".join(lines)
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
        law_tail = _compact_baiten_law_summary(c)
        if law_tail:
            parts.append(law_tail)
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


# ── long_task 资格门（需求#1, P0）────────────────────────────────────────────
# long_task 是「对已有专利对象做分析」的管道，合法前提是请求携带专利引用：
# 查询里的专利号，或（追问意图 + 对话历史里有前序结果）。无引用的文本诉求
# 进管道必然以 no_patents_found 失败——生产 2026-08 四例（公司分析/能力咨询/
# 申请人检索/时间过滤检索）耗时 23s～9分17秒，chat 侧从未获得处理机会。
_PATENT_ID_PATTERNS = (
    re.compile(r"\b\d{8}\b"),                   # US 8 位申请/授权号
    re.compile(r"\b\d{2}/\d{6}\b"),             # US 回执号 30/076,484
    re.compile(r"\bUS\d{6,}\b", re.I),          # US30076484 / US20250103146A1
    re.compile(r"\b20[12]\d{8,9}(?:\.\d)?\b"),  # CN 申请号 202310123456.7
    re.compile(r"\bCN\d{7,12}[A-Z]?\d?\b", re.I),   # CN 公开/公告号
)

# 追问指代：命中关键词只说明「可能是在指代前文」，还必须历史里真有结果。
FOLLOWUP_KEYWORDS = ("这", "上述", "前面", "以上", "其中", "筛选", "挑出",
                     "选出", "哪些", "哪个", "第一个", "第几", "这些", "上面",
                     "刚才", "继续", "接着", "然后", "再分析", "this", "these",
                     "above", "first", "continue")


def _query_has_patent_id(text: str) -> bool:
    """查询文本中出现任一专利号形态。纯函数，永不抛。"""
    raw = str(text or "")
    return any(p.search(raw) for p in _PATENT_ID_PATTERNS)


def _conversation_has_patent_refs(conv_history) -> bool:
    """对话历史里任一消息携带 hidden patent_ids / patent_data。"""
    for msg in conv_history or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("patent_ids") or msg.get("patent_data"):
            return True
    return False


def _is_long_task_eligible(query: str, conv_history=None) -> bool:
    """请求可否进 long_task 管道：必须携带专利引用。

    宽松预检（真正的裁决仍在分类器）：8 位数字误判只保留旧行为，不会
    制造新失败；反之无引用的文本诉求被挡在管道外，秒级转 chat。
    """
    raw = str(query or "")
    if _query_has_patent_id(raw):
        return True
    lowered = raw.lower()
    if not any(k in raw or k in lowered for k in FOLLOWUP_KEYWORDS):
        return False
    return _conversation_has_patent_refs(conv_history)


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
                                  lang: str,
                                  conv_history=None) -> Optional[ToolEntry]:
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
    # Tailored-knowledge pre-route only: the built-in generic deep-analysis
    # entry carries knowledge=None (no question to classify against); every
    # query would match a blank question under the classifier's
    # "宁可命中不可漏判" bias and route EVERYTHING into the deep task.
    entries = [
        e for e in entries
        if e.knowledge is not None
        and str(getattr(e.knowledge, "question", "") or "").strip()
    ]
    if not entries:
        return None
    query_text = str(query or "").strip()
    # Retrieval requests ("获取...档案/文档") are the document-list tools'
    # job — the analysis long task must never answer them.  Determined
    # BEFORE the LLM call so the classifier's "宁可命中不可漏判" bias
    # cannot hijack a document download.
    if _is_retrieval_request(query_text):
        return None
    # 资格门（需求#1）：无专利引用的文本诉求不进分析管道 —— 分类器的
    #「宁可命中不可漏判」偏见会把公司检索/能力咨询也判成长任务。
    if not _is_long_task_eligible(query_text, conv_history):
        _glog = getattr(agent, "logger", None)
        if _glog is not None:
            _glog.info(
                "Long task intent rejected (no patent reference) — "
                "falling back to chat")
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
    dead_filtered = [
        c for c in new_cands if is_dead_status(c.get("status"))]
    if dead_filtered:
        # 诊断日志 (2026-09-01): 08:12 日志 US 20 条命中未进打分窗口,
        # 疑似被 dead 过滤 (申请公开库 abandoned 占比高)。先验证根因
        # 再决定产品行为, 不做未经验证的行为变更。
        _glog = getattr(agent, "logger", None)
        if _glog is not None:
            _glog.info(
                f"dead_filter_diag — filtered={len(dead_filtered)} "
                f"statuses={[str(c.get('status'))[:40] for c in dead_filtered[:5]]} "
                f"granted={[bool(c.get('patent_number')) for c in dead_filtered[:5]]}"
            )
    live = [c for c in new_cands
            if not is_dead_status(c.get("status"))
            and not is_provisional_application(c)]
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
            # 说清缺的是哪个前置条件（标题 json / 向量缓存 / numpy）——
            # 只是"no CPC matches"时运维无从下手（2026-09-14 生产连报两轮）。
            try:
                from sources.long_task.cpc_semantic import cpc_availability
                _why = cpc_availability().get("reason") or "unknown"
            except Exception:
                _why = "availability probe failed"
            _glog.warning(
                "cpc expansion enabled but no CPC matches — "
                f"availability: {_why}")
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
        raw = await _invoke_uspto_with_fallback(agent, entry, q)
    except Exception:
        return None
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
            "source": "cn",
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


_PHRASE_RE = re.compile(r'"([^"]+)"')


def _word_level_query(q: str) -> str | None:
    """Rewrite quoted phrases to word-level AND for applications/search.

    The endpoint matches quoted phrases as ORDER-SENSITIVE exact phrases
    — "RGB LED" returns 200 while "RGB LED driver" returns 404 on the
    same corpus (observed 2026-09-01, five successive user logs).  Word-
    level AND ("RGB AND LED AND driver") matches regardless of word
    order/adjacency.  Returns None when nothing changes.
    """
    phrases = _PHRASE_RE.findall(q or "")
    if not phrases:
        return None
    out = q
    for p in phrases:
        words = p.split()
        if len(words) >= 2:
            out = out.replace(f'"{p}"', " AND ".join(words))
    return out if out != q else None


def _alive_counts(items: list) -> tuple:
    """(alive, total) over raw USPTO items via the shared dead-status rule.

    Pure; returns (0, 0) when the item shape cannot be parsed.
    """
    try:
        from sources.long_task.candidate_metadata import (
            build_candidates, is_dead_status)
        candidates = build_candidates(items or [])
    except Exception:
        return (0, 0)
    if not candidates:
        return (0, 0)
    alive = sum(1 for c in candidates
                if not is_dead_status(c.get("status")))
    return (alive, len(candidates))


async def _uspto_search_by_query(
    q: str, page: int = 1, page_size: int = 20,
) -> tuple[list, str]:
    """POST USPTO applications/search; returns (raw_items, note).

    404 handling is routed by the AND-operator count (2026-09-15):

    - **>2 AND** = the endpoint's dialect overflow (verified 2026-09-07:
      ≤2 operators parse, 3+ always 404 whatever the bracket layout) — the
      query never reached the corpus, so trim trailing conjuncts and resend
      (the only retry that does not silently drop a constraint).
    - **≤2 AND** = the query is well-formed and the corpus genuinely has no
      title-level match ("true zero") — retrying only burns requests
      (production 2026-09-14: 34 requests per 4 usable answers).  Keep the
      quoted-phrase word-level retry (phrase matching here is order-
      sensitive — a different failure mode) and stop.

    After a 200, a page whose candidates are almost all dead (expired /
    abandoned / parked) is re-fetched once under the API default order and
    the better page wins — see REACT_USPTO_SORT_FALLBACK_* above.
    """
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

        try:
            from sources.long_task.search_query_builder import (
                MAX_USPTO_AND_OPS, trim_uspto_and_overflow,
                uspto_and_op_count)
            dialect_overflow = uspto_and_op_count(q) > MAX_USPTO_AND_OPS
        except Exception:
            trim_uspto_and_overflow = None
            dialect_overflow = False

        async def _search(query: str, use_sort: bool = True) -> tuple:
            body = {
                "q": query,
                "pagination": {
                    "offset": max(page - 1, 0) * page_size,
                    "limit": page_size,
                },
                "fields": RECALL_SEARCH_FIELDS,
            }
            if use_sort:
                body["sort"] = [
                    {"field": REACT_USPTO_SORT_FIELD, "order": "desc"}]
            resp = await outbound_http.arequest(
                "POST", USPTO_SEARCH_URL, purpose="dual_patent_search",
                headers=headers, json=body, timeout=30,
            )
            return resp, query

        response, used_q = await _search(q)
        if getattr(response, "status_code", 0) != 200:
            # 引号短语 0 命中 → 词级降级重试一次
            word_q = _word_level_query(q)
            if word_q:
                response, used_q = await _search(word_q)
        trim_note = ""
        if (getattr(response, "status_code", 0) != 200
                and dialect_overflow):
            # 3+ 个 AND 连接组在该端点**必 404**（2026-09-07 实测：≤2 AND 可
            # 解析计数，3+ 无论括号怎么套都 404）。按预算裁掉尾部 AND 合取项
            # 是**唯一不丢约束**的救法，与 dynamic_tool_params 里 KB 路径用的
            # 是同一个纯函数。
            # ⚠️ 这里必须用**比生成侧更严的 1**，不能用默认的 MAX_USPTO_AND_OPS=2。
            # 2026-09-12 生产日志实测：404 掉的正是 2 个 AND 算子（3 个连接组）的
            # 查询，例如
            #   ("cervical rehabilitation" OR "neck exercise")
            #     AND ("head support assembly" OR "head restraint") AND resistance
            # 它在默认预算下**判定为合规、不裁**——于是"重发"与"原发"是同一条，
            # 修复形同空转。既然已经 404 了，就是这条查询过不了，只能裁得更狠。
            trimmed_q = ""
            if trim_uspto_and_overflow is not None:
                try:
                    trimmed_q = trim_uspto_and_overflow(q, max_and_ops=1)
                except Exception:
                    trimmed_q = ""
            if trimmed_q and trimmed_q not in (q, word_q):
                response, used_q = await _search(trimmed_q)
                if getattr(response, "status_code", 0) == 200:
                    trim_note = "uspto 404 retry — AND-budget trim"
        if (USPTO_SPACE_FLATTEN_ENABLED and dialect_overflow
                and getattr(response, "status_code", 0) != 200):
            # ★ 只作**方言超限**的末位兜底的「去括号纯空格词形」：它会把 404 换成**含噪 200**
            # ——空格拼接在该端点是 OR 语义（dynamic_tool_params.py 的实测结论：
            # 单概念基线的命中数之和 == 空格拼接的命中数），约束全丢、结果与
            # 原查询不可比。2026-09-03 起它被当作"救援"，实测是拿噪声换 200；
            # 现降级为末位，且不再计入救援成功。
            try:
                from sources.long_task.search_query_builder import (
                    destructure_uspto_query)
                flat_q = destructure_uspto_query(q)
            except Exception:
                flat_q = ""
            if flat_q and flat_q not in (q, word_q):
                response, used_q = await _search(flat_q)
                if getattr(response, "status_code", 0) == 200:
                    trim_note = "uspto 404 fallback — space-flatten (OR semantics, noisy)"
        if getattr(response, "status_code", 0) != 200:
            # 合规形态 (≤2 AND) 的 404 = 标题域真零：重试链已跳过，标注出来
            # 便于把「真零」与「方言超限」在日志里分开数。
            true_zero = (not dialect_overflow
                         and getattr(response, "status_code", 0) == 404)
            return [], f"USPTO HTTP {response.status_code}" + (
                " (true zero, no retry)" if true_zero else "")
        data = response.json()
        items = _normalize_uspto_items(
            data.get("patentFileWrapperDataBag") or [])
        sort_note = ""
        if (REACT_USPTO_SORT_FALLBACK_ENABLED
                and REACT_USPTO_SORT_FIELD == "_score"):
            alive, total = _alive_counts(items)
            if total and (alive / total) < REACT_USPTO_SORT_FALLBACK_ALIVE_RATIO:
                resp2, _q2 = await _search(used_q, use_sort=False)
                if getattr(resp2, "status_code", 0) == 200:
                    items2 = _normalize_uspto_items(
                        resp2.json().get("patentFileWrapperDataBag") or [])
                    alive2, total2 = _alive_counts(items2)
                    if alive2 > alive:
                        items = items2
                        sort_note = (
                            "uspto sort fallback — default order "
                            f"(alive {alive}/{total} -> {alive2}/{total2})")
        note_bits = [b for b in (trim_note, sort_note) if b]
        suffix = f" ({'; '.join(note_bits)})" if note_bits else ""
        return items, f"USPTO {len(items)} hits" + suffix
    except Exception as exc:
        return [], f"USPTO failed: {exc}"


# ── KB 工具路径的括号-404 降级 (2026-09-03 生产观察) ─────────────────────────
# 线上 ReAct 主要经知识库推送工具 (search_patent_by_key_word / by_assignee
# 等) 走 dynamic_backend_tool_function — 它们不经过 _uspto_search_by_query,
# 没有降级链。同一观察里 8/8 次含括号/引号短语的查询全部 404, 而无括号
# 裸词查询返回 200。因此: 凡 USPTO applications/search 工具返回空且查询
# 带括号或引号 → 自动以空格词形重试一次; 不带括号的查询 (如 "A AND B")
# 保持不变 (有 200 记录)。

def _is_uspto_search_tool(tool_info) -> bool:
    """True for tools whose URL targets USPTO applications/search."""
    url = (getattr(tool_info, "url", "") or "").lower()
    return "uspto" in url and "/applications/search" in url


def _flatten_query_for_uspto(q: str) -> str:
    """Return the plain-space form when *q* carries bracket/quoted
    structure (the structure this endpoint 404s on); otherwise unchanged.
    Pure."""
    if not q or ("(" not in q and '"' not in q):
        return q or ""
    try:
        from sources.long_task.search_query_builder import (
            destructure_uspto_query)
        return destructure_uspto_query(q) or q
    except Exception:
        return q


def _with_query_replaced(payload, new_q: str) -> dict:
    """Deep-copy *payload* with its query slot replaced by *new_q*.

    Understands the shapes invoke payloads take: envelope body.q,
    params-JSON-string, nested query.q, or top-level q.  Returns the
    original payload unchanged when no query slot is found.
    """
    import copy as _copy
    if not isinstance(payload, dict):
        return payload
    out = _copy.deepcopy(payload)

    def _replace_in(d: dict) -> bool:
        if not isinstance(d, dict):
            return False
        body = d.get("body")
        if isinstance(body, dict) and "q" in body:
            body["q"] = new_q
            return True
        nested = d.get("query")
        if isinstance(nested, dict):
            if "q" in nested:
                nested["q"] = new_q
                return True
            if _replace_in(nested):
                return True
        params = d.get("params")
        if isinstance(params, dict):
            if "q" in params:
                params["q"] = new_q
                return True
            if _replace_in(params):
                return True
        if isinstance(params, str) and params.strip():
            try:
                parsed = json.loads(params)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, dict) and "q" in parsed:
                parsed["q"] = new_q
                d["params"] = json.dumps(parsed, ensure_ascii=False)
                return True
        if isinstance(d.get("q"), str):
            d["q"] = new_q
            return True
        for value in d.values():
            if isinstance(value, dict) and _replace_in(value):
                return True
        return False

    _replace_in(out)
    return out


async def _invoke_uspto_with_fallback(agent, entry, q: str) -> list:
    """Invoke one USPTO search query via the KB tool with bracket fallback.

    Calls the tool once; when the result is empty AND the query carries
    bracket/quoted structure, re-invokes the plain-space form (the
    structure's 404s are the observed norm).  Returns the pending raw
    items of the last attempt.  Never raises.
    """
    envelope = _build_uspto_envelope(entry.tool_info, q)
    await asyncio.to_thread(
        entry.tool.invoke, _tool_invoke_payload(agent, envelope))
    pending = getattr(agent, "_pending_raw_items", None) or []
    if pending or not _is_uspto_search_tool(entry.tool_info):
        return pending
    flat = _flatten_query_for_uspto(q)
    if flat and flat != q:
        _glog = getattr(agent, "logger", None)
        if _glog is not None:
            _glog.info(
                "uspto_query_bracket_fallback — "
                f"q0={q[:90]!r} flat={flat[:90]!r}")
        await asyncio.to_thread(
            entry.tool.invoke,
            _tool_invoke_payload(
                agent, _build_uspto_envelope(entry.tool_info, flat)))
        pending = getattr(agent, "_pending_raw_items", None) or []
    return pending


# Legal-status enrichment limits (chat path, 2026-09-03): a single FLZT
# payload carries both the current status and the event timeline, so the
# timeline costs no extra request.  FSWX (复审无效) decisions carry full
# decision text — fetched only for small hit lists (single-number lookups)
# and truncated so the item payload / SMALL-LIST batch stay bounded.
LAW_DETAIL_SMALL_LIST = 3       # ≤3 Baiten candidates → also fetch FSWX
LAW_TIMELINE_MAX_ITEMS = 12     # FLZT events kept per candidate
LAW_REVIEW_MAX_ITEMS = 3        # FSWX decisions kept per candidate
LAW_REVIEW_FULLTEXT_CHARS = 800  # decision fullText cap per decision

# 富化并发上限（2026-09-13 生产事故）。佰腾网关对并发敏感：单轮 30 条候选
# 一起去时约一半 FLZT 调用返回 500
# ``no access for this api: DATA_PAT_PATAFFAIRSDATA_ONE``；而同一个号码在
# 26 秒后重试**成功**——不是权限缺失，是并发节流。此前是「每次检索 10 路」，
# 收拢成单轮一次后变成「单轮 N 路」，去重省下的调用被节流打回，净亏。
# 限并发后单位时间请求数下降，首轮成功率上升。
LAW_ENRICH_CONCURRENCY = int(os.getenv("REACT_LAW_ENRICH_CONCURRENCY", "4"))

# 每次工具调用最多富化多少条（2026-09-13 配额治理）。lawInfos 是**计费/
# 配额**接口，而自动补跑阶梯一次可能收进 30+ 条候选；全查一遍是配额的主要
# 消耗方式。默认对齐摘要展示条数（SEARCH_DIGEST_LIMIT）——超出这个数的行
# 本来也不会出现在模型看到的那段摘要里。
#
# **取舍（明确记录）**：超出上限的候选**没有法律状态**，其状态列在导出文件
# 里会空着。按配额松紧用 env 调整；设 0 表示不限（回到旧行为）。
LAW_ENRICH_MAX_PER_CALL = int(
    os.getenv("REACT_LAW_ENRICH_MAX_PER_CALL", str(SEARCH_DIGEST_LIMIT)))

# 每**请求**的 lawInfos 总预算 —— 配额治理的主要旋钮。按调用限流治不了总量：
# 一次提问会跑 5–6 次工具调用（阶梯 + 自动补跑），每次各自限 20 条仍然合计
# ~50 次；配额是每请求的硬约束，所以必须有总量闸门。默认与摘要展示条数一致
# —— **每个请求只够富化一份完整摘要**。缓存命中不消耗预算，只有真正打到网关
# 的调用才计数。0 = 不限（回到旧行为）。
# 默认 35：2026-09-13 实测该查询导出 30 条 CN，预算 20 时**13 行状态列空白**
# （16/29 有状态）——覆盖不全比多花几次更伤体验。35 覆盖整份导出，且仍比
# 「每轮都富化」的旧行为（~55 次）省 36%。配额紧再往下压。
LAW_ENRICH_MAX_PER_REQUEST = int(
    os.getenv("REACT_LAW_ENRICH_MAX_PER_REQUEST", "35"))


def _compact_baiten_law_summary(c: dict) -> str:
    """One-line legal-status suffix for a digest row (observation only).

    Rows already carry the current ``status`` column; the suffix adds the
    timeline depth and the review/decision count so the model can answer
    "被驳回了吗 / 有没有复审记录" from the digest itself.  The full
    timeline/decision bodies ride the item payloads, never the digest.
    """
    bits: list = []
    timeline = c.get("legal_timeline") or []
    if len(timeline) > 1:
        bits.append(f"{len(timeline)}次状态变更")
    reviews = c.get("review_decisions") or []
    if reviews:
        bit = f"复审/无效决定{len(reviews)}条"
        first_date = str(reviews[0].get("declareDate")
                         or reviews[0].get("declare_date") or "")
        if first_date:
            bit += f"(最近{first_date})"
        bits.append(bit)
    return ("[" + "; ".join(bits) + "]") if bits else ""


# ── 每请求法律状态缓存（2026-09-13 lawInfos 调用量治理）─────────────────────
# 生产日志实测：一次提问的 ~100 次 lawInfos 里有 40 次**完全重复**——自动
# 补跑阶梯把同一条检索式又发了一遍，同一批 10 件专利被重复查了两遍。
#
# 缓存按**线调用**分开记（FLZT / FSWX），因为二者触发策略不同：话题式检索
# （大列表）只查 FLZT；法律状态工具需要时仍会补查 FSWX，但绝不重查昂贵的
# FLZT。空结果也要缓存（查过且为空 ≠ 没查过），否则空结果会被反复重查。
#
# ⚠️ 必须同时加进 create_agent 的 per-request 重置块 —— agent 池复用会让
# 缓存跨请求泄漏（本项目已因漏加重置踩过坑）。

def _law_flzt_cache(agent) -> dict:
    """app_num → 归一化 FLZT 时间线（空列表 = 查过且为空）。纯。"""
    if agent is None:
        return {}
    cache = getattr(agent, "_law_flzt_cache", None)
    if cache is None:
        cache = agent._law_flzt_cache = {}
    return cache


def _law_fswx_cache(agent) -> dict:
    """app_num → FSWX 复审/无效决定（**仅确实查过**时写入）。纯。"""
    if agent is None:
        return {}
    cache = getattr(agent, "_law_fswx_cache", None)
    if cache is None:
        cache = agent._law_fswx_cache = {}
    return cache


async def _enrich_baiten_law_status(client, candidates: list, glog,
                                    agent=None) -> None:
    """Fill Baiten candidates with /openService/law legal data.

    Per candidate (concurrent, 5s cap each so a slow gateway never blocks
    the result list): one FLZT call yields the current ``status`` (latest
    event — existing semantics) AND ``legal_timeline`` (capped events).
    When the candidate list is small enough to be a number lookup rather
    than a topic sweep (≤ LAW_DETAIL_SMALL_LIST), an FSWX (复审无效) call
    additionally attaches ``review_decisions`` with truncated fullText.

    ``agent`` (optional) enables the per-request memo — omit it and every
    candidate hits the wire, which is the historical behavior.

    Any failure degrades to the fields the candidate already carries —
    pure enrichment, never raises.
    """
    if not candidates:
        return
    detail_lookup = len(candidates) <= LAW_DETAIL_SMALL_LIST
    # 并发上限：佰腾网关对并发敏感，不限流会把去重省下的调用换成 500。
    # 每次调用建一个新信号量（不是模块级），避免跨请求持有。
    sem = asyncio.Semaphore(max(1, int(LAW_ENRICH_CONCURRENCY)))
    # 配额上限：只富化前 N 条（候选顺序即摘要展示顺序）。0 = 不限。
    _cap = max(0, int(LAW_ENRICH_MAX_PER_CALL))
    targets = candidates[:_cap] if _cap else candidates
    _max_req = max(0, int(LAW_ENRICH_MAX_PER_REQUEST))

    def _spend() -> bool:
        """从每请求预算里预留一次 lawInfos 调用。

        无 agent（旧调用方式）或不限额时恒真 —— 保持既有行为。缓存命中不
        走这里，因此不消耗预算。asyncio 单线程、检查与自增之间无 await，
        并发下不会超发。
        """
        if agent is None or not _max_req:
            return True
        used = int(getattr(agent, "_law_budget_used", 0) or 0)
        if used >= _max_req:
            return False
        agent._law_budget_used = used + 1
        return True

    async def _app_num(c: dict) -> str:
        return str(c.get("app_num") or c.get("application_number")
                   or "").strip()

    async def _one(c: dict) -> None:
        app_num = await _app_num(c)
        if not app_num:
            return
        flzt = _law_flzt_cache(agent)
        if app_num in flzt:
            timeline = flzt[app_num]
        else:
            try:
                async with sem:
                    if not _spend():
                        return
                    timeline = await asyncio.wait_for(
                        client.query_legal_state_timeline(app_num), timeout=5)
            except Exception as exc:
                if glog is not None:
                    glog.warning(
                        f"baiten law status failed for {app_num}: {exc}")
                return
            # 查过就记，空也算 —— 否则空结果下一轮会被重查。
            flzt[app_num] = timeline or []
        if not timeline:
            return
        law = str((timeline[0] or {}).get("lawStatus") or "").strip()
        if law:
            c["status"] = law
        kept = [
            {"date": str(e.get("date") or ""),
             "lawStatus": str(e.get("lawStatus") or "")}
            for e in timeline[:LAW_TIMELINE_MAX_ITEMS]
            if isinstance(e, dict) and (e.get("date") or e.get("lawStatus"))
        ]
        if kept:
            c["legal_timeline"] = kept

    async def _reviews(c: dict) -> None:
        app_num = await _app_num(c)
        if not app_num:
            return
        fswx = _law_fswx_cache(agent)
        if app_num in fswx:
            decisions = fswx[app_num]
        else:
            try:
                async with sem:
                    if not _spend():
                        return
                    decisions = await asyncio.wait_for(
                        client.query_patent_review(app_num), timeout=5)
            except Exception as exc:
                if glog is not None:
                    glog.warning(
                        f"baiten FSWX review failed for {app_num}: {exc}")
                return
            fswx[app_num] = decisions or []
        # 只在 FSWX **确实返回过**时标记「已查询」——调用失败不能算查过，
        # 否则下游会把"没查成"渲染成"查过且没有"。
        c["reviews_checked"] = True
        kept = []
        for d in (decisions or [])[:LAW_REVIEW_MAX_ITEMS]:
            if not isinstance(d, dict):
                continue
            full_text = str(d.get("fullText") or "").strip()
            kept.append({
                "declareDate": str(d.get("declareDate")
                                   or d.get("declare_date") or ""),
                "declareNum": str(d.get("declareNum")
                                  or d.get("declare_num") or ""),
                "lawBase": str(d.get("lawBase") or d.get("law_base") or ""),
                "fullText": full_text[:LAW_REVIEW_FULLTEXT_CHARS],
            })
        if kept:
            c["review_decisions"] = kept

    await asyncio.gather(*[_one(c) for c in targets])
    if detail_lookup:
        await asyncio.gather(*[_reviews(c) for c in targets])


async def _enrich_cn_candidates_once(agent, items, glog) -> None:
    """一轮工具调用收尾时统一富化佰腾候选（2026-09-13）。

    此前 ``_baiten_search_by_query`` 每次检索都内联 await 富化，而一次
    ``patent_search_dual`` 会跑 2–4 次佰腾检索（首轮 + 自动补跑阶梯），
    每次都把 ~1.5s 串在关键路径上。改为在这里统一跑一遍，配合每请求缓存
    （同号不重查）把总时长与调用数一起压下来。

    **必须在排名之前调用** —— 排名要读 ``status``。纯富化，永不抛。
    """
    cn = [c for c in (items or [])
          if isinstance(c, dict) and is_cn_source(c.get("source"))]
    if not cn:
        return
    client = _baiten_client_or_none(agent)
    if client is None:
        return
    await _enrich_baiten_law_status(client, cn, glog, agent=agent)


# 商业数据供应商名 **绝不允许** 进入 LLM 可见面（2026-09-13 生产实证：模型
# 在回答里写了「中国专利（佰腾）」）。异常/上游文本自带供应商名（baiten_client
# 的异常消息就是），而 notes 会被渲染进 observation —— 构造 note 时中立化。
# **日志仍用原始文本**（_glog），运维排查不受影响。
_VENDOR_TERMS = (
    ("Baiten", "CN source"), ("baiten", "CN source"),
    ("BAITEN", "CN"), ("佰腾", "中国专利"),
)


# 空格兜底（去括号纯空格词形，OR 语义）**默认关闭**（2026-09-13）。
# 生产实证：它每轮返回 20 条噪声，**全部**被后续 dead/评分过滤丢弃（30 条导出
# 里美国只剩 2 条），却照常进入 observation 摘要 —— 模型于是把其中"活着"的
# PCT/美国申请当成 Top 结果报给用户，实测 **9/10 在结果面板里根本不存在**。
# 即：纯成本（每请求 ~16 次 USPTO 调用）+ 污染模型判断，零收益。
# 设 REACT_USPTO_SPACE_FLATTEN=1 恢复旧行为。
USPTO_SPACE_FLATTEN_ENABLED = (
    os.getenv("REACT_USPTO_SPACE_FLATTEN", "0") == "1")


def _neutral_source_text(text) -> str:
    """把商业供应商名替换为中立表述。纯函数，永不抛。"""
    out = str(text or "")
    for vendor, neutral in _VENDOR_TERMS:
        out = out.replace(vendor, neutral)
    return out


async def _baiten_search_by_query(
    q: str, page: int = 1, page_size: int = 20, agent=None,
    enrich: bool = True,
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
    # 轮内复用（2026-09-15, 需求#27）：同一请求里重发的同一条检索式直接返回
    # 上次结果（生产实证：不同 US 同义词组轮询时，CN 阶梯被原样重跑）。
    # 缓存按请求重置（general_agent.create_agent 的重置块），不会跨请求泄漏。
    _cache = getattr(agent, "_search_result_cache", None)
    _cache_key = ("cn", q, int(page or 1), int(page_size or 20))
    if isinstance(_cache, dict) and _cache_key in _cache:
        if _glog is not None:
            _glog.info(
                f"baiten_search_map — query={q[:60]!r} "
                f"(cached, same turn)")
        return _cache[_cache_key]

    def _remember(payload):
        if isinstance(_cache, dict):
            _cache[_cache_key] = payload
        return payload

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
            return [], "CN source not configured (key missing)"
        client = BaitenClient(
            cfg["app_key"], cfg["app_secret"], cfg["gateway_url"])
        # 翻页（2026-09-15）：网关单页硬限 10 条，单查询恒 rows=10 —— 中文
        # 提问的 CN 供给被这一页卡死（生产：total 上千，我们只拿 10 条，
        # 结果被美国专利挤成少数）。按页续取到上限或没有更多为止。
        pages_wanted = max(1, REACT_CN_PAGES_PER_QUERY)
        page_no = int(page or 1)
        items: list = []
        seen_ids: set = set()
        summary: dict = {"total": None, "rows": 0, "keys": []}
        pages_with_data = 0
        last_rows = None
        rows_first = 0
        total_seen = None
        for _page_i in range(pages_wanted):
            body = await client.search(
                q, page=page_no, page_size=page_size,
                api_level=cfg.get("api_level", "ONE"))
            summary = summarize_search_response(body)
            rows_now = int(summary.get("rows") or 0)
            if rows_now == 0:
                break                      # 没有这一页
            if total_seen is None and summary.get("total") is not None:
                total_seen = int(summary["total"])
            if pages_with_data == 0:
                rows_first = rows_now
            for cand in _baiten_results_to_candidates(body):
                pid = str(cand.get("patent_id") or "").strip()
                if pid and pid in seen_ids:
                    continue
                if pid:
                    seen_ids.add(pid)
                items.append(cand)
            pages_with_data += 1
            if last_rows is not None and rows_now < last_rows:
                break                      # 比上一页短 = 到底了
            if total_seen is not None and len(items) >= total_seen:
                break                      # 已取满网关报的总数
            last_rows = rows_now
            page_no += 1
        if items and enrich:
            await _enrich_baiten_law_status(client, items, _glog, agent=agent)
        _total = total_seen
        _last_full = (last_rows is not None
                      and last_rows >= int(page_size or 20))
        if _glog is not None:
            # 需求#9：单查询恒 rows=page_size（页上限）时，真实命中数只有
            # 网关的 total 能回答 —— 一并落日志，别再靠"rows 恒 10"猜。
            _glog.info(
                f"baiten_search_map — query={q[:60]!r} "
                f"rows={last_rows} candidates={len(items)} "
                f"pages={pages_with_data} total={_total}"
                + (" (more available)" if _last_full else "")
            )
        if not pages_with_data:
            return _remember((items, "CN 0 hits (gateway 0 records)"))
        if not items:
            return _remember(([], (
                f"CN 0 candidates (parsed from "
                f"{rows_first} records)"
            )))
        _note = f"CN {len(items)} hits"
        if pages_with_data > 1:
            _note += f" over {pages_with_data} pages"
        if _total is not None and int(_total) > len(items):
            _note += f" (of {_total})"
        elif _total is None and _last_full:
            _note += f" (page-capped at {len(items)})"
        return _remember((items, _note))
    except Exception as exc:
        if _glog is not None:
            _glog.warning(f"baiten_search — failed: {exc}")
        return [], _neutral_source_text(f"CN source failed: {exc}")


# ── Dual-source query resolution + preferred-source auto-ladder ─────────────

# 自动补跑预算 (每源): applications/search 对短语查询命中不稳定
# (2026-09-01 日志: 8 条查询只试了 3 条, "RGB LED" 200 而 "RGB LED driver"
# 404), 提高批次与上限让单概念查询也有机会被尝试。
PATENT_AUTO_LADDER_BATCH = 4  # untried ladder queries auto-run per call
REACT_PATENT_AUTO_LADDER_MAX = int(os.getenv(
    "REACT_PATENT_AUTO_LADDER_MAX", "8"))

# 语言配额（2026-09-15，用户定）：提问**未指定国家**时（=双源工具），按提问
# 语言倾斜两侧供给 —— 中文提问多取中国专利、少取美国专利，英文提问反之。
# 首发检索式两侧照常各跑一次（不让任何一侧空手），收窄的是**非优选源的
# 阶梯补跑预算**与**摘要里非优选源的行数**。
# 生产 2026-09-15 实证：中文提问、池里 CN 30 条，模型仍答成"基本都是美国
# 专利"（US 侧 7 条阶梯检索式取回 83 条候选，CN 侧受网关每查询 10 条硬限）。
REACT_SEARCH_LANG_BALANCE = (
    os.getenv("REACT_SEARCH_LANG_BALANCE", "1") == "1")
# CN 翻页（2026-09-15）：网关单页硬限 10 条，单查询恒 rows=10 —— 中文提问的
# CN 供给被这一页卡死（生产实证：total 上千，我们只拿 10 条）。每查询最多
# 取几页；佰腾按调用计费，调大等于按倍数增加检索调用与后续法律状态富化量。
REACT_CN_PAGES_PER_QUERY = int(os.getenv("REACT_CN_PAGES_PER_QUERY", "2"))
REACT_NONPREFERRED_LADDER_MAX = int(os.getenv(
    "REACT_NONPREFERRED_LADDER_MAX", "3"))
REACT_NONPREFERRED_DIGEST_ROWS = int(os.getenv(
    "REACT_NONPREFERRED_DIGEST_ROWS", "8"))


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
          if isinstance(c, dict) and is_cn_source(c.get("source"))]
    others = [c for c in items
              if not (isinstance(c, dict) and is_cn_source(c.get("source")))]
    return cn + others


def _cn_item_to_pool_candidate(item: dict) -> dict:
    """Map a flat Baiten candidate to the relevance-pool candidate shape.

    The pool consumes patent_id/title/applicant/status/filing_date/
    patent_number/type_code/cpc_codes/_raw; Baiten candidates carry most
    of these natively — filing_date derives from apply_date/pub_date.

    ``app_num`` is carried through as well (需求#29): it is the CN
    application number, the key the Baiten legal-status / retrieval APIs
    actually accept.  Dropping it here is what made a delivered CN
    publication number impossible to read back.
    """
    return {
        "patent_id": str(item.get("patent_id") or ""),
        "title": str(item.get("title") or ""),
        "applicant": str(item.get("applicant") or ""),
        "status": str(item.get("status") or ""),
        "filing_date": str(item.get("apply_date")
                          or item.get("pub_date") or ""),
        "patent_number": str(item.get("patent_number") or ""),
        "app_num": str(item.get("app_num") or ""),
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
            if is_cn_source(item.get("source")):
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


def _is_format_rejection(note: str) -> bool:
    """需求#37: 该结果为「格式被拒」而非「查过但没命中」。

    USPTO applications/search 对超出方言上限的括号/AND 形态直接 404
    （``_uspto_search_by_query`` 把它标成 ``USPTO HTTP 404``），重试链已在中途
    穷尽，这条查询不含任何信息量。 CN 侧网关不产生此标记。
    """
    return "HTTP 404" in str(note or "")


async def _auto_run_patent_ladder(agent, ladder: list, search_fn, merged: list,
                                  notes: list, lang: str, source: str,
                                  page: int, page_size: int,
                                  max_queries: Optional[int] = None,
                                  stop_when_found: bool = False) -> int:
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
    cap = REACT_PATENT_AUTO_LADDER_MAX
    if max_queries is not None:
        cap = min(cap, max(0, int(max_queries)))
    # 需求#37: 被拒的检索式不占额度, 所以按"剩余额度"切片会让批次欠额
    # （额度只剩 1、补跑上限 4 时, 被拒的条目不计数 → 批次只发 1 条就停）。
    # 本批要发多少由"补跑上限"决定, 额度由循环内的 used >= cap 收口。
    take = untried[:PATENT_AUTO_LADDER_BATCH]
    if max_queries is not None:
        take = take[:max(0, int(max_queries))]

    if not take or used >= cap:
        # 无待发条目, 或源头额度已用尽。 未试阶梯必须留痕 —— 之前静默
        # return 0, 线上 total=0 无从排查(2026-08-29 事故); 且"没查"不能被
        # 模型读成"查过了没有"。
        _glog = getattr(agent, "logger", None)
        if _glog is not None:
            _glog.warning(
                f"patent_search_auto_ladder — source={source} auto-ladder "
                f"budget exhausted (used={used}/{cap}), {len(untried)} untried "
                f"ladder queries skipped"
            )
        if untried:
            label = "中国" if source == "cn" else "美国"
            notes.append(
                f"{label}专利阶梯式另有 {len(untried)} 条未执行"
                f"（本轮补跑额度已满）" if lang == "zh"
                else f"{len(untried)} more {source.upper()} ladder queries "
                     f"were not run (per-request auto-ladder budget spent)")
        return 0
    gained = 0
    executed: list = []
    format_rejected = 0
    used_at_entry = used
    for q in take:
        if used >= cap:
            break  # 额度本就有上限 —— 被拒的查询不占额度, 但也不能因此无限跑
        try:
            items, note = await search_fn(q, page=page, page_size=page_size)
        except Exception as exc:
            # 抛异常的查询**没有到达语料**: 不计费, 也不记 tried —— 与 HEAD 的
            # `continue` 一致, 让一次瞬时故障能在本请求内重试。
            notes.append(f"{source}: {exc}")
            continue
        # 需求#37: 计费按**结果**算, 不按发起算。 applications/search 对超限的
        # 括号/AND 形态直接 404 —— 那是格式被拒, 不是"查过但没命中", 扣额度等于
        # 让最松档的合规阶梯被"确定不会成功"的形态挤掉 (2026-09-19 生产:
        # used=3/3 后 2 条未试阶梯被跳过, 该轮美方 us_hits=0)。
        if _is_format_rejection(note):
            format_rejected += 1
        else:
            used_map[source] = used + 1
            used += 1
            agent._patent_auto_used = used_map
        executed.append(q)
        if q not in tried:
            tried.append(q)
        gained += len(items)
        merged.extend(items)
        if note:
            notes.append(note)
        if stop_when_found and gained > 0:
            # 非优选源：取到结果就收手 —— "少取"不等于"一次不中就归零"
            #（2026-09-15 生产：US 首发是真零 + 只放行 1 条补跑 → 整轮 0 条美国）。
            break
    if executed:
        _glog = getattr(agent, "logger", None)
        if _glog is not None:
            _glog.info(
                f"patent_search_auto_ladder — source={source} "
                f"queries={[q[:40] for q in executed]} gained={gained} "
                f"charged={used - used_at_entry} "
                f"format_rejected={format_rejected}"
            )
        label = "中国" if source == "cn" else "美国"
        rejected_tail = (
            f"，其中 {format_rejected} 条因检索式格式被拒(未计额度)"
            if format_rejected and lang == "zh"
            else (f" ({format_rejected} rejected by query syntax, "
                  f"not charged)" if format_rejected else ""))
        notes.append(
            f"已自动补跑{label}专利阶梯式 {len(executed)} 条"
            f"（并入 {gained} 条候选）{rejected_tail}" if lang == "zh"
            else f"Auto-ran {len(executed)} {source.upper()} ladder "
                 f"queries ({gained} candidates merged){rejected_tail}"
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
        # enrich=False：富化移出每轮检索的关键路径，改由本函数收尾统一跑
        # 一次（见下方 _enrich_cn_candidates_once）。此前每次佰腾检索都内联
        # await 富化（~1.5s），一次工具调用会串上 2-4 次。
        return await _baiten_search_by_query(
            q, page=page, page_size=page_size, agent=agent, enrich=False)

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
                    if isinstance(c, dict) and is_cn_source(c.get("source"))]
        return [c for c in merged
                if not (isinstance(c, dict) and is_cn_source(c.get("source")))]

    async def _run_for(source: str):
        q = cn_q if source == "cn" else us_q
        if not q or _source_cands(source):
            return
        ladder = cn_ladder if source == "cn" else us_ladder
        fn = _baiten if source == "cn" else _uspto
        # 语言配额：双源（未指定国家）时收窄非优选源的补跑预算 —— 但只收窄
        # "深度"，保证它有拿到结果的机会（取到即停，最多 REACT_NONPREFERRED_
        # LADDER_MAX 条）。
        _cap = None
        _stop = False
        if REACT_SEARCH_LANG_BALANCE and dual and source != preferred:
            _cap = REACT_NONPREFERRED_LADDER_MAX
            _stop = True
        await _auto_run_patent_ladder(
            agent, ladder, fn, merged, notes, lang, source,
            page, page_size, max_queries=_cap, stop_when_found=_stop)

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
    # 佰腾候选的法律状态在这里统一补一次（此前分散在每次检索内联 await）。
    # 放在排名之前：排名与排序要读 status。
    await _enrich_cn_candidates_once(agent, merged, _glog)

    pending = _merge_pending_items(
        getattr(agent, "_pending_raw_items", None), merged)
    ranked_pending = await _rank_builtin_patent_pool(agent, pending, lang)
    agent._pending_raw_items = _order_pending_for_lang(ranked_pending, lang)
    if _glog is not None:
        cn_hits = len([c for c in merged
                       if isinstance(c, dict) and is_cn_source(c.get("source"))])
        _glog.info(
            f"patent_search_result — us_hits={len(merged) - cn_hits} "
            f"cn_hits={cn_hits} total={len(merged)}"
        )

    # 需求#26：摘要只渲染**可能进入交付集**的结果。此前用 merged 全量渲染，
    # 模型于是会引用随即被失效过滤丢掉的行 —— 2026-09-13 生产实证：回答里
    # 列出的 New York Air Brake / Bendix 两条美国专利，导出文件里根本没有，
    # 用户照着 Top 榜去结果面板里找不到。
    deliverable = [c for c in merged
                   if isinstance(c, dict)
                   and not is_dead_status(c.get("status"))]
    # 语言配额：摘要里非优选源的行数也收窄 —— 模型的行文顺序跟着摘要走。
    _us_rows = _cn_rows = SEARCH_DIGEST_LIMIT
    if REACT_SEARCH_LANG_BALANCE and dual:
        if lang == "zh":
            _us_rows = REACT_NONPREFERRED_DIGEST_ROWS
        else:
            _cn_rows = REACT_NONPREFERRED_DIGEST_ROWS
    digest = _items_digest(deliverable, lang=lang,
                           us_limit=_us_rows, cn_limit=_cn_rows)
    if not digest:
        if merged:
            # 有命中但全部失效 —— 不能笼统说"未返回结果"，那会误导模型。
            digest = ("All hits were filtered out as no longer in force "
                      "(expired / abandoned)." if lang == "en"
                      else "检索有命中，但全部为已失效专利（过期/放弃），"
                           "无有效结果可展示。")
        else:
            digest = ("No results from any source." if lang == "en"
                      else "两个数据源均未返回结果。")
    if notes:
        if _glog is not None:
            _glog.info("patent_search_notes — " + "; ".join(notes))
        # 数据来源/状态噪声 (USPTO HTTP 404 / Baiten N hits / 自动补跑阶梯)
        # 只进日志, 不拼进用户可见的流式 observation 文本 (2026-09-01)。

    # #22c: US hits on a single-CN-publication search are usually citing
    # documents, not family members — the annotation MUST ride the digest
    # (LLM-visible text), because notes above only reach the logs.
    try:
        us_cands = [c for c in merged
                    if not (isinstance(c, dict) and is_cn_source(c.get("source")))]
        cn_cands = [c for c in merged
                    if isinstance(c, dict) and is_cn_source(c.get("source"))]
        citing_note = _us_citing_note(
            getattr(agent, "_last_user_prompt", "") or "",
            cn_q or "", cn_cands, us_cands, lang)
        if citing_note:
            digest = f"{digest}\n\n{citing_note}"
    except Exception:
        pass  # annotation is an enhancement — never breaks the search

    # 需求#25：把本轮**实际执行**的检索式带进模型可见的摘要 —— 用户要
    #「可复现检索式」时，模型必须能逐字复述，而不是凭记忆编造。只列真正
    # 发出过的式子（首轮 + 本请求内的阶梯补跑）。
    try:
        primary = []
        if us_q:
            primary.append(f"[US] {us_q}")
        if cn_q:
            primary.append(f"[CN] {cn_q}")
        extra = []
        seen_q = {us_q, cn_q}
        for q in (getattr(agent, "_tried_queries", None) or []):
            if q and q not in seen_q:
                seen_q.add(q)
                extra.append(q)
        if primary or extra:
            header = ("Query strings actually run this round (copyable):"
                      if lang == "en"
                      else "本轮实际执行的检索式（可复制）：")
            block = "\n".join(f"- {q}" for q in (primary + extra)[:12])
            digest = f"{digest}\n\n{header}\n{block}"
    except Exception:
        pass  # reproducibility block is an enhancement — never breaks the search
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


# ── Deterministic number resolution (USPTO + Baiten CN) ──────────────────────
# Sample #16 (2026-09-03): a bare-number question was closed after ONE
# USPTO 404 — no country disambiguation, no CN cross-check, no guidance.
# The functions below run the parse → primary-source → cross-source
# verification sequence system-side.  The shared per-request budget
# (agent._number_cross_used) bounds gateway calls whether the LLM invoked
# the built-in tool or a KB number tool 404'd and the cross hook fired.

async def _uspto_search_by_number(numbers: list) -> tuple[list, str]:
    """USPTO records for identifier numbers via the recall transport.

    applications/search's quoted free-text query matches the number in
    any bibliographic field (applicationNumberText / patentNumber), so a
    single call covers the grant-vs-application ambiguity of a bare digit
    string without guessing field syntax.
    """
    numbers = [str(n).strip() for n in (numbers or []) if str(n).strip()]
    if not numbers:
        return [], "USPTO: empty number list"
    try:
        from sources.long_task.recall_sources import fetch_by_numbers
        raw = await asyncio.to_thread(fetch_by_numbers, numbers)
        return _normalize_uspto_items(raw), f"USPTO {len(raw)} hits"
    except Exception as exc:
        return [], f"USPTO failed: {exc}"


async def _lookup_number_candidates(
    agent, candidates: list,
) -> tuple[list, list]:
    """Resolve parsed identifiers against their sources.

    Per candidate: its country's source runs first (CN → Baiten, US →
    USPTO); when that source returns nothing, the OTHER source is tried
    with the raw digits — the sample-#16 outcome (single-source 404
    closes the case) is structurally impossible here.  Leg results are
    collected into *notes* so the observation can state exactly what was
    checked.  Shared per-request gateway budget.  Never raises.
    """
    _glog = getattr(agent, "logger", None)
    used = int(getattr(agent, "_number_cross_used", 0) or 0)
    merged: list = []
    notes: list = []
    us_tried: set = set()
    # 需求#36: 中靶记录(号码自身)与非中靶记录必须可区分 —— 前者证明"这个号
    # 是什么", 后者只是"顺带检索到的相关件"。 打标后由调用方决定措辞。
    targets = [l for c in (candidates or []) for l in (c.get("lookups") or [])]

    def _tag_hits(items: list) -> None:
        for item in items:
            if isinstance(item, dict):
                hit = number_hit_kind(item, targets)
                if hit:
                    item["_number_hit"] = hit

    async def _baiten_leg(lookups: list) -> None:
        nonlocal used
        for q in lookups[:2]:
            if used >= NUMBER_CROSS_MAX_QUERIES:
                break
            used += 1
            try:
                items, note = await _baiten_search_by_query(
                    q, page=1, page_size=10, agent=agent)
            except Exception as exc:
                items, note = [], _neutral_source_text(
                    f"CN source failed: {exc}")
            notes.append(f"CN(q={q[:40]!r}) — {note}")
            if items:
                _tag_hits(items)
                merged.extend(items)
                return

    async def _uspto_leg(lookups: list) -> None:
        nonlocal used
        nums = []
        for lookup in lookups:
            digits = re.sub(r"\D", "", str(lookup or ""))
            if digits and digits not in us_tried:
                us_tried.add(digits)
                nums.append(digits)
        if not nums or used >= NUMBER_CROSS_MAX_QUERIES:
            return
        used += 1
        items, note = await _uspto_search_by_number(nums)
        notes.append(f"USPTO(nums={','.join(nums[:2])}) — {note}")
        if items:
            _tag_hits(items)
            merged.extend(items)

    async def _baiten_native_leg(c: dict) -> bool:
        """需求#29: 候选带来源原生键（CN 申请号）时按号直查。

        这是「系统读不回自己刚产出的号码」的正面修法——此前只能把
        CN 公开号当作佰腾的自由文本检索词发出去，而那不是可靠的查询键。
        """
        if str(c.get("native_key_kind") or "") != "app_num":
            return False
        native = str(c.get("native_key") or "").strip()
        if not native:
            return False
        result = await _baiten_law_lookup(agent, native)
        if not result:
            return False
        status = str(result.get("status") or "")
        notes.append(f"CN(app_num={native}) — {status or 'ok'}")
        merged.append({
            "patent_id": str(c.get("display") or native),
            "patent_number": str(c.get("display") or native),
            # 需求#36: 本条是**按原生申请号按键直查**取回的 —— 它就是该号
            # 本身, 无需再比对(号码可能只出现在 native_key 里, 比不到)。
            "_number_hit": native,
            "source": "cn",
            "app_num": native,
            "title": "",
            "status": status,
            "legal_timeline": result.get("timeline") or [],
            "review_decisions": result.get("reviews") or [],
            "reviews_checked": bool(result.get("reviews_checked")),
        })
        return True

    for c in (candidates or [])[:3]:
        country = str(c.get("country") or "")
        lookups = [l for l in (c.get("lookups") or []) if l]
        if not lookups:
            continue
        primary = "cn" if country == "CN" else "us"
        if primary == "cn" and await _baiten_native_leg(c):
            continue  # 原生键已命中 — 无需检索，也无需打对侧
        for source in (primary, "us" if primary == "cn" else "cn"):
            before = len(merged)
            if source == "cn":
                await _baiten_leg(lookups)
            else:
                await _uspto_leg(lookups)
            # 需求#36: 只有**中靶**才收手。 非中靶条目(全文检索的相关件)不能
            # 当作"主源已命中" —— 否则对侧复核被跳过, 而这个号本身可能就在
            # 对侧可查(2026-09-19: US 腿返回非中靶记录后即收手)。
            if any(m.get("_number_hit") for m in merged[before:]):
                break
        if len(merged) > 1:
            merged = _order_hits_first(merged)
    agent._number_cross_used = used
    return merged, notes


def number_hit_kind(item: dict, targets: list) -> str:
    """需求#36: 该记录是否**就是**查询的那个号码。

    ``fetch_by_numbers`` 发的是**全文检索式**（`"N" OR "M"`），不是按键的精确
    查找 —— 匹配落在任意著录字段上，因此"按号查"完全可能返回一堆不含该号的
    记录。 需求#40 已去掉该路径的相关度排序（那让 BM25 把短标题的申请件顶到
    前面），但自由文本匹配这条性质不变，中靶判定仍然必需。
    (2026-09-19 生产: 问 US12253745B2 拿到题名 LEAK DETECTOR 的记录, 模型
    据此断言"不符合", 用户连问三遍)。 判据取记录自带的两个号: US 侧
    ``patent_id`` = applicationNumberText、``patent_number`` = patentNumber;
    CN 侧 ``patent_id`` 是公开号。 数字归一化后等值比较, 非中靶返回 ""。
    """
    target_digits = {re.sub(r"\D", "", str(t or "")) for t in (targets or [])}
    target_digits.discard("")
    if not target_digits:
        return ""
    meta = item.get("applicationMetaData")
    if not isinstance(meta, dict):
        meta = {}
    # 三个号槽, 覆盖原始 USPTO 形态与扁平候选形态: 申请号在
    # applicationNumberText(原始)/patent_id(扁平), 授权号在
    # applicationMetaData.patentNumber(原始)/patent_number(扁平)。
    values = (
        item.get("applicationNumberText"), item.get("patent_id"),
        item.get("patent_number"), meta.get("patentNumber"),
    )
    for value in values:
        digits = re.sub(r"\D", "", str(value or ""))
        if digits and digits in target_digits:
            return digits
    return ""


def _order_hits_first(items: list) -> list:
    """中靶记录排首位, 其余保持原序(稳定排序保证)。"""
    return sorted(items, key=lambda i: 0 if i.get("_number_hit") else 1)


def number_non_hit_declaration(items: list, lang: str) -> str:
    """需求#36: 有记录、但**没有一条是该号码本身**时的边界声明。

    逐条复述这些相关件的题名容易被读成"这个号叫什么名字"（2026-09-19 生产：
    模型据此断言 US12253745B2 = LEAK DETECTOR）。两条入口——patch tool 的
    ``_run_patent_number_resolve`` 与 KB 零命中钩子 ``_auto_number_cross_round``
    ——都必须先把边界说清楚。无记录或已有中靶时返回 ""。
    """
    if not items or any(i.get("_number_hit") for i in items):
        return ""
    if lang == "en":
        return (f"⚠️ No record for the number itself was retrieved — the "
                f"{len(items)} item(s) below are RELATED records and must not "
                f"be reported as this number's bibliographic data.")
    return (f"⚠️ 未取得该号码本身的记录——以下 {len(items)} 条是与该号码"
            f"相关的其他记录，不能视为该号码的著录信息。")


def _pool_candidates_for_items(items: list) -> list:
    """Map mixed USPTO/Baiten display items into pool candidate shape."""
    out = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        if is_cn_source(item.get("source")):
            out.append(_cn_item_to_pool_candidate(item))
        else:
            out.extend(build_candidates([item]))
    return out


def _candidate_confirmation_hints(candidates: list, lang: str) -> str:
    """需求#24: zero-hit candidate card — never close a bare number with a
    plain "not found".  Lists the parsed candidates that were tried (with
    the parser's own country/reason wording) so the user can re-send an
    exact form.  Returns "" when there is nothing worth listing.
    """
    cands = [c for c in (candidates or [])
             if isinstance(c, dict) and str(c.get("display") or "").strip()]
    if not cands:
        return ""
    head = (
        "Did you mean one of these numbers? "
        "Re-send the exact form (prefix and kind letter help):"
        if lang == "en" else
        "您可能查的是（请用完整号码重试，含 CN/US 前缀与类型字母更精确）：")
    lines = [head]
    for c in cands:
        display = str(c.get("display") or "")
        country = str(c.get("country") or "")
        reason = str(c.get("reason") or "").strip()
        if reason:
            lines.append(f"- {display} — {country}: {reason}")
        else:
            lines.append(f"- {display} — {country}")
    return "\n".join(lines)


async def _run_patent_number_resolve(agent, args, lang: str) -> dict:
    """kind='patent_number' executor — deterministic dual-source lookup."""
    number_arg = str((args or {}).get("number") or "").strip()
    source_text = number_arg or (getattr(agent, "_last_user_prompt", "") or "")
    candidates: list = []
    try:
        from sources.patent_number_parser import (
            NUMBER_PARSE_ENABLED, parse_patent_identifiers)
        if NUMBER_PARSE_ENABLED:
            candidates = parse_patent_identifiers(source_text)
    except Exception:
        candidates = []
    if not candidates:
        candidates = getattr(agent, "_number_candidates", None) or []
    if not candidates:
        return {"kind": "observation", "text": (
            "Error: no recognizable patent number in the input — ask for "
            "the full number (CN/US prefix optional) or rephrase as a "
            "keyword search." if lang == "en"
            else "未能识别出专利号格式——请提供完整号码（可含 CN/US 前缀），"
                 "或用关键词描述技术内容进行检索。")}

    # 工具内已完成双源核验 —— 零命中钩子不得再重复执行。
    agent._number_cross_done = True
    merged, notes = await _lookup_number_candidates(agent, candidates)
    _glog = getattr(agent, "logger", None)
    if _glog is not None:
        _glog.info(
            "number_resolve — candidates="
            + str([c.get("display") for c in candidates])
            + " merged=" + str(len(merged))
            + " legs=" + "; ".join(notes))

    # 与 _run_patent_search 相同的尾部: 并入 pending、池化排序、落显示列表。
    pending = _merge_pending_items(
        getattr(agent, "_pending_raw_items", None), merged)
    ranked_pending = await _rank_builtin_patent_pool(agent, pending, lang)
    agent._pending_raw_items = _order_pending_for_lang(ranked_pending, lang)

    digest = _items_digest(merged, lang=lang)
    if not digest:
        checked = ("\n".join(f"- {n}" for n in notes)
                   if notes else "- (no source was queryable)")
        hints = _candidate_confirmation_hints(candidates, lang)
        if lang == "en":
            digest = (
                f"No records found for the number. Sources checked:\n{checked}")
            if hints:
                digest += f"\n\n{hints}"
        else:
            digest = f"未按该号码查到专利记录。已核验的数据源：\n{checked}"
            if hints:
                digest += f"\n\n{hints}"
    else:
        declaration = number_non_hit_declaration(merged, lang)
        if declaration:
            digest = f"{declaration}\n\n{digest}"
    return {"kind": "observation", "text": digest}


# ── 需求#18 法律状态直查（CN 优先）─────────────────────────────────────────
# 与号码工具的区别：号码工具回答"这个号是什么"，本工具回答"这个号现在
# 处于什么法律状态、有没有复审/无效决定"。数据源是佰腾 FLZT（状态事件流）
# 与 FSWX（复审/无效决定），二者都是**按键寻址**的网关方法（app_num 是
# query_law_infos 的文档参数），不涉及自由文本检索。


def _baiten_client_or_none(agent):
    """按配置构造 BaitenClient；未配置或构造失败返回 None。永不抛。"""
    try:
        from sources.baiten_client import BaitenClient
        from sources.long_task.config import get_baiten_config
        cfg = get_baiten_config()
        if not cfg.get("app_key") or not cfg.get("app_secret"):
            return None
        return BaitenClient(cfg["app_key"], cfg["app_secret"],
                            cfg["gateway_url"])
    except Exception as exc:
        # 配置损坏与"未配置"必须可区分 —— 否则用户看到的是"数据源未覆盖"，
        # 而真相是配置坏了，正是需求#18 要避免的能力误报。
        _glog = getattr(agent, "logger", None)
        if _glog is not None:
            _glog.warning(f"baiten client unavailable: {exc}")
        return None


async def _baiten_law_lookup(agent, app_num) -> dict:
    """一个 CN 申请号的法律状态时间线 + 复审/无效决定。

    返回 ``{status, status_date, category, timeline, reviews}``；任一步失败
    降级为 ``{}``，永不抛。计数记在 ``agent._legal_status_used``，
    **刻意不记** ``_number_cross_used`` —— 见 LEGAL_STATUS_MAX_LOOKUPS。
    """
    app_num = str(app_num or "").strip()
    if not app_num:
        return {}
    used = int(getattr(agent, "_legal_status_used", 0) or 0)
    if used >= LEGAL_STATUS_MAX_LOOKUPS:
        return {}
    client = _baiten_client_or_none(agent)
    if client is None:
        return {}
    agent._legal_status_used = used + 1
    _glog = getattr(agent, "logger", None)

    flzt = _law_flzt_cache(agent)
    if app_num in flzt:
        # 检索期已查过这个号 —— 同一请求内状态不会变，不再打网关。
        timeline = flzt[app_num]
    else:
        try:
            timeline = await asyncio.wait_for(
                client.query_legal_state_timeline(app_num), timeout=5)
        except Exception as exc:
            if _glog is not None:
                _glog.warning(f"legal status FLZT failed for {app_num}: {exc}")
            return {}
        flzt[app_num] = timeline or []

    from sources.long_task.legal_status import summarize_timeline
    summary = summarize_timeline(timeline, country="CN")
    if not summary["latest"]:
        return {}
    out: dict = {
        "status": summary["latest"],
        "status_date": summary["latest_date"],
        "category": summary["category"],
        "timeline": [
            {"date": str(e.get("date") or ""),
             "lawStatus": str(e.get("lawStatus") or "")}
            for e in (timeline or [])[:LAW_TIMELINE_MAX_ITEMS]
            if isinstance(e, dict) and e.get("lawStatus")
        ],
    }

    fswx = _law_fswx_cache(agent)
    reviews_checked = False
    if app_num in fswx:
        decisions = fswx[app_num]
        reviews_checked = True
    else:
        try:
            decisions = await asyncio.wait_for(
                client.query_patent_review(app_num), timeout=5)
            reviews_checked = True
            fswx[app_num] = decisions or []
        except Exception as exc:
            if _glog is not None:
                _glog.warning(f"legal status FSWX failed for {app_num}: {exc}")
            decisions = []
    out["reviews_checked"] = reviews_checked
    kept = []
    for d in (decisions or [])[:LAW_REVIEW_MAX_ITEMS]:
        if not isinstance(d, dict):
            continue
        kept.append({
            "declareNum": str(d.get("declareNum")
                              or d.get("declare_num") or ""),
            "declareDate": str(d.get("declareDate")
                               or d.get("declare_date") or ""),
            "lawBase": str(d.get("lawBase") or d.get("law_base") or ""),
            "fullText": str(d.get("fullText") or "")[
                :LAW_REVIEW_FULLTEXT_CHARS],
        })
    if kept:
        out["reviews"] = kept
    return out


def _legal_status_entries(merged: list, notes: list) -> list:
    """把已解析的记录映射成法律状态条目。纯映射，不发起调用。

    多数 CN 记录在检索阶段已被 `_enrich_baiten_law_status` 挂上
    ``status`` / ``legal_timeline`` / ``review_decisions`` —— 这里只读取，
    因此常见的号码查询是零额外网关调用的。

    ``covered`` 由"来源是否真的给了状态"决定，不由国别决定：没拿到就
    不能假装有，也不能假装查过。
    """
    entries = []
    for item in (merged or []):
        if not isinstance(item, dict):
            continue
        is_cn = is_cn_source(item.get("source"))
        display = str(item.get("patent_id")
                      or item.get("patent_number")
                      or item.get("applicationNumberText") or "").strip()
        if not display:
            continue
        status = str(item.get("status") or "").strip()
        entries.append({
            "display": display,
            "app_num": str(item.get("app_num") or "").strip(),
            "country": "CN" if is_cn else "US",
            "status": status,
            "status_date": "",
            "timeline": item.get("legal_timeline") or [],
            "reviews": item.get("review_decisions") or [],
            "reviews_checked": bool(item.get("reviews_checked")),
            "checked": ["cn_legal_status"] if is_cn else ["uspto"],
            "covered": bool(status),
        })
    return entries


# 数据源在 observation 里必须**中立表述**：商业供应商名一旦进入 LLM 可见
# 面，模型会在回答里照抄（2026-09-13 生产实证：「中国专利（佰腾）」）。
# 条目里存**内部键**，展示名在这里映射 —— 数据与展示分离，改文案不动数据。
_SOURCE_DISPLAY = {
    "cn_legal_status": {"zh": "中国专利法律状态库",
                        "en": "China patent legal-status source"},
    "uspto": {"zh": "USPTO", "en": "USPTO"},
}


def _source_display(key, lang: str) -> str:
    labels = _SOURCE_DISPLAY.get(str(key))
    if not labels:
        return str(key)
    return labels["en"] if str(lang) == "en" else labels["zh"]


def _legal_status_digest(entries: list, lang: str) -> str:
    """法律状态 observation 渲染。双语，永不抛。

    每个条目只陈述记录载明的状态与事件；来源没覆盖的部分显式标注
    「未覆盖」并给出官方查询入口 —— 需求#18 的诚实边界要求。
    """
    from sources.long_task.legal_status import (
        LEGAL_STATUS_DISCLAIMER, classify_status, official_portal,
        summarize_review_decisions)
    is_en = str(lang) == "en"
    blocks: list = []
    for e in (entries or []):
        if not isinstance(e, dict):
            continue
        display = str(e.get("display") or "").strip()
        if not display:
            continue
        country = str(e.get("country") or "")
        app_num = str(e.get("app_num") or "").strip()
        lines: list = []
        if app_num:
            lines.append(f"{display}（申请号 {app_num}）" if not is_en
                         else f"{display} (application {app_num})")
        else:
            lines.append(display)

        if e.get("covered"):
            cls = classify_status(e.get("status"), country=country)
            label = cls["en"] if is_en else cls["zh"]
            date = str(e.get("status_date") or "").strip()
            if is_en:
                lines.append(f"Legal status: {label}"
                             + (f" ({date})" if date else ""))
            else:
                lines.append(f"法律状态：{label}"
                             + (f"（{date}）" if date else ""))
            timeline = [t for t in (e.get("timeline") or [])
                        if isinstance(t, dict)
                        and (t.get("date") or t.get("lawStatus"))]
            if timeline:
                lines.append("Timeline:" if is_en else "状态时间线：")
                for t in timeline[:LAW_TIMELINE_MAX_ITEMS]:
                    lines.append(
                        f"- {t.get('date', '')} {t.get('lawStatus', '')}".rstrip())
            if country == "CN":
                rendered = summarize_review_decisions(
                    e.get("reviews") or [], lang=lang)
                if rendered:
                    lines.append(rendered)
                elif e.get("reviews_checked"):
                    lines.append(
                        "No re-examination / invalidation decision records "
                        "found." if is_en
                        else "未检索到复审/无效决定记录。")
                else:
                    # 没查过就不能说"没有"——无据的否定比不回答更糟。
                    lines.append(
                        "Re-examination / invalidation decisions: not "
                        "queried in this lookup." if is_en
                        else "复审/无效决定：本次未查询。")
        else:
            checked = ("、" if not is_en else ", ").join(
                _source_display(k, lang) for k in (e.get("checked") or []))
            portal = official_portal(country)
            if is_en:
                lines.append(
                    "Legal-status data for this number is not covered by "
                    f"this system (sources checked: {checked or '-'}).")
            else:
                lines.append(
                    "未获取到该号码的法律状态（已核验："
                    f"{checked or '-'}）。本系统暂未覆盖该来源的法律状态数据。")
            if portal:
                lines.append(f"Official lookup: {portal}" if is_en
                             else f"官方查询入口：{portal}")
        blocks.append("\n".join(lines))

    if not blocks:
        return ""
    blocks.append(LEGAL_STATUS_DISCLAIMER["en"] if is_en
                  else LEGAL_STATUS_DISCLAIMER["zh"])
    return "\n\n".join(blocks)


async def _fill_missing_status(agent, entries: list) -> list:
    """补齐没带状态的 CN 记录：用其申请号按号直查。

    检索期附带的富化可能失败，于是记录落到这里时 ``covered=False``。但
    只要手里有申请号就还能查——有键不查、转头对用户说"未覆盖"，是能力
    上的谎报。没有键才如实标注。
    """
    for entry in entries:
        if entry.get("covered") or str(entry.get("country")) != "CN":
            continue
        app_num = str(entry.get("app_num") or "").strip()
        if not app_num:
            continue
        result = await _baiten_law_lookup(agent, app_num)
        if not result:
            continue
        entry["status"] = str(result.get("status") or "")
        entry["status_date"] = str(result.get("status_date") or "")
        entry["timeline"] = result.get("timeline") or []
        entry["reviews"] = result.get("reviews") or []
        entry["reviews_checked"] = bool(result.get("reviews_checked"))
        entry["covered"] = bool(entry["status"])
    return entries


async def _run_patent_legal_status(agent, args, lang: str) -> dict:
    """kind='patent_legal_status' executor —— 确定性法律状态查询。"""
    number_arg = str((args or {}).get("number") or "").strip()
    source_text = number_arg or (getattr(agent, "_last_user_prompt", "") or "")
    candidates: list = []
    try:
        from sources.patent_number_parser import (
            NUMBER_PARSE_ENABLED, parse_patent_identifiers)
        if NUMBER_PARSE_ENABLED:
            candidates = parse_patent_identifiers(source_text)
    except Exception:
        candidates = []
    if not candidates:
        candidates = getattr(agent, "_number_candidates", None) or []
    if not candidates:
        return {"kind": "observation", "text": (
            "Error: no recognizable patent number in the input — ask for "
            "the full number (CN/US prefix optional) or rephrase as a "
            "keyword search." if lang == "en"
            else "未能识别出专利号格式——请提供完整号码（可含 CN/US 前缀），"
                 "或用关键词描述技术内容进行检索。")}

    merged, notes = await _lookup_number_candidates(agent, candidates)
    entries = _legal_status_entries(merged, notes)
    entries = await _fill_missing_status(agent, entries)
    digest = _legal_status_digest(entries, lang)

    if not digest:
        # 需求#24 行为延续：禁止以"未找到"直接结案。
        checked = ("\n".join(f"- {n}" for n in notes)
                   if notes else "- (no source was queryable)")
        hints = _candidate_confirmation_hints(candidates, lang)
        if lang == "en":
            digest = (f"No legal-status record found for the number. "
                      f"Sources checked:\n{checked}")
            if hints:
                digest += f"\n\n{hints}"
        else:
            digest = f"未按该号码查到法律状态记录。已核验的数据源：\n{checked}"
            if hints:
                digest += f"\n\n{hints}"

    _glog = getattr(agent, "logger", None)
    if _glog is not None:
        _glog.info(
            "legal_status — candidates="
            + str([c.get("display") for c in candidates])
            + " entries=" + str(len(entries))
            + " covered=" + str(sum(1 for e in entries if e.get("covered")))
            + " legs=" + "; ".join(notes))
    return {"kind": "observation", "text": digest}


async def _auto_number_cross_round(agent, lang) -> Optional[Tuple[list, str, str]]:
    """Zero-hit cross-source verification for number questions.

    Fires once per request when a KB number tool (or any search) returned
    nothing displayable AND the request carried parsed identifiers —
    the deterministic dual-source lookup then still checks the other
    country's source.  Returns the same triple contract as
    ``_auto_ladder_round`` (ranked pool candidates, ranking note,
    executed note) or None when there is nothing to run.  Never raises.
    """
    candidates = getattr(agent, "_number_candidates", None) or []
    if not candidates or getattr(agent, "_number_cross_done", False):
        return None
    agent._number_cross_done = True
    merged, notes = await _lookup_number_candidates(agent, candidates)
    _glog = getattr(agent, "logger", None)
    if _glog is not None:
        _glog.info(
            "number_cross_check — candidates="
            + str([c.get("display") for c in candidates])
            + " merged=" + str(len(merged))
            + " legs=" + "; ".join(notes))
    executed = "\n".join(f"- {n}" for n in notes) if notes else ""
    executed = executed or ("- nothing to check" if lang == "en"
                            else "- 无可用数据源")
    if merged:
        pool_cands = _pool_candidates_for_items(merged)
        ranked, _note = await _rank_pending_pool(
            agent, pool_cands, lang, apply_rerank=False)
        cross_note = (
            f"Cross-source number verification found "
            f"{len(ranked)} record(s):\n{executed}" if lang == "en"
            else f"号码跨源复核命中 {len(ranked)} 条：\n{executed}")
        # 需求#36: 与 patch tool 入口同一条边界, 不得只报"复核命中 N 条"。
        declaration = number_non_hit_declaration(merged, lang)
        if declaration:
            cross_note = f"{declaration}\n\n{cross_note}"
        return ranked, "", cross_note
    cross_note = (
        f"Cross-source number verification found nothing:\n{executed}"
        if lang == "en"
        else f"号码跨源复核无命中（已按候选做双库核验）：\n{executed}")
    return [], "", cross_note


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

        if entry.kind == "patent_number":
            return await _run_patent_number_resolve(agent, args, lang)

        if entry.kind == "patent_legal_status":
            return await _run_patent_legal_status(agent, args, lang)

        if entry.kind == "patent_search":
            return await _run_patent_search(
                agent, args, lang,
                dual=(entry.name == DUAL_PATENT_SEARCH_TOOL_NAME))

        if entry.kind == "search":
            return await _run_search_knowledge(agent, registry, user_id, args, push_filter)

        if entry.kind == "long_task":
            # 资格门复查（需求#1，防御纵深）：即便注册表来自别的构建路径，
            # 无专利引用的文本诉求也不得进入分析管道（否则必然
            # no_patents_found 空转数分钟）。
            if not _is_long_task_eligible(
                    getattr(agent, "_last_user_prompt", "") or "",
                    getattr(agent, "_conversation_history", None)):
                return {
                    "kind": "observation",
                    "text": (
                        "该请求未包含具体专利号或前序结果引用，无需深度分析"
                        "任务，请直接回答用户或改用专利检索工具。"
                        if lang == "zh" else
                        "No patent number or prior-result reference in this "
                        "request — answer directly or use the patent search "
                        "tools instead of submitting a deep task."),
                }
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

        # 括号/引号结构在 applications/search 上 404 是常态 — KB 工具
        # 首次调用落空时, 自动以空格词形重试一次 (观察 2026-09-03:
        # 8/8 括号查询 404, 空格词形 200; 噪声由 relevance gate 兜底)。
        if (not getattr(agent, "_pending_raw_items", None)
                and q_used and _is_uspto_search_tool(entry.tool_info)):
            flat_q = _flatten_query_for_uspto(q_used)
            if flat_q and flat_q != q_used:
                _glog = getattr(agent, "logger", None)
                if _glog is not None:
                    _glog.info(
                        "uspto_query_bracket_fallback — "
                        f"q0={q_used[:90]!r} flat={flat_q[:90]!r}")
                retry_args = _with_query_replaced(invoke_args, flat_q)
                try:
                    result = await asyncio.to_thread(
                        entry.tool.invoke, retry_args)
                except Exception as exc:
                    return {"kind": "observation", "text": f"Error: {exc}"}

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
                # dead-filtered.  For number questions the OTHER country's
                # source is verified first (a single-source 404 must not
                # close the case — sample #16); otherwise execute the
                # untried ladder queries system-side (the agent cannot
                # conclude early while they remain untried), then point
                # the agent at whatever is still left.
                auto = await _auto_number_cross_round(agent, lang)
                if auto is None:
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
