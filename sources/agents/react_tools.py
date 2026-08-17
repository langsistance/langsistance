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
    candidates = build_candidates(items)
    if candidates:
        lines = []
        for c in candidates[:limit]:
            parts = [
                c.get("patent_id") or "?",
                c.get("title") or "(无标题)",
                c.get("applicant") or "?",
                c.get("filing_date") or "?",
                c.get("status") or "?",
            ]
            lines.append(" | ".join(str(p) for p in parts))
        text = "\n".join(lines)
        if len(candidates) > limit:
            note = (f"\n…共 {len(candidates)} 条" if lang == "zh"
                    else f"\n...{len(candidates)} items total")
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
    assignee, or otherwise — merges into the turn's ranked pool)."""
    if not _relevance_pool_applies_tool(agent, tool_info):
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

    def _is_session_sentinel(value) -> bool:
        """True when the LLM pasted a session ID (user_id / query_id) into
        the query slot instead of a search expression — observed in
        production logs (q filled with the user_id)."""
        text = str(value or "").strip()
        if not text:
            return False
        uid = str(getattr(agent, "_last_user_id", "") or "")
        qid = str(getattr(agent, "_last_query_id", "") or "")
        return text == uid or text == qid

    def _blank(value) -> bool:
        return not str(value or "").strip() or _is_session_sentinel(value)

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

        if entry.kind == "search":
            return await _run_search_knowledge(agent, registry, user_id, args, push_filter)

        if entry.kind == "long_task":
            # The loop terminates and core.py's existing long-task branch
            # handles classification + Celery submission.
            return {"kind": "long_task", "text": "",
                    "knowledge": entry.knowledge, "tool_info": entry.tool_info}

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
                ranked, note = await _rank_pending_pool(agent, collected, lang)
                ranked, note = await _auto_second_round(
                    agent, entry, args, ranked, note, lang)
                shown = [c["_raw"] for c in ranked]
                agent._pending_raw_items = shown
                agent._search_ranked = True
                digest = _ranked_digest(ranked, lang=lang)
            else:
                shown, note = _cap_patent_list(entry.tool_info, pending, lang)
                pool = getattr(agent, "_search_pool", None)
                if pool is not None:
                    # The tool function already wrote this legacy result
                    # into _pending_raw_items; restore the turn's ranked
                    # pool as the display list — the legacy result still
                    # feeds the observation digest below.
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
            return {"kind": "observation", "text": text}

        return {"kind": "observation", "text": _summarize_observation(result, lang)}

    return execute_action
