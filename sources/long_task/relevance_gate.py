"""Relevance gating for PHASE0 search results.

Ranks search candidates against the user's original question via the
Flash LLM, keeps only relevant ones, dedupes families, and pages
through additional search results / rewritten queries when the kept
set falls short of the target count.
"""

import json
from copy import deepcopy
from typing import Any

from sources.long_task.candidate_metadata import (
    build_candidates,
    dedupe_candidates,
    ensure_search_fields,
    is_keyword_search_tool,
)
from sources.long_task.search_query_builder import build_search_queries
from sources.long_task.scene_tools import execute_tool

GATE_MIN_SCORE = 3
GATE_MAX_CANDIDATES_PER_CALL = 100
MAX_SEARCH_QUERIES = 3
MAX_PAGES_PER_QUERY = 4
MAX_COLLECTED_CANDIDATES = 300

GATE_SYSTEM_PROMPT = (
    "你是一个专利相关性评分器。判断候选专利与用户原始问题是否相关。\n"
    "评分标准（0-5）：\n"
    "5 — 直接解决用户问题中的技术需求\n"
    "4 — 高度相关，属于同一技术方向\n"
    "3 — 相关，可提供背景或部分覆盖\n"
    "2 — 仅表面相关（共享个别词语但技术方向不同）\n"
    "1 — 基本不相关\n"
    "0 — 完全不相关\n"
    "≥3 分视为相关。依据标题、申请人、CPC 分类码、日期、法律状态判断，"
    "不要猜测未知内容。CPC 分类码是强信号：与用户问题所属技术"
    "领域同族的分类码应显著加分。法律状态也是强信号：已失效"
    "（Expired）、已放弃（Abandoned）、已入库存档（Placed in "
    "storage）的案件没有可实施的权利，最高评 2 分（仅供技术参考）。"
    "外观设计（Design patent，专利号以 D 开头）只保护产品外观，"
    "不涉及技术方案，最高评 2 分。\n"
    'Return JSON: {"scores": [{"id": "<patent_id>", "score": <0-5>}]} '
    "（每个候选一条，id 必须与输入完全一致）"
)


# ── Scoring ─────────────────────────────────────────────────────────────────

def _batch_text(candidates: list[dict], query: str) -> str:
    lines = [
        f"- id={c['patent_id']} | title={c.get('title') or '(no title)'}"
        f" | applicant={c.get('applicant') or '?'}"
        f" | filing={c.get('filing_date') or '?'}"
        f" | status={c.get('status') or '?'}"
        f" | pn={c.get('patent_number') or '?'}"
        f" | cpc={c.get('cpc_codes') or []}"
        for c in candidates
    ]
    return f"用户原始问题：{query}\n\n候选专利：\n" + "\n".join(lines)


def apply_scores(candidates: list[dict], result: Any) -> list[dict]:
    """Attach ``relevance_score`` from LLM output. Never raises."""
    by_id = {str(c["patent_id"]): c for c in candidates}
    scores = (result or {}).get("scores") if isinstance(result, dict) else None
    if not isinstance(scores, list):
        return candidates
    for entry in scores:
        if not isinstance(entry, dict):
            continue
        c = by_id.get(str(entry.get("id") or ""))
        if c is None:
            continue
        try:
            score = int(entry.get("score", -1))
        except (TypeError, ValueError):
            continue
        if 0 <= score <= 5:
            c["relevance_score"] = score
    return candidates


async def score_candidates(
    candidates: list[dict], query: str, provider: Any,
) -> list[dict]:
    """Score candidates in batches via the Flash LLM. Never raises."""
    if not candidates:
        return candidates
    out = []
    for i in range(0, len(candidates), GATE_MAX_CANDIDATES_PER_CALL):
        batch = candidates[i:i + GATE_MAX_CANDIDATES_PER_CALL]
        try:
            result = await provider.complete_json(
                GATE_SYSTEM_PROMPT, _batch_text(batch, query),
            )
        except Exception:
            result = None
        out.extend(apply_scores(batch, result))
    return out


def filter_by_relevance(
    candidates: list[dict], min_score: int = GATE_MIN_SCORE,
) -> list[dict]:
    """Keep candidates at/above the threshold, sorted by score desc.

    Candidates WITHOUT a score (gate LLM failure or partial output) are
    kept — a transient provider error must not zero out a search run;
    the pipeline degrades to legacy behavior instead.
    """
    kept = []
    for c in candidates:
        score = c.get("relevance_score")
        if score is None or (isinstance(score, (int, float)) and score >= min_score):
            kept.append(c)
    kept.sort(key=lambda c: c.get("relevance_score")
              if isinstance(c.get("relevance_score"), (int, float)) else -1,
              reverse=True)
    return kept


# ── Gated search loop ───────────────────────────────────────────────────────

def _page_size(params: dict) -> int:
    body = params.get("body")
    if isinstance(body, dict):
        pag = body.get("pagination")
        if isinstance(pag, dict) and isinstance(pag.get("limit"), int):
            return pag["limit"]
    return 50


def _with_offset(params: dict, offset: int) -> dict:
    out = deepcopy(params)
    body = out.get("body")
    if isinstance(body, dict):
        pag = body.get("pagination")
        if not isinstance(pag, dict):
            pag = {}
            body["pagination"] = pag
        pag["offset"] = offset
    return out


def _total_count(result: dict) -> int:
    data = result.get("data")
    if isinstance(data, dict):
        for k in ("count", "total"):
            v = data.get(k)
            if isinstance(v, (int, float)):
                return int(v)
    return 0


async def run_gated_search(
    selected: dict,
    user_query: str,
    provider: Any,
    rewrite: dict,
    target_count: int,
    task_id: str = "",
    logger: Any = None,
) -> dict:
    """Execute the search tool with rewritten queries, page through
    results, gate by relevance, and dedupe.

    Returns ``{"candidates": [...], "search_meta": {...}}``. Candidate
    dicts carry the full metadata from ``build_candidates`` plus
    ``relevance_score``.
    """
    def _log(msg: str) -> None:
        if logger is not None:
            logger.info(f"[task={task_id}] {msg}")

    tool = selected["tool"]
    base_params = selected.get("params") or {}
    queries = (rewrite or {}).get("queries") or []
    if not queries:
        queries = [None]  # keep the LLM-built params untouched

    keyword_tool = is_keyword_search_tool(tool)
    all_candidates: list[dict] = []
    seen_ids: set[str] = set()
    total_hits = 0
    pages_fetched = 0

    for q in queries[:MAX_SEARCH_QUERIES]:
        offset = 0
        for _page in range(MAX_PAGES_PER_QUERY):
            if len(all_candidates) >= MAX_COLLECTED_CANDIDATES:
                break
            params = ensure_search_fields(base_params)
            if keyword_tool and q:
                body = params.get("body")
                if isinstance(body, dict):
                    body["q"] = q
            params = _with_offset(params, offset)
            _log(f"gated_search request — q={q!r}, offset={offset}")
            result = await execute_tool(tool, params)
            raw_items = result.get("raw_items") or []
            total_hits = max(total_hits, _total_count(result))
            fresh = [c for c in build_candidates(raw_items)
                     if c["patent_id"] not in seen_ids]
            pages_fetched += 1
            if not fresh:
                break
            for c in fresh:
                seen_ids.add(c["patent_id"])
            all_candidates.extend(fresh)
            if len(raw_items) < _page_size(base_params):
                break
            offset += len(raw_items)

    _log(f"gated_search collected — candidates={len(all_candidates)}, "
         f"total_hits={total_hits}, pages={pages_fetched}")

    scored = await score_candidates(all_candidates, user_query, provider)
    kept = filter_by_relevance(scored)
    deduped, dropped = dedupe_candidates(kept)
    final = deduped[:target_count]
    search_meta = {
        "queries_used": [q for q in queries[:MAX_SEARCH_QUERIES] if q],
        "total_hits": total_hits,
        "pages_fetched": pages_fetched,
        "candidates_scored": len(scored),
        "gated_kept": len(kept),
        "gated_dropped": len(scored) - len(kept),
        "deduped_dropped": dropped,
        "final_count": len(final),
    }
    _log(f"gated_search done — meta={json.dumps(search_meta, ensure_ascii=False)}")
    return {"candidates": final, "search_meta": search_meta}


async def phase0_gated_search(
    selected: dict,
    user_query: str,
    provider: Any,
    target_count: int,
    task_id: str = "",
    logger: Any = None,
) -> dict:
    """PHASE0 entry point: rewrite the query, then run the gated search.

    LLM failures (rewrite or scoring) degrade gracefully; execution
    failures propagate like the legacy single-shot path.
    """
    rewrite = await build_search_queries(user_query, provider)
    if logger is not None:
        logger.info(
            f"[task={task_id}] search_rewrite — "
            f"queries={rewrite['queries']}"
        )
    return await run_gated_search(
        selected=selected, user_query=user_query, provider=provider,
        rewrite=rewrite, target_count=target_count,
        task_id=task_id, logger=logger,
    )
