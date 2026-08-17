"""Post-retrieval grounded interpretation of the user question.

The pre-retrieval interpretation (``technical_interpretation``) maps
the question to architecture vocabulary from model knowledge.  Its
players/main lines are knowledge-layer, though — production showed
lighting giants instead of the domain's specialised players.  This
module grounds the interpretation in the actual candidate pool: it
counts applicant/CPC frequencies over the scored candidates, asks the
Flash LLM to cluster the top candidates under the pre-interpretation's
dimension skeleton, and returns per-dimension main lines with
representatives plus supplementary queries/CPC codes for the loop.

Enhancement, not a dependency: any failure degrades to a stats-only
version (applicant frequency + CPC title groups), and a stats failure
returns None so callers keep their flow untouched.  The prompt is
generic — the question is passed at runtime, never baked in.
"""

import asyncio
import json
import os
from typing import Any, Optional

from sources.long_task.search_query_builder import sanitize_uspto_query

GROUNDED_ENABLED = os.getenv("REACT_GROUNDED_ENABLED", "1") == "1"
GROUNDED_MODEL = os.getenv("REACT_GROUNDED_MODEL", "deepseek-v4-flash")
GROUNDED_PROVIDER = os.getenv("REACT_GROUNDED_PROVIDER", "deepseek")
GROUNDED_HEAD = int(os.getenv("REACT_GROUNDED_HEAD", "30"))
GROUNDED_MIN = int(os.getenv("REACT_GROUNDED_MIN", "15"))
MAX_GROUNDED_DIMENSIONS = 3
MAX_GROUNDED_CANDIDATES = 30


def _env_int(name: str, default: int) -> int:
    """Parse an int env var, falling back to *default* on garbage."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


GROUNDED_TIMEOUT = _env_int("REACT_GROUNDED_TIMEOUT", 60)

# Provider construction is expensive and must stay lazy; cache one.
_GROUNDED_PROVIDER_CACHE: dict = {}


def _grounded_provider():
    if "provider" not in _GROUNDED_PROVIDER_CACHE:
        from sources.llm_provider import Provider
        _GROUNDED_PROVIDER_CACHE["provider"] = Provider(
            provider_name=GROUNDED_PROVIDER, model=GROUNDED_MODEL,
            server_address="", is_local=False)
    return _GROUNDED_PROVIDER_CACHE["provider"]


def candidate_stats(candidates: list) -> dict:
    """Applicant and CPC frequencies over the candidates, desc.

    Pure code, zero LLM: this layer survives any model failure.
    """
    applicants: dict = {}
    cpc: dict = {}
    for c in candidates or []:
        if not isinstance(c, dict):
            continue
        name = str(c.get("applicant") or "").strip()
        if name:
            applicants[name] = applicants.get(name, 0) + 1
        for code in (c.get("cpc_codes") or []):
            code = str(code).strip().upper()
            if code:
                cpc[code] = cpc.get(code, 0) + 1

    def _top(counts: dict) -> list:
        return [{"name": k, "count": v} for k, v in
                sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    return {"applicants": _top(applicants), "cpc": _top(cpc)}


def build_synthesis_input(question: str, pre_interp: Optional[dict],
                          candidates: list, stats: dict,
                          cpc_hints: Optional[list] = None) -> dict:
    """Assemble the flash synthesis payload from deterministic facts."""
    pre: dict = {}
    if pre_interp:
        pre["scheme"] = str(pre_interp.get("scheme") or "")
        pre["dimensions"] = [
            {"name": str(d.get("name") or ""),
             "role": str(d.get("role") or ""),
             "terms": [str(t) for t in (d.get("terms") or [])[:6]]}
            for d in (pre_interp.get("dimensions") or [])[:3]
            if isinstance(d, dict)]
        pre["structure_terms"] = [
            str(t) for t in (pre_interp.get("structure_terms") or [])[:10]]
    cands = []
    for c in (candidates or [])[:MAX_GROUNDED_CANDIDATES]:
        if not isinstance(c, dict):
            continue
        cands.append({
            "id": str(c.get("patent_id") or ""),
            "title": str(c.get("title") or ""),
            "applicant": str(c.get("applicant") or ""),
            "cpc": [str(x).upper() for x in (c.get("cpc_codes") or [])[:5]],
            "score": c.get("relevance_score"),
            "filing": str(c.get("filing_date") or ""),
        })
    return {
        "question": str(question),
        "pre_interpretation": pre,
        "candidates": cands,
        "applicant_stats": (stats or {}).get("applicants") or [],
        "cpc_stats": (stats or {}).get("cpc") or [],
        "cpc_hints": [
            {"code": str(h.get("code", "")), "title": str(h.get("title", ""))}
            for h in (cpc_hints or [])[:8] if isinstance(h, dict) and h.get("code")],
    }


GROUNDED_SYSTEM_PROMPT = (
    "你是资深专利检索专家。系统给出一句技术需求、预检索技术解读"
    "（含技术维度骨架）、以及从检索结果抽取的候选专利事实（标题/"
    "申请人/CPC/相关度评分/申请年）与统计（申请人频次、CPC 频次）。\n"
    "你的任务：基于候选数据产出检索后接地解读。只输出 JSON，不要其他文字。\n"
    "规则：\n"
    "1. 以预解读的维度骨架为初始结构，依据候选数据验证/调整/合并/拆分"
    "维度；每个维度输出：name（维度名）、role（分层角色）、line（该"
    "维度在专利文献中的主线描述，一句话，含典型实现方式）、"
    "representatives（1-3 个代表申请人，必须来自候选数据中真实出现"
    "的申请人）、players（该维度活跃申请人 2-5 个，来自数据）、"
    "cpc（该维度对应 CPC 代码 1-3 个，来自数据统计）\n"
    "2. representatives 与 players 只能来自给定的申请人频次与候选"
    "数据，禁止编造；数据不足的维度宁可少给\n"
    "3. supplementary_queries：针对候选覆盖不足或缺失方向的布尔检索式"
    "2-4 条，方案词优先、可直接执行；多词短语加双引号、同组同义词 "
    "OR、组间 AND；每条最多 12 个关键词、250 字符；禁止中文\n"
    "4. supplementary_cpc：主线对应的 CPC 代码 1-4 个，来自 cpc 频次"
    "统计\n"
    "5. players：顶层字段，该领域最活跃的真实申请人 3-5 个（英文"
    "公司名），从 applicant_stats 与候选数据统计得出，禁止编造；"
    "与各维度内 players 可重合\n"
    'Return JSON: {"dimensions": [{"name", "role", "line", '
    '"representatives", "players", "cpc"}], "players": [...], '
    '"supplementary_queries": [...], "supplementary_cpc": [...]}'
)


def _clean_strs(raw: Any, key: str, cap: int) -> list:
    items = raw.get(key) if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return []
    return [str(v).strip() for v in items
            if isinstance(v, str) and str(v).strip()][:cap]


def parse_grounded(raw: Any) -> Optional[dict]:
    """Validate/sanitize the synthesis LLM output.

    None when nothing usable — callers fall back to the stats-only
    version.  Queries go through the same sanitizer as pre-interpretation
    queries; CPC codes are uppercased and deduped.
    """
    if not isinstance(raw, dict):
        return None
    dims: list = []
    for d in (raw.get("dimensions") or []):
        if not isinstance(d, dict):
            continue
        name = str(d.get("name") or "").strip()
        line = str(d.get("line") or "").strip()
        if not name and not line:
            continue
        dims.append({
            "name": name,
            "role": str(d.get("role") or "").strip(),
            "line": line,
            "representatives": _clean_strs(d, "representatives", 3),
            "players": _clean_strs(d, "players", 5),
            "cpc": list(dict.fromkeys(
                c.upper() for c in _clean_strs(d, "cpc", 3) if c and c.upper())),
        })
        if len(dims) >= MAX_GROUNDED_DIMENSIONS:
            break
    players = _clean_strs(raw, "players", 5)
    queries: list = []
    qseen: set = set()
    for q in (raw.get("supplementary_queries") or []):
        if not isinstance(q, str):
            continue
        q = sanitize_uspto_query(q)
        if q and q not in qseen:
            qseen.add(q)
            queries.append(q)
    cpc = list(dict.fromkeys(
        c.upper() for c in (raw.get("supplementary_cpc") or [])
        if isinstance(c, str) and c.strip()))[:4]
    if not dims and not players and not queries:
        return None
    return {"dimensions": dims, "players": players,
            "supplementary_queries": queries[:4],
            "supplementary_cpc": cpc}


_CPC_TITLES: Optional[dict] = None


def _cpc_title(code: str) -> str:
    """Lazy-load the CPC subgroup titles json once; "" on any failure."""
    global _CPC_TITLES
    if _CPC_TITLES is None:
        _CPC_TITLES = {}
        try:
            with open(os.path.join(os.path.dirname(__file__), "..", "..",
                                   "data/cpc/cpc_titles_subgroups.json"),
                      encoding="utf-8") as fh:
                for entry in json.load(fh) or []:
                    if isinstance(entry, dict) and entry.get("code"):
                        _CPC_TITLES[str(entry["code"])] = str(
                            entry.get("title") or "")
        except (OSError, ValueError):
            pass
    return _CPC_TITLES.get(code, "")


def merge_grounded(stats: dict, llm_out: Optional[dict],
                   pre_interp: Optional[dict] = None) -> dict:
    """Merge the synthesis result with the pre-interpretation.

    The pre-interpretation's scheme/structure terms ride along so the
    rubric keeps its architecture vocabulary.  *llm_out* None (flash
    failure) falls back to a stats-only version: applicant frequency
    players plus CPC-title group lines.  Never raises.
    """
    base: dict = {"dimensions": [], "players": [],
                  "supplementary_queries": [], "supplementary_cpc": [],
                  "cpc_hint_lines": []}
    for k in ("scheme", "structure_terms", "independence_terms"):
        v = (pre_interp or {}).get(k)
        if v:
            base[k] = v
    if llm_out:
        out = dict(base)
        out["dimensions"] = llm_out.get("dimensions") or []
        out["players"] = llm_out.get("players") or []
        out["supplementary_queries"] = llm_out.get("supplementary_queries") or []
        out["supplementary_cpc"] = llm_out.get("supplementary_cpc") or []
        return out
    out = dict(base)
    out["players"] = [e["name"] for e in (stats.get("applicants") or [])[:5]]
    for entry in (stats.get("cpc") or [])[:3]:
        title = _cpc_title(entry["name"])
        if title:
            out["cpc_hint_lines"].append(f"{entry['name']} {title}")
    return out


async def synthesize_grounded(question: str, candidates: list,
                              pre_interp: Optional[dict] = None,
                              cpc_hints: Optional[list] = None) -> Optional[dict]:
    """Grounded synthesis via the Flash LLM.  Never raises.

    Stats are computed first; the flash call either produces the full
    grounded interpretation or degrades to the stats-only version.
    Returns None only when disabled or the question is empty — callers
    then keep their pre-interpretation flow untouched.
    """
    if not GROUNDED_ENABLED:
        return None
    question = str(question or "").strip()
    if not question:
        return None
    stats = candidate_stats(candidates or [])
    try:
        provider = _grounded_provider()
        payload = build_synthesis_input(
            question, pre_interp, candidates or [], stats, cpc_hints)
        result = await asyncio.wait_for(
            provider.complete_json(
                GROUNDED_SYSTEM_PROMPT,
                json.dumps(payload, ensure_ascii=False),
                max_retries=1),
            timeout=GROUNDED_TIMEOUT)
        return merge_grounded(stats, parse_grounded(result), pre_interp)
    except Exception:
        return merge_grounded(stats, None, pre_interp)
