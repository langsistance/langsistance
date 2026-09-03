"""Architecture-level technical interpretation of the user question.

The Flash rewrite (``search_query_builder``) produces word-surface
ladders: synonym and carrier variants of the question's literal terms.
Patents often describe the same technology with entirely different
vocabulary, though — the scheme-level wording.  Observed in production:
a Chinese question about independently controllable channel outputs
maps in patent literature to per-channel constant-current loops
(error amplifier + current-sense feedback + per-channel reference),
while the family that implements it titles its patents "human centric
black body dimming" — no shared surface terms at all.

This module runs the question through a stronger model (default
openai/gpt-5.6-terra via openrouter) with a generic interpretation
schema, producing:

- scheme: the circuit/system architecture the question maps to
- structure_terms: the component/circuit-level words that appear in
  that architecture's patents (not literal translations)
- independence_terms / scenarios: auxiliary vocabulary
- queries: ready-to-run boolean search expressions

Enhancement, not a dependency: any failure returns None and callers
keep their existing flow untouched.  The prompt is generic — the
question is passed at runtime, never baked into the prompt.
"""

import asyncio
import json
import os
from typing import Any, Optional

from sources.long_task.search_query_builder import (
    _CJK_RE, sanitize_uspto_query)

INTERPRET_ENABLED = os.getenv("REACT_INTERPRET_ENABLED", "1") == "1"
INTERPRET_PROVIDER = os.getenv("REACT_INTERPRET_PROVIDER", "openrouter")
INTERPRET_MODEL = os.getenv("REACT_INTERPRET_MODEL", "openai/gpt-5.6-terra")


def _env_int(name: str, default: int) -> int:
    """Parse an int env var, falling back to *default* on garbage.

    A hand-edited .env must not take the whole feature down at import
    time (a ValueError here would silently disable interpretation).
    """
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


INTERPRET_TIMEOUT = _env_int("REACT_INTERPRET_TIMEOUT", 45)
MAX_INTERP_QUERIES = 5
# The dimension skeleton is capped at three layers (core device /
# control circuit / application scenario); never let the model's
# output exceed it — parse enforces the cap, not the model.
MAX_DIMENSIONS = 3
MAX_LADDER_QUERIES = 6
# The interpretation chain and the flash rewrite split the ladder head:
# the chain's tightest forms go first, but the rewrite's tail (its
# single-concept OR groups — the form USPTO actually matches) must stay
# reachable within the budget.
MAX_INTERP_LADDER_SLOTS = 3

INTERPRET_SYSTEM_PROMPT = (
    "你是资深专利检索专家，熟悉 US 授权专利的撰写风格。用户会给出一句"
    "技术需求（可能来自非专利领域的中文表述）。你的任务是做专利检索级的"
    "技术解读：把需求映射到专利文献中实际的电路/系统架构模式，并产出"
    "可直接执行的检索词。只输出 JSON，不要其他文字。\n"
    "规则：\n"
    "1. scheme：该需求在专利文献中通常对应的电路/系统架构模式（1-2 句"
    "方案级描述，不是直译——例如「保持温度稳定的装置」对应温控闭环，"
    "「调节电机转速」对应变频/调压驱动拓扑）\n"
    "2. structure_terms：该方案在专利中可能出现的核心结构/元件英文词，"
    "10-15 个，方案级词汇而非直译词（如 error amplifier、voltage "
    "controlled resistor、reference signal、constant current、"
    "feedback loop 这类元件/电路/环路词汇）；禁止用「概念词+control "
    "circuit」式拼词充当\n"
    "3. independence_terms：与「各通道被分别地控制」含义对应的常见英文"
    "表述（如 per-channel、individual、separate loop、independently、"
    "dedicated）\n"
    "4. scenarios：该需求可能出现的专利应用场景（3-5 个，英文）\n"
    "5. queries：用于 US 授权专利全文检索的布尔检索式 3-5 条，方案词"
    "优先、可直接执行；多词短语加双引号、同组同义词用 OR、组间用 "
    "AND；每条最多 12 个关键词、250 字符；禁止出现中文\n"
    "6. dimensions：把技术需求按纵深拆解为 2-3 个技术维度，默认三层"
    "骨架：核心器件/电路层（实现该功能的硬件核心）、控制算法/电路层"
    "（实现该功能的控制逻辑）、场景应用层（该功能落地的应用与接口）；"
    "某层在目标领域无实质内容时并入最接近的层并简要说明；禁止超过 3 "
    "个维度；每个维度输出 name（维度名）、role（分层角色）、terms"
    "（该维度方案级英文词 3-6 个）、queries（该维度布尔检索式 1-2 条，"
    "规则同第 5 条）\n"
    "7. 若提供 cpc_hints（该技术领域命中的专利分类号及分类标题），吸收"
    "其中与该需求相关的分类措辞——分类标题代表专利文献对这类技术的"
    "官方命名\n"
    "8. main_lines：该领域的主要技术路线划分（2-3 条，每条一句话，包含"
    "典型电路结构/实现方式——如某一领域可划分为模拟恒流环路与数字 PWM "
    "驱动两条路线，这类划分帮助检索覆盖不同实现流派；只写该领域真实"
    "存在的路线，禁止编造）\n"
    "9. key_players：该领域的主要申请人（3-5 个，英文公司名，必须是该"
    "领域真实活跃的专利申请人；不确定时宁可少给，禁止编造）\n"
    'Return JSON: {"scheme": "...", "structure_terms": [...], '
    '"independence_terms": [...], "scenarios": [...], "queries": [...], '
    '"dimensions": [{"name", "role", "terms", "queries"}], '
    '"main_lines": [...], "key_players": [...]}'
)

# Provider construction is expensive and must stay lazy (llm_provider
# imports optional backends); cache one provider per (provider, model) pair.
_PROVIDER_CACHE: dict = {}


def _resolve_interpret_provider_model() -> tuple:
    """Resolve the interpretation (provider, model) pair.

    Priority (model-override layer, 2026-09): [MODEL] interpret_* pair >
    env REACT_INTERPRET_PROVIDER / REACT_INTERPRET_MODEL > the original
    hardcoded openrouter + gpt-5.6-terra defaults.  The provider previously
    had NO override point at all — setting only REACT_INTERPRET_MODEL sent
    the model to openrouter regardless.
    """
    try:
        from sources.long_task.config import get_model_overrides, _override_pair
        provider, model = _override_pair(get_model_overrides(), 'interpret')
        if provider and model:
            return provider, model
    except Exception:
        pass  # override layer is optional
    return INTERPRET_PROVIDER, INTERPRET_MODEL


def _interpret_provider(provider_name: str, model: str):
    key = (provider_name, model)
    if key not in _PROVIDER_CACHE:
        from sources.llm_provider import Provider
        _PROVIDER_CACHE[key] = Provider(
            provider_name=provider_name, model=model,
            server_address="", is_local=False)
    return _PROVIDER_CACHE[key]


def _clean_str_list(raw: dict, key: str) -> list:
    """Non-empty string list for a key; [] on missing/foreign shapes."""
    items = raw.get(key) or []
    if not isinstance(items, list):
        return []
    return [str(t).strip() for t in items
            if isinstance(t, str) and str(t).strip()]


def _parse_dimensions(raw: dict) -> list:
    """Sanitize the model's dimension output.

    Hard rules, enforced here not trusted to the model: at most
    MAX_DIMENSIONS entries, first occurrence of each role label wins,
    empty dimensions (no name and no terms) dropped, per-dimension
    queries run through the same sanitizer as top-level queries.
    """
    dims = raw.get("dimensions")
    if not isinstance(dims, list):
        return []
    out: list = []
    seen_roles: set = set()
    for d in dims:
        if not isinstance(d, dict):
            continue
        name = str(d.get("name") or "").strip()
        role = str(d.get("role") or "").strip()
        terms = _clean_str_list(d, "terms")[:10]
        if not name and not terms:
            continue
        if role:
            if role in seen_roles:
                continue
            seen_roles.add(role)
        queries: list = []
        qseen: set = set()
        for q in _clean_str_list(d, "queries")[:4]:
            # Dimension queries must not carry CJK (prompt rule 禁止出现
            # 中文): a CJK-bearing query is rejected outright instead of
            # leaving a dangling Boolean fragment after sanitization.
            if _CJK_RE.search(str(q)):
                continue
            q = sanitize_uspto_query(q)
            if q and q not in qseen:
                qseen.add(q)
                queries.append(q)
        out.append({"name": name, "role": role, "terms": terms,
                    "queries": queries[:2]})
        if len(out) >= MAX_DIMENSIONS:
            break
    return out


def parse_interpretation(raw: Any) -> Optional[dict]:
    """Validate and sanitize the LLM interpretation into a canonical dict.

    Returns None when the output has neither a scheme nor structure
    terms — callers then skip the interpretation entirely.  Queries are
    sanitized the same way as rewrite queries (CJK stripped, length
    capped) so they are safe to hand to the USPTO search API.
    """
    if not isinstance(raw, dict):
        return None
    scheme = str(raw.get("scheme") or "").strip()
    structure_terms = _clean_str_list(raw, "structure_terms")
    if not scheme and not structure_terms:
        return None
    queries: list = []
    seen: set = set()
    for q in _clean_str_list(raw, "queries"):
        q = sanitize_uspto_query(q)
        if q and q not in seen:
            seen.add(q)
            queries.append(q)
    return {
        "scheme": scheme,
        "structure_terms": structure_terms[:15],
        "independence_terms": _clean_str_list(raw, "independence_terms")[:10],
        "scenarios": _clean_str_list(raw, "scenarios")[:6],
        "main_lines": _clean_str_list(raw, "main_lines")[:3],
        "key_players": _clean_str_list(raw, "key_players")[:5],
        "dimensions": _parse_dimensions(raw),
        "queries": queries[:MAX_INTERP_QUERIES],
    }


def _split_and_groups(q: str) -> list:
    """Split a boolean query on top-level AND into its concept groups.

    Parenthesized OR groups are kept whole: ``("a" OR "b") AND ("c")``
    -> ``['("a" OR "b")', '("c")']``.  Top-level AND is detected outside
    any parentheses AND outside double-quoted phrases (``"a AND b"`` is
    one phrase, never a split point).  A query without top-level AND
    returns [q].
    """
    if not q:
        return []
    groups: list = []
    depth = 0
    in_quote = False
    start = 0
    i = 0
    while i < len(q):
        ch = q[i]
        if ch == '"':
            in_quote = not in_quote
        elif ch == "(" and not in_quote:
            depth += 1
        elif ch == ")" and not in_quote:
            depth = max(0, depth - 1)
        elif (depth == 0 and not in_quote and q.startswith("AND", i)
              and i > 0 and i + 3 < len(q)
              and q[i - 1] == " " and q[i + 3] == " "):
            groups.append(q[start:i].strip())
            start = i + 3
            i += 3
            continue
        i += 1
    tail = q[start:].strip()
    if tail:
        groups.append(tail)
    return [g for g in groups if g]


def expand_query_ladder(q: str) -> list:
    """Expand one boolean query into a tight-to-loose chain.

    USPTO's applications/search returns 404 for most multi-concept AND
    combinations but matches single-concept OR groups reliably (observed
    repeatedly in production).  Each concept group dropped at a time so
    the chain always ends in a single-concept OR group — the ladder
    budget can then fall through to a form that actually returns hits.
    A query without top-level AND is returned unchanged.
    """
    groups = _split_and_groups(q)
    if len(groups) <= 1:
        return [q] if q else []
    chain: list = []
    for i in range(len(groups), 0, -1):
        sub = " AND ".join(groups[:i])
        if sub not in chain:
            chain.append(sub)
    return chain


def expand_rewrite_queries(queries: list, limit: int = 8) -> list:
    """Expand every rewrite query into its AND-drop chain, flattened,
    deduped, tightest-first, capped at *limit*.

    USPTO's applications/search 404s most 3-concept AND combinations but
    matches 2-group and single-concept forms reliably — the fallbacks
    must be present in the ladder or the agent/auto-ladder can never
    reach a form that returns hits (production: zh queries get CN
    auto-ladder but US stays at the first 3-concept 404).
    """
    out: list = []
    seen: set = set()
    for q in queries or []:
        for sub in expand_query_ladder(str(q)):
            if sub and sub not in seen:
                seen.add(sub)
                out.append(sub)
    return out[:limit]


def _dimension_queries(interp: Optional[dict]) -> list:
    """Per-dimension ladder head, round-robin interleaved.

    Each dimension contributes its tight-to-loose AND-drop chain; the
    chains are interleaved (d1[0], d2[0], d3[0], d1[1], ...) so the
    ladder head covers every facet before any single dimension's
    fallbacks, capped at MAX_INTERP_LADDER_SLOTS.  A dimension without
    queries contributes nothing.
    """
    dims = (interp or {}).get("dimensions") or []
    chains: list = []
    for d in dims:
        if not isinstance(d, dict):
            continue
        for q in (d.get("queries") or []):
            q = str(q).strip()
            if q:
                chains.append(expand_query_ladder(q))
                break
    interleaved: list = []
    i = 0
    while any(len(ch) > i for ch in chains) \
            and len(interleaved) < MAX_INTERP_LADDER_SLOTS:
        for ch in chains:
            if i < len(ch):
                interleaved.append(ch[i])
                if len(interleaved) >= MAX_INTERP_LADDER_SLOTS:
                    break
        i += 1
    return interleaved


def merge_interpretation_queries(rewrite: dict, interp: Optional[dict],
                                 cap: int = MAX_LADDER_QUERIES) -> dict:
    """Prepend the interpretation's queries to the rewrite ladder.

    With a dimension skeleton, each dimension contributes one ladder
    head entry (round-robin over its AND-drop chains) so retrieval
    covers every facet; without dimensions the flat interpretation
    queries expand into their full chains as before.  The auto-ladder
    and the blank-q injection both pick from the head.  Returns a new
    rewrite dict; the input is never mutated.  Dedupes against existing
    ladder entries.
    """
    out = dict(rewrite or {})
    existing = [q for q in (rewrite or {}).get("queries") or [] if q]
    dims = (interp or {}).get("dimensions") or []
    chain: list = []
    seen: set = set()
    if dims:
        for sub in _dimension_queries(interp):
            if sub and sub not in seen and sub not in existing:
                seen.add(sub)
                chain.append(sub)
    else:
        for q in (interp or {}).get("queries") or []:
            for sub in expand_query_ladder(str(q)):
                if sub and sub not in seen and sub not in existing:
                    seen.add(sub)
                    chain.append(sub)
    chain = chain[:MAX_INTERP_LADDER_SLOTS]
    # The flash rewrite's tail (its single-concept fallbacks) must stay
    # reachable, so the chain never starves it out of the ladder.
    merged = chain + existing
    out["queries"] = merged[:cap]
    return out


def format_interpretation_rubric(interp: Optional[dict]) -> str:
    """Render the interpretation as a scoring-rubric supplement.

    Empty string when there is nothing usable — the caller keeps the
    plain gate prompt.  The rubric tells the scoring LLM that candidates
    matching the architecture (even without the question's literal
    words) belong to the same technical direction.
    """
    if not interp:
        return ""
    dims = interp.get("dimensions") or []
    is_grounded = bool(interp.get("players")) or any(
            isinstance(d, dict) and (d.get("line") or d.get("representatives"))
            for d in dims)
    if is_grounded:
        parts = []
        scheme = interp.get("scheme")
        if scheme:
            parts.append(f"技术方案解读：{scheme}")
        terms = interp.get("structure_terms") or []
        if terms:
            parts.append(
                f"关键结构词：{' / '.join(str(t) for t in terms[:10])}")
        for d in dims[:MAX_DIMENSIONS]:
            seg = str(d.get("name") or "维度")
            role = d.get("role")
            if role:
                seg += f"（{role}）"
            line = d.get("line")
            if line:
                seg += f"：{line}"
            reps = d.get("representatives") or []
            if reps:
                seg += f"；代表申请人：{'、'.join(str(r) for r in reps[:3])}"
            parts.append("· " + seg)
        for line in (interp.get("cpc_hint_lines") or [])[:3]:
            parts.append("· CPC 主线线索：" + str(line))
        players = interp.get("players") or []
        if players:
            parts.append(
                "真实玩家榜（数据驱动，来自检索结果统计）："
                + " / ".join(str(p) for p in players[:5])
                + "。申请人命中该榜且其他相关性信号吻合时，视为同领域"
                "证据，评分可上调 3-5 分（满分 100）。"
            )
        if not parts:
            return ""
        return (
            "评分补充（来自检索后接地解读）：候选专利的标题/CPC/申请人"
            "若命中以下维度/玩家，即使不含查询字面词，也应视为同一技术"
            "方向，评分相应提高。\n" + "\n".join(parts)
        )
    parts = []
    if interp.get("scheme"):
        parts.append(f"技术方案解读：{interp['scheme']}")
    terms = interp.get("structure_terms") or []
    if terms:
        parts.append(f"关键结构词：{' / '.join(terms[:10])}")
    indep = interp.get("independence_terms") or []
    if indep:
        parts.append(f"分别控制表述：{' / '.join(indep[:6])}")
    main_lines = interp.get("main_lines") or []
    if main_lines:
        parts.append("该领域主要技术路线：" + "；".join(main_lines[:3]))
    players = interp.get("key_players") or []
    if players:
        # Weak signal only: applicant identity alone is never relevance
        # evidence (hallucination guard) — it only supports a candidate
        # whose technical direction already matches.
        parts.append(
            "主要申请人（领域内活跃玩家，仅作背景参考）："
            + " / ".join(players[:5])
            + "。申请人命中该名单本身不构成相关性依据；仅当候选的"
            "技术方向与用户问题一致时，可作为补充证据。"
        )
    if not parts:
        return ""
    return (
        "评分补充（来自对用户问题的专利级技术解构）：候选专利的标题/"
        "CPC 若命中以下结构或表述，即使不含查询字面词，也应视为同一"
        "技术方向，评分相应提高。\n" + "\n".join(parts)
    )


async def interpret_query(query: str, cpc_hints: Optional[list] = None,
                          model: Optional[str] = None) -> Optional[dict]:
    """Interpret the question via the strong model.  Never raises.

    *cpc_hints* (matched CPC code/title pairs from the semantic match)
    seed the classification language of the domain.  Any failure —
    provider error, timeout, unparseable output — returns None so the
    caller's flow is unchanged.
    """
    if not INTERPRET_ENABLED:
        return None
    query = str(query or "").strip()
    if not query:
        return None
    payload: dict = {"question": query}
    if cpc_hints:
        payload["cpc_hints"] = [
            {"code": str(h.get("code", "")), "title": str(h.get("title", ""))}
            for h in cpc_hints[:8] if h.get("code")
        ]
    user_content = json.dumps(payload, ensure_ascii=False)
    try:
        resolved_provider, resolved_model = _resolve_interpret_provider_model()
        provider = _interpret_provider(
            resolved_provider, model or resolved_model)
        result = await asyncio.wait_for(
            provider.complete_json(INTERPRET_SYSTEM_PROMPT, user_content,
                                   max_retries=1),
            timeout=INTERPRET_TIMEOUT)
    except Exception:
        return None
    return parse_interpretation(result)
