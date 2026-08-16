"""Architecture-level technical interpretation of the user question.

The Flash rewrite (``search_query_builder``) produces word-surface
ladders: synonym and carrier variants of the question's literal terms.
Patents often describe the same technology with entirely different
vocabulary, though — the scheme-level wording.  Observed in production:
a Chinese question about controlling an amplifier with independent RGB
channel outputs maps in patent literature to per-channel
constant-current loops (error amplifier + current-sense feedback +
per-channel reference), while the family that implements it titles its
patents "human centric black body dimming" — no shared surface terms
at all.

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

from sources.long_task.search_query_builder import sanitize_uspto_query

INTERPRET_ENABLED = os.getenv("REACT_INTERPRET_ENABLED", "1") == "1"
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
    "3. independence_terms：「独立控制/分别控制」在专利文献中的常见英文"
    "表述（如 per-channel、individual、separate loop、independently、"
    "dedicated）\n"
    "4. scenarios：该需求可能出现的专利应用场景（3-5 个，英文）\n"
    "5. queries：用于 US 授权专利全文检索的布尔检索式 3-5 条，方案词"
    "优先、可直接执行；多词短语加双引号、同组同义词用 OR、组间用 "
    "AND；每条最多 12 个关键词、250 字符；禁止出现中文\n"
    "6. 若提供 cpc_hints（该技术领域命中的专利分类号及分类标题），吸收"
    "其中与该需求相关的分类措辞——分类标题代表专利文献对这类技术的"
    "官方命名\n"
    "7. main_lines：该领域的主要技术路线划分（2-3 条，每条一句话，包含"
    "典型电路结构/实现方式——如某一领域可划分为模拟恒流环路与数字 PWM "
    "驱动两条路线，这类划分帮助检索覆盖不同实现流派；只写该领域真实"
    "存在的路线，禁止编造）\n"
    "8. key_players：该领域的主要申请人（3-5 个，英文公司名，必须是该"
    "领域真实活跃的专利申请人；不确定时宁可少给，禁止编造）\n"
    'Return JSON: {"scheme": "...", "structure_terms": [...], '
    '"independence_terms": [...], "scenarios": [...], "queries": [...], '
    '"main_lines": [...], "key_players": [...]}'
)

# Provider construction is expensive and must stay lazy (llm_provider
# imports optional backends); cache one provider per model name.
_PROVIDER_CACHE: dict = {}


def _interpret_provider(model: str):
    if model not in _PROVIDER_CACHE:
        from sources.llm_provider import Provider
        _PROVIDER_CACHE[model] = Provider(
            provider_name="openrouter", model=model,
            server_address="", is_local=False)
    return _PROVIDER_CACHE[model]


def parse_interpretation(raw: Any) -> Optional[dict]:
    """Validate and sanitize the LLM interpretation into a canonical dict.

    Returns None when the output has neither a scheme nor structure
    terms — callers then skip the interpretation entirely.  Queries are
    sanitized the same way as rewrite queries (CJK stripped, length
    capped) so they are safe to hand to the USPTO search API.
    """
    if not isinstance(raw, dict):
        return None

    def _str_list(key: str) -> list:
        items = raw.get(key) or []
        if not isinstance(items, list):
            return []
        return [str(t).strip() for t in items
                if isinstance(t, str) and str(t).strip()]

    scheme = str(raw.get("scheme") or "").strip()
    structure_terms = _str_list("structure_terms")
    if not scheme and not structure_terms:
        return None
    queries: list = []
    seen: set = set()
    for q in _str_list("queries"):
        q = sanitize_uspto_query(q)
        if q and q not in seen:
            seen.add(q)
            queries.append(q)
    return {
        "scheme": scheme,
        "structure_terms": structure_terms[:15],
        "independence_terms": _str_list("independence_terms")[:10],
        "scenarios": _str_list("scenarios")[:6],
        "main_lines": _str_list("main_lines")[:3],
        "key_players": _str_list("key_players")[:5],
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


def merge_interpretation_queries(rewrite: dict, interp: Optional[dict],
                                 cap: int = MAX_LADDER_QUERIES) -> dict:
    """Prepend the interpretation's queries to the rewrite ladder.

    The interpretation queries are architecture-level, so they go at the
    TOP (tightest) of the ladder — the auto-ladder and the blank-q
    injection both pick from the head.  Each query is expanded into its
    AND-drop chain (see ``expand_query_ladder``) so the ladder falls
    through to single-concept OR groups instead of burning its budget on
    404-form combinations.  Returns a new rewrite dict; the input is
    never mutated.  Dedupes against existing ladder entries.
    """
    out = dict(rewrite or {})
    existing = [q for q in (rewrite or {}).get("queries") or [] if q]
    chain: list = []
    seen: set = set()
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
    parts = []
    if interp.get("scheme"):
        parts.append(f"技术方案解读：{interp['scheme']}")
    terms = interp.get("structure_terms") or []
    if terms:
        parts.append(f"关键结构词：{' / '.join(terms[:10])}")
    indep = interp.get("independence_terms") or []
    if indep:
        parts.append(f"独立控制表述：{' / '.join(indep[:6])}")
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
        provider = _interpret_provider(model or INTERPRET_MODEL)
        result = await asyncio.wait_for(
            provider.complete_json(INTERPRET_SYSTEM_PROMPT, user_content,
                                   max_retries=1),
            timeout=INTERPRET_TIMEOUT)
    except Exception:
        return None
    return parse_interpretation(result)
