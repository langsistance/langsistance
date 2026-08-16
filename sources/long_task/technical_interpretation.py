"""Architecture-level technical interpretation of the user question.

The Flash rewrite (``search_query_builder``) produces word-surface
ladders: synonym and carrier variants of the question's literal terms.
Patents often describe the same technology with entirely different
vocabulary, though — the scheme-level wording.  Observed in production:
"控制放大器，独立控制 RGB 颜色输出" maps in patent literature to
per-channel constant-current loops (error amplifier + current-sense
feedback + per-channel reference), and the ERP Power family that
implements it titles its patents "human centric black body dimming" —
no shared surface terms at all.

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
    'Return JSON: {"scheme": "...", "structure_terms": [...], '
    '"independence_terms": [...], "scenarios": [...], "queries": [...]}'
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
        "queries": queries[:MAX_INTERP_QUERIES],
    }


def merge_interpretation_queries(rewrite: dict, interp: Optional[dict],
                                 cap: int = MAX_LADDER_QUERIES) -> dict:
    """Prepend the interpretation's queries to the rewrite ladder.

    The interpretation queries are architecture-level, so they go at the
    TOP (tightest) of the ladder — the auto-ladder and the blank-q
    injection both pick from the head.  Returns a new rewrite dict; the
    input is never mutated.  Dedupes against existing ladder entries.
    """
    out = dict(rewrite or {})
    existing = [q for q in (rewrite or {}).get("queries") or [] if q]
    interp_queries = [q for q in (interp or {}).get("queries") or [] if q]
    merged = [q for q in interp_queries if q not in existing] + existing
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
