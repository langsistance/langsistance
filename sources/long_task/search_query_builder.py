"""LLM-assisted USPTO search query construction for long-task PHASE0.

The Flash LLM translates the user's natural-language (usually Chinese)
question into patent-domain English keyword groups and assembles USPTO
free-form query strings. Pure assembly helpers are separated from the
LLM call so they can be unit-tested without a provider.
"""

import json
import re
from typing import Any

DEFAULT_QUERY_MAX_LENGTH = 250

_CJK_RE = re.compile(r'[　-〿぀-ヿ㐀-䶿一-鿿＀-￯가-힯]')

REWRITE_SYSTEM_PROMPT = (
    "你是一个专利检索式构造专家。把用户的自然语言技术问题改写为 "
    "USPTO Patent Application Search API 的检索式（q 参数）。"
    "本工具面向所有技术领域，不得为特定领域预设关键词。\n\n"
    "步骤：\n"
    "1. 从用户问题中抽取 2-4 个核心技术概念（忽略语气词和通用词）\n"
    "2. 每个概念翻译成该领域专利文献常用的英文术语，并给出至少 5 个"
    "同义/近义关键词（含行业缩写、上位/下位词、英美拼写变体）。"
    "变体必须覆盖同一概念的不同词形/复合结构（如 cool ↔ cooler ↔ "
    "cooling、heat generation ↔ heating——名词短语与动名词复合词都要"
    "给出，不能只给单一词形的多个拼写），并判断该概念的明确程度：\n"
    "   - 明确专有名词/品牌名/具体化合物名 → 精确词即可\n"
    "   - 一般性技术概念 → 关键词集中必须包含至少一个词尾通配符变体\n"
    "   - 判断不清 → 精确词与词尾通配符变体都放入关键词集\n"
    "3. 可以为提高精度添加用户未提及的合理领域限定概念（如应用场景、"
    "设备载体），但每个限定必须有依据，且限定属于最弱概念层级\n"
    "4. 用检索式语法组装查询串：\n"
    "   - 多词短语必须用双引号包裹，如 \"3d printing\"\n"
    "   - 同一概念的同义词用 OR 连接并放在圆括号内："
    '("3d printing" OR "additive manufacturing" OR "rapid prototyping")\n'
    "   - 不同概念之间用 AND 连接\n"
    "   - 支持词尾通配符：filter* 匹配 filter/filtering/filtration，"
    "cataly* 匹配 catalyst/catalysis/catalytic；通配符只在词尾生效，"
    "引号短语内不生效，词首通配符无效\n"
    "   - 为每个概念补充常见词形变体，或直接用通配符覆盖变体\n"
    "   - 每个检索式最多 12 个关键词、250 字符，禁止出现中文\n"
    "5. 输出 2-4 个检索式，必须按松紧排序：\n"
    "   - 第一个为最紧的完整组合式（全部概念 + 领域限定）\n"
    "   - 后续逐级放宽（每级去掉最弱的限定/概念）\n"
    "   - 最后一个为最松的核心概念式：只含一个核心概念的单组检索式\n\n"
    'Return JSON: {"concepts": [{"concept": "中文概念", '
    '"keywords": ["english term", ...]}], '
    '"queries": ["最紧", "较松", "最松"]}'
)


def assemble_query(groups: list[list[str]]) -> str:
    """Join keyword groups into a USPTO free-form query string.

    Each inner list is one concept: its keywords are OR-joined inside
    parentheses; concepts are AND-joined. Multi-word keywords are
    double-quoted automatically.
    """
    parts = []
    for group in groups:
        group = [str(k).strip() for k in group if str(k).strip()]
        if not group:
            continue
        joined = " OR ".join(
            f'"{k}"' if (" " in k and not (k.startswith('"') and k.endswith('"')))
            else k
            for k in group
        )
        parts.append(f"({joined})")
    return " AND ".join(parts)


def sanitize_uspto_query(q: str) -> str:
    """Strip CJK characters, collapse whitespace and cap query length."""
    if not q:
        return ""
    q = _CJK_RE.sub(" ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q[:DEFAULT_QUERY_MAX_LENGTH]


def _validated_rewrite(raw: Any) -> dict:
    """Validate and sanitize LLM output into the canonical rewrite dict."""
    if not isinstance(raw, dict):
        return {"concepts": [], "queries": []}
    queries = raw.get("queries") or []
    if not isinstance(queries, list):
        queries = []
    cleaned = []
    for q in queries:
        q = sanitize_uspto_query(str(q)) if isinstance(q, str) else ""
        if q:
            cleaned.append(q)
    return {"concepts": raw.get("concepts") or [], "queries": cleaned}


async def build_search_queries(query: str, provider: Any) -> dict:
    """Rewrite a user question into USPTO search queries via the Flash LLM.

    Never raises: on any failure returns ``{"concepts": [], "queries": []}``,
    which signals callers to keep their existing query untouched.
    """
    try:
        result = await provider.complete_json(REWRITE_SYSTEM_PROMPT, query)
    except Exception:
        return {"concepts": [], "queries": []}
    return _validated_rewrite(result)


def format_ladder_guidance(rewrite: dict, lang: str = "zh") -> str:
    """Render the rewrite ladder for the loop system prompt.

    Queries are listed tightest-first so the LLM can pick a variant or
    adjust from it.  Returns "" when there is nothing to show.
    """
    queries = (rewrite or {}).get("queries") or []
    if not isinstance(queries, list) or not queries:
        return ""
    if lang == "en":
        header = (
            "Available search queries for the user's question, ordered "
            "tightest to loosest. You may call a search tool with one of "
            "these queries directly, or adjust them based on the result "
            "counts you observe. When hits are fewer than 10, first keep "
            "the current level and retry with synonym / word-form "
            "substitutions of the concept terms, and only loosen by "
            "dropping a constraint if that still fails; aim for hits in "
            "the 10-300 range and tighten by adding constraints when "
            "hits are too many:\n"
        )
    else:
        header = (
            "针对用户问题可用的检索式（由紧到松排列）。你可以直接用其中"
            "任一条调用搜索工具，也可以根据观察到的命中数自行调整"
            "（命中少于 10 条时，先保持当前层级、把概念词换成同义表述"
            "或词形变体重试，仍不足再去掉某组限定放宽；目标命中区间 "
            "10-300，命中过多则添加限定收紧）：\n"
        )
    lines = [header]
    for i, q in enumerate(queries, start=1):
        lines.append(f"{i}. {q}")
    if lang == "en":
        lines.append(
            "\nAlso: if a query returns 0 hits even though the technology "
            "clearly exists (a false zero), add word-ending wildcard "
            "variants to the concept terms and retry."
        )
    else:
        lines.append(
            "\n另外：当某级检索式在相关技术确实存在时仍返回 0 命中"
            "（假性零命中），可给概念词补充词尾通配符变体后重试。"
        )
    return "\n".join(lines)


# ── Title feedback ───────────────────────────────────────────────────────────

FEEDBACK_SYSTEM_PROMPT = (
    "你是专利检索式构造专家。根据用户问题与已命中的专利标题，提炼"
    "该领域专利文献中实际使用的措辞，生成新的紧凑检索式。\n"
    "规则：\n"
    "1. 从标题中提取与用户问题相关的核心措辞与同义表达，这些标题"
    "代表了该领域真实语料中的命名习惯\n"
    "2. 每条检索式由 2-3 个概念组组成，多词短语加双引号，同组同义词"
    "用 OR 连接，概念组之间用 AND 连接\n"
    "3. 覆盖同一概念的不同词形（如 cool ↔ cooler ↔ cooling——名词、"
    "动词、动名词形态互相补充），或用词尾通配符覆盖变体\n"
    "4. 输出 2-4 条检索式，按松紧排序（最紧的在前）；每条最多 12 个"
    "关键词、250 字符，禁止出现中文\n"
    'Return JSON: {"queries": ["最紧", "较松", ...]}'
)


async def build_feedback_queries(question: str, titles: list,
                                 provider: Any) -> list:
    """Refine search queries from already-hit patent titles.

    Textual pseudo-relevance feedback: the hit titles are the domain's
    own vocabulary, so the Flash LLM rewrites the question's concepts
    into the phrasings patents actually use.  Never raises; returns a
    sanitized query list (possibly empty).
    """
    if not titles or provider is None:
        return []
    user_content = json.dumps(
        {"question": question, "hit_titles": [str(t) for t in titles][:10]},
        ensure_ascii=False)
    try:
        result = await provider.complete_json(FEEDBACK_SYSTEM_PROMPT,
                                              user_content)
    except Exception:
        return []
    return _validated_rewrite(result)["queries"]
