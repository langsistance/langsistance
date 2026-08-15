"""LLM-assisted USPTO search query construction for long-task PHASE0.

The Flash LLM translates the user's natural-language (usually Chinese)
question into patent-domain English keyword groups and assembles USPTO
free-form query strings. Pure assembly helpers are separated from the
LLM call so they can be unit-tested without a provider.
"""

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
    "2. 每个概念翻译成该领域专利文献常用的英文术语，并给出 2-5 个"
    "同义/近义关键词（含行业缩写、上位/下位词、英美拼写变体）\n"
    "3. 可以为提高精度添加用户未提及的合理领域限定概念（如应用场景、"
    "设备载体），但每个限定必须有依据，且限定属于最弱概念层级\n"
    "4. 用检索式语法组装查询串：\n"
    "   - 多词短语必须用双引号包裹，如 \"3d printing\"\n"
    "   - 同一概念的同义词用 OR 连接并放在圆括号内："
    '("3d printing" OR "additive manufacturing" OR "rapid prototyping")\n'
    "   - 不同概念之间用 AND 连接\n"
    "   - 支持词尾通配符：air dry* 匹配 air dryer/drying/dried，"
    "dehumidif* 匹配 dehumidifier/dehumidification；通配符只在词尾生效，"
    "引号短语内不生效，词首通配符无效\n"
    "   - 为每个概念补充常见词形变体（dryer/drying、dehumidifier/"
    "dehumidification 等），或直接用通配符覆盖变体\n"
    "   - 每个检索式最多 12 个关键词、250 字符，禁止出现中文\n"
    "5. 输出 2-4 个检索式，必须按松紧排序：\n"
    "   - 第一个为最紧的完整组合式（全部概念 + 领域限定）\n"
    "   - 后续逐级放宽（每级去掉最弱的限定/概念）\n"
    "   - 最后一个为最松的核心概念式\n\n"
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
            "these queries directly, or adjust them (drop a constraint to "
            "loosen, add a constraint to tighten) based on the result "
            "counts you observe:\n"
        )
    else:
        header = (
            "针对用户问题可用的检索式（由紧到松排列）。你可以直接用其中"
            "任一条调用搜索工具，也可以根据观察到的命中数自行调整"
            "（命中为 0 则去掉某组限定放宽，命中过多则添加限定收紧）：\n"
        )
    lines = [header]
    for i, q in enumerate(queries, start=1):
        lines.append(f"{i}. {q}")
    return "\n".join(lines)
