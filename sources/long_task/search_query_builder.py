"""LLM-assisted USPTO search query construction for long-task PHASE0.

The Flash LLM translates the user's natural-language (usually Chinese)
question into patent-domain English keyword groups and assembles USPTO
free-form query strings. Pure assembly helpers are separated from the
LLM call so they can be unit-tested without a provider.
"""

import json
import re
from typing import Any, Optional

DEFAULT_QUERY_MAX_LENGTH = 250

_CJK_RE = re.compile(r'[　-〿぀-ヿ㐀-䶿一-鿿＀-￯가-힯]')

REWRITE_SYSTEM_PROMPT = (
    "你是一个专利检索概念提取专家。把用户的自然语言技术问题分解为 "
    "核心技术概念及其英文检索关键词；检索式阶梯由代码按确定性规则"
    "组装，你无需输出检索式。本工具面向所有技术领域，不得为特定领域"
    "预设关键词。\n\n"
    "步骤：\n"
    "1. 从用户问题中抽取 2-4 个核心技术概念（忽略语气词和通用词），"
    "概念按重要性排序（最重要的放最前）——代码将按此顺序组装由紧到松"
    "的检索式阶梯。概念必须与提问中的独立技术要素一一对应，禁止把两"
    "个技术要素合并成一个概念（例如把要素甲与要素乙合并成「甲乙控制」"
    "）——合并会丢失概念组合的检索结构，导致检索域漂移\n"
    "2. 每个概念翻译成该领域专利文献常用的英文术语，并给出至少 5 个"
    "同义/近义关键词（含行业缩写、上位/下位词、英美拼写变体），"
    "关键词同样按重要性排序（最重要的放最前，代码只取前若干个进入"
    "检索式）。变体必须覆盖同一概念的不同词形/复合结构（如 cool ↔ "
    "cooler ↔ cooling、heat generation ↔ heating——名词短语与动名词"
    "复合词都要给出，不能只给单一词形的多个拼写），并判断该概念的"
    "明确程度：\n"
    "   - 明确专有名词/品牌名/具体化合物名 → 精确词即可\n"
    "   - 一般性技术概念 → 关键词集中必须包含至少一个词尾通配符变体，"
    "且必须有一个裸词根通配符：取自单个词的词根+*（如 cool*，可"
    "同时覆盖 cool/cooler/cooling 各词形，不依赖拼写对错）；不要"
    "写成多词短语+*（如 \"air cool*\" 只匹配该词序的短语，覆盖不到"
    "复合词的其他词序组合）；裸词根是低精度覆盖面保底项，必须排在"
    "关键词列表靠后位置（精确词在前）\n"
    "   - 判断不清 → 精确词与词尾通配符变体都放入关键词集\n"
    "3. 每个概念除了同义/近义关键词，还必须给出 2-5 个「载体词」：在"
    "专利文献中实现该概念功能的器件/电路/系统/方法的实际写法（不是"
    "同义词，而是“这个功能通常由什么实现、专利里叫什么”）。专利"
    "检索中直译词经常查不到真实技术——例如用户说「保持温度稳定的"
    "装置」，载体词可以是 thermostat、temperature regulator、thermal "
    "controller。禁止用「概念词+controller/control circuit/control "
    "device/control system」这类后缀拼词充当载体词——这只是同义复"
    "述，不是实现载体。检验标准：以该词为主题的技术是否天然实现了"
    "该概念的功能；载体词应指向具体器件、电路拓扑、系统类别或应用"
    "场景。载体词必须与用户问题技术相关、有依据，不得与 keywords 重"
    "复，不确定时宁可少给；载体词单独放在 carriers 字段（不进 "
    "keywords 字段）。代码会同时组装直译词版与载体词版两套检索式"
    "阶梯\n"
    "4. 可以为提高精度添加用户未提及的合理领域限定概念（如应用场景、"
    "设备载体），但每个限定必须有依据，且限定概念放在概念列表最后"
    "（最弱层级）\n"
    "5. 关键词规则：词尾通配符只在关键词中生效——filter* 匹配 filter/"
    "filtering/filtration，cataly* 匹配 catalyst/catalysis/catalytic；"
    "词首通配符无效；词尾通配符只用于单词，多词短语不要附加通配符"
    "（会丢失短语语义）；关键词禁止出现中文；多词短语无需加引号"
    "（代码负责）\n"
    'Return JSON: {"concepts": [{"concept": "中文概念", '
    '"keywords": ["english term", ...], '
    '"carriers": ["english term", ...]}, ...]}'
)

MAX_KEYWORDS_PER_GROUP = 5
MAX_ASSEMBLED_QUERY_CHARS = 220


def assemble_query(groups: list[list[str]]) -> str:
    """Join keyword groups into a USPTO free-form query string.

    Each inner list is one concept: its keywords are OR-joined inside
    parentheses; concepts are AND-joined. Multi-word keywords are
    double-quoted automatically — except wildcard terms, where quoting
    would disable the trailing wildcard.
    """
    parts = []
    for group in groups:
        group = [str(k).strip() for k in group if str(k).strip()]
        if not group:
            continue
        joined = " OR ".join(
            f'"{k}"' if (" " in k and "*" not in k
                         and not (k.startswith('"') and k.endswith('"')))
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


def _assemble_ladder(groups: list[list[str]]) -> list[str]:
    """Assemble the tight-to-loose query ladder from concept keyword
    groups, deterministically.

    Level 1 combines every concept; each next level drops the weakest
    (last) concept; the final level is the core concept alone.  The
    keyword cap shrinks until the tightest query fits the length limit,
    so no variant beyond the cap is silently truncated mid-syntax.
    """
    cleaned: list[list[str]] = []
    for group in groups:
        seen: list[str] = []
        for k in group:
            k = sanitize_uspto_query(str(k))
            if k and k not in seen:
                seen.append(k)
        if seen:
            cleaned.append(seen)
    if not cleaned:
        return []
    queries: list[str] = []
    for i in range(len(cleaned), 0, -1):
        # Each level gets its own keyword budget: looser levels have
        # fewer groups and can afford more keywords per group — the
        # tightest level's cap must not starve the fallback query.
        cap = MAX_KEYWORDS_PER_GROUP
        while cap > 1:
            if len(assemble_query([g[:cap] for g in cleaned[:i]])) \
                    <= MAX_ASSEMBLED_QUERY_CHARS:
                break
            cap -= 1
        q = sanitize_uspto_query(
            assemble_query([g[:cap] for g in cleaned[:i]]))
        if q and (not queries or q != queries[-1]):
            queries.append(q)
    return queries[:4]


async def build_search_queries(query: str, provider: Any) -> dict:
    """Rewrite a user question into USPTO search queries via the Flash LLM.

    The LLM produces concepts + keyword lists only; the query ladder is
    assembled in code so no variant can be dropped by the model.  Each
    concept may also carry ``carriers`` (the devices/circuits/systems
    that implement the concept in patent vocabulary); carrier-based
    ladder levels are interleaved after each literal level so the agent
    can switch vocabulary without loosening the concept set.  Never
    raises: on any failure returns ``{"concepts": [], "queries": []}``,
    which signals callers to keep their existing query untouched.
    """
    try:
        result = await provider.complete_json(REWRITE_SYSTEM_PROMPT, query)
    except Exception:
        return {"concepts": [], "queries": []}
    if not isinstance(result, dict):
        return {"concepts": [], "queries": []}
    groups: list[list[str]] = []
    carrier_groups: list[list[str]] = []
    for c in result.get("concepts") or []:
        if not isinstance(c, dict):
            groups.append([])
            carrier_groups.append([])
            continue
        kws = c.get("keywords")
        cars = c.get("carriers")
        groups.append([str(k) for k in kws] if isinstance(kws, list) else [])
        carrier_groups.append(
            [str(k) for k in cars] if isinstance(cars, list) else [])
    literal_ladder = _assemble_ladder(groups)
    carrier_ladder = _assemble_ladder(carrier_groups)
    # Interleave tightest-first: each literal level is followed by its
    # carrier-word variant at the same concept count, so a false-zero on
    # the literal wording has a ready-made substitute at the same level.
    interleaved: list[str] = []
    for i in range(max(len(literal_ladder), len(carrier_ladder))):
        if i < len(literal_ladder):
            interleaved.append(literal_ladder[i])
        if i < len(carrier_ladder):
            interleaved.append(carrier_ladder[i])
    seen: set[str] = set()
    queries: list[str] = []
    for q in interleaved:
        if q not in seen:
            seen.add(q)
            queries.append(q)
    queries = queries[:6]
    if not queries:
        # Legacy fallback: a provider that still returns hand-written
        # queries keeps working.
        queries = _validated_rewrite(result)["queries"]
    return {"concepts": result.get("concepts") or [], "queries": queries}


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
            "tightest to loosest. Adjacent entries pair a literal-wording "
            "query with its carrier-term variant at the same level — "
            "prefer the carrier variant when the literal wording returns "
            "0 hits, because patent vocabulary rarely matches direct "
            "translations. You may call a search tool with one of these "
            "queries directly, or adjust them based on the result counts "
            "you observe. When hits are fewer than 10, first keep the "
            "current level and retry with carrier terms / synonyms from "
            "the concept keyword bank (substitute whole concept terms, "
            "not just word forms), and only loosen by dropping a "
            "constraint if same-level substitutions still fail; aim for "
            "hits in the 10-300 range and tighten by adding constraints "
            "when hits are too many:\n"
        )
    else:
        header = (
            "针对用户问题可用的检索式（由紧到松排列；相邻条目是同一层级"
            "的直译词版与载体词版——直译词命中 0 条时优先直接取用载体词"
            "版，因为专利文献用词经常与直译不一致）。你可以直接用其中"
            "任一条调用搜索工具，也可以根据观察到的命中数自行调整"
            "（命中少于 10 条时，先保持当前层级、优先用概念词库中的"
            "「载体词」整组替换直译词重试，仍不足再换同义表述/词形变体，"
            "最后才去掉某组限定放宽；目标命中区间 10-300，命中过多则"
            "添加限定收紧）：\n"
        )
    lines = [header]
    for i, q in enumerate(queries, start=1):
        lines.append(f"{i}. {q}")
    concepts = (rewrite or {}).get("concepts")
    if isinstance(concepts, list):
        bank_lines = []
        for c in concepts:
            if not isinstance(c, dict):
                continue
            kws = [str(k) for k in (c.get("keywords") or []) if str(k).strip()]
            if not kws:
                continue
            label = c.get("concept") or ("concept" if lang == "en" else "概念")
            line = f"- {label}: " + " / ".join(kws)
            carriers = [str(k) for k in (c.get("carriers") or [])
                        if str(k).strip()]
            if carriers:
                if lang == "en":
                    line += "  | carrier terms: " + " / ".join(carriers)
                else:
                    line += "  ｜载体词: " + " / ".join(carriers)
            bank_lines.append(line)
        if bank_lines:
            if lang == "en":
                lines.append("\nConcept keyword bank (carrier terms are "
                              "the patent-literature wording of the "
                              "concept — substitute them first on low "
                              "hits):")
            else:
                lines.append("\n概念词库（载体词是该概念在专利文献中的"
                              "实际写法，低命中时优先替换它们）：")
            lines.extend(bank_lines)
    if lang == "en":
        lines.append(
            "\nAlso: if a query returns 0 hits even though the technology "
            "clearly exists (a false zero), try the carrier-term variant "
            "of the same level first, then add word-ending wildcard "
            "variants to the concept terms and retry."
        )
    else:
        lines.append(
            "\n另外：当某级检索式在相关技术确实存在时仍返回 0 命中"
            "（假性零命中），先改试同层级的载体词版，仍无效再给概念词"
            "补充词尾通配符变体后重试。"
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
    "5. 若提供 cpc_hints（该技术领域的专利分类标题），吸收其中与用户"
    "问题相关的分类措辞——分类标题代表专利文献对这类技术的官方命名，"
    "是标题之外的另一路词表来源\n"
    'Return JSON: {"queries": ["最紧", "较松", ...]}'
)


async def build_feedback_queries(question: str, titles: list,
                                 provider: Any,
                                 cpc_hints: Optional[list] = None) -> list:
    """Refine search queries from already-hit patent titles.

    Textual pseudo-relevance feedback: the hit titles are the domain's
    own vocabulary, so the Flash LLM rewrites the question's concepts
    into the phrasings patents actually use.  *cpc_hints* (matched CPC
    code/title pairs, plan B route C) add the classification language
    as a second vocabulary source.  Never raises; returns a sanitized
    query list (possibly empty).
    """
    if not titles or provider is None:
        return []
    payload = {"question": question,
               "hit_titles": [str(t) for t in titles][:10]}
    if cpc_hints:
        payload["cpc_hints"] = [
            {"code": str(h.get("code", "")),
             "title": str(h.get("title", ""))}
            for h in cpc_hints[:8] if h.get("code")
        ]
    user_content = json.dumps(payload, ensure_ascii=False)
    try:
        result = await provider.complete_json(FEEDBACK_SYSTEM_PROMPT,
                                              user_content)
    except Exception:
        return []
    return _validated_rewrite(result)["queries"]


# ── Missing-direction feedback ───────────────────────────────────────────────

MISSING_DIRECTION_SYSTEM_PROMPT = (
    "你是专利检索方向推断专家。给出用户问题与当前已命中的若干专利标题，"
    "推断当前候选池缺失的技术方向，生成补充检索式供下一轮检索使用。\n"
    "规则：\n"
    "1. 先判断已命中标题与用户问题是否相关；若明显是噪声（标题相关性弱），"
    "不要提炼噪声标题的措辞，直接根据用户问题推断该技术主题在专利文献中"
    "可能的实际写法\n"
    "2. 优先输出「缺失方向」：与用户问题相关、但当前标题中未出现的器件/"
    "电路/系统/应用场景写法（上位/下位/相邻表述），而不是复述已有标题的"
    "用词\n"
    "3. 若给出 cpc_hints（该技术主题对应的专利分类号及分类标题），优先从"
    "这些分类标题中提取该领域专利文献的实际用词，组合进检索式\n"
    "4. 每条检索式由 2-3 个概念组组成，多词短语加双引号，同组同义词用 OR "
    "连接，概念组之间用 AND 连接\n"
    "5. 命中率优先：三概念 AND 组合经常零命中，至少一半检索式只用 2 个"
    "概念组；概念组内优先使用单个单词或词根通配符，避免罕见的完整短语"
    "（短语类词命中率极低，确需短语时给出单词替代词）\n"
    "6. 输出 2-4 条检索式，按松紧排序（最紧的在前）；每条最多 12 个关键词、"
    "250 字符，禁止出现中文\n"
    'Return JSON: {"queries": ["最紧", "较松", ...]}'
)


async def build_missing_direction_queries(question: str, titles: list,
                                          provider: Any,
                                          cpc_hints: Optional[list] = None) -> list:
    """Infer technical directions missing from the current candidate pool.

    Unlike title refinement (build_feedback_queries), this asks the Flash
    LLM which related phrasings are NOT yet covered by the pool's titles,
    and to derive its own vocabulary from the question when the pool is
    noise.  *cpc_hints* (matched CPC code/title pairs, plan B route C)
    seed the classification language of the domain.  Never raises;
    returns a sanitized query list (possibly empty).
    """
    if not titles or provider is None:
        return []
    payload = {"question": question,
               "hit_titles": [str(t) for t in titles][:10]}
    if cpc_hints:
        payload["cpc_hints"] = [
            {"code": str(h.get("code", "")),
             "title": str(h.get("title", ""))}
            for h in cpc_hints[:8] if h.get("code")
        ]
    user_content = json.dumps(payload, ensure_ascii=False)
    try:
        result = await provider.complete_json(MISSING_DIRECTION_SYSTEM_PROMPT,
                                              user_content)
    except Exception:
        return []
    return _validated_rewrite(result)["queries"]
