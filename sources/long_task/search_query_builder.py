"""LLM-assisted USPTO search query construction for long-task PHASE0.

The Flash LLM translates the user's natural-language (usually Chinese)
question into patent-domain English keyword groups and assembles USPTO
free-form query strings. Pure assembly helpers are separated from the
LLM call so they can be unit-tested without a provider.
"""

import json
import os
import re
from typing import Any, Optional

DEFAULT_QUERY_MAX_LENGTH = 250

_CJK_RE = re.compile(r'[　-〿぀-ヿ㐀-䶿一-鿿＀-￯가-힯]')

# 申请人概念渲染语法 — 见 render_applicant_query。field 形态需先经
# scripts/uspto_applicant_field_probe.py 网关冒烟确认可用再启用
# (2026-09-03 观察: 申请人概念被当普通全文词叠 AND, USPTO 对括号 AND
# 组合又大量 404 — 含申请人限定的组合检索整轮打空)。
USPTO_APPLICANT_SYNTAX = os.getenv(
    "USPTO_APPLICANT_SYNTAX", "phrase").strip().lower()
USPTO_APPLICANT_FIELD = os.getenv(
    "USPTO_APPLICANT_FIELD", "firstApplicantName").strip()

_APPLICANT_ROLES = {"applicant", "assignee", "company", "申请人", "company_name"}
_AND_OR_NOT_RE = re.compile(r"\b(?:AND|OR|NOT)\b", re.IGNORECASE)

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
    "）——合并会丢失概念组合的检索结构，导致检索域漂移。若问题中出现"
    "申请人/公司/机构名（人名、公司名、大学或研究机构名），把它抽为独立"
    "概念并在该概念的 role 字段标 applicant（禁止与任何技术要素合并"
    "；role 缺省为 technical）；申请人概念的 keywords 只给该主体的各种"
    "法定/常用名称变体（全称、简称、历史用名、常见译名），不给通用词"
    "（applicant/company/assignee 之类不是检索词）\n"
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
    '"carriers": ["english term", ...], '
    '"role": "technical" 或 "applicant"(仅申请人概念)}, ...]}'
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


# ── Applicant-anchor semantics (2026-09-03 production observation) ───────────
# An applicant-constrained question was rewritten as an ordinary concept
# AND-ed with the technical concepts; every 4/3/2-concept bracket query
# then 404'd while the plain assignee word full-text matched tens of
# thousands of transfer records.  Fixes here:
#   1. the rewrite marks applicant concepts with role="applicant";
#   2. build_search_queries moves them to the FRONT of the ladder so the
#      AND-drop chain can only ever drop technical groups — the loosest
#      level is the applicant anchor alone, never a bare technical word;
#   3. the anchor renders via render_applicant_query (phrase by default;
#      field syntax behind gateway smoke).

def _concept_role(c: Any) -> str:
    """'applicant' when the concept is a company/assignee, else 'technical'."""
    if not isinstance(c, dict):
        return "technical"
    role = str(c.get("role") or "").strip().lower()
    return "applicant" if role in _APPLICANT_ROLES else "technical"


def order_concepts_by_role(concepts: list) -> list:
    """Move applicant concepts to the front (stable within each group).

    The deterministic ladder drops concepts from the END, so an
    applicant placed first survives until every technical group has been
    dropped — a company-name-alone query is the intended loosest level.
    Pure.
    """
    if not concepts:
        return []
    applicant = [c for c in concepts if _concept_role(c) == "applicant"]
    technical = [c for c in concepts if _concept_role(c) != "applicant"]
    return applicant + technical


def render_applicant_query(keywords: list, syntax: str = "",
                           field: str = "") -> str:
    """Render an applicant/company concept as one query group.

    *syntax*: ``phrase`` (default, quoted OR-group), ``field``
    (``firstApplicantName:(...)`` — enable only after the gateway smoke
    probe confirms the field syntax) or ``space`` (plain words, the
    de-structured fallback that survives the endpoint's bracket-AND
    404s).  Returns "" when there are no usable keywords.
    """
    seen: list[str] = []
    for k in keywords or []:
        k = sanitize_uspto_query(str(k)).strip('"')
        if k and k not in seen:
            seen.append(k)
    if not seen:
        return ""
    syntax = (syntax or USPTO_APPLICANT_SYNTAX).lower()
    if syntax == "space":
        return " ".join(seen)
    if syntax == "field":
        inner = " OR ".join(
            f'"{k}"' if " " in k and "*" not in k else k for k in seen)
        return f"{(field or USPTO_APPLICANT_FIELD)}:({inner})"
    joined = " OR ".join(
        f'"{k}"' if " " in k and "*" not in k else k for k in seen)
    return f"({joined})"


def destructure_uspto_query(q: str) -> str:
    """De-bracket a USPTO query into plain space-joined words.

    applications/search 404s most parenthesized AND/OR structures — a
    2026-09-03 production log shows 4-, 3- AND 2-concept bracket queries
    ALL 404 while the same words space-joined return 200.  Quotes and
    operators are stripped so the retry can never carry the failing
    structure.  Returns "" when nothing remains.
    """
    q = _AND_OR_NOT_RE.sub(" ", q or "")
    q = q.replace("(", " ").replace(")", " ")
    q = q.replace('"', " ")
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
    # 申请人(role=applicant)概念与技术概念分离: 技术组保持原语义组装
    # 阶梯; 申请人锚经 render_applicant_query 单独渲染并拼在每条检索式
    # 最前(锚定不可丢) — 阶梯放宽只发生在技术组之间, 最终最松一级是
    # "申请人锚 AND 首个技术组", 再由末尾的 anchor-alone 收底
    # (2026-09-03 观察: 申请人被当普通概念叠 AND 导致整轮打空)。
    concepts = result.get("concepts") or []
    app_kw_groups: list[list[str]] = []
    groups: list[list[str]] = []
    carrier_groups: list[list[str]] = []
    for c in concepts:
        if not isinstance(c, dict):
            groups.append([])
            carrier_groups.append([])
            continue
        kws = c.get("keywords")
        cars = c.get("carriers")
        kw_list = [str(k) for k in kws] if isinstance(kws, list) else []
        car_list = [str(k) for k in cars] if isinstance(cars, list) else []
        if _concept_role(c) == "applicant":
            app_kw_groups.append(kw_list)
        else:
            groups.append(kw_list)
            carrier_groups.append(car_list)
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
    tech_ladder: list[str] = []
    for q in interleaved:
        if q not in seen:
            seen.add(q)
            tech_ladder.append(q)

    anchors = [a for a in
               (render_applicant_query(kws) for kws in app_kw_groups)
               if a]
    queries: list[str] = []
    if anchors:
        anchor = " AND ".join(anchors)
        queries = [f"{anchor} AND {q}" for q in tech_ladder]
        if anchor not in queries:
            queries.append(anchor)  # 最松一级: 仅申请人名下专利
    else:
        queries = list(tech_ladder)
    queries = queries[:6]
    if not queries:
        # Legacy fallback: a provider that still returns hand-written
        # queries keeps working.
        queries = _validated_rewrite(result)["queries"]
    return {"concepts": concepts, "queries": queries}


def format_ladder_guidance(rewrite: dict, lang: str = "zh",
                           cn_rewrite: dict = None) -> str:
    """Render the rewrite ladder(s) for the loop system prompt.

    Queries are listed tightest-first so the LLM can pick a variant or
    adjust from it.  When *cn_rewrite* is given (dual-source / CN mode),
    the Baiten ladder is rendered with its field-prefix semantics
    (ti=标题 ab=摘要 clm=权利要求); for Chinese questions it comes FIRST
    — the user's language sets the focus source (中文提问侧重中国专利).
    Returns "" when there is nothing to show.
    """
    us_text = _render_ladder_guidance(rewrite, lang, cn=False)
    if not cn_rewrite or not (cn_rewrite.get("queries") or []):
        return us_text
    cn_text = _render_ladder_guidance(cn_rewrite, lang, cn=True)
    if not us_text:
        return cn_text
    if lang == "zh":
        return cn_text + "\n\n" + us_text
    return us_text + "\n\n" + cn_text


def _render_ladder_guidance(rewrite: dict, lang: str, cn: bool) -> str:
    """Render one ladder (USPTO or Baiten-CN) into guidance text."""
    queries = (rewrite or {}).get("queries") or []
    if not isinstance(queries, list) or not queries:
        return ""
    if cn:
        if lang == "en":
            header = (
                "Available Baiten (China patent) search queries for the "
                "user's question, ordered tightest to loosest. Field "
                "prefixes: ti=title, ab=abstract, clm=claims. Chinese has "
                "no word forms, so queries carry no wildcards — synonyms "
                "are OR-joined. You may call the China patent search tool "
                "with one of these directly, or adjust based on hit "
                "counts (aim for 10-300 hits; when too few, substitute "
                "synonym phrasings from the concept bank first, and only "
                "loosen by dropping a constraint group last):\n"
            )
        else:
            header = (
                "针对用户问题可用的佰腾（中国专利）检索式（由紧到松排列）。"
                "字段前缀语义：ti=标题 ab=摘要 clm=权利要求。中文无词形变化，"
                "检索式不使用通配符，同义词用 OR 连接。你可以直接用其中"
                "任一条调用中国专利检索工具，也可以根据观察到的命中数"
                "自行调整（命中少于 10 条时，先保持当前层级、优先替换"
                "同义表述，最后才去掉某组限定放宽；目标命中区间 10-300，"
                "命中过多则添加限定收紧）：\n"
            )
    elif lang == "en":
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
    if cn:
        if lang == "en":
            lines.append(
                "\nAlso: if a query returns 0 hits even though the "
                "technology clearly exists (a false zero), try the "
                "carrier-term variant of the same level first, then "
                "substitute synonym phrasings; wildcard variants are not "
                "available for Chinese."
            )
        else:
            lines.append(
                "\n另外：当某级检索式在相关技术确实存在时仍返回 0 命中"
                "（假性零命中），先改试同层级的载体词版，仍无效再替换"
                "同义表述重试（中文检索不支持通配符）。"
            )
    elif lang == "en":
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


# ── Baiten (佰腾) CN search query assembly ──────────────────────────────────
#
# Chinese has no word forms, so the CN ladder carries no trailing
# wildcards — synonym groups are OR-joined instead, and every query is
# field-prefixed (ti=title / ab=abstract / clm=claims).  Pure functions
# only, so they unit-test without a provider (mirrors the USPTO half).

_BAITEN_FIELDS = ("ti", "ab", "clm")

# Control characters and wildcards have no place in a CN query: strip
# them in sanitize so a wayward LLM output can never corrupt the syntax.
_BAITEN_STRIP_RE = re.compile(r"[\x00-\x1f\x7f*]+")


def sanitize_baiten_query(q: str) -> str:
    """Keep CJK + latin + digits; drop control chars / wildcards; cap length.

    Unlike ``sanitize_uspto_query`` this does NOT strip CJK — Chinese is
    the primary query language here.
    """
    if not q:
        return ""
    q = _BAITEN_STRIP_RE.sub(" ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q[:DEFAULT_QUERY_MAX_LENGTH]


def assemble_baiten_query(groups: list[list[str]], field: str = "ti") -> str:
    """Join keyword groups into a Baiten query string.

    Each inner list is one concept: its keywords are OR-joined inside a
    ``field:(...)`` group; concepts are AND-joined.  Multi-word phrases
    (containing a space) are double-quoted.  No wildcards are emitted —
    Chinese needs none and Baiten's syntax treats them differently.
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
        parts.append(f"{field}:({joined})")
    return " AND ".join(parts)


def _baiten_query_with_budget(groups: list[list[str]], field: str) -> str:
    """Assemble one CN query, shrinking the keyword cap until it fits."""
    cap = MAX_KEYWORDS_PER_GROUP
    while cap > 1:
        if len(assemble_baiten_query([g[:cap] for g in groups], field)) \
                <= MAX_ASSEMBLED_QUERY_CHARS:
            break
        cap -= 1
    return assemble_baiten_query([g[:cap] for g in groups], field)


def _assemble_baiten_ladder(groups: list[list[str]]) -> list[str]:
    """Assemble the tight-to-loose CN query ladder, deterministically.

    One field per concept count (ti → ab → clm → ti → …): with 4 concepts
    the old shape spent every slot in the 6-query cap on 4-concept ANDs
    and the looser (3/2/1-concept) levels never appeared — every ladder
    query then returned 0 hits (observed 2026-08-27: 4-group ANDs across
    CN titles/abstracts).  Each concept-count level now lands inside the
    cap, so the LLM and the auto-ladder always see a loosenable variant.
    """
    cleaned: list[list[str]] = []
    for group in groups:
        seen: list[str] = []
        for k in group:
            k = sanitize_baiten_query(str(k))
            if k and k not in seen:
                seen.append(k)
        if seen:
            cleaned.append(seen)
    if not cleaned:
        return []
    queries: list[str] = []
    for n in range(len(cleaned), 0, -1):
        subset = cleaned[:n]
        field = _BAITEN_FIELDS[(len(cleaned) - n) % len(_BAITEN_FIELDS)]
        q = _baiten_query_with_budget(subset, field)
        if q and (not queries or q != queries[-1]):
            queries.append(q)
    return queries[:6]


REWRITE_SYSTEM_PROMPT_CN = (
    "你是一个专利检索概念提取专家。把用户的自然语言技术问题分解为 "
    "核心技术概念及其检索关键词；检索式阶梯由代码按确定性规则组装，"
    "你无需输出检索式。本工具面向所有技术领域，不得为特定领域预设"
    "关键词。\n\n"
    "步骤：\n"
    "1. 从用户问题中抽取 2-3 个核心技术概念（忽略语气词和通用词），"
    "概念按重要性排序（最重要的放最前）——代码将按此顺序组装由紧到松"
    "的检索式阶梯。概念必须与提问中的独立技术要素一一对应，禁止把两"
    "个技术要素合并成一个概念——合并会丢失概念组合的检索结构，导致"
    "检索域漂移。**概念数上限 3 个**：三概念 AND 组合在中文专利标题上"
    "已经经常 0 命中，4 个概念组 AND 几乎必然 0 命中；宁可把关联技术"
    "要素合并进概念关键词（OR 连接），也不要新增第 4 个概念组\n"
    "2. 每个概念给出 3-8 个同义/近义检索关键词，**以中文为主**（含"
    "全称/简称、上位/下位词、俗名/别称），可以允许少量英文术语（中国"
    "专利文献常见中英混用）。关键词按重要性排序（最重要的放最前，"
    "代码只取前若干个进入检索式）。中文无词形变化，不要使用任何通配符。"
    "**关键词必须优先采用专利标题/摘要中实际出现的 2-4 字短词**（功能"
    "名词、部件名、技术名词），并至少包含若干这样的短词；**禁止直接把"
    "用户问题中的完整长表述**（动宾短语、复合长词组、整句片段）作为"
    "关键词——专利标题极少原样出现用户问法的长表述，长表述检索几乎"
    "必然 0 命中，短词命中率远高于长表述。检验标准：该词作为标题主题词"
    "时是否常见\n"
    "3. 每个概念除了同义/近义关键词，还必须给出 2-5 个「载体词」：在"
    "专利文献中实现该概念功能的器件/电路/系统/方法的实际写法（不是"
    "同义词，而是“这个功能通常由什么实现、专利里叫什么”）。禁止用"
    "「概念词+控制电路/控制装置/控制系统」这类后缀拼词充当载体词——"
    "这只是同义复述，不是实现载体。检验标准：以该词为主题的技术是否"
    "天然实现了该概念的功能；载体词应指向具体器件、电路拓扑、系统"
    "类别或应用场景，**同样优先 2-4 字短词**。载体词必须与用户问题"
    "技术相关、有依据，不得与 keywords 重复，不确定时宁可少给；载体词"
    "单独放在 carriers 字段（不进 keywords 字段）。代码会同时组装直译"
    "词版与载体词版两套检索式阶梯\n"
    "4. 可以为提高精度添加用户未提及的合理领域限定概念（如应用场景、"
    "设备载体），但每个限定必须有依据，且限定概念放在概念列表最后"
    "（最弱层级）\n"
    "5. 关键词规则：中文为主，允许少量英文术语；多词短语无需加引号"
    "（代码负责）；禁止通配符；关键词中不要混入检索式语法符号"
    "（括号、AND、OR 等）\n"
    'Return JSON: {"concepts": [{"concept": "中文概念", '
    '"keywords": ["中文检索词", ...], '
    '"carriers": ["中文载体词", ...]}, ...]}'
)


async def build_baiten_queries(query: str, provider: Any) -> dict:
    """Rewrite a user question into Baiten (CN) search queries.

    Same contract as ``build_search_queries``: the LLM produces concepts
    + keyword lists only, the ladder is assembled in code, and the
    carrier-variant ladder interleaves after each literal level.  Never
    raises — on any failure returns ``{"concepts": [], "queries": []}``.
    """
    try:
        result = await provider.complete_json(REWRITE_SYSTEM_PROMPT_CN, query)
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
    literal_ladder = _assemble_baiten_ladder(groups)
    carrier_ladder = _assemble_baiten_ladder(carrier_groups)
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
    return {"concepts": result.get("concepts") or [], "queries": queries[:6]}
