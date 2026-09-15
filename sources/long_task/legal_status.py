# -*- coding: utf-8 -*-
"""需求#18 法律状态归一化 —— 纯分类学 + 分类器，不依赖网络。

**为什么单独成模块**：``candidate_metadata.is_dead_status`` 是被 7 处消费的
布尔过滤谓词（排序/去重/过滤），而本模块是**展示与意图分类器**。把两者
焊在一起会让「过滤语义」与「工具语义」无法独立演进。

**单一真相源**：美国侧的 dead/alive 判定**委托**给
``candidate_metadata.is_dead_status``（本模块只补 zh/en 展示标签），
保证过滤器与工具永不产生分歧。

**保守方向**：沿用既有原则"不可验证的不隐藏"——未知/空状态一律
``STATUS_UNKNOWN``（非 dead）。误判为"存活"只是少隐藏，误判为"死亡"
会让有效专利沉底，两者代价不对称。

词表是**领域分类学**（专利法律状态术语），不含任何单次提问的词汇。
"""
from __future__ import annotations

from typing import Any

STATUS_ALIVE = "alive"
STATUS_DEAD = "dead"
STATUS_PENDING = "pending"
STATUS_UNKNOWN = "unknown"

# ── 法务语气护栏 ──────────────────────────────────────────────────────────
# 对齐 sources/long_task/prosecution_analyzer.py 既有的红线做法：禁用词
# 与免责声明随 observation 一起下发（LLM 综合答案时读的是 observation，
# 而系统提示的红线在另一轮才生效）。
FORBIDDEN_ADVICE_TERMS: tuple = (
    "建议复审", "不必要", "没有必要", "必然", "一定会", "肯定",
    "recommend", "unnecessary", "will definitely", "should request",
)

LEGAL_STATUS_DISCLAIMER: dict = {
    "zh": "以上为公开法律状态记录的客观整理，不构成法律意见；"
          "是否请求复审应由代理机构结合案情判断。",
    "en": "This is a factual summary of public legal-status records "
          "- not legal advice. Any decision on re-examination should be "
          "made by a qualified patent attorney.",
}

# 各国官方法律状态查询入口。数据源未覆盖时的兜底出口——按国别分类，
# 与提问内容无关。
OFFICIAL_STATUS_PORTALS: dict = {
    "CN": "https://pss-system.cponline.cnipa.gov.cn/",
    "US": "https://ppubs.uspto.gov/pubwebapp/",
    "EP": "https://register.epo.org/",
    "WO": "https://patentscope.wipo.int/",
    "JP": "https://www.j-platpat.inpit.go.jp/",
    "KR": "https://www.kipris.or.kr/",
}


def official_portal(country: str) -> str:
    """官方法律状态查询入口 URL；未收录国别返回空串。纯，永不抛。"""
    return OFFICIAL_STATUS_PORTALS.get(str(country or "").upper(), "")


# ── CN 有序规则表 ────────────────────────────────────────────────────────
# 五元组 (marker, code, category, zh, en)。
#
# **顺序即优先级：首匹配胜出**，因此"否定/更具体"的条目必须排在
# "更宽泛"的条目之前。典型陷阱：``驳回理由通知``（审查意见通知书，
# 申请仍在审，非死案）必须先于 ``驳回`` 匹配，否则在审申请会被误判。
#
# marker 一律是**完整中文法律术语**；``终止``/``权`` 这类过短且歧义的
# 片段不作 marker。
_CN_RULES: tuple = (
    # —— 否定/在审语义优先 ——
    ("驳回理由通知", "CN_OA_NOTICE", STATUS_PENDING, "驳回理由通知",
     "Notice of Reasons for Refusal"),
    ("审查意见通知书", "CN_OA_NOTICE", STATUS_PENDING, "审查意见通知书",
     "Office Action"),
    # —— 权利终止类 ——
    ("专利权终止", "CN_TERMINATED", STATUS_DEAD, "专利权终止",
     "Patent Terminated"),
    ("专利权有效期届满", "CN_EXPIRED", STATUS_DEAD, "专利权有效期届满",
     "Patent Term Expired"),
    ("专利权期满", "CN_EXPIRED", STATUS_DEAD, "专利权期满",
     "Patent Term Expired"),
    ("权利终止", "CN_TERMINATED", STATUS_DEAD, "权利终止",
     "Rights Terminated"),
    # —— 无效类（部分无效仍是有权利的有效专利，不可判死）——
    ("全部无效", "CN_INVALIDATED", STATUS_DEAD, "宣告全部无效",
     "Claims Fully Invalidated"),
    ("部分无效", "CN_PART_INVALIDATED", STATUS_ALIVE, "宣告部分无效",
     "Claims Partially Invalidated"),
    # —— 撤回/放弃/驳回类 ——
    ("视为撤回", "CN_DEEMED_WITHDRAWN", STATUS_DEAD, "视为撤回",
     "Deemed Withdrawn"),
    ("撤回", "CN_WITHDRAWN", STATUS_DEAD, "撤回", "Withdrawn"),
    ("视为放弃", "CN_DEEMED_ABANDONED", STATUS_DEAD, "视为放弃",
     "Deemed Abandoned"),
    ("放弃", "CN_ABANDONED", STATUS_DEAD, "放弃", "Abandoned"),
    ("驳回", "CN_REJECTED", STATUS_DEAD, "驳回", "Rejected"),
    # —— 授权/在审类 ——
    ("授权", "CN_GRANTED", STATUS_ALIVE, "授权", "Granted"),
    ("专利权授予", "CN_GRANTED", STATUS_ALIVE, "专利权授予", "Granted"),
    ("实质审查", "CN_UNDER_EXAM", STATUS_PENDING, "实质审查",
     "Substantive Examination"),
    ("公开", "CN_PUBLISHED", STATUS_PENDING, "公开", "Published"),
    ("受理", "CN_ACCEPTED", STATUS_PENDING, "受理", "Accepted"),
    # —— 权利负担/变动（不影响有效性与可实施性）——
    ("中止", "CN_SUSPENDED", STATUS_ALIVE, "中止", "Suspended"),
    ("质押", "CN_PLEDGED", STATUS_ALIVE, "质押", "Pledged"),
    ("许可备案", "CN_LICENSED", STATUS_ALIVE, "许可备案",
     "License Recorded"),
    ("权利转移", "CN_TRANSFERRED", STATUS_ALIVE, "权利转移",
     "Rights Transferred"),
)

# 匹配前的归一化：剥掉「的」与全部空白。佰腾真实取值形如
# 「专利权的终止 专利权有效期届满」，不做此步则「专利权终止」规则
# 永远匹配不上真实数据。
_CN_STRIP_CHARS = ("的", " ", "　", "\t", "\n", "\r")


def _as_text(value: Any) -> str:
    """仅接受字符串；其余类型一律视为空串（不 str() 化，避免把
    数字/容器变成看似合法的状态文本）。"""
    return value if isinstance(value, str) else ""


def _normalize_cn(text: str) -> str:
    """CN 状态串归一化：去「的」、去空白。"""
    out = text
    for ch in _CN_STRIP_CHARS:
        out = out.replace(ch, "")
    return out


def _has_cjk(text: str) -> bool:
    """CJK 统一表意文字区间 U+4E00–U+9FFF。"""
    return any("一" <= ch <= "鿿" for ch in text)


def _is_cn(country: str, text: str) -> bool:
    """国别判定：显式 CN，或文本本身是中文（中文状态串只可能来自 CN
    源，即便调用方未传 country）。"""
    return str(country or "").upper() == "CN" or _has_cjk(text)


def _unknown(text: str, code: str = "UNKNOWN") -> dict:
    return {"code": code, "category": STATUS_UNKNOWN,
            "zh": text, "en": text, "raw": text}


def classify_status(raw_status: Any, country: str = "") -> dict:
    """把一个来源原生的状态串归一化为
    ``{code, category, zh, en, raw}``。纯函数，永不抛。

    ``raw`` 为空/非字符串 → ``code="UNKNOWN"``。已知国语走对应规则表；
    未收录的状态**不判死**（保守方向）。
    """
    text = _as_text(raw_status)
    if not text.strip():
        return _unknown("")

    if _is_cn(country, text):
        normalized = _normalize_cn(text)
        for marker, code, category, zh, en in _CN_RULES:
            if marker in normalized:
                return {"code": code, "category": category,
                        "zh": zh, "en": en, "raw": text}
        return _unknown(text, "CN_UNMAPPED")

    # 美国/其他：dead/alive 的判定权在 candidate_metadata（单一真相源），
    # 本模块只补展示标签。
    from sources.long_task.candidate_metadata import is_dead_status
    if is_dead_status(text):
        return {"code": "US_DEAD", "category": STATUS_DEAD,
                "zh": "已失效", "en": "Not in force", "raw": text}
    return {"code": "US_UNMAPPED", "category": STATUS_ALIVE,
            "zh": "有效/在审", "en": "In force / pending", "raw": text}


def is_dead_status_i18n(status: Any, country: str = "") -> bool:
    """国别感知的"已失效"判定。

    CN：命中任一 DEAD 规则 → True。
    其他：**逐字节委托** ``candidate_metadata.is_dead_status``，保证
    过滤器与工具结论一致。

    未知/空/非字符串一律 False —— 不可验证的不隐藏。
    """
    text = _as_text(status)
    if not text.strip():
        return False
    if _is_cn(country, text):
        return classify_status(text, country="CN")["category"] == STATUS_DEAD
    from sources.long_task.candidate_metadata import is_dead_status
    return bool(is_dead_status(text))


def summarize_timeline(timeline: Any, country: str = "CN") -> dict:
    """把 ``query_legal_state_timeline`` 的输出归约为当前状态摘要。

    输入形状：``[{date, lawStatusCode, lawStatus, lawStatusDetail}]``。
    **索引 0 即最新**（与 ``BaitenClient.query_law_state`` 既有语义一致）。

    返回 ``{latest, latest_date, latest_category, category, zh, en,
    event_count, dead}``。空/畸形输入 → 全 unknown，永不抛。
    """
    entries = []
    for entry in (timeline or []):
        if not isinstance(entry, dict):
            continue
        law = _as_text(entry.get("lawStatus") or entry.get("law_state")).strip()
        if not law:
            continue
        entries.append({
            "date": _as_text(entry.get("date")
                             or entry.get("notice_date")).strip(),
            "lawStatus": law,
        })
    if not entries:
        return {"latest": "", "latest_date": "", "latest_category":
                STATUS_UNKNOWN, "category": STATUS_UNKNOWN,
                "zh": "", "en": "", "event_count": 0, "dead": False}
    latest = entries[0]
    cls = classify_status(latest["lawStatus"], country=country)
    return {
        "latest": latest["lawStatus"],
        "latest_date": latest["date"],
        "latest_category": cls["category"],
        "category": cls["category"],
        "zh": cls["zh"],
        "en": cls["en"],
        "event_count": len(entries),
        "dead": cls["category"] == STATUS_DEAD,
    }


# 复审/无效决定结论词表（决定记录自身的用语，非提问词汇）。顺序即优先级。
_REVIEW_OUTCOMES: tuple = (
    ("全部无效", {"zh": "宣告专利权全部无效", "en": "Claims fully invalidated"}),
    ("部分无效", {"zh": "宣告专利权部分无效", "en": "Claims partially invalidated"}),
    ("维持", {"zh": "维持专利权有效", "en": "Patent maintained in force"}),
    ("撤回", {"zh": "请求撤回", "en": "Request withdrawn"}),
)


def _review_outcome(text: str) -> dict:
    for marker, labels in _REVIEW_OUTCOMES:
        if marker in text:
            return labels
    return {"zh": "结论未载明", "en": "Outcome not stated"}


def summarize_review_decisions(decisions: Any, lang: str = "zh") -> str:
    """复审/无效决定的**事实**陈述。纯函数，永不抛。

    只陈述决定记录本身载明的内容（决定号/公告日/法律依据/结论），
    **不给任何建议**。无决定 → 空串（调用方据此明说"未检索到记录"）。
    """
    rows = []
    for d in (decisions or []):
        if not isinstance(d, dict):
            continue
        num = _as_text(d.get("declareNum") or d.get("declare_num")).strip()
        date = _as_text(d.get("declareDate") or d.get("declare_date")).strip()
        base = _as_text(d.get("lawBase") or d.get("law_base")).strip()
        body = _as_text(d.get("fullText") or d.get("full_text"))
        if not (num or date or base or body):
            continue
        rows.append({"num": num, "date": date, "base": base,
                     "outcome": _review_outcome(f"{base} {body}")})
    if not rows:
        return ""
    if str(lang) == "en":
        head = f"Re-examination / invalidation decisions ({len(rows)}):"
        lines = [head]
        for r in rows:
            bits = [b for b in (r["date"], r["num"]) if b]
            detail = f" ({'; '.join(bits)})" if bits else ""
            basis = f" basis: {r['base']};" if r["base"] else ""
            lines.append(f"-{basis} outcome: {r['outcome']['en']}{detail}")
        return "\n".join(lines)
    head = f"复审/无效决定 {len(rows)} 条："
    lines = [head]
    for r in rows:
        bits = [b for b in (r["date"], r["num"]) if b]
        detail = f"（{'; '.join(bits)}）" if bits else ""
        basis = f"法律依据：{r['base']}；" if r["base"] else ""
        lines.append(f"- {basis}结论：{r['outcome']['zh']}{detail}")
    return "\n".join(lines)
