"""Patent identifier recognition — the deterministic front end of
number routing.

Background (2026-09-03, sample #16): a bare-number question such as
``117941643`` was searched once against USPTO applications/search
(404 = zero hits) and closed — no country disambiguation, no Baiten CN
cross-check, no format guidance.  The rest of the pipeline was built for
keyword language; nothing recognized a number as a patent identifier.

This module turns raw user text into candidate identifiers:
``parse_patent_identifiers`` → ordered list of candidates carrying
country / id_type / confidence / reason / lookups.  Pure functions only
(no network, no LLM — deterministic and table-testable).  Callers:

- ``decide_number_source`` — L2 hard-signal routing (CN/US prefix or a
  validated CN application number → single source; anything ambiguous →
  None so the caller keeps the default dual/auto routing).
- ``format_number_guidance`` — a system-prompt block listing the parsed
  candidates plus the "no hits anywhere" reply contract.

Guards: long text (document paste for novelty checks) and multi-token
text are deliberately NOT parsed as identifier lookups — the 4500-char
claims paste must keep riding the semantic-search path.
"""

import os
import re

# 总开关: 关闭后 parse_* 全部走空路径, agent 行为与原版一致(一键回退)。
NUMBER_PARSE_ENABLED = os.getenv("REACT_NUMBER_PARSE_ENABLED", "1") == "1"

# ── Candidate keys ───────────────────────────────────────────────────────────

# Each candidate dict:
#   raw        — matched token as found in the text
#   display    — canonical readable form (e.g. "CN117941643A")
#   country    — "CN" | "US" | "EP" | "WO" | "JP" | "" (unknown)
#   id_type    — publication | grant | utility | design | application
#                | reissue | unsupported | ambiguous
#   confidence — "high" | "medium" | "low"
#   reason     — zh explanation used verbatim in the guidance block
#   lookups    — ordered strings to try against data sources

# ── Shape tables ─────────────────────────────────────────────────────────────

_CN_KIND_TYPE = {
    "A": ("publication", "发明公开"),
    "B": ("grant", "发明授权公告"),
    "U": ("utility", "实用新型授权公告"),
    "S": ("design", "外观设计授权公告"),
}

# CN 申请号校验位表: 前 12 位加权和 (weights 2..5×2) 对 11 取余后的映射。
_CN_CHECK_MAP = ("1", "0", "X", "9", "8", "7", "6", "5", "4", "3", "2")
_CN_CHECK_WEIGHTS = (2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5)

_US_DESIGN_SERIES = {"29", "30"}
_US_PROVISIONAL_SERIES = {"60", "61"}

# 号码语境词: 纯数字 token 出现在这些词附近才被当作标识符(长文档防误报)。
_CUE_RE = re.compile(
    r"专利|申请号|公开号|公告号|授权号|专利号|号码|编号|"
    r"patent|application|publication|grant|number|no\.?",
    re.IGNORECASE)

_BARE_WINDOW = 8
_SHORT_TEXT_LEN = 120          # ≤此长度: 裸号全部接受(样本 16 含噪音行约 90 字符)
_DOCUMENT_TEXT_LEN = 600       # 超过: 视为文档粘贴, 不做标识符解析
_MAX_CANDIDATES = 3

# 9 位纯数字以 1 开头 → 中国公开号核心号段(2017 年起 100000000+ 滚动,
# 2023 年前后 ~117xxxxxx)。公开号形如 CN117941643A。
_CN_PUB_BARE_9_RE = re.compile(r"^1[0-9]{8}$")


def _clean_digits(raw: str) -> str:
    """Keep only digits (commas/spaces/slashes/dots are separators)."""
    return re.sub(r"[^0-9]", "", raw or "")


def _strip_edges(text: str) -> str:
    """Trim trailing commas/punctuation the frontends often append."""
    return re.sub(r"[\s,，。.;；:：]+$", "", (text or "").strip())


def _cn_check_valid(digits12: str, check: str) -> bool:
    """Validate a modern CN application-number check digit.  Pure."""
    if len(digits12) != 12 or not digits12.isdigit():
        return False
    total = sum(int(d) * w
                for d, w in zip(digits12, _CN_CHECK_WEIGHTS))
    expected = _CN_CHECK_MAP[total % 11]
    return str(check).upper() == expected


def _cn_pub(core: str, kind: str | None) -> dict | None:
    """CN 公开/公告号 (CN + 7-9 位核心 + kind A/B/U/S, kind 可缺)。"""
    if not (core and core.isdigit() and 7 <= len(core) <= 9):
        return None
    kind = (kind or "").upper()
    id_type, kind_label = _CN_KIND_TYPE.get(
        kind, ("publication", "公开/公告号"))
    display = f"CN{core}{kind}"
    if kind:
        lookups = [display, f"CN{core}", core]
    else:
        display = f"CN{core}"
        lookups = [f"CN{core}A", display, core]
    return {
        "raw": core,
        "display": display,
        "country": "CN",
        "id_type": id_type,
        "confidence": "high",
        "reason": f"中国{kind_label}格式（CN 前缀"
                  + (f"，kind {kind}" if kind else "，缺文献类型代码") + "）",
        "lookups": lookups,
    }


def _cn_application(prefix: bool, digits12: str,
                    check: str | None, dotted: bool) -> dict:
    """CN 申请号 (4 位年份+8 位流水, 2003 后; 校验位可验)。"""
    if not (digits12.isdigit() and len(digits12) == 12
            and 1985 <= int(digits12[:4]) <= 2026):
        year_ok = False
    else:
        year_ok = True
    check = (check or "").upper() or None
    valid = _cn_check_valid(digits12, check) if check else None
    tail = f".{check}" if (dotted and check) else (check or "")
    display = ("CN" if prefix else "") + digits12 + tail
    if check:
        if valid and year_ok:
            confidence, reason = "high", "中国专利申请号（校验位通过）"
        else:
            confidence, reason = "low", "疑似中国专利申请号，校验位不匹配，输入可能有误"
    elif year_ok:
        confidence, reason = "medium", "中国专利申请号（缺校验位，未验证）"
    else:
        confidence, reason = "low", "数字形态不完整，难以归属中国专利申请号"
    lookups = [display]
    if not prefix:
        lookups = [f"CN{digits12}{check or ''}", display]
    else:
        lookups = [display, f"CN{digits12}{check or ''}"]
    if not check:
        lookups.insert(0, f"CN{digits12}")
    return {
        "raw": digits12,
        "display": display,
        "country": "CN",
        "id_type": "application",
        "confidence": confidence,
        "reason": reason,
        "lookups": lookups,
    }


def _us_application(series: str, serial: str, design: bool = False) -> dict:
    display = f"{series}/{serial}"
    label = "外观设计申请" if design else "发明/实用新型申请"
    return {
        "raw": display,
        "display": display,
        "country": "US",
        "id_type": "application",
        "confidence": "medium",
        "reason": f"美国{label}格式（系列码 {series}）",
        "lookups": [_clean_digits(display), display],
    }


def _us_granted(digits: str, kind: str | None = None,
                display: str | None = None) -> dict:
    kind = (kind or "").upper() or None
    display = display or f"US{digits}{kind or ''}"
    return {
        "raw": digits,
        "display": display,
        "country": "US",
        "id_type": "grant",
        "confidence": "high" if kind else "medium",
        "reason": ("美国授权专利号格式"
                   + (f"（{kind} 文献类型）" if kind else "（无文献类型代码）")),
        "lookups": [digits, display],
    }


def _us_design(digits: str, s_flag: bool) -> dict:
    display = f"D{digits}S"
    return {
        "raw": display,
        "display": display,
        "country": "US",
        "id_type": "design",
        "confidence": "high",
        "reason": "美国外观设计专利号（D 号）",
        "lookups": [digits, f"US{display}"],
    }


def _external(prefix: str, digits: str, kind: str | None) -> dict:
    """EP/WO/JP: 识别但本期不接入检索 → 诚实降级, 不进自动复核源。"""
    return {
        "raw": f"{prefix}{digits}",
        "display": f"{prefix}{digits}{kind or ''}",
        "country": prefix,
        "id_type": "unsupported",
        "confidence": "low",
        "reason": f"{prefix} 专利号（当前未接入该源，仅作提示）",
        "lookups": [],
    }


# ── Token scan ───────────────────────────────────────────────────────────────
# 精确形态先扫(design/cnapp/usslash), 通用前缀次之, 裸号最后 — 后扫的跳过
# 已被前扫消费的区间, 避免同串产出重复候选。

_DESIGN_RE = re.compile(r"(?i)\b(?:US)?D\s?([0-9]{5,7})\s?(S)?\b")
_CN_APP_RE = re.compile(
    r"(?i)\b(CN)?\s?((?:19|20)[0-9]{2})([0-9]{8})"
    r"(?:\.([0-9Xx])|([0-9Xx]))?\b")
_US_SLASH_RE = re.compile(r"(?i)\b(US)?\s?([0-9]{2})/([0-9,]{3,7})\b")
_PREFIXED_RE = re.compile(
    r"(?i)\b(CN|US|EP|WO|JP|RE|PP)"
    r"\s*([0-9][0-9,\./ ]{4,14}[0-9]|[0-9]{5,13})"
    r"\s*([A-Z]{1,2}[0-9]?)?\b")
_BARE_TOKEN_RE = re.compile(
    r"(?<![\d.])([0-9][0-9,\./ ]{4,14}[0-9]|[0-9]{5,13})(?![\d.])")


def _scan_tokens(text: str) -> list[tuple]:
    """Collect (kind_hint, start, end) spans from *text* in order."""
    spans: list = []
    for m in _DESIGN_RE.finditer(text):
        spans.append((("design", m.group(1), bool(m.group(2))),
                      m.start(), m.end()))
    for m in _CN_APP_RE.finditer(text):
        digits12 = m.group(2) + m.group(3)
        check = m.group(4) or m.group(5)
        spans.append((("cnapp", bool(m.group(1)), digits12,
                       check, m.group(4) is not None),
                      m.start(), m.end()))
    for m in _US_SLASH_RE.finditer(text):
        spans.append((("usslash", bool(m.group(1)), m.group(2),
                       _clean_digits(m.group(3))), m.start(), m.end()))

    consumed = [(s, e) for _, s, e in spans]
    for m in _PREFIXED_RE.finditer(text):
        if any(m.start() < end and s < m.end() for s, end in consumed):
            continue
        spans.append((("prefix", m.group(1).upper(), m.group(2), m.group(3)),
                      m.start(), m.end()))
    consumed = [(s, e) for _, s, e in spans]
    for m in _BARE_TOKEN_RE.finditer(text):
        if any(m.start() < end and s < m.end() for s, end in consumed):
            continue
        token = m.group(1).strip().rstrip(".,，。;；")
        spans.append((("bare", token), m.start(), m.end()))
    spans.sort(key=lambda t: t[1])
    return spans


def _token_to_candidates(kind_hint: tuple) -> list[dict]:
    head = kind_hint[0]
    if head == "design":
        return [_us_design(kind_hint[1], kind_hint[2])]
    if head == "cnapp":
        return [_cn_application(kind_hint[1], kind_hint[2],
                                kind_hint[3], kind_hint[4])]
    if head == "usslash":
        _, prefixed, series, serial = kind_hint
        if series in _US_DESIGN_SERIES:
            return [_us_application(series, serial, design=True)]
        if series in _US_PROVISIONAL_SERIES:
            return [{
                "raw": f"{series}/{serial}", "display": f"{series}/{serial}",
                "country": "US", "id_type": "application",
                "confidence": "medium",
                "reason": "美国临时申请号格式（系列码 60/61）",
                "lookups": [serial, f"{series}/{serial}"],
            }]
        if series.startswith(("1", "2")):
            return [_us_application(series, serial)]
        return []
    if head == "prefix":
        _, prefix, digits_raw, kind = kind_hint
        digits = _clean_digits(digits_raw)
        if prefix == "CN":
            # 12/13 位完整申请号已由 cnapp 消费; 这里只剩公开/公告号段。
            return [c for c in [_cn_pub(digits, kind)] if c]
        if prefix == "US":
            if len(digits) == 11 and digits.startswith("20"):
                return [{
                    "raw": f"US{digits}", "display": f"US{digits}{kind or ''}",
                    "country": "US", "id_type": "publication",
                    "confidence": "high",
                    "reason": "美国专利公开号格式（US+年份+流水）",
                    "lookups": [digits, f"US{digits}"],
                }]
            return [_us_granted(digits, kind)]
        if prefix in ("RE", "PP"):
            return [{
                "raw": f"{prefix}{digits}", "display": f"{prefix}{digits}{kind or ''}",
                "country": "US",
                "id_type": "reissue" if prefix == "RE" else "ambiguous",
                "confidence": "high",
                "reason": "美国再颁专利号" if prefix == "RE" else "美国植物专利号",
                "lookups": [digits, f"{prefix}{digits}"],
            }]
        return [_external(prefix, digits, kind)]
    # bare
    c = _classify_bare(kind_hint[1])
    return [c] if c else []


def _classify_bare(token: str) -> dict | None:
    """Classify a bare digit token without any country prefix."""
    digits = _clean_digits(token)
    n = len(digits)
    if not (6 <= n <= 13):
        return None
    # 千分位逗号分组 (11,794,164) → 美国授权号风格。
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", token):
        if n in (7, 8):
            return {
                "raw": token, "display": f"US{digits}",
                "country": "US", "id_type": "grant",
                "confidence": "medium",
                "reason": "带千分位的 7-8 位数字，符合美国授权专利号书写习惯",
                "lookups": [digits, f"US{digits}"],
            }
    # 12/13 位 20xx 开头 → CN 申请号 (13 位时末位为校验位)。
    if n in (12, 13) and digits.startswith(("19", "20")):
        check = digits[12] if n == 13 else None
        return _cn_application(False, digits[:12], check, dotted=False)
    # 9 位 1 开头 → CN 公开号核心段 (样本 16: 117941643 → CN117941643A)。
    if n == 9 and _CN_PUB_BARE_9_RE.match(digits):
        return {
            "raw": token, "display": f"CN{digits}",
            "country": "CN", "id_type": "publication",
            "confidence": "medium",
            "reason": ("9 位纯数字以 1 开头，符合中国公开号核心号段"
                       "（如 CN117941643A）；若为美国专利号通常为 8 位"),
            "lookups": [f"CN{digits}A", f"CN{digits}", digits],
        }
    # 其余 6-8 位: 美国授权号/申请号歧义, 同一数字按引用检索。
    return {
        "raw": token, "display": f"US{digits}",
        "country": "US", "id_type": "ambiguous",
        "confidence": "medium",
        "reason": ("纯数字无法区分美国授权号与申请号——两个口径会都尝试，"
                   "仍按该数字在任意著录字段匹配"),
        "lookups": [digits, f"US{digits}"],
    }


def _has_cue(text: str, start: int, end: int) -> bool:
    """语境词出现在 token 前后 *_BARE_WINDOW* 个字符内?"""
    window = text[max(0, start - _BARE_WINDOW): end + _BARE_WINDOW]
    return bool(_CUE_RE.search(window))


# ── Public API ───────────────────────────────────────────────────────────────

def parse_patent_identifiers(
    text: str | None, max_candidates: int = _MAX_CANDIDATES,
) -> list[dict]:
    """Return ordered identifier candidates parsed from *text*.

    Ordering: confidence (high > medium > low) then appearance order.
    Empty when the text carries no recognizable patent identifier —
    callers then keep their existing (keyword/semantic) path untouched.
    Never raises.
    """
    if not NUMBER_PARSE_ENABLED:
        return []
    text = str(text or "").strip()
    if not text:
        return []
    head = text[:_DOCUMENT_TEXT_LEN]
    whole = _strip_edges(text)
    short = len(whole) <= _SHORT_TEXT_LEN
    out: list[dict] = []
    seen: set = set()
    for kind_hint, start, end in _scan_tokens(head):
        if kind_hint[0] == "bare" and not short:
            if not _has_cue(head, start, end):
                continue
        for c in _token_to_candidates(kind_hint):
            if not c or c["display"] in seen:
                continue
            seen.add(c["display"])
            out.append(c)
    order = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda c: order.get(c["confidence"], 3))
    return out[:max_candidates]


def decide_number_source(candidates: list[dict]) -> str | None:
    """Hard-signal source routing: 'cn' | 'uspto' | None.

    Only high-confidence country signals force a single source; a bare
    ambiguous number returns None so the caller keeps dual routing and
    the cross-check resolves by trying both sides.  Pure.
    """
    for c in candidates or []:
        if c.get("country") == "CN" and c.get("confidence") == "high":
            return "cn"
        if c.get("country") == "US" and c.get("confidence") == "high":
            return "uspto"
    return None


def format_number_guidance(
    candidates: list[dict], lang: str = "zh",
) -> str:
    """System-prompt block: parsed candidates + the zero-hit reply contract.

    Empty when there are no candidates.  The contract forbids the bare
    "未找到" close-out the sample-#16 log showed: after both sources come
    back empty the assistant must say what was checked and offer next
    steps.
    """
    candidates = [c for c in (candidates or [])
                  if c.get("country") in ("CN", "US")]
    if not candidates:
        return ""
    lines = []
    for c in candidates:
        lines.append(
            f"- {c.get('display') or c.get('raw')} — 国家 {c.get('country') or '未知'}："
            f"{c.get('reason') or ''}（置信 {c.get('confidence') or '低'}）")
    block = "\n".join(lines)
    if lang == "en":
        return (
            "\n\nUser input carries a patent identifier. Deterministic "
            "parse (not a guess):\n" + block +
            "\nWhen a number lookup returns nothing, verify the OTHER data "
            "source too (CN number → Baiten CN, US number → USPTO; "
            "ambiguous numbers → both). If BOTH return nothing, tell the "
            "user which sources were checked and why the number may not "
            "match, then offer next steps (fix the format / keyword search "
            "/ upload a document for a novelty check / ask where the "
            "number came from). Never close with a bare \"not found\"."
        )
    return (
        "\n\n用户输入包含专利标识号。以下是确定性解析结果（非猜测）：\n"
        + block +
        "\n按号检索返回空时，必须先在另一数据源复核（中国号→佰腾、"
        "美国号→USPTO；歧义号码→两个源都查）。两侧都为空时，向用户说明"
        "已核验的数据源与号码可能不匹配的原因，并给出下一步（修正格式 / "
        "关键词检索 / 上传文档查重 / 询问号码来源），禁止仅回复“未找到”。"
    )
