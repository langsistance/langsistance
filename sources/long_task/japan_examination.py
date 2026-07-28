"""Japan patent examination history analysis pipeline.

Resolves a patent ID (US/CN/EP/JP) to a Japanese application number via the
EPO family API, fetches examination progress data from the JPO IP Data
Platform, and formats it for inclusion in cross-jurisdiction reports.

Unlike China (SIPOP), the JPO API provides a chronological *progress list*
of all examination events rather than structured review decisions.  The
analysis approach is therefore simplified: parse the progress timeline,
translate event names, and present as a structured timeline.

Flow::

    patent_id → [EPO family → JP app_number] → JPO API
    → parse progress events → build timeline → append to report
"""

from __future__ import annotations

import re
from typing import Any

from sources.logger import Logger

_logger = Logger("japan_examination.log")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _truncate(text: str, max_len: int = 200) -> str:
    return text if len(text) <= max_len else text[:max_len] + "..."


def _parse_event_date(raw: str) -> str:
    """Normalize a JPO event date string to YYYY-MM-DD.

    Handles formats: "2020-08-25", "20200825", "2020/08/25".
    """
    cleaned = raw.strip().replace("/", "-").replace(".", "-")
    if len(cleaned) == 8 and cleaned.isdigit():
        cleaned = f"{cleaned[:4]}-{cleaned[4:6]}-{cleaned[6:8]}"
    return cleaned


# ═══════════════════════════════════════════════════════════════════════════════
# JP application number resolution
# ═══════════════════════════════════════════════════════════════════════════════


async def resolve_jp_application_number(
    patent_id: str,
    epo_client: Any | None,
) -> tuple[str | None, dict[str, Any]]:
    """Resolve a patent ID to a Japanese application number via EPO family.

    Args:
        patent_id: Any patent ID (US/CN/EP/JP/WO, etc.).
        epo_client: An ``EPOFamilyClient`` instance (or None).

    Returns:
        (jp_app_number, family_context) tuple.
        jp_app_number is None if no JP member found.
        family_context includes the full PatentFamily info.
    """
    if epo_client is None:
        _logger.warning("japan_resolve — no EPO client, cannot resolve JP member")
        return None, {}

    # Direct JP number — try to normalize
    if re.match(r'^(JP|jp)?\s*\d{4}[\-\.\s]?\d{4,7}', patent_id):
        raw_num = re.sub(
            r'^JP\s*', '', patent_id, flags=re.IGNORECASE,
        ).replace('-', '').replace('.', '').replace(' ', '')
        _logger.info(
            f"japan_resolve_direct — input={patent_id}, normalized={raw_num}"
        )
        return raw_num, {"direct_jp": True, "input_id": patent_id}

    # Resolve via EPO family
    from sources.long_task.patent_family import EPOError

    try:
        family = await epo_client.lookup_family(patent_id)
    except EPOError as e:
        _logger.warning(f"japan_resolve_epo_error — {e}")
        return None, {"error": str(e)}

    jp_member = family.get_representative("JP")
    if not jp_member:
        _logger.info(
            f"japan_resolve_no_jp — input={patent_id}, "
            f"jurisdictions={family.jurisdictions}"
        )
        return None, {
            "jurisdictions": family.jurisdictions,
            "family_id": family.family_id,
        }

    jp_app = jp_member.app_number or ""
    jp_app = re.sub(
        r'^JP\s*', '', jp_app, flags=re.IGNORECASE,
    ).replace('-', '').replace('.', '').replace(' ', '')

    members = []
    for m in family.deduplicated_members:
        members.append({
            "country": m.country,
            "pub_number": m.pub_number,
            "app_number": m.app_number,
            "title": m.title,
            "is_granted": m.is_granted,
        })

    context = {
        "direct_jp": False,
        "input_id": patent_id,
        "jp_app_number": jp_app,
        "jp_pub_number": jp_member.pub_number,
        "family_id": family.family_id,
        "jurisdictions": family.jurisdictions,
        "members": members,
    }

    _logger.info(
        f"japan_resolve_ok — input={patent_id}, jp_app={jp_app}, "
        f"jp_pub={jp_member.pub_number}, "
        f"jurisdictions={family.jurisdictions}"
    )
    return jp_app, context


# ═══════════════════════════════════════════════════════════════════════════════
# Data fetching
# ═══════════════════════════════════════════════════════════════════════════════


async def fetch_examination_data(
    jp_app_number: str,
    jpo_client: Any,
) -> dict[str, Any]:
    """Fetch all available JP examination data for a single application.

    Args:
        jp_app_number: 10-digit JP application number.
        jpo_client: A ``JpoClient`` instance.

    Returns:
        Dict with:
          - progress: full progress list (list of dicts)
          - registration: registration info dict (or None)
          - citations: cited documents list (or None)
          - progress_count: number of progress events
          - has_registration: bool
          - has_citations: bool
    """
    from sources.jpo_client import JpoAPIError, parse_jp_progress_events

    result: dict[str, Any] = {
        "jp_app_number": jp_app_number,
        "progress": [],
        "registration": None,
        "citations": None,
        "progress_count": 0,
        "has_registration": False,
        "has_citations": False,
    }

    # ── Progress (examination timeline) ──
    try:
        progress_raw = await jpo_client.get_patent_progress(jp_app_number)
        events = parse_jp_progress_events(progress_raw)
        result["progress"] = events
        result["progress_count"] = len(events)
        _logger.info(
            f"japan_fetch_progress — app={jp_app_number}, events={len(events)}"
        )
    except JpoAPIError as e:
        _logger.warning(f"japan_fetch_progress_failed — app={jp_app_number}: {e}")

    # ── Registration info ──
    try:
        reg = await jpo_client.get_registration_info(jp_app_number)
        result["registration"] = reg
        result["has_registration"] = bool(reg.get("registrationNumber"))
        _logger.info(
            f"japan_fetch_registration — app={jp_app_number}, "
            f"reg_num={reg.get('registrationNumber', 'N/A')}"
        )
    except JpoAPIError as e:
        _logger.warning(f"japan_fetch_registration_failed — app={jp_app_number}: {e}")

    # ── Citations ──
    try:
        citations = await jpo_client.get_citations(jp_app_number)
        cite_list = citations.get("citeList", citations.get("citationList", []))
        result["citations"] = cite_list if isinstance(cite_list, list) else []
        result["has_citations"] = bool(result["citations"])
        _logger.info(
            f"japan_fetch_citations — app={jp_app_number}, "
            f"cites={len(result['citations'])}"
        )
    except JpoAPIError as e:
        _logger.warning(f"japan_fetch_citations_failed — app={jp_app_number}: {e}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Report formatting
# ═══════════════════════════════════════════════════════════════════════════════


def build_examination_timeline(
    jp_data: dict[str, Any],
    lang: str = "zh",
) -> str:
    """Build a Markdown examination timeline from JP progress events.

    Args:
        jp_data: Result from ``fetch_examination_data()``.
        lang: 'zh' or 'en'.

    Returns:
        Markdown string with timeline table and event descriptions.
    """
    from sources.jpo_client import translate_jp_event

    events = jp_data.get("progress", [])
    if not events:
        if lang == "zh":
            return "未获取到日本审查经过数据。\n"
        else:
            return "No Japanese examination progress data available.\n"

    lines: list[str] = []

    # Header
    if lang == "zh":
        lines.append("### 日本审查经过\n")
    else:
        lines.append("### Japan Examination Progress\n")

    # Summary line
    if lang == "zh":
        lines.append(f"共 {len(events)} 个审查事件。\n")
    else:
        lines.append(f"Total {len(events)} examination events.\n")

    # Timeline table
    if lang == "zh":
        lines.append("| 日期 | 审查事件 | 详情 |")
        lines.append("|------|---------|------|")
    else:
        lines.append("| Date | Event | Detail |")
        lines.append("|------|-------|--------|")

    for evt in events:
        date_str = _parse_event_date(evt.get("event_date", ""))
        event_name = evt.get("event", "")
        translated = translate_jp_event(event_name, lang)
        detail = evt.get("event_detail", "") or evt.get("event_remarks", "") or "—"
        detail = _truncate(detail, 180)

        lines.append(f"| {date_str} | {translated} | {detail} |")

    lines.append("")

    # Key milestones summary
    _key_events = [
        e for e in events
        if any(kw in e.get("event", "")
               for kw in ["拒絶理由通知", "拒絶査定", "特許査定", "特許登録",
                           "審判", "意見書", "手続補正書", "登録査定",
                           "出願審査請求", "取下", "放棄", "無効"])
    ]
    if _key_events:
        if lang == "zh":
            lines.append("**关键审查里程碑：**\n")
        else:
            lines.append("**Key Examination Milestones:**\n")
        for evt in _key_events:
            date_str = _parse_event_date(evt.get("event_date", ""))
            event_name = evt.get("event", "")
            translated = translate_jp_event(event_name, lang)
            lines.append(f"- {date_str} — {translated}")
        lines.append("")

    return "\n".join(lines)


def build_registration_summary(
    jp_data: dict[str, Any],
    lang: str = "zh",
) -> str:
    """Build a Markdown summary of JP registration information.

    Args:
        jp_data: Result from ``fetch_examination_data()``.
        lang: 'zh' or 'en'.

    Returns:
        Markdown string.
    """
    reg = jp_data.get("registration")
    if not reg or not isinstance(reg, dict):
        return ""

    lines: list[str] = []
    if lang == "zh":
        lines.append("### 日本注册信息\n")
    else:
        lines.append("### Japan Registration Information\n")

    reg_num = reg.get("registrationNumber", "")
    reg_date = reg.get("registrationDate", "")
    rights_status = reg.get("rightsStatus", "")

    if lang == "zh":
        lines.append("| 项目 | 内容 |")
        lines.append("|------|------|")
        if reg_num:
            lines.append(f"| 注册号 | {reg_num} |")
        if reg_date:
            lines.append(f"| 注册日期 | {_parse_event_date(reg_date)} |")
        if rights_status:
            lines.append(f"| 权利状态 | {rights_status} |")
    else:
        lines.append("| Item | Detail |")
        lines.append("|------|--------|")
        if reg_num:
            lines.append(f"| Registration No. | {reg_num} |")
        if reg_date:
            lines.append(f"| Registration Date | {_parse_event_date(reg_date)} |")
        if rights_status:
            lines.append(f"| Rights Status | {rights_status} |")

    lines.append("")
    return "\n".join(lines)


def build_citations_summary(
    jp_data: dict[str, Any],
    lang: str = "zh",
) -> str:
    """Build a Markdown summary of JP citation information.

    Args:
        jp_data: Result from ``fetch_examination_data()``.
        lang: 'zh' or 'en'.

    Returns:
        Markdown string.
    """
    citations = jp_data.get("citations")
    if not citations or not isinstance(citations, list) or not citations:
        return ""

    lines: list[str] = []
    if lang == "zh":
        lines.append(f"### 引用文献（共 {len(citations)} 件）\n")
    else:
        lines.append(f"### Cited Documents ({len(citations)} total)\n")

    # Only show first 20 to avoid overwhelming the report
    for i, cite in enumerate(citations[:20]):
        if not isinstance(cite, dict):
            continue
        doc_num = cite.get("citationDocNum", cite.get("documentNumber", "—"))
        doc_type = cite.get("citationDocType", cite.get("documentType", ""))
        if lang == "zh":
            lines.append(f"- {doc_num} ({doc_type})" if doc_type else f"- {doc_num}")
        else:
            lines.append(f"- {doc_num} ({doc_type})" if doc_type else f"- {doc_num}")

    if len(citations) > 20:
        remaining = len(citations) - 20
        if lang == "zh":
            lines.append(f"- ... 及其他 {remaining} 件引用文献")
        else:
            lines.append(f"- ... and {remaining} more citations")

    lines.append("")
    return "\n".join(lines)


def build_japan_section(
    jp_data: dict[str, Any],
    lang: str = "zh",
) -> str:
    """Build the complete Japan examination section for the family report.

    Composes timeline + registration + citations into a single Markdown block.

    Args:
        jp_data: Result from ``fetch_examination_data()``.
        lang: 'zh' or 'en'.

    Returns:
        Complete Markdown section for Japan examination data.
    """
    parts: list[str] = []

    jp_app = jp_data.get("jp_app_number", "")
    if lang == "zh":
        parts.append(f"## 日本审查历史 ({jp_app})\n")
    else:
        parts.append(f"## Japan Examination History ({jp_app})\n")

    # Timeline
    timeline = build_examination_timeline(jp_data, lang)
    if timeline:
        parts.append(timeline)

    # Registration
    reg = build_registration_summary(jp_data, lang)
    if reg:
        parts.append(reg)

    # Citations
    cites = build_citations_summary(jp_data, lang)
    if cites:
        parts.append(cites)

    return "\n".join(parts)
