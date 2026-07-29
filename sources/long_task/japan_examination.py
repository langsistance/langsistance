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
        if not events:
            _logger.warning(
                f"japan_fetch_progress_empty — app={jp_app_number}, "
                f"raw_keys={list(progress_raw.keys()) if isinstance(progress_raw, dict) else type(progress_raw).__name__}"
            )
        else:
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
# AI Analysis
# ═══════════════════════════════════════════════════════════════════════════════


async def generate_table_columns(
    query: str,
    event_count: int,
    provider: Any,
    lang: str = "zh",
) -> list[str]:
    """Phase 1: Flash LLM generates table column definitions for JP analysis.

    The JPO progress API provides a chronological list of examination events
    (filing, examination request, office actions, amendments, decisions).
    Columns are designed around this event-oriented data.
    """
    if lang == "zh":
        system_prompt = (
            "你是一个日本专利审查历史分析专家。根据用户的分析问题和JPO数据特征，"
            "确定分析表格需要哪些列。\n\n"
            "JPO 可用的数据：\n"
            "- 审查经过事件列表（出願審査請求、拒絶理由通知、意見書、手続補正書、"
            "特許査定/拒絶査定、審判請求等）\n"
            "- 注册信息（登録番号、登録日、権利状態）\n"
            "- 引用文献列表\n\n"
            "返回 JSON 格式：{\"columns\": [\"列1\", \"列2\", ...]}\n"
            "列数控制在 5-8 列。\n\n"
            "CRITICAL: 以下列每次分析都必须包含：\n"
            '- "日期" — 事件发生的日期\n'
            '- "事件类型" — 出願/審査請求/拒絶理由/応答/補正/査定/審判\n'
            '- "事件详情" — 具体内容概述\n'
            '- "审查阶段" — 出願/審査/審判/登録\n'
            '- "关键发现" — 该事件的核心信息\n\n'
            "根据用户的具体问题可增加列。"
        )
    else:
        system_prompt = (
            "You are a Japan patent examination history analysis expert. "
            "Determine the columns needed for an analysis table based on the "
            "user's question and available JPO data.\n\n"
            "Available JPO data:\n"
            "- Examination progress events (filing, examination request, "
            "office actions, responses, amendments, decisions, appeals)\n"
            "- Registration information (registration number, date, rights status)\n"
            "- Citation list\n\n"
            'Return JSON: {"columns": ["Col1", "Col2", ...]}\n'
            "Keep to 5-8 columns.\n\n"
            "CRITICAL required columns:\n"
            '- "Date" — event date\n'
            '- "Event Type" — filing/examination request/rejection/response/amendment/decision/appeal\n'
            '- "Description" — what happened\n'
            '- "Examination Phase" — application/examination/appeal/registration\n'
            '- "Key Findings" — core information at this step\n'
            "All column names MUST be in English."
        )

    user_content = (
        f"User question: {query}\n"
        f"Examination events count: {event_count}\n"
        f"Determine the table column definitions."
    )

    result = await provider.complete_json(system_prompt, user_content)
    return result.get("columns", [
        "日期", "事件类型", "事件详情", "审查阶段", "关键发现",
    ])


async def analyze_single_event(
    event: dict[str, Any],
    columns: list[str],
    query: str,
    provider: Any,
    lang: str = "zh",
) -> dict:
    """Phase 2a: AI analyzes one JPO progress event against table columns.

    Args:
        event: A single JPO progress event dict with keys:
               event_date, event, event_detail, event_remarks.
        columns: Table column names from ``generate_table_columns()``.
        query: User's original question.
        provider: Pro LLM provider.
        lang: 'zh' or 'en'.
    """
    from sources.jpo_client import translate_jp_event

    cols_str = "\n".join(f'  "{c}": "..."' for c in columns)

    event_name = event.get("event", "")
    event_date = event.get("event_date", "")
    event_detail = event.get("event_detail", "") or event.get("event_remarks", "") or ""
    translated = translate_jp_event(event_name, lang)

    if lang == "zh":
        system_prompt = (
            "你是一个日本专利审查历史分析专家。根据提供的事件信息，"
            "按照以下维度进行分析：\n\n"
            f"{chr(10).join(f'- {c}' for c in columns)}\n\n"
            f"返回 JSON：\n{{\n{cols_str}\n}}\n\n"
            "分析要求：\n"
            "- 基于实际提供的信息，不要编造\n"
            "- 每个字段 1-3 句话，要有依据\n"
            "- 如果某维度在事件中找不到明确信息，填写\"未在此事件中体现\"\n"
        )
        user_content = (
            f"用户问题：{query}\n\n"
            f"事件日期：{event_date}\n"
            f"事件名称：{event_name}（{translated}）\n"
            f"事件详情：{event_detail[:2000]}\n\n"
            f"按维度分析并返回 JSON。"
        )
    else:
        system_prompt = (
            "You are a Japan patent examination history analysis expert. "
            "Analyze the provided event against the following dimensions:\n\n"
            f"{chr(10).join(f'- {c}' for c in columns)}\n\n"
            f"Return JSON:\n{{\n{cols_str}\n}}\n\n"
            "Requirements:\n"
            "- Base analysis on the actual provided information\n"
            "- 1-3 specific sentences per field\n"
            '- If a dimension cannot be determined, write "Not applicable to this event"\n'
            "Write ALL content in English."
        )
        user_content = (
            f"User question: {query}\n\n"
            f"Event Date: {event_date}\n"
            f"Event Name: {event_name} ({translated})\n"
            f"Event Detail: {event_detail[:2000]}\n\n"
            f"Analyze by dimension and return JSON."
        )

    result = await provider.complete_json(system_prompt, user_content)

    row: dict = {"_event_name": event_name, "_event_date": event_date}
    for col in columns:
        if col in result:
            row[col] = result[col]
        else:
            row[col] = result.get(
                col,
                "未在此事件中体现" if lang == "zh" else "Not applicable to this event",
            )
    return row


async def generate_event_summary(
    event: dict[str, Any],
    row: dict,
    query: str,
    provider: Any,
    lang: str = "zh",
) -> str:
    """Phase 2b: Generate a 2-3 sentence summary of one JPO event."""
    from sources.jpo_client import translate_jp_event

    row_str = "\n".join(
        f"{k}: {v}" for k, v in row.items()
        if not k.startswith("_")
    )
    event_name = event.get("event", "")
    translated = translate_jp_event(event_name, lang)
    event_date = event.get("event_date", "")
    event_detail = event.get("event_detail", "") or event.get("event_remarks", "") or ""

    if lang == "zh":
        system_prompt = (
            "你是一个专利审查分析专家。基于分析结果，用 2-3 句话总结该审查事件的核心发现。"
            "直接输出总结，不要 JSON。"
        )
        user_content = (
            f"用户问题：{query}\n"
            f"事件日期：{event_date}\n"
            f"事件名称：{event_name}（{translated}）\n"
            f"事件详情：{event_detail[:1000]}\n"
            f"分析结果：\n{row_str}\n\n"
            f"请给出简洁总结。"
        )
    else:
        system_prompt = (
            "You are a patent examination analysis expert. Summarize the core "
            "findings of this examination event in 2-3 sentences. "
            "Output directly, no JSON. Write in English."
        )
        user_content = (
            f"User question: {query}\n"
            f"Event Date: {event_date}\n"
            f"Event Name: {event_name} ({translated})\n"
            f"Event Detail: {event_detail[:1000]}\n"
            f"Analysis results:\n{row_str}\n\n"
            f"Please provide a concise summary in English."
        )

    import asyncio
    llm = provider._get_langchain_llm(streaming=True)
    messages = [("system", system_prompt), ("human", user_content)]
    chunks: list[str] = []
    try:
        async def _stream():
            async for chunk in llm.astream(messages):
                if chunk.content:
                    chunks.append(chunk.content)
        await asyncio.wait_for(_stream(), timeout=300)
    except asyncio.TimeoutError:
        pass
    text = "".join(chunks).strip()
    if "</think>" in text:
        text = text[text.rfind("</think>") + len("</think>"):].strip()
    return text or ""


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
