"""EPO patent examination history analysis pipeline.

Resolves a patent ID (US/CN/EP/JP/WO) to an EP application number via the
EPO family API, fetches examination data from the EPO OPS Register and
Published Data APIs, and generates an AI-powered analysis report.

Data sources
------------
- **Register biblio** — bibliographic data (title, applicant, IPC, status)
- **Register events** — legal events timeline (filing, publication, grant, etc.)
- **Register procedural-steps** — examination procedure steps
- **Published Data claims** — current claims text
- **Published Data search report** — Search Opinion / Written Opinion full text
  (available for A1/A3 publications)

Analysis depth
--------------
- Full-text AI analysis: Search Opinion / Written Opinion
- Timeline / structured analysis: procedural-steps (communications, responses,
  amendments — metadata only, no full text)
- Outcome analysis: grant/refusal status from events + biblio

Flow::

    patent_id → [EPO family → EP app_number] → Register + Published Data
    → AI analysis → report section for cross-jurisdiction integration
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sources.logger import Logger

_logger = Logger("epo_examination.log")

# ── Bilingual labels ────────────────────────────────────────────────────────────

_REPORT_TITLES = {
    "zh": "欧洲专利 {patent_id} 审查历史分析报告",
    "en": "Examination History Analysis Report for European Patent {patent_id}",
}

_EXEC_HEADINGS = {
    "zh": "核心审查洞察",
    "en": "Key Examination Insights",
}

_TIMELINE_HEADINGS = {
    "zh": "审查时间线",
    "en": "Examination Timeline",
}

_SEARCH_OPINION_HEADINGS = {
    "zh": "检索意见分析",
    "en": "Search Opinion Analysis",
}

_OUTCOME_HEADINGS = {
    "zh": "审查结论",
    "en": "Examination Outcome",
}


# ── Data classes ────────────────────────────────────────────────────────────────


@dataclass
class EPExaminationEvent:
    """A single examination-relevant event synthesized from register data.

    Combines legal events and procedural-steps into a unified timeline
    ordered by date.
    """

    date: str = ""  # YYYYMMDD
    event_type: str = ""  # "search_report", "communication", "response", "amendment",
    # "oral_proceedings", "grant", "refusal", "appeal", "opposition", "other"
    code: str = ""  # original event/step code
    description: str = ""
    description_en: str = ""


# ── EP application number resolution ────────────────────────────────────────────


async def resolve_ep_application_number(
    patent_id: str,
    epo_client: Any,
) -> tuple[str, dict[str, Any]]:
    """Resolve a patent ID to an EP application number via EPO family.

    Args:
        patent_id: Any patent identifier (US, CN, EP, JP, WO, etc.).
        epo_client: ``EPOClient`` or ``EPOFamilyClient`` instance.

    Returns:
        Tuple of (ep_app_number, family_context_dict).

    Raises:
        ValueError: No EP family member could be found.
    """
    from sources.epo_ops_client import EPOError as OpsEPOError
    from sources.long_task.patent_family import EPOError as FamilyEPOError

    # Direct EP match — normalize to EPODOC format
    if re.match(r'^(EP|ep)?\s*\d{6,8}', patent_id):
        raw = re.sub(
            r'^EP\s*', '', patent_id.strip(), flags=re.IGNORECASE,
        ).replace('.', '').replace('-', '').replace(' ', '')
        _logger.info(
            f"resolve_ep — direct EP match, "
            f"input={patent_id}, normalized={raw}"
        )
        return raw, {"direct_ep": True, "input_id": patent_id}

    # EPO family lookup
    family = None
    try:
        family = await epo_client.lookup_family(patent_id)
    except (OpsEPOError, FamilyEPOError, Exception) as e:
        _logger.warning(f"resolve_ep — EPO family lookup failed for {patent_id}: {e}")
        raise ValueError(
            f"EPO family lookup failed for {patent_id}: {e}"
        ) from e

    ep_member = family.get_representative("EP")
    if not ep_member:
        jurisdictions = family.jurisdictions
        raise ValueError(
            f"No EP family member found for {patent_id}. "
            f"Jurisdictions found: {', '.join(jurisdictions)}"
        )

    ep_app_number = ep_member.app_number
    # Normalize: strip EP prefix, dots, dashes
    ep_app_number = re.sub(
        r'^EP\s*', '', ep_app_number or '', flags=re.IGNORECASE,
    ).replace('.', '').replace('-', '').replace(' ', '')

    _logger.info(
        f"resolve_ep — via EPO family, "
        f"input={patent_id}, family_id={family.family_id}, "
        f"ep_app_number={ep_app_number}, "
        f"ep_pub_number={ep_member.pub_number}, "
        f"jurisdictions={family.jurisdictions}"
    )

    # Build rich family context
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
        "direct_ep": False,
        "input_id": patent_id,
        "ep_app_number": ep_app_number,
        "ep_pub_number": ep_member.pub_number,
        "family_id": family.family_id,
        "jurisdictions": family.jurisdictions,
        "members": members,
    }

    return ep_app_number, context


# ── Data fetching ───────────────────────────────────────────────────────────────


async def fetch_examination_data(
    ep_app_number: str,
    epo_client: Any,
) -> dict[str, Any]:
    """Fetch all available EPO examination data for an EP application.

    Calls 5 endpoints in parallel: register biblio, events, procedural-steps,
    published claims, and search report (via published description).

    Args:
        ep_app_number: EP application number (digits only).
        epo_client: ``EPOClient`` instance.

    Returns:
        Dict with keys:
          - ep_app_number: str
          - biblio: ``EPORegisterBiblio`` or None
          - events: list[``EPORegisterEvent``]
          - procedural_steps: list[``EPOProceduralStep``]
          - claims_text: str
          - search_report_text: str
          - timeline_events: list[``EPExaminationEvent``]  — synthesized timeline
          - has_biblio: bool
          - has_events: bool
          - has_steps: bool
          - has_claims: bool
          - has_search_report: bool
          - status: str  — "GRANTED" / "PENDING" / "REFUSED" / "UNKNOWN"
    """
    import asyncio

    result: dict[str, Any] = {
        "ep_app_number": ep_app_number,
        "biblio": None,
        "events": [],
        "procedural_steps": [],
        "claims_text": "",
        "search_report_text": "",
        "timeline_events": [],
        "has_biblio": False,
        "has_events": False,
        "has_steps": False,
        "has_claims": False,
        "has_search_report": False,
        "status": "UNKNOWN",
    }

    # ── Parallel fetch ──
    async def _safe(coro, label: str):
        try:
            return await coro
        except Exception as e:
            _logger.warning(f"epo_fetch_{label}_failed — app={ep_app_number}: {e}")
            return None

    import asyncio as _asyncio
    biblio, events, steps, claims, search = await _asyncio.gather(
        _safe(epo_client.register_biblio(ep_app_number), "biblio"),
        _safe(epo_client.register_events(ep_app_number), "events"),
        _safe(epo_client.register_procedural_steps(ep_app_number), "steps"),
        _safe(epo_client.published_claims(ep_app_number), "claims"),
        _safe(epo_client.published_search_report_text(ep_app_number), "search_report"),
    )

    if biblio is not None:
        result["biblio"] = biblio
        result["has_biblio"] = True
        result["status"] = biblio.status

    if events is not None:
        result["events"] = events
        result["has_events"] = bool(events)

    if steps is not None:
        result["procedural_steps"] = steps
        result["has_steps"] = bool(steps)

    if claims:
        result["claims_text"] = claims
        result["has_claims"] = True

    if search:
        result["search_report_text"] = search
        result["has_search_report"] = True

    # ── Synthesize unified timeline ──
    result["timeline_events"] = _synthesize_timeline(
        events or [], steps or [], biblio
    )

    _logger.info(
        f"epo_fetch_summary — app={ep_app_number}, "
        f"status={result['status']}, "
        f"events={len(events or [])}, "
        f"steps={len(steps or [])}, "
        f"claims_chars={len(claims or '')}, "
        f"search_report_chars={len(search or '')}, "
        f"timeline_entries={len(result['timeline_events'])}"
    )

    return result


# ── Timeline synthesis ──────────────────────────────────────────────────────────


def _synthesize_timeline(
    events: list[Any],
    steps: list[Any],
    biblio: Any,
) -> list[EPExaminationEvent]:
    """Combine register events and procedural steps into a unified timeline.

    Classifies each entry into a standard type for consistent AI analysis.
    """
    from sources.epo_ops_client import EPORegisterEvent, EPOProceduralStep

    timeline: list[EPExaminationEvent] = []

    # Add biblio milestone: filing date
    if biblio and biblio.filing_date:
        timeline.append(EPExaminationEvent(
            date=biblio.filing_date,
            event_type="filing",
            code="FILING",
            description=f"Application filed — {biblio.title_en or biblio.title}",
            description_en=f"Application filed — {biblio.title_en or biblio.title}",
        ))

    # Add biblio milestone: publication date
    if biblio and biblio.pub_date:
        timeline.append(EPExaminationEvent(
            date=biblio.pub_date,
            event_type="publication",
            code="PUB",
            description="Application published",
            description_en="Application published",
        ))

    # Classify procedural steps
    for s in steps:
        et = _classify_step(s)
        if et:
            timeline.append(et)

    # Add grant/refusal from events
    for e in events:
        et = _classify_event(e)
        if et:
            timeline.append(et)

    # Sort by date ascending
    timeline.sort(key=lambda x: x.date)

    return timeline


def _classify_step(step: Any) -> EPExaminationEvent | None:
    """Classify a procedural step into an EPExaminationEvent."""
    desc = (step.description or "").lower()
    desc_en = (step.description_en or "").lower()
    combined = f"{desc} {desc_en}"

    # Search report
    if any(kw in combined for kw in (
        "search report", "search opinion", "supplementary search",
        "extended search", "european search",
    )):
        return EPExaminationEvent(
            date=step.step_date,
            event_type="search_report",
            code=step.step_code,
            description=step.description,
            description_en=step.description_en,
        )

    # Communication from Examining Division
    if any(kw in combined for kw in (
        "communication", "examining division", "rule 71",
        "rule 94", "summons to oral", "annex to communication",
    )):
        return EPExaminationEvent(
            date=step.step_date,
            event_type="communication",
            code=step.step_code,
            description=step.description,
            description_en=step.description_en,
        )

    # Applicant response / amendment
    if any(kw in combined for kw in (
        "reply", "response", "amendment", "amended claims",
        "applicant", "letter", "statement of grounds",
    )):
        return EPExaminationEvent(
            date=step.step_date,
            event_type="response",
            code=step.step_code,
            description=step.description,
            description_en=step.description_en,
        )

    # Oral proceedings
    if any(kw in combined for kw in (
        "oral proceedings", "hearing", "minutes",
    )):
        return EPExaminationEvent(
            date=step.step_date,
            event_type="oral_proceedings",
            code=step.step_code,
            description=step.description,
            description_en=step.description_en,
        )

    # Opposition
    if any(kw in combined for kw in (
        "opposition", "opponent", "notice of opposition",
    )):
        return EPExaminationEvent(
            date=step.step_date,
            event_type="opposition",
            code=step.step_code,
            description=step.description,
            description_en=step.description_en,
        )

    # Appeal
    if any(kw in combined for kw in (
        "appeal", "board of appeal", "statement of grounds",
    )):
        return EPExaminationEvent(
            date=step.step_date,
            event_type="appeal",
            code=step.step_code,
            description=step.description,
            description_en=step.description_en,
        )

    # Default: include as "other" — still potentially useful
    return EPExaminationEvent(
        date=step.step_date,
        event_type="other",
        code=step.step_code,
        description=step.description,
        description_en=step.description_en,
    )


def _classify_event(event: Any) -> EPExaminationEvent | None:
    """Classify a legal event into an EPExaminationEvent. Only picks key ones."""
    code = (event.event_code or "").upper()
    desc = (event.description or "").lower()

    # Grant
    if code in ("GRANT", "GRANTE", "PUBG", "B1PUB") or any(
        kw in desc for kw in ("grant", "mention of grant", "patent granted")
    ):
        return EPExaminationEvent(
            date=event.event_date,
            event_type="grant",
            code=event.event_code,
            description=event.description,
            description_en=event.description_en,
        )

    # Refusal / withdrawal
    if code in ("REFUS", "WDRN", "WTHD", "LAPS") or any(
        kw in desc for kw in ("refusal", "withdrawn", "lapsed", "abandoned")
    ):
        return EPExaminationEvent(
            date=event.event_date,
            event_type="refusal",
            code=event.event_code,
            description=event.description,
            description_en=event.description_en,
        )

    # Opposition
    if code in ("OPPO", "OPPOS", "OPPON") or "opposition" in desc:
        return EPExaminationEvent(
            date=event.event_date,
            event_type="opposition",
            code=event.event_code,
            description=event.description,
            description_en=event.description_en,
        )

    # Only return key events to avoid noise
    return None


# ── AI Analysis ─────────────────────────────────────────────────────────────────


async def generate_table_columns(
    query: str,
    event_count: int,
    provider: Any,
    lang: str = "zh",
) -> list[str]:
    """Phase 1: Flash LLM generates table column definitions for EPO analysis.

    Columns are designed around: Search Opinion content, procedural timeline,
    and outcome — reflecting what data is actually available from EPO.
    """
    if lang == "zh":
        system_prompt = (
            "你是一个欧洲专利审查历史分析专家。根据用户的分析问题和EPO数据特征，"
            "确定分析表格需要哪些列。\n\n"
            "EPO 可用的数据：\n"
            "- 检索意见/书面意见全文（Search Opinion / Written Opinion）\n"
            "- 审查程序步骤时间线（procedural steps）\n"
            "- 法律事件（filing, grant, opposition, etc.）\n"
            "- 权利要求文本\n\n"
            "返回 JSON 格式：{\"columns\": [\"列1\", \"列2\", ...]}\n"
            "列数控制在 5-8 列。\n\n"
            "CRITICAL: 以下列每次分析都必须包含：\n"
            '- "日期" — 事件/步骤发生的日期\n'
            '- "事件类型" — search_report / communication / response / grant / refusal\n'
            '- "事件描述" — 具体内容概述\n'
            '- "审查阶段" — 检索/实质审查/授权/异议\n'
            '- "关键发现" — 该步骤中的核心信息/争议点\n\n'
            "根据用户的具体问题可增加列。"
        )
    else:
        system_prompt = (
            "You are an EPO patent examination history analysis expert. "
            "Determine the columns needed for an analysis table based on the "
            "user's question and the available EPO data.\n\n"
            "Available EPO data:\n"
            "- Search Opinion / Written Opinion full text\n"
            "- Examination procedural steps timeline\n"
            "- Legal events (filing, grant, opposition, etc.)\n"
            "- Claims text\n\n"
            'Return JSON: {"columns": ["Col1", "Col2", ...]}\n'
            "Keep to 5-8 columns.\n\n"
            "CRITICAL required columns:\n"
            '- "Date" — event/step date\n'
            '- "Event Type" — search_report / communication / response / grant / refusal\n'
            '- "Description" — what happened\n'
            '- "Examination Phase" — search / substantive examination / grant / opposition\n'
            '- "Key Findings" — core information / issues at this step\n'
            "All column names MUST be in English."
        )

    user_content = (
        f"User question: {query}\n"
        f"Timeline events count: {event_count}\n"
        f"Determine the table column definitions."
    )

    result = await provider.complete_json(system_prompt, user_content)
    return result.get("columns", [
        "日期", "事件类型", "事件描述", "审查阶段", "关键发现",
    ])


async def analyze_single_timeline_event(
    event: EPExaminationEvent,
    search_report_text: str,
    claims_text: str,
    columns: list[str],
    query: str,
    provider: Any,
    lang: str = "zh",
) -> dict:
    """Phase 2a: AI analyzes one timeline event against table columns.

    For search_report events, also provides the search report text as context.
    """
    cols_str = "\n".join(f'  "{c}": "..."' for c in columns)

    # Build context
    context_parts = [
        f"Event Date: {event.date}",
        f"Event Type: {event.event_type}",
        f"Event Code: {event.code}",
        f"Description: {event.description_en or event.description}",
    ]
    if event.event_type == "search_report" and search_report_text:
        context_parts.append(
            f"\nSearch Report / Written Opinion excerpt:\n{search_report_text[:6000]}"
        )
    if claims_text:
        context_parts.append(
            f"\nClaims (for reference):\n{claims_text[:2000]}"
        )

    context = "\n".join(context_parts)

    if lang == "zh":
        system_prompt = (
            "你是一个欧洲专利审查历史分析专家。根据提供的事件信息，"
            "按照以下维度进行分析：\n\n"
            f"{chr(10).join(f'- {c}' for c in columns)}\n\n"
            f"返回 JSON：\n{{\n{cols_str}\n}}\n\n"
            "分析要求：\n"
            "- 基于实际提供的信息，不要编造\n"
            "- 每个字段 1-3 句话，要有依据\n"
            "- 如果某维度在事件中找不到明确信息，填写\"未在此步骤中体现\"\n"
        )
    else:
        system_prompt = (
            "You are an EPO patent examination history analysis expert. "
            "Analyze the provided event against the following dimensions:\n\n"
            f"{chr(10).join(f'- {c}' for c in columns)}\n\n"
            f"Return JSON:\n{{\n{cols_str}\n}}\n\n"
            "Requirements:\n"
            "- Base analysis on the actual provided information\n"
            "- 1-3 specific sentences per field\n"
            '- If a dimension cannot be determined, write "Not applicable to this step"\n'
            "Write ALL content in English."
        )

    user_content = (
        f"User question: {query}\n\n"
        f"{context}\n\n"
        f"Analyze by dimension and return JSON."
    )

    result = await provider.complete_json(system_prompt, user_content)

    row: dict = {"_event_type": event.event_type, "_event_date": event.date}
    for col in columns:
        if col in result:
            row[col] = result[col]
        else:
            row[col] = result.get(
                col,
                "未在此步骤中体现" if lang == "zh" else "Not applicable to this step",
            )
    return row


async def generate_event_summary(
    event: EPExaminationEvent,
    row: dict,
    query: str,
    provider: Any,
    lang: str = "zh",
) -> str:
    """Phase 2b: Generate a 2-3 sentence summary of one timeline event."""
    row_str = "\n".join(
        f"{k}: {v}" for k, v in row.items()
        if not k.startswith("_")
    )

    if lang == "zh":
        system_prompt = (
            "你是一个专利审查分析专家。基于分析结果，用 2-3 句话总结该审查事件的核心发现。"
            "直接输出总结，不要 JSON。"
        )
        user_content = (
            f"用户问题：{query}\n"
            f"事件日期：{event.date}\n"
            f"事件类型：{event.event_type}\n"
            f"事件描述：{event.description or event.description_en}\n"
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
            f"Event Date: {event.date}\n"
            f"Event Type: {event.event_type}\n"
            f"Description: {event.description_en or event.description}\n"
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


# ── Timeline formatting ─────────────────────────────────────────────────────────


def build_examination_timeline(
    events: list[EPExaminationEvent],
    biblio: Any,
    lang: str = "zh",
) -> str:
    """Build a Markdown chronological timeline from examination events.

    Args:
        events: Synthesized timeline events (sorted by date).
        biblio: ``EPORegisterBiblio`` or None.
        lang: 'zh' or 'en'.

    Returns:
        Markdown string.
    """
    if not events:
        if lang == "zh":
            return "未获取到欧洲专利审查数据。\n"
        else:
            return "No European patent examination data available.\n"

    lines: list[str] = []

    if lang == "zh":
        lines.append("### 欧洲审查时间线\n")
    else:
        lines.append("### European Examination Timeline\n")

    # Basic info header
    if biblio:
        if lang == "zh":
            if biblio.title_en or biblio.title:
                lines.append(f"**专利名称**：{biblio.title_en or biblio.title}\n")
            if biblio.applicant:
                lines.append(f"**申请人**：{biblio.applicant}\n")
            if biblio.filing_date:
                lines.append(f"**申请日**：{biblio.filing_date}\n")
            if biblio.pub_date:
                lines.append(f"**公开日**：{biblio.pub_date}\n")
            if biblio.status:
                _status_zh = {
                    "GRANTED": "已授权", "PENDING": "审查中",
                    "REFUSED": "已驳回", "WITHDRAWN": "已撤回",
                }
                lines.append(
                    f"**状态**：{_status_zh.get(biblio.status, biblio.status)}\n"
                )
        else:
            if biblio.title_en or biblio.title:
                lines.append(f"**Title**: {biblio.title_en or biblio.title}\n")
            if biblio.applicant:
                lines.append(f"**Applicant**: {biblio.applicant}\n")
            if biblio.filing_date:
                lines.append(f"**Filing Date**: {biblio.filing_date}\n")
            if biblio.pub_date:
                lines.append(f"**Publication Date**: {biblio.pub_date}\n")
            if biblio.status:
                lines.append(f"**Status**: {biblio.status}\n")

    lines.append(f"共 {len(events)} 个审查事件。\n" if lang == "zh"
                 else f"Total {len(events)} examination events.\n")

    # Event-type labels
    _type_labels_zh = {
        "filing": "提交申请", "publication": "公开", "search_report": "检索报告/意见",
        "communication": "审查意见通知", "response": "申请人答复/修改",
        "oral_proceedings": "口头审理", "grant": "授权", "refusal": "驳回/撤回",
        "opposition": "异议", "appeal": "上诉", "other": "其他",
    }
    _type_labels_en = {
        "filing": "Filing", "publication": "Publication", "search_report": "Search Report/Opinion",
        "communication": "Communication", "response": "Applicant Response/Amendment",
        "oral_proceedings": "Oral Proceedings", "grant": "Grant", "refusal": "Refusal/Withdrawal",
        "opposition": "Opposition", "appeal": "Appeal", "other": "Other",
    }

    # Timeline table
    if lang == "zh":
        lines.append("| 日期 | 类型 | 描述 |")
        lines.append("|------|------|------|")
    else:
        lines.append("| Date | Type | Description |")
        lines.append("|------|------|-------------|")

    for evt in events:
        date_display = _format_epo_date(evt.date)
        type_label = _type_labels_zh.get(evt.event_type, evt.event_type) if lang == "zh" \
            else _type_labels_en.get(evt.event_type, evt.event_type)
        desc = (evt.description_en or evt.description)[:200] if lang == "en" \
            else (evt.description or evt.description_en)[:200]
        lines.append(f"| {date_display} | {type_label} | {desc} |")

    lines.append("")

    return "\n".join(lines)


def _format_epo_date(raw: str) -> str:
    """Normalize an EPO date string to YYYY-MM-DD."""
    if not raw:
        return "—"
    cleaned = raw.strip().replace("/", "-").replace(".", "-")
    if len(cleaned) == 8 and cleaned.isdigit():
        return f"{cleaned[:4]}-{cleaned[4:6]}-{cleaned[6:8]}"
    return cleaned[:10]


# ── Report section builder ──────────────────────────────────────────────────────


def build_epo_section(
    epo_data: dict[str, Any],
    lang: str = "zh",
) -> str:
    """Build the complete EPO examination section for the family report.

    Composes timeline + search opinion summary into a single Markdown block.

    Args:
        epo_data: Result from ``fetch_examination_data()``.
        lang: 'zh' or 'en'.

    Returns:
        Complete Markdown section for EPO examination data.
    """
    parts: list[str] = []

    ep_app = epo_data.get("ep_app_number", "")
    if lang == "zh":
        parts.append(f"## 欧洲审查历史 (EP{ep_app})\n")
    else:
        parts.append(f"## European Examination History (EP{ep_app})\n")

    biblio = epo_data.get("biblio")

    # Timeline
    timeline = build_examination_timeline(
        epo_data.get("timeline_events", []), biblio, lang,
    )
    if timeline:
        parts.append(timeline)

    # Search report / opinion summary
    search_text = epo_data.get("search_report_text", "")
    if search_text:
        parts.append(_build_search_opinion_summary(search_text, lang))

    # Outcome
    outcome = _build_outcome_summary(epo_data, lang)
    if outcome:
        parts.append(outcome)

    return "\n".join(parts)


def _build_search_opinion_summary(
    search_text: str,
    lang: str = "zh",
) -> str:
    """Build a summary block for search report / written opinion text.

    Truncates to a reasonable length — the full text is available via the
    analysis pipeline for AI processing.
    """
    heading = _SEARCH_OPINION_HEADINGS.get(lang, "Search Opinion Analysis")
    lines = [f"### {heading}\n"]

    # Show first 3000 chars as a preview
    preview = search_text[:3000]
    lines.append(preview)
    if len(search_text) > 3000:
        if lang == "zh":
            lines.append(f"\n*（检索意见全文共 {len(search_text)} 字符，以上为前3000字符预览）*\n")
        else:
            lines.append(f"\n*(Search opinion full text: {len(search_text)} chars; showing first 3000)*\n")
    lines.append("")

    return "\n".join(lines)


def _build_outcome_summary(
    epo_data: dict[str, Any],
    lang: str = "zh",
) -> str:
    """Build a summary of the final examination outcome."""
    status = epo_data.get("status", "UNKNOWN")
    events = epo_data.get("events", [])
    timeline = epo_data.get("timeline_events", [])

    heading = _OUTCOME_HEADINGS.get(lang, "Examination Outcome")
    lines = [f"### {heading}\n"]

    # Find grant/refusal events from timeline
    outcomes = [e for e in timeline
                if e.event_type in ("grant", "refusal", "opposition")]

    if lang == "zh":
        _status_labels = {
            "GRANTED": "✅ 已授权",
            "PENDING": "⏳ 审查中",
            "REFUSED": "❌ 已驳回",
            "WITHDRAWN": "⬅ 已撤回",
            "UNKNOWN": "❓ 未知",
        }
        lines.append(f"**最终状态**：{_status_labels.get(status, status)}\n")
        if outcomes:
            lines.append("**关键结论事件**：")
            for o in outcomes[-5:]:
                date = _format_epo_date(o.date)
                desc = o.description or o.description_en
                lines.append(f"- {date} — {desc}")
            lines.append("")
    else:
        lines.append(f"**Final Status**: {status}\n")
        if outcomes:
            lines.append("**Key Outcome Events**:")
            for o in outcomes[-5:]:
                date = _format_epo_date(o.date)
                desc = o.description_en or o.description
                lines.append(f"- {date} — {desc}")
            lines.append("")

    return "\n".join(lines)
