"""China patent examination history analysis pipeline.

Resolves a patent ID (US/CN/EP) to a Chinese application number via the
EPO family API, fetches examination review decisions from the sipop.cn
open data platform, and generates an AI-powered analysis report.

Flow::

    patent_id (US/CN/EP) → [EPO family → CN app_number] → SIPOP API
    → classify decisions → AI analysis → report (DOCX + PDF)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from sources.logger import Logger

_logger = Logger("china_examination.log")

# ── Bilingual labels ────────────────────────────────────────────────────────────

_REPORT_TITLES = {
    "zh": "中国专利 {patent_id} 审查历史分析报告",
    "en": "Examination History Analysis Report for Chinese Patent {patent_id}",
}

_EXEC_HEADINGS = {
    "zh": "核心审查洞察",
    "en": "Key Examination Insights",
}

_ANALYSIS_TABLE_HEADINGS = {
    "zh": "审查决定分析表",
    "en": "Examination Decision Analysis Table",
}

_TIMELINE_HEADINGS = {
    "zh": "审查时间线",
    "en": "Examination Timeline",
}


# ── Data classes ────────────────────────────────────────────────────────────────


@dataclass
class ExaminationEvent:
    """A single examination review decision from patentSupport.queryPatentReview."""

    decision_number: str = ""       # e.g. "7971"
    decision_date: str = ""         # YYYYMMDD
    decision: str = ""              # reexamination | invalidation | opposition | overrule | affirmation | part-invalidation
    appeal_type: str = ""           # invalidation | reexamination | opposition
    appellant: str = ""             # the appellant
    assignee: str = ""              # patent assignee
    complainant: str | None = None  # complainant (for invalidation cases)
    invention_title: str = ""
    chief_examiner: str = ""
    leader_examiner: str = ""
    member_examiner: str = ""
    assistant_examiner: str | None = None
    law_reference: str = ""               # cited legal provisions
    decision_main_point: str | None = None
    decision_case_issue: str = ""         # case issues heading + paragraphs
    reasoning: str = ""                   # reasoning heading + paragraphs
    final_decision: str = ""              # final decision heading + paragraphs
    main_classification: str | None = None
    court_house: str = ""
    court_level: str = ""
    court_num: str = ""
    judge_chief: str = ""
    defendant: str = ""
    verdict_reasoning: str = ""
    verdict_holding: str = ""
    verdict_date: str = ""

    @property
    def decision_label_zh(self) -> str:
        """Human-readable Chinese label for the decision type."""
        _map = {
            "reexamination": "复审决定",
            "invalidation": "无效宣告决定",
            "opposition": "异议决定",
            "overrule": "驳回决定",
            "affirmation": "维持决定",
            "part-invalidation": "部分无效决定",
        }
        return _map.get(self.decision, self.decision or "审查决定")

    @property
    def decision_label_en(self) -> str:
        """Human-readable English label for the decision type."""
        _map = {
            "reexamination": "Reexamination Decision",
            "invalidation": "Invalidation Decision",
            "opposition": "Opposition Decision",
            "overrule": "Rejection Decision",
            "affirmation": "Affirmation Decision",
            "part-invalidation": "Partial Invalidation Decision",
        }
        return _map.get(self.decision, self.decision or "Examination Decision")

    @property
    def sort_date(self) -> str:
        """Normalized date for sorting (YYYYMMDD)."""
        return self.decision_date or self.verdict_date or "00000000"


# ── CN application number resolution ────────────────────────────────────────────


def is_cn_patent_id(patent_id: str) -> bool:
    """Check whether *patent_id* is already a CNIPA application number.

    CN application numbers look like: 201710216936.1 or CN201710216936.1
    (year prefix 19XX or 20XX + 8+ digits + optional check digit).
    """
    cleaned = patent_id.strip().upper().replace("CN", "").replace(".", "")
    if len(cleaned) >= 12 and cleaned.isdigit():
        if cleaned.startswith("19") or cleaned.startswith("20"):
            return True
    return False


def normalize_cn_app_number(patent_id: str) -> str:
    """Extract a clean CN application number from a user-provided ID.

    Returns a digit-only string suitable for the SIPOP API.
    """
    cleaned = patent_id.strip()
    # Strip CN prefix
    cleaned = re.sub(r'^CN\s*', '', cleaned, flags=re.IGNORECASE)
    # Keep only digits and the check-digit separator
    cleaned = cleaned.replace('.', '').replace('-', '').replace(' ', '')
    return cleaned


async def resolve_cn_application_number(
    patent_id: str,
    epo_client,
) -> tuple[str, dict[str, Any]]:
    """Resolve a patent ID (US/CN/EP/…) to a Chinese application number.

    Strategy:
    1. If *patent_id* already looks like a CN application number → use directly.
    2. Otherwise, call the EPO family API to find the CN family member.
    3. Return ``(cn_app_number, family_context)``.

    Args:
        patent_id: User-provided patent identifier.
        epo_client: ``EPOFamilyClient`` instance.

    Returns:
        Tuple of (cn_application_number, family_context_dict).
        family_context includes the full PatentFamily info for the report.

    Raises:
        ValueError: No CN family member could be found.
    """
    from sources.long_task.patent_family import EPOError

    if is_cn_patent_id(patent_id):
        cn_app = normalize_cn_app_number(patent_id)
        _logger.info(
            f"resolve_cn — direct CN match, "
            f"input={patent_id}, normalized={cn_app}"
        )
        return cn_app, {"direct_cn": True, "input": patent_id}

    # EPO family lookup
    family = None
    try:
        family = await epo_client.lookup_family(patent_id)
    except EPOError as e:
        _logger.warning(f"resolve_cn — EPO family lookup failed for {patent_id}: {e}")
        raise ValueError(
            f"EPO family lookup failed for {patent_id}: {e}"
        ) from e

    cn_member = family.get_representative("CN")
    if not cn_member:
        jurisdictions = family.jurisdictions
        raise ValueError(
            f"No CN family member found for {patent_id}. "
            f"Jurisdictions found: {', '.join(jurisdictions)}"
        )

    cn_app_number = cn_member.app_number
    # Normalize: strip any country prefix, dots, dashes
    cn_app_number = normalize_cn_app_number(cn_app_number)

    _logger.info(
        f"resolve_cn — via EPO family, "
        f"input={patent_id}, family_id={family.family_id}, "
        f"cn_app_number={cn_app_number}, "
        f"jurisdictions={family.jurisdictions}"
    )

    # Build context with full family info
    members_data = []
    for m in family.deduplicated_members:
        members_data.append({
            "country": m.country,
            "pub_number": m.pub_number,
            "kind": m.pub_kind,
            "is_granted": m.is_granted,
            "title": m.title,
            "app_number": m.app_number,
            "app_date": m.app_date,
        })

    family_context = {
        "direct_cn": False,
        "family_id": family.family_id,
        "jurisdictions": family.jurisdictions,
        "members": members_data,
        "cn_app_number": cn_app_number,
        "cn_pub_number": cn_member.pub_number if cn_member else "",
    }
    return cn_app_number, family_context


# ── Examination data fetching ───────────────────────────────────────────────────


async def fetch_examination_data(
    cn_app_number: str,
    sipop_client,
) -> tuple[list[ExaminationEvent], dict[str, Any], dict[str, Any]]:
    """Fetch all examination-related data for a Chinese patent.

    Calls:
    1. ``patentSupport.queryPatentReview`` → list of review decisions
    2. ``patentSupport.queryLawStateInfo`` → legal status summary
    3. ``patentBase.queryBasicInfo`` → patent bibliographic info

    Args:
        cn_app_number: Cleaned CN application number.
        sipop_client: ``SipopClient`` instance.

    Returns:
        Tuple of (list_of_ExaminationEvent, law_state_dict, basic_info_dict).
    """
    from sources.sipop_client import SipopAPIError, SipopAuthError

    # Fetch review decisions (primary data)
    try:
        review_data = await sipop_client.query_patent_review(cn_app_number)
    except (SipopAPIError, SipopAuthError) as e:
        _logger.error(f"fetch review failed — cn_app={cn_app_number}, error={e}")
        raise
    except Exception as e:
        _logger.error(f"fetch review unexpected error — cn_app={cn_app_number}, error={e}")
        raise

    _logger.info(
        f"fetch examination — cn_app={cn_app_number}, "
        f"review_decisions_count={len(review_data) if isinstance(review_data, list) else 1}"
    )

    # Parse into ExaminationEvent objects
    events = _parse_review_decisions(review_data)
    events.sort(key=lambda e: e.sort_date)

    # Fetch legal status
    law_state: dict[str, Any] = {}
    try:
        law_state = await sipop_client.query_law_state(cn_app_number)
    except Exception as e:
        _logger.warning(f"fetch law_state failed (non-fatal): {e}")

    # Fetch basic info
    basic_info: dict[str, Any] = {}
    try:
        basic_info = await sipop_client.query_basic_info(cn_app_number)
    except Exception as e:
        _logger.warning(f"fetch basic_info failed (non-fatal): {e}")

    return events, law_state, basic_info


def _parse_review_decisions(
    raw_data: list[dict[str, Any]],
) -> list[ExaminationEvent]:
    """Parse raw review decision dicts from SIPOP into ExaminationEvent objects."""
    events: list[ExaminationEvent] = []
    for item in (raw_data or []):
        if not isinstance(item, dict):
            continue

        # Helper: join heading + paragraphs for a section
        def _join_section(heading_key: str, paragraphs_key: str) -> str:
            parts = []
            heading = item.get(heading_key, "")
            if heading and isinstance(heading, str) and heading.strip():
                parts.append(heading.strip())
            para = item.get(paragraphs_key, "")
            if para:
                if isinstance(para, list):
                    text = " ".join(str(p) for p in para)
                elif isinstance(para, str):
                    text = para
                else:
                    text = str(para)
                text = text.strip()
                if text:
                    parts.append(text)
            return "\n\n".join(parts)

        evt = ExaminationEvent(
            decision_number=str(item.get("decisionNumber", "")),
            decision_date=str(item.get("decisionDate", "")),
            decision=str(item.get("decision", "")),
            appeal_type=str(item.get("appealType", "")),
            appellant=str(item.get("appellant", "")),
            assignee=str(item.get("assignee", "")),
            complainant=item.get("complainant"),
            invention_title=str(item.get("inventionTitle", "")),
            chief_examiner=str(item.get("chiefExaminer", "")),
            leader_examiner=str(item.get("leaderExaminer", "")),
            member_examiner=str(item.get("memberExaminer", "")),
            assistant_examiner=item.get("assistantExaminer"),
            law_reference=str(item.get("lawReference", "")),
            decision_main_point=item.get("decisionMainPoint"),
            decision_case_issue=_join_section(
                "decisionCaseIssueHeading", "decisionCaseIssueParagraphs",
            ),
            reasoning=_join_section("reasoningHeading", "reasoningParagraphs"),
            final_decision=_join_section(
                "finalDecisionHeading", "finalDecisionParagraphs",
            ),
            main_classification=item.get("mainClassification"),
            court_house=str(item.get("courtHouse", "")),
            court_level=str(item.get("courtLevel", "")),
            court_num=str(item.get("courtNum", "")),
            judge_chief=str(item.get("judgeChief", "")),
            defendant=str(item.get("defendant", "")),
            verdict_reasoning=str(item.get("verdictReasoning", "")),
            verdict_holding=str(item.get("verdictHolding", "")),
            verdict_date=str(item.get("verdictDate", "")),
        )
        events.append(evt)
    return events


# ── AI Analysis ─────────────────────────────────────────────────────────────────


async def generate_table_columns(
    query: str,
    event_count: int,
    provider: Any,
    lang: str = "zh",
) -> list[str]:
    """Phase 1: Use Flash LLM to generate table column definitions."""
    if lang == "zh":
        system_prompt = (
            "你是一个中国专利审查历史分析专家。根据用户的分析问题，确定分析表格需要哪些列。\n\n"
            "返回 JSON 格式：{\"columns\": [\"列1\", \"列2\", ...]}\n"
            "列数控制在 5-8 列。\n\n"
            "CRITICAL: 以下列每次分析都必须包含：\n"
            '- "决定类型" — 复审/无效/异议/驳回/维持\n'
            '- "决定日期" — 审查决定作出的日期\n'
            '- "决定号" — 审查决定的编号\n'
            '- "决定要点" — 该决定的核心法律问题/争议焦点\n'
            '- "法律依据" — 引用的法律条款\n'
            '- "审理结论" — 合议组/法院的最终裁定\n\n'
            "根据用户的具体问题，可以额外增加列如：\n"
            '- "当事人" — 上诉人/请求人/专利权人\n'
            '- "合议组" — 审查人员组成\n'
            '- "影响" — 该决定对专利保护范围的影响\n'
        )
    else:
        system_prompt = (
            "You are a China patent examination history analysis expert. "
            "Determine the columns needed for an analysis table.\n\n"
            'Return JSON: {"columns": ["Col1", "Col2", ...]}\n'
            "Keep to 5-8 columns.\n\n"
            "CRITICAL required columns:\n"
            '- "Decision Type" — reexamination/invalidation/opposition/rejection/affirmation\n'
            '- "Decision Date" — date of the examination decision\n'
            '- "Decision Number" — decision reference number\n'
            '- "Key Issues" — core legal questions / disputed points\n'
            '- "Legal Basis" — cited legal provisions\n'
            '- "Outcome" — final ruling by the board/court\n'
            "All column names MUST be in English."
        )

    user_content = (
        f"User question: {query}\n"
        f"Examination events count: {event_count}\n"
        f"Determine the table column definitions."
    )

    result = await provider.complete_json(system_prompt, user_content)
    return result.get("columns", [
        "决定类型", "决定日期", "决定号", "决定要点", "法律依据", "审理结论",
    ])


async def analyze_single_event(
    event: ExaminationEvent,
    columns: list[str],
    query: str,
    provider: Any,
    lang: str = "zh",
) -> dict:
    """Analyze one examination event against the table columns using AI."""
    # Build column → value prompt
    cols_str = "\n".join(f'  "{c}": "..."' for c in columns if c != "决定号")

    if lang == "zh":
        system_prompt = (
            "你是一个中国专利审查历史分析专家。根据审查决定的全文，按照以下维度进行分析：\n\n"
            f"{chr(10).join(f'- {c}' for c in columns)}\n\n"
            f"返回 JSON：\n{{\n{cols_str}\n}}\n\n"
            "分析要求：\n"
            "- 基于审查决定的实际内容，不要编造\n"
            "- 每个字段 1-3 句话，要有依据\n"
            "- 如果某维度在决定中找不到明确信息，填写\"未在决定中明确说明\"\n"
            "- 法律依据要引用具体的法条编号"
        )
    else:
        system_prompt = (
            "You are a China patent examination history analysis expert. "
            "Analyze the examination decision against the following dimensions:\n\n"
            f"{chr(10).join(f'- {c}' for c in columns)}\n\n"
            f"Return JSON:\n{{\n{cols_str}\n}}\n\n"
            "Requirements:\n"
            "- Base analysis on the actual decision content, do not fabricate\n"
            "- 1-3 specific sentences per field\n"
            '- If a dimension is not found in the decision, write "Not explicitly stated"\n'
            "- Cite specific legal article numbers for legal basis\n"
            "Write ALL content in English."
        )

    # Build event text for analysis
    event_text = _format_event_for_llm(event)

    user_content = (
        f"User question: {query}\n\n"
        f"Decision Number: {event.decision_number}\n"
        f"Decision Date: {event.decision_date}\n"
        f"Decision Type: {event.decision_label_zh if lang == 'zh' else event.decision_label_en}\n\n"
        f"{event_text[:8000]}\n\n"
        f"Analyze by dimension and return JSON."
    )

    result = await provider.complete_json(system_prompt, user_content)

    # Align keys with column names
    row: dict = {"决定号": event.decision_number}
    for col in columns:
        if col in result:
            row[col] = result[col]
        elif col != "决定号":
            row[col] = result.get(col, "未在决定中明确说明" if lang == "zh" else "Not explicitly stated")
    return row


def _format_event_for_llm(event: ExaminationEvent) -> str:
    """Format an ExaminationEvent as a text block for LLM input."""
    parts: list[str] = []

    if event.invention_title:
        parts.append(f"专利名称：{event.invention_title}")
    if event.appeal_type:
        parts.append(f"程序类型：{event.appeal_type}")
    if event.appellant:
        parts.append(f"上诉人/请求人：{event.appellant}")
    if event.assignee:
        parts.append(f"专利权人：{event.assignee}")
    if event.complainant:
        parts.append(f"请求人：{event.complainant}")

    # Examination panel
    examiners = []
    if event.chief_examiner:
        examiners.append(f"审判长：{event.chief_examiner}")
    if event.leader_examiner:
        examiners.append(f"主审员：{event.leader_examiner}")
    if event.member_examiner:
        examiners.append(f"参审员：{event.member_examiner}")
    if examiners:
        parts.append("合议组：" + "；".join(examiners))

    if event.law_reference:
        parts.append(f"法律依据：{event.law_reference}")

    if event.decision_case_issue:
        parts.append(f"案件争议焦点：\n{event.decision_case_issue}")

    if event.reasoning:
        parts.append(f"审理推理：\n{event.reasoning}")

    if event.final_decision:
        parts.append(f"最终决定：\n{event.final_decision}")

    if event.decision_main_point:
        parts.append(f"决定要点：{event.decision_main_point}")

    # Court judgment data
    if event.court_house:
        parts.append(f"法院：{event.court_house}")
    if event.court_level:
        parts.append(f"审级：{event.court_level}")
    if event.judge_chief:
        parts.append(f"法官：{event.judge_chief}")
    if event.verdict_reasoning:
        parts.append(f"判决推理：\n{event.verdict_reasoning}")
    if event.verdict_holding:
        parts.append(f"判决结果：\n{event.verdict_holding}")

    return "\n\n".join(parts)


async def generate_event_summary(
    event: ExaminationEvent,
    row: dict,
    query: str,
    provider: Any,
    lang: str = "zh",
) -> str:
    """Generate a 2-3 sentence summary of one analysis result."""
    row_str = "\n".join(f"{k}: {v}" for k, v in row.items() if k != "决定号")

    if lang == "zh":
        system_prompt = (
            "你是一个专利审查分析专家。基于分析结果，用 2-3 句话总结该审查决定的核心发现。"
            "直接输出总结，不要 JSON。"
        )
        user_content = (
            f"用户问题：{query}\n"
            f"决定号：{event.decision_number}\n"
            f"决定类型：{event.decision_label_zh}\n"
            f"分析结果：\n{row_str}\n\n"
            f"请给出简洁总结。"
        )
    else:
        system_prompt = (
            "You are a patent examination analysis expert. Summarize the core "
            "findings of this examination decision in 2-3 sentences. "
            "Output directly, no JSON. Write in English."
        )
        user_content = (
            f"User question: {query}\n"
            f"Decision Number: {event.decision_number}\n"
            f"Decision Type: {event.decision_label_en}\n"
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


async def build_examination_timeline(
    events: list[ExaminationEvent],
    law_state: dict[str, Any],
    basic_info: dict[str, Any],
    lang: str = "zh",
) -> str:
    """Build a chronological timeline from examination events as markdown text."""
    events_sorted = sorted(events, key=lambda e: e.sort_date)

    if lang == "zh":
        lines = ["## 审查时间线\n"]

        # Add basic info header
        title = basic_info.get("title", "")
        app_num = basic_info.get("applicationDocNum", "")
        app_date = basic_info.get("applicationDate", "")
        pub_date = basic_info.get("publicationDate", "")
        applicants = basic_info.get("applicant", [])
        applicant_str = ", ".join(applicants) if isinstance(applicants, list) else str(applicants)

        if title:
            lines.append(f"**专利名称**：{title}\n")
        if app_num:
            lines.append(f"**申请号**：{app_num}")
        if app_date:
            lines.append(f"**申请日**：{app_date}")
        if pub_date:
            lines.append(f"**公开日**：{pub_date}")
        if applicant_str:
            lines.append(f"**申请人**：{applicant_str}")

        # Legal status
        law_status = law_state.get("lawStatus", "")
        law_date = law_state.get("date", "")
        if law_status or law_date:
            lines.append(f"\n**当前法律状态**：{law_status}（{law_date}）")

        lines.append("")
        lines.append("| 日期 | 决定类型 | 决定号 | 当事人 | 结论 |")
        lines.append("|------|----------|--------|--------|------|")

        for evt in events_sorted:
            date = evt.decision_date or evt.verdict_date
            if len(date) == 8:
                date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
            party = evt.appellant or evt.assignee or ""
            outcome = _short_outcome(evt, lang="zh")
            lines.append(
                f"| {date} | {evt.decision_label_zh} | {evt.decision_number} "
                f"| {party} | {outcome} |"
            )
    else:
        lines = ["## Examination Timeline\n"]

        title = basic_info.get("title", "")
        app_num = basic_info.get("applicationDocNum", "")
        app_date = basic_info.get("applicationDate", "")
        pub_date = basic_info.get("publicationDate", "")
        applicants = basic_info.get("applicant", [])
        applicant_str = ", ".join(applicants) if isinstance(applicants, list) else str(applicants)

        if title:
            lines.append(f"**Patent Title**: {title}\n")
        if app_num:
            lines.append(f"**Application Number**: {app_num}")
        if app_date:
            lines.append(f"**Filing Date**: {app_date}")
        if pub_date:
            lines.append(f"**Publication Date**: {pub_date}")
        if applicant_str:
            lines.append(f"**Applicant**: {applicant_str}")

        law_status = law_state.get("lawStatus", "")
        law_date = law_state.get("date", "")
        if law_status or law_date:
            lines.append(f"\n**Current Legal Status**: {law_status} ({law_date})")

        lines.append("")
        lines.append("| Date | Decision Type | Number | Party | Outcome |")
        lines.append("|------|---------------|--------|-------|---------|")

        for evt in events_sorted:
            date = evt.decision_date or evt.verdict_date
            if len(date) == 8:
                date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
            party = evt.appellant or evt.assignee or ""
            outcome = _short_outcome(evt, lang="en")
            lines.append(
                f"| {date} | {evt.decision_label_en} | {evt.decision_number} "
                f"| {party} | {outcome} |"
            )

    return "\n".join(lines)


def _short_outcome(event: ExaminationEvent, lang: str = "zh") -> str:
    """Extract a 1-line outcome summary from the event."""
    # Try final decision first
    fd = event.final_decision
    if fd:
        # Take first sentence or first 80 chars
        first_line = fd.split("\n")[0].strip()
        if len(first_line) > 80:
            first_line = first_line[:77] + "..."
        return first_line if first_line else (fd[:80])

    # Fall back to decision main point
    if event.decision_main_point:
        mp = str(event.decision_main_point)
        if len(mp) > 80:
            mp = mp[:77] + "..."
        return mp

    # Fall back to verdict holding
    vh = event.verdict_holding
    if vh:
        if len(vh) > 80:
            vh = vh[:77] + "..."
        return vh

    if lang == "zh":
        return event.decision_label_zh
    return event.decision_label_en
