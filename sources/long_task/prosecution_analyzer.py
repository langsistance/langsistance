"""USPTO prosecution history AI analysis and report generation.

Analyzes downloaded prosecution documents (Office Actions, Applicant Responses,
Amendments, Notice of Allowance) to produce a comprehensive prosecution history
report. Supports bilingual output: Chinese (zh) or English (en) based on the
user's query language.
"""

from __future__ import annotations

import logging
from typing import Any

_logger = logging.getLogger(__name__)


# ── Report section definitions (bilingual) ────────────────────────────────────

REPORT_SECTIONS = {
    "zh": [
        {
            "heading": "审查过程总览",
            "description": "按时间顺序梳理所有 Office Action 和 Applicant Response，呈现审查过程的完整时间线",
        },
        {
            "heading": "审查意见总结",
            "description": "详细分析每一轮 Office Action 中 Examiner 的拒绝理由，包括 §102/§103/§112 等具体法条依据和引用的对比文件",
        },
        {
            "heading": "申请人的争辩点",
            "description": "总结申请人对每一次拒绝的答复策略和核心论点，分析哪些争辩被接受、哪些被驳回",
        },
        {
            "heading": "Claim 和 Specification 修改",
            "description": "逐项对比 Claim 在审查过程中的修改前后变化，分析新增的限制和删除的特征",
        },
        {
            "heading": "授权原因分析",
            "description": "分析专利最终获得授权的原因，说明 Examiner 认可了哪些修改和论点",
        },
        {
            "heading": "经验与启示",
            "description": "从本案审查历史中提炼对专利代理实务有参考价值的策略和经验教训",
        },
    ],
    "en": [
        {
            "heading": "Prosecution History Overview",
            "description": "Chronological timeline of all Office Actions and Applicant Responses throughout the examination process",
        },
        {
            "heading": "Summary of Office Actions",
            "description": "Detailed analysis of each Office Action, including rejection grounds (§102/§103/§112), statutory basis, and cited prior art references",
        },
        {
            "heading": "Applicant's Arguments",
            "description": "Summary of the applicant's response strategy and key arguments for each rejection — which arguments were accepted and which were dismissed",
        },
        {
            "heading": "Claim and Specification Amendments",
            "description": "Itemized before/after comparison of claim amendments throughout prosecution, highlighting added limitations and removed features",
        },
        {
            "heading": "Why the Patent Was Allowed",
            "description": "Analysis of why the patent was ultimately granted — what amendments and arguments the Examiner found persuasive",
        },
        {
            "heading": "Lessons and Insights",
            "description": "Practical takeaways and prosecution strategies derived from this case for patent practitioners",
        },
    ],
}

_REPORT_TITLES = {
    "zh": "美国专利申请 {patent_id} 审查历史分析报告",
    "en": "Prosecution History Analysis Report for U.S. Patent Application {patent_id}",
}


# ── System prompts (bilingual) ────────────────────────────────────────────────

_OUTLINE_SYSTEM_PROMPTS = {
    "zh": (
        "你是一个USPTO专利审查历史分析专家。根据用户问题和已下载的审查文件，"
        "规划一份审查历史分析报告的结构。\n"
        "返回 JSON：{\"title\": \"报告标题\", \"sections\": [{\"heading\": \"章节标题\", \"description\": \"本章内容说明\"}]}\n"
        "章节数 4-7 个，标题简洁。报告必须覆盖以下内容：\n"
        "1. 审查过程时间线\n"
        "2. Office Action 拒绝理由分析\n"
        "3. 申请人答复策略\n"
        "4. Claim 修改对比\n"
        "5. 授权原因\n"
        "6. 经验总结"
    ),
    "en": (
        "You are a USPTO patent prosecution history analysis expert. Based on the user's query "
        "and downloaded prosecution documents, plan the structure of a prosecution history analysis report.\n"
        "Return JSON: {\"title\": \"report title\", \"sections\": [{\"heading\": \"section heading\", \"description\": \"section description\"}]}\n"
        "4-7 sections with concise headings. The report MUST cover:\n"
        "1. Prosecution timeline\n"
        "2. Office Action rejection analysis\n"
        "3. Applicant's response strategy\n"
        "4. Claim amendment comparison\n"
        "5. Reasons for allowance\n"
        "6. Lessons learned"
    ),
}

_SECTION_SYSTEM_PROMPTS = {
    "zh": (
        "你是一个USPTO专利审查历史分析专家。根据给定的审查文件内容，撰写一个报告章节。\n"
        "\n"
        "CRITICAL 写作规则：\n"
        "1. 每个事实性陈述必须注明来源文件。格式：用 **[文件类型: 日期]** 标注来源。\n"
        "   例如：Examiner 认为Claim 1与US9876543组合后缺乏创造性 **[Non-Final Office Action: 2023-03-15]**\n"
        "2. 引用具体的法条依据（§102, §103, §112等）和对比文件的专利号。\n"
        "3. Claim 修改对比必须引用原文并标注修改前后的差异。\n"
        "4. 每个段落至少包含一个来源引用。\n"
        "5. 不要编造信息——只基于提供的审查文件内容。\n"
        "6. 直接输出 Markdown 格式的章节内容，不要输出 JSON。"
    ),
    "en": (
        "You are a USPTO patent prosecution history analysis expert. Write a report section "
        "based on the provided prosecution documents.\n"
        "\n"
        "CRITICAL writing rules:\n"
        "1. Every factual statement MUST cite its source document. Format: **[Document Type: Date]**.\n"
        "   Example: The Examiner rejected Claim 1 as obvious over US9876543 in view of US11223344 **[Non-Final Office Action: 2023-03-15]**\n"
        "2. Cite specific statutory grounds (§102, §103, §112, etc.) and prior art patent numbers.\n"
        "3. Claim amendment comparisons must quote original text and highlight before/after differences.\n"
        "4. Every paragraph must include at least one source citation.\n"
        "5. Do not fabricate information — only use content from the provided prosecution documents.\n"
        "6. Output the section content directly in Markdown format. Do NOT output JSON."
    ),
}


# ── Document analysis helpers ─────────────────────────────────────────────────


def _build_document_context(
    docs_by_category: dict[str, list[Any]],
    max_chars_per_doc: int = 15000,
) -> str:
    """Build a structured context string from downloaded prosecution documents.

    Documents are organized by category with clear headers and separators.
    Each document is truncated to max_chars_per_doc to stay within context limits.
    Documents without extracted text are excluded.

    Args:
        docs_by_category: Dict mapping category name -> list of ProsecutionDoc objects.
        max_chars_per_doc: Maximum characters per individual document.

    Returns:
        Formatted string with all document texts, ready for LLM input.
    """
    # Category display names (bilingual-friendly)
    category_labels = {
        "office_action": "OFFICE ACTION",
        "applicant_response": "APPLICANT RESPONSE",
        "amendment": "AMENDMENT",
        "notice_of_allowance": "NOTICE OF ALLOWANCE",
        "ids": "INFORMATION DISCLOSURE STATEMENT (IDS)",
        "interview_summary": "INTERVIEW SUMMARY",
        "appeal": "APPEAL",
        "rce": "REQUEST FOR CONTINUED EXAMINATION (RCE)",
        "petition": "PETITION",
        "other": "OTHER DOCUMENT",
    }

    parts: list[str] = []
    for category, docs in docs_by_category.items():
        if not docs:
            continue
        label = category_labels.get(category, category.upper())
        for i, doc in enumerate(docs):
            if not doc.text or len(doc.text.strip()) < 50:
                continue
            text = doc.text.strip()
            if len(text) > max_chars_per_doc:
                text = text[:max_chars_per_doc] + "\n\n[... truncated for length ...]"
            parts.append(
                f"=== {label} #{i + 1}: {doc.description} ===\n"
                f"Document Code: {doc.document_code}\n"
                f"Pages: {doc.page_count}\n\n"
                f"{text}"
            )

    if not parts:
        return "（无审查文件内容可用）"  # fallback for no extracted text

    return "\n\n\n".join(parts)


def _truncate_context(context: str, max_chars: int = 200000) -> str:
    """Truncate context to stay within LLM context limits."""
    if len(context) <= max_chars:
        return context
    _logger.warning(
        f"[prosecution] context_truncated — "
        f"original={len(context)}, truncated_to={max_chars}"
    )
    return context[:max_chars] + "\n\n[... remaining documents truncated for length ...]"


# ── Report outline generation ─────────────────────────────────────────────────


async def generate_report_outline(
    document_context: str,
    query: str,
    provider: Any,
    lang: str = "zh",
) -> dict:
    """Generate a dynamic report outline using Flash LLM.

    Args:
        document_context: Concatenated text of all downloaded prosecution documents.
        query: The user's original query.
        provider: LLM provider with a ``complete_json`` method.
        lang: 'zh' or 'en'.

    Returns:
        dict with 'title' and 'sections' keys.
    """
    system_prompt = _OUTLINE_SYSTEM_PROMPTS.get(lang, _OUTLINE_SYSTEM_PROMPTS["en"])
    doc_count = document_context.count("=== ")
    user_content = (
        f"用户问题：{query}\n\n"
        f"已下载审查文件数量：{doc_count}\n\n"
        f"请规划报告结构。"
        if lang == "zh"
        else f"User query: {query}\n\n"
        f"Downloaded prosecution documents: {doc_count}\n\n"
        f"Please plan the report structure."
    )

    try:
        result = await provider.complete_json(system_prompt, user_content)
    except Exception as e:
        _logger.warning(
            f"[prosecution] outline_generation_failed: {e}, using default sections"
        )
        sections = REPORT_SECTIONS.get(lang, REPORT_SECTIONS["en"])
        return {"title": _REPORT_TITLES.get(lang, _REPORT_TITLES["en"]), "sections": sections}

    if not result or not isinstance(result, dict):
        sections = REPORT_SECTIONS.get(lang, REPORT_SECTIONS["en"])
        return {"title": _REPORT_TITLES.get(lang, _REPORT_TITLES["en"]), "sections": sections}

    return result


# ── Report section generation ─────────────────────────────────────────────────


async def generate_report_section(
    section: dict,
    query: str,
    document_context: str,
    provider: Any,
    lang: str = "zh",
) -> str:
    """Write a single report section via streaming LLM call.

    Args:
        section: Dict with 'heading' and 'description'.
        query: The user's original query.
        document_context: Concatenated text of all prosecution documents.
        provider: LLM provider with a ``_get_langchain_llm`` method.
        lang: 'zh' or 'en'.

    Returns:
        Markdown text for this section.
    """
    heading = section.get("heading", "")
    description = section.get("description", "")
    system_prompt = _SECTION_SYSTEM_PROMPTS.get(lang, _SECTION_SYSTEM_PROMPTS["en"])

    user_content = (
        f"用户问题：{query}\n\n"
        f"本章标题：{heading}\n"
        f"本章说明：{description}\n\n"
        f"=== 审查文件内容 ===\n\n"
        f"{_truncate_context(document_context)}\n\n"
        f"请撰写「{heading}」章节内容。要求：\n"
        f"- 每个事实引用来源文件，用 **[文件类型: 日期]** 格式标注\n"
        f"- Markdown 格式，600-1200 字\n"
        f"- 具体有依据，不编造"
        if lang == "zh"
        else f"User query: {query}\n\n"
        f"Section heading: {heading}\n"
        f"Section description: {description}\n\n"
        f"=== Prosecution Document Contents ===\n\n"
        f"{_truncate_context(document_context)}\n\n"
        f"Please write the '{heading}' section. Requirements:\n"
        f"- Cite source documents for every factual claim using **[Document Type]** format\n"
        f"- Markdown format, 600-1200 words\n"
        f"- Specific and evidence-based, do not fabricate"
    )

    try:
        llm = provider._get_langchain_llm(streaming=True)
        messages = [("system", system_prompt), ("human", user_content)]
        chunks = []
        async for chunk in llm.astream(messages):
            if chunk.content:
                chunks.append(chunk.content)
        text = "".join(chunks).strip()
        # Strip reasoning blocks (DeepSeek, MiniMax, Qwen)
        if "</think>" in text:
            text = text[text.rfind("</think>") + len("</think>"):].strip()
        return text or _section_fallback(heading, lang)
    except Exception as e:
        _logger.warning(
            f"[prosecution] section_generation_failed — heading={heading}, error={e}"
        )
        return _section_fallback(heading, lang)


def _section_fallback(heading: str, lang: str) -> str:
    """Fallback message when section generation fails."""
    if lang == "zh":
        return f"（「{heading}」章节生成失败，请重试）"
    return f'(Section "{heading}" generation failed. Please retry.)'


# ── Main report generation ────────────────────────────────────────────────────


async def generate_prosecution_report(
    docs_by_category: dict[str, list[Any]],
    query: str,
    patent_id: str,
    flash_provider: Any,
    pro_provider: Any,
    lang: str = "zh",
) -> str:
    """Generate a comprehensive prosecution history analysis report.

    Two-phase approach:
    1. Flash model generates a dynamic outline
    2. Pro model writes each section with streaming

    Args:
        docs_by_category: Downloaded prosecution docs grouped by category.
        query: The user's original query.
        patent_id: The USPTO application number being analyzed.
        flash_provider: LLM provider for outline generation (Flash tier).
        pro_provider: LLM provider for section writing (Pro tier).
        lang: 'zh' or 'en' — determines output language.

    Returns:
        Complete Markdown report text.
    """
    # Build document context
    document_context = _build_document_context(docs_by_category)
    doc_count = sum(len(docs) for docs in docs_by_category.values())
    docs_with_text = sum(
        1 for docs in docs_by_category.values() for d in docs if d.text
    )

    _logger.info(
        f"[prosecution] report_generation_start — "
        f"patent_id={patent_id}, doc_categories={len(docs_by_category)}, "
        f"total_docs={doc_count}, docs_with_text={docs_with_text}, "
        f"context_chars={len(document_context)}, lang={lang}"
    )

    if docs_with_text == 0:
        if lang == "zh":
            return (
                f"# 专利申请 {patent_id} 审查历史分析报告\n\n"
                f"**警告**：未能从USPTO下载到审查文件。可能原因：\n"
                f"- 该专利申请号不存在或无权访问\n"
                f"- USPTO API 暂时不可用\n"
                f"- 该专利尚未进入审查阶段\n\n"
                f"请检查专利申请号后重试。"
            )
        else:
            return (
                f"# Prosecution History Analysis Report for U.S. Patent Application {patent_id}\n\n"
                f"**Warning**: No prosecution documents could be downloaded from USPTO. Possible reasons:\n"
                f"- The application number does not exist or is not accessible\n"
                f"- USPTO API is temporarily unavailable\n"
                f"- The patent has not yet entered examination\n\n"
                f"Please verify the application number and try again."
            )

    # Phase 1: Generate outline
    _logger.info(f"[prosecution] generating outline — lang={lang}")
    try:
        outline = await generate_report_outline(
            document_context, query, flash_provider, lang
        )
    except Exception as e:
        _logger.warning(f"[prosecution] outline failed: {e}, using defaults")
        sections = REPORT_SECTIONS.get(lang, REPORT_SECTIONS["en"])
        title_template = _REPORT_TITLES.get(lang, _REPORT_TITLES["en"])
        outline = {
            "title": title_template.format(patent_id=patent_id),
            "sections": sections,
        }

    title = outline.get("title", _REPORT_TITLES.get(lang, _REPORT_TITLES["en"]).format(patent_id=patent_id))
    sections = outline.get("sections", REPORT_SECTIONS.get(lang, REPORT_SECTIONS["en"]))

    _logger.info(
        f"[prosecution] outline_ready — title={title[:80]}, sections={len(sections)}"
    )

    # Phase 2: Write each section
    report_parts: list[str] = []
    for idx, section in enumerate(sections):
        heading = section.get("heading", f"Section {idx + 1}")
        _logger.info(
            f"[prosecution] writing_section [{idx + 1}/{len(sections)}] — {heading}"
        )
        try:
            text = await generate_report_section(
                section, query, document_context, pro_provider, lang
            )
        except Exception as e:
            _logger.error(
                f"[prosecution] section [{idx + 1}/{len(sections)}] FAILED: {e}"
            )
            text = _section_fallback(heading, lang)
        report_parts.append(f"## {heading}\n\n{text}")

    report_text = f"# {title}\n\n" + "\n\n".join(report_parts)
    _logger.info(
        f"[prosecution] report_generated — total_chars={len(report_text)}, sections={len(report_parts)}"
    )
    return report_text
