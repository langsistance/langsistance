"""USPTO prosecution history AI analysis and report generation.

Follows the same pattern as patent_analyzer.py + report_generator.py from the
batch patent analysis pipeline:

  Phase 1:  generate_table_columns()      — Flash LLM determines column headers
  Phase 2a: analyze_single_document()     — Pro LLM fills one table row per doc
  Phase 2b: generate_document_summary()   — 2-3 sentence summary per doc
  Phase 3a: generate_executive_summary()  — concise summary at top of report
  Phase 3b: generate_report_outline()     — dynamic section outline
  Phase 3c: generate_report_section()     — streaming section writing
  Phase 3d: generate_prosecution_report() — orchestrator (returns report + table)
"""

from __future__ import annotations

from typing import Any

from sources.logger import Logger

_logger = Logger("prosecution_analyzer.log")


# ── Bilingual labels ──────────────────────────────────────────────────────────

_REPORT_TITLES = {
    "zh": "美国专利申请 {patent_id} 审查历史分析报告",
    "en": "Prosecution History Analysis Report for U.S. Patent Application {patent_id}",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Table column generation (Flash LLM)
# ═══════════════════════════════════════════════════════════════════════════════


async def generate_table_columns(
    query: str,
    doc_count: int,
    provider: Any,
    lang: str = "zh",
) -> list[str]:
    """Phase 1: Use Flash LLM to generate table column definitions.

    Follows the same pattern as patent_analyzer.generate_table_columns().
    """
    if lang == "zh":
        system_prompt = (
            "你是一个专利审查历史分析专家。根据用户的分析问题，确定分析表格需要哪些列。\n\n"
            "返回 JSON 格式：{\"columns\": [\"列1\", \"列2\", ...]}\n"
            "列数控制在 4-7 列。\n\n"
            "CRITICAL: 以下 3 列每次分析都必须包含：\n"
            '- "文件类型"（必须第一列 — Office Action / Response / Amendment / Notice of Allowance）\n'
            '- "文件描述"（USPTO 文件描述，如 Non-Final Office Action）\n'
            '- "核心内容"（该文件的核心信息摘要 — AI 分析结果）\n\n'
            "根据用户的具体问题，在以上必备列之外增加 1-3 列，例如：\n"
            "- 深度分析：增加\"拒绝理由\"、\"引用的对比文件\"\n"
            "- 策略分析：增加\"申请人的争辩\"、\"Claim修改内容\"\n"
            '- 时间线分析：增加"日期"、\"对后续审查的影响\"\n'
        )
    else:
        system_prompt = (
            "You are a patent prosecution history analysis expert. Based on the user's query, "
            "determine what columns the analysis table needs.\n\n"
            'Return JSON: {"columns": ["col1", "col2", ...]}\n'
            "4-7 columns.\n\n"
            "CRITICAL: These 3 columns MUST be included every time:\n"
            '- "Document Type" (must be first — Office Action / Response / Amendment / Notice of Allowance)\n'
            '- "Description" (USPTO document description, e.g. Non-Final Office Action)\n'
            '- "Key Content" (AI-generated summary of the document)\n\n'
            "Add 1-3 additional columns based on the user's query, e.g.:\n"
            '- "Rejection Grounds", "Cited References"\n'
            '- "Applicant Arguments", "Claim Amendments"\n'
            '- "Date", "Impact on Prosecution"\n'
        )

    user_content = (
        f"用户问题：{query}\n审查文件数量：{doc_count}\n请确定分析表格的列定义。"
        if lang == "zh"
        else f"User query: {query}\nDocument count: {doc_count}\nDetermine table columns."
    )

    try:
        result = await provider.complete_json(system_prompt, user_content)
    except Exception as e:
        _logger.warning(f"[prosecution] column_generation_failed: {e}")
        result = {}

    default_cols = (
        ["文件类型", "文件描述", "核心内容"]
        if lang == "zh"
        else ["Document Type", "Description", "Key Content"]
    )
    return result.get("columns", default_cols) if isinstance(result, dict) else default_cols


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2a: Per-document analysis (Pro LLM)
# ═══════════════════════════════════════════════════════════════════════════════


async def analyze_single_document(
    doc_text: str,
    doc_code: str,
    doc_desc: str,
    doc_category: str,
    columns: list[str],
    query: str,
    provider: Any,
    lang: str = "zh",
) -> dict:
    """Phase 2a: Analyze one prosecution document → table row dict.

    Follows the same pattern as patent_analyzer.analyze_single_patent().
    """
    # Skip the first column (文件类型/Document Type) — filled from metadata
    non_first_columns = columns[1:] if len(columns) > 1 else columns
    col_keys = "\n".join(f'  "{c}": "..."' for c in non_first_columns)

    _col_list = "\n".join(f"- {c}" for c in non_first_columns)

    if lang == "zh":
        system_prompt = (
            "你是一个USPTO专利审查文件分析专家。根据以下维度分析这份审查文件：\n\n"
            + _col_list + "\n\n"
            "返回 JSON，**CRITICAL: JSON 的 key 必须严格使用以下列名，一个不能多一个不能少：**\n"
            "{\n"
            '  "file_type": "' + doc_category + '",\n'
            + col_keys + "\n"
            "}\n\n"
            "分析要求：\n"
            "- 基于审查文件原文，不要编造内容\n"
            "- 核心内容：用2-4句话概括该文件的核心信息\n"
            "- 如果涉及拒绝理由，必须引用具体法条（§102/§103/§112等）和对比文件专利号\n"
            "- 如果涉及Claim修改，必须引用原文中的修改前后差异\n"
            "- 每个维度2-4句话，具体有依据\n"
            '- 如果某维度在文件中找不到明确信息，填写"文件中未明确描述"'
        )
        user_content = (
            f"用户问题：{query}\n\n"
            f"文件类型：{doc_category}\n"
            f"文件代码：{doc_code}\n"
            f"文件描述：{doc_desc}\n\n"
            f"审查文件内容：\n{doc_text[:12000]}\n\n"
            f"请按维度分析并返回 JSON。"
        )
    else:
        system_prompt = (
            "You are a USPTO patent prosecution document analysis expert. "
            "Analyze this document according to the following dimensions:\n\n"
            + _col_list + "\n\n"
            "Return JSON — **CRITICAL: use EXACTLY these keys, no more, no less:**\n"
            "{\n"
            '  "file_type": "' + doc_category + '",\n'
            + col_keys + "\n"
            "}\n\n"
            "Requirements:\n"
            "- Base analysis on the document text, do not fabricate\n"
            "- Key Content: 2-4 sentence summary of core information\n"
            "- Cite specific statutory grounds (§102/§103/§112) and patent numbers for rejections\n"
            "- Quote original language when describing claim amendments\n"
            "- 2-4 sentences per dimension, specific and evidence-based\n"
            '- If a dimension cannot be found, write "Not described in this document"'
        )
        user_content = (
            f"User query: {query}\n\n"
            f"Document Type: {doc_category}\n"
            f"Document Code: {doc_code}\n"
            f"Document Description: {doc_desc}\n\n"
            f"Document Content:\n{doc_text[:12000]}\n\n"
            f"Analyze and return JSON."
        )

    try:
        result = await provider.complete_json(system_prompt, user_content)
    except Exception as e:
        _logger.warning(
            f"[prosecution] analyze_failed — code={doc_code}, error={e}"
        )
        return build_failed_row(doc_code, f"analysis error: {e}", columns, lang)

    if not isinstance(result, dict):
        return build_failed_row(doc_code, "LLM returned non-dict", columns, lang)

    # Ensure first column is filled from metadata
    first_col = columns[0] if columns else "文件类型"
    if first_col not in result:
        cat_labels = {
            "office_action": "Office Action",
            "applicant_response": "Applicant Response",
            "amendment": "Amendment",
            "notice_of_allowance": "Notice of Allowance",
            "ids": "IDS",
            "interview_summary": "Interview Summary",
            "appeal": "Appeal",
            "rce": "RCE",
        }
        result[first_col] = cat_labels.get(doc_category, doc_category)

    # Fill missing columns
    for col in columns:
        if col not in result:
            result[col] = (
                "文件中未明确描述"
                if lang == "zh"
                else "Not described in this document"
            )

    return result


def build_failed_row(
    doc_code: str,
    reason: str,
    columns: list[str],
    lang: str = "zh",
) -> dict:
    """Build a table row for a document whose analysis failed."""
    first_col = columns[0] if columns else "文件类型"
    fail_label = "分析失败" if lang == "zh" else "Analysis Failed"
    row = {first_col: fail_label}
    for col in columns[1:]:
        if col == (columns[1] if len(columns) > 1 else ""):
            row[col] = doc_code
        elif col == (columns[2] if len(columns) > 2 else ""):
            row[col] = reason
        else:
            row[col] = "—"
    return row


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2b: Per-document summary (Pro LLM, streaming)
# ═══════════════════════════════════════════════════════════════════════════════


async def generate_document_summary(
    doc_text: str,
    row: dict,
    query: str,
    provider: Any,
    lang: str = "zh",
) -> str:
    """Phase 2b: Generate 2-3 sentence summary of a prosecution document.

    Follows the same pattern as patent_analyzer.generate_patent_summary().
    """
    if lang == "zh":
        system_prompt = (
            "你是一个USPTO专利审查专家。用2-3句话总结这份审查文件的核心内容。"
        )
        user_content = (
            f"用户问题：{query}\n\n"
            f"文件内容：\n{doc_text[:8000]}\n\n"
            f"请用中文简要总结（2-3句，100字以内）："
        )
    else:
        system_prompt = (
            "You are a USPTO patent prosecution expert. "
            "Summarize this document in 2-3 sentences."
        )
        user_content = (
            f"User query: {query}\n\n"
            f"Document content:\n{doc_text[:8000]}\n\n"
            f"Summarize in 2-3 sentences (under 100 words):"
        )

    try:
        llm = provider._get_langchain_llm(streaming=True)
        messages = [("system", system_prompt), ("human", user_content)]
        chunks = []
        async for chunk in llm.astream(messages):
            if chunk.content:
                chunks.append(chunk.content)
        text = "".join(chunks).strip()
        if "</think>" in text:
            text = text[text.rfind("</think>") + len("</think>"):].strip()
        return text or (
            f"（{row.get('文件描述', row.get('Description', '?'))} 的摘要生成失败）"
            if lang == "zh"
            else f"(Summary generation failed for {row.get('Description', '?')})"
        )
    except Exception as e:
        _logger.warning(f"[prosecution] summary_failed: {e}")
        return (
            f"（摘要生成失败）" if lang == "zh" else "(Summary generation failed)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3a: Executive summary (Pro LLM)
# ═══════════════════════════════════════════════════════════════════════════════


async def generate_executive_summary(
    table_rows: list[dict],
    columns: list[str],
    query: str,
    patent_id: str,
    provider: Any,
    lang: str = "zh",
    on_chunk: Any | None = None,
) -> str:
    """Phase 3a: Generate a concise executive summary (1-2 pages).

    Placed at the front of the report before detailed sections.
    """
    if not table_rows:
        return (
            "（无分析数据）" if lang == "zh" else "(No analysis data available)"
        )

    # Build compact data summary for the LLM
    entries = []
    for r in table_rows[:30]:
        first_col = columns[0] if columns else "?"
        doc_type = r.get(first_col, "?")
        desc = r.get(columns[1] if len(columns) > 1 else "", "")
        summary = r.get("_summary", "")
        entries.append(f"- {doc_type} | {desc} | {summary}")
    data_text = "\n".join(entries)

    if lang == "zh":
        system_prompt = (
            "你是一个USPTO专利审查分析专家。根据审查文件分析结果，"
            "撰写一份问题驱动的执行摘要（Executive Summary）。\n\n"
            "在开始写作前，先按以下步骤思考：\n"
            "1. 【理解问题】分析用户的问题。是无效性风险分析、设计规避策略、"
            "审查策略复盘、授权前景评估、还是其他目的？\n"
            "2. 【过滤信息】只包含与用户问题相关的审查文件分析结果。"
            "省略与问题无关的通用信息。\n"
            "3. 【逻辑组织】按问题导向组织内容，展示「事实→分析→结论」的逻辑链。"
            "不要只罗列事实，要解释事实之间的因果关系和对问题的含义。\n"
            "4. 【给出建议】针对用户的具体问题，给出可操作的结论和建议。\n\n"
            "CRITICAL 格式要求：\n"
            "- 使用 Markdown 但保持简洁：**粗体**可以用，但禁止使用 --- 分隔线、"
            "禁止使用 ─── 装饰线、禁止使用装饰性符号\n"
            "- 章节标题用 ### 格式\n"
            "- 列表用 - 开头，不要用特殊符号\n"
            "- 800-1500字，结构化呈现\n"
            "- 不要输出JSON\n"
            "- 每个关键发现注明来源文件类型 **[Office Action]** **[Response]**\n\n"
            "注意：这不是通用摘要。每个事实和结论都必须与用户的问题直接相关。"
            "如果某信息与用户的问题无关，就不要写。"
        )
        user_content = (
            f"专利申请号：{patent_id}\n"
            f"用户问题：{query}\n\n"
            f"审查文件分析结果：\n{data_text}\n\n"
            f"请根据用户问题撰写执行摘要（只包含与问题相关的信息）。"
        )
    else:
        system_prompt = (
            "You are a USPTO patent prosecution analysis expert. "
            "Write a question-driven Executive Summary based on the document analysis.\n\n"
            "Before writing, think through these steps:\n"
            "1. 【Understand】Analyze the user's question. Is it about invalidity risk? "
            "Design-around strategy? Prosecution strategy review? Allowance prospects?\n"
            "2. 【Filter】Only include analysis results relevant to the question. "
            "Omit generic information unrelated to the question.\n"
            "3. 【Organize】Build a \"fact → analysis → conclusion\" logic chain. "
            "Don't just list facts — explain causal relationships and implications for the question.\n"
            "4. 【Advise】Give actionable conclusions and recommendations specific to the question.\n\n"
            "CRITICAL formatting rules:\n"
            "- Use Markdown but keep it clean: **bold** is OK, but NO horizontal rules "
            "(---), NO decorative separators, NO special bullet characters\n"
            "- Section headings use ### format\n"
            "- Lists use plain - prefix, not special symbols\n"
            "- 800-1500 words, structured\n"
            "- Do NOT output JSON\n"
            "- Cite source document types for key findings: **[Office Action]** **[Response]**\n\n"
            "Note: This is NOT a generic summary. Every fact and conclusion must be "
            "directly relevant to the user's question. If information is unrelated, omit it."
        )
        user_content = (
            f"Patent Application: {patent_id}\n"
            f"User Query: {query}\n\n"
            f"Document Analysis Results:\n{data_text}\n\n"
            f"Write an Executive Summary focused on the user's question "
            f"(only include question-relevant information)."
        )

    try:
        llm = provider._get_langchain_llm(streaming=True)
        messages = [("system", system_prompt), ("human", user_content)]
        chunks = []
        async for chunk in llm.astream(messages):
            if chunk.content:
                chunks.append(chunk.content)
                if on_chunk:
                    on_chunk("".join(chunks))
        text = "".join(chunks).strip()
        if "</think>" in text:
            text = text[text.rfind("</think>") + len("</think>"):].strip()
            if on_chunk:
                on_chunk(text)
        return text or _fallback_text("executive_summary", lang)
    except Exception as e:
        _logger.warning(f"[prosecution] executive_summary_failed: {e}")
        return _fallback_text("executive_summary", lang)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3b: Report outline (Flash LLM)
# ═══════════════════════════════════════════════════════════════════════════════


async def generate_report_outline(
    table_rows: list[dict],
    columns: list[str],
    query: str,
    provider: Any,
    lang: str = "zh",
) -> dict:
    """Phase 3b: Generate dynamic report outline based on analysis data.

    Follows the same pattern as report_generator.generate_report_outline().
    """
    row_count = len(table_rows)
    cols_str = ", ".join(columns)
    failed_count = sum(1 for r in table_rows if r.get("_failed"))

    if lang == "zh":
        system_prompt = (
            "你是一个专利审查历史分析报告架构师。根据用户问题和分析结果，规划问题驱动的报告结构。\n"
            "先分析用户的问题类型（无效性分析、审查策略复盘、授权前景评估、技术对比等），"
            "然后设计章节标题，使得每个章节直接服务于回答用户的问题。\n\n"
            "返回 JSON：{\"title\": \"报告标题\", \"sections\": [{\"heading\": \"章节标题\", \"description\": \"本章内容说明及与用户问题的关系\"}]}\n"
            "章节数 4-7 个。章节标题要体现与问题的关联，不要用\"分析结果\"这类通用名。"
            "注意：执行摘要已经写好了，不需要再列。"
        )
        user_content = (
            f"用户问题：{query}\n"
            f"分析维度：{cols_str}\n"
            f"已分析文件数：{row_count}"
            f"{f'（其中 {failed_count} 个分析失败）' if failed_count else ''}\n\n"
            f"请根据用户问题规划报告结构。"
        )
    else:
        system_prompt = (
            "You are a patent prosecution history report architect. "
            "Plan a question-driven report structure based on the user's query and analysis results.\n"
            "First analyze the user's question type (invalidity analysis, prosecution strategy review, "
            "allowance assessment, technology comparison, etc.), then design section headings that "
            "directly serve answering the user's question.\n\n"
            'Return JSON: {"title": "report title", "sections": [{"heading": "...", "description": "content and relevance to question"}]}\n'
            "4-7 sections. Section headings should reflect connection to the question — avoid generic names. "
            "Note: Executive Summary is already written."
        )
        user_content = (
            f"User query: {query}\n"
            f"Analysis dimensions: {cols_str}\n"
            f"Documents analyzed: {row_count}"
            f"{f' ({failed_count} failed)' if failed_count else ''}\n\n"
            f"Plan report structure based on the user's question."
        )

    try:
        result = await provider.complete_json(system_prompt, user_content)
    except Exception as e:
        _logger.warning(f"[prosecution] outline_failed: {e}")
        result = {}

    if not isinstance(result, dict) or not result.get("sections"):
        sections = _default_sections(lang)
        title = _REPORT_TITLES.get(lang, _REPORT_TITLES["en"])
        return {"title": title, "sections": sections}

    return result


def _default_sections(lang: str) -> list[dict]:
    """Fallback section list when outline generation fails."""
    if lang == "zh":
        return [
            {"heading": "审查过程总览", "description": "按时间顺序梳理审查过程"},
            {"heading": "审查意见分析", "description": "Examiner 的拒绝理由和依据"},
            {"heading": "申请人答复策略", "description": "申请人的争辩点和策略"},
            {"heading": "Claim 修改对比", "description": "Claim 修改前后对比"},
            {"heading": "授权原因", "description": "专利最终获得授权的原因"},
            {"heading": "经验与启示", "description": "对专利代理实务的参考价值"},
        ]
    else:
        return [
            {"heading": "Prosecution Overview", "description": "Chronological timeline"},
            {"heading": "Office Action Analysis", "description": "Examiner rejections and grounds"},
            {"heading": "Applicant Response Strategy", "description": "Arguments and strategy"},
            {"heading": "Claim Amendments", "description": "Before/after comparison"},
            {"heading": "Reasons for Allowance", "description": "Why the patent was allowed"},
            {"heading": "Lessons & Insights", "description": "Practical takeaways"},
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3c: Report section writing (Pro LLM, streaming)
# ═══════════════════════════════════════════════════════════════════════════════


async def generate_report_section(
    section: dict,
    query: str,
    table_rows: list[dict],
    columns: list[str],
    provider: Any,
    lang: str = "zh",
    on_chunk: Any | None = None,
) -> str:
    """Phase 3c: Write a single report section via streaming.

    Follows the same pattern as report_generator.generate_report_section().
    """
    heading = section.get("heading", "")
    description = section.get("description", "")

    # Build per-document entries for citation
    entries = []
    for r in table_rows[:30]:
        first_col = columns[0] if columns else "?"
        doc_type = r.get(first_col, str(r.get("_failed", "?")))
        if r.get("_failed"):
            entries.append(f"- {doc_type}: 分析失败" if lang == "zh" else f"- {doc_type}: Analysis failed")
            continue
        parts = [f"**[{doc_type}]**"]
        for col in columns[:8]:
            val = str(r.get(col, "")).strip()
            if val and val != "—":
                parts.append(f"  - {col}: {val}")
        if r.get("_summary"):
            parts.append(f"  - 摘要: {r['_summary']}" if lang == "zh" else f"  - Summary: {r['_summary']}")
        entries.append("\n".join(parts))
    data_summary = "\n\n".join(entries)

    if lang == "zh":
        system_prompt = (
            "你是一个专利审查历史分析报告撰写专家。根据分析数据撰写一个报告章节，"
            "该章节必须直接服务于回答用户的问题。\n\n"
            "在撰写前先思考：\n"
            "1. 本章「{heading}」要回答用户问题的哪个方面？\n"
            "2. 哪些审查文件中的事实与此相关？哪些不相关（省略）？\n"
            "3. 如何组织逻辑链：先给出核心结论，然后用事实支撑，最后给出针对问题的分析？\n"
            "4. 不只是陈述「X 发生了」，而是「X 发生了，这意味着 Y，对你的问题的影响是 Z」。\n\n"
            "用中文，具体有依据。直接输出 Markdown 格式的章节内容，不要输出 JSON。\n\n"
            "CRITICAL 引用规则：\n"
            "1. 报告中提到的每个事实，必须在后面用 **[文件类型]** 标注来源。例如：\n"
            "   - Examiner 认为 Claim 1 与 US9876543 组合后缺乏创造性 **[Non-Final Office Action]**\n"
            "2. 每个段落至少引用 1-2 个来源文件，并解释该事实为什么与用户问题相关。\n"
            "3. 不要虚构信息，只引用数据摘要中给出的内容。\n"
            "4. 引用格式统一用 **[]** 包裹文件类型。\n"
            "5. 在结尾给出针对本章主题、且与用户问题相关的结论。"
        )
        user_content = (
            f"用户问题：{query}\n"
            f"本章标题：{heading}\n"
            f"本章说明：{description}\n\n"
            f"各文件分析结果：\n{data_summary}\n\n"
            f"请撰写「{heading}」章节内容。要求：\n"
            f"- 只选择与用户问题相关的信息，不要罗列所有事实\n"
            f"- 每个事实引用来源，用 **[文件类型]** 格式标注\n"
            f"- Markdown 格式，400-800 字\n"
            f"- 包含逻辑分析和针对问题的结论"
        )
    else:
        system_prompt = (
            "You are a patent prosecution history report writer. "
            "Write a report section that directly serves answering the user's question.\n\n"
            "Before writing, think through:\n"
            "1. What aspect of the user's question does this section '{heading}' address?\n"
            "2. Which prosecution facts are relevant? Which are not (omit)?\n"
            "3. How to organize the logic chain: core finding → supporting facts → analysis → implication?\n"
            "4. Don't just state \"X happened\" — explain \"X happened, which means Y, "
            "and the implication for your question is Z.\"\n\n"
            "Output Markdown directly, do NOT output JSON.\n\n"
            "CRITICAL citation rules:\n"
            "1. Every factual claim MUST cite its source: **[Document Type]**\n"
            "2. Each paragraph must cite at least 1-2 source documents AND explain relevance to the question.\n"
            "3. Do not fabricate — only use content from the data summary.\n"
            "4. Citation format: **[]** wrapping the document type.\n"
            "5. End with a conclusion relevant to both the section topic and the user's question."
        )
        user_content = (
            f"User query: {query}\n"
            f"Section: {heading}\n"
            f"Description: {description}\n\n"
            f"Analysis results:\n{data_summary}\n\n"
            f"Write the '{heading}' section. Requirements:\n"
            f"- Only include information relevant to the user's question\n"
            f"- Cite sources with **[Document Type]** format\n"
            f"- Markdown, 400-800 words\n"
            f"- Include logical analysis and question-relevant conclusions"
        )

    try:
        llm = provider._get_langchain_llm(streaming=True)
        messages = [("system", system_prompt), ("human", user_content)]
        chunks = []
        async for chunk in llm.astream(messages):
            if chunk.content:
                chunks.append(chunk.content)
                if on_chunk:
                    on_chunk("".join(chunks))
        text = "".join(chunks).strip()
        if "</think>" in text:
            text = text[text.rfind("</think>") + len("</think>"):].strip()
            if on_chunk:
                on_chunk(text)
        return text or _fallback_text("section", lang, heading)
    except Exception as e:
        _logger.warning(f"[prosecution] section_failed — heading={heading}: {e}")
        return _fallback_text("section", lang, heading)


def _fallback_text(kind: str, lang: str, heading: str = "") -> str:
    if lang == "zh":
        if kind == "executive_summary":
            return "（执行摘要生成失败，请查看下方详细分析）"
        if kind == "section":
            return f"（「{heading}」章节生成失败，请重试）"
    else:
        if kind == "executive_summary":
            return "(Executive summary generation failed — see detailed analysis below)"
        if kind == "section":
            return f'(Section "{heading}" generation failed.)'
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3d: Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════════


async def generate_prosecution_report(
    table_rows: list[dict],
    columns: list[str],
    query: str,
    patent_id: str,
    flash_provider: Any,
    pro_provider: Any,
    lang: str = "zh",
    summary_updater: Any | None = None,
) -> str:
    """Generate the full prosecution history report.

    Structure:
      1. Executive Summary (written by Pro LLM)
      2. Detailed sections (outline by Flash, sections by Pro)
      3. Analysis table appended at the end

    Args:
        table_rows: Per-document analysis results.
        columns: Table column headers.
        query: User's original query.
        patent_id: USPTO application number.
        flash_provider: LLM provider for outline (Flash tier).
        pro_provider: LLM provider for writing (Pro tier).
        lang: 'zh' or 'en'.

    Returns:
        Complete Markdown report text.
    """
    title_template = _REPORT_TITLES.get(lang, _REPORT_TITLES["en"])
    title = title_template.format(patent_id=patent_id)
    exec_heading = '执行摘要' if lang == 'zh' else 'Executive Summary'

    def _assemble_report(
        exec_summary: str | None,
        completed_parts: list[str],
        current_heading: str | None = None,
        current_text: str = '',
    ) -> str:
        parts = [f"# {title}\n\n"]
        if exec_summary:
            parts.append(f"## {exec_heading}\n\n{exec_summary}\n\n")
        parts.extend(completed_parts)
        if current_heading and current_text:
            parts.append(f"## {current_heading}\n\n{current_text}")
        return "".join(parts)

    _logger.info(
        f"[prosecution] report_start — patent_id={patent_id}, "
        f"rows={len(table_rows)}, columns={columns}, lang={lang}"
    )

    # ── Executive Summary ──
    _logger.info("[prosecution] generating executive_summary")

    def _exec_chunk(partial: str) -> None:
        if summary_updater:
            summary_updater.push(
                _assemble_report(partial, []),
                step_msg='正在撰写执行摘要...' if lang == 'zh' else 'Writing executive summary...',
            )

    exec_summary = await generate_executive_summary(
        table_rows, columns, query, patent_id, pro_provider, lang,
        on_chunk=_exec_chunk if summary_updater else None,
    )
    if summary_updater:
        summary_updater.push(
            _assemble_report(exec_summary, []),
            progress=78,
            step_msg='正在撰写执行摘要...' if lang == 'zh' else 'Writing executive summary...',
            force=True,
        )

    # ── Report outline ──
    _logger.info("[prosecution] generating outline")
    outline = await generate_report_outline(
        table_rows, columns, query, flash_provider, lang,
    )
    sections = outline.get("sections", _default_sections(lang))

    # ── Write each section ──
    report_parts = []
    for idx, section in enumerate(sections):
        heading = section.get("heading", f"Section {idx + 1}")
        sec_pct = 80 + int((idx + 1) / max(len(sections), 1) * 10)
        step_msg = (
            f'正在撰写：{heading}' if lang == 'zh'
            else f'Writing: {heading}'
        )
        if summary_updater:
            summary_updater.progress = sec_pct
            summary_updater.step_msg = step_msg

        def _section_chunk(partial: str, _heading=heading) -> None:
            if summary_updater:
                summary_updater.push(
                    _assemble_report(
                        exec_summary,
                        report_parts,
                        current_heading=_heading,
                        current_text=partial,
                    ),
                    step_msg=step_msg,
                )

        _logger.info(
            f"[prosecution] section [{idx + 1}/{len(sections)}] — {heading}"
        )
        text = await generate_report_section(
            section, query, table_rows, columns, pro_provider, lang,
            on_chunk=_section_chunk if summary_updater else None,
        )
        section_md = f"## {heading}\n\n{text}"
        report_parts.append(section_md)
        if summary_updater:
            summary_updater.push(
                _assemble_report(exec_summary, report_parts),
                progress=sec_pct,
                step_msg=step_msg,
                force=True,
            )

    # ── Build analysis table ──
    table_md = _build_markdown_table(table_rows, columns, lang)

    # ── Assemble full report ──
    report_text = (
        f"# {title}\n\n"
        f"## {'执行摘要' if lang == 'zh' else 'Executive Summary'}\n\n"
        f"{exec_summary}\n\n"
        + "\n\n".join(report_parts)
        + f"\n\n## {'分析数据表' if lang == 'zh' else 'Analysis Data Table'}\n\n"
        + table_md
    )

    _logger.info(
        f"[prosecution] report_done — total_chars={len(report_text)}, "
        f"sections={len(sections)}"
    )
    return report_text


def _build_markdown_table(
    table_rows: list[dict],
    columns: list[str],
    lang: str = "zh",
) -> str:
    """Build a Markdown table from analysis rows."""
    if not table_rows or not columns:
        return "（无数据）" if lang == "zh" else "(No data)"

    # Header
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"

    # Rows (limit to avoid oversized tables)
    rows = []
    for r in table_rows[:50]:
        cells = []
        for col in columns:
            val = str(r.get(col, "—")).replace("\n", " ").replace("|", "\\|")
            # Truncate long cells
            cells.append(val[:200] + ("..." if len(val) > 200 else ""))
        rows.append("| " + " | ".join(cells) + " |")

    return header + "\n" + sep + "\n" + "\n".join(rows)
