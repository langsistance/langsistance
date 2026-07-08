"""Phase 3: Dynamic report generation — executive summary + outline + section-by-section writing."""

from typing import Any


# ── Fallback sections ──────────────────────────────────────────────────────────

def _default_sections(lang: str = "zh") -> list[dict]:
    """Fallback section list when outline generation fails."""
    if lang == "zh":
        return [
            {"heading": "核心发现", "description": "与用户问题直接相关的核心发现和分析"},
            {"heading": "逐专利详细分析", "description": "与用户问题相关的各专利关键技术对比"},
            {"heading": "综合结论与建议", "description": "基于分析的回答和可操作建议"},
        ]
    else:
        return [
            {"heading": "Key Findings", "description": "Core findings directly relevant to the user's question"},
            {"heading": "Per-Patent Detailed Analysis", "description": "Per-patent technology comparison relevant to the query"},
            {"heading": "Conclusions & Recommendations", "description": "Answer and actionable recommendations"},
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3a: Executive summary (Pro LLM, streaming)
# ═══════════════════════════════════════════════════════════════════════════════


async def generate_executive_summary(
    table_rows: list[dict],
    columns: list[str],
    query: str,
    provider: Any,
    lang: str = "zh",
) -> str:
    """Generate a question-driven executive summary at the top of the batch patent report.

    Unlike the per-patent summaries (2-3 sentences each), this provides an
    overall synthesis focused on the user's specific question.
    """
    if not table_rows:
        return "（无分析数据）" if lang == "zh" else "(No analysis data available)"

    # Build compact data summary for the LLM
    entries = []
    for r in table_rows[:20]:
        pid = r.get("patent_id", r.get("专利号", "?"))
        if r.get("_failed"):
            entries.append(
                f"- {pid}: 分析失败" if lang == "zh" else f"- {pid}: Analysis failed"
            )
            continue
        parts = [f"**[{pid}]**"]
        for col in columns:
            if col in ("patent_id", "专利号"):
                continue
            val = str(r.get(col, "")).strip()
            if val and val != "—":
                parts.append(f"  - {col}: {val}")
        if r.get("_summary"):
            parts.append(
                f"  - 摘要: {r['_summary']}" if lang == "zh"
                else f"  - Summary: {r['_summary']}"
            )
        entries.append("\n".join(parts))
    data_text = "\n\n".join(entries)

    if lang == "zh":
        system_prompt = (
            "你是一个专利分析报告专家。根据对多篇专利的分析结果，"
            "撰写一份问题驱动的执行摘要（Executive Summary）。\n\n"
            "在开始写作前，先按以下步骤思考：\n"
            "1. 【理解问题】分析用户的问题。是专利无效性分析、技术趋势分析、"
            "竞争格局分析、专利规避设计、还是侵权风险评估？\n"
            "2. 【过滤信息】只包含与用户问题相关的专利和分析内容。"
            "省略与问题无关的通用描述。\n"
            "3. 【逻辑组织】按问题导向组织内容，展示「事实→分析→结论」的逻辑链。"
            "不要只罗列各专利的事实，要对比、归纳、分析。\n"
            "4. 【给出建议】针对用户的具体问题，给出可操作的结论和建议。\n\n"
            "CRITICAL 格式要求：\n"
            "- Markdown，**粗体**可用，禁止 --- 分隔线、装饰性符号\n"
            "- 章节标题用 ### 格式，列表用 - 开头\n"
            "- 每个关键发现注明来源 **[专利号]**\n"
            "- 800-1500字，不要输出JSON\n"
            "- 只关注与用户问题相关的信息，省略不相关的内容\n\n"
            "注意：这不是通用摘要。每个事实和结论都必须与用户的问题直接相关。"
        )
        user_content = (
            f"用户问题：{query}\n\n"
            f"各专利分析结果：\n{data_text}\n\n"
            f"请根据用户问题撰写执行摘要（只包含与问题相关的信息）。"
        )
    else:
        system_prompt = (
            "You are a patent analysis report expert. Write a question-driven "
            "Executive Summary based on multi-patent analysis results.\n\n"
            "Before writing, think through these steps:\n"
            "1. 【Understand】Analyze the user's question. Is it about invalidity risk? "
            "Technology trend? Competitive landscape? Design-around? Infringement risk?\n"
            "2. 【Filter】Only include patents and analysis relevant to the question. "
            "Omit generic information.\n"
            "3. 【Organize】Build \"fact → analysis → conclusion\" logic chains. "
            "Don't just list per-patent facts — compare, synthesize, and analyze.\n"
            "4. 【Advise】Give actionable conclusions and recommendations specific to the question.\n\n"
            "CRITICAL formatting:\n"
            "- Markdown, **bold** OK, no --- separators or decorative symbols\n"
            "- Use ### for section headings, - for lists\n"
            "- Cite sources with **[patent_id]**\n"
            "- 800-1500 words, do NOT output JSON\n"
            "- Only include question-relevant information\n\n"
            "Note: This is NOT a generic summary. Every fact and conclusion must be "
            "directly relevant to the user's question."
        )
        user_content = (
            f"User query: {query}\n\n"
            f"Patent analysis results:\n{data_text}\n\n"
            f"Write a question-driven Executive Summary "
            f"(only include question-relevant information)."
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
            "（执行摘要生成失败）" if lang == "zh"
            else "(Executive summary generation failed)"
        )
    except Exception as e:
        from sources.logger import Logger
        Logger("report_generator.log").warning(
            f"[report_generator] executive_summary_failed: {e}"
        )
        return (
            "（执行摘要生成失败）" if lang == "zh"
            else "(Executive summary generation failed)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3b: Report outline (Flash LLM)
# ═══════════════════════════════════════════════════════════════════════════════


async def generate_report_outline(
    query: str,
    columns: list[str],
    table_rows: list[dict],
    provider: Any,
    lang: str = "zh",
) -> dict:
    """Generate a dynamic, question-driven report outline based on user query and data.

    Args:
        query: The user's original analysis question.
        columns: List of column names from the analysis table.
        table_rows: List of row dicts from patent analysis (may include _failed rows).
        provider: LLM provider instance with a ``complete_json`` method.
        lang: 'zh' or 'en'.

    Returns:
        dict with keys ``title`` (str) and ``sections`` (list of dicts with
        ``heading`` and ``description``).
    """
    row_count = len(table_rows)
    cols_str = ", ".join(columns)
    failed_count = sum(1 for r in table_rows if r.get("_failed"))

    if lang == "zh":
        system_prompt = (
            "你是一个专利分析报告架构师。根据用户问题，规划问题驱动的报告结构。\n"
            "先分析用户的问题类型（无效性分析、技术趋势、竞争格局、规避设计、侵权风险等），"
            "然后设计章节标题，使得每个章节直接服务于回答用户的问题。\n\n"
            "返回 JSON：{\"title\": \"报告标题\", \"sections\": [{\"heading\": \"章节标题\", \"description\": \"本章内容说明及与问题的关系\"}]}\n"
            "章节数 3-7 个。章节标题要体现与问题的关联，不要用「分析结果」这类通用名。"
            "注意：执行摘要已经写好了，不需要再列。"
        )
        user_content = (
            f"用户问题：{query}\n"
            f"分析维度：{cols_str}\n"
            f"已分析专利数：{row_count}"
            f"{f'（其中 {failed_count} 个分析失败）' if failed_count else ''}\n\n"
            f"请根据用户问题规划报告结构。"
        )
    else:
        system_prompt = (
            "You are a patent analysis report architect. Plan a question-driven "
            "report structure based on the user's query.\n"
            "First analyze the user's question type (invalidity analysis, technology trend, "
            "competitive landscape, design-around, infringement risk, etc.), then design "
            "section headings that directly serve answering the question.\n\n"
            'Return JSON: {"title": "report title", "sections": [{"heading": "section heading", "description": "content and relevance to question"}]}\n'
            "3-7 sections. Section headings should reflect connection to the question. "
            "Note: Executive Summary is already written."
        )
        user_content = (
            f"User query: {query}\n"
            f"Analysis dimensions: {cols_str}\n"
            f"Patents analyzed: {row_count}"
            f"{f' ({failed_count} failed)' if failed_count else ''}\n\n"
            f"Plan report structure based on the user's question."
        )

    try:
        result = await provider.complete_json(system_prompt, user_content)
    except Exception as e:
        from sources.logger import Logger
        Logger("report_generator.log").warning(
            f"[report_generator] outline_failed: {e}"
        )
        result = {}

    if not isinstance(result, dict) or not result.get("sections"):
        sections = _default_sections(lang)
        title = "专利分析报告" if lang == "zh" else "Patent Analysis Report"
        return {"title": title, "sections": sections}

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3c: Report section writing (Pro LLM, streaming)
# ═══════════════════════════════════════════════════════════════════════════════


async def generate_report_section(
    section: dict,
    query: str,
    columns: list[str],
    table_rows: list[dict],
    provider: Any,
    lang: str = "zh",
) -> str:
    """Write a single report section via streaming LLM call.

    Unlike generate_report_outline, this produces free-form Markdown text,
    so it uses streaming instead of complete_json.
    """
    heading = section.get("heading", "")
    description = section.get("description", "")

    # Build a per-patent summary with all columns for accurate citation.
    # Each patent is prefixed with **[patent_id]** so the LLM can cite it.
    patent_entries: list[str] = []
    for r in table_rows[:20]:
        pid = r.get("patent_id", r.get("专利号", "?"))
        if r.get("_failed"):
            patent_entries.append(
                f"- {pid}: 分析失败" if lang == "zh"
                else f"- {pid}: Analysis failed"
            )
            continue
        parts = [f"**[{pid}]**"]
        for col in columns:
            if col in ("patent_id", "专利号"):
                continue
            val = str(r.get(col, "")).strip()
            if val:
                parts.append(f"  - {col}: {val}")
        patent_entries.append("\n".join(parts))
    data_summary = "\n\n".join(patent_entries[:20])

    if lang == "zh":
        system_prompt = (
            "你是一个专利分析报告撰写专家。根据分析数据撰写一个报告章节，"
            "该章节必须直接服务于回答用户的问题。\n\n"
            "在撰写前先思考：\n"
            "1. 本章「{heading}」要回答用户问题的哪个方面？\n"
            "2. 哪些专利的分析结果与此相关？哪些不相关（省略）？\n"
            "3. 如何组织逻辑链：先给出核心结论，然后用各专利的事实支撑，"
            "最后给出针对问题的分析？\n"
            "4. 不只是陈述「某专利有什么特征」，而是「该特征意味着什么，"
            "与用户问题的关联是什么」。\n\n"
            "用中文，具体有依据。直接输出 Markdown 格式的章节内容，不要输出 JSON。\n\n"
            "CRITICAL 引用规则：\n"
            "1. 报告中提到的每一个技术点、技术问题、技术方案、技术效果，都必须在后面"
            "用 **[专利号]** 标注来源专利。例如：\n"
            "   - 采用碳纤维复合材料实现轻量化 **[202310123456.7]**\n"
            "   - 该技术方案被多个专利采用 **[202310123456.7]** **[18331482]**\n"
            "2. 每个段落至少引用 1-2 个专利号，并解释该事实为什么与用户问题相关。"
            "如果一个观点来自多个专利，全部列出。\n"
            "3. 不要虚构专利号，只引用数据摘要中给出的专利号。\n"
            "4. 引用格式统一用 **[]** 包裹专利号，紧跟在被引用的内容后面。\n"
            "5. 在结尾给出针对本章主题、且与用户问题相关的结论。"
        )
        user_content = (
            f"用户问题：{query}\n"
            f"本章标题：{heading}\n"
            f"本章说明：{description}\n\n"
            f"各专利分析结果（每个专利的 **[专利号]** 标注了来源标识，引用时必须使用）：\n"
            f"{data_summary}\n\n"
            f"请撰写「{heading}」章节内容。要求：\n"
            f"- 只选择与用户问题相关的信息，不要罗列所有专利的所有事实\n"
            f"- 每个技术点引用来源专利，用 **[专利号]** 格式标注\n"
            f"- Markdown 格式，400-800 字\n"
            f"- 包含逻辑分析和针对问题的结论，不编造"
        )
    else:
        system_prompt = (
            "You are a patent analysis report writer. Write a report section "
            "that directly serves answering the user's question.\n\n"
            "Before writing, think through:\n"
            "1. What aspect of the user's question does this section '{heading}' address?\n"
            "2. Which patent analysis results are relevant? Which are not (omit)?\n"
            "3. How to organize the logic chain: core finding → per-patent evidence → "
            "analysis → implication for the question?\n"
            "4. Don't just state \"patent X has feature Y\" — explain \"feature Y means Z, "
            "and the implication for the user's question is W.\"\n\n"
            "Output Markdown directly, do NOT output JSON.\n\n"
            "CRITICAL citation rules:\n"
            "1. Every technical point MUST cite its source patent: **[patent_id]**\n"
            "2. Each paragraph must cite at least 1-2 patents AND explain relevance to the question.\n"
            "3. Do not fabricate patent numbers — only cite from the data summary.\n"
            "4. Citation format: **[]** wrapping the patent ID, placed right after the cited content.\n"
            "5. End with a conclusion relevant to both the section topic and the user's question."
        )
        user_content = (
            f"User query: {query}\n"
            f"Section heading: {heading}\n"
            f"Section description: {description}\n\n"
            f"Patent analysis results (cite patents using the **[patent_id]** notation shown):\n"
            f"{data_summary}\n\n"
            f"Write the '{heading}' section. Requirements:\n"
            f"- Only include information relevant to the user's question\n"
            f"- Cite source patents with **[patent_id]** format\n"
            f"- Markdown, 400-800 words\n"
            f"- Include logical analysis and question-relevant conclusions"
        )

    # Use streaming to collect free-text Markdown (not JSON)
    llm = provider._get_langchain_llm(streaming=True)
    messages = [("system", system_prompt), ("human", user_content)]
    chunks = []
    async for chunk in llm.astream(messages):
        if chunk.content:
            chunks.append(chunk.content)
    text = "".join(chunks).strip()
    # Strip <think> block if present
    if "</think>" in text:
        text = text[text.rfind("</think>") + len("</think>"):].strip()
    return text or (
        f"（{heading} 生成失败）" if lang == "zh"
        else f"({heading} generation failed)"
    )
