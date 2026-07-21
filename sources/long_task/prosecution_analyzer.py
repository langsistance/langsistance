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

_EXEC_HEADINGS = {
    "zh": "核心审查洞察",
    "en": "Key Prosecution Insights",
}

_ANALYSIS_TABLE_HEADINGS = {
    "zh": "审查文件分析数据表",
    "en": "Prosecution Document Analysis Table",
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
            '- "核心内容"（该文件对审查策略的关键意义 — AI 分析结果）\n\n'
            "根据用户的具体问题，在以上必备列之外增加 1-3 列，例如：\n"
            "- 深度分析：增加\"驳回理由与对比文件\"、\"审查员的推理逻辑\"\n"
            "- 策略分析：增加\"申请人的争辩策略\"、\"Claim修改及策略目的\"\n"
            '- 授权分析：增加"对授权的影响"、\"关键转折点"\n'
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
            '- "Key Content" (strategic significance of this document — AI-generated analysis)\n\n'
            "Add 1-3 additional columns based on the user's query, e.g.:\n"
            '- "Rejection Grounds & References", "Examiner Reasoning"\n'
            '- "Applicant Argument Strategy", "Claim Amendments & Purpose"\n'
            '- "Impact on Allowance", "Key Turning Point"\n'
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
            "分析要求（从专利律师的审查策略视角）：\n"
            "- 基于审查文件原文，不要编造内容\n"
            "- 核心内容：用2-4句话概括该文件透露的审查策略信息——不只是描述文件类型，"
            "而是该文件在整体审查中的战略意义\n"
            "- 如果涉及拒绝理由：必须引用具体法条（§102/§103/§112等）和对比文件专利号，"
            "并解释审查员的推理逻辑（不只是列法条，要分析审查员为什么认为该对比文件覆盖了哪些特征）\n"
            "- 如果涉及Claim修改：必须引用修改前后的差异，并分析修改的策略目的"
            "（不是为了修改而修改，是为了绕开哪个对比文件的哪个特征）\n"
            "- 每个维度2-4句话，具体有依据\n"
            '- 如果某维度在文件中找不到明确信息，填写"文件中未明确描述"\n\n'
            "CRITICAL 法律语气：\n"
            "- 描述审查员行为时使用客观语言，不要推断审查员的主观意图\n"
            "- 使用「审查员认为」「审查员指出」「根据审查记录」\n"
            "- 不要说「审查员认可」「审查员承认」「审查员同意」\n"
            "- 对 Notice of Allowance：只说「审查员未再提出驳回」，不说「审查员认可了修改」"
        )
        user_content = (
            f"用户问题：{query}\n\n"
            f"文件类型：{doc_category}\n"
            f"文件代码：{doc_code}\n"
            f"文件描述：{doc_desc}\n\n"
            f"审查文件内容：\n{doc_text[:12000]}\n\n"
            f"请按维度分析并返回 JSON。聚焦审查策略洞察。"
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
            "Requirements (from a patent attorney's prosecution strategy perspective):\n"
            "- Base analysis on the document text, do not fabricate\n"
            "- Key Content: 2-4 sentence summary of what this document reveals about"
            " prosecution strategy — not just what the document IS, but its strategic"
            " significance in the overall prosecution\n"
            "- For rejections: Cite specific statutory grounds (§102/§103/§112) AND"
            " patent numbers of prior art references. Explain the examiner's REASONING"
            " — not just what was cited, but WHY the examiner believed the reference"
            " disclosed certain features\n"
            "- For claim amendments: Quote the before/after differences AND analyze the"
            " strategic purpose — what prior art feature was being avoided?\n"
            "- 2-4 sentences per dimension, specific and evidence-based\n"
            '- If a dimension cannot be found, write "Not described in this document"\n\n'
            "CRITICAL Legal Tone:\n"
            "- Describe examiner actions objectively — do not infer subjective intent\n"
            '- Use "the examiner found", "the examiner stated", "based on the record"\n'
            '- NEVER say "the examiner admitted", "the examiner agreed", "the examiner conceded"\n'
            '- For Notice of Allowance: say "the examiner did not raise further rejections", '
            'NOT "the examiner approved the amendments"'
        )
        user_content = (
            f"User query: {query}\n\n"
            f"Document Type: {doc_category}\n"
            f"Document Code: {doc_code}\n"
            f"Document Description: {doc_desc}\n\n"
            f"Document Content:\n{doc_text[:12000]}\n\n"
            f"Analyze and return JSON. Focus on prosecution strategy insights."
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
    """Phase 2b: Generate a prosecution-intelligence-focused summary of a document.

    Unlike generic document summarization, each summary should capture what this
    document reveals about the prosecution strategy — rejection logic, argument
    tactics, claim scope changes, or turning points in the examination.
    """
    # Gather row context for richer prompt
    first_col = ""
    extra_context: list[str] = []
    for k, v in row.items():
        if k in ("_failed", "_failure_reason", "_summary"):
            continue
        if not first_col:
            first_col = str(v)  # doc type
        else:
            val = str(v).strip()
            if val and val != "—":
                extra_context.append(f"  - {k}: {val}")

    if lang == "zh":
        system_prompt = (
            "你是一个USPTO专利审查策略分析师。用3-4句话总结这份审查文件，"
            "聚焦于该文件在整体审查策略中的角色和意义。\n\n"
            "不要只描述「这份文件是什么」。要解释：\n"
            "- 如果是 Office Action：审查员的核心驳回逻辑是什么？引用了哪些关键对比文件？\n"
            "- 如果是 Response/Amendment：申请人采用了什么策略？修改了哪些 Claim？为什么有效？\n"
            "- 如果是 Notice of Allowance：哪些特征最终被认可？什么导致了授权？\n\n"
            "直接输出总结文字，不要输出JSON。"
        )
        user_content = (
            f"用户问题：{query}\n\n"
            f"文件类型：{first_col}\n"
            f"文件内容：\n{doc_text[:8000]}\n\n"
            f"请用中文总结该文件在审查策略中的关键意义（3-4句话，150字以内）："
        )
    else:
        system_prompt = (
            "You are a USPTO patent prosecution strategy analyst. "
            "Summarize this document in 3-4 sentences, focusing on its role and "
            "significance in the overall prosecution strategy.\n\n"
            "Don't just describe what the document IS. Explain:\n"
            "- If Office Action: What was the examiner's core rejection logic? "
            "What key prior art references were cited?\n"
            "- If Response/Amendment: What strategy did the applicant use? "
            "What claims were amended? Why was it effective?\n"
            "- If Notice of Allowance: What features were finally accepted? "
            "What drove allowance?\n\n"
            "Output the summary directly, do NOT output JSON."
        )
        user_content = (
            f"User query: {query}\n\n"
            f"Document Type: {first_col}\n"
            f"Document Content:\n{doc_text[:8000]}\n\n"
            f"Summarize the prosecution strategy significance of this document "
            f"(3-4 sentences, under 150 words):"
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
# Phase 3a: Key Prosecution Insights (Pro LLM, streaming)
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
    """Phase 3a: Generate Key Prosecution Insights — a patent attorney-level
    analysis of the prosecution strategy, rejection reasoning, claim amendments,
    and allowance rationale.

    This is NOT a chronological summary. It is a strategic analysis that helps
    patent professionals quickly understand:
      - What the patent really protects (the novelty-driving features)
      - Why the examiner rejected it (the specific logic and prior art)
      - How the applicant overcame rejections (arguments and claim amendments)
      - What limitations ultimately drove allowance
      - Implications for validity, infringement, and FTO analysis
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
        # Include full row data for richer analysis context
        extra_cols = []
        for col in columns[2:]:
            val = str(r.get(col, "")).strip()
            if val and val != "—":
                extra_cols.append(f"    - {col}: {val}")
        extra = "\n".join(extra_cols)
        entries.append(f"- [{doc_type}] {desc}\n  Key Content: {summary}\n{extra}")
    data_text = "\n".join(entries)

    if lang == "zh":
        system_prompt = (
            "你是一位资深的美国专利律师和专利审查策略分析师。\n\n"
            "**全程使用中文撰写，所有小节标题和内容都用中文。**\n\n"
            "你的任务：从 USPTO 审查历史中提取最有价值的法律洞察，\n"
            "用最短的篇幅解释这个专利为什么最终被授权。\n\n"
            "目标用户：专利律师、专利代理人、IP 专业人士\n"
            "他们不需要专利背景介绍，不需要技术领域概述。\n\n"
            "══════════════════════════════════════\n"
            "输出格式：子弹点式关键洞察（不是论文，不是摘要）\n"
            "══════════════════════════════════════\n\n"
            "**全程用中文撰写，包括所有小节标题和子弹点内容。**\n\n"
            "用 ### 标题分 4 个小节，每节 2-5 个子弹点，子弹点用 - 开头：\n\n"
            "### 1. 专利为什么最终获得授权\n\n"
            "直接回答：什么特征让这个专利最终获得授权？\n"
            "- 不是泛泛的「制造方法」，而是具体的、结构性的限制条件\n"
            "- 用 2-3 个子弹点说清楚决定性特征\n"
            "- 对比该特征在对比文件中为什么不存在\n"
            "格式示例：\n"
            "- 决定性特征是 **[具体的结构/方法限制]**，而非一般的 [概念]。"
            "该限制在 **[对比文件]** 中不存在，因为 [原因]。\n\n"
            "### 2. 审查员最有力的驳回\n\n"
            "聚焦审查员最有力的驳回（通常是 Final OA 中的驳回）：\n"
            "- 引用了哪个对比文件？驳回类型（§102/§103）？\n"
            "- 审查员的逻辑：为什么认为该对比文件覆盖了权利要求？\n"
            "- 审查员和申请人之间的核心争议是什么？\n"
            "- 不要说「审查员认可/承认/同意」——Notice of Allowance 通常没有详细理由\n\n"
            "### 3. 改变局面的关键 Claim 修改\n\n"
            "展示关键 Claim 修改：\n"
            "- 修改前：[原始范围]\n"
            "- 修改后：[修改后范围]\n"
            "- 为什么有效：[这个修改为什么绕开了对比文件]\n"
            "- 用 2-4 个子弹点，清晰对比修改前后的范围变化\n\n"
            "### 4. 审查策略洞察\n\n"
            "对无效/FTO/Claim 解释的实用洞察：\n"
            "- 可能的无效挑战方向：[哪些限制是有效的关键？需要什么对比文件才能攻击？]\n"
            "- FTO 考量：[哪些限制在侵权分析中最重要？]\n"
            "- 审查历史禁反言风险：[审查中哪些陈述可能限制 Claim 范围？]\n"
            "- 用 3-5 个子弹点\n\n"
            "══════════════════════════════════════\n"
            "CRITICAL 法律语气规则\n"
            "══════════════════════════════════════\n\n"
            "你的报告可能被用于法律决策。必须遵守以下规则：\n\n"
            "1. 永远不要使用绝对性语言描述审查员意图：\n"
            "   ❌ 「审查员认可了修改后的权利要求」\n"
            "   ❌ 「The examiner admitted...」\n"
            "   ❌ 「审查员承认/同意/接受」\n"
            "   ✅ 「Based on the prosecution record, the most likely reason for allowance was...」\n"
            "   ✅ 「根据审查记录，授权的最可能原因是...」\n"
            "   ✅ 「The allowance indicates the examiner did not find the amended claims unpatentable」\n\n"
            "2. 对有效性分析使用降级语气：\n"
            "   ❌ 「存在显著脆弱点」「该专利容易被攻击」\n"
            "   ✅ 「Potential challenge points include...」\n"
            "   ✅ 「可能的质疑方向包括...」\n"
            "   ✅ 「To challenge this patent, prior art teaching [feature X] would be needed」\n\n"
            "3. 不要推断审查员的主观意图：\n"
            "   ✅ 描述审查记录中的客观事实\n"
            "   ✅ 使用「likely」「may」「based on the record」「suggests」\n"
            "   ❌ 使用「clearly」「undoubtedly」「admitted」「conceded」\n\n"
            "══════════════════════════════════════\n"
            "写作规则\n"
            "══════════════════════════════════════\n\n"
            "- 全文控制在 800-1200 字（不是 1500-2500）\n"
            "- 子弹点格式（bullet points），不是段落散文\n"
            "- 每个子弹点 1-3 句话，简洁有力\n"
            "- 不要重复任何内容，每个事实只说一次\n"
            "- 不要介绍专利背景（用户已经知道这个专利是什么）\n"
            "- 不要使用 --- 分隔线或装饰性符号\n"
            "- 引用来源用 **[OA]** **[Response]** **[Amendment]**\n"
            "- 优先展示：授权原因 > 驳回逻辑 > Claim 修改 > 诉讼洞察\n"
            "- 不要输出 JSON"
        )
        user_content = (
            f"专利申请号：{patent_id}\n"
            f"用户问题：{query}\n\n"
            f"审查文件分析结果（按时间排列）：\n{data_text}\n\n"
            f"请用中文撰写「核心审查洞察」章节。要求：\n"
            f"- 全程中文，包括小节标题\n"
            f"- 子弹点格式（- 开头），不是段落\n"
            f"- 4 个小节：为什么授权 / 最有力驳回 / 关键Claim修改 / 审查策略洞察\n"
            f"- 800-1200 字，不重复，不介绍背景\n"
            f"- 客观语气：用「审查记录显示」「审查员认为」「likely」「suggests」，不说「admitted」「认可」"
        )
    else:
        system_prompt = (
            "You are an expert US patent attorney and patent prosecution analyst.\n\n"
            "**ALL output MUST be in English — headings, bullets, everything.**\n\n"
            "Your task: Extract the most valuable legal insights from the USPTO\n"
            "prosecution history in the most concise format possible. Explain WHY\n"
            "this patent was ultimately allowed.\n\n"
            "Target audience: Patent attorneys, patent agents, IP professionals.\n"
            "They do NOT need patent background or technology field overview.\n\n"
            "══════════════════════════════════════\n"
            "Output Format: Bullet-Point Key Insights (NOT an essay, NOT a summary)\n"
            "══════════════════════════════════════\n\n"
            "Use ### headings for 4 sections, 2-5 bullet points each, using - prefix:\n\n"
            "### 1. Why This Patent Was Finally Allowed\n\n"
            "Answer directly: What feature made this patent allowable?\n"
            "- NOT a vague \"manufacturing method\" — the specific, structural limitation\n"
            "- 2-3 bullets identifying the decisive feature\n"
            "- Contrast with why this feature was absent from prior art\n"
            "Format example:\n"
            "- The decisive feature was **[specific limitation]** — not the general [concept]. "
            "This limitation was absent from **[prior art reference]** because [reason].\n\n"
            "### 2. Examiner's Strongest Rejection\n\n"
            "Focus on the examiner's most forceful rejection (typically from Final OA):\n"
            "- Which prior art reference? Rejection type (§102/§103)?\n"
            "- Examiner's logic: Why did they believe the reference disclosed the claims?\n"
            "- What was the core dispute between examiner and applicant?\n"
            "- NEVER say \"the examiner admitted/agreed/conceded\" — Notices of Allowance "
            "typically lack detailed reasoning\n\n"
            "### 3. The Amendment That Changed Everything\n\n"
            "Show the key claim amendment:\n"
            "- Before: [original scope]\n"
            "- After: [amended scope]\n"
            "- Why it worked: [how this amendment overcame prior art]\n"
            "- 2-4 bullets with clear before/after comparison\n\n"
            "### 4. Key Prosecution Insights\n\n"
            "Practical insights for invalidity/FTO/claim construction:\n"
            "- Potential invalidity challenge points: which limitations are critical for validity? "
            "What prior art would be needed to attack?\n"
            "- FTO considerations: which limitations matter most for infringement?\n"
            "- Prosecution history estoppel risks: which statements may limit claim scope?\n"
            "- 3-5 bullets\n\n"
            "══════════════════════════════════════\n"
            "CRITICAL: Legal Language Rules\n"
            "══════════════════════════════════════\n\n"
            "Your report may be used for legal decisions. Follow these rules strictly:\n\n"
            "1. NEVER use absolute language about examiner intent:\n"
            "   ❌ \"The examiner admitted the amended claims were allowable\"\n"
            "   ❌ \"The examiner agreed / conceded / acknowledged\"\n"
            "   ✅ \"Based on the prosecution record, the most likely reason for allowance was...\"\n"
            "   ✅ \"The allowance indicates the examiner did not find the amended claims unpatentable "
            "over the cited references\"\n\n"
            "2. Downgrade certainty for validity analysis:\n"
            "   ❌ \"This patent has significant vulnerabilities\"\n"
            "   ❌ \"The patent is easily challengeable\"\n"
            "   ✅ \"Potential challenge points include...\"\n"
            "   ✅ \"To challenge this patent, prior art teaching [feature X] would be needed\"\n\n"
            "3. Do NOT infer examiner's subjective intent:\n"
            "   ✅ Describe objective facts from the prosecution record\n"
            "   ✅ Use \"likely\", \"may\", \"based on the record\", \"suggests\"\n"
            "   ❌ Use \"clearly\", \"undoubtedly\", \"admitted\", \"conceded\"\n\n"
            "══════════════════════════════════════\n"
            "Writing Rules\n"
            "══════════════════════════════════════\n\n"
            "- Target 800-1200 words total (NOT 1500-2500)\n"
            "- Bullet-point format (- prefix), NOT paragraph prose\n"
            "- Each bullet 1-3 sentences, concise and impactful\n"
            "- NEVER repeat content — each fact appears once\n"
            "- Do NOT introduce patent background (user already knows the patent)\n"
            "- NO --- separators or decorative symbols\n"
            "- Cite sources as **[OA]** **[Response]** **[Amendment]**\n"
            "- Priority: Why allowed > Rejection logic > Claim amendments > Litigation insights\n"
            "- Do NOT output JSON"
        )
        user_content = (
            f"Patent Application: {patent_id}\n"
            f"User Query: {query}\n\n"
            f"Prosecution Document Analysis (chronological):\n{data_text}\n\n"
            f"Write the 'Key Prosecution Insights' section in English. Requirements:\n"
            f"- ALL content in English (headings, bullets, analysis)\n"
            f"- Bullet-point format (- prefix), not paragraphs\n"
            f"- 4 sections: Why Allowed / Strongest Rejection / Key Amendment / Prosecution Insights\n"
            f"- 800-1200 words, no repetition, no background introduction\n"
            f"- Use 'Based on the record', 'likely', 'suggests' — never 'admitted', 'agreed', 'conceded'"
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
            "你是一个专利审查历史分析报告架构师。根据用户问题和分析结果，规划报告结构。\n"
            "先分析用户的问题类型（无效性分析、审查策略复盘、授权前景评估、技术对比等），"
            "然后设计章节标题，使得每个章节直接服务于回答用户的问题。\n\n"
            "⚠️ 「核心审查洞察」已经写好放在报告最前面（包含：Why Allowed / "
            "Strongest Rejection / Key Amendment / Prosecution Insights）。\n"
            "你规划的章节是对其中具体细节的展开，绝对不要重复核心审查洞察已经覆盖的内容。\n\n"
            "返回 JSON：{\"title\": \"报告标题\", \"sections\": [{\"heading\": \"章节标题\", \"description\": \"本章内容说明及与用户问题的关系\"}]}\n"
            "章节数 3-5 个（精简！不是越多越好）。每个章节要有明确的、独立的目的，"
            "不要设计互相重叠的章节。章节标题要体现与问题的关联，不要用「分析结果」这类通用名。"
        )
        user_content = (
            f"用户问题：{query}\n"
            f"分析维度：{cols_str}\n"
            f"已分析文件数：{row_count}"
            f"{f'（其中 {failed_count} 个分析失败）' if failed_count else ''}\n\n"
            f"请规划报告结构。注意：核心审查洞察已写好，不要重复其内容。每个章节必须有独立目的，不互相重叠。"
        )
    else:
        system_prompt = (
            "You are a patent prosecution history report architect. "
            "Plan a report structure based on the user's query and analysis results.\n"
            "First analyze the user's question type (invalidity analysis, prosecution strategy review, "
            "allowance assessment, technology comparison, etc.), then design section headings that "
            "directly serve answering the user's question.\n\n"
            "⚠️ 'Key Prosecution Insights' is already written at the top (covering: Why Allowed / "
            "Strongest Rejection / Key Amendment / Prosecution Insights).\n"
            "Your sections elaborate on specific details — do NOT duplicate what's already covered.\n\n"
            'Return JSON: {"title": "report title", "sections": [{"heading": "...", "description": "content and relevance to question"}]}\n'
            "3-5 sections (concise! more is not better). Each section must have a clear, independent purpose. "
            "Do NOT design overlapping sections."
        )
        user_content = (
            f"User query: {query}\n"
            f"Analysis dimensions: {cols_str}\n"
            f"Documents analyzed: {row_count}"
            f"{f' ({failed_count} failed)' if failed_count else ''}\n\n"
            f"Plan report structure. Note: Key Prosecution Insights is already written — "
            f"do not duplicate. Each section must have an independent, non-overlapping purpose."
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
            {"heading": "Claim 修改详细分析", "description": "每次重要 Claim 修改的 Before/After 对比、修改策略目的、对授权的影响"},
            {"heading": "审查时间线与关键转折点", "description": "重要审查事件及其策略意义（只写改变审查走向的事件）"},
            {"heading": "对比文件深度对比", "description": "关键 prior art 与授权权利要求的逐特征对比"},
            {"heading": "审查经验与实务启示", "description": "对专利代理实务的参考价值"},
        ]
    else:
        return [
            {"heading": "Claim Amendment Analysis", "description": "Before/After comparison of each important claim amendment, strategic purpose, and impact on allowance"},
            {"heading": "Prosecution Timeline & Turning Points", "description": "Key events that changed prosecution direction (only strategically significant events)"},
            {"heading": "Prior Art Comparison", "description": "Feature-by-feature comparison of key prior art references vs. allowed claims"},
            {"heading": "Practice Takeaways", "description": "Practical insights for patent prosecution practice"},
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
            "1. 本章「{heading}」要回答用户问题的哪个方面？这个方面在「核心审查洞察」里没有重复吗？\n"
            "2. 哪些审查文件中的事实与此相关？哪些不相关（省略）？\n"
            "3. 如何组织逻辑链：核心结论 → 审查文件事实支撑 → 策略分析 → 对问题的含义？\n"
            "4. 不只是陈述「X 发生了」，而是「X 发生了，这意味着 Y，对问题的意义是 Z」。\n\n"
            "用中文，具体有依据。直接输出 Markdown 格式，不要输出 JSON。\n\n"
            "CRITICAL 引用规则：\n"
            "1. 每个事实必须用 **[文件类型]** 标注来源\n"
            "2. 每个段落至少引用 1-2 个来源文件，并解释对审查策略的意义\n"
            "3. 不要虚构信息，只引用数据摘要中给出的内容\n"
            "4. 引用格式统一用 **[]** 包裹文件类型\n"
            "5. 结尾给出针对本章主题的结论\n\n"
            "CRITICAL 法律语气：\n"
            "- 描述审查员行为时使用「审查员认为」「审查记录显示」「根据审查文件」\n"
            "- 永远不要说「审查员认可」「审查员承认」「审查员同意」\n"
            "- 对不确定性使用「likely」「may」「suggests」「based on the record」\n"
            "- 对有效性分析使用「Potential challenge points」「可能的质疑方向」\n\n"
            "CRITICAL 精炼要求：\n"
            "- 本章与其他章节内容绝对不能重复，如果某个事实已在其他章节提到就不要再说\n"
            "- 300-500 字（不是 400-800），用最少的字传达最高的价值\n"
            "- 如果一段话删除后不影响理解，就删除它\n"
            "- 先说最重要的信息，用户可能只读前几段"
        )
        user_content = (
            f"用户问题：{query}\n"
            f"本章标题：{heading}\n"
            f"本章说明：{description}\n\n"
            f"各文件分析结果：\n{data_summary}\n\n"
            f"请撰写「{heading}」章节。要求：\n"
            f"- 聚焦策略洞察，不是事实罗列\n"
            f"- 每个事实引用来源 **[文件类型]**\n"
            f"- Markdown 格式，300-500 字\n"
            f"- 与其他章节不重复，先写最重要的信息\n"
            f"- 客观语气：不说「审查员认可」，用「审查记录显示」「审查员认为」"
        )
    else:
        system_prompt = (
            "You are a patent prosecution history report writer. "
            "Write a report section that directly serves answering the user's question.\n\n"
            "Before writing, think through:\n"
            "1. What aspect does '{heading}' address? Is this NOT already covered in Key Prosecution Insights?\n"
            "2. Which prosecution facts are relevant? Which are not (omit)?\n"
            "3. How to organize: core finding → prosecution evidence → strategic analysis → implication?\n"
            "4. Don't just state \"X happened\" — explain \"X happened, which means Y, "
            "and the implication is Z.\"\n\n"
            "Output Markdown directly, do NOT output JSON.\n\n"
            "CRITICAL citation rules:\n"
            "1. Every factual claim MUST cite its source: **[Document Type]**\n"
            "2. Each paragraph must cite at least 1-2 source documents AND explain strategic significance\n"
            "3. Do not fabricate — only use content from the data summary\n"
            "4. Citation format: **[]** wrapping the document type\n"
            "5. End with a conclusion relevant to the section topic\n\n"
            "CRITICAL Legal Tone:\n"
            '- Describe examiner actions as "the examiner found", "the record shows", "according to"\n'
            '- NEVER say "the examiner admitted", "agreed", "conceded", "approved"\n'
            '- Use "likely", "may", "suggests", "based on the record" for uncertainty\n'
            '- Use "Potential challenge points" for validity analysis, not "significant vulnerabilities"\n\n'
            "CRITICAL Conciseness:\n"
            "- This section MUST NOT repeat content from other sections or from Key Prosecution Insights\n"
            "- 300-500 words (NOT 400-800), maximum value in minimum words\n"
            "- If removing a sentence doesn't change the insight, remove it\n"
            "- Lead with the most important information — users may only read the first few paragraphs"
        )
        user_content = (
            f"User query: {query}\n"
            f"Section: {heading}\n"
            f"Description: {description}\n\n"
            f"Analysis results:\n{data_summary}\n\n"
            f"Write the '{heading}' section. Requirements:\n"
            f"- Focus on strategy insights, not fact listing\n"
            f"- Cite sources with **[Document Type]** format\n"
            f"- Markdown, 300-500 words\n"
            f"- No overlap with other sections or Key Prosecution Insights — lead with most important info\n"
            f"- Objective tone: 'the record shows', 'the examiner found' — never 'admitted', 'agreed'"
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
            return "（核心审查洞察生成失败，请查看下方详细分析）"
        if kind == "section":
            return f"（「{heading}」章节生成失败，请重试）"
    else:
        if kind == "executive_summary":
            return "(Key Prosecution Insights generation failed — see detailed analysis below)"
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
    exec_heading = _EXEC_HEADINGS.get(lang, _EXEC_HEADINGS["en"])
    analysis_table_heading = _ANALYSIS_TABLE_HEADINGS.get(lang, _ANALYSIS_TABLE_HEADINGS["en"])

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

    # ── Key Prosecution Insights ──
    _logger.info("[prosecution] generating key_prosecution_insights")

    def _exec_chunk(partial: str) -> None:
        if summary_updater:
            summary_updater.push(
                _assemble_report(partial, []),
                step_msg='正在撰写核心审查洞察...' if lang == 'zh' else 'Writing Key Prosecution Insights...',
            )

    exec_summary = await generate_executive_summary(
        table_rows, columns, query, patent_id, pro_provider, lang,
        on_chunk=_exec_chunk if summary_updater else None,
    )
    if summary_updater:
        summary_updater.push(
            _assemble_report(exec_summary, []),
            progress=78,
            step_msg='正在撰写核心审查洞察...' if lang == 'zh' else 'Writing Key Prosecution Insights...',
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
        f"## {exec_heading}\n\n"
        f"{exec_summary}\n\n"
        + "\n\n".join(report_parts)
        + f"\n\n## {analysis_table_heading}\n\n"
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
