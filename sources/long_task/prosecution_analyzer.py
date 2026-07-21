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
            '- 如果某维度在文件中找不到明确信息，填写"文件中未明确描述"'
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
            '- If a dimension cannot be found, write "Not described in this document"'
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
            "你的任务不是按时间顺序总结文档。你的目标是：\n"
            "从 USPTO 审查历史中提取最有价值的法律和技术洞察，\n"
            "解释该专利为什么最终被授权，以及什么使它与现有技术区分开。\n\n"
            "你的分析应该帮助专利专业人士快速理解：\n"
            "- 核心发明策略\n"
            "- 审查员的驳回逻辑\n"
            "- 申请人的争辩理由\n"
            "- Claim 修改策略\n"
            "- 审查中的风险与机会\n\n"
            "══════════════════════════════════════\n"
            "报告结构要求\n"
            "══════════════════════════════════════\n\n"
            "按以下结构撰写，每个小节用 ### 标题：\n\n"
            "### 1. 核心发明与可专利性要点\n\n"
            "不要写通用专利描述。直接回答：\n"
            "- 该发明真正解决的技术问题是什么？\n"
            "- 真正的技术贡献是什么（不是泛泛的「制造方法」，而是具体的关键创新特征）？\n"
            "- 哪个技术特征最终成为专利获得授权的原因？\n"
            "- 聚焦推动新颖性的特征，而非一般性的专利描述。\n\n"
            "示例（好的写法）：\n"
            "「关键发明概念不在于一般的制造工艺，而在于在金属层和抗蚀层之间的"
            "bonding 区域放置的临时条形掩模（temporary strip mask），"
            "使得 bonding 区域在加工后可以选择性暴露。」\n\n"
            "### 2. 审查员的主要驳回策略\n\n"
            "对每次 Office Action，解释：\n"
            "- 驳回类型：35 USC §102（新颖性）、§103（显而易见性）、§112 等\n"
            "- 引用的对比文件：名称、公开号、被驳回的 Claims\n"
            "- 审查员的推理逻辑：审查员认为哪些特征已被公开？哪些 Claim 限制被认为缺失？\n"
            "- 不要简单复制 Office Action 的语言——解释审查员的逻辑。\n\n"
            "### 3. 申请人的答复策略\n\n"
            "解释：\n"
            "- 申请人提出了什么争辩？\n"
            "- 强调了什么技术区别？\n"
            "- 为什么这些争辩成功克服了驳回？\n"
            "- 聚焦审查策略——申请人不是争辩整个方法都是新的，\n"
            "  而是通过缩小权利要求、增加特定结构关系来创造区别。\n\n"
            "### 4. 授权原因分析\n\n"
            "解释审查员最终为什么允许该专利：\n"
            "- 哪些 Claim 特征变得可以授权？\n"
            "- 解决了哪个现有技术问题？\n"
            "- 哪些限制条件最可能驱动了授权？\n\n"
            "### 5. 专利有效性 / 诉讼洞察\n\n"
            "提供专业洞察：\n"
            "- 潜在的有效性质疑考量：还有哪些漏洞？哪些限制对有效性至关重要？\n"
            "  要挑战该专利需要找到什么样的现有技术？\n"
            "- 侵权/FTO 洞察：哪些 Claim 限制在侵权分析中可能最重要？\n"
            "  审查过程中哪些陈述可能限制 Claim 解释（prosecution history estoppel）？\n\n"
            "══════════════════════════════════════\n"
            "写作规则\n"
            "══════════════════════════════════════\n\n"
            "必须遵守：\n"
            "- 聚焦洞察，而非文档摘要\n"
            "- 解释审查策略\n"
            "- 对比 before/after claim 范围\n"
            "- 突出审查员与申请人之间的争议焦点\n"
            "- 识别导致授权的转折点\n"
            "- 每个关键发现注明来源 **[Office Action]** **[Response]** **[Amendment]**\n\n"
            "禁止：\n"
            "- 不要写「申请人在2012年7月27日提交了Amendment」这样的纯时间线描述，\n"
            "  除非同时解释这次修改为什么重要\n"
            "- 避免长篇按时间顺序的摘要\n"
            "- 不要输出 JSON\n"
            "- 不要使用 --- 分隔线或装饰性符号\n\n"
            "输出风格：\n"
            "报告应该读起来像一位经验丰富的专利律师在向另一位专业人士解释审查历史。\n"
            "优先排序：可专利性策略 > 驳回理由 > Claim 修改 > 审查员与申请人的争辩 > 授权理由\n"
            "第一部分应该包含最高价值的洞察，因为很多用户只读报告开头。\n"
            "全文 1500-2500 字。"
        )
        user_content = (
            f"专利申请号：{patent_id}\n"
            f"用户问题：{query}\n\n"
            f"审查文件分析结果（按时间排列）：\n{data_text}\n\n"
            f"请基于以上审查文件分析结果，撰写「核心审查洞察」章节。\n"
            f"记住：这不是摘要，而是对审查策略、驳回逻辑、Claim 修改和授权原因的深度分析。"
        )
    else:
        system_prompt = (
            "You are an expert US patent attorney and patent prosecution analyst.\n\n"
            "Your task is NOT to summarize documents chronologically. Your goal is to\n"
            "extract the most valuable legal and technical insights from the USPTO\n"
            "prosecution history and explain WHY the patent was granted and WHAT\n"
            "made it different from prior art.\n\n"
            "Your analysis should help patent professionals quickly understand:\n"
            "- The core invention strategy\n"
            "- The examiner's rejection reasoning\n"
            "- The applicant's arguments\n"
            "- The claim amendment strategy\n"
            "- Prosecution risks and opportunities\n\n"
            "══════════════════════════════════════\n"
            "Report Structure\n"
            "══════════════════════════════════════\n\n"
            "Structure your output with ### headings:\n\n"
            "### 1. Core Invention and Patentability Point\n\n"
            "Do NOT write generic patent descriptions. Answer directly:\n"
            "- What problem did the invention really solve?\n"
            "- What was the REAL technical contribution (not vague \"manufacturing method\""
            " but the specific key innovative feature)?\n"
            "- Which technical feature became the reason the patent was allowed?\n"
            "- Focus on the novelty-driving features, not general patent descriptions.\n\n"
            "Example of GOOD writing:\n"
            "\"The key inventive concept was not the general manufacturing process, but the\n"
            "introduction of a temporary strip mask positioned at the bonding area between\n"
            "the metal layer and resist layer, which allowed selective exposure of the\n"
            "bonding area after processing.\"\n\n"
            "### 2. Examiner's Main Rejection Strategy\n\n"
            "For every Office Action, explain:\n"
            "- Rejection type: 35 USC §102 (anticipation), §103 (obviousness), §112, etc.\n"
            "- Prior art references: name, publication number, claims rejected\n"
            "- Examiner's reasoning: What features did the examiner believe were already\n"
            "  disclosed? Which claim limitations were considered missing?\n"
            "- Do NOT simply copy Office Action language. Explain the examiner's LOGIC.\n\n"
            "### 3. Applicant's Response Strategy\n\n"
            "Explain:\n"
            "- What arguments did the applicant make?\n"
            "- What technical distinction was emphasized?\n"
            "- Why did the argument overcome the rejection?\n"
            "- Focus on prosecution strategy — the applicant did not argue the entire\n"
            "  method was new, but narrowed claims by adding specific structural\n"
            "  relationships to create distinction from prior art.\n\n"
            "### 4. Allowance Reason Analysis\n\n"
            "Explain why the examiner finally allowed the patent:\n"
            "- Which claim features became allowable?\n"
            "- Which prior art problem was solved?\n"
            "- What limitations likely drove allowance?\n\n"
            "### 5. Patent Validity / Litigation Insights\n\n"
            "Provide professional insights:\n"
            "- Potential Invalidity Considerations: Are there remaining vulnerabilities?\n"
            "  Which limitations are critical for validity? What prior art would need to\n"
            "  be found to challenge this patent?\n"
            "- Infringement/FTO Insights: Which claim limitations are likely important in\n"
            "  infringement analysis? Which statements made during prosecution may limit\n"
            "  claim interpretation (prosecution history estoppel)?\n\n"
            "══════════════════════════════════════\n"
            "Writing Rules\n"
            "══════════════════════════════════════\n\n"
            "MUST:\n"
            "- Focus on insights, not document summaries\n"
            "- Explain prosecution strategy\n"
            "- Compare before/after claim scope\n"
            "- Highlight examiner-applicant disputes\n"
            "- Identify the turning points that led to allowance\n"
            "- Cite source documents: **[Office Action]** **[Response]** **[Amendment]**\n\n"
            "AVOID:\n"
            "- Do NOT write \"The applicant submitted an Amendment on July 27, 2012\"\n"
            "  unless explaining WHY it mattered\n"
            "- Avoid long chronological summaries\n"
            "- Do NOT output JSON\n"
            "- Do NOT use --- separators or decorative symbols\n\n"
            "Output Style:\n"
            "The report should read like an experienced patent attorney explaining the\n"
            "prosecution history to another professional.\n"
            "Priority: Patentability strategy > Rejection reasons > Claim amendments >\n"
            "Examiner-applicant arguments > Allowance rationale.\n"
            "The first section should contain the highest-value insights because many\n"
            "users only read the beginning of the report.\n"
            "Target 1500-2500 words."
        )
        user_content = (
            f"Patent Application: {patent_id}\n"
            f"User Query: {query}\n\n"
            f"Prosecution Document Analysis Results (chronological):\n{data_text}\n\n"
            f"Write the 'Key Prosecution Insights' section based on the above analysis.\n"
            f"Remember: This is NOT a summary — it is a deep analysis of prosecution\n"
            f"strategy, rejection logic, claim amendments, and allowance reasons."
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
            "注意：「核心审查洞察」章节已经写好放在报告最前面（包含核心发明、审查员驳回策略、"
            "申请人答复策略、授权原因分析、有效性/诉讼洞察），请不要再重复这些内容。\n"
            "本章节列表应该是对核心审查洞察的细化展开。\n\n"
            "返回 JSON：{\"title\": \"报告标题\", \"sections\": [{\"heading\": \"章节标题\", \"description\": \"本章内容说明及与用户问题的关系\"}]}\n"
            "章节数 3-6 个。章节标题要体现与问题的关联，不要用「分析结果」这类通用名。"
        )
        user_content = (
            f"用户问题：{query}\n"
            f"分析维度：{cols_str}\n"
            f"已分析文件数：{row_count}"
            f"{f'（其中 {failed_count} 个分析失败）' if failed_count else ''}\n\n"
            f"请根据用户问题规划报告结构（核心审查洞察已写好，不需要再列）。"
        )
    else:
        system_prompt = (
            "You are a patent prosecution history report architect. "
            "Plan a question-driven report structure based on the user's query and analysis results.\n"
            "First analyze the user's question type (invalidity analysis, prosecution strategy review, "
            "allowance assessment, technology comparison, etc.), then design section headings that "
            "directly serve answering the user's question.\n\n"
            "Note: The 'Key Prosecution Insights' section is already written at the top of the report "
            "(covering core invention, examiner rejection strategy, applicant response strategy, "
            "allowance analysis, and validity/litigation insights). Do NOT duplicate these.\n"
            "The sections you plan should elaborate on the details behind the Key Prosecution Insights.\n\n"
            'Return JSON: {"title": "report title", "sections": [{"heading": "...", "description": "content and relevance to question"}]}\n'
            "3-6 sections. Section headings should reflect connection to the question — avoid generic names."
        )
        user_content = (
            f"User query: {query}\n"
            f"Analysis dimensions: {cols_str}\n"
            f"Documents analyzed: {row_count}"
            f"{f' ({failed_count} failed)' if failed_count else ''}\n\n"
            f"Plan report structure based on the user's question "
            f"(Key Prosecution Insights is already written)."
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
            {"heading": "Claim 修改分析", "description": "每次 Claim 修改前后对比，修改原因，以及修改如何影响可专利性"},
            {"heading": "审查时间线", "description": "重要审查事件时间线，每次事件对审查策略的影响"},
            {"heading": "授权原因分析", "description": "专利最终获得授权的原因——哪些特征变得可以授权，解决了什么现有技术问题"},
            {"heading": "专利有效性 / 诉讼洞察", "description": "潜在无效性考量、对未来无效/侵权/FTO分析的参考价值"},
            {"heading": "经验与启示", "description": "对专利代理实务的参考价值"},
        ]
    else:
        return [
            {"heading": "Claim Amendment Analysis", "description": "Before/after comparison of each important claim amendment, why it was made, and how it affected patentability"},
            {"heading": "Prosecution Timeline", "description": "Key prosecution events with analysis of how each event changed prosecution strategy"},
            {"heading": "Reasons for Allowance", "description": "Why the patent was ultimately allowed — which features became allowable, what prior art problem was solved"},
            {"heading": "Validity & Litigation Insights", "description": "Potential invalidity considerations, implications for future invalidity/infringement/FTO analysis"},
            {"heading": "Lessons & Insights", "description": "Practical takeaways for patent prosecution practice"},
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
            "你的写作应该体现专利律师的专业视角——不只是陈述事实，而是分析策略。\n\n"
            "在撰写前先思考：\n"
            "1. 本章「{heading}」要回答用户问题的哪个方面？\n"
            "2. 哪些审查文件中的事实与此相关？哪些不相关（省略）？\n"
            "3. 如何组织逻辑链：先给出核心结论，然后用审查文件中的具体事实支撑，最后给出针对问题的分析？\n"
            "4. 不只是陈述「X 发生了」，而是「X 发生了，这意味着 Y，对你的问题的影响是 Z」。\n"
            "5. 如果是 Claim 修改分析，要对比修改前后的范围变化，解释修改的原因和策略考量。\n"
            "6. 如果是时间线分析，每个事件都要解释「为什么这个事件重要」。\n\n"
            "用中文，具体有依据。直接输出 Markdown 格式的章节内容，不要输出 JSON。\n\n"
            "CRITICAL 引用规则：\n"
            "1. 报告中提到的每个事实，必须在后面用 **[文件类型]** 标注来源。例如：\n"
            "   - Examiner 认为 Claim 1 相对于 US9876543 缺乏创造性 **[Non-Final Office Action]**\n"
            "   - 申请人将 Claim 1 的范围限缩为 bonding 区域有临时掩模 **[Amendment]**\n"
            "2. 每个段落至少引用 1-2 个来源文件，并解释该事实对审查策略的意义。\n"
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
            f"- 聚焦审查策略洞察，不是简单的事实罗列\n"
            f"- 只选择与用户问题相关的信息\n"
            f"- 每个事实引用来源，用 **[文件类型]** 格式标注\n"
            f"- Markdown 格式，400-800 字\n"
            f"- 包含策略分析和针对问题的结论"
        )
    else:
        system_prompt = (
            "You are a patent prosecution history report writer. "
            "Write a report section that directly serves answering the user's question.\n\n"
            "Your writing should reflect a patent attorney's perspective — not just "
            "stating facts, but analyzing prosecution strategy.\n\n"
            "Before writing, think through:\n"
            "1. What aspect of the user's question does this section '{heading}' address?\n"
            "2. Which prosecution facts are relevant? Which are not (omit)?\n"
            "3. How to organize the logic chain: core finding → supporting evidence from "
            "prosecution documents → strategic analysis → implication?\n"
            "4. Don't just state \"X happened\" — explain \"X happened, which means Y, "
            "and the prosecution strategy implication is Z.\"\n"
            "5. If analyzing claim amendments, compare before/after scope, explain the "
            "strategic reason for the amendment.\n"
            "6. If presenting a timeline, each event must explain WHY it was strategically important.\n\n"
            "Output Markdown directly, do NOT output JSON.\n\n"
            "CRITICAL citation rules:\n"
            "1. Every factual claim MUST cite its source: **[Document Type]**\n"
            "2. Each paragraph must cite at least 1-2 source documents AND explain "
            "their significance to the prosecution strategy.\n"
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
            f"- Focus on prosecution strategy insights, not just fact listing\n"
            f"- Only include information relevant to the user's question\n"
            f"- Cite sources with **[Document Type]** format\n"
            f"- Markdown, 400-800 words\n"
            f"- Include strategic analysis and question-relevant conclusions"
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
