"""USPTO prosecution history AI analysis and report generation.

Follows the same pattern as patent_analyzer.py + report_generator.py from the
batch patent analysis pipeline:

  Phase 1:  generate_table_columns()      — Flash LLM determines column headers
  Phase 2a: analyze_single_document()     — Pro LLM fills one table row per doc
  Phase 2b: generate_document_summary()   — prosecution strategy summary per doc
  Phase 3a: generate_executive_summary()  — Key Prosecution Insights (bullet-point)
  Phase 3b: generate_report_outline()     — dynamic section outline
  Phase 3c: generate_report_section()     — streaming section writing
  Phase 3d: generate_claim_chart()        — Claim Limitation vs. Prior Art chart
  Phase 3e: generate_prosecution_report() — orchestrator (returns report + table)
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
            "CRITICAL 法律语气（绝对不可违反）：\n"
            "- 🚫 禁止：「审查员认可」「审查员承认」「审查员同意」「审查员接受」「证明」「证实」\n"
            "- ✅ 使用：「审查员认为」「审查员指出」「审查记录显示」「审查员未再提出驳回」\n"
            "- 对 Notice of Allowance：只说「审查员未再基于已引用对比文件提出驳回」，不说「审查员认可了修改」\n"
            "- 不要推断审查员的主观意图，只描述审查记录中的客观事实"
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
            "CRITICAL Legal Tone (DO NOT VIOLATE):\n"
            '- Describe examiner actions objectively — do not infer subjective intent\n'
            '- 🚫 BANNED: "admitted", "agreed", "conceded", "accepted", "approved", "proved"\n'
            '- ✅ Use: "the examiner found", "the examiner stated", "based on the record", "according to the OA"\n'
            '- For Notice of Allowance: "the examiner did not raise further rejections based on the cited references"'
            ' — NEVER "the examiner approved the amendments"'
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
            # ── FIRST: legal boundaries (primacy effect) ──
            "🚨 开始撰写前，请先阅读以下红线。你的输出将用于法律决策。\n\n"
            "### 绝对禁止使用的词汇和表述\n"
            "你绝对不能使用下列任何词汇或含义相同的表述：\n"
            "❌ 「证明」「证实」「彻底瓦解」「绝对」「唯一原因」「毫无疑问」「必然」「确定」\n"
            "❌ 「审查员认可」「审查员接受」「审查员承认」「审查员同意」「审查员确认」\n"
            "❌ 「极大概率无法获得授权」「可以被无效」「存在显著脆弱点」「很容易被攻击」\n"
            "❌ 「不侵权」「一定侵权」「必然落入保护范围」\n\n"
            "### 必须使用的替代表述\n"
            "✅ 「根据审查记录推断，授权的最可能原因是...」\n"
            "✅ 「基于最后一次 OA 撤回情况和 Notice of Allowance，可推断...」\n"
            "✅ 「主要原因」「最可能因素」「潜在因素」「根据审查记录推断」\n"
            "✅ 「Allowance 表明审查员未再基于已引用对比文件提出驳回」\n"
            "✅ 「可能的质疑方向包括...」「潜在挑战点...」\n"
            "✅ 「若产品结构中 [特征X] 与授权权利要求存在差异，该差异可能成为抗辩点」\n\n"
            "核心原则：你是一名分析师，不是法官。只描述客观事实和合理推断，\n"
            "不做确定性法律结论。区分「已确认事实」和「合理推断」。\n\n"
            "──────────────────────────────────────\n\n"
            # ── Role ──
            "你是一位资深的美国专利律师和专利审查策略分析师。\n"
            "全程使用中文撰写。\n\n"
            "任务：从 USPTO 审查历史中提取最有价值的法律洞察，\n"
            "用最短的篇幅解释这个专利为什么最终被授权。\n\n"
            "目标用户：专利律师、专利代理人、IP 专业人士。\n"
            "不需要专利背景介绍，不需要技术领域概述。\n\n"
            # ── Output structure ──
            "用 ### 标题分 4 个小节，每节 2-5 个子弹点（- 开头），不是段落散文：\n\n"
            "### 1. 专利为什么最终获得授权\n"
            "- 不是泛泛的「制造方法」，而是具体的、结构性的限制条件\n"
            "- 2-3 个子弹点说清楚决定性特征\n"
            "- 格式：决定性特征是 **[具体限制]**，该限制在 **[对比文件]** 中不存在，因为 [原因]\n\n"
            "### 2. 审查员最有力的驳回\n"
            "- 引用哪个对比文件？驳回类型（§102/§103）？\n"
            "- 审查员的核心逻辑是什么？\n"
            "- 审查员和申请人之间的核心争议是什么？\n\n"
            "### 3. 改变局面的关键 Claim 修改\n"
            "- 修改前 vs 修改后，清晰对比范围变化\n"
            "- 为什么这个修改绕开了对比文件\n"
            "- 2-4 个子弹点\n\n"
            "### 4. 审查策略洞察\n"
            "- 可能的无效挑战方向（需要什么对比文件才能攻击）\n"
            "- FTO 考量（哪些限制在侵权分析中最重要；如存在结构差异，可能成为抗辩点）\n"
            "- 审查历史禁反言风险\n"
            "- 3-5 个子弹点\n\n"
            # ── END: pre-output checklist (recency effect) ──
            "──────────────────────────────────────\n"
            "⚠️ 输出前自查清单（逐句检查，缺一不可）：\n"
            "□ 是否出现了「证明」「认可」「接受」「绝对」「唯一」「确定」「必然」？如有，删除重写\n"
            "□ 是否出现了「可以被无效」「不侵权」「容易被攻击」？如有，改为「潜在挑战」「可能成为抗辩点」\n"
            "□ 是否描述了审查员的「主观意图」而非「客观记录」？如有，改为「审查记录显示」\n"
            "□ 是否重复了已在前面章节出现的内容？如有，删除\n"
            "□ 每句话删除后用户会丢失关键信息吗？如果不会，删除\n\n"
            "──────────────────────────────────────\n"
            "写作规则：全文 600-900 字（精简！）、子弹点格式、不重复、不介绍背景、"
            "不用 --- 分隔线、引用用 **[OA]** **[Response]** **[Amendment]**、不输出 JSON"
        )
        user_content = (
            f"专利申请号：{patent_id}\n"
            f"用户问题：{query}\n\n"
            f"审查文件分析结果（按时间排列）：\n{data_text}\n\n"
            f"请用中文撰写「核心审查洞察」章节。\n"
            f"⚠️ 输出前逐句检查：不含「证明」「认可」「接受」「绝对」「确定」「必然」。"
        )
    else:
        system_prompt = (
            # ── FIRST: legal boundaries ──
            "🚨 RED LINES — Your output may be used for legal decisions. "
            "READ BEFORE WRITING:\n\n"
            "### BANNED words (never use):\n"
            '❌ "proved", "confirmed", "destroyed", "absolutely", "sole reason", '
            '"without doubt", "determined", "undoubtedly"\n'
            '❌ "the examiner admitted", "agreed", "conceded", "accepted", '
            '"approved the amendment", "acknowledged"\n'
            '❌ "this patent is invalid", "the patent is easily challengeable", '
            '"does not infringe", "significant vulnerabilities"\n\n'
            "### REQUIRED alternatives:\n"
            '✅ "Based on the prosecution record, the most likely reason for allowance was..."\n'
            '✅ "The allowance indicates the examiner did not find the amended claims '
            'unpatentable over the cited references"\n'
            '✅ "primary reason", "most likely factor", "based on the record", "suggests", "may"\n'
            '✅ "Potential challenge points include..."\n'
            '✅ "A product with [feature X] differing from the claimed [limitation Y] '
            'may present a potential argument against literal infringement"\n\n'
            "Core principle: You are an analyst, not a judge. Describe objective facts "
            "and reasonable inferences. Distinguish 'confirmed facts' from 'reasonable inferences'.\n\n"
            "──────────────────────────────────────\n\n"
            # ── Role ──
            "You are an expert US patent attorney and patent prosecution analyst.\n"
            "ALL output in English.\n\n"
            "Task: Extract the most valuable legal insights from the USPTO prosecution "
            "history. Explain WHY this patent was ultimately allowed.\n\n"
            "Target: Patent attorneys, agents, IP professionals. "
            "No patent background, no technology overview.\n\n"
            # ── Structure ──
            "Use ### headings for 4 sections, 2-5 bullets each (- prefix):\n\n"
            "### 1. Why This Patent Was Finally Allowed\n"
            "- The specific, structural limitation that made it allowable — not vague concepts\n"
            "- Why this limitation was absent from prior art\n"
            "- 2-3 bullets\n\n"
            "### 2. Examiner's Strongest Rejection\n"
            "- Prior art reference, rejection type (§102/§103), examiner's logic\n"
            "- Core dispute between examiner and applicant\n"
            "- 2-4 bullets\n\n"
            "### 3. The Amendment That Changed Everything\n"
            "- Before vs. After, why it overcame prior art\n"
            "- 2-4 bullets\n\n"
            "### 4. Key Prosecution Insights\n"
            "- Potential invalidity challenge directions\n"
            "- FTO considerations (if structural differences exist, they may present arguments)\n"
            "- Prosecution history estoppel risks\n"
            "- 3-5 bullets\n\n"
            # ── END: checklist ──
            "──────────────────────────────────────\n"
            "⚠️ PRE-OUTPUT SELF-CHECK (scan every sentence):\n"
            "□ Any banned word? (proved, admitted, agreed, conceded, absolutely, sole, invalid)\n"
            "□ Any examiner intent inference? Change to 'the record shows'\n"
            "□ Any absolute validity/infringement conclusion? Downgrade to 'potential'/'may'\n"
            "□ Any repeated content? Delete duplicates\n"
            "□ Would the user lose critical info without this sentence? If no, delete\n\n"
            "──────────────────────────────────────\n"
            "Writing: 600-900 words, bullet format, no repetition, no background, "
            "no --- separators, cite as **[OA]** **[Response]** **[Amendment]**, no JSON"
        )
        user_content = (
            f"Patent Application: {patent_id}\n"
            f"User Query: {query}\n\n"
            f"Prosecution Document Analysis (chronological):\n{data_text}\n\n"
            f"Write 'Key Prosecution Insights' in English.\n"
            f"⚠️ Before output: scan for 'proved', 'admitted', 'agreed', 'conceded' — replace all."
        )

    try:
        llm = provider._get_langchain_llm(streaming=True)
        _logger.info(
            f"[prosecution] executive_summary streaming — "
            f"provider={getattr(provider, 'provider_name', '?')}, "
            f"model={getattr(provider, 'model', '?')}, "
            f"streaming={True}"
        )
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
            # ── FIRST: legal boundaries ──
            "🚨 红线：禁止使用「证明」「认可」「接受」「承认」「绝对」「确定」「必然」。\n"
            "用「审查记录显示」「主要原因」「最可能因素」「根据审查记录推断」。\n\n"
            # ── Role ──
            "你是专利审查历史分析报告撰写专家。撰写章节「{heading}」。\n\n"
            "撰写前思考：\n"
            "1. 本章要回答什么问题？核心审查洞察里提到过吗？（如果提过，本章只补充细节）\n"
            "2. 哪些审查文件事实相关？哪些不相关（省略）？\n"
            "3. 逻辑链：核心结论 → 审查文件证据 → 策略分析 → 含义\n"
            "4. 不要只说「X 发生了」，要说「X 发生了，意味着 Y，意义是 Z」\n\n"
            "引用规则：每个事实用 **[文件类型]** 标注来源，每段至少 1-2 个引用。\n"
            "结尾给出针对本章主题的结论。直接输出 Markdown，不要 JSON。\n\n"
            # ── END: checklist ──
            "⚠️ 输出前自查：□不含「证明」「认可」「接受」「绝对」 □不重复核心审查洞察已有内容 □250-400 字 □先说最重要的"
        )
        user_content = (
            f"用户问题：{query}\n"
            f"本章标题：{heading}\n"
            f"本章说明：{description}\n\n"
            f"各文件分析结果：\n{data_summary}\n\n"
            f"撰写「{heading}」章节。250-400 字，不重复核心审查洞察已有内容。"
            f"禁止用「证明」「认可」「接受」「承认」「绝对」。"
        )
    else:
        system_prompt = (
            # ── FIRST: legal boundaries ──
            '🚨 RED LINE: NEVER use "proved", "confirmed", "admitted", "agreed", '
            '"conceded", "accepted", "absolutely", "sole reason", "without doubt".\n'
            'Use "the record shows", "primary reason", "most likely factor", '
            '"based on the prosecution record".\n\n'
            # ── Role ──
            "You are a patent prosecution history report writer. "
            "Write section '{heading}'.\n\n"
            "Before writing, check: Is this already covered in Key Prosecution Insights? "
            "If yes, only add NEW detail not already stated.\n"
            "Organize: core finding → prosecution evidence → strategic analysis → implication.\n\n"
            "Citation: Every fact MUST cite its source: **[Document Type]**. "
            "1-2 citations per paragraph. End with a conclusion.\n"
            "Output Markdown directly, no JSON.\n\n"
            # ── END: checklist ──
            "⚠️ Before output, scan every sentence for banned words. "
            "Target 250-400 words. Lead with most important info."
        )
        user_content = (
            f"User query: {query}\n"
            f"Section: {heading}\n"
            f"Description: {description}\n\n"
            f"Analysis results:\n{data_summary}\n\n"
            f"Write '{heading}'. 250-400 words. Don't repeat Key Prosecution Insights. "
            f"NEVER use 'admitted', 'agreed', 'conceded', 'proved', 'accepted'."
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
# Phase 3d: Claim Limitation Analysis Chart (Pro LLM)
# ═══════════════════════════════════════════════════════════════════════════════


async def generate_claim_chart(
    table_rows: list[dict],
    columns: list[str],
    provider: Any,
    lang: str = "zh",
) -> str:
    """Generate a Claim Limitation Analysis Chart — a structured table mapping
    each key claim limitation against prior art references, showing the
    examiner's position and applicant's response.

    This is the single highest-value deliverable for patent attorneys reviewing
    a prosecution history. It answers: "Which limitation saved this patent?"
    """
    entries = []
    for r in table_rows[:30]:
        first_col = columns[0] if columns else "?"
        doc_type = r.get(first_col, "?")
        parts = [f"[{doc_type}]"]
        for col in columns[1:8]:
            val = str(r.get(col, "")).strip()
            if val and val != "—":
                parts.append(f"  {col}: {val}")
        entries.append("\n".join(parts))
    data_text = "\n\n".join(entries)

    if lang == "zh":
        system_prompt = (
            "你是一个USPTO专利审查分析师。根据审查文件分析结果，"
            "生成一份「Claim 限制 vs 对比文件」对照表。\n\n"
            "目标：展示审查员与申请人围绕每个 Claim 限制的攻防博弈，一眼看清谁在哪个点上赢了。\n\n"
            "输出格式：Markdown 表格，包含以下列：\n"
            "| Claim 限制 | [Ref1] | [Ref2] | 审查员观点 | 申请人观点 | 最终结果 |\n"
            "|-----------|--------|--------|-----------|-----------|--------|\n\n"
            "填写规则：\n"
            "- 对比文件列：[Ref1]/[Ref2] 用实际对比文件名称替换。✓=公开 ✗=未公开 ⚡=有争议 —=不适用\n"
            "- 审查员观点：审查员对该限制的立场（如「Kang 图7F 公开了类似掩模」）\n"
            "- 申请人观点：申请人的反驳或修改策略（如「Kang 的掩模在金属层下方，与本发明相反」）\n"
            "- 最终结果：该限制的最终状态（如「申请人胜出」「未被争议」「修改后克服」）\n\n"
            "只列出与授权/驳回直接相关的关键限制（5-10 行），不要列所有限制。\n"
            "最后一行标注 ★，标记最终驱动授权的决定性限制。\n\n"
            "直接输出 Markdown 表格，不要输出JSON，不要加额外解释段落。"
        )
        user_content = (
            f"审查文件分析结果：\n{data_text}\n\n"
            f"请生成审查员 vs 申请人攻防对照表。展示双方对每个关键限制的立场和最终结果。"
        )
    else:
        system_prompt = (
            "You are a USPTO patent prosecution analyst. Based on the prosecution "
            "document analysis, generate a 'Claim Limitation vs. Prior Art' comparison table.\n\n"
            "Goal: Show the adversarial positions — examiner vs. applicant — for each "
            "key claim limitation, so the reader sees who prevailed on which point.\n\n"
            "Output format: Markdown table with these columns:\n"
            "| Claim Limitation | [Ref1] | [Ref2] | Examiner View | Applicant View | Outcome |\n"
            "|-----------------|--------|--------|--------------|----------------|--------|\n\n"
            "Fill rules:\n"
            "- Reference columns: Use actual reference names. ✓=disclosed ✗=not disclosed ⚡=disputed —=N/A\n"
            "- Examiner View: examiner's position on this limitation (e.g. 'Kang Fig.7F shows similar mask')\n"
            "- Applicant View: applicant's rebuttal or amendment strategy (e.g. 'Kang mask is below metal layer, opposite of invention')\n"
            "- Outcome: final status (e.g. 'Applicant prevailed', 'Not disputed', 'Overcome by amendment')\n\n"
            "Only list key limitations directly relevant to allowance/rejection (5-10 rows).\n"
            "Final row marked ★ indicates the decisive limitation that drove allowance.\n\n"
            "Output Markdown table directly, no JSON, no extra explanatory paragraphs."
        )
        user_content = (
            f"Prosecution document analysis:\n{data_text}\n\n"
            f"Generate Examiner vs Applicant adversarial chart. Show both sides' "
            f"positions on each key limitation and the final outcome."
        )

    try:
        llm = provider._get_langchain_llm(streaming=False)
        messages = [("system", system_prompt), ("human", user_content)]
        resp = await llm.ainvoke(messages)
        text = (resp.content or "").strip()
        if "</think>" in text:
            text = text[text.rfind("</think>") + len("</think>"):].strip()
        return text or (
            "（Claim 对照表生成失败）" if lang == "zh"
            else "(Claim chart generation failed)"
        )
    except Exception as e:
        _logger.warning(f"[prosecution] claim_chart_failed: {e}")
        return (
            "（Claim 对照表生成失败）" if lang == "zh"
            else "(Claim chart generation failed)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3e: Main orchestrator
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
    streaming_provider: Any | None = None,
) -> str:
    """Generate the full prosecution history report.

    Structure:
      1. Executive Summary (written by streaming LLM)
      2. Detailed sections (outline by Flash, sections by streaming LLM)
      3. Analysis table appended at the end

    Args:
        table_rows: Per-document analysis results.
        columns: Table column headers.
        query: User's original query.
        patent_id: USPTO application number.
        flash_provider: LLM provider for outline (Flash tier).
        pro_provider: LLM provider for claim chart (Pro tier, non-streaming).
        lang: 'zh' or 'en'.
        streaming_provider: LLM provider for streaming output (summary + sections).
            Falls back to pro_provider if not set.

    Returns:
        Complete Markdown report text.
    """
    _stream = streaming_provider or pro_provider
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
        table_rows, columns, query, patent_id, _stream, lang,
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
            section, query, table_rows, columns, _stream, lang,
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

    # ── Claim Limitation Chart ──
    _logger.info("[prosecution] generating claim_chart")
    claim_chart_heading = 'Claim 限制 vs 对比文件对照表' if lang == 'zh' else 'Claim Limitation vs. Prior Art Chart'
    try:
        claim_chart_md = await generate_claim_chart(
            table_rows, columns, pro_provider, lang,
        )
        if claim_chart_md and '失败' not in claim_chart_md and 'failed' not in claim_chart_md.lower():
            report_parts.append(f"## {claim_chart_heading}\n\n{claim_chart_md}")
    except Exception as e:
        _logger.warning(f"[prosecution] claim_chart skipped: {e}")

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


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Cross-Jurisdiction Family Prosecution Report
# ═══════════════════════════════════════════════════════════════════════════════


async def generate_family_prosecution_report(
    table_rows: list[dict],
    columns: list[str],
    query: str,
    patent_id: str,
    flash_provider: Any,
    pro_provider: Any,
    lang: str = "zh",
    cn_exam_data: dict | None = None,
    jp_exam_data: dict | None = None,
    ep_exam_data: dict | None = None,
    family_overview: dict | None = None,
    summary_updater: Any | None = None,
) -> str:
    """Generate a cross-jurisdiction patent prosecution history analysis report.

    When CN, JP, or EP examination data is available, produces an integrated
    multi-country report following the professional template.  Falls back to
    the US-only ``generate_prosecution_report()`` when no cross-jurisdiction data.

    Report structure (cross-jurisdiction):
      1. Patent Prosecution Overview (title, applicant, jurisdictions, summary)
      2. Cross-Jurisdiction Examination Timeline (unified timeline per office)
      3. Examination Analysis by Jurisdiction (US + CN + JP + EP separately)
      4. Applicant Response and Claim Strategy
      5. Comparative Jurisdiction Analysis
      6. Final Assessment (grant status, complexity, key insights)
      7. Appendix: analysis tables + claim chart
    """
    _has_cn = bool(cn_exam_data and (cn_exam_data.get('events') or cn_exam_data.get('timeline_md')))
    _has_ep = bool(ep_exam_data and (ep_exam_data.get('events') or ep_exam_data.get('timeline_md')))
    _has_jp = bool(jp_exam_data and (jp_exam_data.get('progress') or jp_exam_data.get('timeline_md')))
    _has_ep = bool(ep_exam_data and (ep_exam_data.get('events') or ep_exam_data.get('timeline_md')))

    # ── Fallback to US-only when no cross-jurisdiction data ──
    if not _has_cn and not _has_jp and not _has_ep:
        _logger.info("[prosecution] family_report_fallback — no CN/JP/EP data, using US-only report")
        return await generate_prosecution_report(
            table_rows, columns, query, patent_id,
            flash_provider, pro_provider, lang,
            summary_updater=summary_updater,
        )

    # ── Build data context ──
    jurisdictions = family_overview.get("jurisdictions", ["US"]) if family_overview else ["US"]
    if "CN" not in jurisdictions and cn_exam_data:
        jurisdictions = list(jurisdictions) + ["CN"]
    if "EP" not in jurisdictions and ep_exam_data:
        jurisdictions = list(jurisdictions) + ["EP"]
    if "EP" not in jurisdictions and ep_exam_data:
        jurisdictions = list(jurisdictions) + ["EP"]

    # US document entries for LLM context
    us_entries = []
    for r in table_rows[:30]:
        doc_type = r.get(columns[0], "?") if columns else "?"
        parts = [f"[{doc_type}]"]
        for col in columns[1:8]:
            val = str(r.get(col, "")).strip()
            if val and val != "—" and len(val) < 300:
                parts.append(f"  {col}: {val}")
        if r.get("_summary"):
            parts.append(f"  Summary: {r['_summary']}" if lang != "zh" else f"  摘要: {r['_summary']}")
        us_entries.append("\n".join(parts))
    us_data_text = "\n\n".join(us_entries)

    # CN data context
    cn_parts = []
    if cn_exam_data.get("cn_app_number"):
        cn_parts.append(
            f"CN Application: {cn_exam_data['cn_app_number']}" if lang != "zh"
            else f"中国申请号: {cn_exam_data['cn_app_number']}"
        )
    if cn_exam_data.get("timeline_md"):
        cn_parts.append(f"### CN Examination Timeline\n{cn_exam_data['timeline_md']}")
    if cn_exam_data.get("claims_md"):
        cn_parts.append(f"### CN Claims\n{cn_exam_data['claims_md']}")
    if cn_exam_data.get("legal_md"):
        cn_parts.append(f"### CN Legal Status\n{cn_exam_data['legal_md']}")
    cn_events_list = cn_exam_data.get("events", [])
    if cn_events_list:
        event_lines = []
        for evt in cn_events_list:
            from sources.long_task.china_examination import ExaminationEvent
            if isinstance(evt, ExaminationEvent):
                label = evt.decision_label_zh if lang == "zh" else evt.decision_label_en
                event_lines.append(
                    f"- {label} — {evt.decision_number} ({evt.decision_date})"
                )
            else:
                event_lines.append(f"- {evt}")
        cn_parts.append(f"### CN Review Decisions\n" + "\n".join(event_lines))
    cn_data_text = "\n\n".join(cn_parts) if cn_parts else (
        "No China examination data available." if lang != "zh"
        else "无中国审查数据。"
    )

    # ── JP data context ──
    jp_data_text = ""
    if jp_exam_data:
        jp_app = jp_exam_data.get("jp_app_number", "")
        jp_parts = [f"JP Application: {jp_app}" if lang != "zh" else f"日本申请号: {jp_app}"]
        if jp_exam_data.get("timeline_md"):
            jp_parts.append(jp_exam_data["timeline_md"])
        if jp_exam_data.get("registration_md"):
            jp_parts.append(jp_exam_data["registration_md"])
        if jp_exam_data.get("citations_md"):
            jp_parts.append(jp_exam_data["citations_md"])
        jp_data_text = "\n\n".join(jp_parts)
    if not jp_data_text:
        jp_data_text = (
            "No Japan examination data available." if lang != "zh"
            else "无日本审查数据。"
        )
    # ── EP data context ──
    ep_data_text = ""
    if ep_exam_data:
        ep_app = ep_exam_data.get("ep_app_number", "")
        ep_parts = [f"EP Application: EP{ep_app}" if lang != "zh" else f"欧洲申请号: EP{ep_app}"]
        if ep_exam_data.get("timeline_md"):
            ep_parts.append("### EP Examination Timeline\n" + ep_exam_data['timeline_md'])
        if ep_exam_data.get("search_report_text"):
            sr_text = ep_exam_data["search_report_text"][:3000]
            sr_heading = "### EP Search Opinion\n" if lang != "zh" else "### 欧洲检索意见\n"
            ep_parts.append(sr_heading + sr_text)
        if ep_exam_data.get("status"):
            status = ep_exam_data["status"]
            status_zh = {
                "GRANTED": "已授权", "PENDING": "审查中",
                "REFUSED": "已驳回", "WITHDRAWN": "已撤回",
            }
            status_label = status_zh.get(status, status) if lang == "zh" else status
            ep_parts.append(
                f"Status: {status_label}" if lang != "zh"
                else f"状态: {status_label}"
            )
        ep_data_text = "\n\n".join(ep_parts)
    if not ep_data_text:
        ep_data_text = (
            "No European examination data available." if lang != "zh"
            else "无欧洲审查数据。"
        )

    # ── EP data context ──
    ep_data_text = ""
    if ep_exam_data:
        ep_app = ep_exam_data.get("ep_app_number", "")
        ep_parts = [f"EP Application: EP{ep_app}" if lang != "zh" else f"欧洲申请号: EP{ep_app}"]
        if ep_exam_data.get("timeline_md"):
            ep_parts.append(f"### EP Examination Timeline\n{ep_exam_data['timeline_md']}")
        if ep_exam_data.get("search_report_text"):
            sr_text = ep_exam_data["search_report_text"][:3000]
            ep_parts.append(
                f"### EP Search Opinion\n{sr_text}" if lang != "zh"
                else f"### 欧洲检索意见\n{sr_text}"
            )
        if ep_exam_data.get("status"):
            status = ep_exam_data["status"]
            status_zh = {
                "GRANTED": "已授权", "PENDING": "审查中",
                "REFUSED": "已驳回", "WITHDRAWN": "已撤回",
            }
            status_label = status_zh.get(status, status) if lang == "zh" else status
            ep_parts.append(
                f"Status: {status_label}" if lang != "zh"
                else f"状态: {status_label}"
            )
        ep_data_text = "\n\n".join(ep_parts)
    if not ep_data_text:
        ep_data_text = (
            "No European examination data available." if lang != "zh"
            else "无欧洲审查数据。"
        )

    # Family overview
    family_text = ""
    if family_overview:
        if lang == "zh":
            family_text = (
                f"输入专利: {family_overview.get('input_id', patent_id)}\n"
                f"同族覆盖 {len(jurisdictions)} 个司法辖区: {', '.join(jurisdictions)}\n"
            )
        else:
            family_text = (
                f"Input patent: {family_overview.get('input_id', patent_id)}\n"
                f"Family spans {len(jurisdictions)} jurisdictions: {', '.join(jurisdictions)}\n"
            )

    title_template = {
        "zh": "专利申请 {patent_id} 全球专利审查情报报告",
        "en": "Global Patent Prosecution Intelligence Report for {patent_id}",
    }
    title = title_template.get(lang, title_template["en"]).format(patent_id=patent_id)

    # ── System prompt ────────────────────────────────────────────────────────
    if lang == "zh":
        system_prompt = (
            # ── ROLE ──
            "你是一位资深专利审查**策略**分析师。你的价值不是描述发生了什么，而是揭示为什么和怎么利用。\n"
            "目标读者：专利律师、IP 管理者、企业专利决策者。\n"
            "他们付费想知道的：\n"
            "- 这个专利为什么能授权？哪个 claim 限制救活了它？\n"
            "- 审查员真正卡在哪里？申请人用什么策略突破？\n"
            "- 如果要无效它，攻击哪里？如果要规避，怎么设计？\n"
            "- 如果我要申请类似专利，应该怎么写 claim？\n\n"
            # ── ANALYSIS RATIO ──
            "⚡ 核心原则：事件描述 30% + 策略分析 70%。\n"
            "- 不要逐个文件总结！不要为每个 OA 写 500 字！\n"
            "- 合并同一审查阶段的文件为一个策略事件\n"
            "- 每个事件 = 日期 + 一句话策略意义（不是什么，而是意味着什么）\n"
            "- ⚠️ 只包含有数据的国家！无数据的国家直接跳过，不写空章节。\n"
            "- 如果某国数据为空 → 直接不出现该国章节，不写「无XX审查数据」。\n\n"
            # ── Report structure ──
            "按以下结构输出 Markdown 报告：\n\n"
            "# Executive Strategy Summary\n"
            "律师打开第一页就知道这个专利的核心价值。\n\n"
            "**核心结论**（3-5 句话，不用表格）：\n"
            "- 本专利在各国的审查历程概况\n"
            "- 最终授权的决定性因素（具体到 claim 限制特征）\n"
            "- 审查最严格的国家\n"
            "- 如果只记住一件事，应该是什么\n\n"
            "**核心授权驱动因素**（表格）：\n"
            "| 国家 | 审查难度 | 关键障碍 | 克服方式 | 驱动因素重要度 |\n"
            "|------|---------|---------|---------|-------------|\n"
            "| US | 高/中/低 | 被哪些引用文献卡住 | 修改了哪个具体的 claim 限制 | ⭐⭐⭐⭐⭐ |\n"
            "| CN | 高/中/低 | 实际驳回理由 or ❓ | 如果数据不足就写 ❓ | ⭐评分 |\n\n"
            "# 1. Patent Family Overview\n"
            "表格：\n"
            "| Field | Detail |\n"
            "|-------|--------|\n"
            "| Patent Title | [title] |\n"
            "| Applicant | [name] |\n"
            "| Priority Date | [date] |\n"
            "| Jurisdictions | [list] |\n"
            "| US Status | [status] |\n"
            "| CN Status | [status] |\n"
            "| JP Status | [status] |\n"
            "| EP Status | [status] |\n\n"
            "# 2. Global Prosecution Timeline\n"
            "统一时间线，按日期排列所有国家关键事件。每行 = 日期 + 国家 + 事件 + 策略意义：\n"
            "| Date | Country | Event | Strategic Significance |\n"
            "|------|---------|-------|----------------------|\n"
            "策略意义列：不要写收到OA，写审查员质疑X特征的创造性，引用Ogawa作为最接近现有技术。\n"
            "控制在 15-20 行。\n\n"
            "# 3. US Prosecution Strategy Analysis\n"
            "这是最深入的部分。分两个子章节：\n\n"
            "## 3.1 Rejection Evolution（驳回理由演变）\n"
            "表格追踪每次 OA 的驳回理由变化：\n"
            "| OA | 日期 | §102/§103 驳回 | 引用文献 | 审查员核心论点 | 是否最终克服 |\n"
            "|----|------|---------------|---------|-------------|-----------|\n"
            "不要列每个撤回的驳回，只列有策略意义的。\n\n"
            "## 3.2 Applicant Strategy（申请人应对策略）\n"
            "分析申请人如何回应每次驳回。关键是：\n"
            "- 争辩了什么（区分技术特征 vs 审查员认为的对比文件特征）\n"
            "- 修改了什么（具体 claim 语言的变化，修改前后对比）\n"
            "- 为什么这个策略有效/无效（哪些争辩被接受，哪些被驳回）\n"
            "格式：\n"
            "| Amendment | 日期 | 修改内容 | 策略目的 | 审查员反应 |\n"
            "|-----------|------|---------|---------|----------|\n\n"
            "# 4. Claim Evolution Analysis ⭐\n"
            "这是报告最有商业价值的部分。律师最想看这个。\n\n"
            "## Claim 1 演变轨迹\n"
            "用修改前到修改后的对比格式展示每次重大修改：\n\n"
            "**Original Claim 1:**\n"
            "```\n"
            "[原始 claim 1 的核心限制特征]\n"
            "```\n"
            "问题：[为什么太宽/为什么被驳回]\n\n"
            "**↓ After [Amendment/RCE]**\n"
            "```\n"
            "[新增/修改的限制特征 — 具体 claim 语言]\n"
            "```\n"
            "策略目的：[是为了绕开哪个引用文献的哪个特征]\n\n"
            "**↓ Final Granted**\n"
            "```\n"
            "[最终授权的核心特征组合]\n"
            "```\n\n"
            "**AI 总结：**\n"
            "申请人逐步将保护范围从 [初始宽泛概念] 收敛到 [具体空间/结构关系]。\n"
            "决定性特征是：[具体特征]，这构成了与 [主要引用文献] 的关键区别。\n\n"
            "## Claims Scope Visualization\n"
            "ASCII 流程图：\n"
            "```\n"
            "Initial: [broad concept]\n"
            "    ↓ Amendment 1 — 原因: [why]\n"
            "  [narrowed concept] + [new limitation]\n"
            "    ↓ RCE Amendment — 原因: [why]\n"
            "  [further narrowed] + [structural feature]\n"
            "    ↓ Final — 结果: [outcome]\n"
            "  [final scope — one sentence]\n"
            "```\n\n"
            "# 5. Prior Art Battle Map ⭐\n"
            "可视化谁攻击了什么到申请人怎么防守。\n\n"
            "| Claim 限制 | 对比文献 | 审查员论点 | 申请人回应 | 结果 |\n"
            "|-----------|---------|----------|----------|------|\n"
            "| [具体特征] | [文献号] | [为什么认为覆盖] | [争辩 or 修改] | ✅ 克服 / ❌ 未克服 / ❓ |\n\n"
            "# 6. China Examination Analysis\n"
            "⚠️ 中国数据通常不如美国丰富。不要编造。诚实标注数据缺口。\n\n"
            "## 审查意见通知书分析\n"
            "如果能获取到中国 OA 信息：\n"
            "| OA | 日期 | 驳回理由 | 引用文献 | 审查员观点（创造性三步法） |\n"
            "|----|------|---------|---------|-------------------------|\n"
            "如果没有 OA 全文 → 直接写「❓ 未获取到中国审查意见通知书全文」。\n\n"
            "## 区别技术特征分析\n"
            "如果能获取到答复意见：\n"
            "| 区别技术特征 | 审查员观点 | 申请人回应 | 最终结果 |\n"
            "|------------|----------|----------|---------|\n"
            "如果没有 → 写「❓ 中国审查答复细节未公开」。\n\n"
            "## 中国审查结论\n"
            "根据可用数据总结：授权/驳回/待审 + 置信度。\n\n"
            "# 7. Japan Examination Analysis\n"
            "日本审查数据（如果有），同样用策略视角分析。\n"
            "如果没有实质审查数据，诚实标注并给出基本状态。\n\n"
            "# 8. European Examination Analysis\n"
            "欧洲审查数据（如果有），重点关注检索意见中的初步可专利性评估。\n\n"
            "# 9. Cross-Jurisdiction Comparison\n"
            "对比各国审查差异：\n"
            "| Dimension | US | CN | JP | EP |\n"
            "|-----------|----|----|----|----|\n"
            "| Examination Rigor | [level + 证据] | [level or ❓] | [level or ❓] | [level or ❓] |\n"
            "| Key Rejection Grounds | [grounds] | [grounds or ❓] | [grounds or ❓] | [grounds or ❓] |\n"
            "| Claim Amendments Required | [extent] | [extent or ❓] | [extent or ❓] | [extent or ❓] |\n"
            "| Allowance Driver | [specific feature] | [or ❓] | [or ❓] | [or ❓] |\n"
            "| Data Quality | [level] | [level] | [level] | [level] |\n\n"
            "# 10. Professional Assessment\n"
            "律师视角的最终判断。\n\n"
            "## 10.1 Invalidity Risk Analysis\n"
            "**高风险点：**\n"
            "- Claim X 的 [特征] — 如果找到 [文献类型 + 特征描述] 的组合，可能构成挑战\n"
            "**中等风险点：**\n"
            "- [如果适用]\n"
            "**防御强度：**\n"
            "- [评估授权权利要求的稳固程度]\n\n"
            "## 10.2 FTO（自由实施）考量\n"
            "- 最容易规避的 claim 限制：[特征]\n"
            "- 最难规避的 claim 限制：[特征]\n"
            "- 竞争产品如果 [具体设计差异]，可能规避独立权利要求\n\n"
            "## 10.3 Drafting Lessons（撰写启示）\n"
            "- [从这个案例中，专利代理人应该学到什么]\n"
            "- [早期申请策略的启示]\n"
            "- [Claim 撰写技巧的启示]\n\n"
            # ── CHINA-SPECIFIC ANALYSIS RULES ──
            "⚡ 中国数据分析特别规则：\n"
            "- 中国审查 ≠ 没有信息。中国有审查意见通知书、答复意见，只是不一定能获取到全文\n"
            "- 如果能获取到 OA 信息 → 分析创造性三步法（区别技术特征 + 实际解决的技术问题 + 是否显而易见）\n"
            "- 如果没有 OA 全文 → 诚实标注 ❓，不要跳过中国部分\n"
            "- 中国授权速度快 ≠ 审查简单。可能只是因为审查员没有找到好的对比文件\n"
            "- 禁止在没有数据时说「中国审查员认为……」\n\n"
            # ── CONFIDENCE MARKERS ──
            "🔴 每条分析必须标注置信度。不能只标注一次——每个事实性陈述都要标注：\n"
            "- ✅ Confirmed — 审查文件明确记载（例如 OA 第3页写明…）\n"
            "- 📋 Supported inference — 基于修改内容和审查记录合理推断\n"
            "- ❓ Unknown — 没有数据支持，不要编造\n\n"
            "规则：\n"
            "- 没有看到 Applicant Remarks/Arguments 时 → 不能说「申请人意图/策略是…」→ 标注 ❓\n"
            "- 「基于授权通知及后续无驳回记录，可以推断审查员接受修改后的权利要求，但具体授权理由未在公开文件中明确说明」<—— 这是正确的 Notice of Allowance 写法\n"
            "- 不要写「审查员认可了修改」→ 写「审查记录显示，修改后审查员未再提出驳回」\n"
            "- 中国数据少时，诚实比猜测更有价值\n\n"
            # ── STYLE ──
            "写作风格：\n"
            "- 律师语气：精确、审慎、证据驱动\n"
            "- 每句话要么提供策略洞察，要么删掉\n"
            "- 表格优于段落；具体特征优于抽象描述\n"
            "- 不要「首先」「其次」「此外」「值得注意的是」\n"
            "- 用主动语态，直接陈述\n\n"
            # ── LEGAL BOUNDARIES ──
            "🚨 绝对红线：\n"
            '- 禁止：「证明」「认可」「接受」「承认」「绝对」「确定」「必然」\n'
            '- 禁止：「可以无效」「不侵权」等法律结论\n'
            '- 用：「审查记录显示」「数据表明」「根据可用信息」「可能」「潜在」「可以推断」\n'
            "- Note: Based on the allowance and absence of further rejection, the examiner appears to have accepted...\n"
            '- 不编造审查员的内心想法\n'
            '- 不对授权专利做有效性结论\n'
            '- 不提供法律建议——只提供策略情报\n\n'
            "直接输出 Markdown，不要 JSON。"
        )
    else:
        system_prompt = (
            # ── ROLE ──
            "You are a senior patent prosecution **strategy** analyst. Your value is NOT describing 'what happened' — it's revealing WHY and HOW TO USE IT.\n"
            "Target audience: patent attorneys, IP managers, corporate patent decision-makers.\n"
            "What they pay for:\n"
            "- WHY was this patent granted? Which claim limitation saved it?\n"
            "- WHERE did the examiner push back hardest? How did the applicant break through?\n"
            "- If I want to INVALIDATE it, where do I attack?\n"
            "- If I want to DESIGN AROUND it, how?\n"
            "- If I'm drafting a similar patent, how should I write the claims?\n\n"
            # ── ANALYSIS RATIO ──
            "⚡ CRITICAL: 30% event description + 70% strategy analysis.\n"
            "- Do NOT summarize every document individually!\n"
            "- Merge documents from the same stage into strategy events\n"
            "- Each event = date + one sentence on strategic significance (not 'what happened' but 'what this MEANS')\n"
            "- ⚠️ ONLY include jurisdictions with actual data! Skip countries with no data entirely.\n"
            "- If a jurisdiction has zero data → do NOT create a section for it. No 'No data available' filler.\n\n"
            # ── Report structure ──
            "Output the following Markdown structure:\n\n"
            "# Executive Strategy Summary\n"
            "Lawyer reads page 1 and knows the patent's strategic value.\n\n"
            "**Core Conclusion** (3-5 sentences, no tables):\n"
            "- Overview of prosecution journey across jurisdictions\n"
            "- Decisive factor for allowance (specific claim limitation)\n"
            "- Strictest jurisdiction\n"
            "- The one thing to remember about this patent\n\n"
            "**Allowance Driver Assessment** (table):\n"
            "| Jurisdiction | Difficulty | Key Obstacle | How Overcome | Driver Importance |\n"
            "|-------------|-----------|-------------|-------------|------------------|\n"
            "| US | High/Med/Low | Which reference blocked | Which specific limitation | ⭐⭐⭐⭐⭐ |\n"
            "| CN | High/Med/Low | Actual rejection or ❓ | Use ❓ if data insufficient | ⭐rating |\n\n"
            "# 1. Patent Family Overview\n"
            "Table:\n"
            "| Field | Detail |\n"
            "|-------|--------|\n"
            "| Patent Title | [title] |\n"
            "| Applicant | [name] |\n"
            "| Priority Date | [date] |\n"
            "| Jurisdictions | [list] |\n"
            "| US Status | [status] |\n"
            "| CN Status | [status] |\n"
            "| JP Status | [status] |\n"
            "| EP Status | [status] |\n\n"
            "# 2. Global Prosecution Timeline\n"
            "Unified timeline — one row per key event across all countries:\n"
            "| Date | Country | Event | Strategic Significance |\n"
            "|------|---------|-------|----------------------|\n"
            "Strategic Significance column: NOT 'Received OA' but 'Examiner challenged inventiveness of feature X, citing Ogawa as closest prior art'.\n"
            "15-20 rows max.\n\n"
            "# 3. US Prosecution Strategy Analysis\n"
            "Deepest analysis. Two sub-sections:\n\n"
            "## 3.1 Rejection Evolution\n"
            "Table tracking rejection grounds across OAs:\n"
            "| OA | Date | §102/§103 Grounds | References Cited | Examiner's Core Argument | Ultimately Overcome? |\n"
            "|----|------|-----------------|------------------|------------------------|---------------------|\n"
            "Only include strategically significant rejections.\n\n"
            "## 3.2 Applicant Strategy\n"
            "How the applicant responded to each rejection:\n"
            "- What was argued (distinguishing features vs. reference features)\n"
            "- What was amended (specific claim language before/after)\n"
            "- Why the strategy worked or failed\n"
            "| Amendment | Date | Change Made | Strategic Purpose | Examiner Response |\n"
            "|-----------|------|------------|------------------|-------------------|\n\n"
            "# 4. Claim Evolution Analysis ⭐\n"
            "HIGHEST VALUE section. Attorneys read this first.\n\n"
            "## Claim 1 Evolution\n"
            "Show each major amendment as before-after comparison:\n\n"
            "**Original Claim 1:**\n"
            "```\n"
            "[core limitations of original independent claim]\n"
            "```\n"
            "Problem: [why too broad / why rejected]\n\n"
            "**↓ After [Amendment/RCE]**\n"
            "```\n"
            "[added/modified limitations — exact claim language]\n"
            "```\n"
            "Strategic purpose: [which reference feature was being distinguished]\n\n"
            "**↓ Final Granted**\n"
            "```\n"
            "[final allowed feature combination]\n"
            "```\n\n"
            "**AI Summary:**\n"
            "Applicant progressively narrowed scope from [initial broad concept] to [specific spatial/structural relationship]. The decisive feature was [specific feature], which created the key distinction over [primary reference].\n\n"
            "## Claims Scope Visualization\n"
            "ASCII flow diagram:\n"
            "```\n"
            "Initial: [broad concept]\n"
            "    ↓ Amendment 1 — reason: [why]\n"
            "  [narrowed concept] + [new limitation]\n"
            "    ↓ RCE Amendment — reason: [why]\n"
            "  [further narrowed] + [structural feature]\n"
            "    ↓ Final — outcome: [result]\n"
            "  [final scope — one sentence]\n"
            "```\n\n"
            "# 5. Prior Art Battle Map ⭐\n"
            "Visualize 'who attacked what → how applicant defended'.\n\n"
            "| Claim Limitation | Prior Art | Examiner's Position | Applicant's Response | Result |\n"
            "|-----------------|-----------|--------------------|---------------------|--------|\n"
            "| [specific feature] | [ref #] | [why it was considered to read on] | [argument or amendment] | ✅/❌/❓ |\n\n"
            "# 6. China Examination Analysis\n"
            "⚠️ CN data is typically less detailed than US. DO NOT fabricate. Mark gaps honestly.\n\n"
            "## Office Action Analysis\n"
            "If CN OA info is available:\n"
            "| OA | Date | Rejection Grounds | References | Examiner's View (3-step inventiveness) |\n"
            "|----|------|-----------------|-----------|--------------------------------------|\n"
            "If OA full text is not available → state '❓ CN Office Action full text not available'.\n\n"
            "## Distinguishing Feature Analysis\n"
            "If response info is available:\n"
            "| Distinguishing Feature | Examiner's View | Applicant's Response | Outcome |\n"
            "|----------------------|----------------|---------------------|--------|\n"
            "If not → state '❓ CN response details not publicly available'.\n\n"
            "## China Conclusion\n"
            "Based on available data: granted/refused/pending + confidence level.\n\n"
            "# 7. Japan Examination Analysis\n"
            "JP examination data with strategy lens, or honest data-gap marking.\n\n"
            "# 8. European Examination Analysis\n"
            "EP examination data — focus on search opinion's preliminary patentability assessment.\n\n"
            "# 9. Cross-Jurisdiction Comparison\n"
            "| Dimension | US | CN | JP | EP |\n"
            "|-----------|----|----|----|----|\n"
            "| Examination Rigor | [level + evidence] | [level or ❓] | [level or ❓] | [level or ❓] |\n"
            "| Key Rejection Grounds | [grounds] | [grounds or ❓] | [grounds or ❓] | [grounds or ❓] |\n"
            "| Claim Amendments Required | [extent] | [extent or ❓] | [extent or ❓] | [extent or ❓] |\n"
            "| Allowance Driver | [specific feature] | [or ❓] | [or ❓] | [or ❓] |\n"
            "| Data Quality | [level] | [level] | [level] | [level] |\n\n"
            "# 10. Professional Assessment\n"
            "Attorney-perspective final judgment.\n\n"
            "## 10.1 Invalidity Risk Analysis\n"
            "**High Risk:**\n"
            "- Claim X [feature] — if a reference combining [type + feature] is found, this may be challenged\n"
            "**Moderate Risk:**\n"
            "- [if applicable]\n"
            "**Defense Strength:**\n"
            "- [assessment of how robust the granted claims are]\n\n"
            "## 10.2 FTO Considerations\n"
            "- Easiest claim limitation to design around: [feature]\n"
            "- Hardest claim limitation to design around: [feature]\n"
            "- A competing product that [specific design difference] may avoid the independent claim\n\n"
            "## 10.3 Drafting Lessons\n"
            "- [What patent prosecutors should learn from this case]\n"
            "- [Early filing strategy implications]\n"
            "- [Claim drafting technique takeaways]\n\n"
            # ── CHINA-SPECIFIC ──
            "⚡ China Analysis Rules:\n"
            "- China examination != no information. CN OAs and responses exist but may not be accessible\n"
            "- If OA info is available → analyze the 3-step inventiveness test (distinguishing features + technical problem solved + obviousness)\n"
            "- If OA full text is NOT available → honestly mark ❓, don't skip the CN section\n"
            "- Fast CN grant != easy examination. It may simply mean no good prior art was found\n"
            '- NEVER say "the CN examiner considered..." without evidence\n\n'
            # ── CONFIDENCE MARKERS ──
            "🔴 EVERY factual statement MUST carry a confidence marker:\n"
            "- ✅ Confirmed — explicitly stated in prosecution documents\n"
            "- 📋 Supported inference — reasonably inferred from amendments/record\n"
            "- ❓ Unknown — no data available; do NOT speculate\n\n"
            "Rules:\n"
            "- Without Applicant Remarks/Arguments → NEVER say 'the applicant intended to...' → mark ❓\n"
            '- For Notice of Allowance: "Based on the allowance and absence of further rejection, the examiner appears to have accepted the amended claims, but the specific reasons for allowance are not explicitly stated in the public record."\n'
            '- NEVER: "the examiner acknowledged/accepted/agreed..." → USE: "the record shows no further rejection was raised after the amendment"\n'
            "- When CN data is sparse, honesty is more valuable than plausible fabrication\n\n"
            # ── STYLE ──
            "Style:\n"
            "- Attorney voice: precise, cautious, evidence-driven\n"
            "- Every sentence either provides strategic insight — or delete it\n"
            "- Tables > paragraphs; specific features > abstract descriptions\n"
            "- No 'It is worth noting that', 'Furthermore', 'Additionally'\n"
            "- Active voice, direct statements\n\n"
            # ── LEGAL ──
            "🚨 RED LINES:\n"
            '- BANNED: "proved", "confirmed [by examiner]", "admitted", "agreed", "conceded", "absolutely", "sole reason", "invalid", "does not infringe"\n'
            '- USE: "the record shows", "data indicates", "based on available information", "may", "potential", "suggests", "appears to have"\n'
            "- Do NOT fabricate examiner's internal thought process\n"
            "- Do NOT make validity conclusions about granted patents\n"
            "- Do NOT provide legal advice — provide strategic intelligence\n\n"
            "Output Markdown directly, no JSON."
        )


    user_content = (
        f"Patent ID: {patent_id}\n"
        f"User query: {query}\n\n"
        f"{family_text}\n\n"
        f"=== US PROSECUTION DATA ===\n"
        f"Analysis dimensions: {', '.join(columns)}\n"
        f"Documents analyzed: {len(table_rows)}\n\n"
        f"{us_data_text}\n\n"
        f"=== CHINA EXAMINATION DATA ===\n"
        f"{cn_data_text}\n\n"
        f"=== JAPAN EXAMINATION DATA ===\n"
        f"{jp_data_text}\n\n"
        f"=== EUROPEAN EXAMINATION DATA ===\n"
        f"{ep_data_text}\n\n"
        f"Generate the AI Patent Prosecution Intelligence Report.\n"
        f"CRITICAL: 70% strategy analysis + 30% event description. "
        f"Do NOT describe what happened — reveal WHY and WHAT IT MEANS. "
        f"Focus on: claim evolution, prior art battle, allowance driver, invalidity risk assessment."
    ) if lang == "zh" else (
        f"Patent ID: {patent_id}\n"
        f"User query: {query}\n\n"
        f"{family_text}\n\n"
        f"=== US PROSECUTION DATA ===\n"
        f"Analysis dimensions: {', '.join(columns)}\n"
        f"Documents analyzed: {len(table_rows)}\n\n"
        f"{us_data_text}\n\n"
        f"=== CHINA EXAMINATION DATA ===\n"
        f"{cn_data_text}\n\n"
        f"=== JAPAN EXAMINATION DATA ===\n"
        f"{jp_data_text}\n\n"
        f"=== EUROPEAN EXAMINATION DATA ===\n"
        f"{ep_data_text}\n\n"
        f"Generate the AI Patent Prosecution Intelligence Report.\n"
        f"CRITICAL: 70% strategy analysis + 30% event description. "
        f"Do NOT describe what happened — reveal WHY and WHAT IT MEANS. "
        f"Focus on: claim evolution, prior art battle, allowance driver, invalidity risk assessment."
    )

    _logger.info(
        f"[prosecution] family_report_start — patent_id={patent_id}, "
        f"us_docs={len(table_rows)}, cn_events={len(cn_events_list)}, "
        f"jurisdictions={jurisdictions}, lang={lang}"
    )

    try:
        llm = pro_provider._get_langchain_llm(streaming=True)
        messages = [("system", system_prompt), ("human", user_content)]
        chunks = []
        async for chunk in llm.astream(messages):
            if chunk.content:
                chunks.append(chunk.content)
                if summary_updater:
                    summary_updater.push(
                        "".join(chunks),
                        step_msg=(
                            "正在撰写跨国审查报告..." if lang == "zh"
                            else "Writing cross-jurisdiction report..."
                        ),
                    )
        text = "".join(chunks).strip()
        if "</think>" in text:
            text = text[text.rfind("</think>") + len("</think>"):].strip()
            if summary_updater:
                summary_updater.push(text, force=True)

        # Strip leading title if LLM generated one (we prepend our own)
        # Remove a leading "# ..." line if present to avoid double titles
        if text.startswith("# "):
            first_nl = text.find("\n")
            if first_nl > 0:
                text = text[first_nl + 1:].strip()

        # Assemble final report
        report = f"# {title}\n\n{text}"
    except Exception as e:
        _logger.warning(f"[prosecution] family_report_failed: {e}")
        # Fallback: generate US report + append CN data
        report = await generate_prosecution_report(
            table_rows, columns, query, patent_id,
            flash_provider, pro_provider, lang,
            summary_updater=summary_updater,
        )
        if cn_exam_data:
            cn_app_num = cn_exam_data.get("cn_app_number", "")
            if lang == "zh":
                report += f"\n\n---\n\n# 中国审查历史 ({cn_app_num})\n\n"
            else:
                report += f"\n\n---\n\n# China Examination History ({cn_app_num})\n\n"
            if cn_exam_data.get("timeline_md"):
                report += cn_exam_data["timeline_md"] + "\n\n"
            if cn_exam_data.get("legal_md"):
                report += cn_exam_data["legal_md"] + "\n\n"
            if cn_exam_data.get("claims_md"):
                report += cn_exam_data["claims_md"] + "\n"

    # Append US analysis table and claim chart at the end
    table_md = _build_markdown_table(table_rows, columns, lang)
    analysis_table_heading = _ANALYSIS_TABLE_HEADINGS.get(lang, _ANALYSIS_TABLE_HEADINGS["en"])
    report += f"\n\n## {analysis_table_heading}\n\n{table_md}"

    _logger.info(
        f"[prosecution] family_report_done — total_chars={len(report)}"
    )
    return report
