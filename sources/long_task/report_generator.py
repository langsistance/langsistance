"""Phase 3: Dynamic report generation — outline + section-by-section writing."""


async def generate_report_outline(query: str, columns: list[str],
                                   table_rows: list[dict], provider) -> dict:
    """Phase 3a: Generate a dynamic report outline based on user query and data.

    Args:
        query: The user's original analysis question.
        columns: List of column names from the analysis table.
        table_rows: List of row dicts from patent analysis (may include _failed rows).
        provider: LLM provider instance with a ``complete_json`` method.

    Returns:
        dict with keys ``title`` (str) and ``sections`` (list of dicts with
        ``heading`` and ``description``).
    """
    row_count = len(table_rows)
    cols_str = ", ".join(columns)
    failed_count = sum(1 for r in table_rows if r.get('_failed'))

    system_prompt = """你是一个专利分析报告架构师。根据用户问题和分析结果，规划报告结构。
返回 JSON：{"title": "报告标题", "sections": [{"heading": "章节标题", "description": "本章内容说明"}]}
章节数 3-7 个，标题简洁。"""

    user_content = f"""用户问题：{query}
分析维度：{cols_str}
已分析专利数：{row_count}{f'（其中 {failed_count} 个分析失败）' if failed_count else ''}

请规划报告结构。"""

    result = await provider.complete_json(system_prompt, user_content)
    return result


async def generate_report_section(section: dict, query: str, columns: list[str],
                                   table_rows: list[dict], provider) -> str:
    """Phase 3b: Write a single report section via streaming LLM call.

    Unlike generate_report_outline, this produces free-form Markdown text,
    so it uses streaming instead of complete_json.
    """
    heading = section.get('heading', '')
    description = section.get('description', '')

    # Build a per-patent summary with all columns for accurate citation.
    # Each patent is prefixed with **[patent_id]** so the LLM can cite it.
    patent_entries: list[str] = []
    for r in table_rows[:20]:
        pid = r.get('patent_id', r.get('专利号', '?'))
        if r.get('_failed'):
            patent_entries.append(f"- {pid}: 分析失败")
            continue
        parts = [f"**[{pid}]**"]
        for col in columns:
            if col in ('patent_id', '专利号'):
                continue
            val = str(r.get(col, '')).strip()
            if val:
                parts.append(f"  - {col}: {val}")
        patent_entries.append("\n".join(parts))
    data_summary = "\n\n".join(patent_entries[:20])

    system_prompt = (
        "你是一个专利分析报告撰写专家。根据给定的分析数据，撰写一个报告章节。"
        "用中文，具体有依据。直接输出 Markdown 格式的章节内容，不要输出 JSON。\n\n"
        "CRITICAL 引用规则：\n"
        "1. 报告中提到的每一个技术点、技术问题、技术方案、技术效果，都必须在后面"
        "用 **[专利号]** 标注来源专利。例如：\n"
        "   - 采用碳纤维复合材料实现轻量化 **[202310123456.7]**\n"
        "   - 该技术方案被多个专利采用 **[202310123456.7]** **[18331482]**\n"
        "2. 每个段落至少引用 1-2 个专利号。如果一个观点来自多个专利，全部列出。\n"
        "3. 不要虚构专利号，只引用数据摘要中给出的专利号。\n"
        "4. 引用格式统一用 **[]** 包裹专利号，紧跟在被引用的内容后面。"
    )
    user_content = f"""用户问题：{query}
本章标题：{heading}
本章说明：{description}

各专利分析结果（每个专利的 **[专利号]** 标注了来源标识，引用时必须使用）：
{data_summary}

请撰写"{heading}"章节内容。要求：
- 每个技术点引用来源专利，用 **[专利号]** 格式标注
- Markdown 格式，400-800 字
- 具体有依据，不编造："""

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
    return text or f"（{heading} 生成失败）"
