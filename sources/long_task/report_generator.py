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

    # Build a compact summary of the table data for context
    data_summary_lines = []
    for r in table_rows[:20]:
        if r.get('_failed'):
            data_summary_lines.append(f"{r.get('patent_id', '?')}: 分析失败")
            continue
        parts = [f"{r.get('patent_id', '?')}:"]
        for col in columns[1:4]:
            val = r.get(col, '')
            if val:
                parts.append(f"{col}={str(val)[:60]}")
        data_summary_lines.append("  ".join(parts))
    data_summary = "\n".join(data_summary_lines[:20])

    system_prompt = (
        "你是一个专利分析报告撰写专家。根据给定的分析数据，撰写一个报告章节。"
        "用中文，具体有依据。直接输出 Markdown 格式的章节内容，不要输出 JSON。"
    )
    user_content = f"""用户问题：{query}
本章标题：{heading}
本章说明：{description}

分析数据摘要：
{data_summary}

请撰写"{heading}"章节内容（Markdown 格式，300-600 字）："""

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
