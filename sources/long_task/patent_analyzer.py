"""Core patent analysis pipeline functions (Phase 1 & 2)."""


def build_failed_row(patent_id: str, reason: str) -> dict:
    """Return a placeholder row for a patent that failed to process."""
    return {
        'patent_id': patent_id,
        '_failed': True,
        '_failure_reason': reason,
    }


async def generate_table_columns(query: str, patent_count: int, provider) -> list[str]:
    """Phase 1: Use Flash to dynamically generate table column definitions."""
    system_prompt = """你是一个专利分析专家。根据用户的分析问题，确定对比表格需要哪些列。
返回 JSON 格式：{"columns": ["列1", "列2", ...]}
列数控制在 4-6 列。"专利号"必须是第一列。
根据问题类型选择列：
- 技术分析：技术领域、核心技术方案、创新点、相关度
- 对比分析：申请人、技术方向、核心方案、差异点
- 风险评估：保护范围、产品对照、风险等级、规避建议"""
    user_content = f"用户问题：{query}\n专利数量：{patent_count}\n请确定分析表格的列定义。"
    result = await provider.complete_json(system_prompt, user_content)
    return result.get('columns', ['专利号', '分析结果'])


async def download_patent_document(patent_id: str, source: str = 'cnipa') -> str:
    """Download patent full text from DI platform (CNIPA) or USPTO.

    Uses existing tool infrastructure — calls the DI platform API
    through dynamic_tool_params.execute_backend_tool_request().
    Returns the patent description text.
    """
    from sources.dynamic_tool_params import _inject_zldjs_auth_params
    from sources.patent_token import ensure_valid_access_token
    import httpx

    access_token = ensure_valid_access_token()

    if source == 'cnipa':
        if not access_token:
            raise RuntimeError("No patent access token available — check DI platform OAuth config")
        url = f"https://open.zldsj.com/api/patent/{patent_id}/fulltext"
        params = {'access_token': access_token}
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            # Extract description from DI platform response
            if 'data' in data and 'description' in data['data']:
                return data['data']['description']
            return str(data)
    else:
        # USPTO path — not implemented yet; scene tools should handle this
        raise NotImplementedError(
            "USPTO patent download is not yet implemented in the fallback path. "
            "Configure a USPTO download knowledge+tool in the scene instead."
        )


async def analyze_single_patent(
    patent_id: str,
    patent_text: str,
    columns: list[str],
    query: str,
    provider,
    timeout: int = 60,
) -> dict:
    """Phase 2b: Analyze one patent and return a table row dict."""
    non_id_columns = [c for c in columns if c != '专利号']
    col_keys = "\n".join(f'  "{c}": "..."' for c in non_id_columns)
    system_prompt = f"""你是一个专利分析专家。根据以下维度分析专利：

{chr(10).join(f"- {c}" for c in non_id_columns)}

返回 JSON，**CRITICAL: JSON 的 key 必须严格使用以下列名，一个不能多一个不能少：**
{{
  "patent_id": "{patent_id}",
{col_keys}
}}
分析要具体、有依据，基于专利文本内容。"""

    user_content = f"""用户问题：{query}

专利号：{patent_id}
专利文本（说明书全文）：
{patent_text[:16000]}

请按维度分析并返回 JSON。"""

    result = await provider.complete_json(system_prompt, user_content)
    result['patent_id'] = patent_id

    # ── Key alignment: ensure all result keys match the expected column names ──
    aligned: dict = {'patent_id': patent_id}

    # Collect LLM-returned keys that are NOT the expected column names
    extra_keys = [k for k in result if k not in columns and k != 'patent_id']

    if len(extra_keys) == len(non_id_columns):
        # LLM used different key names — map by position
        for ek, col_name in zip(extra_keys, non_id_columns):
            aligned[col_name] = result.get(ek, '')
    else:
        # Direct key matching or fallback
        for col in non_id_columns:
            if col in result:
                aligned[col] = result[col]
            else:
                aligned[col] = result.get(col, '（分析数据未生成）')

    return aligned


async def generate_patent_summary(
    patent_id: str,
    row: dict,
    query: str,
    provider,
) -> str:
    """Phase 2c: Generate a short summary for a single analyzed patent."""
    row_str = "\n".join(
        f"{k}: {v}" for k, v in row.items()
        if k not in ('patent_id', '_failed', '_failure_reason', '_summary')
    )
    system_prompt = (
        "你是一个专利分析专家。基于分析结果，用 2-3 句话总结该专利的核心发现。"
        "直接输出总结内容，不要输出 JSON。"
    )
    user_content = (
        f"用户问题：{query}\n"
        f"专利：{patent_id}\n"
        f"分析结果：\n{row_str}\n\n"
        f"请给出简洁总结。"
    )
    # Use streaming for free-text output (not complete_json)
    llm = provider._get_langchain_llm(streaming=True)
    messages = [("system", system_prompt), ("human", user_content)]
    chunks = []
    async for chunk in llm.astream(messages):
        if chunk.content:
            chunks.append(chunk.content)
    text = "".join(chunks).strip()
    if "</think>" in text:
        text = text[text.rfind("</think>") + len("</think>"):].strip()
    return text or ""
