"""Distill a downloaded patent specification into a bounded observation.

The chat loop cannot hold a 20k-200k char specification in context, so the
full text is distilled by an LLM into the standard patent analysis
dimensions (invention point / technical problem / solution / claim gist /
relevance to the user's question).  On any failure the caller falls back
to the truncated full text.
"""

from typing import Any

SPEC_FALLBACK_LIMIT = 16000

DISTILL_SYSTEM_PROMPT = (
    "你是一个专利分析专家。阅读给定的专利说明书全文，提炼出结构化要点，"
    "用于回答用户的专利分析问题。\n\n"
    "要求：\n"
    "1. 每个要点用中文简明表述，聚焦技术实质，不复制原文段落\n"
    "2. 「权利要求要点」提炼独立权利要求的保护范围要点（最多 5 条）\n"
    "3. 「与用户问题的相关性」明确指出该专利与用户问题相关的技术点，"
    "或说明不相关\n"
    "4. 不编造内容——只依据给定的说明书文本\n\n"
    'Return JSON: {"发明点": "...", "解决的技术问题": "...", '
    '"技术方案": "...", "权利要求要点": "...", '
    '"与用户问题的相关性": "..."}'
)


def truncated_fallback(text: str, limit: int = SPEC_FALLBACK_LIMIT) -> str:
    """Return the first *limit* chars of the spec text (fallback observation)."""
    text = str(text or "")
    return text[:limit]


def format_distilled(distilled: dict, lang: str = "zh") -> str:
    """Render a distilled dict into the observation text for the loop."""
    if not isinstance(distilled, dict) or not distilled:
        return ""
    labels = ("发明点", "解决的技术问题", "技术方案",
              "权利要求要点", "与用户问题的相关性")
    lines = []
    for label in labels:
        value = distilled.get(label)
        if isinstance(value, str) and value.strip():
            lines.append(f"**{label}**：{value.strip()}")
    return "\n\n".join(lines)


async def distill_patent_spec(text: str, query: str, provider: Any) -> dict:
    """Distill a spec via the LLM.  Never raises — returns {} on failure."""
    if not text:
        return {}
    user_content = f"用户问题：{query}\n\n专利说明书全文：\n{text}"
    try:
        result = await provider.complete_json(DISTILL_SYSTEM_PROMPT, user_content)
    except Exception:
        return {}
    if not isinstance(result, dict):
        return {}
    return result
