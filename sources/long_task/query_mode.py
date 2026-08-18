"""Query-mode classification: structured/analytical requests vs semantic
technology searches.

Structured requests — identifier retrieval (application / publication /
patent numbers), assignee or explicit-keyword searches, prosecution or
family analysis, document-list retrieval — skip the CPC semantic match
and the architecture interpretation; those two stages only serve
semantic technology searches (natural-language technical descriptions
matched against patent literature).

Classification runs on the already-constructed main agent provider (one
small JSON call, ~1-2s).  Any failure defaults to "semantic" so
technology searches keep the full pipeline — a misclassified structured
query only costs the skip, never breaks retrieval.
"""
import json
import os
from typing import Any

QUERY_MODE_ENABLED = os.getenv("REACT_QUERY_MODE_ENABLED", "1") == "1"

MODE_SYSTEM_PROMPT = (
    "你是一个专利检索意图分类器。判断用户的检索请求属于哪一类：\n"
    "1. structured — 按结构化条件检索或分析：专利申请号、公开号、专利号、"
    "出版物编号等标识符检索；按受让人(assignee)或申请人检索；按明确给出的"
    "关键词检索；审查历史(prosecution history)分析；同族专利或跨国同族"
    "分析；获取某专利的文档/文件清单\n"
    "2. semantic — 技术语义检索：用户用自然语言描述一项技术、功能或结构"
    "特征，需要在专利文献中按技术方案匹配检索\n"
    "只输出 JSON，不要其他文字：{\"mode\": \"structured\" | \"semantic\"}"
)


def _normalize(raw: Any) -> str:
    """Extract the mode from a parsed or raw classifier output."""
    if isinstance(raw, dict):
        candidates = [raw]
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            parsed = None
        candidates = [parsed] if isinstance(parsed, dict) else []
    else:
        candidates = []
    for candidate in candidates:
        mode = str(candidate.get("mode") or "").strip().lower()
        if mode in ("structured", "semantic"):
            return mode
    return "semantic"


async def classify_query_mode(query: str, provider: Any) -> str:
    """Classify the query as 'structured' or 'semantic'.  Never raises —
    any failure returns 'semantic' so tech searches keep the full
    pipeline."""
    if not QUERY_MODE_ENABLED:
        return "semantic"
    query = str(query or "").strip()
    if not query:
        return "semantic"
    try:
        result = await provider.complete_json(MODE_SYSTEM_PROMPT, query)
        return _normalize(result)
    except Exception:
        return "semantic"
