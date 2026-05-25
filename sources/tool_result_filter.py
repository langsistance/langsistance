import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


JsonCall = Callable[[str, str], dict[str, Any] | str | Awaitable[dict[str, Any] | str]]


@dataclass
class FilterResult:
    items: list[Any]
    applied: bool
    original_count: int
    filtered_count: int
    decisions: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


FILTER_SYSTEM_PROMPT = """You are a strict list filter.
Use only the user request and item JSON.
Do not invent facts or assume fields that are not present.
Return one decision for every input index.
If the user request does not contain explicit list filtering criteria, keep every item.
If uncertain, keep the item.
Output JSON only.
"""


def _parse_json_response(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise ValueError(f"LLM JSON response must be dict or str, got {type(value).__name__}")

    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON response must decode to an object")
    return parsed


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


def _decision_index(decision: dict[str, Any]) -> int | None:
    index = decision.get("index")
    if isinstance(index, bool):
        return None
    if isinstance(index, int):
        return index
    if isinstance(index, str) and index.isdigit():
        return int(index)
    return None


def _decision_should_keep(
    decision: dict[str, Any] | None,
    keep_threshold: float,
) -> bool:
    if not decision:
        return True
    if decision.get("keep") is not False:
        return True
    confidence = decision.get("confidence", 0)
    if not isinstance(confidence, (int, float)):
        return True
    return confidence < keep_threshold


def _build_filter_user_content(
    user_prompt: str,
    indexed_items: list[dict[str, Any]],
) -> str:
    return (
        "User request:\n"
        f"{user_prompt}\n\n"
        "Items to classify. Preserve indexes exactly:\n"
        f"{json.dumps(indexed_items, ensure_ascii=False, indent=2)}\n\n"
        "Return this JSON shape exactly:\n"
        "{\n"
        '  "has_filter_requirement": true,\n'
        '  "decisions": [\n'
        '    {"index": 0, "keep": true, "confidence": 0.95, "reason": "short reason"}\n'
        "  ]\n"
        "}\n"
    )


async def filter_tool_result_items(
    items: list[Any],
    user_prompt: str,
    llm_json_call: JsonCall,
    batch_size: int = 10,
    keep_threshold: float = 0.75,
) -> FilterResult:
    """Filter tool result list items with fail-open LLM decisions."""
    original_count = len(items)
    if not items:
        return FilterResult(
            items=[],
            applied=False,
            original_count=0,
            filtered_count=0,
        )

    kept_items: list[Any] = []
    all_decisions: list[dict[str, Any]] = []
    applied = False

    try:
        for start in range(0, original_count, batch_size):
            batch = items[start:start + batch_size]
            indexed_items = [
                {"index": start + offset, "item": item}
                for offset, item in enumerate(batch)
            ]
            raw_response = await _maybe_await(
                llm_json_call(
                    FILTER_SYSTEM_PROMPT,
                    _build_filter_user_content(user_prompt, indexed_items),
                )
            )
            response = _parse_json_response(raw_response)
            has_filter_requirement = response.get("has_filter_requirement") is True
            decisions = response.get("decisions", [])
            if not isinstance(decisions, list):
                raise ValueError("LLM JSON response decisions must be a list")

            normalized_decisions = [
                decision for decision in decisions
                if isinstance(decision, dict) and _decision_index(decision) is not None
            ]
            all_decisions.extend(normalized_decisions)

            if not has_filter_requirement:
                kept_items.extend(batch)
                continue

            applied = True
            decisions_by_index = {
                _decision_index(decision): decision
                for decision in normalized_decisions
            }
            for absolute_index, item in zip(range(start, start + len(batch)), batch):
                decision = decisions_by_index.get(absolute_index)
                if _decision_should_keep(decision, keep_threshold):
                    kept_items.append(item)
    except Exception as exc:
        return FilterResult(
            items=items,
            applied=False,
            original_count=original_count,
            filtered_count=original_count,
            decisions=all_decisions,
            error=str(exc),
        )

    return FilterResult(
        items=kept_items,
        applied=applied,
        original_count=original_count,
        filtered_count=len(kept_items),
        decisions=all_decisions,
    )
