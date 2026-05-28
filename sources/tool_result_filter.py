import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from sources.logger import Logger


JsonCall = Callable[[str, str], dict[str, Any] | str | Awaitable[dict[str, Any] | str]]
logger = Logger("tool_result_filter.log")


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


def _json_for_log(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return repr(value)


def _log_info(message: str) -> None:
    try:
        logger.info(message)
    except Exception:
        pass


def _log_error(message: str) -> None:
    try:
        logger.error(message)
    except Exception:
        pass


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
    _log_info(
        "tool_result_filter filter input items "
        f"({original_count}): {_json_for_log(items)}"
    )
    _log_info(f"tool_result_filter filter criteria/keywords: {user_prompt}")

    if not items:
        _log_info("tool_result_filter filtered result items (0): []")
        return FilterResult(
            items=[],
            applied=False,
            original_count=0,
            filtered_count=0,
        )

    kept_items: list[Any] = []
    all_decisions: list[dict[str, Any]] = []
    applied = False
    latest_raw_response: Any = None

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
            latest_raw_response = raw_response
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
            _log_info(
                "tool_result_filter filter decisions "
                f"for batch {start}-{start + len(batch) - 1}: "
                f"has_filter_requirement={has_filter_requirement}, "
                f"decisions={_json_for_log(normalized_decisions)}"
            )

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
        _log_error(
            "tool_result_filter filter failed open; original items kept. "
            f"error={exc}; decisions={_json_for_log(all_decisions)}; "
            f"raw_response={_json_for_log(latest_raw_response)}; "
            f"items={_json_for_log(items)}"
        )
        return FilterResult(
            items=items,
            applied=False,
            original_count=original_count,
            filtered_count=original_count,
            decisions=all_decisions,
            error=str(exc),
        )

    _log_info(
        "tool_result_filter filtered result items "
        f"({len(kept_items)}): {_json_for_log(kept_items)}"
    )
    return FilterResult(
        items=kept_items,
        applied=applied,
        original_count=original_count,
        filtered_count=len(kept_items),
        decisions=all_decisions,
    )
