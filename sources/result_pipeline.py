import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


COMMON_LIST_KEYS = ("items", "results", "records", "rows", "data", "list")


@dataclass
class LocatedItems:
    path: str
    items: List[Any]


@dataclass
class PipelineSummary:
    total: int
    matched: int


def find_primary_list(data: Any) -> LocatedItems:
    """Find the most likely result list inside an arbitrary JSON response."""
    if isinstance(data, list):
        return LocatedItems("$", data)

    candidates: List[LocatedItems] = []

    def visit(value: Any, path: str, key_name: Optional[str] = None) -> None:
        if isinstance(value, list):
            candidates.append(LocatedItems(path, value))
            for index, child in enumerate(value[:3]):
                visit(child, f"{path}[{index}]")
            return

        if isinstance(value, dict):
            ordered_keys = sorted(
                value.keys(),
                key=lambda key: 0 if str(key).lower() in COMMON_LIST_KEYS else 1,
            )
            for key in ordered_keys:
                child_path = f"{path}.{key}" if path != "$" else f"$.{key}"
                visit(value[key], child_path, str(key))

    visit(data, "$")
    if not candidates:
        return LocatedItems("$", [])

    def score(candidate: LocatedItems) -> tuple[int, int]:
        key_score = 0
        lowered_path = candidate.path.lower()
        for key in COMMON_LIST_KEYS:
            if re.search(rf"(\.|^){re.escape(key)}$", lowered_path):
                key_score = 1
                break
        object_score = 1 if candidate.items and isinstance(candidate.items[0], dict) else 0
        return (key_score + object_score, len(candidate.items))

    return max(candidates, key=score)


def profile_items_schema(items: List[Any], sample_size: int = 10) -> Dict[str, Any]:
    fields: Dict[str, Dict[str, Any]] = {}

    def type_name(value: Any) -> str:
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return "number"
        if isinstance(value, str):
            return "string"
        if isinstance(value, list):
            return "list"
        if isinstance(value, dict):
            return "object"
        if value is None:
            return "null"
        return type(value).__name__

    def add_field(path: str, value: Any) -> None:
        entry = fields.setdefault(path, {"type": type_name(value), "examples": []})
        current_type = type_name(value)
        if entry["type"] != current_type and current_type != "null":
            entry["type"] = "mixed"
        if value is not None and value not in entry["examples"]:
            entry["examples"].append(value)
            entry["examples"] = entry["examples"][:3]

    def flatten(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(child, dict):
                    flatten(child, child_path)
                else:
                    add_field(child_path, child)

    for item in items[:sample_size]:
        flatten(item)

    return {"fields": fields}


def user_intent_requests_filter(user_instruction: str) -> bool:
    lowered = user_instruction.lower()
    keywords = (
        "filter",
        "only",
        "where",
        "match",
        "matching",
        "筛选",
        "过滤",
        "只要",
        "只看",
        "符合",
        "满足",
        "低于",
        "高于",
        "小于",
        "大于",
        "等于",
        "包含",
    )
    return any(keyword in lowered for keyword in keywords)


class ResultPipeline:
    def __init__(self, llm, callback_handler, batch_size: int = 5):
        self.llm = llm
        self.callback_handler = callback_handler
        self.batch_size = batch_size

    async def stream_items(
        self,
        items: List[Any],
        user_instruction: str,
        requires_filter: Optional[bool] = None,
    ) -> PipelineSummary:
        total = len(items)
        if requires_filter is None:
            requires_filter = user_intent_requests_filter(user_instruction)

        if requires_filter:
            return await self._stream_filtered_items(items, user_instruction)

        await self._emit(f"\n\n---\n\n已获取 {total} 条结果，正在整理输出...\n\n")
        matched = 0
        for batch_start in range(0, total, self.batch_size):
            batch = items[batch_start:batch_start + self.batch_size]
            batch_end = min(batch_start + self.batch_size, total)
            await self._emit(f"### 正在整理 {batch_start + 1}-{batch_end} / {total}\n\n")
            await self._format_batch(batch)
            await self._emit("\n\n")
            matched += len(batch)
        await self._emit(f"整理完成：共输出 {matched} 条。\n")
        return PipelineSummary(total=total, matched=matched)

    async def _stream_filtered_items(self, items: List[Any], user_instruction: str) -> PipelineSummary:
        total = len(items)
        matched = 0
        await self._emit(f"\n\n---\n\n已获取 {total} 条结果，正在按你的条件筛选...\n\n")

        for batch_start in range(0, total, self.batch_size):
            batch = items[batch_start:batch_start + self.batch_size]
            batch_end = min(batch_start + self.batch_size, total)
            await self._emit(f"### 正在筛选 {batch_start + 1}-{batch_end} / {total}\n\n")

            kept_items = await self._filter_batch(batch, batch_start, user_instruction)
            if not kept_items:
                await self._emit("未发现匹配项。\n\n")
                continue

            matched += len(kept_items)
            await self._emit(f"找到 {len(kept_items)} 条匹配项：\n\n")
            await self._format_batch(kept_items)
            await self._emit("\n\n")

        await self._emit(f"筛选完成：共检查 {total} 条，匹配 {matched} 条。")
        return PipelineSummary(total=total, matched=matched)

    async def _filter_batch(self, batch: List[Any], batch_start: int, user_instruction: str) -> List[Any]:
        schema_profile = profile_items_schema(batch)
        response = await self.llm.complete_simple(
            system_prompt=(
                "You are a strict JSON filter. Apply the user's filter instruction "
                "to each item independently. Return only valid JSON with a top-level "
                "'matches' array. Each match must include the original global index, "
                "a boolean keep value, and a short reason. Do not output Markdown."
            ),
            user_content=(
                f"User filter instruction:\n{user_instruction}\n\n"
                f"Global start index: {batch_start}\n\n"
                f"Detected item schema:\n{json.dumps(schema_profile, ensure_ascii=False, indent=2)}\n\n"
                f"Items JSON:\n{json.dumps(batch, ensure_ascii=False, indent=2)}"
            ),
        )
        payload = _parse_json_object(response)
        keep_indexes = {
            int(match["index"])
            for match in payload.get("matches", [])
            if isinstance(match, dict) and match.get("keep") is True and "index" in match
        }
        return [
            item
            for offset, item in enumerate(batch)
            if batch_start + offset in keep_indexes
        ]

    async def _format_batch(self, batch: List[Any]) -> None:
        await self.llm.stream_simple(
            system_prompt=(
                "You are presenting result items clearly and concisely. "
                "For each item, extract and present the most important information as clean Markdown. "
                "Use bold field names where helpful. Do not add a preamble or conclusion."
            ),
            user_content=(
                f"Format these {len(batch)} result items as readable Markdown:\n\n"
                f"{json.dumps(batch, ensure_ascii=False, indent=2)}"
            ),
            callback_handler=self.callback_handler,
        )

    async def _emit(self, text: str) -> None:
        if self.callback_handler:
            await self.callback_handler.on_llm_new_token(text)


def _parse_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("Filter LLM response must be a JSON object")
    return parsed
