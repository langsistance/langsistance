from copy import deepcopy
from typing import Any, Dict


PARAMS_FIELD_DESCRIPTION = (
    "Complete API request parameters as the full params JSON. Start from the "
    "tool template, replace or remove only existing query/body fields, and do "
    "not add new keys. If a q value contains fielded search clauses joined by "
    "OR, narrow that q value to the clauses that match the user's requested "
    "field meaning."
)


TEMPLATE_PARAM_RULES = """
        1. You may only change API request parameters that already exist in the template.
           - Start from the full params template
           - Return the full params JSON after applying allowed replacements or removals
           - If query or body is empty in the template, it MUST remain empty
           - If query or body has existing keys, you may replace or remove existing keys based on the user's request
           - You MUST NOT add new keys to query, body, header, or any other params object
           - Do NOT change the JSON structure or nesting
           - Do NOT include user_id or query_id in the params JSON - these are separate parameters

        2. Field semantics:
           - method MUST remain unchanged
           - query contains URL query parameters
           - header contains HTTP headers
           - body contains the HTTP request body

        3. Value replacement rules:
           - Replace a value only if the user query clearly maps to the meaning of an existing field
           - If an existing string value is a search expression with OR alternatives, and the user request clearly targets one of those alternatives, narrow the existing expression to the relevant alternative
           - Treat each fielded search clause inside a q value as an individual alternative when clauses are joined by OR
           - A fielded search clause has its own field path, operator/syntax, and search value; compare the user's requested field meaning against those field paths before deciding what to keep
           - Preserve the existing field/operator syntax when narrowing OR alternatives; replace only the search value and remove unrelated alternatives
           - Generic example: if a template q value is fieldA:"123" OR metadata.fieldB:"123" and the user asks for field B, the q value should become metadata.fieldB:"123"
           - Remove an existing query/body key only if the user request implies it should not be sent
           - If the user query does not mention or imply a field, keep its original value unchanged
           - Do NOT infer or invent information not explicitly expressed by the user
           - Do NOT extract or infer user_id or query_id from the user's request into the params JSON
"""


def normalize_tool_request_params(template_params: Dict[str, Any], llm_params: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp LLM-provided API params to the tool template's allowed query/body keys."""
    normalized = deepcopy(template_params)
    if not isinstance(llm_params, dict):
        llm_params = {}

    normalized["query"] = _normalize_param_section(
        template_params.get("query", {}),
        llm_params.get("query"),
    )
    normalized["body"] = _normalize_param_section(
        template_params.get("body", {}),
        llm_params.get("body"),
    )
    return normalized


def _normalize_param_section(template_section: Any, llm_section: Any) -> Any:
    if isinstance(template_section, dict):
        if not template_section:
            return {}
        if not isinstance(llm_section, dict):
            return deepcopy(template_section)
        return {
            key: llm_section[key]
            for key in template_section
            if key in llm_section
        }

    if template_section in (None, "", [], ()):
        return deepcopy(template_section)

    return deepcopy(llm_section) if llm_section is not None else deepcopy(template_section)


def should_prefetch_tool_result(has_tool_data: bool, query_body_empty: bool) -> bool:
    return not has_tool_data and query_body_empty


def should_expose_dynamic_tool(has_tool_data: bool, query_body_empty: bool) -> bool:
    return not has_tool_data and not query_body_empty
