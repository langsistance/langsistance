from copy import deepcopy
from typing import Any, Dict


TEMPLATE_PARAM_RULES = """
        1. You may only change API request parameters that already exist in the template.
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
