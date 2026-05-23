def should_prefetch_tool_result(has_tool_data: bool, query_body_empty: bool) -> bool:
    return not has_tool_data and query_body_empty


def should_expose_dynamic_tool(has_tool_data: bool, query_body_empty: bool) -> bool:
    return not has_tool_data and not query_body_empty
