def should_prefetch_tool_result(has_tool_data: bool, query_body_empty: bool) -> bool:
    return not has_tool_data and query_body_empty


def should_expose_dynamic_tool(push: int, has_tool_data: bool, query_body_empty: bool) -> bool:
    if push == 1:
        return not has_tool_data and not query_body_empty
    if push == 2:
        return not query_body_empty
    if push == 3:
        return False
    return True
