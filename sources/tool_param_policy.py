TEMPLATE_PARAM_RULES = """
        1. You may modify values in the existing params JSON to match the user's request.
           - Keep the top-level params structure compatible with the template
           - You may add or replace keys inside existing query/body objects when the user's request semantically requires API parameters
           - Do NOT add user_id or query_id to the params JSON - these are separate parameters

        2. Field semantics:
           - method MUST remain unchanged
           - query contains URL query parameters
           - header contains HTTP headers
           - body contains the HTTP request body

        3. Value replacement rules:
           - Build query/body parameter values from the user's semantic request when they clearly map to the API task
           - If the user query does not mention or imply a parameter, keep its original value unchanged
           - Do NOT infer or invent user_id or query_id from the user's request into the params JSON
"""


def should_expose_dynamic_tool(push: int, has_tool_data: bool, query_body_empty: bool) -> bool:
    if push == 1:
        return not has_tool_data
    if push == 2:
        return True
    if push == 3:
        return False
    return True
