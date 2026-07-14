from typing import Dict, Any
import json
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup

from sources.knowledge.knowledge import (
    get_redis_connection,
    get_knowledge_tool,
    select_knowledge_tool_with_llm,
    clean_html_text,
)
from sources.utility import pretty_print, animate_thinking
from sources.agents.agent import Agent
from sources.tools.mcpFinder import MCP_finder
from sources.memory import Memory
from sources.logger import Logger
from sources.dynamic_tool_params import (
    _append_path_to_url,
    _coerce_json_object,
    _is_path_query_body_empty,
    _replace_uspto_download_urls_for_batch,
    execute_backend_tool_request,
)
from sources.tool_result_filter import filter_tool_result_items
from sources.result_export import build_result_artifacts
from sources.workflow.workflow_executor import WorkflowExecutor, is_workflow_knowledge
from sources.http_outbound import outbound_http

from langchain_core.tools import StructuredTool

import os
import time
import re
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

# Headers whose values must never be exposed to the LLM
_SENSITIVE_HEADER_RE = re.compile(
    r'(auth|api.?key|token|secret|credential|password|bearer)',
    re.IGNORECASE
)
_URL_IN_TEXT_RE = re.compile(r'https?://[^\s"\'<>\])}]+')
MAX_BATCH_JSON_CHARS_FOR_LLM = 200000
MAX_MARKDOWN_VALUE_CHARS = 500
MAX_ITEM_CHARS_FOR_LLM = int(os.getenv("GENERAL_AGENT_MAX_ITEM_CHARS", "15000"))
MAX_VALUE_CHARS_THRESHOLD = int(os.getenv("GENERAL_AGENT_MAX_VALUE_CHARS", "10000"))
SMALL_LIST_THRESHOLD = int(os.getenv("GENERAL_AGENT_SMALL_LIST_THRESHOLD", "3"))
USE_LARGE_LIST_SUMMARY = True  # 设为 False 切回逐条批处理模式

def _is_long_task_knowledge(knowledge_item) -> bool:
    """Check if knowledge item represents a long task (type=3)."""
    if knowledge_item is None:
        return False
    k_type = knowledge_item.get('type', 1) if isinstance(knowledge_item, dict) else getattr(knowledge_item, 'type', 1)
    return int(k_type) == 3


def _build_long_task_intent(knowledge_item, tool_info) -> dict:
    """Build a long task intent marker returned by create_agent."""
    return {
        'intent': 'long_task',
        'knowledge': knowledge_item,
        'tool_info': tool_info,
    }


def _json_len(obj) -> int:
    """Return the byte length of *obj* serialised as compact JSON."""
    return len(json.dumps(obj, ensure_ascii=False, default=str))


def _prune_item_for_llm(item, max_item_chars=MAX_ITEM_CHARS_FOR_LLM, max_value_chars=MAX_VALUE_CHARS_THRESHOLD):
    """Recursively prune *item* so it fits within *max_item_chars*.

    Rules (applied depth-first):
    1. If the whole item serialised is 鈮?max_item_chars, return it unchanged.
    2. For a dict: iterate over keys.  For each value:
       - If the value is a **list** and its JSON size exceeds max_value_chars,
         drop the key entirely (the array is too large).
       - If the value is a **dict** and its JSON size exceeds max_value_chars,
         recurse into it to drop oversized nested arrays.
       - If the value is a **string** that exceeds max_value_chars, truncate it.
       - Otherwise keep it as-is.
    3. For a list: apply the same pruning to every element.
    4. Scalars (str / int / float / bool / None) are truncated when over
       max_value_chars, otherwise kept.

    The function returns a **new** object 鈥?the input is never mutated.
    """
    # Fast path: the item already fits.
    if _json_len([item]) <= max_item_chars:
        return item

    if isinstance(item, dict):
        result: dict[str, Any] = {}
        for key, value in item.items():
            if isinstance(value, list):
                if _json_len(value) > max_value_chars:
                    continue  # drop oversized array
                result[key] = value
            elif isinstance(value, dict):
                if _json_len(value) > max_value_chars:
                    # Recurse to strip oversized arrays from the nested dict.
                    result[key] = _prune_item_for_llm(value, max_item_chars, max_value_chars)
                else:
                    result[key] = value
            elif isinstance(value, str):
                if len(value) > max_value_chars:
                    result[key] = (
                        value[:max_value_chars]
                        + f"... [truncated {len(value) - max_value_chars} chars]"
                    )
                else:
                    result[key] = value
            else:
                result[key] = value
        return result

    if isinstance(item, list):
        return [_prune_item_for_llm(i, max_item_chars, max_value_chars) for i in item]

    # Scalar (string / number / bool / None)
    if isinstance(item, str) and len(item) > max_value_chars:
        return item[:max_value_chars] + f"... [truncated {len(item) - max_value_chars} chars]"
    return item


def _collect_urls_from_value(value):
    urls = []
    if isinstance(value, dict):
        for nested_value in value.values():
            urls.extend(_collect_urls_from_value(nested_value))
    elif isinstance(value, list):
        for item in value:
            urls.extend(_collect_urls_from_value(item))
    elif isinstance(value, str):
        urls.extend(match.rstrip(".,;") for match in _URL_IN_TEXT_RE.findall(value))
    return urls


def _format_markdown_value(value) -> str:
    text = str(value)
    if len(text) > MAX_MARKDOWN_VALUE_CHARS:
        hidden_count = len(text) - MAX_MARKDOWN_VALUE_CHARS
        return f"{text[:MAX_MARKDOWN_VALUE_CHARS]}... [truncated {hidden_count} chars]"
    return text


LARGE_LIST_SUMMARY_MAX_VALUE_CHARS = 5000


def _prune_for_summary(items: list, max_value_chars: int = None) -> list:
    """Return a copy of items with keys whose string value exceeds max_value_chars removed."""
    if max_value_chars is None:
        max_value_chars = LARGE_LIST_SUMMARY_MAX_VALUE_CHARS
    result = []
    for item in items:
        if not isinstance(item, dict):
            result.append(item)
            continue
        pruned = {}
        for k, v in item.items():
            if isinstance(v, str) and len(v) > max_value_chars:
                continue
            pruned[k] = v
        result.append(pruned)
    return result

# 瀹氫箟鍙傛暟妯″瀷
class DynamicToolFunction(BaseModel):
    user_id: str = Field(description="user id")
    query_id: str = Field(description="query id")
    params: str = Field(
        description=(
            "params. If the original tool params contain api-key, api_key, "
            "apikey, x-api-key, or another API key field, preserve that key "
            "and its value exactly in the generated params."
        )
    )


class DynamicBackendToolFunction(BaseModel):
    user_id: str = Field(description="user id")
    query_id: str = Field(description="query id")
    params: Dict[str, Any] | str = Field(
        description=(
            "API request parameters as a JSON object. For push=2 tools, "
            "params may contain path, query, and body; path is appended to "
            "the tool URL after replacing template placeholders from the user request. "
            "If the original tool params contain api-key, api_key, apikey, "
            "x-api-key, or another API key field, preserve that key and its value "
            "exactly in the generated params. "
            "Legacy JSON strings are also accepted."
        )
    )


async def _emit_status(callback_handler, message: str):
    """Send a transient status through the callback handler if supported."""
    if callback_handler is None:
        return
    on_status = getattr(callback_handler, "on_status", None)
    if on_status is None:
        return
    try:
        await on_status(message)
    except Exception:
        pass


def _status_callback_for(callback_handler):
    """Create an async status callback suitable for workflow executors."""
    if callback_handler is None:
        return None
    on_status = getattr(callback_handler, "on_status", None)
    if on_status is None:
        return None

    async def emit(message: str):
        try:
            await on_status(message)
        except Exception:
            pass

    return emit


class GeneralAgent(Agent):

    def __init__(self, name, prompt_path, provider, verbose=False):
        """
        The mcp agent is a special agent for using MCPs.
        MCP agent will be disabled if the user does not explicitly set the MCP_FINDER_API_KEY in environment variable.
        """
        super().__init__(name, prompt_path, provider, verbose, None)
        keys = self.get_api_keys()
        self.tools = {
            #"mcp_finder": MCP_finder(keys["mcp_finder"]),
            # add mcp tools here
        }
        self.role = "mcp"
        self.type = "mcp_agent"
        self.memory = Memory(self.load_prompt(prompt_path),
                                recover_last_session=False, # session recovery in handled by the interaction class
                                memory_compression=False,
                                model_provider=provider.get_model_name())
        self.enabled = True
        self.knowledgeTool = {}
        self.logger = Logger("general_agent.log")

    def get_api_keys(self) -> dict:
        """
        Returns the API keys for the tools.
        """
        api_key_mcp_finder = os.getenv("MCP_FINDER_API_KEY")
        if not api_key_mcp_finder or api_key_mcp_finder == "":
            pretty_print("MCP Finder disabled.", color="warning")
            self.enabled = False
        return {
            "mcp_finder": api_key_mcp_finder
        }

    def _sanitize_params_for_llm(self, params_str: str) -> str:
        """Return params JSON with sensitive header values replaced by '****'.

        The real header values stay in tool_info.params for server-side use only.
        This sanitised copy is the only version the LLM ever sees.
        """
        try:
            data = json.loads(params_str)
            if not isinstance(data, dict):
                return params_str
            sanitized = {}
            for k, v in data.items():
                if k == "header" and isinstance(v, dict):
                    sanitized[k] = {
                        hk: "****" if _SENSITIVE_HEADER_RE.search(hk) else hv
                        for hk, hv in v.items()
                    }
                else:
                    sanitized[k] = v
            return json.dumps(sanitized, ensure_ascii=False, indent=2)
        except Exception:
            return params_str
    
    def set_knowledge_tool(self, knowledge_tool: Dict[str, Any]) -> None:
        """Set the knowledge tool dictionary."""
        self.knowledgeTool = knowledge_tool

    def _flatten_dict(self, d: dict) -> str:
        """Flatten a dict into readable 'key: value' pairs, skipping empty values."""
        parts = []
        for k, v in d.items():
            if v is None or v == "" or v == {} or v == []:
                continue
            if isinstance(v, dict):
                sub = self._flatten_dict(v)
                if sub:
                    parts.append(f"{k}: {sub}")
            elif isinstance(v, list):
                if v and not isinstance(v[0], (dict, list)):
                    parts.append(f"{k}: {', '.join(_format_markdown_value(i) for i in v)}")
            else:
                parts.append(f"{k}: {_format_markdown_value(v)}")
        return " | ".join(parts)

    def _render_list_as_md(self, label: str | None, items: list) -> str:
        """Render every item in a list as readable markdown bullets without JSON dumps."""
        header = f"**{label}** ({len(items)} items total):\n\n" if label else f"({len(items)} items total):\n\n"
        blocks = [header]
        for i, item in enumerate(items, 1):
            if isinstance(item, (dict, list)):
                nested_lines = self._render_markdown_node(item, indent_level=1)
                blocks.append("\n".join([f"- **[{i}]**"] + nested_lines))
            else:
                blocks.append(f"- **[{i}]** {self._format_full_markdown_value(item)}")
        return "\n".join(blocks)

    def _format_full_markdown_value(self, value) -> str:
        return str(value)

    def _is_empty_markdown_value(self, value) -> bool:
        return value is None or value == "" or value == {} or value == []

    def _render_markdown_node(self, value, indent_level: int, label: str | None = None) -> list[str]:
        indent = "  " * indent_level

        if isinstance(value, dict):
            lines = []
            child_indent = indent_level
            if label is not None:
                lines.append(f"{indent}- **{label}**:")
                child_indent += 1

            for key, nested_value in value.items():
                if self._is_empty_markdown_value(nested_value):
                    continue
                lines.extend(self._render_markdown_node(nested_value, child_indent, str(key)))
            return lines

        if isinstance(value, list):
            lines = []
            child_indent = indent_level
            if label is not None:
                lines.append(f"{indent}- **{label}**:")
                child_indent += 1

            item_indent = "  " * child_indent
            for index, item in enumerate(value, 1):
                if self._is_empty_markdown_value(item):
                    continue
                if isinstance(item, (dict, list)):
                    lines.append(f"{item_indent}- **[{index}]**")
                    lines.extend(self._render_markdown_node(item, child_indent + 1))
                else:
                    lines.append(f"{item_indent}- **[{index}]** {self._format_full_markdown_value(item)}")
            return lines

        if label is not None:
            return [f"{indent}- **{label}**: {self._format_full_markdown_value(value)}"]
        return [f"{indent}- {self._format_full_markdown_value(value)}"]


    def _get_markdown_formatting_guide(self) -> str:
        """Return a Markdown formatting guide injected into direct-mode system prompts."""
        return """
## Markdown Formatting Guidelines

You MUST follow these formatting rules to ensure beautiful, readable output:

### 1. Structure & Organization
- Use clear heading hierarchy: # for main title, ## for sections, ### for subsections
- Add blank lines between different content blocks for better readability
- Group related information together

### 2. Links
- **Inline links**: Use `[descriptive text](URL)` format
- Make link text meaningful and descriptive (not just "click here")
- Example: `[OpenAI Documentation](https://platform.openai.com/docs)`

### 3. Images
- Display images using: `![alt text](image_URL)`
- Always provide meaningful alt text
- If multiple images, consider organizing them in a list or grid pattern
- Example: `![Product Screenshot](https://example.com/image.jpg)`

### 4. Lists
- Use `-` for unordered lists (more visually appealing than `*`)
- Use numbered lists `1.` for sequential steps
- Add space after list markers
- Indent sub-items with 2-4 spaces
- **Long list handling**: If the tool result is a JSON array with many items, provide a concise summary with count statistics (e.g., total count, counts per category) instead of enumerating every item.

### 5. Emphasis
- Use **bold** for important terms: `**text**`
- Use *italic* for emphasis: `*text*`
- Use `code` for technical terms: `` `code` ``

### 6. Tables (for structured data)
- Use tables for comparing information

### 7. CRITICAL Rules
- **Long list handling**: If the tool result is a JSON list with many items, summarize with count statistics (total, per-category counts) rather than listing each item.
- **No source sections**: Do NOT add "Sources:", "References:", or "Resources:" sections at the end
- **Inline links only**: Integrate links naturally within the content, not as a separate list at the bottom
- **No code block wrapper**: Output DIRECT Markdown content, do NOT wrap your entire response in a code block

**Remember**: Your goal is to make the content scannable, visually appealing, and easy to read.
"""
    def expand_prompt(self, prompt):
        """
        Expands the prompt with the tools available.
        """
        tools_str = self.get_tools_description()
        prompt += f"""
        You can use the following tools and MCPs:
        {tools_str}
        """
        return prompt

    def is_query_and_body_empty(self) -> bool:
        """Return True if path, query, and body in tool_info.params are empty."""
        _, tool_info = self.knowledgeTool
        if not tool_info:
            return True

        try:
            params_data = _coerce_json_object(tool_info.params, "tool_info.params")
            return _is_path_query_body_empty(params_data)
        except ValueError:
            return False

    def generate_fixed_system_prompt(self) -> str:
        """Generate system prompt for fixed (no-parameter) tools."""
        knowledge_item, tool_info = self.knowledgeTool
        self.logger.info(f"knowledge item:{knowledge_item} - tool:{tool_info}")

        # 鑾峰彇缇庡浗涓滈儴鏃跺尯鐨勫綋鍓嶆椂闂?
        eastern_tz = ZoneInfo("America/New_York")
        eastern_time = datetime.now(eastern_tz)

        # 鏍煎紡鍖栦负瀛楃涓诧紙鍖呭惈鏃跺尯淇℃伅锛?
        time_str = eastern_time.strftime("%Y-%m-%d %H:%M:%S %Z")

        # 鑾峰彇knowledge item鐨刟nswer浣滀负涓婁笅鏂?
        context = ""
        if knowledge_item and hasattr(knowledge_item, 'answer') and knowledge_item.answer:
            context = f"""

        Context from knowledge base:
        {knowledge_item.answer}

        Use this context to better understand the task and provide more accurate responses.
        """

        if not tool_info:
            system_prompt = f"""

            You are an intelligent API-enabled assistant. Current time is {time_str}.
            {context}

            If no relevant knowledge is available to complete the user's task, clearly inform the user that no matching knowledge was found and suggest checking the community for shared knowledge or tools that may solve the problem.

            If a tool response indicates that the user is not authenticated, or returns a login page, inform the user that authentication is required before the task can be executed.

            In this case, always append the following tag at the end of your response:

            <Knowledge tool not logged in>

            """
            return system_prompt

        tool_title = tool_info.title
        tool_description = None
        if tool_info.description:
            tool_description = tool_info.description
        else:
            tool_description = tool_title

        # 瑙ｆ瀽宸ュ叿鍙傛暟淇℃伅
        tool_params_info = ""
        if tool_info.params:
            try:
                params_data = json.loads(tool_info.params)
                if isinstance(params_data, dict):
                    tool_params_info = "Tool parameters: user_id, query_id\n"
                    for param_name, param_type in params_data.items():
                        if param_name in ("method", "content-type", "header"):
                            continue
                        tool_params_info += f"  - {param_name} ({param_type})\n"
                else:
                    tool_params_info = f"Tool parameters: {tool_info.params}"
            except json.JSONDecodeError:
                tool_params_info = f"Tool parameters: {tool_info.params}"

        system_prompt = f"""
        You are an intelligent assistant capable of deciding when and how to use APIs to complete tasks.
        {context}

        Based on the user's request and the available context, decide whether invoking a tool is necessary.

        If a tool is required, use the following tool:

        Tool: {tool_title}
        Purpose: {tool_description}
        Input parameters: {tool_params_info}

        Execute the tool with the appropriate parameters and generate the final response strictly based on the tool's output.

        If the task can be completed without invoking the tool, respond directly to the user without calling any tool.

        Do not fabricate tool results. Do not assume tool behavior beyond the provided output.

        Do not return tool parameters, such as the user id and query id.
        Do NOT reveal any API keys, tokens, header values, or authentication credentials in your response.
        """
        # return self.expand_prompt(system_prompt)
        return system_prompt

    def generate_template_system_prompt(self) -> str:
        """Generate system prompt for template-parameter tools (LLM fills params)."""
        knowledge_item, tool_info = self.knowledgeTool
        self.logger.info(f"knowledge item:{knowledge_item} - tool:{tool_info}")

        # 鑾峰彇缇庡浗涓滈儴鏃跺尯鐨勫綋鍓嶆椂闂?
        eastern_tz = ZoneInfo("America/New_York")
        eastern_time = datetime.now(eastern_tz)

        # 鏍煎紡鍖栦负瀛楃涓诧紙鍖呭惈鏃跺尯淇℃伅锛?
        time_str = eastern_time.strftime("%Y-%m-%d %H:%M:%S %Z")

        # 鑾峰彇knowledge item鐨刟nswer浣滀负涓婁笅鏂?
        context = ""
        if knowledge_item and hasattr(knowledge_item, 'answer') and knowledge_item.answer:
            context = f"""

        Context from knowledge base:
        {knowledge_item.answer}

        Use this context to better understand the task and provide more accurate responses.
        """

        if not tool_info:
            system_prompt = f"""

            You are an intelligent API-enabled assistant. Current time is {time_str}.
            {context}

            If no relevant knowledge is available to complete the user's task, clearly inform the user that no matching knowledge was found and suggest checking the community for shared knowledge or tools that may solve the problem.

            If a tool response indicates that the user is not authenticated, or returns a login page, inform the user that authentication is required before the task can be executed.

            In this case, always append the following tag at the end of your response:

            <Knowledge tool not logged in>

            """
            return system_prompt

        tool_title = tool_info.title
        tool_description = None
        if tool_info.description:
            tool_description = tool_info.description
        else:
            tool_description = tool_title

        tool_params_info = "tool requires three parameters:user id - query id - params\n"
        if tool_info.push == 2:
            params_call_instruction = """
        3. params: A JSON object containing ONLY the API request parameters (template below).
           Pass params as a structured object in the tool call, not as a JSON-encoded string.
           Do NOT wrap the whole params object in quotes.
            """
            params_output_rule = """
           - When invoking the tool, provide params as a JSON object, not a string
           - The params object must be valid JSON and must not contain extra closing braces
            """
        else:
            params_call_instruction = """
        3. params: A valid JSON string containing ONLY the API request parameters (template below)
            """
            params_output_rule = """
           - The params string must contain valid, strictly formatted JSON
            """

        path_field_semantics = ""
        path_replacement_rules = ""
        if tool_info.push == 2:
            path_field_semantics = """
           - path contains a relative URL path that will be appended to the configured tool URL
            """
            path_replacement_rules = """
           - For path placeholders like `{applicationNumberText}`, replace the placeholder with the exact value from the user's request or the knowledge context
           - Preserve the rest of the path string, including slashes and fixed suffixes such as `/document`
           - Do not leave unresolved `{...}` placeholders in path. If the required value is missing, ask the user for it instead of calling the tool
           - path must remain a relative URL path, never an absolute URL
            """

        system_prompt = f"""
        You are an intelligent assistant capable of deciding when and how to use APIs to complete tasks.
        {context}

        Based on the user's request and the available context, decide whether invoking a tool is necessary.

        If a tool is required, use the following tool:

        Tool: {tool_title}
        Purpose: {tool_description}
        Input parameters: {tool_params_info}

        IMPORTANT: The tool takes THREE separate parameters:
        1. user_id: The user identifier (provided in the user prompt, DO NOT include in params)
        2. query_id: The query identifier (provided in the user prompt, DO NOT include in params)
        {params_call_instruction}

        The third parameter "params" template: {self._sanitize_params_for_llm(tool_info.params)}

        Your task is to analyze the user's input and modify the third parameter "params" template according to the user's specific requirements. Generate a new JSON object containing only the parameters that need to be changed or specified based on the user's request.

        You MUST follow all rules below without exception:

        1. You may ONLY modify existing values in the JSON.
           - DO NOT add new fields, If a field is empty in the template, then leave it empty.
           - DO NOT change the JSON structure or nesting
           - CRITICAL: DO NOT include user_id or query_id in the params JSON - these are separate parameters

        2. Field semantics:
           - method MUST remain unchanged
           {path_field_semantics}
           - query contains URL query parameters
           - header contains HTTP headers
           - body contains the HTTP request body

        3. Value replacement rules:
           - Replace a value only if the user query clearly maps to the meaning of an existing field
           - If the user query does not mention or imply a field, keep its original value unchanged
           - If the original tool params contain api-key, api_key, apikey, x-api-key, or another API key field, preserve that key and its value exactly in the generated params
           - Do NOT infer or invent information not explicitly expressed by the user
           - DO NOT extract or infer user_id or query_id from the user's request into the params JSON
           {path_replacement_rules}

        4. Output rules:
           - Output ONLY the final, complete JSON for the params parameter
           - DO NOT include explanations, reasoning, comments, or formatting outside JSON
           {params_output_rule}
           - DO NOT include user_id or query_id fields in the JSON output

        Execute the tool with the appropriate parameters and generate the final response strictly based on the tool's output.

        If the task can be completed without invoking the tool, respond directly to the user without calling any tool.

        Do not fabricate tool results. Do not assume tool behavior beyond the provided output.

        Do not return tool parameters, such as the user id and query id.
        Do NOT reveal any API keys, tokens, header values, or authentication credentials in your response.

        ## Markdown Formatting Requirements

        When generating your response based on the tool's output, you MUST format it beautifully using Markdown:

        ### Essential Formatting Rules:

        1. **Structure**: Use clear heading hierarchy (## for main sections, ### for subsections)
        2. **Links**: Convert ALL URLs to descriptive links: `[meaningful text](URL)`
        3. **Images**: Display images using: `![description](image_URL)`
        4. **Lists**: Use `-` for bullet points, `1.` for numbered lists
        5. **Emphasis**: Use **bold** for key terms, *italic* for emphasis, `code` for technical terms
        6. **Tables**: Use tables for structured data comparison
        7. **Spacing**: Add blank lines between content blocks for readability
        8. **Code blocks**: Use fenced code blocks with language specification when showing code
        9. **Long list handling**: If the tool result is a JSON array with many items, provide a concise summary with count statistics (e.g., total count, counts per category) instead of enumerating every item.

        ### Response Structure Template:

        Your response should follow this structure (but output DIRECTLY, not in a code block):

        ## [Main Topic]

        [Brief summary of what the tool returned]

        ### Key Information
        - Important point 1 (with details)
        - Important point 2 (with details)
        - ... (if the tool result is a long JSON list, show a summary with count statistics instead)

        ### Details
        [Organized detailed content 鈥?if the tool result is a long JSON list, show a summary with count statistics]

        [Display images inline where relevant]
        ![Image Description](image_URL)

        **CRITICAL OUTPUT FORMAT**:
        - Output your response as DIRECT Markdown content
        - Do NOT wrap your entire response in a code block
        - Do NOT start with ```markdown or ```
        - Start directly with Markdown formatting (e.g., ## Title or plain text)
        - Only use code blocks for actual code snippets within your content, not for the entire response

        **CRITICAL CONTENT RULES**:
        - If the tool result is a long JSON list, provide a concise summary with count statistics instead of listing every item
        - Do NOT add a "Resources", "Sources", or "References" section at the end of your response
        - Do NOT create a separate list of links at the bottom
        - Integrate all links naturally within the content itself

        **IMPORTANT**: Make your response visually appealing, easy to scan, and professionally formatted. Transform raw data into a beautiful, user-friendly presentation while ensuring ALL content from the tool result is displayed.
        """

        return system_prompt

    def generate_system_prompt(self, tool_data: str = "") -> str:
        """Select and return the appropriate system prompt based on push mode."""
        _, tool_info = self.knowledgeTool

        # 鏍规嵁tool_info.push鐨勫€奸€夋嫨涓嶅悓绯荤粺鎻愮ず璇?
        if not tool_info:
            return self.generate_fixed_system_prompt()

        if tool_info.push == 1:
            if tool_data and tool_data.strip():  # 鍒ゆ柇tool_data鏄惁闈炵┖
                return self.generate_frontend_tool_direct_system_prompt(tool_data)
            else:
                return self.generate_template_system_prompt()
        elif tool_info.push == 2:
            if self.is_query_and_body_empty():
                return self.generate_backend_tool_direct_system_prompt()
            else:
                return self.generate_template_system_prompt()
        elif tool_info.push == 3:
            return self.generate_frontend_tool_direct_system_prompt(tool_data)
        else:
            # 榛樿鎯呭喌涓嬪浐瀹氱殑绯荤粺鎻愮ず璇?
            return self.generate_template_system_prompt()



    def generate_user_prompt(self, prompt, user_id, query_id) -> str:
        user_prompt = f"""
        {prompt},
        user id is {user_id},
        query id is {query_id},
        """
        self.logger.info(f"user prompt:{user_prompt}")

        return user_prompt

    def generate_frontend_tool_direct_system_prompt(self, tool_data: str) -> str:
        """Generate system prompt from pre-fetched tool_data without making an HTTP request."""
        # self.logger.info(f"generate_frontend_tool_direct_system_prompt - tool_data: {tool_data}")

        try:
            # 鑾峰彇knowledge item鐨刟nswer浣滀负涓婁笅鏂?
            knowledge_item, _ = self.knowledgeTool
            context = ""
            if knowledge_item and hasattr(knowledge_item, 'answer') and knowledge_item.answer:
                context = f"""

            Context from knowledge base:
            {knowledge_item.answer}

            Use this context to better understand the task and provide more accurate responses.
            """

            # 瑙ｆ瀽 tool_data 鍐呭
            if "text/html" in tool_data:
                # 濡傛灉鏄?HTML 鍐呭锛屼娇鐢ㄥ叕鐢ㄦ柟娉曟竻鐞嗘枃鏈?
                result_str = clean_html_text(tool_data)
            else:
                # 灏濊瘯灏?tool_data 瑙ｆ瀽涓?JSON
                try:
                    result_data = json.loads(tool_data)
                    # 濡傛灉缁撴灉鏄竴涓?list锛屾彁鍙?raw_items 浠ヤ緵鍚庣画鎵瑰鐞?
                    if isinstance(result_data, list) and result_data:
                        raw_items = result_data
                    elif isinstance(result_data, dict):
                        raw_items = next(
                            (v for v in result_data.values() if isinstance(v, list) and v),
                            None
                        )
                    else:
                        raw_items = None

                    if raw_items:
                        self._pending_raw_items = raw_items
                        result_str = (
                            f"The query returned {len(raw_items)} items. "
                            f"Please write a brief 2鈥? sentence summary of what was found. "
                            f"The complete list will be analyzed and displayed item by item automatically 鈥?"
                            f"do NOT enumerate the items yourself."
                        )
                    elif isinstance(result_data, dict):
                        # dict 鍖呰涓哄崟鍏冪礌鍒楄〃锛岃蛋 Phase 2 蹇犲疄杈撳嚭
                        pruned = _prune_item_for_llm(result_data)
                        self._pending_raw_items = [pruned]
                        result_str = (
                            "The query returned structured data. "
                            "Please write a brief 1-2 sentence summary of what was found. "
                            "The complete data will be displayed automatically - "
                            "do NOT enumerate the fields yourself."
                        )
                    else:
                        result_str = json.dumps(result_data, ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    # 濡傛灉涓嶆槸鏈夋晥鐨?JSON锛屽垯鐩存帴浣跨敤鍘熷鍐呭
                    result_str = tool_data

            # 鑾峰彇鏍煎紡鍖栨寚鍗?
            formatting_guide = self._get_markdown_formatting_guide()

            # 鏋勯€犵郴缁熸彁绀鸿瘝
            system_prompt = f"""
Act as a self-contained intelligent assistant. Follow these instructions strictly:
{context}

## Core Instructions

1.  **Core Principle:** You must perform tasks and generate answers using **only** the data, text, or context that I provide to you within this chat.
2.  **No External Access:** Do not attempt to invoke or use any internal or external tools (such as search functions, code interpreters, calculators, or knowledge retrieval from your base training data) to complete the task.
3.  **Direct Processing:** Analyze, reason, and respond directly based on the provided input. If the necessary information is not contained in my messages, state that clearly instead of making assumptions.
4.  **Privacy Protection:** Do NOT include or output any `user_id`, `query_id`, or similar internal identifiers in your response. These are system metadata and should never appear in user-facing output.
5.  **Long list handling:** If the input data is a JSON array with many items, provide a concise summary with count statistics (e.g., total count, counts per category) instead of enumerating every item.
6.  **No Source Sections:** Do NOT add a "Sources", "References", or "Resources" section at the end of your response. Do NOT create a separate list of links at the bottom.

{formatting_guide}

## Input Data

{result_str}

## Your Task

Generate a beautiful, well-formatted Markdown response based on the above data. Follow ALL the formatting guidelines provided above. Make your response:
- Visually appealing with proper structure
- Easy to scan with clear headings
- Rich with properly formatted links and images (integrated naturally within content)
- Professional and polished
- If the data is a long JSON list, summarize with count statistics rather than listing every item

**CRITICAL OUTPUT FORMAT**:
- Output your response as DIRECT Markdown content
- Do NOT wrap your entire response in a code block
- Do NOT start with ```markdown or ```
- Start directly with Markdown formatting (e.g., ## Title or plain text)
- Only use code blocks for actual code snippets within your content, not for the entire response

**CRITICAL CONTENT RULES**:
- If the Input Data is a long JSON list, provide a summary with count statistics instead of enumerating every item
- Do NOT add a separate "Sources" or "References" section at the end
- Integrate all links naturally within the content

Begin your response now:
            """
            return system_prompt

        except Exception as e:
            self.logger.error(f"Failed to generate frontend tool direct system prompt: {str(e)}")
            return self.generate_template_system_prompt()

    def generate_backend_tool_direct_system_prompt(self) -> str:
        """Make the HTTP request from tool_info, embed the result in the system prompt, and return it."""
        # 鑾峰彇宸ュ叿淇℃伅
        knowledge_item, tool_info = self.knowledgeTool
        self.logger.info(f"generate_tool_direct_system_prompt - tool:{tool_info}")

        try:
            # 鑾峰彇knowledge item鐨刟nswer浣滀负涓婁笅鏂?
            context = ""
            if knowledge_item and hasattr(knowledge_item, 'answer') and knowledge_item.answer:
                context = f"""

            Context from knowledge base:
            {knowledge_item.answer}

            Use this context to better understand the task and provide more accurate responses.
            """

            # 浠?tool_info 涓彁鍙?URL 鍜屽弬鏁版ā鏉?
            url = tool_info.url
            params_data = _coerce_json_object(tool_info.params, "tool_info.params")
            url = _append_path_to_url(url, params_data.get("path", ""))

            # 鑾峰彇 HTTP 鏂规硶鍜?Content-Type
            method = params_data.get("method", "GET").upper()
            content_type = params_data.get("Content-Type", "application/json")

            # 鍑嗗璇锋眰澶?
            headers = {
                "Content-Type": content_type
            }

            # params_data
            user_headers = params_data.get("header", {})
            if isinstance(user_headers, dict):
                headers.update(user_headers)

            # 娣诲姞鏃堕棿鎴冲弬鏁扮粫杩?CDN 缂撳瓨
            cache_bust_params = {"_t": str(int(time.time() * 1000))}

            # 鍙戣捣 HTTP 璇锋眰
            if method not in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
                raise ValueError(f"Unsupported HTTP method: {method}")
            request_kwargs = {
                "params": cache_bust_params,
                "headers": headers,
            }
            if method in {"POST", "PUT", "PATCH"}:
                request_kwargs["json"] = {}
            response = outbound_http.request(method, url, purpose="backend_tool_direct", **request_kwargs)

            # 澶勭悊鍝嶅簲缁撴灉
            if response.status_code == 200:

                content_type = response.headers.get("Content-Type", "").lower()

                if "text/html" in content_type:
                    # 浣跨敤BeautifulSoup绉婚櫎HTML鏍囩
                    result_str = BeautifulSoup(response.content, "html.parser").get_text()
                elif "application/xml" in content_type or "text/xml" in content_type:
                    # 澶勭悊XML鏍煎紡鍝嶅簲
                    try:
                        # 浣跨敤BeautifulSoup瑙ｆ瀽XML骞舵彁鍙栨枃鏈唴瀹?
                        soup = BeautifulSoup(response.content, "xml")
                        # 绉婚櫎XML鏍囩锛屽彧淇濈暀鏂囨湰鍐呭
                        result_str = soup.get_text()
                        # 濡傛灉XML瑙ｆ瀽澶辫触鎴栧唴瀹逛负绌猴紝浣跨敤鍘熷鍐呭
                        if not result_str.strip():
                            result_str = response.text
                    except Exception as xml_e:
                        self.logger.warning(f"XML parsing failed: {str(xml_e)}, using raw content")
                        result_str = response.text
                else:
                    try:
                        result_data = response.json() if response.content else {}
                        if isinstance(result_data, list) and result_data:
                            raw_items = result_data
                        elif isinstance(result_data, dict):
                            raw_items = next(
                                (v for v in result_data.values() if isinstance(v, list) and v),
                                None
                            )
                        else:
                            raw_items = None

                        if raw_items:
                            self._pending_raw_items = raw_items
                            result_str = (
                                f"The query returned {len(raw_items)} items. "
                                f"Please write a brief 2鈥? sentence summary of what was found. "
                                f"The complete list will be analyzed and displayed item by item automatically 鈥?"
                                f"do NOT enumerate the items yourself."
                            )
                        elif isinstance(result_data, dict):
                            # dict 鍖呰涓哄崟鍏冪礌鍒楄〃锛岃蛋 Phase 2 蹇犲疄杈撳嚭
                            pruned = _prune_item_for_llm(result_data)
                            self._pending_raw_items = [pruned]
                            result_str = (
                                "The query returned structured data. "
                                "Please write a brief 1-2 sentence summary of what was found. "
                                "The complete data will be displayed automatically - "
                                "do NOT enumerate the fields yourself."
                            )
                        else:
                            result_str = json.dumps(result_data, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError:
                        # 濡傛灉JSON瑙ｆ瀽澶辫触锛屼娇鐢ㄥ師濮嬪搷搴斿唴瀹?
                        result_str = response.text if response.text else "Empty response"
            else:
                result_str = f"Request failed, status code: {response.status_code}"

            # 鑾峰彇鏍煎紡鍖栨寚鍗?
            formatting_guide = self._get_markdown_formatting_guide()

            # 鏋勯€犵郴缁熸彁绀鸿瘝
            system_prompt = f"""
Act as a self-contained intelligent assistant. Follow these instructions strictly:
{context}

## Core Instructions

1.  **Core Principle:** You must perform tasks and generate answers using **only** the data, text, or context that I provide to you within this chat.
2.  **No External Access:** Do not attempt to invoke or use any internal or external tools (such as search functions, code interpreters, calculators, or knowledge retrieval from your base training data) to complete the task.
3.  **Direct Processing:** Analyze, reason, and respond directly based on the provided input. If the necessary information is not contained in my messages, state that clearly instead of making assumptions.
4.  **Privacy Protection:** Do NOT include or output any `user_id`, `query_id`, or similar internal identifiers in your response. These are system metadata and should never appear in user-facing output.
5.  **Long list handling:** If the input data is a JSON array with many items, provide a concise summary with count statistics (e.g., total count, counts per category) instead of enumerating every item.
6.  **No Source Sections:** Do NOT add a "Sources", "References", or "Resources" section at the end of your response. Do NOT create a separate list of links at the bottom.

{formatting_guide}

## Input Data

{result_str}

## Your Task

Generate a beautiful, well-formatted Markdown response based on the above data. Follow ALL the formatting guidelines provided above. Make your response:
- Visually appealing with proper structure
- Easy to scan with clear headings
- Rich with properly formatted links and images (integrated naturally within content)
- Professional and polished
- If the data is a long JSON list, summarize with count statistics rather than listing every item

**CRITICAL OUTPUT FORMAT**:
- Output your response as DIRECT Markdown content
- Do NOT wrap your entire response in a code block
- Do NOT start with ```markdown or ```
- Start directly with Markdown formatting (e.g., ## Title or plain text)
- Only use code blocks for actual code snippets within your content, not for the entire response

**CRITICAL CONTENT RULES**:
- If the Input Data is a long JSON list, provide a summary with count statistics instead of enumerating every item
- Do NOT add a separate "Sources" or "References" section at the end
- Integrate all links naturally within the content

Begin your response now:
            """
            return system_prompt

        except Exception as e:
            self.logger.error(f"Failed to generate tool direct system prompt: {str(e)}")
            return self.generate_system_prompt(result_str)

    async def get_dynamic_tools(self) -> list:
        try:
            tools = {}
            # 濡傛灉鏈夌煡璇嗗簱涓殑宸ュ叿淇℃伅锛屽垯鍔ㄦ€佹瀯寤篗CP宸ュ叿
            if hasattr(self, 'knowledgeTool') and self.knowledgeTool:
                # 鑾峰彇宸ュ叿淇℃伅
                knowledge_item, tool_info = self.knowledgeTool

                if tool_info:

                    # 鍔ㄦ€佸垱寤哄伐鍏峰嚱鏁?
                    def dynamic_frontend_tool_function(user_id: str, query_id: str, params: str):
                        self.logger.info(f"dynamic_frontend_tool_function user id is {user_id} - query id is {query_id} - param is {params}")
                        try:
                            # 杩炴帴Redis
                            redis_conn = get_redis_connection()

                            # 鏋勯€燫edis閿?
                            redis_key = f"tool_request_{query_id}_{user_id}"

                            # param_dict = {"origin_params": json.loads(tool_info.params)}
                            # if params:
                            #     # 灏嗗弬鏁拌浆鎹负JSON骞跺瓨鍌ㄥ埌Redis
                            #     param_dict["llm_params"] = params

                            params_json = json.dumps(params)

                            redis_conn.set(redis_key, params_json, ex=1200)

                            # 杞璇诲彇tool_response_{query_id}
                            response_key = f"tool_response_{query_id}_{user_id}"
                            timeout = 300  # 5鍒嗛挓瓒呮椂
                            interval = 1  # 姣忕鏌ヨ涓€娆?
                            elapsed = 0

                            while elapsed < timeout:
                                response_value = redis_conn.get(response_key)

                                if response_value is not None:
                                    # 鎴愬姛鑾峰彇鍒板搷搴斿€?
                                    return response_value
                                # 绛夊緟1绉掑悗鍐嶆灏濊瘯
                                time.sleep(interval)
                                elapsed += interval

                            # 瓒呮椂鏈幏鍙栧埌鍝嶅簲鍊?
                            return None
                        except Exception as e:
                            # 濡傛灉Redis鎿嶄綔澶辫触锛岃褰曟棩蹇椾絾浠嶇户缁墽琛屽伐鍏?
                            self.logger.error(f"Failed to write to Redis: {str(e)}")
                            return None

                    def dynamic_backend_tool_function(user_id: str, query_id: str, params: Dict[str, Any] | str):
                        self.logger.info(f"dynamic_backend_tool_function user id is {user_id} - query id is {query_id} - param is {params}")
                        tool_result = execute_backend_tool_request(tool_info, params)
                        raw_items = tool_result.get("raw_items")
                        if raw_items:
                            list_count = len(raw_items)
                            self._pending_raw_items = raw_items
                            return (
                                f"The query returned {list_count} items. "
                                f"Please write a brief 2-3 sentence summary of what was found. "
                                f"The complete list will be analyzed and displayed item by item automatically - "
                                f"do NOT enumerate the items yourself."
                            )
                        data = tool_result.get("data")
                        if isinstance(data, dict):
                            # dict 鍖呰涓哄崟鍏冪礌鍒楄〃锛岃蛋 Phase 2 蹇犲疄杈撳嚭
                            pruned = _prune_item_for_llm(data)
                            self._pending_raw_items = [pruned]
                            return (
                                "The query returned structured data. "
                                "Please write a brief 1-2 sentence summary of what was found. "
                                "The complete data will be displayed automatically - "
                                "do NOT enumerate the fields yourself."
                            )
                        if isinstance(data, list):
                            return json.dumps(data, ensure_ascii=False, indent=2)
                        return data
                        # 浠巘ool_info涓幏鍙朥RL
                        url = tool_info.url

                        # 瑙ｆ瀽鍙傛暟JSON
                        params_data = _coerce_json_object(tool_info.params, "tool_info.params")
                        user_params = _coerce_json_object(params, "LLM tool params")
                        url = _append_path_to_url(
                            url,
                            user_params.get("path", params_data.get("path", ""))
                        )

                        # 鑾峰彇HTTP鏂规硶鍜孋ontent-Type
                        method = params_data.get("method", "GET").upper()
                        content_type = params_data.get("Content-Type", "application/json")

                        # Prepare HTTP headers from server-side tool_info.params only.
                        # Never use LLM-provided header values 鈥?the LLM only sees
                        # sanitised (****) placeholders and must not control auth headers.
                        headers = {
                            "Content-Type": content_type
                        }
                        server_headers = params_data.get("header", {})
                        if isinstance(server_headers, dict):
                            headers.update(server_headers)

                        request_params = user_params.get("query")
                        request_body = user_params.get("body")

                        # 娣诲姞鏃堕棿鎴冲弬鏁扮粫杩?CDN 缂撳瓨
                        if request_params is None:
                            request_params = {}
                        request_params["_t"] = str(int(time.time() * 1000))
                        self.logger.info(f"tool url is {url}")
                        if method not in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
                            raise ValueError(f"Unsupported HTTP method: {method}")
                        request_kwargs = {
                            "params": request_params,
                            "headers": headers,
                        }
                        if method in {"POST", "PUT", "PATCH"}:
                            request_kwargs["json"] = request_body
                        response = outbound_http.request(method, url, purpose="backend_tool", **request_kwargs)

                        # 鎵撳嵃 response 淇℃伅
                        self.logger.info(f"Response status code: {response.status_code}")
                        self.logger.info(f"Response headers: {response.headers}")
                        # self.logger.info(f"Response content: {response.text}")
                        # 澶勭悊鍝嶅簲缁撴灉
                        if response.status_code == 200:
                            content_type = response.headers.get("Content-Type", "").lower()

                            if "text/html" in content_type:
                                # HTML 鍐呭锛屼娇鐢?BeautifulSoup 娓呯悊
                                result = BeautifulSoup(response.content, "html.parser").get_text()
                            elif "application/xml" in content_type or "text/xml" in content_type:
                                # XML 鍐呭锛屽皾璇曡В鏋愬苟鎻愬彇鏂囨湰
                                try:
                                    soup = BeautifulSoup(response.content, "xml")
                                    result = soup.get_text()
                                    if not result.strip():
                                        result = response.text
                                except Exception as xml_e:
                                    self.logger.warning(f"XML parsing failed: {str(xml_e)}, using raw content")
                                    result = response.text
                            else:
                                # JSON 鎴栧叾浠栨牸寮?
                                try:
                                    result_data = response.json() if response.content else None
                                    if isinstance(result_data, (dict, list)):
                                        # Extract raw items list for batch LLM analysis
                                        if isinstance(result_data, list) and result_data:
                                            raw_items = result_data
                                        elif isinstance(result_data, dict):
                                            raw_items = next(
                                                (v for v in result_data.values() if isinstance(v, list) and v),
                                                None
                                            )
                                        else:
                                            raw_items = None

                                        if raw_items:
                                            list_count = len(raw_items)
                                            # Store raw items; invoke_agent will batch-analyze them via LLM
                                            self._pending_raw_items = raw_items
                                            result = (
                                                f"The query returned {list_count} items. "
                                                f"Please write a brief 2鈥? sentence summary of what was found. "
                                                f"The complete list will be analyzed and displayed item by item automatically 鈥?"
                                                f"do NOT enumerate the items yourself."
                                            )
                                        else:
                                            result = json.dumps(result_data, ensure_ascii=False, indent=2)
                                    else:
                                        result = result_data
                                except json.JSONDecodeError:
                                    # JSON 瑙ｆ瀽澶辫触锛岃繑鍥炲師濮嬫枃鏈?
                                    result = response.text if response.text else None
                        else:
                            # 璇锋眰澶辫触锛岃繑鍥為敊璇俊鎭?
                            result = f"Request failed, status code: {response.status_code}"

                        return result

                    # 娓呯悊宸ュ叿鍚嶇О浠ョ鍚圓PI瑕佹眰
                    tool_name = tool_info.title if tool_info.title else "dynamic_knowledge_tool"
                    # 鍙繚鐣欏瓧姣嶃€佹暟瀛椼€佷笅鍒掔嚎鍜岃繛瀛楃
                    cleaned_tool_name = re.sub(r'[^a-zA-Z0-9_-]', '_', tool_name)
                    # 纭繚鍚嶇О涓嶄负绌?
                    if not cleaned_tool_name or cleaned_tool_name.strip() == "":
                        cleaned_tool_name = "dynamic_knowledge_tool"

                    # 鏍规嵁tool_info.push鐨勫€奸€夋嫨涓嶅悓鐨勫伐鍏峰嚱鏁?
                    if tool_info.push == 1 or tool_info.push == 3:
                        tool_func = dynamic_frontend_tool_function
                    elif tool_info.push == 2:
                        tool_func = dynamic_backend_tool_function
                    else:
                        # 榛樿鎯呭喌涓嬩娇鐢ㄥ墠绔伐鍏峰嚱鏁?
                        tool_func = dynamic_frontend_tool_function

                    args_schema = (
                        DynamicBackendToolFunction
                        if tool_info.push == 2
                        else DynamicToolFunction
                    )

                    dynamic_tool = StructuredTool.from_function(
                        func=tool_func,
                        name=cleaned_tool_name,
                        description=tool_info.description if tool_info.description else "Dynamic knowledge tool",
                        args_schema=args_schema
                    )

                    # 鍚堝苟鍔ㄦ€佸伐鍏?
                    tools = [dynamic_tool]
                else:
                    # 濡傛灉娌℃湁鍔ㄦ€佸伐鍏蜂俊鎭紝浣跨敤榛樿閰嶇疆
                    tools = None
            else:
                # 濡傛灉娌℃湁鍔ㄦ€佸伐鍏蜂俊鎭紝浣跨敤榛樿閰嶇疆
                tools = None

            self.logger.info(f"tools{tools}")
            return tools
        except Exception as e:
            raise Exception(f"get_tool failed: {str(e)}") from e

    async def get_tools(self, tool_data: str = "") -> list:
        """Select and return the appropriate LangChain tool list based on push mode."""
        _, tool_info = self.knowledgeTool
        if not tool_info:
            return []

        tools = []
        # 鏍规嵁tool_info.push鐨勫€奸€夋嫨涓嶅悓绯荤粺鎻愮ず璇?
        if tool_info.push == 1:
            # If tool_data is already provided, the system prompt contains the
            # pre-fetched result. Do NOT give the LLM a LangChain tool 鈥?
            # it would call the tool, receive a ToolMessage, and ignore the
            # pre-formatted list we placed in the system prompt.
            if tool_data and tool_data.strip():
                return tools
            if self.is_query_and_body_empty():
                return tools
            else:
                return await self.get_dynamic_tools()
        elif tool_info.push == 2:
            if self.is_query_and_body_empty():
                return tools
            else:
                return await self.get_dynamic_tools()
        elif tool_info.push == 3:
            return tools
        else:
            # 榛樿鎯呭喌涓嬪浐瀹氱殑绯荤粺鎻愮ず璇?
            return await self.get_dynamic_tools()

    async def process(self, user_id, prompt, query_id, speech_module, push_filter=None) -> str | tuple[str, str]:
        if not self.enabled:
            return "general Agent is disabled."
        self._last_user_prompt = prompt
        self._last_query_id = query_id
        conv_history = self.memory.get()
        self.knowledgeTool = await select_knowledge_tool_with_llm(
            user_id,
            prompt,
            self.llm.complete_json,
            push_filter=push_filter,
            conversation_history=conv_history,
        )
        # user_prompt = self.expand_prompt(prompt)
        user_prompt = self.generate_user_prompt(prompt, user_id, query_id)
        system_prompt = self.generate_system_prompt()
        prior = self.memory.get()
        if len(prior) > 6:
            prior = prior[:1] + prior[-1:]
        self.memory.reset(prior)
        self.memory.push('user', user_prompt)
        self.memory.push('system', system_prompt)

        self.logger.info(f"memory.get():{self.memory.get()}")
        self.tools = await self.get_tools()
        working = True
        while working == True:
            self.logger.info(f"tools:{self.tools}")
            animate_thinking("Thinking...", color="status")
            answer, reasoning = await self.llm_request()
            # exec_success, _ = self.execute_modules(answer)
            # answer = self.remove_blocks(answer)
            self.last_answer = answer
            self.status_message = "Ready"
            if len(self.blocks_result) == 0:
                working = False
        return answer, reasoning

    async def create_agent(self, user_id, prompt, query_id, tool_data, callback_handler, push_filter=None):
        #self.knowledgeTool = get_knowledge_tool(user_id,  prompt)
        self._last_user_prompt = prompt
        self._last_query_id = query_id
        if callback_handler:
            await _emit_status(callback_handler, "正在分析您的问题...")
        # Pass conversation history so the LLM routing knowledge selection
        # can understand what "杩?鏉? or "from the results above" refers to.
        conv_history = self.memory.get()
        self.knowledgeTool = await select_knowledge_tool_with_llm(
            user_id,
            prompt,
            self.llm.complete_json,
            push_filter=push_filter,
            conversation_history=conv_history,
        )
        knowledge_item, tool_info = self.knowledgeTool
        self.logger.info(
            f"Knowledge selected: id={getattr(knowledge_item, 'id', None)}, "
            f"type={getattr(knowledge_item, 'type', None) if knowledge_item else None}"
        )
        # Long task detection: type=3 knowledge triggers async Celery pipeline
        if _is_long_task_knowledge(knowledge_item):
            self.logger.info("Long task detected 鈥?returning long_task intent")
            if callback_handler:
                try:
                    await asyncio.wait_for(
                        _emit_status(callback_handler, "正在启动批量专利分析任务..."),
                        timeout=5.0,
                    )
                except Exception:
                    pass
            # Return special marker 鈥?caller (run_pipeline) handles submission
            self._long_task_intent = _build_long_task_intent(knowledge_item, tool_info)
            return self._long_task_intent
        if is_workflow_knowledge(knowledge_item):
            if callback_handler:
                await callback_handler.on_llm_new_token(
                    f"已匹配组合知识：{knowledge_item.question}\n\n"
                )
                await _emit_status(callback_handler, "正在执行组合知识流程...")
            workflow_result = await WorkflowExecutor(
                self.llm,
                status_callback=_status_callback_for(callback_handler),
            ).execute(
                workflow_spec=knowledge_item.params,
                user_prompt=prompt,
                workflow_knowledge=knowledge_item,
            )
            self.knowledgeTool = (knowledge_item, tool_info)
            self._workflow_result = workflow_result
            self.tools = []
            return None
        user_prompt = self.generate_user_prompt(prompt, user_id, query_id)
        system_prompt = self.generate_system_prompt(tool_data)
        # Keep prior conversation context so follow-up queries referencing
        # previous results (e.g. "浠庤繖3鏉′腑绛涢€夊嚭...") can be understood.
        prior = self.memory.get()
        if len(prior) > 6:
            # Keep first message + last 5 messages to bound context window
            prior = prior[:1] + prior[-1:]
        self.memory.reset(prior)
        self.memory.push('user', user_prompt)
        self.memory.push('system', system_prompt)

        self.logger.info(f"memory.get():{self.memory.get()}")
        self.tools = await self.get_tools(tool_data)

        return self.llm.openai_create(self.tools, self.memory.get(), callback_handler)


    async def _stream_workflow_final_result(self, workflow_result, callback_handler):
        output_mode = getattr(workflow_result, "output_mode", "last")
        workflow_question = getattr(workflow_result, "workflow_question", "")
        workflow_instructions = getattr(workflow_result, "workflow_instructions", "")

        if output_mode == "all":
            system_prompt = (
                "You are a professional data presenter serving non-technical readers. "
                "Below are results from multiple data queries about the same topic. "
                "Synthesize ALL results into a single coherent, complete report.\n\n"
                "CRITICAL RULES:\n"
                "1. Do NOT structure your answer by data source or query step. "
                "Present the information as one unified document 鈥?the reader must not "
                "be able to tell which piece of data came from which query.\n"
                "2. Merge duplicate information across sources 鈥?if the same fact appears "
                "in multiple results, present it once.\n"
                "3. ALL meaningful data must be preserved 鈥?do NOT summarize, abbreviate, "
                "or skip any field with a real value. This is the most important rule.\n"
                "4. Group related information logically: titles and abstracts together, "
                "dates together, people and organizations together, legal/classification "
                "together, documents and references together.\n"
                "5. Filter noise: skip API wrapper fields (errorCode, errorDesc, page, "
                "page_row, total, sort_column), empty values, '0'/'鍚? placeholder values, "
                "and internal system IDs (pid).\n"
                "6. Present as a reader-friendly document with clear section headings. "
                "Use comparison tables when comparing multiple records with shared fields.\n"
                "7. Do NOT add any meta-commentary such as 'based on the query results' or "
                "'here is the synthesized report' 鈥?just output the content directly.\n"
                "8. MANDATORY 鈥?EVERY document URL, image URL, and external link found "
                "in the source data MUST appear verbatim in your output. Even if two URLs "
                "look similar, include BOTH if they point to different resources. "
                "Use descriptive link text: [Title](URL) for documents, "
                "![Description](URL) for images. This rule overrides rule #2 鈥?never "
                "merge or skip URLs."
            )
            # Include raw_items from each step so links/images are not lost
            payload = {
                "user_request": getattr(self, "_last_user_prompt", ""),
                "workflow_question": workflow_question,
                "workflow_instructions": workflow_instructions,
                "combined_results": [s.data for s in workflow_result.steps],
                "raw_items_from_all_steps": [
                    getattr(s, "data", None) for s in workflow_result.steps
                ],
            }
        else:
            system_prompt = (
                "You are a self-contained assistant answering from a completed "
                "composed-knowledge workflow result. "
                "Use only the user request and workflow result provided here. "
                "Do not call tools, search externally, or invent missing data. "
                "Return a concise, well-formatted Markdown answer.\n\n"
                "CRITICAL: Every document URL, image URL, and external link in the "
                "workflow result MUST appear in your output. Use descriptive link text: "
                "[Title](URL) for documents, ![Description](URL) for images. "
                "Never omit a link that exists in the source data."
            )
            payload = {
                "user_request": getattr(self, "_last_user_prompt", ""),
                "workflow": {
                    "question": workflow_question,
                    "instructions": workflow_instructions,
                },
                "workflow_result": workflow_result.final_data,
                "raw_items": getattr(workflow_result, "raw_items", None),
            }

        user_content = json.dumps(payload, ensure_ascii=False, indent=2)
        await self.llm.stream_simple(
            system_prompt=system_prompt,
            user_content=user_content,
            callback_handler=callback_handler,
        )

    def _build_formatter_user_content(self, batch, url_checklist: str = "") -> str:
        batch_json = json.dumps(batch, ensure_ascii=False, indent=2)
        return (
            f"Format and analyze these {len(batch)} search result items as readable Markdown:\n\n"
            f"{url_checklist}"
            f"{batch_json}"
        )

    def _build_url_checklist(self, batch) -> str:
        batch_urls = _collect_urls_from_value(batch)
        if not batch_urls:
            return ""
        return (
            "Mandatory URL checklist. Copy every line verbatim into the corresponding item. "
            "Do not omit any URL from this checklist:\n"
            + "\n".join(f"- {url}" for url in batch_urls)
            + "\n\n"
        )

    async def _stream_deterministic_batch(self, batch, callback_handler):
        await callback_handler.on_llm_new_token(self._render_list_as_md(None, batch))

    async def _stream_formatter_batch(self, system_prompt: str, batch, callback_handler) -> None:
        await self.llm.stream_simple(
            system_prompt=system_prompt,
            user_content=self._build_formatter_user_content(
                batch,
                self._build_url_checklist(batch),
            ),
            callback_handler=callback_handler,
        )

    async def _stream_formatter_or_markdown(
        self,
        system_prompt: str,
        batch,
        callback_handler,
        label: str = "",
    ) -> None:
        """Try the formatter LLM; fall back to deterministic markdown on failure."""
        try:
            await self._stream_formatter_batch(system_prompt, batch, callback_handler)
        except Exception as exc:
            self.logger.warning(
                f"formatter LLM failed{label}; "
                f"streaming deterministic markdown fallback. error={exc}"
            )
            await self._stream_deterministic_batch(batch, callback_handler)

    async def _stream_items_individually(
        self,
        batch,
        system_prompt: str,
        callback_handler,
    ) -> None:
        """Process items one by one: LLM 鈫?markdown on failure."""
        for item_index, item in enumerate(batch, start=1):
            item_batch = [item]
            await self._stream_formatter_or_markdown(
                system_prompt, item_batch, callback_handler,
                label=f" for item {item_index}",
            )

    async def _stream_batch_with_retries(
        self,
        batch,
        system_prompt: str,
        callback_handler,
    ) -> None:
        """Try the batch in one LLM call; fall back to individual items; then markdown."""
        batch_json = json.dumps(batch, ensure_ascii=False, indent=2, default=str)
        if len(batch_json) > MAX_BATCH_JSON_CHARS_FOR_LLM:
            self.logger.info(
                "batch JSON too large for formatter LLM; "
                f"retrying by item. chars={len(batch_json)}"
            )
            await self._stream_items_individually(batch, system_prompt, callback_handler)
            return

        try:
            await self._stream_formatter_batch(system_prompt, batch, callback_handler)
        except Exception as exc:
            self.logger.warning(
                "batch formatter LLM failed; "
                f"retrying by item. error={exc}"
            )
            await self._stream_items_individually(batch, system_prompt, callback_handler)

    async def _stream_raw_items(self, raw_items, callback_handler):
        self._pending_raw_items = None
        original_total = len(raw_items)
        user_prompt = getattr(self, "_last_user_prompt", "")
        batch_size = 1

        async def emit_filter_status(event):
            on_status = getattr(callback_handler, "on_status", None)
            if not on_status:
                return
            message = event.get("message", "")
            metadata = {
                key: value
                for key, value in event.items()
                if key != "message"
            }
            await on_status(message, **metadata)

        # 鈹€鈹€ 灏忓垪琛ㄥ揩閫熻矾寰勶細璺宠繃杩囨护锛屼笓鐢?prompt 涓€娆℃€у繝瀹炶緭鍑?鈹€鈹€
        if original_total <= SMALL_LIST_THRESHOLD:
            self.logger.info(
                f"[SMALL-LIST] ({original_total} items), "
                f"using faithful single-call reproduction path"
            )
            pruned = [_prune_item_for_llm(item) for item in raw_items]
            items_for_export = list(raw_items)
            heading = f"## Results ({original_total} items)"
            await callback_handler.on_llm_new_token(
                f"\n\n---\n\n{heading}\n\n"
            )
            _replace_uspto_download_urls_for_batch(pruned)

            url_checklist = self._build_url_checklist(pruned)
            batch_json = json.dumps(pruned, ensure_ascii=False, indent=2, default=str)

            if len(batch_json) <= MAX_BATCH_JSON_CHARS_FOR_LLM:
                # 浠庣涓€鏉?item 鎻愬彇鎵€鏈夊瓧娈靛悕锛岀敓鎴愬己鍒惰緭鍑烘ā鏉?
                first_keys = list(pruned[0].keys()) if pruned else []
                field_checklist = "\n".join(
                    f"- {k}" for k in first_keys
                )

                faithful_system_prompt = (
                    "You are a professional data presenter serving non-technical readers. "
                    "Your output must be readable, well-structured, and free of jargon.\n\n"
                    "CRITICAL RULES:\n\n"
                    "1. TRANSLATE field codes into clear Chinese labels. "
                    "For example: 'apc'鈫?鐢宠浜?, 'ad'鈫?鐢宠鏃?, 'pdt'鈫?涓撳埄绫诲瀷', "
                    "'pk'鈫?鏂囩尞绉嶇被', 'pns'鈫?涓撳埄鍙?, 'lsscn'鈫?娉曞緥鐘舵€?. "
                    "Use your knowledge to interpret every code.\n\n"
                    "2. FILTER noise: skip fields whose value is empty, '0', '鍚?, "
                    "or clearly an internal system ID (like 'pid'). "
                    "Also skip the top-level API wrapper fields (errorCode, errorDesc, "
                    "page_row, page, total, sort_column) 鈥?they are not part of the data.\n\n"
                    "3. GROUP related fields logically: "
                    "titles together, abstracts together, dates together, "
                    "people & organizations together, legal/classification together.\n\n"
                    "4. PRESENT as a reader-friendly document with clear section headings, "
                    "NOT as a flat key-value dump. Use comparison tables when comparing "
                    "multiple records with shared fields.\n\n"
                    "5. ALL meaningful data must be preserved 鈥?do not summarize, "
                    "abbreviate, or skip any non-noise field. If a field has a real value, "
                    "it belongs in the output.\n\n"
                    "6. Every URL must be copied exactly and verbatim. "
                    "Image URLs MUST use ![description](URL) syntax.\n\n"
                    "7. Do NOT add a concluding summary 鈥?let the data speak for itself."
                )
                faithful_user_content = (
                    f"Here are {len(pruned)} data item(s) to present for non-technical readers. "
                    f"Translate all field codes into plain Chinese labels. "
                    f"Group related information logically. "
                    f"Skip empty/noise fields. Keep ALL meaningful data.\n\n"
                    f"Reference 鈥?all fields present in the data:\n"
                    f"{field_checklist}\n\n"
                    f"{url_checklist}"
                    f"{batch_json}"
                )
                try:
                    await self.llm.stream_simple(
                        system_prompt=faithful_system_prompt,
                        user_content=faithful_user_content,
                        callback_handler=callback_handler,
                    )
                except Exception as exc:
                    self.logger.warning(
                        "small-list faithful transcriber failed; "
                        f"falling back to deterministic markdown. error={exc}"
                    )
                    await self._stream_deterministic_batch(pruned, callback_handler)
            else:
                self.logger.info(
                    "small-list pruned JSON too large for single LLM call "
                    f"({len(batch_json)} chars), using deterministic markdown"
                )
                await self._stream_deterministic_batch(pruned, callback_handler)

            await callback_handler.on_llm_new_token("\n\n")

            on_artifacts = getattr(callback_handler, "on_artifacts", None)
            if on_artifacts:
                artifacts = build_result_artifacts(
                    items_for_export,
                    query_id=getattr(self, "_last_query_id", None),
                    original_count=original_total,
                    filter_applied=False,
                )
                if artifacts:
                    await on_artifacts(artifacts)
            return
        # 鈹€鈹€ 澶у垪琛細璧板師鏈夌殑杩囨护 + 鎵归噺鏍煎紡鍖栬矾寰?鈹€鈹€

        filter_result = await filter_tool_result_items(
            raw_items,
            user_prompt,
            self.llm.complete_json,
            batch_size=batch_size,
            status_callback=emit_filter_status,
        )
        pending = filter_result.items

        # Save the original (filtered but un-pruned) items for Excel / CSV
        # export.  The pruning below only affects the LLM input path.
        items_for_export = list(pending)

        # ── 大列表摘要模式：剔除超长字段后做整体总结，跳过逐条批处理 ──
        if USE_LARGE_LIST_SUMMARY:
            summary_items = _prune_for_summary(pending)
            heading = f"## Results — Summary ({len(pending)} items)"
            await callback_handler.on_llm_new_token(
                f"\n\n---\n\n{heading}\n\n"
            )

            summary_system_prompt = (
                "You are a professional data analyst. Create a concise, well-structured summary of the data items below. "
                "Group similar items, highlight key patterns or trends, and present information clearly for non-technical readers. "
                "Use Markdown formatting — including tables where appropriate. Keep it under 600 words. "
                "Do NOT list every item individually; synthesize and summarize. "
                "Focus on: what the data shows overall, key differences between items, any notable outliers."
            )

            try:
                await self.llm.stream_simple(
                    system_prompt=summary_system_prompt,
                    user_content=json.dumps(summary_items, ensure_ascii=False, indent=2, default=str),
                    callback_handler=callback_handler,
                )
            except Exception as exc:
                self.logger.warning(
                    f"large-list summary LLM failed: {exc}"
                )
                await callback_handler.on_llm_new_token(
                    "\n\n*Summary generation failed. Please download the data file below.*"
                )

            await callback_handler.on_llm_new_token(
                "\n\n> \U0001f4e5 For complete data, please download the Excel or CSV file below.\n\n"
            )

            on_artifacts = getattr(callback_handler, "on_artifacts", None)
            if on_artifacts:
                artifacts = build_result_artifacts(
                    items_for_export,
                    query_id=getattr(self, "_last_query_id", None),
                    original_count=original_total,
                    filter_applied=True,
                )
                if artifacts:
                    await on_artifacts(artifacts)
            return


        # Prune each item so that no single element exceeds 15 000 chars.
        # Oversized arrays (> 10 000 chars) are dropped; oversized dicts are
        # recursed into; oversized strings are truncated.
        pending_before_prune = sum(_json_len(item) for item in pending)
        pending = [_prune_item_for_llm(item) for item in pending]
        pending_after_prune = sum(_json_len(item) for item in pending)
        if pending_before_prune != pending_after_prune:
            self.logger.info(
                "tool_result pruned long values before batch formatting; "
                f"chars_before={pending_before_prune}, chars_after={pending_after_prune}"
            )

        total = len(pending)
        heading = (
            f"## Filtered Results ({filter_result.filtered_count} of {filter_result.original_count} items)"
            if filter_result.applied
            else f"## Full Results ({original_total} items)"
        )
        await callback_handler.on_llm_new_token(
            f"\n\n---\n\n{heading}\n\n"
        )
        system_prompt = (
            "You are presenting search result items clearly and concisely. "
            "For each item, extract and present the most important information as clean Markdown. "
            "Use **bold** for field names. Number each item. "
            "Every URL is mandatory: copy every URL from the input exactly and verbatim. "
            "Do not omit, shorten, summarize, translate, decode, re-encode, or alter any URL. "
            "If an item has multiple URLs, include all of them under that item. "
            "If a URL is an image URL, display it using Markdown image syntax exactly as "
            "![alt text](image_URL) so the frontend can render the image inline. "
            "Do NOT add any preamble, summary, or conclusion - output only the formatted items."
        )
        for batch_start in range(0, total, batch_size):
            batch = pending[batch_start:batch_start + batch_size]
            self.logger.info(f"batch: {batch}")
            _replace_uspto_download_urls_for_batch(batch)
            batch_end = min(batch_start + batch_size, total)
            await callback_handler.on_llm_new_token(
                f"### Items {batch_start + 1}-{batch_end}\n\n"
            )
            await self._stream_batch_with_retries(batch, system_prompt, callback_handler)
            await callback_handler.on_llm_new_token("\n\n")

        on_artifacts = getattr(callback_handler, "on_artifacts", None)
        if on_artifacts:
            artifacts = build_result_artifacts(
                items_for_export,
                query_id=getattr(self, "_last_query_id", None),
                original_count=original_total,
                filter_applied=filter_result.applied,
            )
            if artifacts:
                await on_artifacts(artifacts)


    async def invoke_agent(self, agent, callback_handler):
        try:
            workflow_result = getattr(self, "_workflow_result", None)
            if workflow_result is not None:
                self._workflow_result = None
                await _emit_status(callback_handler, "正在整理结果...")
                output_mode = getattr(workflow_result, "output_mode", "last")
                raw_items = getattr(workflow_result, "raw_items", None)
                if output_mode == "all":
                    await self._stream_workflow_final_result(workflow_result, callback_handler)
                elif raw_items:
                    await self._stream_raw_items(raw_items, callback_handler)
                else:
                    await self._stream_workflow_final_result(workflow_result, callback_handler)
                return

            await self.llm.openai_invoke(agent, self.memory.get(), callback_handler)
            # LangGraph agents don't call on_agent_finish 鈥?inject batch analysis here,
            # after all LLM tokens have been streamed but before core.py sends 'end'.
            pending = getattr(self, '_pending_raw_items', None)
            if pending:
                await self._stream_raw_items(pending, callback_handler)
        except Exception as e:
            raise e

if __name__ == "__main__":
    pass
