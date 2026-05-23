from typing import Dict, Any
import json
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup

from sources.knowledge.knowledge import get_redis_connection, get_knowledge_tool, clean_html_text
from sources.utility import pretty_print, animate_thinking
from sources.agents.agent import Agent
from sources.tools.mcpFinder import MCP_finder
from sources.memory import Memory
from sources.logger import Logger
from sources.dynamic_tool_params import _coerce_json_object
from sources.result_pipeline import ResultPipeline, find_primary_list, user_intent_requests_filter
from sources.tool_param_policy import TEMPLATE_PARAM_RULES, should_expose_dynamic_tool

from langchain_core.tools import StructuredTool

import os
import time
import re
import requests
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

# Headers whose values must never be exposed to the LLM
_SENSITIVE_HEADER_RE = re.compile(
    r'(auth|api.?key|token|secret|credential|password|bearer)',
    re.IGNORECASE
)

# 定义参数模型
class DynamicToolFunction(BaseModel):
    user_id: str = Field(description="user id")
    query_id: str = Field(description="query id")
    params: str = Field(description="params")


class DynamicBackendToolFunction(BaseModel):
    user_id: str = Field(description="user id")
    query_id: str = Field(description="query id")
    params: Dict[str, Any] | str = Field(
        description="API request parameters as a JSON object; legacy JSON strings are also accepted"
    )


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
        self._pending_user_instruction = ""

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
                    parts.append(f"{k}: {', '.join(str(i) for i in v)}")
            else:
                parts.append(f"{k}: {v}")
        return " | ".join(parts)

    def _render_list_as_md(self, label: str | None, items: list) -> str:
        """Render every item in a list as readable markdown bullets without JSON dumps."""
        header = f"**{label}** ({len(items)} items total):\n\n" if label else f"({len(items)} items total):\n\n"
        blocks = [header]
        for i, item in enumerate(items, 1):
            if isinstance(item, dict):
                scalar_parts = [
                    f"**{k}**: {v}" for k, v in item.items()
                    if not isinstance(v, (dict, list)) and v is not None and v != ""
                ]
                nested_parts = [
                    (k, v) for k, v in item.items()
                    if isinstance(v, (dict, list)) and v
                ]
                top = f"- **[{i}]** " + (" | ".join(scalar_parts) if scalar_parts else "")
                sub_lines = []
                for nk, nv in nested_parts:
                    if isinstance(nv, dict):
                        flat = self._flatten_dict(nv)
                        if flat:
                            sub_lines.append(f"  - **{nk}**: {flat}")
                    elif isinstance(nv, list):
                        if nv and not isinstance(nv[0], (dict, list)):
                            sub_lines.append(f"  - **{nk}**: {', '.join(str(x) for x in nv)}")
                        elif nv:
                            sub_lines.append(f"  - **{nk}**: [{len(nv)} items]")
                blocks.append("\n".join([top] + sub_lines) if sub_lines else top)
            elif isinstance(item, list):
                if item and not isinstance(item[0], (dict, list)):
                    blocks.append(f"- **[{i}]** {', '.join(str(x) for x in item)}")
                else:
                    blocks.append(f"- **[{i}]** [{len(item)} items]")
            else:
                blocks.append(f"- **[{i}]** {item}")
        return "\n".join(blocks)


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
        """Return True if both query and body in tool_info.params are empty."""
        # 获取工具信息
        _, tool_info = self.knowledgeTool

        try:
            # 解析 params 字段
            params_data = json.loads(tool_info.params)

            # 获取 query 和 body 字段
            query = params_data.get("query", {})
            body = params_data.get("body", {})

            # 判断 query 和 body 是否都为空
            return not query and not body
        except json.JSONDecodeError:
            # 如果 params 不是合法的 JSON，视为非空
            return False

    def generate_fixed_system_prompt(self) -> str:
        """Generate system prompt for fixed (no-parameter) tools."""
        knowledge_item, tool_info = self.knowledgeTool
        self.logger.info(f"knowledge item:{knowledge_item} - tool:{tool_info}")

        # 获取美国东部时区的当前时间
        eastern_tz = ZoneInfo("America/New_York")
        eastern_time = datetime.now(eastern_tz)

        # 格式化为字符串（包含时区信息）
        time_str = eastern_time.strftime("%Y-%m-%d %H:%M:%S %Z")

        # 获取knowledge item的answer作为上下文
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

        # 解析工具参数信息
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

        # 获取美国东部时区的当前时间
        eastern_tz = ZoneInfo("America/New_York")
        eastern_time = datetime.now(eastern_tz)

        # 格式化为字符串（包含时区信息）
        time_str = eastern_time.strftime("%Y-%m-%d %H:%M:%S %Z")

        # 获取knowledge item的answer作为上下文
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

        {TEMPLATE_PARAM_RULES}

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
        [Organized detailed content — if the tool result is a long JSON list, show a summary with count statistics]

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

    def generate_system_prompt(self, tool_data) -> str:
        """Select and return the appropriate system prompt based on push mode."""
        _, tool_info = self.knowledgeTool

        # 根据tool_info.push的值选择不同系统提示词
        if tool_info.push == 1:
            if tool_data and tool_data.strip():  # 判断tool_data是否非空
                return self.generate_frontend_tool_direct_system_prompt(tool_data)
            else:
                return self.generate_template_system_prompt()
        elif tool_info.push == 2:
            return self.generate_template_system_prompt()
        elif tool_info.push == 3:
            return self.generate_frontend_tool_direct_system_prompt(tool_data)
        else:
            # 默认情况下固定的系统提示词
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
            # 获取knowledge item的answer作为上下文
            knowledge_item, _ = self.knowledgeTool
            context = ""
            if knowledge_item and hasattr(knowledge_item, 'answer') and knowledge_item.answer:
                context = f"""

            Context from knowledge base:
            {knowledge_item.answer}

            Use this context to better understand the task and provide more accurate responses.
            """

            # 解析 tool_data 内容
            if "text/html" in tool_data:
                # 如果是 HTML 内容，使用公用方法清理文本
                result_str = clean_html_text(tool_data)
            else:
                # 尝试将 tool_data 解析为 JSON
                try:
                    result_data = json.loads(tool_data)
                    raw_items = find_primary_list(result_data).items

                    if raw_items:
                        self._pending_raw_items = raw_items
                        result_str = (
                            f"The query returned {len(raw_items)} items. "
                            f"Please write a brief 2–3 sentence summary of what was found. "
                            f"The complete list will be analyzed and displayed item by item automatically — "
                            f"do NOT enumerate the items yourself."
                        )
                    else:
                        result_str = json.dumps(result_data, ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    # 如果不是有效的 JSON，则直接使用原始内容
                    result_str = tool_data

            # 获取格式化指南
            formatting_guide = self._get_markdown_formatting_guide()

            # 构造系统提示词
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
        # 获取工具信息
        knowledge_item, tool_info = self.knowledgeTool
        self.logger.info(f"generate_tool_direct_system_prompt - tool:{tool_info}")

        try:
            # 获取knowledge item的answer作为上下文
            context = ""
            if knowledge_item and hasattr(knowledge_item, 'answer') and knowledge_item.answer:
                context = f"""

            Context from knowledge base:
            {knowledge_item.answer}

            Use this context to better understand the task and provide more accurate responses.
            """

            # 从 tool_info 中提取 URL 和参数模板
            url = tool_info.url
            params_data = json.loads(tool_info.params)

            # 获取 HTTP 方法和 Content-Type
            method = params_data.get("method", "GET").upper()
            content_type = params_data.get("Content-Type", "application/json")

            # 准备请求头
            headers = {
                "Content-Type": content_type
            }

            # params_data
            user_headers = params_data.get("header", {})
            if isinstance(user_headers, dict):
                headers.update(user_headers)

            # 添加时间戳参数绕过 CDN 缓存
            cache_bust_params = {"_t": str(int(time.time() * 1000))}

            # 发起 HTTP 请求
            if method == "GET":
                response = requests.get(url, params=cache_bust_params, headers=headers)
            elif method == "POST":
                response = requests.post(url, params=cache_bust_params, headers=headers, json={})
            elif method == "PUT":
                response = requests.put(url, params=cache_bust_params, headers=headers, json={})
            elif method == "DELETE":
                response = requests.delete(url, params=cache_bust_params, headers=headers)
            elif method == "PATCH":
                response = requests.patch(url, params=cache_bust_params, headers=headers, json={})
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            # 处理响应结果
            if response.status_code == 200:

                content_type = response.headers.get("Content-Type", "").lower()

                if "text/html" in content_type:
                    # 使用BeautifulSoup移除HTML标签
                    result_str = BeautifulSoup(response.content, "html.parser").get_text()
                elif "application/xml" in content_type or "text/xml" in content_type:
                    # 处理XML格式响应
                    try:
                        # 使用BeautifulSoup解析XML并提取文本内容
                        soup = BeautifulSoup(response.content, "xml")
                        # 移除XML标签，只保留文本内容
                        result_str = soup.get_text()
                        # 如果XML解析失败或内容为空，使用原始内容
                        if not result_str.strip():
                            result_str = response.text
                    except Exception as xml_e:
                        self.logger.warning(f"XML parsing failed: {str(xml_e)}, using raw content")
                        result_str = response.text
                else:
                    try:
                        result_data = response.json() if response.content else {}
                        raw_items = find_primary_list(result_data).items

                        if raw_items:
                            self._pending_raw_items = raw_items
                            result_str = (
                                f"The query returned {len(raw_items)} items. "
                                f"Please write a brief 2–3 sentence summary of what was found. "
                                f"The complete list will be analyzed and displayed item by item automatically — "
                                f"do NOT enumerate the items yourself."
                            )
                        else:
                            result_str = json.dumps(result_data, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError:
                        # 如果JSON解析失败，使用原始响应内容
                        result_str = response.text if response.text else "Empty response"
            else:
                result_str = f"Request failed, status code: {response.status_code}"

            # 获取格式化指南
            formatting_guide = self._get_markdown_formatting_guide()

            # 构造系统提示词
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
            # 如果有知识库中的工具信息，则动态构建MCP工具
            if hasattr(self, 'knowledgeTool') and self.knowledgeTool:
                # 获取工具信息
                knowledge_item, tool_info = self.knowledgeTool

                if tool_info:

                    # 动态创建工具函数
                    def dynamic_frontend_tool_function(user_id: str, query_id: str, params: str):
                        self.logger.info(f"dynamic_frontend_tool_function user id is {user_id} - query id is {query_id} - param is {params}")
                        try:
                            # 连接Redis
                            redis_conn = get_redis_connection()

                            # 构造Redis键
                            redis_key = f"tool_request_{query_id}_{user_id}"

                            # param_dict = {"origin_params": json.loads(tool_info.params)}
                            # if params:
                            #     # 将参数转换为JSON并存储到Redis
                            #     param_dict["llm_params"] = params

                            params_json = json.dumps(params)

                            redis_conn.set(redis_key, params_json, ex=1200)

                            # 轮询读取tool_response_{query_id}
                            response_key = f"tool_response_{query_id}_{user_id}"
                            timeout = 300  # 5分钟超时
                            interval = 1  # 每秒查询一次
                            elapsed = 0

                            while elapsed < timeout:
                                response_value = redis_conn.get(response_key)

                                if response_value is not None:
                                    # 成功获取到响应值
                                    return response_value
                                # 等待1秒后再次尝试
                                time.sleep(interval)
                                elapsed += interval

                            # 超时未获取到响应值
                            return None
                        except Exception as e:
                            # 如果Redis操作失败，记录日志但仍继续执行工具
                            self.logger.error(f"Failed to write to Redis: {str(e)}")
                            return None

                    def dynamic_backend_tool_function(user_id: str, query_id: str, params: Dict[str, Any] | str):
                        self.logger.info(f"dynamic_backend_tool_function user id is {user_id} - query id is {query_id} - param is {params}")
                        # 从tool_info中获取URL
                        url = tool_info.url

                        # 解析参数JSON
                        params_data = _coerce_json_object(tool_info.params, "tool_info.params")
                        user_params = _coerce_json_object(params, "LLM tool params")

                        # 获取HTTP方法和Content-Type
                        method = params_data.get("method", "GET").upper()
                        content_type = params_data.get("Content-Type", "application/json")

                        # Prepare HTTP headers from server-side tool_info.params only.
                        # Never use LLM-provided header values — the LLM only sees
                        # sanitised (****) placeholders and must not control auth headers.
                        headers = {
                            "Content-Type": content_type
                        }
                        server_headers = params_data.get("header", {})
                        if isinstance(server_headers, dict):
                            headers.update(server_headers)

                        request_params = user_params.get("query")
                        request_body = user_params.get("body")

                        # 添加时间戳参数绕过 CDN 缓存
                        if request_params is None:
                            request_params = {}
                        request_params["_t"] = str(int(time.time() * 1000))

                        if method == "GET":
                            response = requests.get(url, params=request_params, headers=headers)
                        elif method == "POST":
                            response = requests.post(url, params=request_params, headers=headers, json=request_body)
                        elif method == "PUT":
                            response = requests.put(url, params=request_params, headers=headers, json=request_body)
                        elif method == "DELETE":
                            response = requests.delete(url, params=request_params, headers=headers)
                        elif method == "PATCH":
                            response = requests.patch(url, params=request_params, headers=headers, json=request_body)
                        else:
                            raise ValueError(f"Unsupported HTTP method: {method}")

                        # 打印 response 信息
                        self.logger.info(f"Response status code: {response.status_code}")
                        self.logger.info(f"Response headers: {response.headers}")
                        # self.logger.info(f"Response content: {response.text}")
                        # 处理响应结果
                        if response.status_code == 200:
                            content_type = response.headers.get("Content-Type", "").lower()

                            if "text/html" in content_type:
                                # HTML 内容，使用 BeautifulSoup 清理
                                result = BeautifulSoup(response.content, "html.parser").get_text()
                            elif "application/xml" in content_type or "text/xml" in content_type:
                                # XML 内容，尝试解析并提取文本
                                try:
                                    soup = BeautifulSoup(response.content, "xml")
                                    result = soup.get_text()
                                    if not result.strip():
                                        result = response.text
                                except Exception as xml_e:
                                    self.logger.warning(f"XML parsing failed: {str(xml_e)}, using raw content")
                                    result = response.text
                            else:
                                # JSON 或其他格式
                                try:
                                    result_data = response.json() if response.content else None
                                    if isinstance(result_data, (dict, list)):
                                        raw_items = find_primary_list(result_data).items

                                        if raw_items:
                                            list_count = len(raw_items)
                                            # Store raw items; invoke_agent will batch-analyze them via LLM
                                            self._pending_raw_items = raw_items
                                            result = (
                                                f"The query returned {list_count} items. "
                                                f"Please write a brief 2–3 sentence summary of what was found. "
                                                f"The complete list will be analyzed and displayed item by item automatically — "
                                                f"do NOT enumerate the items yourself."
                                            )
                                        else:
                                            result = json.dumps(result_data, ensure_ascii=False, indent=2)
                                    else:
                                        result = result_data
                                except json.JSONDecodeError:
                                    # JSON 解析失败，返回原始文本
                                    result = response.text if response.text else None
                        else:
                            # 请求失败，返回错误信息
                            result = f"Request failed, status code: {response.status_code}"

                        return result

                    # 清理工具名称以符合API要求
                    tool_name = tool_info.title if tool_info.title else "dynamic_knowledge_tool"
                    # 只保留字母、数字、下划线和连字符
                    cleaned_tool_name = re.sub(r'[^a-zA-Z0-9_-]', '_', tool_name)
                    # 确保名称不为空
                    if not cleaned_tool_name or cleaned_tool_name.strip() == "":
                        cleaned_tool_name = "dynamic_knowledge_tool"

                    # 根据tool_info.push的值选择不同的工具函数
                    if tool_info.push == 1 or tool_info.push == 3:
                        tool_func = dynamic_frontend_tool_function
                    elif tool_info.push == 2:
                        tool_func = dynamic_backend_tool_function
                    else:
                        # 默认情况下使用前端工具函数
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

                    # 合并动态工具
                    tools = [dynamic_tool]
                else:
                    # 如果没有动态工具信息，使用默认配置
                    tools = None
            else:
                # 如果没有动态工具信息，使用默认配置
                tools = None

            self.logger.info(f"tools{tools}")
            return tools
        except Exception as e:
            raise Exception(f"get_tool failed: {str(e)}") from e

    async def get_tools(self, tool_data: str = "") -> list:
        """Select and return the appropriate LangChain tool list based on push mode."""
        _, tool_info = self.knowledgeTool

        tools = []
        if should_expose_dynamic_tool(
            push=tool_info.push,
            has_tool_data=bool(tool_data and tool_data.strip()),
            query_body_empty=self.is_query_and_body_empty(),
        ):
            return await self.get_dynamic_tools()
        return tools

    async def process(self, user_id, prompt, query_id, speech_module, push_filter=None) -> str | tuple[str, str]:
        if not self.enabled:
            return "general Agent is disabled."
        self.knowledgeTool = get_knowledge_tool(user_id, prompt, push_filter=push_filter)
        # user_prompt = self.expand_prompt(prompt)
        user_prompt = self.generate_user_prompt(prompt, user_id, query_id)
        system_prompt = self.generate_system_prompt()
        self.memory.reset([])
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
        self._pending_user_instruction = prompt
        self.knowledgeTool = await asyncio.to_thread(get_knowledge_tool, user_id, prompt, push_filter=push_filter)
        user_prompt = self.generate_user_prompt(prompt, user_id, query_id)
        system_prompt = self.generate_system_prompt(tool_data)
        self.memory.reset([])
        self.memory.push('user', user_prompt)
        self.memory.push('system', system_prompt)

        self.logger.info(f"memory.get():{self.memory.get()}")
        self.tools = await self.get_tools(tool_data)

        return self.llm.openai_create(self.tools, self.memory.get(), callback_handler)


    async def invoke_agent(self, agent, callback_handler):
        try:
            await self.llm.openai_invoke(agent, self.memory.get(), callback_handler)
            # LangGraph agents don't call on_agent_finish — inject batch analysis here,
            # after all LLM tokens have been streamed but before core.py sends 'end'.
            pending = getattr(self, '_pending_raw_items', None)
            if pending:
                self._pending_raw_items = None
                user_instruction = getattr(self, "_pending_user_instruction", "")
                pipeline = ResultPipeline(
                    llm=self.llm,
                    callback_handler=callback_handler,
                    batch_size=5,
                )
                await pipeline.stream_items(
                    pending,
                    user_instruction=user_instruction,
                    requires_filter=user_intent_requests_filter(user_instruction),
                )
                return
                total = len(pending)
                batch_size = 5
                await callback_handler.on_llm_new_token(
                    f"\n\n---\n\n## Full Results ({total} items)\n\n"
                )
                system_prompt = (
                    "You are presenting search result items clearly and concisely. "
                    "For each item, extract and present the most important information as clean Markdown. "
                    "Use **bold** for field names. Number each item. "
                    "Do NOT add any preamble, summary, or conclusion — output only the formatted items."
                )
                for batch_start in range(0, total, batch_size):
                    batch = pending[batch_start:batch_start + batch_size]
                    batch_end = min(batch_start + batch_size, total)
                    await callback_handler.on_llm_new_token(
                        f"### Items {batch_start + 1}–{batch_end}\n\n"
                    )
                    batch_json = json.dumps(batch, ensure_ascii=False, indent=2)
                    await self.llm.stream_simple(
                        system_prompt=system_prompt,
                        user_content=(
                            f"Format and analyze these {len(batch)} search result items as readable Markdown:\n\n"
                            f"{batch_json}"
                        ),
                        callback_handler=callback_handler,
                    )
                    await callback_handler.on_llm_new_token("\n\n")
        except Exception as e:
            raise e

if __name__ == "__main__":
    pass
