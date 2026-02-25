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

from langchain_core.tools import StructuredTool

import os
import time
import re
import requests
import asyncio

# 定义参数模型
class DynamicToolFunction(BaseModel):
    user_id: str = Field(description="user id")
    query_id: str = Field(description="query id")
    params: str = Field(description="params")

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
    
    def set_knowledge_tool(self, knowledge_tool: Dict[str, Any]) -> None:
        """
        设置知识工具字典
        Args:
            knowledge_tool (Dict[str, Any]): 知识工具字典
        """
        self.knowledgeTool = knowledge_tool
    
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
        """
        判断 self.knowledgeTool 中的 tool_info 的 params 中的 query 和 body 是否都为空。
        如果都为空，返回 True；否则返回 False。
        """
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
        """
        生成系统提示
        """
        knowledge_item, tool_info = self.knowledgeTool
        self.logger.info(f"knowledge item:{knowledge_item} - tool:{tool_info}")

        # 获取当前时间戳
        current_timestamp = time.time()

        # 转换为本地时间结构
        local_time = time.localtime(current_timestamp)

        # 格式化为字符串
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", local_time)

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
                    tool_params_info = "工具参数要求:user id - query id\n"
                    for param_name, param_type in params_data.items():
                        if param_name == "method" or param_name == "content-type":
                            continue
                        tool_params_info += f"  - {param_name} ({param_type})\n"
                else:
                    tool_params_info = f"工具参数: {tool_info.params}"
            except json.JSONDecodeError:
                tool_params_info = f"工具参数: {tool_info.params}"

        system_prompt = f"""
        You are an intelligent assistant capable of deciding when and how to use APIs to complete tasks.
        {context}

        Based on the user's request and the available context, decide whether invoking a tool is necessary.

        If a tool is required, use the following tool:

        Tool: {tool_title}
        Purpose: {tool_description}
        Input parameters: {tool_params_info}

        Execute the tool with the appropriate parameters and generate the final response strictly based on the tool's output.

        Whenever you provide information, facts, quotes, or data, please always include the specific source links (URLs) used to generate that part of the response.


        Use the standard Markdown inline link syntax: `[anchor text](full_url)`.

        - **Correct:** The global AI market is projected to reach $1.5 trillion by 2030 [source](https://example.com/report2023).

        - **Incorrect:** The global AI market is projected to reach $1.5 trillion by 2030. (Source: https://example.com/report2023)

        If the task can be completed without invoking the tool, respond directly to the user without calling any tool.

        Do not fabricate tool results. Do not assume tool behavior beyond the provided output.

        Do not return tool parameters, such as the user id and query id.
        """
        # return self.expand_prompt(system_prompt)
        return system_prompt

    def generate_template_system_prompt(self) -> str:
        """
        生成系统提示
        """
        knowledge_item, tool_info = self.knowledgeTool
        self.logger.info(f"knowledge item:{knowledge_item} - tool:{tool_info}")

        # 获取当前时间戳
        current_timestamp = time.time()

        # 转换为本地时间结构
        local_time = time.localtime(current_timestamp)

        # 格式化为字符串
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", local_time)

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
        3. params: A JSON object containing ONLY the API request parameters (template below)

        The third parameter "params" template: {tool_info.params}

        Your task is to analyze the user's input and modify the third parameter "params" template according to the user's specific requirements. Generate a new JSON object containing only the parameters that need to be changed or specified based on the user's request.

        You MUST follow all rules below without exception:

        1. You may ONLY modify existing values in the JSON.
           - DO NOT add new fields, If a field is empty in the template, then leave it empty.
           - DO NOT change the JSON structure or nesting
           - CRITICAL: DO NOT include user_id or query_id in the params JSON - these are separate parameters

        2. Field semantics:
           - method MUST remain unchanged
           - query contains URL query parameters
           - header contains HTTP headers
           - body contains the HTTP request body

        3. Value replacement rules:
           - Replace a value only if the user query clearly maps to the meaning of an existing field
           - If the user query does not mention or imply a field, keep its original value unchanged
           - Do NOT infer or invent information not explicitly expressed by the user
           - DO NOT extract or infer user_id or query_id from the user's request into the params JSON

        4. Output rules:
           - Output ONLY the final, complete JSON for the params parameter
           - DO NOT include explanations, reasoning, comments, or formatting outside JSON
           - The output must be valid, strictly formatted JSON
           - DO NOT include user_id or query_id fields in the JSON output

        Execute the tool with the appropriate parameters and generate the final response strictly based on the tool's output.

        Whenever you provide information, facts, quotes, or data, please always include the specific source links (URLs) used to generate that part of the response.

        Use the standard Markdown inline link syntax: `[anchor text](full_url)`.

        - **Correct:** The global AI market is projected to reach $1.5 trillion by 2030 [source](https://example.com/report2023).

        - **Incorrect:** The global AI market is projected to reach $1.5 trillion by 2030. (Source: https://example.com/report2023)

        If the task can be completed without invoking the tool, respond directly to the user without calling any tool.

        Do not fabricate tool results. Do not assume tool behavior beyond the provided output.

        Do not return tool parameters, such as the user id and query id.
        """

        return system_prompt

    def generate_system_prompt(self, tool_data) -> str:
        """
        生成系统提示
        """
        _, tool_info = self.knowledgeTool

        # 根据tool_info.push的值选择不同系统提示词
        if tool_info.push == 1:
            if tool_data and tool_data.strip():  # 判断tool_data是否非空
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
        """
        直接解析 tool_data 并生成系统提示词，无需发起 HTTP 请求。
        """
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
                    result_str = json.dumps(result_data, ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    # 如果不是有效的 JSON，则直接使用原始内容
                    result_str = tool_data

            # 构造系统提示词
            system_prompt = f"""
            Act as a self-contained intelligent assistant. Follow these instructions strictly:
            {context}

            1.  **Core Principle:** You must perform tasks and generate answers using **only** the data, text, or context that I provide to you within this chat.
            2.  **No External Access:** Do not attempt to invoke or use any internal or external tools (such as search functions, code interpreters, calculators, or knowledge retrieval from your base training data) to complete the task.
            3.  **Direct Processing:** Analyze, reason, and respond directly based on the provided input. If the necessary information is not contained in my messages, state that clearly instead of making assumptions.

            Input：
            {result_str}

            Please generate the final result based on the above data.
            """
            return system_prompt

        except Exception as e:
            self.logger.error(f"Failed to generate frontend tool direct system prompt: {str(e)}")
            return self.generate_template_system_prompt()

    def generate_backend_tool_direct_system_prompt(self) -> str:
        """
        直接利用 tool_info 发起 HTTP 请求，将结果写入系统提示词并返回。
        告诉大模型无需调用工具，直接基于返回的数据生成结果。
        """
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

            # 发起 HTTP 请求
            if method == "GET":
                response = requests.get(url, headers=headers)
            elif method == "POST":
                response = requests.post(url, headers=headers, json={})
            elif method == "PUT":
                response = requests.put(url, headers=headers, json={})
            elif method == "DELETE":
                response = requests.delete(url, headers=headers)
            elif method == "PATCH":
                response = requests.patch(url, headers=headers, json={})
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
                        result_str = json.dumps(result_data, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError:
                        # 如果JSON解析失败，使用原始响应内容
                        result_str = response.text if response.text else "Empty response"
            else:
                result_str = f"request failed，status code: {response.status_code}"



            # 构造系统提示词
            system_prompt = f"""
            Act as a self-contained intelligent assistant. Follow these instructions strictly:
            {context}

            1.  **Core Principle:** You must perform tasks and generate answers using **only** the data, text, or context that I provide to you within this chat.
            2.  **No External Access:** Do not attempt to invoke or use any internal or external tools (such as search functions, code interpreters, calculators, or knowledge retrieval from your base training data) to complete the task.
            3.  **Direct Processing:** Analyze, reason, and respond directly based on the provided input. If the necessary information is not contained in my messages, state that clearly instead of making assumptions.

            Input：
            {result_str}

            Please generate the final result based on the above data.
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

                    def dynamic_backend_tool_function(user_id: str, query_id: str, params: str):
                        self.logger.info(f"dynamic_backend_tool_function user id is {user_id} - query id is {query_id} - param is {params}")
                        # 从tool_info中获取URL
                        url = tool_info.url

                        # 解析参数JSON
                        try:
                            params_data = json.loads(tool_info.params)
                            user_params = json.loads(params)
                        except json.JSONDecodeError:
                            raise ValueError(f"Invalid JSON in tool_info.params: {tool_info.params}")

                        # 获取HTTP方法和Content-Type
                        method = params_data.get("method", "GET").upper()
                        content_type = params_data.get("Content-Type", "application/json")

                        # 准备HTTP请求头
                        headers = {
                            "Content-Type": content_type
                        }

                        # 从user_params中获取header并写入headers
                        user_headers = user_params.get("header", {})
                        if isinstance(user_headers, dict):
                            headers.update(user_headers)

                        request_params = user_params.get("query")
                        request_body = user_params.get("body")

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

                        # 返回响应结果
                        return response.json() if response.content else None

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

                    dynamic_tool = StructuredTool.from_function(
                        func=tool_func,
                        name=cleaned_tool_name,
                        description=tool_info.description if tool_info.description else "Dynamic knowledge tool",
                        args_schema=DynamicToolFunction
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

    async def get_tools(self) -> list:
        """
        选择方法
        """
        _, tool_info = self.knowledgeTool

        tools = []
        # 根据tool_info.push的值选择不同系统提示词
        if tool_info.push == 1:
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
            # 默认情况下固定的系统提示词
            return await self.get_dynamic_tools()

    async def process(self,user_id, prompt, query_id, speech_module) -> str | tuple[str, str]:
        if not self.enabled:
            return "general Agent is disabled."
        self.knowledgeTool = get_knowledge_tool(user_id,  prompt)
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

    async def create_agent(self, user_id, prompt, query_id, tool_data, callback_handler):
        #self.knowledgeTool = get_knowledge_tool(user_id,  prompt)
        self.knowledgeTool = await asyncio.to_thread(get_knowledge_tool, user_id,  prompt)
        user_prompt = self.generate_user_prompt(prompt, user_id, query_id)
        system_prompt = self.generate_system_prompt(tool_data)
        self.memory.reset([])
        self.memory.push('user', user_prompt)
        self.memory.push('system', system_prompt)

        self.logger.info(f"memory.get():{self.memory.get()}")
        self.tools = await self.get_tools()

        return self.llm.openai_create(self.tools, self.memory.get(), callback_handler)


    async def invoke_agent(self, agent, callback_handler):
        self.logger.info(f"invoke agent memory:{self.memory.get()}")
        try:
            await self.llm.openai_invoke(agent, self.memory.get(), callback_handler)
        except Exception as e:
            raise e

if __name__ == "__main__":
    pass
