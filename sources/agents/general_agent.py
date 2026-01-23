from typing import Dict, Any
import json
from pydantic import BaseModel, Field

from sources.knowledge.knowledge import get_redis_connection, get_knowledge_tool
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
            "mcp_finder": MCP_finder(keys["mcp_finder"]),
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

        if not tool_info:
            system_prompt = f"""
            
            You are an intelligent API-enabled assistant. Current time is {time_str}.
            
            If no relevant knowledge is available to complete the user’s task, clearly inform the user that no matching knowledge was found and suggest checking the community for shared knowledge or tools that may solve the problem.
            
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

        Based on the user’s request and the available context, decide whether invoking a tool is necessary.

        If a tool is required, use the following tool:

        Tool: {tool_title}
        Purpose: {tool_description}
        Input parameters: {tool_params_info}

        Execute the tool with the appropriate parameters and generate the final response strictly based on the tool’s output.

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

        if not tool_info:
            system_prompt = f"""

            You are an intelligent API-enabled assistant. Current time is {time_str}.

            If no relevant knowledge is available to complete the user’s task, clearly inform the user that no matching knowledge was found and suggest checking the community for shared knowledge or tools that may solve the problem.

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

        Based on the user’s request and the available context, decide whether invoking a tool is necessary.

        If a tool is required, use the following tool:

        Tool: {tool_title}
        Purpose: {tool_description}
        Input parameters: {tool_params_info}
        the third parameter "params" template: {tool_info.params}
    
        Your task is to analyze the user's input and modify the third parameter "params" template according to the user's specific requirements. Generate a new JSON object containing only the parameters that need to be changed or specified based on the user's request.
        
        You MUST follow all rules below without exception:

        1. You may ONLY modify existing values in the JSON.
           - DO NOT add new fields
           - DO NOT change the JSON structure or nesting
        
        2. Field semantics:
           - method MUST remain unchanged
           - query contains URL query parameters
           - header contains HTTP headers
           - body contains the HTTP request body
        
        3. Value replacement rules:
           - Replace a value only if the user query clearly maps to the meaning of an existing field
           - If the user query does not mention or imply a field, keep its original value unchanged
           - Do NOT infer or invent information not explicitly expressed by the user
        
        4. Output rules:
           - Output ONLY the final, complete JSON
           - Do NOT include explanations, reasoning, comments, or formatting outside JSON
           - The output must be valid, strictly formatted JSON
    
        Execute the tool with the appropriate parameters and generate the final response strictly based on the tool's output.
    
        If the task can be completed without invoking the tool, respond directly to the user without calling any tool.
    
        Do not fabricate tool results. Do not assume tool behavior beyond the provided output.
    
        Do not return tool parameters, such as the user id and query id.
        """

        return system_prompt

    def generate_system_prompt(self) -> str:
        """
        生成系统提示
        """
        _, tool_info = self.knowledgeTool

        # 根据tool_info.push的值选择不同系统提示词
        if tool_info.push == 1:
            return self.generate_fixed_system_prompt()
        elif tool_info.push == 2:
            return self.generate_template_system_prompt()
        else:
            # 默认情况下固定的系统提示词
            return self.generate_fixed_system_prompt()



    def generate_user_prompt(self, prompt, user_id, query_id) -> str:
        user_prompt = f"""
        {prompt},
        user id is {user_id},
        query id is {query_id},
        """
        self.logger.info(f"user prompt:{user_prompt}")

        return user_prompt

    # async def get_tools(self) -> dict:
    #
    #     try:
    #         client = MCPClient.from_config_file("tool_config.json")
    #         adapter = LangChainAdapter()
    #         tools = await adapter.create_tools(client)
    #         self.logger.info(f"tools{tools}")
    #         return tools
    #     except Exception as e:
    #         raise Exception(f"get_tool failed: {str(e)}") from e

    async def get_tools(self) -> list:
        try:
            tools = {}
            # 如果有知识库中的工具信息，则动态构建MCP工具
            if hasattr(self, 'knowledgeTool') and self.knowledgeTool:
                # 获取工具信息
                knowledge_item, tool_info = self.knowledgeTool

                if tool_info:

                    # 动态创建工具函数
                    def dynamic_frontend_tool_function(user_id: str, query_id: str, params: str):
                        self.logger.info(f"user id is {user_id} - query id is {query_id} - param is {params}")
                        try:
                            # 连接Redis
                            redis_conn = get_redis_connection()

                            # 构造Redis键
                            redis_key = f"tool_request_{query_id}_{user_id}"


                            param_dict = {"origin_params": json.loads(tool_info.params)}
                            if params:
                                # 将参数转换为JSON并存储到Redis
                                param_dict["llm_params"] = params

                            params_json = json.dumps(param_dict)

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
                    if tool_info.push == 1:
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

    async def create_agent(self, user_id, prompt, query_id, callback_handler):
        self.knowledgeTool = get_knowledge_tool(user_id,  prompt)
        user_prompt = self.generate_user_prompt(prompt, user_id, query_id)
        system_prompt = self.generate_system_prompt()
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
