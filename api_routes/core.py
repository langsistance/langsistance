#!/usr/bin/env python3

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
import os
import uuid
import json
import asyncio

from sources.schemas import QueryResponse
from sources.logger import Logger
from api_routes.models import QueryRequest, QuestionRequest
from sources.knowledge.knowledge import get_knowledge_tool
from sources.user.passport import verify_firebase_token, check_and_increase_usage
from sources.callback.sse_callback import SSECallbackHandler

router = APIRouter()

def register_core_routes(app_logger, interaction_ref, query_resp_history_ref, config_ref, is_generating_flag, think_wrapper_func, create_agent_func):
    """注册核心路由并传递所需的依赖"""

    @router.get("/latest_answer")
    async def get_latest_answer():
        app_logger.info("Latest answer endpoint called")
        if interaction_ref.current_agent is None:
            return JSONResponse(status_code=404, content={"error": "No agent available"})

        uid = str(uuid.uuid4())
        if not any(q["answer"] == interaction_ref.current_agent.last_answer for q in query_resp_history_ref):
            query_resp = {
                "done": "false",
                "answer": interaction_ref.current_agent.last_answer,
                "reasoning": interaction_ref.current_agent.last_reasoning,
                "agent_name": interaction_ref.current_agent.agent_name if interaction_ref.current_agent else "None",
                "success": interaction_ref.current_agent.success,
                "blocks": {f'{i}': block.jsonify() for i, block in enumerate(interaction_ref.get_last_blocks_result())} if interaction_ref.current_agent else {},
                "status": interaction_ref.current_agent.get_status_message if interaction_ref.current_agent else "No status available",
                "uid": uid
            }
            interaction_ref.current_agent.last_answer = ""
            interaction_ref.current_agent.last_reasoning = ""
            query_resp_history_ref.append(query_resp)
            return JSONResponse(status_code=200, content=query_resp)

        if query_resp_history_ref:
            return JSONResponse(status_code=200, content=query_resp_history_ref[-1])
        return JSONResponse(status_code=404, content={"error": "No answer available"})

    @router.post("/query")
    async def process_query(request: QueryRequest, http_request: Request):
        app_logger.info(f"Processing query: {request.query}")
        app_logger.info("Processing start begin")

        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)

        user_id = user['uid']

        allowed = check_and_increase_usage(user_id)
        if not allowed:
            return JSONResponse(status_code=429, content="Daily API usage limit exceeded (100/day)")

        # 如果没有提供 query_id，自动生成一个
        if not request.query_id:
            request.query_id = str(uuid.uuid4())

        query_resp = QueryResponse(
            done="false",
            answer="",
            reasoning="",
            agent_name="Unknown",
            success=False,
            blocks={},
            status="Ready",
            uid=str(uuid.uuid4())
        )

        if is_generating_flag:
            app_logger.warning("Another query is being processed, please wait.")
            return JSONResponse(status_code=429, content=query_resp.jsonify())

        try:
            # is_generating = True  # Uncomment if needed
            # 调用 think_wrapper_func 来处理查询
            success = await think_wrapper_func(user_id, interaction_ref, request.query, request.query_id)

            if not success:
                query_resp.answer = interaction_ref.last_answer
                query_resp.reasoning = interaction_ref.last_reasoning
                return JSONResponse(status_code=400, content=query_resp.jsonify())

            if interaction_ref.current_agent:
                blocks_json = {f'{i}': block.jsonify() for i, block in enumerate(interaction_ref.current_agent.get_blocks_result())}
            else:
                app_logger.error("No current agent found")
                blocks_json = {}
                query_resp.answer = "Error: No current agent"
                return JSONResponse(status_code=400, content=query_resp.jsonify())

            app_logger.info(f"Answer: {interaction_ref.last_answer}")
            app_logger.info(f"Blocks: {blocks_json}")
            query_resp.done = "true"
            query_resp.answer = interaction_ref.last_answer
            query_resp.reasoning = interaction_ref.last_reasoning
            query_resp.agent_name = interaction_ref.current_agent.agent_name
            query_resp.success = interaction_ref.last_success
            query_resp.blocks = blocks_json

            query_resp_dict = {
                "done": query_resp.done,
                "answer": query_resp.answer,
                "agent_name": query_resp.agent_name,
                "success": query_resp.success,
                "blocks": query_resp.blocks,
                "status": query_resp.status,
                "uid": query_resp.uid
            }
            query_resp_history_ref.append(query_resp_dict)

            app_logger.info("Query processed successfully")
            return JSONResponse(status_code=200, content=query_resp.jsonify())

        except Exception as e:
            app_logger.error(f"An error occurred: {str(e)}")
            # sys.exit(1)  # 不应该在路由中退出应用
            return JSONResponse(status_code=500, content={"error": "Internal server error"})
        finally:
            app_logger.info("Processing finished")
            if config_ref.getboolean('MAIN', 'save_session'):
                interaction_ref.save_session()

    @router.get("/screenshot")
    async def get_screenshot():
        app_logger.info("Screenshot endpoint called")
        screenshot_path = ".screenshots/updated_screen.png"
        if os.path.exists(screenshot_path):
            return FileResponse(screenshot_path)
        app_logger.error("No screenshot available")
        return JSONResponse(
            status_code=404,
            content={"error": "No screenshot available"}
        )

    @router.post("/find_knowledge_tool")
    async def find_knowledge_tool(request: QuestionRequest, http_request: Request):
        """根据用户问题查找最相关的知识及其对应的工具"""
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)

        user_id = user['uid']

        app_logger.info(f"Finding knowledge tool for user: {user_id} with question: {request.question}")

        try:
            # 参数校验

            if not request.question:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "message": "question is required"
                    }
                )

            # 调用knowledge.py中的方法获取知识项和工具信息
            knowledge_item, tool_info = get_knowledge_tool(
                user_id,
                request.question,
                request.top_k,
                0
            )

            if not knowledge_item:
                app_logger.info("No matching knowledge found above similarity threshold")
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": False,
                        "message": "No matching knowledge found above similarity threshold"
                    }
                )

            # 构建返回的知识记录对象
            knowledge_response = {
                "userId": knowledge_item.user_id,
                "question": knowledge_item.question,
                "description": knowledge_item.description,
                "answer": knowledge_item.answer,
                "public": knowledge_item.public,
                "modelName": knowledge_item.model_name,
                "toolId": knowledge_item.tool_id,
                "params": knowledge_item.params
            }

            response_data = {
                "success": True,
                "message": "Knowledge and tool found successfully",
                "knowledge": knowledge_response
            }

            if tool_info:
                # 检查tool_info的push字段
                if tool_info.push == 2:
                    # 如果push为2，返回空对象
                    tool_response = {}
                else:
                    # 如果push不为2，保持原有逻辑
                    tool_response = {
                        "id": tool_info.id,
                        "title": tool_info.title,
                        "description": tool_info.description,
                        "url": tool_info.url
                    }

                tool_response["push"] = tool_info.push
                response_data["tool"] = tool_response

            app_logger.info(f"Successfully found knowledge and tool for user: {user_id}")
            return JSONResponse(
                status_code=200,
                content=response_data
            )

        except Exception as e:
            app_logger.error(f"Error in find_knowledge_tool: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": f"Internal server error: {str(e)}"
                }
            )

    @router.post("/query_stream")
    async def process_query_stream(request: QueryRequest, http_request: Request):
        app_logger.info(f"Processing query_stream: {request.query}")

        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)

        user_id = user['uid']

        allowed = check_and_increase_usage(user_id)
        if not allowed:
            return JSONResponse(status_code=429, content="Daily API usage limit exceeded (100/day)")

        query_resp = {
            "done" : "false",
            "success" : "false"
        }

        # 如果没有提供 query_id，自动生成一个
        if not request.query_id:
            app_logger.warning("query id is none.")
            return JSONResponse(status_code=429, content=json.dumps(query_resp))

        if is_generating_flag:
            app_logger.warning("Another query is being processed, please wait.")
            return JSONResponse(status_code=429, content=json.dumps(query_resp))

        async def generate():
            general_agent = await create_agent_func()
            queue = asyncio.Queue()
            handler = SSECallbackHandler(queue)
            openai_agent = await general_agent.create_agent(user_id, request.query, request.query_id, handler)

            async def run_agent():
                try:
                    await general_agent.invoke_agent(openai_agent, handler)
                    await queue.put({'type': 'end', 'content': '[DONE]'})
                except Exception as e:
                    app_logger.error(f"invoke agent fail. An error occurred: {str(e)}")
                    await queue.put({'type': 'error', 'message': str(e)})
                    await queue.put({'type': 'end'})
                finally:
                    handler.queue.put_nowait({'type': 'done'})

            task = asyncio.create_task(run_agent())

            try:
                while True:
                    event = await queue.get()
                    if event['type'] == 'token':
                        # 将token内容进行JSON编码，确保换行符等特殊字符被正确处理
                        token_json = json.dumps(event['content'])
                        yield f"data:{token_json}\n\n"

                    if event['type'] == 'end':
                        break
            finally:
                if not task.done():
                    task.cancel()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )

    @router.get("/copiioai_statistics")
    async def get_statistics():
        """获取今日知识创建和用户提问统计"""
        from datetime import datetime

        # Mock 数据
        today = datetime.now().strftime("%Y-%m-%d")

        stats = {
            "date": today,
            "knowledge_created": 42,  # 今日创建的知识数量
            "knowledge_shared": 41,  # 今日创建的知识数量
            "user_questions": 156,  # 今日用户提问次数
            "active_users": 23,  # 今日活跃用户数
            "knowledge_tools_used": 18,  # 今日知识工具使用次数
            "success_rate": "92.3%"  # 成功率
        }

        return JSONResponse(status_code=200, content=stats)

    return router