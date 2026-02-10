#!/usr/bin/env python3

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
import os
import uuid
import json
import asyncio
from functools import lru_cache
from datetime import datetime, timedelta

from sources.schemas import QueryResponse
from sources.logger import Logger
from api_routes.models import QueryRequest, QuestionRequest
from sources.knowledge.knowledge import get_knowledge_tool
from sources.user.passport import verify_firebase_token, check_and_increase_usage
from sources.callback.sse_callback import SSECallbackHandler

router = APIRouter()

# Agent pool for reusing agent instances
_agent_pool = {}
_agent_pool_lock = asyncio.Lock()
AGENT_POOL_MAX_SIZE = 10
AGENT_POOL_MAX_IDLE_TIME = 300  # 5 minutes

# Token cache with TTL
_token_cache = {}
_token_cache_lock = asyncio.Lock()
TOKEN_CACHE_TTL = 3600  # 1 hour

async def get_cached_token_validation(auth_header: str):
    """Cache token validation results to reduce Firebase API calls"""
    async with _token_cache_lock:
        if auth_header in _token_cache:
            cached_data, expiry = _token_cache[auth_header]
            if datetime.now() < expiry:
                return cached_data
            else:
                del _token_cache[auth_header]

    # Validate token if not cached
    user = await asyncio.to_thread(verify_firebase_token, auth_header)

    async with _token_cache_lock:
        _token_cache[auth_header] = (user, datetime.now() + timedelta(seconds=TOKEN_CACHE_TTL))
        # Cleanup old entries
        current_time = datetime.now()
        expired_keys = [k for k, (_, exp) in _token_cache.items() if current_time >= exp]
        for k in expired_keys:
            del _token_cache[k]

    return user

async def get_or_create_agent(create_agent_func, app_logger):
    """Get an agent from pool or create new one"""
    async with _agent_pool_lock:
        current_time = datetime.now()

        # Clean up expired agents
        expired_keys = [k for k, (_, last_used) in _agent_pool.items()
                       if (current_time - last_used).total_seconds() > AGENT_POOL_MAX_IDLE_TIME]
        for k in expired_keys:
            del _agent_pool[k]
            app_logger.info(f"Removed expired agent from pool: {k}")

        # Try to reuse an existing agent
        for agent_id, (agent, _) in _agent_pool.items():
            _agent_pool[agent_id] = (agent, current_time)
            app_logger.info(f"Reusing agent from pool: {agent_id}")
            return agent

        # Create new agent if pool is not full
        if len(_agent_pool) < AGENT_POOL_MAX_SIZE:
            agent = await create_agent_func()
            agent_id = str(uuid.uuid4())
            _agent_pool[agent_id] = (agent, current_time)
            app_logger.info(f"Created new agent and added to pool: {agent_id}")
            return agent

        # Pool is full, create temporary agent (not cached)
        app_logger.warning("Agent pool is full, creating temporary agent")
        return await create_agent_func()

async def check_usage_async(user_id: str) -> bool:
    """Async wrapper for check_and_increase_usage"""
    return await asyncio.to_thread(check_and_increase_usage, user_id)

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
                        "url": tool_info.url,
                        "params": tool_info.params
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

        # Optimize: Use cached token validation
        auth_header = http_request.headers.get("Authorization")
        try:
            user = await get_cached_token_validation(auth_header)
        except Exception as e:
            app_logger.error(f"Token validation failed: {str(e)}")
            return JSONResponse(status_code=401, content={"error": "Invalid token"})

        user_id = user['uid']

        # Optimize: Make usage check async
        allowed = await check_usage_async(user_id)
        if not allowed:
            return JSONResponse(status_code=429, content={"error": "Daily API usage limit exceeded (100/day)"})

        # 如果没有提供 query_id，自动生成一个
        if not request.query_id:
            app_logger.warning("query id is none.")
            return JSONResponse(status_code=400, content={"error": "query_id is required"})

        if is_generating_flag:
            app_logger.warning("Another query is being processed, please wait.")
            return JSONResponse(status_code=429, content={"error": "Another query is being processed"})

        async def generate():
            # Optimize: Reuse agent from pool instead of creating new one
            general_agent = await get_or_create_agent(create_agent_func, app_logger)
            queue = asyncio.Queue()
            handler = SSECallbackHandler(queue)

            try:
                openai_agent = await general_agent.create_agent(user_id, request.query, request.query_id, request.tool_data, handler)
            except Exception as e:
                app_logger.error(f"Failed to create agent: {str(e)}")
                yield f"data:{json.dumps({'error': 'Failed to create agent'})}\n\n"
                return

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

            # Optimize: Batch tokens to reduce serialization overhead
            token_buffer = []
            buffer_size = 5  # Send 5 tokens at once
            last_flush_time = asyncio.get_event_loop().time()
            flush_interval = 0.05  # Flush every 50ms

            try:
                while True:
                    try:
                        # Try to get event with timeout to enable periodic flushing
                        event = await asyncio.wait_for(queue.get(), timeout=flush_interval)

                        if event['type'] == 'token':
                            token_buffer.append(event['content'])

                            current_time = asyncio.get_event_loop().time()
                            # Flush if buffer is full or enough time has passed
                            if len(token_buffer) >= buffer_size or (current_time - last_flush_time) >= flush_interval:
                                # Send buffered tokens
                                combined = ''.join(token_buffer)
                                token_json = json.dumps(combined)
                                yield f"data:{token_json}\n\n"
                                token_buffer.clear()
                                last_flush_time = current_time

                        elif event['type'] == 'end':
                            # Flush remaining tokens before ending
                            if token_buffer:
                                combined = ''.join(token_buffer)
                                token_json = json.dumps(combined)
                                yield f"data:{token_json}\n\n"
                                token_buffer.clear()
                            break

                        elif event['type'] == 'error':
                            # Flush tokens and send error
                            if token_buffer:
                                combined = ''.join(token_buffer)
                                token_json = json.dumps(combined)
                                yield f"data:{token_json}\n\n"
                                token_buffer.clear()
                            error_json = json.dumps({'error': event.get('message', 'Unknown error')})
                            yield f"data:{error_json}\n\n"
                            break

                    except asyncio.TimeoutError:
                        # Periodic flush even without new tokens
                        if token_buffer:
                            current_time = asyncio.get_event_loop().time()
                            if (current_time - last_flush_time) >= flush_interval:
                                combined = ''.join(token_buffer)
                                token_json = json.dumps(combined)
                                yield f"data:{token_json}\n\n"
                                token_buffer.clear()
                                last_flush_time = current_time

            except Exception as e:
                app_logger.error(f"Error in generate loop: {str(e)}")
                error_json = json.dumps({'error': str(e)})
                yield f"data:{error_json}\n\n"
            finally:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
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