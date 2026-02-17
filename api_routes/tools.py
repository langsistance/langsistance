#!/usr/bin/env python3

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from typing import List
from bs4 import BeautifulSoup
import json
import yaml
import re

from .models import (
    ToolAndKnowledgeCreateRequest, ToolAndKnowledgeCreateResponse,
    ToolUpdateRequest, ToolUpdateResponse,
    ToolDeleteRequest, ToolDeleteResponse,
    ToolQueryResponse, ToolItem,
    ToolFetchRequest, ToolFetchResponse,
    ToolResponseRequest, ToolResponseResponse,
    OpenAPISpecRequest, OpenAPISpecResponse,
    ToolCreateRequest, ToolCreateResponse
)
from sources.knowledge.knowledge import get_db_connection, get_redis_connection, create_tool_and_knowledge_records, get_tool_by_id
from sources.logger import Logger
from sources.user.passport import verify_firebase_token

logger = Logger("backend.log")
router = APIRouter()

@router.post("/create_tool_and_knowledge", response_model=ToolAndKnowledgeCreateResponse)
async def create_tool_and_knowledge(request: ToolAndKnowledgeCreateRequest, http_request:  Request):
    """
    创建tool和knowledge记录接口
    首先创建tool，然后使用tool的id创建knowledge
    """
    auth_header = http_request.headers.get("Authorization")
    user = verify_firebase_token(auth_header)

    user_id = user['uid']

    logger.info(f"Creating tool and knowledge records for user: {user_id}")

    # 参数校验 - Tool部分
    tool_errors = []

    if not request.tool_title or len(request.tool_title) > 100:
        tool_errors.append("tool_title is required and must be no more than 100 characters")

    if not request.tool_description or len(request.tool_description) > 5000:
        tool_errors.append("tool_description is required and must be no more than 5000 characters")

    if not request.tool_url or len(request.tool_url) > 4096:
        tool_errors.append("tool_url is required and must be no more than 4096 characters")

    if len(request.tool_params) > 5000:
        tool_errors.append("tool_params must be no more than 5000 characters")

    if tool_errors:
        logger.error(f"Tool validation errors: {tool_errors}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Tool validation failed",
                "errors": tool_errors,
                "tool_id": None,
                "knowledge_id": None
            }
        )

    # 参数校验 - Knowledge部分
    knowledge_errors = []

    if not request.knowledge_question or len(request.knowledge_question) > 100:
        knowledge_errors.append("knowledge_question is required and must be no more than 100 characters")

    if request.knowledge_description and len(request.knowledge_description) > 5000:
        knowledge_errors.append("knowledge_description is required and must be no more than 5000 characters")

    if not request.knowledge_answer or len(request.knowledge_answer) > 5000:
        knowledge_errors.append("knowledge_answer is required and must be no more than 5000 characters")

    if len(request.knowledge_model_name) > 200:
        knowledge_errors.append("knowledge_model_name must be no more than 200 characters")

    if len(request.knowledge_params) > 5000:
        knowledge_errors.append("knowledge_params must be no more than 5000 characters")

    if knowledge_errors:
        logger.error(f"Knowledge validation errors: {knowledge_errors}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Knowledge validation failed",
                "errors": knowledge_errors,
                "tool_id": None,
                "knowledge_id": None
            }
        )

    # 准备工具数据
    tool_data = {
        'user_id': user_id,
        'title': request.tool_title,
        'description': request.tool_description,
        'url': request.tool_url,
        'push': 1,
        'timeout': request.tool_timeout,
        'params': request.tool_params
    }

    # 准备知识数据
    knowledge_data = {
        'user_id': user_id,
        'question': request.knowledge_question,
        'description': request.knowledge_description,
        'answer': request.knowledge_answer,
        'public': request.knowledge_public,
        'embedding_id': request.knowledge_embeddingId,
        'model_name': request.knowledge_model_name,
        'params': request.knowledge_params
    }

    # 调用核心功能方法
    result = create_tool_and_knowledge_records(tool_data, knowledge_data)

    if result["success"]:
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": result["message"],
                "tool_id": result["tool_id"],
                "knowledge_id": result["knowledge_id"]
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": result["message"],
                "tool_id": result["tool_id"],
                "knowledge_id": result["knowledge_id"]
            }
        )


@router.post("/update_tool", response_model=ToolUpdateResponse)
async def update_tool(request: ToolUpdateRequest,  http_request:  Request):
    """
    更新工具接口（仅允许更新title、description、url和params字段）
    """
    auth_header = http_request.headers.get("Authorization")
    user = verify_firebase_token(auth_header)

    user_id = user['uid']

    logger.info(f"Updating tool {request.toolId} for user: {user_id}")

    # 参数校验
    errors = []

    if request.title is not None and (not request.title or len(request.title) > 100):
        errors.append("title must be between 1 and 100 characters")

    if request.description is not None and len(request.description) > 5000:
        errors.append("description must be no more than 5000 characters")

    if request.url is not None and (not request.url or len(request.url) > 1000):
        errors.append("url must be between 1 and 1000 characters")

    if request.params is not None and len(request.params) > 5000:
        errors.append("params must be no more than 5000 characters")

    if errors:
        logger.error(f"Validation errors: {errors}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Validation failed",
                "errors": errors
            }
        )

    connection = None
    try:
        # 获取数据库连接
        connection = get_db_connection()
        with connection.cursor() as cursor:
            # 首先检查记录是否存在以及用户ID是否匹配
            check_sql = "SELECT user_id FROM tools WHERE id = %s AND status = %s"
            cursor.execute(check_sql, (request.toolId, 1))
            result = cursor.fetchone()

            if not result:
                logger.warning(f"Tool record {request.toolId} not found or inactive")
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "message": "Tool record not found or inactive"
                    }
                )

            # 校验用户ID是否匹配
            record_user_id = result["user_id"]
            if str(record_user_id) != user_id:
                logger.warning(f"User {user_id} not authorized to update tool record {request.toolId}")
                return JSONResponse(
                    status_code=403,
                    content={
                        "success": False,
                        "message": "Not authorized to update this tool record"
                    }
                )

            # 构建更新语句（仅包含允许更新的字段）
            update_fields = []
            update_params = []

            if request.title is not None:
                update_fields.append("title = %s")
                update_params.append(request.title)

            if request.description is not None:
                update_fields.append("description = %s")
                update_params.append(request.description)

            if request.url is not None:
                update_fields.append("url = %s")
                update_params.append(request.url)

            if request.params is not None:
                update_fields.append("params = %s")
                update_params.append(request.params)

            # 如果没有任何字段需要更新
            if not update_fields:
                logger.info(f"No fields to update for tool record {request.toolId}")
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "message": "No fields to update"
                    }
                )

            # 添加工具ID到参数列表
            update_params.append(request.toolId)

            # 更新数据库记录
            update_sql = f"UPDATE tools SET {', '.join(update_fields)} WHERE id = %s"
            logger.info(f"update sql:{update_sql}")
            cursor.execute(update_sql, update_params)
            connection.commit()

            logger.info(f"Tool record {request.toolId} updated successfully")
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Tool record updated successfully"
                }
            )

    except Exception as e:
        logger.error(f"Error updating tool record: {str(e)}")
        if connection:
            connection.rollback()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Internal server error: {str(e)}"
            }
        )
    finally:
        if connection:
            connection.close()

@router.post("/delete_tool", response_model=ToolDeleteResponse)
async def delete_tool(request: ToolDeleteRequest, http_request:  Request):
    """
    删除工具接口（通过修改status字段实现软删除）
    """
    auth_header = http_request.headers.get("Authorization")
    user = verify_firebase_token(auth_header)

    user_id = user['uid']

    logger.info(f"Deleting tool {request.toolId} for user: {user_id}")

    # 参数校验
    errors = []

    if not request.toolId:
        errors.append("toolId is required")

    if errors:
        logger.error(f"Validation errors: {errors}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Validation failed",
                "errors": errors
            }
        )

    connection = None
    try:
        # 获取数据库连接
        connection = get_db_connection()
        with connection.cursor() as cursor:
            # 首先检查记录是否存在以及用户ID是否匹配
            check_sql = "SELECT user_id FROM tools WHERE id = %s"
            cursor.execute(check_sql, (request.toolId,))
            result = cursor.fetchone()

            if not result:
                logger.warning(f"Tool record {request.toolId} not found")
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "message": "Tool record not found"
                    }
                )
            logger.info(f"tool result:{result}")
            # 校验用户ID是否匹配
            record_user_id = result["user_id"]
            if str(record_user_id) != user_id:
                logger.warning(f"User {user_id} not authorized to delete tool record {request.toolId}")
                return JSONResponse(
                    status_code=403,
                    content={
                        "success": False,
                        "message": "Not authorized to delete this tool record"
                    }
                )

            # 软删除数据库记录（将status设置为0表示删除）
            delete_sql = "UPDATE tools SET status = %s WHERE id = %s"
            cursor.execute(delete_sql, (2, request.toolId))
            connection.commit()

            logger.info(f"Tool record {request.toolId} deleted successfully (status set to 0)")
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Tool record deleted successfully"
                }
            )

    except Exception as e:
        logger.error(f"Error deleting tool record: {str(e)}")
        if connection:
            connection.rollback()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Internal server error: {str(e)}"
            }
        )
    finally:
        if connection:
            connection.close()

@router.get("/query_tools", response_model=ToolQueryResponse)
async def query_tool_records(http_request: Request, query: str = "", limit: int = 10, offset: int = 0):
    """
    查询工具记录接口
    """
    auth_header = http_request.headers.get("Authorization")
    user = verify_firebase_token(auth_header)

    user_id = user['uid']

    # logger.info(f"Querying tool records for user: {userId}" + (f" with query: {query}" if query else ""))

    # 参数校验
    errors = []

    if limit <= 0 or limit > 100:
        errors.append("limit must be between 1 and 100")

    if offset < 0:
        errors.append("offset must be greater than or equal to 0")

    if errors:
        logger.error(f"Validation errors: {errors}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Validation failed",
                "errors": errors,
                "data": [],
                "total": 0
            }
        )

    connection = None
    try:
        # 获取数据库连接
        connection = get_db_connection()
        with connection.cursor() as cursor:
            # 构建查询条件
            # 1. 用户ID匹配
            # 2. 公开的工具 或者 用户自己的工具
            # 3. 如果query不为空，则进行title或description模糊匹配
            if query:
                search_pattern = f"%{query}%"
                where_condition = "AND (title LIKE %s OR description LIKE %s)"
                params = [1, user_id, search_pattern, search_pattern]
            else:
                where_condition = ""
                params = [1, user_id]

            count_sql = f"""
                        SELECT COUNT(*) as total
                        FROM tools
                        WHERE status = %s
                          AND user_id = %s
                          {where_condition}
                        """

            cursor.execute(count_sql, params)
            count_result = cursor.fetchone()
            total = count_result['total'] if count_result else 0

            if total == 0:
                # logger.info(f"No tool records found for user: {userId}" + (f" with query: {query}" if query else ""))
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "message": "No records found",
                        "data": [],
                        "total": 0
                    }
                )

            # 查询数据
            if query:
                query_sql = f"""
                            SELECT id,
                                   user_id,
                                   title,
                                   description,
                                   url, push, public, status, timeout, params, create_time, update_time
                            FROM tools
                            WHERE status = %s
                               AND user_id = %s
                              {where_condition}
                            ORDER BY update_time DESC
                                LIMIT %s
                            OFFSET %s
                            """
                params = [1, user_id, search_pattern, search_pattern, limit, offset * limit]
            else:
                query_sql = """
                            SELECT id,
                                   user_id,
                                   title,
                                   description,
                                   url, push, public, status, timeout, params, create_time, update_time
                            FROM tools
                            WHERE status = %s
                               AND user_id = %s
                            ORDER BY update_time DESC
                                LIMIT %s
                            OFFSET %s
                            """
                params = [1, user_id, limit, offset * limit]

            cursor.execute(query_sql, params)
            results = cursor.fetchall()

            # 转换为ToolItem对象列表
            tool_items = []
            for row in results:
                tool_item = ToolItem(
                    id=row['id'],
                    user_id=str(row['user_id']),
                    title=row['title'],
                    description=row['description'],
                    url=row['url'],
                    push=row['push'],
                    public=row['public'],
                    status=row['status'],
                    timeout=row['timeout'],
                    params=row['params']
                )
                tool_items.append(tool_item)

            # logger.info(f"Found {len(tool_items)} tool records for user: {userId}" + (f" with query: {query}" if query else ""))
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Tool records retrieved successfully",
                    "data": [item.dict() for item in tool_items],
                    "total": total
                }
            )

    except Exception as e:
        logger.error(f"Error querying tool records: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Internal server error: {str(e)}",
                "data": [],
                "total": 0
            }
        )
    finally:
        if connection:
            connection.close()

@router.get("/query_public_tools", response_model=ToolQueryResponse)
async def query_public_tools(query: str = "", limit: int = 10, offset: int = 0):
    """
    查询公开工具记录接口
    """
    logger.info(f"Querying public tool" + (f" with query: {query}" if query else ""))

    # 参数校验
    errors = []

    if limit <= 0 or limit > 100:
        errors.append("limit must be between 1 and 100")

    if offset < 0:
        errors.append("offset must be greater than or equal to 0")

    if errors:
        logger.error(f"Validation errors: {errors}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Validation failed",
                "errors": errors,
                "data": [],
                "total": 0
            }
        )

    connection = None
    try:
        # 获取数据库连接
        connection = get_db_connection()
        with connection.cursor() as cursor:
            # 构建查询条件
            # 1. 用户ID匹配
            # 2. 公开的工具 或者 用户自己的工具
            # 3. 如果query不为空，则进行title或description模糊匹配
            if query:
                search_pattern = f"%{query}%"
                where_condition = "AND (title LIKE %s OR description LIKE %s)"
                params = [1, 2, search_pattern, search_pattern]
            else:
                where_condition = ""
                params = [1, 2]

            count_sql = f"""
                        SELECT COUNT(*) as total
                        FROM tools
                        WHERE status = %s
                          AND public = %s
                          {where_condition}
                        """

            cursor.execute(count_sql, params)
            count_result = cursor.fetchone()
            total = count_result['total'] if count_result else 0

            if total == 0:
                logger.info(f"No public tool" + (f" with query: {query}" if query else ""))
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "message": "No records found",
                        "data": [],
                        "total": 0
                    }
                )

            # 查询数据
            if query:
                query_sql = f"""
                            SELECT id,
                                   user_id,
                                   title,
                                   description,
                                   url, push, public, status, timeout, params, create_time, update_time
                            FROM tools
                            WHERE status = %s
                               AND public = %s
                              {where_condition}
                            ORDER BY update_time DESC
                                LIMIT %s
                            OFFSET %s
                            """
                params = [1, 2, search_pattern, search_pattern, limit, offset * limit]
            else:
                query_sql = """
                            SELECT id,
                                   user_id,
                                   title,
                                   description,
                                   url, push, public, status, timeout, params, create_time, update_time
                            FROM tools
                            WHERE status = %s
                               AND public = %s
                            ORDER BY update_time DESC
                                LIMIT %s
                            OFFSET %s
                            """
                params = [1, 2, limit, offset * limit]

            cursor.execute(query_sql, params)
            results = cursor.fetchall()

            # 转换为ToolItem对象列表
            tool_items = []
            for row in results:
                tool_item = ToolItem(
                    id=row['id'],
                    user_id=str(row['user_id']),
                    title=row['title'],
                    description=row['description'],
                    url=row['url'],
                    push=row['push'],
                    public=row['public'],
                    status=row['status'],
                    timeout=row['timeout'],
                    params=row['params']
                )
                tool_items.append(tool_item)

            logger.info(f"Found {len(tool_items)} public tool" + (f" with query: {query}" if query else ""))
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Tool records retrieved successfully",
                    "data": [item.dict() for item in tool_items],
                    "total": total
                }
            )

    except Exception as e:
        logger.error(f"Error querying tool records: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Internal server error: {str(e)}",
                "data": [],
                "total": 0
            }
        )
    finally:
        if connection:
            connection.close()

@router.post("/get_tool_request", response_model=ToolFetchResponse)
async def get_tool_request(request: ToolFetchRequest, http_request: Request):
    """
    根据query_id和user_id从Redis获取工具对象

    Args:
        request: 包含query_id和user_id的请求对象

    Returns:
        ToolFetchResponse: 包含工具对象的响应
    """
    auth_header = http_request.headers.get("Authorization")
    user = verify_firebase_token(auth_header)

    user_id = user['uid']

    logger.info(f"get tool request for user: {user_id} with query_id: {request.query_id}")

    try:
        # 参数校验
        if not request.query_id:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "query_id is required"
                }
            )

        # 创建Redis连接
        redis_conn = None
        try:
            redis_conn = get_redis_connection()
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": f"Failed to connect to Redis: {str(e)}"
                }
            )

        # 构造Redis键
        redis_key = f"tool_request_{request.query_id}_{user_id}"

        # 从Redis获取工具对象
        try:
            tool_data_str = redis_conn.get(redis_key)
            if not tool_data_str:
                logger.warning(f"No tool found in Redis with key: {redis_key}")
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "message": "Tool not found",
                        "tool": None
                    }
                )

            # 尝试解析存储的工具数据
            tool_data = json.loads(tool_data_str)
            logger.info(f"Successfully retrieved tool from Redis with key: {redis_key}")

            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Tool retrieved successfully",
                    "params": tool_data
                }
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tool data from Redis: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Failed to parse tool data"
                }
            )
        except Exception as e:
            logger.error(f"Error retrieving tool from Redis: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": f"Error retrieving tool from Redis: {str(e)}"
                }
            )

    except Exception as e:
        logger.error(f"Error in fetch_tool: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Internal server error: {str(e)}"
            }
        )

@router.post("/save_tool_response", response_model=ToolResponseResponse)
async def save_tool_response(request: ToolResponseRequest, http_request: Request):
    """
    保存工具响应到Redis

    Args:
        request: 包含query_id、user_id和tool_response的请求对象

    Returns:
        ToolResponseResponse: 操作结果响应
    """
    auth_header = http_request.headers.get("Authorization")
    user = verify_firebase_token(auth_header)

    user_id = user['uid']

    try:
        # 参数校验
        if not request.query_id:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "query_id is required"
                }
            )

        if not request.tool_response:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "tool_response is required"
                }
            )

        # 处理tool_response中的html内容，移除HTML标签
        processed_tool_response = request.tool_response
        if isinstance(processed_tool_response, dict) and 'html' in processed_tool_response:
            html_content = processed_tool_response['html']
            if isinstance(html_content, str):
                #使用BeautifulSoup移除HTML标签
                cleaned_html = BeautifulSoup(html_content, "html.parser").get_text()
                # 创建一个新的字典副本
                processed_tool_response = processed_tool_response.copy()
                processed_tool_response['html'] = cleaned_html

        # 创建Redis连接
        redis_conn = None
        try:
            redis_conn = get_redis_connection()
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": f"Failed to connect to Redis: {str(e)}"
                }
            )

        # 构造Redis键
        redis_key = f"tool_response_{request.query_id}_{user_id}"

        # 将tool_response数据存储到Redis
        try:
            tool_response_str = json.dumps(processed_tool_response)
            redis_conn.set(redis_key, tool_response_str, ex=1200)
            logger.info(f"Successfully saved tool response to Redis with key: {redis_key}")

            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Tool response saved successfully"
                }
            )

        except json.JSONEncodeError as e:
            logger.error(f"Failed to serialize tool response data: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Failed to serialize tool response data"
                }
            )
        except Exception as e:
            logger.error(f"Error saving tool response to Redis: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": f"Error saving tool response to Redis: {str(e)}"
                }
            )

    except Exception as e:
        logger.error(f"Error in save_tool_response: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Internal server error: {str(e)}"
            }
        )


@router.get("/query_tool_by_id", response_model=ToolQueryResponse)
async def query_tool_by_id(http_request: Request, tool_id: int):
    """
    根据tool_id查询工具详情接口
    """
    user_id = None
    user_authenticated = False

    # 尝试验证用户身份
    try:
        auth_header = http_request.headers.get("Authorization")
        if auth_header:
            user = verify_firebase_token(auth_header)
            user_id = user['uid']
            user_authenticated = True
    except Exception as e:
        # 用户未登录或token验证失败
        logger.info(f"User not authenticated: {str(e)}")
        pass

    logger.info(f"Querying tool by id: {tool_id}")

    # 参数校验
    if not tool_id:
        logger.error("tool_id is required")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "tool_id is required"
            }
        )

    try:
        # 使用 knowledge.py 中的 get_tool_by_id 方法查询工具
        tool_item = get_tool_by_id(tool_id)

        if not tool_item:
            logger.info(f"No tool found with id: {tool_id}")
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": "Tool not found"
                }
            )

        tool_owner_id = tool_item.user_id

        # 用户未登录，无法访问工具
        if not user_authenticated:
            logger.warning(f"Unauthorized access to private tool {tool_id} by unauthenticated user")
            return JSONResponse(
                status_code=403,
                content={
                    "success": False,
                    "message": "Access denied. This is a private tool."
                }
            )

        # 用户已登录，但不是工具所有者
        if user_id != tool_owner_id:
            logger.warning(f"Unauthorized access to private tool {tool_id} by user {user_id}")
            return JSONResponse(
                status_code=403,
                content={
                    "success": False,
                    "message": "Access denied. This is a private tool."
                }
            )


        logger.info(f"Tool found with id: {tool_id}")
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Tool retrieved successfully",
                "data": tool_item.dict()
            }
        )

    except Exception as e:
        logger.error(f"Error querying tool by id: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Internal server error: {str(e)}"
            }
        )

@router.post("/create_tool_from_openapi", response_model=OpenAPISpecResponse)
async def create_tool_from_openapi(request: OpenAPISpecRequest, http_request: Request):
    """
    从OpenAPI规范创建工具接口

    Args:
        request: 包含OpenAPI规范格式和内容的请求对象
        http_request: HTTP请求对象，用于获取认证信息

    Returns:
        OpenAPISpecResponse: 工具创建结果响应
    """
    connection = None
    try:
        # 验证用户身份
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)
        user_id = user['uid']

        logger.info(f"Creating tool from OpenAPI spec for user: {user_id}")

        # 解析OpenAPI规范
        if request.spec_format.lower() == "json":
            try:
                openapi_spec = json.loads(request.spec_content)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON format: {str(e)}")
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "message": f"Invalid JSON format: {str(e)}"
                    }
                )
        elif request.spec_format.lower() == "yaml":
            try:
                openapi_spec = yaml.safe_load(request.spec_content)
            except yaml.YAMLError as e:
                logger.error(f"Invalid YAML format: {str(e)}")
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "message": f"Invalid YAML format: {str(e)}"
                    }
                )
        else:
            logger.error("Invalid spec format. Must be 'json' or 'yaml'")
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Invalid spec format. Must be 'json' or 'yaml'"
                }
            )

        # 提取必要信息
        title = openapi_spec.get('info', {}).get('title', 'Untitled API')
        description = openapi_spec.get('info', {}).get('description', '')

        # 获取基础URL
        base_url = ""
        if 'servers' in openapi_spec and len(openapi_spec['servers']) > 0:
            base_url = openapi_spec['servers'][0].get('url', '')

        # 验证base_url是否为标准的HTTP域名格式
        if base_url:
            # 验证URL格式的标准正则表达式
            url_pattern = r'^https?://[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*(:\d+)?(/.*)?$'
            if not re.match(url_pattern, base_url):
                logger.error(f"Invalid base_url format: {base_url}")
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "message": f"Invalid base_url format: {base_url}. Must be a standard HTTP/HTTPS domain."
                    }
                )

        # 提取第一个路径作为工具URL的一部分
        path_url = ""
        paths = openapi_spec.get('paths', {})
        if paths:
            # 获取第一个路径
            first_path = list(paths.keys())[0] if paths.keys() else ""
            path_url = first_path

        # 组合完整的URL
        full_url = base_url.rstrip('/') + path_url

        # 提取参数信息
        parameters_info = {}
        if paths:
            first_path = list(paths.keys())[0] if paths.keys() else ""
            path_item = paths.get(first_path, {})

            # 遍历所有HTTP方法(GET, POST, etc.)
            for method, operation in path_item.items():
                if isinstance(operation, dict) and 'parameters' in operation:
                    parameters_info[method] = operation['parameters']
                    break  # 只取第一个有参数的方法

        # 将参数信息序列化为JSON字符串
        params = json.dumps(parameters_info) if parameters_info else "{}"

        # 验证字段长度
        errors = []
        if len(title) > 100:
            errors.append("title must be no more than 100 characters")

        if len(description) > 5000:
            errors.append("description must be no more than 5000 characters")

        if len(full_url) > 4096:
            errors.append("url must be no more than 4096 characters")

        if len(params) > 5000:
            errors.append("spec content is too large, must be no more than 5000 characters")

        if errors:
            logger.error(f"Validation errors: {errors}")
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Validation failed",
                    "errors": errors
                }
            )

        # 插入数据库
        try:
            connection = get_db_connection()
            with connection.cursor() as cursor:
                insert_sql = """
                    INSERT INTO tools 
                    (user_id, title, description, url, push, public, status, timeout, params)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(insert_sql, (
                    user_id,
                    title,
                    description,
                    full_url,  # 使用组合后的完整URL
                    2,  # push后端调用工具为2
                    1,  # public默认为1
                    1,  # status默认为1(有效)
                    30,  # timeout默认为30秒
                    params  # 使用提取的参数信息
                ))
                connection.commit()

                # 获取插入的记录ID
                tool_id = cursor.lastrowid

                logger.info(f"Tool created successfully with ID: {tool_id}")
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "message": "Tool created successfully",
                        "tool_id": tool_id
                    }
                )

        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Database error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": f"Database error: {str(e)}"
                }
            )

    except Exception as e:
        logger.error(f"Error creating tool from OpenAPI spec: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Internal server error: {str(e)}"
            }
        )

    finally:
        if connection:
            connection.close()

@router.post("/create_tool", response_model=ToolCreateResponse)
async def create_tool(request: ToolCreateRequest, http_request: Request):
    """
    创建工具接口
    直接创建工具记录到tools表
    """
    auth_header = http_request.headers.get("Authorization")
    user = verify_firebase_token(auth_header)

    user_id = user['uid']

    logger.info(f"Creating tool for user: {user_id}")

    # 参数校验
    errors = []

    if not request.tool_title or len(request.tool_title) > 100:
        errors.append("tool_title is required and must be no more than 100 characters")

    if not request.tool_description or len(request.tool_description) > 5000:
        errors.append("tool_description is required and must be no more than 5000 characters")

    if not request.tool_url or len(request.tool_url) > 4096:
        errors.append("tool_url is required and must be no more than 4096 characters")

    if len(request.tool_params) > 5000:
        errors.append("tool_params must be no more than 5000 characters")

    if errors:
        logger.error(f"Validation errors: {errors}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Validation failed",
                "errors": errors,
                "tool_id": None,
            }
        )

    connection = None
    try:
        # 获取数据库连接
        connection = get_db_connection()
        with connection.cursor() as cursor:
            # 插入工具数据到数据库
            insert_sql = """
                INSERT INTO tools 
                (user_id, title, description, url, push, public, status, timeout, params)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_sql, (
                user_id,
                request.tool_title,
                request.tool_description,
                request.tool_url,
                1,
                1,
                1,  # status默认为1(有效)
                30,
                request.tool_params
            ))
            connection.commit()

            # 获取插入的记录ID
            tool_id = cursor.lastrowid

            logger.info(f"Tool created successfully with ID: {tool_id}")
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Tool created successfully",
                    "tool_id": tool_id,
                }
            )

    except Exception as e:
        if connection:
            connection.rollback()
        logger.error(f"Database error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Database error: {str(e)}",
                "tool_id": None,
            }
        )
    finally:
        if connection:
            connection.close()


@router.post("/create_tool_from_custom", response_model=OpenAPISpecResponse)
async def create_tool_from_custom(request: OpenAPISpecRequest, http_request: Request):
    """
    从自定义JSON创建工具接口

    Args:
        request: 包含自定义JSON格式的请求对象
        http_request: HTTP请求对象，用于获取认证信息

    Returns:
        ToolCustomCreateResponse: 工具创建结果响应
    """
    connection = None
    try:
        # 验证用户身份
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)
        user_id = user['uid']

        logger.info(f"Creating custom tool for user: {user_id}")

        # 解析传入的JSON内容
        try:
            custom_spec = json.loads(request.spec_content)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": f"Invalid JSON format: {str(e)}"
                }
            )

        # 验证必需的字段
        required_fields = ['url', 'method']
        for field in required_fields:
            if field not in custom_spec:
                logger.error(f"Missing required field: {field}")
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "message": f"Missing required field: {field}"
                    }
                )

        # 提取URL并获取最后一段path作为标题
        url = custom_spec['url']

        # 验证URL格式
        url_pattern = r'^https?://[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*(:\d+)?(/.*)?$'
        if not re.match(url_pattern, url):
            logger.error(f"Invalid URL format: {url}")
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": f"Invalid URL format: {url}. Must be a standard HTTP/HTTPS URL."
                }
            )

        # 从URL中提取最后一段path作为标题
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        path_parts = [part for part in parsed_url.path.split('/') if part]
        title = path_parts[-1] if path_parts else parsed_url.netloc  # 如果没有path，则使用域名

        # 构造params字段，包含除了url以外的所有其他字段
        params_data = {k: v for k, v in custom_spec.items() if k != 'url'}
        params = json.dumps(params_data)

        # 验证字段长度
        errors = []
        if len(title) > 100:
            errors.append("title must be no more than 100 characters")

        if len(url) > 4096:
            errors.append("url must be no more than 4096 characters")

        if len(params) > 5000:
            errors.append("params content is too large, must be no more than 5000 characters")

        if errors:
            logger.error(f"Validation errors: {errors}")
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Validation failed",
                    "errors": errors
                }
            )

        # 插入数据库
        try:
            connection = get_db_connection()
            with connection.cursor() as cursor:
                insert_sql = """
                    INSERT INTO tools 
                    (user_id, title, description, url, push, public, status, timeout, params)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """

                # 设置描述为URL的简化版本或者使用title作为描述
                description = url

                cursor.execute(insert_sql, (
                    user_id,
                    title,
                    description,  # 使用URL作为描述
                    url,  # 完整URL
                    2,  # push设为2
                    1,  # public默认为1
                    1,  # status默认为1(有效)
                    30,  # timeout默认为30秒
                    params  # 包含method, query, header, body的JSON
                ))
                connection.commit()

                # 获取插入的记录ID
                tool_id = cursor.lastrowid

                logger.info(f"Custom tool created successfully with ID: {tool_id}")
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "message": "Custom tool created successfully",
                        "tool_id": tool_id
                    }
                )

        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Database error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": f"Database error: {str(e)}"
                }
            )

    except Exception as e:
        logger.error(f"Error creating custom tool: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Internal server error: {str(e)}"
            }
        )

    finally:
        if connection:
            connection.close()

@router.post("/create_tool_from_url", response_model=OpenAPISpecResponse)
async def create_tool_from_url(urlRequest: UrlRequest, http_request: Request):
    """
    从自定义JSON创建工具接口

    Args:
        request: 包含自定义JSON格式的请求对象
        http_request: HTTP请求对象，用于获取认证信息

    Returns:
        ToolCustomCreateResponse: 工具创建结果响应
    """
    if not urlRequest.url:
        logger.error(f"Validation errors: missing url")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "missing url",
                "tool_id": None,
            }
        )
    url = urlRequest.url

    connection = None
    try:
        # 验证用户身份
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)
        user_id = user['uid']

        logger.info(f"Creating custom tool for user: {user_id}")

        # 验证URL格式
        url_pattern = r'^https?://[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*(:\d+)?(/.*)?$'
        if not re.match(url_pattern, url):
            logger.error(f"Invalid URL format: {url}")
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": f"Invalid URL format: {url}. Must be a standard HTTP/HTTPS URL."
                }
            )

        # 从URL中提取最后一段path作为标题
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        path_parts = [part for part in parsed_url.path.split('/') if part]
        title = path_parts[-1] if path_parts else parsed_url.netloc  # 如果没有path，则使用域名

        # 构造params字段，包含除了url以外的所有其他字段
        params_data = {
            "method": "GET",
            "Content-Type": "text/html",
            "query": {},
            "header": {},
            "body": {}
        }
        params = json.dumps(params_data)

        # 验证字段长度
        errors = []
        if len(title) > 100:
            errors.append("title must be no more than 100 characters")

        if len(url) > 4096:
            errors.append("url must be no more than 4096 characters")

        if len(params) > 5000:
            errors.append("params content is too large, must be no more than 5000 characters")

        if errors:
            logger.error(f"Validation errors: {errors}")
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Validation failed",
                    "errors": errors
                }
            )

        # 插入数据库
        try:
            connection = get_db_connection()
            with connection.cursor() as cursor:
                insert_sql = """
                    INSERT INTO tools 
                    (user_id, title, description, url, push, public, status, timeout, params)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """

                # 设置描述为URL的简化版本或者使用title作为描述
                description = url

                cursor.execute(insert_sql, (
                    user_id,
                    title,
                    description,  # 使用URL作为描述
                    url,  # 完整URL
                    3,  # push设为3
                    1,  # public默认为1
                    1,  # status默认为1(有效)
                    30,  # timeout默认为30秒
                    params  # 包含method, query, header, body的JSON
                ))
                connection.commit()

                # 获取插入的记录ID
                tool_id = cursor.lastrowid

                logger.info(f"Custom tool created successfully with ID: {tool_id}")
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "message": "Custom tool created successfully",
                        "tool_id": tool_id
                    }
                )

        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Database error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": f"Database error: {str(e)}"
                }
            )

    except Exception as e:
        logger.error(f"Error creating custom tool: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Internal server error: {str(e)}"
            }
        )

    finally:
        if connection:
            connection.close()