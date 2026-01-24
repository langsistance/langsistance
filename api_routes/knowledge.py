#!/usr/bin/env python3

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from typing import List

from .models import (
    KnowledgeCreateRequest, KnowledgeCreateResponse,
    KnowledgeDeleteRequest, KnowledgeDeleteResponse,
    KnowledgeUpdateRequest, KnowledgeUpdateResponse,
    KnowledgeQueryResponse, KnowledgeItem,
    KnowledgeCopyRequest, KnowledgeCopyResponse
)
from sources.knowledge.knowledge import get_embedding, get_db_connection, get_redis_connection, create_tool_and_knowledge_records, get_tool_by_id
from sources.logger import Logger
from sources.user.passport import verify_firebase_token, get_user_by_id

logger = Logger("backend.log")
router = APIRouter()

@router.post("/create_knowledge", response_model=KnowledgeCreateResponse)
async def create_knowledge_record(request: KnowledgeCreateRequest, http_request: Request):
    """
    创建知识记录接口
    """
    auth_header = http_request.headers.get("Authorization")
    user = verify_firebase_token(auth_header)

    user_id = user['uid']

    logger.info(f"Creating knowledge record for user: {user_id}")

    # 参数校验
    errors = []

    if not request.question or len(request.question) > 100:
        errors.append("question is required and must be no more than 100 characters")

    if request.description and len(request.description) > 5000:
        errors.append("description is required and must be no more than 5000 characters")

    if not request.answer or len(request.answer) > 5000:
        errors.append("answer is required and must be no more than 5000 characters")

    if request.modelName and len(request.modelName) > 200:
        errors.append("modelName must be no more than 200 characters")

    if len(request.params) > 5000:
        errors.append("params must be no more than 5000 characters")

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
            # 验证tool_id是否属于当前用户
            check_tool_sql = "SELECT id FROM tools WHERE id = %s AND user_id = %s AND status = 1"
            cursor.execute(check_tool_sql, (request.toolId, user_id))
            tool_result = cursor.fetchone()

            if not tool_result:
                logger.warning(f"Tool {request.toolId} not found or not owned by user {user_id}")
                return JSONResponse(
                    status_code=403,
                    content={
                        "success": False,
                        "message": "Tool not found or not owned by current user"
                    }
                )

            # 插入数据
            sql = """
                  INSERT INTO knowledge
                  (user_id, question, description, answer, public, model_name, tool_id, params, status, embedding_id)
                  VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                  """
            cursor.execute(sql, (
                user_id,
                request.question,
                request.description,
                request.answer,
                request.public,
                "gpt-4o-mini",
                request.toolId,
                request.params,
                1,
                0
            ))
            connection.commit()

            # 获取插入的记录ID
            record_id = cursor.lastrowid
            logger.info(f"Knowledge record created successfully with ID: {record_id}")

            query_embedding = get_embedding(request.question + request.answer)

            # 将 embedding 写入 Redis
            try:
                redis_conn = get_redis_connection()
                # 使用记录ID作为键，将embedding存储到Redis中
                redis_key = f"knowledge_embedding_{record_id}"
                redis_conn.set(redis_key, str(query_embedding))
                logger.info(f"Embedding stored in Redis with key: {redis_key}")
            except Exception as redis_error:
                logger.error(f"Failed to store embedding in Redis: {str(redis_error)}")
                # 注意：即使Redis存储失败，我们也不会中断主流程

            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Knowledge record created successfully",
                    "id": record_id
                }
            )

    except Exception as e:
        logger.error(f"Error creating knowledge record: {str(e)}")
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


@router.post("/delete_knowledge", response_model=KnowledgeDeleteResponse)
async def delete_knowledge_record(request: KnowledgeDeleteRequest, http_request: Request):
    """
    删除知识记录接口
    """
    auth_header = http_request.headers.get("Authorization")
    user = verify_firebase_token(auth_header)

    user_id = user['uid']

    logger.info(f"Deleting knowledge record {request.knowledgeId} for user: {user_id}")

    # 参数校验
    errors = []

    if not request.knowledgeId:
        errors.append("knowledgeId is required")

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
            check_sql = "SELECT user_id FROM knowledge WHERE id = %s"
            cursor.execute(check_sql, (request.knowledgeId,))
            result = cursor.fetchone()

            if not result:
                logger.warning(f"Knowledge record {request.knowledgeId} not found")
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "message": "Knowledge record not found"
                    }
                )

            # 校验用户ID是否匹配
            record_user_id = result["user_id"]
            if str(record_user_id) != user_id:
                logger.warning(f"User {user_id} not authorized to delete knowledge record {request.knowledgeId}")
                return JSONResponse(
                    status_code=403,
                    content={
                        "success": False,
                        "message": "Not authorized to delete this knowledge record"
                    }
                )

            # 删除数据库记录
            delete_sql = "UPDATE knowledge SET status = %s WHERE id = %s"
            cursor.execute(delete_sql, (2, request.knowledgeId))
            connection.commit()

            # 删除Redis中的embedding
            try:
                redis_conn = get_redis_connection()
                redis_key = f"knowledge_embedding_{request.knowledgeId}"
                redis_result = redis_conn.delete(redis_key)
                if redis_result:
                    logger.info(f"Embedding deleted from Redis with key: {redis_key}")
                else:
                    logger.warning(f"Embedding not found in Redis with key: {redis_key}")
            except Exception as redis_error:
                logger.error(f"Failed to delete embedding from Redis: {str(redis_error)}")
                # 即使Redis删除失败，我们也不会中断主流程

            logger.info(f"Knowledge record {request.knowledgeId} deleted successfully")
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Knowledge record deleted successfully"
                }
            )

    except Exception as e:
        logger.error(f"Error deleting knowledge record: {str(e)}")
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

@router.post("/update_knowledge", response_model=KnowledgeUpdateResponse)
async def update_knowledge_record(request: KnowledgeUpdateRequest, http_request: Request):
    """
    修改知识记录接口
    """
    auth_header = http_request.headers.get("Authorization")
    user = verify_firebase_token(auth_header)

    user_id = user['uid']

    logger.info(f"Updating knowledge record {request.knowledgeId} for user: {user_id}")

    # 参数校验
    errors = []

    if not request.knowledgeId:
        errors.append("knowledgeId is required")

    if request.question and len(request.question) > 100:
        errors.append("question must be no more than 100 characters")

    if request.description and len(request.description) > 5000:
        errors.append("description must be no more than 5000 characters")

    if request.answer and len(request.answer) > 5000:
        errors.append("answer must be no more than 5000 characters")

    if request.modelName and len(request.modelName) > 200:
        errors.append("modelName must be no more than 200 characters")

    if request.params and len(request.params) > 5000:
        errors.append("params must be no more than 5000 characters")

    if request.toolId and not request.toolId:
        errors.append("toolId must be a valid integer")

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
            check_sql = "SELECT user_id, question, answer FROM knowledge WHERE id = %s AND status = %s"
            cursor.execute(check_sql, (request.knowledgeId, 1))
            result = cursor.fetchone()

            if not result:
                logger.warning(f"Knowledge record {request.knowledgeId} not found or inactive")
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "message": "Knowledge record not found or inactive"
                    }
                )

            # 校验用户ID是否匹配
            record_user_id = result["user_id"]
            if str(record_user_id) != user_id:
                logger.warning(f"User {user_id} not authorized to update knowledge record {request.knowledgeId}")
                return JSONResponse(
                    status_code=403,
                    content={
                        "success": False,
                        "message": "Not authorized to update this knowledge record"
                    }
                )

            # 构建更新语句
            update_fields = []
            update_params = []

            if request.question is not None:
                update_fields.append("question = %s")
                update_params.append(request.question)

            if request.description is not None:
                update_fields.append("description = %s")
                update_params.append(request.description)

            if request.answer is not None:
                update_fields.append("answer = %s")
                update_params.append(request.answer)

            if request.public is not None:
                update_fields.append("public = %s")
                update_params.append(request.public)

            if request.modelName is not None:
                update_fields.append("model_name = %s")
                update_params.append(request.modelName)

            # if request.toolId is not None:
            #     update_fields.append("tool_id = %s")
            #     update_params.append(request.toolId)

            if request.params is not None:
                update_fields.append("params = %s")
                update_params.append(request.params)

            # 如果没有任何字段需要更新
            if not update_fields:
                logger.info(f"No fields to update for knowledge record {request.knowledgeId}")
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "message": "No fields to update"
                    }
                )

            # 添加记录ID到参数列表
            update_params.append(request.knowledgeId)

            # 更新数据库记录
            update_sql = f"UPDATE knowledge SET {', '.join(update_fields)} WHERE id = %s"
            cursor.execute(update_sql, update_params)
            connection.commit()

            # 检查是否需要重新计算embedding (question或answer有变更)
            need_recalculate_embedding = False
            original_question = result["question"]
            original_answer = result["answer"]

            if ((request.question is not None and request.question != original_question) or
                    (request.answer is not None and request.answer != original_answer)):
                need_recalculate_embedding = True

            if need_recalculate_embedding:
                # 获取更新后的question和answer
                new_question = request.question if request.question is not None else original_question
                new_answer = request.answer if request.answer is not None else original_answer

                # 重新计算embedding
                query_embedding = get_embedding(new_question + new_answer)

                # 更新Redis中的embedding
                try:
                    redis_conn = get_redis_connection()
                    redis_key = f"knowledge_embedding_{request.knowledgeId}"
                    redis_conn.set(redis_key, str(query_embedding))
                    logger.info(f"Embedding updated in Redis with key: {redis_key}")
                except Exception as redis_error:
                    logger.error(f"Failed to update embedding in Redis: {str(redis_error)}")
                    # 即使Redis更新失败，我们也不会中断主流程

            logger.info(f"Knowledge record {request.knowledgeId} updated successfully")
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Knowledge record updated successfully"
                }
            )

    except Exception as e:
        logger.error(f"Error updating knowledge record: {str(e)}")
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

@router.get("/query_knowledge", response_model=KnowledgeQueryResponse)
async def query_knowledge_records(http_request: Request, query: str, limit: int = 10, offset: int = 0):
    """
    查询知识记录接口
    """
    # logger.info(f"Querying knowledge records for user: {userId} with query: {query}")

    auth_header = http_request.headers.get("Authorization")
    user = verify_firebase_token(auth_header)

    user_id = user['uid']
    email = user['email']

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
            # 2. 公开的知识 或者 用户自己的知识
            # 3. question、description、answer字段模糊匹配
            if query:
                search_pattern = f"%{query}%"
                where_condition = "AND (question LIKE %s OR description LIKE %s OR answer LIKE %s)"
                params = [1, user_id, search_pattern, search_pattern, search_pattern]
            else:
                where_condition = ""
                params = [1, user_id]

            count_sql = f"""
                        SELECT COUNT(*) as total
                        FROM knowledge
                        WHERE status = %s
                          AND  user_id = %s
                          {where_condition}
                        """

            cursor.execute(count_sql, params)
            count_result = cursor.fetchone()
            total = count_result['total'] if count_result else 0

            if total == 0:
                # logger.info(f"No knowledge records found for user: {userId} with query: {query}")
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
                                   question,
                                   description,
                                   answer, public, model_name, tool_id, params, create_time, update_time
                            FROM knowledge
                            WHERE status = %s
                               AND user_id = %s
                              {where_condition}
                            ORDER BY update_time DESC
                                LIMIT %s
                            OFFSET %s
                            """
                params = [1, user_id, search_pattern, search_pattern, search_pattern, limit, offset]
            else:
                query_sql = """
                            SELECT id,
                                   user_id,
                                   question,
                                   description,
                                   answer, public, model_name, tool_id, params, create_time, update_time
                            FROM knowledge
                            WHERE status = %s
                               AND user_id = %s
                            ORDER BY update_time DESC
                                LIMIT %s
                            OFFSET %s
                            """
                params = [1, user_id, limit, offset]

            cursor.execute(query_sql, params)
            results = cursor.fetchall()

            # 转换为KnowledgeItem对象列表
            knowledge_items = []
            tool_ids = set()  # 收集所有相关的tool_id
            for row in results:
                knowledge_item = KnowledgeItem(
                    id=row['id'],
                    user_id=str(row['user_id']),
                    question=row['question'],
                    description=row['description'],
                    answer=row['answer'],
                    public=row['public'],
                    model_name=row['model_name'] or "",
                    tool_id=row['tool_id'] or 0,
                    params=row['params'] or ""
                )
                # 处理时间字段
                if row['create_time']:
                    knowledge_item.create_time = row['create_time'].isoformat() if hasattr(row['create_time'],
                                                                                           'isoformat') else str(
                        row['create_time'])
                if row['update_time']:
                    knowledge_item.update_time = row['update_time'].isoformat() if hasattr(row['update_time'],
                                                                                           'isoformat') else str(
                        row['update_time'])

                knowledge_items.append(knowledge_item)

                # 收集工具ID用于后续查询
                if row['tool_id']:
                    tool_ids.add(row['tool_id'])

            # 查询对应的工具记录
            tool_items = []
            if tool_ids:

                for tool_id in tool_ids:
                    tool_item = get_tool_by_id(tool_id)
                    if tool_item:
                        tool_items.append(tool_item)

            combined_data = {
                "knowledge": [item.dict() for item in knowledge_items],
                "tools": [item.dict() for item in tool_items]
            }
            # logger.info(f"Found {len(knowledge_items)} knowledge records for user: {userId} with query: {query}")
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Knowledge records retrieved successfully",
                    "data": combined_data,
                    "total": total
                }
            )

    except Exception as e:
        logger.error(f"Error querying knowledge records: {str(e)}")
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

@router.get("/query_public_knowledge", response_model=KnowledgeQueryResponse)
async def query_public_knowledge(query: str, limit: int = 10, offset: int = 0):
    """
    查询公开知识记录接口
    """
    logger.info(f"Querying public knowledge with query: {query}")

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
            # 2. 公开的知识 或者 用户自己的知识
            # 3. question、description、answer字段模糊匹配
            if query:
                search_pattern = f"%{query}%"
                where_condition = "AND (question LIKE %s OR description LIKE %s OR answer LIKE %s)"
                params = [1, 2, search_pattern, search_pattern, search_pattern]
            else:
                where_condition = ""
                params = [1, 2]

            count_sql = f"""
                        SELECT COUNT(*) as total
                        FROM knowledge
                        WHERE status = %s
                          AND  public = %s
                          {where_condition}
                        """

            cursor.execute(count_sql, params)
            count_result = cursor.fetchone()
            total = count_result['total'] if count_result else 0

            if total == 0:
                logger.info(f"No public knowledge with query: {query}")
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
                                   question,
                                   description,
                                   answer, public, model_name, tool_id, params, create_time, update_time
                            FROM knowledge
                            WHERE status = %s
                               AND public = %s
                              {where_condition}
                            ORDER BY update_time DESC
                                LIMIT %s
                            OFFSET %s
                            """
                params = [1, 2, search_pattern, search_pattern, search_pattern, limit, offset]
            else:
                query_sql = """
                            SELECT id,
                                   user_id,
                                   question,
                                   description,
                                   answer, public, model_name, tool_id, params, create_time, update_time
                            FROM knowledge
                            WHERE status = %s
                               AND public = %s
                            ORDER BY update_time DESC
                                LIMIT %s
                            OFFSET %s
                            """
                params = [1, 2, limit, offset]

            cursor.execute(query_sql, params)
            results = cursor.fetchall()

            # 转换为KnowledgeItem对象列表
            knowledge_items = []
            for row in results:
                knowledge_item = KnowledgeItem(
                    id=row['id'],
                    user_id=str(row['user_id']),
                    question=row['question'],
                    description=row['description'],
                    answer=row['answer'],
                    public=row['public'],
                    model_name=row['model_name'] or "",
                    tool_id=row['tool_id'] or 0,
                    params=row['params'] or ""
                )

                # 添加用户邮箱到extra_info字段
                user_info = get_user_by_id(row['user_id'])
                if user_info:
                    knowledge_item.extra_info = {
                        "email": user_info['email']
                    }

                # 处理时间字段
                if row['create_time']:
                    knowledge_item.create_time = row['create_time'].isoformat() if hasattr(row['create_time'],
                                                                                         'isoformat') else str(
                        row['create_time'])
                if row['update_time']:
                    knowledge_item.update_time = row['update_time'].isoformat() if hasattr(row['update_time'],
                                                                                         'isoformat') else str(
                        row['update_time'])

                knowledge_items.append(knowledge_item)

            logger.info(f"Found {len(knowledge_items)} public knowledge with query: {query}")
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Knowledge records retrieved successfully",
                    "data": [item.dict() for item in knowledge_items],
                    "total": total
                }
            )

    except Exception as e:
        logger.error(f"Error querying knowledge records: {str(e)}")
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


@router.post("/copy_knowledge", response_model=KnowledgeCopyResponse)
async def copy_knowledge(request: KnowledgeCopyRequest, http_request: Request):
    """
    复制知识记录接口（包括关联的工具）
    """

    auth_header = http_request.headers.get("Authorization")
    user = verify_firebase_token(auth_header)

    user_id = user['uid']

    logger.info(f"Copy knowledge record for user: {user_id}")

    # 参数校验
    errors = []

    if not request.knowledgeId:
        errors.append("knowledgeId is required")

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

            # 查询要复制的知识记录
            query_sql = """
                        SELECT id, user_id, question, description, answer, 
                               public, model_name, tool_id, params
                        FROM knowledge
                        WHERE id = %s AND status = %s
                        """
            params = [request.knowledgeId, 1]

            cursor.execute(query_sql, params)
            results = cursor.fetchall()
            if len(results) == 0:
                logger.info(f"knowledge not exist: {request.knowledgeId}")
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "message": "knowledge not exist"
                    }
                )

            row = results[0]
            logger.info(f"row:{row}")

            # 准备知识数据
            knowledge_data = {
                'user_id': user_id,  # 新的所有者
                'question': row["question"],
                'description': row["description"],
                'answer': row["answer"],
                'public': 1,  # 设为私有
                'embedding_id': 0,
                'model_name': row["model_name"],
                'params': row["params"]
            }

            # 准备工具数据（如果存在）
            tool_data = None
            if row["tool_id"]:
                # 查询工具详情
                tool_query_sql = """
                    SELECT title, description, url, push, public, timeout, params, user_id
                    FROM tools
                    WHERE id = %s AND status = 1
                """
                cursor.execute(tool_query_sql, (row["tool_id"],))
                tool_result = cursor.fetchone()

                if tool_result:
                    tool_data = {
                        'id': row["tool_id"],
                        'user_id': user_id,  # 新的所有者
                        'title': tool_result['title'],
                        'description': tool_result['description'],
                        'url': tool_result['url'],
                        'push': tool_result['push'],
                        'public': 1,  # 设为私有
                        'timeout': tool_result['timeout'],
                        'params': tool_result['params']
                    }

            # 调用核心方法创建工具和知识记录
            result = create_tool_and_knowledge_records(tool_data, knowledge_data)

            if not result["success"]:
                raise Exception(result["message"])

            new_knowledge_id = result["knowledge_id"]

            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Knowledge record copied successfully",
                    "id": new_knowledge_id
                }
            )

    except Exception as e:
        logger.error(f"Error copy knowledge: {str(e)}")
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


@router.post("/authorize_knowledge_access")
async def authorize_knowledge_access(request: Request, auth_request: dict):
    """
    知识授权接口
    """
    # 验证用户登录态
    auth_header = request.headers.get("Authorization")
    user = verify_firebase_token(auth_header)

    user_id = user['uid']
    email = user['email']
    target_email = auth_request.get("email")
    knowledge_id = auth_request.get("knowledgeId")

    # 参数校验
    errors = []

    if not user_id:
        errors.append("User authentication failed")

    if not target_email or len(target_email) > 255:
        errors.append("Target email is required and must be no more than 255 characters")

    if not knowledge_id:
        errors.append("knowledgeId is required")

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
            # 检查知识是否存在且属于当前用户
            check_knowledge_sql = "SELECT id, user_id, public FROM knowledge WHERE id = %s AND status = 1"
            cursor.execute(check_knowledge_sql, (knowledge_id,))
            knowledge_result = cursor.fetchone()

            if not knowledge_result:
                logger.warning(f"Knowledge record {knowledge_id} not found or inactive")
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "message": "Knowledge record not found or inactive"
                    }
                )

            # 校验用户是否有权限授权该知识
            knowledge_owner_id = str(knowledge_result["user_id"])
            is_public = knowledge_result["public"]

            if knowledge_owner_id != user_id:
                logger.warning(f"User {user_id} not authorized to grant access to knowledge {knowledge_id}")
                return JSONResponse(
                    status_code=403,
                    content={
                        "success": False,
                        "message": "Not authorized to grant access to this knowledge"
                    }
                )

            # 检查目标用户是否存在
            # check_target_user_sql = "SELECT id FROM users WHERE email = %s"
            # cursor.execute(check_target_user_sql, (target_email,))
            # target_user_result = cursor.fetchone()
            #
            # if not target_user_result:
            #     logger.warning(f"Target user with email {target_email} not found")
            #     return JSONResponse(
            #         status_code=404,
            #         content={
            #             "success": False,
            #             "message": "Target user not found"
            #         }
            #     )

            # 获取目标用户的user_id
            # target_user_id = target_user_result["id"]

            # 检查是否已经存在相同的待处理的授权记录
            check_auth_sql = "SELECT id FROM knowledge_share WHERE from_user_id = %s AND to_user_email = %s AND knowledge_id = %s AND status = %s"
            cursor.execute(check_auth_sql, (user_id, target_email, knowledge_id, 1))
            existing_auth = cursor.fetchone()

            if existing_auth:
                logger.info(f"Authorization already exists for user {target_email} on knowledge {knowledge_id}")
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "message": "Authorization already exists"
                    }
                )

            # 插入授权记录，使用目标用户的user_id
            insert_auth_sql = """
                INSERT INTO knowledge_share
                (to_user_email, knowledge_id, from_user_id, from_user_email, status) 
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(insert_auth_sql, (target_email, knowledge_id, user_id, email, 1))
            connection.commit()

            logger.info(
                f"Granted access to user {target_email} (from user_id: {user_id}) for knowledge {knowledge_id}")
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Knowledge access granted successfully"
                }
            )

    except Exception as e:
        logger.error(f"Error authorizing knowledge access: {str(e)}")
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

@router.post("/handle_knowledge_share")
async def handle_knowledge_share(request: Request, handle_request: dict):
    """
    处理知识分享请求（接受或拒绝）
    """
    # 验证用户登录态
    auth_header = request.headers.get("Authorization")
    user = verify_firebase_token(auth_header)

    user_id = user['uid']
    email = user['email']
    knowledge_share_id = handle_request.get("share_id")
    action = handle_request.get("action")  # "accept" 或 "reject"

    # 参数校验
    errors = []

    if not user_id:
        errors.append("User authentication failed")

    if not knowledge_share_id:
        errors.append("shareId is required")

    if not action or action not in ["accept", "reject"]:
        errors.append("action must be either 'accept' or 'reject'")

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
            # 检查分享记录是否存在且是给当前用户的
            check_share_sql = """
                SELECT id, to_user_email, knowledge_id, from_user_id, status
                FROM knowledge_share
                WHERE id = %s AND to_user_email = %s AND status = 1
            """
            cursor.execute(check_share_sql, (knowledge_share_id, email))
            share_result = cursor.fetchone()

            if not share_result:
                logger.warning(f"Knowledge share record {knowledge_share_id} not found or not for current user")
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "message": "Share record not found or not authorized"
                    }
                )

            knowledge_id = share_result["knowledge_id"]

            if action == "reject":
                # 拒绝分享，更新状态为已处理
                update_share_sql = "UPDATE knowledge_share SET status = 2 WHERE id = %s"
                cursor.execute(update_share_sql, (knowledge_share_id,))
                connection.commit()

                logger.info(f"User {user_id} rejected knowledge share {knowledge_share_id}")
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "message": "Knowledge share rejected"
                    }
                )
            else:  # accept
                # 先查询知识详情
                knowledge_query_sql = """
                    SELECT question, description, answer, public, model_name, tool_id, params, user_id
                    FROM knowledge
                    WHERE id = %s AND status = 1
                """
                cursor.execute(knowledge_query_sql, (knowledge_id,))
                knowledge_result = cursor.fetchone()

                if not knowledge_result:
                    logger.warning(f"Knowledge record {knowledge_id} not found or inactive")
                    return JSONResponse(
                        status_code=404,
                        content={
                            "success": False,
                            "message": "Knowledge record not found or inactive"
                        }
                    )

                # 准备知识数据
                knowledge_data = {
                    'user_id': user_id,  # 新的所有者
                    'question': knowledge_result['question'],
                    'description': knowledge_result['description'],
                    'answer': knowledge_result['answer'],
                    'public': 1,  # 设为私有
                    'embedding_id': 0,
                    'model_name': knowledge_result['model_name'],
                    'params': knowledge_result['params']
                }

                # 准备工具数据（如果存在）
                tool_data = None
                if knowledge_result['tool_id']:
                    # 查询工具详情
                    tool_query_sql = """
                        SELECT title, description, url, push, public, timeout, params, user_id
                        FROM tools
                        WHERE id = %s AND status = 1
                    """
                    cursor.execute(tool_query_sql, (knowledge_result['tool_id'],))
                    tool_result = cursor.fetchone()

                    if tool_result:
                        tool_data = {
                            'id': knowledge_result['tool_id'],
                            'user_id': user_id,  # 新的所有者
                            'title': tool_result['title'],
                            'description': tool_result['description'],
                            'url': tool_result['url'],
                            'push': tool_result['push'],
                            'public': 1,  # 设为私有
                            'timeout': tool_result['timeout'],
                            'params': tool_result['params']
                        }

                # 调用核心方法创建工具和知识记录
                result = create_tool_and_knowledge_records(tool_data, knowledge_data)

                if not result["success"]:
                    raise Exception(result["message"])

                new_knowledge_id = result["knowledge_id"]

                # 更新分享记录状态为已处理
                update_share_sql = "UPDATE knowledge_share SET status = 3 WHERE id = %s"
                cursor.execute(update_share_sql, (knowledge_share_id,))
                connection.commit()

                logger.info(
                    f"User {user_id} accepted knowledge share {knowledge_share_id}, created knowledge {new_knowledge_id}")
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "message": "Knowledge share accepted successfully",
                        "knowledgeId": new_knowledge_id
                    }
                )

    except Exception as e:
        logger.error(f"Error handling knowledge share: {str(e)}")
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

@router.get("/query_knowledge_shares", response_model=KnowledgeQueryResponse)
async def query_knowledge_shares(http_request: Request, limit: int = 10, offset: int = 0):
    """
    查询用户收到的知识分享请求
    """
    # 验证用户登录态
    auth_header = http_request.headers.get("Authorization")
    user = verify_firebase_token(auth_header)

    user_id = user['uid']
    email = user['email']

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
            # 先查询knowledge_share表获取分享记录总数
            count_sql = """
                SELECT COUNT(*) as total
                FROM knowledge_share 
                WHERE to_user_email = %s AND status = 1
            """
            cursor.execute(count_sql, (email,))
            count_result = cursor.fetchone()
            total = count_result['total'] if count_result else 0

            if total == 0:
                logger.info(f"No knowledge shares found for user: {email}")
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "message": "No shares found",
                        "data": [],
                        "total": 0
                    }
                )

            # 查询knowledge_share表获取分享记录
            share_query_sql = """
                SELECT id, knowledge_id, from_user_id, from_user_email, to_user_email, status, create_time, update_time
                FROM knowledge_share 
                WHERE to_user_email = %s AND status = 1
                ORDER BY create_time DESC
                LIMIT %s OFFSET %s
            """
            cursor.execute(share_query_sql, (email, limit, offset))
            share_results = cursor.fetchall()

            # 收集所有knowledge_id用于查询知识详情
            knowledge_ids = [str(share["knowledge_id"]) for share in share_results]

            # 根据knowledge_ids查询knowledge表
            if knowledge_ids:
                knowledge_query_sql = f"""
                    SELECT id, user_id, question, description, answer, public, 
                           model_name, tool_id, params, create_time, update_time
                    FROM knowledge
                    WHERE id IN ({','.join(['%s'] * len(knowledge_ids))}) AND status = 1
                """
                cursor.execute(knowledge_query_sql, knowledge_ids)
                knowledge_results = cursor.fetchall()

                # 创建knowledge_id到knowledge记录的映射
                knowledge_map = {k["id"]: k for k in knowledge_results}
            else:
                knowledge_map = {}

            # 组装返回数据
            knowledge_items = []
            for share in share_results:
                knowledge_id = share["knowledge_id"]
                if knowledge_id in knowledge_map:
                    knowledge_data = knowledge_map[knowledge_id]
                    knowledge_item = KnowledgeItem(
                        id=knowledge_data["id"],
                        user_id=user_id,
                        question=knowledge_data["question"],
                        description=knowledge_data["description"],
                        answer=knowledge_data["answer"],
                        public=knowledge_data["public"],
                        model_name=knowledge_data["model_name"] or "",
                        tool_id=knowledge_data["tool_id"] or 0,
                        params=knowledge_data["params"] or "",
                        extra_info={
                            "from_user_email": share["from_user_email"],  # 添加from_user_email字段
                            "share_id": share["id"],
                            "status": share["status"],
                            "to_user_email": share["to_user_email"],
                            "share_create_time": share["create_time"].isoformat() if hasattr(share["create_time"], "isoformat") else str(share["create_time"]),
                            "share_update_time": share["update_time"].isoformat() if hasattr(share["update_time"], "isoformat") else str(share["update_time"])
                        }
                    )

                    # 处理时间字段
                    if knowledge_data["create_time"]:
                        knowledge_item.create_time = knowledge_data["create_time"].isoformat() \
                            if hasattr(knowledge_data["create_time"], "isoformat") else str(
                            knowledge_data["create_time"])
                    if knowledge_data["update_time"]:
                        knowledge_item.update_time = knowledge_data["update_time"].isoformat() \
                            if hasattr(knowledge_data["update_time"], "isoformat") else str(
                            knowledge_data["update_time"])

                    knowledge_items.append(knowledge_item)

            logger.info(f"Found {len(knowledge_items)} knowledge shares for user: {email}")
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Knowledge shares retrieved successfully",
                    "data": [item.dict() for item in knowledge_items],
                    "total": total
                }
            )

    except Exception as e:
        logger.error(f"Error querying knowledge shares: {str(e)}")
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

@router.get("/get_user_shared_knowledge", response_model=KnowledgeQueryResponse)
async def get_user_shared_knowledge(http_request: Request, limit: int = 10, offset: int = 0):
    """
    根据分享人查询知识分享记录
    """
    # 验证用户登录态
    auth_header = http_request.headers.get("Authorization")
    user = verify_firebase_token(auth_header)

    user_id = user['uid']

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
            # 先查询knowledge_share表获取分享记录总数
            count_sql = """
                SELECT COUNT(*) as total
                FROM knowledge_share 
                WHERE from_user_id = %s
            """
            cursor.execute(count_sql, (user_id,))
            count_result = cursor.fetchone()
            total = count_result['total'] if count_result else 0

            if total == 0:
                logger.info(f"No knowledge shares found from user: {user_id}")
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "message": "No shares found",
                        "data": [],
                        "total": 0
                    }
                )

            # 查询knowledge_share表获取分享记录
            share_query_sql = """
                SELECT id, knowledge_id, to_user_email, status, create_time, update_time
                FROM knowledge_share 
                WHERE from_user_id = %s
                ORDER BY create_time DESC
                LIMIT %s OFFSET %s
            """
            cursor.execute(share_query_sql, (user_id, limit, offset))
            share_results = cursor.fetchall()

            # 收集所有knowledge_id用于查询知识详情
            knowledge_ids = [str(share["knowledge_id"]) for share in share_results]

            # 根据knowledge_ids查询knowledge表
            if knowledge_ids:
                knowledge_query_sql = f"""
                    SELECT id, user_id, question, description, answer, public, 
                           model_name, tool_id, params, create_time, update_time
                    FROM knowledge
                    WHERE id IN ({','.join(['%s'] * len(knowledge_ids))})
                """
                cursor.execute(knowledge_query_sql, knowledge_ids)
                knowledge_results = cursor.fetchall()

                # 创建knowledge_id到knowledge记录的映射
                knowledge_map = {k["id"]: k for k in knowledge_results}
            else:
                knowledge_map = {}

            # 组装返回数据
            knowledge_items = []
            for share in share_results:
                knowledge_id = share["knowledge_id"]
                if knowledge_id in knowledge_map:
                    knowledge_data = knowledge_map[knowledge_id]
                    knowledge_item = KnowledgeItem(
                        id=knowledge_data["id"],
                        user_id=str(knowledge_data["user_id"]),
                        question=knowledge_data["question"],
                        description=knowledge_data["description"],
                        answer=knowledge_data["answer"],
                        public=knowledge_data["public"],
                        model_name=knowledge_data["model_name"] or "",
                        tool_id=knowledge_data["tool_id"] or 0,
                        params=knowledge_data["params"] or "",
                        extra_info={
                            "share_id": share["id"],
                            "status": share["status"],
                            "to_user_email": share["to_user_email"],
                            "share_create_time": share["create_time"].isoformat() if hasattr(share["create_time"], "isoformat") else str(share["create_time"]),
                            "share_update_time": share["update_time"].isoformat() if hasattr(share["update_time"], "isoformat") else str(share["update_time"])
                        }
                    )

                    # 处理时间字段
                    if knowledge_data["create_time"]:
                        knowledge_item.create_time = knowledge_data["create_time"].isoformat() \
                            if hasattr(knowledge_data["create_time"], "isoformat") else str(
                            knowledge_data["create_time"])
                    if knowledge_data["update_time"]:
                        knowledge_item.update_time = knowledge_data["update_time"].isoformat() \
                            if hasattr(knowledge_data["update_time"], "isoformat") else str(
                            knowledge_data["update_time"])

                    knowledge_items.append(knowledge_item)

            logger.info(f"Found {len(knowledge_items)} knowledge shares from user: {user_id}")
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Knowledge shares retrieved successfully",
                    "data": [item.dict() for item in knowledge_items],
                    "total": total
                }
            )

    except Exception as e:
        logger.error(f"Error querying knowledge shares by from_user: {str(e)}")
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

@router.post("/cancel_knowledge_share")
async def cancel_knowledge_share(request: Request, cancel_request: dict):
    """
    取消知识分享接口
    根据登录用户的user_id和share_id查询knowledge_share表的from_user_id和id，
    如果存在记录将status的状态改成4
    """
    # 验证用户登录态
    auth_header = request.headers.get("Authorization")
    user = verify_firebase_token(auth_header)

    user_id = user['uid']
    share_id = cancel_request.get("share_id")

    # 参数校验
    errors = []

    if not user_id:
        errors.append("User authentication failed")

    if not share_id:
        errors.append("shareId is required")

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
            # 查询分享记录，验证from_user_id是否为当前用户
            check_share_sql = """
                SELECT id, from_user_id, status
                FROM knowledge_share
                WHERE id = %s AND from_user_id = %s
            """
            cursor.execute(check_share_sql, (share_id, user_id))
            share_result = cursor.fetchone()

            if not share_result:
                logger.warning(f"Knowledge share record {share_id} not found or not owned by user {user_id}")
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "message": "Share record not found or not owned by current user"
                    }
                )

            # 检查当前状态，避免重复取消
            current_status = share_result["status"]
            if current_status == 4:
                logger.info(f"Knowledge share {share_id} is already canceled")
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "message": "Share record is already canceled"
                    }
                )

            # 更新分享记录状态为4（已取消）
            update_share_sql = "UPDATE knowledge_share SET status = 4 WHERE id = %s"
            cursor.execute(update_share_sql, (share_id,))
            connection.commit()

            logger.info(f"User {user_id} canceled knowledge share {share_id}")
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Knowledge share canceled successfully"
                }
            )

    except Exception as e:
        logger.error(f"Error canceling knowledge share: {str(e)}")
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
