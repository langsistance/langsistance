import os
import json
import uuid
from openai import OpenAI
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from sources.logger import Logger
import pymysql
import pymysql.cursors
import redis

from sources.utility import pretty_print

# 设置 OpenAI API 密钥
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    # base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com")  # 设置基础 URL
)

# 存储用户知识库 {user_id: [{"id": str, "question": str, "answer": str, "embedding": list, "params": dict}]}
user_knowledge_bases: Dict[str, List[Dict]] = {}

logger = Logger("knowledge.log")


class UserQuestion(BaseModel):
    user_id: str
    question: str
    top_k: Optional[int] = 3  # 返回最相关的几个结果
    similarity_threshold: Optional[float] = 0.7  # 相似度阈值


class KnowledgeBaseResponse(BaseModel):
    success: bool
    message: str
    item_id: Optional[str] = None



class KnowledgeItem(BaseModel):
    id: int
    user_id: str
    question: str
    description: Optional[str] = None
    answer: str
    public: Optional[int]
    model_name: Optional[str]
    tool_id: Optional[int]
    params: Optional[str] = None
    create_time: Optional[str] = None
    update_time: Optional[str] = None
    extra_info: Optional[Dict[str, Any]] = None


class ToolItem(BaseModel):
    id: int
    user_id: str
    title: str
    description: str
    push: int
    url: Optional[str]
    status: Optional[bool]
    timeout: Optional[int]
    params: Optional[str] = None
    create_time: Optional[str] = None
    update_time: Optional[str] = None


def get_embedding(text: str) -> List[float]:
    """使用 OpenAI 获取文本的嵌入向量"""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error in get_embedding: {str(e)}")
        raise e

def get_user_vector_indices(user_id: str, embeddings_list: List, knowledge_items: List[Dict]):

    user_vector_indices = {user_id: {
        "embeddings": np.array(embeddings_list) if embeddings_list else np.array([]),
        "items": knowledge_items
    }}
    return user_vector_indices

def search_knowledge_base(user_id: str, query_embedding: List[float], user_vector_indices: Dict, top_k: int = 3, threshold: float = 0):
    """在用户知识库中搜索最相关的内容"""
    if user_id not in user_vector_indices or len(user_vector_indices[user_id]["embeddings"]) == 0:
        return []

    # 计算余弦相似度
    similarities = cosine_similarity(
        [query_embedding],
        user_vector_indices[user_id]["embeddings"]
    )[0]
    logger.info(f"similarities: {similarities}")
    # 获取最相似的结果
    results = []
    for i, similarity in enumerate(similarities):
        if similarity >= threshold:
            # 由于items中存储的是KnowledgeItem对象，我们需要创建一个新的字典来存储相似度
            item = user_vector_indices[user_id]["items"][i]
            # 创建一个包含KnowledgeItem字段和相似度的新字典
            result_item = {
                "id": item.id,
                "user_id": item.user_id,
                "question": item.question,
                "description": item.description,
                "answer": item.answer,
                "public": item.public,
                "model_name": item.model_name,
                "tool_id": item.tool_id,
                "params": item.params,
                "create_time": item.create_time,
                "update_time": item.update_time,
                "similarity": float(similarity)
            }
            results.append(result_item)
            logger.info(f"item:{result_item}")
    logger.info(f"results: {results}")
    # 按相似度排序并返回前top_k个结果
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def generate_answer_with_context(question: str, context: List[Dict]) -> str:
    """使用 OpenAI 生成基于上下文的答案"""
    if not context:
        return "抱歉，我在您的知识库中没有找到相关信息。"

    # 构建上下文提示
    context_text = "\n".join([f"问题: {item['question']}\n答案: {item['answer']}" for item in context])

    prompt = f"""
    基于以下上下文信息，请回答用户的问题。如果上下文中的信息不足以回答问题，请如实告知。

    上下文信息:
    {context_text}

    用户问题: {question}

    请基于上述上下文提供准确、简洁的回答:
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个有帮助的助手，基于提供的上下文信息回答问题。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"生成答案时出错: {str(e)}"

def get_db_connection():
    """创建并返回数据库连接"""
    db_config = {
        'host': os.getenv('MYSQL_HOST', 'langsistance_db'),
        'port' : int(os.getenv('MYSQL_PORT', 3306)),
        'user': os.getenv('MYSQL_USER', 'langsistance_user'),
        'password': os.getenv('MYSQL_PASSWORD', ''),
        'database': os.getenv('MYSQL_DATABASE', 'langsistance_db'),
        'charset': 'utf8mb4',
        'cursorclass': pymysql.cursors.DictCursor
    }
    return pymysql.connect(**db_config)

def get_redis_connection():
    """创建并返回 Redis 连接"""
    # 优先从环境变量获取 Redis 配置
    redis_host = os.getenv('REDIS_HOST')
    redis_port = int(os.getenv('REDIS_PORT'))
    logger.info(f"redis_host: {redis_host}, redis_port: {redis_port}")
    try:
        return redis.Redis(host=redis_host, port=redis_port, decode_responses=True, socket_connect_timeout=10, socket_timeout=10)
    except Exception as e:
        logger.error(f"Failed to create Redis connection: {str(e)}")
        raise e



def get_user_knowledge(user_id: str) -> List[KnowledgeItem]:
    """
    根据用户ID从数据库查询有效的知识记录，包括用户拥有的知识和被授权的知识

    Args:
        user_id (str): 用户ID

    Returns:
        List[KnowledgeItem]: 用户的知识记录列表
    """
    try:
        connection = get_db_connection()

        try:
            with connection.cursor() as cursor:
                # 查询用户自己的知识记录 (status=1表示有效)
                user_knowledge_sql = """
                    SELECT id, user_id, question, description, answer, public, model_name, tool_id, params, create_time, update_time
                    FROM knowledge
                    WHERE status = %s
                       AND user_id = %s
                    ORDER BY update_time DESC
                """

                cursor.execute(user_knowledge_sql, (1, user_id))
                user_knowledge_results = cursor.fetchall()

                # 将查询结果转换为KnowledgeItem对象列表
                knowledge_items = []
                for row in user_knowledge_results:
                    knowledge_item = KnowledgeItem(
                        id=row['id'],
                        user_id=str(row['user_id']),
                        question=row['question'],
                        description=row['description'] or "",
                        answer=row['answer'],
                        public=row['public'] or False,
                        model_name=row['model_name'] or "",
                        tool_id=row['tool_id'] or 0,
                        params=row['params'] or "",
                        create_time=row['create_time'].isoformat() if row['create_time'] else None,
                        update_time=row['update_time'].isoformat() if row['update_time'] else None
                    )
                    knowledge_items.append(knowledge_item)

                return knowledge_items
        finally:
            connection.close()

    except Exception as e:
        pretty_print(f"Error querying user knowledge: {str(e)}", color="error")
        return []


def get_knowledge_tool(user_id: str, question: str, top_k: int = 3, similarity_threshold: float = 0) -> Tuple[
    Optional[KnowledgeItem], Optional[ToolItem]]:
    """
    根据用户问题查找最相关的知识及其对应的工具

    Args:
        user_id: 用户ID
        question: 用户问题
        top_k: 返回最相关的几个结果
        similarity_threshold: 相似度阈值

    Returns:
        Tuple[KnowledgeToolItem, ToolItem, float]: 知识项、工具项和相似度
    """
    try:
        # 1. 计算问题的embedding
        query_embedding = get_embedding(question)
        logger.info(f"Generated embedding for question: {question}")

        # 2. 从MySQL查询该用户的所有有效知识记录
        knowledge_results = get_user_knowledge(user_id)

        if not knowledge_results:
            logger.info(f"No knowledge records found for user: {user_id}")
            return None, None

        logger.info(f"Found {len(knowledge_results)} knowledge records for user: {user_id}")

        # 3. 从Redis查询所有知识记录的embedding
        redis_conn = get_redis_connection()
        knowledge_embeddings = {}

        for knowledge in knowledge_results:
            knowledge_id = knowledge.id
            redis_key = f"knowledge_embedding_{knowledge_id}"
            embedding_str = redis_conn.get(redis_key)
            logger.info(f"embedding key is {redis_key}")
            if embedding_str:
                # 将字符串转换回embedding列表
                embedding = eval(embedding_str)  # 注意：在生产环境中应使用更安全的方法如json.loads
                knowledge_embeddings[knowledge_id] = embedding
                logger.info(f"Retrieved embedding for knowledge ID: {knowledge_id}")
            else:
                logger.warning(f"No embedding found in Redis for knowledge ID: {knowledge_id}")

        # 4. 构建用于相似度计算的数据结构
        if not knowledge_embeddings:
            logger.warning("No embeddings found for any knowledge records")
            return None, None

        # 构建向量索引数据结构
        embeddings_list = []
        knowledge_items = []

        for knowledge in knowledge_results:
            knowledge_id = knowledge.id
            if knowledge_id in knowledge_embeddings:
                embeddings_list.append(knowledge_embeddings[knowledge_id])
                knowledge_items.append(knowledge)

        if not embeddings_list:
            logger.warning("No valid embeddings available for similarity calculation")
            return None, None

        # 构建临时的向量索引用于搜索
        temp_user_vector_indices = get_user_vector_indices(user_id, embeddings_list, knowledge_items)
        logger.info(f"temp_user_vector_indices:{temp_user_vector_indices}")
        # 5. 使用search_knowledge_base方法找出最接近的知识
        search_results = search_knowledge_base(
            user_id,  # 使用临时用户ID
            query_embedding,
            temp_user_vector_indices,
            top_k,
            similarity_threshold
        )
        logger.info(f"search_results:{search_results}")
        # 清理临时向量索引
        if user_id in temp_user_vector_indices:
            del temp_user_vector_indices[user_id]

        if not search_results:
            logger.info("No matching knowledge found above similarity threshold")
            return None, None

        # 获取最相似的知识记录
        best_knowledge = search_results[0]

        # 6. 根据知识记录中的tool_id查询对应的工具信息
        tool_info = None
        if best_knowledge["tool_id"]:
            try:
                connection = get_db_connection()
                try:
                    with connection.cursor() as cursor:
                        # 查询工具信息
                        tool_query_sql = """
                                         SELECT id, \
                                                user_id, \
                                                title, \
                                                description, \
                                                url, \
                                                push, \
                                                status, \
                                                timeout, \
                                                params
                                         FROM tools
                                         WHERE id = %s \
                                           AND status = %s
                                         """
                        cursor.execute(tool_query_sql, (best_knowledge['tool_id'], 1))
                        tool_result = cursor.fetchone()

                        if tool_result:
                            tool_info = ToolItem(
                                id=tool_result['id'],
                                user_id=str(tool_result['user_id']),
                                title=tool_result['title'],
                                description=tool_result['description'],
                                url=tool_result['url'],
                                push=tool_result['push'],
                                status=tool_result['status'],
                                timeout=tool_result['timeout'],
                                params=tool_result['params']
                            )
                            logger.info(f"Retrieved tool info for tool ID: {best_knowledge['tool_id']}")
                        else:
                            logger.warning(f"No tool found for tool ID: {best_knowledge['tool_id']}")
                finally:
                    connection.close()
            except Exception as e:
                logger.error(f"Error retrieving tool info: {str(e)}")

        # 构建返回的知识记录对象
        knowledge_item = KnowledgeItem(
            id=best_knowledge['id'],
            user_id=best_knowledge['user_id'],
            question=best_knowledge['question'],
            description=best_knowledge['description'],
            answer=best_knowledge['answer'],
            public=best_knowledge['public'],
            model_name=best_knowledge['model_name'] or "",
            tool_id=best_knowledge['tool_id'] or 0,
            params=best_knowledge['params'] or "",
        )

        return knowledge_item, tool_info

    except Exception as e:
        logger.error(f"Error in find_knowledge_tool: {str(e)}")
        return None, None


def create_tool_and_knowledge_records(tool_data: dict, knowledge_data: dict) -> dict:
    """
    创建工具和知识记录的核心功能

    Args:
        tool_data: 工具相关数据
        knowledge_data: 知识相关数据

    Returns:
        dict: 包含操作结果的字典
    """
    connection = None
    tool_id = None
    knowledge_id = None

    try:
        # 获取数据库连接
        connection = get_db_connection()

        # 开始事务
        connection.begin()

        # 1. 创建 Tool
        with connection.cursor() as cursor:
            # 插入 Tool 数据
            tool_sql = """
                       INSERT INTO tools
                       (user_id, title, description, url, push, status, timeout, params)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                       """
            cursor.execute(tool_sql, (
                tool_data['user_id'],
                tool_data['title'],
                tool_data['description'],
                tool_data['url'],
                tool_data['push'],
                1,  # status
                tool_data['timeout'],
                tool_data['params']
            ))

            # 获取插入的 tool ID
            tool_id = cursor.lastrowid
            logger.info(f"Tool record created successfully with ID: {tool_id}")

        # 2. 创建 Knowledge，使用刚刚创建的 tool_id
        with connection.cursor() as cursor:
            # 插入 Knowledge 数据，使用 tool_id
            knowledge_sql = """
                            INSERT INTO knowledge
                            (user_id, question, description, answer, public, status, embedding_id, model_name, tool_id,
                             params)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """
            cursor.execute(knowledge_sql, (
                knowledge_data['user_id'],
                knowledge_data['question'],
                knowledge_data['description'],
                knowledge_data['answer'],
                knowledge_data['public'],
                1,  # status
                knowledge_data['embedding_id'],
                knowledge_data['model_name'],
                tool_id,  # 使用刚刚创建的 tool_id
                knowledge_data['params']
            ))

            # 获取插入的 knowledge ID
            knowledge_id = cursor.lastrowid

            # 计算并存储 embedding
            query_embedding = get_embedding(knowledge_data['question'] + knowledge_data['answer'])

            # 将 embedding 写入 Redis
            try:
                redis_conn = get_redis_connection()
                # 使用记录ID作为键，将embedding存储到Redis中
                redis_key = f"knowledge_embedding_{knowledge_id}"
                redis_conn.set(redis_key, str(query_embedding))
                logger.info(f"Embedding stored in Redis with key: {redis_key}")
            except Exception as redis_error:
                logger.error(f"Failed to store embedding in Redis: {str(redis_error)}")
                # 注意：即使Redis存储失败，我们也不会中断主流程

        # 提交事务
        connection.commit()

        logger.info(
            f"Tool and knowledge records created successfully. Tool ID: {tool_id}, Knowledge ID: {knowledge_id}")
        return {
            "success": True,
            "message": "Tool and knowledge records created successfully",
            "tool_id": tool_id,
            "knowledge_id": knowledge_id
        }

    except Exception as e:
        logger.error(f"Error creating tool and knowledge records: {str(e)}")
        if connection:
            connection.rollback()
        return {
            "success": False,
            "message": f"Internal server error: {str(e)}",
            "tool_id": tool_id,
            "knowledge_id": knowledge_id
        }
    finally:
        if connection:
            connection.close()


def get_tool_by_id(tool_id: int) -> Optional[ToolItem]:
    """
    根据tool_id查询数据库tools表

    Args:
        tool_id (int): 工具ID

    Returns:
        Optional[ToolItem]: ToolItem对象，如果未找到则返回None
    """
    try:
        connection = get_db_connection()

        try:
            with connection.cursor() as cursor:
                # 查询工具信息
                tool_query_sql = """
                    SELECT id, user_id, title, description, url, push, public, status, timeout, params, create_time, update_time
                    FROM tools
                    WHERE id = %s AND status = 1
                """
                cursor.execute(tool_query_sql, (tool_id,))
                tool_result = cursor.fetchone()

                if tool_result:
                    # 构建ToolItem对象
                    tool_item = ToolItem(
                        id=tool_result['id'],
                        user_id=str(tool_result['user_id']),
                        title=tool_result['title'],
                        description=tool_result['description'],
                        url=tool_result['url'],
                        push=tool_result['push'],
                        status=tool_result['status'],
                        create_time=tool_result['create_time'].isoformat() if tool_result['create_time'] else None,
                        update_time=tool_result['update_time'].isoformat() if tool_result['update_time'] else None,
                        timeout=tool_result['timeout'],
                        params=tool_result['params']
                    )
                    logger.info(f"Retrieved tool info for tool ID: {tool_id}")
                    return tool_item
                else:
                    logger.warning(f"No tool found for tool ID: {tool_id}")
                    return None
        finally:
            connection.close()

    except Exception as e:
        logger.error(f"Error retrieving tool by ID: {str(e)}")
        return None
