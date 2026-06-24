#!/usr/bin/env python3
"""Scene (场景包) management API routes."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .models import (
    SceneItem,
    SceneKnowledgeItem,
    UserSceneStatusItem,
    SceneListResponse,
    SceneKnowledgeResponse,
    UserSceneStatusResponse,
    UserScenesUpdateRequest,
)
from sources.knowledge.knowledge import get_db_connection
from sources.logger import Logger
from sources.user.passport import verify_firebase_token

logger = Logger("backend.log")
router = APIRouter()


@router.get("/scenes/available", response_model=SceneListResponse)
async def list_available_scenes():
    """列出所有场景，包含每个场景下的知识数量。"""
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT s.id, s.name, s.description,
                       COUNT(k.id) AS knowledge_count
                FROM scenes s
                LEFT JOIN knowledge k ON k.scene_id = s.id AND k.status = 1
                GROUP BY s.id
                ORDER BY s.id
            """)
            rows = cursor.fetchall()

            scenes = []
            for row in rows:
                scenes.append(SceneItem(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"] or "",
                    knowledge_count=row["knowledge_count"],
                ))

            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "ok",
                    "scenes": [s.dict() for s in scenes],
                },
            )
    except Exception as e:
        logger.error(f"Error listing scenes: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e), "scenes": []},
        )
    finally:
        if connection:
            connection.close()


@router.get("/scenes/{scene_id}/knowledge", response_model=SceneKnowledgeResponse)
async def get_scene_knowledge(scene_id: int):
    """获取场景下所有知识的 question + description（不含 answer）。"""
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT id, question, description, type
                FROM knowledge
                WHERE scene_id = %s AND status = 1
                ORDER BY type ASC, update_time DESC
            """, (scene_id,))
            rows = cursor.fetchall()

            items = [
                SceneKnowledgeItem(
                    id=row["id"],
                    question=row["question"],
                    description=row["description"] or "",
                    type=row.get("type", 1),
                )
                for row in rows
            ]

            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "ok",
                    "knowledge": [i.dict() for i in items],
                },
            )
    except Exception as e:
        logger.error(f"Error getting scene knowledge: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e), "knowledge": []},
        )
    finally:
        if connection:
            connection.close()


@router.get("/user/scenes", response_model=UserSceneStatusResponse)
@router.get("/user/scenes/status", response_model=UserSceneStatusResponse)
async def get_user_scenes(http_request: Request):
    """获取当前用户订阅的场景及订阅状态。"""
    auth_header = http_request.headers.get("Authorization")
    user = verify_firebase_token(auth_header)
    user_id = user["uid"]

    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT scene_id FROM user_scenes WHERE user_id = %s",
                (user_id,),
            )
            subscribed_ids = {row["scene_id"] for row in cursor.fetchall()}

            # 查用户是否已完成 onboarding
            cursor.execute(
                "SELECT onboarded FROM users WHERE user_id = %s",
                (user_id,),
            )
            user_row = cursor.fetchone()
            onboarded = bool(user_row and user_row.get("onboarded"))

            # 查所有场景
            cursor.execute("""
                SELECT s.id, s.name, s.description,
                       COUNT(k.id) AS knowledge_count
                FROM scenes s
                LEFT JOIN knowledge k ON k.scene_id = s.id AND k.status = 1
                GROUP BY s.id
                ORDER BY s.id
            """)
            rows = cursor.fetchall()

            scenes = []
            for row in rows:
                scenes.append(UserSceneStatusItem(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"] or "",
                    subscribed=row["id"] in subscribed_ids,
                    knowledge_count=row["knowledge_count"],
                ))

            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "ok",
                    "scenes": [s.dict() for s in scenes],
                    "onboarded": onboarded,
                },
            )
    except Exception as e:
        logger.error(f"Error getting user scenes: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e), "scenes": []},
        )
    finally:
        if connection:
            connection.close()


@router.post("/user/scenes")
async def update_user_scenes(http_request: Request, body: UserScenesUpdateRequest):
    """批量更新用户场景订阅（全量替换）。"""
    auth_header = http_request.headers.get("Authorization")
    user = verify_firebase_token(auth_header)
    user_id = user["uid"]

    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            # 先删后插（全量替换）
            cursor.execute(
                "DELETE FROM user_scenes WHERE user_id = %s",
                (user_id,),
            )
            for scene_id in body.scene_ids:
                cursor.execute(
                    "INSERT INTO user_scenes (user_id, scene_id) VALUES (%s, %s)",
                    (user_id, scene_id),
                )
            connection.commit()

            logger.info(f"User {user_id} updated scenes: {body.scene_ids}")
            return JSONResponse(
                status_code=200,
                content={"success": True, "message": "ok"},
            )
    except Exception as e:
        logger.error(f"Error updating user scenes: {str(e)}")
        if connection:
            connection.rollback()
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)},
        )
    finally:
        if connection:
            connection.close()


@router.post("/user/onboarded")
async def mark_onboarded(http_request: Request):
    """标记用户已完成首次场景选择流程。"""
    auth_header = http_request.headers.get("Authorization")
    user = verify_firebase_token(auth_header)
    user_id = user["uid"]

    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE users SET onboarded = 1 WHERE user_id = %s",
                (user_id,),
            )
            connection.commit()
            logger.info(f"User {user_id} marked as onboarded")
            return JSONResponse(
                status_code=200,
                content={"success": True, "message": "ok"},
            )
    except Exception as e:
        logger.error(f"Error marking onboarded: {str(e)}")
        if connection:
            connection.rollback()
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)},
        )
    finally:
        if connection:
            connection.close()
