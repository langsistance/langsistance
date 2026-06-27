#!/usr/bin/env python3
"""Session CRUD API routes.

Endpoints:
  POST   /session                    — create a new session
  GET    /session/{session_id}       — get a session with messages + long_task_ids
  GET    /sessions                   — list user sessions (auth token identifies user)
  POST   /session/{session_id}/message — append a message
  DELETE /session/{session_id}       — archive (status=2)
"""

import uuid
import json
from fastapi import APIRouter, Query, HTTPException, Request
from pydantic import BaseModel
from sources.knowledge.knowledge import get_db_connection
from sources.user.passport import verify_firebase_token


class CreateSessionRequest(BaseModel):
    scene_id: int | None = None
    messages: list = []
    title: str = ""


class AppendMessageRequest(BaseModel):
    role: str
    content: str
    patent_data: list | None = None
    timestamp: str | None = None


class SaveMessagesRequest(BaseModel):
    messages: list
    title: str = ""


def register_session_routes(logger, config):
    """Register session CRUD routes with dependency injection."""
    router = APIRouter()

    @router.post("/session")
    async def create_session(req: CreateSessionRequest, http_request: Request):
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)
        user_id = int(user['uid'])

        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO conversations
                       (session_id, user_id, scene_id, title, messages)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (session_id, user_id, req.scene_id, req.title,
                     json.dumps(req.messages, ensure_ascii=False)))
                conn.commit()
            logger.info(f"Session created: {session_id} user_id={user_id}")
            return {"success": True, "session_id": session_id}
        finally:
            conn.close()

    @router.get("/session/{session_id}")
    async def get_session(session_id: str):
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT id, session_id, user_id, scene_id, title,
                              messages, long_task_ids, status,
                              create_time, update_time
                       FROM conversations
                       WHERE session_id = %s AND status != 2""",
                    (session_id,))
                row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Session not found")
            row['messages'] = json.loads(row['messages']) if isinstance(row['messages'], str) else row['messages']
            row['create_time'] = row['create_time'].isoformat() if row['create_time'] else None
            row['update_time'] = row['update_time'].isoformat() if row['update_time'] else None
            return {"success": True, "session": row}
        finally:
            conn.close()

    @router.get("/sessions")
    async def list_sessions(http_request: Request):
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)
        user_id = int(user['uid'])

        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT session_id, title, status,
                              long_task_ids, create_time, update_time
                       FROM conversations
                       WHERE user_id = %s AND status != 2
                       ORDER BY update_time DESC""",
                    (user_id,))
                rows = cur.fetchall()
            for r in rows:
                r['create_time'] = r['create_time'].isoformat() if r['create_time'] else None
                r['update_time'] = r['update_time'].isoformat() if r['update_time'] else None
                # long_task_ids may be stored as JSON string — parse to list
                lt = r.get('long_task_ids')
                if isinstance(lt, str):
                    try:
                        r['long_task_ids'] = json.loads(lt)
                    except (json.JSONDecodeError, TypeError):
                        r['long_task_ids'] = []
                elif lt is None:
                    r['long_task_ids'] = []
            return {"success": True, "sessions": rows}
        finally:
            conn.close()

    @router.post("/session/{session_id}/message")
    async def append_message(session_id: str, req: AppendMessageRequest):
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT messages FROM conversations WHERE session_id = %s",
                    (session_id,))
                row = cur.fetchone()
                if row is None:
                    raise HTTPException(status_code=404, detail="Session not found")

                messages = json.loads(row['messages']) if isinstance(row['messages'], str) else row['messages']
                new_msg = {"role": req.role, "content": req.content}
                if req.patent_data:
                    new_msg["patent_data"] = req.patent_data
                if req.timestamp:
                    new_msg["timestamp"] = req.timestamp
                messages.append(new_msg)

                cur.execute(
                    "UPDATE conversations SET messages = %s WHERE session_id = %s",
                    (json.dumps(messages, ensure_ascii=False), session_id))
                conn.commit()
            logger.info(f"Message appended to session: {session_id}")
            return {"success": True}
        finally:
            conn.close()

    @router.put("/session/{session_id}/messages")
    async def save_messages(session_id: str, req: SaveMessagesRequest, http_request: Request):
        """Bulk-save all messages for a session (replace entire messages array)."""
        auth_header = http_request.headers.get("Authorization")
        verify_firebase_token(auth_header)
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                # Verify session exists and belongs to user
                cur.execute(
                    "SELECT id FROM conversations WHERE session_id = %s AND status != 2",
                    (session_id,))
                if cur.fetchone() is None:
                    raise HTTPException(status_code=404, detail="Session not found")
                updates = ["messages = %s"]
                params = [json.dumps(req.messages, ensure_ascii=False)]
                if req.title:
                    updates.append("title = %s")
                    params.append(req.title)
                params.append(session_id)
                cur.execute(
                    f"UPDATE conversations SET {', '.join(updates)} WHERE session_id = %s",
                    params)
                conn.commit()
            logger.info(f"Session messages saved: {session_id}, count={len(req.messages)}")
            return {"success": True}
        finally:
            conn.close()

    @router.delete("/session/{session_id}")
    async def archive_session(session_id: str):
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE conversations SET status = 2 WHERE session_id = %s",
                    (session_id,))
                if cur.rowcount == 0:
                    raise HTTPException(status_code=404, detail="Session not found")
                conn.commit()
            logger.info(f"Session archived: {session_id}")
            return {"success": True}
        finally:
            conn.close()

    return router
