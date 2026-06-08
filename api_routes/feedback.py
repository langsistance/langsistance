#!/usr/bin/env python3
"""
Feedback and in-app messaging API routes.

Endpoints:
  POST /submit_feedback   — user submits feedback; email sent to support
  GET  /messages           — list user's messages (newest first)
  GET  /messages/unread_count — unread count for badge
  POST /messages/{id}/read — mark a single message as read
  POST /messages/read_all  — mark all messages as read
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api_routes.models import (
    FeedbackSubmitRequest,
    FeedbackSubmitResponse,
    MessageItem,
    MessagesResponse,
    UnreadCountResponse,
    MarkReadResponse,
)
from sources.user.passport import verify_firebase_token
from sources.knowledge.knowledge import get_db_connection
from sources.feedback_email import send_feedback_notification
from sources.logger import Logger

logger = Logger("feedback.log")


def register_feedback_routes(app_logger, config_ref):
    """Register feedback/messaging routes with dependency injection."""

    router = APIRouter()

    # ── POST /submit_feedback ────────────────────────────────────────────

    @router.post("/submit_feedback")
    async def submit_feedback(request: FeedbackSubmitRequest, http_request: Request):
        """Submit user feedback. Sends notification email to support team."""
        auth_header = http_request.headers.get("Authorization")
        try:
            user = verify_firebase_token(auth_header)
        except Exception as e:
            return JSONResponse(status_code=401, content={"error": "Invalid token"})

        user_id = user["uid"]
        email = user.get("email", "")

        if not request.content or not request.content.strip():
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Feedback content cannot be empty"},
            )

        content = request.content.strip()
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                insert_sql = (
                    "INSERT INTO feedback (user_id, email, content) VALUES (%s, %s, %s)"
                )
                cursor.execute(insert_sql, (user_id, email, content))
                feedback_id = cursor.lastrowid
                conn.commit()

            logger.info(f"Feedback submitted: id={feedback_id} user_id={user_id}")

            # Send email notification to support (fire-and-forget)
            try:
                send_feedback_notification(
                    user_email=email,
                    user_id=str(user_id),
                    feedback_content=content,
                    feedback_id=feedback_id,
                )
            except Exception as email_err:
                logger.warning(f"Feedback saved but email failed: {email_err}")

            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Thank you! Your feedback has been received.",
                    "feedback_id": feedback_id,
                },
            )

        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": "Failed to save feedback"},
            )
        finally:
            if conn:
                conn.close()

    # ── GET /messages ─────────────────────────────────────────────────────

    @router.get("/messages")
    async def get_messages(http_request: Request):
        """Get all messages for the authenticated user, newest first."""
        auth_header = http_request.headers.get("Authorization")
        try:
            user = verify_firebase_token(auth_header)
        except Exception:
            return JSONResponse(status_code=401, content={"error": "Invalid token"})

        user_id = user["uid"]
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                # Get unread count
                count_sql = (
                    "SELECT COUNT(*) AS cnt FROM messages WHERE user_id = %s AND is_read = 0"
                )
                cursor.execute(count_sql, (user_id,))
                unread_count = cursor.fetchone()["cnt"]

                # Get all messages
                list_sql = (
                    "SELECT id, user_id, feedback_id, title, content, is_read, create_time "
                    "FROM messages WHERE user_id = %s ORDER BY create_time DESC"
                )
                cursor.execute(list_sql, (user_id,))
                rows = cursor.fetchall()

                messages = []
                for row in rows:
                    messages.append(
                        {
                            "id": row["id"],
                            "title": row["title"],
                            "content": row["content"],
                            "is_read": bool(row["is_read"]),
                            "create_time": row["create_time"].strftime("%Y-%m-%d %H:%M:%S"),
                            "feedback_id": row.get("feedback_id"),
                        }
                    )

                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "data": messages,
                        "total": len(messages),
                        "unread_count": unread_count,
                    },
                )

        except Exception as e:
            logger.error(f"Error fetching messages: {e}")
            return JSONResponse(
                status_code=500,
                content={"success": False, "data": [], "total": 0, "unread_count": 0},
            )
        finally:
            if conn:
                conn.close()

    # ── GET /messages/unread_count ─────────────────────────────────────────

    @router.get("/messages/unread_count")
    async def get_unread_count(http_request: Request):
        """Get unread message count for the badge indicator."""
        auth_header = http_request.headers.get("Authorization")
        try:
            user = verify_firebase_token(auth_header)
        except Exception:
            return JSONResponse(status_code=401, content={"error": "Invalid token"})

        user_id = user["uid"]
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) AS cnt FROM messages WHERE user_id = %s AND is_read = 0",
                    (user_id,),
                )
                unread_count = cursor.fetchone()["cnt"]

            return JSONResponse(
                status_code=200,
                content={"success": True, "unread_count": unread_count},
            )
        except Exception as e:
            logger.error(f"Error fetching unread count: {e}")
            return JSONResponse(
                status_code=200,
                content={"success": True, "unread_count": 0},
            )
        finally:
            if conn:
                conn.close()

    # ── POST /messages/{message_id}/read ───────────────────────────────────

    @router.post("/messages/{message_id}/read")
    async def mark_message_read(message_id: int, http_request: Request):
        """Mark a single message as read."""
        auth_header = http_request.headers.get("Authorization")
        try:
            user = verify_firebase_token(auth_header)
        except Exception:
            return JSONResponse(status_code=401, content={"error": "Invalid token"})

        user_id = user["uid"]
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                update_sql = (
                    "UPDATE messages SET is_read = 1 "
                    "WHERE id = %s AND user_id = %s AND is_read = 0"
                )
                cursor.execute(update_sql, (message_id, user_id))
                conn.commit()
                affected = cursor.rowcount

            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Marked as read" if affected else "Already read or not found",
                },
            )
        except Exception as e:
            logger.error(f"Error marking message as read: {e}")
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": "Failed to mark as read"},
            )
        finally:
            if conn:
                conn.close()

    # ── POST /messages/read_all ────────────────────────────────────────────

    @router.post("/messages/read_all")
    async def mark_all_read(http_request: Request):
        """Mark all messages as read for the authenticated user."""
        auth_header = http_request.headers.get("Authorization")
        try:
            user = verify_firebase_token(auth_header)
        except Exception:
            return JSONResponse(status_code=401, content={"error": "Invalid token"})

        user_id = user["uid"]
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE messages SET is_read = 1 WHERE user_id = %s AND is_read = 0",
                    (user_id,),
                )
                conn.commit()
                affected = cursor.rowcount

            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": f"{affected} message(s) marked as read",
                },
            )
        except Exception as e:
            logger.error(f"Error marking all messages as read: {e}")
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": "Failed to mark all as read"},
            )
        finally:
            if conn:
                conn.close()

    return router
