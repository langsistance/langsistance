#!/usr/bin/env python3
"""Long task status polling and report download API routes.

Endpoints:
  GET /long_task/{task_id}/status
  GET /long_task/{task_id}/report?format=pdf|docx
"""

from fastapi import APIRouter, Query, HTTPException, Request
from fastapi.responses import Response
from sources.long_task.status_manager import get_task_status
from sources.long_task.storage import create_storage, get_storage_config, LocalReportStorage
from sources.user.passport import verify_firebase_token


def _dispatch_from_mysql(user_id: str, task_id: str, logger) -> None:
    """Read task params from MySQL and dispatch via Celery.  Idempotent — safe
    to call even if the task was already dispatched."""
    import json as _json
    from sources.knowledge.knowledge import get_db_connection as _gdc
    conn = _gdc()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT input_params, session_id, scene_id
                   FROM long_tasks WHERE task_id = %s""",
                (task_id,),
            )
            row = cur.fetchone()
        if row:
            input_params = row.get('input_params')
            stored = _json.loads(input_params) if isinstance(input_params, str) else input_params
            next_params = {
                'query': stored.get('query', ''),
                'patent_ids': stored.get('patent_ids', []),
                'patent_source': stored.get('patent_source', 'auto'),
                'session_id': row.get('session_id') or '',
                'scene_id': row.get('scene_id'),
                'conversation_history': stored.get('conversation_history', []),
                'patent_file_refs': stored.get('patent_file_refs', []),
                'user_id': str(user_id),
            }
            if stored.get('patent_texts'):
                next_params['patent_texts'] = stored['patent_texts']
            # Lazy import to avoid circular dependency at module level
            from celery_worker import execute_patent_analysis
            execute_patent_analysis.delay(task_id=task_id, params=next_params)
            logger.info(f"DISPATCHED task {task_id} via Celery")
    except Exception as e:
        logger.error(f"Failed to dispatch task {task_id}: {e}")
    finally:
        conn.close()


def register_long_task_routes(logger, config):
    """Register long task polling and download routes with dependency injection."""
    router = APIRouter()

    @router.get("/long_task/{task_id}/status")
    async def task_status(task_id: str, http_request: Request):
        """Poll the current status of a long-running task."""
        auth_header = http_request.headers.get("Authorization")
        verify_firebase_token(auth_header)  # Auth gate — any valid user
        logger.info(f"Status poll for task: {task_id}")
        status = get_task_status(task_id)
        return {"success": True, **status}

    @router.post("/long_task/batch_status")
    async def batch_task_status(http_request: Request):
        """Poll status for multiple long-running tasks in one request."""
        auth_header = http_request.headers.get("Authorization")
        verify_firebase_token(auth_header)
        body = await http_request.json()
        task_ids = body.get("task_ids", []) or []
        if not isinstance(task_ids, list):
            task_ids = []
        # Cap at 20 to prevent abuse
        task_ids = task_ids[:20]
        statuses = {}
        for tid in task_ids:
            statuses[tid] = get_task_status(tid)
        return {"success": True, "statuses": statuses}

    @router.post("/long_task/{task_id}/pause")
    async def pause_task(task_id: str, http_request: Request):
        """Pause a running long task at its next checkpoint."""
        auth_header = http_request.headers.get("Authorization")
        verify_firebase_token(auth_header)
        from sources.long_task.status_manager import (
            get_task_status, request_task_pause, is_task_paused,
        )
        status = get_task_status(task_id)
        if status.get('status') not in ('running', 'analyzing', 'searching', 'generating'):
            return {"success": False, "error": "Task is not running"}
        if is_task_paused(task_id):
            return {"success": False, "error": "Task is already paused"}
        request_task_pause(task_id)
        logger.info(f"Pause requested for task: {task_id}")
        return {"success": True, "message": "Pause requested"}

    @router.post("/long_task/{task_id}/resume")
    async def resume_task(task_id: str, http_request: Request):
        """Resume a paused task by re-queuing it. Dispatches immediately if
        no other task is running."""
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)
        from sources.long_task.status_manager import (
            get_task_status, clear_task_pause,
        )
        from sources.long_task.user_queue import requeue_paused_task
        status = get_task_status(task_id)
        if status.get('status') != 'paused':
            return {"success": False, "error": "Task is not paused"}
        clear_task_pause(task_id)
        user_id = str(user.get('uid', ''))
        result = requeue_paused_task(user_id, task_id)
        logger.info(f"Resume requested for task: {task_id}, result={result}")
        if result == 'running':
            # No other task running — dispatch this one immediately
            _dispatch_from_mysql(user_id, task_id, logger)
        return {"success": True, "message": "Task re-queued"}

    @router.post("/long_task/{task_id}/stop")
    async def stop_task(task_id: str, http_request: Request):
        """Permanently stop and discard a long task."""
        auth_header = http_request.headers.get("Authorization")
        verify_firebase_token(auth_header)
        from sources.long_task.status_manager import (
            get_task_status, request_task_stop, is_task_stopped,
        )
        status = get_task_status(task_id)
        if status.get('status') in ('completed', 'failed', 'cancelled'):
            return {"success": False, "error": "Task already in terminal state"}
        if is_task_stopped(task_id):
            return {"success": False, "error": "Stop already requested"}
        request_task_stop(task_id)
        logger.info(f"Stop requested for task: {task_id}")
        return {"success": True, "message": "Stop requested"}

    @router.get("/long_task/{task_id}/report")
    async def download_report(task_id: str, format: str = Query(..., pattern="^(pdf|docx)$"), http_request: Request = None):
        """Download a completed report file for a task."""
        logger.info(f"Report download for task: {task_id}, format: {format}")
        storage = create_storage(get_storage_config())
        filename = f"report.{format}"
        try:
            content = await storage.get(task_id, filename)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Report not found")
        except Exception as e:
            # Primary storage failed — try local fallback
            logger.warning(
                f"Primary storage get failed for {task_id}/{filename}: {e}, "
                f"trying local fallback"
            )
            try:
                local = LocalReportStorage()
                content = await local.get(task_id, filename)
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="Report not found")
            except Exception as e2:
                logger.error(f"Local fallback also failed: {e2}")
                raise HTTPException(status_code=500, detail="Failed to retrieve report")

        media_type = "application/pdf" if format == "pdf" else \
                     "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        return Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="patent_analysis_{task_id}.{format}"'
            },
        )

    return router
