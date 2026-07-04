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
