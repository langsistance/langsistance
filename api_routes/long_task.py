#!/usr/bin/env python3
"""Long task status polling and report download API routes.

Endpoints:
  GET /long_task/{task_id}/status
  GET /long_task/{task_id}/report?format=pdf|docx
"""

from fastapi import APIRouter, Query, HTTPException, Request
from fastapi.responses import Response
from sources.long_task.status_manager import get_task_status
from sources.long_task.storage import create_storage
from sources.user.passport import verify_firebase_token


def register_long_task_routes(logger, config):
    """Register long task polling and download routes with dependency injection."""
    router = APIRouter()

    def _get_storage_config() -> dict:
        return {
            'report_storage_backend': config.get('STORAGE', 'backend',
                                                 fallback='local'),
            'report_storage_local_dir': config.get('STORAGE', 'local_base_dir',
                                                   fallback='/opt/workspace/reports'),
        }

    @router.get("/long_task/{task_id}/status")
    async def task_status(task_id: str, http_request: Request):
        """Poll the current status of a long-running task."""
        auth_header = http_request.headers.get("Authorization")
        verify_firebase_token(auth_header)  # Auth gate — any valid user
        logger.info(f"Status poll for task: {task_id}")
        status = get_task_status(task_id)
        return {"success": True, **status}

    @router.get("/long_task/{task_id}/report")
    async def download_report(task_id: str, format: str = Query(..., pattern="^(pdf|docx)$"), http_request: Request = None):
        """Download a completed report file for a task."""
        logger.info(f"Report download for task: {task_id}, format: {format}")
        storage = create_storage(_get_storage_config())
        filename = f"report.{format}"
        try:
            content = await storage.get(task_id, filename)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Report not found")

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
