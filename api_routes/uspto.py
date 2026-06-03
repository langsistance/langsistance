#!/usr/bin/env python3

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, Response

from sources.uspto_download import (
    fetch_uspto_download_file,
    get_uspto_download_headers,
)
from sources.http_outbound import outbound_http
from sources.logger import Logger


logger = Logger("backend.log")
router = APIRouter()


@router.get("/uspto/download")
async def download_uspto_file(url: str = Query(..., min_length=1)):
    logger.info(f"USPTO lazy download requested: {url}")
    try:
        download_file = fetch_uspto_download_file(
            url,
            fetch_response=lambda download_url, headers: outbound_http.get(
                download_url,
                purpose="download",
                headers=headers,
                timeout=30,
            ),
            request_headers=get_uspto_download_headers(),
        )
    except ValueError as exc:
        logger.warning(f"USPTO lazy download rejected: {exc}")
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception as exc:
        logger.error(f"USPTO lazy download request failed: {exc}")
        return JSONResponse(status_code=502, content={"error": "USPTO download request failed"})

    logger.info(f"USPTO lazy download proxied: {download_file.filename}")
    return Response(
        content=download_file.content,
        media_type=download_file.media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{download_file.filename}"'
        },
    )
