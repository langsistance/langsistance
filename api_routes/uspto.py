#!/usr/bin/env python3

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, RedirectResponse

from sources.uspto_download import (
    get_uspto_download_headers,
    resolve_uspto_download_url,
)
from sources.logger import Logger


logger = Logger("backend.log")
router = APIRouter()


@router.get("/uspto/download")
async def download_uspto_file(url: str = Query(..., min_length=1)):
    logger.info(f"USPTO lazy download requested: {url}")
    try:
        import requests

        resolved_url = resolve_uspto_download_url(
            url,
            fetch_text=lambda download_url, headers: requests.get(
                download_url,
                headers=headers,
                timeout=30,
            ).text,
            request_headers=get_uspto_download_headers(),
        )
    except ValueError as exc:
        logger.warning(f"USPTO lazy download rejected: {exc}")
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception as exc:
        logger.error(f"USPTO lazy download request failed: {exc}")
        return JSONResponse(status_code=502, content={"error": "USPTO download request failed"})

    logger.info(f"USPTO lazy download resolved: {resolved_url}")
    return RedirectResponse(url=resolved_url, status_code=302)
