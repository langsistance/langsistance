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
async def download_uspto_file(
    url: str = Query(..., min_length=1),
    inline: bool = Query(False),
):
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

    disposition = "inline" if inline else "attachment"
    media_type = download_file.media_type
    if (
        inline
        and download_file.filename.lower().endswith(".pdf")
        and media_type.lower() in ("application/octet-stream", "")
    ):
        # USPTO serves some PDFs with an octet-stream content type; with
        # nosniff in play, browsers refuse to render those inline and
        # download instead.  Coerce to application/pdf for the embedded
        # viewer.  The attachment path keeps the upstream type unchanged.
        media_type = "application/pdf"
    logger.info(f"USPTO lazy download proxied: {download_file.filename}")
    return Response(
        content=download_file.content,
        media_type=media_type,
        headers={
            "Content-Disposition": f'{disposition}; filename="{download_file.filename}"'
        },
    )
