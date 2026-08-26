#!/usr/bin/env python3
"""Baiten (佰腾) CN patent document proxy endpoints.

GET /baiten/download — stream a CN patent PDF from the Baiten gateway
through this server so the frontend's inline viewer never sees the
gateway.  The gateway returns the PDF as a base64 JSON string that can
reach tens of MB — the response is streamed, base64-decoded chunk by
chunk into a spooled temp file (8 MB memory threshold) and re-streamed
to the browser, keeping the server's <1 GB RAM budget intact
(memory: server-memory-constraint).

Same contract as /uspto/download: no auth header required (the URL is
embedded in an <iframe> by the results page), URL carries the params.
"""

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse

from sources.logger import Logger


logger = Logger("backend.log")
router = APIRouter()

_STREAM_CHUNK = 64 * 1024


@router.get("/baiten/download")
async def download_baiten_file(
    pub_num: str = Query(..., min_length=1),
    pub_date: str = Query(..., min_length=1),
    inline: bool = Query(True),
):
    from sources.baiten_client import BaitenClient, BaitenError
    from sources.long_task.config import get_baiten_config

    logger.info(f"Baiten download requested: {pub_num} ({pub_date})")

    cfg = get_baiten_config()
    if not cfg["app_key"] or not cfg["app_secret"]:
        logger.warning("Baiten download rejected: not configured")
        return JSONResponse(status_code=400,
                            content={"error": "Baiten not configured"})

    try:
        client = BaitenClient(
            cfg["app_key"], cfg["app_secret"], cfg["gateway_url"])
        spool = await client.get_file(pub_num, pub_date)
    except BaitenError as exc:
        logger.warning(f"Baiten download rejected: {exc}")
        return JSONResponse(status_code=502, content={"error": str(exc)})
    except Exception as exc:
        logger.error(f"Baiten download request failed: {exc}")
        return JSONResponse(
            status_code=502, content={"error": "Baiten download failed"})

    async def stream_spool():
        spool.seek(0)
        try:
            while True:
                chunk = spool.read(_STREAM_CHUNK)
                if not chunk:
                    break
                yield chunk
        finally:
            spool.close()

    disposition = "inline" if inline else "attachment"
    filename = f"{pub_num}.pdf"
    return StreamingResponse(
        stream_spool(),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'{disposition}; filename="{filename}"',
        },
    )
