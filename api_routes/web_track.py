#!/usr/bin/env python3
"""
Lightweight anonymous page-view tracking endpoint (self-hosted beacon).

Endpoints:
  POST /web/page_view  — beacon via navigator.sendBeacon (text/plain body)
  GET  /web/page_view  — beacon via <img> pixel fallback (query params)

Payload fields (both methods):
  vid  — per-browser anonymous visitor id (localStorage, crypto.randomUUID)
  path — current page path (relative, includes query)
  ref  — document.referrer
  src  — utm_source (or other campaign tag), optional

Events land in .logs/analytics.log via sources.analytics.track_event with
event name ``page_view`` (fields vid/path/ref/src at top level, no user_id).
No authentication: this endpoint is public and intentionally minimal —
in-memory per-vid daily cap guards against beacon spam.
"""

from urllib.parse import parse_qsl

from fastapi import APIRouter, Request
from fastapi.responses import Response

from sources.analytics import track_event

# ── Light anti-abuse: per-vid cap, reset on process restart ──
_DAILY_CAP_PER_VID = 500
_seen: dict = {}  # vid -> (date_str, count)


def _check_vid(vid: str) -> bool:
    if not vid or len(vid) < 8 or len(vid) > 128:
        return False
    return all(c.isalnum() or c in "-._" for c in vid)


def _under_cap(vid: str) -> bool:
    """Return True if vid may emit another event today (in-memory)."""
    from datetime import datetime, timezone

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cur = _seen.get(vid)
    if cur is None or cur[0] != day:
        _seen[vid] = [day, 1]
        return True
    if cur[1] >= _DAILY_CAP_PER_VID:
        return False
    cur[1] += 1
    return True


def _clean(value: str | None, default: str, max_len: int) -> str:
    value = (value or "").strip()
    if len(value) > max_len:
        value = value[:max_len]
    return value or default


def register_web_track_routes(app_logger):
    """Register the public page-view beacon endpoint."""

    router = APIRouter()

    @router.api_route("/web/page_view", methods=["POST", "GET"])
    async def page_view(request: Request):
        params: dict = {}
        if request.method == "POST":
            raw = (await request.body()).decode("utf-8", errors="ignore")
            params = dict(parse_qsl(raw, keep_blank_values=True))
        else:
            params = {k: v for k, v in request.query_params.items()}

        vid = _clean(params.get("vid"), "", 128)
        if not _check_vid(vid):
            return Response(status_code=204)  # malformed beacon: drop silently
        if not _under_cap(vid):
            return Response(status_code=204)  # over daily cap: drop silently

        path = _clean(params.get("path"), "/", 300)
        ref = _clean(params.get("ref"), "", 500)
        src = _clean(params.get("src"), "", 64)

        track_event(
            "page_view",
            extra={
                "vid": vid,
                "path": path,
                "ref": ref,
                "src": src,
            },
        )
        return Response(status_code=204)

    return router
