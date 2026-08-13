#!/usr/bin/env python3
"""Patent detail endpoints for the split-view results page.

GET /patent/{source}/{patent_id}/spec    — 说明书分段全文
GET /patent/{source}/{patent_id}/claims  — 权利要求列表

Both endpoints read from Google Patents (patents.google.com) regardless of
the declared *source* — Google Patents indexes US/CN/JP/EP publications with
structured claim/description text, while USPTO PDFs are scanned images that
would require vision-LLM extraction (too heavy for an on-demand endpoint).
``source`` is validated against the known set and carried through for
future data-source routing.

Auth: Firebase bearer token (same pattern as other web routes).
"""

import re

from fastapi import APIRouter, HTTPException, Request

from sources.logger import Logger
from sources.user.passport import verify_firebase_token

VALID_SOURCES = {"uspto", "google_patents"}

_SECTION_HEADING_PATTERN = re.compile(
    r"^[\[【]?[0-9]{4}[\]】]?\s*$"
)
_NATURAL_HEADING_PATTERN = re.compile(
    r"^(技术领域|背景技术|发明内容|附图说明|具体实施方式|"
    r"Technical Field|Background|Summary|Brief Description|"
    r"Detailed Description|Embodiments)\s*$"
)
_SECTION_CHUNK_SIZE = 15


class PatentDetailError(Exception):
    """Base error for patent detail fetch failures."""


def split_description_sections(paragraphs: list[str]) -> list[dict]:
    """Split description paragraphs into sections.

    Paragraphs that look like natural headings (技术领域/背景技术/…) start a
    new section; otherwise paragraphs are chunked into numbered sections of
    ``_SECTION_CHUNK_SIZE``.
    """
    sections: list[dict] = []
    current: dict | None = None
    for para in paragraphs:
        para = (para or "").strip()
        if not para:
            continue
        if _NATURAL_HEADING_PATTERN.match(para):
            if current:
                sections.append(current)
            current = {"heading": para, "paragraphs": []}
            continue
        if current is None:
            current = {"heading": "", "paragraphs": []}
        current["paragraphs"].append(para)
    if current:
        sections.append(current)

    if not sections:
        return []
    # Fallback chunking when no natural headings were found
    if len(sections) == 1 and not sections[0]["heading"]:
        chunks = []
        paras = sections[0]["paragraphs"]
        for i in range(0, len(paras), _SECTION_CHUNK_SIZE):
            end = min(i + _SECTION_CHUNK_SIZE, len(paras))
            chunks.append({
                "heading": f"段落 {i + 1}-{end}",
                "paragraphs": paras[i:end],
            })
        return chunks
    return sections


def build_claims_payload(claims: list[str]) -> dict:
    """Build the claims response payload; first claim is marked independent."""
    if not claims:
        return {"success": False, "claims": []}
    payload_claims = []
    for index, text in enumerate(claims, start=1):
        payload_claims.append({
            "number": index,
            "text": text,
            "independent": index == 1,
        })
    return {"success": True, "claims": payload_claims}


async def _fetch_spec_text(source: str, patent_id: str) -> dict:
    """Fetch description text for *patent_id* and split into sections."""
    from sources.google_patents_client import GooglePatentsClient

    client = GooglePatentsClient(delay=0.5)
    try:
        paragraphs = await client.query_description(patent_id, lang="zh")
    except Exception as exc:
        raise PatentDetailError(str(exc)) from exc
    finally:
        await client.close()

    return {
        "sections": split_description_sections(paragraphs),
        "source_url": f"https://patents.google.com/patent/{patent_id}",
    }


async def _fetch_claims(source: str, patent_id: str) -> dict:
    """Fetch claims for *patent_id* from Google Patents."""
    from sources.google_patents_client import GooglePatentsClient

    client = GooglePatentsClient(delay=0.5)
    try:
        claims = await client.query_claims(patent_id, lang="zh")
    except Exception as exc:
        raise PatentDetailError(str(exc)) from exc
    finally:
        await client.close()

    return build_claims_payload(claims)


def register_patent_detail_routes(logger, config):
    """Register patent detail routes with dependency injection."""
    router = APIRouter()
    logger = Logger("patent_detail.log")

    @router.get("/patent/{source}/{patent_id}/spec")
    async def patent_spec(source: str, patent_id: str, http_request: Request = None):
        if http_request is not None:
            auth_header = http_request.headers.get("Authorization")
            verify_firebase_token(auth_header)
        if source not in VALID_SOURCES:
            raise HTTPException(status_code=400, detail="Unsupported source")
        if not patent_id or len(patent_id) > 40:
            raise HTTPException(status_code=400, detail="Invalid patent_id")
        try:
            payload = await _fetch_spec_text(source, patent_id)
        except PatentDetailError as exc:
            logger.error(f"spec fetch failed — source={source}, id={patent_id}: {exc}")
            raise HTTPException(
                status_code=502,
                detail="Patent specification unavailable",
            )
        return {"success": True, **payload}

    @router.get("/patent/{source}/{patent_id}/claims")
    async def patent_claims(source: str, patent_id: str, http_request: Request = None):
        if http_request is not None:
            auth_header = http_request.headers.get("Authorization")
            verify_firebase_token(auth_header)
        if source not in VALID_SOURCES:
            raise HTTPException(status_code=400, detail="Unsupported source")
        if not patent_id or len(patent_id) > 40:
            raise HTTPException(status_code=400, detail="Invalid patent_id")
        try:
            payload = await _fetch_claims(source, patent_id)
        except PatentDetailError as exc:
            logger.error(f"claims fetch failed — source={source}, id={patent_id}: {exc}")
            raise HTTPException(
                status_code=502,
                detail="Patent claims unavailable",
            )
        if not payload.get("success"):
            # Honest degrade: claims not parseable from the public record
            raise HTTPException(
                status_code=501,
                detail="权利要求暂不可用，请通过 PDF 原文查看",
            )
        return payload

    return router
