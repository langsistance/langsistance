#!/usr/bin/env python3
"""Patent detail endpoints for the split-view results page.

GET /patent/{source}/{patent_id}/spec    — 说明书 PDF（内嵌阅读器代理地址）
GET /patent/{source}/{patent_id}/claims  — 权利要求列表

Documents come from USPTO's file-wrapper download (the same mechanism the
prosecution-history long task uses): resolve the application number, fetch
the PEDS document list and locate the SPEC / CLM documents.  The spec
endpoint returns the USPTO PDF behind the existing lazy-download proxy
(``/uspto/download``) so the frontend embeds the original PDF in a viewer
— the same way patent documents are displayed.  Claims are downloaded and
parsed into a structured list.  ``source`` is validated against the known
set but does not change the download path — USPTO is the single data
source.

Auth: Firebase bearer token (same pattern as other web routes).
"""

import re
from html import unescape

from fastapi import APIRouter, HTTPException, Request

from sources.user.passport import verify_firebase_token

VALID_SOURCES = {"uspto", "google_patents"}

_CLAIM_START_PATTERN = re.compile(r"(?m)^\s*(\d{1,3})\.\s*")

# Dependent claims open by referencing an earlier claim — standard shapes:
# "The method of claim 1", "The apparatus according to claim 2",
# "A method according to any one of claims 1 to 3", "The method as
# claimed in claim 1", "The method of any preceding claim".
_DEPENDENT_OPENERS = re.compile(
    r"^(?:the|a)\s+[\w\s/’-]{2,60}?\s+"
    r"(?:of|according\s+to|as\s+claimed\s+in|as\s+defined\s+in|as\s+recited\s+in)\s+"
    r"(?:any\s+one\s+of\s+)?(?:the\s+)?(?:preceding|previous)?\s*claims?\b",
    re.IGNORECASE,
)

# Chinese dependent openers: 如权利要求1所述 / 根据权利要求1-3中任一项所述 …
_DEPENDENT_OPENERS_CN = re.compile(
    r"^(?:如|根据)权利要求(?:\d+(?:[-–—至到]\d+)?|前述|上述|任一|任一项)"
)

_XML_TAG_PATTERN = re.compile(r"<[^>]+>")


class PatentDetailError(Exception):
    """Base error for patent detail fetch failures."""


def build_claims_payload(claims: list[str]) -> dict:
    """Build the claims response payload; independence follows the opener."""
    if not claims:
        return {"success": False, "claims": []}
    payload_claims = []
    for index, text in enumerate(claims, start=1):
        payload_claims.append({
            "number": index,
            "text": text,
            "independent": _is_independent_claim(text, index == 1),
        })
    return {"success": True, "claims": payload_claims}


def _is_independent_claim(claim_text: str, is_first: bool) -> bool:
    """True when the claim does not open by referencing an earlier claim."""
    if is_first:
        return True
    first_line = (claim_text or "").strip().splitlines()
    if not first_line:
        return True
    # Tolerate a leading claim-number prefix ("2. The method of claim 1…")
    line = re.sub(r"^\d{1,3}\.\s*", "", first_line[0].strip(), count=1)
    return not (_DEPENDENT_OPENERS.match(line) or _DEPENDENT_OPENERS_CN.match(line))


def split_claims_text(text: str) -> list[str]:
    """Split raw claims-document text into individual claims.

    Each claim starts with its number at line start ("1. ...", "2. ...").
    Text before the first claim number is discarded (headers/footers).
    """
    if not text:
        return []
    matches = list(_CLAIM_START_PATTERN.finditer(text))
    if not matches:
        return []
    claims = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            claims.append(body)
    return claims


def _strip_xml_tags(text: str) -> str:
    """Strip XML tags (SPEC.XML / CLM.XML payloads) and unescape entities."""
    if not text or "<" not in text:
        return text
    stripped = _XML_TAG_PATTERN.sub(" ", text)
    stripped = re.sub(r"[ \t]{2,}", " ", stripped)
    return unescape(stripped).strip()


def _find_spec_document(document_bag: list) -> dict | None:
    """Locate the specification document in a USPTO documentBag."""
    if not isinstance(document_bag, list):
        return None
    for doc in document_bag:
        if not isinstance(doc, dict):
            continue
        code = str(doc.get("documentCode", "") or "").strip().upper()
        desc = str(doc.get("documentCodeDescriptionText", "") or "").lower()
        if code == "SPEC" or "specification" in desc:
            return doc
    return None


def _find_claims_document(document_bag: list) -> dict | None:
    """Locate the claims document in a USPTO documentBag."""
    if not isinstance(document_bag, list):
        return None
    for doc in document_bag:
        if not isinstance(doc, dict):
            continue
        code = str(doc.get("documentCode", "") or "").strip().upper()
        if code in ("CLM", "WCLM"):
            return doc
    for doc in document_bag:
        if not isinstance(doc, dict):
            continue
        desc = str(doc.get("documentCodeDescriptionText", "") or "").lower()
        if "claim" in desc:
            return doc
    return None


async def _fetch_spec_pdf(source: str, patent_id: str) -> dict:
    """Resolve the USPTO specification PDF and return its proxy URL.

    The URL points at the existing lazy-download proxy (``/uspto/download``)
    so the frontend can embed the original PDF in an inline viewer without
    exposing the USPTO API key — the same mechanism patent document rows
    already use.
    """
    from sources import uspto_download
    from sources.dynamic_tool_params import _build_uspto_download_proxy_url
    from sources.long_task.text_extractor import (
        USPTO_PDF_PREFERRED_MIME_ORDER,
        get_download_url_from_doc,
    )

    try:
        app_number = await uspto_download.resolve_application_number(patent_id)
    except ValueError as exc:
        raise PatentDetailError(str(exc)) from exc
    document_bag = await uspto_download.fetch_document_bag(app_number)
    if document_bag is None:
        raise PatentDetailError(
            f"USPTO document list unavailable for application {app_number}"
        )
    spec_doc = _find_spec_document(document_bag)
    if spec_doc is None:
        raise PatentDetailError(
            f"No specification document in USPTO file wrapper for {app_number}"
        )
    pdf_url = get_download_url_from_doc(
        spec_doc, mime_order=USPTO_PDF_PREFERRED_MIME_ORDER
    )
    if not pdf_url:
        raise PatentDetailError(
            f"No downloadable specification PDF for application {app_number}"
        )
    return {"pdf_url": _build_uspto_download_proxy_url(pdf_url)}


async def _fetch_claims(source: str, patent_id: str) -> dict:
    """Fetch the USPTO claims document for *patent_id* and parse claims."""
    from sources import uspto_download

    try:
        app_number = await uspto_download.resolve_application_number(patent_id)
    except ValueError as exc:
        raise PatentDetailError(str(exc)) from exc
    document_bag = await uspto_download.fetch_document_bag(app_number)
    if document_bag is None:
        raise PatentDetailError(
            f"USPTO document list unavailable for application {app_number}"
        )
    claims_doc = _find_claims_document(document_bag)
    if claims_doc is None:
        raise PatentDetailError(
            f"No claims document in USPTO file wrapper for {app_number}"
        )
    text = await uspto_download.download_document_text(claims_doc)
    if not text:
        raise PatentDetailError(
            f"Claims text extraction failed for {app_number}"
        )
    return build_claims_payload(split_claims_text(_strip_xml_tags(text)))


def register_patent_detail_routes(logger, config):
    """Register patent detail routes with dependency injection."""
    router = APIRouter()

    @router.get("/patent/{source}/{patent_id}/spec")
    async def patent_spec(source: str, patent_id: str, http_request: Request):
        verify_firebase_token(http_request.headers.get("Authorization"))
        if source not in VALID_SOURCES:
            raise HTTPException(status_code=400, detail="Unsupported source")
        if not patent_id or len(patent_id) > 40:
            raise HTTPException(status_code=400, detail="Invalid patent_id")
        try:
            payload = await _fetch_spec_pdf(source, patent_id)
        except PatentDetailError as exc:
            logger.error(f"spec fetch failed — source={source}, id={patent_id}: {exc}")
            # Expected upstream misses are data conditions, not server
            # errors: return 200 + success:false instead of a 5xx, because
            # Cloudflare swaps origin 5xx responses for its own error page
            # (which carries no CORS headers) and the browser then blocks
            # the response as a CORS failure.
            return {"success": False, "message": "Patent specification unavailable"}
        return {"success": True, **payload}

    @router.get("/patent/{source}/{patent_id}/claims")
    async def patent_claims(source: str, patent_id: str, http_request: Request):
        verify_firebase_token(http_request.headers.get("Authorization"))
        if source not in VALID_SOURCES:
            raise HTTPException(status_code=400, detail="Unsupported source")
        if not patent_id or len(patent_id) > 40:
            raise HTTPException(status_code=400, detail="Invalid patent_id")
        try:
            payload = await _fetch_claims(source, patent_id)
        except PatentDetailError as exc:
            logger.error(f"claims fetch failed — source={source}, id={patent_id}: {exc}")
            return {"success": False, "message": "Patent claims unavailable"}
        if not payload.get("success"):
            # Honest degrade: claims not parseable from the public record.
            # Same 200 + success:false contract — see the spec endpoint.
            return {"success": False, "message": "权利要求暂不可用，请通过 PDF 原文查看"}
        return payload

    return router
