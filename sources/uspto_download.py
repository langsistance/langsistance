import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict

from sources.dynamic_tool_params import (
    USPTO_DOWNLOAD_API_PREFIX,
    _extract_first_url,
)
from sources.logger import Logger


logger = Logger("backend.log")

USPTO_DOCUMENTS_API_URL = (
    "https://api.uspto.gov/api/v1/patent/applications/{app_number}/documents"
)
USPTO_SEARCH_API_URL = "https://api.uspto.gov/api/v1/patent/applications/search"


@dataclass
class UsptoDownloadFile:
    content: bytes
    media_type: str
    filename: str


@dataclass
class UsptoHttpResponse:
    """Response-like object returned by the internal _download seam."""

    status_code: int
    content: bytes
    content_type: str

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


def _api_key() -> str:
    return os.getenv("USPTO_API_KEY") or os.getenv("USPTO_DOWNLOAD_API_KEY") or ""


async def _uspto_headers(extra: dict | None = None) -> dict:
    headers = {"Accept": "application/json"}
    api_key = _api_key()
    if api_key:
        headers["X-API-KEY"] = api_key
    if extra:
        headers.update(extra)
    return headers


async def _request_json(method: str, url: str, body: dict | None = None) -> dict | None:
    """Internal HTTP seam for USPTO JSON APIs (patchable in tests).

    Uses outbound_http so USPTO 400/429 rate-limit responses get the same
    central retry handling as the prosecution pipeline.
    """
    import asyncio

    from sources.http_outbound import outbound_http

    headers = await _uspto_headers()
    if body is not None:
        headers["Content-Type"] = "application/json"
    try:
        if body is not None:
            response = await asyncio.to_thread(
                outbound_http.post,
                url,
                purpose="patent_detail",
                headers=headers,
                json=body,
                timeout=30,
            )
        else:
            response = await asyncio.to_thread(
                outbound_http.get,
                url,
                purpose="patent_detail",
                headers=headers,
                timeout=30,
            )
    except Exception:
        return None
    if response.status_code != 200:
        return None
    try:
        payload = response.json()
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


async def _download(url: str) -> UsptoHttpResponse | None:
    """Internal HTTP seam for document downloads (patchable in tests).

    Uses outbound_http so USPTO 400/429 rate-limit responses get the same
    central retry handling as the prosecution pipeline.
    """
    import asyncio

    from sources.http_outbound import outbound_http

    headers = await _uspto_headers({"Accept": "*/*"})
    try:
        response = await asyncio.to_thread(
            outbound_http.get,
            url,
            purpose="patent_download",
            headers=headers,
            timeout=60,
        )
    except Exception:
        return None
    return UsptoHttpResponse(
        status_code=response.status_code,
        content=response.content,
        content_type=(response.headers.get("Content-Type", "") or "").lower(),
    )


# ── Patent detail (spec / claims) helpers ─────────────────────────────────────
#
# The split-view spec/claims endpoints reuse the prosecution-history
# download mechanism: PEDS document list → locate SPEC/CLM → download →
# text extraction.


async def fetch_document_bag(app_number: str) -> list | None:
    """Fetch the PEDS document list for *app_number*.

    Returns the ``documentBag`` list (possibly empty) or None when the
    API call failed — None distinguishes "id is not a valid application
    number" from "application exists but has no documents".
    """
    payload = await _request_json(
        "GET", USPTO_DOCUMENTS_API_URL.format(app_number=app_number)
    )
    if payload is None:
        return None
    bag = payload.get("documentBag", [])
    return bag if isinstance(bag, list) else []


async def _search_application_number_by_patent_number(patent_number: str) -> str | None:
    """Search PEDS for the application number of a granted patent number."""
    query = f'applicationMetaData.patentNumber:"{patent_number}"'
    payload = await _request_json(
        "POST",
        USPTO_SEARCH_API_URL,
        {
            "q": query,
            "pagination": {"offset": 0, "limit": 1},
            "fields": ["applicationNumberText", "applicationMetaData.patentNumber"],
        },
    )
    if not payload:
        return None
    results = (
        payload.get("patentFileWrapperDataBag")
        or payload.get("results")
        or payload.get("patentFileBag")
        or []
    )
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        return None
    app_number = str(results[0].get("applicationNumberText") or "").strip()
    return app_number or None


async def resolve_application_number(patent_id: str) -> str:
    """Resolve a US patent/application id → applicationNumberText.

    The id may be an application number (usable directly with the PEDS
    documents endpoint) or a granted patent number (requires a PEDS
    search).  Try the documents endpoint with the digits as-is first;
    when that fails or yields no documents, search by patentNumber —
    preserving non-digit prefixes, because design patents are numbered
    ``D754082`` and the D prefix must survive into the search query
    (falling back to the bare digits for sources that store them
    unprefixed).
    """
    normalized = re.sub(r"\s+", "", (patent_id or "").strip())
    digits = "".join(c for c in normalized if c.isdigit())
    if not digits or len(digits) < 6:
        raise ValueError(f"Invalid patent id: {patent_id!r}")

    document_bag = await fetch_document_bag(digits)
    if document_bag:
        return digits

    search_id = normalized.upper()
    app_number = await _search_application_number_by_patent_number(search_id)
    if not app_number and search_id != digits:
        app_number = await _search_application_number_by_patent_number(digits)
    if app_number:
        return app_number
    raise ValueError(f"Could not resolve USPTO application for id {patent_id!r}")


async def download_document_text(doc: dict, mime_order: tuple[str, ...] | None = None) -> str:
    """Download a USPTO documentBag entry and extract its text.

    Follows the prosecution pattern: pick the best downloadOptionBag URL
    (``mime_order``, defaulting to DOCX > XML > PDF for text), follow at
    most one in-body redirect, and extract text via the shared binary
    extractor (PDF text extraction is enabled; image-only scanned PDFs
    return empty).  Failure modes are logged so an empty result is
    diagnosable from backend.log alone.
    """
    import asyncio

    from sources.long_task.text_extractor import (
        extract_text_from_binary,
        get_download_url_from_doc,
    )

    def _doc_summary() -> str:
        code = str((doc or {}).get("documentCode") or "").strip()
        desc = str((doc or {}).get("documentCodeDescriptionText") or "").strip()[:60]
        return f"code={code}, desc={desc}"

    download_url = get_download_url_from_doc(doc or {}, mime_order=mime_order)
    if not download_url:
        logger.warning(f"uspto_download no_url — {_doc_summary()}")
        return ""

    for _hop in range(2):
        response = await _download(download_url)
        if response is None or response.status_code != 200:
            logger.warning(
                f"uspto_download http_fail — "
                f"status={None if response is None else response.status_code}, "
                f"url={download_url[:120]}"
            )
            return ""

        # xmlarchive URLs deliver tar binaries even when labelled XML;
        # binary content mislabelled as text shows NUL bytes.
        force_binary = "xmlarchive" in download_url.lower() or (
            b"\x00" in response.content[:2048]
        )

        if response.content_type and (
            force_binary
            or not any(
                t in response.content_type for t in ("text/", "json", "xml", "html")
            )
        ):
            text = await asyncio.to_thread(
                extract_text_from_binary,
                response.content,
                response.content_type,
                download_url,
                None,  # on_progress
                False,  # skip_pdf_extraction
            )
            text = (text or "").strip()
            if not text:
                logger.warning(
                    f"uspto_download extract_empty — "
                    f"type={response.content_type}, "
                    f"len={len(response.content)}, url={download_url[:120]}"
                )
            return text

        body = response.text.strip()
        redirect_url = _extract_first_url(body)
        if redirect_url and redirect_url != download_url:
            download_url = redirect_url
            continue
        return body

    return ""


def get_uspto_download_headers() -> Dict[str, str]:
    api_key = os.getenv("USPTO_DOWNLOAD_API_KEY") or os.getenv("USPTO_API_KEY")
    if not api_key:
        return {}
    return {"X-API-KEY": api_key}


def _filename_from_content_disposition(content_disposition: str | None) -> str | None:
    if not content_disposition:
        return None

    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', content_disposition)
    if not match:
        return None
    return os.path.basename(match.group(1).strip())


def _filename_from_download_url(download_url: str) -> str:
    path = download_url.split("?", 1)[0].rstrip("/")
    filename = os.path.basename(path)
    return filename or "uspto-download"


def _is_text_response(media_type: str) -> bool:
    normalized = media_type.lower()
    return (
        normalized.startswith("text/")
        or "json" in normalized
        or "xml" in normalized
        or "html" in normalized
    )


def _response_content(response: Any) -> bytes:
    content = getattr(response, "content", b"")
    if isinstance(content, str):
        return content.encode("utf-8")
    return content or b""


def _decode_text_content(content: bytes) -> str | None:
    sample = content[:2048]
    if b"\x00" in sample:
        return None
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not text.strip():
        return None
    return text


def _should_parse_response_text(media_type: str, content: bytes) -> bool:
    if _is_text_response(media_type):
        return True

    text = _decode_text_content(content)
    if not text:
        return False

    normalized = text.lstrip().lower()
    return (
        normalized.startswith("please use redirect url")
        or normalized.startswith("{")
        or normalized.startswith("<")
    )


def fetch_uspto_download_file(
    download_url: str,
    fetch_response: Callable[[str, Dict[str, str]], Any],
    request_headers: Dict[str, str] | None = None,
) -> UsptoDownloadFile:
    if not download_url.startswith(USPTO_DOWNLOAD_API_PREFIX):
        raise ValueError("Unsupported USPTO download URL")

    headers = request_headers or {}
    response = fetch_response(download_url, headers)
    if hasattr(response, "raise_for_status"):
        response.raise_for_status()

    response_headers = getattr(response, "headers", {}) or {}
    content = _response_content(response)

    logger.info(f"USPTO download response status: {getattr(response, 'status_code', 'unknown')}")
    logger.info(f"USPTO download response content length: {len(content)}")

    media_type = response_headers.get("Content-Type", "application/octet-stream")
    filename_url = download_url
    if _should_parse_response_text(media_type, content):
        response_text = _decode_text_content(content) or content.decode("utf-8", errors="replace")
        resolved_url = _extract_first_url(response_text)
        if not resolved_url:
            logger.warning(f"USPTO download non-file response: {response_text[:500]}")
            raise ValueError("USPTO download response did not contain downloadable file content")

        logger.info(f"USPTO download response contained file URL: {resolved_url}")
        filename_url = resolved_url
        resolved_response = fetch_response(resolved_url, headers)
        if hasattr(resolved_response, "raise_for_status"):
            resolved_response.raise_for_status()
        response = resolved_response
        response_headers = getattr(response, "headers", {}) or {}
        content = _response_content(response)
        media_type = response_headers.get("Content-Type", "application/octet-stream")
        logger.info(f"USPTO resolved file response status: {getattr(response, 'status_code', 'unknown')}")
        logger.info(f"USPTO resolved file content length: {len(content)}")

    if not content:
        raise ValueError("USPTO download response did not contain downloadable file content")

    filename = (
        _filename_from_content_disposition(response_headers.get("Content-Disposition"))
        or _filename_from_download_url(filename_url)
    )
    return UsptoDownloadFile(
        content=content,
        media_type=media_type,
        filename=filename,
    )
