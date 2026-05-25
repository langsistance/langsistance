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


@dataclass
class UsptoDownloadFile:
    content: bytes
    media_type: str
    filename: str


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
