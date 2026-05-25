import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict

from sources.dynamic_tool_params import USPTO_DOWNLOAD_API_PREFIX
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
    content = getattr(response, "content", b"")
    if isinstance(content, str):
        content = content.encode("utf-8")

    logger.info(f"USPTO download response status: {getattr(response, 'status_code', 'unknown')}")
    logger.info(f"USPTO download response content length: {len(content)}")

    media_type = response_headers.get("Content-Type", "application/octet-stream")
    filename = (
        _filename_from_content_disposition(response_headers.get("Content-Disposition"))
        or _filename_from_download_url(download_url)
    )
    return UsptoDownloadFile(
        content=content,
        media_type=media_type,
        filename=filename,
    )
