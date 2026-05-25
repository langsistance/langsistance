import os
from typing import Callable, Dict

from sources.dynamic_tool_params import (
    USPTO_DOWNLOAD_API_PREFIX,
    _extract_first_url,
)
from sources.logger import Logger


logger = Logger("backend.log")


def get_uspto_download_headers() -> Dict[str, str]:
    api_key = os.getenv("USPTO_DOWNLOAD_API_KEY") or os.getenv("USPTO_API_KEY")
    if not api_key:
        return {}
    return {"X-API-KEY": api_key}


def resolve_uspto_download_url(
    download_url: str,
    fetch_text: Callable[[str, Dict[str, str]], str],
    request_headers: Dict[str, str] | None = None,
) -> str:
    if not download_url.startswith(USPTO_DOWNLOAD_API_PREFIX):
        raise ValueError("Unsupported USPTO download URL")

    headers = request_headers or {}
    response_text = fetch_text(download_url, headers)
    logger.info(f"USPTO download response_text: {response_text}")
    resolved_url = _extract_first_url(response_text)
    if not resolved_url:
        raise ValueError("USPTO download response did not contain a URL")
    return resolved_url
