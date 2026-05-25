import json
import re
from typing import Any, Callable, Dict
from urllib.parse import urlsplit, urlunsplit

from sources.logger import Logger


USPTO_DOWNLOAD_API_PREFIX = "https://api.uspto.gov/api/v1/download/applications"
_URL_RE = re.compile(r'https?://[^\s"\'<>\])}]+')
logger = Logger("dynamic_tool_params.log")


def _coerce_json_object(value: Any, value_name: str) -> Dict[str, Any]:
    """Return a JSON object from a dict or legacy JSON string."""
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {value_name} at line {exc.lineno}, "
                f"column {exc.colno}: {exc.msg}"
            ) from exc

        if isinstance(parsed, dict):
            return parsed

        raise ValueError(
            f"{value_name} must decode to a JSON object, "
            f"got {type(parsed).__name__}"
        )

    raise ValueError(
        f"{value_name} must be a JSON object or JSON string, "
        f"got {type(value).__name__}"
    )


def _is_empty_param_value(value: Any) -> bool:
    """Return True for values treated as empty tool request parameters."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (dict, list, tuple, set)):
        return len(value) == 0
    return False


def _is_path_query_body_empty(params: Dict[str, Any]) -> bool:
    """Return True when path, query, and body carry no request data."""
    return all(
        _is_empty_param_value(params.get(key))
        for key in ("path", "query", "body")
    )


def _append_path_to_url(base_url: str, path: Any) -> str:
    """Append a relative params.path value to the configured tool URL."""
    if _is_empty_param_value(path):
        return base_url

    if not isinstance(path, str):
        raise ValueError("path must be a URL path string")

    path = path.strip()
    path_parts = urlsplit(path)
    if path_parts.scheme or path_parts.netloc or path_parts.query or path_parts.fragment:
        raise ValueError("path must be a URL path, not an absolute URL or query string")

    path_value = path_parts.path
    if not path_value:
        return base_url

    base_parts = urlsplit(base_url)
    base_path = base_parts.path or ""
    if base_path and base_path != "/":
        combined_path = f"{base_path.rstrip('/')}/{path_value.lstrip('/')}"
    else:
        combined_path = f"/{path_value.lstrip('/')}"

    return urlunsplit(base_parts._replace(path=combined_path))


def _extract_first_url(value: Any) -> str | None:
    """Extract the first URL from a response string."""
    if not isinstance(value, str):
        return None

    match = _URL_RE.search(value)
    if not match:
        return None

    return match.group(0).rstrip(".,;")


def _iter_document_bags(result_data: Any):
    if isinstance(result_data, dict):
        document_bag = result_data.get("documentBag")
        if isinstance(document_bag, list):
            yield document_bag
        for value in result_data.values():
            if isinstance(value, (dict, list)):
                yield from _iter_document_bags(value)
    elif isinstance(result_data, list):
        for item in result_data:
            yield from _iter_document_bags(item)


def _replace_uspto_download_urls(
    result_data: Any,
    headers: Dict[str, Any],
    fetch_text: Callable[[str, Dict[str, Any]], str],
) -> Any:
    """Replace USPTO download API URLs in documentBag with resolved URLs."""
    for document_bag in _iter_document_bags(result_data):
        logger.info(f"documentBag: {document_bag}")
        for document in document_bag:
            if not isinstance(document, dict):
                continue
            download_options = document.get("downloadOptionBag")
            if not isinstance(download_options, list):
                continue
            logger.info(f"downloadOptionBag: {download_options}")
            for option in download_options:
                if not isinstance(option, dict):
                    continue
                download_url = option.get("downloadUrl")
                if isinstance(download_url, str):
                    logger.info(f"downloadUrl: {download_url}")
                if (
                    not isinstance(download_url, str)
                    or not download_url.startswith(USPTO_DOWNLOAD_API_PREFIX)
                ):
                    continue
                try:
                    resolved_text = fetch_text(download_url, headers)
                except Exception:
                    continue
                resolved_url = _extract_first_url(resolved_text)
                if resolved_url:
                    option["downloadUrl"] = resolved_url
                    logger.info(f"replaced downloadUrl: {resolved_url}")

    return result_data


def _replace_uspto_download_urls_for_batch(
    batch: list,
    headers: Dict[str, Any],
    fetch_text: Callable[[str, Dict[str, Any]], str],
) -> list:
    """Resolve USPTO download URLs only for the current display batch."""
    _replace_uspto_download_urls(batch, headers, fetch_text)
    return batch
