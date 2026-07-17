import json
import os
import re
import time
from typing import Any, Dict
from urllib.parse import quote, urlsplit, urlunsplit

from bs4 import BeautifulSoup

from sources.http_outbound import outbound_http

from sources.logger import Logger


USPTO_DOWNLOAD_API_PREFIX = "https://api.uspto.gov/api/v1/download/applications"
USPTO_DOWNLOAD_PROXY_PATH = "/uspto/download"
DEFAULT_COPIIOAI_API_BASE_URL = "https://api.copiioai.com"
ZLDJS_API_HOST = "open.zldsj.com"
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


def _get_copiioai_api_base_url(proxy_base_url: str | None = None) -> str:
    return (
        proxy_base_url
        or os.getenv("COPIIOAI_API_BASE_URL")
        or os.getenv("NEXT_PUBLIC_API_BASE")
        or DEFAULT_COPIIOAI_API_BASE_URL
    ).rstrip("/")


def _build_uspto_download_proxy_url(
    download_url: str,
    proxy_base_url: str | None = None,
) -> str:
    api_base_url = _get_copiioai_api_base_url(proxy_base_url)
    return (
        f"{api_base_url}{USPTO_DOWNLOAD_PROXY_PATH}"
        f"?url={quote(download_url, safe='')}"
    )


def _iter_document_bags(result_data: Any):
    if isinstance(result_data, dict):
        document_bag = result_data.get("documentBag")
        if isinstance(document_bag, list):
            yield document_bag
        for value in result_data.values():
            if isinstance(value, (dict, list)):
                yield from _iter_document_bags(value)
    elif isinstance(result_data, list):
        if any(
            isinstance(item, dict)
            and isinstance(item.get("downloadOptionBag"), list)
            for item in result_data
        ):
            yield result_data
        for item in result_data:
            yield from _iter_document_bags(item)


def _replace_uspto_download_urls(
    result_data: Any,
    proxy_base_url: str | None = None,
) -> Any:
    """Replace USPTO download API URLs in documentBag with lazy proxy URLs."""
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
                proxy_url = _build_uspto_download_proxy_url(download_url, proxy_base_url)
                option["downloadUrl"] = proxy_url
                logger.info(f"replaced downloadUrl: {proxy_url}")

    return result_data


def _replace_uspto_download_urls_for_batch(
    batch: list,
    proxy_base_url: str | None = None,
) -> list:
    """Rewrite USPTO download URLs only for the current display batch."""
    _replace_uspto_download_urls(batch, proxy_base_url)
    return batch


def _extract_raw_items(result_data: Any) -> list | None:
    if isinstance(result_data, list) and result_data:
        return result_data
    if isinstance(result_data, dict):
        return next(
            (value for value in result_data.values() if isinstance(value, list) and value),
            None,
        )
    return None


def _is_zldjs_api_url(url: str) -> bool:
    """Check if the URL targets the open.zldsj.com (DI patent platform) domain."""
    hostname = urlsplit(url).hostname
    return bool(hostname and hostname.lower().rstrip(".") == ZLDJS_API_HOST)


def _inject_zldjs_auth_params(request_params: dict) -> None:
    """Inject client_id, access_token, and scope into query params for open.zldsj.com API requests.

    - client_id   : read from PATENT_CLIENT_ID env var (set in .env)
    - access_token: read from Redis via patent/callback stored token;
                    auto-refreshes if expired or within 1 hour of expiry
    - scope       : fixed value "read_cn"
    """
    from sources.patent_token import ensure_valid_access_token

    client_id = os.getenv("PATENT_CLIENT_ID", "")
    if client_id:
        request_params["client_id"] = client_id

    access_token = ensure_valid_access_token()
    if access_token:
        request_params["access_token"] = access_token

    request_params["scope"] = "read_cn"


def execute_backend_tool_request(tool_info: Any, params: Dict[str, Any] | str | None) -> Dict[str, Any]:
    """Execute a push=2 backend tool and return parsed data plus list metadata."""
    params_data = _coerce_json_object(tool_info.params, "tool_info.params")
    user_params = _coerce_json_object(params or {}, "LLM tool params")
    url = _append_path_to_url(
        tool_info.url,
        user_params.get("path", params_data.get("path", "")),
    )

    method = params_data.get("method", "GET").upper()
    content_type = params_data.get("Content-Type", "application/json")
    headers = {"Content-Type": content_type}
    server_headers = params_data.get("header", {})
    if isinstance(server_headers, dict):
        headers.update(server_headers)

    request_params = user_params.get("query")
    request_body = user_params.get("body")
    if request_params is None:
        request_params = {}
    if not isinstance(request_params, dict):
        raise ValueError("LLM tool params query must be a JSON object")
    request_params["_t"] = str(int(time.time() * 1000))

    # Inject DI patent platform auth params for open.zldsj.com requests
    if _is_zldjs_api_url(url):
        _inject_zldjs_auth_params(request_params)

    # Auto-inject USPTO API key for api.uspto.gov requests
    if "api.uspto.gov" in url and "X-API-Key" not in headers:
        uspto_key = os.getenv("USPTO_API_KEY")
        if uspto_key:
            headers["X-API-Key"] = uspto_key

    timeout = getattr(tool_info, "timeout", None) or 30
    if method not in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
        raise ValueError(f"Unsupported HTTP method: {method}")
    request_kwargs = {
        "params": request_params,
        "headers": headers,
        "timeout": timeout,
    }
    if method in {"POST", "PUT", "PATCH"}:
        if not request_body:
            logger.warning(
                f"backend_tool: {method} request to {url[:120]} has empty body — "
                f"LLM may have failed to generate params. "
                f"param_keys={list(user_params.keys()) if isinstance(user_params, dict) else type(user_params).__name__}"
            )
            return {"data": "Request failed: LLM did not generate valid request parameters (empty body)", "raw_items": None}
        request_kwargs["json"] = request_body

    # 对 open.zldsj.com 请求打印完整请求信息
    if _is_zldjs_api_url(url):
        logger.info(
            f"[ZLDJS REQUEST] {method} {url}\n"
            f"  params: {json.dumps(request_params, ensure_ascii=False)}\n"
            f"  headers: {json.dumps(headers, ensure_ascii=False)}\n"
            f"  body: {json.dumps(request_body, ensure_ascii=False) if request_body else 'None'}"
        )

    # 对 api.uspto.gov 请求打印完整请求信息
    if "api.uspto.gov" in url:
        logger.info(
            f"[USPTO REQUEST] {method} {url}\n"
            f"  params: {json.dumps(request_params, ensure_ascii=False)}\n"
            f"  headers: {json.dumps({k: v for k, v in headers.items() if k.lower() != 'x-api-key'}, ensure_ascii=False)}\n"
            f"  body: {json.dumps(request_body, ensure_ascii=False) if request_body else 'None'}"
        )

    response = outbound_http.request(method, url, purpose="backend_tool", **request_kwargs)

    # 对 open.zldsj.com 请求打印完整返回值
    if _is_zldjs_api_url(url):
        resp_body = ""
        try:
            raw = response.text if response.text else ""
        except Exception:
            raw = ""
        if not raw:
            resp_body = "(empty)"
        elif len(raw) > 50000:
            resp_body = raw[:50000] + f"\n...(truncated, total {len(raw)} chars)"
        else:
            resp_body = raw
        logger.info(
            f"[ZLDJS RESPONSE] status={response.status_code}\n"
            f"  headers: {json.dumps(dict(response.headers), ensure_ascii=False)}\n"
            f"  body: {resp_body}"
        )

    # 对 api.uspto.gov 请求打印响应摘要
    if "api.uspto.gov" in url:
        resp_summary = ""
        try:
            if response.text:
                resp_summary = response.text[:5000]
        except Exception:
            resp_summary = "(unable to decode response body)"
        logger.info(
            f"[USPTO RESPONSE] status={response.status_code}, "
            f"content_type={response.headers.get('Content-Type', '?')}, "
            f"body_len={len(response.content)}, "
            f"body_preview={resp_summary[:2000]}"
        )

    if response.status_code != 200:
        result = f"Request failed, status code: {response.status_code}"
        return {"data": result, "raw_items": None}

    response_content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" in response_content_type:
        result = BeautifulSoup(response.content, "html.parser").get_text()
        return {"data": result, "raw_items": None}
    if "application/xml" in response_content_type or "text/xml" in response_content_type:
        try:
            result = BeautifulSoup(response.content, "xml").get_text()
            if not result.strip():
                result = response.text
        except Exception:
            result = response.text
        return {"data": result, "raw_items": None}

    try:
        result_data = response.json() if response.content else None
    except json.JSONDecodeError:
        result_data = response.text if response.text else None

    # ZLDJS patent API wraps records in {"context": {"records": [...]}}
    # _extract_raw_items only scans one level deep, so handle this explicitly.
    raw_items = _extract_raw_items(result_data)
    if raw_items is None and _is_zldjs_api_url(url) and isinstance(result_data, dict):
        context = result_data.get("context")
        if isinstance(context, dict):
            records = context.get("records")
            if isinstance(records, list) and records:
                raw_items = records

    return {
        "data": result_data,
        "raw_items": raw_items,
    }
