import json
from typing import Any, Dict
from urllib.parse import urlsplit, urlunsplit


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
