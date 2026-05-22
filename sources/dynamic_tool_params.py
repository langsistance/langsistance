import json
from typing import Any, Dict


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
