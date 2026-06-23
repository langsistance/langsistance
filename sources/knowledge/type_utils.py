import json
from typing import Any


def _is_workflow_params(params: Any) -> bool:
    if not params:
        return False
    if isinstance(params, dict):
        return params.get("type") == "workflow"
    if not isinstance(params, str):
        return False
    try:
        parsed = json.loads(params)
    except (TypeError, ValueError):
        return False
    return isinstance(parsed, dict) and parsed.get("type") == "workflow"


def _is_long_task_params(params: Any) -> bool:
    """Check if params indicate a long task (type=3) knowledge entry."""
    if not params:
        return False
    if isinstance(params, dict):
        return params.get("type") == "long_task"
    if not isinstance(params, str):
        return False
    try:
        parsed = json.loads(params)
    except (TypeError, ValueError):
        return False
    return isinstance(parsed, dict) and parsed.get("type") == "long_task"


def infer_knowledge_type(raw_type: Any, params: Any = None) -> int:
    if _is_long_task_params(params):
        return 3
    if _is_workflow_params(params):
        return 2
    try:
        knowledge_type = int(raw_type)
    except (TypeError, ValueError):
        return 1
    return knowledge_type if knowledge_type in (1, 2, 3) else 1
