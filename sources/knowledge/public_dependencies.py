import json
from typing import Any, Dict, List, Optional, Set

from sources.knowledge.type_utils import infer_knowledge_type


def _parse_workflow_params(params: Any) -> Optional[Dict[str, Any]]:
    if isinstance(params, dict):
        return params
    if not isinstance(params, str) or not params.strip():
        return None
    try:
        parsed = json.loads(params)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_workflow_dependency_ids(params: Any) -> List[int]:
    workflow = _parse_workflow_params(params)
    if not workflow or workflow.get("type") != "workflow":
        return []

    steps = workflow.get("steps")
    if not isinstance(steps, list):
        return []

    dependency_ids: List[int] = []
    seen: Set[int] = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        try:
            knowledge_id = int(step.get("knowledge_id"))
        except (TypeError, ValueError):
            continue
        if knowledge_id <= 0 or knowledge_id in seen:
            continue
        dependency_ids.append(knowledge_id)
        seen.add(knowledge_id)
    return dependency_ids


def promote_public_workflow_dependencies(
    cursor,
    user_id: str,
    public: Optional[int],
    knowledge_type: Optional[int],
    params: Any,
) -> List[int]:
    if public != 2 or infer_knowledge_type(knowledge_type, params) != 2:
        return []

    dependency_ids = extract_workflow_dependency_ids(params)
    if not dependency_ids:
        return []

    placeholders = ",".join(["%s"] * len(dependency_ids))
    cursor.execute(
        f"""
        SELECT id, public, `type`, params
        FROM knowledge
        WHERE id IN ({placeholders})
          AND user_id = %s
          AND status = 1
        """,
        (*dependency_ids, user_id),
    )
    rows = cursor.fetchall() or []
    ids_to_promote = [
        int(row["id"])
        for row in rows
        if row.get("public") != 2 and infer_knowledge_type(row.get("type"), row.get("params")) == 1
    ]
    if not ids_to_promote:
        return []

    update_placeholders = ",".join(["%s"] * len(ids_to_promote))
    cursor.execute(
        f"""
        UPDATE knowledge
        SET public = %s
        WHERE user_id = %s
          AND id IN ({update_placeholders})
          AND status = 1
          AND public <> 2
        """,
        (2, user_id, *ids_to_promote),
    )
    return ids_to_promote


def fetch_workflow_dependency_questions(cursor, params: Any) -> List[Dict[str, Any]]:
    dependency_ids = extract_workflow_dependency_ids(params)
    if not dependency_ids:
        return []

    placeholders = ",".join(["%s"] * len(dependency_ids))
    cursor.execute(
        f"""
        SELECT id, question, `type`, params
        FROM knowledge
        WHERE id IN ({placeholders})
          AND public = %s
          AND status = 1
        """,
        (*dependency_ids, 2),
    )
    rows = cursor.fetchall() or []
    rows_by_id = {
        int(row["id"]): row
        for row in rows
        if infer_knowledge_type(row.get("type"), row.get("params")) == 1
    }

    dependencies: List[Dict[str, Any]] = []
    for knowledge_id in dependency_ids:
        row = rows_by_id.get(knowledge_id)
        question = str(row.get("question") or "").strip() if row else ""
        if question:
            dependencies.append({
                "knowledge_id": knowledge_id,
                "question": question,
            })
    return dependencies
