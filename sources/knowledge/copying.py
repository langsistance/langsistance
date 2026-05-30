import copy
import json
from typing import Any, Callable, Optional

from sources.knowledge.type_utils import infer_knowledge_type


EmbeddingWriter = Callable[[int, str, str], None]


class KnowledgeCopyError(ValueError):
    pass


def copy_knowledge_to_user(
    connection,
    source_knowledge_id: int,
    target_user_id: str,
    embedding_writer: Optional[EmbeddingWriter] = None,
) -> Optional[int]:
    connection.begin()
    try:
        with connection.cursor() as cursor:
            source = _fetch_active_knowledge(cursor, source_knowledge_id)
            if not source:
                connection.commit()
                return None

            knowledge_type = infer_knowledge_type(source.get("type"), source.get("params"))
            if knowledge_type == 2:
                new_knowledge_id = _copy_workflow_knowledge(
                    cursor,
                    source,
                    target_user_id,
                    embedding_writer,
                )
            else:
                new_knowledge_id = _copy_normal_knowledge(
                    cursor,
                    source,
                    target_user_id,
                    embedding_writer,
                    require_push2_tool=False,
                )

        connection.commit()
        return new_knowledge_id
    except Exception:
        connection.rollback()
        raise


def _copy_workflow_knowledge(cursor, source: dict, target_user_id: str, embedding_writer: Optional[EmbeddingWriter]) -> int:
    workflow_spec = _parse_workflow_params(source.get("params"))
    dependencies: dict[int, dict] = {}
    dependency_tools: dict[int, dict] = {}
    id_map: dict[int, int] = {}

    for step in workflow_spec["steps"]:
        dependency_id = int(step["knowledge_id"])
        if dependency_id not in dependencies:
            dependency = _fetch_active_knowledge(cursor, dependency_id)
            if not dependency:
                raise KnowledgeCopyError(f"Workflow dependency knowledge {dependency_id} not found")
            if infer_knowledge_type(dependency.get("type"), dependency.get("params")) != 1:
                raise KnowledgeCopyError(f"Workflow dependency knowledge {dependency_id} must be normal knowledge")
            source_tool_id = dependency.get("tool_id") or 0
            if not source_tool_id:
                raise KnowledgeCopyError(f"Knowledge {dependency.get('id')} has no linked tool")
            tool = _fetch_active_tool(cursor, source_tool_id)
            if not tool:
                raise KnowledgeCopyError(f"Tool {source_tool_id} not found")
            if int(tool.get("push") or 1) != 2:
                raise KnowledgeCopyError(f"Workflow dependency tool {source_tool_id} must have push=2")
            dependencies[dependency_id] = dependency
            dependency_tools[dependency_id] = tool

    for step in workflow_spec["steps"]:
        dependency_id = int(step["knowledge_id"])
        if dependency_id not in id_map:
            id_map[dependency_id] = _copy_normal_knowledge(
                cursor,
                dependencies[dependency_id],
                target_user_id,
                embedding_writer,
                require_push2_tool=True,
                source_tool=dependency_tools[dependency_id],
            )
        step["knowledge_id"] = id_map[dependency_id]

    workflow_copy = _build_private_knowledge_copy(source, target_user_id, tool_id=0, knowledge_type=2)
    workflow_copy["params"] = json.dumps(workflow_spec, ensure_ascii=False)
    return _insert_knowledge(cursor, workflow_copy, embedding_writer)


def _copy_normal_knowledge(
    cursor,
    source: dict,
    target_user_id: str,
    embedding_writer: Optional[EmbeddingWriter],
    require_push2_tool: bool,
    source_tool: Optional[dict] = None,
) -> int:
    source_tool_id = source.get("tool_id") or 0
    if not source_tool_id:
        raise KnowledgeCopyError(f"Knowledge {source.get('id')} has no linked tool")

    tool = source_tool or _fetch_active_tool(cursor, source_tool_id)
    if not tool:
        raise KnowledgeCopyError(f"Tool {source_tool_id} not found")

    push_value = int(tool.get("push") or 1)
    if require_push2_tool and push_value != 2:
        raise KnowledgeCopyError(f"Workflow dependency tool {source_tool_id} must have push=2")

    if push_value == 2:
        target_tool_id = source_tool_id
    else:
        target_tool_id = _insert_tool(cursor, tool, target_user_id)

    knowledge_type = infer_knowledge_type(source.get("type"), source.get("params"))
    knowledge_copy = _build_private_knowledge_copy(source, target_user_id, target_tool_id, knowledge_type)
    return _insert_knowledge(cursor, knowledge_copy, embedding_writer)


def _build_private_knowledge_copy(source: dict, target_user_id: str, tool_id: int, knowledge_type: int) -> dict:
    return {
        "user_id": target_user_id,
        "question": source.get("question") or "",
        "description": source.get("description") or "",
        "answer": source.get("answer") or "",
        "public": 1,
        "embedding_id": 0,
        "model_name": source.get("model_name") or "",
        "tool_id": tool_id,
        "params": source.get("params") or "",
        "type": knowledge_type,
    }


def _fetch_active_knowledge(cursor, knowledge_id: int) -> Optional[dict]:
    query_sql = """
        SELECT id, user_id, question, description, answer,
               public, model_name, tool_id, params, `type`
        FROM knowledge
        WHERE id = %s AND status = 1
    """
    cursor.execute(query_sql, (knowledge_id,))
    return cursor.fetchone()


def _fetch_active_tool(cursor, tool_id: int) -> Optional[dict]:
    query_sql = """
        SELECT id, title, description, url, push, public, timeout, params, user_id
        FROM tools
        WHERE id = %s AND status = 1
    """
    cursor.execute(query_sql, (tool_id,))
    return cursor.fetchone()


def _insert_tool(cursor, tool: dict, target_user_id: str) -> int:
    tool_sql = """
        INSERT INTO tools
        (user_id, title, description, url, push, status, timeout, params)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(tool_sql, (
        target_user_id,
        tool.get("title") or "",
        tool.get("description") or "",
        tool.get("url") or "",
        tool.get("push") or 1,
        1,
        tool.get("timeout"),
        tool.get("params") or "",
    ))
    return cursor.lastrowid


def _insert_knowledge(cursor, knowledge: dict, embedding_writer: Optional[EmbeddingWriter]) -> int:
    knowledge_sql = """
        INSERT INTO knowledge
        (user_id, question, description, answer, public, status, embedding_id, model_name, tool_id, params, `type`)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(knowledge_sql, (
        knowledge["user_id"],
        knowledge["question"],
        knowledge["description"],
        knowledge["answer"],
        knowledge["public"],
        1,
        knowledge["embedding_id"],
        knowledge["model_name"],
        knowledge["tool_id"],
        knowledge["params"],
        knowledge["type"],
    ))
    knowledge_id = cursor.lastrowid
    if embedding_writer:
        embedding_writer(knowledge_id, knowledge["question"], knowledge["answer"])
    return knowledge_id


def _parse_workflow_params(params: Any) -> dict:
    if isinstance(params, str):
        try:
            workflow_spec = json.loads(params or "{}")
        except (TypeError, ValueError) as exc:
            raise KnowledgeCopyError("Workflow params must be valid JSON") from exc
    elif isinstance(params, dict):
        workflow_spec = copy.deepcopy(params)
    else:
        raise KnowledgeCopyError("Workflow params must be a JSON object")

    if not isinstance(workflow_spec, dict) or workflow_spec.get("type") != "workflow":
        raise KnowledgeCopyError("Workflow params type must be workflow")
    steps = workflow_spec.get("steps")
    if not isinstance(steps, list) or not steps:
        raise KnowledgeCopyError("Workflow params must include at least one step")

    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise KnowledgeCopyError(f"Workflow step {index} must be an object")
        if "knowledge_id" not in step:
            raise KnowledgeCopyError(f"Workflow step {index} must include knowledge_id")
        try:
            step["knowledge_id"] = int(step["knowledge_id"])
        except (TypeError, ValueError) as exc:
            raise KnowledgeCopyError(f"Workflow step {index} knowledge_id must be an integer") from exc
        if step["knowledge_id"] <= 0:
            raise KnowledgeCopyError(f"Workflow step {index} knowledge_id must be positive")

    return workflow_spec
