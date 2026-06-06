from typing import Any, Iterable, List, Optional, Tuple, Union


def _normalize_push_filter(push_filter: Union[int, str, None]) -> Optional[int]:
    if push_filter is None:
        return None
    if isinstance(push_filter, bool):
        raise ValueError("push_filter must be an integer")
    return int(push_filter)


def _normalize_ids(ids: Optional[Iterable[Any]]) -> List[Any]:
    if ids is None:
        return []
    return list(ids)


def _id_from_row(row: Any) -> Any:
    if isinstance(row, dict):
        return row["id"]
    return row[0]


def fetch_push_tool_ids(cursor: Any, push_filter: Union[int, str, None]) -> Optional[List[Any]]:
    normalized_push_filter = _normalize_push_filter(push_filter)
    if normalized_push_filter is None:
        return None

    cursor.execute(
        """
        SELECT id
        FROM tools
        WHERE status = %s
          AND push = %s
        """,
        (1, normalized_push_filter),
    )
    return [_id_from_row(row) for row in cursor.fetchall()]


def fetch_push_knowledge_ids(
    cursor: Any,
    push_filter: Union[int, str, None],
    push_tool_ids: Optional[Iterable[Any]] = None,
) -> Optional[List[Any]]:
    normalized_push_filter = _normalize_push_filter(push_filter)
    if normalized_push_filter is None:
        return None

    if push_tool_ids is None:
        push_tool_ids = fetch_push_tool_ids(cursor, normalized_push_filter)

    push_condition, push_params = build_knowledge_push_filter_condition(
        normalized_push_filter,
        push_tool_ids,
    )
    cursor.execute(
        f"""
        SELECT id
        FROM knowledge
        WHERE status = %s
        {push_condition}
        """,
        [1, *push_params],
    )
    return [_id_from_row(row) for row in cursor.fetchall()]


def build_knowledge_push_filter_condition(
    push_filter: Union[int, str, None],
    push_tool_ids: Optional[Iterable[Any]] = None,
    knowledge_table: str = "knowledge",
) -> Tuple[str, List[Any]]:
    normalized_push_filter = _normalize_push_filter(push_filter)
    if normalized_push_filter is None:
        return "", []

    normalized_tool_ids = _normalize_ids(push_tool_ids)
    if not normalized_tool_ids:
        return (
            f"""
                          AND {knowledge_table}.type = 2
        """,
            [],
        )

    placeholders = ",".join(["%s"] * len(normalized_tool_ids))
    return (
        f"""
                          AND (
                              {knowledge_table}.type = 2
                              OR {knowledge_table}.tool_id IN ({placeholders})
                          )
        """,
        normalized_tool_ids,
    )


def build_share_push_filter_condition(
    push_filter: Union[int, str, None],
    push_knowledge_ids: Optional[Iterable[Any]] = None,
    share_table: str = "knowledge_share",
) -> Tuple[str, List[Any]]:
    normalized_push_filter = _normalize_push_filter(push_filter)
    if normalized_push_filter is None:
        return "", []

    normalized_knowledge_ids = _normalize_ids(push_knowledge_ids)
    if not normalized_knowledge_ids:
        return (
            """
                  AND 1 = 0
        """,
            [],
        )

    placeholders = ",".join(["%s"] * len(normalized_knowledge_ids))
    return (
        f"""
                  AND {share_table}.knowledge_id IN ({placeholders})
        """,
        normalized_knowledge_ids,
    )
