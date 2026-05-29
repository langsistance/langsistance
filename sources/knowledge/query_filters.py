from typing import Any


def _normalize_push_filter(push_filter: int | str | None) -> int | None:
    if push_filter is None:
        return None
    if isinstance(push_filter, bool):
        raise ValueError("push_filter must be an integer")
    return int(push_filter)


def build_knowledge_push_filter_condition(
    push_filter: int | str | None,
    knowledge_table: str = "knowledge",
) -> tuple[str, list[Any]]:
    normalized_push_filter = _normalize_push_filter(push_filter)
    if normalized_push_filter is None:
        return "", []

    return (
        f"""
                          AND (
                              {knowledge_table}.type = 2
                              OR EXISTS (
                                  SELECT 1
                                  FROM tools
                                  WHERE tools.id = {knowledge_table}.tool_id
                                    AND tools.status = 1
                                    AND tools.push = %s
                              )
                          )
        """,
        [normalized_push_filter],
    )


def build_share_push_filter_condition(
    push_filter: int | str | None,
    share_table: str = "knowledge_share",
) -> tuple[str, list[Any]]:
    normalized_push_filter = _normalize_push_filter(push_filter)
    if normalized_push_filter is None:
        return "", []

    return (
        f"""
                  AND EXISTS (
                      SELECT 1
                      FROM knowledge k
                      WHERE k.id = {share_table}.knowledge_id
                        AND k.status = 1
                        AND (
                            k.type = 2
                            OR EXISTS (
                                SELECT 1
                                FROM tools
                                WHERE tools.id = k.tool_id
                                  AND tools.status = 1
                                  AND tools.push = %s
                            )
                        )
                  )
        """,
        [normalized_push_filter],
    )
