"""Backfill knowledge embeddings from MySQL into Redis.

The application stores knowledge embeddings in Redis under
``knowledge_embedding_{knowledge_id}``. This script rebuilds those cache entries
from active MySQL knowledge rows without deleting or changing MySQL data.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sources.knowledge.embedding_text import build_knowledge_embedding_text


REDIS_KEY_PREFIX = "knowledge_embedding_"


@dataclass
class BackfillFailure:
    knowledge_id: Any
    redis_key: str
    error: str


@dataclass
class BackfillSummary:
    scanned: int = 0
    rebuilt: int = 0
    skipped_existing: int = 0
    would_rebuild: int = 0
    failed: int = 0
    failures: list[BackfillFailure] = field(default_factory=list)


def redis_key_for_knowledge(knowledge_id: Any) -> str:
    return f"{REDIS_KEY_PREFIX}{knowledge_id}"


def fetch_knowledge_rows(
    connection: Any,
    *,
    user_id: str | None = None,
    knowledge_id: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    sql = (
        "SELECT id, question, description, answer "
        "FROM knowledge "
        "WHERE status = 1"
    )
    params: list[Any] = []

    if user_id:
        sql += " AND user_id = %s"
        params.append(user_id)

    if knowledge_id is not None:
        sql += " AND id = %s"
        params.append(knowledge_id)

    sql += " ORDER BY id"

    if limit is not None:
        sql += " LIMIT %s"
        params.append(limit)

    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        return list(cursor.fetchall())


def backfill_knowledge_embeddings(
    rows: Iterable[dict[str, Any]],
    redis_conn: Any,
    embedding_func: Callable[[str], list[float]],
    *,
    force: bool = False,
    dry_run: bool = False,
) -> BackfillSummary:
    summary = BackfillSummary()

    for row in rows:
        summary.scanned += 1
        knowledge_id = row["id"]
        redis_key = redis_key_for_knowledge(knowledge_id)

        try:
            exists = _redis_key_exists(redis_conn, redis_key)
            if exists and not force:
                summary.skipped_existing += 1
                continue

            if dry_run:
                summary.would_rebuild += 1
                continue

            embedding_text = build_knowledge_embedding_text(
                row.get("question", ""),
                row.get("description", ""),
                row.get("answer", ""),
            )
            embedding = embedding_func(embedding_text)
            redis_conn.set(redis_key, str(embedding))
            summary.rebuilt += 1
        except Exception as exc:
            summary.failed += 1
            summary.failures.append(
                BackfillFailure(
                    knowledge_id=knowledge_id,
                    redis_key=redis_key,
                    error=str(exc),
                )
            )

    return summary


def _redis_key_exists(redis_conn: Any, redis_key: str) -> bool:
    if hasattr(redis_conn, "exists"):
        return bool(redis_conn.exists(redis_key))
    return redis_conn.get(redis_key) is not None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _load_env(env_file: str | None) -> None:
    if env_file == "":
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()


def _format_summary(summary: BackfillSummary) -> str:
    return (
        "Backfill complete: "
        f"scanned={summary.scanned}, "
        f"rebuilt={summary.rebuilt}, "
        f"skipped_existing={summary.skipped_existing}, "
        f"would_rebuild={summary.would_rebuild}, "
        f"failed={summary.failed}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill active knowledge embeddings from MySQL into Redis."
    )
    parser.add_argument("--user-id", help="Only backfill knowledge for one user.")
    parser.add_argument(
        "--knowledge-id",
        type=_positive_int,
        help="Only backfill one knowledge row.",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Limit the number of active knowledge rows scanned.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild Redis embeddings even when the key already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show how many rows would be rebuilt without calling embeddings or writing Redis.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Env file to load before connecting. Use an empty value to skip.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _load_env(args.env_file)

    from sources.knowledge.knowledge import (
        get_db_connection,
        get_embedding,
        get_redis_connection,
    )

    db_connection = None
    redis_conn = None
    try:
        db_connection = get_db_connection()
        redis_conn = get_redis_connection()
        rows = fetch_knowledge_rows(
            db_connection,
            user_id=args.user_id,
            knowledge_id=args.knowledge_id,
            limit=args.limit,
        )
        summary = backfill_knowledge_embeddings(
            rows,
            redis_conn,
            get_embedding,
            force=args.force,
            dry_run=args.dry_run,
        )
    finally:
        if db_connection is not None:
            db_connection.close()
        if redis_conn is not None and hasattr(redis_conn, "close"):
            redis_conn.close()

    print(_format_summary(summary))
    for failure in summary.failures:
        print(
            f"Failed knowledge_id={failure.knowledge_id} "
            f"redis_key={failure.redis_key}: {failure.error}"
        )

    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
