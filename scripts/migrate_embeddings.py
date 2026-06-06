#!/usr/bin/env python3
"""
Re-compute embeddings for all active knowledge records and write to Redis.

Usage:
    cd langsistance
    python scripts/migrate_embeddings.py              # migrate all status=1 records
    python scripts/migrate_embeddings.py --dry-run    # preview without writing
    python scripts/migrate_embeddings.py --batch 50   # batch size (default 20)

Before running, make sure config.ini [EMBEDDING] is set to your new provider
(e.g. siliconflow) and the corresponding API key is in .env.
"""

import os
import sys
import argparse
import time

# Ensure langsistance root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.knowledge.knowledge import get_embedding, get_db_connection, get_redis_connection
from sources.knowledge.embedding_text import build_knowledge_embedding_text
from sources.logger import Logger

logger = Logger("embedding_migration.log")


def fetch_knowledge_records(status=1):
    """Fetch all knowledge records with the given status from MySQL."""
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = """
                SELECT id, question, description, answer
                FROM knowledge
                WHERE status = %s
                ORDER BY id
            """
            cursor.execute(sql, (status,))
            rows = cursor.fetchall()
            logger.info(f"Fetched {len(rows)} knowledge records with status={status}")
            return rows
    finally:
        connection.close()


def migrate_embeddings(dry_run=False, batch_size=20):
    """
    Re-compute embeddings for all active knowledge records.

    Args:
        dry_run: If True, only preview — don't write to Redis.
        batch_size: Number of records to process between progress reports.
    """
    records = fetch_knowledge_records(status=1)
    total = len(records)

    if total == 0:
        print("No active knowledge records found. Nothing to migrate.")
        return

    print(f"Found {total} active knowledge records.")
    if dry_run:
        print("DRY RUN — will not write to Redis.\n")
    else:
        print(f"Starting migration with batch size {batch_size}...\n")

    redis_conn = None
    if not dry_run:
        redis_conn = get_redis_connection()

    success_count = 0
    error_count = 0
    start_time = time.time()

    for i, row in enumerate(records, 1):
        knowledge_id = row["id"]
        question = row["question"] or ""
        description = row["description"] or ""
        answer = row["answer"] or ""

        try:
            # Build embedding text the same way as knowledge.py
            embedding_text = build_knowledge_embedding_text(question, description, answer)
            embedding = get_embedding(embedding_text)

            if not dry_run:
                redis_key = f"knowledge_embedding_{knowledge_id}"
                redis_conn.set(redis_key, str(embedding))
                logger.info(f"Stored embedding for knowledge_id={knowledge_id}")

            success_count += 1

        except Exception as e:
            error_count += 1
            logger.error(f"Failed knowledge_id={knowledge_id}: {str(e)}")
            print(f"  [ERROR] id={knowledge_id}: {str(e)}")
            continue

        # Progress report
        if i % batch_size == 0 or i == total:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            status = "DRY RUN" if dry_run else "Migrated"
            print(
                f"  [{status}] {i}/{total} ({100*i//total}%) | "
                f"OK: {success_count} | Errors: {error_count} | "
                f"{rate:.1f} rec/s"
            )

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    if dry_run:
        print(f"DRY RUN complete — {total} records would be migrated.")
    else:
        print(
            f"Migration complete: {success_count} succeeded, {error_count} failed "
            f"in {elapsed:.1f}s ({total/elapsed:.1f} rec/s avg)"
        )
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-compute knowledge embeddings with the current embedding provider"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview only — do not write to Redis"
    )
    parser.add_argument(
        "--batch", type=int, default=20,
        help="Batch size for progress reporting (default: 20)"
    )
    args = parser.parse_args()

    migrate_embeddings(dry_run=args.dry_run, batch_size=args.batch)
