#!/usr/bin/env python3
"""
Re-compute embeddings for all active knowledge records and write to Redis.

Standalone script — no project imports, reads config from .env file.

Usage:
    cd langsistance
    python3 scripts/migrate_embeddings.py --dry-run    # preview first
    python3 scripts/migrate_embeddings.py               # migrate all
"""

import os
import sys
import argparse
import time

# ============================================================================
# Load .env file from project root
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # langsistance/

def load_env():
    """Load .env from project root into os.environ."""
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.exists(env_path):
        print(f"ERROR: .env not found at {env_path}")
        sys.exit(1)
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("'").strip('"')
            if key and val and key not in os.environ:
                os.environ[key] = val

load_env()

# ============================================================================
# Config — edit here if .env doesn't work
# ============================================================================
MYSQL_CONFIG = {
    "host":   "172.31.17.139",
    "port": 3306,
    "user": "copiioai_user",
    "password": "",
    "database": "copiioai_db",
    "charset":  "utf8mb4",
}

REDIS_CONFIG = {
    "host": "172.31.17.139",
    "port": 6379,
}

EMBEDDING_CONFIG = {
    "provider":"siliconflow",
    "model": "BAAI/bge-m3",
    "api_key":  "",
    "base_url":  "https://api.siliconflow.cn/v1",
}

# ============================================================================
# Database & Redis helpers
# ============================================================================

def get_db_connection():
    import pymysql
    import pymysql.cursors
    return pymysql.connect(cursorclass=pymysql.cursors.DictCursor, **MYSQL_CONFIG)


def get_redis_connection():
    import redis
    return redis.Redis(
        host=REDIS_CONFIG["host"],
        port=REDIS_CONFIG["port"],
        decode_responses=True,
        socket_connect_timeout=10,
        socket_timeout=10,
    )


# ============================================================================
# Embedding helpers
# ============================================================================

def build_embedding_text(question, description, answer):
    """Build embedding text — same logic as project's embedding_text.py."""
    def clean(value):
        return " ".join(("" if value is None else str(value)).split())
    parts = []
    for label, val in [("Question", question), ("Routing hint", description), ("Knowledge content", answer)]:
        c = clean(val)
        if c:
            parts.append(f"{label}:\n{c}")
    return "\n\n".join(parts)


def get_embedding(text):
    """Call embedding API. BGE-M3 supports up to 8192 tokens, no truncation needed."""
    from openai import OpenAI
    cfg = EMBEDDING_CONFIG
    if cfg["provider"] == "openai":
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    else:
        client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    return client.embeddings.create(model=cfg["model"], input=text).data[0].embedding


# ============================================================================
# Migration
# ============================================================================

def fetch_knowledge_records(status=1):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, question, description, answer FROM knowledge WHERE status = %s ORDER BY id",
                (status,),
            )
            return cur.fetchall()
    finally:
        conn.close()


def migrate(dry_run=False, batch_size=20):
    records = fetch_knowledge_records(status=1)
    total = len(records)

    if total == 0:
        print("No active knowledge records found.")
        return

    print(f"Found {total} active knowledge records.")
    if dry_run:
        print("DRY RUN — will NOT write to Redis.\n")
    else:
        print(f"Starting migration...\n")

    redis_conn = None if dry_run else get_redis_connection()
    ok, errors, t0 = 0, 0, time.time()

    for i, row in enumerate(records, 1):
        kid = row["id"]
        try:
            text = build_embedding_text(row["question"], row.get("description"), row.get("answer"))
            vec = get_embedding(text)
            if not dry_run:
                redis_conn.set(f"knowledge_embedding_{kid}", str(vec))
            ok += 1
        except Exception as e:
            errors += 1
            print(f"  [ERROR] id={kid}: {e}")
            continue

        if i % batch_size == 0 or i == total:
            elapsed = time.time() - t0
            label = "DRY RUN" if dry_run else "Migrated"
            rate = f"{i / elapsed:.1f} rec/s" if elapsed > 0 else ""
            print(f"  [{label}] {i}/{total} ({100 * i // total}%) | OK: {ok} | Errors: {errors} | {rate}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    if dry_run:
        print(f"DRY RUN complete — {total} records would be migrated.")
    else:
        print(f"Done: {ok} ok, {errors} failed in {elapsed:.1f}s ({total / elapsed:.1f} rec/s)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-compute knowledge embeddings")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch", type=int, default=20)
    args = parser.parse_args()
    migrate(dry_run=args.dry_run, batch_size=args.batch)
