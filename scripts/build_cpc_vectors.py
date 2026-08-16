#!/usr/bin/env python3
"""Build the CPC title vector cache (.npy) for semantic matching.

One-time deploy step: embeds every parsed main-group title with the
configured embedding provider and saves the vectors next to the titles
JSON so match_query_to_cpc() loads them without network calls.

Usage (server, with EMBEDDING_* configured):
    python scripts/build_cpc_vectors.py [--batch 200]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_env(path: str) -> None:
    """Load KEY=VALUE lines from an env file into os.environ (no
    overrides) — the embedding provider credentials usually live in the
    service .env and are not exported in an interactive shell.  Values
    may carry surrounding single/double quotes (common in .env files,
    parsed by python-dotenv for the service); those are stripped."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                value = value.strip()
                if (len(value) >= 2 and value[0] == value[-1]
                        and value[0] in ("'", '"')):
                    value = value[1:-1]
                os.environ.setdefault(key.strip(), value)
    except OSError:
        pass


_load_env(os.path.join(os.path.dirname(__file__), "..", ".env"))

from sources.long_task.cpc_semantic import (
    CPC_TITLES_JSON,
    CPC_VECTORS_NPY,
    load_cpc_titles,
)
from sources.logger import Logger

logger = Logger("cpc_semantic.log")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=200)
    parser.add_argument("--float16", action="store_true", default=True)
    args = parser.parse_args()

    entries = load_cpc_titles(CPC_TITLES_JSON)
    if not entries:
        logger.error(f"No CPC titles found at {CPC_TITLES_JSON}")
        return 1
    titles = [e["title"] for e in entries]
    logger.info(f"building vectors for {len(titles)} titles, batch={args.batch}")

    from sources.knowledge.knowledge import get_embeddings_batch

    vectors = []
    for start in range(0, len(titles), args.batch):
        chunk = titles[start:start + args.batch]
        try:
            vectors.extend(get_embeddings_batch(chunk))
        except Exception as exc:
            logger.error(f"embedding batch failed at {start}: {exc}")
            return 1
        if start % (args.batch * 20) == 0:
            logger.info(f"embedded {start + len(chunk)}/{len(titles)}")

    import numpy as np
    arr = np.asarray(vectors, dtype=np.float16 if args.float16 else np.float32)
    np.save(CPC_VECTORS_NPY, arr)
    logger.info(f"cpc vectors saved — shape={arr.shape} path={CPC_VECTORS_NPY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
