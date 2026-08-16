#!/usr/bin/env python3
"""Build the CPC title vector cache (.npy) for semantic matching.

One-time deploy step: embeds every parsed CPC title of the chosen tier
with the configured embedding provider and saves the vectors next to
the titles JSON so match_query_to_cpc() loads them without network
calls.

Usage (server, with EMBEDDING_* configured):
    python scripts/build_cpc_vectors.py [--groups {main,sub}] [--batch 200]
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

from sources.long_task.cpc_semantic import cpc_paths_for_level, load_cpc_titles
from sources.logger import Logger

logger = Logger("cpc_semantic.log")


def _embed_to_memmap(titles, target_path, batch, float16, get_embeddings):
    """Embed *titles* in batches and stream them straight into a memmap
    at *target_path* + ".tmp", atomically renamed to *target_path* at
    the end.

    Streaming matters: get_embeddings returns lists of Python floats
    and the sub tier holds ~150k x 1024 titles — accumulating every
    batch needs ~5GB of RAM and gets the process OOM-killed (observed
    on the server at ~44k titles).  The memmap keeps peak heap at one
    batch (~1MB).  On any failure the .tmp is removed and an existing
    *target_path* is left untouched.
    """
    import numpy as np

    tmp_path = target_path + ".tmp"
    arr = None
    try:
        for start in range(0, len(titles), batch):
            chunk = titles[start:start + batch]
            vecs = np.asarray(get_embeddings(chunk), dtype=np.float32)
            if arr is None:
                if vecs.ndim != 2 or vecs.shape[0] != len(chunk):
                    raise ValueError(
                        f"embedding batch returned shape {vecs.shape}, "
                        f"expected ({len(chunk)}, dim)")
                dtype = np.float16 if float16 else np.float32
                arr = np.lib.format.open_memmap(
                    tmp_path, mode="w+", dtype=dtype,
                    shape=(len(titles), vecs.shape[1]))
            arr[start:start + len(vecs)] = vecs
            if start % (batch * 20) == 0:
                logger.info(f"embedded {start + len(chunk)}/{len(titles)}")
        arr.flush()
    except Exception:
        del arr
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    del arr  # close the mmap before renaming (Windows)
    os.replace(tmp_path, target_path)
    return len(titles), vecs.shape[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=200)
    parser.add_argument("--groups", choices=("main", "sub"), default="sub",
                        help="CPC tier to embed (default: sub)")
    parser.add_argument("--float16", action="store_true", default=True)
    args = parser.parse_args()

    titles_path, vectors_path = cpc_paths_for_level(args.groups)
    entries = load_cpc_titles(titles_path)
    if not entries:
        logger.error(f"No CPC titles found at {titles_path}")
        return 1
    titles = [e["title"] for e in entries]
    logger.info(
        f"building vectors for {len(titles)} titles "
        f"({args.groups} tier), batch={args.batch}")

    from sources.knowledge.knowledge import get_embeddings_batch

    try:
        rows, dim = _embed_to_memmap(
            titles, vectors_path, args.batch, args.float16,
            get_embeddings_batch)
    except Exception as exc:
        logger.error(f"vector build failed: {exc}")
        return 1
    logger.info(f"cpc vectors saved — shape=({rows}, {dim}) path={vectors_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
