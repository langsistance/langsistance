"""Semantic reranking for the chat-path candidate pool (plan A).

Embeds the user's question and candidate titles with the configured
embedding provider (SiliconFlow BAAI/bge-m3 — multilingual, so a Chinese
question matches English titles semantically) and fuses the cosine
similarity with the LLM relevance scores.

Design constraints:

- Enhancement, not a dependency: any embedding failure degrades to the
  LLM-only ordering — callers always get a usable ranking back.
- Pure fusion math is separated from the provider call so the math is
  unit-testable without network access.
- No per-request state: candidates are re-ranked in place of the return
  value only; the pool itself is never mutated.
"""

import asyncio
import os
import time
from typing import Any, Optional

from sources.logger import Logger

logger = Logger("semantic_rerank.log")

RERANK_ENABLED = os.getenv("REACT_SEMANTIC_RERANK", "0") == "1"
RERANK_TOP_K = int(os.getenv("REACT_SEMANTIC_RERANK_TOPK", "30"))
RERANK_ALPHA = float(os.getenv("REACT_SEMANTIC_RERANK_ALPHA", "0.5"))
# Two-stage scoring: bge-m3 semantically prescores the whole incoming
# batch (one embedding call), and the flash LLM scores only the semantic
# head — the LLM budget lands on the semantically closest candidates
# instead of the newest slice, and deep-window candidates stop being
# pruned unscored.
PRESCORE_ENABLED = os.getenv("REACT_SEMANTIC_PRESCORE", "0") == "1"
# Embedding calls run chunked in a thread pool: one giant synchronous
# batch blocks the event loop for seconds, while chunked to_thread
# calls parallelize and never stall the loop.
SEMANTIC_BATCH_SIZE = int(os.getenv("SEMANTIC_BATCH_SIZE", "64"))

# Startup observability: the first log file line states whether rerank
# is active, so an enabled-but-not-restarted server is immediately
# visible in semantic_rerank.log.
logger.info(
    f"semantic rerank config — enabled={RERANK_ENABLED} "
    f"top_k={RERANK_TOP_K} alpha={RERANK_ALPHA} "
    f"prescore={PRESCORE_ENABLED}"
)


def cosine_similarity(a: Any, b: Any) -> float:
    """Cosine between two equal-length numeric vectors.

    Returns 0.0 on any degenerate input (empty, mismatched lengths,
    non-numeric, zero norm) so callers never crash on provider garbage.
    """
    try:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        norm_a = sum(float(x) * float(x) for x in a) ** 0.5
        norm_b = sum(float(x) * float(x) for x in b) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
    except (TypeError, ValueError):
        return 0.0


def _min_max_normalize(values: list) -> list:
    """Min-max normalize to [0, 1]; all-equal input maps to 0.5."""
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [0.5 for _ in values]
    return [(v - low) / (high - low) for v in values]


def fuse_ranking(candidates: list, semantic_scores: list,
                 alpha: float = RERANK_ALPHA) -> list:
    """Fuse LLM relevance scores with semantic cosine scores and return
    the candidates re-sorted (stable — original order breaks ties).

    The LLM score (0-5, default 0 when unscored) and the semantic score
    (cosine in [-1, 1], mapped to [0, 1]) are each min-max normalized
    over the window, then weighted: fused = alpha*llm + (1-alpha)*sem.
    A mismatched or empty semantic score list returns the original
    order unchanged.
    """
    if not candidates or len(semantic_scores) != len(candidates):
        return list(candidates)
    llm_scores = [c.get("relevance_score") or 0 for c in candidates]
    llm_norm = _min_max_normalize([float(s) for s in llm_scores])
    sem_norm = _min_max_normalize(
        [max(-1.0, min(1.0, float(s))) for s in semantic_scores])
    sem_mapped = [(s + 1.0) / 2.0 for s in sem_norm]
    fused = [
        alpha * llm_norm[i] + (1.0 - alpha) * sem_mapped[i]
        for i in range(len(candidates))
    ]
    ordered = sorted(
        zip(candidates, fused, range(len(candidates))),
        key=lambda item: (item[1], -item[2]),
        reverse=True,
    )
    return [c for c, _, _ in ordered]


def embed_texts(texts: list) -> Optional[list]:
    """Batch-embed texts with the configured provider.

    Returns a list of vectors in input order, or None on any failure —
    reranking is an enhancement, never a hard dependency.
    """
    if not texts:
        return None
    try:
        from sources.knowledge.knowledge import get_embeddings_batch
        return get_embeddings_batch([str(t) for t in texts])
    except Exception:
        logger.warning("semantic rerank embedding failed, skipping")
        return None


async def _embed_chunks(query: str, titles: list) -> Optional[list]:
    """Embed the query once, then the title chunks in parallel threads.

    Returns [query_vec, [chunk vectors...]] or None when the query
    embedding failed.  Chunk-level failures surface as None entries —
    callers decide how strict to be.
    """
    qv = await asyncio.to_thread(embed_texts, [str(query)])
    if qv is None or len(qv) != 1:
        return None
    chunks = [titles[i:i + SEMANTIC_BATCH_SIZE]
              for i in range(0, len(titles), SEMANTIC_BATCH_SIZE)]
    results = await asyncio.gather(
        *(asyncio.to_thread(embed_texts, chunk) for chunk in chunks))
    return qv[0], chunks, results


async def semantic_scores_batch(query: str, candidates: list) -> dict:
    """Semantic cosine scores for every titled candidate.

    Embeds the question once and the titles in concurrent chunks via
    thread pool (never blocks the event loop), returns
    {patent_id: cosine}.  Untitled candidates get no entry; any failure
    returns {} — prescoring is an enhancement, never a hard dependency.
    """
    if not query or not str(query).strip():
        return {}
    titled = [(c, str(c.get("title") or "").strip())
              for c in (candidates or []) if isinstance(c, dict)]
    titled = [(c, t) for c, t in titled if t]
    if len(titled) < 2:
        return {}
    start = time.monotonic()
    embedded = await _embed_chunks(query, [t for _, t in titled])
    if embedded is None:
        return {}
    query_vec, chunks, results = embedded
    scores: dict = {}
    pos = 0
    for ch, vecs in zip(chunks, results):
        if vecs is None or len(vecs) != len(ch):
            pos += len(ch)
            continue
        for (c, _), vec in zip(titled[pos:pos + len(ch)], vecs):
            pid = c.get("patent_id")
            if pid:
                scores[str(pid)] = cosine_similarity(query_vec, vec)
        pos += len(ch)
    logger.info(
        f"semantic prescore — candidates={len(titled)} "
        f"elapsed={round(time.monotonic() - start, 1)}s")
    return scores


async def rerank_candidates(query: str, candidates: list,
                            top_k: int = RERANK_TOP_K,
                            alpha: float = RERANK_ALPHA) -> list:
    """Semantically re-rank the top-*top_k* candidates of a ranked list.

    Candidates without a non-empty title keep their LLM-only order and
    are excluded from the fusion window (their positions never change
    relative to each other).  Embedding failures return the original
    list untouched.  Embedding runs chunked in threads — the event
    loop never blocks on the provider.
    """
    if len(candidates) < 2:
        return list(candidates)
    window = list(candidates[:top_k])
    titled = [c for c in window if str(c.get("title") or "").strip()]
    if len(titled) < 2:
        return list(candidates)
    start = time.monotonic()
    embedded = await _embed_chunks(
        query, [str(c.get("title") or "").strip() for c in titled])
    if embedded is None:
        return list(candidates)
    query_vec, chunks, results = embedded
    sem_scores: list = []
    for ch, vecs in zip(chunks, results):
        if vecs is None or len(vecs) != len(ch):
            return list(candidates)
        sem_scores.extend(cosine_similarity(query_vec, v) for v in vecs)
    logger.info(
        f"semantic rerank — candidates={len(titled)} "
        f"elapsed={round(time.monotonic() - start, 1)}s")
    fused_titled = fuse_ranking(titled, sem_scores, alpha)
    out = []
    fused_iter = iter(fused_titled)
    titled_ids = {id(c) for c in titled}
    for c in window:
        if id(c) in titled_ids:
            out.append(next(fused_iter))
        else:
            out.append(c)
    return out + list(candidates[top_k:])
