"""Relevance-ranked candidate pool for the chat ReAct path.

The chat loop may call the same backend keyword search tool several times
per turn (ladder discipline, tight to loose).  This module merges those
results into one candidate pool, scores newly arrived candidates against
the user's original question with the Flash LLM (same machinery as the
long-task relevance gate), and returns a ranked, family-deduped view for
display and observation digests.

Scoring failures never raise: unscored candidates sink to the bottom of
the ranking, keeping their pool order.
"""
import asyncio
import os
from typing import Any, List

from sources.long_task.candidate_metadata import (
    build_candidates,
    dedupe_candidates,
    is_dead_status,
    is_design_patent,
)
from sources.long_task.relevance_gate import score_candidates

def _env_int(name: str, default: int) -> int:
    """Parse an int env var, falling back to *default* on garbage."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


POOL_MAX_CANDIDATES = 300
SCORE_PER_CALL = int(os.getenv("REACT_SCORE_PER_CALL", "50"))
SCORE_BATCH_SIZE = _env_int("REACT_SCORE_BATCH_SIZE", 10)
# Cap concurrent scoring calls so a large pool cannot hammer the
# gateway into 429s; the semaphore bounds the gather burst.
SCORE_MAX_CONCURRENCY = _env_int("REACT_SCORE_MAX_CONCURRENCY", 6)


async def score_candidates_concurrent(candidates: list, query: str,
                                      provider: Any,
                                      rubric: str = "") -> int:
    """Score unscored candidates in concurrent small batches.

    The Flash provider is slow on 100-entry batches (~50s); 10-entry
    batches gathered concurrently with a semaphore cap cut wall-clock
    substantially without hammering the gateway.  Dead candidates
    (expired/abandoned/PCT storage) are skipped — they rank below every
    live candidate regardless of score, so scoring them is pure waste.
    *rubric* (optional) is the architecture-level interpretation
    supplement passed through to the gate prompt.  Never raises.
    Returns how many candidates gained a score.
    """
    if provider is None:
        return 0
    pending = [c for c in candidates
               if "relevance_score" not in c
               and not is_dead_status(c.get("status"))]
    if not pending:
        return 0
    batches = [
        pending[i:i + SCORE_BATCH_SIZE]
        for i in range(0, len(pending), SCORE_BATCH_SIZE)
    ]
    sem = asyncio.Semaphore(max(1, SCORE_MAX_CONCURRENCY))

    async def _scored(batch: list) -> None:
        async with sem:
            await score_candidates(batch, query, provider, rubric)

    await asyncio.gather(*(_scored(batch) for batch in batches))
    return len(pending) - len([c for c in pending
                               if "relevance_score" not in c])


class SearchPool:
    """Merged candidate pool for one chat turn."""

    def __init__(self, query: str):
        self.query = query
        self._by_id: dict = {}
        self._order: List[str] = []

    def __len__(self) -> int:
        return len(self._order)

    def add(self, raw_items: list) -> int:
        """Merge raw_items into the pool; return the number of NEW
        candidates added (already-known patent_ids are ignored)."""
        new = 0
        for c in build_candidates(raw_items or []):
            pid = c["patent_id"]
            if pid in self._by_id:
                continue
            self._by_id[pid] = c
            self._order.append(pid)
            new += 1
        return new

    def add_from_candidates(self, candidates: list) -> list:
        """Merge pre-built candidate dicts into the pool; return the
        list of NEW candidates added (insertion order)."""
        new: list = []
        for c in candidates or []:
            pid = c.get("patent_id")
            if not pid or pid in self._by_id:
                continue
            self._by_id[pid] = c
            self._order.append(pid)
            new.append(c)
        return new

    def unscored(self) -> list:
        """Candidates without a relevance_score, in pool order."""
        return [self._by_id[pid] for pid in self._order
                if "relevance_score" not in self._by_id[pid]]

    async def score_new(self, provider: Any) -> int:
        """Score unscored candidates via the Flash LLM in concurrent
        small batches.  Never raises.  Returns how many candidates
        gained a score."""
        return await score_candidates_concurrent(
            self.unscored(), self.query, provider)

    def ranked(self, limit: int) -> list:
        """Family-deduped candidates, dead statuses and design patents
        excluded entirely (they can never be a useful technical search
        result and must not reach display or export), then granted,
        relevance_score desc, then filing date; unscored sink last.
        Sliced to *limit*."""
        live = [
            c for c in self._by_id.values()
            if not is_dead_status(c.get("status"))
            and not is_design_patent(c)
        ]
        kept, _ = dedupe_candidates(live)
        return kept[:limit]

    def prune(self) -> None:
        """Keep only the top POOL_MAX_CANDIDATES ranked candidates."""
        kept = self.ranked(POOL_MAX_CANDIDATES)
        self._by_id = {c["patent_id"]: c for c in kept}
        self._order = [c["patent_id"] for c in kept]
