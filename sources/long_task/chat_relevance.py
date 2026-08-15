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
from typing import Any, List

from sources.long_task.candidate_metadata import (
    build_candidates,
    dedupe_candidates,
)
from sources.long_task.relevance_gate import (
    GATE_MAX_CANDIDATES_PER_CALL,
    score_candidates,
)

POOL_MAX_CANDIDATES = 300


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

    def unscored(self) -> list:
        """Candidates without a relevance_score, in pool order."""
        return [self._by_id[pid] for pid in self._order
                if "relevance_score" not in self._by_id[pid]]

    async def score_new(self, provider: Any) -> int:
        """Score unscored candidates via the Flash LLM.

        Batches at GATE_MAX_CANDIDATES_PER_CALL candidates per call and
        runs the batches CONCURRENTLY (the pool can hold several batches
        of unscored candidates after merges or retries).  Never raises.
        Returns how many candidates gained a score.
        """
        if provider is None:
            return 0
        pending = self.unscored()
        if not pending:
            return 0
        batches = [
            pending[i:i + GATE_MAX_CANDIDATES_PER_CALL]
            for i in range(0, len(pending), GATE_MAX_CANDIDATES_PER_CALL)
        ]
        await asyncio.gather(
            *(score_candidates(batch, self.query, provider) for batch in batches)
        )
        return len(pending) - len(self.unscored())

    def ranked(self, limit: int) -> list:
        """Family-deduped candidates ordered granted-first, then
        relevance_score desc, then filing date; unscored sink last.
        Sliced to *limit*."""
        kept, _ = dedupe_candidates(list(self._by_id.values()))
        return kept[:limit]

    def prune(self) -> None:
        """Keep only the top POOL_MAX_CANDIDATES ranked candidates."""
        kept = self.ranked(POOL_MAX_CANDIDATES)
        self._by_id = {c["patent_id"]: c for c in kept}
        self._order = [c["patent_id"] for c in kept]
