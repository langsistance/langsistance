"""Tests for chat-path relevance ranking pool (chat_relevance)."""
import asyncio
import re
import unittest
from unittest import mock

from sources.long_task.chat_relevance import POOL_MAX_CANDIDATES, SearchPool


def _usp_raw_item(app_number, title, applicant="ACME Corp",
                  filing="2024-01-15", continuity_ids=None):
    item = {
        "applicationNumberText": app_number,
        "applicationMetaData": {
            "inventionTitle": title,
            "firstApplicantName": applicant,
            "filingDate": filing,
            "applicationStatusDescriptionText": "Patented Case",
        },
    }
    if continuity_ids:
        item["parentContinuityBag"] = [
            {"parentApplicationNumberText": i} for i in continuity_ids
        ]
    return item


class _FakeProvider:
    def __init__(self, scores=None, fail=False):
        self._scores = scores or {}
        self.fail = fail
        self.calls = 0

    async def complete_json(self, system, user):
        self.calls += 1
        if self.fail:
            raise RuntimeError("down")
        ids = re.findall(r"id=(\d+)", user)
        return {"scores": [
            {"id": i, "score": self._scores.get(i, 3)} for i in ids]}


class _ConcurrentProvider:
    def __init__(self):
        self.calls = 0
        self.inflight = 0
        self.max_inflight = 0

    async def complete_json(self, system, user):
        self.calls += 1
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        await asyncio.sleep(0.05)
        self.inflight -= 1
        ids = re.findall(r"id=(\d+)", user)
        return {"scores": [{"id": i, "score": 3} for i in ids]}


class TestEnvIntFallback(unittest.TestCase):
    def test_garbage_env_falls_back_to_default(self):
        from sources.long_task import chat_relevance as cr
        with mock.patch.dict("os.environ",
                             {"REACT_SCORE_BATCH_SIZE": "junk",
                              "REACT_SCORE_MAX_CONCURRENCY": "abc"}):
            self.assertEqual(
                cr._env_int("REACT_SCORE_BATCH_SIZE", 10), 10)
            self.assertEqual(
                cr._env_int("REACT_SCORE_MAX_CONCURRENCY", 6), 6)


class TestSearchPoolMerge(unittest.TestCase):
    def test_add_flattens_and_merges_new_ids(self):
        pool = SearchPool("测试问题")
        new = pool.add([_usp_raw_item("19511555", "A"),
                        _usp_raw_item("18184836", "B")])
        self.assertEqual(new, 2)
        self.assertEqual(len(pool), 2)

    def test_duplicate_ids_ignored(self):
        pool = SearchPool("测试问题")
        pool.add([_usp_raw_item("19511555", "A")])
        new = pool.add([_usp_raw_item("19511555", "A"),
                        _usp_raw_item("18184836", "B")])
        self.assertEqual(new, 1)
        self.assertEqual(len(pool), 2)

    def test_non_usp_shapes_add_nothing(self):
        pool = SearchPool("测试问题")
        new = pool.add([{"patentNumber": "US10150077B2"}])
        self.assertEqual(new, 0)
        self.assertEqual(len(pool), 0)

    def test_empty_adds_nothing(self):
        pool = SearchPool("测试问题")
        self.assertEqual(pool.add([]), 0)
        self.assertEqual(pool.add(None), 0)


class TestSearchPoolScoring(unittest.IsolatedAsyncioTestCase):
    async def test_scores_only_unscored_candidates(self):
        pool = SearchPool("测试问题")
        pool.add([_usp_raw_item("19511555", "A"),
                  _usp_raw_item("18184836", "B")])
        provider = _FakeProvider({"19511555": 5, "18184836": 2})
        scored = await pool.score_new(provider)
        self.assertEqual(scored, 2)
        self.assertEqual(provider.calls, 1)
        # second run: nothing left to score
        scored2 = await pool.score_new(provider)
        self.assertEqual(scored2, 0)
        self.assertEqual(provider.calls, 1)

    async def test_second_add_scores_only_new_ids(self):
        pool = SearchPool("测试问题")
        pool.add([_usp_raw_item("19511555", "A")])
        provider = _FakeProvider({"19511555": 5})
        await pool.score_new(provider)
        pool.add([_usp_raw_item("18184836", "B")])
        await pool.score_new(provider)
        # two batches: first with 1 id, second with 1 new id
        self.assertEqual(provider.calls, 2)

    async def test_provider_none_scores_nothing(self):
        pool = SearchPool("测试问题")
        pool.add([_usp_raw_item("19511555", "A")])
        self.assertEqual(await pool.score_new(None), 0)

    async def test_provider_failure_returns_zero_no_raise(self):
        pool = SearchPool("测试问题")
        pool.add([_usp_raw_item("19511555", "A")])
        scored = await pool.score_new(_FakeProvider(fail=True))
        self.assertEqual(scored, 0)
        self.assertEqual(pool.unscored().__len__(), 1)


class TestSearchPoolConcurrentScoring(unittest.IsolatedAsyncioTestCase):
    async def test_batches_scored_concurrently(self):
        pool = SearchPool("测试问题")
        pool.add([_usp_raw_item(str(19500000 + i), f"T{i}")
                  for i in range(250)])
        provider = _ConcurrentProvider()
        scored = await pool.score_new(provider)
        # SCORE_BATCH_SIZE=10 → 250/10 = 25 batches; concurrency capped at 6
        self.assertEqual(provider.calls, 25)
        self.assertEqual(provider.max_inflight, 6)
        self.assertEqual(scored, 250)


class TestHeadScoringAndConcurrency(unittest.IsolatedAsyncioTestCase):
    async def test_score_concurrent_uses_small_batches(self):
        from sources.long_task.chat_relevance import (
            SCORE_BATCH_SIZE, score_candidates_concurrent)
        from sources.long_task.candidate_metadata import build_candidates
        items = [_usp_raw_item(str(19500000 + i), f"T{i}")
                 for i in range(60)]
        cands = build_candidates(items)
        provider = _ConcurrentProvider()
        scored = await score_candidates_concurrent(cands, "q", provider)
        self.assertEqual(provider.calls, 6)          # 60/10 → 6 batches
        self.assertEqual(provider.max_inflight, 6)   # min(6 batches, cap 6)
        self.assertEqual(scored, 60)

    async def test_concurrency_capped_by_semaphore(self):
        from sources.long_task import chat_relevance as cr
        from sources.long_task.candidate_metadata import build_candidates
        items = [_usp_raw_item(str(19500000 + i), f"T{i}")
                 for i in range(100)]
        cands = build_candidates(items)
        provider = _ConcurrentProvider()
        with mock.patch.object(cr, "SCORE_MAX_CONCURRENCY", 2):
            scored = await cr.score_candidates_concurrent(cands, "q", provider)
        self.assertEqual(provider.calls, 10)         # 100/10 → 10 batches
        self.assertEqual(provider.max_inflight, 2)   # capped at 2
        self.assertEqual(scored, 100)

    async def test_add_from_candidates_returns_new_list(self):
        pool = SearchPool("测试问题")
        new = pool.add_from_candidates(
            [{"patent_id": "19511555", "title": "A"},
             {"patent_id": "18184836", "title": "B"}])
        self.assertEqual([c["patent_id"] for c in new],
                         ["19511555", "18184836"])
        again = pool.add_from_candidates(
            [{"patent_id": "19511555", "title": "A"},
             {"patent_id": "17222222", "title": "C"}])
        self.assertEqual([c["patent_id"] for c in again], ["17222222"])


class TestDeadCandidatesNotScored(unittest.IsolatedAsyncioTestCase):
    async def test_dead_candidates_skipped_by_scoring(self):
        from sources.long_task.chat_relevance import score_candidates_concurrent
        from sources.long_task.candidate_metadata import build_candidates
        dead = _usp_raw_item("19511555", "Dead dryer")
        dead["applicationMetaData"]["applicationStatusDescriptionText"] = \
            "Provisional Application Expired"
        cands = build_candidates([dead])
        provider = _ConcurrentProvider()
        scored = await score_candidates_concurrent(cands, "q", provider)
        self.assertEqual(scored, 0)
        self.assertEqual(provider.calls, 0)
        self.assertNotIn("relevance_score", cands[0])

    async def test_live_candidates_still_scored_alongside_dead(self):
        from sources.long_task.chat_relevance import score_candidates_concurrent
        from sources.long_task.candidate_metadata import build_candidates
        dead = _usp_raw_item("19511555", "Dead dryer")
        dead["applicationMetaData"]["applicationStatusDescriptionText"] = \
            "Abandoned  --  Failure to Respond to an Office Action"
        cands = build_candidates([dead, _usp_raw_item("18184836", "Live dryer")])
        provider = _ConcurrentProvider()
        scored = await score_candidates_concurrent(cands, "q", provider)
        self.assertEqual(scored, 1)
        self.assertEqual(provider.calls, 1)
        self.assertNotIn("relevance_score", cands[0])
        self.assertEqual(cands[1]["relevance_score"], 3)


class TestSearchPoolRanking(unittest.TestCase):
    def test_ranked_excludes_dead_entirely(self):
        pool = SearchPool("测试问题")
        dead = _usp_raw_item("19511555", "Expired granted dryer")
        dead["applicationMetaData"]["applicationStatusDescriptionText"] = \
            "Patent Expired Due to NonPayment of Maintenance Fees Under 37 CFR 1.362"
        dead["applicationMetaData"]["patentNumber"] = "9123456"
        pool.add([dead, _usp_raw_item("18184836", "Pending dry air control")])
        pool._by_id["19511555"]["relevance_score"] = 5
        pool._by_id["18184836"]["relevance_score"] = 5
        ranked = pool.ranked(10)
        self.assertEqual([c["patent_id"] for c in ranked], ["18184836"])

    def test_ranked_excludes_dead_even_when_scored_high(self):
        pool = SearchPool("测试问题")
        dead = _usp_raw_item("19511555", "Expired scored")
        dead["applicationMetaData"]["applicationStatusDescriptionText"] = \
            "Provisional Application Expired"
        pool.add([dead, _usp_raw_item("18184836", "Live unscored")])
        pool._by_id["19511555"]["relevance_score"] = 5
        # live candidate stays unscored
        ranked = pool.ranked(10)
        self.assertEqual([c["patent_id"] for c in ranked], ["18184836"])

    def test_ranked_excludes_design_patents(self):
        pool = SearchPool("测试问题")
        design = _usp_raw_item("19511555", "Servo amplifier")
        design["applicationMetaData"]["patentNumber"] = "D9123456"
        design["applicationMetaData"]["applicationTypeCode"] = "DES"
        pool.add([design, _usp_raw_item("18184836", "Live utility")])
        pool._by_id["19511555"]["relevance_score"] = 5
        ranked = pool.ranked(10)
        self.assertEqual([c["patent_id"] for c in ranked], ["18184836"])

    def test_ranked_excludes_dead_before_dedupe(self):
        # A dead parent must not shadow its live child in the same family.
        pool = SearchPool("测试问题")
        dead_parent = _usp_raw_item("19511555", "Family root")
        dead_parent["applicationMetaData"]["applicationStatusDescriptionText"] = \
            "Provisional Application Expired"
        pool.add([
            dead_parent,
            _usp_raw_item("18184836", "Family root", continuity_ids=["19511555"]),
        ])
        ranked = pool.ranked(10)
        self.assertEqual([c["patent_id"] for c in ranked], ["18184836"])

    def test_ranked_sorts_by_score_desc_unscored_sink(self):
        pool = SearchPool("测试问题")
        pool.add([_usp_raw_item("19511555", "A"),
                  _usp_raw_item("18184836", "B"),
                  _usp_raw_item("17222222", "C")])
        pool._by_id["19511555"]["relevance_score"] = 1
        pool._by_id["18184836"]["relevance_score"] = 5
        # C stays unscored
        # The 1-scored A is filtered from the displayed list by the
        # noise floor (default min_score=2).
        ranked = pool.ranked(10)
        self.assertEqual([c["patent_id"] for c in ranked],
                         ["18184836", "17222222"])

    def test_ranked_slices_to_limit(self):
        pool = SearchPool("测试问题")
        pool.add([_usp_raw_item(str(19500000 + i), f"T{i}") for i in range(120)])
        self.assertEqual(len(pool.ranked(100)), 100)

    def test_ranked_dedupes_family_members(self):
        pool = SearchPool("测试问题")
        pool.add([
            _usp_raw_item("18184836", "Continuation child",
                          continuity_ids=["19511555"]),
            _usp_raw_item("19511555", "Original parent"),
        ])
        pool._by_id["18184836"]["relevance_score"] = 4
        pool._by_id["19511555"]["relevance_score"] = 5
        ranked = pool.ranked(10)
        self.assertEqual([c["patent_id"] for c in ranked], ["19511555"])

    def test_ranked_dedupes_identical_titles(self):
        # 高分 (>= DEDUPE_HIGH_SCORE) 豁免标题去重 — 申请公开库同标题的
        # continuation 申请是独立技术记录, 用户宁可多看 (2026-09-01)。
        pool = SearchPool("测试问题")
        pool.add([_usp_raw_item("19511555", "Same title"),
                  _usp_raw_item("18184836", "Same title")])
        pool._by_id["19511555"]["relevance_score"] = 5
        pool._by_id["18184836"]["relevance_score"] = 4
        ranked = pool.ranked(10)
        self.assertEqual(sorted(c["patent_id"] for c in ranked),
                         ["18184836", "19511555"])

    def test_ranked_low_score_identical_title_still_deduped(self):
        # 低分同标题候选仍被高分候选去重 (高分豁免只保护高分不被砍)
        pool = SearchPool("测试问题")
        pool.add([_usp_raw_item("19511555", "Same title"),
                  _usp_raw_item("18184836", "Same title")])
        pool._by_id["19511555"]["relevance_score"] = 5
        pool._by_id["18184836"]["relevance_score"] = 2
        ranked = pool.ranked(10)
        self.assertEqual([c["patent_id"] for c in ranked], ["19511555"])


class TestRankedNoiseFilter(unittest.TestCase):
    def _pool(self):
        from sources.long_task.chat_relevance import SearchPool
        pool = SearchPool("q")
        pool.add([_usp_raw_item("19511555", "A"),
                  _usp_raw_item("18184836", "B"),
                  _usp_raw_item("17222222", "C"),
                  _usp_raw_item("16111111", "D")])
        pool._by_id["19511555"]["relevance_score"] = 0
        pool._by_id["18184836"]["relevance_score"] = 1
        pool._by_id["17222222"]["relevance_score"] = 2
        # D stays unscored
        return pool

    def test_ranked_filters_zero_and_one_scored_noise(self):
        ranked = self._pool().ranked(10)
        ids = [c["patent_id"] for c in ranked]
        self.assertNotIn("19511555", ids)
        self.assertNotIn("18184836", ids)
        self.assertIn("17222222", ids)
        self.assertIn("16111111", ids)  # unscored survives

    def test_ranked_min_score_zero_keeps_noise(self):
        ranked = self._pool().ranked(10, min_score=0)
        ids = [c["patent_id"] for c in ranked]
        self.assertIn("19511555", ids)

    def test_prune_physically_keeps_low_scores(self):
        pool = self._pool()
        pool.prune()
        self.assertIn("19511555", pool._by_id)  # pool never gates on score


class TestSearchPoolPrune(unittest.TestCase):
    def test_prune_keeps_top_max(self):
        pool = SearchPool("测试问题")
        pool.add([_usp_raw_item(str(19500000 + i), f"T{i}")
                  for i in range(POOL_MAX_CANDIDATES + 5)])
        pool.prune()
        self.assertEqual(len(pool), POOL_MAX_CANDIDATES)

    def test_prune_caps_only_never_gates(self):
        pool = SearchPool("测试问题")
        pool.add([_usp_raw_item("19511555", "keep"),
                  _usp_raw_item("18184836", "drop")])
        pool._by_id["19511555"]["relevance_score"] = 5
        pool._by_id["18184836"]["relevance_score"] = 0
        pool.prune()
        self.assertIn("19511555", pool._by_id)
        self.assertIn("18184836", pool._by_id)  # prune only caps, never gates


class TestSemanticScoreOrdering(unittest.TestCase):
    """Unscored candidates with a semantic_score (two-stage prescore)
    rank by semantic desc instead of sinking in insertion order — the
    deep end of the recall window becomes visible instead of pruned."""

    def test_unscored_semantic_orders_before_unscored_plain(self):
        from sources.long_task.candidate_metadata import _sort_key
        sem = {"patent_id": "a", "semantic_score": 0.8}
        plain = {"patent_id": "b"}
        self.assertGreater(_sort_key(sem), _sort_key(plain))

    def test_scored_rank_above_unscored_semantic(self):
        from sources.long_task.candidate_metadata import _sort_key
        scored_zero = {"patent_id": "a", "relevance_score": 0}
        sem_high = {"patent_id": "b", "semantic_score": 0.9}
        self.assertGreater(_sort_key(scored_zero), _sort_key(sem_high))

    def test_pool_ranks_unscored_by_semantic_desc(self):
        pool = SearchPool("干燥空气")
        pool.add([_usp_raw_item("19511555", "A"),
                  _usp_raw_item("18184836", "B"),
                  _usp_raw_item("10123456", "C")])
        pool._by_id["18184836"]["semantic_score"] = 0.9
        pool._by_id["10123456"]["semantic_score"] = 0.5
        ranked = pool.ranked(10)
        # no LLM scores at all — order by semantic desc, then insertion
        self.assertEqual([c["patent_id"] for c in ranked],
                         ["18184836", "10123456", "19511555"])


if __name__ == "__main__":
    unittest.main()
