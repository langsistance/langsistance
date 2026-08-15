"""Tests for chat-path relevance ranking pool (chat_relevance)."""
import asyncio
import re
import unittest

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
        self.assertEqual(provider.calls, 3)          # 3 batches of ≤100
        self.assertEqual(provider.max_inflight, 3)   # gathered, not sequential
        self.assertEqual(scored, 250)


class TestSearchPoolRanking(unittest.TestCase):
    def test_ranked_sorts_by_score_desc_unscored_sink(self):
        pool = SearchPool("测试问题")
        pool.add([_usp_raw_item("19511555", "A"),
                  _usp_raw_item("18184836", "B"),
                  _usp_raw_item("17222222", "C")])
        pool._by_id["19511555"]["relevance_score"] = 1
        pool._by_id["18184836"]["relevance_score"] = 5
        # C stays unscored
        ranked = pool.ranked(10)
        self.assertEqual([c["patent_id"] for c in ranked],
                         ["18184836", "19511555", "17222222"])

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
        pool = SearchPool("测试问题")
        pool.add([_usp_raw_item("19511555", "Same title"),
                  _usp_raw_item("18184836", "Same title")])
        pool._by_id["19511555"]["relevance_score"] = 5
        pool._by_id["18184836"]["relevance_score"] = 4
        ranked = pool.ranked(10)
        self.assertEqual([c["patent_id"] for c in ranked], ["19511555"])


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


if __name__ == "__main__":
    unittest.main()
