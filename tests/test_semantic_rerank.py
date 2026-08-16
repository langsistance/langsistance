"""Tests for semantic reranking (plan A): pure fusion math.

The embedding call itself is provider-dependent and exercised through
mocked embed_texts in the wiring tests; everything here is pure.
"""
import unittest

from sources.long_task.semantic_rerank import cosine_similarity, fuse_ranking


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors_score_one(self):
        self.assertAlmostEqual(cosine_similarity([1, 2, 3], [1, 2, 3]), 1.0)

    def test_orthogonal_vectors_score_zero(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_opposite_vectors_score_negative_one(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1.0)

    def test_degenerate_inputs_score_zero(self):
        self.assertEqual(cosine_similarity([], [1]), 0.0)
        self.assertEqual(cosine_similarity([1], [1, 2]), 0.0)
        self.assertEqual(cosine_similarity([0, 0], [1, 0]), 0.0)
        self.assertEqual(cosine_similarity(None, [1]), 0.0)


def _cand(pid, score, title="t"):
    c = {"patent_id": pid, "title": title}
    if score is not None:
        c["relevance_score"] = score
    return c


class TestFuseRanking(unittest.TestCase):
    def test_semantic_flips_llm_order_when_strong(self):
        cands = [_cand("a", 5), _cand("b", 1)]
        # b is semantically far closer; alpha=0.3 → semantics dominate
        fused = fuse_ranking(cands, [0.2, 0.95], alpha=0.3)
        self.assertEqual([c["patent_id"] for c in fused], ["b", "a"])

    def test_llm_order_kept_when_semantics_agree(self):
        cands = [_cand("a", 5), _cand("b", 1)]
        fused = fuse_ranking(cands, [0.9, 0.1], alpha=0.5)
        self.assertEqual([c["patent_id"] for c in fused], ["a", "b"])

    def test_alpha_one_ignores_semantics(self):
        cands = [_cand("a", 5), _cand("b", 1)]
        fused = fuse_ranking(cands, [0.1, 0.9], alpha=1.0)
        self.assertEqual([c["patent_id"] for c in fused], ["a", "b"])

    def test_empty_semantic_scores_keep_original_order(self):
        cands = [_cand("a", 5), _cand("b", 1)]
        fused = fuse_ranking(cands, [], alpha=0.5)
        self.assertEqual([c["patent_id"] for c in fused], ["a", "b"])

    def test_unscored_candidates_treated_as_zero(self):
        cands = [_cand("a", None), _cand("b", 3)]
        fused = fuse_ranking(cands, [0.9, 0.1], alpha=0.5)
        self.assertEqual([c["patent_id"] for c in fused], ["b", "a"])

    def test_equal_llm_scores_semantics_decide(self):
        cands = [_cand("a", 2), _cand("b", 2)]
        fused = fuse_ranking(cands, [0.1, 0.9], alpha=0.5)
        self.assertEqual([c["patent_id"] for c in fused], ["b", "a"])

    def test_input_order_stable_on_ties(self):
        cands = [_cand("a", 5), _cand("b", 5)]
        fused = fuse_ranking(cands, [0.5, 0.5], alpha=0.5)
        self.assertEqual([c["patent_id"] for c in fused], ["a", "b"])

    def test_semantic_scores_length_mismatch_keeps_order(self):
        cands = [_cand("a", 5), _cand("b", 1)]
        fused = fuse_ranking(cands, [0.9], alpha=0.5)
        self.assertEqual([c["patent_id"] for c in fused], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
