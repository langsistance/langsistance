"""Tests for applicant-anchored query assembly (search_query_builder).

Production observation (2026-09-03): an applicant-constrained question
("检索某公司发表的专利") was rewritten as an ordinary concept AND-ed
with the technical concepts; every multi-concept bracket query then 404'd
against applications/search while the plain company-name full-text query
returned tens of thousands of irrelevant transfer records.  These tests
pin the fixes: role-marked applicant concepts become a bind-once anchor
that the loosening ladder can never drop, and bracket queries have a
plain-space de-structured fallback.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock

from sources.long_task import search_query_builder as sqb


def _run(coro):
    return asyncio.run(coro)


class _FakeProvider:
    def __init__(self, payload):
        self._payload = payload
        self.complete_json = AsyncMock(return_value=payload)


_APPLICANT = {
    "concept": "申请人限定", "role": "applicant",
    "keywords": ["BASF SE", "BASF"], "carriers": [],
}
_TECH_A = {
    "concept": "成分甲", "role": "technical",
    "keywords": ["lactose", "alpha-lactose"],
    "carriers": ["tablet formulation"],
}
_TECH_B = {
    "concept": "成分乙", "role": "technical",
    "keywords": ["povidone"], "carriers": [],
}


class TestApplicantAnchorAssembly(unittest.TestCase):
    def test_anchor_binds_every_ladder_level(self):
        out = _run(sqb.build_search_queries(
            "请检索申请人 X 的成分甲与成分乙专利",
            _FakeProvider({"concepts": [_APPLICANT, _TECH_A, _TECH_B]})))
        queries = out["queries"]
        self.assertTrue(queries)
        for q in queries[:-1]:
            self.assertTrue(q.startswith('("BASF SE" OR BASF) AND ('),
                            q)
        # 最松一级 = 申请人锚单独存在(技术组全部放宽掉之后仍不丢申请人)。
        self.assertEqual(queries[-1], '("BASF SE" OR BASF)')
        # 放宽链(直译/载体词交错为既有语义): 全技术组 → 载体词级 → 单技术组。
        self.assertIn("povidone", queries[0])
        self.assertIn("tablet formulation", queries[1])
        self.assertNotIn("povidone", queries[-2])
        self.assertIn("lactose", queries[-2])

    def test_no_applicant_keeps_legacy_ladder(self):
        out = _run(sqb.build_search_queries(
            "成分甲与成分乙", _FakeProvider({"concepts": [_TECH_A, _TECH_B]})))
        queries = out["queries"]
        self.assertEqual(queries, [
            "(lactose OR alpha-lactose) AND (povidone)",
            '("tablet formulation")',
            "(lactose OR alpha-lactose)",
        ])

    def test_carrier_levels_stay_interleaved_under_anchor(self):
        out = _run(sqb.build_search_queries(
            "申请人 X 的成分甲",
            _FakeProvider({"concepts": [_APPLICANT, _TECH_A]})))
        queries = out["queries"]
        self.assertTrue(any("tablet formulation" in q for q in queries))
        for q in queries[:-1]:
            self.assertTrue(q.startswith('("BASF SE" OR BASF) AND '))

    def test_legacy_handwritten_queries_fallback(self):
        out = _run(sqb.build_search_queries(
            "旧 provider", _FakeProvider({
                "concepts": [], "queries": ["legacy q one", "legacy q two"]})))
        self.assertEqual(out["queries"], ["legacy q one", "legacy q two"])


class TestRenderApplicantQuery(unittest.TestCase):
    def test_phrase_default(self):
        self.assertEqual(
            sqb.render_applicant_query(["BASF SE", "BASF"]),
            '("BASF SE" OR BASF)')

    def test_field_syntax(self):
        self.assertEqual(
            sqb.render_applicant_query(["BASF SE", "BASF"],
                                       syntax="field",
                                       field="firstApplicantName"),
            'firstApplicantName:("BASF SE" OR BASF)')

    def test_space_syntax(self):
        self.assertEqual(
            sqb.render_applicant_query(["BASF SE", "BASF"], syntax="space"),
            "BASF SE BASF")

    def test_empty_keywords(self):
        self.assertEqual(sqb.render_applicant_query([]), "")
        self.assertEqual(sqb.render_applicant_query(["  "]), "")


class TestDestructureUsptoQuery(unittest.TestCase):
    def test_bracket_and_query_flattens(self):
        q = '("BASF SE" OR BASF) AND (lactose OR "alpha-lactose") AND (povidone)'
        self.assertEqual(
            sqb.destructure_uspto_query(q),
            "BASF SE BASF lactose alpha-lactose povidone")

    def test_operators_and_parens_stripped(self):
        q = "(foo AND bar) OR (baz NOT qux)"
        self.assertEqual(sqb.destructure_uspto_query(q), "foo bar baz qux")

    def test_blank(self):
        self.assertEqual(sqb.destructure_uspto_query(""), "")
        self.assertEqual(sqb.destructure_uspto_query("((()))"), "")

    def test_length_capped(self):
        long_q = " ".join(["word"] * 200)
        out = sqb.destructure_uspto_query(long_q)
        self.assertLessEqual(len(out), sqb.DEFAULT_QUERY_MAX_LENGTH)


class TestOrderConceptsByRole(unittest.TestCase):
    def test_applicant_moves_front_stable(self):
        concepts = [{"concept": "技术", "keywords": ["a"]},
                    {"concept": "公司", "role": "applicant", "keywords": ["b"]},
                    {"concept": "机构", "role": "applicant", "keywords": ["c"]},
                    {"concept": "技术2", "keywords": ["d"]}]
        ordered = sqb.order_concepts_by_role(concepts)
        roles = [sqb._concept_role(c) for c in ordered]
        self.assertEqual(roles, ["applicant", "applicant",
                                 "technical", "technical"])
        self.assertEqual([c["concept"] for c in ordered],
                         ["公司", "机构", "技术", "技术2"])

    def test_role_tolerance(self):
        self.assertEqual(
            sqb._concept_role({"role": "Assignee"}), "applicant")
        self.assertEqual(sqb._concept_role({"role": "申请人"}), "applicant")
        self.assertEqual(sqb._concept_role({}), "technical")


if __name__ == "__main__":
    unittest.main()
