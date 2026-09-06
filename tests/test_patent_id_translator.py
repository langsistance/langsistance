"""Tests for sources/patent_id_translator — resolvable verdict + candidates.

Feature (spec §5.2): once a number is *recognised* by the parser, this layer
decides whether any downstream resolver can meaningfully consume it and hands
each office the ordered candidate strings to try.  PCT is normally
unresolvable-with-guidance (its real WO publication must come by reverse
lookup, an experimental channel that is unimplemented → fail-closed); the
naive single-WO-number guess was proven wrong on the server (2026-09-06),
so no direct-guess candidate is ever built.
"""

import asyncio
import inspect
import sys
import unittest
from unittest import mock

from sources.patent_id_translator import (
    TranslateResult,
    resolve_us_pub_number,
    translate,
    verdict_of,
    _us_docdb_candidates,
)


def _awaited(coro):
    return asyncio.run(coro)


def _mock_resolve_return(pub=None, app=None, grant=None):
    m = mock.AsyncMock()
    m.return_value = (pub, app, grant)
    return m


class TestVerdictTable(unittest.TestCase):
    """§5.2 判定表逐行 — judgement outcome by parser classification."""

    def test_us_grant_resolvable(self):
        with mock.patch("sources.patent_id_translator.resolve_us_pub_number",
                        _mock_resolve_return("US20220294065A1", "17/123456",
                                             "12506212")):
            r = _awaited(translate("US12506212"))
            self.assertIsInstance(r, TranslateResult)
            self.assertEqual(r.verdict, "resolvable")
            self.assertEqual(r.evidence, "local")
            self.assertIn("US12506212", r.candidates["epo_docdb"])

    def test_us_application_resolvable_low_when_no_grant(self):
        # reverse returns nothing → still resolvable vs USPTO, epo docdb weak.
        with mock.patch("sources.patent_id_translator.resolve_us_pub_number",
                        _mock_resolve_return()):
            r = _awaited(translate("17/027,484"))
            self.assertEqual(r.verdict, "resolvable")
            self.assertEqual(r.confidence, "low")
            self.assertTrue(r.candidates["uspto"])

    def test_cn_resolvable(self):
        r = _awaited(translate("CN117941643A", scenario="families"))
        self.assertEqual(r.verdict, "resolvable")
        self.assertTrue(r.candidates["cnipa"])

    def test_wo_user_provided_resolvable_local(self):
        r = _awaited(translate("WO2021/059064A1"))
        self.assertEqual(r.verdict, "resolvable")
        self.assertEqual(r.evidence, "local")
        self.assertIn("WO2021059064", r.candidates["epo_docdb"])

    def test_pct_unresolvable_with_guidance(self):
        r = _awaited(translate("PCTUS2021059064", reverse_lookup=False))
        self.assertEqual(r.verdict, "unresolvable")
        self.assertIn("WO 公开号", r.reason)
        self.assertIn("国家阶段", r.reason)

    def test_pct_unresolvable_never_direct_guess(self):
        # PCT→WO automatic channel exists only via reverse_lookup; without it
        # no WO docdb candidate is fabricated.
        r = _awaited(translate("PCTUS2021059064"))
        self.assertEqual(r.verdict, "unresolvable")
        self.assertEqual(r.candidates.get("epo_docdb", []), [])

    def test_bare_9plus_unsupported_unresolvable(self):
        r = _awaited(translate("2021059064"))
        self.assertEqual(r.verdict, "unresolvable")


class TestReverseLookupFlag(unittest.TestCase):
    def test_reverse_lookup_true_channel_unimplemented_failclosed(self):
        # Ask for the PCT→WO reverse channel: still not implemented → we must
        # answer unresolvable, NOT raise / fabricate.
        r = _awaited(translate("PCTUS2021059064", reverse_lookup=True))
        self.assertEqual(r.verdict, "unresolvable")

    def test_translate_api_surface(self):
        # translate is async; verdict_of synchronous (no network).
        self.assertTrue(inspect.iscoroutinefunction(translate))
        self.assertFalse(inspect.iscoroutinefunction(verdict_of))


class TestInternalExceptionPropagates(unittest.TestCase):
    def test_translate_resolve_exception_propagates(self):
        # Fail-open contract (spec §7): translator exceptions must NOT be
        # coerced into a verdict — they surface for the caller to pass through.
        m = mock.AsyncMock(side_effect=RuntimeError("boom"))
        with mock.patch("sources.patent_id_translator.resolve_us_pub_number", m):
            with self.assertRaises(RuntimeError):
                _awaited(translate("US12506212"))


class TestVerdictOfSync(unittest.TestCase):
    """verdict_of matches translate (local sample), never touches network."""
    def test_consistent_with_translate(self):
        cases = {
            "US12506212": "resolvable",
            "17/027,484": "resolvable",
            "CN117941643A": "resolvable",
            "WO2021/059064A1": "resolvable",
            "PCTUS2021059064": "unresolvable",
            "2021059064": "unresolvable",
        }
        for pid, expected in cases.items():
            with self.subTest(pid=pid):
                self.assertEqual(verdict_of(pid, "families"), expected)

    def test_verdict_of_pure_local_no_network_calls(self):
        # verdict_of must not hit resolve_us_pub_number (network).
        with mock.patch("sources.patent_id_translator.resolve_us_pub_number") as m:
            self.assertEqual(verdict_of("US12506212"), "resolvable")
            m.assert_not_called()


class TestUsDocdbOrdering(unittest.TestCase):
    """families Phase 0 candidate ordering must be preserved on extraction:
    [orig, US{grant}, no-kind pub, full pub, US.xxx.kind, app_text]."""
    def test_ordering_matches_families_phase0(self):
        ordered = _us_docdb_candidates(
            "17027484", "US20220294065A1", "12506212", "17/027484")
        self.assertEqual(ordered, [
            "17027484",
            "US12506212",
            "US20220294065",
            "US20220294065A1",
            "US.20220294065.A1",
            "17/027484",
        ])

    def test_no_duplicates(self):
        ordered = _us_docdb_candidates(
            "17027484", "US20220294065A1", "12506212", "12506212")
        self.assertEqual(len(ordered), len(set(ordered)))


if __name__ == "__main__":
    unittest.main()
