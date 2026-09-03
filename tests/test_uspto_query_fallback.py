"""Tests for the KB-tool bracket-404 fallback (react_tools).

Production observation (2026-09-03, second run): the applicant-anchored
ladder was now generated correctly, but every bracket/quoted query the
KB push tools sent to applications/search 404'd (8/8) while the same
words space-joined returned 200.  The helpers here give the KB-tool
invoke path (which bypasses _uspto_search_by_query) the same fallback:
empty bracket result → one plain-space re-invoke.
"""

import asyncio
import unittest
from types import SimpleNamespace

from sources.agents import react_tools


def _run(coro):
    return asyncio.run(coro)


class _FakeTool:
    """Sync invoke that mirrors the dynamic backend tool contract:
    writes the hit list onto agent._pending_raw_items, keyed by the q
    that reaches the envelope body."""

    def __init__(self, agent, hits_by_q):
        self.agent = agent
        self.hits_by_q = hits_by_q
        self.calls = []

    def invoke(self, payload):
        params = (payload or {}).get("params") or {}
        body = params.get("body") or {}
        q = body.get("q") or ""
        self.calls.append(q)
        hits = self.hits_by_q.get(q)
        if hits is not None:
            self.agent._pending_raw_items = hits
        else:
            self.agent._pending_raw_items = []
        return "ok"


def _entry(agent, hits_by_q, url="https://api.uspto.gov/api/v1/patent/applications/search"):
    return SimpleNamespace(
        tool=_FakeTool(agent, hits_by_q),
        tool_info=SimpleNamespace(
            url=url,
            params='{"body": {"q": "", "fields": []}, "method": "POST"}',
        ),
    )


def _agent():
    return SimpleNamespace(
        logger=None, _pending_raw_items=None,
        _last_user_id="u1", _last_query_id="q1",
    )


class TestHelpers(unittest.TestCase):
    def test_is_uspto_search_tool(self):
        info = SimpleNamespace(
            url="https://api.uspto.gov/api/v1/patent/applications/search")
        self.assertTrue(react_tools._is_uspto_search_tool(info))
        info = SimpleNamespace(url="https://example.com/applications/search")
        self.assertFalse(react_tools._is_uspto_search_tool(info))
        self.assertFalse(react_tools._is_uspto_search_tool(None))

    def test_flatten_bracket_query(self):
        q = '("Acme Corp" OR Acme) AND (widget OR "widget maker")'
        self.assertEqual(react_tools._flatten_query_for_uspto(q),
                         "Acme Corp Acme widget widget maker")

    def test_flatten_keeps_plain_forms(self):
        # 无括号/引号的查询保持不变 — "A AND B" 有 200 记录。
        for q in ("Acme AND widget", "lactose povidone BASF", "Acme"):
            self.assertEqual(react_tools._flatten_query_for_uspto(q), q)

    def test_flatten_blank(self):
        self.assertEqual(react_tools._flatten_query_for_uspto(""), "")
        self.assertEqual(react_tools._flatten_query_for_uspto(None), "")

    def test_with_query_replaced_body_q(self):
        payload = {"user_id": "u", "query_id": "q",
                   "params": {"body": {"q": "old", "fields": []}}}
        out = react_tools._with_query_replaced(payload, "new words")
        self.assertEqual(out["params"]["body"]["q"], "new words")
        self.assertEqual(payload["params"]["body"]["q"], "old")  # 原对象不变

    def test_with_query_replaced_nested_and_top(self):
        out = react_tools._with_query_replaced({"query": {"q": "a"}}, "b")
        self.assertEqual(out["query"]["q"], "b")
        out = react_tools._with_query_replaced({"q": "a"}, "b")
        self.assertEqual(out["q"], "b")
        out = react_tools._with_query_replaced(
            {"params": '{"q": "a", "x": 1}'}, "b")
        self.assertEqual(out["params"], '{"q": "b", "x": 1}')

    def test_with_query_replaced_unknown_shape_untouched(self):
        payload = {"nope": 1}
        self.assertEqual(react_tools._with_query_replaced(payload, "b"),
                         payload)


class TestInvokeWithFallback(unittest.TestCase):
    def test_bracket_empty_then_flat_hits(self):
        agent = _agent()
        flat_hits = [{"applicationNumberText": "1", "title": "T"}]
        entry = _entry(agent, {"Acme widget": flat_hits})
        hits = _run(react_tools._invoke_uspto_with_fallback(
            agent, entry, '("Acme") AND (widget)'))
        self.assertEqual(len(hits), 1)
        self.assertEqual(entry.tool.calls,
                         ['("Acme") AND (widget)', "Acme widget"])

    def test_first_call_hits_no_retry(self):
        agent = _agent()
        hits = [{"applicationNumberText": "1", "title": "T"}]
        entry = _entry(agent, {'("Acme")': hits})
        out = _run(react_tools._invoke_uspto_with_fallback(
            agent, entry, '("Acme")'))
        self.assertEqual(len(out), 1)
        self.assertEqual(len(entry.tool.calls), 1)

    def test_plain_query_no_retry(self):
        agent = _agent()
        entry = _entry(agent, {"Acme AND widget": []})
        out = _run(react_tools._invoke_uspto_with_fallback(
            agent, entry, "Acme AND widget"))
        self.assertEqual(out, [])
        self.assertEqual(len(entry.tool.calls), 1)

    def test_non_uspto_tool_no_retry(self):
        agent = _agent()
        entry = _entry(agent, {}, url="https://elsewhere.example/x")
        out = _run(react_tools._invoke_uspto_with_fallback(
            agent, entry, '("Acme") AND (widget)'))
        self.assertEqual(out, [])
        self.assertEqual(len(entry.tool.calls), 1)

    def test_both_empty_returns_empty(self):
        agent = _agent()
        entry = _entry(agent, {})
        out = _run(react_tools._invoke_uspto_with_fallback(
            agent, entry, '("Acme")'))
        self.assertEqual(out, [])
        self.assertEqual(len(entry.tool.calls), 2)


if __name__ == "__main__":
    unittest.main()
