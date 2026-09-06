# -*- coding: utf-8 -*-
"""Resolvability gates at the three non-chat entry points (spec §5.3 入口 2-4 /
plan Task 4).

Families deep-analysis resubmission entry points must refuse *deterministically
unresolvable* ids BEFORE they create a task row / dispatch any executor:

- ``submit_long_task`` (family): unresolvable → 4xx with a structural detail
  ``{error, code:ERR_UNRESOLVABLE_ID, guidance}`` — nothing inserted into
  ``long_tasks``, ``execute_family_analysis.delay`` never called.
- ``retry_long_task`` (family type): unresolvable → 4xx.  Because the block sits
  above the new-task INSERT, the stored original failed task is left untouched
  and no replacement pending row is created.  Resolvable family ids resubmit and
  dispatch exactly as before; non-family types are not gated.
- USPTO ``details`` (claims/spec): a foreign / non-US id must NOT be stripped of
  its prefix and sent to ``resolve_application_number``.  It returns a guided
  error on the route's established error contract (success:false + message that
  carries the ERR_UNRESOLVABLE_ID code + failure_guidance text), resolver never
  awaited.  CN ids keep bailing to Baiten and US ids keep reaching the resolver.

Side note (deliberate deviation, see final report): ``verdict_of`` answers
"resolvable by *some* deep-analysis executor" — for it a WO public number is
resolvable, because a WO seeding an EPO family query is legit for submit/retry.
The USPTO-document detail routes only ever resolve US file-wrapper applications,
so their gate is narrower (parse-country: US passes, every other origin blocks).
Those two behaviours differ and both are anchored by tests here.
"""
import asyncio
import logging
import sys
import types
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

# api_routes.long_task / api_routes.patent_detail import sources.user.passport at
# module level (server-only deps).  Passport stubbed as the sibling suites do.
_fake_passport = types.ModuleType("sources.user.passport")
_fake_passport.verify_firebase_token = lambda hdr=None: {"uid": "1"}
sys.modules.setdefault("sources.user.passport", _fake_passport)

# `celery_worker` cannot import in the local venv (heavy server deps).  submit
# lazily does `from celery_worker import execute_family_analysis` — provide a
# fake module so we can assert `.delay` without importing the real one.
_fake_celery = types.ModuleType("celery_worker")
sys.modules.setdefault("celery_worker", _fake_celery)


def _make_clients():
    """Build and RETURN two routed TestClients (long_task + patent_detail)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api_routes.long_task import register_long_task_routes
    from api_routes.patent_detail import register_patent_detail_routes

    long_app = FastAPI()
    long_app.include_router(register_long_task_routes(
        logging.getLogger("test"), MagicMock()))
    long_client = TestClient(long_app)

    detail_app = FastAPI()
    detail_app.include_router(register_patent_detail_routes(
        MagicMock(), MagicMock()))
    detail_client = TestClient(detail_app, raise_server_exceptions=False)
    return long_client, detail_client


_LONG, _DETAIL = _make_clients()
_BODY = '{"query":"分析同族差异","patent_id":"%s","patent_source":"epo","lang":"zh"}'


def _mk_conn():
    """Connection cursor records every (sql, params) and returns no rows."""
    conn = MagicMock()
    cur = MagicMock()
    calls = []
    cur.execute.side_effect = lambda sql, *a: calls.append((sql, a))
    cur.fetchone.return_value = None
    conn.cursor.return_value.__enter__.return_value = cur
    conn._calls = calls
    conn._cur = cur
    return conn


def _rows_of(conn, needle):
    return [sql for (sql, _) in conn._calls if needle in sql]


# ────────────────────────────────── submit gate ───────────────────────────────

class TestSubmitResolvabilityGate(unittest.TestCase):
    def setUp(self):
        _reset_celery()
        self.db = _mk_conn()
        self._patch_db = patch(
            "sources.knowledge.knowledge.get_db_connection", return_value=self.db)
        self._patch_q = patch(
            "sources.long_task.user_queue.try_start_user_task",
            return_value="running")

    def tearDown(self):
        if getattr(self, "_started", False):
            self._patch_db.stop()
            self._patch_q.stop()
        _reset_celery()

    def _post(self, patent_id, scenario="family"):
        with patch("api_routes.long_task.verify_firebase_token",
                   return_value={"uid": "1"}), \
             self._patch_db, self._patch_q:
            self._started = True
            return _LONG.post("/long_task/submit", json={
                "scenario": scenario, "patent_id": patent_id,
                "patent_source": "epo", "lang": "zh",
            })

    def test_family_unresolvable_is_422_no_db_no_delay(self):
        # PCT international number (no grant path) must refuse before DB / delay.
        resp = self._post("PCTUS2021059064")
        self.assertEqual(resp.status_code, 422)
        detail = resp.json()["detail"]
        self.assertEqual(detail["code"], "ERR_UNRESOLVABLE_ID")
        self.assertNotEqual(detail["error"], "")
        self.assertIn("WO 公开号", detail["guidance"])
        # 不建任务行 / 不 delay
        self.assertEqual(_rows_of(self.db, "INSERT INTO long_tasks"), [])
        self.assertFalse(_fake_celery.execute_family_analysis.delay.called)

    def test_family_resolvable_us_still_submits(self):
        resp = self._post("19519846")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])
        self.assertEqual(len(_rows_of(self.db, "INSERT INTO long_tasks")), 1)
        self.assertTrue(getattr(_fake_celery.execute_family_analysis, "delay")
                        .called)

    def test_family_body_has_guidance_when_forced_unresolvable(self):
        with patch("api_routes.long_task.verdict_of",
                   new=lambda pid, sc="": "unresolvable"):
            resp = self._post("US12000123B2")
        self.assertEqual(resp.status_code, 422)

    def test_family_verdict_exception_falls_through(self):
        # verdict raising → gate inactive (translator upgrade must not regress).
        with patch("api_routes.long_task.verdict_of",
                   side_effect=RuntimeError("boom")):
            resp = self._post("17429113")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])

    def test_family_verdict_none_legacy_pass(self):
        # No recognisable number (verdict None) → gate says nothing → old path.
        with patch("api_routes.long_task.verdict_of",
                   new=lambda pid, sc="": None):
            resp = self._post("acme technotronics 2021 division")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])

    def test_family_real_wo_still_allowed(self):
        # Even the real translator marks a user WO public number resolvable — it
        # is a legitimate EPO family seed, so submit family must let it through.
        resp = self._post("WO2021059064")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])


# ────────────────────────────────── retry gate ────────────────────────────────

class TestRetryResolvabilityGate(unittest.TestCase):
    def setUp(self):
        self.db = _mk_conn()
        _reset_celery()

    def tearDown(self):
        _reset_celery()

    def _seed(self, task_type, patent_id):
        self.db._cur.fetchone.return_value = {
            "session_id": "sess_old", "scene_id": None,
            "task_type": task_type,
            "input_params": (_BODY % patent_id) if patent_id else '{"query":"q"}',
        }

    def _retry(self):
        with patch("api_routes.long_task.verify_firebase_token",
                   return_value={"uid": "9"}), \
             patch("sources.knowledge.knowledge.get_db_connection",
                   return_value=self.db), \
             patch("sources.long_task.user_queue.try_start_user_task",
                   return_value="running"), \
             patch("api_routes.long_task._dispatch_retry_task") as dispatch:
            resp = _LONG.post("/long_task/lt_old/retry")
        return resp, dispatch

    def test_retry_unresolvable_family_refused_no_new_row(self):
        self._seed("family_analysis", "PCTUS2021059064")
        resp, dispatch = self._retry()
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["detail"]["code"], "ERR_UNRESOLVABLE_ID")
        dispatch.assert_not_called()
        # no new pending task inserted (gate sits above the INSERT)
        self.assertEqual(_rows_of(self.db, "INSERT INTO long_tasks"), [])

    def test_retry_resolvable_family_still_dispatches(self):
        self._seed("family_analysis", "19519846")
        resp, dispatch = self._retry()
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])
        dispatch.assert_called_once()
        self.assertEqual(len(_rows_of(self.db, "INSERT INTO long_tasks")), 1)

    def test_retry_verdict_none_legacy_pass(self):
        self._seed("family_analysis", "17371126")
        with patch("api_routes.long_task.verdict_of",
                   new=lambda pid, sc="": None):
            resp, dispatch = self._retry()
        self.assertEqual(resp.status_code, 200)
        dispatch.assert_called_once()

    def test_retry_patent_analysis_unaffected(self):
        # Non-family (search) resume is A9's concern, not this gate.
        self._seed("patent_analysis", "")
        resp, dispatch = self._retry()
        self.assertEqual(resp.status_code, 200)
        dispatch.assert_called_once()


# ─────────────────────────── USPTO detail gate ────────────────────────────────

def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestUsptoDetailResolvabilityGate(unittest.TestCase):
    def test_uspto_claims_foreign_is_blocked_no_resolve(self):
        # Non-US id reaching USPTO resolution → PatentDetailError carrying the
        # code + format guidance; resolve_application_number never awaited.
        from api_routes.patent_detail import _fetch_claims, PatentDetailError
        for pid in ("WO2021059064", "PCTUS2021059064", "2021059064",
                    "EP30012345"):
            with patch(
                "sources.uspto_download.resolve_application_number",
                new=AsyncMock(side_effect=AssertionError("must not resolve")),
            ):
                with self.assertRaises(PatentDetailError) as ctx:
                    _run(_fetch_claims("uspto", pid))
            msg = str(ctx.exception)
            self.assertIn("ERR_UNRESOLVABLE_ID", msg)
            self.assertIn("WO 公开号", msg)

    def test_uspto_claims_legal_us_resolves(self):
        from api_routes.patent_detail import _fetch_claims
        claims = "1. A widget.\n2. The widget of claim 1.\n"
        with patch(
            "sources.long_task.text_extractor.get_download_url_from_doc",
            return_value="x/clm.xmlarchive",
        ), patch(
            "sources.uspto_download.fetch_document_bag",
            new=AsyncMock(return_value=[
                {"documentCode": "CLM",
                 "documentCodeDescriptionText": "Claims"}]),
        ), patch(
            "sources.uspto_download.download_document_text",
            new=AsyncMock(return_value=claims),
        ), patch(
            "sources.uspto_download.resolve_application_number",
            new=AsyncMock(return_value="19519846"),
        ) as resolve:
            result = _run(_fetch_claims("uspto", "19519846"))
        self.assertTrue(result["success"])
        resolve.assert_awaited_once()

    def test_uspto_spec_foreign_blocked_no_resolve(self):
        from api_routes.patent_detail import _fetch_spec_pdf, PatentDetailError
        with patch(
            "sources.uspto_download.resolve_application_number",
            new=AsyncMock(side_effect=AssertionError("must not resolve")),
        ):
            with self.assertRaises(PatentDetailError) as ctx:
                _run(_fetch_spec_pdf("uspto", "WO2021059064", "20210506"))
        msg = str(ctx.exception)
        self.assertIn("ERR_UNRESOLVABLE_ID", msg)
        self.assertIn("WO 公开号", msg)

    def test_route_claims_foreign_returns_success_false_message(self):
        # Route-level contract for the guided error is 200 + success:false where
        # the message exposes the code + guidance (never a 5xx; Cloudflare).
        resp = _DETAIL.get("/patent/uspto/WO2021059064/claims",
                           headers={"Authorization": "Bearer test"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertFalse(body["success"])
        self.assertIn("ERR_UNRESOLVABLE_ID", body["message"])
        self.assertIn("WO 公开号", body["message"])


def _reset_celery():
    from unittest.mock import MagicMock as _M
    _fake_celery.execute_family_analysis = _M()
    _fake_celery.execute_prosecution_analysis = _M()
    _fake_celery.execute_family_analysis.delay = _M()
    _fake_celery.execute_prosecution_analysis.delay = _M()


if __name__ == "__main__":
    unittest.main()
