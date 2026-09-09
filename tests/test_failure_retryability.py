# -*- coding: utf-8 -*-
"""需求#17: deterministic (structural) failures must not be retried.

The batch executor's catch block used to ``raise self.retry`` on ANY
exception, so a bad id / bad parameter burned the full retry budget (with
30-60s delays) before terminal failure.  ``is_retryable_failure`` splits
transient causes (remote 5xx / transport / credentials — retry can help)
from deterministic ones (unresolvable ids, parse/parameter errors — retry
cannot help), and the worker + retry endpoint honour the split.
"""
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
sys.modules.setdefault("firebase_admin", MagicMock())

import celery_worker  # noqa: E402  (imports cleanly off-server)
from sources.long_task import status_manager  # noqa: E402
from sources.long_task.status_manager import (  # noqa: E402
    is_retryable_failure,
)


class TestIsRetryableFailure(unittest.TestCase):
    def test_transient_remote_5xx(self):
        for err in ("HTTP 500 Internal Server Error",
                    "remote returned HTTP 502 Bad Gateway",
                    "upstream 503 Service Unavailable"):
            self.assertTrue(is_retryable_failure(err), err)

    def test_transient_transport(self):
        for err in ("Connection reset by peer",
                    "could not connect: connection refused",
                    "timed out after 30 seconds",
                    "network is unreachable",
                    "gateway timeout while proxying",
                    "too many requests (429)"):
            self.assertTrue(is_retryable_failure(err), err)

    def test_transient_credentials(self):
        for err in ("401 Unauthorized — token expired",
                    "403 Forbidden",
                    "OAuth access_token invalid"):
            self.assertTrue(is_retryable_failure(err), err)

    def test_structural_unresolvable_id(self):
        for err in ("CLIENT.InvalidCountryCode: PCTUS2021059064",
                    "HTTP 404 — could not resolve USPTO application for id",
                    "docdb 404 for publication WO2021/000001"):
            self.assertFalse(is_retryable_failure(err), err)

    def test_structural_parameter_and_parse(self):
        for err in ("ValueError: cannot parse patent number ''",
                    "JSONDecodeError: Expecting value",
                    "unexpected scenario 'direct_ids' with empty ids"):
            self.assertFalse(is_retryable_failure(err), err)

    def test_empty_error_not_retryable(self):
        self.assertFalse(is_retryable_failure(""))

    def test_task_type_does_not_override_transient_marker(self):
        # task_type is accepted for future per-type policy; markers win.
        self.assertTrue(is_retryable_failure("HTTP 503", "family_analysis"))
        self.assertFalse(is_retryable_failure("InvalidCountryCode",
                                              "family_analysis"))


class _FakeWorker:
    """Minimal celery task stand-in: retry() raises when the budget is gone."""

    MaxRetriesExceededError = None  # set below

    def __init__(self):
        self.retries = 0

    def retry(self, exc=None):
        self.retries += 1
        raise self.MaxRetriesExceededError()


def _make_worker():
    from celery.exceptions import MaxRetriesExceededError
    _FakeWorker.MaxRetriesExceededError = MaxRetriesExceededError
    return _FakeWorker()


class TestBatchFailureExit(unittest.TestCase):
    """celery_worker._handle_batch_failure splits deterministic vs transient."""

    def test_structural_failure_terminates_without_retry(self):
        worker = _make_worker()
        with patch.object(celery_worker, "_release_user_queue_and_notify") as rel:
            result = celery_worker._handle_batch_failure(
                worker, "lt_b1", {"user_id": "7"},
                ValueError("cannot parse patent number"))
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result.get("terminal"))
        self.assertEqual(worker.retries, 0)
        rel.assert_called_once_with("7", "lt_b1",
                                    "cannot parse patent number")

    def test_transient_failure_retries_then_raises(self):
        worker = _make_worker()
        with patch.object(celery_worker, "_release_user_queue_and_notify") as rel, \
             patch.object(celery_worker, "_pipeline_logger") as _log, \
             patch("sources.long_task.status_manager.set_task_failed") as _stf:
            with pytest.raises(worker.MaxRetriesExceededError):
                celery_worker._handle_batch_failure(
                    worker, "lt_b2", {"user_id": "7"}, RuntimeError("HTTP 503"))
        self.assertEqual(worker.retries, 1)
        rel.assert_called_once_with("7", "lt_b2", "HTTP 503")
        _stf.assert_called_once_with("lt_b2", "HTTP 503")


# ── retry endpoint honours the deterministic-failure pre-check ──────────────

def _api_client():
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from api_routes.long_task import register_long_task_routes
    import logging

    app = FastAPI()
    router = register_long_task_routes(logging.getLogger("test"), MagicMock())
    app.include_router(router)
    return TestClient(app)


def _task_row(task_type: str = "patent_analysis") -> dict:
    return {
        "session_id": "sess_1",
        "scene_id": None,
        "task_type": task_type,
        "input_params": json.dumps({
            "query": "批量分析", "lang": "zh", "user_id": "1",
        }, ensure_ascii=False),
    }


class TestRetryEndpointPreflight(unittest.TestCase):
    def test_retry_refuses_deterministic_failure(self):
        client = _api_client()
        with patch("api_routes.long_task.verify_firebase_token",
                   return_value={"uid": "1"}), \
             patch("api_routes.long_task.get_task_status",
                   return_value={"status": "failed",
                                 "error_message": "InvalidCountryCode: "
                                                  "PCTUS2021059064"}), \
             patch("sources.knowledge.knowledge.get_db_connection") as _db, \
             patch("sources.long_task.user_queue.try_start_user_task") as _start, \
             patch("api_routes.long_task._dispatch_retry_task") as _dispatch:
            _db.return_value = MagicMock()
            cur = _db.return_value.cursor.return_value.__enter__.return_value
            cur.fetchone.return_value = _task_row("patent_analysis")
            resp = client.post("/long_task/lt_old/retry")
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["detail"]["code"], "ERR_OTHER")
        _dispatch.assert_not_called()
        _start.assert_not_called()

    def test_retry_allows_transient_failure(self):
        client = _api_client()
        with patch("api_routes.long_task.verify_firebase_token",
                   return_value={"uid": "1"}), \
             patch("api_routes.long_task.get_task_status",
                   return_value={"status": "failed",
                                 "error_message": "HTTP 503 upstream"}), \
             patch("sources.knowledge.knowledge.get_db_connection") as _db, \
             patch("sources.long_task.user_queue.try_start_user_task",
                   return_value="running") as _start, \
             patch("api_routes.long_task._dispatch_retry_task") as _dispatch:
            _db.return_value = MagicMock()
            cur = _db.return_value.cursor.return_value.__enter__.return_value
            cur.fetchone.return_value = _task_row()
            resp = client.post("/long_task/lt_old/retry")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])
        _dispatch.assert_called_once()

    def test_retry_allows_no_error_record(self):
        # Old tasks without a Redis failure record keep the historical path.
        client = _api_client()
        with patch("api_routes.long_task.verify_firebase_token",
                   return_value={"uid": "1"}), \
             patch("api_routes.long_task.get_task_status",
                   return_value={"status": "failed"}), \
             patch("sources.knowledge.knowledge.get_db_connection") as _db, \
             patch("sources.long_task.user_queue.try_start_user_task",
                   return_value="running") as _start, \
             patch("api_routes.long_task._dispatch_retry_task") as _dispatch:
            _db.return_value = MagicMock()
            cur = _db.return_value.cursor.return_value.__enter__.return_value
            cur.fetchone.return_value = _task_row()
            resp = client.post("/long_task/lt_old/retry")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])
        _dispatch.assert_called_once()

    def test_retry_redis_unavailable_degrades_to_historical_path(self):
        # Our own Redis outage must never block a user retry.
        client = _api_client()
        with patch("api_routes.long_task.verify_firebase_token",
                   return_value={"uid": "1"}), \
             patch("api_routes.long_task.get_task_status",
                   side_effect=ConnectionError("no redis")), \
             patch("sources.knowledge.knowledge.get_db_connection") as _db, \
             patch("sources.long_task.user_queue.try_start_user_task",
                   return_value="running") as _start, \
             patch("api_routes.long_task._dispatch_retry_task") as _dispatch:
            _db.return_value = MagicMock()
            cur = _db.return_value.cursor.return_value.__enter__.return_value
            cur.fetchone.return_value = _task_row()
            resp = client.post("/long_task/lt_old/retry")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])
        _dispatch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
