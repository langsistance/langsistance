# -*- coding: utf-8 -*-
"""R1/R2 upload-reuse detection rules (需求#7, P4).

Repo unit scope (resolution): the prompt-bearing upload pipeline in
``api_routes.core._handle_file_upload_query`` depends on streaming, auth and
Celery — not unit-testable here.  So this file nails down the session_anchor
rules the upload path is wired to call:

- ``suggest_reuse`` (R1): same-session same-file re-upload hint within the
  10-minute sliding window, ext/case-insensitive, stale or mismatched ⇒ None.
- ``load_session_anchor``: the loader R2/route caller uses; missing key ⇒ None.

The core.py wiring correctness is verified by py_compile + the surrounding
regression suite + code-audit (see task-5-report).
"""
import json
import time

import pytest

from sources.long_task import session_anchor as sa
from sources.long_task.session_anchor import (
    load_session_anchor,
    suggest_reuse,
)


class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        v = self.store.get(key)
        return v.encode() if isinstance(v, str) else v

    def set(self, key, value, ex=None):
        self.store[key] = value
        return True


@pytest.fixture
def fake_redis(monkeypatch):
    r = FakeRedis()
    monkeypatch.setattr(sa, "_get_redis", lambda: r)
    return r


def _anchor(**kw):
    base = dict(version=1, anchor_type="file", session_id="sess_1",
                task_id="lt_1", target="文件A.pdf", target_summary="s",
                source="cnipa", result_ids=["CN1"], result_titles=None,
                updated_at=time.time())
    base.update(kw)
    return base


def _seed(redis, payload):
    redis.store[f"sess:{payload['session_id']}:anchor"] = json.dumps(
        payload, ensure_ascii=False)


def test_reuse_hint_returned_for_same_file_case_and_ext_insensitive(fake_redis):
    _seed(fake_redis, _anchor(target="文件A.pdf"))
    # ext/case-insensitive match → same logical file
    hint = suggest_reuse("sess_1", "文件A.PDF")
    assert hint and hint["task_id"] == "lt_1"
    assert hint["anchor_type"] == "file"


def test_no_hint_for_different_file(fake_redis):
    _seed(fake_redis, _anchor(target="文件A.pdf"))
    assert suggest_reuse("sess_1", "文件B.pdf") is None


def test_no_hint_when_anchor_stale(fake_redis):
    _seed(fake_redis, _anchor(target="文件A.pdf",
                              updated_at=time.time() - 3600))
    assert suggest_reuse("sess_1", "文件A.pdf", window_sec=60) is None


def test_no_hint_when_anchor_is_topic_not_file(fake_redis):
    # A number/topic anchor never yields an R1 file-reuse hint.
    _seed(fake_redis, _anchor(anchor_type="topic", target="CN114948588A",
                              source="cnipa"))
    assert suggest_reuse("sess_1", "CN114948588A") is None


def test_loader_returns_none_for_missing_or_empty(fake_redis):
    assert load_session_anchor("sess_x") is None
    # empty session id short-circuits without touching Redis
    assert load_session_anchor("") is None
