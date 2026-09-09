# -*- coding: utf-8 -*-
"""W3 微信登录单元测试（决策 D3）。

code2session 传输层 mock；users 存取用内存 fake 连接；Redis 用内存 fake。
不触网、不依赖服务器依赖。
"""
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# api_routes.wechat_auth 经 Logger/track_event 无服务器依赖；wechat_login 亦无。
# 桩 passport 仅为防止未来该模块顶层 import 链触发 firebase 初始化。
_fake_passport = types.ModuleType("sources.user.passport")
_fake_passport.verify_firebase_token = lambda *a, **k: {"uid": "1"}
_fake_passport.check_and_increase_usage = lambda *a, **k: True
_fake_passport.ensure_local_user_record = lambda *a, **k: None
sys.modules.setdefault("sources.user.passport", _fake_passport)

from sources.user import wechat_login  # noqa: E402
from sources.user.wechat_login import (  # noqa: E402
    MAX_USER_ID_ATTEMPTS,
    WX_TOKEN_TTL_SECONDS,
    WechatAuthError,
    code2session,
    ensure_wechat_user_record,
    issue_wechat_token,
    verify_wechat_token,
)


# ── fake Redis ───────────────────────────────────────────────────────────────

class _FakeRedis:
    def __init__(self):
        self.store = {}

    def setex(self, key, ttl, value):
        self.store[key] = value

    def get(self, key):
        return self.store.get(key)

    def delete(self, key):
        self.store.pop(key, None)


# ── fake DB（users 表内存实现，含 W3 ALTER 后的可空语义）────────────────────

class _FakeCursor:
    def __init__(self, rows, conn):
        self._rows = rows
        self._conn = conn
        self._last = None

    def execute(self, sql, params=()):
        self._last = (sql, params)

    def fetchone(self):
        sql, params = self._last
        if sql.startswith("SELECT user_id FROM users"):
            oid = params[0]
            for row in self._conn.users:
                if (row.get("oauth_provider") == "wechat"
                        and row.get("oauth_provider_id") == oid):
                    return {"user_id": row["user_id"]}
            return None
        if sql.startswith("SELECT COUNT(*) AS cnt"):
            uid = params[0]
            return {"cnt": 1 if any(r["user_id"] == uid
                                    for r in self._conn.users) else 0}
        if sql.startswith("SELECT user_id, email FROM users"):
            return None
        return None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeConn:
    def __init__(self):
        self.users = []
        self.committed = 0

    def cursor(self):
        return _FakeCursor(None, self)

    def commit(self):
        self.committed += 1
        # 由测试侧通过 _apply_pending 显式落库以模拟真实 INSERT

    def close(self):
        pass


def _record_into(conn, sql, params):
    """SQL 解释器：只认识本模块用到的 INSERT。"""
    if sql.startswith("INSERT INTO users"):
        user_id, provider_id = params
        conn.users.append({
            "user_id": user_id,
            "oauth_provider": "wechat",
            "oauth_provider_id": provider_id,
        })
    elif sql.startswith("INSERT IGNORE INTO user_scenes"):
        pass


# 让 _FakeCursor 的 INSERT 也落库：改写 execute 捕获 INSERT 交给 conn
_orig_execute = _FakeCursor.execute


def _execute_with_insert(self, sql, params=()):
    _orig_execute(self, sql, params)
    if sql.startswith("INSERT INTO users"):
        _record_into(self._conn, sql, params)


_FakeCursor.execute = _execute_with_insert


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


def _code2session_ok(**over):
    payload = {"openid": "openid_abc123", "session_key": "sk_xx", **over}
    return _Resp(payload)


# ── tests ────────────────────────────────────────────────────────────────────

class TestCode2Session(unittest.TestCase):
    def test_missing_env_config_fails_fast(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(WechatAuthError) as ctx:
                asyncio_run(code2session("c1"))
            self.assertIn("WECHAT_APPID", str(ctx.exception))

    def test_success_returns_openid(self):
        with patch.dict("os.environ",
                        {"WECHAT_APPID": "app1", "WECHAT_SECRET": "sec1"}), \
             patch("sources.http_outbound.outbound_http") as m:
            m.get.return_value = _code2session_ok(unionid="union_1")
            payload = asyncio_run(code2session("code_x"))
        self.assertEqual(payload["openid"], "openid_abc123")
        self.assertEqual(payload["unionid"], "union_1")
        m.get.assert_called_once()
        kwargs = m.get.call_args
        self.assertEqual(kwargs[1]["purpose"], "wechat")
        self.assertIn("js_code", kwargs[1]["params"])
        self.assertEqual(kwargs[1]["params"]["js_code"], "code_x")

    def test_wechat_errorcode_raises(self):
        with patch.dict("os.environ",
                        {"WECHAT_APPID": "app1", "WECHAT_SECRET": "sec1"}), \
             patch("sources.http_outbound.outbound_http") as m:
            m.get.return_value = _Resp(
                {"errcode": 40013, "errmsg": "invalid appid"}, status=200)
            with self.assertRaises(WechatAuthError) as ctx:
                asyncio_run(code2session("code_bad"))
            self.assertIn("40013", str(ctx.exception))

    def test_transport_error_wrapped(self):
        with patch.dict("os.environ",
                        {"WECHAT_APPID": "app1", "WECHAT_SECRET": "sec1"}), \
             patch("sources.http_outbound.outbound_http") as m:
            m.get.side_effect = RuntimeError("connection refused")
            with self.assertRaises(WechatAuthError) as ctx:
                asyncio_run(code2session("c"))
            self.assertIn("transport", str(ctx.exception))


class TestEnsureWechatUserRecord(unittest.TestCase):
    def _conn_redis(self):
        return _FakeConn(), _FakeRedis()

    def test_creates_new_user_with_unionid(self):
        conn, redis = self._conn_redis()
        uid = ensure_wechat_user_record(
            "open_1", unionid="union_1",
            db_factory=lambda: conn, redis_client=redis,
            random_bits=lambda bits: 0xAAAA,
        )
        self.assertEqual(uid, 0xAAAA)
        self.assertEqual(len(conn.users), 1)
        row = conn.users[0]
        self.assertEqual(row["oauth_provider"], "wechat")
        self.assertEqual(row["oauth_provider_id"], "union_1")
        # 缓存写了两把钥匙（unionid + openid）
        self.assertEqual(redis.get("wx:uid:union_1"), uid)
        self.assertEqual(redis.get("wx:uid:open_1"), uid)

    def test_existing_by_unionid_returns_same_id(self):
        conn, redis = self._conn_redis()
        first = ensure_wechat_user_record(
            "open_1", unionid="union_1",
            db_factory=lambda: conn, redis_client=redis,
            random_bits=lambda bits: 0xBBBB,
        )
        second = ensure_wechat_user_record(
            "open_1", unionid="union_1",
            db_factory=lambda: conn, redis_client=redis,
            random_bits=lambda bits: 0xCCCC,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(conn.users), 1)

    def test_legacy_openid_record_reachable_with_unionid(self):
        conn, redis = self._conn_redis()
        ensure_wechat_user_record(
            "open_legacy", db_factory=lambda: conn, redis_client=redis,
            random_bits=lambda bits: 0x1111,
        )
        # 老记录只有 openid —— 现在带 unionid 再来，应命中同一用户
        uid = ensure_wechat_user_record(
            "open_legacy", unionid="union_new",
            db_factory=lambda: conn, redis_client=redis,
            random_bits=lambda bits: 0x2222,
        )
        self.assertEqual(uid, 0x1111)
        self.assertEqual(len(conn.users), 1)

    def test_duplicate_candidate_id_retries(self):
        conn, redis = self._conn_redis()
        conn.users.append({"user_id": 0xA1, "oauth_provider": "wechat",
                           "oauth_provider_id": "other"})
        seq = iter([0xA1, 0xB2])
        uid = ensure_wechat_user_record(
            "open_dup", db_factory=lambda: conn, redis_client=redis,
            random_bits=lambda bits: next(seq),
        )
        self.assertEqual(uid, 0xB2)


class TestWechatToken(unittest.TestCase):
    def test_issue_and_verify_roundtrip(self):
        redis = _FakeRedis()
        with patch.object(wechat_login, "_default_redis", return_value=redis):
            token = issue_wechat_token(0xCAFE, "open_1")
        self.assertTrue(token.startswith("wx_"))
        with patch.object(wechat_login, "_default_redis", return_value=redis):
            wx = verify_wechat_token(token)
        self.assertIsNotNone(wx)
        self.assertEqual(wx["user_id"], 0xCAFE)
        self.assertEqual(wx["openid"], "open_1")
        self.assertEqual(wx["provider"], "wechat")

    def test_unknown_token_returns_none(self):
        redis = _FakeRedis()
        with patch.object(wechat_login, "_default_redis", return_value=redis):
            self.assertIsNone(verify_wechat_token("wx_nope"))
            self.assertIsNone(verify_wechat_token("Bearer wx_x"))
            self.assertIsNone(verify_wechat_token(""))

    def test_redis_unavailable_fails_closed(self):
        with patch.object(wechat_login, "_default_redis",
                          side_effect=RuntimeError("no redis")):
            self.assertIsNone(verify_wechat_token("wx_x"))
            with self.assertRaises(WechatAuthError):
                issue_wechat_token(1, "open_1")


class TestAuthWechatEndpoint(unittest.TestCase):
    def _endpoint(self):
        from api_routes.wechat_auth import auth_wechat
        return auth_wechat

    def test_endpoint_returns_token_and_user_id(self):
        body = SimpleNamespace(code="wx_login_code")
        conn = _FakeConn()
        with patch.dict("os.environ",
                        {"WECHAT_APPID": "app1", "WECHAT_SECRET": "sec1"}), \
             patch("sources.http_outbound.outbound_http") as m, \
             patch.object(wechat_login, "_default_redis") as redis_factory, \
             patch.object(wechat_login, "_default_db_factory",
                          return_value=conn) as _db, \
             patch("api_routes.wechat_auth.track_event") as _ev:
            redis = _FakeRedis()
            redis_factory.return_value = redis
            m.get.return_value = _code2session_ok(unionid="union_e")
            result = asyncio_run(self._endpoint()(body))
        self.assertTrue(result["token"].startswith("wx_"))
        self.assertIsInstance(result["user_id"], int)
        _ev.assert_called_once()
        self.assertEqual(_ev.call_args[0][0], "auth:wechat")

    def test_endpoint_code2session_error_is_401(self):
        body = SimpleNamespace(code="bad_code")
        with patch.dict("os.environ",
                        {"WECHAT_APPID": "app1", "WECHAT_SECRET": "sec1"}), \
             patch("sources.http_outbound.outbound_http") as m:
            m.get.return_value = _Resp({"errcode": 40029, "errmsg": "invalid code"})
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as ctx:
                asyncio_run(self._endpoint()(body))
            self.assertEqual(ctx.value.status_code, 401)


def asyncio_run(coro):
    import asyncio
    return asyncio.run(coro)


if __name__ == "__main__":
    unittest.main()
