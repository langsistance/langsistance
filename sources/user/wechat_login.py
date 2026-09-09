# -*- coding: utf-8 -*-
"""WeChat mini-program login (决策 D3, W3).

微信小程序端 wx.login() 得 code → 本模块经 code2session 换 openid/unionid →
ensure_wechat_user_record 在 users 表建/取同 user_id（oauth_provider='wechat',
oauth_provider_id=unionid|openid，firebase_uid/email 为 NULL —— 依赖
deploy/china/w3_users_wechat_migration.sql 的 ALTER）→ issue_wechat_token
发 wx_ 前缀 token 存 Redis 7 天。passport.verify_firebase_token 对 wx_ 前缀
分发到 verify_wechat_token，既有端点零改动。

镜像 sources/user/local_user.py 的写法（db_factory/redis_client/random_bits
可注入以便测试）。
"""
import asyncio
import json
import os
import random
import secrets
from typing import Callable, Optional

WX_TOKEN_TTL_SECONDS = 604800  # 7 天（决策 D3）
WX_TOKEN_PREFIX = "wx:token:"
WX_UID_CACHE_PREFIX = "wx:uid:"
MAX_USER_ID_ATTEMPTS = 5
CODE2SESSION_URL = "https://api.weixin.qq.com/sns/jscode2session"
DEFAULT_HTTP_TIMEOUT_SECONDS = 10.0


class WechatAuthError(RuntimeError):
    """code2session 失败 / 配置缺失 / 微信返回错误码。"""


def _wechat_credentials() -> tuple[str, str]:
    """(appid, secret) 来自环境变量；缺失即抛可读错误（fail fast）。"""
    appid = os.getenv("WECHAT_APPID", "").strip()
    secret = os.getenv("WECHAT_SECRET", "").strip()
    if not appid or not secret:
        raise WechatAuthError(
            "WECHAT_APPID/WECHAT_SECRET not set in environment"
        )
    return appid, secret


def _default_db_factory():
    from sources.knowledge.knowledge import get_db_connection

    return get_db_connection()


def _default_redis():
    from sources.knowledge.knowledge import get_redis_connection

    return get_redis_connection()


async def code2session(code: str) -> dict:
    """Exchange a wx.login() code for openid/session_key (optionally unionid).

    Returns the WeChat JSON payload (openid required by contract; unionid
    only when present).  Raises :class:`WechatAuthError` on config /
    transport / WeChat errorcode failures — never returns a partial result.
    """
    appid, secret = _wechat_credentials()
    params = {
        "appid": appid,
        "secret": secret,
        "js_code": (code or "").strip(),
        "grant_type": "authorization_code",
    }
    from sources.http_outbound import outbound_http

    try:
        resp = await asyncio.to_thread(
            outbound_http.get,
            CODE2SESSION_URL,
            purpose="wechat",
            params=params,
            timeout=DEFAULT_HTTP_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # transport / timeout — 可重试类
        raise WechatAuthError(f"code2session transport failed: {exc}") from exc
    try:
        payload = resp.json()
    except Exception as exc:
        raise WechatAuthError(
            f"code2session bad response (status={getattr(resp, 'status_code', '?')})"
        ) from exc
    if payload.get("errcode"):
        raise WechatAuthError(
            f"code2session error {payload.get('errcode')}: "
            f"{payload.get('errmsg', '')}"
        )
    if not payload.get("openid"):
        raise WechatAuthError("code2session returned no openid")
    return payload


def _cache_wx_uid(redis_client, provider_id: str, user_id, ttl: int) -> None:
    if redis_client is not None:
        try:
            redis_client.setex(
                f"{WX_UID_CACHE_PREFIX}{provider_id}", ttl, user_id)
        except Exception:
            pass  # 缓存失败不阻塞登录


def ensure_wechat_user_record(
    openid: str,
    unionid: Optional[str] = None,
    *,
    db_factory: Optional[Callable[[], object]] = None,
    redis_client=None,
    random_bits: Optional[Callable[[int], int]] = None,
    use_cache: bool = True,
    cache_ttl_seconds: int = WX_TOKEN_TTL_SECONDS,
) -> int:
    """Find-or-create the users row for a WeChat identity; returns user_id.

    Lookup order: stored provider_id = unionid (when given) first, then
    openid — a legacy record keyed by openid stays reachable once unionid
    becomes available.  New rows insert oauth_provider='wechat' with
    oauth_provider_id = unionid|openid and NULL firebase_uid/email (requires
    the W3 ALTERs); default scene (scene_id=1) is subscribed like local
    users.  Mirrors ``ensure_local_user_record`` semantics.
    """
    openid = (openid or "").strip()
    if not openid:
        raise WechatAuthError("openid is required")
    provider_id = (unionid or "").strip() or openid
    cache_keys_bare = [provider_id]
    if unionid:
        cache_keys_bare.append(openid)

    if use_cache and redis_client is not None:
        for oid in cache_keys_bare:
            try:
                cached = redis_client.get(f"{WX_UID_CACHE_PREFIX}{oid}")
            except Exception:
                cached = None
            if cached:
                return int(cached)

    make_connection = db_factory or _default_db_factory
    make_random_id = random_bits or random.getrandbits
    conn = make_connection()
    try:
        cursor = conn.cursor()
        # unionid 优先、openid 兜底（老记录可能只存了 openid）
        lookups = [provider_id]
        if unionid and provider_id != openid:
            lookups.append(openid)
        for oid in lookups:
            cursor.execute(
                "SELECT user_id FROM users "
                "WHERE oauth_provider = 'wechat' AND oauth_provider_id = %s",
                (oid,),
            )
            row = cursor.fetchone()
            if row:
                _cache_wx_uid(redis_client, oid, row["user_id"],
                              cache_ttl_seconds)
                return int(row["user_id"])

        attempts = 0
        user_id = None
        while attempts < MAX_USER_ID_ATTEMPTS:
            candidate = make_random_id(64)
            cursor.execute(
                "SELECT COUNT(*) AS cnt FROM users WHERE user_id = %s",
                (candidate,),
            )
            row = cursor.fetchone()
            if row and row["cnt"] > 0:
                attempts += 1
                continue
            user_id = candidate
            break
        if user_id is None:
            raise WechatAuthError(
                "Failed to generate unique user_id after 5 attempts")

        cursor.execute(
            "INSERT INTO users (user_id, oauth_provider, oauth_provider_id) "
            "VALUES (%s, 'wechat', %s)",
            (user_id, provider_id),
        )
        conn.commit()
        try:
            cursor.execute(
                "INSERT IGNORE INTO user_scenes (user_id, scene_id) "
                "VALUES (%s, 1)",
                (user_id,),
            )
            conn.commit()
        except Exception:
            pass  # 非致命：订阅失败不阻塞登录
        # 缓存键写入用裸标识（_cache_wx_uid 内部会加前缀）
        for oid in cache_keys_bare:
            _cache_wx_uid(redis_client, oid, user_id, cache_ttl_seconds)
        return int(user_id)
    finally:
        conn.close()


def issue_wechat_token(user_id: int, openid: str) -> str:
    """Issue a ``wx_`` bearer token bound to *user_id* (Redis, 7d TTL)."""
    token = "wx_" + secrets.token_hex(24)
    payload = json.dumps(
        {"user_id": int(user_id), "openid": (openid or "")[:128],
         "provider": "wechat"},
        ensure_ascii=False,
    )
    try:
        redis_client = _default_redis()
        redis_client.setex(
            f"{WX_TOKEN_PREFIX}{token}", WX_TOKEN_TTL_SECONDS, payload)
    except Exception as exc:
        raise WechatAuthError(f"failed to store wechat token: {exc}") from exc
    return token


def verify_wechat_token(token: str) -> Optional[dict]:
    """Verify a ``wx_`` token; returns {user_id, openid, provider} or None.

    Expired / unknown / Redis-unavailable tokens all return None (fail
    closed) so callers answer 401 uniformly.
    """
    token = (token or "").strip()
    if not token.startswith("wx_"):
        return None
    try:
        redis_client = _default_redis()
        raw = redis_client.get(f"{WX_TOKEN_PREFIX}{token}")
    except Exception:
        return None
    if not raw:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    try:
        return {
            "user_id": int(payload["user_id"]),
            "openid": str(payload.get("openid") or ""),
            "provider": "wechat",
        }
    except (KeyError, TypeError, ValueError):
        return None
