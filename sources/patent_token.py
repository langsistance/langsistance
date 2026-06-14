#!/usr/bin/env python3
"""
DI 开放平台 — 中国专利 access_token 管理服务

Token 生命周期
--------------
- access_token  有效期 7 个自然日（604800 秒）
- refresh_token 有效期 30 个自然日（2592000 秒）
- DI 平台通过回调接口推送 token，本模块负责 Redis 持久化与主动刷新。

Redis Key 设计
---------------
- patent:access_token   — 当前有效的 access_token
- patent:refresh_token  — 当前有效的 refresh_token
- patent:token_expires_at — access_token 过期时间戳（epoch）
- patent:refresh_expires_at — refresh_token 过期时间戳（epoch）
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import redis

from sources.logger import Logger

logger = Logger("patent_token.log")

# ── 常量 ──────────────────────────────────────────────────────────────────────

ACCESS_TOKEN_TTL = 7 * 24 * 3600    # 7 天
REFRESH_TOKEN_TTL = 30 * 24 * 3600  # 30 天
ACCESS_TOKEN_REFRESH_BUFFER = 3600  # 提前 1 小时刷新

import threading
_patent_refresh_lock = threading.Lock()

REDIS_KEY_ACCESS = "patent:access_token"
REDIS_KEY_REFRESH = "patent:refresh_token"
REDIS_KEY_ACCESS_EXPIRES = "patent:token_expires_at"
REDIS_KEY_REFRESH_EXPIRES = "patent:refresh_expires_at"

# ── Redis 连接 ────────────────────────────────────────────────────────────────


def _get_redis() -> redis.Redis:
    """创建 Redis 连接（读取环境变量 REDIS_HOST / REDIS_PORT）"""
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    return redis.Redis(
        host=host,
        port=port,
        decode_responses=True,
        socket_connect_timeout=10,
        socket_timeout=10,
    )


# ── 数据模型 ──────────────────────────────────────────────────────────────────


@dataclass
class PatentTokenStore:
    access_token: str
    refresh_token: str
    access_expires_at: float  # epoch 秒
    refresh_expires_at: float  # epoch 秒
    open_id: Optional[str] = None
    uid: Optional[str] = None

    @property
    def is_access_expired(self) -> bool:
        return time.time() >= self.access_expires_at

    @property
    def is_refresh_expired(self) -> bool:
        return time.time() >= self.refresh_expires_at

    @property
    def access_ttl_seconds(self) -> float:
        return max(0.0, self.access_expires_at - time.time())


# ── 存储操作 ──────────────────────────────────────────────────────────────────


def store_tokens(
    access_token: str,
    refresh_token: str,
    expires_in: Optional[int] = None,
    open_id: Optional[str] = None,
    uid: Optional[str] = None,
) -> PatentTokenStore:
    """
    将 DI 平台回调推送的 token 持久化到 Redis。

    Parameters
    ----------
    access_token : str
        DI 平台下发的 access_token
    refresh_token : str
        DI 平台下发的 refresh_token
    expires_in : int, optional
        access_token 有效期（秒），默认 7 天
    open_id : str, optional
        平台用户 open_id
    uid : str, optional
        平台用户 uid

    Returns
    -------
    PatentTokenStore
        已存储的 token 对象
    """
    if expires_in is None:
        expires_in = ACCESS_TOKEN_TTL

    now = time.time()
    access_expires_at = now + expires_in
    refresh_expires_at = now + REFRESH_TOKEN_TTL

    r = _get_redis()
    try:
        r.setex(REDIS_KEY_ACCESS, expires_in, access_token)
        r.setex(REDIS_KEY_REFRESH, REFRESH_TOKEN_TTL, refresh_token)
        r.setex(REDIS_KEY_ACCESS_EXPIRES, expires_in, str(access_expires_at))
        r.setex(REDIS_KEY_REFRESH_EXPIRES, REFRESH_TOKEN_TTL, str(refresh_expires_at))

        if open_id:
            r.setex("patent:open_id", REFRESH_TOKEN_TTL, open_id)
        if uid:
            r.setex("patent:uid", REFRESH_TOKEN_TTL, uid)

        logger.info(
            f"Token stored: access_expires_at={access_expires_at}, "
            f"refresh_expires_at={refresh_expires_at}"
        )
    except Exception as exc:
        logger.error(f"Failed to store tokens in Redis: {exc}")
        raise
    finally:
        r.close()

    return PatentTokenStore(
        access_token=access_token,
        refresh_token=refresh_token,
        access_expires_at=access_expires_at,
        refresh_expires_at=refresh_expires_at,
        open_id=open_id,
        uid=uid,
    )


def load_tokens() -> Optional[PatentTokenStore]:
    """
    从 Redis 读取当前存储的 token。

    Returns
    -------
    PatentTokenStore or None
        如果 token 不存在或已过期则返回 None
    """
    r = _get_redis()
    try:
        access_token = r.get(REDIS_KEY_ACCESS)
        refresh_token = r.get(REDIS_KEY_REFRESH)
        access_exp_str = r.get(REDIS_KEY_ACCESS_EXPIRES)
        refresh_exp_str = r.get(REDIS_KEY_REFRESH_EXPIRES)
    except Exception as exc:
        logger.error(f"Failed to read tokens from Redis: {exc}")
        return None
    finally:
        r.close()

    if not access_token or not refresh_token:
        logger.info("No tokens found in Redis")
        return None

    access_expires_at = float(access_exp_str) if access_exp_str else 0.0
    refresh_expires_at = float(refresh_exp_str) if refresh_exp_str else 0.0

    # 检查 refresh_token 是否已过期（完全失效）
    if time.time() >= refresh_expires_at:
        logger.warning("Refresh token has expired, tokens are unusable")
        return None

    store = PatentTokenStore(
        access_token=access_token,
        refresh_token=refresh_token,
        access_expires_at=access_expires_at,
        refresh_expires_at=refresh_expires_at,
    )

    if store.is_access_expired:
        logger.info("Access token expired but refresh token still valid")

    return store


def get_access_token() -> Optional[str]:
    """
    获取当前有效的 access_token。

    Returns
    -------
    str or None
    """
    store = load_tokens()
    if store is None:
        return None
    return store.access_token


def get_refresh_token() -> Optional[str]:
    """
    获取当前存储的 refresh_token（可能已过期）。
    """
    r = _get_redis()
    try:
        return r.get(REDIS_KEY_REFRESH)
    except Exception:
        return None
    finally:
        r.close()


def ensure_valid_access_token() -> Optional[str]:
    """
    获取有效的 access_token，如果过期或接近过期则自动刷新。

    刷新流程：
    1. 检查当前 access_token 是否有效（预留 1 小时缓冲）
    2. 如过期/接近过期，使用 refresh_token 调用 DI 平台刷新接口
    3. 刷新成功后，将新 token 覆盖写入 Redis
    4. 返回有效的 access_token

    线程安全：使用锁防止并发刷新。

    Returns
    -------
    str or None
        有效的 access_token；如果没有任何 token 且无法刷新则返回 None
    """
    store = load_tokens()

    # 没有任何 token
    if store is None:
        logger.info("No patent tokens available in Redis")
        return None

    # access_token 仍在有效期内（含缓冲），直接返回
    if store.access_ttl_seconds > ACCESS_TOKEN_REFRESH_BUFFER:
        return store.access_token

    # 需要刷新
    logger.info(
        f"Patent access_token expired or near expiry "
        f"(ttl={store.access_ttl_seconds:.0f}s), triggering refresh..."
    )

    if store.is_refresh_expired:
        logger.warning("Patent refresh_token also expired, cannot refresh")
        # 返回旧 token，让调用方自行处理
        return store.access_token if not store.is_access_expired else None

    # 使用锁防止并发刷新
    with _patent_refresh_lock:
        # 双重检查：获取锁后再次确认 token 状态
        # （可能在等待锁期间已被其他线程刷新）
        store_after_lock = load_tokens()
        if store_after_lock is not None and store_after_lock.access_ttl_seconds > ACCESS_TOKEN_REFRESH_BUFFER:
            logger.info("Patent token already refreshed by another thread")
            return store_after_lock.access_token

        result = call_di_refresh_api(store.refresh_token)

        if result is None:
            logger.error("Patent token refresh API call failed")
            # 返回旧 access_token（可能已过期，但别无选择）
            return store.access_token

        # 刷新响应中包含新 token，直接存储
        if "access_token" in result:
            new_access = result.get("access_token")
            new_refresh = result.get("refresh_token", store.refresh_token)
            expires_in = result.get("expires_in", ACCESS_TOKEN_TTL)

            try:
                stored = store_tokens(
                    access_token=new_access,
                    refresh_token=new_refresh,
                    expires_in=expires_in,
                )
                logger.info(
                    f"Patent tokens refreshed successfully, "
                    f"new access_token expires in {expires_in}s"
                )
                return stored.access_token
            except Exception as exc:
                logger.error(f"Failed to store refreshed tokens: {exc}")
                # 存储失败但拿到了新 token，仍然返回
                return new_access

        # 响应中不包含 token（走 callback 路径）
        logger.info("Patent refresh triggered (callback path), using existing token")
        return store.access_token


def clear_tokens() -> bool:
    """清除所有存储的 token（用于 debug / 重置）。"""
    r = _get_redis()
    try:
        keys = [
            REDIS_KEY_ACCESS,
            REDIS_KEY_REFRESH,
            REDIS_KEY_ACCESS_EXPIRES,
            REDIS_KEY_REFRESH_EXPIRES,
            "patent:open_id",
            "patent:uid",
        ]
        r.delete(*keys)
        logger.info("All patent tokens cleared from Redis")
        return True
    except Exception as exc:
        logger.error(f"Failed to clear tokens: {exc}")
        return False
    finally:
        r.close()


# ── DI 平台 API 调用 ──────────────────────────────────────────────────────────


def _get_oauth_config() -> dict:
    """
    读取 DI 平台 OAuth 配置。

    优先级：环境变量 > config.ini
    """
    import configparser

    client_id = os.getenv("PATENT_CLIENT_ID", "")
    client_secret = os.getenv("PATENT_CLIENT_SECRET", "")
    refresh_url = os.getenv("PATENT_REFRESH_URL", "")

    # 环境变量未配置时回退到 config.ini
    if not client_id or not client_secret or not refresh_url:
        try:
            cfg = configparser.ConfigParser()
            cfg.read("config.ini")
            if not client_id:
                client_id = cfg.get("PATENT", "client_id", fallback="")
            if not client_secret:
                client_secret = cfg.get("PATENT", "client_secret", fallback="")
            if not refresh_url:
                refresh_url = cfg.get(
                    "PATENT", "refresh_url",
                    fallback="https://open.zldsj.com/oauth2/token",
                )
        except Exception:
            pass

    if not refresh_url:
        refresh_url = "https://open.zldsj.com/oauth2/token"

    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_url": refresh_url,
    }


def call_di_refresh_api(refresh_token: str) -> Optional[dict]:
    """
    调用 DI 开放平台的 refresh_token 接口换取新的 access_token。

    DI 平台预期在刷新成功后通过回调接口再次推送新 token，
    此方法主动调用刷新接口作为触发。

    Parameters
    ----------
    refresh_token : str
        当前持有的 refresh_token

    Returns
    -------
    dict or None
        DI 平台返回的 token 数据；失败返回 None
    """
    import urllib.request
    import urllib.parse
    import json

    cfg = _get_oauth_config()
    if not cfg["client_id"] or not cfg["client_secret"]:
        logger.error("PATENT_CLIENT_ID or PATENT_CLIENT_SECRET not configured")
        return None

    params = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "client_id": cfg["client_id"],
        "client_secret": cfg["client_secret"],
        "refresh_token": refresh_token,
    }).encode("utf-8")

    req = urllib.request.Request(
        cfg["refresh_url"],
        data=params,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            logger.info(f"DI refresh API response: {json.dumps(body, ensure_ascii=False)}")
            return body
    except urllib.error.HTTPError as exc:
        logger.error(f"DI refresh API HTTP error: {exc.code} {exc.reason}")
        return None
    except Exception as exc:
        logger.error(f"DI refresh API call failed: {exc}")
        return None
