#!/usr/bin/env python3
"""
中国专利 — DI 开放平台 OAuth 回调 & Token 管理接口

路由清单
--------
POST /patent/callback    DI 平台回调推送 access_token + refresh_token
GET  /patent/token       获取当前 access_token 状态
POST /patent/refresh     手动触发 refresh_token 刷新
POST /patent/clear       清除存储的 token（调试用）
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api_routes.models import (
    PatentTokenCallback,
    PatentTokenResponse,
    PatentRefreshRequest,
    PatentRefreshResponse,
)
from sources.patent_token import (
    store_tokens,
    load_tokens,
    get_access_token,
    get_refresh_token,
    clear_tokens,
    call_di_refresh_api,
    ACCESS_TOKEN_TTL,
)
from sources.logger import Logger

logger = Logger("backend.log")
router = APIRouter()


# ── POST /patent/callback ─────────────────────────────────────────────────────


@router.post("/patent/callback")
async def patent_callback(request: Request):
    """
    DI 开放平台 OAuth 回调接口。

    DI 平台在用户授权后（以及 refresh_token 刷新后）通过此接口
    将 access_token 和 refresh_token 推送给我们的后端。

    请求体 (JSON):
        access_token  : str   — 访问令牌（有效期 7 天）
        refresh_token : str   — 刷新令牌（有效期 30 天）
        token_type    : str   — 固定 "Bearer"（可选）
        expires_in    : int   — access_token 有效期秒数（可选，默认 604800）
        scope         : str   — 授权范围（可选）
        open_id       : str   — 平台用户 open_id（可选）
        uid           : str   — 平台用户 uid（可选）

    响应:
        200: token 存储成功
        400: 请求体缺失必要字段
        500: Redis 存储失败
    """
    logger.info("Patent callback received")

    try:
        body = await request.json()
    except Exception:
        logger.warning("Patent callback: failed to parse JSON body")
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Invalid JSON body"},
        )

    logger.info(f"Patent callback body keys: {list(body.keys()) if body else 'None'}")

    access_token = body.get("access_token")
    refresh_token = body.get("refresh_token")

    if not access_token or not refresh_token:
        logger.warning(
            f"Patent callback: missing required fields. "
            f"has_access_token={bool(access_token)}, "
            f"has_refresh_token={bool(refresh_token)}"
        )
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Missing required fields: access_token and refresh_token",
            },
        )

    try:
        store = store_tokens(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=body.get("expires_in"),
            open_id=body.get("open_id"),
            uid=body.get("uid"),
        )
    except Exception as exc:
        logger.error(f"Patent callback: Redis storage failed: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Failed to store tokens: {str(exc)}",
            },
        )

    logger.info(
        f"Patent callback: tokens stored successfully. "
        f"access_expires_at={datetime.fromtimestamp(store.access_expires_at, tz=timezone.utc).isoformat()}"
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "message": "Tokens stored successfully",
            "access_token_expires_at": datetime.fromtimestamp(
                store.access_expires_at, tz=timezone.utc
            ).isoformat(),
            "refresh_token_expires_at": datetime.fromtimestamp(
                store.refresh_expires_at, tz=timezone.utc
            ).isoformat(),
        },
    )


# ── GET /patent/token ─────────────────────────────────────────────────────────


@router.get("/patent/token", response_model=PatentTokenResponse)
async def patent_token_status():
    """
    查询当前 access_token 的状态。

    返回当前存储的 access_token 及其过期时间。
    不会暴露 refresh_token 的值。

    响应:
        200: 返回 token 状态（即使已过期也会返回状态说明）
    """
    store = load_tokens()

    if store is None:
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "message": "No tokens available. Please authorize via DI platform first.",
                "access_token": None,
                "expires_at": None,
                "has_refresh_token": False,
            },
        )

    expires_at_iso = datetime.fromtimestamp(
        store.access_expires_at, tz=timezone.utc
    ).isoformat()

    if store.is_access_expired:
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Access token has expired. Use POST /patent/refresh to refresh.",
                "access_token": None,  # 不返回已过期的 token
                "expires_at": expires_at_iso,
                "has_refresh_token": True,
            },
        )

    ttl_days = store.access_ttl_seconds / 86400
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "message": f"Access token valid ({ttl_days:.1f} days remaining)",
            "access_token": store.access_token,
            "expires_at": expires_at_iso,
            "has_refresh_token": True,
        },
    )


# ── POST /patent/refresh ──────────────────────────────────────────────────────


@router.post("/patent/refresh", response_model=PatentRefreshResponse)
async def patent_refresh_token(request: PatentRefreshRequest):
    """
    使用 refresh_token 向 DI 平台申请新的 access_token。

    DI 平台的处理流程：
    1. 本接口使用存储的 refresh_token（或请求中提供的）调用 DI refresh API
    2. DI 平台验证 refresh_token 有效后，通过 /patent/callback 推送新 token
    3. 本接口返回是否成功触发刷新

    如果请求成功但 callback 尚未收到新 token，可以稍后通过
    GET /patent/token 查询新的 access_token。

    请求体 (JSON):
        refresh_token : str, optional — 如不传则使用 Redis 中存储的

    响应:
        200: 刷新请求已发送（token 将通过 callback 更新）
        400: 无 refresh_token 可用
        502: DI 平台刷新接口调用失败
    """
    refresh_token = request.refresh_token or get_refresh_token()

    if not refresh_token:
        logger.warning("Patent refresh: no refresh_token available")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "No refresh_token available. Please authorize via DI platform first.",
            },
        )

    # 检查 refresh_token 是否过期
    store = load_tokens()
    if store is not None and store.is_refresh_expired:
        logger.warning("Patent refresh: refresh_token has expired")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Refresh token has expired (30 days). Please re-authorize on DI platform.",
            },
        )

    logger.info("Calling DI refresh API...")
    result = call_di_refresh_api(refresh_token)

    if result is None:
        logger.error("Patent refresh: DI API call failed")
        return JSONResponse(
            status_code=502,
            content={
                "success": False,
                "message": "DI platform refresh API call failed. Please check credentials and network.",
            },
        )

    # DI 平台可能直接在 refresh 响应中返回新 token（备用路径）
    # 如果响应中包含 access_token，直接存储
    if "access_token" in result:
        new_access = result.get("access_token")
        new_refresh = result.get("refresh_token", refresh_token)
        expires_in = result.get("expires_in", ACCESS_TOKEN_TTL)

        try:
            stored = store_tokens(
                access_token=new_access,
                refresh_token=new_refresh,
                expires_in=expires_in,
            )
            logger.info("Patent refresh: new tokens stored from refresh API response")
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Tokens refreshed successfully",
                    "access_token": new_access,
                    "expires_in": expires_in,
                },
            )
        except Exception as exc:
            logger.error(f"Patent refresh: failed to store new tokens: {exc}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": f"Failed to store refreshed tokens: {str(exc)}",
                },
            )

    # 如果没有直接返回 token，则等待 DI 平台通过 callback 推送
    logger.info("Patent refresh: refresh triggered, waiting for callback")
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "message": "Refresh request sent to DI platform. New tokens will arrive via callback.",
        },
    )


# ── POST /patent/clear ────────────────────────────────────────────────────────


@router.post("/patent/clear")
async def patent_clear_tokens():
    """
    清除所有存储的 patent token（调试/重置用）。

    响应:
        200: 清除成功或失败信息
    """
    ok = clear_tokens()
    return JSONResponse(
        status_code=200,
        content={
            "success": ok,
            "message": "Tokens cleared" if ok else "Failed to clear tokens",
        },
    )
