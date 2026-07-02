#!/usr/bin/env python3
"""
Firebase Auth REST API proxy endpoints.

国内浏览器无法直连 *.googleapis.com，所以由后端（部署在海外）代为转发。
前端调本路由，后端 httpx 请求 Firebase REST，返回的 idToken 是 Firebase 真签的，
原有 passport.verify_firebase_token 校验逻辑无需任何改动。
"""

import os
import httpx
from fastapi import APIRouter, HTTPException
from firebase_admin import auth as fb_admin_auth
from pydantic import BaseModel, EmailStr

from sources.crypto_utils import decrypt_password
from sources.http_outbound import outbound_http
from sources.logger import Logger
from sources.user.local_user import ensure_local_user_record
# 触发 firebase_admin.initialize_app 的调用（在 passport 模块顶层）
from sources.user import passport as passport_module  # noqa: F401

logger = Logger("backend.log")
router = APIRouter()

IDENTITY_TOOLKIT = "https://identitytoolkit.googleapis.com/v1/accounts"
SECURE_TOKEN = "https://securetoken.googleapis.com/v1/token"

HTTP_TIMEOUT = httpx.Timeout(15.0, connect=5.0)

# Map raw Firebase error codes to user-friendly messages.
# Keep in sync with frontend i18n locales (auth.errors section).
_FIREBASE_ERROR_MAP: dict[str, str] = {
    # Login errors
    "INVALID_PASSWORD": "Incorrect password. Please try again.",
    "INVALID_LOGIN_CREDENTIALS": "Invalid email or password. Please check and try again.",
    "EMAIL_NOT_FOUND": "No account found with this email. Please sign up first.",
    "USER_DISABLED": "This account has been disabled. Please contact support.",
    "TOO_MANY_ATTEMPTS_TRY_LATER": "Too many failed attempts. Please try again later or reset your password.",
    # Signup errors
    "EMAIL_EXISTS": "An account with this email already exists. Please sign in instead.",
    "WEAK_PASSWORD": "Password is too weak. Please use at least 6 characters.",
    "INVALID_EMAIL": "Please enter a valid email address.",
    "MISSING_PASSWORD": "Please enter a password.",
    "MISSING_EMAIL": "Please enter an email address.",
    # Token errors
    "EXPIRED_OOB_CODE": "This reset link has expired. Please request a new one.",
    "INVALID_OOB_CODE": "Invalid reset link. Please request a new one.",
    "INVALID_ID_TOKEN": "Session expired. Please sign in again.",
    "TOKEN_EXPIRED": "Session expired. Please sign in again.",
    "USER_NOT_FOUND": "Account not found. Please sign up first.",
}


def _friendly_auth_error(firebase_message: str) -> str:
    """Return a user-friendly message for a Firebase error code."""
    return _FIREBASE_ERROR_MAP.get(firebase_message, firebase_message)


def _api_key() -> str:
    key = os.getenv("FIREBASE_WEB_API_KEY")
    if not key:
        logger.error("FIREBASE_WEB_API_KEY not set in environment")
        raise HTTPException(status_code=500, detail="Auth service misconfigured")
    return key


class EmailPasswordRequest(BaseModel):
    email: EmailStr
    encryptedPassword: str


class RefreshRequest(BaseModel):
    refreshToken: str


class ResetPasswordRequest(BaseModel):
    email: EmailStr


async def _firebase_post(url: str, payload: dict) -> dict:
    params = {"key": _api_key()}
    try:
        resp = await outbound_http.apost(
            url,
            purpose="auth",
            params=params,
            json=payload,
            timeout=HTTP_TIMEOUT,
        )
    except httpx.HTTPError as e:
        logger.error(f"Firebase REST call failed: {e}")
        raise HTTPException(status_code=502, detail="Authentication service is temporarily unavailable. Please try again later.")

    if resp.status_code >= 400:
        try:
            err = resp.json().get("error", {})
            msg = err.get("message", "AUTH_ERROR")
        except Exception:
            msg = "AUTH_ERROR"
        logger.info(f"Firebase REST returned {resp.status_code}: {msg}")
        friendly = _friendly_auth_error(msg)
        raise HTTPException(status_code=resp.status_code, detail=friendly)

    return resp.json()


@router.post("/auth/signup")
async def auth_signup(body: EmailPasswordRequest):
    logger.info(f"/auth/signup attempt: {body.email}")
    password = decrypt_password(body.encryptedPassword)
    data = await _firebase_post(
        f"{IDENTITY_TOOLKIT}:signUp",
        {"email": body.email, "password": password, "returnSecureToken": True},
    )
    uid = data["localId"]

    # 国内邮箱收 Firebase 验证邮件不稳定（163/QQ 常常被拒），
    # 这里用 Admin SDK 直接标记为已验证，跳过 passport.py 的 email_verified 检查。
    try:
        fb_admin_auth.update_user(uid, email_verified=True)
    except Exception as e:
        logger.error(f"Failed to mark {uid} as email_verified: {e}")
        raise HTTPException(status_code=500, detail="signup post-process failed")

    # 标记 verified 后，重新签一个 idToken 让新 claims 生效
    fresh = await _firebase_post(
        f"{IDENTITY_TOOLKIT}:signInWithPassword",
        {"email": body.email, "password": password, "returnSecureToken": True},
    )
    ensure_local_user_record(
        uid,
        fresh.get("email", body.email),
        redis_client=getattr(passport_module, "redis_client", None),
        use_cache=False,
    )
    logger.info(f"/auth/signup ok: {body.email} -> {uid}")
    return {
        "idToken": fresh["idToken"],
        "refreshToken": fresh["refreshToken"],
        "expiresIn": int(fresh["expiresIn"]),
        "localId": fresh["localId"],
        "email": fresh["email"],
    }


@router.post("/auth/login")
async def auth_login(body: EmailPasswordRequest):
    logger.info(f"/auth/login attempt: {body.email}")
    password = decrypt_password(body.encryptedPassword)
    data = await _firebase_post(
        f"{IDENTITY_TOOLKIT}:signInWithPassword",
        {"email": body.email, "password": password, "returnSecureToken": True},
    )
    logger.info(f"/auth/login ok: {body.email} -> {data.get('localId')}")
    return {
        "idToken": data["idToken"],
        "refreshToken": data["refreshToken"],
        "expiresIn": int(data["expiresIn"]),
        "localId": data["localId"],
        "email": data["email"],
    }


@router.post("/auth/refresh")
async def auth_refresh(body: RefreshRequest):
    data = await _firebase_post(
        SECURE_TOKEN,
        {"grant_type": "refresh_token", "refresh_token": body.refreshToken},
    )
    return {
        "idToken": data["id_token"],
        "refreshToken": data["refresh_token"],
        "expiresIn": int(data["expires_in"]),
        "localId": data["user_id"],
    }


@router.post("/auth/reset")
async def auth_reset(body: ResetPasswordRequest):
    await _firebase_post(
        f"{IDENTITY_TOOLKIT}:sendOobCode",
        {"requestType": "PASSWORD_RESET", "email": body.email},
    )
    return {"ok": True}
