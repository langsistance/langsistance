#!/usr/bin/env python3
"""
WeChat mini-program auth endpoint (决策 D3, W3).

POST /auth/wechat — body {code} (wx.login 的临时凭证)
  1. code2session（api.weixin.qq.com）→ openid/unionid
  2. ensure_wechat_user_record → users 同表同 user_id（oauth_provider='wechat'）
  3. issue_wechat_token → wx_ 前缀 token（Redis 7 天）

返回 token 供小程序以 ``Authorization: Bearer wx_...`` 调既有全部端点；
passport.verify_firebase_token 的 wx_ 分支完成校验，既有端点零改动。
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from sources.analytics import track_event
from sources.logger import Logger
from sources.user.wechat_login import (
    WechatAuthError,
    code2session,
    ensure_wechat_user_record,
    issue_wechat_token,
)

logger = Logger("backend.log")
router = APIRouter()


class WechatCodeRequest(BaseModel):
    code: str


@router.post("/auth/wechat")
async def auth_wechat(body: WechatCodeRequest):
    """Exchange a wx.login() code for a backend wx_ bearer token."""
    code = (body.code or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="code is required")
    try:
        session = await code2session(code)
        openid = session["openid"]
        unionid = session.get("unionid")
        user_id = ensure_wechat_user_record(
            openid,
            unionid=unionid,
            redis_client=None,  # ensure 内部自行取 Redis（失败降级）
            use_cache=False,  # 注册/登录即时性优先，跳过读缓存
        )
        token = issue_wechat_token(user_id, openid)
    except WechatAuthError as exc:
        logger.info(f"/auth/wechat failed: {exc}")
        raise HTTPException(status_code=401, detail=str(exc))
    logger.info(f"/auth/wechat ok: user_id={user_id}")
    track_event("auth:wechat", user_id=str(user_id),
                extra={"has_unionid": bool(unionid)})
    return {"token": token, "user_id": user_id}
