"""Baiten (佰腾) open platform — HTTP API client for patent legal-status queries.

Reverse-engineered from ``cube-sdk-ext-1.0.jar`` (2018-09-13 build) and
live-tested against the current gateway.

Architecture
------------
The SDK is a fork of the **Alibaba TOP (Taobao Open Platform) SDK**.  The
current API uses **path-based routing** (not the ``method`` POST parameter):

- Gateway: ``http://open.baiten.cn/router``
- Method routing: ``/router{apiMethodName}`` (e.g. ``/router/openService/law``)
- The ``method`` param is **included in the signature** but **omitted from the
  POST body** (it's expressed as the URL path instead).

Signature: ``MD5(appSecret + sorted_key_value_concat + appSecret)`` → uppercase hex
(sorted over ALL params including method, app_key, timestamp, v, sign_method, format)

API methods
-----------
- ``/openService/law`` — patent legal-status query (CubeOpenLawRequest →
  CubePatentGetLawResponse).  Returns **snake_case** keys from the router.
- ``/extService/law`` — older variant (CubePatentLawRequest), may be retired.

Response model
--------------
- ``patent_laws`` (list): basic legal-status events {notice_date, law_state, law_info}
- ``patent_law_declare`` / ``patentLawDeclare_list`` (list): reexamination /
  invalidation decisions: {appNum, inTitle, declareNum, declareDate,
  reDeclarePerson, ineffectivePerson, patentee, mainExamingPerson, chargeMan,
  examingPerson, mainClassNum, lawBase, declareProintvarhcar, **fullText**, ...}
- ``patent_laws_count`` (map): status event counts by category
- ``custom_info`` (object): customs record filing info

Reference
---------
- SDK: cube-sdk-ext-1.0.jar (decompiled from bytecode)
- Docs: https://open.baiten.cn/interface/lawInfos
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone, timedelta
from typing import Any

import httpx

from sources.logger import Logger

_logger = Logger("baiten_client.log")

# ── Constants (from SDK bytecode and live testing) ─────────────────────────

BAITEN_GATEWAY_URL = "http://open.baiten.cn/router"
BAITEN_REQUEST_TIMEOUT = 30

# SDK version (from Constants.SDK_VERSION)
_SDK_VERSION = "top-sdk-java-20140926"

# Date format (from Constants.DATE_TIME_FORMAT)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Charset (from Constants.CHARSET_UTF8)
_CHARSET = "UTF-8"

# API method name (from CubeOpenLawRequest.getApiMethodName())
_API_METHOD_LAW = "/openService/law"

# Sign method (from Constants.SIGN_METHOD_MD5)
_SIGN_METHOD = "md5"

# Response format (from Constants.FORMAT_JSON)
_FORMAT = "json"

# API version
_API_VERSION = "1.0"

# lawCategory enum values (from CubePatentLawRequest + LawCategory)
_LAW_CATEGORY = {
    "FLZT": "法律状态",
    "XKBA": "许可备案",
    "QLZY": "权利转移",
    "ZYQL": "权利质押",
    "FSWX": "复审无效",
    "HGBA": "海关备案",
}


# ── Error types ─────────────────────────────────────────────────────────────────


class BaitenError(Exception):
    """Base error for Baiten API failures."""


class BaitenAuthError(BaitenError):
    """Authentication failed (bad APP_KEY / APP_SECRET)."""


class BaitenAPIError(BaitenError):
    """API returned a non-success response."""


# ── Client ──────────────────────────────────────────────────────────────────────


class BaitenClient:
    """Async HTTP client for the Baiten (佰腾) open platform.

    Uses TOP-style MD5 signature with path-based method routing.

    Usage::

        client = BaitenClient(app_key="...", app_secret="...")

        # Reexamination / invalidation (复审无效)
        result = await client.query_law_infos("CN201080053868", "FSWX")

        # General legal status timeline (法律状态)
        result = await client.query_law_infos("CN201080053868", "FLZT")
    """

    def __init__(
        self,
        app_key: str,
        app_secret: str,
        gateway_url: str = BAITEN_GATEWAY_URL,
    ) -> None:
        if not app_key or not app_secret:
            raise BaitenAuthError(
                "Baiten app_key and app_secret are required. "
                "Set BAITEN_APP_KEY / BAITEN_APP_SECRET env vars."
            )
        self._app_key = app_key
        self._app_secret = app_secret
        self._gateway_url = gateway_url.rstrip("/")

    # ── Public API ────────────────────────────────────────────────────────────

    async def query_law_infos(
        self, app_num: str, law_category: str = "FSWX",
    ) -> dict[str, Any]:
        """Query patent legal-status / reexamination data.

        Calls ``/openService/law`` (CubeOpenLawRequest) via
        ``POST {gateway}/openService/law``.

        Args:
            app_num: CN application number (e.g. ``"CN201080053868"``).
            law_category: One of ``FSWX`` (复审无效), ``FLZT`` (法律状态),
                ``XKBA`` (许可备案), ``QLZY`` (权利转移), ``ZYQL`` (权利质押),
                ``HGBA`` (海关备案).

        Returns:
            Dict with keys (snake_case from router):
            - ``patent_laws``: list of {notice_date, law_state, law_info}
            - ``patent_law_declare``: single declare record (may be null)
            - ``patentLawDeclare_list``: list of reexamination decisions
              with fields: appNum, inTitle, declareNum, declareDate,
              reDeclarePerson, ineffectivePerson, patentee, mainExamingPerson,
              chargeMan, examingPerson, mainClassNum, lawBase,
              declareProintvarhcar, **fullText**, etc.
            - ``patent_laws_count``: map of status counts by category
            - ``custom_info``: customs record filing info
        """
        if law_category not in _LAW_CATEGORY:
            raise BaitenAPIError(
                f"Unknown law_category: {law_category!r}. "
                f"Valid values: {list(_LAW_CATEGORY)}"
            )

        # Build TOP-style params (method included in signature)
        all_params = self._build_top_params(law_category, app_num)

        # POST body excludes 'method' — it's expressed in the URL path
        api_method = all_params.pop("method")
        api_method_path = api_method  # e.g. "/openService/law"
        if api_method_path.startswith("/"):
            api_method_path = api_method_path[1:]  # "openService/law"

        form_data = self._encode_params(all_params)

        url = f"{self._gateway_url}/{api_method_path}"

        _logger.info(
            f"baiten_request — url={url}, "
            f"lawCategory={law_category}, appNum={app_num}"
        )

        async with httpx.AsyncClient(timeout=BAITEN_REQUEST_TIMEOUT) as client:
            response = await client.post(
                url,
                content=form_data,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
                    "User-Agent": "top-sdk-java",
                    "Accept": "application/json",
                },
            )

        if response.status_code != 200:
            raise BaitenAPIError(
                f"Baiten API HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )

        body = response.json()
        if not isinstance(body, dict):
            raise BaitenAPIError(
                f"Baiten returned non-JSON response: {response.text[:300]}"
            )

        # Check error response
        error_code = body.get("code", "")
        if error_code and str(error_code) != "200":
            msg = body.get("msg", "Unknown error")
            raise BaitenAPIError(
                f"Baiten API error code={error_code}: {msg}"
            )

        _logger.info(
            f"baiten_response — lawCategory={law_category}, "
            f"appNum={app_num}, "
            f"has_patent_laws={bool(body.get('patent_laws'))}, "
            f"has_declareList={bool(body.get('patentLawDeclare_list'))}"
        )
        return body

    # ── Convenience: SIPOP-compatible wrappers ────────────────────────────────

    async def query_patent_review(
        self, app_num: str, country: str = "CN",
    ) -> list[dict[str, Any]]:
        """SIPOP-compatible — fetch reexamination/invalidation decisions.

        Calls ``query_law_infos(app_num, "FSWX")`` and extracts
        ``patentLawDeclare_list``.

        Each returned dict contains fields like:
        declareNum, declareDate, reDeclarePerson, ineffectivePerson,
        patentee, lawBase, mainExamingPerson, **fullText** (the complete
        decision text), etc.
        """
        data = await self.query_law_infos(app_num, "FSWX")
        declare_list = data.get("patentLawDeclare_list")
        if declare_list and isinstance(declare_list, list):
            return declare_list
        # Fallback: try patent_laws (legal status events)
        patent_laws = data.get("patent_laws")
        if patent_laws and isinstance(patent_laws, list):
            return patent_laws
        return []

    async def query_law_state(
        self, app_num: str,
    ) -> dict[str, Any]:
        """SIPOP-compatible — fetch current legal status summary.

        Calls ``query_law_infos(app_num, "FLZT")``.
        """
        data = await self.query_law_infos(app_num, "FLZT")
        patent_laws = data.get("patent_laws", [])
        if patent_laws and len(patent_laws) > 0:
            latest = patent_laws[0]
            return {
                "date": latest.get("notice_date", ""),
                "lawStatus": latest.get("law_state", ""),
                "lawStatusCode": "",
            }
        return {}

    async def query_legal_state_timeline(
        self, app_num: str, country: str = "CN",
    ) -> list[dict[str, Any]]:
        """SIPOP-compatible — fetch full legal-status timeline.

        Calls ``query_law_infos(app_num, "FLZT")``.
        Returns list of {notice_date, law_state, law_info}.
        """
        data = await self.query_law_infos(app_num, "FLZT")
        patent_laws = data.get("patent_laws")
        if patent_laws and isinstance(patent_laws, list):
            normalized = []
            for entry in patent_laws:
                normalized.append({
                    "date": entry.get("notice_date", ""),
                    "lawStatusCode": "",
                    "lawStatus": entry.get("law_state", ""),
                    "lawStatusDetail": entry.get("law_info", ""),
                })
            return normalized
        return []

    # ── Internal: TOP-style request building ──────────────────────────────────

    def _build_top_params(
        self, law_category: str, app_num: str,
    ) -> dict[str, str]:
        """Build the complete parameter map for a TOP-style API call.

        Combines protocol-mandatory params (method, app_key, timestamp, v,
        sign_method, format) with application-specific params (app_num,
        law_category), then computes the MD5 signature over ALL params.

        The ``method`` param is returned for use in the URL path; the caller
        removes it from the POST body.
        """
        # Beijing time (GMT+8) — from Constants.DATE_TIMEZONE
        now = datetime.now(timezone(timedelta(hours=8)))
        timestamp = now.strftime(_DATE_FORMAT)

        # Protocol-mandatory parameters (from DefaultCubeClient)
        protocol = {
            "method": _API_METHOD_LAW,
            "app_key": self._app_key,
            "timestamp": timestamp,
            "v": _API_VERSION,
            "sign_method": _SIGN_METHOD,
            "format": _FORMAT,
        }

        # Application parameters (from CubePatentLawRequest.getTextParams())
        application = {
            "app_num": app_num,
            "law_category": law_category,
        }

        # Merge all params for signing
        all_params = {**protocol, **application}
        sign = self._top_sign(all_params, self._app_secret)
        all_params["sign"] = sign

        return all_params

    # ── Internal: TOP signature algorithm ─────────────────────────────────────

    @staticmethod
    def _top_sign(params: dict[str, str], secret: str) -> str:
        """Compute the Alibaba TOP-style MD5 signature.

        Algorithm (from CubeUtils.signTopRequest):
        1. Sort parameter keys alphabetically.
        2. Concatenate key+value for each non-empty value (no separator).
        3. Sign: ``MD5(secret + concatenated + secret)`` → uppercase hex.

        This matches the bytecode of:
        - ``CubeUtils.signTopRequest()``
        - ``Coder.encryptMD5()``
        - ``Coder.byte2hex()``
        """
        sorted_keys = sorted(params.keys())
        query_parts = []
        for k in sorted_keys:
            v = params[k]
            if v:  # StringUtils.areNotEmpty check
                query_parts.append(f"{k}{v}")
        query = "".join(query_parts)

        raw = f"{secret}{query}{secret}"
        md5 = hashlib.md5(raw.encode(_CHARSET))

        # byte2hex: lowercase → uppercase (from Coder.byte2hex)
        return md5.hexdigest().upper()

    @staticmethod
    def _encode_params(params: dict[str, str]) -> str:
        """URL-encode parameters as application/x-www-form-urlencoded."""
        from urllib.parse import urlencode
        return urlencode(params)
