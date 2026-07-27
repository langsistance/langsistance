"""SIPOP.cn — China Patent Open Data Platform API client.

Authentication uses appKey + MD5 signature (not OAuth2).  Every request
is a POST to the single gateway URL with method routing via the ``method``
parameter.

Signing formula (16-bit MD5)::

    sign = md5(timestamp + param_json + app_secret)

Reference: 数据开放平台接口调用说明 (2024.03, v1.1)
Base URL: http://open.sipop.cn/dataplatpro/api/route
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

import httpx

from sources.logger import Logger

_logger = Logger("sipop_client.log")

# ── Constants ───────────────────────────────────────────────────────────────────

SIPOP_BASE_URL = "http://open.sipop.cn/dataplatpro/api/route"
SIPOP_VERSION = "1.0"
SIPOP_REQUEST_TIMEOUT = 30

# ── API method constants ────────────────────────────────────────────────────────

METHOD_QUERY_BASIC_INFO = "patentBase.queryBasicInfo"
METHOD_QUERY_FULL_TEXT = "patentBase.queryFullTxt"
METHOD_QUERY_LAW_STATE = "patentSupport.queryLawStateInfo"
METHOD_QUERY_LEGAL_STATE = "patentSupport.queryLegalState"
METHOD_QUERY_PATENT_REVIEW = "patentSupport.queryPatentReview"
METHOD_QUERY_PATENT_TRANSFER = "patentSupport.queryPatentTransfer"
METHOD_QUERY_PATENT_LICENSE = "patentSupport.queryPatentLicense"
METHOD_QUERY_PATENT_PLEDGE = "patentSupport.queryPatentPledge"
METHOD_QUERY_FAMILY_INFO = "extra.queryFamilyInfo"
METHOD_QUERY_REFER_INFO = "extra.queryReferInfo"
METHOD_SENIOR_SEARCH = "patentFullSenior.seniorSearch"
METHOD_QUERY_BY_EXPRESSION = "patentFullSenior.queryByExpression"


# ── Client ──────────────────────────────────────────────────────────────────────


class SipopError(Exception):
    """Base error for SIPOP API failures."""


class SipopAuthError(SipopError):
    """Authentication failed (bad appKey/appSecret)."""


class SipopAPIError(SipopError):
    """API returned a non-success code."""


class SipopClient:
    """Async HTTP client for the sipop.cn China patent open data platform.

    All requests go through a single POST endpoint with method routing.
    Authentication params (appKey, sign, timestamp, v) are injected
    automatically — callers only need to provide ``method`` and ``param``.

    Usage::

        client = SipopClient(app_key="...", app_secret="...")
        result = await client.call_api(
            method="patentSupport.queryPatentReview",
            param={"applicationDocNum": "201710216936", "country": "CN"},
        )
        for decision in result.get("data", []):
            print(decision["decisionNumber"], decision["decision"])
    """

    def __init__(self, app_key: str, app_secret: str) -> None:
        if not app_key or not app_secret:
            raise SipopAuthError(
                "SIPOP app_key and app_secret are required. "
                "Set [SIPOP] in config.ini or SIPOP_APP_KEY / SIPOP_APP_SECRET env vars."
            )
        self._app_key = app_key
        self._app_secret = app_secret

    # ── Public API ────────────────────────────────────────────────────────────

    async def call_api(self, method: str, param: dict[str, Any]) -> dict[str, Any]:
        """Call the sipop.cn API and return the parsed JSON ``data`` field.

        Args:
            method: API method name (e.g. ``"patentSupport.queryPatentReview"``).
            param: API-specific parameters dict (e.g. ``{"applicationDocNum": "..."}``).

        Returns:
            The ``data`` field from the API response.

        Raises:
            SipopAuthError: appKey/appSecret invalid (code 1004).
            SipopAPIError: API returned a non-success code or malformed response.
        """
        request_body = self._build_request(method, param)
        _log_safe = {**request_body, "sign": request_body.get("sign", "")[:8] + "****"}
        _logger.info(f"sipop_request — method={method}, params={_log_safe}")

        async with httpx.AsyncClient(timeout=SIPOP_REQUEST_TIMEOUT) as client:
            response = await client.post(
                SIPOP_BASE_URL,
                data=request_body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        if response.status_code != 200:
            raise SipopAPIError(
                f"SIPOP API HTTP {response.status_code}: "
                f"{_truncate(response.text)}"
            )

        body = response.json()
        if not isinstance(body, dict):
            raise SipopAPIError(f"SIPOP API returned non-JSON response: {_truncate(response.text)}")

        code = str(body.get("code", ""))
        msg = body.get("msg", "")

        if code == "1004":
            raise SipopAuthError(f"SIPOP auth failed (sign error): {msg}")
        if code == "1001":
            raise SipopAuthError(f"SIPOP auth failed (invalid appKey): {msg}")

        # Code 1100 = no results found (query succeeded but data is empty).
        # This is a normal business case — the patent may not have any
        # examination review / legal event records on the platform.
        if code == "1100":
            _logger.info(
                f"sipop_response — method={method}, code=1100 (no results), "
                f"msg={msg}"
            )
            return {} if method != METHOD_QUERY_PATENT_REVIEW else []

        if code != "1000":
            raise SipopAPIError(f"SIPOP API error code={code}: {msg}")

        _logger.info(
            f"sipop_response — method={method}, "
            f"result={body.get('result')}, total={body.get('total')}, "
            f"count={body.get('count')}"
        )
        return body.get("data", {})

    async def query_patent_review(
        self, application_doc_num: str, country: str = "CN",
    ) -> list[dict[str, Any]]:
        """Query examination review decisions for a Chinese patent.

        Wraps ``patentSupport.queryPatentReview``.

        Args:
            application_doc_num: CN application number (e.g. ``"201710216936"``).
            country: Country code, default ``"CN"``.

        Returns:
            List of review decision dicts, each containing:
            decisionNumber, decisionDate, decision, appealType,
            appellant, assignee, inventionTitle, chiefExaminer,
            lawReference, reasoningParagraphs, finalDecisionParagraphs, etc.
        """
        data = await self.call_api(METHOD_QUERY_PATENT_REVIEW, {
            "applicationDocNum": application_doc_num,
            "country": country,
        })
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "data" in data:
            items = data["data"]
            return items if isinstance(items, list) else [items]
        return [data] if data else []

    async def query_law_state(
        self, application_doc_num: str,
    ) -> dict[str, Any]:
        """Query legal status summary for a Chinese patent.

        Wraps ``patentSupport.queryLawStateInfo``.

        Returns:
            Dict with date, lawStatusCode, lawStatus fields.
        """
        return await self.call_api(METHOD_QUERY_LAW_STATE, {
            "applicationDocNum": application_doc_num,
        })

    async def query_full_text(
        self, application_doc_num: str,
    ) -> dict[str, Any]:
        """Query full text (claims + description) for a Chinese patent.

        Wraps ``patentBase.queryFullTxt``.

        Returns:
            Dict with applicationDocNum, claim (list[str]), description (list[str]),
            descriptionFigure (list[str]) fields.
        """
        return await self.call_api(METHOD_QUERY_FULL_TEXT, {
            "applicationDocNum": application_doc_num,
        })

    async def query_legal_state_timeline(
        self, application_doc_num: str, country: str = "CN",
    ) -> list[dict[str, Any]]:
        """Query the full legal-status timeline for a patent.

        Wraps ``patentSupport.queryLegalState``.  Unlike ``query_law_state``
        which returns a single current-status record, this returns the
        complete history of legal events (designations, grants, withdrawals,
        etc.) — especially useful for PCT / foreign-origin patents.

        Returns:
            List of legal-status events, each with date, lawStatusCode,
            lawStatus, lawStatusDetail, lawStatusEffect.
        """
        data = await self.call_api(METHOD_QUERY_LEGAL_STATE, {
            "applicationDocNum": application_doc_num,
            "country": country,
        })
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "data" in data:
            items = data["data"]
            return items if isinstance(items, list) else []
        return []

    async def query_basic_info(
        self, application_doc_num: str,
    ) -> dict[str, Any]:
        """Query basic bibliographic info for a Chinese patent.

        Wraps ``patentBase.queryBasicInfo``.

        Returns:
            Dict with title, applicationDate, publicationDate, applicant,
            inventor, ipcMain, abstractDesc, etc.
        """
        return await self.call_api(METHOD_QUERY_BASIC_INFO, {
            "applicationDocNum": application_doc_num,
        })

    # ── Internal: request building ────────────────────────────────────────────

    def _build_request(self, method: str, param: dict[str, Any]) -> dict[str, str]:
        """Build the full POST form body with auth params injected.

        This is the unified HTTP egress layer — every request to
        ``open.sipop.cn`` gets appKey, timestamp, sign, and v auto-injected.
        """
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        param_json = json.dumps(param, ensure_ascii=False, separators=(",", ":"))
        sign = self._build_sign(timestamp, param_json)
        return {
            "appKey": self._app_key,
            "method": method,
            "param": param_json,
            "timestamp": timestamp,
            "sign": sign,
            "v": SIPOP_VERSION,
        }

    def _build_sign(self, timestamp: str, param_json: str) -> str:
        """Compute 16-bit lowercase MD5 sign.

        Formula: md5(timestamp + param_json + app_secret) → 16-char hex.
        """
        raw = f"{timestamp}{param_json}{self._app_secret}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()


# ── Helpers ──────────────────────────────────────────────────────────────────────


def _truncate(text: str, max_len: int = 300) -> str:
    return text if len(text) <= max_len else text[:max_len] + "..."
