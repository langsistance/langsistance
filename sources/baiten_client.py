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
import re
import tempfile
from datetime import datetime, timezone, timedelta
from typing import Any

import httpx

from sources.logger import Logger

_logger = Logger("baiten_client.log")

# Shared POST headers for the TOP-style gateway.
_REQUEST_HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
    "User-Agent": "top-sdk-java",
    "Accept": "application/json",
}

# ── Constants (from SDK bytecode and live testing) ─────────────────────────

BAITEN_GATEWAY_URL = "http://open.baiten.cn/router"
BAITEN_REQUEST_TIMEOUT = 30

# SDK version (from Constants.SDK_VERSION)
_SDK_VERSION = "top-sdk-java-20140926"

# Date format (from Constants.DATE_TIME_FORMAT)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Charset (from Constants.CHARSET_UTF8)
_CHARSET = "UTF-8"

# API method names (from SDK request classes; live-probed against the
# gateway 2026-08-26 — a missing required param returns Spring "Required
# ... parameter 'x' is not present" BEFORE auth, which is how the wire
# contracts below were confirmed without a valid key).
_API_METHOD_LAW = "/openService/law"           # verified: app_num, law_category
_API_METHOD_SEARCH = "/openService/search"     # verified: query, level, page_index, page_size
_API_METHOD_CLAIMS = "/openService/claims"     # verified: app_num, pat_type
_API_METHOD_DOWNLOAD = "/openService/download"  # verified: pub_num, pub_date

# ── 以下 2 个 method 路径在网关上返回 404（2026-08-26 实测），
#    真实路径未知——SDK 类名（CubeOpenGetDocRequest/CubeOpenGetSpecRequest）
#    与线上路径命名不一致。search 响应自带 an/pd 时 getDoc 可绕开；spec
#    文本通道已延后（patent_detail 只渲染 pdf_url）。──
_API_METHOD_GET_DOC = "/openService/getDoc"    # UNVERIFIED — 404 on gateway
_API_METHOD_SPEC = "/openService/spec"         # UNVERIFIED — 404 on gateway

# PDF base64 spool: keep this many bytes in memory before spilling to disk.
# The server has <1GB RAM; a multi-page spec PDF (tens of MB, 1.33x base64)
# must never be buffered whole (memory: server-memory-constraint).
_SPOOL_MAX_MEMORY = 8 * 1024 * 1024  # 8 MB

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


def _error_preview(text: str) -> str:
    """Short, useful preview of a non-200 response body.

    The gateway is Spring/Tomcat: a bad request returns an HTML error
    page whose ``<b>Message</b>`` line names the missing/invalid request
    parameter ("Required String parameter 'level' is not present") —
    exactly the signal needed to fix the wire contract.  Extracting it
    beats dumping 300 chars of HTML style blocks into logs/notes.
    """
    if not text:
        return "(empty body)"
    if "<" in text and ("<title>" in text or "<html" in text.lower()):
        import html as _html
        m = re.search(r"<b>Message</b>\s*(.*?)</p>", text,
                      re.IGNORECASE | re.DOTALL)
        if m:
            msg = _html.unescape(re.sub(r"\s+", " ", m.group(1))).strip()
            if msg:
                return msg[:300]
        m = re.search(r"<title>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
        if m:
            return _html.unescape(m.group(1)).strip()[:300]
    return text[:300]


def summarize_search_response(body: dict) -> dict:
    """Defensive summary of a Baiten search response for diagnostics.

    Distinguishes "gateway returned 0 records" from "records present but
    the candidate mapping dropped them": ``rows`` counts the raw hit
    entries (``data.fieldValues`` per the SDK shape, ``grouped_hits``
    per the live gateway error envelope), ``total`` is the gateway's
    count field when present (``total_hits`` on the wire), ``keys``
    shows the top-level shape so a schema drift is visible in a single
    log line.  Pure — never raises.
    """
    if not isinstance(body, dict):
        return {"total": None, "rows": 0, "keys": []}
    data = body.get("data")
    rows = data.get("fieldValues") if isinstance(data, dict) else None
    if rows is None:
        rows = body.get("fieldValues")
    if rows is None:
        rows = body.get("grouped_hits")
    if not isinstance(rows, list):
        rows = []
    total = body.get("total")
    if total is None:
        total = body.get("total_hits")
    if total is None and isinstance(data, dict):
        total = data.get("total")
    return {
        "total": total,
        "rows": len(rows),
        "keys": list(body.keys())[:8],
    }


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

        body = await self._request_json(
            _API_METHOD_LAW,
            {"app_num": app_num, "law_category": law_category},
        )

        _logger.info(
            f"baiten_response — lawCategory={law_category}, "
            f"appNum={app_num}, "
            f"has_patent_laws={bool(body.get('patent_laws'))}, "
            f"has_declareList={bool(body.get('patentLawDeclare_list'))}"
        )
        return body

    # ── Baiten open-platform methods (search / doc / claims / spec / file) ──
    #
    # Method paths and parameter names below are inferred from the SDK
    # request classes (cube-sdk-ext-1.0.jar) and are marked 待实测 until
    # the live gateway confirms them (proposal §八 ①-③).  All failures
    # raise BaitenAPIError so callers can degrade gracefully.

    async def search(
        self, query_string: str, source: str = "15", page: int = 1,
        page_size: int = 20, api_level: str = "ONE",
    ) -> dict[str, Any]:
        """Basic full-text search (source=15 → China patent library).

        Wire contract verified against the live gateway 2026-08-26:
        ``query`` / ``level`` / ``page_index`` / ``page_size`` (NOT the
        SDK-style queryString/apiLevel/page/pageSize — those get a Spring
        400 "Required ... parameter is not present").  Returns the
        gateway payload; the caller maps the hit rows (fieldValues /
        grouped_hits — shape confirmed on first valid-key success) into
        candidate structures.
        """
        body = await self._request_json(
            _API_METHOD_SEARCH,
            {
                "query": query_string,
                "source": source,
                "page_index": page,
                "page_size": page_size,
                "level": api_level,
            },
        )
        _logger.info(
            f"baiten_search — query={query_string[:80]}, source={source}, "
            f"page={page}"
        )
        _logger.info(
            f"baiten_search_response — {summarize_search_response(body)}"
        )
        return body

    async def get_doc(self, doc_id: str) -> dict[str, Any]:
        """Full bibliographic record for one patent (docId = pn or id)."""
        body = await self._request_json(
            _API_METHOD_GET_DOC, {"docId": doc_id},
        )
        _logger.info(f"baiten_get_doc — docId={doc_id}")
        return body

    async def get_claims(
        self, app_num: str, pat_type: str = "APP",
    ) -> dict[str, Any]:
        """Structured claims (patentClaimses[] hierarchy) for one patent.

        Wire contract verified 2026-08-26: ``app_num`` / ``pat_type``.
        pat_type: "AUTH" (granted) or "APP" (application).
        """
        body = await self._request_json(
            _API_METHOD_CLAIMS,
            {"app_num": app_num, "pat_type": pat_type},
        )
        _logger.info(f"baiten_get_claims — appNum={app_num}, patType={pat_type}")
        return body

    async def get_spec(self, doc_id: str) -> dict[str, Any]:
        """Specification text (docId = CN application number)."""
        body = await self._request_json(
            _API_METHOD_SPEC, {"docId": doc_id},
        )
        _logger.info(f"baiten_get_spec — docId={doc_id}")
        return body

    async def get_file(
        self, pub_num: str, pub_date: str, file_category: str = "PDF",
    ):
        """Download a patent PDF as a streaming spooled file.

        Wire contract verified 2026-08-26: method path
        ``/openService/download`` with ``pub_num`` / ``pub_date`` (the
        SDK-style pubNum/pubDate names get a Spring 400).

        The gateway returns ``fileByte`` as a base64 JSON string that can
        reach tens of MB (1.33x expansion) — the response is streamed and
        base64-decoded chunk-by-chunk into a ``tempfile.SpooledTemporaryFile``
        (8 MB memory threshold, then disk), so peak memory stays far below
        the server's 1 GB budget.

        Returns:
            ``tempfile.SpooledTemporaryFile`` rewound to position 0.
            The caller owns it and must close it when done.
        """
        all_params = self._build_top_params(
            _API_METHOD_DOWNLOAD,
            {
                "pub_num": pub_num,
                "pub_date": pub_date,
                "file_category": file_category,
            },
        )
        api_method = all_params.pop("method")
        api_method_path = api_method[1:] if api_method.startswith("/") else api_method
        form_data = self._encode_params(all_params)
        url = f"{self._gateway_url}/{api_method_path}"

        _logger.info(
            f"baiten_file — url={url}, pubNum={pub_num}, pubDate={pub_date}"
        )

        spool = tempfile.SpooledTemporaryFile(max_size=_SPOOL_MAX_MEMORY)
        try:
            async with httpx.AsyncClient(timeout=BAITEN_REQUEST_TIMEOUT) as client:
                async with client.stream(
                    "POST", url, content=form_data, headers=_REQUEST_HEADERS,
                ) as response:
                    if response.status_code != 200:
                        preview = (await response.aread())[:300]
                        raise BaitenAPIError(
                            f"Baiten API HTTP {response.status_code}: "
                            f"{preview!r}"
                        )
                    found = await self._stream_base64_field(
                        response.aiter_bytes(), b'"fileByte"', spool,
                    )
            if not found:
                raise BaitenAPIError(
                    f"Baiten file response carries no fileByte field "
                    f"(pubNum={pub_num}, pubDate={pub_date})"
                )
        except Exception:
            spool.close()
            raise
        spool.seek(0)
        return spool

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
        self, method: str, application: dict[str, Any],
    ) -> dict[str, str]:
        """Build the complete parameter map for a TOP-style API call.

        Combines protocol-mandatory params (method, app_key, timestamp, v,
        sign_method, format) with application-specific params, then
        computes the MD5 signature over ALL params.

        The ``method`` param is returned for use in the URL path; the caller
        removes it from the POST body.
        """
        # Beijing time (GMT+8) — from Constants.DATE_TIMEZONE
        now = datetime.now(timezone(timedelta(hours=8)))
        timestamp = now.strftime(_DATE_FORMAT)

        # Protocol-mandatory parameters (from DefaultCubeClient)
        protocol = {
            "method": method,
            "app_key": self._app_key,
            "timestamp": timestamp,
            "v": _API_VERSION,
            "sign_method": _SIGN_METHOD,
            "format": _FORMAT,
        }

        # Merge all params for signing
        all_params = {**protocol, **application}
        sign = self._top_sign(all_params, self._app_secret)
        all_params["sign"] = sign

        return all_params

    async def _request_json(self, method: str, params: dict[str, Any]) -> dict:
        """POST one JSON-returning API method with shared error handling.

        Signs and encodes *params*, routes to ``{gateway}/{method}`` and
        raises ``BaitenAPIError`` on HTTP errors or a non-200 gateway
        ``code``.  Returns the parsed JSON object.
        """
        all_params = self._build_top_params(method, params)
        api_method = all_params.pop("method")
        api_method_path = api_method[1:] if api_method.startswith("/") else api_method
        form_data = self._encode_params(all_params)
        url = f"{self._gateway_url}/{api_method_path}"

        _logger.info(f"baiten_request — url={url}, method={method}")

        async with httpx.AsyncClient(timeout=BAITEN_REQUEST_TIMEOUT) as client:
            response = await client.post(
                url,
                content=form_data,
                headers=_REQUEST_HEADERS,
            )

        if response.status_code != 200:
            raise BaitenAPIError(
                f"Baiten API HTTP {response.status_code}: "
                f"{_error_preview(response.text)}"
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
        return body

    @staticmethod
    async def _stream_base64_field(
        chunks, field: bytes, out,
    ) -> bool:
        """Stream-scan a JSON response for a base64 string field.

        Decodes the field's base64 content chunk-by-chunk into *out*
        (a binary file-like), keeping at most 4 undecoded bytes in memory
        and the rest written straight through.  The base64 alphabet
        (A-Za-z0-9+/=) contains neither ``"`` nor ``\\``, so the first
        unescaped quote after the opening one terminates the value —
        no full JSON parse is needed, and peak memory stays O(chunk).

        Returns True when the field was found and decoded; False when the
        stream ended without it (e.g. an error JSON with no fileByte).
        """
        import base64 as _b64
        import binascii as _binascii

        b64_buf = bytearray()
        tail = bytearray()  # carry the half-split field marker across chunks
        state = "scan"      # scan → value_start → collect → done
        value = False

        async for chunk in chunks:
            data = bytes(tail) + chunk
            del tail[:]
            if state == "scan":
                idx = data.find(field)
                if idx < 0:
                    tail.extend(data[-len(field) + 1:])
                    continue
                data = data[idx + len(field):]
                state = "value_start"
            if state == "value_start":
                data = data.lstrip(b" \t\r\n:")  # colon/whitespace after key
                if not data:
                    continue
                if data[0] != 34:  # 34 = '"' — null/absent value
                    return False
                data = data[1:]
                state = "collect"
            if state == "collect":
                end = data.find(b'"')
                if end < 0:
                    b64_buf.extend(data)
                else:
                    b64_buf.extend(data[:end])
                    state = "done"
                # decode aligned runs, keep <4-byte remainder for next chunk
                n = len(b64_buf) - (len(b64_buf) % 4)
                if n:
                    try:
                        out.write(_b64.b64decode(bytes(b64_buf[:n]),
                                                 validate=False))
                    except (_binascii.Error, ValueError):
                        out.write(bytes(b64_buf[:n]))
                    del b64_buf[:n]
                if state == "done":
                    value = True
                    break
        return value

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
