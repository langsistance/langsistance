"""JPO — Japan Patent Office IP Data Platform API client.

Authentication uses OAuth2 password grant (not MD5 signing).  Tokens are
refreshed automatically when expired.

Reference:
  https://ip-data.jpo.go.jp/pages/top_e.html
  https://www.jpo.go.jp/e/system/laws/koho/internet/document/api-patent_info/

Base URL: https://ip-data.jpo.go.jp
Auth endpoint: POST /auth/token
API prefix: /api
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import httpx

from sources.logger import Logger

_logger = Logger("jpo_client.log")

# ── Constants ───────────────────────────────────────────────────────────────────

JPO_BASE_URL = "https://ip-data.jpo.go.jp"
JPO_AUTH_PATH = "/auth/token"
JPO_API_PREFIX = "/api"
JPO_REQUEST_TIMEOUT = 60  # some document endpoints return large ZIPs
JPO_TOKEN_LIFETIME_SEC = 3300   # 55 min, actual lifetime is 1 h — refresh early
JPO_MAX_RETRIES = 2

# ── Errors ─────────────────────────────────────────────────────────────────────


class JpoError(Exception):
    """Base error for JPO API failures."""


class JpoAuthError(JpoError):
    """Authentication failed (bad credentials or token expired)."""


class JpoAPIError(JpoError):
    """API returned a non-success status code."""


# ── Client ──────────────────────────────────────────────────────────────────────


class JpoClient:
    """Async HTTP client for the JPO IP Data Platform.

    Manages OAuth2 token lifecycle automatically — callers only provide
    method-specific parameters.

    Usage::

        client = JpoClient(username="...", password="...")
        progress = await client.get_patent_progress("202080061975")
        for bib in progress.get("bibliographyInformation", []):
            for doc in bib.get("documentList", []):
                print(doc["documentCode"], doc["documentDescription"])
    """

    def __init__(self, username: str, password: str) -> None:
        if not username or not password:
            raise JpoAuthError(
                "JPO username and password are required. "
                "Set [JPO] in config.ini or JPO_USERNAME / JPO_PASSWORD env vars."
            )
        self._username = username
        self._password = password
        self._access_token: str | None = None
        self._token_expires_at: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_patent_progress(
        self, application_number: str,
    ) -> dict[str, Any]:
        """Retrieve full patent examination progress history.

        GET /api/patent/v1/app_progress/{applicationNumber}

        Args:
            application_number: 10-digit JP application number.

        Returns:
            Dict with bibliographic fields plus ``bibliographyInformation`` —
            a list of ``{numberType, number, documentList[]}`` entries where
            each document carries ``documentCode``, ``documentDescription``,
            ``legalDate`` and ``documentNumber``.  There is NO event list;
            the document list IS the examination history.
        """
        return await self._get(
            f"{JPO_API_PREFIX}/patent/v1/app_progress/{application_number}"
        )

    async def get_patent_progress_simple(
        self, application_number: str,
    ) -> dict[str, Any]:
        """Simplified patent progress (no div/priority info).

        GET /api/patent/v1/app_progress_simple/{applicationNumber}
        """
        return await self._get(
            f"{JPO_API_PREFIX}/patent/v1/app_progress_simple/{application_number}"
        )

    async def get_registration_info(
        self, application_number: str,
    ) -> dict[str, Any]:
        """Retrieve patent registration information.

        GET /api/patent/v1/registration_info/{applicationNumber}

        Returns registration number, date, rights status etc.
        """
        return await self._get(
            f"{JPO_API_PREFIX}/patent/v1/registration_info/{application_number}"
        )

    async def get_citations(
        self, application_number: str,
    ) -> dict[str, Any]:
        """Retrieve cited document information.

        GET /api/patent/v1/cite_doc_info/{applicationNumber}

        Returns list of documents cited by the examiner.
        """
        return await self._get(
            f"{JPO_API_PREFIX}/patent/v1/cite_doc_info/{application_number}"
        )

    async def get_refusal_reasons(
        self, application_number: str,
    ) -> dict[str, Any]:
        """Retrieve notices of reasons for refusal (拒絶理由通知書).

        GET /api/patent/v1/app_doc_cont_refusal_reason/{applicationNumber}

        Returns the refusal reason documents (XML in ZIP or direct content).
        """
        return await self._get(
            f"{JPO_API_PREFIX}/patent/v1/app_doc_cont_refusal_reason"
            f"/{application_number}"
        )

    async def get_amendments(
        self, application_number: str,
    ) -> dict[str, Any]:
        """Retrieve written opinions / amendments (意見書・補正書).

        GET /api/patent/v1/app_doc_cont_opinion_amendment/{applicationNumber}
        """
        return await self._get(
            f"{JPO_API_PREFIX}/patent/v1/app_doc_cont_opinion_amendment"
            f"/{application_number}"
        )

    async def lookup_number_relation(
        self, relation_type: str, case_number: str,
    ) -> dict[str, Any]:
        """Cross-reference application / publication / registration numbers.

        GET /api/patent/v1/case_number_reference/{relationType}/{caseNumber}

        Args:
            relation_type: 'application' | 'publication' | 'registration'.
            case_number: Number to look up.
        """
        return await self._get(
            f"{JPO_API_PREFIX}/patent/v1/case_number_reference"
            f"/{relation_type}/{case_number}"
        )

    async def get_family(
        self, relation: str, case_number: str,
    ) -> dict[str, Any]:
        """Retrieve patent family information (OPD).

        GET /api/patent/v1/family/{relation}/{caseNumber}

        Args:
            relation: 'application' | 'publication'.
            case_number: JP application or publication number.
        """
        return await self._get(
            f"{JPO_API_PREFIX}/patent/v1/family/{relation}/{case_number}"
        )

    # ── Internal: HTTP + auth ────────────────────────────────────────────────

    async def _authenticate(self) -> None:
        """Obtain or refresh the OAuth2 access token."""
        _logger.info("jpo_auth — requesting new access token")

        async with httpx.AsyncClient(timeout=JPO_REQUEST_TIMEOUT) as client:
            resp = await client.post(
                f"{JPO_BASE_URL}{JPO_AUTH_PATH}",
                data={
                    "grant_type": "password",
                    "username": self._username,
                    "password": self._password,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        if resp.status_code != 200:
            body = _safe_body(resp)
            raise JpoAuthError(
                f"JPO auth failed — HTTP {resp.status_code}: {body}"
            )

        data = resp.json()
        token = data.get("access_token")
        if not token:
            raise JpoAuthError(
                f"JPO auth response missing access_token: "
                f"{_safe_body(resp)}"
            )

        self._access_token = token
        self._token_expires_at = time.time() + JPO_TOKEN_LIFETIME_SEC
        _logger.info(
            f"jpo_auth_ok — token_len={len(token)}, "
            f"expires_in={JPO_TOKEN_LIFETIME_SEC}s"
        )

    async def _get(self, path: str) -> dict[str, Any]:
        """Perform an authenticated GET request with automatic token refresh.

        All JPO API responses follow the same envelope::

            {"result": {"statusCode": "100", ...}}
            {"result": {"statusCode": "200", "errorMessage": "..."}}

        Code '100' = success; anything else is an error.
        """
        for attempt in range(JPO_MAX_RETRIES + 1):
            await self._ensure_token()

            url = f"{JPO_BASE_URL}{path}"
            _logger.info(f"jpo_request — path={path}")

            try:
                async with httpx.AsyncClient(timeout=JPO_REQUEST_TIMEOUT) as client:
                    resp = await client.get(
                        url,
                        headers={
                            "Authorization": f"Bearer {self._access_token}",
                            "Accept": "application/json",
                        },
                    )
            except httpx.TimeoutException:
                if attempt < JPO_MAX_RETRIES:
                    _logger.warning(f"jpo_timeout — path={path}, retrying")
                    continue
                raise JpoAPIError(f"JPO API timed out after {JPO_REQUEST_TIMEOUT}s: {path}")

            if resp.status_code == 401:
                _logger.warning(
                    f"jpo_401 — path={path}, invalidating token"
                )
                self._access_token = None
                self._token_expires_at = 0.0
                continue

            if resp.status_code != 200:
                raise JpoAPIError(
                    f"JPO API HTTP {resp.status_code}: {_safe_body(resp)}"
                )

            body = resp.json()
            if not isinstance(body, dict):
                raise JpoAPIError(
                    f"JPO API returned non-JSON: {_safe_body(resp)}"
                )

            result = body.get("result", body)
            if not isinstance(result, dict):
                raise JpoAPIError(
                    f"JPO API missing result envelope: {_safe_body(resp)}"
                )

            status_code = str(result.get("statusCode", ""))
            if status_code == "100":
                # JPO API wraps actual data inside a "data" key.
                # Unwrap it so callers get the data directly.
                result_data = result.get("data", result)
                _logger.info(
                    f"jpo_response_ok — path={path}"
                )
                return result_data

            # statusCode != 100 → API error
            error_msg = result.get("errorMessage", "unknown error")
            if status_code in ("200", "201", "202", "400"):
                # 200-series are business-logic errors (e.g. "not found")
                raise JpoAPIError(
                    f"JPO API error statusCode={status_code}: {error_msg}"
                )

            raise JpoAPIError(
                f"JPO API unexpected statusCode={status_code}: {error_msg}"
            )

        raise JpoAuthError(
            f"JPO token refresh failed after {JPO_MAX_RETRIES} retries"
        )

    async def _ensure_token(self) -> None:
        if self._access_token and time.time() < self._token_expires_at:
            return
        await self._authenticate()


# ── Helpers ──────────────────────────────────────────────────────────────────────


def _safe_body(resp: httpx.Response, max_len: int = 300) -> str:
    try:
        text = resp.text
    except Exception:
        return "(body unavailable)"
    return text if len(text) <= max_len else text[:max_len] + "..."


def normalize_jp_application_number(raw: str) -> str:
    """Normalize a JP application number to the 10-digit format expected by JPO API.

    Handles various input formats:
      - "JP2020-123456" → "2020123456"
      - "2020-123456"   → "2020123456"
      - "2020123456"    → "2020123456"

    The JPO API expects a bare 10-digit number: YYYYNNNNNN (4-digit year + 6-digit serial).
    """
    import re

    num = raw.strip()
    # Strip "JP" prefix
    num = re.sub(r'^JP\s*', '', num, flags=re.IGNORECASE)
    # Remove separators
    num = num.replace('-', '').replace('.', '').replace(' ', '').replace('/', '')
    # Remove any trailing kind codes (A, B1, B2, etc.)
    num = re.sub(r'[A-Z]\d*$', '', num)
    # Pad to 10 digits if needed (some older numbers are shorter)
    if len(num) < 10:
        _logger.warning(
            f"jpo_app_number_short — raw={raw}, normalized={num}, len={len(num)}"
        )
    return num


def parse_jp_progress_events(
    progress_data: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    """Parse the JPO ``app_progress`` response into examination events.

    The JPO API has NO ``progressList`` key.  Real examination events live
    in ``bibliographyInformation[].documentList[]`` — one entry per
    document (書類), each carrying:

      - ``documentCode``        — 中間書類コード (e.g. "A621..." = 拒絶理由通知書)
      - ``documentDescription`` — 書類名 (human-readable, e.g. "拒絶理由通知書")
      - ``legalDate``           — 受付日・発送日・作成日 (YYYYMMDD)
      - ``documentNumber``      — 書類番号 (11 digits)

    Returns:
        ``(events, synthesized)`` — ``events`` are REAL events from the
        document list (sorted by date).  When the document list is empty
        (bibliographic-only response), a fallback timeline is synthesized
        from the bibliographic dates and ``synthesized`` is ``True`` so
        callers can treat the data as degraded instead of real.
    """
    events: list[dict[str, Any]] = []

    bib_entries = progress_data.get("bibliographyInformation", [])
    if isinstance(bib_entries, list):
        for bib_entry in bib_entries:
            if not isinstance(bib_entry, dict):
                continue
            doc_list = bib_entry.get("documentList", [])
            if not isinstance(doc_list, list):
                continue
            for doc in doc_list:
                if not isinstance(doc, dict):
                    continue
                event_date = str(doc.get("legalDate", "") or "").strip()
                event_code = str(doc.get("documentCode", "") or "").strip()
                event_desc = str(doc.get("documentDescription", "") or "").strip()
                if not event_date and not event_code and not event_desc:
                    continue  # fully empty placeholder row
                events.append({
                    "event": event_desc or f"書類コード {event_code}",
                    "event_date": event_date,
                    "event_category": "",
                    "event_detail": "",
                    "event_remarks": "",
                    "event_code": event_code,
                    "event_number": str(doc.get("documentNumber", "") or "").strip(),
                    "document_number": str(doc.get("documentNumber", "") or "").strip(),
                })

    # ── Fallback: synthesize from bibliographic dates ──────────────────────
    if not events:
        events = _synthesize_progress_from_biblio(progress_data)
        return events, True

    # Sort by date (empty dates last, stable for same-date documents)
    events.sort(key=lambda e: (e.get("event_date", "") == "", e.get("event_date", "")))
    return events, False


def _synthesize_progress_from_biblio(
    data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build a minimal timeline from bibliographic date fields.

    Used when the JPO ``app_progress`` endpoint returns only bibliographic
    data (no document list).  Every synthesized event carries
    ``synthesized=True`` so downstream consumers can label the data as
    degraded instead of presenting it as a real examination history.
    """
    events: list[dict[str, Any]] = []

    # Filing date
    filing = data.get("filingDate", "")
    if filing:
        events.append({
            "event": "出願",
            "event_date": filing,
            "event_category": "A01",
            "event_detail": data.get("inventionTitle", ""),
            "event_remarks": "",
            "synthesized": True,
        })

    # Publication date
    pub_date = data.get("publicationDate", "")
    if pub_date:
        events.append({
            "event": "出願公開",
            "event_date": pub_date,
            "event_category": "B01",
            "event_detail": data.get("publicationNumber", ""),
            "event_remarks": "",
            "synthesized": True,
        })

    # Registration (grant) date
    reg_date = data.get("registrationDate", "")
    reg_num = data.get("registrationNumber", "")
    if reg_date:
        events.append({
            "event": "特許登録",
            "event_date": reg_date,
            "event_category": "G01",
            "event_detail": reg_num or "",
            "event_remarks": "",
            "synthesized": True,
        })

    return events


# ── Common Japanese examination event translations ─────────────────────────────

JPO_EVENT_TRANSLATIONS: dict[str, dict[str, str]] = {
    "出願":                 {"zh": "提交申请",            "en": "Application Filed"},
    "出願公開":             {"zh": "申请公开",            "en": "Application Published"},
    "出願審査請求":         {"zh": "请求实质审查",        "en": "Request for Examination"},
    "審査請求":             {"zh": "请求实质审查",        "en": "Request for Examination"},
    "拒絶理由通知":         {"zh": "驳回理由通知",        "en": "Notice of Reasons for Refusal"},
    "拒絶査定":             {"zh": "驳回决定",            "en": "Decision of Refusal"},
    "拒絶査定不服審判請求": {"zh": "驳回决定上诉",        "en": "Appeal Against Decision of Refusal"},
    "特許査定":             {"zh": "授权决定",            "en": "Decision to Grant"},
    "特許登録":             {"zh": "专利注册",            "en": "Patent Registration"},
    "登録査定":             {"zh": "注册决定",            "en": "Decision to Register"},
    "設定登録":             {"zh": "设定注册",            "en": "Registration"},
    "手続補正書":           {"zh": "手续补正书",          "en": "Written Amendment"},
    "意見書":               {"zh": "意见陈述书",          "en": "Written Opinion"},
    "査定":                 {"zh": "审查决定",            "en": "Examination Decision"},
    "審判":                 {"zh": "审判",                "en": "Trial/Appeal"},
    "審決":                 {"zh": "审决",                "en": "Trial Decision"},
    "取下":                 {"zh": "撤回",                "en": "Withdrawal"},
    "放棄":                 {"zh": "放弃",                "en": "Abandonment"},
    "無効":                 {"zh": "无效",                "en": "Invalidation"},
    "異議":                 {"zh": "异议",                "en": "Opposition"},
    "優先権主張":           {"zh": "优先权主张",          "en": "Priority Claim"},
    "出願分割":             {"zh": "分案申请",            "en": "Divisional Application"},
    "変更":                 {"zh": "变更",                "en": "Change/Conversion"},
    "中間処理":             {"zh": "中间处理",            "en": "Intermediate Processing"},
    "国内移行":             {"zh": "进入国家阶段",        "en": "National Phase Entry"},
    "国際出願":             {"zh": "国际申请",            "en": "International Application"},
    "受理":                 {"zh": "受理",                "en": "Accepted/Received"},
    "送付":                 {"zh": "发送",                "en": "Sent/Dispatched"},
    "納付":                 {"zh": "缴纳",                "en": "Paid"},
    "登録":                 {"zh": "注册",                "en": "Registration"},
    "年金":                 {"zh": "年费",                "en": "Annual Fee"},
    "存続":                 {"zh": "维持有效",            "en": "Maintained"},
    "抹消":                 {"zh": "注销",                "en": "Cancelled"},
    "移転":                 {"zh": "转让",                "en": "Transfer/Assignment"},
    "消滅":                 {"zh": "失效",                "en": "Lapsed/Expired"},
}


def translate_jp_event(event_name: str, lang: str = "zh") -> str:
    """Translate a Japanese examination event name to Chinese or English."""
    # Direct match
    if event_name in JPO_EVENT_TRANSLATIONS:
        return JPO_EVENT_TRANSLATIONS[event_name].get(lang, event_name)
    # Partial match (event name may contain additional text)
    for jp_key, translations in JPO_EVENT_TRANSLATIONS.items():
        if jp_key in event_name:
            return translations.get(lang, event_name)
    return event_name
