"""EPO OPS API client for patent-family lookup.

Authenticates via OAuth2 Client Credentials grant and provides a single
``lookup_family()`` method that returns a ``PatentFamily``.

OAuth2 flow
-----------
1. POST https://ops.epo.org/3.2/auth/accesstoken
   Authorization: Basic base64(consumer_key:consumer_secret)
   Body: grant_type=client_credentials
2. Token expires after 20 minutes — the client caches and auto-refreshes.

API reference: https://developers.epo.org/apis/ops-v32
"""

from __future__ import annotations

import base64
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

import httpx

from sources.logger import Logger
from sources.long_task.family_member import FamilyMember, PatentFamily

logger = Logger("patent_family.log")

# ── Constants ───────────────────────────────────────────────────────────────────

EPO_TOKEN_URL = "https://ops.epo.org/3.2/auth/accesstoken"
EPO_FAMILY_URL = (
    "https://ops.epo.org/3.2/rest-services/family/publication/docdb/{pub_number}/biblio"
)
EPO_TOKEN_TTL_SECONDS = 19 * 60  # refresh 1 min before the 20-min expiry
EPO_REQUEST_TIMEOUT = 30  # seconds

# XML namespace used by OPS family responses
_OPS_NS = "http://ops.epo.org"
_EXCHANGE_NS = "http://www.epo.org/exchange"

# ── Per-jurisdiction examination analysis scope ─────────────────────────────────
# 审查分析有数据源支持的四局。
ANALYZABLE_JURISDICTIONS: tuple = ("US", "CN", "EP", "JP")

# EP 族的成员国（DE/GB/FR…）共用同一份 EPO 审查程序，合并成 EP 一个分析对象。
EP_MEMBER_STATES = frozenset({
    "AT", "BE", "BG", "CH", "CY", "CZ", "DE", "DK", "EE", "ES", "FI",
    "FR", "GB", "GR", "HR", "HU", "IE", "IS", "IT", "LI", "LT", "LU",
    "MC", "MT", "NL", "NO", "PL", "PT", "RO", "SE", "SI", "SK", "TR",
})


def _office_of(country: str) -> str:
    """把成员国代码归一到分析局：EP 的成员国归 EP，其余按原样。

    EP 族里的成员可能以 DE/GB/FR 等国家码出现（EPO 的 DOCDB 就这么给），
    它们接受的是**同一套** EPO 审查程序，必须合并，否则同一份审查过程会被
    当成多份、报告里出现重复章节。
    """
    code = str(country or "").strip().upper()
    if code in EP_MEMBER_STATES:
        return "EP"
    return code


def analyzable_jurisdictions(family) -> dict:
    """该族的哪些成员国能支撑一次审查分析 → {office: representative_member}。

    纯函数，无 I/O。存在的理由：``execute_family_analysis`` 原先硬取美国成员，
    拿不到就整任务失败 —— 哪怕族里明明有 CN/JP/EP 成员，而这三个局的分析器
    都已存在。用户问「查这件中国专利的全球审查历史」时，若该族恰好没有美国
    同族，得到的是「未找到美国同族成员」这一句与诉求无关的话。

    返回空 dict 表示**没有任何可分析的辖区**（如只有 WO/KR 成员），调用方据此
    如实报告覆盖范围，而不是报一个具体的"缺某个国"。
    """
    if family is None:
        return {}
    out: dict = {}
    for member in family.deduplicated_members:
        office = _office_of(member.country)
        if office not in ANALYZABLE_JURISDICTIONS:
            continue
        existing = out.get(office)
        # 同一局多条时优先取已授权的那条（审查过程最完整）。
        if existing is None or (member.is_granted and not existing.is_granted):
            out[office] = member
    return out


# ── Error types ──────────────────────────────────────────────────────────────────


class EPOError(Exception):
    """Base error for EPO OPS API failures."""


class EPOAuthError(EPOError):
    """OAuth2 token request failed (bad credentials, network, etc.)."""


class EPOFamilyError(EPOError):
    """Family lookup request failed."""


# ── Client ───────────────────────────────────────────────────────────────────────


@dataclass
class _TokenCache:
    """In-memory OAuth2 token with expiry."""
    access_token: str = ""
    expires_at: float = 0.0  # monotonic timestamp


class EPOFamilyClient:
    """Thin async client for the EPO OPS family API.

    Usage::

        client = EPOFamilyClient(consumer_key="...", consumer_secret="...")
        family = await client.lookup_family("US12506212")
        for member in family.deduplicated_members:
            print(member.country, member.pub_number, member.is_granted)
    """

    def __init__(
        self,
        consumer_key: str,
        consumer_secret: str,
        *,
        token_url: str = EPO_TOKEN_URL,
        family_url_template: str = EPO_FAMILY_URL,
    ) -> None:
        self._consumer_key = consumer_key
        self._consumer_secret = consumer_secret
        self._token_url = token_url
        self._family_url_template = family_url_template
        self._token_cache = _TokenCache()

    # ── Public API ────────────────────────────────────────────────────────────

    async def lookup_family(self, pub_number: str) -> PatentFamily:
        """Look up the patent family for *pub_number* (e.g. ``"US12506212"``).

        Returns a ``PatentFamily`` with all members parsed from the EPO OPS
        XML response.
        """
        token = await self._ensure_token()
        url = self._family_url_template.format(pub_number=pub_number)

        async with httpx.AsyncClient(timeout=EPO_REQUEST_TIMEOUT) as client:
            response = await client.get(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/xml",
                },
            )

        if response.status_code == 401 or response.status_code == 403:
            # Token may have expired early — force refresh and retry once
            self._token_cache = _TokenCache()
            token = await self._ensure_token()
            async with httpx.AsyncClient(timeout=EPO_REQUEST_TIMEOUT) as client:
                response = await client.get(
                    url,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/xml",
                    },
                )

        if response.status_code != 200:
            raise EPOFamilyError(
                f"EPO family lookup failed: HTTP {response.status_code} "
                f"for {pub_number}: {_truncate(response.text)}"
            )

        return _parse_family_xml(response.text, pub_number)

    # ── OAuth2 token management ───────────────────────────────────────────────

    async def _ensure_token(self) -> str:
        """Return a valid access token, refreshing if necessary."""
        now = time.monotonic()
        if self._token_cache.access_token and now < self._token_cache.expires_at:
            return self._token_cache.access_token

        credentials = f"{self._consumer_key}:{self._consumer_secret}"
        encoded = base64.b64encode(credentials.encode("ascii")).decode("ascii")

        async with httpx.AsyncClient(timeout=EPO_REQUEST_TIMEOUT) as client:
            response = await client.post(
                self._token_url,
                headers={
                    "Authorization": f"Basic {encoded}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={"grant_type": "client_credentials"},
            )

        if response.status_code != 200:
            raise EPOAuthError(
                f"EPO OAuth2 token request failed: HTTP {response.status_code}: "
                f"{_truncate(response.text)}"
            )

        body = response.json()
        access_token = body.get("access_token", "")
        if not access_token:
            raise EPOAuthError("EPO OAuth2 response missing access_token field")

        self._token_cache = _TokenCache(
            access_token=access_token,
            expires_at=time.monotonic() + EPO_TOKEN_TTL_SECONDS,
        )
        logger.info("epo_token_refreshed")
        return access_token


# ── XML parser ───────────────────────────────────────────────────────────────────


def _parse_family_xml(xml_text: str, query_pub_number: str) -> PatentFamily:
    """Parse an EPO OPS family-biblio XML response into a ``PatentFamily``."""
    root = ET.fromstring(xml_text)

    family_elem = root.find(f"{{{_OPS_NS}}}patent-family")
    if family_elem is None:
        raise EPOFamilyError("EPO family response missing <patent-family> element")

    family_id = ""
    total_count = int(family_elem.get("total-result-count", "0"))
    members: list[FamilyMember] = []

    for fm_elem in family_elem.findall(f"{{{_OPS_NS}}}family-member"):
        fid = fm_elem.get("family-id", "")
        if fid and not family_id:
            family_id = fid

        member = _parse_one_family_member(fm_elem, family_id)
        if member:
            members.append(member)

    return PatentFamily(
        query_pub_number=query_pub_number,
        family_id=family_id,
        total_count=total_count,
        members=members,
    )


def _parse_one_family_member(
    fm_elem: ET.Element, family_id: str
) -> FamilyMember | None:
    """Parse a single ``<ops:family-member>`` element into a ``FamilyMember``.

    Returns ``None`` if the member lacks a recognisable publication-reference
    (should not happen in practice).
    """
    # ── Publication reference (required) ────────────────────────────────────
    # NOTE: Elements without prefix inside <ops:family-member> belong to the
    # *default* XML namespace (http://www.epo.org/exchange), not the ops: ns.
    pub_ref = fm_elem.find(f"{{{_EXCHANGE_NS}}}publication-reference")
    if pub_ref is None:
        return None

    pub_doc_id = pub_ref.find(f"{{{_EXCHANGE_NS}}}document-id[@document-id-type='docdb']")
    if pub_doc_id is None:
        return None

    country = _text(pub_doc_id, "country")
    pub_number = _text(pub_doc_id, "doc-number")
    pub_kind = _text(pub_doc_id, "kind")
    pub_date = _text(pub_doc_id, "date")

    if not country or not pub_number:
        return None

    # ── Application reference ───────────────────────────────────────────────
    app_ref = fm_elem.find(f"{{{_EXCHANGE_NS}}}application-reference")
    app_number = ""
    app_date = ""
    if app_ref is not None:
        app_doc_id = app_ref.find(
            f"{{{_EXCHANGE_NS}}}document-id[@document-id-type='docdb']"
        )
        if app_doc_id is not None:
            app_number = _text(app_doc_id, "doc-number")
            app_date = _text(app_doc_id, "date")

    # ── Title from exchange-document (optional) ──────────────────────────────
    title = ""
    exchange = fm_elem.find(f"{{{_EXCHANGE_NS}}}exchange-document")
    if exchange is not None:
        biblio = exchange.find(f"{{{_EXCHANGE_NS}}}bibliographic-data")
        if biblio is not None:
            for t_elem in biblio.findall(f"{{{_EXCHANGE_NS}}}invention-title"):
                if t_elem.get("lang") == "en":
                    title = (t_elem.text or "").strip()
                    break

    return FamilyMember(
        country=country,
        pub_number=pub_number,
        pub_kind=pub_kind,
        pub_date=pub_date,
        app_number=app_number,
        app_date=app_date,
        title=title,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────────


def _text(parent: ET.Element, tag: str) -> str:
    """Return the text content of *tag* inside *parent*, or ``""``."""
    child = parent.find(f"{{{_EXCHANGE_NS}}}{tag}")
    if child is not None and child.text:
        return child.text.strip()
    return ""


def _truncate(text: str, max_len: int = 300) -> str:
    return text if len(text) <= max_len else text[:max_len] + "..."
