"""General async client for the EPO Open Patent Services (OPS) API v3.2.

Extends the OAuth2 token management from ``patent_family.py`` to cover all
OPS services needed for examination-history analysis:

- **Family** — patent family lookup (existing functionality, kept for BWC)
- **Register** — application register: biblio, events, procedural-steps
- **Published Data** — published documents: biblio, claims, description, fulltext

OAuth2 flow (same as patent_family.py)
---------------------------------------
1. POST https://ops.epo.org/3.2/auth/accesstoken
   Authorization: Basic base64(consumer_key:consumer_secret)
   Body: grant_type=client_credentials
2. Token expires after 20 minutes — cached and auto-refreshed at 19 min.

API reference: https://developers.epo.org/apis/ops-v32
"""

from __future__ import annotations

import base64
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import httpx

from sources.logger import Logger

logger = Logger("epo_ops.log")

# ── Constants ───────────────────────────────────────────────────────────────────

EPO_TOKEN_URL = "https://ops.epo.org/3.2/auth/accesstoken"
EPO_BASE = "https://ops.epo.org/3.2/rest-services"
EPO_TOKEN_TTL_SECONDS = 19 * 60
EPO_REQUEST_TIMEOUT = 30

# XML namespaces
_OPS_NS = "http://ops.epo.org"
_EXCHANGE_NS = "http://www.epo.org/exchange"

# Accept header to request XML (register API returns XML by default)
_ACCEPT_XML = "application/xml"


# ── Error types ──────────────────────────────────────────────────────────────────


class EPOError(Exception):
    """Base error for EPO OPS API failures."""


class EPOAuthError(EPOError):
    """OAuth2 token request failed."""


class EPORegisterError(EPOError):
    """Register API request failed."""


class EPOPublishedDataError(EPOError):
    """Published Data API request failed."""


# ── Dataclasses ──────────────────────────────────────────────────────────────────


@dataclass
class EPORegisterBiblio:
    """Parsed bibliographic data from the EPO Register."""

    app_number: str = ""
    pub_number: str = ""
    title: str = ""
    title_en: str = ""
    applicant: str = ""
    ipc_classes: list[str] = None  # type: ignore[assignment]
    filing_date: str = ""
    pub_date: str = ""
    grant_date: str = ""
    status: str = ""  # e.g. "GRANTED", "PENDING", "REFUSED", "WITHDRAWN"
    raw: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.ipc_classes is None:
            self.ipc_classes = []
        if self.raw is None:
            self.raw = {}


@dataclass
class EPORegisterEvent:
    """A single legal event from the EPO Register."""

    event_code: str = ""
    event_date: str = ""  # YYYYMMDD
    description: str = ""
    description_en: str = ""
    category: str = ""  # e.g. "EXAMINATION", "GRANT", "OPPOSITION", "SURRENDER"


@dataclass
class EPOProceduralStep:
    """A single procedural step from the EPO Register."""

    step_code: str = ""
    step_date: str = ""  # YYYYMMDD
    description: str = ""
    description_en: str = ""


# ── Helpers ──────────────────────────────────────────────────────────────────────


def _text(parent: ET.Element, tag: str, ns: str = _EXCHANGE_NS) -> str:
    """Return text content of *tag* inside *parent*, or ``""``."""
    child = parent.find(f"{{{ns}}}{tag}")
    if child is not None and child.text:
        return child.text.strip()
    return ""


def _truncate(text: str, max_len: int = 300) -> str:
    return text if len(text) <= max_len else text[:max_len] + "..."


# ── Client ───────────────────────────────────────────────────────────────────────


@dataclass
class _TokenCache:
    """In-memory OAuth2 token with expiry."""
    access_token: str = ""
    expires_at: float = 0.0


class EPOClient:
    """Async client for EPO OPS API v3.2 — Register, Published Data, Family.

    Usage::

        client = EPOClient(consumer_key="...", consumer_secret="...")
        biblio = await client.register_biblio("EP12345678")
        events = await client.register_events("EP12345678")
        steps  = await client.register_procedural_steps("EP12345678")
    """

    def __init__(
        self,
        consumer_key: str,
        consumer_secret: str,
        *,
        token_url: str = EPO_TOKEN_URL,
    ) -> None:
        self._consumer_key = consumer_key
        self._consumer_secret = consumer_secret
        self._token_url = token_url
        self._token_cache = _TokenCache()

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

    async def _get_xml(self, url: str, purpose: str = "general") -> str:
        """Make an authenticated GET request and return the XML response body.

        Automatically retries once on 401/403 with a fresh token.
        """
        token = await self._ensure_token()

        async with httpx.AsyncClient(timeout=EPO_REQUEST_TIMEOUT) as client:
            response = await client.get(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": _ACCEPT_XML,
                },
            )

        if response.status_code in (401, 403):
            self._token_cache = _TokenCache()
            token = await self._ensure_token()
            async with httpx.AsyncClient(timeout=EPO_REQUEST_TIMEOUT) as client:
                response = await client.get(
                    url,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": _ACCEPT_XML,
                    },
                )

        if response.status_code != 200:
            raise EPOError(
                f"EPO {purpose} failed: HTTP {response.status_code} "
                f"for {_truncate(url)}: {_truncate(response.text)}"
            )

        return response.text

    # ── Register API ──────────────────────────────────────────────────────────

    async def _register(self, app_number: str, constituent: str) -> str:
        """Low-level call to the Register API.

        Args:
            app_number: EP application number (digits only, e.g. "12345678").
            constituent: One of "biblio", "events", "procedural-steps".

        Returns:
            Raw XML response string.
        """
        url = (
            f"{EPO_BASE}/register/application/epodoc/"
            f"EP{app_number}/{constituent}"
        )
        return await self._get_xml(url, purpose=f"register/{constituent}")

    async def register_biblio(self, app_number: str) -> EPORegisterBiblio | None:
        """Fetch bibliographic data from the EPO Register.

        Returns an ``EPORegisterBiblio`` or ``None`` if the application
        cannot be found.
        """
        try:
            xml_text = await self._register(app_number, "biblio")
        except EPOError:
            return None

        root = ET.fromstring(xml_text)
        rd = root.find(f"{{{_OPS_NS}}}register-document")
        if rd is None:
            return None

        biblio = _parse_register_biblio(rd, app_number)
        logger.info(
            f"epo_register_biblio — app={app_number}, "
            f"title={_truncate(biblio.title_en, 80)}, status={biblio.status}"
        )
        return biblio

    async def register_events(self, app_number: str) -> list[EPORegisterEvent]:
        """Fetch legal events from the EPO Register.

        Returns a list of ``EPORegisterEvent``, sorted by date descending.
        """
        try:
            xml_text = await self._register(app_number, "events")
        except EPOError:
            return []

        root = ET.fromstring(xml_text)
        events: list[EPORegisterEvent] = []
        for doc in root.findall(f"{{{_OPS_NS}}}register-document"):
            events.extend(_parse_register_events(doc))

        events.sort(key=lambda e: e.event_date, reverse=True)
        logger.info(
            f"epo_register_events — app={app_number}, count={len(events)}"
        )
        return events

    async def register_procedural_steps(
        self, app_number: str
    ) -> list[EPOProceduralStep]:
        """Fetch examination procedural steps from the EPO Register.

        Returns a list of ``EPOProceduralStep``, sorted by date descending.
        """
        try:
            xml_text = await self._register(app_number, "procedural-steps")
        except EPOError:
            return []

        root = ET.fromstring(xml_text)
        steps: list[EPOProceduralStep] = []
        for doc in root.findall(f"{{{_OPS_NS}}}register-document"):
            steps.extend(_parse_procedural_steps(doc))

        steps.sort(key=lambda s: s.step_date, reverse=True)
        logger.info(
            f"epo_procedural_steps — app={app_number}, count={len(steps)}"
        )
        return steps

    # ── Published Data API ────────────────────────────────────────────────────

    async def _published_data(self, pub_number: str, constituent: str) -> str:
        """Low-level call to the Published Data API.

        Args:
            pub_number: EP publication number (e.g. "4000000" for EP4000000).
            constituent: e.g. "biblio", "claims", "description".
        """
        url = (
            f"{EPO_BASE}/published-data/publication/epodoc/"
            f"EP{pub_number}/{constituent}"
        )
        return await self._get_xml(url, purpose=f"published-data/{constituent}")

    async def published_biblio(self, pub_number: str) -> dict[str, Any]:
        """Fetch bibliographic data for a published EP document.

        Returns the raw parsed data as a dict (keys: title, pub_date, kind, etc.).
        """
        try:
            xml_text = await self._published_data(pub_number, "biblio")
        except EPOError:
            return {}

        root = ET.fromstring(xml_text)
        bib_data = _parse_published_biblio(root)
        logger.info(
            f"epo_published_biblio — pub={pub_number}, "
            f"kind={bib_data.get('kind', '?')}"
        )
        return bib_data

    async def published_claims(self, pub_number: str) -> str:
        """Fetch claims text for a published EP document.

        Returns the claims as a plain-text string (XML tags stripped).
        """
        try:
            xml_text = await self._published_data(pub_number, "claims")
        except EPOError:
            return ""

        text = _strip_xml_tags(xml_text)
        logger.info(
            f"epo_published_claims — pub={pub_number}, chars={len(text)}"
        )
        return text

    async def published_description(self, pub_number: str) -> str:
        """Fetch description text for a published EP document.

        Returns the description as a plain-text string (XML tags stripped).
        Includes the Search Report section for A1/A3 publications.
        """
        try:
            xml_text = await self._published_data(pub_number, "description")
        except EPOError:
            return ""

        text = _strip_xml_tags(xml_text)
        logger.info(
            f"epo_published_description — pub={pub_number}, chars={len(text)}"
        )
        return text

    async def published_search_report_text(self, pub_number: str) -> str:
        """Extract the Search Report / Written Opinion text from a publication.

        For EP A1/A3 publications, the description endpoint returns the full
        specification including the search report.  For B1 (granted) documents
        the search report is not included — callers should look up the
        corresponding A1/A3 publication instead.

        Returns the search-report portion of the text, or the full description
        if we cannot isolate the search-report section.
        """
        text = await self.published_description(pub_number)
        if not text:
            return ""

        # Try to isolate the Search Report section.
        # In EP publications the search report / written opinion usually
        # appears after the description and claims, marked by headings like
        # "EUROPEAN SEARCH REPORT", "European search opinion", "SEARCH REPORT".
        _markers = [
            r"(?i)EUROPEAN\s+SEARCH\s+REPORT",
            r"(?i)European\s+search\s+opinion",
            r"(?i)SEARCH\s+REPORT",
            r"(?i)WRITTEN\s+OPINION.*SEARCH",
        ]
        for marker in _markers:
            m = re.search(marker, text)
            if m:
                # Return from the marker to end of text
                return text[m.start():]

        # No marker found — return the full text (may be a B1 without search report)
        logger.warning(
            f"epo_search_report — pub={pub_number}, "
            f"no search-report marker found in {len(text)} chars"
        )
        return ""


# ── XML Parsers ─────────────────────────────────────────────────────────────────


def _safe_int(element: ET.Element, tag: str, ns: str = _EXCHANGE_NS) -> int | None:
    child = element.find(f"{{{ns}}}{tag}")
    if child is not None and child.text:
        try:
            return int(child.text.strip())
        except (ValueError, TypeError):
            return None
    return None


def _find_biblio_element(root_or_doc: ET.Element) -> ET.Element | None:
    """Walk down to the <exchange:bibliographic-data> element.

    Handles both <ops:register-document> and <ops:exchange-documents> wrappers.
    """
    # Try direct <bibliographic-data> inside <register-document> or <exchange-document>
    for wrapper_tag in (
        f"{{{_EXCHANGE_NS}}}exchange-document",
        f"{{{_OPS_NS}}}bibliographic-data",
    ):
        wrapper = root_or_doc.find(wrapper_tag)
        if wrapper is not None:
            bd = wrapper.find(f"{{{_EXCHANGE_NS}}}bibliographic-data")
            if bd is None:
                bd = wrapper  # maybe wrapper IS the biblio data
            return bd

    # Try at root level
    bd = root_or_doc.find(f"{{{_EXCHANGE_NS}}}bibliographic-data")
    if bd is not None:
        return bd

    # Deep search
    for elem in root_or_doc.iter():
        if elem.tag == f"{{{_EXCHANGE_NS}}}bibliographic-data":
            return elem

    return None


def _parse_register_biblio(
    register_doc: ET.Element, app_number: str
) -> EPORegisterBiblio:
    """Parse a <ops:register-document> into EPORegisterBiblio."""
    biblio = EPORegisterBiblio(app_number=app_number)

    bd = _find_biblio_element(register_doc)
    if bd is None:
        return biblio

    # Publication reference
    pub_ref = bd.find(f"{{{_EXCHANGE_NS}}}publication-reference")
    if pub_ref is not None:
        pub_doc = pub_ref.find(
            f"{{{_EXCHANGE_NS}}}document-id[@document-id-type='docdb']"
        )
        if pub_doc is not None:
            biblio.pub_number = _text(pub_doc, "doc-number")
            biblio.pub_date = _text(pub_doc, "date")

    # Application reference
    app_ref = bd.find(f"{{{_EXCHANGE_NS}}}application-reference")
    if app_ref is not None:
        app_doc = app_ref.find(
            f"{{{_EXCHANGE_NS}}}document-id[@document-id-type='docdb']"
        )
        if app_doc is not None:
            biblio.filing_date = _text(app_doc, "date")

    # Title — prefer English
    for t in bd.findall(f"{{{_EXCHANGE_NS}}}invention-title"):
        lang = t.get("lang", "")
        text = (t.text or "").strip()
        if lang == "en" and not biblio.title_en:
            biblio.title_en = text
        if not biblio.title:
            biblio.title = text

    # IPC classes
    for ipc in bd.findall(f"{{{_EXCHANGE_NS}}}classification-ipc"):
        text = (ipc.text or "").strip()
        if text:
            biblio.ipc_classes.append(text)

    # Applicant / Assignee
    for party in bd.findall(f"{{{_EXCHANGE_NS}}}party"):
        # Look for applicant
        if party.get("change-type") == "applicant" or party.get("party-type") == "applicant":
            name = party.find(f"{{{_EXCHANGE_NS}}}party-name")
            if name is not None and name.text:
                biblio.applicant = name.text.strip()
                break

    # Status — try to determine from bibliographic data
    # Check if there's a B1 publication (grant)
    if biblio.pub_number:
        biblio.status = "GRANTED"  # presence in register implies published
    else:
        biblio.status = "PENDING"

    return biblio


def _parse_register_events(
    register_doc: ET.Element,
) -> list[EPORegisterEvent]:
    """Parse <exchange:events> from a register document."""
    events: list[EPORegisterEvent] = []

    events_elem = register_doc.find(f"{{{_EXCHANGE_NS}}}events")
    if events_elem is None:
        # Try inside exchange-document
        exc = register_doc.find(f"{{{_EXCHANGE_NS}}}exchange-document")
        if exc is not None:
            events_elem = exc.find(f"{{{_EXCHANGE_NS}}}events")

    if events_elem is None:
        return events

    for ev in events_elem.findall(f"{{{_EXCHANGE_NS}}}event"):
        code = ev.get("event-code", "")
        category = ev.get("category", "")

        # Dates
        date_elem = ev.find(f"{{{_EXCHANGE_NS}}}event-date")
        date_str = date_elem.text.strip() if date_elem is not None and date_elem.text else ""

        # Description
        desc_parts: list[tuple[str, str]] = []
        for desc in ev.findall(f"{{{_EXCHANGE_NS}}}event-description"):
            lang = desc.get("lang", "")
            text = (desc.text or "").strip()
            desc_parts.append((lang, text))

        desc_en = next((t for l, t in desc_parts if l == "en"), "")
        desc_all = "; ".join(t for _, t in desc_parts) if desc_parts else ""

        events.append(EPORegisterEvent(
            event_code=code,
            event_date=date_str,
            description=desc_all,
            description_en=desc_en,
            category=category,
        ))

    return events


def _parse_procedural_steps(
    register_doc: ET.Element,
) -> list[EPOProceduralStep]:
    """Parse <ops:procedural-steps> from a register document."""
    steps: list[EPOProceduralStep] = []

    ps_elem = register_doc.find(f"{{{_OPS_NS}}}procedural-steps")
    if ps_elem is None:
        # Try inside exchange-document
        exc = register_doc.find(f"{{{_EXCHANGE_NS}}}exchange-document")
        if exc is not None:
            ps_elem = exc.find(f"{{{_OPS_NS}}}procedural-steps")

    if ps_elem is None:
        return steps

    for ps in ps_elem.findall(f"{{{_OPS_NS}}}procedural-step"):
        code = ps.get("step-code", "") or ps.get("code", "")

        # Date
        date_elem = ps.find(f"{{{_OPS_NS}}}step-date")
        if date_elem is None:
            date_elem = ps.find(f"{{{_EXCHANGE_NS}}}event-date")
        date_str = date_elem.text.strip() if date_elem is not None and date_elem.text else ""

        # Description
        desc_parts: list[tuple[str, str]] = []
        for desc in ps.findall(f"{{{_OPS_NS}}}step-description"):
            lang = desc.get("lang", "")
            text = (desc.text or "").strip()
            desc_parts.append((lang, text))
        # Also try exchange namespace
        if not desc_parts:
            for desc in ps.findall(f"{{{_EXCHANGE_NS}}}event-description"):
                lang = desc.get("lang", "")
                text = (desc.text or "").strip()
                desc_parts.append((lang, text))

        desc_en = next((t for l, t in desc_parts if l == "en"), "")
        desc_all = "; ".join(t for _, t in desc_parts) if desc_parts else ""

        steps.append(EPOProceduralStep(
            step_code=code,
            step_date=date_str,
            description=desc_all,
            description_en=desc_en,
        ))

    return steps


def _parse_published_biblio(root: ET.Element) -> dict[str, Any]:
    """Parse a published-data biblio XML response into a flat dict."""
    result: dict[str, Any] = {}

    bd = _find_biblio_element(root)
    if bd is None:
        return result

    # Title
    for t in bd.findall(f"{{{_EXCHANGE_NS}}}invention-title"):
        lang = t.get("lang", "")
        if lang == "en":
            result.setdefault("title_en", (t.text or "").strip())
        result.setdefault("title", (t.text or "").strip())

    # Publication reference
    pub_ref = bd.find(f"{{{_EXCHANGE_NS}}}publication-reference")
    if pub_ref is not None:
        pub_doc = pub_ref.find(
            f"{{{_EXCHANGE_NS}}}document-id[@document-id-type='docdb']"
        )
        if pub_doc is not None:
            result["kind"] = _text(pub_doc, "kind")
            result["pub_date"] = _text(pub_doc, "date")
            result["pub_number"] = _text(pub_doc, "doc-number")

    # Abstract
    for ab in bd.findall(f"{{{_EXCHANGE_NS}}}abstract"):
        lang = ab.get("lang", "")
        for p in ab.findall(f"{{{_EXCHANGE_NS}}}p"):
            if p.text:
                key = f"abstract_{lang}" if lang else "abstract"
                result[key] = (p.text or "").strip()
                break

    return result


def _strip_xml_tags(xml_text: str) -> str:
    """Remove XML tags from *xml_text*, returning plain text.

    Handles the OPS XML wrapper that contains the actual document text
    inside <exchange:...> elements.  Strips all tag markup and collapses
    whitespace.
    """
    # Remove XML processing instructions and comments
    text = re.sub(r'<\?xml[^>]*\?>', '', xml_text)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # Strip all tags (keep their text content)
    text = re.sub(r'<[^>]+>', ' ', text)

    # Decode common XML entities
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&apos;', "'")

    # Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()
