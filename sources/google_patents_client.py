"""Google Patents page scraping client.

Extracts patent full text (claims, description, abstract, bibliographic data)
from patents.google.com HTML pages.  No API key or authentication required.

Rate limiting is enforced on a shared httpx.AsyncClient to stay within
Google's acceptable-use thresholds for public patent data access.

Reference: https://patents.google.com/patent/{pub_number}/{lang}
            robots.txt allows ``/patent/`` path.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx
from bs4 import BeautifulSoup

from sources.logger import Logger

_logger = Logger("google_patents.log")

# ── Constants ───────────────────────────────────────────────────────────────────

GOOGLE_PATENTS_BASE = "https://patents.google.com/patent"
DEFAULT_DELAY_SECONDS = 2.0
REQUEST_TIMEOUT = 30
BACKOFF_DELAY_SECONDS = 10.0


# ── Error types ─────────────────────────────────────────────────────────────────


class GooglePatentsError(Exception):
    """Base error for Google Patents scraping failures."""


class GooglePatentsNotFoundError(GooglePatentsError):
    """Patent page returned 404 — publication number may be invalid."""


class GooglePatentsRateLimitError(GooglePatentsError):
    """Persistent 429 — rate limiting triggered even after backoff."""


# ── Client ──────────────────────────────────────────────────────────────────────


class GooglePatentsClient:
    """Async scraper for patents.google.com patent detail pages.

    Uses a single shared ``httpx.AsyncClient`` for connection pooling.
    Enforces a configurable delay between requests to avoid rate limiting.

    Usage::

        client = GooglePatentsClient(delay=2.0)
        claims = await client.query_claims("CN107041743B", lang="zh")
        desc   = await client.query_description("CN107041743B", lang="zh")
        info   = await client.query_basic_info("CN107041743B", lang="zh")
        await client.close()
    """

    def __init__(self, delay: float = DEFAULT_DELAY_SECONDS) -> None:
        self._delay = delay
        self._last_request_time: float = 0.0
        self._client: httpx.AsyncClient | None = None
        self._headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; PatentAnalysis/1.0; "
                "+https://copiioai.com)"
            ),
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

    # ── Public API ────────────────────────────────────────────────────────────

    async def query_claims(
        self, pub_number: str, lang: str = "zh",
    ) -> list[str]:
        """Extract patent claims.

        Returns:
            List of claim texts, one per claim (including dependent claims).
            Empty list if the claims section is not found.
        """
        html = await self._fetch_page(pub_number, lang)
        soup = self._parse_html(html)
        claims_div = soup.find("div", class_="claims")
        if not claims_div:
            _logger.warning(
                f"google_patents: no claims section for {pub_number}"
            )
            return []

        claims: list[str] = []
        for claim_block in claims_div.find_all("div", class_="claim"):
            texts: list[str] = []
            for ct in claim_block.find_all("div", class_="claim-text"):
                t = ct.get_text(" ", strip=True)
                if t:
                    texts.append(t)
            if texts:
                claims.append(" ".join(texts))

        _logger.info(
            f"google_patents: claims — pub={pub_number}, count={len(claims)}"
        )
        return claims

    async def query_description(
        self, pub_number: str, lang: str = "zh",
    ) -> list[str]:
        """Extract patent description / specification.

        Returns:
            List of description paragraphs.  Empty list if not found.
        """
        html = await self._fetch_page(pub_number, lang)
        soup = self._parse_html(html)
        desc_div = soup.find("div", class_="description")
        if not desc_div:
            _logger.warning(
                f"google_patents: no description section for {pub_number}"
            )
            return []

        paras = desc_div.find_all("div", class_="description-paragraph")
        texts = [p.get_text(" ", strip=True) for p in paras]
        texts = [t for t in texts if t]  # filter empty
        _logger.info(
            f"google_patents: description — pub={pub_number}, "
            f"paragraphs={len(texts)}"
        )
        return texts

    async def query_abstract(
        self, pub_number: str, lang: str = "zh",
    ) -> str:
        """Extract patent abstract."""
        html = await self._fetch_page(pub_number, lang)
        soup = self._parse_html(html)
        abstract_div = soup.find("div", class_="abstract")
        if not abstract_div:
            return ""
        return abstract_div.get_text(" ", strip=True)

    async def query_basic_info(
        self, pub_number: str, lang: str = "zh",
    ) -> dict[str, Any]:
        """Extract bibliographic / meta information.

        Returns a dict whose keys mirror SIPOP's ``queryBasicInfo`` response
        as closely as possible::

            {
                "title": str,
                "applicationDocNum": str,
                "publicationDocNum": str,
                "applicationDate": str,       # YYYYMMDD
                "publicationDate": str,        # YYYYMMDD
                "applicant": [str, ...],
                "inventor": [str, ...],
                "abstractDesc": str,
                "ipcMain": str,                # may be empty
            }
        """
        html = await self._fetch_page(pub_number, lang)
        soup = self._parse_html(html)

        # ── Title ──────────────────────────────────────────────────────────
        title = ""
        title_meta = soup.find("meta", attrs={"name": "DC.title"})
        if title_meta and title_meta.get("content"):
            title = title_meta["content"].strip()

        # ── Dates (from meta) ──────────────────────────────────────────────
        app_date = ""
        pub_date = ""
        for meta in soup.find_all("meta"):
            name = meta.get("name", "")
            scheme = meta.get("scheme", "")
            content = meta.get("content", "")
            if name == "DC.date":
                if scheme == "dateSubmitted":
                    app_date = content.replace("-", "")  # 2017-04-05 → 20170405
                elif scheme == "":
                    pub_date = content.replace("-", "")

        # ── Inventor & Applicant (from meta) ───────────────────────────────
        inventors: list[str] = []
        applicants: list[str] = []
        for meta in soup.find_all("meta"):
            name = meta.get("name", "")
            scheme = meta.get("scheme", "")
            content = meta.get("content", "")
            if name == "DC.contributor":
                if scheme == "inventor":
                    inventors.append(content.strip())
                elif scheme == "assignee":
                    applicants.append(content.strip())

        # ── Abstract ───────────────────────────────────────────────────────
        abstract = ""
        abstract_div = soup.find("div", class_="abstract")
        if abstract_div:
            abstract = abstract_div.get_text(" ", strip=True)

        # ── Publication number ─────────────────────────────────────────────
        pub_doc_num = pub_number
        pn_meta = soup.find("meta", attrs={"name": "citation_patent_number"})
        if pn_meta and pn_meta.get("content"):
            pub_doc_num = pn_meta["content"].strip()

        result = {
            "title": title,
            "applicationDocNum": "",  # Google Patents pages don't reliably expose app number
            "publicationDocNum": pub_doc_num,
            "applicationDate": app_date,
            "publicationDate": pub_date,
            "applicant": applicants,
            "inventor": inventors,
            "abstractDesc": abstract,
            "ipcMain": "",  # Google Patents HTML doesn't expose IPC directly
        }

        _logger.info(
            f"google_patents: basic_info — pub={pub_number}, "
            f"title={title[:60] if title else '(none)'}"
        )
        return result

    async def query_legal_events(
        self, pub_number: str, lang: str = "en",
    ) -> list[dict[str, str]]:
        """Extract legal-status event timeline from the *Legal Events* table.

        Each event includes a date, a WIPO ST.36 event code, and a
        human-readable title.  Duplicate rows (an artefact of the page
        markup) are collapsed.

        Returns a list of dicts::

            [
                {"date": "2017-08-15", "code": "PB01", "title": "Publication"},
                ...
            ]

        An empty list is returned when the patent page has no *Legal Events*
        section (common for older CN patents).
        """
        html = await self._fetch_page(pub_number, lang)
        soup = self._parse_html(html)
        section_heading = soup.find("h2", string="Legal Events")
        if not section_heading:
            _logger.info(
                f"google_patents: no Legal Events section for {pub_number}"
            )
            return []

        section = section_heading.find_parent("section")
        if not section:
            return []

        table = section.find("table")
        if not table:
            return []

        rows = table.find_all("tr", itemprop="legalEvents")
        events: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for tr in rows:
            date_el = tr.find("time", itemprop="date")
            code_el = tr.find(attrs={"itemprop": "code"})
            title_el = tr.find(attrs={"itemprop": "title"})

            date_val = date_el.get("datetime", "") if date_el else ""
            code_val = code_el.text.strip() if code_el else ""
            title_val = title_el.text.strip() if title_el else ""

            key = (date_val, code_val)
            if key not in seen:
                seen.add(key)
                events.append({
                    "date": date_val,
                    "code": code_val,
                    "title": title_val,
                })

        _logger.info(
            f"google_patents: legal_events — pub={pub_number}, "
            f"count={len(events)}"
        )
        return events

    async def query_legal_status_text(
        self, pub_number: str, lang: str = "en",
    ) -> str:
        """Extract the single-line legal-status summary (e.g. *Active*, *Expired*).

        This is the IFI-assigned status shown near the top of the patent page,
        **not** the full event timeline.  For the timeline use
        :meth:`query_legal_events`.

        Returns an empty string when the page has no legal-status markup
        or the patent page does not exist (404).
        """
        try:
            html = await self._fetch_page(pub_number, lang)
        except GooglePatentsNotFoundError:
            _logger.info(
                f"google_patents: legal_status_text — "
                f"{pub_number} not found (404)"
            )
            return ""
        soup = self._parse_html(html)
        # The legal-status is on a <dd itemprop="legalStatusIfi">
        dd = soup.find("dd", itemprop="legalStatusIfi")
        if dd:
            span = dd.find("span", itemprop="status")
            if span:
                return span.text.strip()
        return ""

    async def close(self) -> None:
        """Close the shared HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-init the shared httpx AsyncClient."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(REQUEST_TIMEOUT),
                headers=self._headers,
                follow_redirects=True,
            )
        return self._client

    async def _rate_limit(self) -> None:
        """Enforce minimum delay between requests."""
        loop = asyncio.get_running_loop()
        now = loop.time()
        elapsed = now - self._last_request_time
        if elapsed < self._delay:
            await asyncio.sleep(self._delay - elapsed)
        self._last_request_time = loop.time()

    async def _fetch_page(self, pub_number: str, lang: str) -> str:
        """Fetch a patent detail page with rate limiting and retry."""
        await self._rate_limit()

        client = await self._get_client()
        url = f"{GOOGLE_PATENTS_BASE}/{pub_number}/{lang}"
        _logger.info(f"google_patents: GET {url}")

        resp = await client.get(url)

        # Single retry with longer backoff on 429
        if resp.status_code == 429:
            _logger.warning(
                f"google_patents: 429 rate limited for {pub_number}, "
                f"backing off {BACKOFF_DELAY_SECONDS}s"
            )
            await asyncio.sleep(BACKOFF_DELAY_SECONDS)
            resp = await client.get(url)

        if resp.status_code == 404:
            raise GooglePatentsNotFoundError(
                f"Patent {pub_number} not found on Google Patents (404)"
            )
        if resp.status_code == 429:
            raise GooglePatentsRateLimitError(
                f"Persistent 429 — Google Patents rate limiting {pub_number}"
            )
        if resp.status_code != 200:
            raise GooglePatentsError(
                f"Google Patents HTTP {resp.status_code} for {pub_number}: "
                f"{resp.text[:200]}"
            )

        return resp.text

    @staticmethod
    def _parse_html(html: str) -> BeautifulSoup:
        return BeautifulSoup(html, "html.parser")
