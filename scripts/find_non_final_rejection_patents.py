#!/usr/bin/env python3
"""
Find 10 US granted patents from major tech companies that went through
a complete Non-Final Rejection cycle (CTNF → Amendment → NOA).

Outputs: company, patent number, app number, title, grant date.

Reuses existing USPTO infrastructure:
- outbound_http client (retry + rate limiting)
- _classify_single_document (patent document type classification)

Usage:
    # Add USPTO_API_KEY to .env, then:
    python scripts/find_non_final_rejection_patents.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Load .env (same pattern as api.py and backfill_knowledge_embeddings.py)
try:
    from dotenv import load_dotenv

    _env_file = os.path.join(_project_root, ".env")
    if os.path.isfile(_env_file):
        load_dotenv(_env_file)
    else:
        load_dotenv()
except ImportError:
    pass

from sources.http_outbound import outbound_http
from sources.long_task.prosecution_downloader import _classify_single_document
from sources.logger import Logger

logger = Logger("patent_finder.log")

# ── Configuration ──────────────────────────────────────────────────────────────

TARGET_COUNT = 10
PAGE_SIZE = 100
MAX_PAGES_PER_COMPANY = 15  # up to 1500 per company

# Each entry: display name + USPTO Lucene assignee query string.
# Using exact-match phrases with the company's current legal entity name.
COMPANIES: list[dict[str, str]] = [
    {"name": "Apple", "assignee": '"Apple Inc."'},
    {"name": "Tesla", "assignee": '"Tesla Inc."'},
    {"name": "NVIDIA", "assignee": '"NVIDIA Corporation"'},
    {"name": "SpaceX", "assignee": '"Space Exploration Technologies Corp."'},
    {"name": "Samsung", "assignee": '"Samsung Electronics Co. Ltd."'},
    {"name": "Qualcomm", "assignee": '"Qualcomm Incorporated"'},
]

USPTO_SEARCH_URL = "https://api.uspto.gov/api/v1/patent/applications/search"
USPTO_DOCS_URL_TEMPLATE = (
    "https://api.uspto.gov/api/v1/patent/applications/{app_number}/documents"
)

# Only consider patents granted on or after this date (5 years ago)
CUTOFF_DATE = datetime.now(timezone.utc) - timedelta(days=5 * 365)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _get_headers(content_type: bool = True) -> dict[str, str]:
    """Build request headers with USPTO API key from .env."""
    headers: dict[str, str] = {"Accept": "application/json"}
    if content_type:
        headers["Content-Type"] = "application/json"
    uspto_key = os.getenv("USPTO_API_KEY", "").strip()
    if uspto_key:
        headers["X-API-Key"] = uspto_key
    return headers


def _extract_items(data: Any) -> list[dict[str, Any]]:
    """Extract a list of items from a USPTO API response.

    The API wraps results in various top-level keys (e.g. 'patentApplications',
    'applicationBag', 'results'); this scans for the first non-empty list value.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                items: list[dict[str, Any]] = v
                return items
    return []


async def _search_uspto(
    assignee_query: str, offset: int = 0, limit: int = PAGE_SIZE
) -> list[dict[str, Any]]:
    """Search USPTO for patents matching an assignee Lucene query.

    Returns a list of patent application records (may include both pending and
    granted applications — callers should filter by patentNumber presence).
    """
    search_body = {
        "q": f"assignee:{assignee_query}",
        "pagination": {"offset": offset, "limit": limit},
        "fields": [
            "applicationNumberText",
            "applicationMetaData.patentNumber",
            "applicationMetaData.inventionTitle",
            "applicationMetaData.patentGrantDate",
        ],
    }

    try:
        resp = await asyncio.to_thread(
            outbound_http.post,
            USPTO_SEARCH_URL,
            purpose="patent_search",
            headers=_get_headers(),
            json=search_body,
            timeout=30,
        )
    except Exception as exc:
        logger.error(f"USPTO search error: {type(exc).__name__}: {exc}")
        return []

    if resp.status_code != 200:
        logger.error(
            f"USPTO search HTTP {resp.status_code} — "
            f"assignee={assignee_query}, offset={offset}"
        )
        return []

    data = resp.json() if resp.text else {}
    return _extract_items(data)


async def _get_document_list(app_number: str) -> list[dict[str, Any]]:
    """Fetch the document bag for a patent application."""
    url = USPTO_DOCS_URL_TEMPLATE.format(app_number=app_number)

    try:
        resp = await asyncio.to_thread(
            outbound_http.get,
            url,
            purpose="patent_docs",
            headers=_get_headers(content_type=False),
            timeout=20,
        )
    except Exception as exc:
        logger.warning(
            f"Document list error for {app_number}: {type(exc).__name__}: {exc}"
        )
        return []

    if resp.status_code != 200:
        logger.warning(
            f"Document list HTTP {resp.status_code} for {app_number}"
        )
        return []

    data = resp.json() if resp.text else {}
    if isinstance(data, dict):
        return data.get("documentBag", [])
    return []


def _has_all_document_types(document_bag: list[dict[str, Any]]) -> bool:
    """Return True if the document bag contains CTNF + Amendment + NOA.

    Uses the project's existing _classify_single_document() so the matching
    rules stay identical to production prosecution analysis.
    """
    has_ctnf = False
    has_amendment = False
    has_noa = False

    for doc in document_bag:
        if not isinstance(doc, dict):
            continue
        classified = _classify_single_document(doc)
        if classified is None:
            continue

        # CTNF: office_action category with the specific document code
        if (
            classified.category == "office_action"
            and classified.document_code.upper() == "CTNF"
        ):
            has_ctnf = True

        # Amendment: claims filed in response to examination
        if classified.category == "amendment":
            has_amendment = True

        # NOA: notice of allowance = grant imminent
        if classified.category == "notice_of_allowance":
            has_noa = True

        if has_ctnf and has_amendment and has_noa:
            return True

    return False


def _parse_grant_date(raw_date: str | None) -> datetime | None:
    """Parse a USPTO grant-date string to a timezone-aware UTC datetime."""
    if not raw_date:
        return None
    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%m/%d/%Y",
    ):
        try:
            return datetime.strptime(raw_date, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


# ── Main loop ──────────────────────────────────────────────────────────────────


async def find_patents() -> list[dict[str, Any]]:
    """Search across companies until TARGET_COUNT matches are found."""
    matches: list[dict[str, Any]] = []

    for company in COMPANIES:
        if len(matches) >= TARGET_COUNT:
            break

        label = f"{company['name']} ({company['assignee']})"
        logger.info(f"search_start — company={company['name']}")
        print(f"\n{'—' * 60}")
        print(f"Searching: {label}")
        print(f"Progress: {len(matches)}/{TARGET_COUNT} found so far")

        patents_checked = 0

        for page in range(MAX_PAGES_PER_COMPANY):
            if len(matches) >= TARGET_COUNT:
                break

            offset = page * PAGE_SIZE
            results = await _search_uspto(company["assignee"], offset=offset)

            if not results:
                logger.info(
                    f"search_empty — company={company['name']}, page={page + 1}"
                )
                break

            logger.info(
                f"search_page — company={company['name']}, "
                f"page={page + 1}, count={len(results)}"
            )

            for item in results:
                if len(matches) >= TARGET_COUNT:
                    break
                if not isinstance(item, dict):
                    continue

                # ── Extract grant number ──
                app_meta = item.get("applicationMetaData")
                if isinstance(app_meta, dict):
                    patent_number = app_meta.get("patentNumber", "")
                    title = app_meta.get("inventionTitle", "")
                    grant_date_raw = app_meta.get("patentGrantDate", "")
                else:
                    patent_number = item.get("patentNumber", "")
                    title = item.get("inventionTitle", "")
                    grant_date_raw = item.get("patentGrantDate", "")

                # Skip non-granted applications
                if not patent_number:
                    continue

                # ── Extract application number ──
                app_number = item.get("applicationNumberText", "") or item.get(
                    "appNumberText", ""
                )
                if not app_number:
                    app_number = "".join(
                        c for c in str(item.get("appNumber", "")) if c.isdigit()
                    )
                if not app_number:
                    continue

                # ── Date filter (client-side, safer than relying on API query) ──
                grant_date = _parse_grant_date(grant_date_raw)
                if grant_date and grant_date < CUTOFF_DATE:
                    continue

                patents_checked += 1

                # ── Fetch documents and classify ──
                documents = await _get_document_list(app_number)
                if not documents:
                    continue

                if _has_all_document_types(documents):
                    match = {
                        "company": company["name"],
                        "patent_number": patent_number,
                        "app_number": app_number,
                        "title": title,
                        "grant_date": (
                            grant_date.strftime("%Y-%m-%d")
                            if grant_date
                            else "unknown"
                        ),
                    }
                    matches.append(match)
                    print(
                        f"  ✅ #{len(matches):>2}  "
                        f"US{match['patent_number']}  |  "
                        f"{match['app_number']}  |  "
                        f"{match['grant_date']}  |  "
                        f"{match['title'][:70]}"
                    )
                    logger.info(
                        f"match — #{len(matches)} company={company['name']} "
                        f"patent={patent_number} app={app_number} "
                        f"grant_date={match['grant_date']}"
                    )

            # If API returned fewer results than page size, this company is done
            if len(results) < PAGE_SIZE:
                logger.info(
                    f"search_exhausted — company={company['name']}, "
                    f"total_checked={patents_checked}"
                )
                break

        print(f"  Checked {patents_checked} patents from {company['name']}")

    return matches


# ── Main ───────────────────────────────────────────────────────────────────────


async def main() -> None:
    print("=" * 80)
    print("Non-Final Rejection Patent Finder")
    print("Filter: CTNF (Non-Final Rejection) + Amendment + NOA (Notice of Allowance)")
    print(f"Date range: granted >= {CUTOFF_DATE.strftime('%Y-%m-%d')} (last 5 years)")
    print(f"Companies: {', '.join(c['name'] for c in COMPANIES)}")
    print(f"Target: {TARGET_COUNT} patents")
    print("=" * 80)

    api_key = os.getenv("USPTO_API_KEY", "").strip()
    if not api_key:
        print("\nERROR: USPTO_API_KEY not configured.")
        print("Add it to the .env file:")
        print("    USPTO_API_KEY=your_key_here")
        sys.exit(1)

    matches = await find_patents()

    # ── Final report ──
    print()
    print("=" * 100)
    print(f"RESULTS  —  {len(matches)} patents with CTNF + Amendment + NOA")
    print("=" * 100)

    if matches:
        print()
        print(
            f"{'#':<4} {'Company':<12} {'Patent #':<14} {'App #':<12} "
            f"{'Grant Date':<12} Title"
        )
        print("-" * 100)
        for i, m in enumerate(matches, 1):
            print(
                f"{i:<4} {m['company']:<12} US{m['patent_number']:<13} "
                f"{m['app_number']:<12} {m['grant_date']:<12} {m['title'][:60]}"
            )
        print()
        print(f"Total: {len(matches)} patents found.")
    else:
        print("\nNo matching patents found across all companies.")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
