#!/usr/bin/env python3
"""
Find 10 US granted patents from major tech companies that went through
a complete Non-Final Rejection cycle (CTNF → Amendment → NOA).

Outputs: company, patent number, app number, title, grant date.

Standalone — no dependency on the langsistance sources/ modules.
Only requires: requests (or httpx), and optionally python-dotenv.

Usage:
    # Add USPTO_API_KEY to .env, then:
    python3 scripts/find_non_final_rejection_patents.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any

# ── Logging to stdout ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("patent_finder")

# ── .env loading ────────────────────────────────────────────────────────────────

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv

    _env_file = os.path.join(_project_root, ".env")
    if os.path.isfile(_env_file):
        load_dotenv(_env_file)
    else:
        load_dotenv()
except ImportError:
    pass

# ── Configuration ──────────────────────────────────────────────────────────────

TARGET_COUNT = 10
PAGE_SIZE = 100
MAX_PAGES_PER_COMPANY = 15

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

CUTOFF_DATE = datetime.now(timezone.utc) - timedelta(days=5 * 365)

# USPTO rate limiting
USPTO_RETRY_DELAY = 1.0
USPTO_MAX_RETRIES = 10


# ── Document classification (inlined from prosecution_downloader.py) ────────────

def _matches_any_description(description: str, substrings: list[str]) -> bool:
    desc_lower = description.lower()
    return any(sub in desc_lower for sub in substrings)


def _classify_document(doc: dict) -> dict | None:
    """Classify a single USPTO documentBag entry into a category.

    Inlined from prosecution_downloader._classify_single_document().
    Returns dict with keys: category, priority, document_code, description;
    or None if the document lacks metadata.
    """
    code = str(doc.get("documentCode", "") or doc.get("documentTypeCode", "")).strip()
    desc = str(
        doc.get("documentCodeDescriptionText", "")
        or doc.get("documentTypeName", "")
    ).strip()

    if not code and not desc:
        return None

    # Priority 1 rules (inlined)
    p1 = {
        "office_action": {
            "codes": {"CTNF", "CTFR", "CTAV", "CTRS", "CTEQ"},
            "descriptions": [
                "non-final office action", "final office action", "office action",
                "restriction requirement", "non-final rejection", "final rejection",
                "ex parte quayle", "advisory action", "examiner's action",
            ],
        },
        "amendment": {
            "codes": {"CLM", "WCLM"},
            "descriptions": [
                "claims", "amendment", "preliminary amendment",
                "after final amendment", "amendment after final",
                "amendment under", "supplemental amendment", "amended",
            ],
        },
        "notice_of_allowance": {
            "codes": {"NOA"},
            "descriptions": [
                "notice of allowance", "notice of allowability", "issue notification",
            ],
        },
    }

    for category, rules in p1.items():
        codes = rules.get("codes", set())
        descriptions = rules.get("descriptions", [])
        if code.upper() in codes or _matches_any_description(desc, descriptions):
            return {
                "category": category,
                "priority": 1,
                "document_code": code.upper(),
                "description": desc,
            }

    # Default — everything else (we don't need priority 2/3 for filtering)
    return {
        "category": "other",
        "priority": 2,
        "document_code": code.upper(),
        "description": desc,
    }


# ── HTTP helpers ────────────────────────────────────────────────────────────────


def _get_headers(content_type: bool = True) -> dict[str, str]:
    headers: dict[str, str] = {"Accept": "application/json"}
    if content_type:
        headers["Content-Type"] = "application/json"
    key = os.getenv("USPTO_API_KEY", "").strip()
    if key:
        headers["X-API-Key"] = key
    return headers


def _http_post(url: str, json_body: dict, timeout: int = 30) -> Any:
    """POST with USPTO retry logic."""
    import requests

    headers = _get_headers()
    last_status = None
    for attempt in range(USPTO_MAX_RETRIES):
        try:
            resp = requests.post(url, headers=headers, json=json_body, timeout=timeout)
        except requests.RequestException as e:
            log.warning(f"HTTP POST error (attempt {attempt + 1}): {e}")
            time.sleep(USPTO_RETRY_DELAY)
            continue

        if resp.status_code in (400, 429) and attempt + 1 < USPTO_MAX_RETRIES:
            last_status = resp.status_code
            log.info(f"USPTO {resp.status_code} retry {attempt + 1}/{USPTO_MAX_RETRIES}")
            time.sleep(USPTO_RETRY_DELAY)
            continue

        if resp.status_code != 200:
            log.error(f"USPTO HTTP {resp.status_code} for {url[:100]}")
            return None

        return resp.json() if resp.text else {}

    log.error(f"USPTO retries exhausted (last status={last_status})")
    return None


def _http_get(url: str, timeout: int = 20) -> Any:
    """GET with USPTO retry logic."""
    import requests

    headers = _get_headers(content_type=False)
    last_status = None
    for attempt in range(USPTO_MAX_RETRIES):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            log.warning(f"HTTP GET error (attempt {attempt + 1}): {e}")
            time.sleep(USPTO_RETRY_DELAY)
            continue

        if resp.status_code in (400, 429) and attempt + 1 < USPTO_MAX_RETRIES:
            last_status = resp.status_code
            log.info(f"USPTO {resp.status_code} retry {attempt + 1}/{USPTO_MAX_RETRIES}")
            time.sleep(USPTO_RETRY_DELAY)
            continue

        if resp.status_code != 200:
            log.warning(f"USPTO HTTP {resp.status_code} for {url[:100]}")
            return None

        return resp.json() if resp.text else {}

    log.error(f"USPTO retries exhausted (last status={last_status})")
    return None


# ── USPTO API helpers ──────────────────────────────────────────────────────────


def _extract_items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
    return []


async def _search_uspto(
    assignee_query: str, offset: int = 0, limit: int = PAGE_SIZE
) -> list[dict[str, Any]]:
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
    data = await asyncio.to_thread(_http_post, USPTO_SEARCH_URL, search_body)
    return _extract_items(data) if data else []


async def _get_document_list(app_number: str) -> list[dict[str, Any]]:
    url = USPTO_DOCS_URL_TEMPLATE.format(app_number=app_number)
    data = await asyncio.to_thread(_http_get, url)
    if isinstance(data, dict):
        return data.get("documentBag", [])
    return []


def _has_all_document_types(document_bag: list[dict[str, Any]]) -> bool:
    has_ctnf = False
    has_amendment = False
    has_noa = False

    for doc in document_bag:
        if not isinstance(doc, dict):
            continue
        classified = _classify_document(doc)
        if classified is None:
            continue

        if (
            classified["category"] == "office_action"
            and classified["document_code"] == "CTNF"
        ):
            has_ctnf = True
        if classified["category"] == "amendment":
            has_amendment = True
        if classified["category"] == "notice_of_allowance":
            has_noa = True

        if has_ctnf and has_amendment and has_noa:
            return True

    return False


def _parse_grant_date(raw_date: str | None) -> datetime | None:
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
    matches: list[dict[str, Any]] = []

    for company in COMPANIES:
        if len(matches) >= TARGET_COUNT:
            break

        label = f"{company['name']} ({company['assignee']})"
        log.info(f"Searching: {label}")
        print(f"\n{'—' * 60}")
        print(f"Searching: {label}")
        print(f"Progress: {len(matches)}/{TARGET_COUNT}")

        patents_checked = 0

        for page in range(MAX_PAGES_PER_COMPANY):
            if len(matches) >= TARGET_COUNT:
                break

            offset = page * PAGE_SIZE
            results = await _search_uspto(company["assignee"], offset=offset)

            if not results:
                log.info(f"  No results page {page + 1}, company done.")
                break

            log.info(f"  Page {page + 1}: {len(results)} results")

            for item in results:
                if len(matches) >= TARGET_COUNT:
                    break
                if not isinstance(item, dict):
                    continue

                app_meta = item.get("applicationMetaData")
                if isinstance(app_meta, dict):
                    patent_number = app_meta.get("patentNumber", "")
                    title = app_meta.get("inventionTitle", "")
                    grant_date_raw = app_meta.get("patentGrantDate", "")
                else:
                    patent_number = item.get("patentNumber", "")
                    title = item.get("inventionTitle", "")
                    grant_date_raw = item.get("patentGrantDate", "")

                if not patent_number:
                    continue

                app_number = item.get("applicationNumberText", "") or item.get(
                    "appNumberText", ""
                )
                if not app_number:
                    app_number = "".join(
                        c for c in str(item.get("appNumber", "")) if c.isdigit()
                    )
                if not app_number:
                    continue

                grant_date = _parse_grant_date(grant_date_raw)
                if grant_date and grant_date < CUTOFF_DATE:
                    continue

                patents_checked += 1

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
                            grant_date.strftime("%Y-%m-%d") if grant_date else "unknown"
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
                    log.info(
                        f"MATCH #{len(matches)}: {company['name']} "
                        f"US{patent_number} / {app_number}"
                    )

            if len(results) < PAGE_SIZE:
                break

        print(f"  Checked {patents_checked} patents from {company['name']}")

    return matches


# ── Main ───────────────────────────────────────────────────────────────────────


async def main() -> None:
    print("=" * 80)
    print("Non-Final Rejection Patent Finder")
    print("Filter: CTNF + Amendment + NOA")
    print(f"Date range: granted >= {CUTOFF_DATE.strftime('%Y-%m-%d')}")
    print(f"Companies: {', '.join(c['name'] for c in COMPANIES)}")
    print(f"Target: {TARGET_COUNT} patents")
    print("=" * 80)

    api_key = os.getenv("USPTO_API_KEY", "").strip()
    if not api_key:
        print("\nERROR: USPTO_API_KEY not configured.")
        print("Add it to .env:  USPTO_API_KEY=your_key_here")
        sys.exit(1)

    matches = await find_patents()

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
        print("\nNo matching patents found.")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
