"""Recall-expansion sources for the patent search loop.

Two transports widen recall beyond the keyword ladder:

- family fetch — USPTO applications/search free-text q with the patent
  / application numbers collected from the pool candidates' continuity
  bags (childContinuityBag / parentContinuityBag);
- CPC fetch — the local MCF index (built monthly by
  scripts/build_cpc_index.py) resolves the matched CPC codes to patent
  numbers; their metadata comes through the same number search.

Every function degrades to empty results on any failure — recall
expansion is an enhancement, never a hard dependency.
"""

import os
from typing import Any

from sources.http_outbound import outbound_http
from sources.logger import Logger
from sources.long_task.candidate_metadata import build_candidates

logger = Logger("recall_sources.log")

USPTO_SEARCH_URL = "https://api.uspto.gov/api/v1/patent/applications/search"
CPC_INDEX_DB = os.path.join(
    os.getenv("CPC_DATA_DIR", "data/cpc"), "cpc_index.db")

# Enough metadata to build candidates and continue the family chain.
RECALL_SEARCH_FIELDS = [
    "applicationNumberText",
    "applicationMetaData.inventionTitle",
    "applicationMetaData.firstApplicantName",
    "applicationMetaData.applicationStatusDescriptionText",
    "applicationMetaData.filingDate",
    "applicationMetaData.grantDate",
    "applicationMetaData.patentNumber",
    "applicationMetaData.cpcClassificationBag",
    "parentContinuityBag",
    "childContinuityBag",
]

MAX_FAMILY_NUMBERS = int(os.getenv("REACT_RECALL_MAX_FAMILY_NUMBERS", "12"))
RECALL_NUMBER_BATCH = int(os.getenv("REACT_RECALL_NUMBER_BATCH", "20"))
RECALL_CPC_PER_CODE = int(os.getenv("REACT_RECALL_CPC_PER_CODE", "50"))


def collect_family_refs(candidates: list, limit: int = MAX_FAMILY_NUMBERS) -> dict:
    """Collect family member numbers from candidates' continuity bags.

    Returns {"patents": [...], "applications": [...]} deduped and capped
    at *limit* each.  Numbers already held by the pool (application
    number or granted patent number of any candidate) are excluded.
    Pure — never raises.
    """
    known = set()
    for c in candidates or []:
        if not isinstance(c, dict):
            continue
        for key in ("patent_id", "patent_number"):
            v = c.get(key)
            if v:
                known.add(str(v).strip())
    patents: list = []
    applications: list = []
    for c in candidates or []:
        if not isinstance(c, dict):
            continue
        raw = c.get("_raw")
        if not isinstance(raw, dict):
            continue
        for bag_key, num_key, app_key in (
                ("childContinuityBag", "childPatentNumber",
                 "childApplicationNumberText"),
                ("parentContinuityBag", "parentPatentNumber",
                 "parentApplicationNumberText")):
            bag = raw.get(bag_key)
            if not isinstance(bag, list):
                continue
            for entry in bag:
                if not isinstance(entry, dict):
                    continue
                num = str(entry.get(num_key) or "").strip()
                app = str(entry.get(app_key) or "").strip()
                if num and num not in known and num not in patents:
                    patents.append(num)
                if app and app not in known and app not in applications:
                    applications.append(app)
    return {"patents": patents[:limit], "applications": applications[:limit]}


def records_to_candidates(records: list) -> list[dict]:
    """Convert recalled patent records into pool candidates.

    USPTO-shaped items (applicationNumberText + applicationMetaData)
    flatten via build_candidates; minimal records only need
    patent_number / patent_title / assignee / dates.  Unknown shapes and
    records without an identifiable number are skipped.  Pure — never
    raises.
    """
    candidates = build_candidates(records)
    seen = {c["patent_id"] for c in candidates}
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        num = str(rec.get("patent_number")
                  or rec.get("patentNumber")
                  or rec.get("patent_id") or "").strip()
        if not num or num in seen:
            continue
        candidates.append({
            "patent_id": num,
            "title": str(rec.get("title")
                         or rec.get("patent_title")
                         or rec.get("inventionTitle") or "").strip(),
            "applicant": str(rec.get("assignee")
                             or rec.get("assignee_organization")
                             or rec.get("applicant") or "").strip(),
            "status": str(rec.get("status") or "").strip(),
            "filing_date": str(rec.get("filing_date")
                               or rec.get("patent_date") or "").strip(),
            "grant_date": "",
            "patent_number": num,
            "type_code": "",
            "cpc_codes": [],
            "_raw": rec,
        })
        seen.add(num)
    return candidates


def fetch_by_numbers(numbers: list, timeout: int = 30) -> list:
    """Fetch USPTO records for the given patent/application numbers via
    free-text OR queries, chunked in batches of
    RECALL_NUMBER_BATCH — the query string must stay comfortably short
    for the API.  [] on any failure; one failed chunk never hides the
    others."""
    numbers = [str(n).strip() for n in (numbers or []) if str(n).strip()]
    if not numbers:
        return []
    headers = {"Content-Type": "application/json"}
    uspto_key = os.getenv("USPTO_API_KEY")
    if uspto_key:
        headers["X-API-Key"] = uspto_key
    items: list = []
    for start in range(0, len(numbers), RECALL_NUMBER_BATCH):
        chunk = numbers[start:start + RECALL_NUMBER_BATCH]
        q = " OR ".join(f'"{n}"' for n in chunk)
        body: dict[str, Any] = {
            "q": q,
            "pagination": {"offset": 0, "limit": len(chunk) + 2},
            "fields": RECALL_SEARCH_FIELDS,
            "sort": [{"field": "_score", "order": "desc"}],
        }
        try:
            response = outbound_http.request(
                "POST", USPTO_SEARCH_URL, purpose="recall_family",
                headers=headers, json=body, timeout=timeout)
            if getattr(response, "status_code", 0) != 200:
                logger.warning(
                    f"recall family fetch failed — "
                    f"status={getattr(response, 'status_code', None)}")
                continue
            data = response.json()
            batch_items = data.get("patentFileWrapperDataBag") or []
            items.extend(batch_items)
            logger.info(f"recall family fetch — q={q[:60]!r} "
                        f"hits={len(batch_items)}")
        except Exception as exc:
            logger.warning(f"recall family fetch failed: {exc}")
    return items


def fetch_by_cpc(codes: list, timeout: int = 30) -> list:
    """Resolve CPC recall via the local MCF index (built monthly by
    scripts/build_cpc_index.py), then fetch the matching patents'
    metadata through the number search.

    Main-group hints (/00) match all their subgroups; subgroup hints
    match exactly.  Newest patents are preferred.  [] when the index is
    absent or nothing matches — recall expansion must never hard-fail.
    """
    codes = [str(c).strip() for c in (codes or []) if str(c).strip()]
    if not codes:
        return []
    if not os.path.exists(CPC_INDEX_DB):
        logger.warning(
            "recall cpc index absent — run scripts/build_cpc_index.py "
            f"(expected at {CPC_INDEX_DB})")
        return []
    patents: list = []
    seen = set()
    try:
        import sqlite3
        conn = sqlite3.connect(CPC_INDEX_DB)
        try:
            for code in codes:
                base = code.split("/")[0]
                if code.endswith("/00") and len(base) >= 4:
                    rows = conn.execute(
                        "SELECT DISTINCT patent FROM cpc_patents "
                        "WHERE cpc LIKE ? "
                        "ORDER BY length(patent) DESC, patent DESC LIMIT ?",
                        (base + "/%", RECALL_CPC_PER_CODE))
                else:
                    rows = conn.execute(
                        "SELECT DISTINCT patent FROM cpc_patents "
                        "WHERE cpc = ? "
                        "ORDER BY length(patent) DESC, patent DESC LIMIT ?",
                        (code, RECALL_CPC_PER_CODE))
                for (patent,) in rows:
                    if patent not in seen:
                        seen.add(patent)
                        patents.append(patent)
        finally:
            conn.close()
    except Exception as exc:
        logger.warning(f"cpc index lookup failed: {exc}")
        return []
    if not patents:
        return []
    logger.info(f"recall cpc fetch — codes={codes} "
                f"index_hits={len(patents)}")
    return fetch_by_numbers(patents, timeout=timeout)
