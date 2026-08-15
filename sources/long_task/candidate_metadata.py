"""Candidate extraction and enrichment for USPTO search results.

raw_items from the USPTO applications/search response are
``patentFileWrapperDataBag`` entries. This module flattens the useful
metadata (title, applicant, dates, status, CPC codes) into compact
candidate dicts used by the relevance gate, dedupe, and report.
"""

import re
from copy import deepcopy
from typing import Any

# USPTO response fields worth requesting when they are not already in
# the tool's fields template.
SEARCH_FIELDS_TO_ENSURE = [
    "applicationMetaData.inventionTitle",
    "applicationMetaData.firstApplicantName",
    "applicationMetaData.applicationStatusDescriptionText",
    "applicationMetaData.filingDate",
    "applicationMetaData.grantDate",
    "applicationMetaData.patentNumber",
    "applicationMetaData.cpcClassificationBag",
    "parentContinuityBag",
]


def is_keyword_search_tool(tool: Any) -> bool:
    """True when the tool's title indicates a keyword search tool."""
    title = (getattr(tool, "title", "") or "").lower()
    return "key" in title or "keyword" in title


def is_uspto_tool(tool: Any) -> bool:
    """True when the tool's URL targets api.uspto.gov."""
    url = (getattr(tool, "url", "") or "").lower()
    return "uspto" in url


def ensure_search_fields(params: dict) -> dict:
    """Return a deep copy of tool params with required USPTO fields added.

    Only touches ``body.fields``. The input dict is never mutated.
    """
    out = deepcopy(params)
    body = out.get("body") if isinstance(out.get("body"), dict) else {}
    fields = body.get("fields")
    if not isinstance(fields, list):
        return out
    for f in SEARCH_FIELDS_TO_ENSURE:
        if f not in fields:
            fields.append(f)
    return out


def _first_str(d: dict, *keys: str) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _meta(item: dict) -> dict:
    m = item.get("applicationMetaData")
    if not isinstance(m, dict):
        m = {}
    return m


def _extract_cpc_codes(m: dict) -> list[str]:
    """Defensively collect CPC codes from applicationMetaData."""
    codes: list[str] = []
    bag = m.get("cpcClassificationBag") or m.get("classificationBag")
    if isinstance(bag, list):
        for entry in bag:
            if isinstance(entry, dict):
                for k in ("cpcClassCode", "classificationCode", "cpcCode"):
                    v = entry.get(k)
                    if isinstance(v, str) and v.strip():
                        codes.append(v.strip())
                        break
    return list(dict.fromkeys(codes))


def build_candidates(raw_items: list) -> list[dict]:
    """Flatten USPTO raw_items into compact candidate dicts."""
    candidates = []
    for item in raw_items or []:
        if not isinstance(item, dict):
            continue
        m = _meta(item)
        pid = str(m.get("applicationNumberText") or "").strip()
        if not pid:
            continue
        candidates.append({
            "patent_id": pid,
            "title": _first_str(m, "inventionTitle"),
            "applicant": _first_str(m, "firstApplicantName"),
            "status": _first_str(m, "applicationStatusDescriptionText"),
            "filing_date": _first_str(m, "filingDate"),
            "grant_date": _first_str(m, "grantDate"),
            "patent_number": _first_str(m, "patentNumber"),
            "cpc_codes": _extract_cpc_codes(m),
            "_raw": item,
        })
    return candidates


# ── Dedupe ──────────────────────────────────────────────────────────────────

def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (title or "").lower())


def _continuity_ids(item: dict) -> set[str]:
    """Collect application numbers referenced by parentContinuityBag."""
    ids: set[str] = set()
    bag = item.get("parentContinuityBag")
    if isinstance(bag, list):
        for entry in bag:
            if isinstance(entry, dict):
                for k in ("parentApplicationNumberText",
                          "childApplicationNumberText"):
                    v = entry.get(k)
                    if isinstance(v, str) and v.strip():
                        ids.add(v.strip())
    return ids


def _sort_key(c: dict) -> tuple:
    granted = 1 if c.get("patent_number") else 0
    score = c.get("relevance_score")
    if not isinstance(score, (int, float)):
        score = -1
    return (granted, score, c.get("filing_date") or "")


def dedupe_candidates(candidates: list[dict]) -> tuple[list[dict], int]:
    """Remove family/near-duplicate candidates, keeping the best one.

    Two candidates are duplicates when one's patent_id appears in the
    other's ``parentContinuityBag``, or their normalized titles are
    identical. Preference: granted > higher relevance_score > newer
    filing date (ordering by ``_sort_key`` descending).
    """
    ordered = sorted(candidates, key=_sort_key, reverse=True)
    title_groups: dict[str, str] = {}
    seen_ids: set[str] = set()
    kept: list[dict] = []
    dropped = 0
    for c in ordered:
        pid = c["patent_id"]
        dup = any(rel in seen_ids for rel in _continuity_ids(c.get("_raw") or {}))
        nt = _norm_title(c.get("title", ""))
        if not dup and nt:
            if nt in title_groups:
                dup = True
        if dup:
            dropped += 1
            continue
        seen_ids.add(pid)
        if nt and nt not in title_groups:
            title_groups[nt] = pid
        kept.append(c)
    return kept, dropped
