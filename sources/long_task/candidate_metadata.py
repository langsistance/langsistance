"""Candidate extraction and enrichment for USPTO search results.

raw_items from the USPTO applications/search response are
``patentFileWrapperDataBag`` entries. This module flattens the useful
metadata (title, applicant, dates, status, CPC codes) into compact
candidate dicts used by the relevance gate, dedupe, and report.
"""

import json
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


def is_identifying_number_tool(tool: Any) -> bool:
    """True for tools that fetch a single application by number.

    The LLM uses these to verify candidate details one by one.  Repeated
    calls with no cap (observed: 8+ verification calls, each followed by
    a full ~2.5s semantic rerank) stall the request — the loop caps them
    per request via REACT_VERIFY_CALL_MAX.
    """
    title = (getattr(tool, "title", "") or "").lower()
    return "identifying_number" in title


def is_uspto_tool(tool: Any) -> bool:
    """True when the tool's URL targets api.uspto.gov."""
    url = (getattr(tool, "url", "") or "").lower()
    return "uspto" in url


def is_documents_tool(tool: Any) -> bool:
    """True for USPTO document-list tools.

    Checks the tool URL *and* its params template path — the production
    documents tool stores the placeholder path in params
    (url=``.../applications``, path=``"{applicationNumberText}/documents"``),
    so a URL-only check misses it and the document list gets treated as a
    search pool (observed: 68 documents replaced by recall patents).
    """
    url = (getattr(tool, "url", "") or "").lower()
    if "documents" in url:
        return True
    try:
        from sources.dynamic_tool_params import _coerce_json_object
        params = _coerce_json_object(
            getattr(tool, "params", "") or "", "tool_info.params")
        path = str(params.get("path", "") or "").lower()
        return "documents" in path
    except Exception:
        return False


def ensure_search_fields(params: dict) -> dict:
    """Return a deep copy of tool params with required USPTO fields added.

    Only touches ``body.fields``. The input dict is never mutated.
    Some knowledge-base templates store ``fields`` as a JSON string;
    coerce it to a list so field completion cannot be silently skipped.
    """
    out = deepcopy(params)
    body = out.get("body") if isinstance(out.get("body"), dict) else {}
    fields = body.get("fields")
    if isinstance(fields, str):
        try:
            parsed = json.loads(fields)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, list):
            fields = parsed
            body["fields"] = fields
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
        # Real USPTO responses carry applicationNumberText at the TOP
        # level of each patentFileWrapperDataBag item; read that first
        # and keep the nested read as a fallback for other source shapes.
        pid = str(item.get("applicationNumberText")
                  or m.get("applicationNumberText") or "").strip()
        if not pid:
            continue
        candidates.append({
            "patent_id": pid,
            # Title may ride top-level (title / titleOfInvention, after
            # _normalize_uspto_items lifting) or nested in applicationMetaData
            # (inventionTitle / titleOfInvention — schema varies by version).
            "title": (_first_str(item, "title", "inventionTitle",
                                 "titleOfInvention")
                      or _first_str(m, "inventionTitle", "titleOfInvention")),
            "applicant": _first_str(m, "firstApplicantName"),
            "status": _first_str(m, "applicationStatusDescriptionText"),
            "filing_date": _first_str(m, "filingDate"),
            "grant_date": _first_str(m, "grantDate"),
            "patent_number": _first_str(m, "patentNumber"),
            "type_code": _first_str(m, "applicationTypeCode"),
            "cpc_codes": _extract_cpc_codes(m),
            "_raw": item,
        })
    return candidates


# ── Legal status ────────────────────────────────────────────────────────────

DEAD_STATUS_MARKERS = ("expired", "abandon", "placed in storage")


def is_dead_status(status: Any) -> bool:
    """True when a USPTO status string means no enforceable rights.

    Covers expired/abandoned cases and PCT applications parked in
    storage (never entered the US national stage).  Unknown or empty
    statuses are treated as alive — never hide what we cannot verify.
    """
    if not isinstance(status, str) or not status.strip():
        return False
    lowered = status.lower()
    return any(m in lowered for m in DEAD_STATUS_MARKERS)


# ── Dedupe ──────────────────────────────────────────────────────────────────

def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (title or "").lower())


# Candidates at or above this relevance score are never dropped for having
# the same normalized title as another candidate — a high score means the
# user is likely interested, and the application-publication corpus
# contains many same-title continuation/divisional filings that are
# distinct technical records (observed 2026-09-01: 5 of 6 US hits for a
# dry-air question collapsed to 1 by title dedupe alone).
DEDUPE_HIGH_SCORE = 4


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


def is_design_patent(c: dict) -> bool:
    """True when the candidate is a design patent (D-numbered patent or
    applicationTypeCode DES) — design rights protect appearance, not
    the technical solution a user searches for."""
    pn = str(c.get("patent_number") or "").strip().upper()
    code = str(c.get("type_code") or "").strip().upper()
    return pn.startswith("D") or code == "DES"


def _sort_key(c: dict) -> tuple:
    alive = 0 if is_dead_status(c.get("status")) else 1
    utility = 0 if is_design_patent(c) else 1
    granted = 1 if c.get("patent_number") else 0
    score = c.get("relevance_score")
    if not isinstance(score, (int, float)):
        score = -1
    # Two-stage prescore: unscored candidates with a semantic_score
    # order by it instead of sinking in insertion order — the deep end
    # of the recall window stays visible instead of being pruned.
    sem = c.get("semantic_score")
    if not isinstance(sem, (int, float)):
        sem = -2.0
    return (alive, utility, granted, score, sem,
            c.get("filing_date") or "")


def dedupe_candidates(candidates: list[dict]) -> tuple[list[dict], int]:
    """Remove family/near-duplicate candidates, keeping the best one.

    A candidate is a duplicate when:
      1. its patent_id appears in another kept candidate's
         ``parentContinuityBag`` (true family — same application chain), or
      2. its normalized title AND normalized applicant match a kept
         candidate — same title under a DIFFERENT applicant is a different
         technical record and is kept (observed: USPTO application-publication
         corpus is full of same-title continuation/divisional filings).

    Candidates with ``relevance_score >= DEDUPE_HIGH_SCORE`` are exempt
    from the title/applicant rule (high score = the user likely wants it),
    but never from the continuity rule.

    Preference among duplicates: live status > granted > higher
    relevance_score > newer filing date (ordering by ``_sort_key``
    descending).
    """
    ordered = sorted(candidates, key=_sort_key, reverse=True)
    title_groups: dict[tuple, str] = {}
    seen_ids: set[str] = set()
    kept: list[dict] = []
    dropped = 0
    for c in ordered:
        pid = c["patent_id"]
        dup = any(rel in seen_ids for rel in _continuity_ids(c.get("_raw") or {}))
        nt = _norm_title(c.get("title", ""))
        high_score = (
            isinstance(c.get("relevance_score"), (int, float))
            and c["relevance_score"] >= DEDUPE_HIGH_SCORE
        )
        if not dup and nt and not high_score:
            key = (nt, _norm_title(c.get("applicant", "")))
            if key in title_groups:
                dup = True
        if dup:
            dropped += 1
            continue
        seen_ids.add(pid)
        if nt:
            key = (nt, _norm_title(c.get("applicant", "")))
            if key not in title_groups:
                title_groups[key] = pid
        kept.append(c)
    return kept, dropped
