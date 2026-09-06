"""Patent-number translation layer — verdict + per-office candidate lookup lists.

Where ``patent_number_parser`` only *recognises* a number (country / type /
confidence / reason), this module decides whether any downstream resolver
(USPTO / EPO OPS docdb / Baiten CN / regional executors) can meaningfully
consume it, and hands each office the ordered list of concrete identifier
strings to try (``TranslateResult.candidates``).

The decision is a pure local table keyed off the parser output — the only
network dependency is the optional US reverse lookup (``resolve_us_pub_number``,
extracted unchanged from celery_worker families Phase 0).  All callers obey the
fail-open contract (spec §7): ``translate`` / ``verdict_of`` only flag
*deterministically* unresolvable ids; an internal exception raises upstream so
the old path is preserved.

Background (2026-09-06): a PCT international application number is not itself
a resolvable publication — its docdb / national-phase siblings must be derived
by reverse lookup, and the naive "same WO number" guess was empirically
disproved on the server (WO2021059064 = an unrelated ABB case vs the real
WO2023075806A1 for PCTUS2021059064).  Hence PCT → unresolvable-with-guidance
(default) and never a direct-guess candidate; evidence never reports
``direct_guess``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from sources.patent_number_parser import parse_patent_identifiers
from sources.patent_id_utils import extract_us_patent_digits

# ── Reverse-lookup experimental channel (spec §5.2) ─────────────────────────
# PCT→WO automatic reverse lookup is deferred to Phase A later (implementation
# of GooglePatentsClient reverse search).  Until then the channel is
# unimplemented and translate() falls closed to unresolvable.
PATENT_REVERSE_LOOKUP_ENABLED = False

REASON_PCT = (
    "PCT 国际申请号本身不能直接做同族/审查查询。"
    "请提供 WO 公开号或进入国家阶段的申请号后再发起分析。"
)
_REASON_WO = "WO 公开号可直接做 EPO 同族查询。"


# helper for stripping the trailing kind code off a US publication string.
_KIND_RE = re.compile(r"[A-Z]\d*$")

# helper for WO display → bare docdb form: "WO2021/059064" → "WO2021059064".
_WO_DOCTDB_RE = re.compile(r"^WO?\s*(\d+)\s*(?:[/\s]+)(\d+)", re.IGNORECASE)


@dataclass
class TranslateResult:
    """Verdict + per-office candidate strings for one patent id.

    immutable, in-memory only (never persisted).  candidates maps:
      epo_docdb — ordered docdb id strings to try against EPO OPS, first = best
      uspto     — USPTO search id strings
      cnipa     — Baiten CN id strings
    """
    patent_id: str
    verdict: str = "unresolvable"           # "resolvable" | "unresolvable"
    candidates: dict = field(default_factory=dict)
    evidence: str = ""                       # "local" | "reverse_lookup" | ""
    reason: str = ""
    confidence: str = "low"


def _log():
    # Lazy module logger: same file the long-task pipeline uses, but created on
    # first use so importing this module (parser tests, gate callers) performs
    # no filesystem side effect.
    if _log._singleton is None:  # noqa: SLF001 private naming allowed here
        from sources.logger import Logger
        _log._singleton = Logger("long_task_pipeline.log")  # noqa: SLF001
    return _log._singleton  # noqa: SLF001


_log._singleton = None  # noqa: SLF001


async def resolve_us_pub_number(
    app_id: str,
    tid: str,
    id_type: str = "application_number",
) -> tuple[str | None, str | None, str | None]:
    """Resolve a US patent identifier → (pub_number, appNumberText, grant_number).

    Searches USPTO by the field appropriate for *id_type*:
    - application_number → applicationNumberText
    - publication_number → earliestPublicationNumber
    - grant_number       → patentNumber

    Returns a 3-tuple; any element may be None when not found.  Network
    exceptions are swallowed (returns all-None) — fail closed, never raises.

    Cross-office note (spec §5.2): this is the USPTO-search reverse lookup
    used to build EPO docdb candidates.  It is distinct from
    ``uspto_download.resolve_application_number`` (PEDS documents) which keeps
    its own callers in patent_detail routes — do not merge the two.

    Extracted byte-equivalently from celery_worker families Phase 0 — do not
    change its query building / field-parsing semantics here.
    """
    try:
        import json as _json, re as _re, os as _os_fb
        import httpx as _httpx

        # Normalise to the format USPTO expects for each field:
        _up = app_id.upper()
        _digits = extract_us_patent_digits(_up)
        if not _digits or len(_digits) < 6:
            return None, None, None

        _esc = lambda s: _re.sub(  # noqa: E731
            r'(["\\+\-!(){}[\]^~*?:/]|&&|\|\|)', r'\\\1', s,
        )

        if id_type == "grant_number":
            _query = f'applicationMetaData.patentNumber:"{_esc(_digits)}"'
        elif id_type == "publication_number":
            # earliestPublicationNumber WITH US prefix + kind code.
            _pub_full = _up if _up.startswith("US") else f"US{_up}"
            _pub_digits = f"US{_digits}"
            _query = (
                f'applicationMetaData.earliestPublicationNumber:"{_esc(_pub_full)}"'
                f' OR applicationMetaData.earliestPublicationNumber:"{_esc(_pub_digits)}"'
            )
        else:
            _query = f'applicationNumberText:"{_esc(_digits)}"'
        # Always include applicationNumberText as fallback.
        if "applicationNumberText" not in _query:
            _query += f' OR applicationNumberText:"{_esc(_digits)}"'

        _body = {
            "q": _query,
            "pagination": {"offset": 0, "limit": 1},
            "fields": [
                "applicationNumberText",
                "applicationMetaData.patentNumber",
                "applicationMetaData.earliestPublicationNumber",
                "applicationMetaData.inventionTitle",
            ],
        }
        _hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
        _uk = _os_fb.getenv("USPTO_API_KEY", "")
        if _uk:
            _hdrs["X-API-Key"] = _uk

        async with _httpx.AsyncClient(timeout=15) as _cl:
            _resp = await _cl.post(
                "https://api.uspto.gov/api/v1/patent/applications/search",
                headers=_hdrs, json=_body,
            )
        _log().info(
            f"[task={tid}] uspto_search — status={_resp.status_code}, "
            f"id_type={id_type}, digits={_digits}"
        )
        if _resp.status_code != 200:
            _log().warning(
                f"[task={tid}] uspto_search failed — status={_resp.status_code}"
            )
            return None, None, None

        _data = _resp.json() if _resp.text else {}
        _results = (
            _data.get("patentFileWrapperDataBag", None)
            or _data.get("results", None)
            or _data.get("patentFileBag", [])
        )
        if not _results:
            return None, None, None

        _hit = _results[0] if isinstance(_results, list) else _results
        _app_text = _hit.get("applicationNumberText", "") or ""
        _pub = (
            _hit.get("applicationMetaData", {}).get("earliestPublicationNumber", "")
            if isinstance(_hit, dict) else ""
        ) or ""
        _grant = (
            _hit.get("patentNumber", "")
            or (_hit.get("applicationMetaData", {}).get("patentNumber", "")
                if isinstance(_hit, dict) else "")
        ) or ""
        _log().info(
            f"[task={tid}] uspto_search result — "
            f"pub={_pub}, grant={_grant}, app_text={_app_text}"
        )
        return _pub or None, _app_text or None, _grant or None
    except Exception as _fe:
        _log().warning(
            f"[task={tid}] uspto_search error — {type(_fe).__name__}: {_fe}"
        )
        return None, None, None


def _us_docdb_candidates(
    orig: str, pub_number: str | None, grant: str | None,
    app_text: str | None,
) -> list:
    """Ordered EPO docdb id strings for a US origin id, best-guess first.

    Mirrors familias Phase 0 candidate ordering so a consumer fed these
    strings tries EPO in the historically proven order, unchanged:
        [orig, US{grant}, no-kind pub, pub (with kind), US.xxxx.kind, app_text]
    """
    out = []
    for c in (orig,):
        if c:
            out.append(c)
    if grant:
        _g = f"US{grant}"
        if _g not in out:
            out.append(_g)
    if pub_number:
        _no_kind = _KIND_RE.sub("", pub_number)
        _kind = pub_number[len(_no_kind):]
        _dotted = (f"US.{_no_kind[2:]}.{_kind}" if _kind
                   else f"US.{_no_kind[2:]}")
        for c in (_no_kind, pub_number, _dotted):
            if c not in out:
                out.append(c)
    if app_text and app_text not in out:
        out.append(app_text)
    return out


# ── Local verdict decision (shared by translate & verdict_of) ───────────────
# Returns verdict/evidence/reason/confidence plus the coarse "uspto resolvable"
# / "needs us reverse lookup" hints.  Purely local — no network.

def _decide(candidate: dict | None) -> dict:
    """Local-only classification of one parser candidate → action dict."""
    if not candidate:
        return {
            "verdict": "unresolvable", "evidence": "",
            "reason": "未能识别为可解析的专利号。",
            "confidence": "low",
            "uspto_resolvable": False, "needs_us_reverse": False,
            "lookups": [],
        }
    country = candidate.get("country")
    id_type = candidate.get("id_type")
    lookups = list(candidate.get("lookups") or [])
    conf = candidate.get("confidence") or "low"
    # US: any US shape (grant/pub/app/design/reissue/ambiguous ≤ digits) is
    # resolvable against USPTO; EPO docdb candidate may need a reverse lookup
    # to disambiguate grant vs application.
    if country == "US":
        return {
            "verdict": "resolvable", "evidence": "local",
            "reason": candidate.get("reason") or "美国号码可查 USPTO。",
            "confidence": conf,
            "uspto_resolvable": True,
            "needs_us_reverse": True,
            "lookups": lookups or [extract_us_patent_digits(candidate.get("raw") or "")],
        }
    if country == "CN":
        return {
            "verdict": "resolvable", "evidence": "local",
            "reason": candidate.get("reason") or "中国号码可查佰腾。",
            "confidence": conf,
            "uspto_resolvable": False, "needs_us_reverse": False,
            "lookups": lookups,
        }
    if id_type == "wo":  # user-provided WO publication number.
        return {
            "verdict": "resolvable", "evidence": "local",
            "reason": _REASON_WO,
            "confidence": conf or "high",
            "uspto_resolvable": False, "needs_us_reverse": False,
            "lookups": lookups,  # ["WOYYYYNNNNNN"]
        }
    if id_type == "pct":
        return {
            # reverse_lookup handled by translate; verdict here is the
            # fail-closed default (guided, no direct guess).
            "verdict": "unresolvable", "evidence": "",
            "reason": REASON_PCT,
            "confidence": conf or "high",
            "uspto_resolvable": False, "needs_us_reverse": False,
            "lookups": [],
        }
    # unsupported / foreign prefix (EP/JP/…) / anything else.
    return {
        "verdict": "unresolvable", "evidence": "",
        "reason": candidate.get("reason")
        or "该号码当前无法直接解析，请补全为 WO/国家号后再查。",
        "confidence": "low",
        "uspto_resolvable": False, "needs_us_reverse": False,
        "lookups": [],
    }


def _top_candidate(patent_id: str) -> dict | None:
    """First parser candidate for *patent_id* (best confidence first)."""
    parsed = parse_patent_identifiers(str(patent_id or "").strip())
    return parsed[0] if parsed else None


def _us_resolver_id_type(parser_id_type: str | None) -> str:
    """Map parser id_type vocabulary → resolve_us_pub_number query branch.

    grant → grant_number (patentNumber), publication → publication_number
    (earliestPublicationNumber); ambiguous / application / design / others keep
    the legacy default (applicationNumberText, which the resolver always unions
    in as fallback anyway).  Mirrors the id_type-aware querying the pre-extract
    families Phase 0 code performed.
    """
    if parser_id_type == "grant":
        return "grant_number"
    if parser_id_type == "publication":
        return "publication_number"
    return "application_number"


def verdict_of(patent_id: str, scenario: str = "") -> str | None:
    """Synchronous lightweight resolvable verdict (no network side effects).

    Deliberately offline: local parser + regex only.  Used by the detail-route
    / submit / chat pre-check gates on non-async call paths, where a full
    ``translate`` (which may reverse-lookup a US application number) is overkill.

    Returns "resolvable" / "unresolvable"; None when nothing deterministic can
    be asserted (no recognisable patent number) so gates fall through to the
    legacy path instead of blocking plain text (review 5f07835 MEDIUM-1).
    """
    del scenario  # verdict is scenario-independent for now.
    candidate = _top_candidate(patent_id)
    if candidate is None:
        return None
    return _decide(candidate)["verdict"]


def unresolvable_gate_error(patent_id: str) -> str:
    """A6 shared gate (spec §5.3): publication-format guidance text iff *patent_id*
    is a deterministically unresolvable shape; ``""`` means pass-through.

    One-line quick gate the CN / EP / JP examination resolvers place right
    before they delegate a non-local id to the EPO ``lookup_family`` remote call,
    so a PCT / unsupported-bare / foreign office shape never white-sends a wasted
    remote family query.  Fail-open contract (spec §7): returns ``""`` when
    nothing deterministic can be asserted (no recognisable number → None verdict)
    or when the translator itself raises — in both cases callers keep their legacy
    delegation unchanged.  Purely local / offline; builds generic next-step copy
    from ``failure_guidance`` (reason_code ERR_UNRESOLVABLE_ID, spec §5.4).
    """
    try:
        if verdict_of(patent_id) != "unresolvable":
            return ""
    except Exception:
        _log().warning(
            f"A6 gate — verdict_of raised for {patent_id!r}; pass-through")
        return ""
    from sources.long_task.status_manager import (
        failure_guidance, ERR_UNRESOLVABLE_ID)
    return failure_guidance("", ERR_UNRESOLVABLE_ID, lang="zh")


# ── Public async entry ───────────────────────────────────────────────────────

async def translate(
    patent_id: str, scenario: str = "", reverse_lookup: bool = False,
) -> TranslateResult:
    """Build a TranslateResult for one patent id.

    Deterministic for resolvable ids; PCT/unsupported become unresolvable with
    guidance.  Internal exceptions propagate to the caller (fail-open contract,
    spec §7) rather than being coerced into a bogus verdict.
    """
    del scenario
    candidate = _top_candidate(patent_id)
    decision = _decide(candidate)
    base = TranslateResult(
        patent_id=patent_id,
        verdict=decision["verdict"],
        evidence=decision["evidence"],
        reason=decision["reason"],
        confidence=decision["confidence"],
    )

    if decision["verdict"] == "unresolvable":
        # PCT/unsupported hard unresolvable with guidance.  Even when the user
        # opts into reverse_lookup, the experimental PCT→WO channel is not yet
        # implemented (spec §5.2) → answers unresolvable, never a WO direct
        # guess (constant-guess disproved on the server, 2026-09-06).
        return base

    # resolvable — fill office candidate lists from local parse / optional US
    # reverse lookup.
    office = {
        "epo_docdb": [], "uspto": decision["lookups"], "cnipa": [],
    }
    confidence = decision["confidence"]
    if candidate is None:
        return base

    if decision["needs_us_reverse"] and candidate.get("country") == "US":
        # US: epo docdb candidate benefits from an (optional) USPTO reverse
        # lookup that recovers grant / earliest-publication forms.  The first
        # orig attempts the original (prefixed) id, matching families Phase 0.
        # Pass the parsed id_type through so the query uses the grant/publication
        # branch instead of always defaulting to applicationNumberText
        # (review 5f07835 MEDIUM-2).
        pub, app, grant = await resolve_us_pub_number(
            patent_id, tid="",
            id_type=_us_resolver_id_type(candidate.get("id_type")),
        )
        office["epo_docdb"] = _us_docdb_candidates(
            patent_id, pub, grant, app,
        )
        if not (grant or pub):
            # no grant found — still resolvable against USPTO, but the downstream
            # EPO candidate cannot be pinned down → drop confidence.
            confidence = _lowest(confidence, "low")
    elif decision["uspto_resolvable"]:
        office["epo_docdb"] = list(decision["lookups"])

    if candidate.get("country") == "CN":
        office["cnipa"] = decision["lookups"]
    if candidate.get("id_type") == "wo":
        # Construct the EPO docdb candidate locally from the parsed shape —
        # independent of parser lookups (the WO source isn't wired into the
        # generic search legs).  display "WO2021/059064" → "WO2021059064".
        office["epo_docdb"] = [_wo_docdb(candidate.get("display") or patent_id)]
    return replace(base, confidence=confidence, candidates=office)


def _wo_docdb(display: str) -> str:
    """Bare EPO docdb form from a WO display string ("WOYYYYNNNNNN")."""
    m = _WO_DOCTDB_RE.search(str(display))
    if m:
        return f"WO{m.group(1)}{m.group(2)}"
    # Fall back to digits-only extraction for already-compact inputs.
    return "WO" + extract_us_patent_digits(display or "")


def _lowest(a: str, b: str) -> str:
    """Return the more conservative (lower) confidence between labels.

    high=0 < medium=1 < low=2; "lowest" = the numerically largest rank.
    """
    scale = {"high": 0, "medium": 1, "low": 2}
    return a if scale.get(a, 2) >= scale.get(b, 2) else b
