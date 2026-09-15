#!/usr/bin/env python3
"""Probe whether USPTO applications/search honors the `sort` clause, and
what the top of a loose single-phrase query actually contains.

Background (2026-09-13): after the space-flatten fallback was switched
off (REACT_USPTO_SPACE_FLATTEN=0), the production logs still showed
`dead_filter_diag — filtered=20` with every entry a Provisional
Application Expired, this time from the auto-ladder's own query (a
single quoted phrase, e.g. "humidity controller").  Two candidate
mechanisms, and they predict different observations:

  H1 sort ignored — the endpoint silently drops `_score` and serves a
     fixed default order, so every query returns the same slice of the
     corpus regardless of relevance.
  H2 sort honored but BM25 favors provisionals — title-level corpus,
     provisional titles are short, and BM25's field-length norm ranks
     short fields above long ones for the same term match, so a loose
     query's head is structurally provisional-heavy.

Deciding between them matters: H1 is a transport bug (fix the request),
H2 is a corpus property (fix the query or filter the type server-side).

Probes:
  S. same query under three sort variants — `_score`, filingDate, and
     no `sort` key at all.  Identical orderings => the clause is a
     no-op (H1); differing orderings => the clause is honored.
  T. two semantically distant phrases — overlapping result sets => the
     ranking is not discriminating (H1); disjoint => it is (H2).
  U. type-code / status histogram of the loose query's top 20 — the
     direct measurement of "20/20 Provisional".
  V. whether the provisional type can be excluded server-side
     (`applicationTypeCode` field tag) — this is the natural remedy if
     H2 holds; the 2026-09-07 matrix found TTL/ABST/SPEC field tags are
     rejected, so this needs its own measurement rather than an
     assumption.
  W. whether the endpoint honors the `fields` restriction at all.  The
     production dual-search requests RECALL_SEARCH_FIELDS, which does
     NOT list applicationMetaData.applicationTypeCode — so if the
     restriction is honored, `type_code` is always empty on that path
     and is_provisional_application() can never fire there (only the
     status-string filter catches provisionals).  If the restriction is
     ignored and the full item comes back, the type filter is live and
     that gap does not exist.

Read-only: every request is a POST to the search endpoint, limit<=20,
no writes, no other API paths touched.

Usage (anywhere with a valid key):
    USPTO_API_KEY=... python scripts/uspto_sort_probe.py
    USPTO_API_KEY=... python scripts/uspto_sort_probe.py --only S,T

Stdlib only; no repo imports.  Memory-light.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

URL = "https://api.uspto.gov/api/v1/patent/applications/search"

# Loose single-phrase queries, the shape the auto-ladder's loosest rungs
# take.  Two of them are deliberately far apart semantically: if the
# ranking works at all, their result sets must barely intersect.
LOOSE_Q = '"humidity controller"'
DISTANT_Q = '"beverage container"'

FIELDS = [
    "applicationNumberText",
    "applicationMetaData.inventionTitle",
    "applicationMetaData.applicationTypeCode",
    "applicationMetaData.applicationStatusDescriptionText",
    "applicationMetaData.filingDate",
    "applicationMetaData.patentNumber",
]

SORT_SCORE = [{"field": "_score", "order": "desc"}]
SORT_FILING = [{"field": "applicationMetaData.filingDate", "order": "desc"}]

# Verbatim copy of sources/long_task/recall_sources.RECALL_SEARCH_FIELDS —
# the field list the production dual-search leg actually sends.  Kept as a
# literal rather than an import so the probe stays stdlib-only.
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


def _load_env(path: str) -> None:
    """Load KEY=VALUE lines into os.environ without overwriting."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                value = value.strip()
                if (len(value) >= 2 and value[0] == value[-1]
                        and value[0] in ("'", '"')):
                    value = value[1:-1]
                os.environ.setdefault(key.strip(), value)
    except OSError:
        pass


_load_env(os.path.join(os.path.dirname(__file__), "..", ".env"))


def _post(q: str, api_key: str, limit: int = 20,
          sort: list | None = None, offset: int = 0) -> dict:
    """POST one search.  `sort=None` omits the key entirely so the
    endpoint's own default order is observed."""
    body: dict = {
        "q": q,
        "pagination": {"offset": offset, "limit": limit},
        "fields": FIELDS,
    }
    if sort is not None:
        body["sort"] = sort
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-API-Key": api_key})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            bag = payload.get("patentFileWrapperDataBag") or []
            return {"status": resp.status, "count": payload.get("count"),
                    "items": [_digest(i) for i in bag]}
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = json.loads(exc.read().decode("utf-8")).get(
                "detailedMessage", "")
        except Exception:
            pass
        return {"status": exc.code, "count": None, "items": [],
                "detail": detail[:110]}
    except Exception as exc:  # network etc.
        return {"status": "ERR", "count": None, "items": [],
                "detail": f"{type(exc).__name__}: {exc}"[:110]}


def _digest(item: dict) -> dict:
    m = item.get("applicationMetaData") or {}
    return {
        "app": str(item.get("applicationNumberText") or "").strip(),
        "type": str(m.get("applicationTypeCode") or "").strip(),
        "status": str(m.get("applicationStatusDescriptionText") or "").strip(),
        "filed": str(m.get("filingDate") or "").strip(),
        "pn": str(m.get("patentNumber") or "").strip(),
        "title": str(m.get("inventionTitle") or "").strip()[:44],
    }


def _apps(res: dict) -> list:
    return [i["app"] for i in res.get("items", []) if i["app"]]


def _order_equal(a: list, b: list) -> bool:
    return bool(a) and a == b


# Mirror of sources/long_task/candidate_metadata.DEAD_STATUS_MARKERS.
_DEAD_MARKERS = ("expired", "abandon", "placed in storage")


def _is_dead(status: str) -> bool:
    lowered = (status or "").lower()
    return any(m in lowered for m in _DEAD_MARKERS)


def _histogram(items: list, key: str) -> list:
    counts: dict = {}
    for i in items:
        counts[i[key] or "(empty)"] = counts.get(i[key] or "(empty)", 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])


def _show(label: str, res: dict) -> None:
    print(f"  {label:<26} status={res['status']!s:<5} "
          f"count={res['count']!s:<10} returned={len(res.get('items', []))}")
    if res.get("detail"):
        print(f"    detail: {res['detail']}")
    for i in res.get("items", [])[:5]:
        print(f"    {i['app']:<12} {i['type']:<4} "
              f"{i['status'][:34]:<34} {i['title']}")


def probe_s(key: str) -> dict:
    """Same query, three sort variants — is the clause honored?"""
    print("=== S. sort variants on one loose phrase ===")
    print(f"q = {LOOSE_Q}")
    s_score = _post(LOOSE_Q, key, sort=SORT_SCORE)
    _show("S1 sort=_score", s_score)
    s_filing = _post(LOOSE_Q, key, sort=SORT_FILING)
    _show("S2 sort=filingDate", s_filing)
    s_none = _post(LOOSE_Q, key, sort=None)
    _show("S3 sort omitted", s_none)

    a1, a2, a3 = _apps(s_score), _apps(s_filing), _apps(s_none)
    print()
    print("  verdict (top-20 applicationNumberText order):")
    if not (a1 and a2 and a3):
        print("    INCONCLUSIVE — at least one variant returned nothing "
              "(check status/detail above).")
    elif _order_equal(a1, a2) and _order_equal(a1, a3):
        print("    ALL THREE IDENTICAL → the `sort` clause is a no-op on "
              "this endpoint (H1).")
    elif a1 != a3:
        print("    S1 differs from S3 → `_score` ordering is applied "
              "(H2 territory; inspect the histogram in U).")
    if a1 and a3:
        overlap = len(set(a1) & set(a3))
        print(f"    S1∩S3 = {overlap}/{len(a1)} of S1")
    return {"score": s_score, "filing": s_filing, "none": s_none}


def probe_t(key: str, s_score: dict) -> None:
    """Two distant phrases — do their sets intersect?"""
    print()
    print("=== T. semantically distant phrase ===")
    print(f"q = {DISTANT_Q}")
    d = _post(DISTANT_Q, key, sort=SORT_SCORE)
    _show("T1 sort=_score", d)
    a1, a2 = _apps(s_score), _apps(d)
    if a1 and a2:
        inter = set(a1) & set(a2)
        print(f"    S1∩T1 = {len(inter)}/{len(a1)} — "
              + ("OVERLAPPING, ranking not discriminating (H1)"
                 if inter else "disjoint, ranking discriminates (H2)"))


def probe_u(s_score: dict) -> None:
    """What is the loose query's head actually made of?"""
    print()
    print("=== U. composition of the loose query's top 20 ===")
    items = s_score.get("items", [])
    if not items:
        print("    no items — run S first.")
        return
    for label, key in (("type code", "type"), ("status", "status")):
        print(f"    by {label}:")
        for value, n in _histogram(items, key):
            print(f"      {n:>3}  {value}")
    provisionals = sum(1 for i in items
                       if i["type"].upper() == "P"
                       or "provisional" in i["status"].lower())
    print(f"    provisional-like: {provisionals}/{len(items)}")
    print("    (if ~all are provisional AND S showed identical orderings, "
          "H1+H2 combine:\n     a fixed default slice of a corpus that is "
          "provisional-dense)")


def probe_v(key: str) -> None:
    """Can the type be constrained server-side?"""
    print()
    print("=== V. server-side type constraint ===")
    variants = [
        ("V1 dotted field tag",
         f'{LOOSE_Q} AND applicationMetaData.applicationTypeCode:UTL'),
        ("V2 short field tag",
         f'{LOOSE_Q} AND applicationTypeCode:UTL'),
        ("V3 exclude provisionals",
         f'{LOOSE_Q} AND NOT applicationMetaData.applicationTypeCode:P'),
        ("V4 bare type P count",
         "applicationMetaData.applicationTypeCode:P"),
    ]
    for label, q in variants:
        res = _post(q, key, limit=3, sort=SORT_SCORE)
        print(f"  {label}: status={res['status']!s} count={res['count']!s}")
        if res.get("detail"):
            print(f"    detail: {res['detail']}")
        for i in res.get("items", [])[:3]:
            print(f"    {i['app']:<12} {i['type']:<4} {i['status'][:34]}")


def probe_w(key: str) -> None:
    """Is the `fields` restriction honored?  Raw item shape, no digest."""
    print()
    print("=== W. fields restriction honored? ===")
    body = {
        "q": LOOSE_Q,
        "pagination": {"offset": 0, "limit": 2},
        "fields": RECALL_SEARCH_FIELDS,
        "sort": SORT_SCORE,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-API-Key": key})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"  status={exc.code} — cannot inspect item shape")
        return
    except Exception as exc:
        print(f"  {type(exc).__name__}: {exc}")
        return
    bag = payload.get("patentFileWrapperDataBag") or []
    if not bag:
        print("  no items returned — cannot inspect item shape")
        return
    item = bag[0]
    meta = item.get("applicationMetaData") or {}
    print(f"  requested {len(RECALL_SEARCH_FIELDS)} fields, "
          f"item has {len(item)} top-level keys")
    print(f"  top-level keys: {sorted(item.keys())}")
    print(f"  meta keys:      {sorted(meta.keys())}")
    returned = "applicationTypeCode" in meta
    print(f"  applicationTypeCode present: {returned}")
    if returned:
        print("    → `fields` is IGNORED (or merged) — type_code is live "
              "on the production path,\n      so the provisional type "
              "filter does fire there.")
    else:
        print("    → `fields` is HONORED — type_code is always empty on "
              "the production dual-search\n      path, so "
              "is_provisional_application() can never fire there and only "
              "the\n      status-string filter catches provisionals.")


# Ladder-rung-shaped phrases from unrelated domains, to separate "this
# query is unlucky" from "the corpus head is death-heavy".
SAMPLE_PHRASES = [
    '"humidity controller"',
    '"beverage container"',
    '"LED driver"',
    '"air dryer"',
    '"semiconductor wafer"',
    '"injection molding"',
]


def probe_y(key: str) -> None:
    """Live-candidate yield of `_score` vs the API default, per phrase.

    `dead_filter_diag` reports `filtered=N` whenever a merge introduces N
    dead candidates.  Because dead candidates are dropped before the
    scoring head is sliced, a page whose 20 slots are all dead yields
    ZERO live candidates — the search "returns hits" yet contributes
    nothing displayable.  This group measures, per phrase, how many of
    the 20 slots survive under each ordering, which is the decision table
    for whether overriding the API default is worth it.
    """
    print()
    print("=== Y. live-candidate yield: _score vs API default "
          "(offset 0, limit 20) ===")
    print(f"  {'phrase':<24}{'count':<8}{'_score live':<14}"
          f"{'default live':<14}{'default ='}")
    for phrase in SAMPLE_PHRASES:
        r_score = _post(phrase, key, limit=20, sort=SORT_SCORE)
        r_default = _post(phrase, key, limit=20, sort=None)
        if not r_score.get("items") or not r_default.get("items"):
            print(f"  {phrase:<24}{r_score['status']!s:<8}— (no items)")
            continue
        def _live(res):
            items = res["items"]
            return sum(1 for i in items if not _is_dead(i["status"])), \
                sum(1 for i in items if "provisional" in i["status"].lower())
        live_s, prov_s = _live(r_score)
        live_d, prov_d = _live(r_default)
        print(f"  {phrase:<24}{r_score['count']!s:<8}"
              f"{live_s}/{len(r_score['items'])} ({prov_s}p){'':<4}"
              f"{live_d}/{len(r_default['items'])} ({prov_d}p){'':<4}"
              f"filingDate desc")
    print("  read: `_score live` = how many of the 20 slots survive the dead")
    print("  filter under the ordering the code forces.  0 means the whole")
    print("  page is discarded as noise — the search hit nothing usable even")
    print("  though `count` is large.")


def probe_x(key: str) -> None:
    """Page-position composition — where does the dead tail start?

    `_auto_run_patent_ladder` passes the CALLER's `page` through to every
    ladder rung (react_tools.py:2807), so a rung reads whatever offset
    the first round used rather than the head of its own result set.  If
    the ranking is honest, ranks 1-20 and 21-40 of a small result set
    have very different death rates — which is exactly the shape that
    turns into `dead_filter_diag filtered=<page_size>` when the ladder
    runs on a deep page.
    """
    print()
    print("=== X. composition by page position ===")
    print(f"q = {LOOSE_Q}   sort=_score")
    for offset in (0, 20, 40):
        res = _post(LOOSE_Q, key, limit=20, sort=SORT_SCORE, offset=offset)
        items = res.get("items", [])
        if not items:
            print(f"  offset {offset:<3} status={res['status']} returned=0"
                  + (f" — {res['detail']}" if res.get("detail") else ""))
            continue
        dead = sum(1 for i in items if _is_dead(i["status"]))
        prov = sum(1 for i in items if "provisional" in i["status"].lower())
        print(f"  offset {offset:<3} status={res['status']} "
              f"count={res['count']} returned={len(items)} "
              f"dead={dead}/{len(items)} provisional={prov}/{len(items)}")
        for i in items[:3]:
            print(f"      {i['app']:<12} {i['type']:<4} {i['status'][:40]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None,
                        help="comma list of probe groups, e.g. S,U or S,T,U,V")
    parser.add_argument("--key", default=None,
                        help="USPTO_API_KEY (default: env USPTO_API_KEY)")
    parser.add_argument("--q", default=None,
                        help="override the loose phrase under test, e.g. "
                             "--q '\"LED driver\"'")
    args = parser.parse_args()

    if args.q:
        global LOOSE_Q
        LOOSE_Q = args.q

    api_key = args.key or os.getenv("USPTO_API_KEY", "")
    if not api_key:
        print("USPTO_API_KEY not set (env or --key) — cannot run.",
              file=sys.stderr)
        return 2

    only = (args.only or "S,T,U,V,W").split(",")
    s_score = probe_s(api_key)["score"] if "S" in only else {"items": []}
    if "T" in only:
        probe_t(api_key, s_score)
    if "U" in only:
        probe_u(s_score)
    if "V" in only:
        probe_v(api_key)
    if "W" in only:
        probe_w(api_key)
    if "X" in only:
        probe_x(api_key)
    if "Y" in only:
        probe_y(api_key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
