#!/usr/bin/env python3
"""Probe which query dialect the USPTO applications/search API actually
understands.

Background (probed 2026-09-07; production logs 2026-09-03/07): the
endpoint searches a TITLE-level corpus with real boolean support up to
a hard stop — queries carrying 3+ AND operators 404 with "No matching
records found" whatever the parentheses; OR groups and quoted phrases
are fine.  Space-joined words are OR semantics, NOT implicit AND
(single-word counts wafer=25k temperature=61k pressure=80k control=431k
sum to the space-join counts wafer temperature=86k / 4-word=580k), so
flattening a query to space-joined words loosens it to OR noise and can
never rescue AND meaning.  Field tags (TTL/ABST/SPEC/inventionTitle)
are rejected.  The production recall ladder died rung by rung because
every rung carried 4+ AND conjuncts while only ≤2-AND queries parse and
count.

This script POSTs a dialect matrix and prints status/count per query.
Rows A-C were the first round; D/E (single-word baselines + poison
isolation + safe dialect shapes) the second.

Usage (anywhere with a valid key — e.g. the backend server):
    USPTO_API_KEY=... python scripts/uspto_dialect_matrix_probe.py
    USPTO_API_KEY=... python scripts/uspto_dialect_matrix_probe.py --only A

Stdlib only; no repo imports.  Memory-light (limit=1 per request).
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

URL = "https://api.uspto.gov/api/v1/patent/applications/search"

# Each row: (id, label, q).  Group A compares operator shapes on terms
# that certainly exist in the corpus; group B checks field scope/corpus;
# group C checks that 404 still means zero for clean queries.
PROBES = [
    # ── A. operator / structure semantics (same vocabulary) ──────────
    ("A1",  "space-join 5 words",      "semiconductor wafer temperature pressure control"),
    ("A2",  "explicit AND x5",         "semiconductor AND wafer AND temperature AND pressure AND control"),
    ("A3",  "paren pair + AND term",   "(semiconductor AND wafer) AND temperature"),
    ("A4",  "single paren pair AND",   "(semiconductor AND wafer)"),
    ("A5",  "explicit AND x2",         "semiconductor AND wafer"),
    ("A6",  "space-join 4 words",      "wafer temperature pressure control"),
    ("A7",  "explicit AND x4",         "wafer AND temperature AND pressure AND control"),
    ("A8",  "quoted phrases, AND",     '"process chamber" AND "wafer temperature"'),
    ("A9",  "quoted phrases, space",   '"process chamber" "wafer temperature"'),
    ("A10", "plain words of phrases",  "process chamber wafer temperature"),
    ("A11", "full ladder rung (404 obs)", '("process chamber" AND "wafer temperature") '
                                          'AND (manometer OR "pressure transducer")'),
    ("A12", "paren OR clause",         "wafer AND (temperature OR thermal)"),
    ("A13", "2-term space join",       "wafer temperature"),
    # ── B. field qualification / corpus scope ────────────────────────
    ("B1",  "TTL field, phrase",       "TTL:(wafer temperature)"),
    ("B2",  "TTL field, AND pair",     "TTL:semiconductor AND TTL:wafer"),
    ("B3",  "ABST field AND",          "ABST:(wafer AND temperature)"),
    ("B4",  "SPEC full text?",         "SPEC:wafer"),
    ("B5",  "SPEC full-text AND x4",   "SPEC:(wafer AND temperature AND pressure AND control)"),
    ("B6",  "inventionTitle field",    'inventionTitle:"wafer temperature"'),
    # ── C. 404 semantics on clean queries ────────────────────────────
    ("C1",  "gibberish single word",   "zzqwxpqrsemifab"),
    ("C2",  "gibberish AND pair",      "zzqwxpqrsemifab AND wafer"),
    # ── D. single-word baselines (pin space=OR vs AND semantics) ─────
    ("D1",  "single word: wafer",      "wafer"),
    ("D2",  "single word: temperature","temperature"),
    ("D3",  "single word: pressure",   "pressure"),
    ("D4",  "single word: control",    "control"),
    # ── E. poison isolation & candidate safe dialects ────────────────
    ("E1",  "bare 3-AND chain",        "wafer AND temperature AND pressure"),
    ("E2",  "single paren 4-AND",      "(wafer AND temperature AND pressure AND control)"),
    ("E3",  "chained parens 3-term",   "(wafer AND temperature) AND pressure"),
    ("E4",  "chained parens 4-term",   "(wafer AND temperature) AND (pressure AND control)"),
    ("E5",  "phrase alone",            '"wafer temperature"'),
    ("E6",  "phrase AND word",         '"wafer temperature" AND pressure'),
    ("E7",  "paren phrase AND word",   '("wafer temperature") AND pressure'),
    ("E8",  "bare 3-AND w/ common word","semiconductor AND wafer AND control"),
    ("E9",  "paren2 AND common word",  "(semiconductor AND wafer) AND control"),
    ("E10", "nested parens",           "semiconductor AND (wafer AND control)"),
    ("E11", "phrase AND word b",       '"closed loop" AND wafer'),
    ("E12", "paren-OR 3-term",         "(temperature OR thermal) AND wafer"),
]


def _post(q: str, api_key: str) -> dict:
    body = {
        "q": q,
        "pagination": {"offset": 0, "limit": 1},
        "fields": ["applicationMetaData.inventionTitle"],
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-API-Key": api_key})
    try:
        with urllib.request.urlopen(req, timeout=40) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return {"status": resp.status, "count": payload.get("count")}
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = json.loads(exc.read().decode("utf-8")).get(
                "detailedMessage", "")
        except Exception:
            pass
        return {"status": exc.code, "count": None,
                "detail": detail[:90]}
    except Exception as exc:  # network etc.
        return {"status": "ERR", "count": None,
                "detail": str(exc)[:90]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", default=None,
        help="comma list of probe ids/groups to run, e.g. A,B or A1,A3")
    parser.add_argument("--key", default=None,
                        help="USPTO_API_KEY (default: env USPTO_API_KEY)")
    args = parser.parse_args()

    api_key = args.key or os.getenv("USPTO_API_KEY", "")
    if not api_key:
        print("USPTO_API_KEY not set (env or --key) — cannot run.", file=sys.stderr)
        return 2

    wanted = [p for p in PROBES
              if not args.only
              or p[0] in args.only.split(",")
              or (len(p[0]) == 2 and p[0][0] in args.only.split(","))]
    if not wanted:
        print(f"--only {args.only!r} matched no probes.", file=sys.stderr)
        return 2

    print(f"{'id':<5}{'dialect':<28}{'status':<8}{'count':<12}detail")
    print("-" * 78)
    for pid, label, q in wanted:
        r = _post(q, api_key)
        print(f"{pid:<5}{label:<28}{r['status']!s:<8}"
              f"{str(r['count']):<12}{r.get('detail', '')}")
    print()
    print("Reading (2026-09-07 matrix results, both rounds):")
    print("  A5/A4=7609 and A3=84 (2 ANDs parse and count precisely);")
    print("  A2/A7/E2/E4 (3+ ANDs in any paren layout) all 404 -> the")
    print("  endpoint's hard stop is 2 AND operators; OR groups (A12) and")
    print("  quoted phrases (E5=136, E11=9) are fine inside that budget.")
    print("  E6/E7 404s are genuine title-corpus zeros, not parser poison.")
    print("  D1-D4 sum to the A13/A6 space-join counts -> space = OR, so")
    print("  space-flattening a query is loosening to noise, never a")
    print("  precision rescue.")
    print("  B1-B6 all 404 -> field tags are unsupported: TTL/ABST/SPEC")
    print("  cannot force full-text search; the corpus is title-level.")
    print("  C1/C2 404 -> genuine zero on clean single-token queries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
