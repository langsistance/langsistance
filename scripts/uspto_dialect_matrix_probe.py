#!/usr/bin/env python3
"""Probe which query dialect the USPTO applications/search API actually
understands.

Background (2026-09-03 / 2026-09-07 production logs): the endpoint
answers many parenthesized/phrase AND-OR queries — and bare multi-term
AND chains — with HTTP 404 "No matching records found" (a FALSE zero:
the same words space-joined return 200), while other shapes work.
The query ladder is generated in a PPS-style dialect (parens, quotes,
explicit AND/OR) that this endpoint partially rejects, so a whole
recall ladder can collapse into false zeros.

This script POSTs a dialect matrix and prints status/count per query.
The result decides how the ladder generator (search_query_builder)
should phrase queries:

  - space-joined plain words returning 200 with counts => the generator
    should emit that shape (or the flatten fallback must always fire);
  - TTL:/ABST:/SPEC: field qualification working => phrase the ladder
    with field tags for precision;
  - phrase queries ("...") working only space-joined => drop quotes.

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
    print("Reading:")
    print("  A6/A1 200 but A7/A2 (explicit AND chain) 404 -> emit space-joined"
          " word bags; explicit AND is the poison.")
    print("  A8 404 but A9/A10 200 -> quotes break the endpoint; drop quotes.")
    print("  B4 SPEC count ~millions -> full text exists and SPEC: works;")
    print("    B5 SPEC AND-chain count > 0 -> use SPEC:(...) for multi-concept.")
    print("  B1/B2/B3/B6 200 -> field tags work; use TTL:/ABST: to narrow.")
    print("  C1/C2 404 with detail 'No matching records' -> genuine zero;")
    print("    keep 404-as-zero-hit ONLY for shapes proven clean above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
