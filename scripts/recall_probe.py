#!/usr/bin/env python3
"""Probe which recall-expansion routes the USPTO-family APIs support.

PatentsView (search.patentsview.org) has been folded into the USPTO
Open Data Portal as of 2026, so the recall expansion needs a verified
transport.  This script probes, on the server (which holds the
credentials):

  1. applications/search free-text q=<application number>   (family route)
  2. applications/search free-text q=<patent number>        (family route)
  3. GET applications/{applicationNumber}                    (single-record)
  4. applications/search free-text q=<CPC subgroup code>    (recheck)
  5. Patent Public Search (ppubs) CPC query, when PPS_API_KEY is set

Usage (server, venv):
    python scripts/recall_probe.py
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_env(path: str) -> None:
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

USPTO_KEY = os.getenv("USPTO_API_KEY", "")
SEARCH_URL = "https://api.uspto.gov/api/v1/patent/applications/search"
FIELDS = ["applicationNumberText", "applicationMetaData.inventionTitle",
          "applicationMetaData.firstApplicantName",
          "applicationMetaData.applicationStatusDescriptionText",
          "applicationMetaData.patentNumber",
          "applicationMetaData.filingDate",
          "applicationMetaData.cpcClassificationBag",
          "parentContinuityBag", "childContinuityBag"]


def _post_json(url, body, headers, timeout=30):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}
    except Exception as e:
        return None, {"error": f"{type(e).__name__}: {e}"}


def _get_json(url, headers, timeout=30):
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}
    except Exception as e:
        return None, {"error": f"{type(e).__name__}: {e}"}


def _search(q: str) -> dict:
    status, data = _post_json(
        SEARCH_URL,
        {"q": q, "pagination": {"offset": 0, "limit": 3},
         "fields": FIELDS,
         "sort": [{"field": "_score", "order": "desc"}]},
        {"X-API-Key": USPTO_KEY})
    items = data.get("patentFileWrapperDataBag") or []
    return {"status": status, "count": data.get("count"),
            "titles": [(_item_title(i) or "?") for i in items[:3]],
            "app_numbers": [_app_number(i) for i in items[:3] if _app_number(i)]}


def _item_title(item: dict) -> str:
    m = item.get("applicationMetaData") or {}
    return (m.get("inventionTitle") or "").strip()


def _app_number(item: dict) -> str:
    return (item.get("applicationNumberText") or "").strip()


def main() -> int:
    print(f"USPTO_API_KEY set: {bool(USPTO_KEY)}")
    print()

    print("=== probe 1: q=<application number> (family fetch) ===")
    print(json.dumps(_search("16884540"), ensure_ascii=False))
    print()

    print("=== probe 2: q=<patent number> (family fetch) ===")
    p2 = _search("11882632")
    print(json.dumps(p2, ensure_ascii=False))
    print()

    # single-record GET, fed by whatever application number probe 2 found
    app_no = ""
    if p2.get("status") == 200:
        status, data = _post_json(
            SEARCH_URL,
            {"q": "11882632", "pagination": {"offset": 0, "limit": 1},
             "fields": ["applicationNumberText"]},
            {"X-API-Key": USPTO_KEY})
        items = data.get("patentFileWrapperDataBag") or []
        if items:
            app_no = _app_number(items[0])
    if app_no:
        print(f"=== probe 3: GET applications/{app_no} (single record) ===")
        status, data = _get_json(
            f"https://api.uspto.gov/api/v1/patent/applications/{app_no}",
            {"X-API-Key": USPTO_KEY})
        print(json.dumps(
            {"status": status, "keys": list((data or {}).keys())[:10],
             "has_title": bool(_item_title(data or {}))},
            ensure_ascii=False))
    else:
        print("=== probe 3: SKIPPED (no application number from probe 2) ===")
    print()

    print("=== probe 4: q=<CPC subgroup code> (recheck) ===")
    print(json.dumps(_search("H05B45/20"), ensure_ascii=False))
    print()

    print("=== probe 5: Patent Public Search (ppubs) CPC query ===")
    pps_key = os.getenv("PPS_API_KEY", "")
    if not pps_key:
        print("SKIPPED — PPS_API_KEY not set in .env")
    else:
        status, data = _post_json(
            "https://ppubs.uspto.gov/dirsearch-public/searches/"
            "searchWithBeFamily",
            {"searchText": "cpc/H05B45/20",
             "databaseFilters": [{"databaseName": "USPAT",
                                  "countryCode": "US"}],
             "sort": [{"fieldName": "patentNumber", "order": "desc"}],
             "pagination": {"offset": 0, "limit": 3}},
            {"X-API-KEY": pps_key, "X-Search-Scope": "US-PGPUB,USPAT"})
        print(json.dumps({"status": status,
                          "keys": list((data or {}).keys())[:10]},
                         ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    sys.exit(main())
