"""Probe the USPTO applications/search API for CPC field-qualified queries.

Decides how the chat-path CPC relevance feedback can inject CPC
constraints into USPTO envelope queries (sources/agents/react_tools.py).

Probes:
  1. target reachability — "air dryer" AND humidity, pages through the
     first 500 hits by _score (limit is capped at 100 per page by the
     API); reports whether application 15273791 (US10150077B2, AIR
     DRYER CONTROL USING HUMIDITY) is reachable and at which rank.
  2. full field path — AND applicationMetaData.cpcClassificationBag
     .cpcClassCode:"B01D53/261"
  3. short field name — AND cpcClassCode:"B01D53/261"
  4. bare CPC symbol — B01D53/261 as a plain query term
  5. field-qualified title — inventionTitle:"air dryer" (sanity check:
     does the q DSL support field qualification at all)
  6. CPC data availability — "air dryer" AND humidity with
     applicationMetaData.cpcClassificationBag requested; reports whether
     the response actually carries CPC codes

Reading: probes 2/3 returning HTTP 200 with count > 0 means CPC field
qualification works and feedback can append an AND clause to q.  Probe
4 working (2/3 failing) means CPC symbols can be OR-ed into q as plain
terms.  All failing means the fallback (CPC definition text -> keyword
expansion) is required.  Probe 6 failing means candidate CPC codes must
come from another data source entirely.

Usage:
    USPTO_API_KEY=... python scripts/uspto_cpc_probe.py
"""
import json
import os
import sys
import urllib.request

USPTO_SEARCH_URL = "https://api.uspto.gov/api/v1/patent/applications/search"
PAGE_LIMIT = 100
FIELDS = ["applicationNumberText", "applicationMetaData.inventionTitle"]
TARGET_APP = "15273791"

PROBES = {
    "1_target_reachable": '"air dryer" AND humidity',
    "2_fieldpath_cpc": (
        '("dry air" OR desiccant OR "air dryer") AND '
        'applicationMetaData.cpcClassificationBag.cpcClassCode:"B01D53/261"'),
    "3_shortfield_cpc": '"dry air" AND cpcClassCode:"B01D53/261"',
    "4_bare_cpc": 'B01D53/261',
    "5_field_title": 'inventionTitle:"air dryer"',
    "6_cpc_in_response": '"air dryer" AND humidity',
}


def _load_env(path: str) -> None:
    """Load KEY=VALUE lines from an env file into os.environ (no
    overrides)."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except OSError:
        pass


def _api_key() -> str:
    key = os.getenv("USPTO_API_KEY") or os.getenv("USPTO_DOWNLOAD_API_KEY")
    if key:
        return key
    _load_env(".env")
    _load_env("../.env")
    key = os.getenv("USPTO_API_KEY") or os.getenv("USPTO_DOWNLOAD_API_KEY")
    if not key:
        sys.exit("USPTO_API_KEY not found in env or .env — set it first")
    return key


def _search(key: str, q: str, offset: int, limit: int,
            fields: list = FIELDS) -> dict:
    body = json.dumps({
        "q": q,
        "pagination": {"offset": offset, "limit": limit},
        "fields": fields,
        "sort": [{"field": "_score", "order": "desc"}],
    })
    req = urllib.request.Request(
        USPTO_SEARCH_URL, data=body.encode("utf-8"),
        headers={"Content-Type": "application/json", "X-API-Key": key})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)


def _row(item: dict) -> str:
    meta = item.get("applicationMetaData") or {}
    app = item.get("applicationNumberText") or ""
    title = (meta.get("inventionTitle") or "")[:70]
    return f"  {app} | {title}"


def _probe_target_reachable(key: str, q: str) -> None:
    """Page through the first 500 hits looking for TARGET_APP."""
    rank = 0
    for page in range(5):
        try:
            data = _search(key, q, offset=page * PAGE_LIMIT,
                           limit=PAGE_LIMIT)
        except urllib.error.HTTPError as exc:
            print(f"HTTP {exc.code}: {exc.read()[:300]!r}")
            return
        except Exception as exc:
            print(f"request failed: {exc}")
            return
        bag = data.get("patentFileWrapperDataBag") or []
        if page == 0:
            print(f"count: {data.get('count')}")
        for item in bag:
            rank += 1
            if item.get("applicationNumberText") == TARGET_APP:
                print(f"{TARGET_APP} rank #{rank}:")
                print(_row(item))
                return
        if len(bag) < PAGE_LIMIT:
            break
    print(f"{TARGET_APP}: NOT in top {rank}")


def _probe_cpc_in_response(key: str, q: str) -> None:
    fields = FIELDS + ["applicationMetaData.cpcClassificationBag"]
    try:
        data = _search(key, q, offset=0, limit=3, fields=fields)
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read()[:300]!r}")
        return
    except Exception as exc:
        print(f"request failed: {exc}")
        return
    bag = data.get("patentFileWrapperDataBag") or []
    print(f"count: {data.get('count')}, returned: {len(bag)}")
    for item in bag[:3]:
        meta = item.get("applicationMetaData") or {}
        cpc = meta.get("cpcClassificationBag")
        codes = ([e.get("cpcClassCode") for e in cpc
                  if isinstance(e, dict) and e.get("cpcClassCode")]
                 if isinstance(cpc, list) else None)
        print(f"  {item.get('applicationNumberText')} | cpc={codes}")


def main() -> None:
    key = _api_key()
    for name, q in PROBES.items():
        print(f"==================== {name} ====================")
        if name.startswith("1"):
            _probe_target_reachable(key, q)
        elif name.startswith("6"):
            _probe_cpc_in_response(key, q)
        else:
            try:
                data = _search(key, q, offset=0, limit=5)
            except urllib.error.HTTPError as exc:
                print(f"HTTP {exc.code}: {exc.read()[:300]!r}")
                continue
            except Exception as exc:
                print(f"request failed: {exc}")
                continue
            bag = data.get("patentFileWrapperDataBag") or []
            print(f"count: {data.get('count')}")
            for item in bag[:5]:
                print(_row(item))


if __name__ == "__main__":
    main()
