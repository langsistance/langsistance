"""Probe the USPTO applications/search API for CPC field-qualified queries.

Decides how the chat-path CPC relevance feedback can inject CPC
constraints into USPTO envelope queries (sources/agents/react_tools.py).

Probes:
  1. target reachability — "air dryer" AND humidity, top 500 by _score;
     reports whether application 15273791 (US10150077B2, AIR DRYER
     CONTROL USING HUMIDITY) is reachable and at which rank.
  2. full field path — AND applicationMetaData.cpcClassificationBag
     .cpcClassCode:"B01D53/261"
  3. short field name — AND cpcClassCode:"B01D53/261"
  4. bare CPC symbol — B01D53/261 as a plain query term

Reading: probes 2/3 returning HTTP 200 with count > 0 means CPC field
qualification works and feedback can append an AND clause to q.  Probe
4 working (2/3 failing) means CPC symbols can be OR-ed into q as plain
terms.  All failing means the fallback (CPC definition text -> keyword
expansion) is required.

Usage:
    USPTO_API_KEY=... python scripts/uspto_cpc_probe.py
"""
import json
import os
import sys
import urllib.request

USPTO_SEARCH_URL = "https://api.uspto.gov/api/v1/patent/applications/search"
FIELDS = ["applicationNumberText", "applicationMetaData.inventionTitle"]
TARGET_APP = "15273791"

PROBES = {
    "1_target_reachable": '"air dryer" AND humidity',
    "2_fieldpath_cpc": (
        '("dry air" OR desiccant OR "air dryer") AND '
        'applicationMetaData.cpcClassificationBag.cpcClassCode:"B01D53/261"'),
    "3_shortfield_cpc": '"dry air" AND cpcClassCode:"B01D53/261"',
    "4_bare_cpc": 'B01D53/261',
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


def _search(key: str, q: str, limit: int) -> dict:
    body = json.dumps({
        "q": q,
        "pagination": {"offset": 0, "limit": limit},
        "fields": FIELDS,
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


def main() -> None:
    key = _api_key()
    for name, q in PROBES.items():
        print(f"==================== {name} ====================")
        limit = 500 if name.startswith("1") else 5
        try:
            data = _search(key, q, limit)
        except urllib.error.HTTPError as exc:
            print(f"HTTP {exc.code}: {exc.read()[:300]!r}")
            continue
        except Exception as exc:
            print(f"request failed: {exc}")
            continue
        bag = data.get("patentFileWrapperDataBag") or []
        print(f"count: {data.get('count')}")
        if name.startswith("1"):
            hit = [i for i in bag
                   if i.get("applicationNumberText") == TARGET_APP]
            if hit:
                print(f"{TARGET_APP} rank #{bag.index(hit[0]) + 1}:")
                print(_row(hit[0]))
            else:
                print(f"{TARGET_APP}: NOT in top {len(bag)}")
        else:
            for i in bag[:5]:
                print(_row(i))


if __name__ == "__main__":
    main()
