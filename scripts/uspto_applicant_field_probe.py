#!/usr/bin/env python3
"""Probe applications/search applicant-field query syntax (run on server).

Why: 2026-09-03 production log — an applicant-constrained question's
ladder 404'd end to end (multi-concept bracket AND) while the plain
company-name full-text query returned tens of thousands of transfer
records.  The fix (search_query_builder.render_applicant_query) supports
three anchor syntaxes; which one the live gateway accepts for the
applicant FIELD must be verified against the real endpoint before
``USPTO_APPLICANT_SYNTAX`` is switched from the safe ``phrase`` default.

Usage (server, ~/langsistance):
    python scripts/uspto_applicant_field_probe.py [applicant] [techword]

Query terms default to neutral placeholders — pass a real company name
and one technology word as argv when re-checking a specific case.  Every
row prints as it completes (streaming; well under the server RAM
budget).  Reads USPTO_API_KEY from the environment when present.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

URL = "https://api.uspto.gov/api/v1/patent/applications/search"
FIELDS = [
    "applicationNumberText",
    "applicationMetaData.inventionTitle",
    "applicationMetaData.firstApplicantName",
    "applicationMetaData.patentNumber",
]


def probe(name: str, q: str) -> dict:
    body = {
        "q": q,
        "pagination": {"offset": 0, "limit": 3},
        "fields": FIELDS,
        "sort": [{"field": "_score", "order": "desc"}],
    }
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("USPTO_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            first = ""
            bag = data.get("patentFileWrapperDataBag") or []
            if bag and isinstance(bag[0], dict):
                meta = bag[0].get("applicationMetaData") or {}
                first = str(meta.get("inventionTitle")
                            or bag[0].get("applicationNumberText"))[:60]
            print(f"{name:44s} 200 count={data.get('count'):>8} "
                  f"first={first}", flush=True)
            return {"name": name, "q": q, "ok": True,
                    "count": data.get("count")}
    except urllib.error.HTTPError as exc:
        preview = exc.read().decode()[:120].replace("\n", " ")
        print(f"{name:44s} {exc.code} {preview}", flush=True)
        return {"name": name, "q": q, "ok": False,
                "status": exc.code}
    except Exception as exc:
        print(f"{name:44s} ERROR {type(exc).__name__}: {exc}", flush=True)
        return {"name": name, "q": q, "ok": False, "error": str(exc)}


def main() -> int:
    applicant = sys.argv[1] if len(sys.argv) > 1 else "acme"
    tech = sys.argv[2] if len(sys.argv) > 2 else "adhesive"
    name_parts = applicant.split()
    anchor = '"' + applicant + '"'
    word_anchor = applicant.replace(" ", " ")

    rows = [
        # 基线: 引号短语(当前默认 anchor 语法) — 决定 phrase 模式是否可直接用
        (f"A1 phrase anchor", anchor),
        (f"A2 phrase anchor AND tech", f"{anchor} AND {tech}"),
        # 字段语法候选 — 哪个被网关接受
        (f"B1 firstApplicantName:(bare)", f"firstApplicantName:({word_anchor})"),
        (f"B2 firstApplicantName:(quoted)", f'firstApplicantName:("{applicant}")'),
        (f"B3 dotted field", f"applicationMetaData.firstApplicantName:({word_anchor})"),
        (f"B4 assigneeBag field", f"assigneeBag.assigneeNameText:({word_anchor})"),
        (f"B5 field AND tech",
         f"firstApplicantName:({word_anchor}) AND {tech}"),
        # 保底: 去括号纯空格词形(已知 200, 噪声靠 gate 过滤)
        (f"C1 space fallback AND tech", f"{word_anchor} {tech}"),
    ]
    results = [probe(name, q) for name, q in rows]
    ok_rows = [r for r in results if r.get("ok")]
    print("\n== summary ==", flush=True)
    for r in ok_rows:
        count = r.get("count") or 0
        if isinstance(count, int) and 0 < count < 50000:
            print(f"USABLE  {r['name']} (count={count})", flush=True)
    print("\nDecision:", flush=True)
    print(" 若 B1/B3 任一 200 且 count 显著小于 A2 → "
          "export USPTO_APPLICANT_SYNTAX=field 启用字段锚定", flush=True)
    print(" 若仅 A1/A2 可用 → 保持默认 phrase(USPTO_APPLICANT_SYNTAX=phrase)",
          flush=True)
    print(" 若两者都 404 而 C1 可用 → export USPTO_APPLICANT_SYNTAX=space",
          flush=True)
    print(" field 字段名如需调整: export USPTO_APPLICANT_FIELD=<字段>",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
