# -*- coding: utf-8 -*-
"""US 外观(design) B 方案可行性探针 —— V1-V5 一次性实测脚本(v2)。

v2 修订(2026-09-06 首跑反馈):
  1. 加载 .env(V1 全 401 根因: 独立脚本未 dotenv)；
  2. 设计页 id 从 XHR 结果取(真实 id 带 S1 后缀, 如 patent/USD504889S1/en)；
  3. V4 计数按 Google id 正则识别设计件(US D 与 EM/CN S 号), 并尝试 type 过滤；
  4. V5 用已证可用的 publication 日期过滤按年代段抽样并验证页面图可得。

用途/输出/运行方式同 v1(见文件头注释):
  PYTHONUTF8=1 python scripts/design_feasibility_probe.py -o probe_results.json
"""
import argparse
import asyncio
import json
import os
import re
import sys

import httpx


def _load_dotenv() -> None:
    for name in (".env",):
        try:
            with open(name, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key, val = key.strip(), val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
        except OSError:
            return


_load_dotenv()

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
USPTO_SEARCH = "https://api.uspto.gov/api/v1/patent/applications/search"
GP_XHR = "https://patents.google.com/xhr/query"
GP_PAGE = "https://patents.google.com/patent/{pid}/en"

# V4 抽样品名词 —— 仅探针工具数据, 不进入产品提示词。
SAMPLE_TERMS = ["robot toy", "bottle", "office chair", "desk lamp", "headphones"]
# V1 探针号: Apple 2005 外观 D504,889(经 XHR 已证存在于 Google Patents)。
PROBE_D = "D504889"
PROBE_DIGITS = "504889"
DESIGN_LOOKUP = "USD504889S"

# Google id 设计件识别: patent/USD...S1/en、EM...S/en、CN...S/en 等。
_DESIGN_ID_RE = re.compile(r"/(?:USD?\d+|[A-Z]{2}\d+)[A-Z]?S\d*/en", re.I)


def log(msg: str) -> None:
    print(f"[probe] {msg}", file=sys.stderr, flush=True)


async def _uspto_search(client: httpx.AsyncClient, query: str,
                        limit: int = 3) -> dict:
    key = os.getenv("USPTO_API_KEY", "")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if key:
        headers["X-API-Key"] = key
    body = {
        "q": query,
        "pagination": {"offset": 0, "limit": limit},
        "fields": [
            "applicationNumberText",
            "applicationMetaData.patentNumber",
            "applicationMetaData.earliestPublicationNumber",
            "applicationMetaData.inventionTitle",
        ],
    }
    resp = await client.post(USPTO_SEARCH, headers=headers, json=body)
    out = {"http": resp.status_code, "api_key_present": bool(key)}
    if resp.status_code == 200:
        data = resp.json()
        results = (
            data.get("patentFileWrapperDataBag", None)
            or data.get("results", None)
            or data.get("patentFileBag", [])
        )
        items = results if isinstance(results, list) else []
        out["hit_count"] = len(items)
        out["first_keys"] = sorted(items[0].keys()) if items else []
        meta = (items[0].get("applicationMetaData") or {}) if items else {}
        out["meta_keys"] = sorted(meta.keys())
        out["meta_classification_present"] = any(
            "class" in k.lower() for k in meta) if meta else None
        out["sample_patent_number"] = meta.get("patentNumber") if meta else None
        out["sample_title"] = meta.get("inventionTitle") if meta else None
    else:
        out["note"] = (f"HTTP {resp.status_code} (api_key_present={bool(key)}); "
                       "401 且 key 缺失=确认 .env 无该键; 401 且 key 在=键无效")
    return out


async def v1(client: httpx.AsyncClient) -> dict:
    log("V1 USPTO D 号段查询形态(.env 已加载)…")
    return {
        "v1_patentNumber_with_D": await _uspto_search(
            client, f'applicationMetaData.patentNumber:"{PROBE_D}"'),
        "v1_patentNumber_digits_only": await _uspto_search(
            client, f'applicationMetaData.patentNumber:"{PROBE_DIGITS}"'),
        "v1_title_query_toy": await _uspto_search(
            client, 'applicationMetaData.inventionTitle:"toy"'),
    }


async def _xhr(client: httpx.AsyncClient, url_query: str) -> dict:
    """XHR 检索; url_query 为 url 参数原文(未编码由 params 处理)。"""
    try:
        resp = await client.get(GP_XHR, params={"url": url_query, "exp": ""},
                                headers={"User-Agent": UA})
        rec = {"http": resp.status_code}
        if resp.status_code != 200:
            rec["body_head"] = resp.text[:150]
            return rec
        data = resp.json()
        cluster = (((data.get("results") or {}).get("cluster") or [{}])[0]
                   .get("result") or [])
        rec["total"] = ((data.get("results") or {}).get("total_num_results"))
        rec["items"] = []
        for r in cluster[:10]:
            pat = r.get("patent", {})
            rec["items"].append({
                "id": r.get("id", ""),
                "pub": pat.get("publication_number", ""),
                "title": (pat.get("title") or "").strip()[:80],
                "cls": sorted(
                    set((pat.get("classifications") or "").split(";"))
                ) if pat.get("classifications") else [],
            })
        return rec
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


async def _gp_page(client: httpx.AsyncClient, pid: str) -> dict:
    out = {"pid": pid, "http": None}
    try:
        resp = await client.get(GP_PAGE.format(pid=pid),
                                headers={"User-Agent": UA})
        out["http"] = resp.status_code
        if resp.status_code != 200:
            return out
        html = resp.text
        out["has_locarno"] = "Locarno" in html
        out["has_uspc"] = "United States Patent Classification" in html
        imgs = re.findall(
            r'https://patentimages\.storage\.googleapis\.com/[^"\'>\s]+',
            html)
        out["patentimages_count"] = len(imgs)
        out["first_image"] = imgs[0] if imgs else None
        m = re.search(r"Locarno Classification.{0,600}", html, re.S)
        out["locarno_snippet"] = (re.sub(r"<[^>]+>", " ", m.group(0))[:400]
                                  if m else None)
        return out
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out


async def v2_v3(client: httpx.AsyncClient) -> dict:
    log("V2/V3 设计号 XHR → 页 id → Locarno/USPC/图直链…")
    lookup = await _xhr(client, f"q={DESIGN_LOOKUP}")
    pid = None
    if lookup.get("items"):
        pid = lookup["items"][0].get("id", "").removeprefix("patent/")
    page = await _gp_page(client, pid) if pid else {"pid": None,
                                                    "note": "lookup 无命中"}
    img_check = None
    if page.get("first_image"):
        try:
            r = await client.get(page["first_image"],
                                 headers={"User-Agent": UA})
            img_check = {"http": r.status_code, "bytes": len(r.content),
                         "content_type": r.headers.get("content-type")}
        except Exception as exc:  # noqa: BLE001
            img_check = {"error": f"{type(exc).__name__}: {exc}"}
    # XHR 是否支持 design 过滤: 尝试若干参数形态, 对比总量变化。
    filters = {}
    variants = {
        "type_param": "q=toy snake&type=DESIGN",
        "q_type_token": "q=toy snake type:DESIGN",
        "plain": "q=toy snake",
    }
    for label, q in variants.items():
        r = await _xhr(client, q)
        filters[label] = {"total": r.get("total"), "http": r.get("http")}
    return {"v2_xhr_lookup": lookup, "v2_page": page,
            "v3_image_direct": img_check, "filter_trials": filters}


async def v4_v5(client: httpx.AsyncClient) -> dict:
    log("V4 品名词设计召回 + V5 年代段图可得性…")
    terms = []
    for term in SAMPLE_TERMS:
        r = await _xhr(client, f"q={term}")
        items = r.get("items", [])
        designs = [it for it in items if _DESIGN_ID_RE.search(it.get("id", ""))]
        us_designs = [it for it in items if re.search(r"/USD?\d+S\d*/en", it.get("id", ""))]
        terms.append({
            "term": term, "http": r.get("http"), "total": r.get("total"),
            "design_in_top10": len(designs), "us_design_in_top10": len(us_designs),
            "sample_ids": [it["id"] for it in items[:5]],
        })
    eras = {}
    for label, after, before in [
        ("era_1990s", "publication:19900101", "publication:19991231"),
        ("era_2000s", "publication:20000101", "publication:20091231"),
        ("era_2010s", "publication:20100101", "publication:20191231"),
    ]:
        q = f"q=(toy)&after={after}&before={before}"
        r = await _xhr(client, q)
        pid = None
        for it in r.get("items", []):
            if _DESIGN_ID_RE.search(it.get("id", "")):
                pid = it["id"].removeprefix("patent/")
                break
        page = await _gp_page(client, pid) if pid else {"pid": None}
        eras[label] = {
            "total": r.get("total"),
            "first_design_id": pid,
            "page_http": page.get("http"),
            "patentimages_count": page.get("patentimages_count"),
            "first_image": page.get("first_image"),
        }
    return {"v4_term_sample": terms, "v5_era_pages": eras}


async def main(out_path: str) -> None:
    results: dict = {"vision_config": {}, "v1": {}, "v2_v3": {}, "v4_v5": {}}
    try:
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read("config.ini", encoding="utf-8")
        lt = cfg["LONG_TASK"] if cfg.has_section("LONG_TASK") else {}
        results["vision_config"] = {
            "vision_enabled": lt.get("vision_enabled"),
            "vision_provider": lt.get("vision_provider"),
            "vision_model": lt.get("vision_model"),
        }
    except Exception as exc:  # noqa: BLE001
        results["vision_config"] = {"error": f"{type(exc).__name__}: {exc}"}

    async with httpx.AsyncClient(timeout=25,
                                 follow_redirects=True) as client:
        results["v1"] = await v1(client)
        results["v2_v3"] = await v2_v3(client)
        results["v4_v5"] = await v4_v5(client)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    log(f"done → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="design_probe_results.json")
    args = ap.parse_args()
    asyncio.run(main(args.output))
