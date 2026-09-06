# -*- coding: utf-8 -*-
"""US 外观(design) B 方案可行性探针 —— V1-V5 一次性实测脚本。

用途: 回填 docs/superpowers/specs/2026-09-06-us-design-clearance-feasibility.md
§5 待验清单。流式/低内存(不落库, 随取随弃), 输出 JSON 结论表, 不触碰密钥明文。

待验项:
  V1  USPTO applications/search 对 D 号段的查询形态与字段支持
      (patentNumber 带 D 前缀 vs 纯 digits; 结果里分类字段的键名)
  V2  Google Patents 设计专利页/检索: Locarno 与 USPC-D 可得性; XHR 的
      design 检索/过滤参数行为
  V3  patentimages.storage.googleapis.com 直链 200 验证(取真实页 <img> src)
  V4  5 个通用品名词检索的 design 召回抽样(XHR, 统计 Top 内 USD 占比)
  V5  年代段设计专利页面/图可得性(尽力而为; 过滤参数不支持则如实报告)

运行(服务器, 有 .env 凭据):
  PYTHONUTF8=1 python scripts/design_feasibility_probe.py -o probe_results.json
  进度打 stderr, 结论 JSON 写 -o 指定文件。
"""
import argparse
import asyncio
import json
import os
import re
import sys

import httpx

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
USPTO_SEARCH = "https://api.uspto.gov/api/v1/patent/applications/search"
GP_XHR = "https://patents.google.com/xhr/query"
GP_PAGE = "https://patents.google.com/patent/{pid}/en"
GP_UA = UA

# V4 抽样品名词 —— 仅探针工具数据, 不进入产品提示词(仓库红线: 检索增强不固化提问词)。
SAMPLE_TERMS = ["robot toy", "bottle", "office chair", "desk lamp", "headphones"]

# V1 探针: D 号与 digits(带 D 前缀设计号, 已知 2005 年 Apple 外观 D504,889)。
PROBE_D = "D504889"
PROBE_DIGITS = "504889"


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
    out = {"http": resp.status_code}
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
        out["note"] = f"HTTP {resp.status_code}; 401=无 key 或 key 无效"
    return out


async def v1(client: httpx.AsyncClient) -> dict:
    log("V1 USPTO D 号段查询形态…")
    return {
        "v1_patentNumber_with_D": await _uspto_search(
            client, f'applicationMetaData.patentNumber:"{PROBE_D}"'),
        "v1_patentNumber_digits_only": await _uspto_search(
            client, f'applicationMetaData.patentNumber:"{PROBE_DIGITS}"'),
        "v1_title_query_toy": await _uspto_search(
            client, 'applicationMetaData.inventionTitle:"toy"'),
    }


async def _gp_page(client: httpx.AsyncClient, pid: str) -> dict:
    out = {"pid": pid, "http": None}
    try:
        resp = await client.get(GP_PAGE.format(pid=pid), headers={"User-Agent": GP_UA})
        out["http"] = resp.status_code
        if resp.status_code != 200:
            return out
        html = resp.text
        out["has_locarno"] = "Locarno" in html
        out["has_uspc"] = bool(re.search(r"USPC|United States Patent Classification", html))
        imgs = re.findall(
            r'https://patentimages\.storage\.googleapis\.com/[^"\'>\s]+',
            html)
        out["patentimages_count"] = len(imgs)
        out["first_image"] = imgs[0] if imgs else None
        # 分类段落取样(供人工确认 Locarno 号段写法)。
        m = re.search(r"Locarno Classification.{0,600}", html, re.S)
        out["locarno_snippet"] = (re.sub(r"<[^>]+>", " ", m.group(0))[:400]
                                  if m else None)
        return out
    except Exception as exc:  # noqa: BLE001 探针容忍一切
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out


async def v2_v3(client: httpx.AsyncClient) -> dict:
    log("V2/V3 Google Patents 设计页字段与图直链…")
    page = await _gp_page(client, "USD504889S")
    img_check = None
    if page.get("first_image"):
        try:
            r = await client.get(page["first_image"],
                                 headers={"User-Agent": GP_UA})
            img_check = {"http": r.status_code, "bytes": len(r.content),
                         "content_type": r.headers.get("content-type")}
        except Exception as exc:  # noqa: BLE001
            img_check = {"error": f"{type(exc).__name__}: {exc}"}
    xhr = {}
    for label, q in [("design_number", "USD504889S"), ("plain_term", "toy snake")]:
        try:
            resp = await client.get(
                GP_XHR,
                params={"url": f"q={q}", "exp": ""},
                headers={"User-Agent": GP_UA})
            xhr[label] = {"http": resp.status_code, "body_head": resp.text[:200]}
        except Exception as exc:  # noqa: BLE001
            xhr[label] = {"error": f"{type(exc).__name__}: {exc}"}
    return {"v2_page": page, "v3_image_direct": img_check, "xhr_raw": xhr}


async def v4_v5(client: httpx.AsyncClient) -> dict:
    log("V4/V5 品名词 design 召回抽样与年代过滤尝试…")
    terms = []
    for term in SAMPLE_TERMS:
        try:
            resp = await client.get(
                GP_XHR,
                params={"url": f"q={term}", "exp": ""},
                headers={"User-Agent": GP_UA})
            rec = {"term": term, "http": resp.status_code}
            if resp.status_code == 200:
                data = resp.json()
                results = (((data.get("results") or {}).get("cluster") or [{}])[0]
                           .get("result") or [])
                pubs = [r.get("patent", {}).get("publication_number", "")
                        for r in results]
                rec["top_total"] = len(pubs)
                rec["design_count_in_top"] = sum(
                    1 for p in pubs if p.upper().startswith("USD"))
                rec["sample_pubs"] = pubs[:5]
            terms.append(rec)
        except Exception as exc:  # noqa: BLE001
            terms.append({"term": term, "error": f"{type(exc).__name__}: {exc}"})
    # V5: 尝试按公开日区间过滤(Google Patents XHR url 内联 after/before)。
    date_filter = {}
    try:
        url = "q=(toy)&after=publication:19900101&before=publication:19991231"
        resp = await client.get(GP_XHR, params={"url": url, "exp": ""},
                                headers={"User-Agent": GP_UA})
        date_filter = {"http": resp.status_code,
                       "body_head": resp.text[:200],
                       "note": "若 http=200 且命中为 1990s 件则过滤可用; "
                               "否则该过滤器不受支持(报告为准)"}
    except Exception as exc:  # noqa: BLE001
        date_filter = {"error": f"{type(exc).__name__}: {exc}"}
    return {"v4_term_sample": terms, "v5_date_filter": date_filter}


async def main(out_path: str) -> None:
    results: dict = {"vision_config": {}, "v1": {}, "v2_v3": {}, "v4_v5": {}}
    # 记录生效视觉配置(不打印密钥)。
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
