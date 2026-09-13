#!/usr/bin/env python3
"""Probe whether the 80-call lawInfos fan-out per query can be eliminated.

Background (2026-09-13): one user query issued ~80 ``/openService/law``
(FLZT) calls, because ``_enrich_baiten_law_status`` makes ONE call per
candidate and the gateway hard-caps ``page_size`` at 10 — so every Baiten
search costs 10 calls.  A production log also showed ~40% of those calls
being exact repeats (the auto-ladder re-issued the same query).

Before optimising, three questions must be settled **against the live
gateway** (the API docs answer none of them definitively):

  probe 1  Does ``/openService/search`` already return legal-status fields
           in ``documents[].field_values`` that we are discarding?  Our
           client sends no field selection, so whatever the gateway returns
           lands in ``_raw``.  If a status field is already there, the whole
           enrichment can be deleted — 80 calls become 0.

  probe 2  Does ``/openService/law`` accept MULTIPLE appNum values in one
           call (comma / space / semicolon)?  The docs show a single String,
           but a gateway that happily parses a list would collapse 10 calls
           into 1.  A single-value call is run first as the control.

  probe 3  What is the FULL ``patent_laws[]`` / ``patentLawDeclare_list[]``
           field set?  We currently read only ``law_state`` and
           ``notice_date``; everything else is dropped.

The probe is READ-ONLY (no writes, no state changes) and small enough for
the <1 GB server budget.

Usage (server, venv — the gateway credentials live in .env):
    python scripts/baiten_status_probe.py
    python scripts/baiten_status_probe.py --query 'ti:(散热)' \
        --app-nums CN202510962081.1 CN202310742799.0
"""
import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Defaults are the exact values observed in the 2026-09-13 production log —
# a probe wants an input already known to return data, so a zero result
# means "the field is absent", not "the query missed".
DEFAULT_QUERY = "ti:(压缩空气 OR 空气压缩机 OR 干燥机) AND ti:(湿度 OR 除湿 OR 露点)"
DEFAULT_APP_NUMS = ["CN202510962081.1", "CN202310742799.0"]

# Field-name hints for probe 1 — a search row carrying any of these means we
# are paying for lawInfos to re-fetch something we already had.  Deliberately
# loose: this drives a human eyeball, not a decision.
_STATUS_HINTS = ("law", "status", "state", "legal", "right",
                 "法律", "状态", "有效", "失效", "终止", "撤回", "驳回")

_RAW_DUMP_CHARS = 1800


def _load_env(path):
    """Load .env without overriding the process env.

    Values are stripped of wrapping quotes — the server's .env carries
    them, and an un-stripped value produced a 401 in an earlier probe
    (see build_cpc_vectors.py).
    """
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


_ROOT = os.path.join(os.path.dirname(__file__), "..")
_load_env(os.path.join(_ROOT, ".env"))


def _hr(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def _clip(value, limit=_RAW_DUMP_CHARS):
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + " …(truncated {} chars)".format(len(text) - limit)


def _client():
    from sources.baiten_client import BaitenClient
    from sources.long_task.config import get_baiten_config
    cfg = get_baiten_config()
    if not cfg.get("app_key") or not cfg.get("app_secret"):
        print("!! BAITEN_APP_KEY / BAITEN_APP_SECRET 未配置 —— 无法 probe")
        sys.exit(2)
    print("gateway = {}".format(cfg.get("gateway_url")))
    print("app_key = {}…{} (len={})".format(
        cfg["app_key"][:4], cfg["app_key"][-2:], len(cfg["app_key"])))
    return BaitenClient(cfg["app_key"], cfg["app_secret"],
                        cfg["gateway_url"])


def _docs(body):
    data = body.get("data")
    if isinstance(data, dict) and isinstance(data.get("documents"), list):
        return data["documents"]
    if isinstance(body.get("documents"), list):
        return body["documents"]
    return []


def _unwrap(row):
    for key in ("field_values", "fieldValues"):
        wrapped = row.get(key)
        if isinstance(wrapped, dict):
            return wrapped
    return row


async def probe_search_fields(client, query):
    """Does the search response already carry legal status?"""
    _hr("probe 1 — /openService/search 返回的字段全集（法律状态是否已在其中？）")
    print("query = {!r}".format(query))
    body = await client.search(query, page=1, page_size=10)
    docs = _docs(body)
    summary = body.get("total_hits", body.get("total"))
    print("total_hits = {}, rows = {}".format(summary, len(docs)))
    if not docs:
        print("!! 0 rows —— 无法判断字段集，换一个 query 重跑")
        return set()

    fv_keys, hl_keys = set(), set()
    for row in docs:
        wrapped = _unwrap(row)
        fv_keys |= {str(k) for k in wrapped.keys()}
        hl = row.get("hl_field_values") or row.get("hlFieldValues")
        if isinstance(hl, dict):
            hl_keys |= {str(k) for k in hl.keys()}
    print("field_values 键（{} 条并集）: {}".format(
        len(docs), sorted(fv_keys)))
    if hl_keys:
        print("hl_field_values 键: {}".format(sorted(hl_keys)))

    hits = sorted(k for k in fv_keys
                  if any(h in k.lower() for h in _STATUS_HINTS))
    print("疑似法律状态字段: {}".format(hits or "无"))
    print("--- 首条原文 ---")
    print(_clip(_unwrap(docs[0])))
    return fv_keys


async def probe_batch_app_num(client, app_nums):
    """Can one /openService/law call cover several application numbers?"""
    _hr("probe 2 — /openService/law 是否支持一次传多个 appNum")
    if len(app_nums) < 2:
        print("!! 需要至少两个申请号才能对比，已跳过")
        return
    a, b = app_nums[0], app_nums[1]
    forms = [
        ("对照: 单值 a", a),
        ("对照: 单值 b", b),
        ("逗号 a,b", "{},{}".format(a, b)),
        ("空格 a b", "{} {}".format(a, b)),
        ("分号 a;b", "{};{}".format(a, b)),
    ]
    singles = []
    batched = []
    for label, value in forms:
        try:
            body = await client._request_json(
                "/openService/law",
                {"app_num": value, "law_category": "FLZT"})
        except Exception as exc:
            print("- {:<14} → 失败: {}".format(label, exc))
            continue
        laws = body.get("patent_laws") or []
        declares = body.get("patentLawDeclare_list") or []
        # patent_laws[] carries no appNum of its own; the declare list does.
        apps = sorted({str(x.get("appNum") or "") for x in declares
                       if isinstance(x, dict) and x.get("appNum")}) \
            if declares else []
        print("- {:<14} → patent_laws={} 条, declare={} 条, 含申请号={}".format(
            label, len(laws), len(declares), apps or "-"))
        if label.startswith("对照"):
            singles.append(len(laws))
        else:
            batched.append((label, len(laws), len(apps)))

    print()
    if singles and batched:
        want = sum(singles)
        best = max(batched, key=lambda t: t[1])
        label, got, covered = best
        if got >= want and covered >= 2:
            print("判读：{} 拿到 {} 条 ≈ 单值之和 {}，且覆盖 2 个申请号".format(
                label, got, want))
            print("      → 网关支持批量，10 次调用可压成 1 次。")
        else:
            print("判读：批量形态最多只拿到 {} 条（单值之和 {}，覆盖 {} 个号）"
                  .format(got, want, covered))
            print("      → 不支持批量，走记忆化路线。")
    else:
        print("判读：样本不足，人工看上面的行。")


async def probe_law_fields(client, app_num):
    """Full field inventory of one FLZT + one FSWX payload."""
    _hr("probe 3 — /openService/law 的完整字段集（我们目前只用 2 个）")
    for category in ("FLZT", "FSWX"):
        try:
            body = await client.query_law_infos(app_num, category)
        except Exception as exc:
            print("[{}] 失败: {}".format(category, exc))
            continue
        print("[{}] 顶层键: {}".format(category, sorted(body.keys())))
        laws = body.get("patent_laws") or []
        if laws and isinstance(laws[0], dict):
            print("[{}] patent_laws[0] 全文: {}".format(
                category, _clip(laws[0], 900)))
        declares = body.get("patentLawDeclare_list") or []
        if declares and isinstance(declares[0], dict):
            print("[{}] patentLawDeclare_list[0] 键: {}".format(
                category, sorted(declares[0].keys())))
        count = body.get("patent_laws_count")
        if count:
            print("[{}] patent_laws_count: {}".format(
                category, _clip(count, 400)))
        custom = body.get("custom_info")
        if custom:
            print("[{}] custom_info: {}".format(
                category, _clip(custom, 300)))


async def cross_check(client, query, limit=3):
    """Side-by-side: what the search row already tells us vs. what lawInfos adds."""
    _hr("probe 4 — 交叉对照：搜索行字段 vs lawInfos 补出来的东西")
    body = await client.search(query, page=1, page_size=10)
    docs = [_unwrap(r) for r in _docs(body)]
    if not docs:
        print("!! 搜索 0 行，跳过")
        return
    print("对比前 {} 条：搜索行已有的键 → lawInfos 单独补出来的状态".format(
        min(limit, len(docs))))
    for row in docs[:limit]:
        pn = str(row.get("pn") or "")
        an = str(row.get("an") or "")
        try:
            law = await client.query_law_infos(an, "FLZT")
        except Exception as exc:
            print("- {} ({}) → lawInfos 失败: {}".format(pn, an, exc))
            continue
        laws = law.get("patent_laws") or []
        latest = ""
        if laws and isinstance(laws[0], dict):
            latest = str(laws[0].get("law_state") or "")
        print("- {} ({})".format(pn, an))
        print("    搜索行返回的键: {}".format(sorted(row.keys())))
        print("    lawInfos 得到: patent_laws={} 条, 最新状态={!r}".format(
            len(laws), latest))


async def main():
    parser = argparse.ArgumentParser(
        description="Baiten legal-status fan-out probe (read-only)")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--app-nums", nargs="*", default=DEFAULT_APP_NUMS)
    args = parser.parse_args()

    client = _client()
    await probe_search_fields(client, args.query)
    await probe_batch_app_num(client, args.app_nums)
    await probe_law_fields(client, args.app_nums[0])
    await cross_check(client, args.query)

    _hr("结论怎么读")
    print("probe 1 出现法律状态字段  → 直接删掉富化，80 次调用归零")
    print("probe 2 批量有数据        → 按批查，10 次压成 1 次")
    print("两者都不成立              → 退回本请求内按 app_num 记忆化")
    print("                            （生产日志实测约 40% 是纯重复）")
    print()
    print("把以上完整输出贴回来即可定方案。")


if __name__ == "__main__":
    asyncio.run(main())
