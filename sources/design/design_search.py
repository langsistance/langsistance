"""design_search XHR 检索: 阶梯 / US 过滤 / 退避 (design P1 T3)。

规格: docs/superpowers/specs/2026-09-06-us-design-clearance-design.md §5.3; task-3-brief.md。
- build_ladder: 紧→松生成 ≤CAP 条查询词 (en_name+kw1 → en_name → 各关键词单独放宽)。
- parse_design_hits: 从 XHR JSON 抽 US 件 (id patent/USD 开头), EM/CN 记日志项不返回。
  US 件带 status: 有 grant/publication 日期 → "active" (由调用方/risk 以 today 判届满);
  缺日期 → "unknown", 设计上不参与 active 判定 (Task 7 按 active-only 留省)。
- search_designs: 阶梯逐阶 fetch (阶间 1s 节奏), 命中 US 即停; 503/429 逐阶内
  指数退避重试 ≤3, 仍耗尽再冷却 _COOLDOWN 终试一次 (突发限流数秒回落, 避免
  误判服务不可用); 终试仍 429/503 → 该阶 rate_limited=True 上抛终止。
fetch 由调用方注入 (真实 httpx 包装在 Task 7); 本模块只拼 query 串并 await fetch。
仅标准库, 纯函数无 IO (退避用 asyncio.sleep, jitter 由调用侧放缩)。零产品词固化。
"""
import asyncio
import json
from urllib.parse import quote

from sources.design.design_risk import DesignCandidate

_CAP = 6           # 阶梯最大阶数 (spec §5.3 ≤6)
_RETRYABLE = {429, 503}
_DELAYS = (2.0, 4.0, 8.0)   # 指数退避基数 (s), 乘 jitter 后实际睡
_MAX_RETRIES = 3            # ≤3 次重试 = 单阶至多 4 次请求
_COOLDOWN = 12.0            # 单阶耗尽后的冷却终试 (突发限流常于数秒内回落)
_STEP_PACE = 1.0            # 阶间节奏 (s), 避免阶梯突发打满 Google XHR 限流


def build_ladder(en_name: str, keywords: list[str]) -> list[str]:
    """紧→松 ≤_CAP 阶候选词: en_name+kw1 → en_name → 各关键词单独放宽。

    返回裸词 (加引号与 type=DESIGN 由 search_designs 拼接), 精确去重保留先序。
    """
    name = (en_name or "").strip()
    first = keywords[0] if keywords else ""
    terms = []
    if first and name:
        terms.append(f"{name} {first}")
    if name:
        terms.append(name)
    seen = set()
    deduped = []
    for t in terms + list(keywords):
        if t not in seen:
            seen.add(t)
            deduped.append(t)
            if len(deduped) >= _CAP:
                break
    return deduped or [name]


def _country_code(pub: str) -> str:
    """出版物号前导字母段即地区码 (USD→过滤前, EM/CN/… 作 em_cn 日志)。"""
    head = ""
    for ch in pub:
        if ch.isalpha():
            head += ch
        else:
            break
    return head


def _date_part(raw: str) -> str:
    """取 publication_date 前 10 位为 YYYY-MM-DD, 缺省 ""。"""
    return raw[:10] if raw else ""


def parse_design_hits(xhr_json: dict) -> tuple[list[DesignCandidate], list[dict]]:
    """XHR JSON → (US 全量设计件, em_cn 日志项)。

    US 判定: 顶层 item id 以 'patent/USD' 开头。非 US 件记 {"id","country"} 供日志。
    解析字段取专利 item: patent.publication_number / title / publication_date / assignee。
    返回 (cands, em_cn); 无 results → 空。
    """
    cands: list[DesignCandidate] = []
    em_cn: list[dict] = []
    clusters = xhr_json.get("results", {}).get("cluster", []) if isinstance(
        xhr_json, dict) else []
    for cluster in clusters:
        for item in cluster.get("result", []):
            raw_id = item.get("id", "")
            pat = item.get("patent", {}) or {}
            pub = str(pat.get("publication_number", "") or "")
            if not raw_id.startswith("patent/USD"):
                em_cn.append({"id": pub, "country": _country_code(pub)})
                continue
            date = _date_part(str(pat.get("publication_date", "") or ""))
            status = "unknown" if not date else "active"
            cands.append(DesignCandidate(
                pub=pub,
                title=str(pat.get("title", "") or ""),
                grant_date=date,
                assignee=str(pat.get("assignee", "") or ""),
                status=status,
            ))
    return cands, em_cn


def _build_query(term: str, country_param: str) -> str:
    """拼 XHR url 参数原文: q="{term}"&type=DESIGN (+country 由 flag 控制)。

    NB: 表达式内的引号转义拆出行外计算——f-string 表达式部分在 Python
    <3.12 不允许反斜杠(服务器 celery 跑 3.11, 曾致 import 即 SyntaxError)。
    """
    quoted_term = quote('"' + term + '"')
    q = f"q={quoted_term}&type=DESIGN"
    if country_param:
        q += f"&country={quote(country_param)}"
    return q


def _to_hits(text: str) -> tuple[list[DesignCandidate], list[dict], bool]:
    """解析响应体 → (cands, em_cn, ok)。非可解析 JSON 视空命中。"""
    try:
        data = json.loads(text) if text else {}
    except (json.JSONDecodeError, TypeError):
        return [], [], False
    if not isinstance(data, dict):
        return [], [], False
    return (*parse_design_hits(data), True)


async def _fetch_once(fetch, query: str):
    """单次 fetch; 非 200 但非可重试码也返回, 由调用方按 (未命中 / 可重试) 分类。"""
    return await fetch(query)


async def _run_query(fetch, query: str, jitter: float):
    """单阶请求循环: 拿到 200 即停 (匹配解析); 否则退避重试 ≤_MAX_RETRIES,
    仍耗尽再冷却 _COOLDOWN 后终试一次 —— 突发限流 (Google XHR 对单 IP 短窗
    突发常 429) 在数秒内回落, 冷却终试可消除误判"服务不可用"。

    返回 (cands, em_cn, ok, rate_limited)。
    - ok: 拿到 200 并成功解析 (即使 0 命中)。
    - rate_limited: 退避+冷却终试后仍以 429/503 收尾, 置整次限流标志。
    - 非 200 且非 429/503 (如 500/404) 不强推, 归为"该阶请求未获结构化数据", 不判定限流。
    """
    retryable = False
    for attempt in range(_MAX_RETRIES + 1):
        status, text = await _fetch_once(fetch, query)
        if status == 200:
            cands, em_cn, ok = _to_hits(text)
            return cands, em_cn, ok, False
        if status in _RETRYABLE:
            retryable = True
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_DELAYS[attempt] * jitter)
    if retryable:
        await asyncio.sleep(_COOLDOWN * jitter)
        status, text = await _fetch_once(fetch, query)
        if status == 200:
            cands, em_cn, ok = _to_hits(text)
            return cands, em_cn, ok, False
        if status not in _RETRYABLE:
            return [], [], False, False   # 冷却后非可重试码 → 该阶无结构化数据
    return [], [], False, retryable


async def search_designs(
    fetch,
    en_name: str,
    keywords: list[str],
    country_param: str = "US",
    jitter: float = 1.0,
) -> dict:
    """组合检索 → dict。

    阶梯紧→松；某阶返回 ≥1 件 US 候选即终止 (used_queries = 已推进阶数)。
    单阶 429/503 退避重试 ≤2 仍败 → rate_limited=True、忽略后续阶直接上抛。
    candidates/em_cn 聚合推进过程中所见件；0 命中走完阶梯 → 空候选。
    """
    ladder = build_ladder(en_name, keywords)
    seen_ids: set[str] = set()
    cands_all: list[DesignCandidate] = []
    em_cn_all: list[dict] = []
    rate_limited = False
    used = 0
    for idx, term in enumerate(ladder):
        used += 1
        if idx > 0 and _STEP_PACE > 0:
            await asyncio.sleep(_STEP_PACE * jitter)   # 阶间节奏防突发限流
        query = _build_query(term, country_param)
        cands, em_cn, ok, limited = await _run_query(fetch, query, jitter)
        for hit in em_cn:
            if hit["id"] not in seen_ids:
                seen_ids.add(hit["id"])
                em_cn_all.append(hit)
        if limited:
            rate_limited = True
            break
        if not ok:
            continue                    # 非 200/非可重试 → 该阶无结构化数据, 继续放宽
        cands_all.extend(cands)
        if cands:
            break                       # 命中 US → 停
    return {
        "candidates": cands_all,
        "em_cn": em_cn_all,
        "used_queries": used,
        "rate_limited": rate_limited,
    }
