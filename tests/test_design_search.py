"""design_search XHR 检索 (design P1 T3) — 契约测试。

- fet 全程 mock (异步注入), 无真实网络。
- 样例词仅用通用品名词 (robot toy 等数据夹具), 生产代码零产品词。
- 覆盖: build_ladder / parse_design_hits / search_designs 退避与 rate_limited。
"""
import asyncio
import json

import pytest

from sources.design.design_risk import DesignCandidate
from sources.design.design_search import (
    build_ladder,
    parse_design_hits,
    search_designs,
)

# brief 复刻: mock fetch 协程便捷包装。
def _run(fetch, en_name, keywords, country_param=""):
    return asyncio.run(
        search_designs(fetch, en_name, keywords, country_param=country_param))


def _hit(pid, pub, date, title, assignee=""):
    """构造单条 XHR result item (fixture 字段镜像 brief 样例)。"""
    return {
        "id": f"patent/{pid}/en",
        "patent": {
            "publication_number": pub,
            "title": title,
            "publication_date": date,
            "assignee": assignee,
        },
    }


# ---------- build_ladder ----------

def test_ladder_order_and_cap():
    lad = build_ladder("Robot toy", ["robot dog", "mechanical dog"])
    assert len(lad) <= 6
    assert lad[0] == "Robot toy robot dog"          # brief 钉死首阶 = en_name+kw1
    assert lad[1] == "Robot toy"                    # 次阶 = en_name 单飞
    assert "mechanical dog" in lad                   # 后续单特征词池逐个放宽


def test_ladder_dedup_and_empty_keywords():
    # keywords 空 → 至少 en_name 一阶可查 (search rate_limited 需 used_queries>=1)。
    assert build_ladder("Robot toy", []) == ["Robot toy"]
    # 关键词池内重复词精确去重: 组合+ename 各一次, 池重复不再展开。
    assert build_ladder("Pet", ["dog", "dog", "cat"]) == ["Pet dog", "Pet", "dog", "cat"]


def test_ladder_never_exceeds_cap():
    many = [f"variant {i}" for i in range(30)]
    lad = build_ladder("Robot toy", many)
    assert len(lad) <= 6 and lad[0] == "Robot toy variant 0"


# ---------- parse_design_hits ----------

def test_parse_us_only_and_status():
    xhr = {"results": {"cluster": [{"result": [
        _hit("USD1A", "USD1A", "2023-01-01", "Toy"),
        _hit("EM150000001S", "EM150000001S", "2023-02-01", "Toy"),
        _hit("CN309123456S", "CN309123456S", "2023-03-01", "Toy"),
    ]}]}}
    cands, em_cn = parse_design_hits(xhr)
    assert [c.pub for c in cands] == ["USD1A"]
    assert len(em_cn) == 2
    # 每件含 id+country 的日志项 (brief: 记录 {id,country} 不返回进候选)。
    assert {e["country"] for e in em_cn} == {"EM", "CN"}


def test_parse_none_or_empty_json_via_search_ok():
    # 解析遇无 results 键 → 空候选 (search 0命中才走下一阶梯)。
    cands, em_cn = parse_design_hits({})
    assert cands == [] and em_cn == []


def test_parse_keeps_date_and_missing_date_unknown():
    dated = {
        "results": {"cluster": [{"result": [
            _hit("USD2B", "USD2BS", "2023-11-30", "Mug", "ACME"),
        ]}]}}
    cands, _ = parse_design_hits(dated)
    assert cands[0].title == "Mug"
    assert cands[0].assignee == "ACME"
    assert cands[0].pub == "USD2BS"          # pub 用 publication_number
    assert cands[0].grant_date == "2023-11-30"
    assert cands[0].status == "active"        # 有日期 → 由今日后由 risk/T7 判; 此处非 unknown

    nodated = {
        "results": {"cluster": [{"result": [
            _hit("USD3C", "USD3CS", "", "No date"),
        ]}]}}
    cands2, _ = parse_design_hits(nodated)
    assert cands2[0].status == "unknown"      # 缺日期 → 不解 expiry, 交由 T7 active-only


def test_parse_date_truncated_to_day():
    xhr = {"results": {"cluster": [{"result": [
        _hit("USD4D", "USD4DS", "2023-01-01T00:00:00Z", "Watch"),
    ]}]}}
    cands, _ = parse_design_hits(xhr)
    assert cands[0].grant_date == "2023-01-01"


# ---------- search_designs ----------

def test_search_success_returns_candidates_and_used_queries():
    xhr = json.dumps({"results": {"cluster": [{"result": [
        _hit("USD5E", "USD5ES", "2022-01-01", "Robot"),
    ]}]}})

    async def fetch(url_query):
        assert "type=DESIGN" in url_query         # 必须带 type=DESIGN
        assert "companion" in url_query            # 首阶 term 关键词被带上 (空/引号转义, 字母原样)
        return 200, xhr

    res = _run(fetch, "Robot toy", ["robot companion"])
    assert res["rate_limited"] is False
    assert res["used_queries"] == 1                # 首阶命中即停
    assert [c.pub for c in res["candidates"]] == ["USD5ES"]


def test_search_rate_limit_sets_flag(monkeypatch):
    async def no_sleep(seconds):
        return None
    monkeypatch.setattr(asyncio, "sleep", no_sleep)   # 快进正退避, 防 1s/3s 真睡

    async def boom(url_query):
        return 503, "sorry"

    res = _run(boom, "Robot toy", [])
    assert res["rate_limited"] is True
    assert res["used_queries"] == 1
    assert res["candidates"] == []


def test_search_503_then_200_recovers_not_rate_limited(monkeypatch):
    async def no_sleep(seconds):
        return None
    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    xhr = json.dumps({"results": {"cluster": [{"result": [
        _hit("USD6F", "USD6FS", "2022-01-01", "Robot"),
    ]}]}})
    calls = {"n": 0}

    async def fetch(url_query):
        calls["n"] += 1
        if calls["n"] == 1:
            return 503, "retry-me"                # 第一次限流
        return 200, xhr                            # 重试回 200

    res = _run(fetch, "Robot toy", [])
    assert res["rate_limited"] is False
    assert res["used_queries"] == 1               # 同一 query 重试不计多次阶梯
    assert [c.pub for c in res["candidates"]] == ["USD6FS"]


def test_search_ladder_loosens_until_hit_then_stops():
    # 首二阶梯 0 命中 → 第三阶梯才返回两件 US; used_queries 反映实际发起的阶梯次数。
    empty = json.dumps({"results": {}})
    xhr = json.dumps({"results": {"cluster": [{"result": [
        _hit("USD7", "USD7S", "2021-01-01", "Carrier"),
        _hit("EM987K", "EM987KS", "2021-02-01", "Carrier"),
    ]}]}})
    seq = {"i": 0}

    async def fetch(url_query):
        seq["i"] += 1
        return 200, (empty if seq["i"] < 3 else xhr)

    res = _run(fetch, "Robot shelf", ["spring shelf"])
    assert res["used_queries"] == 3
    # EM 件被滤出, US 件保留。
    assert [c.pub for c in res["candidates"]] == ["USD7S"]
    assert len(res["em_cn"]) == 1
    assert res["rate_limited"] is False


def test_search_empty_parse_env_returns_empty():
    async def fetch(url_query):
        return 200, json.dumps({"results": {}})

    res = _run(fetch, "Robot toy", [])
    assert res["candidates"] == []
    assert res["em_cn"] == []
    assert res["rate_limited"] is False


def test_cooldown_final_attempt_rescues_burst_throttle(monkeypatch):
    """2026-09-07: 单发 200、管线却限流 → Google XHR 突发 429 数秒回落。
    退避耗尽后冷却终试一次: 终试 200 → 不判 rate_limited, 命中照常解析。"""
    async def no_sleep(seconds):
        return None
    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    xhr = json.dumps({"results": {"cluster": [{"result": [
        _hit("USD9", "USD9S", "2022-03-01", "Robot"),
    ]}]}})
    calls = {"n": 0}

    async def fetch(url_query):
        calls["n"] += 1
        if calls["n"] <= 4:                     # 退避 3 次重试仍 429 (共 4 次请求)
            return 429, "burst"
        return 200, xhr                         # 冷却后终试回 200

    res = _run(fetch, "Robot toy", [])
    assert calls["n"] == 5                      # 4 次退避 + 1 次冷却终试
    assert res["rate_limited"] is False
    assert [c.pub for c in res["candidates"]] == ["USD9S"]


def test_cooldown_still_blocked_then_rate_limited(monkeypatch):
    async def no_sleep(seconds):
        return None
    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    calls = {"n": 0}

    async def fetch(url_query):
        calls["n"] += 1
        return 503, "still down"

    res = _run(fetch, "Robot toy", [])
    assert calls["n"] == 5                      # 4 次退避 + 1 次冷却终试仍 503
    assert res["rate_limited"] is True
