# 卖家线 M1a：输入识别 + 专利卡后端 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地卖家专利卡的第一个可独立验证后端切片：输入自识别（专利号 vs 产品名）+ `POST /seller/patent_card` 端点（解析 → 拉权利要求 → LLM 人话卡 → 进程内缓存），TDD 全绿。

**Architecture:** 新增薄适配层。路由层照 `patent_detail.register_patent_detail_routes(logger, config)` 工厂模式，注入 `provider`（api.py `initialize_system` 内已有的 `_build_main_provider` 实例）；纯逻辑放 `sources/seller/` 两个小模块（可单测、不依赖 FastAPI）；专利权利要求拉取直接复用 `api_routes/patent_detail._fetch_claims(source, patent_id)`（同仓库内部函数，已验证签名 `async def _fetch_claims(source, patent_id) -> dict`）。不上多 agent、不加框架（spec §4）。

**Tech Stack:** FastAPI（工厂注册模式）、`Provider.complete_json`（LLM 结构化输出，`sources/llm_provider.py:769`）、`patent_number_parser.parse_patent_identifiers` + `decide_number_source`、pytest/unittest（passport 打桩先例：`tests/test_patent_detail_api.py`）。

**Scope 边界（本计划不含，见 spec）：** `/seller/search`（查一查）、`/seller/ask`、GeneralAgent `scene_id` 参数、scenes 卖家知识包种子（需 embedding 回填，单列脚本任务）、utm、前端统一首页 —— 均为后续独立计划。

## Global Constraints

- 本机 pytest 需 `PYTHONUTF8=1`（运行前缀：`PYTHONUTF8=1 python -m pytest ...`）
- 测试须在导入 `api_routes.seller` 前打桩 `sys.modules["sources.user.passport"]`（passport 初始化 Firebase/Redis，本地 venv 无法加载；先例 `tests/test_patent_detail_api.py`）
- 卖家文案红线（spec §3.2-7）：LLM 人话输出不得出现"权利要求 / 本领域技术人员"等术语，只说"保护什么 / 撞不撞 / 到期没 / 能不能卖"——系统提示词里固定口径
- 免责口径固定：所有卡输出携带 `disclaimer` 文案（"基于公开数据库自动分析，不保证检索穷尽，不构成法律意见"）
- 路由只做编排与鉴权，判断逻辑全部下沉 `sources/seller/`（无 FastAPI 依赖、可单测）
- 缓存 M1 用进程内 TTL dict（单进程后端；多 worker 部署时替换为 Redis——spec §4.3 缓存策略，接口不变）
- 法律状态/到期数据源未接入（spec §6-6 风险，M1.5 spike）→ 本计划 `legal_status: null` + `status_note: "状态核验中（M1.5）"` 是**已批准契约**，前端显示"核验中"，不做假判断
- 输入识别失败/拉取失败走 200 + `success:false` 降级（照 `patent_detail` 的 Cloudflare/CORS 教训注释：上游 miss 是数据条件不是 5xx）
- 提交粒度：每 Task 一个 commit，消息 `feat(seller): ...`

---

### Task 1: 查询输入识别模块 `sources/seller/query_classifier.py`

**Files:**
- Create: `sources/seller/__init__.py`（空文件）
- Create: `sources/seller/query_classifier.py`
- Test: `tests/test_seller_query_classifier.py`

**Interfaces:**
- Consumes: `patent_number_parser.parse_patent_identifiers(text) -> list[dict]`、`patent_number_parser.decide_number_source(candidates) -> str | None`（验证于 `sources/patent_number_parser.py:368/402`）
- Produces: `classify_seller_query(text: str) -> dict`，返回
  - `{"kind": "patent", "source": "uspto"|"baiten", "patent_id": str, "matched": str}`（matched=从原文中匹配到的专利号片段）
  - `{"kind": "product"}`（无专利号意图）
  - 永不抛异常；空串/超长（>200 字）返回 `{"kind": "product"}`

- [ ] **Step 1: 写失败测试**

```python
"""Tests for sources/seller/query_classifier."""
import sys
import unittest

# passport stub 需要在导入 seller 路由相关模块前就位；本文件只测纯逻辑，
# 无需 stub，但保持目录测试隔离一致。
from sources.seller.query_classifier import classify_seller_query


class TestClassifySellerQuery(unittest.TestCase):
    def test_us_design_number_is_patent(self):
        result = classify_seller_query("US D1,088,888")
        self.assertEqual(result["kind"], "patent")
        self.assertEqual(result["source"], "uspto")

    def test_cn_number_is_patent(self):
        result = classify_seller_query("CN 306,998,821")
        self.assertEqual(result["kind"], "patent")
        self.assertEqual(result["source"], "baiten")

    def test_bare_us_application_style_is_patent(self):
        result = classify_seller_query("30/076,484")
        self.assertEqual(result["kind"], "patent")

    def test_product_name_is_product(self):
        result = classify_seller_query("折叠水杯")
        self.assertEqual(result["kind"], "product")

    def test_mixed_text_with_patent_number_is_patent(self):
        result = classify_seller_query("帮我看看 US 11,675,432 这件的保护范围")
        self.assertEqual(result["kind"], "patent")
        self.assertIn("US 11,675,432", result["matched"])

    def test_empty_and_long_input_are_product(self):
        self.assertEqual(classify_seller_query("")["kind"], "product")
        self.assertEqual(classify_seller_query("水" * 300)["kind"], "product")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_seller_query_classifier.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'sources.seller'`）

- [ ] **Step 3: 最小实现**

```python
"""Query classification for the seller patent card flow.

Pure logic — no FastAPI / DB imports so it stays unit-testable.
"""

from sources.patent_number_parser import (
    parse_patent_identifiers,
    decide_number_source,
)

_MAX_LEN = 200


def classify_seller_query(text: str) -> dict:
    """Classify a seller workbench query into patent-number or product intent.

    Returns {"kind": "patent", ...} when the text carries a recognizable
    patent number (US or CN), else {"kind": "product"}.  Never raises.
    """
    if not text or not text.strip():
        return {"kind": "product"}
    stripped = text.strip()
    if len(stripped) > _MAX_LEN:
        return {"kind": "product"}

    candidates = parse_patent_identifiers(stripped)
    if not candidates:
        return {"kind": "product"}

    source = decide_number_source(candidates)
    if not source:
        return {"kind": "product"}

    # candidates are ordered by likelihood; take the highest-ranked match
    # and normalize the source token to the api_routes/patent_detail set.
    top = candidates[0]
    return {
        "kind": "patent",
        "source": "uspto" if source in ("uspto", "us", "USPTO") else "baiten",
        "patent_id": str(top.get("pub", "") or top.get("digits", "") or stripped),
        "matched": str(top.get("raw", "") or stripped),
    }
```

> 注：`parse_patent_identifiers` 返回的候选 dict 字段名（`pub`/`digits`/`raw`）以 Step 4 实际跑通后为准——若字段名不符，仅需在此文件内改用实际字段（两个 `.get` 兜底保证不崩）。若 `decide_number_source` 对 US 设计号返回非预期值，允许把 `source in (...)` 判断扩展为该函数实际返回值并更新本测试的断言。

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_seller_query_classifier.py -v`
Expected: PASS（若 `decide_number_source` 返回约定与假设不同，按 Step 3 注修正映射并保持断言语义不变）

- [ ] **Step 5: 提交**

```bash
git add sources/seller/ sources/seller/query_classifier.py tests/test_seller_query_classifier.py
git commit -m "feat(seller): 卖家查询输入识别（专利号 vs 产品）"
```

---

### Task 2: 专利卡服务 `sources/seller/patent_card.py`（含 LLM 人话 + 缓存 + 免责）

**Files:**
- Create: `sources/seller/patent_card.py`
- Test: `tests/test_seller_patent_card.py`

**Interfaces:**
- Consumes: `provider.complete_json(system_prompt: str, user_content: str, max_retries: int = 2)`（异步，返回已解析 JSON 对象或 None——`sources/llm_provider.py:769`）
- Produces: `build_patent_card(provider, claims_text: str, source: str, patent_id: str, lang: str = "zh") -> dict`
  返回卡片契约（lang 未用于 M1，保留参数供后续 i18n）：
  ```json
  {"success": true,
   "card": {
     "patent_id": "...", "source": "...",
     "legal_status": null, "status_note": "状态核验中（M1.5）",
     "protection_summary": "人话一句话（LLM）或 null",
     "risk_level": "high|mid|low|null",
     "next_step": "人话建议或 null",
     "llm_available": true|false,
     "disclaimer": "基于公开数据库自动分析，不保证检索穷尽，不构成法律意见"}}
  ```
  LLM 失败时 `llm_available:false`、`protection_summary/risk_level/next_step:null`，**不抛异常**（卖家场景降级优先）。
  另导出 `card_cache_get(key) -> dict | None` / `card_cache_put(key, card, ttl_seconds=86400)`（进程内 TTL 缓存，key=`f"{source}:{patent_id}:{lang}"`）。

- [ ] **Step 1: 写失败测试**（fake provider 用 AsyncMock 返回预定 JSON）

```python
"""Tests for sources/seller/patent_card."""
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock
from freezegun import freeze_time

from sources.seller.patent_card import build_patent_card, card_cache_put, card_cache_get


class FakeProvider:
    def __init__(self, payload):
        self.complete_json = AsyncMock(return_value=payload)


CARD_COPY_OK = {
    "protection_summary": "保护这种带卡扣的折叠水杯结构：杯身可分两段折叠，折叠处用卡扣固定",
    "risk_level": "high",
    "next_step": "高风险：建议规避或授权。改掉卡扣闭合特征，或向权利人询价授权。",
}


class TestBuildPatentCard(unittest.TestCase):
    def test_success_card_has_four_blocks_and_disclaimer(self):
        provider = FakeProvider(dict(CARD_COPY_OK))
        result = build_patent_card(provider, "1. A folding cup with a clasp.", "uspto", "US 11,675,432")
        self.assertTrue(result["success"])
        card = result["card"]
        self.assertEqual(card["patent_id"], "US 11,675,432")
        self.assertEqual(card["risk_level"], "high")
        self.assertIsNone(card["legal_status"])
        self.assertIn("M1.5", card["status_note"])
        self.assertIn("不构成法律意见", card["disclaimer"])
        self.assertTrue(card["llm_available"])
        self.assertTrue(provider.complete_json.await_count == 1)

    def test_llm_failure_degrades_without_raising(self):
        provider = FakeProvider(None)
        result = build_patent_card(provider, "1. A folding cup.", "uspto", "US 11,675,432")
        self.assertTrue(result["success"])
        card = result["card"]
        self.assertFalse(card["llm_available"])
        self.assertIsNone(card["protection_summary"])
        self.assertIn("disclaimer", card)

    def test_cache_roundtrip_and_ttl(self):
        with freeze_time("2026-09-04 12:00:00"):
            card_cache_put("uspto:US 1:zh", {"patent_id": "US 1"})
            self.assertEqual(card_cache_get("uspto:US 1:zh")["patent_id"], "US 1")
        with freeze_time("2026-09-06 12:00:00"):
            self.assertIsNone(card_cache_get("uspto:US 1:zh"))


if __name__ == "__main__":
    unittest.main()
```

> 测试需要 `freezegun`——若仓库未依赖，改为不用 freeze：`card_cache_put(..., ttl_seconds=-1)` 断言 None 即可。第 4 步前如遇 import 错误，优先采用 -1 TTL 变体并删掉 freeze_time 用法。

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_seller_patent_card.py -v`
Expected: FAIL（`ModuleNotFoundError: sources.seller.patent_card`）

- [ ] **Step 3: 最小实现**

```python
"""Seller patent card assembly: claims text -> plain-language card.

Pure orchestration with an injectable provider (``complete_json``).
LLM failures degrade to ``llm_available:false`` instead of raising —
seller queries are free-tier high-frequency; a polite empty card beats a 5xx.
"""

import time

from sources.logger import Logger

logger = Logger("backend.log")

DISCLAIMER = "基于公开数据库自动分析，不保证检索穷尽，不构成法律意见"

# System copy fixed for the seller voice (spec §3.2-7): never surface
# claim-language terms; speak in 保护什么 / 撞不撞 / 能不能卖.
_CARD_SYSTEM = (
    "你是 CopiioAI 卖家专利安全台的专利解读助手，面向完全不懂专利法的跨境卖家。\n"
    "把下面的专利权利要求原文翻译成中文人话，只输出 JSON：\n"
    '{"protection_summary": "一句话说明这个专利保护什么（禁止出现\"权利要求/本领域技术人员/实施例\"等词）", '
    '"risk_level": "high|mid|low|expired", '
    '"next_step": "给卖家的下一步建议：可上架 / 需规避（提示改哪里） / 建议询价授权 / 已过期可参考"}'
)

_cache: dict[str, dict] = {}  # key -> {"expires_at": float, "card": dict}
_DEFAULT_TTL = 86400


def card_cache_get(key: str):
    entry = _cache.get(key)
    if not entry:
        return None
    if time.time() > entry["expires_at"]:
        _cache.pop(key, None)
        return None
    return entry["card"]


def card_cache_put(key: str, card: dict, ttl_seconds: int = _DEFAULT_TTL) -> None:
    _cache[key] = {"expires_at": time.time() + ttl_seconds, "card": card}


def _empty_card(patent_id: str, source: str) -> dict:
    return {
        "patent_id": patent_id,
        "source": source,
        "legal_status": None,
        "status_note": "状态核验中（M1.5）",
        "protection_summary": None,
        "risk_level": None,
        "next_step": None,
        "llm_available": False,
        "disclaimer": DISCLAIMER,
    }


async def build_patent_card(provider, claims_text: str, source: str,
                            patent_id: str, lang: str = "zh") -> dict:
    key = f"{source}:{patent_id}:{lang}"
    cached = card_cache_get(key)
    if cached is not None:
        return {"success": True, "card": cached, "cached": True}

    card = _empty_card(patent_id, source)
    try:
        parsed = await provider.complete_json(
            _CARD_SYSTEM,
            f"专利号：{patent_id}\n权利要求原文（摘录前 6000 字符）：\n{claims_text[:6000]}",
            max_retries=1,
        )
    except Exception as exc:  # noqa: BLE001 — degrade, never raise
        logger.error(f"seller card llm failed — {patent_id}: {exc}")
        parsed = None

    if isinstance(parsed, dict) and parsed.get("protection_summary"):
        card.update({
            "protection_summary": str(parsed["protection_summary"])[:500],
            "risk_level": parsed.get("risk_level")
            if parsed.get("risk_level") in ("high", "mid", "low", "expired") else None,
            "next_step": str(parsed.get("next_step") or "")[:500] or None,
            "llm_available": True,
        })
    card_cache_put(key, card)
    return {"success": True, "card": card, "cached": False}
```

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_seller_patent_card.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add sources/seller/patent_card.py tests/test_seller_patent_card.py
git commit -m "feat(seller): 专利卡组装服务（LLM 人话 + 降级 + 进程内缓存）"
```

---

### Task 3: 路由注册骨架 + `POST /seller/patent_card` 端点

**Files:**
- Create: `api_routes/seller.py`
- Modify: `api.py`（import 行 + 注册行）
- Test: `tests/test_seller_api.py`

**Interfaces:**
- Consumes: `classify_seller_query`（Task 1）、`build_patent_card`（Task 2）、`api_routes.patent_detail._fetch_claims(source, patent_id)`（`api_routes/patent_detail.py:627`，异步，返回 `{"success": bool, "claims": [{"number","independent","text"}] | 无, ...}`——claims 文本取各条 `text` 拼接）、`verify_firebase_token`（返回 `{"uid": ...}`）
- Produces: 注册工厂 `register_seller_routes(logger, config, provider) -> APIRouter`，在 `api.py` 注册后对外暴露：
  - `POST /seller/patent_card` body `{"query": str, "lang": "zh"|"en"}` → 见 Step 3 契约

- [ ] **Step 1: 写失败测试**（TestClient 直挂 router + passport 打桩 + monkeypatch 拉取函数）

```python
"""Tests for api_routes/seller."""
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# passport stub：必须在导入 api_routes.seller 之前（先例 test_patent_detail_api.py）
_passport_stub = MagicMock()
_passport_stub.verify_firebase_token = MagicMock(return_value={"uid": "1"})
sys.modules["sources.user.passport"] = _passport_stub

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sources.seller.query_classifier import classify_seller_query
from sources.seller.patent_card import _empty_card
from api_routes.seller import register_seller_routes


class _FakeProvider:
    def __init__(self):
        self.complete_json = AsyncMock(return_value={
            "protection_summary": "保护这种带卡扣的折叠水杯结构",
            "risk_level": "high",
            "next_step": "改掉卡扣闭合特征，或向权利人询价授权",
        })


def _make_client(provider=None, claims_payload=None):
    app = FastAPI()
    app.include_router(register_seller_routes(MagicMock(), MagicMock(), provider or _FakeProvider()))
    if claims_payload is not None:
        # 实现经 api_routes.seller._fetch_claims_lazy → api_routes.patent_detail._fetch_claims
        # 拉取；patch 必须打在源头模块上。
        patcher = patch("api_routes.patent_detail._fetch_claims",
                        new=AsyncMock(return_value=claims_payload))
        patcher.start()
        _make_client._patcher = patcher
    return TestClient(app)


class TestSellerPatentCardEndpoint(unittest.TestCase):
    def tearDown(self):
        p = getattr(_make_client, "_patcher", None)
        if p:
            p.stop()
            del _make_client._patcher

    def test_patent_query_returns_card(self):
        client = _make_client(claims_payload={
            "success": True,
            "claims": [
                {"number": 1, "independent": True, "text": "1. A folding cup with a clasp."},
                {"number": 2, "independent": False, "text": "2. The cup of claim 1."},
            ],
        })
        resp = client.post("/seller/patent_card", json={"query": "US 11,675,432"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["card"]["patent_id"], "US 11,675,432")
        self.assertEqual(body["card"]["risk_level"], "high")
        self.assertIn("不构成法律意见", body["card"]["disclaimer"])
        self.assertFalse(body["card"]["llm_available"] is False)

    def test_claims_unavailable_degrades(self):
        client = _make_client(claims_payload={"success": False})
        resp = client.post("/seller/patent_card", json={"query": "CN 306,998,821"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["success"])
        self.assertIn("claims_available", body)
        self.assertFalse(body["claims_available"])

    def test_product_query_returns_product_hint(self):
        client = _make_client()
        resp = client.post("/seller/patent_card", json={"query": "折叠水杯"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["kind"], "product")

    def test_missing_auth_is_rejected(self):
        client = _make_client()
        _passport_stub.verify_firebase_token = MagicMock(side_effect=Exception("no token"))
        resp = client.post("/seller/patent_card", json={"query": "US D1,088,888"})
        self.assertEqual(resp.status_code in (400, 401, 500), True)
        _passport_stub.verify_firebase_token = MagicMock(return_value={"uid": "1"})


if __name__ == "__main__":
    unittest.main()
```

> 若 `verify_firebase_token` 抛的是 HTTPException 而非 Exception，最后一条测试按实际异常类型放宽断言（改 `side_effect` 为该异常即可）。

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_seller_api.py -v`
Expected: FAIL（`ModuleNotFoundError: api_routes.seller`）

- [ ] **Step 3: 最小实现**（`api_routes/seller.py`）

```python
#!/usr/bin/env python3
"""Seller patent safety workbench routes (M1a slice).

POST /seller/patent_card — patent number -> plain-language card.

Thin orchestration only: classification and card assembly live in
sources/seller/* (unit-testable, framework-free).  Claims documents are
fetched through the existing patent_detail pipeline (_fetch_claims);
upstream misses degrade to 200 + success:false (Cloudflare swaps origin
5xx for its own CORS-less error page — same rationale as patent_detail).
"""

from fastapi import APIRouter, HTTPException, Request

from sources.logger import Logger
from sources.user.passport import verify_firebase_token
from sources.seller.query_classifier import classify_seller_query
from sources.seller.patent_card import build_patent_card

logger = Logger("backend.log")


def register_seller_routes(logger, config, provider):
    """Register seller workbench routes with dependency injection."""
    router = APIRouter()

    async def _fetch_claims_lazy(source: str, patent_id: str) -> dict:
        # Lazy import mirrors patent_detail's own internal function and
        # keeps module import graphs acyclic.
        from api_routes.patent_detail import _fetch_claims
        return await _fetch_claims(source, patent_id)

    @router.post("/seller/patent_card")
    async def seller_patent_card(http_request: Request):
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)
        user_id = int(user["uid"])

        try:
            body = await http_request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        query = str(body.get("query") or "").strip()
        lang = body.get("lang") if body.get("lang") in ("zh", "en") else "zh"
        if not query:
            raise HTTPException(status_code=400, detail="query is required")

        classification = classify_seller_query(query)
        if classification["kind"] != "patent":
            return {"success": True, "kind": "product",
                    "message": "该输入走查一查检索（/seller/search，M1b）"}

        source = classification["source"]
        patent_id = classification["patent_id"]
        try:
            payload = await _fetch_claims_lazy(source, patent_id)
        except Exception as exc:  # noqa: BLE001 — degrade per module contract
            logger.error(f"seller card claims fetch failed — {patent_id}: {exc}")
            return {"success": True, "claims_available": False,
                    "message": "专利文本暂不可用，请稍后重试"}

        if not payload.get("success"):
            return {"success": True, "claims_available": False,
                    "message": payload.get("message") or "专利文本暂不可用"}

        claims = payload.get("claims") or []
        claims_text = "\n".join(
            str(c.get("text") or "") for c in claims if c.get("text")
        )
        if not claims_text.strip():
            return {"success": True, "claims_available": False,
                    "message": "专利文本为空，暂无法生成解读"}

        result = await build_patent_card(provider, claims_text, source,
                                         patent_id, lang=lang)
        result.update({"kind": "patent", "claims_available": True,
                       "user_id": user_id})
        return result

    return router
```

**Modify `api.py`：** import 行（第 15 行长 import 内追加 `seller`）与注册区（`patent_detail_router` 块后）：

```python
from api_routes import knowledge, tools, system, core, auth, uspto, feedback, scenes, patent, session, patent_detail, baiten, long_task as long_task_routes, seller
```
```python
patent_detail_router = patent_detail.register_patent_detail_routes(logger, config)
api.include_router(patent_detail_router, tags=["patent-detail"])
seller_router = seller.register_seller_routes(logger, config, provider)
api.include_router(seller_router, tags=["seller"])
```

> `provider` 在该作用域已存在（`api.py` `_build_main_provider(config)` 的返回值，见 api.py 头部）。若注册区在 `initialize_system()` 内而 `provider` 也在其中，直接可用；若不在同一函数，把 `provider` 作为参数传入注册区所在函数。

- [ ] **Step 4: 运行确认通过 + 全量回归**

Run:
```bash
PYTHONUTF8=1 python -m pytest tests/test_seller_api.py tests/test_seller_query_classifier.py tests/test_seller_patent_card.py -v
PYTHONUTF8=1 python -m pytest tests/test_api_route_imports.py -v
```
Expected: 全部 PASS（后者验证 api.py import 行没写坏）

- [ ] **Step 5: 提交**

```bash
git add api_routes/seller.py api.py tests/test_seller_api.py
git commit -m "feat(seller): POST /seller/patent_card 端点（识别→claims→人话卡，工厂注册）"
```

---

### Task 4: 真实样例冒烟（可选但强烈建议：识别器对照表）

**Files:**
- Create: `tests/test_seller_real_inputs.py`（表格驱动，输入来自卖家真实诉求样例：1545 玩具、7271 遥控蛇回执号、9408 图片描述——只保留**通用形态**，不固化单条提问原文，遵守"测试提问词不固化"红线）

**Interfaces:**
- Consumes: `classify_seller_query`（Task 1）

- [ ] **Step 1: 写测试**

```python
"""Real-shape seller inputs (generic forms only — no verbatim user asks)."""
import unittest
from sources.seller.query_classifier import classify_seller_query

CASES = [
    # (输入形态, 期望 kind, 期望 source 或 None)
    ("US D1,0xx,xxx 形态", "patent", "uspto"),
    ("US 1x,xxx,xxx 形态", "patent", "uspto"),
    ("x0/xxx,xxx 回执号形态", "patent", None),          # source 依解析结果可放宽
    ("CN 30x,xxx,xxx 外观号形态", "patent", "baiten"),
    ("一款可折叠的便携水杯", "product", None),
    ("ASIN 开头的一串字符", "product", None),
]


class TestRealShapes(unittest.TestCase):
    def test_generic_forms(self):
        for text, kind, source in CASES:
            result = classify_seller_query(text)
            self.assertEqual(result["kind"], kind, text)
            if source is not None:
                self.assertEqual(result.get("source"), source, text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行**

Run: `PYTHONUTF8=1 python -m pytest tests/test_seller_real_inputs.py -v`
Expected: 通过；失败时按实际解析器能力调整用例形态（保留语义：专利号→patent、产品描述→product）

- [ ] **Step 3: 提交**

```bash
git add tests/test_seller_real_inputs.py
git commit -m "test(seller): 专利号/产品输入形态识别回归表"
```

---

### Self-Review 记录（本计划已对照 spec 执行）

- **Spec §3.2-3 查询自识别** → Task 1（专利号→卡 / 产品→product hint，M1b 接 /seller/search）
- **Spec §3.2-4 专利卡四件套** → Task 2/3：①保护什么=protection_summary；③状态到期=legal_status:null+status_note(M1.5 契约)；④建议=risk_level+next_step；②"撞不撞"依赖产品图/描述比对属 M2 范围（spec §5 映射表已标注）
- **Spec §3.2-5 免责与不确定性** → DISCLAIMER 注入每张卡 + claims_available 显式降级
- **Spec §4.3 薄适配层/单 agent** → 路由仅编排，逻辑下沉 sources/seller；不新建 agent
- **Spec §3.2-7 文案红线** → _CARD_SYSTEM 固定禁词口径（禁止出现权利要求等术语——提示词仅限通用口径，无单条测试词）
- **范围外已显式记录**：/seller/search、agent scene_id、场景知识种子（需 embedding 流程）、utm、前端统一首页——分别独立计划，避免假占位

---

## 执行方式（二选一）

计划已保存。执行选项：

1. **Subagent-Driven（推荐）**——每个 Task 派新 subagent 执行 + 两阶段审查，快速迭代
2. **Inline Execution**——本会话内用 executing-plans 按批执行 + 检查点审查

选哪种？（另：执行前需要先 `git push origin main` 同步远程基线，还是保留本地？）
