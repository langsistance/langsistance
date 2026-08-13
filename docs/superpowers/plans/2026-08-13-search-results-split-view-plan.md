# Search Results Split-View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将智慧问答批量检索结果做成 Eureka 式专业展示——独立结果页三栏布局（聊天窄侧边栏 + 结果列表 + 详情面板），每行专利提供 详情/说明书/权利要求/审查历史 四个操作，聊天与结果页双向联动。

**Architecture:** 复用现有 artifact SSE 通道新增 `format=json` 的列表载荷（带列角色标注），前端解码渲染；说明书/权利要求走两个按需详情端点（Google Patents 客户端），审查历史复用现有长任务管线（新增直接提交端点）。结果数据随会话消息裁剪持久化。

**Tech Stack:** Python FastAPI + Celery（后端）；Next.js App Router + React（前端 nextjs/）；node:test 跑前端纯函数测试；pytest 跑后端测试。

**Spec:** `docs/superpowers/specs/2026-08-13-search-results-split-view-design.md`（§2 数据流、§3 传输格式、§4 交互、§5 API 契约、§7 持久化）

## Global Constraints

- 分支：`feature/search-results-split-view`（已从 main 拉出）；提交信息用 conventional commits（feat:/fix:/refactor:），不附加 Co-Authored-By
- 后端改动遵循现有模式：路由用 `register_*_routes(logger, config)` 工厂 + 依赖注入；状态/日志走 `sources.logger.Logger`
- SSE 层零改动——json artifact 走现有 `artifact_start/chunk/end` 通道
- 列角色集合封闭：`title, patent_id, application_number, assignee, inventors, filing_date, publication_date, ipc, abstract, document_title, document_date, url, text`——未识别列 role=text 且在详情 tab 以键值表展示，任何数据源不丢字段
- 旧消息/旧会话无 `results` 字段时优雅回退（仅显示现有下载按钮），不报错
- 说明书/权利要求按需拉取，不随 artifact 下发；审查历史按钮触发长任务，复用现有 `/long_task/{id}/status`、`/long_task/{id}/report` 轮询与下载
- 前端纯函数测试：`node --test lib/`；组件无单测设施，用 `npm run build` 验证；后端测试：pytest（本地 venv 缺依赖时在服务器执行）
- 前端 i18n：新文案必须同时加入 `frontend/nextjs/lib/app-i18n/locales/zh.ts` 与 `en.ts` 的 `chat` 段

## File Map

**后端（langsistance）**

| 文件 | 责任 |
|---|---|
| `sources/result_export.py` | 新增 `infer_column_role()`、json artifact 构建；`build_result_artifacts()` 增 `source` 参数 |
| `sources/agents/general_agent.py` | 三个调用点传 `source`（新增 `_infer_result_source()` 辅助函数） |
| `api_routes/patent_detail.py`（新） | `GET /patent/{source}/{patent_id}/spec`、`GET /patent/{source}/{patent_id}/claims` |
| `api_routes/long_task.py` | 新增 `POST /long_task/submit` |
| `api.py` | 注册 patent_detail 路由 |
| `tests/test_result_export_roles.py`（新） | 角色推断 + json artifact 测试 |
| `tests/test_general_agent_result_source.py`（新） | source 推断测试 |
| `tests/test_patent_detail_api.py`（新） | spec/claims 端点测试（mock 客户端） |
| `tests/test_long_task_submit.py`（新） | submit 端点测试 |

**前端（frontend/nextjs）**

| 文件 | 责任 |
|---|---|
| `lib/chatSession.js` | 新增 `decodeResultsArtifact()`——json artifact 解码 → `message.results` |
| `lib/results.js`（新） | 行模型构建（role→卡片字段）、持久化裁剪 `pruneResultsForPersistence()` |
| `services/api.ts` | 新增 `fetchPatentSpec()`、`fetchPatentClaims()`、`submitLongTask()` |
| `contexts/ChatContext.tsx` | 新增 `resultsSetId` 共享状态 |
| `lib/useChatStream.ts`（新） | 从 chat/page.tsx 提取的 send + SSE + 轮询 hook（聊天页与结果页共用） |
| `components/app/ResultCard.tsx`（新） | 聊天气泡内结果卡片 |
| `app/app/(auth)/results/page.tsx`（新） | 结果页三栏布局 |
| `components/app/results/ResultList.tsx`、`ResultRow.tsx`、`DetailPanel.tsx`、`SpecTab.tsx`、`ClaimsTab.tsx`、`ProsecutionTab.tsx`、`DocTab.tsx`（新） | 列表与详情面板组件 |
| `app/app/(auth)/chat/page.tsx` | 渲染 ResultCard + 跳转入口 + 持久化 results 字段 |
| `lib/chatSession.test.mjs`、`lib/results.test.mjs`（新） | 前端纯函数测试 |

---

## Phase 1: 后端

### Task 1: result_export.py — 列角色推断 + JSON artifact

**Files:**
- Modify: `sources/result_export.py`（在 `_export_min_rows` 之后插入新代码；修改 `build_result_artifacts` 签名与返回）
- Test: `tests/test_result_export_roles.py`（新建）

**Interfaces:**
- Produces:
  - `infer_column_role(key: str) -> str` — 输入展平后的列名（如 `applicationMetaData.patentTitle`），输出封闭角色集合之一
  - `build_result_artifacts(items, *, source="uspto", query_id=None, original_count=None, filter_applied=False, generated_at=None, lang="zh")` — 现返回 3 个 artifact dict（csv/xlsx/**json**）；json 的 `content` 为 `{"source", "columns": [{"key","label","role"}], "rows": [...]}` 的 UTF-8 bytes

- [ ] **Step 1: Write the failing tests**

创建 `tests/test_result_export_roles.py`：

```python
"""Tests for result_export column-role inference and JSON artifact."""
import json
import unittest

from sources.result_export import (
    build_result_artifacts,
    infer_column_role,
)

VALID_ROLES = {
    "title", "patent_id", "application_number", "assignee", "inventors",
    "filing_date", "publication_date", "ipc", "abstract",
    "document_title", "document_date", "url", "text",
}


class TestInferColumnRole(unittest.TestCase):
    def test_known_roles_across_sources(self):
        cases = {
            "patentTitle": "title",
            "applicationMetaData.patentTitle": "title",
            "inventionTitle": "title",
            "patentNumber": "patent_id",
            "publicationNumber": "patent_id",
            "applicationNumberText": "application_number",
            "application_number": "application_number",
            "assigneeEntityName": "assignee",
            "applicant": "assignee",
            "inventors": "inventors",
            "inventorName": "inventors",
            "filingDate": "filing_date",
            "publicationDate": "publication_date",
            "grantDate": "publication_date",
            "ipcClass": "ipc",
            "cpcClass": "ipc",
            "abstract": "abstract",
            "abstractText": "abstract",
            "documentTitle": "document_title",
            "documentDate": "document_date",
            "pdfUrl": "url",
            "download_url": "url",
        }
        for key, expected in cases.items():
            with self.subTest(key=key):
                self.assertEqual(infer_column_role(key), expected)

    def test_unknown_keys_fall_back_to_text(self):
        self.assertEqual(infer_column_role("someCustomField"), "text")
        self.assertEqual(infer_column_role(""), "text")
        self.assertEqual(infer_column_role("value"), "text")

    def test_always_returns_role_from_closed_set(self):
        for key in ["patentTitle", "xyz", "applicationMetaData.claims",
                    "documentList", "numberOfPages", "title"]:
            self.assertIn(infer_column_role(key), VALID_ROLES)


class TestBuildResultArtifactsJson(unittest.TestCase):
    def _items(self):
        return [
            {
                "patentTitle": "一种图像处理方法",
                "patentNumber": "US12000123B2",
                "applicationNumberText": "17638216",
                "assigneeEntityName": "华为",
                "filingDate": "2022-02-25",
                "abstractText": "本申请公开了一种图像处理方法。",
            },
        ]

    def test_includes_json_artifact_with_roles_and_source(self):
        artifacts = build_result_artifacts(self._items() * 6, source="uspto")

        formats = {a["format"] for a in artifacts}
        self.assertIn("json", formats)
        self.assertIn("csv", formats)
        self.assertIn("xlsx", formats)

        json_artifact = next(a for a in artifacts if a["format"] == "json")
        payload = json.loads(json_artifact["content"].decode("utf-8"))
        self.assertEqual(payload["source"], "uspto")
        self.assertEqual(json_artifact["row_count"], 6)
        roles = {c["key"]: c["role"] for c in payload["columns"]}
        self.assertEqual(roles["patentTitle"], "title")
        self.assertEqual(roles["applicationNumberText"], "application_number")
        self.assertIn("label", payload["columns"][0])
        self.assertEqual(payload["rows"][0]["patentTitle"], "一种图像处理方法")

    def test_respects_min_rows_threshold(self):
        # Default threshold is 6 — 1 item produces no artifacts at all
        artifacts = build_result_artifacts(self._items(), source="uspto")
        self.assertEqual(artifacts, [])

    def test_source_defaults_to_uspto(self):
        items = self._items() * 6
        artifacts = build_result_artifacts(items)
        payload = json.loads(
            next(a for a in artifacts if a["format"] == "json")["content"]
            .decode("utf-8")
        )
        self.assertEqual(payload["source"], "uspto")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_result_export_roles.py -v`（本地无 pytest 时：服务器上执行）
Expected: FAIL — `ImportError: cannot import name 'infer_column_role'` / `build_result_artifacts` 缺 `source` 参数

- [ ] **Step 3: Write minimal implementation**

在 `sources/result_export.py` 的 `_export_min_rows()` 之后插入：

```python
# ── Column roles for the structured JSON artifact ─────────────────────────
# Closed set consumed by the frontend results view.  Unknown keys are "text".
_ROLE_SUFFIXES: list[tuple[str, str]] = [
    # (role, lowercase suffix) — first match wins, checked in order
    ("document_title", "documenttitle"),
    ("document_date", "documentdate"),
    ("application_number", "applicationnumbertext"),
    ("application_number", "applicationnumber"),
    ("application_number", "application_number"),
    ("patent_id", "patentnumber"),
    ("patent_id", "publicationnumber"),
    ("assignee", "assigneeentityname"),
    ("assignee", "assignee"),
    ("assignee", "applicant"),
    ("inventors", "inventorname"),
    ("inventors", "inventors"),
    ("filing_date", "filingdate"),
    ("filing_date", "applicationdate"),
    ("publication_date", "publicationdate"),
    ("publication_date", "grantdate"),
    ("ipc", "ipcclass"),
    ("ipc", "cpcclass"),
    ("ipc", "ipc"),
    ("abstract", "abstracttext"),
    ("abstract", "abstract"),
    ("title", "patenttitle"),
    ("title", "inventiontitle"),
    ("title", "title"),
    ("url", "pdfurl"),
    ("url", "downloadurl"),
    ("url", "download_url"),
    ("url", "document_url"),
]


def infer_column_role(key: str) -> str:
    """Map a flattened result column key to a frontend rendering role.

    Keys may carry prefixes (e.g. ``applicationMetaData.patentTitle``) —
    only the last path segment is compared.  Unknown keys map to ``text``.
    """
    segment = str(key or "").lower().rsplit(".", 1)[-1].strip()
    for role, suffix in _ROLE_SUFFIXES:
        if segment == suffix:
            return role
    return "text"
```

修改 `build_result_artifacts` 签名并追加 json artifact（`lang` 归一化之后、`metadata` 构建之前加 `source` 参数；`return [...]` 改为包含第三个 dict）：

```python
def build_result_artifacts(
    items: list[Any],
    *,
    source: str = "uspto",
    query_id: str | None = None,
    original_count: int | None = None,
    filter_applied: bool = False,
    generated_at: datetime | None = None,
    lang: str = "zh",
) -> list[dict[str, Any]]:
    """Build CSV, XLSX and structured JSON artifacts from result items.

    ``source`` is one of ``uspto`` / ``google_patents`` / ``uspto_documents``
    and rides along in the JSON payload so the frontend can drive per-row
    detail actions.
    """
```

在 `common = {...}` 之前追加：

```python
    json_payload = {
        "source": source,
        "columns": [
            {
                "key": col,
                "label": uspto_field_label(col, lang),
                "role": infer_column_role(col),
            }
            for col in columns
        ],
        "rows": rows,
    }
    json_content = json.dumps(
        json_payload, ensure_ascii=False,
    ).encode("utf-8")
```

return 列表追加第三个 dict：

```python
        {
            **common,
            "artifact_id": f"{uuid.uuid4().hex}-json",
            "format": "json",
            "filename": f"{base_name}.json",
            "mime_type": "application/json",
            "content": json_content,
        },
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_result_export_roles.py -v`
Expected: PASS（12 个测试全绿）

- [ ] **Step 5: Commit**

```bash
git add sources/result_export.py tests/test_result_export_roles.py
git commit -m "feat: structured JSON result artifact with column roles"
```

---

### Task 2: general_agent.py — 传递 source

**Files:**
- Modify: `sources/agents/general_agent.py:1975,2065,2158`（三个 `build_result_artifacts(...)` 调用点）+ 新增模块级辅助函数（放在 `_extract_patent_ids_from_items` 之后）
- Test: `tests/test_general_agent_result_source.py`（新建）

**Interfaces:**
- Consumes: Task 1 的 `build_result_artifacts(source=...)`
- Produces: `_infer_result_source(tool_info) -> str` — `uspto` | `google_patents` | `uspto_documents`

- [ ] **Step 1: Write the failing test**

创建 `tests/test_general_agent_result_source.py`：

```python
"""Tests for _infer_result_source in general_agent."""
import unittest
from types import SimpleNamespace

from sources.agents.general_agent import _infer_result_source


def _tool(url: str) -> SimpleNamespace:
    return SimpleNamespace(url=url)


class TestInferResultSource(unittest.TestCase):
    def test_google_patents(self):
        self.assertEqual(
            _infer_result_source(_tool("https://patents.google.com/x")),
            "google_patents",
        )

    def test_uspto_documents(self):
        self.assertEqual(
            _infer_result_source(
                _tool("https://api.uspto.gov/api/v1/patent/applications/1/documents")
            ),
            "uspto_documents",
        )

    def test_uspto_default(self):
        self.assertEqual(
            _infer_result_source(
                _tool("https://api.uspto.gov/api/v1/patent/applications/search")
            ),
            "uspto",
        )
        self.assertEqual(_infer_result_source(_tool("")), "uspto")
        self.assertEqual(_infer_result_source(None), "uspto")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_general_agent_result_source.py -v`
Expected: FAIL — `ImportError: cannot import name '_infer_result_source'`

- [ ] **Step 3: Write minimal implementation**

在 `_extract_patent_ids_from_items` 函数之后插入：

```python
def _infer_result_source(tool_info) -> str:
    """Infer the search result source from the backend tool URL."""
    url = (getattr(tool_info, "url", "") or "").lower()
    if "patents.google.com" in url:
        return "google_patents"
    if "documents" in url:
        return "uspto_documents"
    return "uspto"
```

三个调用点（第 1975/2065/2158 行）的 `build_result_artifacts(` 调用中加一行（每处相同）：

```python
                artifacts = build_result_artifacts(
                    items_for_export,
                    source=_infer_result_source(self.knowledgeTool[1]),
                    query_id=getattr(self, "_last_query_id", None),
                    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_general_agent_result_source.py -v && python -m py_compile sources/agents/general_agent.py`
Expected: PASS + COMPILE OK

- [ ] **Step 5: Commit**

```bash
git add sources/agents/general_agent.py tests/test_general_agent_result_source.py
git commit -m "feat: pass result source through to structured JSON artifact"
```

---

### Task 3: patent_detail.py — 说明书/权利要求端点

**Files:**
- Create: `api_routes/patent_detail.py`
- Test: `tests/test_patent_detail_api.py`（新建，直接测 handler 内部逻辑函数 + FastAPI TestClient 冒烟）

**Interfaces:**
- Produces（`register_patent_detail_routes(logger, config)` 返回 router，注册两个端点）:
  - `GET /patent/{source}/{patent_id}/spec` → `200 {success, sections: [{heading, paragraphs[]}], source_url}` | `400`（source 非法）| `502`（上游失败）
  - `GET /patent/{source}/{patent_id}/claims` → `200 {success, claims: [{number, text, independent}]}` | `501`（解析不出权利要求）
  - 模块级纯函数：`split_description_sections(paragraphs: list[str]) -> list[dict]`、`build_claims_payload(claims: list[str]) -> dict`

- [ ] **Step 1: Write the failing tests**

创建 `tests/test_patent_detail_api.py`：

```python
"""Tests for patent detail endpoints (spec / claims)."""
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from api_routes.patent_detail import (
    build_claims_payload,
    split_description_sections,
)


class TestSplitDescriptionSections(unittest.TestCase):
    def test_chunks_paragraphs_without_headings(self):
        paras = [f"paragraph {i}" for i in range(40)]
        sections = split_description_sections(paras)
        self.assertEqual(len(sections), 3)  # 15 + 15 + 10
        self.assertEqual(sections[0]["heading"], "段落 1-15")
        self.assertEqual(len(sections[0]["paragraphs"]), 15)

    def test_uses_natural_headings_when_present(self):
        paras = [
            "技术领域", "本申请涉及电池。",
            "背景技术", "现有技术存在不足。",
            "发明内容", "提供一种新的结构。",
        ]
        sections = split_description_sections(paras)
        headings = [s["heading"] for s in sections]
        self.assertIn("技术领域", headings)
        self.assertIn("背景技术", headings)
        self.assertIn("发明内容", headings)

    def test_empty_paragraphs(self):
        self.assertEqual(split_description_sections([]), [])


class TestBuildClaimsPayload(unittest.TestCase):
    def test_marks_first_claim_independent(self):
        payload = build_claims_payload(["1. 一种机器人。", "2. 如权利要求1所述。"])
        self.assertEqual(payload["success"], True)
        self.assertEqual(len(payload["claims"]), 2)
        self.assertTrue(payload["claims"][0]["independent"])
        self.assertFalse(payload["claims"][1]["independent"])
        self.assertEqual(payload["claims"][0]["number"], 1)

    def test_empty_claims(self):
        self.assertEqual(build_claims_payload([]), {"success": False, "claims": []})


class TestSpecHandlerLogic(unittest.IsolatedAsyncioTestCase):
    async def test_spec_fetches_google_description_and_splits(self):
        fake_client = MagicMock()
        fake_client.query_description = AsyncMock(return_value=["para1", "para2"])
        fake_client.close = AsyncMock()

        from api_routes.patent_detail import _fetch_spec_text

        with patch(
            "api_routes.patent_detail.GooglePatentsClient",
            return_value=fake_client,
        ):
            result = await _fetch_spec_text("uspto", "US12000123B2")

        self.assertEqual(len(result["sections"]), 1)
        self.assertIn("patents.google.com", result["source_url"])

    async def test_spec_raises_on_backend_failure(self):
        fake_client = MagicMock()
        fake_client.query_description = AsyncMock(
            side_effect=Exception("boom")
        )
        fake_client.close = AsyncMock()

        from api_routes.patent_detail import _fetch_spec_text, PatentDetailError

        with patch(
            "api_routes.patent_detail.GooglePatentsClient",
            return_value=fake_client,
        ):
            with self.assertRaises(PatentDetailError):
                await _fetch_spec_text("uspto", "US12000123B2")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_patent_detail_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'api_routes.patent_detail'`

- [ ] **Step 3: Write minimal implementation**

创建 `api_routes/patent_detail.py`（完整文件）：

```python
#!/usr/bin/env python3
"""Patent detail endpoints for the split-view results page.

GET /patent/{source}/{patent_id}/spec    — 说明书分段全文
GET /patent/{source}/{patent_id}/claims  — 权利要求列表

Both endpoints read from Google Patents (patents.google.com) regardless of
the declared *source* — Google Patents indexes US/CN/JP/EP publications with
structured claim/description text, while USPTO PDFs are scanned images that
would require vision-LLM extraction (too heavy for an on-demand endpoint).
``source`` is validated against the known set and carried through for
future data-source routing.

Auth: Firebase bearer token (same pattern as other web routes).
"""

import re

from fastapi import APIRouter, HTTPException, Request

from sources.logger import Logger
from sources.user.passport import verify_firebase_token

VALID_SOURCES = {"uspto", "google_patents"}

_SECTION_HEADING_PATTERN = re.compile(
    r"^[\[【]?[0-9]{4}[\]】]?\s*$"
)
_NATURAL_HEADING_PATTERN = re.compile(
    r"^(技术领域|背景技术|发明内容|附图说明|具体实施方式|"
    r"Technical Field|Background|Summary|Brief Description|"
    r"Detailed Description|Embodiments)\s*$"
)
_SECTION_CHUNK_SIZE = 15


class PatentDetailError(Exception):
    """Base error for patent detail fetch failures."""


def split_description_sections(paragraphs: list[str]) -> list[dict]:
    """Split description paragraphs into sections.

    Paragraphs that look like natural headings (技术领域/背景技术/…) start a
    new section; otherwise paragraphs are chunked into numbered sections of
    ``_SECTION_CHUNK_SIZE``.
    """
    sections: list[dict] = []
    current: dict | None = None
    for para in paragraphs:
        para = (para or "").strip()
        if not para:
            continue
        if _NATURAL_HEADING_PATTERN.match(para):
            if current:
                sections.append(current)
            current = {"heading": para, "paragraphs": []}
            continue
        if current is None:
            current = {"heading": "", "paragraphs": []}
        current["paragraphs"].append(para)
    if current:
        sections.append(current)

    if not sections:
        return []
    # Fallback chunking when no natural headings were found
    if len(sections) == 1 and not sections[0]["heading"]:
        chunks = []
        paras = sections[0]["paragraphs"]
        for i in range(0, len(paras), _SECTION_CHUNK_SIZE):
            end = min(i + _SECTION_CHUNK_SIZE, len(paras))
            chunks.append({
                "heading": f"段落 {i + 1}-{end}",
                "paragraphs": paras[i:end],
            })
        return chunks
    return sections


def build_claims_payload(claims: list[str]) -> dict:
    """Build the claims response payload; first claim is marked independent."""
    if not claims:
        return {"success": False, "claims": []}
    payload_claims = []
    for index, text in enumerate(claims, start=1):
        payload_claims.append({
            "number": index,
            "text": text,
            "independent": index == 1,
        })
    return {"success": True, "claims": payload_claims}


async def _fetch_spec_text(source: str, patent_id: str) -> dict:
    """Fetch description text for *patent_id* and split into sections."""
    from sources.google_patents_client import GooglePatentsClient

    client = GooglePatentsClient(delay=0.5)
    try:
        paragraphs = await client.query_description(patent_id, lang="zh")
    except Exception as exc:
        raise PatentDetailError(str(exc)) from exc
    finally:
        await client.close()

    return {
        "sections": split_description_sections(paragraphs),
        "source_url": f"https://patents.google.com/patent/{patent_id}",
    }


async def _fetch_claims(source: str, patent_id: str) -> dict:
    """Fetch claims for *patent_id* from Google Patents."""
    from sources.google_patents_client import GooglePatentsClient

    client = GooglePatentsClient(delay=0.5)
    try:
        claims = await client.query_claims(patent_id, lang="zh")
    except Exception as exc:
        raise PatentDetailError(str(exc)) from exc
    finally:
        await client.close()

    return build_claims_payload(claims)


def register_patent_detail_routes(logger, config):
    """Register patent detail routes with dependency injection."""
    router = APIRouter()
    logger = Logger("patent_detail.log")

    @router.get("/patent/{source}/{patent_id}/spec")
    async def patent_spec(source: str, patent_id: str, http_request: Request = None):
        if http_request is not None:
            auth_header = http_request.headers.get("Authorization")
            verify_firebase_token(auth_header)
        if source not in VALID_SOURCES:
            raise HTTPException(status_code=400, detail="Unsupported source")
        if not patent_id or len(patent_id) > 40:
            raise HTTPException(status_code=400, detail="Invalid patent_id")
        try:
            payload = await _fetch_spec_text(source, patent_id)
        except PatentDetailError as exc:
            logger.error(f"spec fetch failed — source={source}, id={patent_id}: {exc}")
            raise HTTPException(
                status_code=502,
                detail="Patent specification unavailable",
            )
        return {"success": True, **payload}

    @router.get("/patent/{source}/{patent_id}/claims")
    async def patent_claims(source: str, patent_id: str, http_request: Request = None):
        if http_request is not None:
            auth_header = http_request.headers.get("Authorization")
            verify_firebase_token(auth_header)
        if source not in VALID_SOURCES:
            raise HTTPException(status_code=400, detail="Unsupported source")
        if not patent_id or len(patent_id) > 40:
            raise HTTPException(status_code=400, detail="Invalid patent_id")
        try:
            payload = await _fetch_claims(source, patent_id)
        except PatentDetailError as exc:
            logger.error(f"claims fetch failed — source={source}, id={patent_id}: {exc}")
            raise HTTPException(
                status_code=502,
                detail="Patent claims unavailable",
            )
        if not payload.get("success"):
            # Honest degrade: claims not parseable from the public record
            raise HTTPException(
                status_code=501,
                detail="权利要求暂不可用，请通过 PDF 原文查看",
            )
        return payload

    return router
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_patent_detail_api.py -v && python -m py_compile api_routes/patent_detail.py`
Expected: PASS + COMPILE OK

- [ ] **Step 5: Commit**

```bash
git add api_routes/patent_detail.py tests/test_patent_detail_api.py
git commit -m "feat: patent spec and claims detail endpoints"
```

---

### Task 4: long_task.py — POST /long_task/submit

**Files:**
- Modify: `api_routes/long_task.py`（在 `register_long_task_routes` 内新增端点；模块级新增 `_normalize_submit_patent_id` 纯函数）
- Test: `tests/test_long_task_submit.py`（新建）

**Interfaces:**
- Produces:
  - `_normalize_submit_patent_id(raw: str, scenario: str) -> str` — 去空格/前缀，prosecution 要求恰 8 位数字，family 要求 ≥6 位；非法时 `raise ValueError`
  - `POST /long_task/submit` body `{scenario: "prosecution"|"family", patent_id: str, query?: str, lang?: "zh"|"en", session_id?: str}` → `200 {success, task_id, session_id, status: "queued"|"running"}` | `400` | `401`

- [ ] **Step 1: Write the failing tests**

创建 `tests/test_long_task_submit.py`：

```python
"""Tests for the long task submit endpoint helpers."""
import unittest

from api_routes.long_task import _normalize_submit_patent_id


class TestNormalizeSubmitPatentId(unittest.TestCase):
    def test_strips_us_prefix(self):
        self.assertEqual(
            _normalize_submit_patent_id(" US17638216 ", "prosecution"),
            "17638216",
        )

    def test_prosecution_requires_exactly_8_digits(self):
        with self.assertRaises(ValueError):
            _normalize_submit_patent_id("US12000123B2", "prosecution")
        with self.assertRaises(ValueError):
            _normalize_submit_patent_id("12345", "prosecution")

    def test_family_accepts_publication_numbers(self):
        self.assertEqual(
            _normalize_submit_patent_id("US12000123B2", "family"),
            "US12000123B2",
        )

    def test_rejects_garbage(self):
        with self.assertRaises(ValueError):
            _normalize_submit_patent_id("", "family")
        with self.assertRaises(ValueError):
            _normalize_submit_patent_id("<>script", "family")

    def test_rejects_unknown_scenario(self):
        with self.assertRaises(ValueError):
            _normalize_submit_patent_id("17638216", "batch")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_long_task_submit.py -v`
Expected: FAIL — `ImportError: cannot import name '_normalize_submit_patent_id'`

- [ ] **Step 3: Write minimal implementation**

在 `api_routes/long_task.py` 的 `register_long_task_routes` 定义之前插入：

```python
def _normalize_submit_patent_id(raw: str, scenario: str) -> str:
    """Validate and normalize a patent ID for the submit endpoint.

    - prosecution: must be an 8-digit US application number (prefix stripped)
    - family: any publication/application number with >= 6 alphanumerics
    """
    if scenario not in ("prosecution", "family"):
        raise ValueError(f"Unknown scenario: {scenario}")
    value = (raw or "").strip()
    if not value:
        raise ValueError("patent_id is required")
    if scenario == "prosecution":
        digits = "".join(ch for ch in value if ch.isdigit())
        if len(digits) != 8:
            raise ValueError(
                "Prosecution analysis requires an 8-digit US application number"
            )
        return digits
    clean = re.sub(r"[^A-Za-z0-9]", "", value)
    if len(clean) < 6:
        raise ValueError("patent_id too short for family analysis")
    return value
```

在 `register_long_task_routes` 内、`recover_long_task` 之后插入端点（注意：`long_task.py` 顶部需加 `import re`；`re` 已由 `_normalize_submit_patent_id` 使用）：

```python
    @router.post("/long_task/submit")
    async def submit_long_task(http_request: Request):
        """Directly submit a prosecution/family long task (results-page button).

        Body: {scenario: "prosecution"|"family", patent_id, query?, lang?,
               session_id?}
        Reuses the existing queue + Celery dispatch; polling/download go
        through the existing status/report endpoints.
        """
        import json as _json
        import uuid as _uuid

        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)
        user_id = int(user["uid"])

        try:
            body = await http_request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        scenario = body.get("scenario", "")
        try:
            patent_id = _normalize_submit_patent_id(
                body.get("patent_id", ""), scenario,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        query = str(body.get("query") or "").strip() or (
            f"分析专利 {patent_id} 的审查历史" if scenario == "prosecution"
            else f"分析 {patent_id} 及其全球同族的审查差异"
        )
        lang = body.get("lang") if body.get("lang") in ("zh", "en") else "zh"

        from sources.knowledge.knowledge import get_db_connection
        from sources.long_task.user_queue import try_start_user_task

        task_id = f"lt_{_uuid.uuid4().hex[:12]}"
        session_id = str(body.get("session_id") or "").strip()

        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                if not session_id:
                    session_id = f"sess_{_uuid.uuid4().hex[:12]}"
                    cur.execute(
                        """INSERT INTO conversations
                           (session_id, user_id, title, messages, long_task_ids)
                           VALUES (%s, %s, %s, %s, %s)""",
                        (session_id, user_id, query[:60],
                         _json.dumps([], ensure_ascii=False),
                         _json.dumps([task_id])),
                    )
                else:
                    cur.execute(
                        """SELECT long_task_ids FROM conversations
                           WHERE session_id = %s AND user_id = %s AND status != 2""",
                        (session_id, user_id),
                    )
                    row = cur.fetchone()
                    if not row:
                        raise HTTPException(status_code=400, detail="Unknown session_id")
                    existing = _json.loads(row["long_task_ids"]) if isinstance(
                        row["long_task_ids"], str
                    ) else (row["long_task_ids"] or [])
                    existing.append(task_id)
                    cur.execute(
                        """UPDATE conversations SET long_task_ids = %s,
                           update_time = NOW() WHERE session_id = %s""",
                        (_json.dumps(existing), session_id),
                    )

                task_type = (
                    "prosecution_analysis" if scenario == "prosecution"
                    else "family_analysis"
                )
                cur.execute(
                    """INSERT INTO long_tasks
                       (task_id, session_id, user_id, task_type, input_params, status)
                       VALUES (%s, %s, %s, %s, %s, 'pending')""",
                    (task_id, session_id, user_id, task_type,
                     _json.dumps({
                         "query": query,
                         "patent_id": patent_id,
                         "patent_source": "uspto",
                         "lang": lang,
                     }, ensure_ascii=False)),
                )
                conn.commit()
        finally:
            conn.close()

        celery_params = {
            "query": query,
            "session_id": session_id,
            "user_id": str(user_id),
            "scenario": "prosecution" if scenario == "prosecution" else "families",
            "patent_id": patent_id,
            "patent_source": "uspto",
            "patent_id_type": (
                "application_number" if scenario == "prosecution" else "unknown"
            ),
            "lang": lang,
        }

        queue_result = try_start_user_task(str(user_id), task_id)
        status = "running"
        if queue_result == "running":
            if scenario == "prosecution":
                from celery_worker import execute_prosecution_analysis
                execute_prosecution_analysis.delay(task_id=task_id, params=celery_params)
            else:
                from celery_worker import execute_family_analysis
                execute_family_analysis.delay(task_id=task_id, params=celery_params)
        else:
            status = "queued"
        logger.info(
            f"submit_long_task — task_id={task_id}, scenario={scenario}, "
            f"patent_id={patent_id}, queue={queue_result}"
        )
        return {
            "success": True,
            "task_id": task_id,
            "session_id": session_id,
            "status": status,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_long_task_submit.py -v && python -m py_compile api_routes/long_task.py`
Expected: PASS + COMPILE OK

- [ ] **Step 5: Commit**

```bash
git add api_routes/long_task.py tests/test_long_task_submit.py
git commit -m "feat: direct long task submit endpoint for prosecution and family analysis"
```

---

### Task 5: api.py — 注册 patent_detail 路由

**Files:**
- Modify: `api.py:179`（`api.include_router(patent.router, tags=["patent"])` 之后）
- Test: `tests/test_api_route_imports.py`（已有，检查并确保通过）

- [ ] **Step 1: Add the registration lines**

在 `api.py` 的 patent 路由注册之后追加：

```python
from api_routes import patent_detail
patent_detail_router = patent_detail.register_patent_detail_routes(logger, config)
api.include_router(patent_detail_router, tags=["patent-detail"])
```

- [ ] **Step 2: Run import/route test**

Run: `pytest tests/test_api_route_imports.py -v`（本地缺依赖则在服务器执行）
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add api.py
git commit -m "feat: register patent detail routes"
```

## Phase 2: 前端数据层

### Task 6: chatSession.js — json artifact 解码 → message.results

**Files:**
- Modify: `frontend/nextjs/lib/chatSession.js`（末尾追加函数）
- Test: `frontend/nextjs/lib/chatSession.test.mjs`（追加测试）

**Interfaces:**
- Produces:
  - `decodeResultsArtifact(messages, messageId) -> messages` — 找该消息下 `format === 'json'` 且 `complete === true` 的 artifact，base64 解码 chunks → `JSON.parse` → 挂 `msg.results = {setId: artifactId, source, columns, rows}`；解码失败或不存在时原样返回（幂等，每次重算不重复追加）

- [ ] **Step 1: Write the failing tests**

在 `frontend/nextjs/lib/chatSession.test.mjs` 末尾追加（`node:test` 语法，文件顶部已 import 各函数，需把 `decodeResultsArtifact` 加入 import）：

```javascript
test('decodes complete JSON artifact into message.results', () => {
  const assistant = createChatMessage('assistant', 'answer')
  const messages = [assistant]
  const payload = JSON.stringify({
    source: 'uspto',
    columns: [{ key: 'patentTitle', label: '标题', role: 'title' }],
    rows: [{ patentTitle: '一种图像处理方法' }],
  })
  const b64 = Buffer.from(payload, 'utf-8').toString('base64')

  let withArtifact = addAssistantArtifactStart(messages, assistant.id, {
    artifact_id: 'art-json',
    format: 'json',
    filename: 'r.json',
    mime_type: 'application/json',
    row_count: 1,
    column_count: 1,
  })
  withArtifact = addAssistantArtifactChunk(withArtifact, assistant.id, 'art-json', b64)
  withArtifact = addAssistantArtifactEnd(withArtifact, assistant.id, 'art-json')

  const decoded = decodeResultsArtifact(withArtifact, assistant.id)
  assert.ok(decoded[0].results)
  assert.equal(decoded[0].results.setId, 'art-json')
  assert.equal(decoded[0].results.source, 'uspto')
  assert.equal(decoded[0].results.rows[0].patentTitle, '一种图像处理方法')
})

test('decodeResultsArtifact leaves message untouched when no JSON artifact', () => {
  const assistant = createChatMessage('assistant', 'answer')
  const messages = [assistant]

  const withCsv = addAssistantArtifactStart(messages, assistant.id, {
    artifact_id: 'art-csv', format: 'csv', filename: 'r.csv',
  })
  const decoded = decodeResultsArtifact(withCsv, assistant.id)
  assert.equal(decoded[0].results, undefined)
})

test('decodeResultsArtifact survives malformed JSON', () => {
  const assistant = createChatMessage('assistant', 'answer')
  let messages = addAssistantArtifactStart([assistant], assistant.id, {
    artifact_id: 'art-bad', format: 'json', filename: 'r.json',
  })
  messages = addAssistantArtifactChunk(messages, assistant.id, 'art-bad', '%%%%')
  messages = addAssistantArtifactEnd(messages, assistant.id, 'art-bad')

  const decoded = decodeResultsArtifact(messages, assistant.id)
  assert.equal(decoded[0].results, undefined)
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend/nextjs && node --test lib/chatSession.test.mjs`
Expected: FAIL — `ReferenceError: decodeResultsArtifact is not defined`

- [ ] **Step 3: Write minimal implementation**

`frontend/nextjs/lib/chatSession.js` 末尾追加：

```javascript
function base64ChunksToText(chunks) {
  try {
    if (typeof Buffer !== 'undefined') {
      return Buffer.from(chunks.join(''), 'base64').toString('utf-8')
    }
    const binary = window.atob(chunks.join(''))
    const bytes = new Uint8Array(binary.length)
    for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i)
    return new TextDecoder().decode(bytes)
  } catch {
    return null
  }
}

/**
 * Decode a complete format=json artifact into message.results.
 * Idempotent — returns messages unchanged when there is nothing to decode
 * (or the payload is malformed), so it is safe to call on every update.
 */
export function decodeResultsArtifact(messages, messageId) {
  return messages.map((msg) => {
    if (msg.id !== messageId || msg.results) return msg
    const artifacts = Array.isArray(msg.artifacts) ? msg.artifacts : []
    const jsonArtifact = artifacts.find(
      (artifact) => artifact.format === 'json' && artifact.complete,
    )
    if (!jsonArtifact) return msg
    const text = base64ChunksToText(jsonArtifact.chunks || [])
    if (!text) return msg
    try {
      const payload = JSON.parse(text)
      if (!payload || typeof payload !== 'object' || !Array.isArray(payload.rows)) {
        return msg
      }
      return {
        ...msg,
        results: {
          setId: jsonArtifact.artifactId,
          source: payload.source || 'uspto',
          columns: Array.isArray(payload.columns) ? payload.columns : [],
          rows: payload.rows,
        },
      }
    } catch {
      return msg
    }
  })
}
```

同时把文件顶部注释 `/* eslint-disable */` 不动；`window` 分支仅浏览器环境触发，node 测试走 `Buffer` 分支。

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend/nextjs && node --test lib/chatSession.test.mjs`
Expected: PASS（新 3 个 + 既有全部）

- [ ] **Step 5: Commit**

```bash
git add frontend/nextjs/lib/chatSession.js frontend/nextjs/lib/chatSession.test.mjs
git commit -m "feat: decode JSON artifact into structured message results"
```

---

### Task 7: lib/results.js — 行模型与持久化裁剪

**Files:**
- Create: `frontend/nextjs/lib/results.js`
- Test: `frontend/nextjs/lib/results.test.mjs`（新建）

**Interfaces:**
- Produces:
  - `findRoleColumn(columns, role) -> {key} | null`
  - `buildRowModel(row, columns, source) -> {id, title, meta: [{label, value}], patentId, applicationNumber, url, isDocument, fields: [[label, value]]}` — 专利行 isDocument=false；有 document_title/document_date 角色时 isDocument=true
  - `pruneResultsForPersistence(results, {maxRows = 50, abstractLimit = 500} = {}) -> {setId, source, columns, rows}` — abstract 角色列截断、行数上限、只保留 role≠text 的展示列

- [ ] **Step 1: Write the failing tests**

创建 `frontend/nextjs/lib/results.test.mjs`：

```javascript
import test from 'node:test'
import assert from 'node:assert/strict'

import {
  buildRowModel,
  findRoleColumn,
  pruneResultsForPersistence,
} from './results.js'

const COLUMNS = [
  { key: 'patentTitle', label: '标题', role: 'title' },
  { key: 'patentNumber', label: '专利号', role: 'patent_id' },
  { key: 'applicationNumberText', label: '申请号', role: 'application_number' },
  { key: 'assigneeEntityName', label: '申请人', role: 'assignee' },
  { key: 'publicationDate', label: '公开日', role: 'publication_date' },
  { key: 'abstractText', label: '摘要', role: 'abstract' },
  { key: 'customThing', label: '自定义', role: 'text' },
]

const ROW = {
  patentTitle: '一种图像处理方法',
  patentNumber: 'US12000123B2',
  applicationNumberText: '17638216',
  assigneeEntityName: '华为',
  publicationDate: '2024-06-01',
  abstractText: '摘要文字。',
  customThing: 'x',
}

test('findRoleColumn returns the first column matching a role', () => {
  assert.deepEqual(findRoleColumn(COLUMNS, 'title'), COLUMNS[0])
  assert.deepEqual(findRoleColumn(COLUMNS, 'patent_id'), COLUMNS[1])
  assert.equal(findRoleColumn(COLUMNS, 'url'), null)
})

test('buildRowModel builds title/meta/patent identifiers for patent rows', () => {
  const model = buildRowModel(ROW, COLUMNS, 'uspto')
  assert.equal(model.title, '一种图像处理方法')
  assert.equal(model.patentId, 'US12000123B2')
  assert.equal(model.applicationNumber, '17638216')
  assert.equal(model.isDocument, false)
  assert.equal(model.meta.length, 2) // patent_id + assignee
  assert.equal(model.meta[0].label, '专利号')
  assert.ok(model.fields.length >= 6) // all non-empty fields incl. text role
})

test('buildRowModel detects document rows by role', () => {
  const docColumns = [
    { key: 'documentTitle', label: '文档标题', role: 'document_title' },
    { key: 'documentDate', label: '日期', role: 'document_date' },
    { key: 'pdfUrl', label: '链接', role: 'url' },
  ]
  const docRow = { documentTitle: 'Issue Notification', documentDate: '2025-09-24', pdfUrl: 'https://x/y.pdf' }
  const model = buildRowModel(docRow, docColumns, 'uspto_documents')
  assert.equal(model.isDocument, true)
  assert.equal(model.title, 'Issue Notification')
  assert.equal(model.url, 'https://x/y.pdf')
})

test('buildRowModel tolerates empty rows', () => {
  const model = buildRowModel({}, COLUMNS, 'uspto')
  assert.equal(model.title, '')
  assert.equal(model.patentId, '')
  assert.equal(model.fields.length, 0)
})

test('pruneResultsForPersistence truncates abstracts and caps rows', () => {
  const longAbstract = '字'.repeat(1200)
  const rows = Array.from({ length: 60 }, (_, i) => ({
    ...ROW,
    patentNumber: `US${i}`,
    abstractText: longAbstract,
  }))
  const pruned = pruneResultsForPersistence(
    { setId: 's1', source: 'uspto', columns: COLUMNS, rows },
  )
  assert.equal(pruned.rows.length, 50)
  assert.ok(pruned.rows[0].abstractText.length <= 500)
  assert.equal(pruned.rows[0].patentNumber, 'US0')
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend/nextjs && node --test lib/results.test.mjs`
Expected: FAIL — `Cannot find module './results.js'`

- [ ] **Step 3: Write minimal implementation**

创建 `frontend/nextjs/lib/results.js`：

```javascript
/**
 * Results payload helpers shared by the chat result card and the
 * split-view results page.  Works purely on the structured payload
 * decoded from the format=json artifact.
 */

export function findRoleColumn(columns, role) {
  const list = Array.isArray(columns) ? columns : []
  return list.find((col) => col && col.role === role) || null
}

export function columnValue(row, column) {
  if (!row || !column) return ''
  const value = row[column.key]
  return value === undefined || value === null ? '' : String(value)
}

const META_ROLES = [
  ['patent_id', 'patent_id'],
  ['assignee', 'assignee'],
  ['publication_date', 'publication_date'],
]

export function buildRowModel(row, columns, source) {
  const list = Array.isArray(columns) ? columns : []
  const docTitleCol = findRoleColumn(list, 'document_title')
  const isDocument = Boolean(docTitleCol)

  const titleCol = isDocument ? docTitleCol : findRoleColumn(list, 'title')
  const title = columnValue(row, titleCol)

  const meta = []
  for (const [role] of META_ROLES) {
    const col = findRoleColumn(list, role)
    if (!col) continue
    const value = columnValue(row, col)
    if (value) meta.push({ label: col.label || col.key, value })
  }

  const fields = []
  for (const col of list) {
    const value = columnValue(row, col)
    if (!value) continue
    if (isDocument && col.role === 'document_title') continue
    fields.push([col.label || col.key, value])
  }

  const patentIdCol = findRoleColumn(list, 'patent_id')
  const appNumCol = findRoleColumn(list, 'application_number')
  const urlCol = findRoleColumn(list, 'url')

  return {
    id: String(columnValue(row, patentIdCol) || title || fields[0]?.[1] || 'row'),
    title,
    meta,
    patentId: columnValue(row, patentIdCol),
    applicationNumber: columnValue(row, appNumCol),
    url: columnValue(row, urlCol),
    source,
    isDocument,
    fields,
  }
}

export const MAX_PERSIST_ROWS = 50
export const MAX_PERSIST_ABSTRACT_CHARS = 500

export function pruneResultsForPersistence(
  results,
  { maxRows = MAX_PERSIST_ROWS, abstractLimit = MAX_PERSIST_ABSTRACT_CHARS } = {},
) {
  if (!results || !Array.isArray(results.rows)) return results
  const columns = Array.isArray(results.columns) ? results.columns : []
  const abstractCols = columns.filter((col) => col && col.role === 'abstract')
  const displayCols = columns.filter((col) => col && col.role !== 'text')
  const displayKeys = new Set(displayCols.map((col) => col.key))

  const rows = results.rows.slice(0, maxRows).map((row) => {
    const pruned = {}
    for (const key of Object.keys(row)) {
      if (!displayKeys.has(key)) continue
      let value = row[key]
      if (abstractCols.some((col) => col.key === key) && typeof value === 'string') {
        value = value.slice(0, abstractLimit)
      }
      pruned[key] = value
    }
    return pruned
  })

  return {
    setId: results.setId,
    source: results.source,
    columns: displayCols,
    rows,
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend/nextjs && node --test lib/results.test.mjs`
Expected: PASS（6 个测试）

- [ ] **Step 5: Commit**

```bash
git add frontend/nextjs/lib/results.js frontend/nextjs/lib/results.test.mjs
git commit -m "feat: results row model and persistence pruning helpers"
```

---

### Task 8: services/api.ts — 详情与提交请求函数

**Files:**
- Modify: `frontend/nextjs/services/api.ts`（在 `getLongTaskReportUrl` 之后追加）

**Interfaces:**
- Produces:
  - `fetchPatentSpec(source: string, patentId: string) => Promise<{sections: Array<{heading: string, paragraphs: string[]}>, source_url: string}>`
  - `fetchPatentClaims(source: string, patentId: string) => Promise<{claims: Array<{number: number, text: string, independent: boolean}>}>`
  - `submitLongTask(payload: {scenario: 'prosecution'|'family', patentId: string, query?: string, lang?: string, sessionId?: string}) => Promise<{task_id: string, session_id: string, status: string}>`

- [ ] **Step 1: Add the functions**

`frontend/nextjs/services/api.ts` 末尾追加（复用文件内已有 `get`/`post` 辅助与 `BASE_URL`）：

```typescript
export interface PatentSpecSection {
  heading: string
  paragraphs: string[]
}

export interface PatentSpecResponse {
  sections: PatentSpecSection[]
  source_url: string
}

export interface PatentClaim {
  number: number
  text: string
  independent: boolean
}

export interface PatentClaimsResponse {
  claims: PatentClaim[]
}

export interface SubmitLongTaskResponse {
  success: boolean
  task_id: string
  session_id: string
  status: string
}

export async function fetchPatentSpec(
  source: string,
  patentId: string,
): Promise<PatentSpecResponse> {
  return get<PatentSpecResponse & { success: boolean }>(
    `/patent/${encodeURIComponent(source)}/${encodeURIComponent(patentId)}/spec`,
  )
}

export async function fetchPatentClaims(
  source: string,
  patentId: string,
): Promise<PatentClaimsResponse> {
  return get<PatentClaimsResponse & { success: boolean }>(
    `/patent/${encodeURIComponent(source)}/${encodeURIComponent(patentId)}/claims`,
  )
}

export async function submitLongTask(payload: {
  scenario: 'prosecution' | 'family'
  patentId: string
  query?: string
  lang?: string
  sessionId?: string
}): Promise<SubmitLongTaskResponse> {
  return post<SubmitLongTaskResponse>('/long_task/submit', {
    scenario: payload.scenario,
    patent_id: payload.patentId,
    ...(payload.query ? { query: payload.query } : {}),
    ...(payload.lang ? { lang: payload.lang } : {}),
    ...(payload.sessionId ? { session_id: payload.sessionId } : {}),
  })
}
```

- [ ] **Step 2: Build check**

Run: `cd frontend/nextjs && npx tsc --noEmit`（或 `npm run build`）
Expected: 无新增类型错误

- [ ] **Step 3: Commit**

```bash
git add frontend/nextjs/services/api.ts
git commit -m "feat: patent detail fetch and long task submit API functions"
```

---

### Task 9: ChatContext.tsx — resultsSetId 共享状态

**Files:**
- Modify: `frontend/nextjs/contexts/ChatContext.tsx`

**Interfaces:**
- Produces:
  - `useChatSession()` 返回值新增 `resultsSetId: string | null` 与 `setResultsSetId: Dispatch<SetStateAction<string | null>>`
  - `ChatMessage` 接口新增 `results?: {setId: string, source: string, columns: any[], rows: any[]}`

- [ ] **Step 1: Apply the changes**

`frontend/nextjs/contexts/ChatContext.tsx` 中：

接口 `ChatMessage` 追加：

```typescript
  results?: {
    setId: string
    source: string
    columns: Array<{ key: string; label: string; role: string }>
    rows: Array<Record<string, unknown>>
  }
```

`ChatContextValue` 接口追加：

```typescript
  resultsSetId: string | null
  setResultsSetId: Dispatch<SetStateAction<string | null>>
```

`ChatProvider` 内追加状态并注入 value：

```typescript
  const [resultsSetId, setResultsSetId] = useState<string | null>(null)
```

value 对象中追加：

```typescript
        resultsSetId,
        setResultsSetId,
```

- [ ] **Step 2: Build check**

Run: `cd frontend/nextjs && npx tsc --noEmit`
Expected: 无新增类型错误（`useChatSession` 消费方不改也兼容——新增字段非破坏性）

- [ ] **Step 3: Commit**

```bash
git add frontend/nextjs/contexts/ChatContext.tsx
git commit -m "feat: shared results set selection state in chat context"
```

## Phase 3: 前端 UI

### Task 10: 提取 useChatStream hook（聊天页重构，行为不变）

**Files:**
- Create: `frontend/nextjs/lib/useChatStream.ts`
- Modify: `frontend/nextjs/app/app/(auth)/chat/page.tsx`（删除被搬走的代码，改用 hook）

**Interfaces:**
- Produces:
  - `useChatStream() => { send, abort, transientStatus, selectedFiles, setSelectedFiles, addFiles, removeFile, isDragOver, setIsDragOver }`
  - hook 内部使用 `useChatSession()`、`useAuth()`、`useI18n()`；send 的 SSE 处理与聊天页现有 `send()` 完全一致，唯一新增行为：**每个 `artifact_end` 事件后调用 `setMessages(m => decodeResultsArtifact(m, assistantId))`**
- Consumes: Task 6 的 `decodeResultsArtifact`；`queryStream`/`queryStreamWithFiles`/`pollLongTaskBatchStatus` 等现有 services

**注意**：这是纯行为保持重构，无单测设施——用 build + 手动回归验证。

- [ ] **Step 1: Create the hook file by moving code**

创建 `frontend/nextjs/lib/useChatStream.ts`，内容 = 从 `chat/page.tsx` 原样搬入以下成员（import 相应补齐）：

| 从 page.tsx 搬入 | 在 hook 中的形态 |
|---|---|
| `cleanGarbledText` 函数（第 86-93 行） | 模块级函数，原样 |
| `send()` 函数体（第 395-617 行） | hook 内部函数；`input`/`messages`/`streaming`/`sessionId` 等改为从 `useChatSession()` 取；`selectedFiles` 用 hook 本地 state；`requireAuth`/`t` 从 hooks 取；SSE 循环中每个 `artifact_end` 分支追加一行 `setMessages((m) => decodeResultsArtifact(m, assistantId))` |
| `stopLongTaskPolling` / `ensureGlobalPollLoop` / `startLongTaskPolling`（第 715-896 行） | 原样搬入，closure 引用 hook 内 `activeTasksRef` / `globalPollTimerRef` / `longTaskReceivedRef` / `transientStatus` setter |
| `activeTasksRef` / `globalPollTimerRef` / `longTaskReceivedRef` 三个 ref（第 81-83 行） | hook 内 `useRef`，原样 |
| `transientStatus` state（第 85 行） | hook 内 `useState`，原样 |
| 文件选择逻辑：`MAX_FILE_SIZE/MAX_FILE_COUNT/ALLOWED_EXTENSIONS/ALLOWED_MIMES` 常量、`addFiles/removeFile/openFilePicker?`（第 636-663、665-667 行）、`isDragOver` state | hook 内；`openFilePicker`/`fileInputRef` 留在页面（页面持有 input 元素） |

hook 返回签名与卸载清理：

```typescript
'use client'

import { useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import { queryStream, queryStreamWithFiles, pollLongTaskBatchStatus, getLongTaskReportUrl, getSession, saveSessionMessages } from '@/services/api'
import { pollRecoverLongTask } from '@/lib/longTaskRecovery'
import { useI18n } from '@/lib/app-i18n'
import { useAuth } from '@/contexts/AuthContext'
import { useChatSession, type ChatMessage } from '@/contexts/ChatContext'
import { decodeResultsArtifact } from '@/lib/chatSession'
import {
  addAssistantArtifactChunk, addAssistantArtifactEnd, addAssistantArtifactStart,
  addAssistantPatentIds, createChatId, createChatMessage,
  updateAssistantMessage, replaceAssistantMessage,
} from '@/lib/chatSession'

// ... 原样搬入的 cleanGarbledText ...

export function useChatStream() {
  const { t, lang } = useI18n()
  const { user, requireAuth } = useAuth()
  const {
    messages, setMessages, input, setInput,
    streaming, setStreaming, streamingId, setStreamingId,
    abortRef, sessionId, setSessionId,
  } = useChatSession()

  const [transientStatus, setTransientStatus] = useState('')
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [isDragOver, setIsDragOver] = useState(false)
  const activeTasksRef = useRef<Map<string, string>>(new Map())
  const globalPollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const longTaskReceivedRef = useRef(false)
  const sendRef = useRef<(files?: File[]) => Promise<void>>(async () => {})
  sendRef.current = send

  // ... 原样搬入的 send / stopLongTaskPolling / ensureGlobalPollLoop /
  //     startLongTaskPolling（内部按上表改名）...

  useEffect(() => {
    return () => {
      // 原样搬入页面第 899-902 行的清理逻辑
      activeTasksRef.current.clear()
      if (globalPollTimerRef.current) {
        clearInterval(globalPollTimerRef.current)
        globalPollTimerRef.current = null
      }
    }
  }, [])

  return {
    send: (files: File[] = []) => {
      if (files.length > 0) setSelectedFiles((prev) => [...prev, ...files])
      return sendRef.current()
    },
    abort: () => abortRef.current?.abort(),
    transientStatus,
    selectedFiles,
    setSelectedFiles,
    addFiles,
    removeFile,
    isDragOver,
    setIsDragOver,
  }
}
```

搬入的 `send()` 中 `const currentFiles = selectedFiles` 逻辑保持原样（hook 内 selectedFiles 已存在）。

- [ ] **Step 2: Rewire chat/page.tsx**

`app/app/(auth)/chat/page.tsx`：
- 删除上表列出的所有已搬走代码（`send`、轮询函数、refs、`transientStatus`、文件常量与处理函数、`isDragOver`）
- 保留：UI 渲染、`handleKeyDown`、`handleInput`、`fileInputRef`/`openFilePicker`、场景提示、session 加载 effect、持久化 effect、滚动逻辑
- 组件顶部改为：

```typescript
  const { t, lang } = useI18n()
  const { user, requireAuth } = useAuth()
  const { messages, setMessages, input, setInput, streaming, streamingId, sessionId, setSessionId } = useChatSession()
  const {
    send, abort, transientStatus, selectedFiles, setSelectedFiles,
    addFiles, removeFile, isDragOver, setIsDragOver,
  } = useChatStream()
```

- `onChange={e => { if (e.target.files) addFiles(e.target.files); e.target.value = '' }}` 等处引用不变（hook 导出的 addFiles 签名与原来一致）
- 发送按钮 `onClick={() => send()}`

- [ ] **Step 3: Build check**

Run: `cd frontend/nextjs && npm run build`
Expected: build 成功，无类型错误

- [ ] **Step 4: Manual regression（行为保持）**

本地起 dev（`npm run dev`）验证：发消息 → 流式回复 → 长任务卡片轮询 → 文件上传 → 会话恢复，全部与重构前一致。

- [ ] **Step 5: Commit**

```bash
git add frontend/nextjs/lib/useChatStream.ts frontend/nextjs/app/app/'(auth)'/chat/page.tsx
git commit -m "refactor: extract chat stream hook for reuse in results page"
```

---

### Task 11: ResultCard + 聊天页集成 + results 持久化

**Files:**
- Create: `frontend/nextjs/components/app/ResultCard.tsx`
- Modify: `frontend/nextjs/app/app/(auth)/chat/page.tsx`（渲染卡片、session 持久化含 results、恢复水合）
- Modify: `frontend/nextjs/lib/app-i18n/locales/zh.ts`、`en.ts`（chat 段新增 2 个 key）

**Interfaces:**
- Produces: `<ResultCard results={msg.results} sessionId={string | null} />` — 点击跳转 `/app/results?set={setId}&session_id={sessionId || ''}`

- [ ] **Step 1: Add i18n keys**

`locales/zh.ts` 的 `chat` 段追加：

```typescript
    resultsCardTitle: '🔍 检索到 {count} 条结果',
    resultsViewButton: '在结果页查看',
```

`locales/en.ts` 的 `chat` 段追加：

```typescript
    resultsCardTitle: '🔍 Found {count} results',
    resultsViewButton: 'View in results page',
```

- [ ] **Step 2: Create ResultCard.tsx**

```tsx
'use client'

import { useRouter } from 'next/navigation'
import { useI18n } from '@/lib/app-i18n'

interface ResultsPayload {
  setId: string
  source: string
  columns: Array<{ key: string; label: string; role: string }>
  rows: Array<Record<string, unknown>>
}

export default function ResultCard({ results, sessionId }: { results: ResultsPayload; sessionId: string | null }) {
  const { t } = useI18n()
  const router = useRouter()

  function openResultsPage() {
    const params = new URLSearchParams({ set: results.setId })
    if (sessionId) params.set('session_id', sessionId)
    router.push(`/app/results?${params.toString()}`)
  }

  return (
    <div className="result-card">
      <div className="result-card-header">
        <span className="result-card-title">
          {t('chat.resultsCardTitle').replace('{count}', String(results.rows.length))}
        </span>
        <span className="result-card-source">{results.source}</span>
      </div>
      <button className="result-card-button" onClick={openResultsPage}>
        {t('chat.resultsViewButton')}
      </button>
    </div>
  )
}
```

（样式类 `result-card*` 追加到 `frontend/nextjs/styles/app.css`：卡片圆角边框、标题行、主按钮色与现有 `--color-accent` 一致。）

- [ ] **Step 3: Wire chat page**

`chat/page.tsx` 渲染处（`<MarkdownMessage .../>` 之前）追加：

```tsx
            {msg.role === 'assistant' ? (
              <>
                {(msg as any).results && (
                  <ResultCard results={(msg as any).results} sessionId={sessionId} />
                )}
                <MarkdownMessage ... />
              </>
            ) : (...)}
```

import 追加：`import ResultCard from '@/components/app/ResultCard'`、`import { pruneResultsForPersistence } from '@/lib/results'`

持久化 effect（`toSave` 映射）追加 results 字段：

```typescript
        const toSave = messages.map(m => ({
          role: m.role,
          content: m.content,
          ...(m.taskId ? { taskId: m.taskId } : {}),
          ...(m.resultSummary ? { resultSummary: m.resultSummary } : {}),
          ...(m.patent_ids ? { patent_ids: m.patent_ids } : {}),
          ...((m as any).results
            ? { results: pruneResultsForPersistence((m as any).results) }
            : {}),
        }))
```

session 恢复 effect（`loaded` 映射）追加：

```typescript
              results: (m as any).results || undefined,
```

- [ ] **Step 4: Build check + node tests**

Run: `cd frontend/nextjs && node --test lib/ && npm run build`
Expected: 测试 PASS + build 成功

- [ ] **Step 5: Commit**

```bash
git add frontend/nextjs/components/app/ResultCard.tsx frontend/nextjs/styles/app.css frontend/nextjs/app/app/'(auth)'/chat/page.tsx frontend/nextjs/lib/app-i18n/locales/zh.ts frontend/nextjs/lib/app-i18n/locales/en.ts
git commit -m "feat: result card in chat with split-view entry and persistence"
```

---

### Task 12: 结果页骨架 — 路由 + 三栏布局 + 聊天侧边栏

**Files:**
- Create: `frontend/nextjs/app/app/(auth)/results/page.tsx`
- Modify: `frontend/nextjs/lib/app-i18n/locales/zh.ts`、`en.ts`（新增 results 段）

**Interfaces:**
- Consumes: `useChatSession`（Task 9）、`useChatStream`（Task 10）、`ResultCard`（Task 11）、`ResultList`/`DetailPanel`（Task 13/14 占位 — 本任务先用最小占位组件，后续任务替换）
- Produces: `/app/results?set={setId}&session_id={sid}` 页面骨架：左聊天窄栏（消息列表 + 输入框 + 发送）、中结果列表占位、右详情面板占位；无 setId → 空态引导

- [ ] **Step 1: Add i18n keys**

`locales/zh.ts` 追加 `results` 段（en.ts 同步英文）：

```typescript
  results: {
    emptyTitle: '还没有可浏览的检索结果',
    emptyHint: '回到对话页发起一次检索，然后在结果卡片中点击"在结果页查看"。',
    backToChat: '回对话页',
    sidebarPlaceholder: '继续追问…',
    selectedCount: '{count} 条结果',
  },
```

- [ ] **Step 2: Create the page**

`frontend/nextjs/app/app/(auth)/results/page.tsx`：

```tsx
'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { useI18n } from '@/lib/app-i18n'
import { useChatSession, type ChatMessage } from '@/contexts/ChatContext'
import { useChatStream } from '@/lib/useChatStream'
import { getSession } from '@/services/api'
import ResultCard from '@/components/app/ResultCard'
import ResultList from '@/components/app/results/ResultList'
import DetailPanel from '@/components/app/results/DetailPanel'

export default function ResultsPage() {
  const { t } = useI18n()
  const router = useRouter()
  const searchParams = useSearchParams()
  const { messages, setMessages, sessionId, setSessionId, resultsSetId, setResultsSetId, input, setInput, streaming } = useChatSession()
  const { send, selectedFiles, setSelectedFiles, addFiles, removeFile, isDragOver, setIsDragOver } = useChatStream()

  const setId = searchParams.get('set') || resultsSetId
  const loadedRef = useRef(false)

  // Hydrate conversation when arriving with a session_id but no messages
  useEffect(() => {
    const sid = searchParams.get('session_id')
    if (!sid || loadedRef.current || messages.length > 0) return
    loadedRef.current = true
    ;(async () => {
      try {
        const data = await getSession(sid)
        if (!Array.isArray(data.messages)) return
        setMessages(data.messages
          .filter((m: any) => m.role && m.content)
          .map((m: any, i: number) => ({
            id: `hist_${i}_${Date.now()}`,
            role: m.role,
            content: m.content,
            taskId: m.taskId || undefined,
            resultSummary: m.resultSummary || undefined,
            patent_ids: m.patent_ids || undefined,
            results: m.results || undefined,
            artifacts: [],
          })))
        setSessionId(sid)
      } catch {
        // Session unavailable — stay in empty state
      }
    })()
  }, [searchParams, messages.length, setMessages, setSessionId])

  useEffect(() => {
    if (setId) setResultsSetId(setId)
  }, [setId, setResultsSetId])

  const activeMessage: ChatMessage | undefined = useMemo(() => {
    if (!setId) return undefined
    return messages.find((m) => (m as any).results?.setId === setId)
  }, [messages, setId])

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  if (!activeMessage) {
    return (
      <div className="page active results-page">
        <div className="results-empty">
          <h3>{t('results.emptyTitle')}</h3>
          <p>{t('results.emptyHint')}</p>
          <button onClick={() => router.push('/app/chat')}>{t('results.backToChat')}</button>
        </div>
      </div>
    )
  }

  return (
    <div className="page active results-page">
      <div className="results-layout">
        <aside className="results-chat-sidebar">
          <div className="results-chat-messages">
            {messages.map((msg) => (
              <div key={msg.id} className={`results-chat-item ${msg.role}`}>
                {msg.role === 'assistant' && (msg as any).results ? (
                  <ResultCard results={(msg as any).results} sessionId={sessionId} />
                ) : (
                  <span>{msg.content.length > 120 ? msg.content.slice(0, 120) + '…' : msg.content}</span>
                )}
              </div>
            ))}
          </div>
          <div className="results-chat-input">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={t('results.sidebarPlaceholder')}
              rows={2}
            />
            <button onClick={() => send()} disabled={streaming || !input.trim()}>→</button>
          </div>
        </aside>
        <ResultList results={(activeMessage as any).results} onOpenRow={() => {}} />
        <DetailPanel row={null} />
      </div>
    </div>
  )
}
```

（样式类追加到 `styles/app.css`：`.results-page` 全高；`.results-layout` grid 三栏 280px/1fr/1fr；窄栏消息列表滚动。占位的 `ResultList`/`DetailPanel` 在 Task 13/14 中实现——本任务先创建最小占位组件：`export default function ResultList(props: any) { return <div className="results-list-placeholder">…</div> }`、`DetailPanel` 同理，保证 build 通过。）

- [ ] **Step 3: Build check**

Run: `cd frontend/nextjs && npm run build`
Expected: build 成功

- [ ] **Step 4: Commit**

```bash
git add frontend/nextjs/app/app/'(auth)'/results/page.tsx frontend/nextjs/components/app/results/ frontend/nextjs/styles/app.css frontend/nextjs/lib/app-i18n/locales/zh.ts frontend/nextjs/lib/app-i18n/locales/en.ts
git commit -m "feat: results page skeleton with chat sidebar"
```

---

### Task 13: ResultList + ResultRow

**Files:**
- Create: `frontend/nextjs/components/app/results/ResultList.tsx`、`ResultRow.tsx`
- Modify: `frontend/nextjs/app/app/(auth)/results/page.tsx`（选中行状态、`onOpenRow` 接线）
- Modify: `frontend/nextjs/lib/app-i18n/locales/zh.ts`、`en.ts`（排序/筛选文案）

**Interfaces:**
- Produces:
  - `ResultList({ results, activeRowId, onSelect, onOpenTab })` — `results` 为 payload；内部排序（`relevance`=原始顺序/`date`=publication_date/`assignee`）与来源筛选；行点击 = onSelect(rowModel)，按钮点击 = onOpenTab(rowModel, 'details'|'spec'|'claims'|'prosecution')
  - `ResultRow({ model, active, onSelect, onOpenTab })`

- [ ] **Step 1: Add i18n keys**

`locales/zh.ts` 的 `results` 段追加（en.ts 同步）：

```typescript
    sortRelevance: '相关度',
    sortDate: '日期',
    sortAssignee: '申请人',
    filterAll: '全部来源',
    rowDetails: '详情',
    rowSpec: '说明书',
    rowClaims: '权要',
    rowProsecution: '审查历史',
    rowDocView: '查看原文',
    rowDocInfo: '详情',
```

- [ ] **Step 2: Create ResultRow.tsx**

```tsx
'use client'

import { useI18n } from '@/lib/app-i18n'

interface RowModel {
  id: string
  title: string
  meta: Array<{ label: string; value: string }>
  patentId: string
  applicationNumber: string
  url: string
  source: string
  isDocument: boolean
  fields: Array<[string, string]>
}

export default function ResultRow({
  model, active, onSelect, onOpenTab,
}: {
  model: RowModel
  active: boolean
  onSelect: (model: RowModel) => void
  onOpenTab: (model: RowModel, tab: string) => void
}) {
  const { t } = useI18n()

  return (
    <div className={`result-row${active ? ' active' : ''}`} onClick={() => onSelect(model)}>
      <div className="result-row-title">{model.title || '—'}</div>
      <div className="result-row-meta">
        {model.meta.map((item) => (
          <span key={item.label} className="result-row-meta-item">
            {item.label}: {item.value}
          </span>
        ))}
      </div>
      <div className="result-row-actions" onClick={(e) => e.stopPropagation()}>
        {model.isDocument ? (
          <>
            <button onClick={() => onOpenTab(model, 'doc')}>{t('results.rowDocInfo')}</button>
            {model.url && (
              <a href={model.url} target="_blank" rel="noopener noreferrer">
                {t('results.rowDocView')}
              </a>
            )}
          </>
        ) : (
          <>
            <button onClick={() => onOpenTab(model, 'details')}>{t('results.rowDetails')}</button>
            <button onClick={() => onOpenTab(model, 'spec')}>{t('results.rowSpec')}</button>
            <button onClick={() => onOpenTab(model, 'claims')}>{t('results.rowClaims')}</button>
            <button onClick={() => onOpenTab(model, 'prosecution')}>{t('results.rowProsecution')}</button>
          </>
        )}
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Create ResultList.tsx**

```tsx
'use client'

import { useMemo, useState } from 'react'
import { useI18n } from '@/lib/app-i18n'
import { buildRowModel, findRoleColumn } from '@/lib/results'
import ResultRow from './ResultRow'

interface ResultsPayload {
  setId: string
  source: string
  columns: Array<{ key: string; label: string; role: string }>
  rows: Array<Record<string, unknown>>
}

export default function ResultList({
  results, activeRowId, onSelect, onOpenTab,
}: {
  results: ResultsPayload
  activeRowId: string | null
  onSelect: (model: ReturnType<typeof buildRowModel>) => void
  onOpenTab: (model: ReturnType<typeof buildRowModel>, tab: string) => void
}) {
  const { t } = useI18n()
  const [sort, setSort] = useState<'relevance' | 'date' | 'assignee'>('relevance')
  const [sourceFilter, setSourceFilter] = useState<string>('all')

  const models = useMemo(() => {
    const dateCol = findRoleColumn(results.columns, 'publication_date')
    const assigneeCol = findRoleColumn(results.columns, 'assignee')
    const list = results.rows.map((row) => buildRowModel(row, results.columns, results.source))
    if (sort === 'date' && dateCol) {
      list.sort((a, b) => String(b.fields.find(([k]) => k === (dateCol.label || dateCol.key))?.[1] || '').localeCompare(String(a.fields.find(([k]) => k === (dateCol.label || dateCol.key))?.[1] || '')))
    } else if (sort === 'assignee' && assigneeCol) {
      list.sort((a, b) => a.meta.find((m) => m.label === (assigneeCol.label || assigneeCol.key))?.value.localeCompare(b.meta.find((m) => m.label === (assigneeCol.label || assigneeCol.key))?.value || ''))
    }
    if (sourceFilter !== 'all') {
      return list.filter((m) => m.source === sourceFilter)
    }
    return list
  }, [results, sort, sourceFilter])

  const sources = useMemo(() => {
    const set = new Set(models.map((m) => m.source))
    return Array.from(set)
  }, [models])

  return (
    <div className="results-list">
      <div className="results-list-toolbar">
        <span className="results-list-count">{t('results.selectedCount').replace('{count}', String(models.length))}</span>
        <select value={sort} onChange={(e) => setSort(e.target.value as any)} aria-label="sort">
          <option value="relevance">{t('results.sortRelevance')}</option>
          <option value="date">{t('results.sortDate')}</option>
          <option value="assignee">{t('results.sortAssignee')}</option>
        </select>
        <select value={sourceFilter} onChange={(e) => setSourceFilter(e.target.value)} aria-label="source filter">
          <option value="all">{t('results.filterAll')}</option>
          {sources.map((source) => (
            <option key={source} value={source}>{source}</option>
          ))}
        </select>
      </div>
      <div className="results-list-scroll">
        {models.map((model) => (
          <ResultRow
            key={model.id + model.title}
            model={model}
            active={activeRowId === model.id}
            onSelect={onSelect}
            onOpenTab={onOpenTab}
          />
        ))}
      </div>
    </div>
  )
}
```

- [ ] **Step 4: Wire results page state**

`results/page.tsx` 中：

```tsx
  const [activeRowId, setActiveRowId] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<string>('details')
```

替换占位渲染：

```tsx
        <ResultList
          results={(activeMessage as any).results}
          activeRowId={activeRowId}
          onSelect={(model) => { setActiveRowId(model.id); setActiveTab(model.isDocument ? 'doc' : 'details') }}
          onOpenTab={(model, tab) => { setActiveRowId(model.id); setActiveTab(tab) }}
        />
        <DetailPanel row={activeRow} tab={activeTab} onTabChange={setActiveTab} />
```

其中 `activeRow` 从选中行计算：

```tsx
  const activeRow = useMemo(() => {
    if (!activeMessage || !activeRowId) return null
    const results = (activeMessage as any).results
    if (!results) return null
    const row = results.rows.find((r: any) => buildRowModel(r, results.columns, results.source).id === activeRowId)
    return row ? buildRowModel(row, results.columns, results.source) : null
  }, [activeMessage, activeRowId])
```

（import `buildRowModel`）

- [ ] **Step 5: Build check**

Run: `cd frontend/nextjs && npm run build`
Expected: build 成功

- [ ] **Step 6: Commit**

```bash
git add frontend/nextjs/components/app/results/ResultList.tsx frontend/nextjs/components/app/results/ResultRow.tsx frontend/nextjs/app/app/'(auth)'/results/page.tsx frontend/nextjs/lib/app-i18n/locales/zh.ts frontend/nextjs/lib/app-i18n/locales/en.ts
git commit -m "feat: results list with role-driven rows, sort and source filter"
```

---

### Task 14: DetailPanel — 详情 tab + 文档 tab

**Files:**
- Create: `frontend/nextjs/components/app/results/DetailPanel.tsx`、`DocTab.tsx`
- Modify: `frontend/nextjs/lib/app-i18n/locales/zh.ts`、`en.ts`（tab 名文案）

**Interfaces:**
- Produces: `DetailPanel({ row, tab, onTabChange })` — tab 集：`details`（行数据键值卡）、`doc`（文档信息，`row.isDocument` 时）、`spec`/`claims`/`prosecution`（Task 15/16 实现后挂载）

- [ ] **Step 1: Add i18n keys**

`locales/zh.ts` 的 `results` 段追加（en.ts 同步）：

```typescript
    tabDetails: '详情',
    tabSpec: '说明书',
    tabClaims: '权利要求',
    tabProsecution: '审查历史',
    tabDoc: '文档信息',
    fieldTableEmpty: '暂无更多字段',
```

- [ ] **Step 2: Create DetailPanel.tsx + DocTab.tsx**

`DetailPanel.tsx`：

```tsx
'use client'

import { useI18n } from '@/lib/app-i18n'
import DocTab from './DocTab'
import SpecTab from './SpecTab'
import ClaimsTab from './ClaimsTab'
import ProsecutionTab from './ProsecutionTab'

const TAB_ORDER = ['details', 'doc', 'spec', 'claims', 'prosecution']

export default function DetailPanel({
  row, tab, onTabChange,
}: {
  row: any
  tab: string
  onTabChange: (tab: string) => void
}) {
  const { t } = useI18n()
  if (!row) {
    return <div className="results-detail-empty">← {t('results.emptyHint')}</div>
  }
  const availableTabs = TAB_ORDER.filter((key) => {
    if (key === 'doc') return row.isDocument
    if (key === 'details') return !row.isDocument
    if (key === 'prosecution') return !row.isDocument
    return !row.isDocument
  })

  return (
    <div className="results-detail">
      <div className="results-detail-tabs">
        {availableTabs.map((key) => (
          <button
            key={key}
            className={tab === key ? 'active' : ''}
            onClick={() => onTabChange(key)}
          >
            {t(`results.tab${key.charAt(0).toUpperCase() + key.slice(1)}`)}
          </button>
        ))}
      </div>
      <div className="results-detail-body">
        {tab === 'details' && (
          <div className="results-detail-card">
            <h3>{row.title}</h3>
            {row.fields.length > 0 ? (
              <table className="results-field-table">
                <tbody>
                  {row.fields.map(([label, value]: [string, string]) => (
                    <tr key={label}>
                      <td className="results-field-label">{label}</td>
                      <td>{value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p>{t('results.fieldTableEmpty')}</p>
            )}
          </div>
        )}
        {tab === 'doc' && <DocTab row={row} />}
        {tab === 'spec' && <SpecTab row={row} />}
        {tab === 'claims' && <ClaimsTab row={row} />}
        {tab === 'prosecution' && <ProsecutionTab row={row} />}
      </div>
    </div>
  )
}
```

`DocTab.tsx`：

```tsx
'use client'

import { useI18n } from '@/lib/app-i18n'

export default function DocTab({ row }: { row: any }) {
  const { t } = useI18n()
  return (
    <div className="results-detail-card">
      <h3>{row.title}</h3>
      {row.meta.map((item: { label: string; value: string }) => (
        <div key={item.label} className="results-doc-meta">
          <span className="results-field-label">{item.label}</span>
          <span>{item.value}</span>
        </div>
      ))}
      {row.url && (
        <a className="results-doc-link" href={row.url} target="_blank" rel="noopener noreferrer">
          {t('results.rowDocView')}
        </a>
      )}
    </div>
  )
}
```

（SpecTab/ClaimsTab/ProsecutionTab 本任务先创建最小占位：`export default function SpecTab({ row }: { row: any }) { return <div className="results-detail-card">说明书（Task 15 实现）</div> }`，保证 build；Task 15/16 替换为完整实现。）

- [ ] **Step 3: Build check**

Run: `cd frontend/nextjs && npm run build`
Expected: build 成功

- [ ] **Step 4: Commit**

```bash
git add frontend/nextjs/components/app/results/DetailPanel.tsx frontend/nextjs/components/app/results/DocTab.tsx frontend/nextjs/components/app/results/SpecTab.tsx frontend/nextjs/components/app/results/ClaimsTab.tsx frontend/nextjs/components/app/results/ProsecutionTab.tsx frontend/nextjs/lib/app-i18n/locales/zh.ts frontend/nextjs/lib/app-i18n/locales/en.ts
git commit -m "feat: detail panel with tab navigation and document tab"
```

---

### Task 15: SpecTab + ClaimsTab — 按需拉取、缓存、错误重试

**Files:**
- Modify: `frontend/nextjs/components/app/results/SpecTab.tsx`、`ClaimsTab.tsx`（替换占位）
- Modify: `frontend/nextjs/lib/app-i18n/locales/zh.ts`、`en.ts`

**Interfaces:**
- Consumes: Task 8 的 `fetchPatentSpec(source, patentId)`、`fetchPatentClaims(source, patentId)`；row 提供 `patentId`/`applicationNumber`/`source`
- Produces: tab 内 `loading | error | data` 三态；`source_url` 顶部 [PDF 原文] 链接；说明书分段渲染；权利要求列表独立权要高亮 + 一键复制

- [ ] **Step 1: Add i18n keys**

`locales/zh.ts` 的 `results` 段追加（en.ts 同步）：

```typescript
    specLoading: '正在加载说明书…',
    specError: '说明书加载失败',
    specPdfLink: 'PDF 原文',
    claimIndependent: '独立权利要求',
    claimsLoading: '正在加载权利要求…',
    claimsError: '权利要求加载失败',
    claimCopy: '复制',
    retry: '重试',
```

- [ ] **Step 2: Implement SpecTab.tsx**

```tsx
'use client'

import { useEffect, useState } from 'react'
import { useI18n } from '@/lib/app-i18n'
import { fetchPatentSpec, type PatentSpecResponse } from '@/services/api'

export default function SpecTab({ row }: { row: any }) {
  const { t } = useI18n()
  const [state, setState] = useState<'loading' | 'error' | 'data'>('loading')
  const [data, setData] = useState<PatentSpecResponse | null>(null)
  const [retryKey, setRetryKey] = useState(0)

  useEffect(() => {
    let cancelled = false
    setState('loading')
    const identifier = row.patentId || row.applicationNumber
    if (!identifier) {
      setState('error')
      return
    }
    fetchPatentSpec(row.source, identifier)
      .then((payload) => {
        if (cancelled) return
        setData(payload)
        setState('data')
      })
      .catch(() => {
        if (!cancelled) setState('error')
      })
    return () => { cancelled = true }
  }, [row.source, row.patentId, row.applicationNumber, retryKey])

  if (state === 'loading') return <div className="results-detail-card">{t('results.specLoading')}</div>
  if (state === 'error' || !data) {
    return (
      <div className="results-detail-card results-error">
        <p>{t('results.specError')}</p>
        <button onClick={() => setRetryKey((k) => k + 1)}>{t('results.retry')}</button>
      </div>
    )
  }

  return (
    <div className="results-detail-card">
      {data.source_url && (
        <a className="results-spec-pdf" href={data.source_url} target="_blank" rel="noopener noreferrer">
          {t('results.specPdfLink')}
        </a>
      )}
      {data.sections.map((section) => (
        <section key={section.heading} className="results-spec-section">
          <h4>{section.heading}</h4>
          {section.paragraphs.map((para, index) => (
            <p key={index}>{para}</p>
          ))}
        </section>
      ))}
    </div>
  )
}
```

- [ ] **Step 3: Implement ClaimsTab.tsx**

```tsx
'use client'

import { useEffect, useState } from 'react'
import { useI18n } from '@/lib/app-i18n'
import { fetchPatentClaims, type PatentClaimsResponse } from '@/services/api'
import { copyTextToClipboard } from '@/lib/clipboard'

export default function ClaimsTab({ row }: { row: any }) {
  const { t } = useI18n()
  const [state, setState] = useState<'loading' | 'error' | 'data'>('loading')
  const [data, setData] = useState<PatentClaimsResponse | null>(null)
  const [retryKey, setRetryKey] = useState(0)
  const [copiedNumber, setCopiedNumber] = useState<number | null>(null)

  useEffect(() => {
    let cancelled = false
    setState('loading')
    const identifier = row.patentId || row.applicationNumber
    if (!identifier) {
      setState('error')
      return
    }
    fetchPatentClaims(row.source, identifier)
      .then((payload) => {
        if (cancelled) return
        setData(payload)
        setState('data')
      })
      .catch(() => {
        if (!cancelled) setState('error')
      })
    return () => { cancelled = true }
  }, [row.source, row.patentId, row.applicationNumber, retryKey])

  if (state === 'loading') return <div className="results-detail-card">{t('results.claimsLoading')}</div>
  if (state === 'error' || !data) {
    return (
      <div className="results-detail-card results-error">
        <p>{t('results.claimsError')}</p>
        <button onClick={() => setRetryKey((k) => k + 1)}>{t('results.retry')}</button>
      </div>
    )
  }

  async function handleCopy(number: number, text: string) {
    if (await copyTextToClipboard(text)) {
      setCopiedNumber(number)
      setTimeout(() => setCopiedNumber(null), 2000)
    }
  }

  return (
    <div className="results-detail-card">
      {data.claims.map((claim) => (
        <div key={claim.number} className={`results-claim${claim.independent ? ' independent' : ''}`}>
          <div className="results-claim-header">
            <span>{claim.independent ? t('results.claimIndependent') : `#${claim.number}`}</span>
            <button onClick={() => handleCopy(claim.number, claim.text)}>
              {copiedNumber === claim.number ? '✓' : t('results.claimCopy')}
            </button>
          </div>
          <p>{claim.text}</p>
        </div>
      ))}
    </div>
  )
}
```

- [ ] **Step 4: Build check**

Run: `cd frontend/nextjs && npm run build`
Expected: build 成功

- [ ] **Step 5: Commit**

```bash
git add frontend/nextjs/components/app/results/SpecTab.tsx frontend/nextjs/components/app/results/ClaimsTab.tsx frontend/nextjs/lib/app-i18n/locales/zh.ts frontend/nextjs/lib/app-i18n/locales/en.ts
git commit -m "feat: on-demand spec and claims tabs with loading and error states"
```

---

### Task 16: ProsecutionTab — 提交长任务 + 内嵌进度与报告

**Files:**
- Modify: `frontend/nextjs/components/app/results/ProsecutionTab.tsx`（替换占位）
- Modify: `frontend/nextjs/lib/app-i18n/locales/zh.ts`、`en.ts`

**Interfaces:**
- Consumes: Task 8 的 `submitLongTask`；现有 `pollLongTaskBatchStatus`、`getLongTaskReportUrl`、`LongTaskProgress` 组件；`useChatSession().sessionId`
- Produces: 两个入口卡片（🇺🇸 美国审查历史 / 🌐 跨国同族审查历史），美国入口在 `row.applicationNumber` 非 8 位数字时置灰 + tooltip；提交后轮询（1.5s 间隔）并渲染 `LongTaskProgress`；完成后显示报告 + [下载 PDF/Word]；同一专利两次分析独立 taskId 可切换

- [ ] **Step 1: Add i18n keys**

`locales/zh.ts` 的 `results` 段追加（en.ts 同步）：

```typescript
    prosecutionUs: '美国审查历史分析',
    prosecutionFamily: '跨国同族审查历史分析',
    prosecutionUsUnavailable: '需要 8 位美国申请号',
    prosecutionSubmitting: '正在提交分析任务…',
    prosecutionSubmitError: '提交失败，请稍后重试',
    prosecutionRunning: '分析进行中…',
```

- [ ] **Step 2: Implement ProsecutionTab.tsx**

```tsx
'use client'

import { useEffect, useRef, useState } from 'react'
import { useI18n } from '@/lib/app-i18n'
import { submitLongTask, pollLongTaskBatchStatus, getLongTaskReportUrl } from '@/services/api'
import { useChatSession } from '@/contexts/ChatContext'
import LongTaskProgress from '@/components/app/LongTaskProgress'

interface TaskState {
  taskId: string
  kind: 'prosecution' | 'family'
  status: string
  progress: number | null
  currentStep: string
  resultSummary: string | null
  error: string | null
}

export default function ProsecutionTab({ row }: { row: any }) {
  const { t } = useI18n()
  const { sessionId } = useChatSession()
  const [tasks, setTasks] = useState<TaskState[]>([])
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState(false)
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const hasUsAppNumber = /^\d{8}$/.test(row.applicationNumber || '')

  async function handleSubmit(kind: 'prosecution' | 'family') {
    setSubmitting(true)
    setSubmitError(false)
    try {
      const res = await submitLongTask({
        scenario: kind,
        patentId: kind === 'prosecution' ? row.applicationNumber : (row.patentId || row.applicationNumber),
        query: kind === 'prosecution'
          ? `分析专利 ${row.patentId || row.applicationNumber} 的审查历史`
          : `分析 ${row.patentId || row.applicationNumber} 及其全球同族申请的审查差异`,
        lang: 'zh',
        ...(sessionId ? { sessionId } : {}),
      })
      setTasks((prev) => [...prev, {
        taskId: res.task_id, kind, status: res.status, progress: 0,
        currentStep: '', resultSummary: null, error: null,
      }])
      setActiveTaskId(res.task_id)
    } catch {
      setSubmitError(true)
    } finally {
      setSubmitting(false)
    }
  }

  useEffect(() => {
    const openTasks = tasks.filter((task) => !['completed', 'failed'].includes(task.status))
    if (openTasks.length === 0) return

    async function pollAll() {
      const ids = openTasks.map((task) => task.taskId)
      try {
        const batch = await pollLongTaskBatchStatus(ids)
        setTasks((prev) => prev.map((task) => {
          const data = batch[task.taskId]
          if (!data) return task
          if (data.status === 'completed' || data.status === 'success') {
            return { ...task, status: 'completed', progress: 100, resultSummary: data.result_summary || task.resultSummary }
          }
          if (data.status === 'failed' || data.status === 'error') {
            return { ...task, status: 'failed', error: data.error_message || 'failed' }
          }
          return { ...task, status: data.status, progress: data.progress ?? task.progress, currentStep: data.current_step || task.currentStep }
        }))
      } catch {
        // Transient poll error — keep polling
      }
    }

    pollAll()
    pollTimerRef.current = setInterval(pollAll, 1500)
    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current)
      pollTimerRef.current = null
    }
  }, [tasks.map((task) => task.taskId + task.status).join(',')])

  const activeTask = tasks.find((task) => task.taskId === activeTaskId) || tasks[tasks.length - 1]

  return (
    <div className="results-detail-card">
      <div className="results-prosecution-entries">
        <button
          className="results-prosecution-entry"
          disabled={!hasUsAppNumber || submitting}
          title={hasUsAppNumber ? '' : t('results.prosecutionUsUnavailable')}
          onClick={() => handleSubmit('prosecution')}
        >
          🇺🇸 {t('results.prosecutionUs')}
        </button>
        <button
          className="results-prosecution-entry"
          disabled={submitting}
          onClick={() => handleSubmit('family')}
        >
          🌐 {t('results.prosecutionFamily')}
        </button>
      </div>

      {submitting && <p>{t('results.prosecutionSubmitting')}</p>}
      {submitError && <p className="results-error-text">{t('results.prosecutionSubmitError')}</p>}

      {tasks.length > 1 && (
        <div className="results-prosecution-switch">
          {tasks.map((task) => (
            <button
              key={task.taskId}
              className={activeTask?.taskId === task.taskId ? 'active' : ''}
              onClick={() => setActiveTaskId(task.taskId)}
            >
              {task.kind === 'prosecution' ? '🇺🇸' : '🌐'} {task.status}
            </button>
          ))}
        </div>
      )}

      {activeTask && (
        <div className="results-prosecution-task">
          {activeTask.status === 'failed' ? (
            <p className="results-error-text">{activeTask.error || t('results.prosecutionSubmitError')}</p>
          ) : activeTask.status === 'completed' ? (
            <div>
              {activeTask.resultSummary && (
                <LongTaskProgress
                  content={activeTask.resultSummary}
                  resultSummary={activeTask.resultSummary}
                  streaming={false}
                />
              )}
              <div className="results-prosecution-downloads">
                <a href={getLongTaskReportUrl(activeTask.taskId, 'pdf')} download>PDF</a>
                <a href={getLongTaskReportUrl(activeTask.taskId, 'docx')} download>Word</a>
              </div>
            </div>
          ) : (
            <LongTaskProgress
              content={`${t('results.prosecutionRunning')} [${activeTask.progress ?? 0}%] ${activeTask.currentStep} Task ID: ${activeTask.taskId}`}
              streaming={true}
            />
          )}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 3: Build check**

Run: `cd frontend/nextjs && npm run build`
Expected: build 成功（`LongTaskProgress` 的 Props 已核验：`{content: string, resultSummary?, streaming, analysisType?, tableColumns?, familyOverview?, jurisdictions?}`——上文用法一致）

- [ ] **Step 4: Commit**

```bash
git add frontend/nextjs/components/app/results/ProsecutionTab.tsx frontend/nextjs/lib/app-i18n/locales/zh.ts frontend/nextjs/lib/app-i18n/locales/en.ts
git commit -m "feat: prosecution analysis tab with long task submit and progress"
```

---

### Task 17: 端到端验证清单 + 收尾

**Files:** 无新代码（验证任务；若发现 bug 则按 fix: 提交修复）

- [ ] **Step 1: Backend full test run（服务器）**

```bash
pytest tests/ -v
```

Expected: 全部通过（含 Task 1-5 新增的测试与既有测试）

- [ ] **Step 2: Frontend full test + build**

```bash
cd frontend/nextjs && node --test lib/ && npm run build
```

Expected: 全部通过

- [ ] **Step 3: 手动端到端验证（对照规格 §4/§6/§7）**

1. 对话页提问触发检索（USPTO 工具，如"检索 Tesla 的电池相关专利"）→ 聊天里出现结果卡片（N 条 + [在结果页查看] + 原有下载按钮仍在）
2. 点击 [在结果页查看] → 结果页三栏打开，列表显示行卡片（标题/元信息/四按钮）
3. 点"详情" → 右栏著录项键值表（零请求）
4. 点"说明书" → 骨架屏 → 分段全文；[PDF 原文] 打开 Google Patents 页
5. 点"权要" → 权利要求列表、独立权要高亮、复制按钮
6. 点"审查历史" → 🇺🇸 亮（8 位申请号行）→ 提交 → 内嵌进度 → 完成 → 报告 + PDF/Word 下载；非美行 🇺🇸 置灰 + tooltip，🌐 可用
7. 侧边栏继续追问"再检索 XX 的专利" → 新结果卡片出现在侧边栏 → 列表切换到新结果集；点旧卡片切回
8. 刷新页面（带 session_id）→ 结果页恢复；无 session 直接访问 `/app/results` → 空态引导
9. 旧会话（修复前产生、无 results 字段）→ 聊天正常显示下载按钮，无卡片（回退行为）
10. USPTO 文档检索工具 → 文档行形态（标题/日期/[查看原文]）

- [ ] **Step 4: 提交收尾 commit（若验证中发现并修复了问题）**

```bash
git add <fixed files>
git commit -m "fix: <描述>"
```

- [ ] **Step 5: 合并与部署（用户确认后）**

```bash
git checkout main && git merge --no-ff feature/search-results-split-view
# 服务器: git pull origin main && docker compose --profile backend up -d --build
```

---

## Self-Review（对规格的覆盖检查）

- 规格 §2 数据流全景 → Task 1/2（json artifact + source）、Task 5（路由注册）、Task 6（前端解码）、Task 12（结果页）✅
- §3 传输格式/列角色 → Task 1（封闭角色集 + 阈值）、Task 7（role→行模型）✅
- §4.1 列表（排序/筛选/按钮态/高亮）→ Task 13 ✅
- §4.2 详情面板四 tab → Task 14/15/16 ✅（详情零接口、说明书/权要按需、审查历史双入口 + 8 位号置灰 + 独立 task 切换）
- §4.3 文档行形态 → Task 7（isDocument）+ Task 13（文档按钮）+ Task 14（DocTab）✅
- §4.4 双向联动 → Task 10（hook 复用）+ Task 12（侧边栏 + 结果集切换）✅
- §5 API 契约 → Task 3（spec/claims 端点——实现为 Google Patents 后端，契约与规格一致）、Task 4（submit 端点）✅
- §6 错误处理矩阵 → Task 15（加载/错误/重试/PDF 兜底）、Task 16（提交失败/队列）、Task 6（解析失败回退）✅
- §7 持久化 → Task 7（prune）+ Task 11（saveSessionMessages results + 恢复水合）✅
- §10 v1 范围外 → 未引入 CNIPA/后端分页/AI 翻译/分享 ✅
- 与规格的**有意偏差**：说明书/权要端点在 uspto 源下也走 Google Patents（USPTO PDF 为扫描件，需 vision LLM 才能抽文本，不适合按需端点）——端点契约与错误语义不变，已在 Task 3 代码注释中说明


