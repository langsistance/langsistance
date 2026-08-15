# 聊天路径专利检索 + 自主下载分析 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让聊天 ReAct 路径的专利检索类查询具备通用质量：搜索观察携带真实条目摘要、关键词检索确定性改写、内置 `fetch_patent_spec` 工具让 agent 自主下载说明书全文并提炼分析。

**Architecture:** 三件独立单元：① 从 `celery_worker.py` 抽取 USPTO 说明书下载实现为共享模块（长任务与聊天路径共用同一实现）；② 新增 `patent_distill.py`（全文提炼 + 16k 截断降级）；③ `react_tools.py` 增加 `_items_digest` 观察增强、确定性改写接入（复用 Phase 1 `search_query_builder`）、`fetch_patent_spec` 常驻工具与执行分支。

**Tech Stack:** Python 3.14（async/await）、unittest + `IsolatedAsyncioTestCase` + `AsyncMock`、LangChain `StructuredTool`、既有 `Provider.complete_json(system_prompt, user_content, max_retries=2)`、USPTO API（api.uspto.gov）。

## Global Constraints

- Python 3.14：禁用 `asyncio.get_event_loop()`；异步测试用 `IsolatedAsyncioTestCase`。
- 测试运行：`cd E:\online\workspace\copiioai\langsistance && PYTHONUTF8=1 python -m pytest tests/<file> -v`；全量：`PYTHONUTF8=1 python -m pytest tests/ -q --continue-on-collection-errors`。用系统 python（C:/Python314），venv 未装 pytest；`PYTHONUTF8=1` 必带。
- 全量基线（本环境系统 python）：**369 passed / 26 failed / 9 errors / 1 warning**——26+9 为预存环境问题（缺 ollama/markdownify 等），验收标准 = 零新增失败，失败数不变。
- **通用性（D5）**：任何模块不得为特定查询/技术领域硬编码关键词、阈值或分支；改写与提炼输入均为用户原始问题。
- 所有新 LLM 调用均 try/except 包裹，失败必须降级到 spec 第 8 节规定的行为，循环永不因新功能崩溃。
- 新模块只依赖标准库 + 既有 `sources` 模块；零新第三方依赖。
- 提交规范：`feat:` / `fix:`，无 attribution trailer。
- 分支：从当前 HEAD 拉出 `feature/chat-patent-analysis`。
- 前端零改动（步骤条/时间线已实现待部署；本计划不涉及 frontend/）。

## File Structure

| 文件 | 职责 |
|---|---|
| `sources/long_task/uspto_download.py` | **新建**。从 celery_worker 抽取的 USPTO 说明书下载（共享模块） |
| `sources/long_task/patent_distill.py` | **新建**。全文提炼 + 16k 截断降级 + 观察格式化 |
| `sources/agents/react_tools.py` | **修改**。`_items_digest`、`_maybe_rewrite_search_query`、`fetch_patent_spec` 注册与执行分支 |
| `sources/agents/general_agent.py` | **修改**。`create_agent` 状态重置加一行 `self._search_rewrite = None` |
| `celery_worker.py` | **修改**。删除被抽取函数，改为 import 共享模块（行为逐字节不变） |
| `tests/test_uspto_download.py` | **新建** |
| `tests/test_patent_distill.py` | **新建** |
| `tests/test_react_tools.py` | **修改**（追加 3 组新测试） |

---

### Task 1: 抽取共享下载模块 `uspto_download.py`

**Files:**
- Create: `sources/long_task/uspto_download.py`
- Modify: `celery_worker.py`（删除被抽取函数 + 改 import）
- Test: `tests/test_uspto_download.py`

**Interfaces:**
- Produces（Task 5 依赖）:
  - `async def download_uspto_patent_text(patent_id: str, spec_selector_provider=None, logger=None) -> tuple[str | None, bytes | None]` — `(text, None)` 提取成功；`(None, binary)` 文本失败但缓存二进制；`(None, None)` 完全失败。**永不抛异常。**

- [ ] **Step 1: 写失败测试**

创建 `tests/test_uspto_download.py`：

```python
"""Tests for uspto_download — shared USPTO spec download module."""
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from sources.long_task.uspto_download import (
    download_uspto_patent_text,
    normalize_app_number,
)


class TestNormalizeAppNumber(unittest.TestCase):
    def test_strips_us_prefix_and_punctuation(self):
        self.assertEqual(normalize_app_number("US 19/511,555"), "19511555")
        self.assertEqual(normalize_app_number("19511555"), "19511555")

    def test_too_short_returns_empty(self):
        self.assertEqual(normalize_app_number("12345"), "")
        self.assertEqual(normalize_app_number(""), "")


class TestDownloadUsptoPatentText(unittest.IsolatedAsyncioTestCase):
    async def test_no_documents_returns_none_none(self):
        with patch("sources.long_task.uspto_download._uspto_get_with_retry",
                   new=AsyncMock(return_value=_resp(200, {"documentBag": []}))):
            text, binary = await download_uspto_patent_text("19511555")
        self.assertEqual((text, binary), (None, None))

    async def test_spec_text_extracted(self):
        doc_list = {"documentBag": [{
            "documentCode": "SPEC",
            "downloadOptionBag": [{
                "downloadUrl": "https://api.uspto.gov/api/v1/download/applications/19511555/x/doc.docx",
                "mimeTypeIdentifier": "MS_WORD",
            }],
        }]}
        with patch("sources.long_task.uspto_download._uspto_get_with_retry",
                   new=AsyncMock(return_value=_resp(200, doc_list))):
            with patch("sources.long_task.uspto_download._download_uspto_spec_with_redirect",
                       new=AsyncMock(return_value=("SPEC TEXT BODY", None))):
                text, binary = await download_uspto_patent_text("19511555")
        self.assertIn("SPEC TEXT BODY", text)
        self.assertIsNone(binary)

    async def test_binary_fallback_when_text_extraction_empty(self):
        doc_list = {"documentBag": [{
            "documentCode": "SPEC",
            "downloadOptionBag": [{
                "downloadUrl": "https://api.uspto.gov/api/v1/download/applications/19511555/x.pdf",
                "mimeTypeIdentifier": "PDF",
            }],
        }]}
        with patch("sources.long_task.uspto_download._uspto_get_with_retry",
                   new=AsyncMock(return_value=_resp(200, doc_list))):
            with patch("sources.long_task.uspto_download._download_uspto_spec_with_redirect",
                       new=AsyncMock(return_value=(None, b"PDFBYTES"))):
                text, binary = await download_uspto_patent_text("19511555")
        self.assertIsNone(text)
        self.assertEqual(binary, b"PDFBYTES")

    async def test_invalid_app_number_returns_none_none(self):
        text, binary = await download_uspto_patent_text("abc")
        self.assertEqual((text, binary), (None, None))

    async def test_internal_error_never_raises(self):
        with patch("sources.long_task.uspto_download._uspto_get_with_retry",
                   new=AsyncMock(side_effect=RuntimeError("boom"))):
            text, binary = await download_uspto_patent_text("19511555")
        self.assertEqual((text, binary), (None, None))


def _resp(status, data):
    resp = MagicMock()
    resp.status_code = status
    resp.text = str(data) if not isinstance(data, dict) else ""
    resp.content = b""
    if isinstance(data, dict):
        resp.json = lambda: data
    return resp


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_uspto_download.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 创建共享模块（原文搬移 + 四处手术）**

创建 `sources/long_task/uspto_download.py`。**搬移来源**（celery_worker.py，逐字复制函数体，不重写逻辑）：

| 来源（celery_worker.py） | 搬到共享模块后的形态 |
|---|---|
| `_download_uspto_patent_direct`（:5298-5494） | 公开函数 `download_uspto_patent_text(patent_id, spec_selector_provider=None, logger=None)` |
| `_uspto_get_with_retry`（:5998-6010） | 保持同名内部函数 |
| `_download_uspto_spec_with_redirect`（:6013-约 6120 至该函数结束） | 保持同名内部函数 |
| `_guess_format_from_url`（:5597-5606） | 保持同名内部函数 |

模块骨架（手术点已标注）：

```python
"""USPTO specification download shared by the long-task pipeline and the
ReAct chat loop.  Extracted verbatim from celery_worker.py so both paths
use the same implementation; behavior is byte-identical to the original.
"""

import os
from typing import Any

from sources.logger import Logger

from sources.long_task.text_extractor import (
    extract_text_from_binary,
    get_download_url_from_doc,
)

_get_download_url_from_doc = get_download_url_from_doc

_DEFAULT_LOGGER = Logger("uspto_download.log")


def normalize_app_number(patent_id: str) -> str:
    """Strip commas, slashes, non-digits; return '' when too short."""
    app_number = (patent_id or "").strip().replace(",", "").replace("/", "")
    app_number = "".join(c for c in app_number if c.isdigit())
    if len(app_number) < 8:
        return ""
    return app_number


async def download_uspto_patent_text(
    patent_id: str,
    spec_selector_provider=None,
    logger: Any = None,
) -> tuple[str | None, bytes | None]:
    """Download USPTO specification text directly (two-step).

    Step 1: GET /api/v1/patent/applications/{appNumber}/documents
    Step 2: collect SPEC docs, LLM may pick preferred, download all,
            concatenate extracted text.

    Returns (text, binary):
      - (text, None)          — text extracted successfully
      - (None, binary_bytes)  — all specs failed text extraction, binary cached
      - (None, None)          — download failed entirely

    Never raises — all failures degrade to (None, None).
    """
    # 【手术 1】原函数体整体移入，把模块级 _pipeline_logger 替换为局部 _log：
    #   def _log(msg): (logger or _DEFAULT_LOGGER).info(msg)
    #   def _warn(msg): (logger or _DEFAULT_LOGGER).warning(msg)
    # 原函数体中：
    #   _pipeline_logger.info(...)  → _log(...)
    #   _pipeline_logger.warning(...) → _warn(...)
    # 【手术 2】app_number 规范化改用 normalize_app_number()：
    #   app_number = normalize_app_number(patent_id)
    #   if not app_number: _warn(...); return (None, None)
    # 【手术 3】SPEC 选择段的 flash_provider 参数名 → spec_selector_provider
    # 【手术 4】其余逻辑（documents 拉取、SPEC 收集、LLM 选择、多 SPEC
    #   下载拼接、binary fallback、最外层 try/except）逐字保留
    pass  # ← 以原函数体替换此占位
```

说明：本任务为**逐字搬移**，以 celery_worker.py 当前代码为准（实施时先 Read 原函数再搬移），上面标注的是四类机械替换点；搬移后新模块必须通过 Step 1 的全部测试。

- [ ] **Step 4: 修改 celery_worker.py（删 + 改 import）**

4a. 在 `_download_patent_via_scene_or_fallback` 的 USPTO 直连调用处（约 :5273）：

```python
    # ── Step 2: Direct USPTO API for US patents ──
    if patent_source == 'uspto':
        from sources.long_task.uspto_download import download_uspto_patent_text
        uspto_text, uspto_binary = await download_uspto_patent_text(
            patent_id, flash_provider, _pipeline_logger,
        )
```

替换原 `uspto_text, uspto_binary = await _download_uspto_patent_direct(patent_id, flash_provider)` 一行。

4b. 删除 celery_worker.py 中被搬移的函数：`_download_uspto_patent_direct`（:5298-5494）、`_uspto_get_with_retry`、`_download_uspto_spec_with_redirect`、`_guess_format_from_url`。**删除前先 `Grep` 每个名字在 celery_worker.py 的剩余引用**：若某函数还有其他调用点（如 prosecution 下载路径），保留一个转发别名（`_name = uspto_download._name`）而不是删除；若零引用则直接删。把检查结果写进报告。

- [ ] **Step 5: 运行测试确认通过 + 全量回归**

Run: `PYTHONUTF8=1 python -m pytest tests/test_uspto_download.py -v && PYTHONUTF8=1 python -m pytest tests/ -q --continue-on-collection-errors 2>&1 | tail -2`
Expected: 新测试 7 passed；全量 369 + 7 = 376 passed / 26 failed / 9 errors（失败数与基线完全一致，零新增）

- [ ] **Step 6: Commit**

```bash
git add sources/long_task/uspto_download.py celery_worker.py tests/test_uspto_download.py
git commit -m "feat: extract USPTO spec download into shared module for long task and chat paths"
```

---

### Task 2: 说明书提炼模块 `patent_distill.py`

**Files:**
- Create: `sources/long_task/patent_distill.py`
- Test: `tests/test_patent_distill.py`

**Interfaces:**
- Consumes: `Provider.complete_json(system_prompt, user_content, max_retries=2)`（async）
- Produces（Task 5 依赖）:
  - `async def distill_patent_spec(text: str, query: str, provider) -> dict` — 失败返回 `{}`，永不抛异常
  - `def format_distilled(distilled: dict, lang: str = "zh") -> str`
  - `def truncated_fallback(text: str, limit: int = 16000) -> str`
  - 常量 `SPEC_FALLBACK_LIMIT = 16000`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_patent_distill.py`：

```python
"""Tests for patent_distill — spec distillation into loop observations."""
import unittest

from sources.long_task.patent_distill import (
    SPEC_FALLBACK_LIMIT,
    distill_patent_spec,
    format_distilled,
    truncated_fallback,
)


class _FakeProvider:
    def __init__(self, response):
        self._response = response
        self.calls = []

    async def complete_json(self, system, user):
        self.calls.append(user)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class TestTruncatedFallback(unittest.TestCase):
    def test_caps_at_16000(self):
        text = truncated_fallback("x" * 50000)
        self.assertEqual(len(text), SPEC_FALLBACK_LIMIT)

    def test_short_text_untouched(self):
        self.assertEqual(truncated_fallback("short"), "short")


class TestFormatDistilled(unittest.TestCase):
    def test_renders_all_sections(self):
        text = format_distilled({
            "发明点": "A", "解决的技术问题": "B", "技术方案": "C",
            "权利要求要点": "D", "与用户问题的相关性": "E",
        })
        for label in ("发明点", "解决的技术问题", "技术方案",
                      "权利要求要点", "与用户问题的相关性"):
            self.assertIn(label, text)

    def test_missing_keys_skipped(self):
        text = format_distilled({"发明点": "A"})
        self.assertIn("发明点", text)
        self.assertNotIn("技术方案", text)

    def test_empty_distilled_returns_empty(self):
        self.assertEqual(format_distilled({}), "")


class TestDistillPatentSpec(unittest.IsolatedAsyncioTestCase):
    async def test_returns_structured_result(self):
        provider = _FakeProvider({
            "发明点": "x", "解决的技术问题": "y", "技术方案": "z",
        })
        result = await distill_patent_spec("SPEC TEXT", "查询", provider)
        self.assertEqual(result["发明点"], "x")

    async def test_provider_failure_returns_empty(self):
        result = await distill_patent_spec(
            "SPEC TEXT", "查询", _FakeProvider(RuntimeError("down")))
        self.assertEqual(result, {})

    async def test_garbage_response_returns_empty(self):
        result = await distill_patent_spec("SPEC TEXT", "查询", _FakeProvider("junk"))
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_patent_distill.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 写实现**

创建 `sources/long_task/patent_distill.py`：

```python
"""Distill a downloaded patent specification into a bounded observation.

The chat loop cannot hold a 20k-200k char specification in context, so the
full text is distilled by an LLM into the standard patent analysis
dimensions (invention point / technical problem / solution / claim gist /
relevance to the user's question).  On any failure the caller falls back
to the truncated full text.
"""

from typing import Any

SPEC_FALLBACK_LIMIT = 16000

DISTILL_SYSTEM_PROMPT = (
    "你是一个专利分析专家。阅读给定的专利说明书全文，提炼出结构化要点，"
    "用于回答用户的专利分析问题。\n\n"
    "要求：\n"
    "1. 每个要点用中文简明表述，聚焦技术实质，不复制原文段落\n"
    "2. 「权利要求要点」提炼独立权利要求的保护范围要点（最多 5 条）\n"
    "3. 「与用户问题的相关性」明确指出该专利与用户问题相关的技术点，"
    "或说明不相关\n"
    "4. 不编造内容——只依据给定的说明书文本\n\n"
    'Return JSON: {"发明点": "...", "解决的技术问题": "...", '
    '"技术方案": "...", "权利要求要点": "...", '
    '"与用户问题的相关性": "..."}'
)


def truncated_fallback(text: str, limit: int = SPEC_FALLBACK_LIMIT) -> str:
    """Return the first *limit* chars of the spec text (fallback observation)."""
    text = str(text or "")
    return text[:limit]


def format_distilled(distilled: dict, lang: str = "zh") -> str:
    """Render a distilled dict into the observation text for the loop."""
    if not isinstance(distilled, dict) or not distilled:
        return ""
    labels = ("发明点", "解决的技术问题", "技术方案",
              "权利要求要点", "与用户问题的相关性")
    lines = []
    for label in labels:
        value = distilled.get(label)
        if isinstance(value, str) and value.strip():
            lines.append(f"**{label}**：{value.strip()}")
    return "\n\n".join(lines)


async def distill_patent_spec(text: str, query: str, provider: Any) -> dict:
    """Distill a spec via the LLM.  Never raises — returns {} on failure."""
    if not text:
        return {}
    user_content = f"用户问题：{query}\n\n专利说明书全文：\n{text}"
    try:
        result = await provider.complete_json(DISTILL_SYSTEM_PROMPT, user_content)
    except Exception:
        return {}
    if not isinstance(result, dict):
        return {}
    return result
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_patent_distill.py -v`
Expected: PASS（8 passed）

- [ ] **Step 5: Commit**

```bash
git add sources/long_task/patent_distill.py tests/test_patent_distill.py
git commit -m "feat: add patent spec distillation with truncated fallback for chat loop"
```

---

### Task 3: 搜索观察内容增强 `_items_digest`

**Files:**
- Modify: `sources/agents/react_tools.py`（新增 `_items_digest` + 执行器 knowledge 分支观察改造）
- Test: `tests/test_react_tools.py`（追加测试类）

**Interfaces:**
- Consumes: Task 无（Task 2 之前）；`sources.long_task.candidate_metadata.build_candidates(raw_items) -> list[dict]`
- Produces: `_items_digest(raw_items, limit=20, lang="zh") -> str`（纯函数，Task 3/5 内部使用）

- [ ] **Step 1: 写失败测试**

在 `tests/test_react_tools.py` 末尾追加：

```python
# ── Search observation digest ────────────────────────────────────────────────

from sources.agents.react_tools import _items_digest


def _usp_raw_item(app_number, title, applicant="ACME Corp", filing="2024-01-15"):
    return {
        "applicationMetaData": {
            "applicationNumberText": app_number,
            "inventionTitle": title,
            "firstApplicantName": applicant,
            "filingDate": filing,
            "applicationStatusDescriptionText": "Patented Case",
        },
    }


class TestItemsDigest(unittest.TestCase):
    def test_formats_usp_items(self):
        items = [_usp_raw_item("19511555", "Air dryer humidity control",
                               applicant="New York Air Brake")]
        text = _items_digest(items)
        self.assertIn("19511555", text)
        self.assertIn("Air dryer humidity control", text)
        self.assertIn("New York Air Brake", text)

    def test_caps_at_20_with_total_note(self):
        items = [_usp_raw_item(str(19500000 + i), f"Title {i}") for i in range(30)]
        text = _items_digest(items)
        self.assertNotIn("Title 20", text)  # 21st item excluded
        self.assertIn("共 30 条", text)

    def test_non_usp_falls_back_to_truncated_json(self):
        text = _items_digest([{"patentNumber": "US10150077B2",
                               "inventionTitle": "Air dryer"}])
        self.assertIn("US10150077B2", text)

    def test_empty_returns_empty(self):
        self.assertEqual(_items_digest([]), "")
        self.assertEqual(_items_digest(None), "")


class TestSearchObservationContent(unittest.TestCase):
    """Search results observation carries real items, not just counts."""

    def test_executor_returns_digest_for_raw_items(self):
        agent = _FakeAgent()
        entry_k = _Knowledge(3, ktype=1)
        agent.get_dynamic_tool_for = _make_tool_with_pending(agent)
        registry, tools = asyncio.run(
            _registry_with_one_knowledge(agent, entry_k))
        executor = asyncio.run(make_action_executor(agent, registry, None))
        agent._pending_raw_items = [
            _usp_raw_item("19511555", "Air dryer humidity control"),
            _usp_raw_item("18184836", "Moisture control enclosure"),
        ]
        result = asyncio.run(executor("uspto_search", {"params": "{}"}, 1))
        self.assertEqual(result["kind"], "observation")
        self.assertIn("19511555", result["text"])
        self.assertIn("Air dryer humidity control", result["text"])
        self.assertIn("完整列表已展示", result["text"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_tools.py -v`
Expected: FAIL（ImportError: cannot import name '_items_digest'）

- [ ] **Step 3: 写实现**

3a. `react_tools.py` 顶部 import 区追加：

```python
from sources.long_task.candidate_metadata import build_candidates
```

3b. `react_tools.py` 中 `_cap_patent_list` 之前新增：

```python
SEARCH_DIGEST_LIMIT = 20
SEARCH_DIGEST_CHARS = 3000


def _items_digest(raw_items, limit: int = SEARCH_DIGEST_LIMIT,
                  lang: str = "zh") -> str:
    """Serialize search raw_items into a bounded digest for the LLM.

    USPTO-shaped items are flattened via build_candidates into
    ``申请号 | 标题 | 申请人 | 申请日 | 状态`` lines.  Non-USPTO shapes
    fall back to a truncated JSON dump.
    """
    items = raw_items or []
    if not items:
        return ""
    candidates = build_candidates(items)
    if candidates:
        lines = []
        for c in candidates[:limit]:
            parts = [
                c.get("patent_id") or "?",
                c.get("title") or "(无标题)",
                c.get("applicant") or "?",
                c.get("filing_date") or "?",
                c.get("status") or "?",
            ]
            lines.append(" | ".join(str(p) for p in parts))
        text = "\n".join(lines)
        if len(candidates) > limit:
            note = (f"\n…共 {len(candidates)} 条" if lang == "zh"
                    else f"\n...{len(candidates)} items total")
            text += note
        return text[:SEARCH_DIGEST_CHARS]
    import json
    try:
        dumped = json.dumps(items, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        dumped = str(items)
    return dumped[:SEARCH_DIGEST_CHARS]
```

3c. `make_action_executor` knowledge 分支（`entry.kind == "knowledge"` 的 invoke 之后）——把现有 pending 分支的观察文本改为摘要。当前代码块：

```python
        pending = getattr(agent, "_pending_raw_items", None)
        if pending:
            capped, note = _cap_patent_list(entry.tool_info, pending, lang)
            agent._pending_raw_items = capped
            if lang == "en":
                text = f"Tool returned {len(capped)} record(s) ({note}); the full list is displayed afterwards."
            else:
                text = f"工具返回 {len(capped)} 条记录（{note}），完整列表稍后展示。"
            return {"kind": "observation", "text": text}
```

替换为：

```python
        pending = getattr(agent, "_pending_raw_items", None)
        if pending:
            capped, note = _cap_patent_list(entry.tool_info, pending, lang)
            agent._pending_raw_items = capped
            digest = _items_digest(capped, lang=lang)
            if lang == "en":
                text = (f"Search results ({len(capped)} records, {note}):\n"
                        f"{digest}\n\n"
                        "The full list is displayed to the user.")
            else:
                text = (f"检索结果（{len(capped)} 条，{note}）：\n"
                        f"{digest}\n\n"
                        "完整列表已展示给用户。")
            return {"kind": "observation", "text": text}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_tools.py -v`
Expected: PASS（既有 12 个 + 新增 5 个 = 17 passed；既有测试中 `test_knowledge_action_sets_pending_and_caps` 断言 `self.assertIn("已截断", result["text"])` 仍成立——新文本保留了截断说明）

- [ ] **Step 5: Commit**

```bash
git add sources/agents/react_tools.py tests/test_react_tools.py
git commit -m "feat: feed real search item digests into ReAct loop observations"
```

---

### Task 4: 确定性查询改写接入

**Files:**
- Modify: `sources/agents/react_tools.py`（新增 `_maybe_rewrite_search_query` + 执行器调用点）
- Modify: `sources/agents/general_agent.py`（`create_agent` 状态重置加一行）
- Test: `tests/test_react_tools.py`（追加测试类）

**Interfaces:**
- Consumes: Task 1 无依赖；`sources.long_task.search_query_builder.build_search_queries(query, provider) -> {"queries": [...]}`（Phase 1 已交付模块）；`sources.long_task.candidate_metadata.is_keyword_search_tool(tool)`
- Produces: `async def _maybe_rewrite_search_query(agent, tool_info, args) -> dict`（内部）；`agent._search_rewrite` 属性（None 或改写结果 dict）

- [ ] **Step 1: 写失败测试**

在 `tests/test_react_tools.py` 末尾追加：

```python
# ── Deterministic search query rewriting ─────────────────────────────────────

from sources.agents.react_tools import _maybe_rewrite_search_query


class _RewriteAgent(_FakeAgent):
    def __init__(self):
        super().__init__()
        self._last_user_prompt = "工业在线干燥空气源提供与湿度控制"
        self._search_rewrite = None

        class _LLM:
            def __init__(self):
                self.calls = 0

            async def complete_json(self, system, user):
                self.calls += 1
                return {"queries": [
                    '("compressed air dryer" OR "air dryer") AND ("humidity control" OR "dew point")',
                ]}
        self.llm = _LLM()


class TestMaybeRewriteSearchQuery(unittest.IsolatedAsyncioTestCase):
    async def test_rewrites_q_key(self):
        agent = _RewriteAgent()
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"q": "raw noise patent"})
        self.assertIn('"compressed air dryer"', out["q"])

    async def test_rewrites_query_key(self):
        agent = _RewriteAgent()
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"query": "raw"})
        self.assertIn('"air dryer"', out["query"])

    async def test_rewrites_params_json_string(self):
        agent = _RewriteAgent()
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"),
            {"params": '{"q": "raw", "pagination": {"offset": 0, "limit": 50}}'})
        import json
        parsed = json.loads(out["params"])
        self.assertIn('"compressed air dryer"', parsed["q"])
        self.assertEqual(parsed["pagination"]["limit"], 50)

    async def test_cached_across_calls(self):
        agent = _RewriteAgent()
        await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"q": "a"})
        await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"q": "b"})
        self.assertEqual(agent.llm.calls, 1)

    async def test_skips_non_keyword_tools(self):
        agent = _RewriteAgent()
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("get_patent_documents_application_number"), {"q": "raw"})
        self.assertEqual(out, {"q": "raw"})

    async def test_rewrite_failure_keeps_original_args(self):
        class _FailingLLM:
            async def complete_json(self, system, user):
                raise RuntimeError("down")
        agent = _RewriteAgent()
        agent.llm = _FailingLLM()
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"q": "raw"})
        self.assertEqual(out, {"q": "raw"})

    async def test_empty_queries_keep_original_args(self):
        class _EmptyLLM:
            async def complete_json(self, system, user):
                return {"queries": []}
        agent = _RewriteAgent()
        agent.llm = _EmptyLLM()
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"q": "raw"})
        self.assertEqual(out, {"q": "raw"})

    async def test_args_without_query_key_unchanged(self):
        agent = _RewriteAgent()
        out = await _maybe_rewrite_search_query(
            agent, _ToolInfo("search_patent_by_key_word"), {"other": "x"})
        self.assertEqual(out, {"other": "x"})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_tools.py -v`
Expected: FAIL（ImportError: cannot import name '_maybe_rewrite_search_query'）

- [ ] **Step 3: 写实现**

3a. `react_tools.py` import 区追加：

```python
from sources.long_task.candidate_metadata import (
    build_candidates,
    is_keyword_search_tool,
)
```

3b. `make_action_executor` 中 `execute_action` 的 knowledge 分支，`entry.tool.invoke` 之前改写 args。当前代码：

```python
        try:
            result = await asyncio.to_thread(entry.tool.invoke, args)
        except Exception as exc:
            return {"kind": "observation", "text": f"Error: {exc}"}
```

替换为：

```python
        try:
            args = await _maybe_rewrite_search_query(agent, entry.tool_info, args)
            result = await asyncio.to_thread(entry.tool.invoke, args)
        except Exception as exc:
            return {"kind": "observation", "text": f"Error: {exc}"}
```

3c. `react_tools.py` 中 `make_action_executor` 之前新增：

```python
async def _maybe_rewrite_search_query(agent, tool_info, args) -> dict:
    """Deterministically rewrite the q of keyword-search tool calls.

    Uses the user's ORIGINAL question (agent._last_user_prompt) — not the
    LLM's possibly garbled q — via the shared search_query_builder.  The
    result is cached on the agent for the whole loop run.  Applies only to
    backend (push=2) keyword search tools; every failure keeps the original
    args untouched.
    """
    if getattr(tool_info, "push", None) != 2 or not is_keyword_search_tool(tool_info):
        return args
    cached = getattr(agent, "_search_rewrite", None)
    if cached is None:
        from sources.long_task.search_query_builder import build_search_queries
        try:
            cached = await build_search_queries(
                getattr(agent, "_last_user_prompt", "") or "", agent.llm,
            )
        except Exception:
            cached = {"queries": []}
        agent._search_rewrite = cached
    queries = (cached or {}).get("queries") or []
    if not queries:
        return args
    rewritten = queries[0]
    out = dict(args or {})
    if "q" in out:
        out["q"] = rewritten
        return out
    if "query" in out:
        out["query"] = rewritten
        return out
    if "params" in out:
        try:
            import json
            if isinstance(out["params"], str):
                p = json.loads(out["params"])
            elif isinstance(out["params"], dict):
                p = dict(out["params"])
            else:
                return args
            if "q" in p:
                p["q"] = rewritten
            elif "query" in p:
                p["query"] = rewritten
            else:
                return args
            out["params"] = json.dumps(p, ensure_ascii=False)
            return out
        except (ValueError, TypeError):
            return args
    return args
```

3d. `general_agent.py` 的 `create_agent` 状态重置区（`self._react_loop_ran = False` 之后）加一行：

```python
        self._search_rewrite = None   # deterministic q rewrite cache, per request
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_tools.py -v`
Expected: PASS（17 + 8 = 25 passed）

- [ ] **Step 5: Commit**

```bash
git add sources/agents/react_tools.py sources/agents/general_agent.py tests/test_react_tools.py
git commit -m "feat: deterministically rewrite keyword search queries in ReAct chat path"
```

---

### Task 5: 内置下载分析工具 `fetch_patent_spec`

**Files:**
- Modify: `sources/agents/react_tools.py`（工具注册 + 执行分支）
- Test: `tests/test_react_tools.py`（追加测试类）

**Interfaces:**
- Consumes:
  - Task 1: `sources.long_task.uspto_download.download_uspto_patent_text(patent_id, spec_selector_provider=None, logger=None) -> tuple[str|None, bytes|None]`（async）
  - Task 2: `sources.long_task.patent_distill.distill_patent_spec(text, query, provider) -> dict`、`format_distilled(distilled, lang) -> str`、`truncated_fallback(text) -> str`
- Produces: 常量 `FETCH_PATENT_SPEC_TOOL_NAME = "fetch_patent_spec"`（循环内注册，前端步骤行自动显示）

- [ ] **Step 1: 写失败测试**

在 `tests/test_react_tools.py` 末尾追加：

```python
# ── fetch_patent_spec built-in tool ──────────────────────────────────────────

from sources.agents.react_tools import (
    FETCH_PATENT_SPEC_TOOL_NAME,
    build_tool_set,
)


class _SpecAgent(_FakeAgent):
    def __init__(self):
        super().__init__()
        self._last_user_prompt = "分析这篇专利的技术方案"
        self.llm = _RewriteAgent().llm  # reuse complete_json stub


class TestFetchPatentSpecRegistered(unittest.TestCase):
    @patch("sources.agents.react_tools.get_knowledge_tool_candidates")
    def test_tool_always_registered(self, mock_candidates):
        mock_candidates.return_value = []
        agent = _FakeAgent()
        registry, tools = asyncio.run(
            build_tool_set(agent, "u1", "任意问题", push_filter=None))
        self.assertIn(FETCH_PATENT_SPEC_TOOL_NAME, registry)
        self.assertEqual(
            registry[FETCH_PATENT_SPEC_TOOL_NAME].kind, "patent_spec")
        self.assertEqual(len(tools), len(registry))


class TestFetchPatentSpecExecution(unittest.IsolatedAsyncioTestCase):
    def _registry(self, agent):
        entry = type("E", (), {
            "name": FETCH_PATENT_SPEC_TOOL_NAME, "kind": "patent_spec",
            "knowledge": None, "tool_info": None, "tool": None,
        })()
        return {FETCH_PATENT_SPEC_TOOL_NAME: entry}

    async def test_downloads_distills_and_returns_observation(self):
        agent = _SpecAgent()
        executor = asyncio.run(
            make_action_executor(agent, self._registry(agent), None))
        with patch("sources.agents.react_tools.download_uspto_patent_text",
                   new=AsyncMock(return_value=("FULL SPEC TEXT", None))):
            with patch("sources.agents.react_tools.distill_patent_spec",
                       new=AsyncMock(return_value={
                           "发明点": "a", "技术方案": "b",
                           "权利要求要点": "c",
                       })):
                result = asyncio.run(
                    executor(FETCH_PATENT_SPEC_TOOL_NAME,
                             {"patent_id": "19511555"}, 2))
        self.assertEqual(result["kind"], "observation")
        self.assertIn("发明点", result["text"])
        self.assertIn("c", result["text"])

    async def test_download_failure_returns_error_observation(self):
        agent = _SpecAgent()
        executor = asyncio.run(
            make_action_executor(agent, self._registry(agent), None))
        with patch("sources.agents.react_tools.download_uspto_patent_text",
                   new=AsyncMock(return_value=(None, None))):
            result = asyncio.run(
                executor(FETCH_PATENT_SPEC_TOOL_NAME,
                         {"patent_id": "19511555"}, 2))
        self.assertTrue(result["text"].startswith("Error:"))
        self.assertIn("说明书", result["text"])

    async def test_distill_failure_falls_back_to_truncated_text(self):
        agent = _SpecAgent()
        executor = asyncio.run(
            make_action_executor(agent, self._registry(agent), None))
        with patch("sources.agents.react_tools.download_uspto_patent_text",
                   new=AsyncMock(return_value=("x" * 50000, None))):
            with patch("sources.agents.react_tools.distill_patent_spec",
                       new=AsyncMock(return_value={})):
                result = asyncio.run(
                    executor(FETCH_PATENT_SPEC_TOOL_NAME,
                             {"patent_id": "19511555"}, 2))
        self.assertEqual(len(result["text"]), 16000)

    async def test_binary_only_result_returns_error(self):
        agent = _SpecAgent()
        executor = asyncio.run(
            make_action_executor(agent, self._registry(agent), None))
        with patch("sources.agents.react_tools.download_uspto_patent_text",
                   new=AsyncMock(return_value=(None, b"PDF"))):
            result = asyncio.run(
                executor(FETCH_PATENT_SPEC_TOOL_NAME,
                         {"patent_id": "19511555"}, 2))
        self.assertTrue(result["text"].startswith("Error:"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_tools.py -v`
Expected: FAIL（ImportError: cannot import name 'FETCH_PATENT_SPEC_TOOL_NAME'）

- [ ] **Step 3: 写实现**

3a. `react_tools.py` 常量区追加：

```python
FETCH_PATENT_SPEC_TOOL_NAME = "fetch_patent_spec"


class _PatentIdArgs(BaseModel):
    patent_id: str = Field(description="USPTO application number (8 digits, e.g. 19511555)")


async def _fetch_patent_spec_stub(patent_id: str) -> str:
    raise NotImplementedError("executed via dispatch, not directly")
```

3b. `build_tool_set` 的 `add` 注册区（`search_tool` 注册之前）插入：

```python
    spec_tool = StructuredTool.from_function(
        func=_fetch_patent_spec_stub,
        name=FETCH_PATENT_SPEC_TOOL_NAME,
        description=(
            "Download and analyze the specification (说明书) of one USPTO "
            "patent application by its application number. Use this when the "
            "user asks for the technical solution, claims, or details of a "
            "specific patent. Returns a structured analysis of the full text."
        ),
        args_schema=_PatentIdArgs,
    )
    add(ToolEntry(name=FETCH_PATENT_SPEC_TOOL_NAME, kind="patent_spec",
                  knowledge=None, tool_info=None, tool=spec_tool))
```

3c. `make_action_executor` 的 `execute_action` 中，`entry.kind == "search"` 分支之前插入：

```python
        if entry.kind == "patent_spec":
            return await _run_patent_spec(agent, args, lang)
```

3d. `make_action_executor` 之前新增执行器：

```python
async def _run_patent_spec(agent, args, lang: str) -> dict:
    """Download one patent's specification and distill it into the loop."""
    patent_id = str((args or {}).get("patent_id") or "").strip()
    if not patent_id:
        return {"kind": "observation", "text": "Error: missing patent_id"}
    from sources.long_task.patent_distill import (
        distill_patent_spec, format_distilled, truncated_fallback,
    )
    from sources.long_task.uspto_download import download_uspto_patent_text

    text, binary = await download_uspto_patent_text(
        patent_id,
        spec_selector_provider=getattr(agent, "llm", None),
        logger=getattr(agent, "logger", None),
    )
    if not text:
        if binary is not None:
            return {"kind": "observation",
                    "text": "Error: 说明书为扫描件，暂无法自动提取文本分析"}
        return {"kind": "observation",
                "text": f"Error: 说明书下载失败（专利号 {patent_id}）"}
    query = getattr(agent, "_last_user_prompt", "") or ""
    distilled = await distill_patent_spec(text, query, agent.llm)
    if distilled:
        return {"kind": "observation", "text": format_distilled(distilled, lang)}
    return {"kind": "observation", "text": truncated_fallback(text)}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_react_tools.py -v`
Expected: PASS（25 + 5 = 30 passed；既有 `test_workflow_type_2_skipped_and_top_n_respected` 的 `len(tools) == len(registry)` 断言不受影响——新工具计入两侧）

- [ ] **Step 5: Commit**

```bash
git add sources/agents/react_tools.py tests/test_react_tools.py
git commit -m "feat: add fetch_patent_spec built-in tool with download and distillation"
```

---

### Task 6: 全量回归 + 多领域基准验收准备

**Files:** 无代码改动（控制器跑回归 + 用户手动验收）

- [ ] **Step 1: 全量回归**

Run: `PYTHONUTF8=1 python -m pytest tests/ -q --continue-on-collection-errors 2>&1 | tail -2`
Expected: **402 passed / 26 failed / 9 errors**（基线 369 + 新测试 7 + 8 + 5 + 8 + 5 = 33；实施者报告以实际计数为准，核心判据：**失败与错误数 = 26/9 与基线完全一致，零新增**）

- [ ] **Step 2: 提交回归记录**

```bash
cat >> .superpowers/sdd/progress.md << 'EOF'
（任务完成记录由控制器维护，实施者跳过此步）
EOF
```

- [ ] **Step 3: 手动验收清单（用户执行，部署后）**

部署后端 api-test + 前端 test.copiioai.com 后，按 spec 第 12 节执行 5 领域基准查询集：

| # | 查询 | 领域 |
|---|---|---|
| 1 | 帮我查找工业中在线干燥空气源提供和设备内部环境湿度精准控制的相关专利（回归用例，对照 880 万噪声基线） | 工业除湿 |
| 2 | 电动汽车动力电池热失控预警和散热结构相关专利 | 新能源/电池 |
| 3 | 半导体工艺腔室的温度控制与晶圆加热相关专利 | 半导体设备 |
| 4 | AR 眼镜光学波导与全息显示相关专利 | 消费电子/光学 |
| 5 | 医学影像的 AI 辅助诊断算法相关专利 | 医疗 AI |

每查询检查：清单切题率 ≥80%、每篇有真实相关性分析（非套话）、改写日志（USPTO REQUEST 的 q 为 OR 组英文形态）、步骤行显示；任一查询追问某篇 → fetch_patent_spec 深挖回答基于真实提炼内容。

- [ ] **Step 4: 记录验收结果到会话文件**（/save-session，控制器执行）

---

## Self-Review

**1. Spec coverage:**
- 单元 1（观察增强）→ Task 3 ✓；单元 2（确定性改写 + 缓存 + create_agent 重置）→ Task 4 ✓；单元 3（共享模块 → Task 1、提炼 → Task 2、内置工具 → Task 5）✓；错误处理降级表 → 各任务测试覆盖（下载失败/提炼失败/改写失败/二进制扫描件）✓；验收（spec 第 12 节）→ Task 6 ✓；前端零改动 ✓；长任务行为不变 → Task 1 回归判据 ✓
- D5 通用性：无任何领域硬编码——提炼 prompt 为专利通用维度、改写输入为用户原始问题、digest 为 schema 驱动 ✓

**2. Placeholder scan:** Task 1 Step 3 有一处 `pass  # ← 以原函数体替换此占位`——这是**逐字搬移任务的刻意设计**（搬移源在 celery_worker.py 当前代码，计划转录 300 行反而会产生漂移风险），四类机械手术点已精确标注；其余任务全部含完整代码。无 TBD/TODO。

**3. Type consistency:**
- `download_uspto_patent_text(patent_id, spec_selector_provider=None, logger=None) -> tuple[str|None, bytes|None]`：Task 1 定义，Task 5 `_run_patent_spec` 以同名同签名调用 ✓
- `distill_patent_spec(text, query, provider) -> dict`（失败 `{}`）：Task 2 定义，Task 5 消费 ✓；`format_distilled(distilled, lang) -> str`、`truncated_fallback(text, limit=16000)` ✓
- `_items_digest(raw_items, limit=20, lang="zh") -> str`：Task 3 定义，仅 Task 3 使用 ✓
- `_maybe_rewrite_search_query(agent, tool_info, args) -> dict`：Task 4 定义并接入 execute_action ✓；`agent._search_rewrite` 由 Task 4 写入并在 general_agent.create_agent 重置 ✓
- Task 5 测试 patch 目标 `sources.agents.react_tools.download_uspto_patent_text` / `distill_patent_spec` 与实现中的函数内 import 一致（函数内 `from ... import` 在调用时解析模块属性，patch 生效）✓
- `FETCH_PATENT_SPEC_TOOL_NAME` 常量：Task 5 定义并测试引用 ✓

**4. 回归风险:** Task 1 删除 celery_worker 函数前强制 Grep 引用检查（有其他引用则留别名）；Task 3 观察文本改造保留了「已截断」措辞使既有断言成立；Task 5 注册新工具不影响既有 `len(tools) == len(registry)` 断言（两侧同时增加）✓
