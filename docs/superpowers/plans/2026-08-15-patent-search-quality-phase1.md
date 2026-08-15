# 专利检索质量 Phase 1（关键词路线）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复长任务管线 PHASE0 的专利检索质量——查询改写（专利域英文术语 + OR 组）、相关性闸门、同族去重、CPC 字段增强，并在报告中披露检索策略与法律状态等元数据。

**Architecture:** 新增 3 个聚焦模块（`search_query_builder.py` 查询改写、`candidate_metadata.py` 候选元数据提取/字段补全/去重、`relevance_gate.py` 相关性评分与受控翻页搜索），`celery_worker.py` PHASE0 仅做薄接线：USPTO 关键词搜索工具改走 `phase0_gated_search`（改写 → 受控翻页 → 评分 → 去重 → 截断），非 USPTO 工具与所有失败场景走原有 legacy 路径不变。报告层（Task 5）把候选元数据并入行数据并追加「检索说明与局限」章节。

**Tech Stack:** Python 3.14（async/await）、unittest + `IsolatedAsyncioTestCase` + `unittest.mock`、现有 `Provider.complete_json`（Flash LLM）、USPTO Patent Application Search API（`q` 支持 `"phrase"`、`(A OR B)`、AND/OR、通配符、`offset`/`limit` 分页；**无 relevance 排序**——排序由相关性闸门承担）。

## Global Constraints

- Python 3.14：禁用 `asyncio.get_event_loop()`；测试一律用 `IsolatedAsyncioTestCase` + `AsyncMock`。
- 测试运行：`cd E:\online\workspace\copiioai\langsistance && PYTHONUTF8=1 python -m pytest tests/test_xxx.py -v`（用系统 python `C:/Python314`；repo venv 未装 pytest；Windows GBK 控制台必须带 `PYTHONUTF8=1`）。
- 测试风格：沿用现有 `unittest`（非 pytest 风格裸函数），参照 `tests/test_dynamic_tool_params.py`、`tests/test_report_generator.py`。
- 新模块只允许依赖 `sources.long_task` 内既有模块与标准库，不得引入新第三方包。
- **范围仅 USPTO，项目不检索中国专利**：CNIPA 旧代码路径零改动、零新逻辑（仅保留兼容，防止旧场景数据报错）；直接专利号模式不变。
- **优化面向通用专利检索场景**：任何模块不得为特定查询（如干燥空气）硬编码关键词、阈值或分支；验收用多领域查询基准集，而非单一查询。
- 所有 LLM 调用必须 try/except 包裹，失败时降级到 legacy 行为，**绝不因新逻辑抛异常导致任务失败**。
- 提交规范：`feat: ...` / `test: ...`，每任务一个 commit（TDD：先测后码）。
- 分支：从当前 HEAD 拉出 `feature/patent-search-quality`（当前分支 `feature/agent-react-loop` 的 ReAct 工作未合并，勿混入）。

## File Structure

| 文件 | 职责 |
|---|---|
| `sources/long_task/search_query_builder.py` | **新建**。LLM 查询改写 + 纯函数检索式组装/清洗（`assemble_query`、`sanitize_uspto_query`） |
| `sources/long_task/candidate_metadata.py` | **新建**。USPTO raw_items → 候选元数据；`ensure_search_fields`（补 CPC/标题/状态字段）；`dedupe_candidates`（同族/同标题去重）；`is_keyword_search_tool` / `is_uspto_tool` |
| `sources/long_task/relevance_gate.py` | **新建**。批量相关性评分（`score_candidates` / `apply_scores` / `filter_by_relevance`）+ 受控翻页搜索 `run_gated_search` + PHASE0 入口 `phase0_gated_search` |
| `celery_worker.py` | **修改**。PHASE0 接线（约 `:836-883`）：USPTO 工具走 gated 路径，产出 `patent_meta_map` / `search_meta`；PHASE2 `_analyze_one` 合并 `_row["_meta"]`（约 `:1064`）；PHASE3 追加方法论章节（约 `:1331`） |
| `sources/long_task/report_generator.py` | **修改**。`_meta_lines` 元数据行、exec summary/section 数据含元数据、`append_methodology_section` |
| `tests/test_search_query_builder.py` | **新建** |
| `tests/test_candidate_metadata.py` | **新建** |
| `tests/test_relevance_gate.py` | **新建** |
| `tests/test_report_generator.py` | **修改**（追加元数据与方法论章节测试） |

数据流：`select_tool`（不变，仍负责选工具）→ `phase0_gated_search`（改写 q → `run_gated_search` 翻页执行 → `build_candidates` → `score_candidates` → `filter_by_relevance` → `dedupe_candidates` → 截断 target_count）→ `patent_ids` / `patent_meta_map` / `search_meta` → PHASE2（`_meta` 合并入行）→ PHASE3（方法论章节）。

---

### Task 1: 查询改写模块 `search_query_builder.py`

**Files:**
- Create: `sources/long_task/search_query_builder.py`
- Test: `tests/test_search_query_builder.py`

**Interfaces:**
- Consumes: `Provider.complete_json(system_prompt, user_content)`（async，返回 dict 或任意）
- Produces:
  - `build_search_queries(query: str, provider) -> dict` — async，**永不抛异常**；失败返回 `{"concepts": [], "queries": []}`（调用方据此保持原查询）
  - `assemble_query(groups: list[list[str]]) -> str` — 纯函数
  - `sanitize_uspto_query(q: str) -> str` — 纯函数

- [ ] **Step 1: 写失败测试**

创建 `tests/test_search_query_builder.py`：

```python
"""Tests for search_query_builder — USPTO query rewriting helpers."""
import unittest

from sources.long_task.search_query_builder import (
    assemble_query,
    build_search_queries,
    sanitize_uspto_query,
)


class _FakeProvider:
    def __init__(self, response):
        self._response = response
        self.calls = []

    async def complete_json(self, system, user):
        self.calls.append((system, user))
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class TestAssembleQuery(unittest.TestCase):
    def test_single_concept_quotes_multiword_terms(self):
        self.assertEqual(
            assemble_query([["compressed air dryer", "air dryer", "desiccant"]]),
            '("compressed air dryer" OR "air dryer" OR desiccant)',
        )

    def test_multi_concept_joins_with_and(self):
        self.assertEqual(
            assemble_query([
                ["compressed air dryer", "desiccant dryer"],
                ["humidity control", "dew point", "dehumidif*"],
            ]),
            '("compressed air dryer" OR "desiccant dryer")'
            ' AND ("humidity control" OR "dew point" OR dehumidif*)',
        )

    def test_skips_empty_groups(self):
        self.assertEqual(
            assemble_query([[], ["dew point", "humidity"]]),
            '("dew point" OR humidity)',
        )


class TestSanitizeUsptoQuery(unittest.TestCase):
    def test_removes_cjk_characters(self):
        self.assertEqual(
            sanitize_uspto_query('"compressed air dryer" AND 湿度控制'),
            '"compressed air dryer" AND',
        )

    def test_caps_length(self):
        long_q = " AND ".join([f'term{i}' for i in range(50)])
        result = sanitize_uspto_query(long_q)
        self.assertLessEqual(len(result), 250)

    def test_empty_input(self):
        self.assertEqual(sanitize_uspto_query(""), "")


class TestBuildSearchQueries(unittest.IsolatedAsyncioTestCase):
    async def test_returns_validated_queries(self):
        provider = _FakeProvider({
            "concepts": [
                {"concept": "干燥空气源", "keywords": ["air dryer", "desiccant dryer"]},
            ],
            "queries": [
                '("compressed air dryer" OR "air dryer") AND 湿度',
                '"desiccant dryer"',
            ],
        })
        result = await build_search_queries("工业在线干燥空气源", provider)
        self.assertEqual(len(result["queries"]), 2)
        self.assertEqual(
            result["queries"][0],
            '("compressed air dryer" OR "air dryer") AND',
        )

    async def test_provider_failure_returns_empty(self):
        provider = _FakeProvider(RuntimeError("boom"))
        result = await build_search_queries("任意查询", provider)
        self.assertEqual(result, {"concepts": [], "queries": []})

    async def test_garbage_response_returns_empty_queries(self):
        provider = _FakeProvider({"queries": [123, None, "   "]})
        result = await build_search_queries("任意查询", provider)
        self.assertEqual(result["queries"], [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_search_query_builder.py -v`
Expected: FAIL（`ModuleNotFoundError: sources.long_task.search_query_builder`）

- [ ] **Step 3: 写最小实现**

创建 `sources/long_task/search_query_builder.py`：

```python
"""LLM-assisted USPTO search query construction for long-task PHASE0.

The Flash LLM translates the user's natural-language (usually Chinese)
question into patent-domain English keyword groups and assembles USPTO
free-form query strings. Pure assembly helpers are separated from the
LLM call so they can be unit-tested without a provider.
"""

import re
from typing import Any

DEFAULT_QUERY_MAX_LENGTH = 250

_CJK_RE = re.compile(r'[一-鿿]')

REWRITE_SYSTEM_PROMPT = (
    "你是一个专利检索式构造专家。把用户的自然语言技术问题改写为 "
    "USPTO Patent Application Search API 的检索式（q 参数）。"
    "本工具面向所有技术领域，不得为特定领域预设关键词。\n\n"
    "步骤：\n"
    "1. 从用户问题中抽取 2-4 个核心技术概念（忽略语气词和通用词）\n"
    "2. 每个概念翻译成该领域专利文献常用的英文术语，并给出 2-5 个"
    "同义/近义关键词（含行业缩写、上位/下位词、英美拼写变体）\n"
    "3. 用检索式语法组装查询串：\n"
    "   - 多词短语必须用双引号包裹，如 \"3d printing\"\n"
    "   - 同一概念的同义词用 OR 连接并放在圆括号内："
    '("3d printing" OR "additive manufacturing" OR "rapid prototyping")\n'
    "   - 不同概念之间用 AND 连接\n"
    "   - 支持通配符，如 \"thermal runaway\" AND (battery OR \"energy storage\")\n"
    "   - 每个检索式最多 12 个关键词、250 字符，禁止出现中文\n"
    "4. 输出 1-3 个检索式：第一个为完整概念组合式；后续为放宽的变体"
    "（去掉较次要的概念），用于扩大召回。\n\n"
    'Return JSON: {"concepts": [{"concept": "中文概念", '
    '"keywords": ["english term", ...]}], '
    '"queries": ["query1", "query2", ...]}'
)


def assemble_query(groups: list[list[str]]) -> str:
    """Join keyword groups into a USPTO free-form query string.

    Each inner list is one concept: its keywords are OR-joined inside
    parentheses; concepts are AND-joined. Multi-word keywords are
    double-quoted automatically.
    """
    parts = []
    for group in groups:
        group = [str(k).strip() for k in group if str(k).strip()]
        if not group:
            continue
        joined = " OR ".join(
            f'"{k}"' if (" " in k and not (k.startswith('"') and k.endswith('"')))
            else k
            for k in group
        )
        parts.append(f"({joined})")
    return " AND ".join(parts)


def sanitize_uspto_query(q: str) -> str:
    """Strip CJK characters, collapse whitespace and cap query length."""
    if not q:
        return ""
    q = _CJK_RE.sub(" ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q[:DEFAULT_QUERY_MAX_LENGTH]


def _validated_rewrite(raw: Any) -> dict:
    """Validate and sanitize LLM output into the canonical rewrite dict."""
    if not isinstance(raw, dict):
        return {"concepts": [], "queries": []}
    queries = raw.get("queries") or []
    if not isinstance(queries, list):
        queries = []
    cleaned = []
    for q in queries:
        q = sanitize_uspto_query(str(q)) if isinstance(q, str) else ""
        if q:
            cleaned.append(q)
    return {"concepts": raw.get("concepts") or [], "queries": cleaned}


async def build_search_queries(query: str, provider: Any) -> dict:
    """Rewrite a user question into USPTO search queries via the Flash LLM.

    Never raises: on any failure returns ``{"concepts": [], "queries": []}``,
    which signals callers to keep their existing query untouched.
    """
    try:
        result = await provider.complete_json(REWRITE_SYSTEM_PROMPT, query)
    except Exception:
        return {"concepts": [], "queries": []}
    return _validated_rewrite(result)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_search_query_builder.py -v`
Expected: PASS（9 passed）

- [ ] **Step 5: Commit**

```bash
git add sources/long_task/search_query_builder.py tests/test_search_query_builder.py
git commit -m "feat: add LLM query rewriting for USPTO search with OR-group assembly"
```

---

### Task 2: 候选元数据模块 `candidate_metadata.py`

**Files:**
- Create: `sources/long_task/candidate_metadata.py`
- Test: `tests/test_candidate_metadata.py`

**Interfaces:**
- Consumes: Task 1 无直接依赖；`raw_items` 为 USPTO `patentFileWrapperDataBag` 列表
- Produces:
  - `build_candidates(raw_items: list) -> list[dict]` — 每项含 `patent_id/title/applicant/status/filing_date/grant_date/cpc_codes/patent_number/_raw`
  - `ensure_search_fields(params: dict) -> dict` — 返回**深拷贝**后的 params，`body.fields` 补全
  - `dedupe_candidates(candidates: list[dict]) -> tuple[list[dict], int]` — 返回 (去重后列表, 去重数)，保留高相关性/已授权/新申请者
  - `is_keyword_search_tool(tool) -> bool`、`is_uspto_tool(tool) -> bool`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_candidate_metadata.py`：

```python
"""Tests for candidate_metadata — USPTO raw_items flattening + dedupe."""
import unittest

from sources.long_task.candidate_metadata import (
    build_candidates,
    dedupe_candidates,
    ensure_search_fields,
    is_keyword_search_tool,
    is_uspto_tool,
)


def _usp_item(app_number, title, applicant="ACME Corp", filing="2024-01-15",
              status="Patented Case", cpc=None, grant=None, continuity=None):
    meta = {
        "applicationNumberText": app_number,
        "inventionTitle": title,
        "firstApplicantName": applicant,
        "filingDate": filing,
        "applicationStatusDescriptionText": status,
    }
    if cpc:
        meta["cpcClassificationBag"] = [{"cpcClassCode": c} for c in cpc]
    if grant:
        meta["grantDate"] = grant
        meta["patentNumber"] = grant.replace(",", "")
    item = {"applicationMetaData": meta}
    if continuity:
        item["parentContinuityBag"] = continuity
    return item


class _FakeTool:
    def __init__(self, title, url):
        self.title = title
        self.url = url


class TestBuildCandidates(unittest.TestCase):
    def test_extracts_standard_metadata(self):
        items = [_usp_item("19511555", "Hydrogen Supply System",
                           applicant="Robert Bosch GmbH", cpc=["F02M 21/02"])]
        candidates = build_candidates(items)
        self.assertEqual(len(candidates), 1)
        c = candidates[0]
        self.assertEqual(c["patent_id"], "19511555")
        self.assertEqual(c["title"], "Hydrogen Supply System")
        self.assertEqual(c["applicant"], "Robert Bosch GmbH")
        self.assertEqual(c["filing_date"], "2024-01-15")
        self.assertEqual(c["status"], "Patented Case")
        self.assertEqual(c["cpc_codes"], ["F02M 21/02"])

    def test_skips_items_without_application_number(self):
        candidates = build_candidates([
            {"applicationMetaData": {"inventionTitle": "No number"}},
            _usp_item("18184836", "Valid one"),
        ])
        self.assertEqual([c["patent_id"] for c in candidates], ["18184836"])

    def test_handles_non_dict_items(self):
        candidates = build_candidates([None, "junk", 42])
        self.assertEqual(candidates, [])


class TestEnsureSearchFields(unittest.TestCase):
    def test_adds_missing_fields_without_mutating_input(self):
        params = {"body": {"q": "air dryer", "fields": ["applicationMetaData.inventionTitle"]}}
        out = ensure_search_fields(params)
        self.assertIn("applicationMetaData.cpcClassificationBag", out["body"]["fields"])
        self.assertIn("parentContinuityBag", out["body"]["fields"])
        self.assertNotIn("applicationMetaData.cpcClassificationBag",
                         params["body"]["fields"])

    def test_keeps_fields_without_body(self):
        out = ensure_search_fields({"query": {}})
        self.assertEqual(out, {"query": {}})


class TestDedupeCandidates(unittest.TestCase):
    def test_same_title_deduped_keeps_higher_score(self):
        candidates = [
            {"patent_id": "17361306", "title": "Hybrid Level EVSE",
             "relevance_score": 3, "filing_date": "2021-06-28", "_raw": {}},
            {"patent_id": "16821726", "title": "Hybrid Level EVSE",
             "relevance_score": 4, "filing_date": "2020-03-17", "_raw": {}},
        ]
        kept, dropped = dedupe_candidates(candidates)
        self.assertEqual([c["patent_id"] for c in kept], ["16821726"])
        self.assertEqual(dropped, 1)

    def test_continuity_relationship_deduped(self):
        candidates = [
            {"patent_id": "19511555", "title": "Hydrogen supply system",
             "relevance_score": 5, "filing_date": "2024-08-05",
             "_raw": {"parentContinuityBag": [
                 {"parentApplicationNumberText": "PCTEP2024072117",
                  "childApplicationNumberText": "19511555"}]}},
            {"patent_id": "19504130", "title": "Unrelated fuel injection",
             "relevance_score": 4, "filing_date": "2024-06-01",
             "_raw": {"parentContinuityBag": [
                 {"parentApplicationNumberText": "19511555"}]}},
        ]
        kept, dropped = dedupe_candidates(candidates)
        # 19511555 scores higher and is referenced by 19504130's parent bag
        self.assertEqual([c["patent_id"] for c in kept], ["19511555"])
        self.assertEqual(dropped, 1)

    def test_distinct_titles_kept(self):
        candidates = [
            {"patent_id": "11111111", "title": "Air dryer control",
             "relevance_score": 3, "filing_date": "2020-01-01", "_raw": {}},
            {"patent_id": "22222222", "title": "Moisture control enclosure",
             "relevance_score": 3, "filing_date": "2020-01-01", "_raw": {}},
        ]
        kept, dropped = dedupe_candidates(candidates)
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, 0)


class TestToolPredicates(unittest.TestCase):
    def test_keyword_tool_detection(self):
        self.assertTrue(is_keyword_search_tool(
            _FakeTool("search_patent_by_key_word", "https://api.uspto.gov/api/v1/patent/applications/search")))
        self.assertFalse(is_keyword_search_tool(
            _FakeTool("get_patent_documents_application_number",
                      "https://api.uspto.gov/api/v1/patent/applications")))

    def test_uspto_tool_detection(self):
        self.assertTrue(is_uspto_tool(
            _FakeTool("search_patent_by_key_word", "https://api.uspto.gov/api/v1/patent/applications/search")))
        self.assertFalse(is_uspto_tool(
            _FakeTool("cnipa_search", "https://open.zldsj.com/api/search")))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_candidate_metadata.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 写最小实现**

创建 `sources/long_task/candidate_metadata.py`：

```python
"""Candidate extraction and enrichment for USPTO search results.

raw_items from the USPTO applications/search response are
``patentFileWrapperDataBag`` entries. This module flattens the useful
metadata (title, applicant, dates, status, CPC codes) into compact
candidate dicts used by the relevance gate, dedupe, and report.
"""

import re
from copy import deepcopy
from typing import Any

# USPTO response fields worth requesting when they are not already in
# the tool's fields template.
SEARCH_FIELDS_TO_ENSURE = [
    "applicationMetaData.inventionTitle",
    "applicationMetaData.firstApplicantName",
    "applicationMetaData.applicationStatusDescriptionText",
    "applicationMetaData.filingDate",
    "applicationMetaData.grantDate",
    "applicationMetaData.patentNumber",
    "applicationMetaData.cpcClassificationBag",
    "parentContinuityBag",
]


def is_keyword_search_tool(tool: Any) -> bool:
    """True when the tool's title indicates a keyword search tool."""
    title = (getattr(tool, "title", "") or "").lower()
    return "key" in title or "keyword" in title


def is_uspto_tool(tool: Any) -> bool:
    """True when the tool's URL targets api.uspto.gov."""
    url = (getattr(tool, "url", "") or "").lower()
    return "uspto" in url


def ensure_search_fields(params: dict) -> dict:
    """Return a deep copy of tool params with required USPTO fields added.

    Only touches ``body.fields``. The input dict is never mutated.
    """
    out = deepcopy(params)
    body = out.get("body") if isinstance(out.get("body"), dict) else {}
    fields = body.get("fields")
    if not isinstance(fields, list):
        return out
    for f in SEARCH_FIELDS_TO_ENSURE:
        if f not in fields:
            fields.append(f)
    return out


def _first_str(d: dict, *keys: str) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _meta(item: dict) -> dict:
    m = item.get("applicationMetaData")
    if not isinstance(m, dict):
        m = {}
    return m


def _extract_cpc_codes(m: dict) -> list[str]:
    """Defensively collect CPC codes from applicationMetaData."""
    codes: list[str] = []
    bag = m.get("cpcClassificationBag") or m.get("classificationBag")
    if isinstance(bag, list):
        for entry in bag:
            if isinstance(entry, dict):
                for k in ("cpcClassCode", "classificationCode", "cpcCode"):
                    v = entry.get(k)
                    if isinstance(v, str) and v.strip():
                        codes.append(v.strip())
                        break
    return list(dict.fromkeys(codes))


def build_candidates(raw_items: list) -> list[dict]:
    """Flatten USPTO raw_items into compact candidate dicts."""
    candidates = []
    for item in raw_items or []:
        if not isinstance(item, dict):
            continue
        m = _meta(item)
        pid = str(m.get("applicationNumberText") or "").strip()
        if not pid:
            continue
        candidates.append({
            "patent_id": pid,
            "title": _first_str(m, "inventionTitle"),
            "applicant": _first_str(m, "firstApplicantName"),
            "status": _first_str(m, "applicationStatusDescriptionText"),
            "filing_date": _first_str(m, "filingDate"),
            "grant_date": _first_str(m, "grantDate"),
            "patent_number": _first_str(m, "patentNumber"),
            "cpc_codes": _extract_cpc_codes(m),
            "_raw": item,
        })
    return candidates


# ── Dedupe ──────────────────────────────────────────────────────────────────

def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (title or "").lower())


def _continuity_ids(item: dict) -> set[str]:
    """Collect application numbers referenced by parentContinuityBag."""
    ids: set[str] = set()
    bag = item.get("parentContinuityBag")
    if isinstance(bag, list):
        for entry in bag:
            if isinstance(entry, dict):
                for k in ("parentApplicationNumberText",
                          "childApplicationNumberText"):
                    v = entry.get(k)
                    if isinstance(v, str) and v.strip():
                        ids.add(v.strip())
    return ids


def _sort_key(c: dict) -> tuple:
    granted = 1 if c.get("patent_number") else 0
    score = c.get("relevance_score")
    if not isinstance(score, (int, float)):
        score = -1
    return (granted, score, c.get("filing_date") or "")


def dedupe_candidates(candidates: list[dict]) -> tuple[list[dict], int]:
    """Remove family/near-duplicate candidates, keeping the best one.

    Two candidates are duplicates when one's patent_id appears in the
    other's ``parentContinuityBag``, or their normalized titles are
    identical. Preference: granted > higher relevance_score > newer
    filing date (ordering by ``_sort_key`` descending).
    """
    ordered = sorted(candidates, key=_sort_key, reverse=True)
    title_groups: dict[str, str] = {}
    seen_ids: set[str] = set()
    kept: list[dict] = []
    dropped = 0
    for c in ordered:
        pid = c["patent_id"]
        dup = any(rel in seen_ids for rel in _continuity_ids(c.get("_raw") or {}))
        nt = _norm_title(c.get("title", ""))
        if not dup and nt:
            if nt in title_groups:
                dup = True
        if dup:
            dropped += 1
            continue
        seen_ids.add(pid)
        if nt and nt not in title_groups:
            title_groups[nt] = pid
        kept.append(c)
    return kept, dropped
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_candidate_metadata.py -v`
Expected: PASS（10 passed）

- [ ] **Step 5: Commit**

```bash
git add sources/long_task/candidate_metadata.py tests/test_candidate_metadata.py
git commit -m "feat: add USPTO candidate metadata extraction, field completion and family dedupe"
```

---

### Task 3: 相关性闸门模块 `relevance_gate.py`

**Files:**
- Create: `sources/long_task/relevance_gate.py`
- Test: `tests/test_relevance_gate.py`

**Interfaces:**
- Consumes:
  - Task 1: `build_search_queries(query, provider) -> {"queries": [...]}`
  - Task 2: `build_candidates`、`dedupe_candidates`、`ensure_search_fields`、`is_keyword_search_tool`、`is_uspto_tool`
  - `sources.long_task.scene_tools.execute_tool(tool, params) -> {"data": ..., "raw_items": [...]}`（async）
  - `Provider.complete_json`（async）
- Produces（Task 4 依赖）:
  - `phase0_gated_search(selected, user_query, provider, target_count, task_id="", logger=None) -> {"candidates": list[dict], "search_meta": dict}`（async）
  - `apply_scores(candidates, result) -> list[dict]`、`filter_by_relevance(candidates, min_score=3) -> list[dict]`（纯函数）

- [ ] **Step 1: 写失败测试**

创建 `tests/test_relevance_gate.py`：

```python
"""Tests for relevance_gate — candidate scoring and gated search loop."""
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from sources.long_task.relevance_gate import (
    apply_scores,
    filter_by_relevance,
    phase0_gated_search,
    run_gated_search,
)
from sources.long_task.search_query_builder import build_search_queries


class _FakeProvider:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def complete_json(self, system, user):
        self.calls.append(user)
        if not self._responses:
            return {}
        r = self._responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _usp_item(app_number, title):
    return {
        "applicationMetaData": {
            "applicationNumberText": app_number,
            "inventionTitle": title,
            "firstApplicantName": "ACME",
            "filingDate": "2024-01-15",
            "applicationStatusDescriptionText": "Patented Case",
        },
        "parentContinuityBag": [],
    }


class TestApplyScores(unittest.TestCase):
    def test_attaches_scores_by_id(self):
        candidates = [
            {"patent_id": "11111111", "title": "A"},
            {"patent_id": "22222222", "title": "B"},
        ]
        result = {"scores": [
            {"id": "11111111", "score": 5},
            {"id": "22222222", "score": 1},
            {"id": "99999999", "score": 5},
        ]}
        out = apply_scores(candidates, result)
        self.assertEqual(out[0]["relevance_score"], 5)
        self.assertEqual(out[1]["relevance_score"], 1)

    def test_ignores_invalid_scores(self):
        candidates = [{"patent_id": "11111111", "title": "A"}]
        result = {"scores": [
            {"id": "11111111", "score": 99},
            {"id": "11111111", "score": "five"},
        ]}
        out = apply_scores(candidates, result)
        self.assertNotIn("relevance_score", out[0])

    def test_garbage_result_keeps_candidates(self):
        candidates = [{"patent_id": "11111111", "title": "A"}]
        out = apply_scores(candidates, None)
        self.assertEqual(out, candidates)


class TestFilterByRelevance(unittest.TestCase):
    def test_keeps_above_threshold_sorted_desc(self):
        candidates = [
            {"patent_id": "1", "relevance_score": 2},
            {"patent_id": "2", "relevance_score": 5},
            {"patent_id": "3", "relevance_score": 3},
        ]
        kept = filter_by_relevance(candidates)
        self.assertEqual([c["patent_id"] for c in kept], ["2", "3"])


class TestRunGatedSearch(unittest.IsolatedAsyncioTestCase):
    async def test_overrides_keyword_tool_query_and_scores(self):
        tool = MagicMock()
        tool.title = "search_patent_by_key_word"
        tool.url = "https://api.uspto.gov/api/v1/patent/applications/search"
        selected = {
            "tool": tool,
            "params": {
                "body": {
                    "q": "bad literal translation",
                    "fields": ["applicationMetaData.inventionTitle"],
                    "pagination": {"offset": 0, "limit": 50},
                }
            },
        }
        page = {
            "data": {"count": 2},
            "raw_items": [_usp_item("11111111", "Air dryer humidity control"),
                          _usp_item("22222222", "EV charging station")],
        }
        with patch("sources.long_task.relevance_gate.execute_tool",
                   new=AsyncMock(return_value=page)) as mock_exec:
            provider = _FakeProvider([
                {"scores": [
                    {"id": "11111111", "score": 5},
                    {"id": "22222222", "score": 1},
                ]},
            ])
            out = await run_gated_search(
                selected=selected,
                user_query="工业在线干燥空气源",
                provider=provider,
                rewrite={"queries": ['("air dryer" OR desiccant)']},
                target_count=10,
            )
        # q was replaced with the rewritten query
        sent_params = mock_exec.call_args[0][1]
        self.assertEqual(
            sent_params["body"]["q"], '("air dryer" OR desiccant)')
        # fields were completed
        self.assertIn("applicationMetaData.cpcClassificationBag",
                      sent_params["body"]["fields"])
        # only the relevant candidate survives
        self.assertEqual([c["patent_id"] for c in out["candidates"]],
                         ["11111111"])
        self.assertEqual(out["search_meta"]["gated_dropped"], 1)
        self.assertEqual(out["search_meta"]["final_count"], 1)

    async def test_keeps_original_query_when_rewrite_empty(self):
        tool = MagicMock()
        tool.title = "search_patent_by_key_word"
        tool.url = "https://api.uspto.gov/api/v1/patent/applications/search"
        selected = {
            "tool": tool,
            "params": {"body": {"q": "original llm query",
                                "pagination": {"offset": 0, "limit": 50}}},
        }
        page = {
            "data": {"count": 1},
            "raw_items": [_usp_item("11111111", "Air dryer control")],
        }
        with patch("sources.long_task.relevance_gate.execute_tool",
                   new=AsyncMock(return_value=page)) as mock_exec:
            provider = _FakeProvider([
                {"scores": [{"id": "11111111", "score": 5}]},
            ])
            out = await run_gated_search(
                selected=selected, user_query="干燥空气",
                provider=provider, rewrite={"queries": []},
                target_count=10,
            )
        sent_params = mock_exec.call_args[0][1]
        self.assertEqual(sent_params["body"]["q"], "original llm query")
        self.assertEqual(len(out["candidates"]), 1)

    async def test_pages_through_results_until_target(self):
        tool = MagicMock()
        tool.title = "search_patent_by_key_word"
        tool.url = "https://api.uspto.gov/api/v1/patent/applications/search"
        selected = {
            "tool": tool,
            "params": {"body": {"q": "old", "pagination": {"offset": 0, "limit": 2}}},
        }
        page1 = {
            "data": {"count": 3},
            "raw_items": [_usp_item("11111111", "Relevant dryer one"),
                          _usp_item("22222222", "Noise patent")],
        }
        # Short page (1 item < limit 2) signals the last page → loop stops
        page2 = {
            "data": {"count": 3},
            "raw_items": [_usp_item("33333333", "Relevant dryer two")],
        }
        with patch("sources.long_task.relevance_gate.execute_tool",
                   new=AsyncMock(side_effect=[page1, page2])) as mock_exec:
            provider = _FakeProvider([
                {"scores": [
                    {"id": "11111111", "score": 5},
                    {"id": "22222222", "score": 1},
                ]},
                {"scores": [
                    {"id": "33333333", "score": 4},
                ]},
            ])
            out = await run_gated_search(
                selected=selected, user_query="干燥空气",
                provider=provider,
                rewrite={"queries": ['"air dryer"']},
                target_count=2,
            )
        self.assertEqual(mock_exec.call_count, 2)
        # second page offset advanced by page size
        self.assertEqual(mock_exec.call_args_list[1][0][1]["body"]["pagination"]["offset"], 2)
        self.assertEqual([c["patent_id"] for c in out["candidates"]],
                         ["11111111", "33333333"])

    async def test_scoring_failure_keeps_candidates_unscored(self):
        tool = MagicMock()
        tool.title = "search_patent_by_key_word"
        tool.url = "https://api.uspto.gov/api/v1/patent/applications/search"
        selected = {
            "tool": tool,
            "params": {"body": {"q": "old", "pagination": {"offset": 0, "limit": 50}}},
        }
        page = {
            "data": {"count": 1},
            "raw_items": [_usp_item("11111111", "Air dryer control")],
        }
        with patch("sources.long_task.relevance_gate.execute_tool",
                   new=AsyncMock(return_value=page)):
            provider = _FakeProvider([RuntimeError("provider down")])
            out = await run_gated_search(
                selected=selected, user_query="干燥空气",
                provider=provider, rewrite={"queries": []},
                target_count=10,
            )
        # Gate LLM down → candidates kept unscored (degrade to legacy
        # behavior), pipeline must not raise
        self.assertEqual([c["patent_id"] for c in out["candidates"]],
                         ["11111111"])
        self.assertEqual(out["search_meta"]["candidates_scored"], 1)


class TestPhase0GatedSearch(unittest.IsolatedAsyncioTestCase):
    async def test_rewrites_then_runs_gated_search(self):
        tool = MagicMock()
        tool.title = "search_patent_by_key_word"
        tool.url = "https://api.uspto.gov/api/v1/patent/applications/search"
        selected = {
            "tool": tool,
            "params": {"body": {"q": "old", "pagination": {"offset": 0, "limit": 50}}},
        }
        page = {
            "data": {"count": 1},
            "raw_items": [_usp_item("11111111", "Air dryer control")],
        }
        with patch("sources.long_task.relevance_gate.execute_tool",
                   new=AsyncMock(return_value=page)) as mock_exec:
            # first provider response = rewrite, second = gate scores
            provider = _FakeProvider([
                {"queries": ['("air dryer" OR desiccant)']},
                {"scores": [{"id": "11111111", "score": 5}]},
            ])
            out = await phase0_gated_search(
                selected=selected, user_query="工业干燥空气源",
                provider=provider, target_count=10,
            )
        sent_params = mock_exec.call_args[0][1]
        self.assertEqual(sent_params["body"]["q"], '("air dryer" OR desiccant)')
        self.assertEqual(out["search_meta"]["final_count"], 1)

    async def test_rewrite_failure_keeps_llm_query(self):
        tool = MagicMock()
        tool.title = "search_patent_by_key_word"
        tool.url = "https://api.uspto.gov/api/v1/patent/applications/search"
        selected = {
            "tool": tool,
            "params": {"body": {"q": "llm built query",
                                "pagination": {"offset": 0, "limit": 50}}},
        }
        page = {
            "data": {"count": 1},
            "raw_items": [_usp_item("11111111", "Air dryer control")],
        }
        with patch("sources.long_task.relevance_gate.execute_tool",
                   new=AsyncMock(return_value=page)) as mock_exec:
            provider = _FakeProvider([
                RuntimeError("rewrite down"),
                {"scores": [{"id": "11111111", "score": 5}]},
            ])
            out = await phase0_gated_search(
                selected=selected, user_query="干燥空气",
                provider=provider, target_count=10,
            )
        sent_params = mock_exec.call_args[0][1]
        self.assertEqual(sent_params["body"]["q"], "llm built query")
        self.assertEqual(len(out["candidates"]), 1)


if __name__ == "__main__":
    unittest.main()
```

注意：`run_gated_search` 内部 import `execute_tool`，因此 patch 目标必须是 `sources.long_task.relevance_gate.execute_tool`。

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_relevance_gate.py -v`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 写最小实现**

创建 `sources/long_task/relevance_gate.py`：

```python
"""Relevance gating for PHASE0 search results.

Ranks search candidates against the user's original question via the
Flash LLM, keeps only relevant ones, dedupes families, and pages
through additional search results / rewritten queries when the kept
set falls short of the target count.
"""

import json
from copy import deepcopy
from typing import Any

from sources.long_task.candidate_metadata import (
    build_candidates,
    dedupe_candidates,
    ensure_search_fields,
    is_keyword_search_tool,
)
from sources.long_task.search_query_builder import build_search_queries
from sources.long_task.scene_tools import execute_tool

GATE_MIN_SCORE = 3
GATE_MAX_CANDIDATES_PER_CALL = 100
MAX_SEARCH_QUERIES = 3
MAX_PAGES_PER_QUERY = 4
MAX_COLLECTED_CANDIDATES = 300

GATE_SYSTEM_PROMPT = (
    "你是一个专利相关性评分器。判断候选专利与用户原始问题是否相关。\n"
    "评分标准（0-5）：\n"
    "5 — 直接解决用户问题中的技术需求\n"
    "4 — 高度相关，属于同一技术方向\n"
    "3 — 相关，可提供背景或部分覆盖\n"
    "2 — 仅表面相关（共享个别词语但技术方向不同）\n"
    "1 — 基本不相关\n"
    "0 — 完全不相关\n"
    "≥3 分视为相关。只依据标题、申请人、日期判断，不要猜测未知内容。\n"
    'Return JSON: {"scores": [{"id": "<patent_id>", "score": <0-5>}]} '
    "（每个候选一条，id 必须与输入完全一致）"
)


# ── Scoring ─────────────────────────────────────────────────────────────────

def _batch_text(candidates: list[dict], query: str) -> str:
    lines = [
        f"- id={c['patent_id']} | title={c.get('title') or '(no title)'}"
        f" | applicant={c.get('applicant') or '?'}"
        f" | filing={c.get('filing_date') or '?'}"
        for c in candidates
    ]
    return f"用户原始问题：{query}\n\n候选专利：\n" + "\n".join(lines)


def apply_scores(candidates: list[dict], result: Any) -> list[dict]:
    """Attach ``relevance_score`` from LLM output. Never raises."""
    by_id = {str(c["patent_id"]): c for c in candidates}
    scores = (result or {}).get("scores") if isinstance(result, dict) else None
    if not isinstance(scores, list):
        return candidates
    for entry in scores:
        if not isinstance(entry, dict):
            continue
        c = by_id.get(str(entry.get("id") or ""))
        if c is None:
            continue
        try:
            score = int(entry.get("score", -1))
        except (TypeError, ValueError):
            continue
        if 0 <= score <= 5:
            c["relevance_score"] = score
    return candidates


async def score_candidates(
    candidates: list[dict], query: str, provider: Any,
) -> list[dict]:
    """Score candidates in batches via the Flash LLM. Never raises."""
    if not candidates:
        return candidates
    out = []
    for i in range(0, len(candidates), GATE_MAX_CANDIDATES_PER_CALL):
        batch = candidates[i:i + GATE_MAX_CANDIDATES_PER_CALL]
        try:
            result = await provider.complete_json(
                GATE_SYSTEM_PROMPT, _batch_text(batch, query),
            )
        except Exception:
            result = None
        out.extend(apply_scores(batch, result))
    return out


def filter_by_relevance(
    candidates: list[dict], min_score: int = GATE_MIN_SCORE,
) -> list[dict]:
    """Keep candidates at/above the threshold, sorted by score desc.

    Candidates WITHOUT a score (gate LLM failure or partial output) are
    kept — a transient provider error must not zero out a search run;
    the pipeline degrades to legacy behavior instead.
    """
    kept = []
    for c in candidates:
        score = c.get("relevance_score")
        if score is None or (isinstance(score, (int, float)) and score >= min_score):
            kept.append(c)
    kept.sort(key=lambda c: c.get("relevance_score")
              if isinstance(c.get("relevance_score"), (int, float)) else -1,
              reverse=True)
    return kept


# ── Gated search loop ───────────────────────────────────────────────────────

def _page_size(params: dict) -> int:
    body = params.get("body")
    if isinstance(body, dict):
        pag = body.get("pagination")
        if isinstance(pag, dict) and isinstance(pag.get("limit"), int):
            return pag["limit"]
    return 50


def _with_offset(params: dict, offset: int) -> dict:
    out = deepcopy(params)
    body = out.get("body")
    if isinstance(body, dict):
        pag = body.get("pagination")
        if not isinstance(pag, dict):
            pag = {}
            body["pagination"] = pag
        pag["offset"] = offset
    return out


def _total_count(result: dict) -> int:
    data = result.get("data")
    if isinstance(data, dict):
        for k in ("count", "total"):
            v = data.get(k)
            if isinstance(v, (int, float)):
                return int(v)
    return 0


async def run_gated_search(
    selected: dict,
    user_query: str,
    provider: Any,
    rewrite: dict,
    target_count: int,
    task_id: str = "",
    logger: Any = None,
) -> dict:
    """Execute the search tool with rewritten queries, page through
    results, gate by relevance, and dedupe.

    Returns ``{"candidates": [...], "search_meta": {...}}``. Candidate
    dicts carry the full metadata from ``build_candidates`` plus
    ``relevance_score``.
    """
    def _log(msg: str) -> None:
        if logger is not None:
            logger.info(f"[task={task_id}] {msg}")

    tool = selected["tool"]
    base_params = selected.get("params") or {}
    queries = (rewrite or {}).get("queries") or []
    if not queries:
        queries = [None]  # keep the LLM-built params untouched

    keyword_tool = is_keyword_search_tool(tool)
    all_candidates: list[dict] = []
    seen_ids: set[str] = set()
    total_hits = 0
    pages_fetched = 0

    for q in queries[:MAX_SEARCH_QUERIES]:
        offset = 0
        for _page in range(MAX_PAGES_PER_QUERY):
            if len(all_candidates) >= MAX_COLLECTED_CANDIDATES:
                break
            params = ensure_search_fields(base_params)
            if keyword_tool and q:
                body = params.get("body")
                if isinstance(body, dict):
                    body["q"] = q
            params = _with_offset(params, offset)
            _log(f"gated_search request — q={q!r}, offset={offset}")
            result = await execute_tool(tool, params)
            raw_items = result.get("raw_items") or []
            total_hits = max(total_hits, _total_count(result))
            fresh = [c for c in build_candidates(raw_items)
                     if c["patent_id"] not in seen_ids]
            pages_fetched += 1
            if not fresh:
                break
            for c in fresh:
                seen_ids.add(c["patent_id"])
            all_candidates.extend(fresh)
            if len(raw_items) < _page_size(base_params):
                break
            offset += len(raw_items)

    _log(f"gated_search collected — candidates={len(all_candidates)}, "
         f"total_hits={total_hits}, pages={pages_fetched}")

    scored = await score_candidates(all_candidates, user_query, provider)
    kept = filter_by_relevance(scored)
    deduped, dropped = dedupe_candidates(kept)
    final = deduped[:target_count]
    search_meta = {
        "queries_used": [q for q in queries[:MAX_SEARCH_QUERIES] if q],
        "total_hits": total_hits,
        "pages_fetched": pages_fetched,
        "candidates_scored": len(scored),
        "gated_kept": len(kept),
        "gated_dropped": len(scored) - len(kept),
        "deduped_dropped": dropped,
        "final_count": len(final),
    }
    _log(f"gated_search done — meta={json.dumps(search_meta, ensure_ascii=False)}")
    return {"candidates": final, "search_meta": search_meta}


async def phase0_gated_search(
    selected: dict,
    user_query: str,
    provider: Any,
    target_count: int,
    task_id: str = "",
    logger: Any = None,
) -> dict:
    """PHASE0 entry point: rewrite the query, then run the gated search.

    LLM failures (rewrite or scoring) degrade gracefully; execution
    failures propagate like the legacy single-shot path.
    """
    rewrite = await build_search_queries(user_query, provider)
    if logger is not None:
        logger.info(
            f"[task={task_id}] search_rewrite — "
            f"queries={rewrite['queries']}"
        )
    return await run_gated_search(
        selected=selected, user_query=user_query, provider=provider,
        rewrite=rewrite, target_count=target_count,
        task_id=task_id, logger=logger,
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_relevance_gate.py -v`
Expected: PASS（10 passed）

- [ ] **Step 5: Commit**

```bash
git add sources/long_task/relevance_gate.py tests/test_relevance_gate.py
git commit -m "feat: add relevance gate and paged gated search for PHASE0"
```

---

### Task 4: `celery_worker.py` PHASE0 接线

**Files:**
- Modify: `celery_worker.py`（三处小改：局部变量初始化 `:586` 附近；PHASE0 搜索块 `:836-883`；无新增测试文件——接线逻辑全部落在 Task 1-3 已测模块内，本任务以全量回归验证）

**Interfaces:**
- Consumes: Task 3 `phase0_gated_search`；Task 2 `is_uspto_tool`；既有 `extract_patent_ids` / `extract_patent_id_url_map` / `extract_patent_id_pid_map` / `execute_tool`
- Produces（Task 5 依赖）:
  - 局部变量 `patent_meta_map: dict[str, dict]`（patent_id → 候选元数据，含 `title/applicant/status/filing_date/grant_date/cpc_codes/relevance_score`）
  - 局部变量 `search_meta: dict`（`queries_used/total_hits/pages_fetched/candidates_scored/gated_kept/gated_dropped/deduped_dropped/final_count`）
  - `patent_ids` / `id_url_map` 行为不变（仍是 id 列表与 id→url 映射）

- [ ] **Step 1: 初始化局部变量**

在 `celery_worker.py` 约 `:586-587`（`id_url_map` / `id_pid_map` 初始化处）追加两行：

```python
    id_url_map = {}          # patent_id → document_url from search results
    id_pid_map = {}          # patent_id → pid from CNIPA search results
    patent_meta_map = {}     # patent_id → candidate metadata (USPTO gated search)
    search_meta = {}         # gated-search transparency data for the report
```

- [ ] **Step 2: 替换 PHASE0 搜索块**

将 `celery_worker.py` `:836-883` 的 `if selected:` 块整体替换为：

```python
        if selected:
            update_task_status(task_id, 'searching_patents', 2,
                               _t('tool_search', batch_lang, reason=selected.get("reason", "")))
            from sources.long_task.candidate_metadata import is_uspto_tool
            from sources.long_task.scene_tools import extract_patent_id_pid_map
            params['patent_source'] = _infer_source_from_tool(
                selected['tool'], params.get('patent_source', 'cnipa'),
            )
            source_max = _get_max_patents_for_source(
                params.get('patent_source', 'cnipa'), max_patents,
                max_patents_cnipa, max_patents_uspto,
            )
            if is_uspto_tool(selected['tool']):
                from sources.long_task.relevance_gate import phase0_gated_search
                try:
                    gated = await phase0_gated_search(
                        selected=selected,
                        user_query=params['query'],
                        provider=flash_provider,
                        target_count=source_max,
                        task_id=task_id,
                        logger=_pipeline_logger,
                    )
                except Exception as e:
                    _pipeline_logger.warning(
                        f"[task={task_id}] PHASE0 gated_search_failed — {e}"
                    )
                    gated = None
                if gated:
                    candidates = gated.get('candidates') or []
                    patent_ids = [c['patent_id'] for c in candidates]
                    id_url_map = {
                        c['patent_id']: (
                            c.get('_raw', {}).get('document_url')
                            or c.get('_raw', {}).get('fulltext_url')
                            or c.get('_raw', {}).get('download_url')
                            or c.get('_raw', {}).get('url')
                            or ""
                        )
                        for c in candidates
                    }
                    patent_meta_map = {c['patent_id']: c for c in candidates}
                    search_meta = gated.get('search_meta') or {}
                    _pipeline_logger.info(
                        f"[task={task_id}] PHASE0 gated_search_result — "
                        f"final_count={len(patent_ids)}, "
                        f"meta={json.dumps(search_meta, ensure_ascii=False)}"
                    )
                else:
                    result = await execute_tool(selected['tool'], selected['params'])
                    raw_items = result.get('raw_items', []) or []
                    patent_ids = extract_patent_ids(raw_items)
                    id_url_map = extract_patent_id_url_map(raw_items)
            else:
                # CNIPA / non-USPTO tools: legacy single-shot path unchanged
                result = await execute_tool(selected['tool'], selected['params'])
                raw_items = result.get('raw_items', []) or []
                patent_ids = extract_patent_ids(raw_items)
                id_url_map = extract_patent_id_url_map(raw_items)
                id_pid_map = extract_patent_id_pid_map(raw_items)
        if patent_ids:
            # Safety net only — gated search already caps at source_max
            patent_ids = patent_ids[:source_max]
            total = len(patent_ids)
            if not (checkpoint and checkpoint.get('pending')):
                pending = patent_ids
                table_rows = []
            else:
                _pipeline_logger.info(
                    f"[task={task_id}] CHECKPOINT_RESUME_SCENE — "
                    f"keeping checkpoint state: table_rows={len(table_rows)}, "
                    f"pending={len(pending)}"
                )
            update_task_status(task_id, 'searching_patents', 5,
                               _t('search_complete', batch_lang, count=len(patent_ids)),
                               patent_ids=patent_ids)
        else:
            set_task_failed(task_id, _t('no_patents_found', batch_lang))
            user_id_for_analytics = params.get('user_id', '')
            if user_id_for_analytics:
                track_event("long_task:fail", user_id=user_id_for_analytics,
                            task_id=task_id,
                            extra={"error": "no_patents_found"})
            return {'status': 'failed', 'task_id': task_id,
                    'error': 'No patents found matching the search criteria'}
```

替换时注意：旧代码中被替换范围内的 `from sources.long_task.scene_tools import extract_patent_id_pid_map`（原 `:843`）一并移除，由上方新代码统一引入；原 `:853-855` 的 `_infer_source_from_tool` 与 `:857-860` 的 `source_max` 计算被上移到 `if selected:` 开头，删除原位置的重复代码。`extract_patent_ids` / `extract_patent_id_url_map` 由文件中原有的 `:814-819` 顶部 import 提供，无需新增。

- [ ] **Step 3: 运行全量后端回归**

Run: `PYTHONUTF8=1 python -m pytest tests/ -q`
Expected: 既有 121 项 + 新增 29 项全部 PASS（1 个无关 Starlette 警告允许存在）。若出现 `json` 未定义等 NameError，确认 `celery_worker.py` 顶部已 `import json`（现有代码 `:782` 已在用，应已存在）。

- [ ] **Step 4: 静态自检语法**

Run: `PYTHONUTF8=1 python -m py_compile celery_worker.py sources/long_task/relevance_gate.py sources/long_task/candidate_metadata.py sources/long_task/search_query_builder.py`
Expected: 无输出（成功）

- [ ] **Step 5: Commit**

```bash
git add celery_worker.py
git commit -m "feat: wire PHASE0 USPTO search through gated search with legacy fallback"
```

---

### Task 5: 报告元数据与检索说明章节

**Files:**
- Modify: `sources/long_task/report_generator.py`
- Modify: `celery_worker.py`（`_analyze_one` 合并 `_meta`：约 `:1064`；PHASE3 追加方法论章节：约 `:1331`）
- Test: `tests/test_report_generator.py`（追加测试类）

**Interfaces:**
- Consumes: Task 4 的 `patent_meta_map`、`search_meta`
- Produces:
  - `append_methodology_section(sections: list[dict], search_meta: dict, lang: str = "zh") -> list[dict]` — 纯函数
  - `_meta_lines(row: dict, lang: str = "zh") -> list[str]` — 纯函数
  - 行数据新增 `_meta` 键（dict，仅当 `patent_meta_map` 有对应条目时非空）

- [ ] **Step 1: 写失败测试**

在 `tests/test_report_generator.py` 末尾追加：

```python
# ── Metadata lines & methodology section ─────────────────────────────────────

from sources.long_task.report_generator import (
    _meta_lines,
    append_methodology_section,
)


class TestMetaLines(unittest.TestCase):
    def test_renders_available_meta_fields(self):
        lines = _meta_lines({
            "_meta": {
                "title": "Air dryer control using humidity",
                "applicant": "New York Air Brake",
                "status": "Patented Case",
                "filing_date": "2016-03-01",
                "cpc_codes": ["B60T 17/00", "F26B 21/08"],
                "patent_number": "10150077",
            },
        })
        joined = "\n".join(lines)
        self.assertIn("标题: Air dryer control using humidity", joined)
        self.assertIn("申请人: New York Air Brake", joined)
        self.assertIn("法律状态: Patented Case", joined)
        self.assertIn("CPC 分类号: B60T 17/00, F26B 21/08", joined)

    def test_empty_meta_renders_nothing(self):
        self.assertEqual(_meta_lines({}), [])
        self.assertEqual(_meta_lines({"foo": "bar"}), [])


class TestAppendMethodologySection(unittest.TestCase):
    def test_appends_section_with_meta(self):
        sections = [{"heading": "核心发现", "description": "x"}]
        out = append_methodology_section(sections, {
            "queries_used": ['("air dryer" OR desiccant)'],
            "total_hits": 598,
            "pages_fetched": 2,
            "candidates_scored": 100,
            "gated_kept": 12,
            "gated_dropped": 88,
            "deduped_dropped": 1,
            "final_count": 10,
        }, lang="zh")
        self.assertEqual(len(out), 2)
        self.assertEqual(out[-1]["heading"], "检索说明与局限")
        desc = out[-1]["description"]
        self.assertIn("598", desc)
        self.assertIn("相关性评分", desc)
        self.assertIn("不构成法律意见", desc)

    def test_skips_when_no_meta(self):
        out = append_methodology_section([{"heading": "a", "description": "b"}], {})
        self.assertEqual(len(out), 1)

    def test_english_variant(self):
        out = append_methodology_section([], {
            "queries_used": ['"air dryer"'],
            "total_hits": 10, "pages_fetched": 1,
            "candidates_scored": 5, "gated_kept": 3,
            "gated_dropped": 2, "deduped_dropped": 0,
            "final_count": 3,
        }, lang="en")
        self.assertEqual(out[-1]["heading"], "Search Methodology & Limitations")
        self.assertIn("10", out[-1]["description"])


class TestSummaryIncludesMeta(unittest.IsolatedAsyncioTestCase):
    async def test_exec_summary_prompt_includes_meta_fields(self):
        captured = []

        class RecordingProvider:
            def __init__(self):
                self.mock_llm = MagicMock()

            def _get_langchain_llm(self, streaming=False):
                async def _astream(*args, **kwargs):
                    captured.append(args[0])
                    class Chunk:
                        content = "### 核心发现\n\n摘要内容 **[11111111]**"
                    yield Chunk()
                self.mock_llm.astream = _astream
                return self.mock_llm

        provider = RecordingProvider()
        await generate_executive_summary(
            table_rows=[{
                "专利号": "11111111", "发明点": "x", "技术方案": "y",
                "_meta": {"title": "Air dryer", "applicant": "ACME",
                          "status": "Patented Case"},
            }],
            columns=["专利号", "发明点", "技术方案"],
            query="技术趋势",
            provider=provider,
            lang="zh",
        )
        human_msg = captured[0][1]
        self.assertIn("标题: Air dryer", human_msg)
        self.assertIn("申请人: ACME", human_msg)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONUTF8=1 python -m pytest tests/test_report_generator.py -v`
Expected: FAIL（ImportError: cannot import name '_meta_lines'）

- [ ] **Step 3: 修改 `report_generator.py`**

在文件末尾追加两个纯函数，并在 `generate_executive_summary` / `generate_report_section` 的 entry 组装处各加一行元数据（zh/en 共用——`_meta_lines` 内部用固定中文标签，与既有报告语言无关时保持中文标签即可）：

3a. 文件末尾追加：

```python
# ── Metadata lines & methodology section ─────────────────────────────────────

def _meta_lines(row: dict, lang: str = "zh") -> list[str]:
    """Render ``row["_meta"]`` (candidate metadata) as bullet lines."""
    meta = row.get("_meta")
    if not isinstance(meta, dict):
        return []
    labels = {
        "title": "标题",
        "applicant": "申请人",
        "status": "法律状态",
        "filing_date": "申请日",
        "grant_date": "授权日",
        "patent_number": "专利号",
        "cpc_codes": "CPC 分类号",
    }
    lines = []
    for key, label in labels.items():
        v = meta.get(key)
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v)
        if v:
            lines.append(f"  - {label}: {v}")
    return lines


def append_methodology_section(
    sections: list[dict], search_meta: dict, lang: str = "zh",
) -> list[dict]:
    """Append a fixed 'search methodology & limitations' section."""
    if not search_meta:
        return sections
    if lang == "zh":
        heading = "检索说明与局限"
        parts = []
        queries = search_meta.get("queries_used") or []
        if queries:
            parts.append(f"本次检索使用检索式：{'；'.join(str(q) for q in queries)}。")
        parts.append(
            f"共检索到 {search_meta.get('total_hits', 0)} 条记录，"
            f"翻阅 {search_meta.get('pages_fetched', 0)} 页，"
            f"对 {search_meta.get('candidates_scored', 0)} 个候选专利进行相关性评分，"
            f"保留 {search_meta.get('gated_kept', 0)} 条"
            f"（过滤 {search_meta.get('gated_dropped', 0)} 条，"
            f"去重 {search_meta.get('deduped_dropped', 0)} 条），"
            f"最终分析 {search_meta.get('final_count', 0)} 件。"
        )
        parts.append(
            "本报告基于 USPTO 公开数据自动生成，检索结果与相关性评分为 AI 辅助判断，"
            "不构成法律意见；如需查全率保障，建议补充人工复核。"
        )
    else:
        heading = "Search Methodology & Limitations"
        parts = []
        queries = search_meta.get("queries_used") or []
        if queries:
            parts.append(
                f"Searches used the following queries: {'; '.join(str(q) for q in queries)}."
            )
        parts.append(
            f"Total hits: {search_meta.get('total_hits', 0)} across "
            f"{search_meta.get('pages_fetched', 0)} pages; "
            f"{search_meta.get('candidates_scored', 0)} candidates scored for relevance; "
            f"{search_meta.get('gated_kept', 0)} kept "
            f"({search_meta.get('gated_dropped', 0)} filtered, "
            f"{search_meta.get('deduped_dropped', 0)} deduplicated); "
            f"{search_meta.get('final_count', 0)} analyzed."
        )
        parts.append(
            "This report is generated automatically from public USPTO data. "
            "Search results and relevance scoring are AI-assisted and do not "
            "constitute legal advice."
        )
    return sections + [{"heading": heading, "description": " ".join(parts)}]
```

3b. `generate_executive_summary` 的 `parts` 组装处（约 `:57-60`，`for col in columns:` 循环之后、`if r.get("_summary"):` 之前）插入：

```python
        parts.extend(_meta_lines(r))
```

3c. `generate_report_section` 的 `parts` 组装处（约 `:266-271`，同样的 `for col in columns:` 循环之后）插入：

```python
        parts.extend(_meta_lines(r))
```

- [ ] **Step 4: 修改 `celery_worker.py` 两处**

4a. `_analyze_one` 内、`_table_rows[_i] = _row`（约 `:1064`）之前插入：

```python
            _row["_meta"] = patent_meta_map.get(_patent_id, {})
```

4b. PHASE3 `sections = outline.get('sections', ...)`（约 `:1331`）之后插入：

```python
    if search_meta:
        from sources.long_task.report_generator import append_methodology_section
        sections = append_methodology_section(sections, search_meta, lang=batch_lang)
```

- [ ] **Step 5: 运行测试确认通过**

Run: `PYTHONUTF8=1 python -m pytest tests/test_report_generator.py -v && PYTHONUTF8=1 python -m pytest tests/ -q`
Expected: 新增测试 PASS；全量回归全绿

- [ ] **Step 6: Commit**

```bash
git add sources/long_task/report_generator.py celery_worker.py tests/test_report_generator.py
git commit -m "feat: enrich report with candidate metadata and search methodology section"
```

---

### Task 6: 全量回归 + 多领域基准验收

**Files:** 无代码改动（验收为手动执行）

**验收原则：** 验证的是通用检索质量，不是单一查询。基准集覆盖多个技术领域；每个查询按统一指标打分；任一查询可单独失败并定位断点，不因单点失败阻塞整体。

- [ ] **Step 1: 后端全量回归**

Run: `PYTHONUTF8=1 python -m pytest tests/ -q`
Expected: 全部 PASS（121 旧 + 35 新 = 156 项，允许 1 个无关 Starlette 警告）

- [ ] **Step 2: 部署到测试环境**

部署后端到 api-test.copiioai.com（沿用既有部署流程）。

- [ ] **Step 3: 基准查询集（5 个不同技术领域，可扩充至 8-10 个）**

在 test.copiioai.com 逐条提交以下查询（1 号是本次回归用例，其余为新增领域样例；建议之后从真实用户查询日志采样扩充）：

| # | 查询 | 领域 |
|---|---|---|
| 1 | 帮我查找工业中在线干燥空气源提供和设备内部环境湿度精准控制的相关专利 | 工业除湿（回归用例） |
| 2 | 电动汽车动力电池热失控预警和散热结构相关专利 | 新能源/电池 |
| 3 | 半导体工艺腔室的温度控制与晶圆加热相关专利 | 半导体设备 |
| 4 | AR 眼镜光学波导与全息显示相关专利 | 消费电子/光学 |
| 5 | 医学影像的 AI 辅助诊断算法相关专利 | 医疗 AI |

- [ ] **Step 4: 每查询验收清单（5 项全查）**

检查 `long_task_pipeline.log` 与最终报告，逐查询打分：

| 检查项 | 通过标准 |
|---|---|
| `search_rewrite` 日志 | 检索式为 OR 组 + 引号短语 + 英文形态（如 `("3d printing" OR "additive manufacturing") AND (...)…`），无中文字符、无逐字直译长短语 |
| `gated_search done` meta | `candidates_scored > 0`，`gated_dropped ≥ 0`；`total_hits` 记录在案 |
| 最终 10 件专利相关率 | **≥ 8/10 与查询主题直接相关**；不得出现跨领域噪声（如查询电池却出现食品加工/金融类专利） |
| 同族去重 | 最终列表内无同标题/同族重复 |
| 报告质量 | 含「检索说明与局限」章节（披露检索式/命中数/过滤数/免责声明）；行内含法律状态/申请人/CPC 元数据 |

- [ ] **Step 5: 回归用例专项对照（仅查询 1）**

查询 1 的最终列表应可识别出 workbuddy 报告中的美国专利（以标题+申请人识别，号码形式不计）：US10150077B2（New York Air Brake 双塔干燥器）、US20170347473A1（Eaton 外壳水分控制）、US5934368A（Hitachi 防凝露）、US6205796B1（IBM 亚露点冷却）、US7905096B1（Lenovo 机架除湿）、US8189334B2（Lenovo 除湿冷却）、US11594802B2（Thales 天线防凝露）、US20170282883A1（西门子压缩空气膨胀除湿）。参考线：≥6/8；未达线时单独定位断点（改写质量 / 闸门误杀 / 检索式语法），不作为整体通过判据。

- [ ] **Step 6: 整体判定与记录**

- 整体通过：**5/5 查询达到 Step 4 的相关率标准**（查询 1 的专项召回为参考指标）
- 任一查询失败：记录断点位置与根因到会话文件（/save-session），形成「Phase 2 语义索引启动判据」或针对性修复任务
- 验收记录提交：

```bash
git add -A && git commit -m "docs: record patent search Phase 1 acceptance results" || true
```

---

## Self-Review

**1. Spec coverage**（对应此前讨论的 A 类 5 项 + 决策）：
- 查询改写（概念抽取 + 术语映射 + OR 组）→ Task 1 ✓
- 相关性闸门（截断前打分 + 翻页补足）→ Task 3（`run_gated_search` 翻页 + `filter_by_relevance`）✓
- 同族去重 → Task 2（`dedupe_candidates`，覆盖同标题 + parentContinuityBag 两类）✓
- CPC 字段增强 → Task 2（`ensure_search_fields` + `_extract_cpc_codes`）✓
- 报告升级（元数据 + 检索说明与局限）→ Task 5 ✓
- US-only 范围（不查中国专利：CNIPA 零改动零新逻辑、多源合并不在本计划）→ Task 4 的 `is_uspto_tool` 分支 ✓
- **通用场景而非单查询优化**：改写 prompt 无领域预设关键词、模块无硬编码阈值分支 → Task 1 prompt + 全模块 ✓；验收为 5 领域基准查询集（统一指标：相关率 ≥8/10、无跨领域噪声、去重、报告披露），干燥空气仅作回归用例的参考对照 → Task 6 ✓
- 语义检索（B 类）→ 明确不在此计划，Task 6 验收结果作为 Phase 2 启动判据 ✓

**2. Placeholder scan:** 无 TBD/TODO；每个 Step 均有完整代码与期望输出。

**3. Type consistency:**
- `build_search_queries` 返回 `{"concepts": [], "queries": []}`，Task 3 `run_gated_search` 按 `rewrite["queries"]` 消费 ✓
- `build_candidates` 产出键（`patent_id/title/applicant/status/filing_date/grant_date/patent_number/cpc_codes/_raw`）与 Task 3 评分文本、Task 5 `_meta_lines` 标签键一致 ✓
- `phase0_gated_search` 返回 `{"candidates", "search_meta"}`，Task 4 按此解构；`search_meta` 键与 Task 5 `append_methodology_section` 消费键一致（`queries_used/total_hits/pages_fetched/candidates_scored/gated_kept/gated_dropped/deduped_dropped/final_count`）✓
- `dedupe_candidates` 依赖 `relevance_score`（Task 3 在去重前调用 `score_candidates` 写入）✓
- Task 4 局部变量 `patent_meta_map` / `search_meta` 在 Task 5 的 `_analyze_one` / PHASE3 同函数作用域内使用 ✓

**4. 回归风险:** `id_pid_map` 提取被保留在 CNIPA legacy 分支；`source_max` 上移到 `if selected:` 内提前计算，USPTO 截断语义不变（gated 已截断 + 安全切片兜底）；`checkpoint` 恢复路径不受影响（`pending`/`table_rows` 赋值逻辑未变）。
